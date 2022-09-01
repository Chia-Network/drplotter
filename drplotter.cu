/*
 ============================================================================
 Name        : drplotter.cu
 Author      : NH
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <cuda_fp16.h>

// for mmap
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h> /* mmap() is defined in this header */
#include <fcntl.h>

#include "chia/util.hpp"
#include "chia/chacha8.h"
#include "nick_globals.hpp"
#include "attack.hpp"
#include "phase2.hpp"



const uint16_t THREADS_FOR_MATCHING = 256; // 386 is 10600ms matching. 256 is 9761ms matching. 237 is...10109

int cmd_read = 0;

using milli = std::chrono::milliseconds;
int64_t total_gpu_time_ms = 0;
int64_t total_transfer_in_time_ms = 0;
int64_t total_transfer_out_time_ms = 0;
int64_t total_chacha_time_ms = 0;
int64_t total_match_time_ms = 0;
uint64_t total_transfer_in_bytes = 0;
uint64_t total_transfer_out_bytes = 0;
int64_t table_gpu_time_ms = 0;
int64_t table_transfer_in_time_ms = 0;
int64_t table_transfer_out_time_ms = 0;
int64_t table_match_time_ms = 0;
uint64_t table_transfer_in_bytes = 0;
uint64_t table_transfer_out_bytes = 0;

// global memory
char *host_criss_cross_blocks; // aka host_meta_blocks
char *host_refdata_blocks;
char *device_buffer_A;
char *device_buffer_B;

char *device_buffer_C;
char *device_buffer_T3_base;
char *device_buffer_refdata;

int* device_block_entry_counts; // [BATCHES];
int* device_local_kbc_num_entries;
uint32_t host_criss_cross_entry_counts[BATCHES * BATCHES]; // kbc counts for each block


#include "nick_blake3.hpp"


template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_calc_only(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {
	// match: 25804 ms
	// phase 1: 4366ms
	__shared__ uint Lys[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ uint Rys[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ Index_Match matches[512];//KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int yr_yl_bid_m_results[kB*2];
	__shared__ int yr_yl_cid_mod_kC[kC*2];


	uint32_t kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	const uint8_t doPrint = 1;//(global_kbc_L_bucket_id < 10) ? 1 : 0; // start_kbc_L > 0 ? 1: 0; // 0 is none, 1 is basic, 2 is detailed

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (num_L == 0) {
		return;
	}

	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		Rys[pos_R] = (R_entry.y / kC) + ((R_entry.y % kC) << 8); // do mod and div entries too in bitmask.
	}
	for (int pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
		Lys[pos_L] = (L_entry.y / kC) + ((L_entry.y % kC) << 8);
	}
	const int16_t parity = (global_kbc_L_bucket_id) % 2;
	for (int i=threadIdx.x;i<kB*2;i+=blockDim.x) {
		int16_t yr_bid_minus_yl_bid = (i-kB);
		int16_t m = yr_bid_minus_yl_bid;
		if (m < 0) m+=kB;
		int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC;
		//if (blockIdx.x == 0) printf("m: %d  parity:%d m2_parity_squared: %d\n", m, parity, m2_parity_squared);
		yr_yl_bid_m_results[i] = (m << 8) + m2_parity_squared;
	}
	for (int i=threadIdx.x;i<kC*2;i+=blockDim.x) {
		int16_t yr_cid_minus_yl_cid = i-kC;
		if (yr_cid_minus_yl_cid < 0) yr_cid_minus_yl_cid += kC;
		yr_yl_cid_mod_kC[i] = yr_cid_minus_yl_cid;
	}

	if (threadIdx.x == 0) {
		total_matches = 0;
	}
	__syncthreads(); // all written initialize data should sync

	//   For any 0 <= m < kExtraBitsPow:
	//   yl / kBC + 1 = yR / kBC   AND
	//   (yr % kBC) / kC - (yl % kBC) / kC = m   (mod kB)  -> MEANS yr/kC can only match with the 64 slots including and to the right of yl/kC
	//   (yr % kBC) % kC - (yl % kBC) % kC = (2m + (yl/kBC) % 2)^2   (mod kC)

	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		const uint yr_data = Rys[pos_R];
		//int16_t yr_kbc = yr_data;// & 0b01111111111111111;
		const int16_t yr_bid = yr_data & 0b011111111; // yr_kbc / kC; // values [0..kB]
		const int16_t yr_cid = (yr_data >> 8);//yr_kbc % kC;//(yr_data >> 24);
		for (int pos_L = 0; pos_L < num_L; pos_L++) {
			// do L_entry and R_entry match?
			const uint yl_data = Lys[pos_L];
			//int16_t yl_kbc = yl_data;// & 0b01111111111111111;
			const int8_t yl_bid = yl_data & 0b011111111; //yl_kbc / kC; values [0..kB]
			const int8_t yl_cid = yl_data >> 8;//yl_kbc % kC;//(yl_data >> 24);

			int16_t m_results = yr_yl_bid_m_results[yr_bid-yl_bid+kB];
			int16_t m = m_results >> 8;//& 0b011111111;
			int16_t m2_parity_squared = (m_results & 0b011111111);
			int16_t formula_two = yr_yl_cid_mod_kC[yr_cid - yl_cid + kC];

			//int16_t formula_one = yr_bid - yl_bid; // this should actually give m
			//if (formula_one < 0) {
			//	formula_one += kB;
			//}
			//int16_t m = formula_one;
			//if (m >= kB) {
			//	m -= kB;
			//}

			//int16_t m = (yr_bid - yl_bid);
			//if (m < 0) m+=kB;
			//if (m >= kB) m-=kB;

			//if (m < 64) {
				// passed first test
				//const int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC; // values [0..127]
				//int16_t formula_two = yr_cid - yl_cid;
				//if (formula_two < 0) formula_two += kC;

				if ((m < 64) && (formula_two == m2_parity_squared)) {
					// we have a match.
					int num_matches = atomicAdd(&total_matches,1);
					//if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
					//	printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
					//} else {
						Index_Match match = { };
						match.idxL = pos_L;
						match.idxR = pos_R;//value >> 4;
						matches[num_matches] = match;
					//}
				}
			//}
						/*




			uint16_t m = (yr_bid - yl_bid) % kB; // 77ms w/o branch mod test, big jump w/ mod. - 158ms
			uint16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC;
			uint16_t formula_two = (yr_cid - yl_cid) % kC;
			//if (m < 0) {
			//	m += kB;
			//}// else if (m >= kB) m-=kB;
			if ((m < 64) && (m2_parity_squared == formula_two)) {
				//uint16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC;
				//uint16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC;
				int num_matches = atomicAdd(&total_matches,1);
				if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
					printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
				} else {
					Index_Match match = { };
					match.idxL = pos_L;
					match.idxR = pos_R;//value >> 4;
					matches[num_matches] = match;
				}
			}*/
		}
	}

	/*for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		int16_t yr_kbc = R_entry.y;
		int16_t yr_bid = yr_kbc / kC; // values [0..kB]
		for (uint16_t pos_L = 0; pos_L < num_L; pos_L++) {
			// do L_entry and R_entry match?
			BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
			int16_t yl_kbc = L_entry.y;
			int16_t yl_bid = yl_kbc / kC; // values [0..kB]
			int16_t formula_one = yr_bid - yl_bid; // this should actually give m
			if (formula_one < 0) {
				formula_one += kB;
			}
			int16_t m = formula_one;
			if (m >= kB) {
				m -= kB;
			}
			if (m < 64) {
				// passed first test
				int16_t yl_cid = yl_kbc % kC; // % kBC % kC = %kC since kBC perfectly divisible by kC
				int16_t yr_cid = yr_kbc % kC;
				int16_t parity = (global_kbc_L_bucket_id) % 2;
				int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC; // values [0..127]
				int16_t formula_two = yr_cid - yl_cid;
				if (formula_two < 0) {
					formula_two += kC;
				}
				if (formula_two == m2_parity_squared) {
					// we have a match.
					int num_matches = atomicAdd(&total_matches,1);
					if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
						printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
					} else {
						Index_Match match = { };
						match.idxL = pos_L;
						match.idxR = pos_R;//value >> 4;
						matches[num_matches] = match;
					}
				}
			}
		}
	}

	__syncthreads();*/


	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			printf("Bucket L %u Total matches: %u\n", kbc_L_bucket_id, total_matches);
		}
		if (total_matches > KBC_MAX_ENTRIES_PER_BUCKET) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET);
			total_matches = KBC_MAX_ENTRIES_PER_BUCKET;
		}
	}

	__syncthreads();

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
				//if ((calc_y == 21557) && (L_Entry.meta[0] == 3620724289) && (R_Entry.meta[0] == 2663198278)) {
			//	printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
				//Ly is:[20932] Lx: [322482289] Rx: [3382886636]  f result:[273114646565]
				//if (blake_result == 56477140042) {
				//	printf(" ---** BLAKE CORRECT **\n");
				//} else {
				//	printf(" ---** BLAKE WRONG :(((( \n");
				//}
				// Ly is:[21557] Lx: [3620724289] Rx: [2663198278]  f result:[56477140042]
				//}
			//}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			if (global_kbc_L_bucket_id == 1) {
				uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
				uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
				printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		if (table < 6) {
			uint64_t batch_bucket = blake_result >> (38-6);
			const uint64_t block_mod = (uint64_t) 1 << (38-6);
			pair.y = (uint32_t) (blake_result % block_mod);
			int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
			uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
			if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
				printf("ERROR: results address overflow\n");
			} else {
				bucketed_out[pair_address] = pair;
			}
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	//if (threadIdx.x == 0) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);
		//if ((global_kbc_L_bucket_id % 25000 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

	//}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if (threadIdx.x == 0) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
	// table 1 4865ms match time to beat.
	// with shared mem for pos_L/R is 3942 - win!
	// formula improvement (one branch) - 3810ms
	// removal of max kbc test in m loop - 3639ms +33% faster.
	// shared compute buffers to prevent % and division - 2280ms!
	//   -- now getting dangerously close to best algo time of 1606ms :)
}


template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_test(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {
	// T1 match: 1714 ms -> with delaying extras: 1630
	//Total tables time: 73726 ms
	//        match: 10015 ms -> 9705ms with delaying extras
	__shared__ int total_matches;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	uint8_t doPrint = 2;

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
		if (doPrint > 1) {
			printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
			if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
				printf("ERROR numL or numR > max entries\n");
				return;
			}
			if ((num_L == 0) || (num_R == 0) ) {
				printf("ERROR: numL and numR are 0\n");
				return;
			}
		}
	}

	//   For any 0 <= m < kExtraBitsPow:
	//   yl / kBC + 1 = yR / kBC   AND
	//   (yr % kBC) / kC - (yl % kBC) / kC = m   (mod kB)  -> MEANS (1) yr/kC can only match with the 64 slots including and to the right of yl/kC
	//   (yr % kBC) % kC - (yl % kBC) % kC = (2m + (yl/kBC) % 2)^2   (mod kC)

	// yr_kc's : [0..127] -> contains what? Either y-value, then compute matching m, or contains %kC
	// if /kC distance yr to yl is 5, m = 5, then diff %kC must be (20^2)%kC = 400 % kC =

	//  000001111111111000000 yl1
	//  000111111111100000000 y12
	//  000000011111111111000 yl3

	const uint16_t parity = global_kbc_L_bucket_id % 2;
	for (int16_t Ry = threadIdx.x; Ry < kBC; Ry+=blockDim.x) {
		int16_t yr_kbc = Ry;
		int16_t yr_bid = yr_kbc / kC; // values [0..kB]
		for (int16_t Ly = 0; Ly < kBC; Ly++) {
			int16_t yl_kbc = Ly;
			int16_t yl_bid = yl_kbc / kC; // values [0..kB]
			int16_t formula_one = yr_bid - yl_bid; // this should actually give m
			if (formula_one < 0) {
				formula_one += kB;
			}
			int16_t m = formula_one;
			if (m >= kB) {
				m -= kB;
			}
			if (m < 64) {
				// passed first test
				int16_t yl_cid = yl_kbc % kC; // % kBC % kC = %kC since kBC perfectly divisible by kC
				int16_t yr_cid = yr_kbc % kC;
				int16_t parity = (global_kbc_L_bucket_id) % 2;
				int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC; // values [0..127]
				int16_t formula_two = yr_cid - yl_cid;
				if (formula_two < 0) {
					formula_two += kC;
				}
				if (formula_two == m2_parity_squared) {
					// we have a match.
					printf("match Ly:%u Ry:%u\n", Ly, Ry);
					atomicAdd(&total_matches,1);
				}
			}
		}
	}
	if (threadIdx.x == 0) {
		printf("Done. Total matche: %u", total_matches);
	}

}


template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {
	// T1 match: 1714 ms -> with delaying extras: 1630
	//Total tables time: 73726 ms
	//        match: 10015 ms -> 9705ms with delaying extras
	const uint16_t NUM_RMAPS = (kBC/2)+1;
	__shared__ int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
	__shared__ uint32_t nick_rmap_extras_rl[32];
	__shared__ uint16_t nick_rmap_extras_ry[32];
	__shared__ uint16_t nick_rmap_extras_pos[32];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int num_extras;
	__shared__ int y_duplicate_counts;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	uint8_t doPrint = 1;

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
		num_extras = 0;
		y_duplicate_counts = 0;
		if (doPrint > 1) {
			printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
			if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
				printf("ERROR numL or numR > max entries\n");
				return;
			}
			if ((num_L == 0) || (num_R == 0) ) {
				printf("ERROR: numL and numR are 0\n");
				return;
			}
		}
	}
	// unfortunately to clear we have to do this
	for (int i = threadIdx.x; i < NUM_RMAPS; i += blockDim.x) {
		nick_rmap[i] = 0;
	}
	__syncthreads(); // all written initialize data should sync


	/*bool printandquit = ((global_kbc_L_bucket_id == 0));
		if (printandquit) {
			if (threadIdx.x == 0) {

				printf("R_y list:\n");
				for (size_t pos_R = 0; pos_R < num_R; pos_R++) {
					uint16_t r_y = kbc_R_entries[pos_R].y;
					printf("[x:%u y:%u]\n",kbc_R_entries[pos_R].meta[0], r_y);
				}
				printf("L_y list num %u:\n", num_L);
				for (size_t pos_L = 0; pos_L < num_L; pos_L++) {
					uint16_t l_y = kbc_L_entries[pos_L].y;
					printf("[x:%u y:%u]\n",kbc_L_entries[pos_L].meta[0], l_y);
				}
			}
		}*/
	//__syncthreads();
	uint16_t parity = global_kbc_L_bucket_id % 2;


	for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		uint16_t r_y = R_entry.y;

		// r_y's share a block across two adjacent values, so kbc_map just works out which part it's in.
		int kbc_map = r_y / 2;
		const int kbc_box_shift = (r_y % 2) * 15;
		int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)

		int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
		rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;
		if (rmap_value == 0) {
			// if we added to an empty spot, what we do is add the pos_R here in the lower 9 bits of the box
			// and ONLY for this one.
			atomicAdd(&nick_rmap[kbc_map], (pos_R << kbc_box_shift));
			//if (printandquit) {
			//	printf("r_y: %u   pos:%u\n", r_y, pos_R);
			//}
		} else {
			// we hit duplicate entry...add this to a row
			int slot = atomicAdd(&num_extras, 1);
			nick_rmap_extras_ry[slot] = r_y;
			nick_rmap_extras_pos[slot] = pos_R;
		}

	}

	__syncthreads(); // wait for all threads to write r_bid entries

	// benchmark: 66ms at this point
	//if ((nick_rmap_extras_ry[threadIdx.x % 32] + nick_rmap_extras_pos[threadIdx.x % 32]) == 2334534423) printf("bogus");
	//return;

	// load parity tables into shared
	/*if (printandquit) {
		if (threadIdx.x == 0) {
			printf("num extras bucket %u : %u   parity: %u \n", global_kbc_L_bucket_id, num_extras, parity);

			for (int i=0;i<kBC;i++) {
				int kbc_map = i / 2;
				const int kbc_box_shift = (i % 2) * 15;
				int rmap_value = nick_rmap[kbc_map];
				rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

				//uint16_t rmap_value = nick_rmap[i];
				uint16_t pos = (rmap_value & 0b0111111111);
				if (rmap_value > 0) {
					printf("kbc:%i  value:%u pos:%u\n", i, rmap_value, pos);
				}
			}

		}

	}
	__syncthreads();*/


	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
			//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
			BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
			uint16_t l_y = L_entry.y;
			uint16_t indJ = l_y / kC;
			//printf("scanning for pos_L: %u\n", pos_L);

			// this part is killer, this does add bulk of time.
			// weird simplfiying the math doesn't help much unless you pragma unroll it
			// might be too much branching inside too.
				// setup code for loop increment "optimization"
				//uint16_t indJ_mod_kB_times_kC = ((indJ + 0) % kB) * kC;
				//uint16_t start_parity_add = 4 + parity * 4;
				//uint16_t parity_base = (parity + l_y) % kC;
				//const uint16_t m_switch_kb = kB - indJ; // calculate point at which indJ + m is %kb!
			for (int m=0;m<64;m++) {

				//uint16_t r_target = L_targets[parity][l_y][m]; // this performs so badly because this lookup
					// is super-inefficient.

				// 27.58ms
				uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);





					// a cute "optimization" but saves no time whatsoever...27.7ms instead of 27.58ms :/
					//if (m_switch_kb == m) indJ_mod_kB_times_kC = ((indJ + m) % kB) * kC; // 323ms // 490
					//uint16_t r_target = indJ_mod_kB_times_kC + parity_base;
					//indJ_mod_kB_times_kC += kC; // 256ms
					//parity_base += start_parity_add;
					//if (parity_base >= kC) parity_base -= kC;
					//start_parity_add += 8;
					//if (start_parity_add >= kC) start_parity_add -= kC;
					//if (test_target != r_target) {
					//	printf("Ly: %u m: %u target: %u test_target: %u \n", l_y, m, r_target, test_target);
					//}


				//if (r_target + indJ == m) bogus_match_counter++;
				//if (bogus_match_counter >= KBC_MAX_ENTRIES_PER_BUCKET) {
				//		printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, bogus_match_counter);
				//}

				// find which box our r_target is in, extra the 15bit value from that box
				int kbc_map = r_target / 2;
				const int kbc_box_shift = (r_target % 2) * 15;
				int rmap_value = nick_rmap[kbc_map];
				rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

				if (rmap_value > 0) {
					// the pos_R is the lower 9 bits of that 15bit boxed value
					uint16_t pos_R = rmap_value & 0b0111111111;
					uint16_t count = rmap_value / 1024;

					//if (printandquit) {
					//	printf("L_y: %u  r_target hit: %u   pos_R:%u\n", l_y, r_target, pos_R);
					//}
					int num_matches = atomicAdd(&total_matches,1);//count);
					if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
						printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
					} else {
						Index_Match match = { };
						match.idxL = pos_L;
						match.idxR = pos_R;
						matches[num_matches] = match;


						// handle edge cases
						// TODO: let's push these into separate array
						// then test them later.
						if (count > 1) {
							int slot = atomicAdd(&y_duplicate_counts, 1);
							nick_rmap_extras_rl[slot] = (r_target << 16) + pos_L;
							// add the extras
							/*int extra_match = 0;
							for (int slot = 0; slot < num_extras; slot++) {
								if (nick_rmap_extras_ry[slot] == r_target) {
									uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
									match.idxR = extra_pos_R;//value >> 4;
									int num_matches = atomicAdd(&total_matches,1);
									matches[num_matches] = match;
									//extra_match++;
									//matches[num_matches+extra_match] = match;
									//if (doPrint > 1) {
									//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
									//}
								}
							}*/

							//if (global_kbc_L_bucket_id < 10) {
							//	if (extra_match != count-1) {
							//		printf("ERRORRRR! EXTRA MATCHES %u DOES NOT MATCH COUNT-1 %u\n", extra_match, count);
							//	} else {
							//		printf("BUCKET L %u SUCCESSFULLY ADDED EXTRA COUNTS %u\n", global_kbc_L_bucket_id, count);
							//	}
							//}
						}
					}
				}
			}
		}

	__syncthreads();

	// up until this point matching takes 976ms total for k32
	// it's 936ms with only the total matches counter (so about 40ms for appending match data)
	// 745ms with a bogus counter (so no shared atomic conflict)
	// it's 586ms with only m computations and bogus counter (no lookups) - so rmap lookups add 140ms
	// it's 128ms with only 1m -- so calculations are adding 460ms!!!
	// in summary:
	// -- 460ms : m loop calculations - moreso the actual m loop than the math inside!
	// -- 140ms : rmap lookups (bank conflict improvements possible)
	// -- 128ms : data reads
	//     - 66ms rmap setup
	//     - 62ms reading y values back in
	// --  40ms : match atomic shared counter (vs non atomic shared counter)
	//if (threadIdx.x == 0) {
	//	if (total_matches == 1342343) printf("bogus");
	//}
	//return;

	// do the extras

	//int num_matches = atomicAdd(&total_matches,num_extras); // warning can only let thread 0 do this otherwise all will add!
	for (int slot=threadIdx.x; slot<num_extras; slot+=blockDim.x) {
		for (int i = 0; i < y_duplicate_counts; i++) {
			uint32_t value = nick_rmap_extras_rl[i];
			uint16_t r_target = value >> 16;
			uint16_t pos_L = value & 0x0FFFF;
			if (nick_rmap_extras_ry[slot] == r_target) {
				uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
				Index_Match match = { };
				match.idxL = pos_L;
				match.idxR = extra_pos_R;
				int num_matches = atomicAdd(&total_matches,1);
				matches[num_matches] = match;
				//matches[total_matches+slot] = match;
				//if (doPrint > 1) {
				//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
				//}
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			//printf("Bucket L %u Total matches: %u   duplicate counts: %u non_dupes: %u\n", kbc_L_bucket_id, total_matches, duplicate_counts, non_duplicate_counts);
		}
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
	}

	__syncthreads();

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms

			blake_result = 23;
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);

			//if (global_kbc_L_bucket_id == 1) {
				//printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
			//}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
			//	uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
			//	uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
			//	printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			//}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		if (table < 6) {
			uint64_t batch_bucket = blake_result >> (38-6); // 27.52ms for 1/64 of kbcs
			//uint64_t batch_bucket = threadIdx.x % 64; // 25.3ms with blake computation, 20ms without. So blake adds 5ms for 1/64 of values;
			//uint64_t batch_bucket = 0; // 18ms per 1/64 of values, and our block counts aren't even optimized since global locking on atomic adds
			// so...in theory could reduce from 27ms time down to sub 18ms, and then do blake pass on seperate scan, which *should* be faster.
			// since we write less blocks/data in here
			const uint64_t block_mod = (uint64_t) 1 << (38-6);
			pair.y = (uint32_t) (blake_result % block_mod);
			int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
			uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
			//if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			//	printf("ERROR: results address overflow\n");
			//} else {
				// up to here takes 1508ms. Seems 1508-976 = 532ms for blake results
				// quite substantial!
				bucketed_out[pair_address] = pair;
				// including the write-out is 1696ms
			//}
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	if ((doPrint >=1) && (threadIdx.x == 0)) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

		if ((global_kbc_L_bucket_id % 1000000 == 0) || (global_kbc_L_bucket_id < 10)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);


	}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if ((doPrint >= 1) && (threadIdx.x == 0)) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_direct_to_host(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		char *host_criss_cross, int *out_bucket_counts) {
	// T1 match: 1714 ms -> with delaying extras: 1630
	//Total tables time: 73726 ms
	//        match: 10015 ms -> 9705ms with delaying extras
	const uint16_t NUM_RMAPS = (kBC/2)+1;
	__shared__ int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
	__shared__ uint32_t nick_rmap_extras_rl[32];
	__shared__ uint16_t nick_rmap_extras_ry[32];
	__shared__ uint16_t nick_rmap_extras_pos[32];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int num_extras;
	__shared__ int y_duplicate_counts;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	uint8_t doPrint = 1;

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
		num_extras = 0;
		y_duplicate_counts = 0;
		if (doPrint > 1) {
			printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
			if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
				printf("ERROR numL or numR > max entries\n");
				return;
			}
			if ((num_L == 0) || (num_R == 0) ) {
				printf("ERROR: numL and numR are 0\n");
				return;
			}
		}
	}
	// unfortunately to clear we have to do this
	for (int i = threadIdx.x; i < NUM_RMAPS; i += blockDim.x) {
		nick_rmap[i] = 0;
	}
	__syncthreads(); // all written initialize data should sync

	//bool printandquit = ((global_kbc_L_bucket_id == 75000));




	//	if (printandquit) {
			//printf("R_y list:\n");
			//for (size_t pos_R = 0; pos_R < num_R; pos_R++) {
			//	uint16_t r_y = kbc_R_entries[pos_R].y;
			//	printf("%u\n",r_y);
			//}
			//if (threadIdx.x == 0) {
			//	printf("L_y list num %u:\n", num_L);
			//	for (size_t pos_L = 0; pos_L < num_L; pos_L++) {
			//		uint16_t l_y = kbc_L_entries[pos_L].y;
			//		printf("%u\n",l_y);
			//	}
			//}
	//	}
	//__syncthreads();
	uint16_t parity = global_kbc_L_bucket_id % 2;

	for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		uint16_t r_y = R_entry.y;

		// r_y's share a block across two adjacent values, so kbc_map just works out which part it's in.
		int kbc_map = r_y / 2;
		const int kbc_box_shift = (r_y % 2) * 15;
		int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)

		int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
		rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;
		if (rmap_value == 0) {
			// if we added to an empty spot, what we do is add the pos_R here in the lower 9 bits of the box
			// and ONLY for this one.
			atomicAdd(&nick_rmap[kbc_map], (pos_R << kbc_box_shift));
			//if (printandquit) {
			//	printf("r_y: %u   pos:%u\n", r_y, pos_R);
			//}
		} else {
			// we hit duplicate entry...add this to a row
			int slot = atomicAdd(&num_extras, 1);
			nick_rmap_extras_ry[slot] = r_y;
			nick_rmap_extras_pos[slot] = pos_R;
		}

	}

	__syncthreads(); // wait for all threads to write r_bid entries

	// load parity tables into shared
	/*if (printandquit) {
		if (threadIdx.x == 0) {
			printf("num extras bucket %u : %u   parity: %u \n", global_kbc_L_bucket_id, num_extras, parity);

			for (int i=0;i<kBC;i++) {
				int kbc_map = i / 2;
				const int kbc_box_shift = (i % 2) * 15;
				int rmap_value = nick_rmap[kbc_map];
				rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

				//uint16_t rmap_value = nick_rmap[i];
				uint16_t pos = (rmap_value & 0b0111111111);
				if (rmap_value > 0) {
					printf("kbc:%i  value:%u pos:%u\n", i, rmap_value, pos);
				}
			}

		}

	}
	__syncthreads();*/

	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
		BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
		uint16_t l_y = L_entry.y;
		//printf("scanning for pos_L: %u\n", pos_L);

		for (int m=0;m<64;m++) {

			//uint16_t r_target = L_targets[parity][l_y][m]; // this performs so badly because this lookup
				// is super-inefficient.

			uint16_t indJ = l_y / kC;
			uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);

			// find which box our r_target is in, extra the 15bit value from that box
			int kbc_map = r_target / 2;
			const int kbc_box_shift = (r_target % 2) * 15;
			int rmap_value = nick_rmap[kbc_map];
			rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

			if (rmap_value > 0) {
				// the pos_R is the lower 9 bits of that 15bit boxed value
				uint16_t pos_R = rmap_value & 0b0111111111;
				uint16_t count = rmap_value / 1024;

				//if (printandquit) {
				//	printf("L_y: %u  r_target hit: %u   pos_R:%u\n", l_y, r_target, pos_R);
				//}
				int num_matches = atomicAdd(&total_matches,1);//count);
				if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
					printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
				} else {
					Index_Match match = { };
					match.idxL = pos_L;
					match.idxR = pos_R;
					matches[num_matches] = match;


					// handle edge cases
					// TODO: let's push these into separate array
					// then test them later.
					if (count > 1) {
						int slot = atomicAdd(&y_duplicate_counts, 1);
						nick_rmap_extras_rl[slot] = (r_target << 16) + pos_L;
						// add the extras
						/*int extra_match = 0;
						for (int slot = 0; slot < num_extras; slot++) {
							if (nick_rmap_extras_ry[slot] == r_target) {
								uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
								match.idxR = extra_pos_R;//value >> 4;
								int num_matches = atomicAdd(&total_matches,1);
								matches[num_matches] = match;
								//extra_match++;
								//matches[num_matches+extra_match] = match;
								//if (doPrint > 1) {
								//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
								//}
							}
						}*/
						//if (global_kbc_L_bucket_id < 10) {
						//	if (extra_match != count-1) {
						//		printf("ERRORRRR! EXTRA MATCHES %u DOES NOT MATCH COUNT-1 %u\n", extra_match, count);
						//	} else {
						//		printf("BUCKET L %u SUCCESSFULLY ADDED EXTRA COUNTS %u\n", global_kbc_L_bucket_id, count);
						//	}
						//}
					}
				}
			}
		}
	}

	__syncthreads();

	// do the extras

	//int num_matches = atomicAdd(&total_matches,num_extras); // warning can only let thread 0 do this otherwise all will add!
	for (int slot=threadIdx.x; slot<num_extras; slot+=blockDim.x) {
		for (int i = 0; i < y_duplicate_counts; i++) {
			uint32_t value = nick_rmap_extras_rl[i];
			uint16_t r_target = value >> 16;
			uint16_t pos_L = value & 0x0FFFF;
			if (nick_rmap_extras_ry[slot] == r_target) {
				uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
				Index_Match match = { };
				match.idxL = pos_L;
				match.idxR = extra_pos_R;
				int num_matches = atomicAdd(&total_matches,1);
				matches[num_matches] = match;
				//matches[total_matches+slot] = match;
				//if (doPrint > 1) {
				//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
				//}
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			//printf("Bucket L %u Total matches: %u   duplicate counts: %u non_dupes: %u\n", kbc_L_bucket_id, total_matches, duplicate_counts, non_duplicate_counts);
		}
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
	}

	__syncthreads();

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
				//printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
			//}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
			//	uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
			//	uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
			//	printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			//}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		if (table < 6) {
			uint64_t batch_bucket = blake_result >> (38-6);
			const uint64_t block_mod = (uint64_t) 1 << (38-6);
			pair.y = (uint32_t) (blake_result % block_mod);
			int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);

			uint64_t criss_cross_id;
			uint64_t cross_row_id = batch_id;
			uint64_t cross_column_id = batch_bucket;
			if ((table % 2) == 1) {
				criss_cross_id = (cross_row_id * BATCHES  + cross_column_id);
			} else {
			  	criss_cross_id = (cross_column_id * BATCHES  + cross_row_id);
			}
			uint64_t host_block_entry_start_position = criss_cross_id * HOST_MAX_BLOCK_ENTRIES;
			uint64_t host_bytes_start = host_block_entry_start_position * HOST_UNIT_BYTES;

			BUCKETED_ENTRY_OUT *host_block = (BUCKETED_ENTRY_OUT *) &host_criss_cross[host_bytes_start];
			host_block[block_slot] = pair;
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	if (threadIdx.x == 0) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);
		if ((global_kbc_L_bucket_id % 1000000 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

	}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if (threadIdx.x == 0) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_with_backref(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out,
		char *bucketed_ref_out, int *out_bucket_counts) {
	// T1 match: 1714 ms -> with delaying extras: 1630
	//Total tables time: 73726 ms
	//        match: 10015 ms -> 9705ms with delaying extras
	const uint16_t NUM_RMAPS = (kBC/2)+1;
	__shared__ int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
	__shared__ uint32_t nick_rmap_extras_rl[32];
	__shared__ uint16_t nick_rmap_extras_ry[32];
	__shared__ uint16_t nick_rmap_extras_pos[32];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int num_extras;
	__shared__ int y_duplicate_counts;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	uint8_t doPrint = 1;

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
		num_extras = 0;
		y_duplicate_counts = 0;
		if (doPrint > 1) {
			printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
			if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
				printf("ERROR numL or numR > max entries\n");
				return;
			}
			if ((num_L == 0) || (num_R == 0) ) {
				printf("ERROR: numL and numR are 0\n");
				return;
			}
		}
	}
	// unfortunately to clear we have to do this
	for (int i = threadIdx.x; i < NUM_RMAPS; i += blockDim.x) {
		nick_rmap[i] = 0;
	}
	__syncthreads(); // all written initialize data should sync

	//bool printandquit = ((global_kbc_L_bucket_id == 75000));




	//	if (printandquit) {
			//printf("R_y list:\n");
			//for (size_t pos_R = 0; pos_R < num_R; pos_R++) {
			//	uint16_t r_y = kbc_R_entries[pos_R].y;
			//	printf("%u\n",r_y);
			//}
			//if (threadIdx.x == 0) {
			//	printf("L_y list num %u:\n", num_L);
			//	for (size_t pos_L = 0; pos_L < num_L; pos_L++) {
			//		uint16_t l_y = kbc_L_entries[pos_L].y;
			//		printf("%u\n",l_y);
			//	}
			//}
	//	}
	//__syncthreads();
	uint16_t parity = global_kbc_L_bucket_id % 2;

	for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		uint16_t r_y = R_entry.y;

		// r_y's share a block across two adjacent values, so kbc_map just works out which part it's in.
		int kbc_map = r_y / 2;
		const int kbc_box_shift = (r_y % 2) * 15;
		int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)

		int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
		rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;
		if (rmap_value == 0) {
			// if we added to an empty spot, what we do is add the pos_R here in the lower 9 bits of the box
			// and ONLY for this one.
			atomicAdd(&nick_rmap[kbc_map], (pos_R << kbc_box_shift));
			//if (printandquit) {
			//	printf("r_y: %u   pos:%u\n", r_y, pos_R);
			//}
		} else {
			// we hit duplicate entry...add this to a row
			int slot = atomicAdd(&num_extras, 1);
			nick_rmap_extras_ry[slot] = r_y;
			nick_rmap_extras_pos[slot] = pos_R;
		}

	}

	__syncthreads(); // wait for all threads to write r_bid entries

	// load parity tables into shared
	/*if (printandquit) {
		if (threadIdx.x == 0) {
			printf("num extras bucket %u : %u   parity: %u \n", global_kbc_L_bucket_id, num_extras, parity);

			for (int i=0;i<kBC;i++) {
				int kbc_map = i / 2;
				const int kbc_box_shift = (i % 2) * 15;
				int rmap_value = nick_rmap[kbc_map];
				rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

				//uint16_t rmap_value = nick_rmap[i];
				uint16_t pos = (rmap_value & 0b0111111111);
				if (rmap_value > 0) {
					printf("kbc:%i  value:%u pos:%u\n", i, rmap_value, pos);
				}
			}

		}

	}
	__syncthreads();*/

	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
		BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
		uint16_t l_y = L_entry.y;
		//printf("scanning for pos_L: %u\n", pos_L);

		for (int m=0;m<64;m++) {

			//uint16_t r_target = L_targets[parity][l_y][m]; // this performs so badly because this lookup
				// is super-inefficient.

			uint16_t indJ = l_y / kC;
			uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);

			// find which box our r_target is in, extra the 15bit value from that box
			int kbc_map = r_target / 2;
			const int kbc_box_shift = (r_target % 2) * 15;
			int rmap_value = nick_rmap[kbc_map];
			rmap_value = (rmap_value >> kbc_box_shift) & 0b0111111111111111;

			if (rmap_value > 0) {
				// the pos_R is the lower 9 bits of that 15bit boxed value
				uint16_t pos_R = rmap_value & 0b0111111111;
				uint16_t count = rmap_value / 1024;

				//if (printandquit) {
				//	printf("L_y: %u  r_target hit: %u   pos_R:%u\n", l_y, r_target, pos_R);
				//}
				int num_matches = atomicAdd(&total_matches,1);//count);
				if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
					printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
				} else {
					Index_Match match = { };
					match.idxL = pos_L;
					match.idxR = pos_R;
					matches[num_matches] = match;


					// handle edge cases
					// TODO: let's push these into separate array
					// then test them later.
					if (count > 1) {
						int slot = atomicAdd(&y_duplicate_counts, 1);
						nick_rmap_extras_rl[slot] = (r_target << 16) + pos_L;
						// add the extras
						/*int extra_match = 0;
						for (int slot = 0; slot < num_extras; slot++) {
							if (nick_rmap_extras_ry[slot] == r_target) {
								uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
								match.idxR = extra_pos_R;//value >> 4;
								int num_matches = atomicAdd(&total_matches,1);
								matches[num_matches] = match;
								//extra_match++;
								//matches[num_matches+extra_match] = match;
								//if (doPrint > 1) {
								//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
								//}
							}
						}*/
						//if (global_kbc_L_bucket_id < 10) {
						//	if (extra_match != count-1) {
						//		printf("ERRORRRR! EXTRA MATCHES %u DOES NOT MATCH COUNT-1 %u\n", extra_match, count);
						//	} else {
						//		printf("BUCKET L %u SUCCESSFULLY ADDED EXTRA COUNTS %u\n", global_kbc_L_bucket_id, count);
						//	}
						//}
					}
				}
			}
		}
	}

	__syncthreads();

	// do the extras

	//int num_matches = atomicAdd(&total_matches,num_extras); // warning can only let thread 0 do this otherwise all will add!
	for (int slot=threadIdx.x; slot<num_extras; slot+=blockDim.x) {
		for (int i = 0; i < y_duplicate_counts; i++) {
			uint32_t value = nick_rmap_extras_rl[i];
			uint16_t r_target = value >> 16;
			uint16_t pos_L = value & 0x0FFFF;
			if (nick_rmap_extras_ry[slot] == r_target) {
				uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
				Index_Match match = { };
				match.idxL = pos_L;
				match.idxR = extra_pos_R;
				int num_matches = atomicAdd(&total_matches,1);
				matches[num_matches] = match;
				//matches[total_matches+slot] = match;
				//if (doPrint > 1) {
				//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
				//}
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			//printf("Bucket L %u Total matches: %u   duplicate counts: %u non_dupes: %u\n", kbc_L_bucket_id, total_matches, duplicate_counts, non_duplicate_counts);
		}
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
	}

	__syncthreads();

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
				//printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
			//}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
			//	uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
			//	uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
			//	printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			//}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		//printf("table %u blake result: %llu\n", table, blake_result);
		uint64_t batch_bucket = blake_result >> (38-6);
		const uint64_t block_mod = (uint64_t) 1 << (38-6);
		int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
		uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
		if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			printf("ERROR: results address overflow\n");
		} else {
			if (table < 6) {
				// our last table 6 doesn't write into hostmem criss cross, it just does backref with extra y instead.
				pair.y = (uint32_t) (blake_result % block_mod);
				bucketed_out[pair_address] = pair;
			}
		}

		//// TODO: export Lx's to save into table, these are x1,x3 denoting 2 pairs that can be compressed into kbc buckets
		// we *could* do double the data in table 3, but then we need extra buffers and memory that we don't have
		if (table == 2) {
			// this task can be left to the CPU to deal with the batch buckets and write baseref to file.
		}
		if ((table == 3) || (table == 4) || (table == 5) || (table == 6)) {

			if (table == 6) {
				// last table does backref with extra y truncated to most significant k bits.
				T6BackRef ref = {};
				ref.prev_block_ref_L = L_Entry.blockposref;
				ref.prev_block_ref_R = R_Entry.blockposref;
				ref.y = (uint32_t) (blake_result >> kExtraBits); // get top 32 most significant bits, since calc_y is 38 bits.
				//printf("blake y result table 6: %llu  -> %u\n", blake_result, ref.y);

				T6BackRef *out = (T6BackRef *) bucketed_ref_out;
				//if ((ref.prev_block_ref_L == 0) && (ref.prev_block_ref_R == 0)) {
				//	printf("Both refs are 0!\n");
				//}
				out[pair_address] = ref;
			} else if (table == 3) {
				T3BaseRef ref = {};
				ref.Lx1 = L_Entry.meta[0];
				ref.Lx2 = L_Entry.meta[2];
				ref.Lx3 = R_Entry.meta[0];
				ref.Lx4 = R_Entry.meta[2];
				T3BaseRef *out = (T3BaseRef *) bucketed_ref_out;
				out[pair_address] = ref;
			} else if ((table == 3) || (table == 4) || (table == 5)) {
				BackRef ref = {};
				ref.prev_block_ref_L = L_Entry.blockposref;
				ref.prev_block_ref_R = R_Entry.blockposref;
				BackRef *out = (BackRef *) bucketed_ref_out;
				//if ((ref.prev_block_ref_L == 0) && (ref.prev_block_ref_R == 0)) {
				//	printf("Both refs are 0!\n");
				//}
				out[pair_address] = ref;
			}
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	if (threadIdx.x == 0) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);
		if ((global_kbc_L_bucket_id % 1000000 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

	}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if (threadIdx.x == 0) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_rmap_working(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {
	//  match: 10000 ms
	//  table 1 match match: 1633 ms, potentially 2.5x faster than orig method
	//          with extras: 1841 ms - win!
	//     with extras hashed counters (working): 2144 ms
	// Total tables time: 77112 ms
	//        match: 12505 ms
	// TODO: TRY THIS AS GLOBAL MEMORY COVERING BATCH SIZE
	//__shared__ __half nick_rmap_counts[kBC]; // 30226 bytes
	const int RMAP_NUM_COUNTS_PER_BOX = 8; // whether 8  per box, 7, 4, bit counts 4 etc doesn't change result measurably I don't think.
	const int RMAP_BITS_FOR_COUNTS = 4;
	const int RMAP_COUNT_MASK = 0b01111;
	const int NUM_RMAP_COUNTS = (15113 / RMAP_NUM_COUNTS_PER_BOX)+1;
	__shared__ int nick_rmap_counts[NUM_RMAP_COUNTS]; // kBC / 2, sharing bits [12bits pos, 3 bits counter][12 bits pos, 3 bits counter]
	//__shared__ int16_t nick_rmap_counts[kBC]; // 30226 bytes
	__shared__ uint16_t nick_rmap_positions[kBC];
	__shared__ uint16_t nick_rmap_extras_ry[100];
	__shared__ uint16_t nick_rmap_extras_pos[100];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int num_extras;

	//__shared__ int non_duplicate_counts;
	//__shared__ int duplicate_counts;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	//if (global_kbc_L_bucket_id > 0) {
	//	return;
	//}

	uint8_t doPrint = 1;//(global_kbc_L_bucket_id < 10) ? 1 : 0; // start_kbc_L > 0 ? 1: 0; // 0 is none, 1 is basic, 2 is detailed
	//if (global_kbc_L_bucket_id == 75000) {
	//	doPrint = 100;
	//}

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
		num_extras = 0;
		//non_duplicate_counts = 0;
		//duplicate_counts = 0;
		if (doPrint > 1) {
			printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
			if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
				printf("ERROR numL or numR > max entries\n");
				return;
			}
			if ((num_L == 0) || (num_R == 0) ) {
				printf("ERROR: numL and numR are 0\n");
				return;
			}
		}
	}
	// unfortunately to clear we have to do this 236 times for 64 threads
	for (int i = threadIdx.x; i < NUM_RMAP_COUNTS; i += blockDim.x) {
		nick_rmap_counts[i] = 0;
	}
	__syncthreads(); // all written initialize data should sync

	//bool printandquit = ((global_kbc_L_bucket_id == 75000));




	//	if (printandquit) {
			//printf("R_y list:\n");
			//for (size_t pos_R = 0; pos_R < num_R; pos_R++) {
			//	uint16_t r_y = kbc_R_entries[pos_R].y;
			//	printf("%u\n",r_y);
			//}
			//if (threadIdx.x == 0) {
			//	printf("L_y list num %u:\n", num_L);
			//	for (size_t pos_L = 0; pos_L < num_L; pos_L++) {
			//		uint16_t l_y = kbc_L_entries[pos_L].y;
			//		printf("%u\n",l_y);
			//	}
			//}
	//	}
	//__syncthreads();
	uint16_t parity = global_kbc_L_bucket_id % 2;

	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		uint16_t r_y = R_entry.y;
		//int16_t rmap_value = nick_rmap_counts[r_y];
		//uint8_t rmap_count = rmap_value & 0b0111;

		// TODO: ok, let's make it MUCH easier, and have the atomic adds on 3 bits only
		// and cut kbc_map into 15 bit counts (5 counts) each. Gives us plenty of space now
		// to have separate rmap_positions entries, and greaty simplifies code (hopefully).
		// however...may be slower!
		//int kbc_map = r_y / 2;
		//const int kbc_box_shift = (r_y % 2) * 12;
		//int add = 1 << kbc_box_shift;
		//int rmap_value = atomicAdd(&nick_rmap_counts[kbc_map],add);
		//rmap_value = (rmap_value >> kbc_box_shift) & 0x0000FFFF;
		//int rmap_count = rmap_value & 0b0111;

		int kbc_map = r_y / RMAP_NUM_COUNTS_PER_BOX;
		const int kbc_box_shift = (r_y % RMAP_NUM_COUNTS_PER_BOX) * RMAP_BITS_FOR_COUNTS; // 3 bits each, gives up to 111 = 7 duplicates

		int add = 1 << kbc_box_shift;
		int rmap_value = atomicAdd(&nick_rmap_counts[kbc_map],add);
		int rmap_count = (rmap_value >> kbc_box_shift) & RMAP_COUNT_MASK;

		if (rmap_count == 0) {
			nick_rmap_positions[r_y] = pos_R;
			//int add_value = (pos_R << 3) << kbc_box_shift;
			//atomicAdd(&nick_rmap_counts[kbc_map], add_value);
			//int16_t new_value = atomicAdd(&nick_rmap_counts[r_y], add_value); // encode position
			//if ((printandquit) && (r_y == 1725)) {
			//	nick_rmap_counts[r_y] = add + 1;
				//unsigned short prev = atomicAdd(&nick_rmap_counts[r_y],add);
				//printf("***** add value is: %u  prev:%u\n", add, prev);
				//prev = atomicAdd(&nick_rmap_counts[r_y],1);
				//printf("***** add value is: %u  prev:%u\n", add, prev);
			//}
			//nick_rmap_counts[r_y] = 1 + (pos_R << 3);
		} else {
			// we hit duplicate entry...
			int slot = atomicAdd(&num_extras, 1);
			nick_rmap_extras_ry[slot] = r_y;
			nick_rmap_extras_pos[slot] = pos_R;
		}
	}

	__syncthreads(); // wait for all threads to write r_bid entries

	// load parity tables into shared
	/*if (doPrint > 1) {
		if (threadIdx.x == 0) {
			printf("num extras bucket %u : %u   parity: %u \n", global_kbc_L_bucket_id, num_extras, parity);
			if (printandquit) {
				for (int i=1700;i<1750;i++) {
					//unsigned short value = nick_rmap_counts[i];
					//unsigned short count = value & 0b0111;
					//printf("kbc:%u  value:%u count:%u\n", i, value, count);

					int kbc_map = i / 2;
					int kbc_box_shift = (i % 2) * 12;
					int rmap_value = (nick_rmap_counts[kbc_map]) >> kbc_box_shift;
					int rmap_count = rmap_value & (0b0111);
					int pos = (rmap_value & 0b0111111111000) >> 3;
					printf("kbc:%i  value:%u count:%u pos:%u\n", i, rmap_value, rmap_count,pos);
				}
			}
		}

	}
	__syncthreads();*/

	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
		BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];
		uint16_t l_y = L_entry.y;
		//printf("scanning for pos_L: %u\n", pos_L);

		for (int m=0;m<64;m++) {

			//uint16_t r_target = L_targets[parity][l_y][m]; // this performs so badly because this lookup
				// is super-inefficient.

			uint16_t indJ = l_y / kC;
			uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);

			//if (r_target != r_target_calc) {
			//	printf("CALC ERROR r_target calc %u does not match r_target %u\n", r_target_calc, r_target);
			//}

			//uint16_t value = nick_rmap[r_target];
			//uint8_t count = value & 0x000F;
			//__half value = nick_rmap_counts[r_target];
			//int16_t value = nick_rmap_counts[r_target];
			//unsigned short value = nick_rmap_counts[r_target];
			//unsigned short count = value & 0b0111;

			//int kbc_map = r_target / 2;
			//int kbc_box_shift = (r_target % 2) * 12;
			//int value = (nick_rmap_counts[kbc_map] >> kbc_box_shift) & 0x0000FFFF;
			//int count = value & (0b0111);

			const int kbc_map = r_target / RMAP_NUM_COUNTS_PER_BOX;
			const int kbc_box_shift = (r_target % RMAP_NUM_COUNTS_PER_BOX) * RMAP_BITS_FOR_COUNTS; // 3 bits each.

			int rmap_value = nick_rmap_counts[kbc_map];
			int count = (rmap_value >> kbc_box_shift) & RMAP_COUNT_MASK;

			//if ((printandquit) && (l_y == 13414)) {
				// bool superdebug = l_y == 13414  r_target hit: 1725
			//	printf("  m: %u   r_target: %u   count:%u\n", m, r_target, count);
			//}
			if (count > 0) {
				//uint16_t pos_R = value >> 3;
				uint16_t pos_R = nick_rmap_positions[r_target];
				//if (printandquit) {
				//	printf("L_y: %u  r_target hit: %u\n", l_y, r_target);
				//}
				//printf("      has match\n");
				int num_matches = atomicAdd(&total_matches,1);
				if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
					printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
				} else {
					Index_Match match = { };
					match.idxL = pos_L;
					match.idxR = pos_R;//nick_rmap_positions[r_target];//value >> 4;
					matches[num_matches] = match;
					//atomicAdd(&non_duplicate_counts,1);

					// handle edge cases
					// TODO: let's push these into separate array
					// then test them later.
					if (count > 1) {
						// add the extras
						//int extra_match = 0;
						for (int slot = 0; slot < num_extras; slot++) {
							if (nick_rmap_extras_ry[slot] == r_target) {
								uint16_t extra_pos_R = nick_rmap_extras_pos[slot];
								match.idxR = extra_pos_R;//value >> 4;
								int num_matches = atomicAdd(&total_matches,1);
								matches[num_matches] = match;
								//extra_match++;
								//matches[num_matches+extra_match] = match;
								//atomicAdd(&duplicate_counts,1);
								//if (doPrint > 1) {
								//	printf("Collected extra match pos_R: %u from r_y: %u in slot:%u \n", extra_pos_R, r_target, slot);
								//}
							}
						}
						//if (global_kbc_L_bucket_id < 10) {
						//	if (extra_match != count-1) {
						//		printf("ERRORRRR! EXTRA MATCHES %u DOES NOT MATCH COUNT-1 %u\n", extra_match, count);
						//	} else {
						//		printf("BUCKET L %u SUCCESSFULLY ADDED EXTRA COUNTS %u\n", global_kbc_L_bucket_id, count);
						//	}
						//}
					}
				}
			}
		}
	}

	__syncthreads();


	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			//printf("Bucket L %u Total matches: %u   duplicate counts: %u non_dupes: %u\n", kbc_L_bucket_id, total_matches, duplicate_counts, non_duplicate_counts);
		}
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
	}

	__syncthreads();

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			if (global_kbc_L_bucket_id == 1) {
				//if ((calc_y == 21557) && (L_Entry.meta[0] == 3620724289) && (R_Entry.meta[0] == 2663198278)) {
					printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
					//Ly is:[20932] Lx: [322482289] Rx: [3382886636]  f result:[273114646565]
					//if (blake_result == 56477140042) {
					//	printf(" ---** BLAKE CORRECT **\n");
					//} else {
					//	printf(" ---** BLAKE WRONG :(((( \n");
					//}
					// Ly is:[21557] Lx: [3620724289] Rx: [2663198278]  f result:[56477140042]
				//}
			}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			if (global_kbc_L_bucket_id == 1) {
				uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
				uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
				printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		uint64_t batch_bucket = blake_result >> (38-6);
		const uint64_t block_mod = (uint64_t) 1 << (38-6);
		pair.y = (uint32_t) (blake_result % block_mod);
		int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
		uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
		if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			printf("ERROR: results address overflow\n");
		} else {
			bucketed_out[pair_address] = pair;
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	if (threadIdx.x == 0) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);
		if ((global_kbc_L_bucket_id % 25000 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

	}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if (threadIdx.x == 0) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
}


template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_find_tx_matches_orig(uint16_t table, uint32_t batch_id, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {
	const uint16_t MAX_BIDS = 16;
	__shared__ uint16_t R_bids[kC*MAX_BIDS]; // kC is 127, this gives us 127*8 * 2 bytes = 2kb
	__shared__ int R_bids_count[kC]; // size 127 bytes
	__shared__ int R_bid_positions[kC*MAX_BIDS];//RBid_Entry R_bid_entries[kC*MAX_BIDS]; // size 127 * 8 * 6 bytes = 6kb
	__shared__ uint8_t matching_shifts_c[64]; // 128 bytes
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	//*********************
	//Total tables time: 86822 ms
	//        match: 22397 ms
	//      phase 1: 3930ms
	//__shared__ Bucketed_kBC_Entry kbc_L_entries[400]; // will copy global to here, unfortunately not faster :(
	//__shared__ Bucketed_kBC_Entry kbc_R_entries[400];

	//end_kbc_R = end_kbc_R - start_kbc_L;
	//start_kbc_L = 0;
	//if (threadIdx.x == 0) {
	//	printf("doing block inside kernel %u\n", start_kbc_L);
	//}

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	// doPrint 1 = end matches and bucket counts, 2 = a little debug, 3 = lots.
	const uint8_t doPrint = 1;//(global_kbc_L_bucket_id < 10) ? 1 : 0; // start_kbc_L > 0 ? 1: 0; // 0 is none, 1 is basic, 2 is detailed



	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	//printf("threadId: %u  startScan: %u skipScan: %u", threadId, threadStartScan, threadSkipScan);
	if (threadIdx.x == 0) {
		// only do this once, should be in constant memory
		/*for (uint16_t parity = 0; parity < 2; parity++) {
			for (uint16_t r = 0; r < 64; r++) {
				uint16_t v = ((2 * r + parity) * (2 * r + parity)) % kC;
				matching_shifts_c[parity][r] = v;
				//printf("matching shifts %u %u = %u\n", parity, r, v);
			}
		}*/
		total_matches = 0;
	}

	uint16_t max_bids_found = 0;

	//const uint32_t start_L = kbc_start_addresses[kbc_L_bucket_id];
	//const uint32_t start_R = kbc_start_addresses[kbc_R_bucket_id];
	//const int num_L = start_R - start_L;
	//const int num_R = (start_R < kBC_NUM_BUCKETS) ? kbc_start_addresses[kbc_R_bucket_id+1] - start_R : total_entries_count - start_R;
	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries[(kbc_L_bucket_id+1)];

	if (threadIdx.x == 0) {
		if (doPrint > 1) printf("find matches global kbc bucket L: %u local_b_id:%u num_L %u num_R %u\n", global_kbc_L_bucket_id, kbc_L_bucket_id, num_L, num_R);
		if ((num_L >= KBC_MAX_ENTRIES_PER_BUCKET) || (num_R >= KBC_MAX_ENTRIES_PER_BUCKET)) {
			printf("ERROR numL or numR > max entries\n");
			return;
		}
		if ((num_L == 0) || (num_R == 0) ) {
			printf("ERROR: numL and numR are 0\n");
			return;
		}
	}

	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries[start_R];

	uint16_t parity = global_kbc_L_bucket_id % 2;
	for (int r = threadIdx.x; r < 64; r += blockDim.x) {
		uint16_t v = ((2 * r + parity) * (2 * r + parity)) % kC;
		matching_shifts_c[r] = v; // this is a wash...doesn't save much if anything
	}
	for (int i = threadIdx.x; i < kC; i += blockDim.x) {
		R_bids_count[i] = 0;
	}

	__syncthreads(); // all written initialize data should sync



	//Bucketed_kBC_Entry L_entry = kbc_local_entries[0];
	BUCKETED_ENTRY_IN temp_entry = kbc_L_entries[0];

	//uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(temp_entry, global_kbc_L_bucket_id);
	//uint16_t parity = (calc_y / kBC) % 2;


	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		//Bucketed_kBC_Entry R_entry = kbc_local_entries[MAX_KBC_ENTRIES+pos_R];
		BUCKETED_ENTRY_IN R_entry = kbc_R_entries[pos_R];
		//global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;
		//calc_y = CALC_Y_BUCKETED_KBC_ENTRY(R_entry, global_kbc_L_bucket_id+1);
		uint16_t y_kC = R_entry.y % kC; // should be same as calc_y % kC ?
		uint16_t y_mod_kBC_div_kC = R_entry.y / kC; // should be same as R_entry.y / kC


		int num_bids = atomicAdd(&R_bids_count[y_kC],1);
		if (num_bids >= MAX_BIDS) {
			printf("ERROR KBC LOCAL MAX BIDS EXCEEDED %u in global bucket %u\n", num_bids, global_kbc_L_bucket_id);
			//printf("\nR_entry y:%u  meta[0]:%u  y_kC:%u  y_mod_kBC_div_kC: %u  into slot: %u\n ", R_entry.y, R_entry.meta[0], y_kC, y_mod_kBC_div_kC, num_bids);
		} else {
			// uint8_t num_bids = R_bids_count[y_kC]++;
			R_bids[y_kC*MAX_BIDS + num_bids] = y_mod_kBC_div_kC;
			//R_bid_entries[y_kC*MAX_BIDS + num_bids].x = R_entry.x;
			R_bid_positions[y_kC*MAX_BIDS + num_bids] = pos_R;
		}

		//if (doPrint>2) printf("R_entry x:%u  y:%u  y_kC:%u  y_mod_kBC_div_kC: %u  into slot: %u\n ", R_entry.x, R_entry.y, y_kC, y_mod_kBC_div_kC, num_bids);


		if (max_bids_found > num_bids) {
			max_bids_found = num_bids;
		}
	}


	__syncthreads(); // wait for all threads to write r_bid entries

	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
		BUCKETED_ENTRY_IN L_entry = kbc_L_entries[pos_L];

		//if (doPrint>2) printf("CHECKING pos_L:%u entry x:%u for match\n", pos_L, L_entry.x);
		uint16_t yl_bid = L_entry.y / kC;
		uint16_t yl_cid = L_entry.y % kC;

		for (uint8_t m = 0; m < 64; m++) {
			uint16_t target_bid = (yl_bid + m);
			// TODO: benchmark if matching_shifts array is actually faster...doubt it.
			uint16_t target_cid = yl_cid + matching_shifts_c[m]; // turns out it's a wash
			//uint16_t target_cid = yl_cid + ((2 * m + parity) * (2 * m + parity)) % kC;

			// This is faster than %
			if (target_bid >= kB) {
				target_bid -= kB;
			}
			if (target_cid >= kC) { // check if rid of %k on = part above.
				target_cid -= kC;
			}

			uint16_t num_bids = R_bids_count[target_cid];
			if (num_bids > MAX_BIDS) {
				printf("PRUNING NUM BIDS FROM %u TO %u", num_bids, MAX_BIDS);
				num_bids = MAX_BIDS;
			}
			// this inner loop is inefficient as num bids can vary...maybe push into list?
			for (uint32_t i = 0; i < num_bids; i++) {
				uint16_t R_bid = R_bids[target_cid*MAX_BIDS + i];

				if (target_bid == R_bid) {
					int pos_R = R_bid_positions[target_cid*MAX_BIDS + i];
					int num_matches = atomicAdd(&total_matches,1);
					if (num_matches >= KBC_MAX_ENTRIES_PER_BUCKET) {
						printf("PRUNED: exceeded matches allowed per bucket MAX:%u current:%u\n", KBC_MAX_ENTRIES_PER_BUCKET, num_matches);
					} else {
						Index_Match match = { };
						match.idxL = pos_L;
						match.idxR = pos_R;
						matches[num_matches] = match;
					}
					//if (doPrint>2) {
					//	printf("Thread %u pos_L:%u Match #%u found Lx:%u, Rx:%u\n", threadId, pos_L, num_matches, L_entry.x, R_entry.x);
					//}
					//printf("          Match found Lx:%u, Rx:%u\n", match.Lx, match.Rx);
				}
			}
		}

	}

	__syncthreads();


	if (threadIdx.x == 0) {
		if (doPrint>1) {
			// only do this once, should be in constant memory
			//if (doPrint>2) {
			//	printf("match list\n");
			//	for (int i=0;i<total_matches;i++) {
			//		Index_Match match = matches[i];
			//		printf("match %u = Lx %u   Rx %u   y %u\n", i, match.Lx, match.Rx, match.y);
			//	}
			//}
			printf("Bucket L %u Total matches: %u\n", kbc_L_bucket_id, total_matches);
		}
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
	}

	__syncthreads();

	/*if ((global_kbc_L_bucket_id == 0) && (threadIdx.x == 0)) {

		printf("Bucket match calc verification bucket %u num_matches: %u", global_kbc_L_bucket_id, total_matches);
		for (int i=0;i < total_matches;i++) {
			Index_Match match = matches[i];
			BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
			BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];

			printf("L_Entry y %u   R_Entry y %u\n", L_Entry.y, R_Entry.y);
			int16_t yr_kbc = R_Entry.y;
			int16_t yr_bid = yr_kbc / kC; // values [0..kB]
			int16_t yl_kbc = L_Entry.y;
			int16_t yl_bid = yl_kbc / kC; // values [0..kB]
			int16_t formula_one = yr_bid - yl_bid; // this should actually give m
			if (formula_one < 0) {
				formula_one += kB;
			}
			int16_t m = formula_one;
			if (m >= kB) {
				m -= kB;
			}
			printf("     m value calc: %u\n", m);
			if (m < 64) {
				// passed first test
				int16_t yl_cid = yl_kbc % kC; // % kBC % kC = %kC since kBC perfectly divisible by kC
				int16_t yr_cid = yr_kbc % kC;
				int16_t parity = (global_kbc_L_bucket_id) % 2;
				int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC; // values [0..127]
				int16_t formula_two = yr_cid - yl_cid;
				if (formula_two < 0) {
					formula_two += kC;
				}
				printf("     formula two %u   <-> %u  m2_parity %u\n", formula_two, m2_parity_squared);
				if (formula_two == m2_parity_squared) {
					// we have a match.
					printf("       MATCH OK\n");
				} else {
					printf("      FAILED TO MATCH\n");
				}
			}
		}

	}*/

	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match match = matches[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_L_entries[match.idxL];
		BUCKETED_ENTRY_IN R_Entry = kbc_R_entries[match.idxR];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, global_kbc_L_bucket_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			if (global_kbc_L_bucket_id == 1) {
				//if ((calc_y == 21557) && (L_Entry.meta[0] == 3620724289) && (R_Entry.meta[0] == 2663198278)) {
					printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
					//Ly is:[20932] Lx: [322482289] Rx: [3382886636]  f result:[273114646565]
					//if (blake_result == 56477140042) {
					//	printf(" ---** BLAKE CORRECT **\n");
					//} else {
					//	printf(" ---** BLAKE WRONG :(((( \n");
					//}
					// Ly is:[21557] Lx: [3620724289] Rx: [2663198278]  f result:[56477140042]
				//}
			}

		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			if (global_kbc_L_bucket_id == 1) {
				uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
				uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
				printf("Got y %llu idxL:%u idxR:%u Lx: %llu Rx: %llu and f_result: %llu\n", calc_y, match.idxL, match.idxR, Lx, Rx, blake_result);
			}
		} else if (table == 3) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 4, pair.meta);
		} else if (table == 4) {
			const uint32_t meta[8] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2], L_Entry.meta[3],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2], R_Entry.meta[3]
			};
			nick_blake3(meta, 8, calc_y, &blake_result, 3, pair.meta);
		} else if (table == 5) {
			const uint32_t meta[6] = {
					L_Entry.meta[0], L_Entry.meta[1], L_Entry.meta[2],
					R_Entry.meta[0], R_Entry.meta[1], R_Entry.meta[2],
			};
			nick_blake3(meta, 6, calc_y, &blake_result, 2, pair.meta);
		} else if (table == 6) {
			const uint32_t meta[4] = {
					L_Entry.meta[0], L_Entry.meta[1],
					R_Entry.meta[0], R_Entry.meta[1]
			};
			nick_blake3(meta, 4, calc_y, &blake_result, 0, NULL);
		}
		uint64_t batch_bucket = blake_result >> (38-6);
		const uint64_t block_mod = (uint64_t) 1 << (38-6);
		pair.y = (uint32_t) (blake_result % block_mod);
		int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
		uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
		if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			printf("ERROR: results address overflow\n");
		} else {
			bucketed_out[pair_address] = pair;
		}

		// do we have a double bucket to write into?
		//uint32_t double_bucket_id = 0;
		//uint32_t kbc_bucket_id = blake_result / kBC;
		//uint64_t batch_bucket_min_kbc = (batch_bucket << 32) / kBC;
		//uint64_t batch_bucket_max_kbc = ((batch_bucket+1) << 32) / kBC;
		//if (kbc_bucket_id == batch_bucket_min_kbc) {
		//	double_bucket_id = batch_bucket - 1;
		//} else if (kbc_bucket_id == batch_bucket_max_kbc) {
		//	double_bucket_id = batch_bucket + 1;
		//}
	}

	if (threadIdx.x == 0) {
		//if ((doPrint > 0) && (global_kbc_L_bucket_id < 10 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);
		if ((global_kbc_L_bucket_id % 25000 == 0)) printf(" matches kbc bucket: %u num_L:%u num_R:%u pairs:%u\n", global_kbc_L_bucket_id, num_L, num_R, total_matches);

	}
	/*
	kBC bucket id: 0 L entries: 222 R entries: 242 matches: 219
	 kBC bucket id: 1 L entries: 242 R entries: 257 matches: 248
	 kBC bucket id: 2 L entries: 257 R entries: 204 matches: 222
	 kBC bucket id: 3 L entries: 204 R entries: 243 matches: 185
	Total matches: 4294859632

	 Computing table 3
	Bucket 0 uniform sort. Ram: 7.678GiB, u_sort min: 2.250GiB, qs min: 0.563GiB.
 	 kBC bucket id: 0 L entries: 228 R entries: 253 matches: 276
 	 kBC bucket id: 1 L entries: 253 R entries: 230 matches: 227
 	 kBC bucket id: 2 L entries: 230 R entries: 232 matches: 212
 	 kBC bucket id: 3 L entries: 232 R entries: 237 matches: 221
 	 Total matches: 4294848520
	 */
	if (threadIdx.x == 0) {
		if (table == 1) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==222) && (num_R==242) && (total_matches==219)) {
					printf("- TABLE 1 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 1 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 240 R entries: 233 matches: 232
			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==240) && (num_R==233) && (total_matches==232)) {
					printf("- TABLE 1 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 1 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
		if (table == 2) {
			if (global_kbc_L_bucket_id == 0) {
				if ((num_L==228) && (num_R==253) && (total_matches==276)) {
					printf("- TABLE 2 MATCHES CORRECT -\n");
				} else {
					printf("*** TABLE 2 MATCHES WRONG! ***\n");
				}
			}
			//kBC bucket id: 4000000 L entries: 241 R entries: 238 matches: 224

			if (global_kbc_L_bucket_id == 4000000) {
				if ((num_L==241) && (num_R==238) && (total_matches==224)) {
					printf("- TABLE 2 bucket 4000000 MATCHES CORRECT num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				} else {
					printf("*** TABLE 2 bucket 4000000 MATCHES WRONG! num_L:%u num_R:%u matches:%u-\n", num_L, num_R, total_matches);
				}
			}
		}
	}
}

#define KBCFILTER_WITH_XINCLUDES(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	for (int j=0;j<64;j++) { \
		if (include_xs[j] == (x+i)) { printf("including x %u\n", (x+i)); \
	if ((kbc_bucket_id >= KBC_START) && (kbc_bucket_id <= KBC_END)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START; \
		int slot = atomicAdd(&kbc_local_num_entries[local_kbc_bucket_id],1); \
		F1_Bucketed_kBC_Entry entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		kbc_local_entries[entries_address] = entry; \
	} \
	} } \
}

//if ((x + i) < 256) { printf("x: %u  y:%llu  kbc:%u\n", (x+i), y, kbc_bucket_id); }
#define KBCFILTER(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	for (int j=0;j<64;j++) { \
		if (include_xs[j] == (x+i)) { printf("including x %u\n", (x+i)); \
	if ((kbc_bucket_id >= KBC_START) && (kbc_bucket_id <= KBC_END)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START; \
		int slot = atomicAdd(&kbc_local_num_entries[local_kbc_bucket_id],1); \
		F1_Bucketed_kBC_Entry entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		kbc_local_entries[entries_address] = entry; \
	} \
	} } \
}

//if ((x + i) < 256) { printf("x: %u  y:%llu  kbc:%u\n", (x+i), y, kbc_bucket_id); }
//if (((x+i) % (1024*1024)) == 0) { printf("x: %u  chacha: %u y:%llu  kbc:%u\n", (x+i), chacha_y, y, kbc_bucket_id); }
//if (kbc_bucket_id == 0) { printf("x: %u  chacha: %u y:%llu  kbc:%u\n", (x+i), chacha_y, y, kbc_bucket_id); }

#define KBCFILTER(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
if ((kbc_bucket_id >= KBC_START) && (kbc_bucket_id <= KBC_END)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START; \
		int slot = atomicAdd(&kbc_local_num_entries[local_kbc_bucket_id],1); \
		F1_Bucketed_kBC_Entry entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		kbc_local_entries[entries_address] = entry; \
	} \
}

__global__
void gpu_chacha8_get_k32_keystream_into_local_kbc_entries(const uint32_t N,
		const __restrict__ uint32_t *input, F1_Bucketed_kBC_Entry *kbc_local_entries, int *kbc_local_num_entries,
		uint32_t KBC_START, uint32_t KBC_END)
{
	uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group
	/*const uint32_t include_xs[64] = {602009779,2127221679,3186459061,443532047,1234434947,1652736830,396228306,464118917,
	                             3981993340,3878862024,1730679522,3234011360,521197720,2635193875,2251292298,608281027,
	                             1468569780,2075860307,2880258779,999340005,1240438978,4293399624,4226635802,1031429862,
	                             2391120891,3533658526,3823422504,3983813271,4180778279,2403148863,2441456056,319558395,
	                             2338010591,196206622,1637393731,853158574,2704638588,2368357012,1703808356,451208700,
	                             2145291166,2741727812,3305809226,1748168268,415625277,3051905493,4257489502,1429077635,
	                             2438113590,3028543211,3993396297,2678430597,458920999,889121073,3577485087,1822568056,
	                             2222781147,1942400192,195608354,1460166215,2544813525,3231425778,2958837604,2710532969};*/

	uint32_t x_group = index;
	//for (uint32_t x_group = index; x_group <= end_n; x_group += stride) {
		uint32_t x = x_group << 4;//  *16;
		uint32_t pos = x_group;

		x0 = input[0];x1 = input[1];x2 = input[2];x3 = input[3];x4 = input[4];x5 = input[5];x6 = input[6];x7 = input[7];
		x8 = input[8];x9 = input[9];x10 = input[10];x11 = input[11];
		x12 = pos; x13 = 0; // pos never bigger than 32 bit pos >> 32;
		x14 = input[14];x15 = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(x0, x4, x8, x12);QUARTERROUND(x1, x5, x9, x13);QUARTERROUND(x2, x6, x10, x14);QUARTERROUND(x3, x7, x11, x15);
			QUARTERROUND(x0, x5, x10, x15);QUARTERROUND(x1, x6, x11, x12);QUARTERROUND(x2, x7, x8, x13);QUARTERROUND(x3, x4, x9, x14);
		}

		x0 += input[0];x1 += input[1];x2 += input[2];x3 += input[3];x4 += input[4];
		x5 += input[5];x6 += input[6];x7 += input[7];x8 += input[8];x9 += input[9];
		x10 += input[10];x11 += input[11];x12 += pos; // j12;//x13 += 0;
		x14 += input[14];x15 += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(x0);BYTESWAP32(x1);BYTESWAP32(x2);BYTESWAP32(x3);BYTESWAP32(x4);BYTESWAP32(x5);
		BYTESWAP32(x6);BYTESWAP32(x7);BYTESWAP32(x8);BYTESWAP32(x9);BYTESWAP32(x10);BYTESWAP32(x11);
		BYTESWAP32(x12);BYTESWAP32(x13);BYTESWAP32(x14);BYTESWAP32(x15);

		//uint64_t y = x0 << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = x0 >> 22; // gives bucket id 0..1023
		KBCFILTER(x0,0);KBCFILTER(x1,1);KBCFILTER(x2,2);KBCFILTER(x3,3);
		KBCFILTER(x4,4);KBCFILTER(x5,5);KBCFILTER(x6,6);KBCFILTER(x7,7);
		KBCFILTER(x8,8);KBCFILTER(x9,9);KBCFILTER(x10,10);KBCFILTER(x11,11);
		KBCFILTER(x12,12);KBCFILTER(x13,13);KBCFILTER(x14,14);KBCFILTER(x15,15);
	//}
}

__global__
void gpu_print_kbc_counts(int *local_kbc_counts) {
	for (int i = 0; i < 10/*KBC_LOCAL_NUM_BUCKETS*/; i++) {
		printf("kbc bucket: %u  num:%u\n", i, local_kbc_counts[i]);
	}
}


template <typename BUCKETED_ENTRY>
__global__
void gpu_print_kbc_bucket_contents(BUCKETED_ENTRY *entries, int *local_kbc_counts) {
	for (uint32_t kbc_bucket_id = 0; kbc_bucket_id < 4/*KBC_LOCAL_NUM_BUCKETS*/; kbc_bucket_id++) {
		int num = local_kbc_counts[kbc_bucket_id];
		uint64_t add_Y = CALC_KBC_BUCKET_ADD_Y(kbc_bucket_id);
		printf("kbc bucket: %u  num:%u\n", kbc_bucket_id, num);
		for (int idxL=0;idxL<num;idxL++) {
			BUCKETED_ENTRY entry = entries[kbc_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET + idxL];
			uint64_t calc_y = (uint64_t) entry.y + add_Y;
			printf("   y: %u   calc_y:%llu   meta0:%u meta1:%u \n", entry.y, calc_y, entry.meta[0], entry.meta[1]);
		}
	}
}

template <typename BUCKETED_ENTRY>
__global__
void gpu_merge_block_buckets_into_kbc_buckets(
		const uint32_t KBC_START_ID, // determined by batch_id
		const BUCKETED_ENTRY *in, uint64_t batch_bucket_add_Y, const uint32_t N,
		BUCKETED_ENTRY *local_kbc_entries, int *local_kbc_counts)
{
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	//for (int i = 0; i < N; i++) {
	if (i < N) {
		// TODO: try just reading out entries and see if they match when going in

		BUCKETED_ENTRY block_entry = in[i];
		uint64_t calc_y = (uint64_t) block_entry.y + batch_bucket_add_Y;
		uint32_t kbc_id = calc_y / kBC;
		uint32_t KBC_END_ID = KBC_START_ID + KBC_LOCAL_NUM_BUCKETS;
		if ((kbc_id < KBC_START_ID) || (kbc_id > KBC_END_ID)) {
			printf(" i:%u  entry.y:%u  add_Y:%llu calc_y:%llu OUT OF RANGE: kbc id: %u   KBC_LOCAL_NUM_BUCKETS:%u START:%u  END:%u\n", i, block_entry.y, batch_bucket_add_Y, calc_y, kbc_id, KBC_LOCAL_NUM_BUCKETS, KBC_START_ID, KBC_END_ID);
		}

		uint32_t local_kbc_id = kbc_id - KBC_START_ID;
		int slot = atomicAdd(&local_kbc_counts[local_kbc_id],1);
		uint32_t destination_address = local_kbc_id * KBC_MAX_ENTRIES_PER_BUCKET + slot;

		//printf("block_id:%u [i: %u] entry.y:%u  kbc_id:%u   local_kbc:%u   slot:%u   dest:%u\n",
		//		block_id, i, block_entry.y, kbc_id, local_kbc_id, slot, destination_address);

		if (slot > KBC_MAX_ENTRIES_PER_BUCKET) {
			printf("OVERFLOW: slot > MAX ENTRIES PER BUCKET\n");
		}
		if (destination_address > DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			printf("OVERFLOW: destination_address overflow > DEVICE_BUFFER_ALLOCATED_ENTRIES %u\n", destination_address);
		}
		block_entry.y = calc_y % kBC; // hah! Don't forget to map it to kbc bucket form.
		local_kbc_entries[destination_address] = block_entry;
	}
}

template <typename BUCKETED_ENTRY, typename BUCKETED_ENTRY_BLOCKPOSREF>
__global__
void gpu_merge_block_buckets_into_kbc_buckets_with_blockposref(
		const uint32_t KBC_START_ID, const uint32_t block_id, // determined by batch_id
		const BUCKETED_ENTRY *in, uint64_t batch_bucket_add_Y, const uint32_t N,
		BUCKETED_ENTRY_BLOCKPOSREF *local_kbc_entries, int *local_kbc_counts,
		int metasize)
{
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	//for (int i = 0; i < N; i++) {
	if (i < N) {
		// TODO: try just reading out entries and see if they match when going in

		BUCKETED_ENTRY block_entry = in[i];
		BUCKETED_ENTRY_BLOCKPOSREF backref_entry = {};
		//size_t n = sizeof(block_entry.meta)/sizeof(block_entry.meta[0]);
		for (int s=0;s<metasize;s++) backref_entry.meta[s] = block_entry.meta[s];

		backref_entry.blockposref = (block_id << (32 - 6)) + i; // encode block_id on top 6 bits, and refpos on lower bits.
		uint64_t calc_y = (uint64_t) block_entry.y + batch_bucket_add_Y;
		backref_entry.y = calc_y % kBC;
		uint32_t kbc_id = calc_y / kBC;
		uint32_t KBC_END_ID = KBC_START_ID + KBC_LOCAL_NUM_BUCKETS;
		if ((kbc_id < KBC_START_ID) || (kbc_id > KBC_END_ID)) {
			printf(" i:%u  entry.y:%u  add_Y:%llu calc_y:%llu OUT OF RANGE: kbc id: %u   KBC_LOCAL_NUM_BUCKETS:%u START:%u  END:%u\n", i, block_entry.y, batch_bucket_add_Y, calc_y, kbc_id, KBC_LOCAL_NUM_BUCKETS, KBC_START_ID, KBC_END_ID);
		}

		uint32_t local_kbc_id = kbc_id - KBC_START_ID;
		int slot = atomicAdd(&local_kbc_counts[local_kbc_id],1);
		uint32_t destination_address = local_kbc_id * KBC_MAX_ENTRIES_PER_BUCKET + slot;

		//printf("block_id:%u [i: %u] entry.y:%u  kbc_id:%u   local_kbc:%u   slot:%u   dest:%u\n",
		//		block_id, i, block_entry.y, kbc_id, local_kbc_id, slot, destination_address);

		if (slot > KBC_MAX_ENTRIES_PER_BUCKET) {
			printf("OVERFLOW: slot > MAX ENTRIES PER BUCKET\n");
		}
		if (destination_address > DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			printf("OVERFLOW: destination_address overflow > DEVICE_BUFFER_ALLOCATED_ENTRIES %u\n", destination_address);
		}
		//block_entry.y = calc_y % kBC; // hah! Don't forget to map it to kbc bucket form.
		local_kbc_entries[destination_address] = backref_entry;
	}
}

void transferBlocksFromHostToDevice(const uint16_t table, const uint32_t batch_id,
		char *device_buffer_in, char *device_buffer_kbc, const size_t DEVICE_ENTRY_SIZE) {
	uint32_t KBC_START = MIN_KBC_BUCKET_FOR_BATCH(batch_id);

	// consider compressing stream to cpu
	// https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/

	// clear local kbc's!
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));

	uint64_t device_bytes_start = 0;
	uint32_t total_entries_copied = 0;
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
		//std::cout << "\n   Preparing batch:" << batch_id << " block:" << block_id << " for host->device" << std::endl;
		uint32_t criss_cross_id = getCrissCrossBlockId(table,batch_id,block_id);
		//std::cout << "      criss_cross_id:" << criss_cross_id << std::endl;
		uint32_t num_entries_to_copy = host_criss_cross_entry_counts[criss_cross_id];
		//std::cout << "        num_entries_to_copy: " << num_entries_to_copy << std::endl;
		uint64_t host_block_entry_start_position = getCrissCrossBlockEntryStartPosition(criss_cross_id);
		uint64_t host_bytes_start = host_block_entry_start_position * HOST_UNIT_BYTES;
		//std::cout << "        host_block_entry_start_position: " << host_block_entry_start_position << std::endl;
		//std::cout << "        host_bytes_start: " << host_bytes_start << std::endl;
		total_entries_copied += num_entries_to_copy;

		if (num_entries_to_copy > HOST_MAX_BLOCK_ENTRIES) {
			std::cout << "OVERFLOW: num_entries_to_copy " << num_entries_to_copy << " > HOST_MAX_BLOCK_ENTRIES " << HOST_MAX_BLOCK_ENTRIES << std::endl;
		}

		size_t bytes_to_copy = num_entries_to_copy*DEVICE_ENTRY_SIZE;
		if (device_bytes_start + bytes_to_copy > DEVICE_BUFFER_ALLOCATED_BYTES) {
			std::cout << "ERROR: DEVICE BUFFER OVERFLOW\n size wanted: " << (device_bytes_start + bytes_to_copy) << " size available:" << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
		}
		if (host_bytes_start + bytes_to_copy > HOST_ALLOCATED_BYTES) {
			std::cout << "ERROR: HOST MEM OVERFLOW\n size wanted: " << (host_bytes_start + bytes_to_copy) << " size available:" << HOST_ALLOCATED_BYTES << std::endl;
		}

		/*
			Total tables time: 73825 ms
        match: 10377 ms
   ----------
transfer time: 61610 ms
        bytes: 687109273160 (639GB)


		******- no pci transfer, do direct fro mhost...saved 7s or 10% (ok, we don't include writing to disk) ***************
Total tables time: 66989 ms
        match: 10358 ms
   ----------
transfer time: 54805 ms
        bytes: 687109273464 (639GB)
*********************
			        */

		//std::cout << "   Copying " << num_entries_to_copy
		//		<< " entries from device_bytes_start: " << device_bytes_start
		//		<< "             to host_bytes_start: " << host_bytes_start
		//		<< "                    bytes length: " << bytes_to_copy << std::endl;
		//std::cout << "   Block_id: " << block_id << " device->host bytes:" << bytes_to_copy << " entries:" << num_entries_to_copy << std::endl;
		const bool use_direct_from_host = true;
		if (!use_direct_from_host) {
			CUDA_CHECK_RETURN(cudaMemcpy(&device_buffer_in[device_bytes_start], &host_criss_cross_blocks[host_bytes_start],bytes_to_copy,cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		//std::cout << "   done.\n";

		// now for our block, determine what the kbc counts were, and merge entries ordered into global kbc's.
		// gpu_map_in_buffer_to_global_kbc_for_batch(device_buffer_in, device_buffer_out, num_entries_to_copy);
		int blockSize = 256;
		int numBlocks = (num_entries_to_copy + blockSize - 1) / (blockSize);
		uint64_t batch_bucket_add_Y = CALC_BATCH_BUCKET_ADD_Y(batch_id);//(((uint64_t) 1) << (38-6)) * ((uint64_t) batch_id);
		if (table == 2) {
			Tx_Bucketed_Meta2 *in;
			if (use_direct_from_host) in = (Tx_Bucketed_Meta2 *) &host_criss_cross_blocks[host_bytes_start];
			else in = (Tx_Bucketed_Meta2 *) &device_buffer_in[device_bytes_start];
			Tx_Bucketed_Meta2 *local_kbc_entries = (Tx_Bucketed_Meta2 *) &device_buffer_kbc[0];
			gpu_merge_block_buckets_into_kbc_buckets<Tx_Bucketed_Meta2><<<numBlocks,blockSize>>>(
								KBC_START,
								in, batch_bucket_add_Y, num_entries_to_copy,
								local_kbc_entries, device_local_kbc_num_entries);
		} else if ((table == 3) || (table == 4)) {
			Tx_Bucketed_Meta4 *in;
			if (use_direct_from_host) in = (Tx_Bucketed_Meta4 *) &host_criss_cross_blocks[host_bytes_start];
			else in = (Tx_Bucketed_Meta4 *) &device_buffer_in[device_bytes_start];
			//Tx_Bucketed_Meta4 *in = (Tx_Bucketed_Meta4 *) &device_buffer_in[device_bytes_start];
			//Tx_Bucketed_Meta4 *local_kbc_entries = (Tx_Bucketed_Meta4 *) &device_buffer_kbc[0];
			Tx_Bucketed_Meta4_Blockposref *local_kbc_entries = (Tx_Bucketed_Meta4_Blockposref *) &device_buffer_kbc[0];
			gpu_merge_block_buckets_into_kbc_buckets_with_blockposref<Tx_Bucketed_Meta4,Tx_Bucketed_Meta4_Blockposref><<<numBlocks,blockSize>>>(
					KBC_START,block_id,
					in, batch_bucket_add_Y, num_entries_to_copy,
					local_kbc_entries, device_local_kbc_num_entries,
					4);
		} else if (table == 5) {
			Tx_Bucketed_Meta3 *in;
			if (use_direct_from_host) in = (Tx_Bucketed_Meta3 *) &host_criss_cross_blocks[host_bytes_start];
			else in = (Tx_Bucketed_Meta3 *) &device_buffer_in[device_bytes_start];
			//Tx_Bucketed_Meta3 *in = (Tx_Bucketed_Meta3 *) &device_buffer_in[device_bytes_start];
			Tx_Bucketed_Meta3_Blockposref *local_kbc_entries = (Tx_Bucketed_Meta3_Blockposref *) &device_buffer_kbc[0];
			gpu_merge_block_buckets_into_kbc_buckets_with_blockposref<Tx_Bucketed_Meta3,Tx_Bucketed_Meta3_Blockposref><<<numBlocks,blockSize>>>(
					KBC_START,block_id,
					in, batch_bucket_add_Y, num_entries_to_copy,
					local_kbc_entries, device_local_kbc_num_entries,
					3);
		} else if (table == 6) {
			Tx_Bucketed_Meta2 *in;
			if (use_direct_from_host) in = (Tx_Bucketed_Meta2 *) &host_criss_cross_blocks[host_bytes_start];
			else in = (Tx_Bucketed_Meta2 *) &device_buffer_in[device_bytes_start];
			//Tx_Bucketed_Meta2 *in = (Tx_Bucketed_Meta2 *) &device_buffer_in[device_bytes_start];
			Tx_Bucketed_Meta2_Blockposref *local_kbc_entries = (Tx_Bucketed_Meta2_Blockposref *) &device_buffer_kbc[0];
			gpu_merge_block_buckets_into_kbc_buckets_with_blockposref<Tx_Bucketed_Meta2,Tx_Bucketed_Meta2_Blockposref><<<numBlocks,blockSize>>>(
					KBC_START,block_id,
					in, batch_bucket_add_Y, num_entries_to_copy,
					local_kbc_entries, device_local_kbc_num_entries,
					2);
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		device_bytes_start += bytes_to_copy;
		table_transfer_in_bytes += bytes_to_copy;
	}
	//std::cout << "\nTotal entries copied in batch " << batch_id << ": " << total_entries_copied << std::endl;
}



int mmap_fdout;
char *mmap_address;
void setupMMap(size_t desired_size_bytes) {

	int mode = 0x0777;

	std::string filename = "/mnt/kioxia/tmp/test-mmap.tmp";

	std::cout << "Setting up MMap with " << desired_size_bytes << " bytes in file: " << filename << std::endl;

	if ((mmap_fdout = open (filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode )) < 0) {
		std::cout << "can't create " << filename << " for writing" << std::endl;
		return;
	}

	/* go to the location corresponding to the last byte */
	if (lseek (mmap_fdout, desired_size_bytes, SEEK_SET) == -1) {
		printf ("lseek error");
		return;
	}

	/* write a dummy byte at the last location */
	if (write (mmap_fdout, "", 1) != 1) {
		printf ("write error");
		return;
	}

	if ((mmap_address = (char *) mmap (0, desired_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mmap_fdout, 0)) == (caddr_t) -1) {
		printf ("mmap error for output");
	    return;
	}

	std::cout << "MMap done." << std::endl;
}

inline void writeHostMemToMMap(uint32_t address, char *host_mem, uint32_t bytes_to_copy) {
	//std::string filename = "/mnt/kioxia/tmp/test" + std::to_string(table) + "-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	//std::cout << "Writing to file " << filename << std::endl;
	//FILE* pFile;
	//pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
	//fwrite(host_mem, 1, bytes_to_copy, pFile);
	//fclose(pFile);
	memcpy(mmap_address, host_mem, bytes_to_copy);
}



void convertAndWriteT2HostMemToBlockFiles(
		uint16_t batch_id, uint16_t block_id,
		Tx_Bucketed_Meta4 *t2_data, // will take meta[0] and meta[2] for Lx1 and Lx2
		uint32_t num_entries_to_copy) {

	if (num_entries_to_copy == 0) {
		return;
	}
	// first convert to memory
	T2BaseRef *t2_base = (T2BaseRef *) host_refdata_blocks;
	for (int i=0;i<num_entries_to_copy;i++) {
		T2BaseRef entry = {};
		entry.Lx1 = t2_data[i].meta[0];
		entry.Lx2 = t2_data[i].meta[2];
		if ((entry.Lx1 == 0) && (entry.Lx2 == 0)) {
			std::cout << "error: Lx1/Lx2 entries are 0 for batch_id:" << batch_id << " block_id:" << block_id << std::endl;
		}
		t2_base[i] = entry;
	}

	// then flush to disk...

	uint32_t bytes_to_copy = sizeof(T2BaseRef) * num_entries_to_copy;

	std::string filename = "/mnt/kioxia/tmp/T2-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	//if (batch_id == 0) {
	//	std::cout << "Writing to file [" << filename << "]";
	//} else {
	//	std::cout << " [" << filename << "]";
	//}
	FILE* pFile;
	pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
	fwrite(&num_entries_to_copy, sizeof(uint32_t), 1, pFile); // write the num entries first.
	fwrite(t2_base, 1, bytes_to_copy, pFile);
	fclose(pFile);
	//if (batch_id == BATCHES-1) {
	//	std::cout << " done." << std::endl;
	//}
}


void writeT3BaseDataToBlockFiles(uint16_t batch_id, uint16_t block_id, char *t3_base_ref,
		uint32_t num_entries_to_copy, uint32_t bytes_to_copy) {
	if (num_entries_to_copy == 0) {
		return;
	}
	std::string filename = "/mnt/kioxia/tmp/T3BaseRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	FILE* pFile;
	pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
	fwrite(&num_entries_to_copy, sizeof(uint32_t), 1, pFile); // write the num entries first.
	fwrite(t3_base_ref, 1, bytes_to_copy, pFile);
	fclose(pFile);
}

void writeHostRefdataToBlockFiles(uint16_t table, uint16_t batch_id, uint16_t block_id, char *host_ref, uint32_t num_entries_to_copy, uint32_t bytes_to_copy) {
	if (num_entries_to_copy == 0) {
		return;
	}
	std::string filename = "/mnt/kioxia/tmp/T" + std::to_string(table) + "BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	//if (batch_id == 0) {
	//	std::cout << "Writing backref to file [" << filename << "]";
	//} else {
	//	std::cout << " [" << filename << "]";
	//}
	FILE* pFile;
	pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
	fwrite(&num_entries_to_copy, sizeof(uint32_t), 1, pFile); // write the num entries first.
	fwrite(host_ref, 1, bytes_to_copy, pFile);
	fclose(pFile);
	//if (batch_id == BATCHES-1) {
	//	std::cout << " done." << std::endl;
	//}
	//if (table == 6) {
	//	T6BackRef *t6_data = (T6BackRef *) host_ref;
	//	uint32_t num_entries = num_entries_to_copy;
		//std::cout << "Num entries T6 block " << block_id << " num_entries: " << num_entries << "   bytes:" << bytes_to_copy << std::endl;
		//for (int i=0;i<10;i++) {
		//	T6BackRef entry = t6_data[i];
			//printf("T6 BackRef L:%u R:%u y:%u\n", entry.prev_block_ref_L, entry.prev_block_ref_R, entry.y);
		//}
	//}
}


void writeT2HostMemToBlockFiles(uint16_t table, uint16_t batch_id, uint16_t block_id, char *host_mem, uint32_t bytes_to_copy) {
	std::string filename = "/mnt/kioxia/tmp/T2-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	//if (batch_id == 0) {
	//	std::cout << "Writing to file [" << filename << "]";
	//} else {
	//	std::cout << " [" << filename << "]";
	//}
	FILE* pFile;
	pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
	fwrite(host_mem, 1, bytes_to_copy, pFile);
	fclose(pFile);
	//if (batch_id == BATCHES-1) {
	//	std::cout << " done." << std::endl;
	//}
}

FILE* t2File;
void writeT2HostMemToBatchFiles(uint16_t table, uint16_t batch_id, uint16_t block_id, char *host_mem, uint32_t bytes_to_copy) {
	std::string filename = "/mnt/kioxia/tmp/T2-largefile-" + std::to_string(batch_id) + ".tmp";
	if (block_id == 0) {
		std::cout << "Opening file " << filename << std::endl;
		t2File = fopen(filename.c_str(), "wb");
	}
	std::cout << ".";
	fwrite(host_mem, 1, bytes_to_copy, t2File);
	if (block_id == BATCHES) {
		std::cout << "Closing file " << filename << std::endl;
		fclose(t2File);
	}
}

void writeT2HostMemToTableFiles(uint16_t table, uint16_t batch_id, uint16_t block_id, char *host_mem, uint32_t bytes_to_copy) {
	std::string filename = "/mnt/kioxia/tmp/T2-table-batches.tmp";
	if ((batch_id == 0) && (block_id == 0)) {
		std::cout << "Opening file " << filename << std::endl;
		t2File = fopen(filename.c_str(), "wb");
	}
	std::cout << "Writing to file " << filename << std::endl;
	fwrite(host_mem, 1, bytes_to_copy, t2File);
	if ((batch_id == (BATCHES-1)) && (block_id == (BATCHES-1))) {
		std::cout << "Closing file " << filename << std::endl;
		fclose(t2File);
	}
}

bool doWriteT2BaseData = false;
bool doWriteT3BaseData = true;
bool doWriteRefData = true;
bool doWriteT6Data = true;

uint32_t max_block_entries_copied_device_to_host = 0;
void transferBucketedBlocksFromDeviceToHost(const uint16_t table, const uint32_t batch_id,
		char *device_buffer, const size_t DEVICE_ENTRY_SIZE,
		char *device_refdata, const int* block_counts) {

	const bool doPrint = false;
	uint64_t batch_bytes_transfered = 0;
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {

		//std::cout << "\n   Preparing batch:" << batch_id << " block:" << block_id << " for transfer" << std::endl;

		uint32_t criss_cross_id = getCrissCrossBlockId(table,batch_id,block_id);
		if (doPrint) std::cout << "      criss_cross_id:" << criss_cross_id << std::endl;

		uint32_t num_entries_to_copy = block_counts[block_id];
		//std::cout << "        num_entries_to_copy: " << num_entries_to_copy << std::endl;

		uint64_t host_block_entry_start_position = getCrissCrossBlockEntryStartPosition(criss_cross_id);
		uint64_t host_bytes_start = host_block_entry_start_position * HOST_UNIT_BYTES;
		if (doPrint) std::cout << "        host_block_entry_start_position: " << host_block_entry_start_position << std::endl;
		if (doPrint) std::cout << "        host_bytes_start: " << host_bytes_start << std::endl;

		uint32_t device_entry_start = block_id * HOST_MAX_BLOCK_ENTRIES; // device bucketed block entry pos
		host_criss_cross_entry_counts[criss_cross_id] = num_entries_to_copy;

		if (doPrint) std::cout << "        device_entry_start: " << device_entry_start << std::endl;
		if (num_entries_to_copy > HOST_MAX_BLOCK_ENTRIES) {
			std::cout << "OVERFLOW: num_entries_to_copy " << num_entries_to_copy << " > HOST_MAX_BLOCK_ENTRIES " << HOST_MAX_BLOCK_ENTRIES << std::endl;
		}
		if (max_block_entries_copied_device_to_host < num_entries_to_copy) {
			max_block_entries_copied_device_to_host = num_entries_to_copy; // helps determine HOST_MAX_BLOCK_ENTRIES value.
		}

		uint64_t device_bytes_start = device_entry_start * DEVICE_ENTRY_SIZE;
		size_t bytes_to_copy = num_entries_to_copy*DEVICE_ENTRY_SIZE;
		if (device_bytes_start + bytes_to_copy > DEVICE_BUFFER_ALLOCATED_BYTES) {
			std::cout << "ERROR: DEVICE BUFFER OVERFLOW\n size wanted: " << (device_bytes_start + bytes_to_copy) << " size available:" << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
		}
		if (host_bytes_start + bytes_to_copy > HOST_ALLOCATED_BYTES) {
			std::cout << "ERROR: HOST MEM OVERFLOW\n size wanted: " << (host_bytes_start + bytes_to_copy) << " size available:" << HOST_ALLOCATED_BYTES << std::endl;
		}
		//if (doPrint) std::cout << "   Copying " << num_entries_to_copy
		//		<< " entries from device_bytes_start: " << device_bytes_start
		//		<< "             to host_bytes_start: " << host_bytes_start
		//		<< "                    bytes length: " << bytes_to_copy << std::endl;
		//std::cout << "   Block_id: " << block_id << " device->host bytes:" << bytes_to_copy << " entries:" << num_entries_to_copy << std::endl;

		if (table < 6) {
			// we only copy criss cross memory if it's not the last table, since that only exports back ref data and no forward propagation.
			CUDA_CHECK_RETURN(cudaMemcpy(&host_criss_cross_blocks[host_bytes_start],&device_buffer[device_bytes_start],bytes_to_copy,cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			batch_bytes_transfered += bytes_to_copy;
		}
		if (doPrint) std::cout << "   done.\n";

		//if (table == 1) {
			// oof...mmap is 47000ms transfer for T1
		//	memcpy(mmap_address + total_transfered_bytes, &host_criss_cross_blocks[host_bytes_start], bytes_to_copy);
			//writeHostMemToMMap(total_transfered_bytes, &host_criss_cross_blocks[host_bytes_start], bytes_to_copy);
		//}



		// for T2 we dump to file, since this becomes the baseline with 4 meta entries for x's.
		/*if (table == 2) {
			// 42241 ms - a wash whether we write 4 x's in T2 or use 2'xs in T1 and write a 64bit ref here.
			// BUT- our goals is to get kbc's, so base level T2 can just write 2 kbc entries (only 50 bits (25 kbc * 2) but we need
			// CPU to process the entry and split into the proper reference buckets at this stage. 64 batches already splits 18m kbc's
			// down into 285k kbc's so should help with mem buffer. OR...we could do this in GPU and just use CPU to dumb copy, ay?
			// BUUUTTT - we need enough spare memory so have to do it at end of entire first phase process.
			// tables was 56..yeesh
			if (doWriteT2BaseData) {
				Tx_Bucketed_Meta4 *t2_data = (Tx_Bucketed_Meta4 *) &host_criss_cross_blocks[host_bytes_start];
				convertAndWriteT2HostMemToBlockFiles(batch_id, block_id, t2_data, num_entries_to_copy);
			}
		}*/
		if (table == 3) {
			if (doWriteT3BaseData) {
				uint64_t refdata_bytes_start;
				size_t refdata_bytes_to_copy;
				refdata_bytes_start = device_entry_start * sizeof(T3BaseRef);
				refdata_bytes_to_copy = num_entries_to_copy*sizeof(T3BaseRef);

				if (refdata_bytes_start + bytes_to_copy > DEVICE_BUFFER_ALLOCATED_BYTES) {
					std::cout << "ERROR: DEVICE REFDATA OVERFLOW\n size wanted: " << (refdata_bytes_start + refdata_bytes_to_copy) << " size available:" << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
				}
				CUDA_CHECK_RETURN(cudaMemcpy(&host_refdata_blocks[refdata_bytes_start],&device_refdata[refdata_bytes_start],refdata_bytes_to_copy,cudaMemcpyDeviceToHost));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				// now write to files
				writeT3BaseDataToBlockFiles(batch_id, block_id,
						&host_refdata_blocks[refdata_bytes_start],
						num_entries_to_copy, refdata_bytes_to_copy);
			}
		}
		if (table > 3) {
			// transfer back ref
			if (table == 6) doWriteRefData = doWriteT6Data;
			if (doWriteRefData) {
				uint64_t refdata_bytes_start;
				size_t refdata_bytes_to_copy;
				if (table == 6) {
					refdata_bytes_start = device_entry_start * sizeof(T6BackRef);
					refdata_bytes_to_copy = num_entries_to_copy*sizeof(T6BackRef);
				} else {
					refdata_bytes_start = device_entry_start * sizeof(BackRef);
					refdata_bytes_to_copy = num_entries_to_copy*sizeof(BackRef);
				}
				if (refdata_bytes_start + bytes_to_copy > DEVICE_BUFFER_ALLOCATED_BYTES) {
					std::cout << "ERROR: DEVICE REFDATA OVERFLOW\n size wanted: " << (refdata_bytes_start + refdata_bytes_to_copy) << " size available:" << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
				}
				CUDA_CHECK_RETURN(cudaMemcpy(&host_refdata_blocks[refdata_bytes_start],&device_refdata[refdata_bytes_start],refdata_bytes_to_copy,cudaMemcpyDeviceToHost));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				// now write to files
				writeHostRefdataToBlockFiles(table, batch_id, block_id, &host_refdata_blocks[refdata_bytes_start], num_entries_to_copy, refdata_bytes_to_copy);
			}
		}

	}
	//fclose(pFile);
	//std::cout << "Waiting for writes to finish...";
	//for(uint8_t i=0;i<BATCHES;i++) { threads[i].join(); std::cout << "[" << i << "]";}

	if (doPrint) std::cout << "\nTotal bytes for batch copied: " << batch_bytes_transfered << std::endl;
	table_transfer_out_bytes += batch_bytes_transfered;
}



void doT1Batch(uint32_t batch_id, int* local_kbc_num_entries, const uint32_t KBC_START, const uint32_t KBC_END) {
	// 1) gpu scan kbs into (F1_Bucketed_kBC_Entry *) bufferA
	// 2) gpu find_f1_matches from (F1_Bucketed_kBC_Entry *) bufferA to (T1_Pairing_Chunk *) bufferB
	// 3) gpu exclusive scan kbc_counts to get kbc_memory_positions by blocks, and kbc_block_counts
	// 4) gpu cp (T1_Pairing_Chunk *) bufferB into (T1_Bucketed_kBC_Entry *) bufferA
	// 5) device to host transfer bufferA

	std::cout << "   doF1Batch: " << batch_id << std::endl <<
				 "     SPANNING FOR BUCKETS    count:" << (KBC_END - KBC_START + 1) << "  KBC_START: " << KBC_START << "   KBC_END: " << KBC_END << std::endl;

	auto batch_start = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();

	F1_Bucketed_kBC_Entry *local_kbc_entries = (F1_Bucketed_kBC_Entry *) device_buffer_A;

	// 1) gpu scan kbs into (F1_Bucketed_kBC_Entry *) bufferA
	//std::cout << "   Generating F1 results into kbc buckets...";
	start = std::chrono::high_resolution_clock::now();
	int blockSize = 256; // # of threads per block, maximum is 1024.
	const uint64_t calc_N = UINT_MAX;
	const uint64_t calc_blockSize = blockSize;
	const uint64_t calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
	int numBlocks = calc_numBlocks;
	//std::cout << "  Block configuration: [blockSize:" << blockSize << "  numBlocks:" << numBlocks << "]" << std::endl;
	// don't forget to clear counter...will only use a portion of this memory so should be fast access.
	CUDA_CHECK_RETURN(cudaMemset(local_kbc_num_entries, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
	gpu_chacha8_get_k32_keystream_into_local_kbc_entries<<<numBlocks, blockSize>>>(calc_N, chacha_input,
			local_kbc_entries, local_kbc_num_entries, KBC_START, KBC_END);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	finish = std::chrono::high_resolution_clock::now();
	total_chacha_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
	//std::cout << "   done.     " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";

	// 2) gpu find_f1_matches from (F1_Bucketed_kBC_Entry *) bufferA to (T1_Pairing_Chunk *) bufferB
	std::cout << "   Finding matches...";
	cudaEvent_t mstart, mstop;
	float milliseconds = 0;
	cudaEventCreate(&mstart);
	cudaEventCreate(&mstop);

	start = std::chrono::high_resolution_clock::now();

	Tx_Bucketed_Meta1 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta1 *) device_buffer_A;
	Tx_Bucketed_Meta2 *bucketed_out = (Tx_Bucketed_Meta2 *) device_buffer_B;

	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599
	cudaEventRecord(mstart);
	gpu_find_tx_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(1, batch_id, KBC_START, KBC_END,
			bucketed_kbc_entries_in, local_kbc_num_entries,
			bucketed_out, device_block_entry_counts);
	cudaEventRecord(mstop);
	cudaEventSynchronize(mstop);
	cudaEventElapsedTime(&milliseconds, mstart, mstop);
	std::cout << "gpu_find_tx_matches time: " << milliseconds << " ms\n";
	//gpu_find_tx_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(1, batch_id, KBC_START, KBC_END,
	//		bucketed_kbc_entries_in, local_kbc_num_entries,
	//		host_criss_cross_blocks, device_block_entry_counts);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	finish = std::chrono::high_resolution_clock::now();
	total_match_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
	std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";


	// 4) gpu cp (T1_Pairing_Chunk *) bufferB into (T1_Bucketed_kBC_Entry *) bufferA
	total_gpu_time_ms += std::chrono::duration_cast<milli>(finish - batch_start).count();
	//std::cout << "     transferBucketedBlocksFromDeviceToHost\n";
	start = std::chrono::high_resolution_clock::now();
	transferBucketedBlocksFromDeviceToHost(1, batch_id, device_buffer_B, sizeof(Tx_Bucketed_Meta2), NULL, device_block_entry_counts);
	finish = std::chrono::high_resolution_clock::now();
	table_transfer_out_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
	//std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";
}

void doTxBatch(uint16_t table, uint32_t batch_id) {
	// 1) host to device transfer -> bufferB = (T1_Bucketed_kBC_Entry *) bufferB
	// 2) gpu find_f1_matches from (T1_Bucketed_kBC_Entry *) bufferB to (T2_Pairing_Chunk *) bufferA
	// 3) gpu exclusive scan kbc_counts to get kbc_memory_positions by blocks, and kbc_block_counts
	// 4) gpu cp (T2_Pairing_Chunk *) bufferB into (T2_Bucketed_kBC_Entry *) bufferA
	// 5) device to host transfer bufferA
	auto batch_start = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();

	size_t transfer_in_size = 0;
	size_t transfer_out_size = 0;
	if (table == 2) {
		transfer_in_size = sizeof(Tx_Bucketed_Meta2);
		transfer_out_size = sizeof(Tx_Bucketed_Meta4);
	}
	else if (table == 3) {
		transfer_in_size = sizeof(Tx_Bucketed_Meta4);
		transfer_out_size = sizeof(Tx_Bucketed_Meta4);
	}
	else if (table == 4) {
		transfer_in_size = sizeof(Tx_Bucketed_Meta4);
		transfer_out_size = sizeof(Tx_Bucketed_Meta3);
	}
	else if (table == 5) {
		transfer_in_size = sizeof(Tx_Bucketed_Meta3);
		transfer_out_size = sizeof(Tx_Bucketed_Meta2);
	}
	else if (table == 6) {
		transfer_in_size = sizeof(Tx_Bucketed_Meta2);
		transfer_out_size = 0;
		// TODO: T6 could transfer to hostmem or to the backref blocks table
		// since we will then read from backref blocks tables for all backrefs across tables.
	}

	start = std::chrono::high_resolution_clock::now();
	transferBlocksFromHostToDevice(table, batch_id, device_buffer_B, device_buffer_A, transfer_in_size);
	finish = std::chrono::high_resolution_clock::now();
	table_transfer_in_time_ms += std::chrono::duration_cast<milli>(finish - start).count();

	//gpu_print_kbc_counts<<<1,1>>>(device_local_kbc_num_entries);


	// 2) gpu find_f1_matches from (F1_Bucketed_kBC_Entry *) bufferA to (T1_Pairing_Chunk *) bufferB
	//std::cout << "   Finding matches...";
	start = std::chrono::high_resolution_clock::now();

	//if (batch_id == 0) {
	//	gpu_print_kbc_bucket_contents<Tx_Bucketed_Meta2><<<1,1>>>(bucketed_kbc_entries_in, device_local_kbc_num_entries);
	//}

	const uint32_t KBC_START = MIN_KBC_BUCKET_FOR_BATCH(batch_id);
	const uint32_t next_batch = batch_id + 1;
	const uint32_t KBC_END = MIN_KBC_BUCKET_FOR_BATCH(next_batch);

	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts, 0, (BATCHES)*sizeof(int)));
	if (table == 2) {
		Tx_Bucketed_Meta2 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta2 *) device_buffer_A;
		Tx_Bucketed_Meta4 *bucketed_out = (Tx_Bucketed_Meta4 *) device_buffer_B;
		gpu_find_tx_matches<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(table, batch_id, KBC_START, KBC_END,
				bucketed_kbc_entries_in, device_local_kbc_num_entries,
				bucketed_out, device_block_entry_counts);
	} else if (table == 3) {
		// at table 3 we start pulling in backref to table 2
		//Tx_Bucketed_Meta4 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta4 *) device_buffer_A;
		Tx_Bucketed_Meta4_Blockposref *bucketed_kbc_entries_in = (Tx_Bucketed_Meta4_Blockposref *) device_buffer_A;
		Tx_Bucketed_Meta4 *bucketed_out = (Tx_Bucketed_Meta4 *) device_buffer_B;
		gpu_find_tx_matches_with_backref<Tx_Bucketed_Meta4_Blockposref,Tx_Bucketed_Meta4><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(table, batch_id, KBC_START, KBC_END,
				bucketed_kbc_entries_in, device_local_kbc_num_entries,
				bucketed_out, device_buffer_refdata, device_block_entry_counts);
	} else if (table == 4) {
		//Tx_Bucketed_Meta4 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta4 *) device_buffer_A;
		Tx_Bucketed_Meta4_Blockposref *bucketed_kbc_entries_in = (Tx_Bucketed_Meta4_Blockposref *) device_buffer_A;
		Tx_Bucketed_Meta3 *bucketed_out = (Tx_Bucketed_Meta3 *) device_buffer_B;
		gpu_find_tx_matches_with_backref<Tx_Bucketed_Meta4_Blockposref,Tx_Bucketed_Meta3><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(table, batch_id, KBC_START, KBC_END,
				bucketed_kbc_entries_in, device_local_kbc_num_entries,
				bucketed_out, device_buffer_refdata, device_block_entry_counts);
	} else if (table == 5) {
		//Tx_Bucketed_Meta3 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta3 *) device_buffer_A;
		Tx_Bucketed_Meta3_Blockposref *bucketed_kbc_entries_in = (Tx_Bucketed_Meta3_Blockposref *) device_buffer_A;
		Tx_Bucketed_Meta2 *bucketed_out = (Tx_Bucketed_Meta2 *) device_buffer_B;
		gpu_find_tx_matches_with_backref<Tx_Bucketed_Meta3_Blockposref,Tx_Bucketed_Meta2><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(table, batch_id, KBC_START, KBC_END,
				bucketed_kbc_entries_in, device_local_kbc_num_entries,
				bucketed_out, device_buffer_refdata, device_block_entry_counts);
	} else if (table == 6) {
		//Tx_Bucketed_Meta2 *bucketed_kbc_entries_in = (Tx_Bucketed_Meta2 *) device_buffer_A;
		Tx_Bucketed_Meta2_Blockposref *bucketed_kbc_entries_in = (Tx_Bucketed_Meta2_Blockposref *) device_buffer_A;
		Tx_Bucketed_Meta2 *NOT_USED = (Tx_Bucketed_Meta2 *) device_buffer_B;
		gpu_find_tx_matches_with_backref<Tx_Bucketed_Meta2_Blockposref,Tx_Bucketed_Meta2><<<(KBC_END - KBC_START), THREADS_FOR_MATCHING>>>(table, batch_id, KBC_START, KBC_END,
			bucketed_kbc_entries_in, device_local_kbc_num_entries,
			NOT_USED, device_buffer_refdata, device_block_entry_counts);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	finish = std::chrono::high_resolution_clock::now();
	table_match_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
	//std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";


	// 4) gpu cp (T1_Pairing_Chunk *) bufferB into (T1_Bucketed_kBC_Entry *) bufferA
	//if (table < 6) {
		//std::cout << "     transferBucketedBlocksFromDeviceToHost\n";
		start = std::chrono::high_resolution_clock::now();
		transferBucketedBlocksFromDeviceToHost(table, batch_id, device_buffer_B, transfer_out_size, device_buffer_refdata, device_block_entry_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		finish = std::chrono::high_resolution_clock::now();

		table_transfer_out_time_ms += std::chrono::duration_cast<milli>(finish - start).count();

		//std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";
	//} else if (table == 6) {
		// TODO: handle final T6 file...maybe this can write into hostmem instead of to file.
	//}

}

void doT1() {

	std::cout << "doT1  BATCHES:" << BATCHES  << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();
	auto finish =  std::chrono::high_resolution_clock::now(); // just to allocate

	// what's faster, 0.4% of kbc's, or 0.63% of xs'
	for (uint32_t batch_id = 0; batch_id < BATCHES; batch_id++) {

		uint32_t KBC_START = MIN_KBC_BUCKET_FOR_BATCH(batch_id);
		uint32_t KBC_END = MIN_KBC_BUCKET_FOR_BATCH(batch_id+1)-1;

		auto batch_start = std::chrono::high_resolution_clock::now();
		//if (batch_id < 2)
			doT1Batch(batch_id, device_local_kbc_num_entries, KBC_START, KBC_END);
		finish =  std::chrono::high_resolution_clock::now();
		//std::cout << "  ** T1 batch " << batch_id << " finished ** " << std::chrono::duration_cast<milli>(finish - batch_start).count() << " ms\n";
	}

	finish = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "T1 Total time: " << std::chrono::duration_cast<milli>(finish - total_start).count() << " ms\n";
	std::cout << "     gpu time: " << total_gpu_time_ms << " ms\n";
	std::cout << "       chacha: " << total_chacha_time_ms << " ms\n";
	std::cout << "        match: " << total_match_time_ms << " ms\n";
	std::cout << "   ----------  " << std::endl;
	std::cout << "transfer time: " << table_transfer_out_time_ms << " ms\n";
	std::cout << "        bytes: " << table_transfer_out_bytes << " (" << (table_transfer_out_bytes/(1024*1024*1024)) << "GB)\n";
	std::cout << "*********************" << std::endl;

	total_transfer_in_time_ms += table_transfer_in_time_ms;
	total_transfer_out_time_ms += table_transfer_out_time_ms;
	total_transfer_in_bytes += table_transfer_in_bytes;
	total_transfer_out_bytes += table_transfer_out_bytes;
}

void doTx(uint16_t table) {
	std::cout << "do Table " << table <<"   BATCHES:" << BATCHES << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();
	auto finish =  std::chrono::high_resolution_clock::now(); // just to allocate

	table_match_time_ms = 0;
	table_transfer_in_time_ms = 0;
	table_transfer_out_time_ms = 0;
	table_transfer_in_bytes = 0;
	table_transfer_out_bytes = 0;

	for (uint32_t batch_id = 0; batch_id < BATCHES; batch_id++) {
		auto batch_start = std::chrono::high_resolution_clock::now();
		doTxBatch(table, batch_id);
		finish =  std::chrono::high_resolution_clock::now();
		//std::cout << "  ** T" << table << " batch " << batch_id << " finished ** " << std::chrono::duration_cast<milli>(finish - batch_start).count() << " ms\n";
	}

	finish = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "T" << table << " time: " << std::chrono::duration_cast<milli>(finish - total_start).count() << " ms\n";
	std::cout << "        match: " << table_match_time_ms << " ms\n";
	std::cout << "   ----------  " << std::endl;
	std::cout << "transfer in time: " << table_transfer_in_time_ms << " ms\n";
	std::cout << "         in bytes: " << table_transfer_in_bytes << " (" << (table_transfer_in_bytes/(1024*1024*1024)) << "GB)\n";
	std::cout << "transfer out time: " << table_transfer_out_time_ms << " ms\n";
	std::cout << "        out bytes: " << table_transfer_out_bytes << " (" << (table_transfer_out_bytes/(1024*1024*1024)) << "GB)\n";
	std::cout << "*********************" << std::endl;
	total_match_time_ms += table_match_time_ms;
	total_transfer_in_time_ms += table_transfer_in_time_ms;
	total_transfer_out_time_ms += table_transfer_out_time_ms;
	total_transfer_in_bytes += table_transfer_in_bytes;
	total_transfer_out_bytes += table_transfer_out_bytes;
}




void setupMemory() {

	//setupMMap(HOST_ALLOCATED_BYTES); // potentially useful if going to do random reads/writes to stored data

	std::cout << "      device_block_entry_counts (" << BATCHES << "): " << BATCHES << " size:" << (sizeof(int)*BATCHES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_block_entry_counts, BATCHES*sizeof(int)));

	std::cout << "      device_local_kbc_num_entries " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));

	//Tx_Pairing_Chunk_Meta4 *device_buffer_A;
	std::cout << "      device_buffer_A " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  DEVICE_BUFFER_UNIT_BYTES << ") = " << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_A, DEVICE_BUFFER_ALLOCATED_BYTES));

	//Tx_Pairing_Chunk_Meta4 *device_buffer_B;
	std::cout << "      device_buffer_B " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  DEVICE_BUFFER_UNIT_BYTES << ") = " << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_B, DEVICE_BUFFER_ALLOCATED_BYTES));


	std::cout << "      device_buffer_C " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  DEVICE_BUFFER_UNIT_BYTES << ") = " << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_C, DEVICE_BUFFER_ALLOCATED_BYTES));

	std::cout << "      device_buffer_refdata " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  BACKREF_UNIT_BYTES << ") = " << BACKREF_ALLOCATED_BYTES << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_refdata, BACKREF_ALLOCATED_BYTES));

	std::cout << "      HOST host_refdata_blocks ENTRIES: " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " ALLOCATED ENTRIES: " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " UNIT BYTES: " << BACKREF_UNIT_BYTES << " = " << (BACKREF_ALLOCATED_BYTES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_refdata_blocks, BACKREF_ALLOCATED_BYTES)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();

	std::cout << "      HOST host_criss_cross_blocks MAX_ENTRIES: " << HOST_MAX_BLOCK_ENTRIES << " ALLOCATED ENTRIES: " << HOST_ALLOCATED_ENTRIES << " UNIT BYTES: " << HOST_UNIT_BYTES << " = " << (HOST_ALLOCATED_BYTES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_criss_cross_blocks, HOST_ALLOCATED_BYTES)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();
}



void freeMemory() {
	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(device_buffer_A));
	CUDA_CHECK_RETURN(cudaFree(device_buffer_B));
	CUDA_CHECK_RETURN(cudaFree(device_buffer_C));

	//CUDA_CHECK_RETURN(cudaFree(device_block_entry_counts));
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries));
	CUDA_CHECK_RETURN(cudaFreeHost(host_criss_cross_blocks));
	std::cout << "   memory freed." << std::endl;
}




void doPhase3Compression() {
	// our phase 3 compression then needs to take all pruned batches for T2, and write blocks of kbc's compressed with ANS.
	// it also needs to take T6_Backref table, load all into memory, and sort by y, and put into blocks with new backref into criss cross back ref to table 2 kbc sets.p
}

#include "k29_plotter.hpp"

int main(int argc, char *argv[])
{
	std::cout << "DrPlotter v0.1d" << std::endl;
	chacha_setup();

	cmd_read = 0;

	if (cmd_read == 2) {
		//attack_it();
		doPhase2Pruning();
		exit(EXIT_SUCCESS);
	}
	if (cmd_read == 3) {
		do_k29();
		exit(EXIT_SUCCESS);
	}


	doWriteT2BaseData = false;
	doWriteT3BaseData = false;
	doWriteRefData = false;
	doWriteT6Data = false;
	setupMemory();


	auto total_start = std::chrono::high_resolution_clock::now();
	doT1();
	doTx(2);
	doTx(3);
	doTx(4);
	doTx(5);
	doTx(6);
	auto total_end = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "Total tables time: " << std::chrono::duration_cast<milli>(total_end - total_start).count() << " ms\n";
	std::cout << "        match: " << total_match_time_ms << " ms\n";
	std::cout << "   ----------  " << std::endl;
	std::cout << "transfer in time: " << total_transfer_in_time_ms << " ms\n";
	std::cout << "        bytes: " << total_transfer_in_bytes << " (" << (total_transfer_in_bytes/(1024*1024*1024)) << "GB)\n";
	std::cout << "transfer out time: " << total_transfer_out_time_ms << " ms\n";
		std::cout << "        bytes: " << total_transfer_out_bytes << " (" << (total_transfer_out_bytes/(1024*1024*1024)) << "GB)\n";
	std::cout << "*********************" << std::endl;
	std::cout << "Max block entries used: " << max_block_entries_copied_device_to_host << " VS HOST_MAX_BLOCK_ENTRIES:" << HOST_MAX_BLOCK_ENTRIES << std::endl;
	std::cout << " freeing memory...";
	freeMemory();
	std::cout << "end." << std::endl;
	exit(EXIT_SUCCESS);
}
