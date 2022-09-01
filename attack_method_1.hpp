/*
 * attack_method_1.hpp
 *
 *  Created on: Nov 2, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_1_HPP_
#define ATTACK_METHOD_1_HPP_




#define ATTACK_KBCFILTER_LR1LR2(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
		if ((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) { \
			uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L; \
			int slot = atomicAdd(&kbc_local_num_entries_L[local_kbc_bucket_id],1); \
			Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_L[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) { \
			uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R; \
			int slot = atomicAdd(&kbc_local_num_entries_R[local_kbc_bucket_id],1); \
			Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_R[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_L2) && (kbc_bucket_id <= KBC_END_L2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L2; \
					int slot = atomicAdd(&kbc_local_num_entries_L2[local_kbc_bucket_id],1); \
					Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_L2[entries_address] = entry; \
				} \
				if ((kbc_bucket_id >= KBC_START_R2) && (kbc_bucket_id <= KBC_END_R2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R2; \
					int slot = atomicAdd(&kbc_local_num_entries_R2[local_kbc_bucket_id],1); \
					Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_R2[entries_address] = entry; \
				} \
}

__global__
void gpu_chacha8_k32_kbc_ranges_LR1LR2(const uint32_t N,
		const __restrict__ uint32_t *input,
		Tx_Bucketed_Meta1 *kbc_local_entries_L, int *kbc_local_num_entries_L, uint32_t KBC_START_L, uint32_t KBC_END_L,
		Tx_Bucketed_Meta1 *kbc_local_entries_R, int *kbc_local_num_entries_R, uint32_t KBC_START_R, uint32_t KBC_END_R,
		Tx_Bucketed_Meta1 *kbc_local_entries_L2, int *kbc_local_num_entries_L2, uint32_t KBC_START_L2, uint32_t KBC_END_L2,
		Tx_Bucketed_Meta1 *kbc_local_entries_R2, int *kbc_local_num_entries_R2, uint32_t KBC_START_R2, uint32_t KBC_END_R2)
{
	uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group

	for (uint32_t x_group = index; x_group <= end_n; x_group += stride) {
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
		x10 += input[10];x11 += input[11];x12 += x_group; // j12;//x13 += 0;
		x14 += input[14];x15 += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(x0);BYTESWAP32(x1);BYTESWAP32(x2);BYTESWAP32(x3);BYTESWAP32(x4);BYTESWAP32(x5);
		BYTESWAP32(x6);BYTESWAP32(x7);BYTESWAP32(x8);BYTESWAP32(x9);BYTESWAP32(x10);BYTESWAP32(x11);
		BYTESWAP32(x12);BYTESWAP32(x13);BYTESWAP32(x14);BYTESWAP32(x15);

		//uint64_t y = x0 << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = x0 >> 22; // gives bucket id 0..1023
		ATTACK_KBCFILTER_LR1LR2(x0,0);ATTACK_KBCFILTER_LR1LR2(x1,1);ATTACK_KBCFILTER_LR1LR2(x2,2);ATTACK_KBCFILTER_LR1LR2(x3,3);
		ATTACK_KBCFILTER_LR1LR2(x4,4);ATTACK_KBCFILTER_LR1LR2(x5,5);ATTACK_KBCFILTER_LR1LR2(x6,6);ATTACK_KBCFILTER_LR1LR2(x7,7);
		ATTACK_KBCFILTER_LR1LR2(x8,8);ATTACK_KBCFILTER_LR1LR2(x9,9);ATTACK_KBCFILTER_LR1LR2(x10,10);ATTACK_KBCFILTER_LR1LR2(x11,11);
		ATTACK_KBCFILTER_LR1LR2(x12,12);ATTACK_KBCFILTER_LR1LR2(x13,13);ATTACK_KBCFILTER_LR1LR2(x14,14);ATTACK_KBCFILTER_LR1LR2(x15,15);
	}
}

template <typename BUCKETED_ENTRY>
__global__
void gpu_attack_merge_block_buckets_into_kbc_buckets_with_kbc_count_limit(
		const uint32_t KBC_START_ID, // determined by batch_id
		const BUCKETED_ENTRY *in, uint64_t batch_bucket_add_Y, const uint32_t N,
		BUCKETED_ENTRY *local_kbc_entries, int *local_kbc_counts,
		const uint32_t MAX_KBC_ENTRIES)
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
		uint32_t destination_address = local_kbc_id * MAX_KBC_ENTRIES + slot;

		//printf("block_id:%u [i: %u] entry.y:%u  kbc_id:%u   local_kbc:%u   slot:%u   dest:%u\n",
		//		block_id, i, block_entry.y, kbc_id, local_kbc_id, slot, destination_address);

		if (slot > MAX_KBC_ENTRIES) {
			printf("OVERFLOW: slot > MAX ENTRIES PER BUCKET\n");
		}
		//if (destination_address > DEVICE_BUFFER_ALLOCATED_ENTRIES) {
		//	printf("OVERFLOW: destination_address overflow > DEVICE_BUFFER_ALLOCATED_ENTRIES %u\n", destination_address);
		//}
		block_entry.y = calc_y % kBC; // hah! Don't forget to map it to kbc bucket form.
		local_kbc_entries[destination_address] = block_entry;
	}
}

__global__
void gpu_display_t2_match_results(Tx_Bucketed_Meta4 *T2_batch_match_results, int *device_T2_block_entry_counts, uint32_t MAX_ENTRIES_PER_BLOCK) {
	printf("GPU DISPLAY T2 MATCH RESULTS:\n");
	int total_counts = 0;
	for (int i=0;i<BATCHES;i++) {
		int num_results = device_T2_block_entry_counts[i];
		total_counts += num_results;
		uint32_t block_start_entry = MAX_ENTRIES_PER_BLOCK * i;
		for (int j=0;j<num_results;j++) {
			Tx_Bucketed_Meta4 entry = T2_batch_match_results[block_start_entry + j];
			if ((entry.meta[0] == 1320788535) || (entry.meta[0] == 434033488)) {
				printf("  block %d entry %d   x1:%u  x2:%u  x3:%u  x4:%u\n", i, j, entry.meta[0], entry.meta[1], entry.meta[2], entry.meta[3]);
			}
		}
	}
	printf("  TOTAL: %d\n", total_counts);
}

// std::vector<uint32_t> solution_xs = {1320788535,3465356684,2131394289,606438761,434033488,2479909174,3785038649,1942582046,438483300,2306941967,2327418650,184663264,3396904066,3057226705,2120150435,441715922,10459628,1281656413,88943898,810187686,112052271,2540716951,3073359813,4019528057,504026248,1706169436,2772410422,1772771468,607317630,4168020964,4286528917,2472944651,3546546119,1799281226,1202952199,1278165962,4062613743,2747217422,1182029562,1339760739,613483600,3661736730,1251588944,3140803170,2503085418,2541929248,4159128725,2325034733,4257771109,2804935474,2997421030,150533389,709945445,4159463930,714122558,1939000200,3291628318,1878268201,2874051942,2826426895,2146970589,4276159281,3509962078,2808839331};
/*
 * Pair 0 x:1320788535 y:76835538515  kBC:5084069
  Pair 1 x:3465356684 y:76835558195  kBC:5084070

  Pair 2 x:2131394289 y:227752410271  kBC:15069966
  Pair 3 x:606438761 y:227752417481  kBC:15069967

  Pair 4 x:434033488 y:274225910406  kBC:18145034
  Pair 5 x:2479909174 y:274225916708  kBC:18145035

  Pair 6 x:3785038649 y:213830149496  kBC:14148756
  Pair 7 x:1942582046 y:213830170524  kBC:14148757

  Pair 8 x:438483300 y:248522697030  kBC:16444299
  Pair 9 x:2306941967 y:248522719906  kBC:16444300
  Pair 10 x:2327418650 y:23832869730  kBC:1576978
  Pair 11 x:184663264 y:23832892290  kBC:1576979
  Pair 12 x:3396904066 y:31837336818  kBC:2106619
  Pair 13 x:3057226705 y:31837353261  kBC:2106620
  Pair 14 x:2120150435 y:22313127263  kBC:1476419
  Pair 15 x:441715922 y:22313149126  kBC:1476420
 */
void attack_method_1(uint32_t bits) {

	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();

	uint64_t BITS_DIVISOR = 1 << bits;

	uint64_t target_kbc_L1 = 5084069;
	uint64_t target_kbc_R1 = 15069966;
	uint64_t bucket_L1 = ((target_kbc_L1 + 1) * BITS_DIVISOR) / kBC_NUM_BUCKETS;
	uint64_t bucket_R1 = ((target_kbc_R1 + 1) * BITS_DIVISOR) / kBC_NUM_BUCKETS;
	uint64_t KBC_START_L1 = (bucket_L1*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_END_L1 = ((bucket_L1+1)*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_START_R1 = (bucket_R1*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_END_R1 = ((bucket_R1+1)*kBC_NUM_BUCKETS) / BITS_DIVISOR;

	uint64_t target_kbc_L2 = 18145034;
	uint64_t target_kbc_R2 = 14148756;
	uint64_t bucket_L2 = ((target_kbc_L2 + 1) * BITS_DIVISOR) / kBC_NUM_BUCKETS;
	uint64_t bucket_R2 = ((target_kbc_R2 + 1) * BITS_DIVISOR) / kBC_NUM_BUCKETS;
	uint64_t KBC_START_L2 = (bucket_L2*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_END_L2 = ((bucket_L2+1)*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_START_R2 = (bucket_R2*kBC_NUM_BUCKETS) / BITS_DIVISOR;
	uint64_t KBC_END_R2 = ((bucket_R2+1)*kBC_NUM_BUCKETS) / BITS_DIVISOR;

	//Pair 0 x:1320788535 y:76835538515  kBC:5084069
	//  Pair 1 x:3465356684 y:76835558195  kBC:5084070
	//  Pair 2 x:2131394289 y:227752410271  kBC:15069966
	//  Pair 3 x:606438761 y:227752417481  kBC:15069967

	uint64_t KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS; // +1 is for including last R bucket space

	uint64_t MAX_KBCS_POST_T1 = 16; // reduce if smaller selection based on initial t0 range.
	uint32_t BLOCK_MAX_ENTRIES_T2 = HOST_MAX_BLOCK_ENTRIES / 16;
	//uint32_t NUM_EXPECTED_ENTRIES_T1_MATCHES = 67108864;
	uint32_t NUM_EXPECTED_ENTRIES_T2_MATCHES = 1048576;
	if (bits == 6) {
		KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS;
		//NUM_EXPECTED_ENTRIES_T1_MATCHES = 67108864;
		MAX_KBCS_POST_T1 = 16;
		NUM_EXPECTED_ENTRIES_T2_MATCHES = 1048576;
		BLOCK_MAX_ENTRIES_T2 = NUM_EXPECTED_ENTRIES_T2_MATCHES / 32;
	} else if (bits == 7) {
		KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS / 2;
		//NUM_EXPECTED_ENTRIES_T1_MATCHES = 33554432;
		MAX_KBCS_POST_T1 = 14;
		NUM_EXPECTED_ENTRIES_T2_MATCHES = 262144;
		BLOCK_MAX_ENTRIES_T2 = NUM_EXPECTED_ENTRIES_T2_MATCHES / 32;
	} else if (bits == 8) {
		KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS / 4;
		//NUM_EXPECTED_ENTRIES_T1_MATCHES = 16777216;
		MAX_KBCS_POST_T1 = 12;
		NUM_EXPECTED_ENTRIES_T2_MATCHES = 65536;
		BLOCK_MAX_ENTRIES_T2 = NUM_EXPECTED_ENTRIES_T2_MATCHES / 32;
	} else if (bits == 9) {
		KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS / 8;
		//NUM_EXPECTED_ENTRIES_T1_MATCHES = 8388608;
		MAX_KBCS_POST_T1 = 10;
		NUM_EXPECTED_ENTRIES_T2_MATCHES = 16384;
		BLOCK_MAX_ENTRIES_T2 = NUM_EXPECTED_ENTRIES_T2_MATCHES / 32;
	} else if (bits == 10) {
		KBC_ATTACK_NUM_BUCKETS = KBC_LOCAL_NUM_BUCKETS / 16;
		//NUM_EXPECTED_ENTRIES_T1_MATCHES = 4194304;
		MAX_KBCS_POST_T1 = 8;
		NUM_EXPECTED_ENTRIES_T2_MATCHES = 4096;
		BLOCK_MAX_ENTRIES_T2 = NUM_EXPECTED_ENTRIES_T2_MATCHES / 32;
	}

	std::cout << "Attack Method 1 " << std::endl
			  << "   L0 kbc range " << KBC_START_L1 << " to " << KBC_END_L1 << " = " << (KBC_END_L1-KBC_START_L1) << "kbcs " << (100.0*(double)(KBC_END_L1-KBC_START_L1)/(double)kBC_LAST_BUCKET_ID) << "%" << std::endl
			  << "   R0 kbc range " << KBC_START_R1 << " to " << KBC_END_R1 << " = " << (KBC_END_R1-KBC_START_R1) << "kbcs " << (100.0*(double)(KBC_END_R1-KBC_START_R1)/(double)kBC_LAST_BUCKET_ID) << "%" << std::endl
			  << "   KBC_ATTACK_NUM_BUCKETS: " << KBC_ATTACK_NUM_BUCKETS << std::endl
			  << "   MAX BCS POST T1: " << MAX_KBCS_POST_T1 << std::endl
			  << "   BLOCK_MAX_ENTRIES_T2: " << BLOCK_MAX_ENTRIES_T2 << std::endl;


	char *device_buffer;

	int* device_local_kbc_num_entries_L1;
	int* device_local_kbc_num_entries_R1;
	int* device_local_kbc_num_entries_L2;
	int* device_local_kbc_num_entries_R2;
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_L1;
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_R1;
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_L2;
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_R2;

	int* device_block_entry_counts_L;
	int* device_block_entry_counts_R;
	Tx_Bucketed_Meta2 *T1_L_batch_match_results;
	Tx_Bucketed_Meta2 *T1_R_batch_match_results;

	int* device_T2_block_entry_counts;
	Tx_Bucketed_Meta4 *T2_batch_match_results;


	const uint64_t T0_KBC_DEVICE_BUFFER_ALLOCATED_ENTRIES = KBC_ATTACK_NUM_BUCKETS * KBC_MAX_ENTRIES_PER_BUCKET;

	const uint64_t CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED = T0_KBC_DEVICE_BUFFER_ALLOCATED_ENTRIES * sizeof(Tx_Bucketed_Meta2);
		std::cout << "  CHACHA BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED: " << CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED << std::endl;
		std::cout << "                                               * 4 =  " << (CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED * 4) << std::endl;
	const uint64_t T1_BATCH_MATCH_RESULTS_BYTES_NEEDED = DEVICE_BUFFER_ALLOCATED_ENTRIES * sizeof(Tx_Bucketed_Meta2);
	std::cout << "KBC RESULTS T1 L NEEDED: " << T1_BATCH_MATCH_RESULTS_BYTES_NEEDED << std::endl;
	const uint64_t T2_BATCH_MATCH_RESULTS_BYTES_NEEDED = (BLOCK_MAX_ENTRIES_T2 * BATCHES) * sizeof(Tx_Bucketed_Meta4);
		std::cout << "  T2_BATCH_MATCH_RESULTS_BYTES_NEEDED: " << T2_BATCH_MATCH_RESULTS_BYTES_NEEDED << std::endl;


	const uint64_t TOTAL_BYTES_NEEDED =
			      4 * CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED
				+ 2 * T1_BATCH_MATCH_RESULTS_BYTES_NEEDED
				+     T2_BATCH_MATCH_RESULTS_BYTES_NEEDED;

	std::cout << "      device_buffer TOTAL BYTES: " <<  TOTAL_BYTES_NEEDED << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer, TOTAL_BYTES_NEEDED));
	uint64_t MEM_POS = 0;

	T0_local_kbc_entries_L1 = (Tx_Bucketed_Meta1 *) &device_buffer[MEM_POS];
	T0_local_kbc_entries_R1 = (Tx_Bucketed_Meta1 *) &device_buffer[MEM_POS + CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED];
	T0_local_kbc_entries_L2 = (Tx_Bucketed_Meta1 *) &device_buffer[MEM_POS + CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED*2];
	T0_local_kbc_entries_R2 = (Tx_Bucketed_Meta1 *) &device_buffer[MEM_POS + CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED*3];
	MEM_POS += 4 * CHACHA_LOCAL_KBC_ENTRIES_BYTES_NEEDED;

	T1_L_batch_match_results = (Tx_Bucketed_Meta2 *) &device_buffer[MEM_POS];
	T1_R_batch_match_results = (Tx_Bucketed_Meta2 *) &device_buffer[MEM_POS + T1_BATCH_MATCH_RESULTS_BYTES_NEEDED];
	MEM_POS += 2 * T1_BATCH_MATCH_RESULTS_BYTES_NEEDED;
	T2_batch_match_results = (Tx_Bucketed_Meta4 *) &device_buffer[MEM_POS];
	MEM_POS +=  T2_BATCH_MATCH_RESULTS_BYTES_NEEDED;

	std::cout << "      device_block_entry_counts_L (" << BATCHES << "): " << BATCHES << " size:" << (sizeof(int)*BATCHES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_block_entry_counts_L, BATCHES*sizeof(int)));
	std::cout << "      device_block_entry_counts_R (" << BATCHES << "): " << BATCHES << " size:" << (sizeof(int)*BATCHES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_block_entry_counts_R, BATCHES*sizeof(int)));
	std::cout << "      device_T2_block_entry_counts (" << BATCHES << "): " << BATCHES << " size:" << (sizeof(int)*BATCHES) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_T2_block_entry_counts, BATCHES*sizeof(int)));


	auto alloc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   alloc time: " << std::chrono::duration_cast<milli>(alloc_finish - attack_start).count() << " ms\n";

	auto compute_only_start = std::chrono::high_resolution_clock::now();
	std::cout << "Doing chacha\n";


	int blockSize = 128; // # of threads per block, maximum is 1024.
	const uint64_t calc_N = UINT_MAX;
	const uint64_t calc_blockSize = blockSize;
	const uint64_t calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
	int numBlocks = calc_numBlocks;



	// don't forget to clear counter...will only use a portion of this memory so should be fast access.
	std::cout << "      device_local_kbc_num_entries_L1 " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_L1, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_R1 " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_R1, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_L2 " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_L2, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_R2 " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_R2, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));

	std::cout << "Doing T1" << std::endl;

	// we use only attack range for local num buckets
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L1, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R1, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L2, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R2, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));

	auto t1_start = std::chrono::high_resolution_clock::now();
	auto chacha_start = std::chrono::high_resolution_clock::now();
	gpu_chacha8_k32_kbc_ranges_LR1LR2<<<numBlocks, blockSize>>>(calc_N, chacha_input,
			T0_local_kbc_entries_L1, device_local_kbc_num_entries_L1, KBC_START_L1, KBC_END_L1,
			T0_local_kbc_entries_R1, device_local_kbc_num_entries_R1, KBC_START_R1, KBC_END_R1,
			T0_local_kbc_entries_L2, device_local_kbc_num_entries_L2, KBC_START_L2, KBC_END_L2,
			T0_local_kbc_entries_R2, device_local_kbc_num_entries_R2, KBC_START_R2, KBC_END_R2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto chacha_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   chacha L1 time: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";

	auto matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts_L, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599
	gpu_attack_find_t1_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_L1 - KBC_START_L1), 256>>>(1, KBC_START_L1, KBC_END_L1,
				T0_local_kbc_entries_L1, device_local_kbc_num_entries_L1,
				T1_L_batch_match_results, device_block_entry_counts_L);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 L time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";

	matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts_R, 0, (BATCHES)*sizeof(int)));
	gpu_attack_find_t1_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_R1 - KBC_START_R1), 256>>>(1, KBC_START_R1, KBC_END_R1,
				T0_local_kbc_entries_R1, device_local_kbc_num_entries_R1,
				T1_R_batch_match_results, device_block_entry_counts_R);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 R time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";

	auto t1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "      T1 total time: " << std::chrono::duration_cast<milli>(t1_finish - t1_start).count() << " ms\n";



	auto mergekbcs_start = std::chrono::high_resolution_clock::now();
	// clear our local kbc num entries as these will be written with new data

	// don't use T0 buckets anymore, so overwrite/reuse their memory space.
	Tx_Bucketed_Meta2 *T1_local_kbc_entries_L = (Tx_Bucketed_Meta2 *) &T0_local_kbc_entries_L1[0];
	Tx_Bucketed_Meta2 *T1_local_kbc_entries_R = (Tx_Bucketed_Meta2 *) &T0_local_kbc_entries_R1[0];

	// clump block-0-batch_id_L block-0-batch_id_R into same group and solve.

	auto matchTx_start = std::chrono::high_resolution_clock::now();
	auto matchTx_finish = std::chrono::high_resolution_clock::now();
	auto mergeTx_start = std::chrono::high_resolution_clock::now();
	auto mergeTx_finish = std::chrono::high_resolution_clock::now();
	uint64_t total_match_time_micros = 0;
	uint64_t total_merge_time_micros = 0;
	uint32_t global_block_counts[BATCHES] = {0};
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
		CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L1, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
		CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R1, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
		uint32_t KBC_MERGE_BUCKET_START = MIN_KBC_BUCKET_FOR_BATCH(block_id);
		const uint32_t KBC_START = MIN_KBC_BUCKET_FOR_BATCH(block_id);
		const uint32_t KBC_END = MIN_KBC_BUCKET_FOR_BATCH(block_id+1);

		uint32_t num_entries_to_copy = device_block_entry_counts_L[block_id];
		int blockSize = 256;
		int numBlocks = (num_entries_to_copy + blockSize - 1) / (blockSize);
		uint64_t batch_bucket_add_Y = CALC_BATCH_BUCKET_ADD_Y(block_id);//(((uint64_t) 1) << (38-6)) * ((uint64_t) batch_id);

		uint32_t block_address = block_id * HOST_MAX_BLOCK_ENTRIES;
		Tx_Bucketed_Meta2 *in = &T1_L_batch_match_results[block_address];

		//std::cout << "batch " << batch_id << " num_entries: " << num_entries_to_copy << std::endl;
		mergeTx_start = std::chrono::high_resolution_clock::now();
		gpu_attack_merge_block_buckets_into_kbc_buckets_with_kbc_count_limit<Tx_Bucketed_Meta2><<<numBlocks,blockSize>>>(
					KBC_MERGE_BUCKET_START,
					in, batch_bucket_add_Y, num_entries_to_copy,
					T1_local_kbc_entries_L, device_local_kbc_num_entries_L1,
					MAX_KBCS_POST_T1);

		num_entries_to_copy = device_block_entry_counts_R[block_id];
		numBlocks = (num_entries_to_copy + blockSize - 1) / (blockSize);
		in = &T1_R_batch_match_results[block_address];

			//std::cout << "batch " << batch_id << " num_entries: " << num_entries_to_copy << std::endl;
		gpu_attack_merge_block_buckets_into_kbc_buckets_with_kbc_count_limit<Tx_Bucketed_Meta2><<<numBlocks,blockSize>>>(
						KBC_MERGE_BUCKET_START,
						in, batch_bucket_add_Y, num_entries_to_copy,
						T1_local_kbc_entries_R, device_local_kbc_num_entries_R1,
						MAX_KBCS_POST_T1);

		// TODO: find matches in entries_L against entries_R...should be <16, avg around 3-4
		// only have 2m entries...so...could sort 1mL's against 1mR's?
		//auto matchTx_start = std::chrono::high_resolution_clock::now();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		mergeTx_finish = std::chrono::high_resolution_clock::now();
		total_merge_time_micros += std::chrono::duration_cast< std::chrono::microseconds >( mergeTx_finish - mergeTx_start ).count();


		/*CUDA_CHECK_RETURN(cudaMemset(device_T2_block_entry_counts, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599


		matchTx_start = std::chrono::high_resolution_clock::now();
		gpu_attack_find_tx_LR_matches<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<(KBC_END - KBC_START), 8>>>(1, KBC_START, KBC_END,
						T1_local_kbc_entries_L, device_local_kbc_num_entries_L1,
						T1_local_kbc_entries_R, device_local_kbc_num_entries_R1,
						T2_batch_match_results, device_T2_block_entry_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		matchTx_finish = std::chrono::high_resolution_clock::now();
		total_match_time_micros += std::chrono::duration_cast< std::chrono::microseconds >( matchTx_finish - matchTx_start ).count();
*/
		//total_match_time_ms += std::chrono::duration_cast<microseconds>(matchTx_finish - matchTx_start).count();
		//for (int i = 0; i < BATCHES; i++) {
		//	global_block_counts[i] += device_T2_block_entry_counts[i];
		//}

	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	std::cout << "   match t2 LR sum time: " << (total_match_time_micros/1000) << "ms" << std::endl;
	std::cout << "   merge t2 LR sum time: " << (total_merge_time_micros/1000) << "ms" << std::endl;
	auto mergekbcs_finish = std::chrono::high_resolution_clock::now();
	std::cout << "      T2 total time: " << std::chrono::duration_cast<milli>(mergekbcs_finish - mergekbcs_start).count() << " ms\n";
		//gpu_list_local_kbc_entries<<<1,1>>>(device_local_kbc_num_entries_L);


	auto compute_only_finish = std::chrono::high_resolution_clock::now();


	uint32_t total_counts = 0;
	for (int i=0;i<BATCHES;i++) {
		//std::cout << " device_T2_block_entry_counts[" << i << "] : " << device_T2_block_entry_counts[i] << std::endl;
		//total_counts += global_block_counts[i];
	}
	std::cout << "Total T2 block entry counts: " << total_counts << std::endl;

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_L1));
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_R1));
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_L2));
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_R2));
	//CUDA_CHECK_RETURN(cudaFree(device_block_entry_counts));
	CUDA_CHECK_RETURN(cudaFree(device_buffer));

	auto attack_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "   attack total time: " << std::chrono::duration_cast<milli>(attack_finish - attack_start).count() << " ms\n";
	std::cout << "end." << std::endl;
}


#endif /* ATTACK_METHOD_1_HPP_ */
