/*
 * attack.hpp
 *
 *  Created on: Oct 26, 2021
 *      Author: nick
 */

#ifndef ATTACK_HPP_
#define ATTACK_HPP_


#include "nick_blake3.hpp"
//#include <thrust/device_ptr.h>
//#include <thrust/sort.h>
//#include <thrust/unique.h>

#define ATTACK_KBCFILTER(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	if ((kbc_bucket_id >= KBC_START) && (kbc_bucket_id <= KBC_END)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START; \
		int slot = atomicAdd(&kbc_local_num_entries[local_kbc_bucket_id],1); \
		Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		kbc_local_entries[entries_address] = entry; \
	} \
}

__global__
void gpu_chacha8_k32_kbc_ranges(const uint32_t N,
		const __restrict__ uint32_t *input, Tx_Bucketed_Meta1 *kbc_local_entries, int *kbc_local_num_entries,
		uint32_t KBC_START, uint32_t KBC_END)
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
		ATTACK_KBCFILTER(x0,0);ATTACK_KBCFILTER(x1,1);ATTACK_KBCFILTER(x2,2);ATTACK_KBCFILTER(x3,3);
		ATTACK_KBCFILTER(x4,4);ATTACK_KBCFILTER(x5,5);ATTACK_KBCFILTER(x6,6);ATTACK_KBCFILTER(x7,7);
		ATTACK_KBCFILTER(x8,8);ATTACK_KBCFILTER(x9,9);ATTACK_KBCFILTER(x10,10);ATTACK_KBCFILTER(x11,11);
		ATTACK_KBCFILTER(x12,12);ATTACK_KBCFILTER(x13,13);ATTACK_KBCFILTER(x14,14);ATTACK_KBCFILTER(x15,15);
	}
}

__device__ int gpu_xs_L_count = 0;
__device__ int gpu_xs_R_count = 0;

#define ATTACK_WRITEXS_LR(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	if ((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) { \
		int slot = atomicAdd(&local_num_xs_L,1); \
		local_xs_L[slot] = x+i; \
		local_ys_L[slot] = chacha_y; \
	} \
	if ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) { \
		int slot = atomicAdd(&local_num_xs_R,1); \
		local_xs_R[slot] = x+i; \
		local_ys_R[slot] = chacha_y; \
	} \
}

__global__
void gpu_chacha8_k32_kbc_ranges_LR_write_xy(const uint32_t N,
		const __restrict__ uint32_t *input,
		uint32_t *xs_L, uint32_t *ys_L, uint32_t *xs_L_count, uint32_t KBC_START_L, uint32_t KBC_END_L,
		uint32_t *xs_R, uint32_t *ys_R, uint32_t *xs_R_count, uint32_t KBC_START_R, uint32_t KBC_END_R)
{
	__shared__ uint32_t local_xs_L[256];
	__shared__ uint32_t local_ys_L[256];
	__shared__ uint32_t local_xs_R[256];
	__shared__ uint32_t local_ys_R[256];
	__shared__ int local_num_xs_L;
	__shared__ int local_num_xs_R;
	__shared__ int global_L_slot;
	__shared__ int global_R_slot;

	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group

	if (threadIdx.x == 0) {
		local_num_xs_L = 0;
		local_num_xs_R = 0;
	}
	__syncthreads();
	const int j = 33*threadIdx.x;
	for (uint32_t x_group = index; x_group <= end_n; x_group += stride) {
		uint32_t x = x_group << 4;//  *16;
		uint32_t pos = x_group;

		datax[j+0] = input[0];datax[j+1] = input[1];datax[j+2] = input[2];datax[j+3] = input[3];datax[j+4] = input[4];datax[j+5] = input[5];datax[j+6] = input[6];datax[j+7] = input[7];
		datax[j+8] = input[8];datax[j+9] = input[9];datax[j+10] = input[10];datax[j+11] = input[11];
		datax[j+12] = pos; datax[j+13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[j+14] = input[14];datax[j+15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[j+0], datax[j+4], datax[j+8], datax[j+12]);QUARTERROUND(datax[j+1], datax[j+5], datax[j+9], datax[j+13]);
			QUARTERROUND(datax[j+2], datax[j+6], datax[j+10], datax[j+14]);QUARTERROUND(datax[j+3], datax[j+7], datax[j+11], datax[j+15]);
			QUARTERROUND(datax[j+0], datax[j+5], datax[j+10], datax[j+15]);QUARTERROUND(datax[j+1], datax[j+6], datax[j+11], datax[j+12]);
			QUARTERROUND(datax[j+2], datax[j+7], datax[j+8], datax[j+13]);QUARTERROUND(datax[j+3], datax[j+4], datax[j+9], datax[j+14]);
		}

		datax[j+0] += input[0];datax[j+1] += input[1];datax[j+2] += input[2];datax[j+3] += input[3];datax[j+4] += input[4];
		datax[j+5] += input[5];datax[j+6] += input[6];datax[j+7] += input[7];datax[j+8] += input[8];datax[j+9] += input[9];
		datax[j+10] += input[10];datax[j+11] += input[11];datax[j+12] += x_group; // j12;//datax[j+13] += 0;
		datax[j+14] += input[14];datax[j+15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[j+0]);BYTESWAP32(datax[j+1]);BYTESWAP32(datax[j+2]);BYTESWAP32(datax[j+3]);BYTESWAP32(datax[j+4]);BYTESWAP32(datax[j+5]);
		BYTESWAP32(datax[j+6]);BYTESWAP32(datax[j+7]);BYTESWAP32(datax[j+8]);BYTESWAP32(datax[j+9]);BYTESWAP32(datax[j+10]);BYTESWAP32(datax[j+11]);
		BYTESWAP32(datax[j+12]);BYTESWAP32(datax[j+13]);BYTESWAP32(datax[j+14]);BYTESWAP32(datax[j+15]);

		//uint64_t y = datax[j+0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[j+0] >> 22; // gives bucket id 0..1023
		ATTACK_WRITEXS_LR(datax[j+0],0);ATTACK_WRITEXS_LR(datax[j+1],1);ATTACK_WRITEXS_LR(datax[j+2],2);ATTACK_WRITEXS_LR(datax[j+3],3);
		ATTACK_WRITEXS_LR(datax[j+4],4);ATTACK_WRITEXS_LR(datax[j+5],5);ATTACK_WRITEXS_LR(datax[j+6],6);ATTACK_WRITEXS_LR(datax[j+7],7);
		ATTACK_WRITEXS_LR(datax[j+8],8);ATTACK_WRITEXS_LR(datax[j+9],9);ATTACK_WRITEXS_LR(datax[j+10],10);ATTACK_WRITEXS_LR(datax[j+11],11);
		ATTACK_WRITEXS_LR(datax[j+12],12);ATTACK_WRITEXS_LR(datax[j+13],13);ATTACK_WRITEXS_LR(datax[j+14],14);ATTACK_WRITEXS_LR(datax[j+15],15);

	}
	// without global writes it has maximum speed of 21ms
	// these global writes up it to 26ms.
	// hope here is that sorting won't take long, so that sorted entries are under total 35ms
	// and then the matching *should* be quicker than when it's bucketed
	__syncthreads();
	if (threadIdx.x == 0) {
		//printf("finished with %u %u counts\n", local_num_xs_L, local_num_xs_R);
		global_L_slot = atomicAdd(&xs_L_count[0],local_num_xs_L);
		global_R_slot = atomicAdd(&xs_R_count[0],local_num_xs_R);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < local_num_xs_L; i+=blockDim.x) {
		xs_L[i+global_L_slot] = local_xs_L[i];
	}
	for (int i = threadIdx.x; i < local_num_xs_L; i+=blockDim.x) {
		ys_L[i+global_L_slot] = local_ys_L[i];
	}
	for (int i = threadIdx.x; i < local_num_xs_R; i+=blockDim.x) {
		xs_R[i+global_R_slot] = local_xs_R[i];
	}
	for (int i = threadIdx.x; i < local_num_xs_R; i+=blockDim.x) {
		ys_R[i+global_R_slot] = local_ys_R[i];
	}
}

__global__
void gpu_merge_f1xypairs_into_kbc_buckets(
		const uint32_t KBC_START_ID, // determined by batch_id
		const uint64_t *in, const uint32_t N,
		Tx_Bucketed_Meta1 *local_kbc_entries, int *local_kbc_counts)
{
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	//for (int i = 0; i < N; i++) {

	if (i < N) {
		uint64_t value = in[i];
		uint32_t x = value >> 32;
		uint32_t chacha_y = value;
		uint64_t calc_y = (((uint64_t) chacha_y) << 6) + (x >> 26);
		uint32_t kbc_id = calc_y / kBC;
		uint32_t KBC_END_ID = KBC_START_ID + KBC_LOCAL_NUM_BUCKETS / 256;
		if ((kbc_id >= KBC_START_ID) || (kbc_id < KBC_END_ID)) {


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
		Tx_Bucketed_Meta1 kbc_entry = {};
		kbc_entry.y = calc_y % kBC;
		kbc_entry.meta[0] = x;
		local_kbc_entries[destination_address] = kbc_entry;
		}
	}
}

__global__
void gpu_chacha8_k32_kbc_ranges_LR_write_xypairs(const uint32_t N,
		const __restrict__ uint32_t *input,
		uint64_t *xys_L, uint32_t *xs_L_count, uint32_t KBC_START_L, uint32_t KBC_END_L,
		uint64_t *xys_R, uint32_t *xs_R_count, uint32_t KBC_START_R, uint32_t KBC_END_R)
{
	__shared__ uint32_t local_xs_L[256];
	__shared__ uint32_t local_ys_L[256];
	__shared__ uint32_t local_xs_R[256];
	__shared__ uint32_t local_ys_R[256];
	__shared__ int local_num_xs_L;
	__shared__ int local_num_xs_R;
	__shared__ int global_L_slot;
	__shared__ int global_R_slot;

	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[256*17]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group

	if (threadIdx.x == 0) {
		local_num_xs_L = 0;
		local_num_xs_R = 0;
	}
	__syncthreads();
	const int j = 17*threadIdx.x;
	for (uint32_t x_group = index; x_group <= end_n; x_group += stride) {
		uint32_t x = x_group << 4;//  *16;
		uint32_t pos = x_group;

		datax[j+0] = input[0];datax[j+1] = input[1];datax[j+2] = input[2];datax[j+3] = input[3];datax[j+4] = input[4];datax[j+5] = input[5];datax[j+6] = input[6];datax[j+7] = input[7];
		datax[j+8] = input[8];datax[j+9] = input[9];datax[j+10] = input[10];datax[j+11] = input[11];
		datax[j+12] = pos; datax[j+13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[j+14] = input[14];datax[j+15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[j+0], datax[j+4], datax[j+8], datax[j+12]);QUARTERROUND(datax[j+1], datax[j+5], datax[j+9], datax[j+13]);
			QUARTERROUND(datax[j+2], datax[j+6], datax[j+10], datax[j+14]);QUARTERROUND(datax[j+3], datax[j+7], datax[j+11], datax[j+15]);
			QUARTERROUND(datax[j+0], datax[j+5], datax[j+10], datax[j+15]);QUARTERROUND(datax[j+1], datax[j+6], datax[j+11], datax[j+12]);
			QUARTERROUND(datax[j+2], datax[j+7], datax[j+8], datax[j+13]);QUARTERROUND(datax[j+3], datax[j+4], datax[j+9], datax[j+14]);
		}

		datax[j+0] += input[0];datax[j+1] += input[1];datax[j+2] += input[2];datax[j+3] += input[3];datax[j+4] += input[4];
		datax[j+5] += input[5];datax[j+6] += input[6];datax[j+7] += input[7];datax[j+8] += input[8];datax[j+9] += input[9];
		datax[j+10] += input[10];datax[j+11] += input[11];datax[j+12] += x_group; // j12;//datax[j+13] += 0;
		datax[j+14] += input[14];datax[j+15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[j+0]);BYTESWAP32(datax[j+1]);BYTESWAP32(datax[j+2]);BYTESWAP32(datax[j+3]);BYTESWAP32(datax[j+4]);BYTESWAP32(datax[j+5]);
		BYTESWAP32(datax[j+6]);BYTESWAP32(datax[j+7]);BYTESWAP32(datax[j+8]);BYTESWAP32(datax[j+9]);BYTESWAP32(datax[j+10]);BYTESWAP32(datax[j+11]);
		BYTESWAP32(datax[j+12]);BYTESWAP32(datax[j+13]);BYTESWAP32(datax[j+14]);BYTESWAP32(datax[j+15]);

		//uint64_t y = datax[j+0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[j+0] >> 22; // gives bucket id 0..1023
		ATTACK_WRITEXS_LR(datax[j+0],0);ATTACK_WRITEXS_LR(datax[j+1],1);ATTACK_WRITEXS_LR(datax[j+2],2);ATTACK_WRITEXS_LR(datax[j+3],3);
		ATTACK_WRITEXS_LR(datax[j+4],4);ATTACK_WRITEXS_LR(datax[j+5],5);ATTACK_WRITEXS_LR(datax[j+6],6);ATTACK_WRITEXS_LR(datax[j+7],7);
		ATTACK_WRITEXS_LR(datax[j+8],8);ATTACK_WRITEXS_LR(datax[j+9],9);ATTACK_WRITEXS_LR(datax[j+10],10);ATTACK_WRITEXS_LR(datax[j+11],11);
		ATTACK_WRITEXS_LR(datax[j+12],12);ATTACK_WRITEXS_LR(datax[j+13],13);ATTACK_WRITEXS_LR(datax[j+14],14);ATTACK_WRITEXS_LR(datax[j+15],15);

	}
	// without global writes it has maximum speed of 21ms
	// these global writes up it to 26ms.
	// hope here is that sorting won't take long, so that sorted entries are under total 35ms
	// and then the matching *should* be quicker than when it's bucketed
	__syncthreads();
	if (threadIdx.x == 0) {
		//printf("finished with %u %u counts\n", local_num_xs_L, local_num_xs_R);
		global_L_slot = atomicAdd(&xs_L_count[0],local_num_xs_L);
		global_R_slot = atomicAdd(&xs_R_count[0],local_num_xs_R);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < local_num_xs_L; i+=blockDim.x) {
		xys_L[i+global_L_slot] = (((uint64_t) local_xs_L[i]) << 32) + local_ys_L[i];
	}
	for (int i = threadIdx.x; i < local_num_xs_R; i+=blockDim.x) {
		xys_R[i+global_R_slot] = (((uint64_t) local_xs_R[i]) << 32) + local_ys_R[i];
	}
}


#define ATTACK_KBCFILTER_LR(chacha_y,i) \
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
}



__global__
void gpu_chacha8_k32_kbc_ranges_LR(const uint32_t N,
		const __restrict__ uint32_t *input,
		Tx_Bucketed_Meta1 *kbc_local_entries_L, int *kbc_local_num_entries_L, uint32_t KBC_START_L, uint32_t KBC_END_L,
		Tx_Bucketed_Meta1 *kbc_local_entries_R, int *kbc_local_num_entries_R, uint32_t KBC_START_R, uint32_t KBC_END_R)
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
		ATTACK_KBCFILTER_LR(x0,0);ATTACK_KBCFILTER_LR(x1,1);ATTACK_KBCFILTER_LR(x2,2);ATTACK_KBCFILTER_LR(x3,3);
		ATTACK_KBCFILTER_LR(x4,4);ATTACK_KBCFILTER_LR(x5,5);ATTACK_KBCFILTER_LR(x6,6);ATTACK_KBCFILTER_LR(x7,7);
		ATTACK_KBCFILTER_LR(x8,8);ATTACK_KBCFILTER_LR(x9,9);ATTACK_KBCFILTER_LR(x10,10);ATTACK_KBCFILTER_LR(x11,11);
		ATTACK_KBCFILTER_LR(x12,12);ATTACK_KBCFILTER_LR(x13,13);ATTACK_KBCFILTER_LR(x14,14);ATTACK_KBCFILTER_LR(x15,15);
	}
}



template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_find_t1_matches(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
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

	const uint8_t doPrint = 0;

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

			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
				//printf("Got y %llu idxL:%u idxR:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, match.idxL, match.idxR, L_Entry.meta[0], R_Entry.meta[0], blake_result);
			//}


			uint64_t batch_bucket = blake_result >> (38-6); // setting this to 0 (seq.) changes from 57ms to 48ms.
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
}



template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_find_t1_matches_out_kbc(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		BUCKETED_ENTRY_OUT *kbc_out, unsigned int *out_kbc_counts, const uint32_t MAX_KBC_ENTRIES) {
	// T1 match: 1714 ms -> with delaying extras: 1630
	//Total tables time: 73726 ms
	//        match: 10015 ms -> 9705ms with delaying extras
	const uint16_t NUM_RMAPS = (kBC/2)+1;
	__shared__ int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
	__shared__ uint32_t nick_rmap_extras_rl[32];
	__shared__ uint16_t nick_rmap_extras_ry[32];
	__shared__ uint16_t nick_rmap_extras_pos[32];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ BUCKETED_ENTRY_IN kbc_L_entries[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ BUCKETED_ENTRY_IN kbc_R_entries[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int num_extras;
	__shared__ int y_duplicate_counts;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	const uint8_t doPrint = 0;

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

	for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		kbc_R_entries[pos_R] = kbc_local_entries[start_R+pos_R];
	}
	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		kbc_L_entries[pos_L] = kbc_local_entries[start_L+pos_L];
	}


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
			//printf("match list\n");
			//for (int i=0;i<total_matches;i++) {
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



			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			//nick_blake3_old(pair.meta[0], pair.meta[1], calc_y, &blake_result); // adds 500ms
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);

			//uint32_t batch_bucket = blake_result >> (38-6); // setting this to 0 (seq.) changes from 57ms to 48ms.

			//if ((pair.meta[0] == 1320788535) || (pair.meta[0] == 2131394289)) {
			//	printf("Got y %llu batch:%u Lx: %u Rx: %u and f_result: %llu\n", calc_y, batch_bucket, L_Entry.meta[0], R_Entry.meta[0], blake_result);
			//}

			uint32_t kbc_bucket = blake_result / kBC;

			pair.y = (uint32_t) (blake_result % kBC);
		//if (batch_bucket == 49) {
			//int block_slot = atomicAdd(&out_kbc_counts[kbc_bucket],1);

			// slightly faster and more memory efficient anyway
			uint32_t kbc_bitmask_bucket = kbc_bucket / 8; \
			uint32_t kbc_bitmask_shift = 4*(kbc_bucket % 8); \
			unsigned int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
			unsigned int bitadd = atomicAdd(&out_kbc_counts[kbc_bitmask_bucket],kbc_bitmask_add); \
			uint32_t block_slot = bitadd; \
			block_slot = (block_slot >> (kbc_bitmask_shift)) & 0b01111; \

/*
 * Doing T1
   chacha L1 time: 35 ms
   match T1 L time: 18 ms
   match T1 R time: 18 ms
   match T2 L time: 22 ms
Freeing memory...
GPU DISPLAY T2 MATCH RESULTS:
  block 22 entry 3140   x1:1320788535  x2:3465356684  x3:2131394289  x4:606438761
  TOTAL: 262341

  Doing T1
   chacha L1 time: 36 ms
   match T1 L time: 19 ms
   match T1 R time: 19 ms
   match T2 L time: 22 ms
Freeing memory...
GPU DISPLAY T2 MATCH RESULTS:
  block 22 entry 3140   x1:1320788535  x2:3465356684  x3:2131394289  x4:606438761
  TOTAL: 262341
 */

			if (block_slot > MAX_KBC_ENTRIES) {
				printf("block_slot > MAX %u\n", block_slot);
			} else {
				uint32_t pair_address = kbc_bucket * MAX_KBC_ENTRIES + block_slot;
			//if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
				//printf("ERROR: results address overflow\n");
			//} else {
				kbc_out[pair_address] = pair;
			//}
			}
		//} // TOKENPOD


	}
}



template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_find_tx_LR_matches(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries_L, const int *kbc_local_num_entries_L,
		const BUCKETED_ENTRY_IN *kbc_local_entries_R, const int *kbc_local_num_entries_R,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts) {

	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET]; // TODO: this could be smaller
	__shared__ int total_matches;

	int kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;
	uint32_t global_kbc_L_bucket_id = kbc_L_bucket_id + start_kbc_L;

	const uint8_t doPrint = 0;

	if (gridDim.x != (end_kbc_R - start_kbc_L)) {
		printf("ERROR: GRIDDIM %u MUST EQUAL NUMBER OF KBCS TO SCAN %u\n", gridDim.x, end_kbc_R - start_kbc_L);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = kbc_L_bucket_id*KBC_MAX_ENTRIES_PER_BUCKET;
	const uint32_t start_R = (kbc_L_bucket_id+1)*KBC_MAX_ENTRIES_PER_BUCKET;
	const int num_L = kbc_local_num_entries_L[kbc_L_bucket_id];
	const int num_R = kbc_local_num_entries_R[(kbc_L_bucket_id+1)];
	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_local_entries_L[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_local_entries_R[start_R];

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
	if ((num_L == 0) || (num_R == 0)) {
		return;
	}

	__syncthreads(); // all written initialize data should sync

	//   For any 0 <= m < kExtraBitsPow:
	//   yl / kBC + 1 = yR / kBC   AND
	//   (yr % kBC) / kC - (yl % kBC) / kC = m   (mod kB)  AND
	//   (yr % kBC) % kC - (yl % kBC) % kC = (2m + (yl/kBC) % 2)^2   (mod kC)

	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
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
			uint32_t pair_address = batch_bucket * HOST_MAX_BLOCK_ENTRIES + block_slot;
			if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
				printf("ERROR: results address overflow\n");
			} else {
				//bucketed_out[pair_address] = pair;
			}
		}
	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_find_tx_LR_matches_global(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_global_entries_L, const unsigned int *kbc_global_num_entries_L,
		const BUCKETED_ENTRY_IN *kbc_global_entries_R, const unsigned int *kbc_global_num_entries_R,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts,
		uint32_t KBC_MAX_ENTRIES, uint32_t BLOCK_MAX_ENTRIES) {

	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET]; // TODO: this could be smaller
	__shared__ int total_matches;
	//__shared__ int num_L;
	//__shared__ int num_R;

	int global_kbc_L_bucket_id = blockIdx.x; // NOTE: localized so starts at 0... //  + start_kbc_L;

	const uint8_t doPrint = 0;

	if (gridDim.x != kBC_NUM_BUCKETS) {
		printf("ERROR: GRIDDIM %u MUST EQUAL KBC NUM BUCKETS %u\n", gridDim.x, kBC_NUM_BUCKETS);
	}
	int numThreadsInBlock = blockDim.x;
	int threadId = threadIdx.x;
	int threadStartScan = threadId;
	int threadSkipScan = numThreadsInBlock;

	const uint32_t start_L = global_kbc_L_bucket_id*KBC_MAX_ENTRIES;
	const uint32_t start_R = (global_kbc_L_bucket_id+1)*KBC_MAX_ENTRIES;


	//if (threadIdx.x == 0) {
		uint32_t kbc_bitmask_bucket = global_kbc_L_bucket_id / 8;
		uint32_t kbc_bitmask_shift = 4*(global_kbc_L_bucket_id % 8);
		uint32_t bitvalue = kbc_global_num_entries_L[kbc_bitmask_bucket];
		const unsigned int num_L = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;
	//}
	//if (threadIdx.x == 1) {
		kbc_bitmask_bucket = (global_kbc_L_bucket_id + 1) / 8;
		kbc_bitmask_shift = 4*((global_kbc_L_bucket_id + 1) % 8);
		bitvalue = kbc_global_num_entries_R[kbc_bitmask_bucket];
		const unsigned int num_R = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;
	//}
	//__syncthreads();
	//const int num_L = kbc_global_num_entries_L[global_kbc_L_bucket_id];
	//const int num_R = kbc_global_num_entries_R[(global_kbc_L_bucket_id+1)];
	if ((num_L == 0) || (num_R == 0)) {
		return;
	}

	const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_global_entries_L[start_L];
	const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_global_entries_R[start_R];

	if (threadIdx.x == 0) {
		total_matches = 0;
	}
	__syncthreads(); // all written initialize data should sync

	//   For any 0 <= m < kExtraBitsPow:
	//   yl / kBC + 1 = yR / kBC   AND
	//   (yr % kBC) / kC - (yl % kBC) / kC = m   (mod kB)  AND
	//   (yr % kBC) % kC - (yl % kBC) % kC = (2m + (yl/kBC) % 2)^2   (mod kC)

	for (int pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
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
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET);
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
			//printf("Got t2 match x1: %u x2: %u x3: %u x4: %u\n", L_Entry.meta[0], L_Entry.meta[1], R_Entry.meta[0], R_Entry.meta[1]);

			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
			//if (global_kbc_L_bucket_id == 1) {
			//	uint64_t Lx = (((uint64_t) pair.meta[0]) << 32) + pair.meta[1];
			//	uint64_t Rx = (((uint64_t) pair.meta[2]) << 32) + pair.meta[3];
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
			uint32_t pair_address = batch_bucket * BLOCK_MAX_ENTRIES + block_slot;
			//if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
			//	printf("ERROR: results address overflow\n");
			//} else {
				bucketed_out[pair_address] = pair;
			//}
		}
	}
}

template <typename BUCKETED_ENTRY>
__global__
void gpu_attack_merge_block_buckets_into_kbc_buckets(
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
		//uint32_t KBC_END_ID = KBC_START_ID + KBC_LOCAL_NUM_BUCKETS;
		//if ((kbc_id < KBC_START_ID) || (kbc_id > KBC_END_ID)) {
		//	printf(" i:%u  entry.y:%u  add_Y:%llu calc_y:%llu OUT OF RANGE: kbc id: %u   KBC_LOCAL_NUM_BUCKETS:%u START:%u  END:%u\n", i, block_entry.y, batch_bucket_add_Y, calc_y, kbc_id, KBC_LOCAL_NUM_BUCKETS, KBC_START_ID, KBC_END_ID);
		//}

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


__global__
void gpu_list_local_kbc_entries(int *kbc_num_entries, int from, int to, int skip) {
	for (int i=from;i<to;i+=skip) {
		int num = kbc_num_entries[i];
		printf("kbc %u : %u\n", i, num);
	}
}

__global__
void gpu_list_local_kbc_entries_bitmask(unsigned int *kbc_num_entries, int from, int to, int skip) {
	for (int i=from;i<to;i+=skip) {
		uint32_t kbc_bitmask_bucket = i / 8;
		uint32_t kbc_bitmask_shift = 4*(i % 8);
		uint32_t bitvalue = kbc_num_entries[kbc_bitmask_bucket];
		const unsigned int num = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;

		printf("kbc %u : %u\n", i, num);
	}
}

//#include "attack_method_kbc_list.hpp"
#include "attack_method_lxs.hpp"
//#include "attack_method_2.hpp" // this is current working one
//#include "attack_method_xpairbits.hpp"

void attack_it() {
	std::cout << "Attack it!" << std::endl;

	//uint32_t bits = 10;
	//attack_method_2(bits);


	//attack_method_xpairbits();
	attack_method_lxs(6000000);
	return;

	//auto sort_start = std::chrono::high_resolution_clock::now();
	//thrust::device_ptr<uint32_t> device_xs_L_ptr(device_xs_L);
	//thrust::device_ptr<uint32_t> device_ys_L_ptr(device_ys_L);
	//thrust::sort_by_key(device_ys_L_ptr, device_ys_L_ptr + xs_count_L[0], device_xs_L_ptr);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//auto sort_finish = std::chrono::high_resolution_clock::now();
	//std::cout << "   sort time: " << std::chrono::duration_cast<milli>(sort_finish - sort_start).count() << " ms\n";
	// why is 2nd sort 31ms and first sort 8ms!?!?
	//sort_start = std::chrono::high_resolution_clock::now();
	//thrust::device_ptr<uint32_t> device_xs_R_ptr(device_xs_R);
	//thrust::device_ptr<uint32_t> device_ys_R_ptr(device_ys_R);
	//thrust::sort_by_key(device_ys_R_ptr, device_ys_R_ptr + xs_count_R[0], device_xs_R_ptr);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//sort_finish = std::chrono::high_resolution_clock::now();
	//std::cout << "   sort time: " << std::chrono::duration_cast<milli>(sort_finish - sort_start).count() << " ms\n";


	/*auto matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts_L, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599
	gpu_attack_find_t1_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_L - KBC_START_L), 256>>>(1, batch_id_L, KBC_START_L, KBC_END_L,
			T0_local_kbc_entries_L, device_local_kbc_num_entries_L,
			T1_L_batch_match_results, device_block_entry_counts_L);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 L time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";

	matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts_R, 0, (BATCHES)*sizeof(int)));
	gpu_attack_find_t1_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_R - KBC_START_R), 256>>>(1, batch_id_R, KBC_START_R, KBC_END_R,
				T0_local_kbc_entries_R, device_local_kbc_num_entries_R,
				T1_R_batch_match_results, device_block_entry_counts_R);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 R time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";

	auto t1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "      T1 total time: " << std::chrono::duration_cast<milli>(t1_finish - t1_start).count() << " ms\n";



	auto mergekbcs_start = std::chrono::high_resolution_clock::now();
	// clear our local kbc num entries as these will be written with new data


	Tx_Bucketed_Meta2 *T1_local_kbc_entries_L = (Tx_Bucketed_Meta2 *) &device_local_kbc_entries_L[0]; // will replace...
	Tx_Bucketed_Meta2 *T1_local_kbc_entries_R = (Tx_Bucketed_Meta2 *) &device_local_kbc_entries_R[0];
	// clump block-0-batch_id_L block-0-batch_id_R into same group and solve.
	auto matchTx_start = std::chrono::high_resolution_clock::now();
	auto matchTx_finish = std::chrono::high_resolution_clock::now();
	auto mergeTx_start = std::chrono::high_resolution_clock::now();
	auto mergeTx_finish = std::chrono::high_resolution_clock::now();
	uint64_t total_match_time_micros = 0;
	uint64_t total_merge_time_micros = 0;
	uint32_t global_block_counts[BATCHES] = {0};
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
		CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
		CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));
		uint32_t KBC_MERGE_BUCKET_START = MIN_KBC_BUCKET_FOR_BATCH(block_id);
		uint32_t num_entries_to_copy = device_block_entry_counts_L[block_id];
		int blockSize = 256;
		int numBlocks = (num_entries_to_copy + blockSize - 1) / (blockSize);
		uint64_t batch_bucket_add_Y = CALC_BATCH_BUCKET_ADD_Y(block_id);//(((uint64_t) 1) << (38-6)) * ((uint64_t) batch_id);

		uint32_t block_address = block_id * HOST_MAX_BLOCK_ENTRIES;
		Tx_Bucketed_Meta2 *in = &T1_L_batch_match_results[block_address];

		//std::cout << "batch " << batch_id << " num_entries: " << num_entries_to_copy << std::endl;
		mergeTx_start = std::chrono::high_resolution_clock::now();
		gpu_attack_merge_block_buckets_into_kbc_buckets<Tx_Bucketed_Meta2><<<numBlocks,blockSize>>>(
				KBC_MERGE_BUCKET_START,
				in, batch_bucket_add_Y, num_entries_to_copy,
				T1_local_kbc_entries_L, device_local_kbc_num_entries_L);

		num_entries_to_copy = device_block_entry_counts_R[block_id];
		numBlocks = (num_entries_to_copy + blockSize - 1) / (blockSize);
		in = &T1_R_batch_match_results[block_address];

		//std::cout << "batch " << batch_id << " num_entries: " << num_entries_to_copy << std::endl;
		gpu_attack_merge_block_buckets_into_kbc_buckets<Tx_Bucketed_Meta2><<<numBlocks,blockSize>>>(
						KBC_MERGE_BUCKET_START,
						in, batch_bucket_add_Y, num_entries_to_copy,
						T1_local_kbc_entries_R, device_local_kbc_num_entries_R);

		// TODO: find matches in entries_L against entries_R...should be <16, avg around 3-4
		// only have 2m entries...so...could sort 1mL's against 1mR's?
		//auto matchTx_start = std::chrono::high_resolution_clock::now();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		mergeTx_finish = std::chrono::high_resolution_clock::now();
		total_merge_time_micros += std::chrono::duration_cast< std::chrono::microseconds >( mergeTx_finish - mergeTx_start ).count();


		CUDA_CHECK_RETURN(cudaMemset(device_T2_block_entry_counts, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599

		// yes this can be ram optimized to contrain MAX_ENTRIES to a fraction (at least 1/16th the size)
		// yikes...577ms...terrible...CPU WOULD BE FASTER!!!
		matchTx_start = std::chrono::high_resolution_clock::now();
		gpu_attack_find_tx_LR_matches<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<(KBC_END_L - KBC_START_L), 8>>>(1, batch_id_L, KBC_START_L, KBC_END_L,
					T1_local_kbc_entries_L, device_local_kbc_num_entries_L,
					T1_local_kbc_entries_R, device_local_kbc_num_entries_R,
					T2_batch_match_results, device_T2_block_entry_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		matchTx_finish = std::chrono::high_resolution_clock::now();
		total_match_time_micros += std::chrono::duration_cast< std::chrono::microseconds >( matchTx_finish - matchTx_start ).count();

		//total_match_time_ms += std::chrono::duration_cast<microseconds>(matchTx_finish - matchTx_start).count();
		for (int i = 0; i < BATCHES; i++) {
			global_block_counts[i] += device_T2_block_entry_counts[i];
		}

	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	std::cout << "   match t2 LR sum time: " << (total_match_time_micros/1000) << "ms" << std::endl;
	std::cout << "   merge t2 LR sum time: " << (total_merge_time_micros/1000) << "ms" << std::endl;
	auto mergekbcs_finish = std::chrono::high_resolution_clock::now();
	std::cout << "      T2 total time: " << std::chrono::duration_cast<milli>(mergekbcs_finish - mergekbcs_start).count() << " ms\n";
	//gpu_list_local_kbc_entries<<<1,1>>>(device_local_kbc_num_entries_L);
*/

	/*{
		auto matchT2_start = std::chrono::high_resolution_clock::now();
		Tx_Bucketed_Meta2 *t2bucketed_kbc_entries_in = (Tx_Bucketed_Meta2 *) device_buffer_A;
		Tx_Bucketed_Meta4 *t2bucketed_out = (Tx_Bucketed_Meta4 *) device_buffer_B;

		CUDA_CHECK_RETURN(cudaMemset(device_block_entry_counts, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599

		gpu_attack_find_t1_matches<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<(KBC_END - KBC_START), 256>>>(2, batch_id, KBC_START, KBC_END,
				t2bucketed_kbc_entries_in, device_local_kbc_num_entries,
				t2bucketed_out, device_block_entry_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto matchT2_finish = std::chrono::high_resolution_clock::now();

		std::cout << "   match T2 time: " << std::chrono::duration_cast<milli>(matchT2_finish - matchT2_start).count() << " ms\n";
		//gpu_list_local_kbc_entries<<<1,1>>>(device_local_kbc_num_entries);
	}
*/

}


#endif /* ATTACK_HPP_ */
