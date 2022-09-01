/*
 * k29_plotter.hpp
 *
 *  Created on: Mar 25, 2022
 *      Author: nick
 */

#ifndef K29_PLOTTER_HPP_
#define K29_PLOTTER_HPP_

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

const uint32_t kXX_BITS = 29;

const uint64_t k29_DEVICE_BUFFER_A_BYTES = 8589934592; // 8GB total buffer
const uint32_t k29_MAX_X_VALUE = 1 << kXX_BITS;
const uint64_t k29_MAX_Y_VALUE = 4294967296; // hack, set to 32 bit value of chacha

const uint32_t k29_CHACHA_SPLIT_BUCKETS = 1024; // after 10 starts dropping
const uint64_t k29_CHACHA_SPLIT_BUCKET_DIVISOR = k29_MAX_Y_VALUE / (k29_CHACHA_SPLIT_BUCKETS);
const uint64_t k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET = 2 * k29_MAX_X_VALUE / k29_CHACHA_SPLIT_BUCKETS;

uint *xchachas_bucket_counts;
uint *global_kbc_counts;

const uint32_t k29_BATCHES = 1;
const uint32_t k29_BC_NUM_BUCKETS = 568381;//1136761;//2273523;
const uint64_t k29_BC_BUCKET_DIVISOR = k29_MAX_Y_VALUE / k29_BC_NUM_BUCKETS;
const uint32_t k29_BC_LAST_BUCKET_ID = 1136761 - 1;//2273522;
const uint32_t k29_BCS_PER_BATCH = (k29_BC_NUM_BUCKETS / BATCHES)+1;
const uint32_t k29_BC_LOCAL_NUM_BUCKETS = k29_BCS_PER_BATCH + 1; // +1 is for including last R bucket space

const uint64_t k29_DEVICE_BUFFER_UNIT_BYTES = 32; // Tx_pairing_chunk_meta4 is 24 bytes, w/ backref is 32 bytes
const uint64_t k29_DEVICE_BUFFER_ALLOCATED_ENTRIES = KBC_LOCAL_NUM_BUCKETS * KBC_MAX_ENTRIES_PER_BUCKET; // HOST_MAX_BLOCK_ENTRIES * BATCHES;// DEVICE_BUFFER_ALLOCATED_ENTRIES = 120 * ((uint64_t) 1 << 32) / (100*BATCHES);
const uint64_t k29_DEVICE_BUFFER_ALLOCATED_BYTES = DEVICE_BUFFER_ALLOCATED_ENTRIES * DEVICE_BUFFER_UNIT_BYTES;


#define ATTACK_CHACHAS_k29_YS_ONLY(datax_slot) \
{ \
	int x_value = pos + datax_slot; \
	chacha_ys[x_value] = datax[datax_slot]; \
	chacha_xs[x_value] = x_value; \
}

#define ATTACK_CHACHAS_k29_TO_KBC(datax_slot) \
{ \
	uint32_t x_value = pos + datax_slot; \
	uint32_t chacha_y = datax[datax_slot]; \
	uint32_t Ly = chacha_y; \
	uint32_t bucket_id = Ly / k29_BC_BUCKET_DIVISOR; \
	xchacha_pair pair = { x_value, chacha_y }; \
	int slot = atomicAdd(&xchachas_bucket_counts[bucket_id],1); \
	if (slot > k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET) printf("Overflow k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET %u SLOT %u\n", k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET, slot); \
	else { \
		xchachas_buckets[KBC_MAX_ENTRIES_PER_BUCKET * bucket_id + slot] = pair; \
	} \
}

#define ATTACK_CHACHAS_k29_BUCKETADD(datax_slot) \
{ \
	uint32_t chacha_y = datax[datax_slot]; \
	uint32_t Ly = chacha_y; \
	uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR; \
	int slot = atomicAdd(&shared_counts[bucket_id],1); \
}

#define ATTACK_CHACHAS_k29_SORTEDADD(datax_slot) \
{ \
	uint32_t x_value = pos + datax_slot; \
	uint32_t chacha_y = datax[datax_slot]; \
	uint32_t Ly = chacha_y; \
	uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR; \
	int slot = shared_counts_offsets[bucket_id] + atomicAdd(&shared_counts[bucket_id],1); \
	shared_sorted_xs[slot] = x_value; shared_sorted_chachas[slot] = chacha_y; \
}

#define ATTACK_CHACHAS_k29_SORTEDADD_FILTERED(datax_slot) \
{ \
	uint32_t x_value = pos + datax_slot; \
	uint32_t chacha_y = datax[datax_slot]; \
	uint32_t Ly = chacha_y; \
	uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR; \
	if ((bucket_id >= filter_min) && (bucket_id < filter_max)) { \
		xchacha_pair pair = { x_value, chacha_y }; \
		int slot = shared_counts_offsets[bucket_id] + atomicAdd(&shared_counts[bucket_id],1); \
		shared_sorted_xchachas[slot] = pair; \
	} \
}

#define ATTACK_CHACHAS_k29_BUCKETSET(datax_slot) \
{ \
	uint32_t x_value = pos + datax_slot; \
	uint32_t chacha_y = datax[datax_slot]; \
	uint32_t Ly = chacha_y; \
	uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR; \
	xchacha_pair pair = { x_value, chacha_y }; \
	int slot = global_counts[bucket_id] + atomicAdd(&shared_counts[bucket_id],1); \
	if (slot > k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET) printf("Overflow k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET %u SLOT %u\n", k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET, slot); \
	else { \
		xchachas_buckets[k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET * bucket_id + slot] = pair; \
	} \
}

__global__
void gpu_chacha8_k29_bucketadd(const uint32_t N,
		const __restrict__ uint32_t *input, xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	__shared__ uint shared_counts[k29_CHACHA_SPLIT_BUCKETS];
	__shared__ uint global_counts[k29_CHACHA_SPLIT_BUCKETS];



	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	uint32_t base_group = blockIdx.x * blockDim.x;
	//uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	if (x_group < end_n) {

		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			shared_counts[i] = 0;
		}
		__syncthreads();

		uint32_t pos = x_group * 16;// + X_START/16;
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		ATTACK_CHACHAS_k29_BUCKETADD(0);ATTACK_CHACHAS_k29_BUCKETADD(1);ATTACK_CHACHAS_k29_BUCKETADD(2);ATTACK_CHACHAS_k29_BUCKETADD(3);
		ATTACK_CHACHAS_k29_BUCKETADD(4);ATTACK_CHACHAS_k29_BUCKETADD(5);ATTACK_CHACHAS_k29_BUCKETADD(6);ATTACK_CHACHAS_k29_BUCKETADD(7);
		ATTACK_CHACHAS_k29_BUCKETADD(8);ATTACK_CHACHAS_k29_BUCKETADD(9);ATTACK_CHACHAS_k29_BUCKETADD(10);ATTACK_CHACHAS_k29_BUCKETADD(11);
		ATTACK_CHACHAS_k29_BUCKETADD(12);ATTACK_CHACHAS_k29_BUCKETADD(13);ATTACK_CHACHAS_k29_BUCKETADD(14);ATTACK_CHACHAS_k29_BUCKETADD(15);

		__syncthreads();
		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
			shared_counts[i] = 0;
		}

		__syncthreads();

		// now recompute and this time add to global array
		pos = x_group * 16;// + X_START/16;

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		ATTACK_CHACHAS_k29_BUCKETSET(0);ATTACK_CHACHAS_k29_BUCKETSET(1);ATTACK_CHACHAS_k29_BUCKETSET(2);ATTACK_CHACHAS_k29_BUCKETSET(3);
		ATTACK_CHACHAS_k29_BUCKETSET(4);ATTACK_CHACHAS_k29_BUCKETSET(5);ATTACK_CHACHAS_k29_BUCKETSET(6);ATTACK_CHACHAS_k29_BUCKETSET(7);
		ATTACK_CHACHAS_k29_BUCKETSET(8);ATTACK_CHACHAS_k29_BUCKETSET(9);ATTACK_CHACHAS_k29_BUCKETSET(10);ATTACK_CHACHAS_k29_BUCKETSET(11);
		ATTACK_CHACHAS_k29_BUCKETSET(12);ATTACK_CHACHAS_k29_BUCKETSET(13);ATTACK_CHACHAS_k29_BUCKETSET(14);ATTACK_CHACHAS_k29_BUCKETSET(15);

	}
}

// we do computes and tally up number in each bucket
// if number in a bucket exceeds the 128 bytes (i.e. 128/8 bytes = 16)
// and we have at least 2 buckets with said bytes, then write those out to global.
__global__
void gpu_chacha8_k29_threshold_counters(const uint32_t N,
		const __restrict__ uint32_t *input, xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	__shared__ uint shared_counts[k29_CHACHA_SPLIT_BUCKETS];
	__shared__ uint global_counts[k29_CHACHA_SPLIT_BUCKETS];



	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	uint32_t base_group = blockIdx.x * blockDim.x;
	//uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	if (x_group < end_n) {

		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			shared_counts[i] = 0;
		}
		__syncthreads();

		uint32_t pos = x_group * 16;// + X_START/16;
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		ATTACK_CHACHAS_k29_BUCKETADD(0);ATTACK_CHACHAS_k29_BUCKETADD(1);ATTACK_CHACHAS_k29_BUCKETADD(2);ATTACK_CHACHAS_k29_BUCKETADD(3);
		ATTACK_CHACHAS_k29_BUCKETADD(4);ATTACK_CHACHAS_k29_BUCKETADD(5);ATTACK_CHACHAS_k29_BUCKETADD(6);ATTACK_CHACHAS_k29_BUCKETADD(7);
		ATTACK_CHACHAS_k29_BUCKETADD(8);ATTACK_CHACHAS_k29_BUCKETADD(9);ATTACK_CHACHAS_k29_BUCKETADD(10);ATTACK_CHACHAS_k29_BUCKETADD(11);
		ATTACK_CHACHAS_k29_BUCKETADD(12);ATTACK_CHACHAS_k29_BUCKETADD(13);ATTACK_CHACHAS_k29_BUCKETADD(14);ATTACK_CHACHAS_k29_BUCKETADD(15);

		__syncthreads();
		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
			shared_counts[i] = 0;
		}

		__syncthreads();

		// now recompute and this time add to global array
		pos = x_group * 16;// + X_START/16;

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		ATTACK_CHACHAS_k29_BUCKETSET(0);ATTACK_CHACHAS_k29_BUCKETSET(1);ATTACK_CHACHAS_k29_BUCKETSET(2);ATTACK_CHACHAS_k29_BUCKETSET(3);
		ATTACK_CHACHAS_k29_BUCKETSET(4);ATTACK_CHACHAS_k29_BUCKETSET(5);ATTACK_CHACHAS_k29_BUCKETSET(6);ATTACK_CHACHAS_k29_BUCKETSET(7);
		ATTACK_CHACHAS_k29_BUCKETSET(8);ATTACK_CHACHAS_k29_BUCKETSET(9);ATTACK_CHACHAS_k29_BUCKETSET(10);ATTACK_CHACHAS_k29_BUCKETSET(11);
		ATTACK_CHACHAS_k29_BUCKETSET(12);ATTACK_CHACHAS_k29_BUCKETSET(13);ATTACK_CHACHAS_k29_BUCKETSET(14);ATTACK_CHACHAS_k29_BUCKETSET(15);

	}
}

__global__
void gpu_chacha8_k29_bucketadd_256threads_warp_buckets(const uint32_t N,
		const __restrict__ uint32_t *input, xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	__shared__ uint32_t warp_bucket_ys[32];
	__shared__ uint32_t warp_bucket_xs[32];
	__shared__ int warp_bucket_counts[256/32]; // 8 different sets of warp buckets
	// idea here is to process with warps
	int warp_id = threadIdx.x % 32;
	uint32_t chacha_result = 23; // computed...
	if ((chacha_result % 32) == warp_id) {
		// add result to our bucket
		int count = atomicAdd(&warp_bucket_counts[warp_id],1);
		if (count == 16) {
			// 8 * (4x2) = 128 bytes, full bandwidth write
		}
	}
	// 256 threads, one bucket add at a time = 256 entries each loop.
	// we need 128 bytes to make a full bandwidth global write
	// = 128/8 = 16 entries from a bucket.

}

__global__
void gpu_chacha8_k29_bucketadd_256threads_upto1024buckets(const uint32_t N,
		// 1024 buckets = 176GB/s (240GB/s possible), 512 buckets =  276GB/s, 256 buckets =  293GB/s, 8 buckets =  337GB/s
		// note we lose ~1ms on innefficient prefix sums so this can improve +20% for 1024 buckets
		// with only shared counters we get 400GB/s, so this does take significant time and could be optimized
		// against having bank conflicts for instance.
		// possibly by doing 32 passes(!) where each thread focuses on it's own bank for shared memory. Yikes.
		const __restrict__ uint32_t *input, xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	__shared__ int shared_counts[k29_CHACHA_SPLIT_BUCKETS];
	__shared__ int global_counts[k29_CHACHA_SPLIT_BUCKETS];
	__shared__ int shared_counts_offsets[k29_CHACHA_SPLIT_BUCKETS];
	// a 256 thread x 16 pass gives 4096 values total.
	// for 1024 buckets that's only 4 values average per bucket. We want to write 128 bytes = 128/8 = 16 entries minimum.
	// so want minimum multiple of 4 so we average 16 entries
	// our shared space only allows 32k / 8 = 4096 entries
	// 1024 buckets = 176GB/s
	// 512 buckets =  276GB/s
	// 256 buckets =  293GB/s
	//   8 buckets =  337GB/s

	//__shared__ xchacha_pair shared_sorted_xchachas[4096];// 32k
	__shared__ uint32_t shared_sorted_xs[4096];// 16k <- tried to resolve bank conflicts but didn't do much
	__shared__ uint32_t shared_sorted_chachas[4096];// 16k

	if (blockDim.x != 256) printf("ERROR BLOCKDIM MUST BE 256\n");
	if (k29_CHACHA_SPLIT_BUCKETS > 1024) printf("ERROR SPLIT BUCKETS MUST BE <1024\n");

	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	uint32_t base_group = blockIdx.x * blockDim.x;
	//uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / (16); // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	if (x_group < end_n) {

		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			shared_counts[i] = 0;
		}
		__syncthreads();

		uint32_t pos = x_group * 16;// this is incorrect but shouldn't really matter that much
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		ATTACK_CHACHAS_k29_BUCKETADD(0);ATTACK_CHACHAS_k29_BUCKETADD(1);ATTACK_CHACHAS_k29_BUCKETADD(2);ATTACK_CHACHAS_k29_BUCKETADD(3);
		ATTACK_CHACHAS_k29_BUCKETADD(4);ATTACK_CHACHAS_k29_BUCKETADD(5);ATTACK_CHACHAS_k29_BUCKETADD(6);ATTACK_CHACHAS_k29_BUCKETADD(7);
		ATTACK_CHACHAS_k29_BUCKETADD(8);ATTACK_CHACHAS_k29_BUCKETADD(9);ATTACK_CHACHAS_k29_BUCKETADD(10);ATTACK_CHACHAS_k29_BUCKETADD(11);
		ATTACK_CHACHAS_k29_BUCKETADD(12);ATTACK_CHACHAS_k29_BUCKETADD(13);ATTACK_CHACHAS_k29_BUCKETADD(14);ATTACK_CHACHAS_k29_BUCKETADD(15);

		__syncthreads();

		/*
		 * 1.6 - 2.06746 ms with only bucket adds and shared to global counts
		 * 2.43 - 2.64 with our single thread prefix sum = +0.8 to 0.64
		 * then 5.94 total after writing out. = 180GB/s but minus 0.64 = 12% faster which is 200GB/s
		 */
		if (threadIdx.x == 0) {
			// yes this can be sped up, it adds 1.6ms/multiple - i.e. mult = 1 = +1.6ms, 2 = +0.8ms etc.
			shared_counts_offsets[0] = 0;
			//int min = shared_counts[0]; int max = shared_counts[0]; int num_above_16 = 0;
			for (int i=1;i<k29_CHACHA_SPLIT_BUCKETS;i++) {
				//if (min > shared_counts[i]) min = shared_counts[i];
				//if (max < shared_counts[i]) max = shared_counts[i];
				//if (shared_counts[i] >= 16) num_above_16++;
				//printf(" %i ", shared_counts[i]);
				shared_counts_offsets[i] = shared_counts[i-1] + shared_counts_offsets[i-1];
			}
			//printf("min: %i max: %i above16: %i\n", min, max,num_above_16);
		}
		__syncthreads();


		/*if ((base_group == 0) && (threadIdx.x == 0)) {
			printf("base group %u : ",base_group);
			for (int i=0;i<1024;i++) printf("%u ",shared_counts[i]);
			printf("\n");
			for (int i=0;i<1024;i++) printf("%u ",shared_counts_offsets[i]);
			printf("\n");
		}
		__syncthreads();*/

		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
			shared_counts[i] = 0;
		}
		__syncthreads();

		// now recompute and add sorted into array
		pos = x_group * 16;
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		// makes it 3.61 from 2.41 so yes did add a lot.
		ATTACK_CHACHAS_k29_SORTEDADD(0);ATTACK_CHACHAS_k29_SORTEDADD(1);ATTACK_CHACHAS_k29_SORTEDADD(2);ATTACK_CHACHAS_k29_SORTEDADD(3);
		ATTACK_CHACHAS_k29_SORTEDADD(4);ATTACK_CHACHAS_k29_SORTEDADD(5);ATTACK_CHACHAS_k29_SORTEDADD(6);ATTACK_CHACHAS_k29_SORTEDADD(7);
		ATTACK_CHACHAS_k29_SORTEDADD(8);ATTACK_CHACHAS_k29_SORTEDADD(9);ATTACK_CHACHAS_k29_SORTEDADD(10);ATTACK_CHACHAS_k29_SORTEDADD(11);
		ATTACK_CHACHAS_k29_SORTEDADD(12);ATTACK_CHACHAS_k29_SORTEDADD(13);ATTACK_CHACHAS_k29_SORTEDADD(14);ATTACK_CHACHAS_k29_SORTEDADD(15);

		// now push to global
		__syncthreads();
		for (int i=threadIdx.x;i<4096;i+=blockDim.x) {

			uint32_t x = shared_sorted_xs[i];
			uint32_t Ly = shared_sorted_chachas[i];//pair.chacha;
			xchacha_pair pair = {}; pair.x = x; pair.chacha = Ly;
			uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
			int slot = global_counts[bucket_id] + atomicAdd(&shared_counts[bucket_id],1);
			if (slot > k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET) printf("Overflow k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET %u SLOT %u\n", k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET, slot);
			else xchachas_buckets[k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET * bucket_id + slot] = pair;
		}

		//gpu_chacha8_k29_bucketadd time: 10.9147 ms w/ 1024 buckets no multipasses, w/o writing is 3.6ms so writes take 7ms
		//Effective Bandwidth (GB/s): 196.752304
	}
}





__global__
void gpu_chacha8_k29_linear(const uint32_t N,
		const __restrict__ uint32_t *input, uint32_t *chacha_xs, uint32_t *chacha_ys)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	if (x_group < end_n) {
		uint32_t pos = x_group * 16;// + X_START/16;
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		ATTACK_CHACHAS_k29_YS_ONLY(0);ATTACK_CHACHAS_k29_YS_ONLY(1);ATTACK_CHACHAS_k29_YS_ONLY(2);ATTACK_CHACHAS_k29_YS_ONLY(3);
		ATTACK_CHACHAS_k29_YS_ONLY(4);ATTACK_CHACHAS_k29_YS_ONLY(5);ATTACK_CHACHAS_k29_YS_ONLY(6);ATTACK_CHACHAS_k29_YS_ONLY(7);
		ATTACK_CHACHAS_k29_YS_ONLY(8);ATTACK_CHACHAS_k29_YS_ONLY(9);ATTACK_CHACHAS_k29_YS_ONLY(10);ATTACK_CHACHAS_k29_YS_ONLY(11);
		ATTACK_CHACHAS_k29_YS_ONLY(12);ATTACK_CHACHAS_k29_YS_ONLY(13);ATTACK_CHACHAS_k29_YS_ONLY(14);ATTACK_CHACHAS_k29_YS_ONLY(15);
	}
}

__global__
void gpu_chacha8_k29_to_kbc(const uint32_t N,
		const __restrict__ uint32_t *input, xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	if (x_group < end_n) {
		uint32_t pos = x_group * 16;// + X_START/16;
		//printf("x group pos = %u\n", pos);

		datax[0] = input[0];datax[1] = input[1];datax[2] = input[2];datax[3] = input[3];datax[4] = input[4];datax[5] = input[5];datax[6] = input[6];datax[7] = input[7];
		datax[8] = input[8];datax[9] = input[9];datax[10] = input[10];datax[11] = input[11];
		datax[12] = pos; datax[13]= 0; // pos never bigger than 32 bit pos >> 32;
		datax[14] = input[14];datax[15] = input[15];

		#pragma unroll
		for (int i = 0; i < 4; i++) {
			QUARTERROUND(datax[0], datax[4], datax[8], datax[12]);QUARTERROUND(datax[1], datax[5], datax[9], datax[13]);
			QUARTERROUND(datax[2], datax[6], datax[10], datax[14]);QUARTERROUND(datax[3], datax[7], datax[11], datax[15]);
			QUARTERROUND(datax[0], datax[5], datax[10], datax[15]);QUARTERROUND(datax[1], datax[6], datax[11], datax[12]);
			QUARTERROUND(datax[2], datax[7], datax[8], datax[13]);QUARTERROUND(datax[3], datax[4], datax[9], datax[14]);
		}

		datax[0] += input[0];datax[1] += input[1];datax[2] += input[2];datax[3] += input[3];datax[4] += input[4];
		datax[5] += input[5];datax[6] += input[6];datax[7] += input[7];datax[8] += input[8];datax[9] += input[9];
		datax[10] += input[10];datax[11] += input[11];datax[12] += x_group; // j12;//datax[13] += 0;
		datax[14] += input[14];datax[15] += input[15];

		// convert to little endian/big endian whatever, chia needs it like this
		BYTESWAP32(datax[0]);BYTESWAP32(datax[1]);BYTESWAP32(datax[2]);BYTESWAP32(datax[3]);BYTESWAP32(datax[4]);BYTESWAP32(datax[5]);
		BYTESWAP32(datax[6]);BYTESWAP32(datax[7]);BYTESWAP32(datax[8]);BYTESWAP32(datax[9]);BYTESWAP32(datax[10]);BYTESWAP32(datax[11]);
		BYTESWAP32(datax[12]);BYTESWAP32(datax[13]);BYTESWAP32(datax[14]);BYTESWAP32(datax[15]);

		//uint64_t y = datax[0] << 6 + x >> 26;  for 2^10 (1024 buckets) is >> (38-10) => 28, >> 28 -> x >> 22
		//int nick_bucket_id; //  = datax[0] >> 22; // gives bucket id 0..1023
		ATTACK_CHACHAS_k29_TO_KBC(0);ATTACK_CHACHAS_k29_TO_KBC(1);ATTACK_CHACHAS_k29_TO_KBC(2);ATTACK_CHACHAS_k29_TO_KBC(3);
		ATTACK_CHACHAS_k29_TO_KBC(4);ATTACK_CHACHAS_k29_TO_KBC(5);ATTACK_CHACHAS_k29_TO_KBC(6);ATTACK_CHACHAS_k29_TO_KBC(7);
		ATTACK_CHACHAS_k29_TO_KBC(8);ATTACK_CHACHAS_k29_TO_KBC(9);ATTACK_CHACHAS_k29_TO_KBC(10);ATTACK_CHACHAS_k29_TO_KBC(11);
		ATTACK_CHACHAS_k29_TO_KBC(12);ATTACK_CHACHAS_k29_TO_KBC(13);ATTACK_CHACHAS_k29_TO_KBC(14);ATTACK_CHACHAS_k29_TO_KBC(15);
	}
}

__global__
void gpu_chacha_ys_bucket_direct(const uint32_t N, const __restrict__ uint32_t *chacha_ys,
		xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N;

	if (x < end_n) {
		uint32_t chacha_y = chacha_ys[x];
		uint32_t Ly = chacha_y; // (((uint64_t) chacha_y) << 6) + (x >> 26);
		uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
		int slot = atomicAdd(&xchachas_bucket_counts[bucket_id],1);
		xchacha_pair pair = { x, chacha_y };
		xchachas_buckets[bucket_id * k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET + slot] = pair;
	}
}

__global__
void gpu_chacha_ys_bucket_shared_counts(const uint32_t N, const __restrict__ uint32_t *chacha_ys,
		xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	__shared__ uint shared_counts[k29_CHACHA_SPLIT_BUCKETS];
	__shared__ uint global_counts[k29_CHACHA_SPLIT_BUCKETS];

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N;

	if (x < end_n) {

		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			shared_counts[i] = 0;
		}
		__syncthreads();

		uint32_t chacha_y = chacha_ys[x];
		uint64_t Ly = chacha_y; // (((uint64_t) chacha_y) << 6) + (x >> 26);
		uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
		xchacha_pair pair = { x, chacha_y };
		atomicAdd(&shared_counts[bucket_id],1);

		__syncthreads();
		for (int i=threadIdx.x;i<k29_CHACHA_SPLIT_BUCKETS;i+=blockDim.x) {
			global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
			shared_counts[i] = 0;
		}

		__syncthreads();
		for (int i=threadIdx.x;i<blockDim.x;i+=blockDim.x) {
			//printf("writing slot %u into global slot %u\n",i,base_x + i);
			//xchacha_pair pair = shared_chachas[i];
			//uint32_t bucket_id = pair.chacha / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
			uint slot = global_counts[bucket_id] + atomicAdd(&shared_counts[bucket_id],1);
			if (slot > k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET) printf("Overflow k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET %u SLOT %u\n", k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET, slot);
			else xchachas_buckets[k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET * bucket_id + slot] = pair; // shared_chachas[i];
		}
	}


}

__global__
void gpu_test_cache_cp(const uint32_t N, const uint32_t cache_size_bytes, uint32_t *cache)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N;
	//if (threadIdx.x == 0) {
	//	printf("gridDim.x: %u  blockIdx.x: %u  our block id: %u  total blocks: %u\n", gridDim.x, blockIdx.x, block_id, total_blocks);
	// }
	const uint32_t CACHE_SIZE = 1024*1024;
	if (x < end_n) {
		//uint32_t address = x*1; // x*1: 1 write - 862GB/s vs 1444G/s cached = 1.67x
		//uint32_t address = x*4; // x*4: 1 write - 171GB/s in no cache zone, and 404GB/s in cache zone = 2.3x
		uint32_t address = x*64 + 1; //   Xt*8: 1 write - 85GB/s in no cache zone,  and 204GB/s in cache zone = 2.4x (2.63ms)
		 // *8 with 2 writes (8 byte) :  85GB/s same as 1 write, so effective 170GB/s cache zone: 137GB/s effective 274GB/s in cache zone
		//  *8 with 4 writes (16 byte):  81GB/s..so x4 = 240GB/s
		//  *64 with 1 writes (4 byte):  38GB/s ..  cache:  127GB/s in cache,
		//  *64 with 4 writes (16 byte): 32GB/s full random write effective x4 = 120GB/s, cache doesn't seem to help here.
		//       also writing at address*64+1 didn't affect write speed strangely.
		 // *64 with 6 writes (24 byte): 21GB/s - x6 = 120GB/s
		//  *64 with 8 writes (32 byte): 16GB/s - x8 = 128GB/s
		 // *8 with 32 bytes: 17.9ms
		const int BOUNDS = false ? CACHE_SIZE : N;
		cache[(address + 0) % BOUNDS] = x;
		cache[(address + 1) % BOUNDS] = x;
		cache[(address + 2) % BOUNDS] = x;
		cache[(address + 3) % BOUNDS] = x;

		//float4 val;
		//const float4* myinput = cache+address;
		//asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(myinput));

		//cache[(address + 4) % BOUNDS] = x;
		//cache[(address + 5) % BOUNDS] = x;
		//cache[(address + 6) % BOUNDS] = x;
		//cache[(address + 7) % BOUNDS] = x;

		//cache[(x*8) % (N)] = x;
		// x*1: 862GB/s vs 1444G/s cached = 1.67x
		// x*4: 171GB/s in no cache zone, and 404GB/s in cache zone = 2.3x
		// x*8: 85GB/s in no cache zone,  and 204GB/s in cache zone = 2.4x
	}
}




__global__
void gpu_count_xpairs_kbc_buckets(
		const xchacha_pair *xchachas_buckets, const uint *xchachas_block_counts, uint *global_kbc_counts)
{
	uint32_t block_id = blockIdx.x;
	const uint32_t num_in_block = xchachas_block_counts[block_id];
	const uint32_t block_offset = k29_CHACHA_SPLIT_MAX_ENTRIES_PER_BUCKET * block_id;

	for (int i=block_offset + threadIdx.x;i<block_offset + num_in_block;i+=blockDim.x) {
		xchacha_pair pair = xchachas_buckets[i];
		uint32_t Ly = pair.chacha; // (((uint64_t) chacha_y) << 6) + (x >> 26);
		//uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
		uint32_t kbc_id = pair.chacha / k29_BC_BUCKET_DIVISOR; // hack for k28 size kBC;
		//printf("x: %u  chacha: %u  bucket: %u  kbc_id:%u\n", pair.x, pair.chacha, bucket_id, kbc_id);
		int slot = atomicAdd(&global_kbc_counts[kbc_id],1);
	}
}


__global__ void gpu_check_xpairs(const xchacha_pair *xchachas_in, const uint32_t N) {
	if (threadIdx.x == 0) {
		for (int i=0;i<N;i++) {
			xchacha_pair pair = xchachas_in[i];

			uint32_t Ly = pair.chacha; // (((uint64_t) chacha_y) << 6) + (x >> 26);
			uint32_t bucket_id = Ly / k29_CHACHA_SPLIT_BUCKET_DIVISOR;
			printf("%u = %u  bucket: %u\n", pair.x, pair.chacha, bucket_id);
		}
	}
}

#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>


void do_k29_T1() {

	std::cout << "do k29 T1  BATCHES:" << k29_BATCHES  << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();
	auto finish =  std::chrono::high_resolution_clock::now(); // just to allocate

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin);


	int blockSize; // # of threads per block, maximum is 1024.
	uint64_t calc_N;
	uint64_t calc_blockSize;
	uint64_t calc_numBlocks;
	int numBlocks;

	// first phase is writing chacha results
	uint32_t *chacha_ys = (uint32_t *) &device_buffer_A[0]; // set ys to beginning of device buffer A
	uint32_t *chacha_xs = (uint32_t *) &device_buffer_A[k29_MAX_X_VALUE*4]; // set ys to beginning of device buffer A
	xchacha_pair *xchachas_buckets = (xchacha_pair *) &device_buffer_A[k29_MAX_X_VALUE*4];
	float milliseconds = 0;

	std::cout << "   gpu_chacha8_k29_bucketadd    ys num:" << calc_N << std::endl;
		blockSize = 256; // # of threads per block, maximum is 1024.
		calc_N = k29_MAX_X_VALUE;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;
		std::cout << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		CUDA_CHECK_RETURN(cudaMemset(global_kbc_counts, 0, k29_BC_NUM_BUCKETS*sizeof(int)));
		cudaEventRecord(start);
		//gpu_chacha8_k29_to_kbc<<<numBlocks,blockSize>>>(calc_N, chacha_input,xchachas_buckets, global_kbc_counts);
		// cuda event total time: 65.0044 ms
		//gpu_chacha8_k29_bucketadd time: 23.0625 ms
		//Effective Bandwidth (GB/s): 46.557852
		//gpu_chacha8_k29_bucketadd time: 23.0697 ms
		//Effective Bandwidth (GB/s): 46.543388

		//gpu_chacha8_k29_bucketadd<<<numBlocks,blockSize>>>(calc_N, chacha_input,xchachas_buckets, xchachas_bucket_counts);
		//gpu_chacha8_k29_bucketadd_256threads_upto1024buckets<<<numBlocks,blockSize>>>(calc_N, chacha_input,xchachas_buckets, xchachas_bucket_counts);
		gpu_chacha8_k29_bucketadd_256threads_upto1024buckets<<<numBlocks,blockSize>>>(calc_N, chacha_input,xchachas_buckets, xchachas_bucket_counts);
		//gpu_chacha8_k29_bucketadd_256threads_upto1024buckets<<<numBlocks,blockSize>>>(calc_N, chacha_input,xchachas_buckets, xchachas_bucket_counts);

		// counter list counts  SUM:134217728   MAX:132347 id: 0 count: 131044 6.06925 ms (GB/s): 176.706432

		cudaEventRecord(stop);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "gpu_chacha8_k29_bucketadd_256threads_upto1024buckets time: " << milliseconds << " ms\n";
		printf("Effective Bandwidth (GB/s): %f\n", calc_N*8/milliseconds/1e6);

		//1024 buckets multiple 1 gpu_chacha8_k29_bucketadd time: 11.008 ms Effective Bandwidth (GB/s): 195.083904
/*
	cudaEventRecord(start);
	// 1 block per split bucket, threads will have to work out how much to parse
	gpu_count_xpairs_kbc_buckets<<<k29_CHACHA_SPLIT_BUCKETS,blockSize>>>(xchachas_buckets, xchachas_bucket_counts, global_kbc_counts);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "gpu_count_xpairs_kbc_buckets time: " << milliseconds << " ms\n";
	printf("Effective Bandwidth (GB/s): %f\n", k29_MAX_X_VALUE*8/milliseconds/1e6);

	thrust::device_ptr<uint> device_kbc_counts(global_kbc_counts);
	cudaEventRecord(start);
	thrust::exclusive_scan(device_kbc_counts, device_kbc_counts + k29_BC_NUM_BUCKETS, device_kbc_counts);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "exclusive scan kbc_buckets time: " << milliseconds << " ms\n";
*/

		std::cout << "   gpu_test_cache    ys num:" << calc_N << std::endl;
		blockSize = 256; // # of threads per block, maximum is 1024.
		calc_N = k29_MAX_X_VALUE;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
		numBlocks = calc_numBlocks;
		std::cout << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		cudaEventRecord(start);
		gpu_test_cache_cp<<<numBlocks,blockSize>>>(calc_N, calc_N, chacha_ys);
		cudaEventRecord(stop);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);//auto sort_start = std::chrono::high_resolution_clock::now();
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "cache linear test " << calc_N << " time: " << milliseconds << " ms\n";
		printf("Effective Bandwidth (GB/s): %f\n", calc_N*4/milliseconds/1e6);


	{
		// thrust linear then sort method

		std::cout << "   gpu_chacha8_k29_linear    ys num:" << calc_N << std::endl;

		blockSize = 256; // # of threads per block, maximum is 1024.
		calc_N = k29_MAX_X_VALUE;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;
		std::cout << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		cudaEventRecord(start);
		gpu_chacha8_k29_linear<<<numBlocks,blockSize>>>(calc_N, chacha_input,chacha_xs,chacha_ys);
		cudaEventRecord(stop);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);//auto sort_start = std::chrono::high_resolution_clock::now();
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "write chachas time: " << milliseconds << " ms\n";
		printf("Effective Bandwidth (GB/s): %f\n", calc_N*8/milliseconds/1e6);

		/*auto sort_start = std::chrono::high_resolution_clock::now();
		cudaEventRecord(start);
		thrust::device_ptr<uint32_t> device_xs_L_ptr(chacha_xs);
		thrust::device_ptr<uint32_t> device_ys_L_ptr(chacha_ys);
		thrust::sort_by_key(device_ys_L_ptr, device_ys_L_ptr + calc_N, device_xs_L_ptr);
		cudaEventRecord(stop);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto sort_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   sort time: " << std::chrono::duration_cast<milli>(sort_finish - sort_start).count() << " ms\n";
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "thrust sort " << calc_N << " time: " << milliseconds << " ms\n";
		printf("Effective Bandwidth (GB/s): %f\n", calc_N*8*2/milliseconds/1e6);*/
	}

	{// Declare, allocate, and initialize device-accessible pointers for sorting data

		// Determine temporary device storage requirements
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				chacha_ys, chacha_ys, chacha_xs, chacha_xs, k29_MAX_X_VALUE);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run sorting operation
		cudaEventRecord(start);
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				chacha_ys, chacha_ys, chacha_xs, chacha_xs, k29_MAX_X_VALUE);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		// thrust is 13ms
		std::cout << "cuda sort " << calc_N << " time: " << milliseconds << " ms\n";
		// d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
		// d_values_out          <-- [5, 4, 3, 1, 2, 0, 6]
	}


	/*std::cout << "   gpu_chacha split buckets (num: " << k29_CHACHA_SPLIT_BUCKETS << " divisor:" << k29_CHACHA_SPLIT_BUCKET_DIVISOR << ")   ys num:" << calc_N << std::endl;
	blockSize = 1024; // # of threads per block, maximum is 1024.
	calc_N = k29_MAX_X_VALUE;
	calc_blockSize = blockSize;
	calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
	numBlocks = calc_numBlocks;
	cudaEventRecord(start);
	// erm...thrust sort on 268435456 is 27ms...just saying, beats most other timings.
	//       and sort on 33million elements (1/8 of total) is 3ms. In other words...worth doing...
	//gpu_chacha_ys_bucket_shared_counts<<<numBlocks,blockSize>>>(calc_N, chacha_ys, xchachas_buckets, xchachas_bucket_counts);
	//counter list counts  SUM:268435456   MAX:33561699
	//   gpu_chacha split buckets (num: 1024 divisor:4194304)      ys num:268435456 time: 47.1419 ms (GB/s): 68.330432
	//   gpu_chacha split buckets (num: 128 divisor:33554432)      ys num:268435456 time: 38.0465 ms (GB/s): 84.665424
	//   gpu_chacha_ys_bucket_shared (num: 32 divisor:134217728)   ys num:268435456 time: 17.9118 ms (GB/s): 179.838096
	//   gpu_chacha split buckets (num: 8 divisor:536870912)       ys num:268435456 time: 6.79731 ms (GB/s): 473.896960
	//   -> note 8*8*8*8 = 4096, and would take 27ms, which is less than 1024 @ 47ms
	//gpu_chacha_ys_bucket_direct<<<numBlocks,blockSize>>>(calc_N, chacha_ys, xchachas_buckets, xchachas_bucket_counts);
	//   gpu_chacha_ys_bucket_direct (num: 1136761 divisor:3778)   ys num:268435456 time: 102.703 ms   (GB/s): 31.364442
	//   gpu_chacha_ys_bucket_direct (num: 1024 divisor:4194304)   ys num:268435456 time: 48.5682 ms   (GB/s): 66.323720
	//   gpu_chacha_ys_bucket_direct (num: 128 divisor:33554432)   ys num:268435456 time: 73.6359 ms (GB/s): 43.745292
	//   gpu_chacha_ys_bucket_direct (num: 32 divisor:134217728)   ys num:268435456 time: 85.1845 ms   (GB/s): 37.814688
	cudaEventRecord(stop);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "bucket chachas time: " << milliseconds << " ms\n";
	printf("Effective Bandwidth (GB/s): %f\n", (calc_N*(4+8))/milliseconds/1e6);*/

	//gpu_get_max_counts_from_counter_list<<<1,1>>>(xchachas_bucket_counts, k29_CHACHA_SPLIT_BUCKETS, true);
	//<<<1,1>>>(global_kbc_counts, 1024, false);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, begin, end);
	std::cout << "cuda event total time: " << milliseconds << " ms\n";
}


void setup_memory_k29() {

	//setupMMap(HOST_ALLOCATED_BYTES); // potentially useful if going to do random reads/writes to stored data

	//std::cout << "      device_block_entry_counts (" << k29_BATCHES << "): " << k29_BATCHES << " size:" << (sizeof(int)*k29_BATCHES) << std::endl;
	//CUDA_CHECK_RETURN(cudaMallocManaged(&device_block_entry_counts, k29_BATCHES*sizeof(int)));

	std::cout << "      device_local_kbc_num_entries " << k29_BC_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*k29_BC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&global_kbc_counts, k29_BC_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(global_kbc_counts, 0, k29_BC_NUM_BUCKETS*sizeof(int)));

	//Tx_Pairing_Chunk_Meta4 *device_buffer_A;
	std::cout << "      device_buffer_A " << k29_DEVICE_BUFFER_A_BYTES << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_A, k29_DEVICE_BUFFER_A_BYTES));

	std::cout << "      xchachas_bucket_counts k29_CHACHA_SPLIT_BUCKETS:" << k29_CHACHA_SPLIT_BUCKETS << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&xchachas_bucket_counts, k29_CHACHA_SPLIT_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(xchachas_bucket_counts, 0, k29_CHACHA_SPLIT_BUCKETS*sizeof(int)));


	//Tx_Pairing_Chunk_Meta4 *device_buffer_B;
	//std::cout << "      device_buffer_B " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  DEVICE_BUFFER_UNIT_BYTES << ") = " << DEVICE_BUFFER_ALLOCATED_BYTES << std::endl;
	//CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_B, DEVICE_BUFFER_ALLOCATED_BYTES));

	//std::cout << "      device_buffer_refdata " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  BACKREF_UNIT_BYTES << ") = " << BACKREF_ALLOCATED_BYTES << std::endl;
	//CUDA_CHECK_RETURN(cudaMalloc(&device_buffer_refdata, BACKREF_ALLOCATED_BYTES));

	//std::cout << "      HOST host_refdata_blocks ENTRIES: " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " ALLOCATED ENTRIES: " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " UNIT BYTES: " << BACKREF_UNIT_BYTES << " = " << (BACKREF_ALLOCATED_BYTES) << std::endl;
	//CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_refdata_blocks, BACKREF_ALLOCATED_BYTES)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();

	//std::cout << "      HOST host_criss_cross_blocks MAX_ENTRIES: " << HOST_MAX_BLOCK_ENTRIES << " ALLOCATED ENTRIES: " << HOST_ALLOCATED_ENTRIES << " UNIT BYTES: " << HOST_UNIT_BYTES << " = " << (HOST_ALLOCATED_BYTES) << std::endl;
	//CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_criss_cross_blocks, HOST_ALLOCATED_BYTES)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();
}

void do_k29() {
	std::cout << "****** PROGRAM START K29 V0.1 *********" << std::endl;

	setup_memory_k29();


	auto total_start = std::chrono::high_resolution_clock::now();
	do_k29_T1();
	std::cout << " freeing memory...";
	freeMemory();
	std::cout << "end." << std::endl;
	exit(EXIT_SUCCESS);
}




#endif /* K29_PLOTTER_HPP_ */
