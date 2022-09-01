/*
 * attack_method_xpairbits.hpp
 *
 *  Created on: Dec 5, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_XPAIRBITS_HPP_
#define ATTACK_METHOD_XPAIRBITS_HPP_

const uint32_t MAX_LXS_PER_KBC_BUCKET = 16; // 24 for 110,000,000

const uint32_t XPAIR_BITS = 8;
const uint32_t MAX_RX_MATCHES = (1 << (32 - XPAIR_BITS))*2;
const uint32_t CHACHA_NUM_BATCHES_BITS = 3;
const uint32_t CHACHA_NUM_BATCHES = 1 << CHACHA_NUM_BATCHES_BITS;
const uint32_t CHACHA_TOTAL_ENTRIES_PER_BATCH = (1 << (32 - XPAIR_BITS - CHACHA_NUM_BATCHES_BITS));
const uint32_t CHACHA_BUCKET_BITS = 4; // ACROSS ALL BATCHES
const uint32_t CHACHA_NUM_BUCKETS = (1 << CHACHA_BUCKET_BITS);
const uint32_t CHACHA_BUCKET_DIVISOR = (1 << (32 - CHACHA_BUCKET_BITS));
const uint32_t CHACHA_SPLIT_BUCKET_DIVISOR = (1 << (32 - CHACHA_BUCKET_BITS - CHACHA_NUM_BATCHES_BITS));
const uint32_t CHACHA_MAX_ENTRIES_PER_BUCKET = (11 * (CHACHA_TOTAL_ENTRIES_PER_BATCH / CHACHA_NUM_BUCKETS)) / 10;
const uint64_t CHACHA_OUT_MAX_ENTRIES_NEEDED = (CHACHA_NUM_BUCKETS * CHACHA_MAX_ENTRIES_PER_BUCKET);

struct xchacha_pair {
	uint32_t x;
	uint32_t chacha;
};

#define KBC_MASK_SHIFT 4
#define KBC_MASK_MOD 8
#define KBC_MASK_BITS 0b001111
#define ATTACK_INTO_KBC_YS_BITMASK(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	uint32_t kbc_bitmask_bucket = kbc_bucket_id / KBC_MASK_MOD; \
	uint32_t kbc_bitmask_shift = KBC_MASK_SHIFT * (kbc_bucket_id % KBC_MASK_MOD); \
	uint32_t add = 1 << kbc_bitmask_shift; \
	uint slot_value = atomicAdd(&kbc_global_num_entries_L[kbc_bitmask_bucket],add); \
	uint slot = (slot_value >> kbc_bitmask_shift) & KBC_MASK_BITS; \
	if (slot > MAX_LXS_PER_KBC_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u\n", MAX_LXS_PER_KBC_BUCKET, slot); } \
	uint32_t entries_address = kbc_bucket_id * MAX_LXS_PER_KBC_BUCKET + slot; \
	kbc_global_Ly_entries_L[entries_address] = y; \
	kbc_x_entries[entries_address] = (x + i); \
}

__global__
void gpu_chacha8_set_Lxs_into_kbc_ys_mask(const uint32_t N,
		const __restrict__ uint32_t *input,
		uint16_t *kbc_global_Ly_entries_L, uint32_t *kbc_x_entries, unsigned int *kbc_global_num_entries_L, uint32_t MAX_LXS_PER_KBC_BUCKET)
{
	uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group

	for (uint32_t x_group = index; x_group < end_n; x_group += stride) {
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
		ATTACK_INTO_KBC_YS_BITMASK(x0,0);ATTACK_INTO_KBC_YS_BITMASK(x1,1);ATTACK_INTO_KBC_YS_BITMASK(x2,2);ATTACK_INTO_KBC_YS_BITMASK(x3,3);
		ATTACK_INTO_KBC_YS_BITMASK(x4,4);ATTACK_INTO_KBC_YS_BITMASK(x5,5);ATTACK_INTO_KBC_YS_BITMASK(x6,6);ATTACK_INTO_KBC_YS_BITMASK(x7,7);
		ATTACK_INTO_KBC_YS_BITMASK(x8,8);ATTACK_INTO_KBC_YS_BITMASK(x9,9);ATTACK_INTO_KBC_YS_BITMASK(x10,10);ATTACK_INTO_KBC_YS_BITMASK(x11,11);
		ATTACK_INTO_KBC_YS_BITMASK(x12,12);ATTACK_INTO_KBC_YS_BITMASK(x13,13);ATTACK_INTO_KBC_YS_BITMASK(x14,14);ATTACK_INTO_KBC_YS_BITMASK(x15,15);
	}
}

__global__
void gpu_list_xchachas(const uint32_t N, const xchacha_pair *xchachas)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		xchacha_pair pair = xchachas[index];
		uint64_t y = (((uint64_t) pair.chacha) << 6) + (pair.x >> 26);
		uint32_t kbc_bucket_id = uint32_t (y / kBC);
		printf("set xchachas kbc mask index: %u  x: %u  chacha: %u   y: %llu   kbc_bucket_id: %u\n",
					index, pair.x, pair.chacha, y, kbc_bucket_id);
	}
}

__global__
void gpu_chacha8_set_xchachas_into_kbc_ys_mask(const uint32_t N,
		const xchacha_pair *xchachas,
		uint16_t *kbc_global_Ly_entries_L, uint32_t *kbc_x_entries, unsigned int *kbc_global_num_entries_L, uint32_t MAX_LXS_PER_KBC_BUCKET)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		xchacha_pair pair = xchachas[index];
		uint64_t y = (((uint64_t) pair.chacha) << 6) + (pair.x >> 26);
		uint32_t kbc_bucket_id = uint32_t (y / kBC);
		if (index < 10)
		printf("set xchachas kbc mask index: %u  x: %u  chacha: %u   y: %llu   kbc_bucket_id: %u\n",
							index, pair.x, pair.chacha, y, kbc_bucket_id);
		//uint32_t kbc_bitmask_bucket = kbc_bucket_id / KBC_MASK_MOD;
		//uint32_t kbc_bitmask_shift = KBC_MASK_SHIFT * (kbc_bucket_id % KBC_MASK_MOD);
		//uint32_t add = 1 << kbc_bitmask_shift;
		//uint slot_value = atomicAdd(&kbc_global_num_entries_L[kbc_bitmask_bucket],add);
		//uint slot = (slot_value >> kbc_bitmask_shift) & KBC_MASK_BITS;

		uint slot = atomicAdd(&kbc_global_num_entries_L[kbc_bucket_id],1);

		if (index < 10) {
			printf("set xchachas kbc mask index: %u  x: %u  chacha: %u   y: %llu   kbc_bucket_id: %u\n",
					index, pair.x, pair.chacha, y, kbc_bucket_id);
		}

		//if (slot > MAX_LXS_PER_KBC_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u\n", MAX_LXS_PER_KBC_BUCKET, slot); }
		//uint32_t entries_address = kbc_bucket_id * MAX_LXS_PER_KBC_BUCKET + slot;
		//kbc_global_Ly_entries_L[entries_address] = y % kBC;
		//kbc_x_entries[entries_address] = pair.x;
	}

}

__global__ void gpu_get_max_counts_from_counter_list(unsigned int *kbc_counts, const int NUM, const bool printAll) {
	__shared__ unsigned int max_kbc_count;
	__shared__ unsigned int sum_kbc_count;
	if (threadIdx.x == 0) {
		max_kbc_count = 0;
		sum_kbc_count = 0;
	}
	__syncthreads();
	for (uint32_t i=threadIdx.x;i<NUM;i+=blockDim.x) {
		//uint32_t kbc_bitmask_bucket = i / KBC_MASK_MOD;
		//uint32_t kbc_bitmask_shift = KBC_MASK_SHIFT * (i % KBC_MASK_MOD);
		//uint slot_value = kbc_counts[kbc_bitmask_bucket];
		//unsigned int kbc_count = (slot_value >> kbc_bitmask_shift) & KBC_MASK_BITS;
		unsigned int kbc_count = kbc_counts[i];
		if (printAll) printf("id: %u count: %u\n", i, kbc_count);
		atomicMax(&max_kbc_count, kbc_count);
		atomicAdd(&sum_kbc_count, kbc_count);
	}
	__syncthreads();
	if (threadIdx.x == 0) printf("counter list counts  SUM:%u   MAX:%u\n", sum_kbc_count, max_kbc_count);
}

#define ATTACK_BUCKETBATCH_CHACHAS32_PAIR(chacha_y,i) \
{ \
	if ((chacha_y >= BATCH_CHACHA_RANGE_MIN) && (chacha_y <= BATCH_CHACHA_RANGE_MAX)) { \
		xchacha_pair pair = { base_x + i, chacha_y }; \
		int slot = atomicAdd(&local_filter_count,1); \
		if (slot > MAX_SHARED_CHACHAS) printf("MAX_SHARED_CHACHAS %u OVERFLOW %u\n", MAX_SHARED_CHACHAS, slot); \
		shared_chachas[slot] = pair; \
		uint32_t split_bucket_id = (chacha_y - BATCH_CHACHA_RANGE_MIN) / CHACHA_SPLIT_BUCKET_DIVISOR; \
		atomicAdd(&shared_counts[split_bucket_id],1); \
	} \
}

// run with 128 blocksize, more doesn't matter.
template<int NUM_SPLIT_BUCKETS>
__global__
void gpu_chacha8_k32_compute_chachas32_filter_buckets_bychachabatchrange(const uint32_t N,
		const uint32_t BATCH_CHACHA_RANGE_MIN, const uint32_t BATCH_CHACHA_RANGE_MAX,
		const uint32_t CHACHA_MAX_PER_SPLIT_BUCKET, const uint32_t CHACHA_SPLIT_BUCKET_DIVISOR,
		const __restrict__ uint32_t *input,
		xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	const uint32_t MAX_SHARED_CHACHAS = 128*8; // try to bring down as much as can
	__shared__ xchacha_pair shared_chachas[MAX_SHARED_CHACHAS]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.
	__shared__ uint shared_counts[NUM_SPLIT_BUCKETS];
	__shared__ uint global_counts[NUM_SPLIT_BUCKETS];
	__shared__ uint local_filter_count;

	//if (blockDim.x > 128) printf("MUST HAVE BLOCKSIZE 128 (RECOMMENDED) OR LESS, OR INCREASED SHARED MEM TO MORE\n");

	//uint32_t base_group = blockIdx.x * blockDim.x;

	uint32_t x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	uint32_t base_x = x_group * 32;
	const uint32_t end_n = N / 32; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	for (int i=threadIdx.x;i<NUM_SPLIT_BUCKETS;i+=blockDim.x) {
		shared_counts[i] = 0;
	}
	if (threadIdx.x == 0) {
		local_filter_count = 0;
	}
	__syncthreads();

	const int j = 0;
	if (x_group < end_n) {
		uint32_t pos = x_group * 2;// + X_START/16;
		//printf("x group pos = %u\n", pos);

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
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+0],0);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+1],1);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+2],2);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+3],3);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+4],4);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+5],5);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+6],6);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+7],7);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+8],8);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+9],9);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+10],10);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+11],11);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+12],12);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+13],13);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+14],14);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+15],15);

		pos += 1;

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
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+0],16+0);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+1],16+1);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+2],16+2);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+3],16+3);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+4],16+4);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+5],16+5);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+6],16+6);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+7],16+7);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+8],16+8);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+9],16+9);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+10],16+10);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+11],16+11);
		ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+12],16+12);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+13],16+13);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+14],16+14);ATTACK_BUCKETBATCH_CHACHAS32_PAIR(datax[j+15],16+15);
	}
	// at this point we have 128*32 = 4096 entries
	// now we have to sort them into the buckets
	// we already have the shared counts set from the ATTACK macro
	__syncthreads();
	for (int i=threadIdx.x;i<NUM_SPLIT_BUCKETS;i+=blockDim.x) {
		global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
		shared_counts[i] = 0;
	}
	// now just scan our filtered entries and bucket them
	__syncthreads();
	for (int i=threadIdx.x;i<local_filter_count;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n",i,base_x + i);

		// remember, these are *already* bucketed to some range
		xchacha_pair pair = shared_chachas[i];
		uint32_t split_bucket_id = (pair.chacha - BATCH_CHACHA_RANGE_MIN) / CHACHA_SPLIT_BUCKET_DIVISOR;
		uint slot = global_counts[split_bucket_id] + atomicAdd(&shared_counts[split_bucket_id],1);
		if (slot > CHACHA_MAX_PER_SPLIT_BUCKET) printf("Overflow CHACHA_MAX_PER_BUCKET %u SLOT %u\n", CHACHA_MAX_PER_SPLIT_BUCKET, slot);
		else xchachas_buckets[CHACHA_MAX_PER_SPLIT_BUCKET * split_bucket_id + slot] = shared_chachas[i];
	}
}

#define CHECK_MATCH() \
{ \
	int16_t yr_kbc = Ry % kBC; \
	int16_t yr_bid = yr_kbc / kC; \
	int16_t yl_bid = yl_kbc / kC; \
	int16_t formula_one = yr_bid - yl_bid; \
	if (formula_one < 0) { \
		formula_one += kB; \
	} \
	int16_t m = formula_one; \
	if (m >= kB) { \
		m -= kB; \
	} \
	if (m < 64) { \
		int16_t yl_cid = yl_kbc % kC; \
		int16_t yr_cid = yr_kbc % kC;\
		int16_t parity = (kbc_bucket_id_L) % 2; \
		int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC; \
		int16_t formula_two = yr_cid - yl_cid; \
		if (formula_two < 0) { \
			formula_two += kC; \
		} \
		if (formula_two == m2_parity_squared) { \
			isMatch = true; \
		} \
	} \
}

__global__
void gpu_chacha8_filter_rxs_from_bucket_batch(
		const uint32_t N,
		const xchacha_pair* __restrict__ xchachas,
		const uint16_t* __restrict__ kbc_global_Ly_entries_L,
		const unsigned int* __restrict__ kbc_global_num_entries_L,
		uint32_t MAX_LXS_PER_KBC_BUCKET,
		uint32_t * __restrict__ rxs,
		int *rx_count)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		xchacha_pair entry = xchachas[i];
		uint64_t Ry = (((uint64_t) entry.chacha) << 6) + (entry.x >> 26);
		int kbc_bucket_id_R = (uint32_t (Ry / kBC));
		if (kbc_bucket_id_R > 0) {
			int kbc_bucket_id_L = kbc_bucket_id_R - 1;
			//printf("entry x:%u chacha:%u\n", entry.x, entry.chacha, kbc_bucket_id_L);
			//int num = kbc_global_num_entries_L[kbc_bucket_id_L];

			//uint num = kbc_global_num_entries_L[kbc_bucket_id_L];
			uint32_t kbc_bitmask_bucket = kbc_bucket_id_L / KBC_MASK_MOD;
			uint32_t kbc_bitmask_shift = KBC_MASK_SHIFT * (kbc_bucket_id_L % KBC_MASK_MOD);
			uint slot_value =kbc_global_num_entries_L[kbc_bitmask_bucket];
			uint num = (slot_value >> kbc_bitmask_shift) & KBC_MASK_BITS;
			for (int nm=0;nm<num;nm++) {
				bool isMatch = false;
				int16_t yl_kbc = kbc_global_Ly_entries_L[kbc_bucket_id_L * MAX_LXS_PER_KBC_BUCKET + nm];
				CHECK_MATCH();
				if (isMatch) {
					int slot = atomicAdd(&rx_count[0],1);
					rxs[slot] = entry.x;
				}
			}
		}
	}
}

void attack_method_xpairbits() {
	std::cout << "ATTACK METHOD X PAIR BITS: " << XPAIR_BITS << std::endl;

	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();

	unsigned int *device_global_kbc_num_entries_L;
	uint16_t *kbc_Ly_entries; // the y % kbc bucketed entries
	uint32_t *kbc_x_entries;  // the associated x value for the y pairing

	// alloc for lx's
	std::cout << "      kbc_Ly_entries MAX_LXS: " << MAX_LXS_PER_KBC_BUCKET << " TOTAL BYTES: " <<  (MAX_LXS_PER_KBC_BUCKET * sizeof(uint64_t) * kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_Ly_entries, (MAX_LXS_PER_KBC_BUCKET * sizeof(uint16_t) * kBC_NUM_BUCKETS)));
	std::cout << "      kbc_x_entries MAX_LXS: " << MAX_LXS_PER_KBC_BUCKET << " TOTAL BYTES: " <<  (MAX_LXS_PER_KBC_BUCKET * sizeof(uint64_t) * kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_x_entries, (MAX_LXS_PER_KBC_BUCKET * sizeof(uint32_t) * kBC_NUM_BUCKETS)));

	std::cout << "      device_global_kbc_num_entries_L size:" << (sizeof(int)*kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_global_kbc_num_entries_L, sizeof(int)*kBC_NUM_BUCKETS));
	CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, kBC_NUM_BUCKETS*sizeof(int)));

	xchacha_pair *lxchachas, *rxchachas;
	uint *lxchachas_bucket_counts, *rxchachas_bucket_counts;
	CUDA_CHECK_RETURN(cudaMallocManaged(&lxchachas_bucket_counts, CHACHA_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(lxchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&rxchachas_bucket_counts, CHACHA_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(rxchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));

	std::cout << "      lxchachas size:" << (sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&lxchachas, sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED));
	std::cout << "      rxchachas size:" << (sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&rxchachas, sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED));

	uint32_t *rx_match_list;
	int *rx_match_count;
	std::cout << "      rx_match_list MAX_RX_MATCHES: " << MAX_RX_MATCHES << " TOTAL BYTES: " <<  (MAX_RX_MATCHES * sizeof(uint32_t)) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&rx_match_list, (MAX_RX_MATCHES * sizeof(uint32_t))));
	CUDA_CHECK_RETURN(cudaMallocManaged(&rx_match_count, sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(rx_match_count, 0, sizeof(int)));



	auto alloc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   alloc time: " << std::chrono::duration_cast<milli>(alloc_finish - attack_start).count() << " ms\n";

	auto compute_only_start = std::chrono::high_resolution_clock::now();

	int blockSize; // # of threads per block, maximum is 1024.
	uint64_t calc_N;
	uint64_t calc_blockSize;
	uint64_t calc_numBlocks;
	int numBlocks;

	// FIRST SET LXS into global memory, these stay put for each chacha round
	/*blockSize = 256; // # of threads per block, maximum is 1024.
	calc_N = 1 << (32 - XPAIR_BITS);
	calc_blockSize = blockSize;
	calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
	numBlocks = calc_numBlocks;

	std::cout << "   gpu_chacha8_set_Lxs_into_kbc_ys num:" << calc_N << std::endl;
	auto lxintokbc_start = std::chrono::high_resolution_clock::now();
	gpu_chacha8_set_Lxs_into_kbc_ys_mask<<<numBlocks,blockSize>>>(calc_N, chacha_input,
			kbc_Ly_entries, kbc_x_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto lxintokbc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   gpu_chacha8_set_Lxs_into_kbc_ys time: " << std::chrono::duration_cast<milli>(lxintokbc_finish - lxintokbc_start).count() << " ms\n";
	gpu_get_max_counts_from_counter_list<<<1,1024>>>(device_global_kbc_num_entries_L, kBC_NUM_BUCKETS, false);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());*/

	auto chacha_batches_start = std::chrono::high_resolution_clock::now();
	int64_t total_chacha_ms = 0;
	uint32_t sum_counts = 0;
	for (uint64_t chacha_batch_id = 0; chacha_batch_id < 1/*CHACHA_NUM_BATCHES*/; chacha_batch_id++) {
		//std::cout << "Doing chacha batch " << chacha_batch_id << std::endl;
		uint64_t BATCH_CHACHA_DIVISOR = (1 << (32 - CHACHA_NUM_BATCHES_BITS));
		uint64_t BATCH_CHACHA_RANGE_MIN = ((uint64_t) (chacha_batch_id + 0)) * BATCH_CHACHA_DIVISOR;
		uint64_t BATCH_CHACHA_RANGE_MAX = ((uint64_t) (chacha_batch_id + 1)) * BATCH_CHACHA_DIVISOR - 1; // use -1 since rnage is inclusive, also helps stay in 32-bit range rather than wrap to 0 for last batch

		//std::cout << "   BATCH_CHACHA_DIVISOR : " << BATCH_CHACHA_DIVISOR << std::endl;
		//std::cout << "   BATCH_CHACHA_RANGE   : " << BATCH_CHACHA_RANGE_MIN << " <-> " << BATCH_CHACHA_RANGE_MAX << std::endl;
		//std::cout << "   BATCH_CHACHA_TOTAL_ENTRIES : " << CHACHA_TOTAL_ENTRIES_PER_BATCH << std::endl;
		//std::cout << "   CHACHA_MAX_ENTRIES_PER_BUCKET : " << CHACHA_MAX_ENTRIES_PER_BUCKET << std::endl;
		//std::cout << "   CHACHA_SPLIT_BUCKET_DIVISOR : " << CHACHA_SPLIT_BUCKET_DIVISOR << std::endl;

		blockSize = 128; // # of threads per block, maximum is 1024.
		calc_N = 1 << (32 - XPAIR_BITS); //CHACHA_TOTAL_ENTRIES_PER_BATCH;
		uint32_t CHACHA_X_START = chacha_batch_id * calc_N;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 32);
		numBlocks = calc_numBlocks;
		CUDA_CHECK_RETURN(cudaMemset(lxchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));
		auto chacha_start = std::chrono::high_resolution_clock::now();
		//std::cout << "   calc_N   : " << calc_N << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		gpu_chacha8_k32_compute_chachas32_filter_buckets_bychachabatchrange<CHACHA_NUM_BUCKETS><<<numBlocks,blockSize>>>(calc_N,
							BATCH_CHACHA_RANGE_MIN, BATCH_CHACHA_RANGE_MAX,
							CHACHA_MAX_ENTRIES_PER_BUCKET, CHACHA_SPLIT_BUCKET_DIVISOR,
							chacha_input,
							lxchachas, lxchachas_bucket_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto chacha_finish = std::chrono::high_resolution_clock::now();
		total_chacha_ms += std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count();
		gpu_get_max_counts_from_counter_list<<<1,1>>>(lxchachas_bucket_counts, CHACHA_NUM_BUCKETS, true);
		//auto chacha_rs_start = std::chrono::high_resolution_clock::now();


		for (uint chacha_bucket_id=0;chacha_bucket_id<CHACHA_NUM_BUCKETS;chacha_bucket_id++) {
			std::cout << " chacha bucket id " << chacha_bucket_id << std::endl;
			gpu_list_xchachas<<<1,1>>>(10, &lxchachas[chacha_bucket_id]);
			blockSize = 256; // # of threads per block, maximum is 1024.
			calc_N = lxchachas_bucket_counts[chacha_bucket_id];
			sum_counts += calc_N;
			calc_blockSize = blockSize;
			calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
			numBlocks = calc_numBlocks;
			std::cout << "Setting kbcs calc_N: " << calc_N << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
			//gpu_chacha8_set_xchachas_into_kbc_ys_mask<<<numBlocks,blockSize>>>(calc_N, &lxchachas[chacha_bucket_id],
			//	kbc_Ly_entries, kbc_x_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET);
		}
		std::cout << "sum counts: " << sum_counts << std::endl;
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//std::cout << "   gpu_chacha8_k32_write_chachas32_buckets results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
	}

	gpu_get_max_counts_from_counter_list<<<1,1>>>(device_global_kbc_num_entries_L, 100, true);//kBC_NUM_BUCKETS, false);


	/*for (uint64_t chacha_batch_id = 0; chacha_batch_id < CHACHA_NUM_BATCHES; chacha_batch_id++) {
		//std::cout << "Doing chacha batch " << chacha_batch_id << std::endl;
		uint64_t BATCH_CHACHA_DIVISOR = (1 << (32 - CHACHA_NUM_BATCHES_BITS));
		uint64_t BATCH_CHACHA_RANGE_MIN = ((uint64_t) (chacha_batch_id + 0)) * BATCH_CHACHA_DIVISOR;
		uint64_t BATCH_CHACHA_RANGE_MAX = ((uint64_t) (chacha_batch_id + 1)) * BATCH_CHACHA_DIVISOR - 1; // use -1 since rnage is inclusive, also helps stay in 32-bit range rather than wrap to 0 for last batch

		//std::cout << "   BATCH_CHACHA_DIVISOR : " << BATCH_CHACHA_DIVISOR << std::endl;
		//std::cout << "   BATCH_CHACHA_RANGE   : " << BATCH_CHACHA_RANGE_MIN << " <-> " << BATCH_CHACHA_RANGE_MAX << std::endl;
		//std::cout << "   BATCH_CHACHA_TOTAL_ENTRIES : " << CHACHA_TOTAL_ENTRIES_PER_BATCH << std::endl;
		//std::cout << "   CHACHA_MAX_ENTRIES_PER_BUCKET : " << CHACHA_MAX_ENTRIES_PER_BUCKET << std::endl;
		//std::cout << "   CHACHA_SPLIT_BUCKET_DIVISOR : " << CHACHA_SPLIT_BUCKET_DIVISOR << std::endl;

		blockSize = 128; // # of threads per block, maximum is 1024.
		calc_N = 1 << (32 - XPAIR_BITS); //CHACHA_TOTAL_ENTRIES_PER_BATCH;
		uint32_t CHACHA_X_START = 0;//chacha_batch_id * calc_N;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 32);
		numBlocks = calc_numBlocks;
		CUDA_CHECK_RETURN(cudaMemset(rxchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));
		auto chacha_start = std::chrono::high_resolution_clock::now();
		//std::cout << "   calc_N   : " << calc_N << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		gpu_chacha8_k32_compute_chachas32_filter_buckets_bychachabatchrange<CHACHA_NUM_BUCKETS><<<numBlocks,blockSize>>>(calc_N,
							BATCH_CHACHA_RANGE_MIN, BATCH_CHACHA_RANGE_MAX,
							CHACHA_MAX_ENTRIES_PER_BUCKET, CHACHA_SPLIT_BUCKET_DIVISOR,
							chacha_input,
							rxchachas, rxchachas_bucket_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto chacha_finish = std::chrono::high_resolution_clock::now();
		total_chacha_ms += std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count();
		//gpu_get_max_counts_from_counter_list<<<1,1>>>(xchachas_bucket_counts, CHACHA_NUM_BUCKETS, true);
		//auto chacha_rs_start = std::chrono::high_resolution_clock::now();

		for (uint chacha_bucket_id=0;chacha_bucket_id<CHACHA_NUM_BUCKETS;chacha_bucket_id++) {
			blockSize = 256; // # of threads per block, maximum is 1024.
			calc_N = lxchachas_bucket_counts[chacha_bucket_id];
			calc_blockSize = blockSize;
			calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
			numBlocks = calc_numBlocks;
			//std::cout << "Setting kbcs calc_N: " << calc_N << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
			gpu_chacha8_filter_rxs_from_bucket_batch<<<numBlocks,blockSize>>>(
							calc_N,
							&rxchachas[chacha_bucket_id],
							kbc_Ly_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET,
							rx_match_list, rx_match_count);
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//std::cout << "   gpu_chacha8_k32_write_chachas32_buckets results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
	}*/



	auto compute_only_finish = std::chrono::high_resolution_clock::now();

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(kbc_Ly_entries));
	CUDA_CHECK_RETURN(cudaFree(kbc_x_entries));
	CUDA_CHECK_RETURN(cudaFree(device_global_kbc_num_entries_L));

	auto attack_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   found " << rx_match_count[0] << " matches" << std::endl;
	std::cout << "   total chachas time: " << total_chacha_ms << " ms\n";
	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "   attack total time: " << std::chrono::duration_cast<milli>(attack_finish - attack_start).count() << " ms\n";
	std::cout << "end." << std::endl;
}


#endif /* ATTACK_METHOD_XPAIRBITS_HPP_ */
