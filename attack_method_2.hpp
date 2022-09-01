/*
 * attack_method_2.hpp
 *
 *  Created on: Nov 4, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_2_HPP_
#define ATTACK_METHOD_2_HPP_

#define ATTACK_KBCFILTER_LR1LR2slower(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	uint32_t local_kbc_bucket_id = 30000000; \
	int slot = -1; \
	int *num_list; \
	Tx_Bucketed_Meta1 *entries_list; \
	if ((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) { \
		local_kbc_bucket_id = kbc_bucket_id - KBC_START_L; \
		num_list = kbc_local_num_entries_L; \
		entries_list = kbc_local_entries_L; \
	} \
	if ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) { \
		local_kbc_bucket_id = kbc_bucket_id - KBC_START_R; \
		num_list = kbc_local_num_entries_R; \
		entries_list = kbc_local_entries_R; \
	} \
	if ((kbc_bucket_id >= KBC_START_L2) && (kbc_bucket_id <= KBC_END_L2)) { \
		local_kbc_bucket_id = kbc_bucket_id - KBC_START_L2; \
		num_list = kbc_local_num_entries_L2; \
		entries_list = kbc_local_entries_L2; \
	} \
	if ((kbc_bucket_id >= KBC_START_R2) && (kbc_bucket_id <= KBC_END_R2)) { \
		local_kbc_bucket_id = kbc_bucket_id - KBC_START_R2; \
		num_list = kbc_local_num_entries_R2; \
		entries_list = kbc_local_entries_R2; \
	} \
	if (local_kbc_bucket_id < 30000000) { \
		slot = atomicAdd(&num_list[local_kbc_bucket_id],1); \
		Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		entries_list[entries_address] = entry; \
	} \
}

/*
 * uint32_t kbc_bitmask_bucket = local_kbc_bucket_id / 3; \
		int kbc_bitmask_add = 1 << (kbc_bitmask_bucket*9); \
		int bitadd = atomicAdd(&kbc_local_num_entries_L[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t slot = bitadd; \
		slot = (slot >> (kbc_bitmask_bucket*9)) & 0b0111111111; \

		 TOTAL: 262341
 */
//with bitmask kbcs
#define ATTACK_KBCFILTER_LR1LR2bitmask(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	if ((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L; \
		uint32_t kbc_bitmask_bucket = local_kbc_bucket_id / 3; \
		uint32_t kbc_bitmask_shift = 9*(local_kbc_bucket_id % 3); \
		int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
		int bitadd = atomicAdd(&kbc_local_num_entries_L[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t slot = bitadd; \
		slot = (slot >> (kbc_bitmask_shift)) & 0b0111111111; \
			Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_L[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) { \
			uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R; \
			uint32_t kbc_bitmask_bucket = local_kbc_bucket_id / 3; \
		uint32_t kbc_bitmask_shift = 9*(local_kbc_bucket_id % 3); \
		int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
		int bitadd = atomicAdd(&kbc_local_num_entries_R[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t slot = bitadd; \
		slot = (slot >> (kbc_bitmask_shift)) & 0b0111111111; \
			Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_R[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_L2) && (kbc_bucket_id <= KBC_END_L2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L2; \
					uint32_t kbc_bitmask_bucket = local_kbc_bucket_id / 3; \
		uint32_t kbc_bitmask_shift = 9*(local_kbc_bucket_id % 3); \
		int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
		int bitadd = atomicAdd(&kbc_local_num_entries_L2[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t slot = bitadd; \
		slot = (slot >> (kbc_bitmask_shift)) & 0b0111111111; \
					Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_L2[entries_address] = entry; \
				} \
				if ((kbc_bucket_id >= KBC_START_R2) && (kbc_bucket_id <= KBC_END_R2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R2; \
					uint32_t kbc_bitmask_bucket = local_kbc_bucket_id / 3; \
		uint32_t kbc_bitmask_shift = 9*(local_kbc_bucket_id % 3); \
		int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
		int bitadd = atomicAdd(&kbc_local_num_entries_R2[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t slot = bitadd; \
		slot = (slot >> (kbc_bitmask_shift)) & 0b0111111111; \
					Tx_Bucketed_Meta1 entry = { (x+i), (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_R2[entries_address] = entry; \
				} \
}

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

#define ATTACK_KBCFILTER_LR1LR2_CHACHA(chacha_y,x) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
		if ((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) { \
			uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L; \
			int slot = atomicAdd(&kbc_local_num_entries_L[local_kbc_bucket_id],1); \
			Tx_Bucketed_Meta1 entry = { x, (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_L[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) { \
			uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R; \
			int slot = atomicAdd(&kbc_local_num_entries_R[local_kbc_bucket_id],1); \
			Tx_Bucketed_Meta1 entry = { x, (uint32_t) (y % kBC) }; \
			if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
			uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
			kbc_local_entries_R[entries_address] = entry; \
		} \
		if ((kbc_bucket_id >= KBC_START_L2) && (kbc_bucket_id <= KBC_END_L2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_L2; \
					int slot = atomicAdd(&kbc_local_num_entries_L2[local_kbc_bucket_id],1); \
					Tx_Bucketed_Meta1 entry = { x, (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_L2[entries_address] = entry; \
				} \
				if ((kbc_bucket_id >= KBC_START_R2) && (kbc_bucket_id <= KBC_END_R2)) { \
					uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START_R2; \
					int slot = atomicAdd(&kbc_local_num_entries_R2[local_kbc_bucket_id],1); \
					Tx_Bucketed_Meta1 entry = { x, (uint32_t) (y % kBC) }; \
					if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
					uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
					kbc_local_entries_R2[entries_address] = entry; \
				} \
}

#define ATTACK_KBCSTREAM_LR1LR2(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + ((base_x + i) >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	if (((kbc_bucket_id >= KBC_START_L) && (kbc_bucket_id <= KBC_END_L)) \
			|| ((kbc_bucket_id >= KBC_START_R) && (kbc_bucket_id <= KBC_END_R)) \
			|| ((kbc_bucket_id >= KBC_START_L2) && (kbc_bucket_id <= KBC_END_L2)) \
			|| ((kbc_bucket_id >= KBC_START_R2) && (kbc_bucket_id <= KBC_END_R2))) { \
		xchacha_pair pair = { base_x + i, chacha_y }; \
		int slot = atomicAdd(&local_filter_count,1); \
		if (slot > MAX_SHARED_CHACHAS) printf("MAX_SHARED_CHACHAS %u OVERFLOW %u\n", MAX_SHARED_CHACHAS, slot); \
		shared_chachas[slot] = pair; \
	} \
}
struct xchacha_pair {
	uint32_t x;
	uint32_t chacha;
};

__global__
void gpu_chacha8_k32_kbc_ranges_LR1LR2(const uint32_t N,
		const __restrict__ uint32_t *input,
		Tx_Bucketed_Meta1 *kbc_local_entries_L, int *kbc_local_num_entries_L, uint32_t KBC_START_L, uint32_t KBC_END_L,
		Tx_Bucketed_Meta1 *kbc_local_entries_R, int *kbc_local_num_entries_R, uint32_t KBC_START_R, uint32_t KBC_END_R,
		Tx_Bucketed_Meta1 *kbc_local_entries_L2, int *kbc_local_num_entries_L2, uint32_t KBC_START_L2, uint32_t KBC_END_L2,
		Tx_Bucketed_Meta1 *kbc_local_entries_R2, int *kbc_local_num_entries_R2, uint32_t KBC_START_R2, uint32_t KBC_END_R2)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	const uint32_t MAX_SHARED_CHACHAS = 128*8; // try to bring down as much as can
	__shared__ xchacha_pair shared_chachas[MAX_SHARED_CHACHAS]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.
	__shared__ uint local_filter_count;

	//if (blockDim.x > 128) printf("MUST HAVE BLOCKSIZE 128 (RECOMMENDED) OR LESS, OR INCREASED SHARED MEM TO MORE\n");

	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 32;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 32; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

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
		ATTACK_KBCSTREAM_LR1LR2(datax[j+0],0);ATTACK_KBCSTREAM_LR1LR2(datax[j+1],1);ATTACK_KBCSTREAM_LR1LR2(datax[j+2],2);ATTACK_KBCSTREAM_LR1LR2(datax[j+3],3);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+4],4);ATTACK_KBCSTREAM_LR1LR2(datax[j+5],5);ATTACK_KBCSTREAM_LR1LR2(datax[j+6],6);ATTACK_KBCSTREAM_LR1LR2(datax[j+7],7);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+8],8);ATTACK_KBCSTREAM_LR1LR2(datax[j+9],9);ATTACK_KBCSTREAM_LR1LR2(datax[j+10],10);ATTACK_KBCSTREAM_LR1LR2(datax[j+11],11);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+12],12);ATTACK_KBCSTREAM_LR1LR2(datax[j+13],13);ATTACK_KBCSTREAM_LR1LR2(datax[j+14],14);ATTACK_KBCSTREAM_LR1LR2(datax[j+15],15);

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
		ATTACK_KBCSTREAM_LR1LR2(datax[j+0],16+0);ATTACK_KBCSTREAM_LR1LR2(datax[j+1],16+1);ATTACK_KBCSTREAM_LR1LR2(datax[j+2],16+2);ATTACK_KBCSTREAM_LR1LR2(datax[j+3],16+3);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+4],16+4);ATTACK_KBCSTREAM_LR1LR2(datax[j+5],16+5);ATTACK_KBCSTREAM_LR1LR2(datax[j+6],16+6);ATTACK_KBCSTREAM_LR1LR2(datax[j+7],16+7);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+8],16+8);ATTACK_KBCSTREAM_LR1LR2(datax[j+9],16+9);ATTACK_KBCSTREAM_LR1LR2(datax[j+10],16+10);ATTACK_KBCSTREAM_LR1LR2(datax[j+11],16+11);
		ATTACK_KBCSTREAM_LR1LR2(datax[j+12],16+12);ATTACK_KBCSTREAM_LR1LR2(datax[j+13],16+13);ATTACK_KBCSTREAM_LR1LR2(datax[j+14],16+14);ATTACK_KBCSTREAM_LR1LR2(datax[j+15],16+15);
	}
	// at this point we have 128*32 = 4096 entries
	// now we have to sort them into the buckets
	// we already have the shared counts set from the ATTACK macro
	// now just scan our filtered entries and bucket them
	__syncthreads();
	for (int i=threadIdx.x;i<local_filter_count;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n",i,base_x + i);
		// remember, these are *already* bucketed to some range
		xchacha_pair pair = shared_chachas[i];
		ATTACK_KBCFILTER_LR1LR2_CHACHA(pair.chacha, pair.x);
	}
}




__global__
void gpu_chacha8_k32_kbc_ranges_LR1LR2_orig(const uint32_t N,
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



__global__
void gpu_attack_get_kbcs_with_pairs_from_global_kbcs(
		const unsigned int *kbc_global_num_entries_L,
		const unsigned int *kbc_global_num_entries_R,
		unsigned int *kbc_pairs_list_L_bucket_ids, int *pairs_count) {

	uint32_t global_kbc_L_bucket_id = blockIdx.x*blockDim.x+threadIdx.x;

	if (global_kbc_L_bucket_id < (kBC_NUM_BUCKETS-1)) {

		uint32_t kbc_bitmask_bucket = global_kbc_L_bucket_id / 8;
		uint32_t kbc_bitmask_shift = 4*(global_kbc_L_bucket_id % 8);
		uint32_t bitvalue = kbc_global_num_entries_L[kbc_bitmask_bucket];
		const unsigned int num_L = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;

		kbc_bitmask_bucket = (global_kbc_L_bucket_id + 1) / 8;
		kbc_bitmask_shift = 4*((global_kbc_L_bucket_id + 1) % 8);
		bitvalue = kbc_global_num_entries_R[kbc_bitmask_bucket];
		const unsigned int num_R = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;

		if ((num_L > 0) && (num_R > 0)) {

			int slot = atomicAdd(&pairs_count[0], 1);
			//printf("found kbc %u with two blocks > 0 slot %u \n", global_kbc_L_bucket_id,slot);
			kbc_pairs_list_L_bucket_ids[slot] = global_kbc_L_bucket_id;
		}
	}
}

struct Match_Attack_Pair_Index {
	uint32_t bucket_L_id; // could compress this to fit in 32 bit
	uint16_t idx_L;
	uint16_t idx_R;
};

template <typename BUCKETED_ENTRY_IN>
__global__
void gpu_attack_process_t1_pairs(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		Match_Attack_Pair_Index *match_list, int *match_counts) {
	// testmatch count: 33532242
	//   testmatch T1 L time: 9 ms
	const uint16_t NUM_RMAPS = (kBC/2)+1;
		__shared__ unsigned int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
		__shared__ uint32_t nick_rmap_extras_rl[32];
		__shared__ uint16_t nick_rmap_extras_ry[32];
		__shared__ uint16_t nick_rmap_extras_pos[32];
		__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int global_match_slot;
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

	// bucket sort the r positions!
	for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
		BUCKETED_ENTRY_IN R_entry = kbc_local_entries[start_R+pos_R];
		uint16_t r_y = R_entry.y;

		// r_y's share a block across two adjacent values, so kbc_map just works out which part it's in.
		unsigned int kbc_map = r_y / 2;
		const unsigned int kbc_box_shift = (r_y % 2) * 15;
		int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)

		unsigned int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
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
	//for (uint16_t pos_R = threadStartScan; pos_R < num_R; pos_R+=threadSkipScan) {
	//	kbc_R_entries[pos_R] = kbc_local_entries[start_R+pos_R];
	//}
	//for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
	//	kbc_L_entries[pos_L] = kbc_local_entries[start_L+pos_L];
	//}




	uint16_t parity = global_kbc_L_bucket_id % 2;

	__syncthreads(); // wait for all threads to write r_bid entries

	//testmatch count: 33271871
	//   testmatch T1 L time: 9 ms

	for (uint16_t pos_L = threadStartScan; pos_L < num_L; pos_L+=threadSkipScan) {
		//Bucketed_kBC_Entry L_entry = kbc_local_entries[pos_L];
		BUCKETED_ENTRY_IN L_entry = kbc_local_entries[start_L+pos_L];
		uint16_t l_y = L_entry.y;

		//bool doPrint = (L_entry.meta[0] == 601683299);

		//uint16_t base_indJ = l_y / kC;
		//uint16_t indJ_plus_m_mod_kB = base_indJ % kB;
		//uint16_t indJ_plus_m_mod_kB_times_kC = indJ_plus_m_mod_kB * kC;
		//uint16_t m_2_plus_parity_squared_iter = (parity + l_y) % kC;
		//uint16_t m_2_plus_parity_start_add = parity == 0 ? 4 : 8; // this increments by 8 each time
		//if (doPrint) {
		//	printf("Starting values:\n");
		//	printf("                           l_y: %u\n",l_y);
		//	printf("                        parity: %u\n",parity);
		///	printf("            indJ_plus_m_mod_kB: %u\n",indJ_plus_m_mod_kB);
		//	printf("   indJ_plus_m_mod_kB_times_kC: %u\n",indJ_plus_m_mod_kB_times_kC);
		//	printf("  m_2_plus_parity_squared_iter: %u\n",m_2_plus_parity_squared_iter);
		//	printf("     m_2_plus_parity_start_add: %u\n",m_2_plus_parity_start_add);
		//}
		for (int m=0;m<64;m++) {


			/*
			 * sadly these no division optimations turned out to be slower than a single calculation line
			 * uint16_t r_target = indJ_plus_m_mod_kB_times_kC + m_2_plus_parity_squared_iter;

			// this gets updated at end of loop.
			indJ_plus_m_mod_kB += 1;
			if (indJ_plus_m_mod_kB >= kB) {
				indJ_plus_m_mod_kB = 0;
				indJ_plus_m_mod_kB_times_kC = 0;
			} else {
				indJ_plus_m_mod_kB_times_kC += kC;
			}

			m_2_plus_parity_squared_iter += m_2_plus_parity_start_add;
			m_2_plus_parity_start_add += 8; // adds 8 extra each round compounding
			if (m_2_plus_parity_squared_iter >= kC) m_2_plus_parity_squared_iter -= kC;
			if (m_2_plus_parity_start_add >= kC) m_2_plus_parity_start_add -= kC;
*/
			uint16_t indJ = l_y / kC;
			uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);

			//if (!(test_target == r_target)) printf("fail: meta[0] %u\n",L_entry.meta[0]);
			//if (doPrint) {

			//	printf(" Test target result   : %u ",test_target);
			//	if (r_target == test_target) printf(" SUCCESS!\n"); else printf(" FAIL.\n");
			//	printf(" Desired target result: %u\n",r_target);

			//	printf("\nNext values m:%u\n",m+1);

			//	printf("            indJ_plus_m_mod_kB: %u\n",indJ_plus_m_mod_kB);
			//	printf("   indJ_plus_m_mod_kB_times_kC: %u\n",indJ_plus_m_mod_kB_times_kC);
			//	printf("  m_2_plus_parity_squared_iter: %u\n",m_2_plus_parity_squared_iter);
			//	printf("     m_2_plus_parity_start_add: %u\n",m_2_plus_parity_start_add);
			//}


			//uint16_t r_target = L_targets[parity][l_y][m]; // this performs so badly because this lookup
				// is super-inefficient.


			// find which box our r_target is in, extra the 15bit value from that box
			unsigned int kbc_map = r_target / 2;
			const unsigned int kbc_box_shift = (r_target % 2) * 15;
			unsigned int rmap_value = nick_rmap[kbc_map];
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
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
		global_match_slot = atomicAdd(&match_counts[0],total_matches);
	}

	__syncthreads();


	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match shared_match = matches[i];
		Match_Attack_Pair_Index match = { };
		match.bucket_L_id = global_kbc_L_bucket_id;
		match.idx_L = shared_match.idxL;
		match.idx_R = shared_match.idxR;
		// *could* coelesce pair.meta[0..4] values here and y, instead of splitting y list.
		// suspect splitting y list would be faster.
		match_list[global_match_slot + i] = match;
	}
}


template <typename BUCKETED_ENTRY_IN>
__global__
void gpu_attack_process_t1_pairs_orig(uint16_t table, uint32_t start_kbc_L, uint32_t end_kbc_R,
		const BUCKETED_ENTRY_IN *kbc_local_entries, const int *kbc_local_num_entries,
		Match_Attack_Pair_Index *match_list, int *match_counts) {
	// testmatch count: 33532242
	//   testmatch T1 L time: 12 ms
	const uint16_t NUM_RMAPS = (kBC/2)+1;
	__shared__ unsigned int nick_rmap[NUM_RMAPS]; // positions and counts. Use 30 bits, 15 bits each entry with lower 9 bits for pos, 1024+ for count
	__shared__ uint32_t nick_rmap_extras_rl[32];
	__shared__ uint16_t nick_rmap_extras_ry[32];
	__shared__ uint16_t nick_rmap_extras_pos[32];
	__shared__ Index_Match matches[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ BUCKETED_ENTRY_IN kbc_L_entries[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ BUCKETED_ENTRY_IN kbc_R_entries[KBC_MAX_ENTRIES_PER_BUCKET];
	__shared__ int total_matches;
	__shared__ int global_match_slot;
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
		unsigned int kbc_map = r_y / 2;
		const unsigned int kbc_box_shift = (r_y % 2) * 15;
		unsigned int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)

		unsigned int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
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
			unsigned int kbc_map = r_target / 2;
			const unsigned int kbc_box_shift = (r_target % 2) * 15;
			int add = 1024 << kbc_box_shift; // we add from 10th bit up (shifted by the box it's in)
			unsigned int rmap_value = atomicAdd(&nick_rmap[kbc_map],add); // go ahead and add the counter (which will add in bits 10 and above)
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
		if (total_matches > (KBC_MAX_ENTRIES_PER_BUCKET-1)) {
			printf("PRUNING MATCHES FROM %u to %u\n", total_matches, KBC_MAX_ENTRIES_PER_BUCKET-1);
			total_matches = (KBC_MAX_ENTRIES_PER_BUCKET-1);
		}
		global_match_slot = atomicAdd(&match_counts[0],total_matches);
	}

	__syncthreads();


	// now we go through all our matches and output to next round.
	for (int i=threadIdx.x;i < total_matches;i+=blockDim.x) {
		Index_Match shared_match = matches[i];
		Match_Attack_Pair_Index match = { };
		match.bucket_L_id = global_kbc_L_bucket_id;
		match.idx_L = shared_match.idxL;
		match.idx_R = shared_match.idxR;
		// *could* coelesce pair.meta[0..4] values here and y, instead of splitting y list.
		// suspect splitting y list would be faster.
		match_list[global_match_slot + i] = match;
	}
}


template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_process_t1_matches_list(
		const int MATCHES_COUNT, Match_Attack_Pair_Index *match_list,
		const BUCKETED_ENTRY_IN *kbc_local_entries,
		BUCKETED_ENTRY_OUT *kbc_out, unsigned int *out_kbc_counts,
		const uint32_t KBC_START_L1, const uint32_t KBC_MAX_ENTRIES) {

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < MATCHES_COUNT) {
		Match_Attack_Pair_Index match = match_list[i];
		BUCKETED_ENTRY_OUT pair = {};
		uint32_t local_bucket_id = match.bucket_L_id - KBC_START_L1;
		//printf("reading match %u : bucketL %u  idx_L %u    idx_R %u\n", i, local_bucket_id, match.idx_L, match.idx_R);
		BUCKETED_ENTRY_IN L_Entry = kbc_local_entries[local_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + match.idx_L];
		BUCKETED_ENTRY_IN R_Entry = kbc_local_entries[(local_bucket_id+1) * KBC_MAX_ENTRIES_PER_BUCKET + match.idx_R];

		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, match.bucket_L_id); // make sure this is global bucket id

		pair.meta[0] = L_Entry.meta[0];
		pair.meta[1] = R_Entry.meta[0];
		nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);

		uint32_t kbc_bucket = blake_result / kBC;

		pair.y = (uint32_t) (blake_result % kBC);

		uint32_t kbc_bitmask_bucket = kbc_bucket / 8; \
		uint32_t kbc_bitmask_shift = 4*(kbc_bucket % 8); \
		unsigned int kbc_bitmask_add = 1 << (kbc_bitmask_shift); \
		unsigned int bitadd = atomicAdd(&out_kbc_counts[kbc_bitmask_bucket],kbc_bitmask_add); \
		uint32_t block_slot = bitadd; \
		block_slot = (block_slot >> (kbc_bitmask_shift)) & 0b01111; \

		if (block_slot > KBC_MAX_ENTRIES) {
			printf("block_slot > MAX %u\n", block_slot);
		} else {
			uint32_t pair_address = kbc_bucket * KBC_MAX_ENTRIES + block_slot;
			//if (pair_address >= DEVICE_BUFFER_ALLOCATED_ENTRIES) {
				//printf("ERROR: results address overflow\n");
			//} else {
				kbc_out[pair_address] = pair;
			//}
			}
		//}


	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_process_global_kbc_pairs_list(
		const int PAIRS_COUNT, unsigned int *kbc_pairs_list_L_bucket_ids,
		const BUCKETED_ENTRY_IN *kbc_global_entries_L, const unsigned int *kbc_global_num_entries_L,
		const BUCKETED_ENTRY_IN *kbc_global_entries_R, const unsigned int *kbc_global_num_entries_R,
		Match_Attack_Pair_Index *match_list, int *match_counts,
		const uint32_t KBC_MAX_ENTRIES) {

	// NOTE: possible optimization is to only get y elements of a list instead of ALL the meta...
	// requires splitting the meta and y fields into two separate lists. Alternatively we copy
	// all the meta chunk in this round.

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < PAIRS_COUNT) {
		unsigned int global_kbc_L_bucket_id = kbc_pairs_list_L_bucket_ids[i];

		uint32_t kbc_bitmask_bucket = global_kbc_L_bucket_id / 8;
		uint32_t kbc_bitmask_shift = 4*(global_kbc_L_bucket_id % 8);
		uint32_t bitvalue = kbc_global_num_entries_L[kbc_bitmask_bucket];
		const unsigned int num_L = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;

		kbc_bitmask_bucket = (global_kbc_L_bucket_id + 1) / 8;
		kbc_bitmask_shift = 4*((global_kbc_L_bucket_id + 1) % 8);
		bitvalue = kbc_global_num_entries_R[kbc_bitmask_bucket];
		const unsigned int num_R = (bitvalue >> (kbc_bitmask_shift)) & 0b01111;

		if ((num_L == 0) || (num_R == 0)) {
			printf("ERROR: PAIRS LIST SHOULD NOT HAVE 0 COUNTS\n");
			return; // shouldn't ever happen with a pairs list...
		}

		const uint32_t start_L = global_kbc_L_bucket_id*KBC_MAX_ENTRIES;
		const uint32_t start_R = (global_kbc_L_bucket_id+1)*KBC_MAX_ENTRIES;

		const BUCKETED_ENTRY_IN *kbc_L_entries = &kbc_global_entries_L[start_L];
		const BUCKETED_ENTRY_IN *kbc_R_entries = &kbc_global_entries_R[start_R];

	//   For any 0 <= m < kExtraBitsPow:
	//   yl / kBC + 1 = yR / kBC   AND
	//   (yr % kBC) / kC - (yl % kBC) / kC = m   (mod kB)  AND
	//   (yr % kBC) % kC - (yl % kBC) % kC = (2m + (yl/kBC) % 2)^2   (mod kC)

		for (int pos_R = 0; pos_R < num_R; pos_R+=1) {
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
						int slot = atomicAdd(&match_counts[0],1);
						Match_Attack_Pair_Index match = { };
						match.bucket_L_id = global_kbc_L_bucket_id;
						match.idx_L = pos_L;
						match.idx_R = pos_R;
						// *could* coelesce pair.meta[0..4] values here and y, instead of splitting y list.
						// suspect splitting y list would be faster.
						match_list[slot] = match;
					}
				}
			}
		}
	}
}

template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_process_matches_list(
		uint16_t table,
		const int MATCHES_COUNT, Match_Attack_Pair_Index *match_list,
		const BUCKETED_ENTRY_IN *kbc_global_entries_L,
		const BUCKETED_ENTRY_IN *kbc_global_entries_R,
		BUCKETED_ENTRY_OUT *bucketed_out, int *out_bucket_counts,
		const uint32_t KBC_MAX_ENTRIES, const uint32_t BLOCK_MAX_ENTRIES) {

	// NOTE: possible optimization is to only get y elements of a list instead of ALL the meta...
	// requires splitting the meta and y fields into two separate lists. Alternatively we copy
	// all the meta chunk in this round.

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < MATCHES_COUNT) {
		Match_Attack_Pair_Index match = match_list[i];
		BUCKETED_ENTRY_OUT pair = {};
		BUCKETED_ENTRY_IN L_Entry = kbc_global_entries_L[match.bucket_L_id * KBC_MAX_ENTRIES + match.idx_L];
		BUCKETED_ENTRY_IN R_Entry = kbc_global_entries_R[(match.bucket_L_id+1) * KBC_MAX_ENTRIES + match.idx_R];
		uint64_t blake_result;
		uint64_t calc_y = CALC_Y_BUCKETED_KBC_ENTRY(L_Entry, match.bucket_L_id);
		if (table == 1) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = R_Entry.meta[0];
			nick_blake3(pair.meta, 2, calc_y, &blake_result, 0, NULL);
		} else if (table == 2) {
			pair.meta[0] = L_Entry.meta[0];
			pair.meta[1] = L_Entry.meta[1];
			pair.meta[2] = R_Entry.meta[0];
			pair.meta[3] = R_Entry.meta[1];
			nick_blake3(pair.meta, 4, calc_y, &blake_result, 0, NULL);
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
		}
		uint64_t batch_bucket = blake_result >> (38-6);
		const uint64_t block_mod = (uint64_t) 1 << (38-6);
		pair.y = (uint32_t) (blake_result % block_mod);
		int block_slot = atomicAdd(&out_bucket_counts[batch_bucket],1);
		uint32_t pair_address = batch_bucket * BLOCK_MAX_ENTRIES + block_slot;
		bucketed_out[pair_address] = pair;
	}
}


void attack_method_2(uint32_t bits) {

	// attack method 2 does:

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

	// kbc bucket bitmask: e.g. if 10 bits = 1024 buckets
	// set [64][64] with appropriate bit
	// when chacha, do kbc_bucket and translate to kbc_bit
	// then kbc_bit & [64] for the check, to get true/false
	// then need to find which array to write to. Oh.
	// maybe easier to make array [0..1024] of (Array *), where NULL is in ones not used
	// and just do kbc_array = array[kbc_bucket]
	// if !NULL DO....

	std::cout << "ATTACK METHOD 2" << std::endl;
	std::cout << "   BITS: " << bits << " DIVISOR:" << BITS_DIVISOR
			<< "        target_kbc_L1 " << target_kbc_L1 << " -> bucket L1 " << bucket_L1
			<< "        kbc range: "<< KBC_START_L1 << " - " << (KBC_END_L1) << "kbcs " << std::endl;

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
		MAX_KBCS_POST_T1 = 12;
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
	uint64_t T0_KBC_DEVICE_BUFFER_ALLOCATED_ENTRIES = KBC_ATTACK_NUM_BUCKETS * KBC_MAX_ENTRIES_PER_BUCKET;
		std::cout	  << "   L0 kbc range " << KBC_START_L1 << " to " << KBC_END_L1 << " = " << (KBC_END_L1-KBC_START_L1) << "kbcs " << (100.0*(double)(KBC_END_L1-KBC_START_L1)/(double)kBC_LAST_BUCKET_ID) << "%" << std::endl
			  << "   R0 kbc range " << KBC_START_R1 << " to " << KBC_END_R1 << " = " << (KBC_END_R1-KBC_START_R1) << "kbcs " << (100.0*(double)(KBC_END_R1-KBC_START_R1)/(double)kBC_LAST_BUCKET_ID) << "%" << std::endl
			  << "   KBC_ATTACK_NUM_BUCKETS: " << KBC_ATTACK_NUM_BUCKETS << std::endl
			  << "   MAX BCS POST T1: " << MAX_KBCS_POST_T1 << std::endl
			  << "   BLOCK_MAX_ENTRIES_T2: " << BLOCK_MAX_ENTRIES_T2 << std::endl;


	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();

	char *device_buffer;
	int* device_local_kbc_num_entries_L;
	int* device_local_kbc_num_entries_R;
	int* device_local_kbc_num_entries_L2;
	int* device_local_kbc_num_entries_R2;
	int* device_T2_block_entry_counts;

	const uint64_t T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED = kBC_NUM_BUCKETS * MAX_KBCS_POST_T1 * sizeof(Tx_Bucketed_Meta2);
	std::cout << "  T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED: " << T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED << std::endl;
	std::cout << "                                               * 2 =  " << (T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED * 2) << std::endl;

	const uint64_t T2_BATCH_MATCH_RESULTS_BYTES_NEEDED = (BLOCK_MAX_ENTRIES_T2 * BATCHES) * sizeof(Tx_Bucketed_Meta4);
	std::cout << "  T2_BATCH_MATCH_RESULTS_BYTES_NEEDED: " << T2_BATCH_MATCH_RESULTS_BYTES_NEEDED << std::endl;
		const uint64_t BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED = T0_KBC_DEVICE_BUFFER_ALLOCATED_ENTRIES * sizeof(Tx_Bucketed_Meta2);
	std::cout << "  CHACHA BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED: " << BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED << std::endl;
	std::cout << "                                               * 4 =  " << (BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED * 4) << std::endl;

	const uint64_t TOTAL_BYTES_NEEDED =
			  4 * BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED
			+ 2 * T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED
			+     T2_BATCH_MATCH_RESULTS_BYTES_NEEDED;

	Tx_Bucketed_Meta4 *T2_batch_match_results;
	char *device_local_kbc_entries_L;
	char *device_local_kbc_entries_R;
	char *device_local_kbc_entries_L2;
	char *device_local_kbc_entries_R2;

	Tx_Bucketed_Meta2 *T1_L_kbc_match_results;
	Tx_Bucketed_Meta2 *T1_R_kbc_match_results;
	unsigned int *device_global_kbc_num_entries_L;
	unsigned int *device_global_kbc_num_entries_R;

	//std::cout << "      T1_L_batch_match_results " << DEVICE_BUFFER_ALLOCATED_ENTRIES << " * (UNIT BYTES:" <<  sizeof(Tx_Bucketed_Meta2) << ") = " << (DEVICE_BUFFER_ALLOCATED_ENTRIES * sizeof(Tx_Bucketed_Meta2)) << std::endl;
	//CUDA_CHECK_RETURN(cudaMalloc(&device_buffer, DEVICE_BUFFER_ALLOCATED_ENTRIES * sizeof(Tx_Bucketed_Meta2)));

	std::cout << "      device_local_kbc_num_entries_L " << KBC_ATTACK_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_L, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_R " << KBC_ATTACK_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_R, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_L2 " << KBC_ATTACK_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_L2, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	std::cout << "      device_local_kbc_num_entries_R2 " << KBC_ATTACK_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_local_kbc_num_entries_R2, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));

	// 32 bit...limit to 4 bits = 16 max, = 8 entries per kbc
	std::cout << "      device_global_kbc_num_entries_L " << (kBC_NUM_BUCKETS/8) << " = " << ((kBC_NUM_BUCKETS/8)*sizeof(int)) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_global_kbc_num_entries_L, (kBC_NUM_BUCKETS/8)*sizeof(int)));
	std::cout << "      device_global_kbc_num_entries_R " << (kBC_NUM_BUCKETS/8) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_global_kbc_num_entries_R, (kBC_NUM_BUCKETS/8)*sizeof(int)));


	std::cout << "      device_buffer TOTAL BYTES: " <<  TOTAL_BYTES_NEEDED << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_buffer, TOTAL_BYTES_NEEDED));
	uint64_t MEM_POS = 0;
	device_local_kbc_entries_L = &device_buffer[MEM_POS];
	device_local_kbc_entries_R = &device_buffer[MEM_POS + BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED];
	device_local_kbc_entries_L2 = &device_buffer[MEM_POS + BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED*2];
	device_local_kbc_entries_R2 = &device_buffer[MEM_POS + BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED*3];
	MEM_POS += 4 * BATCH_LOCAL_KBC_ENTRIES_BYTES_NEEDED;
	T1_L_kbc_match_results = (Tx_Bucketed_Meta2 *) &device_buffer[MEM_POS];
	T1_R_kbc_match_results = (Tx_Bucketed_Meta2 *) &device_buffer[MEM_POS + T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED];
	MEM_POS += 2 * T1_BATCH_MATCH_KBC_RESULTS_BYTES_NEEDED;
	T2_batch_match_results = (Tx_Bucketed_Meta4 *) &device_buffer[MEM_POS];
	MEM_POS +=  T2_BATCH_MATCH_RESULTS_BYTES_NEEDED;

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





	Tx_Bucketed_Meta1 *T0_local_kbc_entries_L = (Tx_Bucketed_Meta1 *) &device_local_kbc_entries_L[0]; // will replace...
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_R = (Tx_Bucketed_Meta1 *) &device_local_kbc_entries_R[0];
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_L2 = (Tx_Bucketed_Meta1 *) &device_local_kbc_entries_L2[0]; // will replace...
	Tx_Bucketed_Meta1 *T0_local_kbc_entries_R2 = (Tx_Bucketed_Meta1 *) &device_local_kbc_entries_R2[0];

	std::cout << "Note: sizeof(Tx_Bucketed_Meta1) is " << sizeof(Tx_Bucketed_Meta2)*8 << "bits, when it should be 96 bits" << std::endl;

	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_L2, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(device_local_kbc_num_entries_R2, 0, KBC_ATTACK_NUM_BUCKETS*sizeof(int)));

	std::cout << "Doing T1" << std::endl;
	auto t1_start = std::chrono::high_resolution_clock::now();
	auto chacha_start = std::chrono::high_resolution_clock::now();
	//gpu_chacha8_k32_kbc_ranges_LR<<<numBlocks, blockSize>>>(calc_N, chacha_input,
	//			T0_local_kbc_entries_L, device_local_kbc_num_entries_L, KBC_START_L1, KBC_END_L1,
	//			T0_local_kbc_entries_R, device_local_kbc_num_entries_R, KBC_START_R1, KBC_END_R1);
	gpu_chacha8_k32_kbc_ranges_LR1LR2<<<numBlocks, blockSize>>>(calc_N, chacha_input,
					T0_local_kbc_entries_L, device_local_kbc_num_entries_L, KBC_START_L1, KBC_END_L1,
					T0_local_kbc_entries_R, device_local_kbc_num_entries_R, KBC_START_R1, KBC_END_R1,
					T0_local_kbc_entries_L2, device_local_kbc_num_entries_L2, KBC_START_L2, KBC_END_L2,
					T0_local_kbc_entries_R2, device_local_kbc_num_entries_R2, KBC_START_R2, KBC_END_R2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto chacha_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   chacha L1 time: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
	//gpu_list_local_kbc_entries<<<1,1>>>(device_local_kbc_num_entries_L2, 0, 100, 1);

	Match_Attack_Pair_Index *match_list;
	int *match_counts;
	CUDA_CHECK_RETURN(cudaMalloc(&match_list, 67108864*sizeof(Match_Attack_Pair_Index)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&match_counts, sizeof(unsigned int)));
	match_counts[0] = 0;
	auto testmatchT1_start = std::chrono::high_resolution_clock::now();
	gpu_attack_process_t1_pairs<Tx_Bucketed_Meta1><<<(KBC_END_L1 - KBC_START_L1), 256>>>(1, KBC_START_L1, KBC_END_L1,
			T0_local_kbc_entries_L, device_local_kbc_num_entries_L,
			match_list,match_counts);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto testmatchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   testmatch count: " << match_counts[0] << std::endl;
	std::cout << "   testmatch T1 L time: " << std::chrono::duration_cast<milli>(testmatchT1_finish - testmatchT1_start).count() << " ms\n";

	// CODE BELOW CRASHES

	int matchT1_count = match_counts[0];
	const int matchT1_blockSize = 256;
	const int matchT1_numBlocks = (matchT1_count + matchT1_blockSize - 1) / matchT1_blockSize;
	auto bestmatchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, (kBC_NUM_BUCKETS/8)*sizeof(int)));
	gpu_attack_process_t1_matches_list<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<matchT1_numBlocks,matchT1_blockSize>>>(
	//gpu_attack_process_t1_matches_list<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<matchT1_count,1>>>(
			matchT1_count, match_list,
			T0_local_kbc_entries_L,
			T1_L_kbc_match_results, device_global_kbc_num_entries_L,
			KBC_START_L1, MAX_KBCS_POST_T1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto bestmatchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   blake match T1 L time: " << std::chrono::duration_cast<milli>(bestmatchT1_finish - bestmatchT1_start).count() << " ms\n";
	std::cout << "   FINAL match T1 L time: " << std::chrono::duration_cast<milli>(bestmatchT1_finish - testmatchT1_start).count() << " ms\n";

	//gpu_list_local_kbc_entries_bitmask<<<1,1>>>(device_global_kbc_num_entries_L, 0, 100, 1);

	auto matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, (kBC_NUM_BUCKETS/8)*sizeof(int)));
	gpu_attack_find_t1_matches_out_kbc<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_L1 - KBC_START_L1), 256>>>(1, KBC_START_L1, KBC_END_L1,
			T0_local_kbc_entries_L, device_local_kbc_num_entries_L,
			T1_L_kbc_match_results, device_global_kbc_num_entries_L, MAX_KBCS_POST_T1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 L time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";
	//gpu_list_local_kbc_entries_bitmask<<<1,1>>>(device_global_kbc_num_entries_L, 0, 100, 1);

	matchT1_start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_R, 0, (kBC_NUM_BUCKETS/8)*sizeof(int)));
	gpu_attack_find_t1_matches_out_kbc<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2><<<(KBC_END_R1 - KBC_START_R1), 256>>>(1, KBC_START_R1, KBC_END_R1,
				T0_local_kbc_entries_R, device_local_kbc_num_entries_R,
				T1_R_kbc_match_results, device_global_kbc_num_entries_R, MAX_KBCS_POST_T1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	matchT1_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T1 R time: " << std::chrono::duration_cast<milli>(matchT1_finish - matchT1_start).count() << " ms\n";

	//gpu_list_local_kbc_entries_bitmask<<<1,1>>>(device_global_kbc_num_entries_R, 0, 100, 1);

	// TODO: need to do a "pairing" pass, where we just scan through each bucket and spit out a list of kbc pairs
	// then, on a second pass, process the pairs with compute method. This way all threads are going to be working
	// and it should be near instant.
	// NOTE: will have to handle pairing pass having more than one entry
	// ALSO TRY: single pass where we compute on the fly, but probably it will store all the 0 entries
	//   e.g. T2 9 bit, expect 16000 matches from 18188177 buckets = 1 in 1100 buckets

	// after t1 pairs output to kbc list, for t2 pairing we first filter all eligible bucket ids.
	unsigned int *kbc_pairs_list_L_bucket_ids;
	int *pairs_count;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_pairs_list_L_bucket_ids, kBC_NUM_BUCKETS*sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_count, sizeof(unsigned int)));
	pairs_count[0] = 0;
	//CUDA_CHECK_RETURN(cudaMemset(pairs_count, 0, sizeof(int)));

	auto pairingT2_start = std::chrono::high_resolution_clock::now();
	const int pair_blockSize = 256; // # of threads per block, maximum is 1024.
	const uint32_t pair_numBlocks = (kBC_NUM_BUCKETS + pair_blockSize - 1) / pair_blockSize;
	gpu_attack_get_kbcs_with_pairs_from_global_kbcs<<<pair_numBlocks,pair_blockSize>>>(
			device_global_kbc_num_entries_L,device_global_kbc_num_entries_R,
			kbc_pairs_list_L_bucket_ids, pairs_count);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto pairingT2_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   pairs count: " << pairs_count[0] << std::endl;
	std::cout << "   pairing T2 L time: " << std::chrono::duration_cast<milli>(pairingT2_finish - pairingT2_start).count() << " ms\n";

	//Match_Attack_Pair_Index *match_list;
	//int *match_counts;
	//CUDA_CHECK_RETURN(cudaMalloc(&match_list, 2*NUM_EXPECTED_ENTRIES_T2_MATCHES*sizeof(Match_Attack_Pair_Index)));
	//CUDA_CHECK_RETURN(cudaMallocManaged(&match_counts, sizeof(unsigned int)));
	match_counts[0] = 0;


	auto processT2_start = std::chrono::high_resolution_clock::now();
	int process_count = pairs_count[0];
	const int process_blockSize = 256;
	const int process_numBlocks = (process_count + process_blockSize - 1) / process_blockSize;
	gpu_attack_process_global_kbc_pairs_list<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<process_numBlocks,process_blockSize>>>(
			process_count, kbc_pairs_list_L_bucket_ids,
			T1_L_kbc_match_results, device_global_kbc_num_entries_L,
			T1_R_kbc_match_results, device_global_kbc_num_entries_R,
			match_list, match_counts,
			MAX_KBCS_POST_T1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto processT2_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   t2 match_counts: " << match_counts[0] << std::endl;
	std::cout << "   process T2 L time: " << std::chrono::duration_cast<milli>(processT2_finish - processT2_start).count() << " ms\n";

	CUDA_CHECK_RETURN(cudaMemset(device_T2_block_entry_counts, 0, (BATCHES)*sizeof(int))); // 128 is 2046, 384 is 1599

	auto matchT2_start = std::chrono::high_resolution_clock::now();
	int matches_count = match_counts[0];
	const int match_blockSize = 256;
	const int match_numBlocks = (matches_count + match_blockSize - 1) / match_blockSize;
	gpu_attack_process_matches_list<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<match_numBlocks,match_blockSize>>>(
			2,
			matches_count, match_list,
			T1_L_kbc_match_results,
			T1_R_kbc_match_results,
			T2_batch_match_results, device_T2_block_entry_counts,
			MAX_KBCS_POST_T1, BLOCK_MAX_ENTRIES_T2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto matchT2_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T2 L time: " << std::chrono::duration_cast<milli>(matchT2_finish - matchT2_start).count() << " ms\n";

	/*
	 *   process T2 L time: 0 ms
   match T2 L time: 12 ms
Freeing memory...
GPU DISPLAY T2 MATCH RESULTS:
  block 22 entry 198   x1:1320788535  x2:3465356684  x3:2131394289  x4:606438761
  TOTAL: 16498
	 */

	/*auto matchT2_start = std::chrono::high_resolution_clock::now();
	gpu_attack_find_tx_LR_matches_global<Tx_Bucketed_Meta2,Tx_Bucketed_Meta4><<<kBC_NUM_BUCKETS, 8>>>(2, 0, kBC_NUM_BUCKETS,
			T1_L_kbc_match_results, device_global_kbc_num_entries_L,
			T1_R_kbc_match_results, device_global_kbc_num_entries_R,
			T2_batch_match_results, device_T2_block_entry_counts,
			MAX_KBCS_POST_T1, BLOCK_MAX_ENTRIES_T2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto matchT2_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   match T2 L time: " << std::chrono::duration_cast<milli>(matchT2_finish - matchT2_start).count() << " ms\n";
*/
	auto compute_only_finish = std::chrono::high_resolution_clock::now();

	gpu_display_t2_match_results<<<1,1>>>(T2_batch_match_results, device_T2_block_entry_counts, BLOCK_MAX_ENTRIES_T2);

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_L));
	CUDA_CHECK_RETURN(cudaFree(device_local_kbc_num_entries_R));
	//CUDA_CHECK_RETURN(cudaFree(device_block_entry_counts));
	CUDA_CHECK_RETURN(cudaFree(device_buffer));

	auto attack_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "   attack total time: " << std::chrono::duration_cast<milli>(attack_finish - attack_start).count() << " ms\n";
	std::cout << "end." << std::endl;
}




#endif /* ATTACK_METHOD_2_HPP_ */
