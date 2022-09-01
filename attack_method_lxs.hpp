/*
 * attack_method_lxs.hpp
 *
 *  Created on: Nov 6, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_LXS_HPP_
#define ATTACK_METHOD_LXS_HPP_

#include <cuda/barrier> // memcpy_async

const uint32_t CHACHA_NUM_BATCHES_BITS = 3;
const uint32_t CHACHA_NUM_BATCHES = 1 << CHACHA_NUM_BATCHES_BITS;
const uint32_t CHACHA_TOTAL_ENTRIES_PER_BATCH = UINT_MAX / CHACHA_NUM_BATCHES;
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

// MASKED method for counter 10 bits, should help cache by 3x
#define KBCFILTER_mask(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	if ((kbc_bucket_id >= KBC_START) && (kbc_bucket_id <= KBC_END)) { \
		uint32_t local_kbc_bucket_id = kbc_bucket_id - KBC_START; \
		int kbc_bitmask_bucket = local_kbc_bucket_id / 10; \
		int kbc_bit_slot = local_kbc_bucket_id % 10; \
		unsigned int kbc_mask = 1 << kbc_bit_slot; \
		unsigned int add = atomicAdd(&kbc_local_num_entries[kbc_bitmask_bucket],kbc_mask); \
		unsigned int slot = (add >> kbc_bit_slot) & 0b01111111111; \
		F1_Bucketed_kBC_Entry entry = { (x+i), (uint32_t) (y % kBC) }; \
		if (slot >= KBC_MAX_ENTRIES_PER_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u", KBC_MAX_ENTRIES_PER_BUCKET, slot); } \
		uint32_t entries_address = local_kbc_bucket_id * KBC_MAX_ENTRIES_PER_BUCKET + slot; \
		kbc_local_entries[entries_address] = entry; \
	} \
}

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
		const __restrict__ uint32_t *input, F1_Bucketed_kBC_Entry *kbc_local_entries, unsigned int *kbc_local_num_entries,
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
		KBCFILTER(x0,0);KBCFILTER(x1,1);KBCFILTER(x2,2);KBCFILTER(x3,3);
		KBCFILTER(x4,4);KBCFILTER(x5,5);KBCFILTER(x6,6);KBCFILTER(x7,7);
		KBCFILTER(x8,8);KBCFILTER(x9,9);KBCFILTER(x10,10);KBCFILTER(x11,11);
		KBCFILTER(x12,12);KBCFILTER(x13,13);KBCFILTER(x14,14);KBCFILTER(x15,15);
	}
}

#define ATTACK_INTO_KBC_YS(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	int slot = atomicAdd(&kbc_global_num_entries_L[kbc_bucket_id],1); \
	if (slot >= MAX_LXS_PER_KBC_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u\n", MAX_LXS_PER_KBC_BUCKET, slot); } \
	uint32_t entries_address = kbc_bucket_id * MAX_LXS_PER_KBC_BUCKET + slot; \
	kbc_global_Ly_entries_L[entries_address] = y; \
	kbc_x_entries[entries_address] = (x + i); \
}

// can hold 6 entries of 5 bits each = 5*6 = 30 bits.
#define KBC_MASK_SHIFT 5
#define KBC_MASK_MOD 6
#define KBC_MASK_BITS 0b011111
#define ATTACK_INTO_KBC_YS_BITMASK(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	uint32_t kbc_bucket_id = uint32_t (y / kBC); \
	uint32_t kbc_bitmask_bucket = kbc_bucket_id / KBC_MASK_MOD; \
	uint32_t kbc_bitmask_shift = KBC_MASK_SHIFT * (kbc_bucket_id % KBC_MASK_MOD); \
	uint32_t add = 1 << kbc_bitmask_shift; \
	uint slot_value = atomicAdd(&kbc_global_num_entries_L[kbc_bitmask_bucket],add); \
	uint slot = (slot_value >> kbc_bitmask_shift) & KBC_MASK_BITS; \
	if (slot >= MAX_LXS_PER_KBC_BUCKET) { printf("ERROR KBC OVERFLOW MAX:%u actual:%u\n", MAX_LXS_PER_KBC_BUCKET, slot); } \
	uint32_t entries_address = kbc_bucket_id * MAX_LXS_PER_KBC_BUCKET + slot; \
	kbc_global_Ly_entries_L[entries_address] = y; \
	kbc_x_entries[entries_address] = (x + i); \
}

#define CHACHA_OUT(chacha_y,i) \
{ \
	chachas[x+i] = chacha_y; \
}

// uint16_t indJ = l_y / kC;
// uint16_t r_target = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + l_y) % kC);
// OK, so we get all our Lx's and get their Ly's, and then compute their target Lys,
// but then we have to write this to huge data of global_target_rys which is 38 bits.
// even with 1 bit per entry it's too much data, unless we remove bottom bits and get some false positives.
// 2^38 bits / 8 = 2^34 bits, >> 2 = 2^32 bits...means we can do 4 Lx passes and 4 Rx passes...interesting...
// will have to do binary tree search for rxs...fuck.
__global__
void gpu_chacha8_only_chacha_results(const uint32_t N,
		const __restrict__ uint32_t *input,
		uint32_t *chachas)
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
		CHACHA_OUT(x0,0);CHACHA_OUT(x1,1);CHACHA_OUT(x2,2);CHACHA_OUT(x3,3);
		CHACHA_OUT(x4,4);CHACHA_OUT(x5,5);CHACHA_OUT(x6,6);CHACHA_OUT(x7,7);
		CHACHA_OUT(x8,8);CHACHA_OUT(x9,9);CHACHA_OUT(x10,10);CHACHA_OUT(x11,11);
		CHACHA_OUT(x12,12);CHACHA_OUT(x13,13);CHACHA_OUT(x14,14);CHACHA_OUT(x15,15);
	}
}

#define CHACHA_BUCKET_OUT(chacha_y,i) \
{ \
	uint32_t rx_bucket = chacha_y / CHACHA_BUCKET_DIVISOR; \
	if ((rx_bucket > CHACHA_BUCKET_RANGE_MIN) && (rx_bucket <= CHACHA_BUCKET_RANGE_MAX)) { \
		rx_bucket = rx_bucket - CHACHA_BUCKET_RANGE_MIN; \
		uint slot = atomicAdd(&shared_rx_counts[rx_bucket],1); \
		if (slot > MAX_ENTRIES_PER_LOCAL_BUCKET) printf("CHACHA BUCKET OUT SLOT OVERFLOW %u\n", slot); \
		chachas_buffer[rx_bucket * NUM_LOCAL_BUCKETS + slot] = chacha_y; \
		xs_buffer[rx_bucket * NUM_LOCAL_BUCKETS + slot] = (x+i); \
	} \
}
//printf("PASSED FILTER   local rx bucket %u   slot %u\n", chacha_y, rx_bucket+CHACHA_BUCKET_MIN, rx_bucket, slot); \
	printf("chacha y: %u rx_bucket %u \n", chacha_y, rx_bucket); \ chachas[address] = chacha_y; \
		//rxs[address] = (x+i); \

#define ATTACK_WRITE_CHACHAS32_PAIR(chacha_y,i) \
{ \
	xchacha_pair pair = { base_x + i, chacha_y }; \
	shared_chachas[threadIdx.x*32+i] = pair; \
	const uint32_t bucket_id = pair.chacha >> (32 - CHACHA_BUCKET_BITS); \
	atomicAdd(&shared_counts[bucket_id],1); \
}

// run with 128 blocksize, more doesn't matter.
template<int NUM_BUCKETS>
__global__
void gpu_chacha8_k32_write_chachas32_buckets(const uint32_t N, const uint32_t X_START,
		const uint32_t CHACHA_MAX_PER_BUCKET,
		const __restrict__ uint32_t *input,
		xchacha_pair *xchachas_buckets, uint *xchachas_bucket_counts)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	__shared__ xchacha_pair shared_chachas[128*32]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.
	__shared__ uint shared_counts[NUM_BUCKETS];
	__shared__ uint global_counts[NUM_BUCKETS];

	if (blockDim.x > 128) printf("MUST HAVE BLOCKSIZE 128 (RECOMMENDED) OR LESS, OR INCREASED SHARED MEM TO MORE\n");

	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 32;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 32; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
		shared_counts[i] = 0;
	}
	__syncthreads();

	const int j = 0;
	if (x_group < end_n) {
		uint32_t pos = x_group * 2 + X_START/16;
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
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+0],0);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+1],1);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+2],2);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+3],3);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+4],4);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+5],5);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+6],6);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+7],7);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+8],8);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+9],9);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+10],10);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+11],11);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+12],12);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+13],13);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+14],14);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+15],15);

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
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+0],16+0);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+1],16+1);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+2],16+2);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+3],16+3);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+4],16+4);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+5],16+5);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+6],16+6);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+7],16+7);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+8],16+8);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+9],16+9);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+10],16+10);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+11],16+11);
		ATTACK_WRITE_CHACHAS32_PAIR(datax[j+12],16+12);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+13],16+13);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+14],16+14);ATTACK_WRITE_CHACHAS32_PAIR(datax[j+15],16+15);

	}
	__syncthreads();
	for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
		global_counts[i] = atomicAdd(&xchachas_bucket_counts[i],shared_counts[i]);
		shared_counts[i] = 0;
	}
	__syncthreads();
	for (int i=threadIdx.x;i<blockDim.x*32;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n",i,base_x + i);
		xchacha_pair pair = shared_chachas[i];

		uint32_t bucket_id = pair.chacha >> (32 - CHACHA_BUCKET_BITS); // 16 buckets
		uint slot = global_counts[bucket_id] + atomicAdd(&shared_counts[bucket_id],1);
		if (slot > CHACHA_MAX_PER_BUCKET) printf("Overflow CHACHA_MAX_PER_BUCKET %u SLOT %u\n", CHACHA_MAX_PER_BUCKET, slot);
		else xchachas_buckets[CHACHA_MAX_ENTRIES_PER_BUCKET * bucket_id + slot] = shared_chachas[i];
	}
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

	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 32;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
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


__global__
void gpu_chacha8_tag_rxs_from_chacha(const uint32_t N,
		const __restrict__ uint32_t *input,
		const uint16_t *kbc_global_Ly_entries_L, const unsigned int *kbc_global_num_entries_L, const uint32_t MAX_LXS_PER_KBC_BUCKET,
		uint32_t *chachas)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < N) {
		uint32_t chacha_y = chachas[x];
		uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26);
		int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1;
		if (kbc_bucket_id_L > 0) {
			int num = kbc_global_num_entries_L[kbc_bucket_id_L];
			for (int nm=0;nm<num;nm++) {
				bool isMatch = false;
				int16_t yl_kbc = kbc_global_Ly_entries_L[kbc_bucket_id_L * MAX_LXS_PER_KBC_BUCKET + nm];
				int16_t yr_kbc = Ry % kBC;
				int16_t yr_bid = yr_kbc / kC;
				int16_t yl_bid = yl_kbc / kC;
				int16_t formula_one = yr_bid - yl_bid;
				if (formula_one < 0) {
					formula_one += kB;
				}
				int16_t m = formula_one;
				if (m >= kB) {
					m -= kB;
				}
				if (m < 64) {
					int16_t yl_cid = yl_kbc % kC;
					int16_t yr_cid = yr_kbc % kC;
					int16_t parity = (kbc_bucket_id_L) % 2;
					int16_t m2_parity_squared = (((2 * m) + parity) * ((2 * m) + parity)) % kC;
					int16_t formula_two = yr_cid - yl_cid;
					if (formula_two < 0) {
						formula_two += kC;
					}
					if (formula_two == m2_parity_squared) {
						isMatch = true;
					}
				}
				if (isMatch) {
					chachas[x] = 0;
				}
			}
		}
	}

}

__global__
void gpu_chacha8_filter_rxs_from_chacha(const uint32_t N, const uint32_t *chachas, uint32_t *rxs, int *rx_count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < N) {
		uint32_t chacha_y = chachas[x];
		if (chacha_y == 0) {
			int slot = atomicAdd(&rx_count[0], 1);
			rxs[slot] = x;
		}
	}

}

__global__
void gpu_chacha8_set_Lxs_into_kbc_ys(const uint32_t N,
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
		ATTACK_INTO_KBC_YS(x0,0);ATTACK_INTO_KBC_YS(x1,1);ATTACK_INTO_KBC_YS(x2,2);ATTACK_INTO_KBC_YS(x3,3);
		ATTACK_INTO_KBC_YS(x4,4);ATTACK_INTO_KBC_YS(x5,5);ATTACK_INTO_KBC_YS(x6,6);ATTACK_INTO_KBC_YS(x7,7);
		ATTACK_INTO_KBC_YS(x8,8);ATTACK_INTO_KBC_YS(x9,9);ATTACK_INTO_KBC_YS(x10,10);ATTACK_INTO_KBC_YS(x11,11);
		ATTACK_INTO_KBC_YS(x12,12);ATTACK_INTO_KBC_YS(x13,13);ATTACK_INTO_KBC_YS(x14,14);ATTACK_INTO_KBC_YS(x15,15);
	}
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



#define ATTACK_FILTER_RXS(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	if ((kbc_bucket_id_L > KBC_MIN_RANGE) && (kbc_bucket_id_L <= KBC_MAX_RANGE)) { \
		int num = kbc_global_num_entries_L[kbc_bucket_id_L]; \
		for (int nm=0;nm<num;nm++) { \
			isMatch = false; \
			int16_t yl_kbc = kbc_global_Ly_entries_L[kbc_bucket_id_L * MAX_LXS_PER_KBC_BUCKET + nm]; \
			CHECK_MATCH(); \
			if (isMatch) { \
				int slot = atomicAdd(&rx_count[0],1); \
				rxs[slot] = (x+i); \
			} \
		} \
	} \
}

#define ATTACK_FILTER_RXS_single_match(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	isMatch = false; \
	if (kbc_bucket_id_L > 0) { \
		uint64_t Ly = kbc_global_Ly_entries_L[kbc_bucket_id_L * MAX_LXS_PER_KBC_BUCKET]; \
		if (Ly > 0) { \
			CHECK_MATCH(); \
		} \
	} \
	if (isMatch) { \
		int slot = atomicAdd(&rx_count[0],1); \
		rxs[slot] = (x+i); \
	} \
}


__global__
void gpu_chacha8_filter_rxs(const uint32_t N,
		const __restrict__ uint32_t *input,
		const uint16_t* __restrict__ kbc_global_Ly_entries_L, const unsigned int* __restrict__ kbc_global_num_entries_L, uint32_t MAX_LXS_PER_KBC_BUCKET,
		uint32_t * __restrict__ rxs, int *rx_count,
		const uint32_t KBC_MIN_RANGE, const uint32_t KBC_MAX_RANGE)
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
		bool isMatch = false;
		ATTACK_FILTER_RXS(x0,0);ATTACK_FILTER_RXS(x1,1);ATTACK_FILTER_RXS(x2,2);ATTACK_FILTER_RXS(x3,3);
		ATTACK_FILTER_RXS(x4,4);ATTACK_FILTER_RXS(x5,5);ATTACK_FILTER_RXS(x6,6);ATTACK_FILTER_RXS(x7,7);
		ATTACK_FILTER_RXS(x8,8);ATTACK_FILTER_RXS(x9,9);ATTACK_FILTER_RXS(x10,10);ATTACK_FILTER_RXS(x11,11);
		ATTACK_FILTER_RXS(x12,12);ATTACK_FILTER_RXS(x13,13);ATTACK_FILTER_RXS(x14,14);ATTACK_FILTER_RXS(x15,15);
	}
}

__global__
void gpu_chacha8_filter_rxs_from_bucket_batch_async(
		const uint32_t N,
		const xchacha_pair* __restrict__ xchachas,
		const uint16_t* __restrict__ kbc_global_Ly_entries_L,
		const unsigned int* __restrict__ kbc_global_num_entries_L,
		uint32_t MAX_LXS_PER_KBC_BUCKET,
		uint32_t * __restrict__ rxs,
		int *rx_count)
{
	__shared__ uint16_t copy_Ly_entries[64];

	cuda::barrier<cuda::thread_scope_system> bar;
	init(&bar, 1);

	int num;
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		xchacha_pair entry = xchachas[i];
		uint64_t Ry = (((uint64_t) entry.chacha) << 6) + (entry.x >> 26);
		int kbc_bucket_id_R = (uint32_t (Ry / kBC));
		if (kbc_bucket_id_R > 0) {
			int kbc_bucket_id_L = kbc_bucket_id_R - 1;
			//printf("entry x:%u chacha:%u\n", entry.x, entry.chacha, kbc_bucket_id_L);
			num = kbc_global_num_entries_L[kbc_bucket_id_L];
			cuda::memcpy_async(&copy_Ly_entries[0],
							        &kbc_global_Ly_entries_L[kbc_bucket_id_L * MAX_LXS_PER_KBC_BUCKET], sizeof(uint16_t)*num, bar);
			bar.arrive_and_wait();
			for (int nm=0;nm<num;nm++) {
				bool isMatch = false;

				int16_t yl_kbc = copy_Ly_entries[nm];
				CHECK_MATCH();
				if (isMatch) {
					int slot = atomicAdd(&rx_count[0],1);
					rxs[slot] = entry.x;
				}
			}
		}
	}
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

#define ATTACK_SET_BITMASK(chacha_y,i) \
{ \
	uint64_t y = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id = (uint32_t (y / kBC)); \
	int kbc_bitmask_bucket = kbc_bucket_id / 32; \
	int kbc_bit_slot = kbc_bucket_id % 32; \
	unsigned int kbc_mask = 1 << kbc_bit_slot; \
	atomicOr(&kbc_global_bitmask[kbc_bitmask_bucket],kbc_mask); \
}

#define ATTACK_FILTER_BITMASK_batch64(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	if (kbc_bucket_id_L > 0) { \
		int kbc_bitmask_bucket = kbc_bucket_id_L / 32; \
		int kbc_bit_slot = kbc_bucket_id_L % 32; \
		unsigned int kbc_mask = 1 << kbc_bit_slot; \
		unsigned int kbc_value = kbc_global_bitmask[kbc_bitmask_bucket]; \
		if ((kbc_mask & kbc_value) > 0) { \
			uint32_t batch_id = kbc_bucket_id_L >> (32-6); \
			int slot = atomicAdd(&rx_count[batch_id],1); \
			rxs[batch_id * RX_MAX_ENTRIES_PER_BATCH + slot] = (x+i); \
		} \
	} \
}

#define ATTACK_FILTER_BITMASK(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	if (kbc_bucket_id_L > 0) { \
		int kbc_bitmask_bucket = kbc_bucket_id_L / 32; \
		int kbc_bit_slot = kbc_bucket_id_L % 32; \
		unsigned int kbc_mask = 1 << kbc_bit_slot; \
		unsigned int kbc_value = kbc_global_bitmask[kbc_bitmask_bucket]; \
		if ((kbc_mask & kbc_value) > 0) { \
			int slot = atomicAdd(&rx_local_count,1); \
			shared_rxs[slot] = (x+i); \
		} \
	} \
}

#define ATTACK_FILTER_BITMASK_origbeforeaddingshared(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	if (kbc_bucket_id_L > 0) { \
		int kbc_bitmask_bucket = kbc_bucket_id_L / 32; \
		int kbc_bit_slot = kbc_bucket_id_L % 32; \
		unsigned int kbc_mask = 1 << kbc_bit_slot; \
		unsigned int kbc_value = kbc_global_bitmask[kbc_bitmask_bucket]; \
		if ((kbc_mask & kbc_value) > 0) { \
			int slot = atomicAdd(&rx_count[0],1); \
			rxs[slot] = (x+i); \
		} \
	} \
}

__global__
void gpu_chacha8_set_Lxs_into_kbc_bitmask(const uint32_t N,
		const __restrict__ uint32_t *input,
		unsigned int* kbc_global_bitmask)
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
		ATTACK_SET_BITMASK(x0,0);ATTACK_SET_BITMASK(x1,1);ATTACK_SET_BITMASK(x2,2);ATTACK_SET_BITMASK(x3,3);
		ATTACK_SET_BITMASK(x4,4);ATTACK_SET_BITMASK(x5,5);ATTACK_SET_BITMASK(x6,6);ATTACK_SET_BITMASK(x7,7);
		ATTACK_SET_BITMASK(x8,8);ATTACK_SET_BITMASK(x9,9);ATTACK_SET_BITMASK(x10,10);ATTACK_SET_BITMASK(x11,11);
		ATTACK_SET_BITMASK(x12,12);ATTACK_SET_BITMASK(x13,13);ATTACK_SET_BITMASK(x14,14);ATTACK_SET_BITMASK(x15,15);
	}
}



__global__
void gpu_chacha8_filter_rxs_by_kbc_bitmask(const uint32_t N,
		const __restrict__ uint32_t *input,
		const unsigned int* __restrict__ kbc_global_bitmask,
		uint32_t * __restrict__ rxs, int *rx_count,
		const uint32_t RX_BATCHES, const uint32_t RX_MAX_ENTRIES_PER_BATCH)
{
	uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	__shared__ uint32_t shared_rxs[1024];
	__shared__ int rx_local_count;
	__shared__ int global_slot;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	int stride = blockDim.x * gridDim.x;
	const uint32_t end_n = N / 16; // 16 x's in each group
	if (threadIdx.x == 0) {
		rx_local_count = 0;
	}
	for (uint32_t x_group = index; x_group <= end_n; x_group += stride) {
		uint32_t x = x_group << 4;//  *16;
		uint32_t pos = x_group;
		__syncthreads();

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
		ATTACK_FILTER_BITMASK(x0,0);ATTACK_FILTER_BITMASK(x1,1);ATTACK_FILTER_BITMASK(x2,2);ATTACK_FILTER_BITMASK(x3,3);
		ATTACK_FILTER_BITMASK(x4,4);ATTACK_FILTER_BITMASK(x5,5);ATTACK_FILTER_BITMASK(x6,6);ATTACK_FILTER_BITMASK(x7,7);
		ATTACK_FILTER_BITMASK(x8,8);ATTACK_FILTER_BITMASK(x9,9);ATTACK_FILTER_BITMASK(x10,10);ATTACK_FILTER_BITMASK(x11,11);
		ATTACK_FILTER_BITMASK(x12,12);ATTACK_FILTER_BITMASK(x13,13);ATTACK_FILTER_BITMASK(x14,14);ATTACK_FILTER_BITMASK(x15,15);

		__syncthreads();
		if (threadIdx.x == 0) {
			global_slot = atomicAdd(&rx_count[0],rx_local_count);
			rx_local_count = 0;
		}
		__syncthreads();
		for (int i=threadIdx.x;i<rx_local_count;i+=blockDim.x) {
			rxs[global_slot+i] = shared_rxs[i];
		}

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
	for (int i=threadIdx.x;i<NUM;i+=blockDim.x) {
		unsigned int kbc_count = kbc_counts[i];
		if (printAll) printf("id: %u count: %u\n", i, kbc_count);
		atomicMax(&max_kbc_count, kbc_count);
		atomicAdd(&sum_kbc_count, kbc_count);
	}
	__syncthreads();
	if (threadIdx.x == 0) printf("counter list counts  SUM:%u   MAX:%u\n", sum_kbc_count, max_kbc_count);
}

void attack_method_lxs(uint32_t num_lxs) {
	num_lxs =  110000000;
	std::cout << "ATTACK METHOD LXS: " << num_lxs << std::endl;

	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();

	unsigned int *device_global_kbc_num_entries_L;
	uint16_t *kbc_Ly_entries; // the y % kbc bucketed entries
	uint32_t *kbc_x_entries;  // the associated x value for the y pairing
	uint32_t *rx_match_list;

	const uint32_t NUM_KBC_RANGE_BATCHES = 1;
	const uint32_t MAX_LXS_PER_KBC_BUCKET = 24; // 24 for 110,000,000
	const uint32_t MAX_RX_MATCHES = num_lxs;
	const uint32_t RX_BATCHES = 1;
	const uint32_t RX_MAX_ENTRIES_PER_BATCH = MAX_RX_MATCHES / RX_BATCHES;
	std::cout   << "CHACHA NUM BATCHES    : " << CHACHA_NUM_BATCHES << std::endl
				<< "CHACHA_TOTAL_ENTRIES_PER_BATCH : " << CHACHA_TOTAL_ENTRIES_PER_BATCH << std::endl
			    << "CHACHA BUCKETS        : " << CHACHA_NUM_BUCKETS << std::endl
			    << "CHACHA_MAX_ENTRIES_PER_BUCKET : " << CHACHA_MAX_ENTRIES_PER_BUCKET << std::endl
			    << "CHACHA_OUT_MAX_ENTRIES_NEEDED : " << CHACHA_OUT_MAX_ENTRIES_NEEDED << std::endl
	            << "MAX_RX_MATCHES: " << MAX_RX_MATCHES << std::endl
				<< "RX_BATCHES: " << RX_BATCHES << std::endl
			  << "RX_MAX_ENTRIES_PER_BATCH: " << RX_MAX_ENTRIES_PER_BATCH << std::endl;

	xchacha_pair *xchachas;
	uint *xchachas_bucket_counts;
	CUDA_CHECK_RETURN(cudaMallocManaged(&xchachas_bucket_counts, CHACHA_NUM_BUCKETS*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(xchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));

	std::cout << "      xchachas size:" << (sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&xchachas, sizeof(xchacha_pair)*CHACHA_OUT_MAX_ENTRIES_NEEDED));

	// alloc for lx's
	std::cout << "      kbc_Ly_entries MAX_LXS: " << MAX_LXS_PER_KBC_BUCKET << " TOTAL BYTES: " <<  (MAX_LXS_PER_KBC_BUCKET * sizeof(uint64_t) * kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_Ly_entries, (MAX_LXS_PER_KBC_BUCKET * sizeof(uint16_t) * kBC_NUM_BUCKETS)));
	std::cout << "      kbc_x_entries MAX_LXS: " << MAX_LXS_PER_KBC_BUCKET << " TOTAL BYTES: " <<  (MAX_LXS_PER_KBC_BUCKET * sizeof(uint64_t) * kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_x_entries, (MAX_LXS_PER_KBC_BUCKET * sizeof(uint32_t) * kBC_NUM_BUCKETS)));

	std::cout << "      device_global_kbc_num_entries_L size:" << (sizeof(int)*kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&device_global_kbc_num_entries_L, sizeof(int)*kBC_NUM_BUCKETS));
	CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, kBC_NUM_BUCKETS*sizeof(int)));

	// alloc for rx's
	int *rx_match_count;
	std::cout << "      rx_match_list MAX_RX_MATCHES: " << MAX_RX_MATCHES << " TOTAL BYTES: " <<  (MAX_RX_MATCHES * sizeof(uint32_t)) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&rx_match_list, (MAX_RX_MATCHES * sizeof(uint32_t))));
	CUDA_CHECK_RETURN(cudaMallocManaged(&rx_match_count, RX_BATCHES*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMemset(rx_match_count, 0, RX_BATCHES*sizeof(int)));

	auto alloc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   alloc time: " << std::chrono::duration_cast<milli>(alloc_finish - attack_start).count() << " ms\n";

	auto compute_only_start = std::chrono::high_resolution_clock::now();

	int blockSize; // # of threads per block, maximum is 1024.
	uint64_t calc_N;
	uint64_t calc_blockSize;
	uint64_t calc_numBlocks;
	int numBlocks;

/*	std::cout << "   gpu_chacha8_set_Lxs_into_kbc_bitmask \n";
		int blockSize = 16; // # of threads per block, maximum is 1024.
		uint64_t calc_N = num_lxs;
		uint64_t calc_blockSize = blockSize;
		uint64_t calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		int numBlocks = calc_numBlocks;

		auto chacha_start = std::chrono::high_resolution_clock::now();
		gpu_chacha8_set_Lxs_into_kbc_bitmask<<<numBlocks,blockSize>>>(calc_N, chacha_input,
				device_global_kbc_num_entries_L);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto chacha_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   - gpu_chacha8_set_Lxs_into_kbc_bitmask results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";

	F1_Bucketed_kBC_Entry *local_kbc_entries = (F1_Bucketed_kBC_Entry *) rx_match_list;
		chacha_start = std::chrono::high_resolution_clock::now();
		// 1) gpu scan kbs into (F1_Bucketed_kBC_Entry *) bufferA
		//std::cout << "   Generating F1 results into kbc buckets...";
		blockSize = 128; // # of threads per block, maximum is 1024.
		calc_N = UINT_MAX;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;
		//std::cout << "  Block configuration: [blockSize:" << blockSize << "  numBlocks:" << numBlocks << "]" << std::endl;
		// don't forget to clear counter...will only use a portion of this memory so should be fast access.

		CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, 10000000*sizeof(int)));
		gpu_chacha8_get_k32_keystream_into_local_kbc_entries<<<numBlocks, blockSize>>>(calc_N, chacha_input,
				local_kbc_entries, device_global_kbc_num_entries_L, 0, 2000000);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		chacha_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   - gpu_chacha8_get_k32_keystream_into_local_kbc_entries results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";


	std::cout << "   gpu_chacha8_filter_rxs_by_kbc_bitmask \n";
		blockSize = 256; // # of threads per block, maximum is 1024.
		calc_N = UINT_MAX;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;

		chacha_start = std::chrono::high_resolution_clock::now();
		gpu_chacha8_filter_rxs_by_kbc_bitmask<<<numBlocks,blockSize>>>(calc_N, chacha_input,
						device_global_kbc_num_entries_L,
						rx_match_list, rx_match_count,
						RX_BATCHES, RX_MAX_ENTRIES_PER_BATCH);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		chacha_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   gpu_chacha8_filter_rxs_by_kbc_bitmask results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
		std::cout << "   found " << rx_match_count[0] << " RXS" << std::endl;

*/

	// FIRST SET LXS into global memory, these stay put for each chacha round
	blockSize = 128; // # of threads per block, maximum is 1024.
	calc_N = num_lxs;
	calc_blockSize = blockSize;
	calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
	numBlocks = calc_numBlocks;

	std::cout << "   gpu_chacha8_set_Lxs_into_kbc_ys num:" << calc_N << std::endl;
	auto lxintokbc_start = std::chrono::high_resolution_clock::now();
	gpu_chacha8_set_Lxs_into_kbc_ys_mask<<<numBlocks,blockSize>>>(calc_N, chacha_input,
			kbc_Ly_entries, kbc_x_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET);

	/* Doing chacha batch 7
	   gpu_chacha8_k32_write_chachas32_buckets results: 32 ms
	   chacha Rxs time: 37 ms
	   found 90582467 matches
	Freeing memory...
	   total chachas time: 248 ms
	   total Rxs time: 302 ms
	   compute only time: 654 ms  attack total time: 692 ms */
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto lxintokbc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   gpu_chacha8_set_Lxs_into_kbc_ys time: " << std::chrono::duration_cast<milli>(lxintokbc_finish - lxintokbc_start).count() << " ms\n";
	gpu_get_max_counts_from_counter_list<<<1,1024>>>(device_global_kbc_num_entries_L, kBC_NUM_BUCKETS, false);

	int64_t total_chacha_ms = 0;
	int64_t total_rx_ms = 0;
	for (uint64_t chacha_batch_id = 0; chacha_batch_id < CHACHA_NUM_BATCHES; chacha_batch_id++) {
		std::cout << "Doing chacha batch " << chacha_batch_id << std::endl;
		uint64_t BATCH_CHACHA_DIVISOR = (1 << (32 - CHACHA_NUM_BATCHES_BITS));
		uint64_t BATCH_CHACHA_RANGE_MIN = ((uint64_t) (chacha_batch_id + 0)) * BATCH_CHACHA_DIVISOR;
		uint64_t BATCH_CHACHA_RANGE_MAX = ((uint64_t) (chacha_batch_id + 1)) * BATCH_CHACHA_DIVISOR - 1; // use -1 since rnage is inclusive, also helps stay in 32-bit range rather than wrap to 0 for last batch
		//if (chacha_batch_id == CHACHA_NUM_BATCHES - 1) BATCH_CHACHA_RANGE_MAX = UINT_MAX;

		//std::cout << "   BATCH_CHACHA_DIVISOR : " << BATCH_CHACHA_DIVISOR << std::endl;
		//std::cout << "   BATCH_CHACHA_RANGE   : " << BATCH_CHACHA_RANGE_MIN << " <-> " << BATCH_CHACHA_RANGE_MAX << std::endl;
		//std::cout << "   BATCH_CHACHA_TOTAL_ENTRIES : " << CHACHA_TOTAL_ENTRIES_PER_BATCH << std::endl;
		//std::cout << "   CHACHA_MAX_ENTRIES_PER_BUCKET : " << CHACHA_MAX_ENTRIES_PER_BUCKET << std::endl;
		//std::cout << "   CHACHA_SPLIT_BUCKET_DIVISOR : " << CHACHA_SPLIT_BUCKET_DIVISOR << std::endl;


		blockSize = 128; // # of threads per block, maximum is 1024.
		calc_N = UINT_MAX;//CHACHA_TOTAL_ENTRIES_PER_BATCH;
		uint32_t CHACHA_X_START = 0;//chacha_batch_id * calc_N;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 32);
		numBlocks = calc_numBlocks;
		CUDA_CHECK_RETURN(cudaMemset(xchachas_bucket_counts, 0, CHACHA_NUM_BUCKETS*sizeof(int)));
		auto chacha_start = std::chrono::high_resolution_clock::now();
		//std::cout << "   calc_N   : " << calc_N << " numBlocks: " << numBlocks << " blockSize: " << blockSize << std::endl;
		gpu_chacha8_k32_compute_chachas32_filter_buckets_bychachabatchrange<CHACHA_NUM_BUCKETS><<<numBlocks,blockSize>>>(calc_N,
				BATCH_CHACHA_RANGE_MIN, BATCH_CHACHA_RANGE_MAX,
				CHACHA_MAX_ENTRIES_PER_BUCKET, CHACHA_SPLIT_BUCKET_DIVISOR,
				chacha_input,
				xchachas, xchachas_bucket_counts);


		//gpu_chacha8_only_chacha_results<<<numBlocks,blockSize>>>(calc_N, chacha_input,
		//				chachas);
		//gpu_chacha8_k32_write_chachas32_buckets<CHACHA_NUM_BUCKETS><<<numBlocks,blockSize>>>(calc_N, CHACHA_X_START,
		//		CHACHA_MAX_ENTRIES_PER_BUCKET,
		//		chacha_input,
		//		xchachas, xchachas_bucket_counts);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto chacha_finish = std::chrono::high_resolution_clock::now();
		total_chacha_ms += std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count();
		std::cout << "   gpu_chacha8_k32_write_chachas32_buckets results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
		//gpu_get_max_counts_from_counter_list<<<1,1>>>(xchachas_bucket_counts, CHACHA_NUM_BUCKETS, true);
		auto chacha_rs_start = std::chrono::high_resolution_clock::now();
		for (uint chacha_bucket_id=0;chacha_bucket_id<CHACHA_NUM_BUCKETS;chacha_bucket_id++) {
			blockSize = 256; // # of threads per block, maximum is 1024.
			calc_N = xchachas_bucket_counts[chacha_bucket_id];
			calc_blockSize = blockSize;
			calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
			numBlocks = calc_numBlocks;
			//std::cout << "Doing chacha bucket " << chacha_bucket_id << " (calc_N: " << calc_N << ")" << std::endl;
			gpu_chacha8_filter_rxs_from_bucket_batch<<<numBlocks,blockSize>>>(
					calc_N,
					&xchachas[chacha_bucket_id],
					kbc_Ly_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET,
					rx_match_list, rx_match_count);
			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		/*
		blockSize = 128; // # of threads per block, maximum is 1024.
		calc_N = UINT_MAX/CHACHA_NUM_BATCHES;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;


		std::cout << "Doing " << NUM_KBC_RANGE_BATCHES << " range batches of gpu_chacha_filter_rxs" << std::endl;
		for (int kbc_range_batch=0;kbc_range_batch < NUM_KBC_RANGE_BATCHES; kbc_range_batch++) {
			const uint32_t KBC_MIN_RANGE = ((kbc_range_batch+0) * 18188177) / (NUM_KBC_RANGE_BATCHES);
			const uint32_t KBC_MAX_RANGE = ((kbc_range_batch+1) * 18188177) / (NUM_KBC_RANGE_BATCHES);
			std::cout << "range KBC_MIN: " << KBC_MIN_RANGE << " - " << KBC_MAX_RANGE << std::endl;
			gpu_chacha8_filter_rxs<<<numBlocks,blockSize>>>(calc_N, chacha_input,
					kbc_Ly_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET,
					rx_match_list, rx_match_count,
					KBC_MIN_RANGE, KBC_MAX_RANGE);
		}
*/

			//calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize); numBlocks = calc_numBlocks;
			//gpu_chacha8_tag_rxs_from_chacha<<<numBlocks,blockSize>>>(calc_N, chacha_input,
			//		kbc_Ly_entries, device_global_kbc_num_entries_L, MAX_LXS_PER_KBC_BUCKET,
			//		chachas);
			//gpu_chacha8_filter_rxs_from_chacha<<<numBlocks,blockSize>>>(calc_N,chachas,rx_match_list,rx_match_count);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto chacha_rs_finish = std::chrono::high_resolution_clock::now();
		total_rx_ms += std::chrono::duration_cast<milli>(chacha_rs_finish - chacha_rs_start).count();
		std::cout << "   chacha Rxs time: " << std::chrono::duration_cast<milli>(chacha_rs_finish - chacha_rs_start).count() << " ms\n";
		std::cout << "   found " << rx_match_count[0] << " matches" << std::endl;


	}






	auto compute_only_finish = std::chrono::high_resolution_clock::now();

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(kbc_Ly_entries));
	CUDA_CHECK_RETURN(cudaFree(device_global_kbc_num_entries_L));

	auto attack_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   total chachas time: " << total_chacha_ms << " ms\n";
	std::cout << "   total Rxs time: " << total_rx_ms << " ms\n";
	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "   attack total time: " << std::chrono::duration_cast<milli>(attack_finish - attack_start).count() << " ms\n";
	std::cout << "end." << std::endl;
}





#endif /* ATTACK_METHOD_LXS_HPP_ */
