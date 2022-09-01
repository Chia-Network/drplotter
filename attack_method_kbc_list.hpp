/*
 * attack_method_kbc_list.hpp
 *
 *  Created on: Nov 7, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_KBC_LIST_HPP_
#define ATTACK_METHOD_KBC_LIST_HPP_

#define ATTACK_FILTER_BITMASK(chacha_y,i) \
{ \
	uint64_t Ry = (((uint64_t) chacha_y) << 6) + (x >> 26); \
	int kbc_bucket_id_L = (uint32_t (Ry / kBC)) - 1; \
	if (kbc_bucket_id_L > 0) { \
		int kbc_bitmask_bucket = kbc_bucket_id_L / 32; \
		unsigned int kbc_bit_slot = kbc_bucket_id_L % 32; \
		unsigned int kbc_mask = 1 << kbc_bit_slot; \
		unsigned int kbc_value = kbc_global_bitmask[kbc_bitmask_bucket]; \
		if ((kbc_mask & kbc_value) > 0) { \
			int slot = atomicAdd(&count[0],1); \
			xs[slot] = (x+i); \
			chachas[slot] = chacha_y; \
		} \
	} \
}

__global__
void gpu_chacha8_filter_rxs_by_kbc_bitmask(const uint32_t N,
		const __restrict__ uint32_t *input,
		const unsigned int* __restrict__ kbc_global_bitmask,
		uint32_t * __restrict__ xs, uint32_t * __restrict__ chachas, int *count)
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
		ATTACK_FILTER_BITMASK(x0,0);ATTACK_FILTER_BITMASK(x1,1);ATTACK_FILTER_BITMASK(x2,2);ATTACK_FILTER_BITMASK(x3,3);
		ATTACK_FILTER_BITMASK(x4,4);ATTACK_FILTER_BITMASK(x5,5);ATTACK_FILTER_BITMASK(x6,6);ATTACK_FILTER_BITMASK(x7,7);
		ATTACK_FILTER_BITMASK(x8,8);ATTACK_FILTER_BITMASK(x9,9);ATTACK_FILTER_BITMASK(x10,10);ATTACK_FILTER_BITMASK(x11,11);
		ATTACK_FILTER_BITMASK(x12,12);ATTACK_FILTER_BITMASK(x13,13);ATTACK_FILTER_BITMASK(x14,14);ATTACK_FILTER_BITMASK(x15,15);
	}
}

__global__
void gpu_set_kbc_bitmask_from_kbc_list(const uint32_t N,
		uint32_t *kbc_list, unsigned int* kbc_bitmask)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		uint32_t kbc_bucket_id = kbc_list[i];
		int kbc_bitmask_bucket = kbc_bucket_id / 32;
		int kbc_bit_slot = kbc_bucket_id % 32;
		unsigned int kbc_mask = 1 << kbc_bit_slot;
		atomicOr(&kbc_bitmask[kbc_bitmask_bucket],kbc_mask);
		//printf("kbc slot %u value %u SET mask bucket: %u  bitslot:%u\n",i, kbc_bucket_id, kbc_bitmask_bucket, kbc_bit_slot);
		// don't forget buckets needed for rx's.
		kbc_bitmask_bucket = (kbc_bucket_id+1) / 32;
		kbc_bit_slot = (kbc_bucket_id+1) % 32;
		kbc_mask = 1 << kbc_bit_slot;
		atomicOr(&kbc_bitmask[kbc_bitmask_bucket],kbc_mask);
		//printf("kbc %u SET mask bucket: %u  bitslot:%u\n",kbc_bucket_id+1, kbc_bitmask_bucket, kbc_bit_slot);
	}
}

__global__
void gpu_count_kbc_mask_bits(unsigned int* kbc_bitmask)
{
	int count = 0;
	for (int kbc_bucket_id_L=0;kbc_bucket_id_L<kBC_NUM_BUCKETS;kbc_bucket_id_L++) {
		int kbc_bitmask_bucket = kbc_bucket_id_L / 32;
		int kbc_bit_slot = kbc_bucket_id_L % 32;
		unsigned int kbc_mask = 1 << kbc_bit_slot;
		unsigned int kbc_value = kbc_bitmask[kbc_bitmask_bucket];
		if ((kbc_mask & kbc_value) > 0) {
			count++;
		}
	}
	printf("Counted kbc masks: %u\n",count);
}

#include <bits/stdc++.h>

void attack_method_kbc_list(uint32_t bits) {

	const uint32_t NUM_L_KBCS = 208147; // T4 16-bit entry list size
	std::cout << "ATTACK METHOD KBC LIST NUM: " << NUM_L_KBCS << std::endl;

	/* Tried, really tried, but the bitmask slows it down too much, all those x's checking 4 billion times against
	 * ram and then doing a simple xs/ys add, even so it's 109ms just to filter the xs compared to kbc bit scan method
	 * that's done with that phase and sorted into buckets at 40ms tops.
	 * DrPlotter v0.1d
Attack it!
ATTACK METHOD KBC LIST NUM: 208147
      kbc list bytes size:832588
      kbc_bitmask:832588
      expected xs:106571264 size: 426285056
               chachas:106571264 size: 426285056
Generating kbc list (step:87)
 num uniques:208146    duplicates: 0
setting kbc mask
   gpu_chacha8_set_Lxs_into_kbc_bitmask results: 1 ms
Counted kbc masks: 411613
getting filtered xs/chachas list
   gpu_chacha8_filter_rxs_by_kbc_bitmask time: 109 ms
   xs count: 97190536
Freeing memory...
   compute only time: 287 ms
end.
	 *
	 */

	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();

	// first we "read" the kbc list on host

	const uint32_t EXPECTED_XS = NUM_L_KBCS*2*256;
	uint32_t *kbc_list;
	unsigned int *kbc_bitmask;
	int *xs_count;
	uint32_t *xs;
	uint32_t *chachas;

	std::cout << "      kbc list bytes size:" << (sizeof(uint32_t)*NUM_L_KBCS) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&kbc_list, sizeof(uint32_t)*NUM_L_KBCS));
	std::cout << "      kbc_bitmask:" << (sizeof(unsigned int)*NUM_L_KBCS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&kbc_bitmask, kBC_NUM_BUCKETS*sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemset(kbc_bitmask, 0, kBC_NUM_BUCKETS*sizeof(unsigned int)));
	std::cout << "      expected xs:" << EXPECTED_XS << " size: " << (sizeof(uint32_t)*EXPECTED_XS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&xs, EXPECTED_XS*sizeof(uint32_t)));
	std::cout << "               chachas:" << EXPECTED_XS << " size: " << (sizeof(uint32_t)*EXPECTED_XS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&chachas, EXPECTED_XS*sizeof(uint32_t)));
	CUDA_CHECK_RETURN(cudaMallocManaged(&xs_count, 1024)); // 1024 blocks maybe?

	auto compute_only_start = std::chrono::high_resolution_clock::now();

	int step = kBC_NUM_BUCKETS / NUM_L_KBCS;
	std::cout << "Generating kbc list (step:" << step << ")" << std::endl;
	for (int i=0;i<NUM_L_KBCS;i++) {
		int value = rand() % kBC_NUM_BUCKETS;//i*step;
		//std::cout << " setting kbc " << value << std::endl;
		kbc_list[i] = value; // just set distribution but consistent for testing.
	}
	//std::sort(kbc_list, kbc_list + NUM_L_KBCS);
	int duplicates = 0;
	int uniques = 0;
	for (int i=1;i<NUM_L_KBCS;i++) {
		if (kbc_list[i] == kbc_list[i-1]) duplicates++;
		else uniques++;
	}
	std::cout << " num uniques:" << uniques << "    duplicates: " << duplicates << std::endl;

	std::cout << "setting kbc mask" << std::endl;
		int blockSize = 256; // # of threads per block, maximum is 1024.
		uint64_t calc_N = NUM_L_KBCS;
		uint64_t calc_blockSize = blockSize;
		uint64_t calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize);
		int numBlocks = calc_numBlocks;

		auto time_start = std::chrono::high_resolution_clock::now();
		gpu_set_kbc_bitmask_from_kbc_list<<<numBlocks,blockSize>>>(calc_N, kbc_list, kbc_bitmask);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		auto time_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   gpu_chacha8_set_Lxs_into_kbc_bitmask results: " << std::chrono::duration_cast<milli>(time_finish - time_start).count() << " ms\n";

	gpu_count_kbc_mask_bits<<<1,1>>>(kbc_bitmask);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::cout << "getting filtered xs/chachas list" << std::endl;
		blockSize = 256; // # of threads per block, maximum is 1024.
		calc_N = UINT_MAX;
		calc_blockSize = blockSize;
		calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
		numBlocks = calc_numBlocks;
		xs_count[0] = 0;
		time_start = std::chrono::high_resolution_clock::now();
		gpu_chacha8_filter_rxs_by_kbc_bitmask<<<numBlocks,blockSize>>>(calc_N,chacha_input,
			kbc_bitmask, xs, chachas, &xs_count[0]);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		time_finish = std::chrono::high_resolution_clock::now();
		std::cout << "   gpu_chacha8_filter_rxs_by_kbc_bitmask time: " << std::chrono::duration_cast<milli>(time_finish - time_start).count() << " ms\n";
		std::cout << "   xs count: " << xs_count[0] << "\n";


	auto compute_only_finish = std::chrono::high_resolution_clock::now();

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(kbc_bitmask));
	CUDA_CHECK_RETURN(cudaFree(xs));
	CUDA_CHECK_RETURN(cudaFree(chachas));

	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "end." << std::endl;

}

#endif /* ATTACK_METHOD_KBC_LIST_HPP_ */
