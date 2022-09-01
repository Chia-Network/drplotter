/*
 * attack_method_lxs2.hpp
 *
 *  Created on: Nov 8, 2021
 *      Author: nick
 */

#ifndef ATTACK_METHOD_LXS2_HPP_
#define ATTACK_METHOD_LXS2_HPP_

//#include <thrust/device_ptr.h>
//#include <thrust/sort.h>
//#include <thrust/unique.h>



struct xchacha_pair {
	uint32_t x;
	uint32_t chacha;
};

// TODO: try increasing the buckets as we go down the iterations
// suspect we can benefit more from cache when flipping back and forth vs the chacha generation
// which likely eats a lot of the cache? Or I had a huge bug somewhere.

const uint32_t DUMBSORT_BUCKET_BITS = 4;
const uint32_t DUMBSORT_NUM_BUCKETS = 1 << DUMBSORT_BUCKET_BITS;
const uint32_t PHASE_3_DUMBSORT_MAX_PER_BUCKET = 42;//32;
const uint32_t PHASE_2_DUMBSORT_MAX_PER_BUCKET = 42*16;//512;
const uint32_t PHASE_1_DUMBSORT_MAX_PER_BUCKET = 42*16*16;//8192; // 8601 was largest found, using a multiple of 256 so going for 8704
const uint32_t DUMBSORT_BATCHES_TILE_SPACE = PHASE_1_DUMBSORT_MAX_PER_BUCKET * DUMBSORT_NUM_BUCKETS;
const uint32_t GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK = 65536;
const uint32_t DUMBSORT_SPACE_NEEDED_FOR_SCRATCH = ((1 << (32-6)) / GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK) * DUMBSORT_BATCHES_TILE_SPACE;


////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#define SHARED_SIZE_LIMIT 1024U

__device__ inline void Comparator(
    uint &keyA,
    uint &valA,
    uint &keyB,
    uint &valB,
    uint dir
)
{
    uint t;

    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Monolithic Bacther's sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeSortShared(uint32_t *chachas, uint32_t *out_chachas, uint32_t *out_xs)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for one or more small vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    uint dir = 1;
        uint arrayLength = 1024;

        //Offset to the beginning of subbatch and load data
        chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
        out_chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
        out_xs += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
        s_key[threadIdx.x +                       0] = chachas[                      0];
        s_val[threadIdx.x +                       0] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = chachas[(SHARED_SIZE_LIMIT / 2)];
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2);

    for (uint size = 2; size <= arrayLength; size <<= 1)
    {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                Comparator(
                    s_key[pos - stride], s_val[pos - stride],
                    s_key[pos +      0], s_val[pos +      0],
                    dir
                );
        }
    }

    cg::sync(cta);
    out_chachas[                      0] = s_key[threadIdx.x +                       0];
        out_xs[                      0] = s_val[threadIdx.x +                       0];
        out_chachas[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
        out_xs[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

}

// threads must be SHARED_SIZE_LIMIT/2
__global__ void nickSortShared(uint32_t *chachas, uint32_t *out_chachas, uint32_t *out_xs)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for one or more short vectors
    __shared__ uint order[SHARED_SIZE_LIMIT*2]; // we're going to use top 16 and bottom 16 to store indexes
    __shared__ uint bucket_counts[1024];
    __shared__ uint s_key[SHARED_SIZE_LIMIT]; // the sort values
    __shared__ uint s_val[SHARED_SIZE_LIMIT]; // stores the xs
    __shared__ uint sorted_val[SHARED_SIZE_LIMIT];
    __shared__ uint sorted_key[SHARED_SIZE_LIMIT];

    uint dir = 1;
    uint arrayLength = 1024;

    //Offset to the beginning of subbatch and load data
    chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    out_chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    out_xs += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    uint32_t chacha = chachas[0];
    uint16_t index = threadIdx.x;
    bucket_counts[threadIdx.x]   = 0;
    order[threadIdx.x] = 0;
    order[threadIdx.x + 1024] = 0;
    //order[threadIdx.x*2+1] = 0;
    s_key[threadIdx.x +                       0] = chachas[                      0];
    s_val[threadIdx.x +                       0] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;

    cg::sync(cta);
    uint16_t bucket_id = chacha >> (32 - 10);
        int add = atomicAdd(&bucket_counts[bucket_id],1);
        if (add < 4) {
        	uint pos = bucket_id * 2 + add;
        	uint value = index << ((pos & 0b01)*16);
        	atomicAdd(&order[pos], value);
        }
        // from [ 1 3 2 0 0 1 0 2 ]
        //  to> [ 0 1 4 6 6 6 7 7 ]
        // then each thread, reads its scan offset, and that's the shared start + the counts to copy into global memory
        //      [ 1 3 2 0 0 1 0 2 ]
        //      [ 0 1 3 5 5 0 1 0 ]
        //      [ 0 1 4 6


      // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda


    if (threadIdx.x == 0) {
        printf("buckets counts:\n");
        for (int i=0;i<SHARED_SIZE_LIMIT;i++) {
           printf("bucket %u - counts: %u\n", i, bucket_counts[i]);
        }
        printf("buckets order:\n");
        for (int pos=0;pos<SHARED_SIZE_LIMIT*2;pos++) {
           printf("pos %u - index value: %u\n", pos, order[pos]);
        }
      }


    // should be sorted now.
    //out_chachas[                      0] = sorted_key[threadIdx.x +                       0];
    //out_xs[                      0] = sorted_val[threadIdx.x +                       0];

    //__syncthreads();
    //if (threadIdx.x == 0) {
    //	printf("results sort:\n");
    //    for (int i=0;i<SHARED_SIZE_LIMIT;i++) {
   //     	printf("i %u - x: %u   chacha: %u\n", i, sorted_val[i], sorted_key[i]);
   //     }
   // }

}

__global__ void prescan(float *g_odata, float *g_idata, int n) {
	extern __shared__ float temp[];  // allocated on invocation
	int thid = threadIdx.x; int offset = 1;

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
    	__syncthreads();
    	if (thid < d)    {
    		int ai = offset*(2*thid+1)-1;
    		int bi = offset*(2*thid+2)-1;
    		temp[bi] += temp[ai];
    	}
    	offset *= 2;
    }

     if (thid == 0) { temp[n - 1] = 0; } // clear the last element
     for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    	 {
    	 offset >>= 1;
    	 __syncthreads();
    	 if (thid < d)      {
    		 int ai = offset*(2*thid+1)-1;
    		 int bi = offset*(2*thid+2)-1;
    		 float t = temp[ai]; temp[ai] = temp[bi]; temp[bi] += t;
    	 }
    	 }
     __syncthreads();

    g_odata[2*thid] = temp[2*thid];
    // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
}

// threads must be SHARED_SIZE_LIMIT/2
__global__ void bitonicSortShared(uint32_t *chachas, uint32_t *out_chachas, uint32_t *out_xs)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT]; // the sort values
    __shared__ uint s_val[SHARED_SIZE_LIMIT]; // stores the xs


    uint dir = 1;
    uint arrayLength = 1024;

    //Offset to the beginning of subbatch and load data
    chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    out_chachas += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    out_xs += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = chachas[                      0];
    s_val[threadIdx.x +                       0] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = chachas[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2);

    //__syncthreads();
    //if (threadIdx.x == 0) {
    //	printf("doing bitonic sort, start list: \n");
    //	for (int i=0;i<SHARED_SIZE_LIMIT;i++) {
    //		printf("i %u - x: %u   chacha: %u\n", i, s_val[i], s_key[i]);
    //	}
   // }


    for (uint size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
        }
    }

    cg::sync(cta);

    // should be sorted now.
    out_chachas[                      0] = s_key[threadIdx.x +                       0];
    out_xs[                      0] = s_val[threadIdx.x +                       0];
    out_chachas[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    out_xs[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

    //__syncthreads();
    //if (threadIdx.x == 0) {
    //	printf("results sort:\n");
    //    for (int i=0;i<SHARED_SIZE_LIMIT;i++) {
    //    	printf("i %u - x: %u   chacha: %u\n", i, s_val[i], s_key[i]);
    //    }
    //}

}

__global__
void gpu_show_chacha_xs_lists(uint32_t START, uint32_t NUM, uint32_t *chachas, uint32_t *xs) {
	printf("gpu_show_chacha_xs_lists:\n");
	for (int i=START;i<NUM;i++) {
		printf("i %u   x: %u  chacha: %u\n", i, xs[i], chachas[i]);
	}
}

__global__
void gpu_write_chachas_into_buckets_dumb_batches(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		xchacha_pair *results, xchacha_pair *results2)
{
	// highest performance bucket bits 4, with 1024 threads, num per block 65536. Then all blocks work with L2 cache?
	const uint32_t NUM_BUCKETS = DUMBSORT_NUM_BUCKETS;
	const uint32_t BUCKET_DIVISOR = 1 << (32-DUMBSORT_BUCKET_BITS); // 32bit chacha into 8 bit NUM buckets
	const uint32_t NUM_THREADS = blockDim.x;
	uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;
	const uint32_t GLOBAL_TILE_START = x_group * DUMBSORT_BATCHES_TILE_SPACE;

	__shared__ int buffer_counts[NUM_BUCKETS];
	__shared__ int buffer_counts_phase2[NUM_BUCKETS*NUM_BUCKETS];
	__shared__ int buffer_counts_phase3[NUM_BUCKETS];

	if (x_start < N) {
		//if (threadIdx.x == 0) {
		//	printf("x start: %u global_bucket_start_pos: %u vs before %u\n", x_start, global_bucket_start_pos, x_start / blockDim.x);
		//}
		for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts[i] = 0;
		}
		for (int i=threadIdx.x;i<NUM_BUCKETS*NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts_phase2[i] = 0;
		}
		__syncthreads();

		uint32_t batch_id = 0;
		for (batch_id = 0; batch_id < NUM_BATCHES_OF_THREADS; batch_id++) {
			uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
			uint32_t chacha = chachas[x];
			xchacha_pair entry = { x, chacha };

			uint32_t bucket_id = chacha / BUCKET_DIVISOR;
			//printf("chacha %u - bucket id: %u\n", chacha, bucket_id);
			if (bucket_id >= NUM_BUCKETS) printf("BUCKET OUT OF RANGE ERROR: %u", bucket_id);

			int slot = atomicAdd(&buffer_counts[bucket_id],1);
			if (slot > PHASE_1_DUMBSORT_MAX_PER_BUCKET) printf("PHASE 1 DUMBSORT OVERFLOW: %u\n", slot);

			uint32_t results_address = GLOBAL_TILE_START + bucket_id * PHASE_1_DUMBSORT_MAX_PER_BUCKET + slot;
			if (results_address < DUMBSORT_SPACE_NEEDED_FOR_SCRATCH) {
				results[results_address] = entry;
			} else {
				printf("results address overflow %u - global start pos: %u bucket %u slot %u DUMBSORT_SPACE_NEEDED_FOR_SCRATCH: %u\n",
						results_address, GLOBAL_TILE_START, bucket_id, slot, DUMBSORT_SPACE_NEEDED_FOR_SCRATCH);
			}
		}

		__syncthreads();
		//if (threadIdx.x == 0) {
		//	printf("end phase 1, buffer counts:\n");
		//	for (int i=0;i<NUM_BUCKETS;i++) {
		//		printf("  bucket %u : %u\n", i, buffer_counts[i]);
		//	}
		//}

		// phase 2...now read from buckets and sort into small buckets...hohoho.
		for (int read_bucket_id=0;read_bucket_id < NUM_BUCKETS; read_bucket_id++) {
			const uint32_t NUM_ENTRIES = buffer_counts[read_bucket_id];
			const uint32_t BUCKET_DIVISOR2 = 1 << (32 - DUMBSORT_BUCKET_BITS*2);
			const uint32_t SUB_2_TILE_START = GLOBAL_TILE_START + read_bucket_id * PHASE_1_DUMBSORT_MAX_PER_BUCKET;
			const uint32_t SUB_2_BUFFER_COUNT_OFFSET = read_bucket_id*NUM_BUCKETS;

			//for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			//	buffer_counts_phase2[i] = 0;
			//}
			//__syncthreads();

			for (batch_id = 0; batch_id < (NUM_ENTRIES + NUM_THREADS - 1) / NUM_THREADS; batch_id++) {

				uint32_t pos = batch_id * NUM_THREADS + threadIdx.x;
				if (pos < NUM_ENTRIES) {

					uint32_t read_address = SUB_2_TILE_START + pos;
					xchacha_pair entry = results[read_address];

					uint32_t local_bucket_id = (entry.chacha / BUCKET_DIVISOR2) % NUM_BUCKETS;
					//printf("chacha %u - bucket id: %u\n", entry.chacha, local_bucket_id);

					int slot = atomicAdd(&buffer_counts_phase2[SUB_2_BUFFER_COUNT_OFFSET + local_bucket_id],1);
					if (slot > PHASE_2_DUMBSORT_MAX_PER_BUCKET) printf("PHASE 2 DUMBSORT OVERFLOW: %u\n", slot);

					uint32_t results_address2 = SUB_2_TILE_START + local_bucket_id * PHASE_2_DUMBSORT_MAX_PER_BUCKET + slot;
					results2[results_address2] = entry;
				}
			}

			__syncthreads();
			//if (threadIdx.x == 0) {
			//	printf("end phase 2-%u, buffer counts:\n",read_bucket_id);
			//	for (int i=0;i<NUM_BUCKETS;i++) {
			//		printf("  bucket %u : %u\n", i, buffer_counts_phase2[SUB_2_BUFFER_COUNT_OFFSET + i]);
			//	}
			//}
		}

		// phase 3...now read from buckets and sort into small buckets...hohoho.
		for (int read_bucket_id = 0; read_bucket_id < NUM_BUCKETS; read_bucket_id++) {
			const uint32_t SUB_2_TILE_START = GLOBAL_TILE_START + read_bucket_id * PHASE_1_DUMBSORT_MAX_PER_BUCKET;
			const uint32_t SUB_2_BUFFER_COUNT_OFFSET = read_bucket_id*NUM_BUCKETS;
			for (int read_bucket_id_phase2=0;read_bucket_id_phase2 < NUM_BUCKETS; read_bucket_id_phase2++) {
				const uint32_t NUM_PHASE2_ENTRIES = buffer_counts_phase2[SUB_2_BUFFER_COUNT_OFFSET + read_bucket_id_phase2];
				const uint32_t BUCKET_DIVISOR3 = 1 << (32 - DUMBSORT_BUCKET_BITS*3);
				const uint32_t SUB_3_TILE_START = SUB_2_TILE_START + read_bucket_id_phase2 * PHASE_2_DUMBSORT_MAX_PER_BUCKET;


				for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
					buffer_counts_phase3[i] = 0;
				}
				__syncthreads();

				for (batch_id = 0; batch_id < (NUM_PHASE2_ENTRIES + NUM_THREADS - 1) / NUM_THREADS; batch_id++) {

					uint32_t pos2 = batch_id * NUM_THREADS + threadIdx.x;
					if (pos2 < NUM_PHASE2_ENTRIES) {

						uint32_t read_address_2 = SUB_3_TILE_START + pos2;
						xchacha_pair entry = results2[read_address_2];

						uint32_t local_bucket_id_3 = (entry.chacha / BUCKET_DIVISOR3) % NUM_BUCKETS;
						//printf("chacha %u - bucket id: %u\n", entry.chacha, local_bucket_id_3);

						int slot = atomicAdd(&buffer_counts_phase3[local_bucket_id_3],1);
						if (slot > PHASE_3_DUMBSORT_MAX_PER_BUCKET) printf("PHASE 3 DUMBSORT OVERFLOW: %u\n", slot);

						uint32_t results_address3 = SUB_3_TILE_START + local_bucket_id_3 * PHASE_3_DUMBSORT_MAX_PER_BUCKET + slot;
						results[results_address3] = entry;
					}
				}

				__syncthreads();

				//if (threadIdx.x == 0) {
				//	printf("end phase 3-2:[%u]-1[%u], buffer counts:\n",read_bucket_id_phase2,read_bucket_id);
				//	for (int i=0;i<NUM_BUCKETS;i++) {
				//		printf("  bucket %u : %u\n", i, buffer_counts_phase3[i]);
				//	}
				//}
			}
		}
	}
}

__global__
void gpu_write_chachas_into_buckets_dumb_batches_orig1phaseonly(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		uint32_t const MAX_TOTAL_GROUPED_ENTRIES, xchacha_pair *results, unsigned int *results_counts)
{
	// highest performance bucket bits 4, with 1024 threads, num per block 65536. Then all blocks work with L2 cache?
	const uint32_t NUM_BUCKET_BITS = 4; // 4 = 16 buckets
	const uint32_t NUM_BUCKETS = 1 << NUM_BUCKET_BITS;
	const uint32_t BUCKET_DIVISOR = 1 << (32-NUM_BUCKET_BITS); // 32bit chacha into 8 bit NUM buckets
	const uint32_t NUM_THREADS = blockDim.x;
	const uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	const uint32_t GLOBAL_BUCKET_MAX_ENTRIES = MAX_TOTAL_GROUPED_ENTRIES / NUM_BUCKETS;
	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;
	uint32_t global_bucket_start_pos = x_group * PHASE_1_DUMBSORT_MAX_PER_BUCKET;

	__shared__ int buffer_counts[NUM_BUCKETS];

	if (x_start < N) {
		//if (threadIdx.x == 0) {
		//	printf("x start: %u global_bucket_start_pos: %u vs before %u\n", x_start, global_bucket_start_pos, x_start / blockDim.x);
		//}
		for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts[i] = 0;
		}
		__syncthreads();
		//https://stackoverflow.com/questions/42620649/sorting-algorithm-with-cuda-inside-or-outside-kernels
		//https://stackoverflow.com/questions/5510715/thrust-inside-user-written-kernels
		//https://stackoverflow.com/questions/22339936/sorting-many-small-arrays-in-cuda

		uint32_t batch_id = 0;
		// simplest algorith, works 167ms with 32 buckets, but only 50ms with 8 buckets
		// want to reduce this down to same with 32 buckets as 8, means we take 4 batches and sort it.
		for (batch_id = 0; batch_id < NUM_BATCHES_OF_THREADS; batch_id++) {
			uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
			uint32_t chacha = chachas[x];
			xchacha_pair entry = { x, chacha };

			uint32_t bucket_id = chacha / BUCKET_DIVISOR;
			if (bucket_id >= NUM_BUCKETS) printf("BUCKET OUT OF RANGE ERROR: %u", bucket_id);

			int slot = atomicAdd(&buffer_counts[bucket_id],1);
			uint32_t results_address = global_bucket_start_pos + bucket_id * GLOBAL_BUCKET_MAX_ENTRIES + slot;
			if (results_address < 134217728) {
				results[results_address] = entry;
			} else {
				printf("results address overflow %u - global start pos: %u bucket %u slot %u globalmaxentries: %u\n",
						results_address, global_bucket_start_pos, bucket_id, slot, GLOBAL_BUCKET_MAX_ENTRIES);
			}
			//__syncthreads(); // holy fuck a sync threads increases from 50ms to 85!!!!! That's why!
			//for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			//	atomicAdd(&results_counts[i], buffer_counts[i]);
			//	buffer_counts[i] = 0;
			//}
			//__syncthreads();
		}
		__syncthreads();
		for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			atomicAdd(&results_counts[i], buffer_counts[i]+1);
			if (buffer_counts[i] > PHASE_1_DUMBSORT_MAX_PER_BUCKET)
				printf("BUFFER OVERFLOW: %u was over max per bucket\n",buffer_counts[i], PHASE_1_DUMBSORT_MAX_PER_BUCKET);
		}
	}
}


__global__
void gpu_write_chachas_into_buckets_with_single_row_depthflush(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		uint32_t const MAX_TOTAL_GROUPED_ENTRIES, xchacha_pair *results, unsigned int *results_counts)
{
	// note num threads should be equal or higher than NUM_BUCKETS
	// 256 has max depth of 23, 512 has max depth of 11. need keep some space for other variables.

	// good settings: NUM_BUCKETS 512, BUCKET DEPTH 11, FLUSH DEPTH 6 (15ms)
	//                            256,              22,            12 (11ms)
	// the bigger the span betwen flush depth and bucket depth, the less likely hashes will overflow before the rest can fill up.
	const uint32_t BUCKET_BITS = 5;
	const uint32_t FLUSH_DEPTH = 128;
	const uint32_t BUCKET_DEPTH = FLUSH_DEPTH+32; // give some room for overflow - careful too much and it slows down!
	// I tried with a buffer overflow instead of padding...but...it performed slightly slower and that's without
	// moving the buckets back in. seems like loops on threads not being perfect multiples when writing is more
	// forgiving than though? Can try again.

	const uint32_t NUM_BUCKETS = 1 << BUCKET_BITS;
	const uint32_t BUCKET_DIVISOR = 1 << (32-BUCKET_BITS); // 32bit chacha into 8 bit NUM buckets
	const uint32_t GLOBAL_BUCKET_MAX_ENTRIES = MAX_TOTAL_GROUPED_ENTRIES / NUM_BUCKETS;

	__shared__ int buffer_counts[NUM_BUCKETS];
	__shared__ int global_counts[NUM_BUCKETS];
	__shared__ uint32_t chachas_buffer[NUM_BUCKETS*BUCKET_DEPTH];
	__shared__ uint16_t xs_buffer[NUM_BUCKETS*BUCKET_DEPTH]; // 4 entries per bucket
	__shared__ int num_ready;
	__shared__ int batch_id;
	__shared__ int bucket_to_flush;

	// 49152 bytes total shared memory = 384 chunks of 128 bytes. Means we can use 384 buckets to fill shared memory.
	// let's try first with 256 buckets.
	//__shared__ int flush;

	const uint32_t NUM_THREADS = blockDim.x;
	const uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	//if ((NUM_PER_BLOCK % NUM_THREADS) > 0) printf("CONFIG ERROR: NUM PER BLOCK MUST BE MULTIPLE OF NUM THREADS\n");

	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;

	if (x_start < N) {
		if (threadIdx.x == 0) {
			num_ready = 0;
			batch_id = 0;
		}
		// make sure all values start right!
		for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts[i] = 0;
			global_counts[i] = 0;
		}
		__syncthreads();

		// go through each batch of data
		while (batch_id < NUM_BATCHES_OF_THREADS) {
			while ((num_ready == 0) && (batch_id < NUM_BATCHES_OF_THREADS)) {
				// thread is of course threadIdx.x
				uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
				uint32_t chacha = chachas[x];

				//if (threadIdx.x == 0) {
				//	printf("BATCH_ID %u of %u - x starts: %u num_ready: %u\n",batch_id, NUM_BATCHES_OF_THREADS, x, num_ready);
				//}
				__syncthreads();

				uint32_t bucket_id = chacha / BUCKET_DIVISOR;
				uint32_t slot = atomicAdd(&buffer_counts[bucket_id], 1);
				uint32_t address = bucket_id * BUCKET_DEPTH + slot;

				//printf("      xchacha pair x:%u chacha:%u into bucket:%u slot:%u \n", x, chachas[x], bucket_id, slot);
				if (address > NUM_BUCKETS*BUCKET_DEPTH) {
					printf("ERROR ADDRESS %u  --  batch: %u  bucket_id: %u  slot: %u\n", address, batch_id, bucket_id, slot);
				} else {
					//xchacha_pair entry = { x, chacha };
					chachas_buffer[address] = chacha;
					xs_buffer[address] = x;
				}

				if (slot == (FLUSH_DEPTH-1)) {
					atomicAdd(&num_ready, 1);
					bucket_to_flush = bucket_id; // doesn't matter if this gets overwritten by another thread
					// point is we want to get first bucket and if there is more we fetch it from list.
					//printf("-> bucket %u slot is FLUSH ready, incremented num_ready counter to %u\n", bucket_id, num_ready);
				}

				__syncthreads();
				if (threadIdx.x == 0) {
					//for (int i=0;i<NUM_BUCKETS;i++) {
					//	printf("bucket %u entries: %u\n", i, buffer_counts[i]);
					//}
					//printf("NUM READY after batch %u is %u\n", batch_id, num_ready);
					batch_id++;
				}
				__syncthreads();
			}

			// all buffers should be full OR batch processing is over and we have some to flush
			__syncthreads();

			while (num_ready > 0) {
				// flush those ready
				const int num_to_flush = buffer_counts[bucket_to_flush];
				if (threadIdx.x == 0) {
					global_counts[bucket_to_flush] += num_to_flush;
					//global_counts[bucket_to_flush] = atomicAdd(&results_counts[bucket_to_flush],num_to_flush);
				//	printf("FLUSHING! %u buckets are ready, flushing bucket %u\n", num_ready, bucket_to_flush);
				}

				__syncthreads();

				for (int i=threadIdx.x;i<num_to_flush;i+=blockDim.x) {
					uint32_t buffer_address = bucket_to_flush * BUCKET_DEPTH + i;
					uint32_t chacha = chachas_buffer[buffer_address];
					uint32_t x = xs_buffer[buffer_address] + x_start;
					xchacha_pair entry = { x, chacha };

					const int global_pos = global_counts[bucket_to_flush] + i;
					uint32_t global_address = bucket_to_flush * GLOBAL_BUCKET_MAX_ENTRIES + global_pos;
					results[global_address] = entry;
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					num_ready--;
					//printf("num ready set to %u\n", num_ready);
					buffer_counts[bucket_to_flush] = 0;
				}

				__syncthreads();

				if (num_ready > 0) {
					// find next bucket to flush! doesn't matter if multiple threads overwrite,
					// just want one of them
					for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
						if (buffer_counts[i] >= FLUSH_DEPTH)
							bucket_to_flush = i;
					}
				}
			}

			__syncthreads();

		}
		if (batch_id == NUM_BATCHES_OF_THREADS) {
			// we finished entering all our data, now check left-over buckets.
			// TODO: check each bucket count and write out data to global.
			//if (threadIdx.x == 0) {
			//	printf("BATCHES COMPLETED: todo finish flushing rest of buffers\n");
			//}
			for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
				if (buffer_counts[i] > 0) atomicAdd(&results_counts[i], buffer_counts[i]);
			}

		}
	}
}

__global__
void gpu_write_chachas_into_buckets_with_single_row_depthflush_ORIG(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		uint32_t const MAX_TOTAL_GROUPED_ENTRIES, xchacha_pair *results, unsigned int *results_counts)
{
	// note num threads should be equal or higher than NUM_BUCKETS
	// 256 has max depth of 23, 512 has max depth of 11. need keep some space for other variables.

	// good settings: NUM_BUCKETS 512, BUCKET DEPTH 11, FLUSH DEPTH 6 (15ms)
	//                            256,              22,            12 (11ms)
	// the bigger the span betwen flush depth and bucket depth, the less likely hashes will overflow before the rest can fill up.
	const uint32_t BUCKET_BITS = 5;
	const uint32_t FLUSH_DEPTH = 128;
	const uint32_t BUCKET_DEPTH = FLUSH_DEPTH+32; // give some room for overflow

	const uint32_t NUM_BUCKETS = 1 << BUCKET_BITS;
	const uint32_t BUCKET_DIVISOR = 1 << (32-BUCKET_BITS); // 32bit chacha into 8 bit NUM buckets
	const uint32_t GLOBAL_BUCKET_MAX_ENTRIES = MAX_TOTAL_GROUPED_ENTRIES / NUM_BUCKETS;

	__shared__ int buffer_counts[NUM_BUCKETS];
	__shared__ int global_counts[NUM_BUCKETS];
	__shared__ xchacha_pair buffer[NUM_BUCKETS*BUCKET_DEPTH]; // 4 entries per bucket
	__shared__ int num_ready;
	__shared__ int batch_id;
	__shared__ int bucket_to_flush;

	// 49152 bytes total shared memory = 384 chunks of 128 bytes. Means we can use 384 buckets to fill shared memory.
	// let's try first with 256 buckets.
	//__shared__ int flush;

	const uint32_t NUM_THREADS = blockDim.x;
	const uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	//if ((NUM_PER_BLOCK % NUM_THREADS) > 0) printf("CONFIG ERROR: NUM PER BLOCK MUST BE MULTIPLE OF NUM THREADS\n");

	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;

	if (x_start < N) {
		if (threadIdx.x == 0) {
			num_ready = 0;
			batch_id = 0;
		}
		// make sure all values start right!
		for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts[i] = 0;
			global_counts[i] = 0;
		}
		__syncthreads();

		// go through each batch of data
		while (batch_id < NUM_BATCHES_OF_THREADS) {
			while ((num_ready == 0) && (batch_id < NUM_BATCHES_OF_THREADS)) {
				// thread is of course threadIdx.x
				uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
				uint32_t chacha = chachas[x];

				//if (threadIdx.x == 0) {
				//	printf("BATCH_ID %u of %u - x starts: %u num_ready: %u\n",batch_id, NUM_BATCHES_OF_THREADS, x, num_ready);
				//}
				__syncthreads();

				uint32_t bucket_id = chacha / BUCKET_DIVISOR;
				uint32_t slot = atomicAdd(&buffer_counts[bucket_id], 1);
				uint32_t address = bucket_id * BUCKET_DEPTH + slot;

				//printf("      xchacha pair x:%u chacha:%u into bucket:%u slot:%u \n", x, chachas[x], bucket_id, slot);
				if (address > NUM_BUCKETS*BUCKET_DEPTH) {
					printf("ERROR ADDRESS %u  --  batch: %u  bucket_id: %u  slot: %u\n", address, batch_id, bucket_id, slot);
				} else {
					xchacha_pair entry = { x, chacha };
					buffer[address] = entry;
				}

				if (slot == (FLUSH_DEPTH-1)) {
					atomicAdd(&num_ready, 1);
					bucket_to_flush = bucket_id; // doesn't matter if this gets overwritten by another thread
					// point is we want to get first bucket and if there is more we fetch it from list.
					//printf("-> bucket %u slot is FLUSH ready, incremented num_ready counter to %u\n", bucket_id, num_ready);
				}

				__syncthreads();
				if (threadIdx.x == 0) {
					//for (int i=0;i<NUM_BUCKETS;i++) {
					//	printf("bucket %u entries: %u\n", i, buffer_counts[i]);
					//}
					//printf("NUM READY after batch %u is %u\n", batch_id, num_ready);
					batch_id++;
				}
				__syncthreads();
			}

			// all buffers should be full OR batch processing is over and we have some to flush
			__syncthreads();

			while (num_ready > 0) {
				// flush those ready
				const int num_to_flush = buffer_counts[bucket_to_flush];
				if (threadIdx.x == 0) {
					global_counts[bucket_to_flush] += num_to_flush;
					//global_counts[bucket_to_flush] = atomicAdd(&results_counts[bucket_to_flush],num_to_flush);
				//	printf("FLUSHING! %u buckets are ready, flushing bucket %u\n", num_ready, bucket_to_flush);
				}

				__syncthreads();

				for (int i=threadIdx.x;i<num_to_flush;i+=blockDim.x) {
					uint32_t buffer_address = bucket_to_flush * BUCKET_DEPTH + i;
					xchacha_pair entry = buffer[buffer_address];

					const int global_pos = global_counts[bucket_to_flush] + i;
					uint32_t global_address = bucket_to_flush * GLOBAL_BUCKET_MAX_ENTRIES + global_pos;
					results[global_address] = entry;
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					num_ready--;
					//printf("num ready set to %u\n", num_ready);
					buffer_counts[bucket_to_flush] = 0;
				}

				__syncthreads();

				if (num_ready > 0) {
					// find next bucket to flush! doesn't matter if multiple threads overwrite,
					// just want one of them
					for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
						if (buffer_counts[i] >= FLUSH_DEPTH)
							bucket_to_flush = i;
					}
				}
			}

			__syncthreads();

		}
		if (batch_id == NUM_BATCHES_OF_THREADS) {
			// we finished entering all our data, now check left-over buckets.
			// TODO: check each bucket count and write out data to global.
			//if (threadIdx.x == 0) {
			//	printf("BATCHES COMPLETED: todo finish flushing rest of buffers\n");
			//}
			for (int i=threadIdx.x;i<NUM_BUCKETS;i+=blockDim.x) {
				if (buffer_counts[i] > 0) atomicAdd(&results_counts[i], buffer_counts[i]);
			}

		}
	}
}



__global__
void gpu_write_chachas_into_buckets_with_buffer_batches(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		uint32_t const MAX_PER_RESULTS_BUCKET, xchacha_pair *results, unsigned int *results_counts)
{
	// note num threads should be equal or higher than NUM_BUCKETS
	// 256 has max depth of 23, 512 has max depth of 11. need keep some space for other variables.

	// good settings: NUM_BUCKETS 512, BUCKET DEPTH 11, FLUSH DEPTH 6 (15ms)
	//                            256,              22,            12 (11ms)
	// the bigger the span betwen flush depth and bucket depth, the less likely hashes will overflow before the rest can fill up.
	const uint32_t NUM_BUCKETS = 32;
	const uint32_t BUCKET_DIVISOR = 1 << (32-5); // 32bit chacha into 8 bit NUM buckets
	const uint32_t BUCKET_DEPTH = 128; // *should* be able to set this freely, as the active window should modulo effectively.
	const uint32_t FLUSH_DEPTH = 32; // cache does best with a flush depth of 8, but even 6 is ok, 4 is 1st benefit jump.

	__shared__ int buffer_counts[NUM_BUCKETS];
	__shared__ int global_counts[NUM_BUCKETS];
	__shared__ xchacha_pair buffer[NUM_BUCKETS*BUCKET_DEPTH]; // 4 entries per bucket
	__shared__ int num_ready;
	__shared__ int active_buffer_pos; // this is the moving position/window in the buffer
	__shared__ int eviction_needed;
	__shared__ uint32_t batch_id;

	// 49152 bytes total shared memory = 384 chunks of 128 bytes. Means we can use 384 buckets to fill shared memory.
	// let's try first with 256 buckets.
	//__shared__ int flush;

	const uint32_t NUM_THREADS = blockDim.x;
	const uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	//if ((NUM_PER_BLOCK % NUM_THREADS) > 0) printf("CONFIG ERROR: NUM PER BLOCK MUST BE MULTIPLE OF NUM THREADS\n");

	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;

	if (x_start < N) {
		if (threadIdx.x == 0) {
			num_ready = 0;
			active_buffer_pos = 0;
			eviction_needed = 0;
			batch_id = 0;
		}
		// make sure all values start right!
		for (int i=threadIdx.x;i < NUM_BUCKETS;i+=blockDim.x) {
			buffer_counts[i] = 0;
			global_counts[i] = 0;
		}
		__syncthreads();

		// go through each batch of data
		while (batch_id < NUM_BATCHES_OF_THREADS) {
			while ((num_ready < NUM_BUCKETS) && (batch_id < NUM_BATCHES_OF_THREADS) && (eviction_needed == 0)) {
				// thread is of course threadIdx.x
				uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
				uint32_t chacha = chachas[x];

				//if (threadIdx.x == 0) {
				//	printf("BATCH_ID %u of %u - x starts: %u num_ready: %u\n",batch_id, NUM_BATCHES_OF_THREADS, x, num_ready);
				//}
				//__syncthreads();

				uint32_t bucket_id = chacha / BUCKET_DIVISOR;
				uint32_t slot = atomicAdd(&buffer_counts[bucket_id], 1);
				uint32_t address = bucket_id * BUCKET_DEPTH + (slot + active_buffer_pos) % BUCKET_DEPTH;

				//printf("      xchacha pair x:%u chacha:%u into bucket:%u slot:%u \n", x, chachas[x], bucket_id, slot);

				if (address > NUM_BUCKETS*BUCKET_DEPTH) {
					printf("ERROR ADDRESS %u  --  batch: %u  bucket_id: %u  slot: %u\n", address, batch_id, bucket_id, slot);
				} else {
					xchacha_pair entry = { x, chacha };
					buffer[address] = entry;
				}

				if (slot == (FLUSH_DEPTH-1)) {
					atomicAdd(&num_ready, 1);
					//printf("-> bucket %u slot is FLUSH ready, incremented num_ready counter to %u\n", bucket_id, num_ready);
				} else if (slot == (BUCKET_DEPTH-1)) {
					// one bucket got full, so it's time to evict all
					atomicAdd(&eviction_needed, 1); // atomic not really necessary
					//printf("-> bucket %u slot reached max bucket depth %u, set eviction needed to %u\n", bucket_id, BUCKET_DEPTH-1, eviction_needed);
				}

				__syncthreads();
				if (threadIdx.x == 0) {
					//for (int i=0;i<NUM_BUCKETS;i++) {
					//	printf("bucket %u entries: %u\n", i, buffer_counts[i]);
					//}
					//printf("NUM READY after batch %u is %u\n", batch_id, num_ready);
					batch_id++;
				}
				__syncthreads();
			}

			// all buffers should be full OR batch processing is over and we have some to flush
			__syncthreads();

			if (num_ready == NUM_BUCKETS) {
				// flush all up to FLUSH_DEPTH
				//if (threadIdx.x == 0) {
				//	printf("FLUSHING! %u buckets are ready\n", num_ready);
				//}
				for (uint32_t bucket_id=threadIdx.x;bucket_id < NUM_BUCKETS;bucket_id+=blockDim.x) {
					// now increment all global counts to reserve space for flush
					global_counts[bucket_id] = atomicAdd(&results_counts[bucket_id], FLUSH_DEPTH);
					// decrement our buffer counts by the flush depth
					int new_count = buffer_counts[bucket_id] - FLUSH_DEPTH;
					buffer_counts[bucket_id] = new_count;
					if (new_count < FLUSH_DEPTH) atomicSub(&num_ready,1);
				}

				__syncthreads();

				// and write all stuffs
				for (uint32_t i=threadIdx.x;i < NUM_BUCKETS * FLUSH_DEPTH;i+=blockDim.x) {
					uint32_t bucket_id_for_thread = (i / FLUSH_DEPTH) % NUM_BUCKETS;
					int local_pos = i % FLUSH_DEPTH;
					uint32_t buffer_address = bucket_id_for_thread * BUCKET_DEPTH + ((local_pos + active_buffer_pos) % BUCKET_DEPTH);
					xchacha_pair entry = buffer[buffer_address];

					int global_pos = global_counts[bucket_id_for_thread] + local_pos;
					uint32_t global_address = bucket_id_for_thread * MAX_PER_RESULTS_BUCKET + global_pos;
					if (global_address > 256 * MAX_PER_RESULTS_BUCKET) {
						printf("global address out of bounds bucket_id: %u global_pos:%u\n", bucket_id_for_thread, global_pos);
					} else {
						//printf("global address bucket_id: %u global_pos:%u\n", bucket_id_for_thread, global_pos);
						results[global_address] = entry;
					}
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					// switch active buffer position now
					active_buffer_pos = (active_buffer_pos + FLUSH_DEPTH) % BUCKET_DEPTH;
					//printf("  - active_buffer_pos now set to %u\n", active_buffer_pos);
					//for (int i=0;i<NUM_BUCKETS;i++) {
					//	printf("bucket %u entries: %u\n", i, buffer_counts[i]);
					//}
					//printf("NUM READY after batch %u is %u\n", batch_id, num_ready);
				}

				__syncthreads();

			} else if (batch_id == NUM_BATCHES_OF_THREADS) {
				// we finished entering all our data, now check left-over buckets.
				// TODO: check each bucket count and write out data to global.
				//if (threadIdx.x == 0) {
				//	printf("BATCHES COMPLETED: todo finish flushing rest of buffers\n");
				//}

			} else if (eviction_needed > 0) {
				if (threadIdx.x == 0) {
					//printf("HANDLE EVICTION CASE\n");
					for (int i=0;i<NUM_BUCKETS;i++) {
						if (buffer_counts[i] >= BUCKET_DEPTH) {
							eviction_needed = i;
							num_ready = num_ready - 1;
							buffer_counts[i] = 0; // okay, kind of a bug b/c if more than one eviction then we lose entries
							// but for now it's just to test performance.
						}
					}
				}
				__syncthreads();

				for (int i=threadIdx.x;i<BUCKET_DEPTH;i+=blockDim.x) {
					uint32_t bucket_id_for_thread = eviction_needed;
					uint32_t buffer_address = bucket_id_for_thread * BUCKET_DEPTH + i;
					xchacha_pair entry = buffer[buffer_address];

					int global_pos = global_counts[bucket_id_for_thread] + i;
					uint32_t global_address = bucket_id_for_thread * MAX_PER_RESULTS_BUCKET + global_pos;
					results[global_address] = entry;
				}

				// afterwards clear eviction flag
				__syncthreads();

				if (threadIdx.x == 0) {
					eviction_needed = 0;
				}

				__syncthreads();
			}
		}
	}
}



__global__
void gpu_filter_chachas_into_global_kbc_bucket(const uint32_t N, const uint32_t X_START, const __restrict__ uint32_t *chachas,
		uint16_t *out_kbc_ys, uint32_t *out_kbc_xs, unsigned int *kbc_counts) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < N) {
		uint64_t y = (((uint64_t) chachas[x]) << 6) + (x >> 26);
		uint16_t kbc_y = y % kBC;
		uint32_t kbc_bucket_id = y / kBC;
		//printf("x: %u  kbc: %u\n", x, kbc_bucket_id);
		unsigned int kbc_shift = kbc_bucket_id % 32;
		unsigned int kbc_add_slot = 1 << kbc_shift;
		unsigned int value = atomicAdd(&kbc_counts[kbc_bucket_id/32], kbc_add_slot);
		unsigned int slot = (value >> kbc_shift) & 31;
		//kbc_counts[kbc_bucket_id/32] = slot+1;
		// THE ATOMIC ADDS ARE THE PROBLEM!
		//unsigned int slot = atomicAdd(&kbc_counts[kbc_bucket_id % (32768*32)],1);// = slot+1;
		out_kbc_ys[kbc_bucket_id * 32 + slot] = kbc_y;
		out_kbc_xs[kbc_bucket_id * 32 + slot] = x;
	}
}

__global__ void gpu_get_max_counts_from_counter_list(unsigned int *kbc_counts, const int NUM) {
	__shared__ unsigned int max_kbc_count;
	__shared__ unsigned int sum_kbc_count;
	if (threadIdx.x == 0) {
		max_kbc_count = 0;
		sum_kbc_count = 0;
	}
	__syncthreads();
	for (int i=threadIdx.x;i<NUM;i+=blockDim.x) {
		unsigned int kbc_count = kbc_counts[i];
		//if (kbc_count > 150) printf("kbc: %u count: %u\n", i, kbc_count);
		atomicMax(&max_kbc_count, kbc_count);
		atomicAdd(&sum_kbc_count, kbc_count);
	}
	if (threadIdx.x == 0) printf("counter list counts  SUM:%u   MAX:%u\n", sum_kbc_count, max_kbc_count);
}

__global__ void gpu_show_chachas(const uint32_t N, const uint32_t step, uint32_t *chachas) {
	for (int i=0;i<N;i+=step) {
		printf("x: %u chacha: %u\n", i, chachas[i]);
	}
}

// threadIdx.x of 0 gets x=0,1,2,3,4...15
//                1 gets x=16......31

#define ATTACK_WRITE_CHACHAS(chacha_y,i) \
{ \
	shared_chachas[threadIdx.x*16+i] = chacha_y; \
}

#define ATTACK_WRITE_CHACHAS32(chacha_y,i) \
{ \
	shared_chachas[threadIdx.x*32+i] = chacha_y; \
}

#define ATTACK_WRITE_CHACHAS32_PAIR(chacha_y,i) \
{ \
	xchacha_pair pair = { base_x + i, chacha_y }; \
	shared_chachas[threadIdx.x*32+i] = pair; \
}

#define ATTACK_WRITE_CHACHAS_COALESCED(chacha_y,i) \
{ \
	chachas[base_x+threadIdx.x+i*blockDim.x] = chacha_y; \
}

__global__
void gpu_chacha8_k32_write_chachas(const uint32_t N, const uint32_t X_START,
		const __restrict__ uint32_t *input,
		uint32_t *chachas)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	__shared__ uint32_t shared_chachas[256*16]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.

	int base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	const int j = 0;
	if (x_group < end_n) {
		//uint32_t x = x_group << 4;//  *16;
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
		ATTACK_WRITE_CHACHAS(datax[j+0],0);ATTACK_WRITE_CHACHAS(datax[j+1],1);ATTACK_WRITE_CHACHAS(datax[j+2],2);ATTACK_WRITE_CHACHAS(datax[j+3],3);
		ATTACK_WRITE_CHACHAS(datax[j+4],4);ATTACK_WRITE_CHACHAS(datax[j+5],5);ATTACK_WRITE_CHACHAS(datax[j+6],6);ATTACK_WRITE_CHACHAS(datax[j+7],7);
		ATTACK_WRITE_CHACHAS(datax[j+8],8);ATTACK_WRITE_CHACHAS(datax[j+9],9);ATTACK_WRITE_CHACHAS(datax[j+10],10);ATTACK_WRITE_CHACHAS(datax[j+11],11);
		ATTACK_WRITE_CHACHAS(datax[j+12],12);ATTACK_WRITE_CHACHAS(datax[j+13],13);ATTACK_WRITE_CHACHAS(datax[j+14],14);ATTACK_WRITE_CHACHAS(datax[j+15],15);
	}

	__syncthreads();
	for (int i=threadIdx.x;i<blockDim.x*16;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n", threadIdx.x*16 + i, base_x + threadIdx.x*blockDim.x + i);
		chachas[base_x + i] = shared_chachas[i];
	}
}

// can run optimal with 32 blocksize BUT results are out of order since each thread
// writes sequentially and to regain order needed to % blockSize
__global__
void gpu_chacha8_k32_write_chachas_global_coalesced(const uint32_t N, const uint32_t X_START,
		const __restrict__ uint32_t *input,
		uint32_t *chachas)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;


	int base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 16;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 16; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	const int j = 0;
	if (x_group < end_n) {
		//uint32_t x = x_group << 4;//  *16;
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
		ATTACK_WRITE_CHACHAS_COALESCED(datax[j+0],0);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+1],1);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+2],2);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+3],3);
		ATTACK_WRITE_CHACHAS_COALESCED(datax[j+4],4);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+5],5);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+6],6);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+7],7);
		ATTACK_WRITE_CHACHAS_COALESCED(datax[j+8],8);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+9],9);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+10],10);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+11],11);
		ATTACK_WRITE_CHACHAS_COALESCED(datax[j+12],12);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+13],13);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+14],14);ATTACK_WRITE_CHACHAS_COALESCED(datax[j+15],15);
	}
}

// run with 128 blocksize, more doesn't matter.
__global__
void gpu_chacha8_k32_write_chachas32(const uint32_t N, const uint32_t X_START,
		const __restrict__ uint32_t *input,
		uint32_t *chachas)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	__shared__ uint32_t shared_chachas[128*32]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.

	if (blockDim.x > 128) printf("MUST HAVE BLOCKSIZE 128 (RECOMMENDED) OR LESS, OR INCREASED SHARED MEM TO MORE\n");

	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 32;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 32; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

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
		ATTACK_WRITE_CHACHAS32(datax[j+0],0);ATTACK_WRITE_CHACHAS32(datax[j+1],1);ATTACK_WRITE_CHACHAS32(datax[j+2],2);ATTACK_WRITE_CHACHAS32(datax[j+3],3);
		ATTACK_WRITE_CHACHAS32(datax[j+4],4);ATTACK_WRITE_CHACHAS32(datax[j+5],5);ATTACK_WRITE_CHACHAS32(datax[j+6],6);ATTACK_WRITE_CHACHAS32(datax[j+7],7);
		ATTACK_WRITE_CHACHAS32(datax[j+8],8);ATTACK_WRITE_CHACHAS32(datax[j+9],9);ATTACK_WRITE_CHACHAS32(datax[j+10],10);ATTACK_WRITE_CHACHAS32(datax[j+11],11);
		ATTACK_WRITE_CHACHAS32(datax[j+12],12);ATTACK_WRITE_CHACHAS32(datax[j+13],13);ATTACK_WRITE_CHACHAS32(datax[j+14],14);ATTACK_WRITE_CHACHAS32(datax[j+15],15);

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
		ATTACK_WRITE_CHACHAS32(datax[j+0],16+0);ATTACK_WRITE_CHACHAS32(datax[j+1],16+1);ATTACK_WRITE_CHACHAS32(datax[j+2],16+2);ATTACK_WRITE_CHACHAS32(datax[j+3],16+3);
		ATTACK_WRITE_CHACHAS32(datax[j+4],16+4);ATTACK_WRITE_CHACHAS32(datax[j+5],16+5);ATTACK_WRITE_CHACHAS32(datax[j+6],16+6);ATTACK_WRITE_CHACHAS32(datax[j+7],16+7);
		ATTACK_WRITE_CHACHAS32(datax[j+8],16+8);ATTACK_WRITE_CHACHAS32(datax[j+9],16+9);ATTACK_WRITE_CHACHAS32(datax[j+10],16+10);ATTACK_WRITE_CHACHAS32(datax[j+11],16+11);
		ATTACK_WRITE_CHACHAS32(datax[j+12],16+12);ATTACK_WRITE_CHACHAS32(datax[j+13],16+13);ATTACK_WRITE_CHACHAS32(datax[j+14],16+14);ATTACK_WRITE_CHACHAS32(datax[j+15],16+15);

	}

	__syncthreads();
	for (int i=threadIdx.x;i<blockDim.x*32;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n",i,base_x + i);
		chachas[base_x + i] = shared_chachas[i];
	}
}

// run with 128 blocksize, more doesn't matter.
__global__
void gpu_chacha8_k32_write_chachas32_buckets(const uint32_t N, const uint32_t X_START,
		const uint32_t CHACHA_NUM_BUCKETS, const uint32_t CHACHA_MAX_PER_BUCKET,
		const __restrict__ uint32_t *input,
		xchacha_pair *chachas_buckets)
{
	uint32_t datax[16]; // shared memory can go as fast as 32ms but still slower than 26ms with local
	//__shared__ uint32_t datax[33*256]; // each thread (256 max) gets its own shared access starting at 32 byte boundary.
	//uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	__shared__ xchacha_pair shared_chachas[128*32]; // *possibly* using 32 to prevent some bank conflicts can help, but don't thing so.
	__shared__ int counts[32];

	if (blockDim.x > 128) printf("MUST HAVE BLOCKSIZE 128 (RECOMMENDED) OR LESS, OR INCREASED SHARED MEM TO MORE\n");

	uint32_t base_group = blockIdx.x * blockDim.x;
	uint32_t base_x = base_group * 32;
	int x_group = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;
	const uint32_t end_n = N / 32; // 16 x's in each group
	//printf("blockIdx.x: %u blockDim.x: %u gridDim.x: %u base_x: %u  x_group:%u\n", blockIdx.x, blockDim.x, gridDim.x, base_x, x_group);

	const int j = 0;
	if (x_group < end_n) {
		for (int i=threadIdx.x;i<32;i+=blockDim.x) {
			counts[i] = 0;
		}

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
	const uint32_t TEST_BUCKET_BITS = 5;
	const uint32_t TEST_MAX_PER_BUCKET = (1 << (32-TEST_BUCKET_BITS-6))*2;
	for (int i=threadIdx.x;i<blockDim.x*32;i+=blockDim.x) {
		//printf("writing slot %u into global slot %u\n",i,base_x + i);
		xchacha_pair pair = shared_chachas[i];

		uint32_t bucket_id = pair.chacha >> (32 - TEST_BUCKET_BITS); // 16 buckets
		int slot = atomicAdd(&counts[bucket_id],1);
		chachas_buckets[TEST_MAX_PER_BUCKET * bucket_id + base_x + slot] = shared_chachas[i];
	}
}

__global__
void gpu_filter_chachas(
		const uint32_t NUM_PER_BLOCK, const uint32_t N, uint32_t *chachas,
		xchacha_pair *results, xchacha_pair *results2)
{
	// highest performance bucket bits 4, with 1024 threads, num per block 65536. Then all blocks work with L2 cache?
	const uint32_t NUM_BUCKETS = DUMBSORT_NUM_BUCKETS;
	const uint32_t BUCKET_DIVISOR = 1 << (32-DUMBSORT_BUCKET_BITS); // 32bit chacha into 8 bit NUM buckets
	const uint32_t NUM_THREADS = blockDim.x;
	uint32_t NUM_BATCHES_OF_THREADS = NUM_PER_BLOCK / NUM_THREADS; // note num per block must be multiple of num threads
	uint32_t x_group = blockIdx.x;
	uint32_t x_start = x_group * NUM_PER_BLOCK;
	const uint32_t GLOBAL_TILE_START = x_group * DUMBSORT_BATCHES_TILE_SPACE;

	__shared__ int filter_count;


	if (x_start < N) {
		//if (threadIdx.x == 0) {
		//	printf("x start: %u global_bucket_start_pos: %u vs before %u\n", x_start, global_bucket_start_pos, x_start / blockDim.x);
		//}
		if (threadIdx.x == 0) filter_count = 0;
		__syncthreads();

		uint32_t batch_id = 0;
		for (batch_id = 0; batch_id < NUM_BATCHES_OF_THREADS; batch_id++) {
			uint32_t x = x_start + batch_id * NUM_THREADS + threadIdx.x;
			uint32_t chacha = chachas[x];
			xchacha_pair entry = { x, chacha };

			uint32_t bucket_id = chacha / BUCKET_DIVISOR;
			//printf("chacha %u - bucket id: %u\n", chacha, bucket_id);
			if (bucket_id >= NUM_BUCKETS) printf("BUCKET OUT OF RANGE ERROR: %u", bucket_id);
			if (bucket_id == 0) {
				int slot = atomicAdd(&filter_count,1);
				uint32_t results_address = GLOBAL_TILE_START + bucket_id * PHASE_1_DUMBSORT_MAX_PER_BUCKET + slot;
				if (results_address < DUMBSORT_SPACE_NEEDED_FOR_SCRATCH) {
					results[results_address] = entry;
				} else {
					printf("results address overflow %u - global start pos: %u bucket %u slot %u DUMBSORT_SPACE_NEEDED_FOR_SCRATCH: %u\n",
							results_address, GLOBAL_TILE_START, bucket_id, slot, DUMBSORT_SPACE_NEEDED_FOR_SCRATCH);
				}
			}
		}
	}
}

/*template <typename BUCKETED_ENTRY_IN, typename BUCKETED_ENTRY_OUT>
__global__
void gpu_attack_process_global_kbc_pairs_list(
		const int PAIRS_COUNT, unsigned int *kbc_pairs_list_L_bucket_ids,
		const BUCKETED_ENTRY_IN *kbc_global_entries_L, const unsigned int *kbc_global_num_entries_L,
		const uint32_t *rx_list, const uint RX_START, const uint RX_END,
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
}*/


void attack_method_lxs(uint32_t num_lxs) {

	std::cout << "ATTACK METHOD LXS - SORT XS/YS! " << num_lxs << std::endl;

	using milli = std::chrono::milliseconds;
	auto attack_start = std::chrono::high_resolution_clock::now();


	const uint32_t NUM_LXS = 20000000;
	const uint32_t BATCHES = 64;
	const uint32_t NUM_PER_BATCH = UINT_MAX / BATCHES;
	const uint32_t KBC_MAX_BUCKET_SIZE = 32; // SHOULD BE MAX 19 FOR BATCHES 64
	// for our bucketing sort, we have a total number of grouped entries and divvy that up into 256 stripes to get
	// our max per entry
	const uint32_t MAX_TOTAL_GROUPED_ENTRIES = DUMBSORT_BATCHES_TILE_SPACE;
	//const uint32_t MAX_ENTRIES_PER_GROUPING = MAX_TOTAL_GROUPED_ENTRIES / 256;


	auto alloc_start = std::chrono::high_resolution_clock::now();
	int blockSize; uint64_t calc_N;uint64_t calc_blockSize;uint64_t calc_numBlocks;int numBlocks;

	uint32_t *chachas;
	xchacha_pair *xchachas_buffer_1;
	xchacha_pair *xchachas_buffer_2;
	uint32_t *batched_chachas;
	uint32_t *batched_xs;
	unsigned int *xchachas_counts;
	uint16_t *out_kbc_ys;
	uint32_t *out_kbc_xs;
	unsigned int *global_kbc_counts;

	std::cout << " NUM BATCHES:   " << BATCHES << std::endl;
	std::cout << " NUM PER BATCH: " << NUM_PER_BATCH << std::endl;
	std::cout << " KBC MAX BUCKET SIZE:" << KBC_MAX_BUCKET_SIZE << std::endl;
	std::cout << " MAX_TOTAL_GROUPED_ENTRIES: " << MAX_TOTAL_GROUPED_ENTRIES << std::endl;

	std::cout << "      chachas size:" << (sizeof(uint32_t)*NUM_PER_BATCH) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&chachas, sizeof(uint32_t)*NUM_PER_BATCH));
	CUDA_CHECK_RETURN(cudaMemset(chachas, 0, sizeof(uint32_t)*NUM_PER_BATCH));

	std::cout << "      xchachas_grouped size: " << (sizeof(xchacha_pair)*DUMBSORT_SPACE_NEEDED_FOR_SCRATCH) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&xchachas_buffer_1, sizeof(xchacha_pair)*DUMBSORT_SPACE_NEEDED_FOR_SCRATCH));
	CUDA_CHECK_RETURN(cudaMalloc(&xchachas_buffer_2, sizeof(xchacha_pair)*DUMBSORT_SPACE_NEEDED_FOR_SCRATCH));
	CUDA_CHECK_RETURN(cudaMalloc(&xchachas_counts, sizeof(int)*1024)); // can be tuned to less, for now this is general
	CUDA_CHECK_RETURN(cudaMemset(xchachas_counts, 0, 1024));
	batched_chachas = (uint32_t *) &xchachas_buffer_1[0];
	batched_xs = (uint32_t *) &xchachas_buffer_2[0];


	//std::cout << "      out_kbc_ys size:" << (sizeof(uint16_t)*KBC_MAX_BUCKET_SIZE*kBC_NUM_BUCKETS) << std::endl;
	//CUDA_CHECK_RETURN(cudaMalloc(&out_kbc_ys, sizeof(uint16_t)*KBC_MAX_BUCKET_SIZE*kBC_NUM_BUCKETS));
	//std::cout << "      out_kbc_xs size:" << (sizeof(uint32_t)*KBC_MAX_BUCKET_SIZE*kBC_NUM_BUCKETS) << std::endl;
	//CUDA_CHECK_RETURN(cudaMalloc(&out_kbc_xs, sizeof(uint32_t)*KBC_MAX_BUCKET_SIZE*kBC_NUM_BUCKETS));

	std::cout << "      global_kbc_counts size:" << (sizeof(int)*kBC_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&global_kbc_counts, sizeof(int)*kBC_NUM_BUCKETS));
	CUDA_CHECK_RETURN(cudaMemset(global_kbc_counts, 0, kBC_NUM_BUCKETS*sizeof(int)));

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
				static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	} else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	//int device_id = 0;
	//cudaSetDevice(device_id);
	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, device_id);
	//printf("\nDevice %d: \"%s\"\n", device_id, deviceProp.name);
	//cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, deviceProp.persistingL2CacheMaxSize);
	//std::cout << " persisting cache size: " << deviceProp.persistingL2CacheMaxSize << std::endl;
	//std::cout << " accessPolicyMaxWindowSize: " << deviceProp.accessPolicyMaxWindowSize << std::endl;
	//cudaStream_t stream;
	//cudaStreamCreate(&stream);
	//cudaStreamAttrValue attr;
	//attr.accessPolicyWindow.base_ptr = global_kbc_counts;
	//attr.accessPolicyWindow.num_bytes = kBC_NUM_BUCKETS*sizeof(int) / 32;
	//attr.accessPolicyWindow.hitRatio = 1.0;
	//attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
	//attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
	//cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&attr);

	auto alloc_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   alloc time: " << std::chrono::duration_cast<milli>(alloc_finish - attack_start).count() << " ms\n";

	auto compute_only_start = std::chrono::high_resolution_clock::now();


	auto chacha_start = std::chrono::high_resolution_clock::now();
	blockSize = 128; // # of threads per block, maximum is 1024.
	calc_N = UINT_MAX / BATCHES;
	calc_blockSize = blockSize;
	calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 32);
	numBlocks = calc_numBlocks;

	// NEW ALGORITHM!!!!
	// 1) LOAD ALL LX'S INTO GLOBAL_KBC_L1_BUCKETED_YS
	// 2) GO THROUGH EACH RX IN ORDER (NO SORTING!) AND FIND L VALUES IN BUCKETS AND CHECK FOR MATCHES. THAT'S IT.
	// will the cache be fast enough???????? or will sorting be better!?!?!?
	// can experiment with different local_kbc sizes to see if lx's fit in cache we get sufficient performance


	const int groupingBlockSize = 1024;

	//const uint32-t GROUPING_BATCH_MAX_ENTRIES_PER_BUCKET = 65536 / 8;
	int groupingNumBlocks = (NUM_PER_BATCH + groupingBlockSize - 1) / (GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK);

	int bitonicThreads = 512;
	int bitonicBlocks = NUM_PER_BATCH / 1024; // should be 65536
	std::cout << "GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK: " << GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK << "  NUM BLOCKS: " << groupingNumBlocks << std::endl;
	uint32_t X_START = 0;
	for (uint32_t batch_id=0;batch_id < 1; batch_id++) {
		X_START = batch_id * (1 << (32-6));

		gpu_chacha8_k32_write_chachas32<<<numBlocks, blockSize>>>(calc_N, X_START, chacha_input, chachas); // 24ms

		//bitonicSortShared<<<bitonicBlocks,(SHARED_SIZE_LIMIT / 2)>>>(chachas, batched_chachas, batched_xs);
		nickSortShared<<<1,SHARED_SIZE_LIMIT>>>(chachas, batched_chachas, batched_xs);
		//gpu_show_chacha_xs_lists<<<1,1>>>(0,10,batched_chachas, batched_xs);
		//gpu_show_chacha_xs_lists<<<1,1>>>(1024,10,batched_chachas, batched_xs);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//CUDA_CHECK_RETURN(cudaMemset(xchachas_counts, 0, sizeof(int)*1024));
		//gpu_filter_chachas<<<groupingNumBlocks,groupingBlockSize>>>(
		//				GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK, NUM_PER_BATCH, chachas,
		//				xchachas_buffer_1, xchachas_buffer_2);

		//gpu_write_chachas_into_buckets_dumb_batches<<<groupingNumBlocks,groupingBlockSize>>>(
		//		GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK, NUM_PER_BATCH, chachas,
		//		xchachas_buffer_1, xchachas_buffer_2);

		//gpu_write_chachas_into_buckets_with_single_row_depthflush<<<groupingNumBlocks,groupingBlockSize>>>(
		//				GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK, NUM_PER_BATCH, chachas,
		//				MAX_TOTAL_GROUPED_ENTRIES, xchachas_grouped, xchachas_counts);

		/*
		 * GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK: 65536
   - gpu_chacha8_k32_write_chachas 4294967232 in 64 BATCHES results: 1158 ms
Freeing memory...
counter list counts  SUM:67108864   MAX:263782
		 */

		//gpu_write_chachas_into_buckets_with_buffer_batches<<<groupingNumBlocks,groupingBlockSize>>>(
		//	GROUPING_BATCH_NUM_ENTRIES_PER_BLOCK, NUM_PER_BATCH, chachas,
		//	MAX_ENTRIES_PER_GROUPING, xchachas_grouped, xchachas_counts);

		// stupid thrust, not a 100% deal break but close to being too slow
		//thrust::device_ptr<uint32_t> device_xs_R_ptr(out_kbc_xs);
		//thrust::device_ptr<uint32_t> device_ys_R_ptr(chachas);
		//thrust::sort_by_key(device_ys_R_ptr, device_ys_R_ptr + calc_N, device_xs_R_ptr);
		//thrust::sort(device_ys_R_ptr, device_ys_R_ptr + calc_N);

		//CUDA_CHECK_RETURN(cudaMemset(global_kbc_counts, 0, kBC_NUM_BUCKETS*sizeof(int))); // 30ms
		//gpu_filter_chachas_into_global_kbc_bucket<<<numBlocks*32, blockSize>>>(calc_N, X_START, chachas,
		//		out_kbc_ys, out_kbc_xs, global_kbc_counts); // 56ms
		//gpu_get_max_count_in_global_kbc_bucket<<<1,256>>>(global_kbc_counts);

	}


	//CUDA_CHECK_RETURN(cudaMemset(device_global_kbc_num_entries_L, 0, 10000000*sizeof(int)));
	//gpu_chacha8_get_k32_keystream_into_local_kbc_entries<<<numBlocks, blockSize>>>(calc_N, chacha_input,
	//		local_kbc_entries, device_global_kbc_num_entries_L, 0, 2000000);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	auto chacha_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   - gpu_chacha8_k32_write_chachas " << (calc_N*BATCHES) << " in " << BATCHES << " BATCHES results: " << std::chrono::duration_cast<milli>(chacha_finish - chacha_start).count() << " ms\n";
	gpu_get_max_counts_from_counter_list<<<1,1>>>(xchachas_counts, 256);
	//gpu_show_chachas<<<1,1>>>(NUM_PER_BATCH, 10000, chachas);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//gpu_get_max_counts_from_counter_list<<<1,256>>>(global_kbc_counts, kBC_NUM_BUCKETS);

	auto compute_only_finish = std::chrono::high_resolution_clock::now();

	std::cout << "Freeing memory..." << std::endl;
	CUDA_CHECK_RETURN(cudaFree(chachas));
	CUDA_CHECK_RETURN(cudaFree(out_kbc_ys));
	CUDA_CHECK_RETURN(cudaFree(out_kbc_xs));
	CUDA_CHECK_RETURN(cudaFree(global_kbc_counts));



	auto attack_finish = std::chrono::high_resolution_clock::now();
	std::cout << "   compute only time: " << std::chrono::duration_cast<milli>(compute_only_finish - compute_only_start).count() << " ms\n";
	std::cout << "   attack total time: " << std::chrono::duration_cast<milli>(attack_finish - attack_start).count() << " ms\n";
	std::cout << "end." << std::endl;
}







#endif /* ATTACK_METHOD_LXS2_HPP_ */
