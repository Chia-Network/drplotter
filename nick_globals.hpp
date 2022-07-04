#ifndef NICK_GLOBALS_HPP_
#define NICK_GLOBALS_HPP_

#include <iostream>
#include <stdlib.h>

using std::string;

const uint32_t BATCHES = 64;
const uint64_t BATCHBC = (uint64_t) 1 << (38 - 6);

const uint32_t KBC_MAX_ENTRIES_PER_BUCKET = 400;
const uint32_t kBC_NUM_BUCKETS = 18188177;
const uint32_t kBC_LAST_BUCKET_ID = 18188176;
const uint16_t kBC = 15113;
const uint32_t KBCS_PER_BATCH = (kBC_NUM_BUCKETS / BATCHES)+1;
const uint32_t KBC_LOCAL_NUM_BUCKETS = KBCS_PER_BATCH + 1; // +1 is for including last R bucket space

#define CALC_BATCH_BUCKET_ADD_Y(batch_id) ((((uint64_t) 1) << (38-6)) * ((uint64_t) batch_id))
#define CALC_KBC_BUCKET_ADD_Y(kbc_bucket_id) (((uint64_t) kBC) * ((uint64_t) kbc_bucket_id))

#define MIN_KBC_BUCKET_FOR_BATCH(batch_id) \
	( (uint32_t) ((((uint64_t) 1 << 32) * ((uint64_t) (batch_id))) / ((uint64_t) kBC) ));


const uint64_t HOST_UNIT_BYTES = 20; //12// Bytes used for biggest host entry.
const uint64_t HOST_MAX_BLOCK_ENTRIES = 1114112;//1114112; // MUST be multiple of 32 so it works with bit masking // 1052614 (min calculated) // 1258291; // (120 * ((uint64_t) 1 << 32)) / (100*(BATCHES * BATCHES));
const uint64_t HOST_ALLOCATED_ENTRIES = HOST_MAX_BLOCK_ENTRIES * BATCHES * BATCHES;
const uint64_t HOST_ALLOCATED_BYTES = HOST_UNIT_BYTES * HOST_ALLOCATED_ENTRIES;

const uint64_t DEVICE_BUFFER_UNIT_BYTES = 32; // Tx_pairing_chunk_meta4 is 24 bytes, w/ backref is 32 bytes
const uint64_t DEVICE_BUFFER_ALLOCATED_ENTRIES = KBC_LOCAL_NUM_BUCKETS * KBC_MAX_ENTRIES_PER_BUCKET; // HOST_MAX_BLOCK_ENTRIES * BATCHES;// DEVICE_BUFFER_ALLOCATED_ENTRIES = 120 * ((uint64_t) 1 << 32) / (100*BATCHES);
const uint64_t DEVICE_BUFFER_ALLOCATED_BYTES = DEVICE_BUFFER_ALLOCATED_ENTRIES * DEVICE_BUFFER_UNIT_BYTES;
const uint64_t BACKREF_UNIT_BYTES = 12; // backref w/y for last table is 12 bytes
const uint64_t BACKREF_ALLOCATED_BYTES = DEVICE_BUFFER_ALLOCATED_ENTRIES * BACKREF_UNIT_BYTES;


const uint64_t CROSS_MATRIX_BC = (2097152 * 128) + kBC - ((2097152 * 128) % kBC);
const uint64_t CROSS_MATRIX_NUM_BUCKETS = 1024; // each batch splits into buckets, the max per bucket is dependent on size of batch
const uint64_t CROSS_MATRIX_BATCH_MAX_ENTRIES_PER_BUCKET = (119 * ((uint64_t)1 << 32)) / (100*(CROSS_MATRIX_NUM_BUCKETS * BATCHES));
const uint64_t CROSS_MATRIX_ALLOCATED_SPACE_PER_BATCH = CROSS_MATRIX_BATCH_MAX_ENTRIES_PER_BUCKET * CROSS_MATRIX_NUM_BUCKETS;
const uint64_t CROSS_MATRIX_ALLOCATED_SPACE = CROSS_MATRIX_ALLOCATED_SPACE_PER_BATCH * BATCHES;








static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

uint32_t *chacha_input;

// output from F(x) -> chacha
struct F1_Bucketed_kBC_Entry {
	uint32_t x;
	uint32_t y;
};

struct T1_Match {
	uint32_t Lx;
	uint32_t Rx;
	uint32_t y;
};

struct T1_Pairing_Chunk {
	uint32_t Lx;
	uint32_t Rx;
	uint32_t y;
};

struct Tx_Bucketed_Final_Y {
	uint32_t y;
};

struct Tx_Bucketed_Meta1 {
	uint32_t meta[1];
	uint32_t y;
};

struct Tx_Bucketed_Meta2 {
	uint32_t meta[2];
	uint32_t y;
};

struct Tx_Bucketed_Meta3 {
	uint32_t meta[3];
	uint32_t y;
};

struct Tx_Bucketed_Meta4 {
	uint32_t meta[4];
	uint32_t y;
};

struct Tx_Bucketed_Meta2_Blockposref {
	uint32_t meta[2];
	uint32_t y;
	uint32_t blockposref;
};

struct Tx_Bucketed_Meta3_Blockposref {
	uint32_t meta[3];
	uint32_t y;
	uint32_t blockposref;
};

struct Tx_Bucketed_Meta4_Blockposref {
	uint32_t meta[4];
	uint32_t y;
	uint32_t blockposref;
};


struct Tx_Pairing_Chunk_Meta2 {
	uint64_t y;
	uint32_t meta[2];
	//uint16_t idxL;
	//uint16_t idxR;
	//uint32_t p_b_id;
};

struct Tx_Pairing_Chunk_Meta3 {
	uint64_t y;
	uint32_t meta[2];
	//uint16_t idxL;
	//uint16_t idxR;
	//uint32_t p_b_id;
};

struct Tx_Pairing_Chunk_Meta4 {
	uint64_t y;
	uint32_t meta[4];
	//uint16_t idxL;
	//uint16_t idxR;
	//uint32_t p_b_id;
};

struct Index_Match {
	uint16_t idxL;
	uint16_t idxR;
};

// our base pairing struct T3.
struct T2BaseRef {
	uint32_t Lx1;
	uint32_t Lx2;
};

struct T3BaseRef {
	uint32_t Lx1;
	uint32_t Lx2;
	uint32_t Lx3;
	uint32_t Lx4;
};

struct T2BaseRefWithUsed {
	uint32_t Lx1;
	uint32_t Lx2;
	bool used;
};

struct BackRef {
	uint32_t prev_block_ref_L; // (block_id(L) << (32 - 6)) + block_pos
	uint32_t prev_block_ref_R; // (block_id(R) << (32 - 6)) + block_pos
};

struct T6BackRef { // 12 bytes
	uint32_t prev_block_ref_L; // (block_id(L) << (32 - 6)) + block_pos
	uint32_t prev_block_ref_R; // (block_id(R) << (32 - 6)) + block_pos
	uint32_t y;
};

struct T6FinalEntry {
	uint32_t refL; // 6,6,6 = 24
	uint32_t refR; // 6,6,6 = 24
	uint32_t y;    // 32
};

struct T4FinalEntry {
	uint32_t Lx1,Lx2,Lx3,Lx4,Lx5,Lx6,Lx7,Lx8;
};


struct RBid_Entry {
	uint32_t x;
	uint16_t pos;
};

// chia specific constants
const uint32_t K_SIZE = 32;
const uint64_t K_MAX = ((uint64_t) 1 << K_SIZE);
const uint64_t K_MAX_Y = K_MAX << 6;
const uint8_t kExtraBits = 6;
const uint16_t kB = 119;
const uint16_t kC = 127;
const uint32_t nickBC = (2097152 * 128) + kBC - ((2097152 * 128) % kBC);
const uint32_t NICK_BUCKET_MAX_ENTRIES = 34000 * 128;
const uint32_t NICK_NUM_BUCKETS = 1024;

// code below is WRONG! 2nd clause only uses batch_id
//#define CRISS_CROSS_BLOCK_ID(table, batch_id, block_id) \
//(((table % 2) == 1) ? batch_id * BATCHES  + block_id : batch_id * BATCHES + batch_id)


uint64_t getCrissCrossBlockId(uint8_t table, uint32_t batch_id, uint32_t block_id) {
    uint64_t cross_row_id = batch_id;
    uint64_t cross_column_id = block_id;
    if ((table % 2) == 1) {
        return (cross_row_id * BATCHES  + cross_column_id);
    } else {
        return (cross_column_id * BATCHES  + cross_row_id);
    }
}

inline uint64_t getCrissCrossBlockEntryStartPosition(uint64_t criss_cross_id) {
    return criss_cross_id * HOST_MAX_BLOCK_ENTRIES;
}


string Strip0x(const string &hex)
{
    if (hex.size() > 1 && (hex.substr(0, 2) == "0x" || hex.substr(0, 2) == "0X")) {
        return hex.substr(2);
    }
    return hex;
}

void HexToBytes(const string &hex, uint8_t *result)
{
    for (uint32_t i = 0; i < hex.length(); i += 2) {
        string byteString = hex.substr(i, 2);
        uint8_t byte = (uint8_t)strtol(byteString.c_str(), NULL, 16);
        result[i / 2] = byte;
    }
}

void chacha_setup() {
	string id = "022fb42c08c12de3a6af053880199806532e79515f94e83461612101f9412f9e";

	uint8_t enc_key[32];

	id = Strip0x(id);
	std::array<uint8_t, 32> id_bytes;
	HexToBytes(id, id_bytes.data());
	uint8_t* orig_key = id_bytes.data();

	enc_key[0] = 1;
	memcpy(enc_key + 1, orig_key, 31);

	CUDA_CHECK_RETURN(cudaMallocManaged(&chacha_input, 16*sizeof(uint32_t)));
	// Setup ChaCha8 context with zero-filled IV
	chacha8_keysetup_data(chacha_input, enc_key, 256, NULL);

}

// chacha specific macros end

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


#endif /* NICK_GLOBALS_HPP_ */

/*void doF1Original() {
	const uint32_t N = UINT_MAX;

	const uint32_t N_PER_BATCH = (N) / BATCHES;
	uint32_t KBCS_PER_BATCH = (kBC_LAST_BUCKET_ID + 1) / BATCHES;
	uint32_t KBC_START = 0;
	uint32_t KBC_END = KBC_START + KBCS_PER_BATCH;
	const uint32_t KBC_LOCAL_NUM_BUCKETS = KBC_END - KBC_START + 1; // +1 is for including last R bucket space
	const uint32_t KBC_ALLOCATED_SPACE = KBC_MAX_ENTRIES_PER_BUCKET * KBC_LOCAL_NUM_BUCKETS;
	const uint64_t HOST_COPY_BUFFER_ALLOCATED_SPACE = KBC_ALLOCATED_SPACE; // should then cp to disk anyway.(((uint64_t) 1) << 32);
	const uint32_t MAX_RESULTS = KBC_ALLOCATED_SPACE;

	std::cout << "doF1  N:" << N << "  BATCHES:" << BATCHES << "   N per batch:" << N_PER_BATCH << std::endl;
	std::cout << "   CROSS_MATRIX_NUM_BUCKETS:" << CROSS_MATRIX_NUM_BUCKETS << " MAX_ENTRIES BATCH:" << CROSS_MATRIX_BATCH_MAX_ENTRIES_PER_BUCKET << std::endl;
	std::cout << "   gpu memory alloc:" << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();

	int* local_kbc_num_entries;
	std::cout << "      local_kbc_num_entries " << KBC_LOCAL_NUM_BUCKETS << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*KBC_LOCAL_NUM_BUCKETS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&local_kbc_num_entries, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));

	int* global_kbc_num_entries;
	std::cout << "      global_kbc_num_entries " << (kBC_LAST_BUCKET_ID+1) << " * (max per bucket: " << KBC_MAX_ENTRIES_PER_BUCKET << ") size:" << (sizeof(int)*(kBC_LAST_BUCKET_ID+1)) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&global_kbc_num_entries, (kBC_LAST_BUCKET_ID+1)*sizeof(int)));


	// todo: allocate this as fx, then map to f1, f2 etc.
	Tx_Bucketed_Meta4 *device_bucketed_meta_entries;
	std::cout << "      device_bucketed_meta_entries " << KBC_ALLOCATED_SPACE << " * (Tx_Bucketed_Meta4:" <<  sizeof(Tx_Bucketed_Meta4) << ") = " << (sizeof(Tx_Bucketed_Meta4)*KBC_ALLOCATED_SPACE) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_bucketed_meta_entries, KBC_ALLOCATED_SPACE*sizeof(Tx_Bucketed_Meta4)));

	Tx_Pairing_Chunk_Meta4 *device_pairing_chunks;
	std::cout << "      device_pairing_chunks " << MAX_RESULTS << " * (Tx_Pairing_Chunk_Meta4:" << sizeof(Tx_Pairing_Chunk_Meta4) << ") = " << (sizeof(Tx_Pairing_Chunk_Meta4)*MAX_RESULTS) << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc(&device_pairing_chunks, MAX_RESULTS*sizeof(Tx_Pairing_Chunk_Meta4)));
	int *pairing_chunks_count;
	std::cout << "      t1_pairing_chunks_count (1)" << std::endl;
	CUDA_CHECK_RETURN(cudaMallocManaged(&pairing_chunks_count, sizeof(int)));

	Tx_Pairing_Chunk_Meta4 *host_copy_buffer;
	std::cout << "      HOST t1_pairing_chunks * " << HOST_COPY_BUFFER_ALLOCATED_SPACE << " size:" << (sizeof(Tx_Pairing_Chunk_Meta4)*HOST_COPY_BUFFER_ALLOCATED_SPACE) << std::endl;
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_copy_buffer, HOST_COPY_BUFFER_ALLOCATED_SPACE * sizeof(Tx_Pairing_Chunk_Meta4))); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();

	Tx_Pairing_Chunk_Meta4 *host_criss_cross_store;
	//std::cout << "      HOST host_criss_cross_store * " << CROSS_MATRIX_ALLOCATED_SPACE << " size:" << (sizeof(Tx_Bucketed_Meta4)*CROSS_MATRIX_ALLOCATED_SPACE) << std::endl;
	//host_criss_cross_store = (Tx_Bucketed_Meta4 *) malloc(CROSS_MATRIX_ALLOCATED_SPACE * sizeof(Tx_Bucketed_Meta4));
	//CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_criss_cross_store, CROSS_MATRIX_ALLOCATED_SPACE * sizeof(Tx_Bucketed_Meta4))); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();


	uint32_t *criss_cross_num_entries; // host num_buckets for each batch
	std::cout << "      HOST criss_cross_num_entries * " << (CROSS_MATRIX_NUM_BUCKETS * BATCHES) << " size:" << (sizeof(uint32_t) * CROSS_MATRIX_NUM_BUCKETS * BATCHES) << std::endl;
	criss_cross_num_entries = (uint32_t *) malloc(CROSS_MATRIX_NUM_BUCKETS * BATCHES * sizeof(uint32_t));


	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "    mem allocation done. " << std::chrono::duration_cast<milli>(finish - total_start).count() << " ms\n";

	auto total_start_without_memory = std::chrono::high_resolution_clock::now();

	int blockSize = 64; // # of threads per block, maximum is 1024.
	const uint64_t calc_N = N;
	const uint64_t calc_blockSize = blockSize;
	const uint64_t calc_numBlocks = (calc_N + calc_blockSize - 1) / (blockSize * 16);
	int numBlocks = calc_numBlocks;
	std::cout << "  Block configuration: [blockSize:" << blockSize << "  numBlocks:" << numBlocks << "]" << std::endl;

	//batches = 2;
	int64_t total_compute_time_ms = 0;
	int64_t total_transfer_time_ms = 0;
	uint32_t total_f2_results_count = 0;

	// map for table 1.
	{
		T1_Pairing_Chunk *t1_pairing_chunks = (T1_Pairing_Chunk *) device_pairing_chunks;
		F1_Bucketed_kBC_Entry *local_kbc_entries = (F1_Bucketed_kBC_Entry *) device_bucketed_meta_entries;
		T1_Pairing_Chunk *host_t1_pairing_chunks = (T1_Pairing_Chunk *) host_copy_buffer;
		uint32_t batches_to_go = BATCHES;
		while (batches_to_go > 0) {

			std::cout << "   gpuScanIntoKbcBuckets BATCHES to go: " << batches_to_go << std::endl <<
					"     SPANNING FOR BUCKETS    count:" << (KBC_END - KBC_START + 1) << "  KBC_START: " << KBC_START << "   KBC_END: " << KBC_END << std::endl;
			std::cout << "   Generating F1 results into kbc buckets...";
			auto batch_start = std::chrono::high_resolution_clock::now();
			auto start = std::chrono::high_resolution_clock::now();

			// don't forget to clear counter...
			CUDA_CHECK_RETURN(cudaMemset(local_kbc_num_entries, 0, KBC_LOCAL_NUM_BUCKETS*sizeof(int)));

			gpu_chacha8_get_k32_keystream_into_local_kbc_entries<<<numBlocks, blockSize>>>(N, chacha_input,
					local_kbc_entries, local_kbc_num_entries, KBC_START, KBC_END);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			auto finish = std::chrono::high_resolution_clock::now();
			std::cout << "   done.     " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";

			std::cout << "   Finding matches...";
			(*pairing_chunks_count) = 0; // set...
			CUDA_CHECK_RETURN(cudaMemset(global_kbc_num_entries, 0, (kBC_LAST_BUCKET_ID+1)*sizeof(int)));
			gpu_find_f1_matches<<<(KBC_LOCAL_NUM_BUCKETS-1), 256>>>(KBC_START, KBC_END,
					local_kbc_entries, local_kbc_num_entries,
					t1_pairing_chunks, pairing_chunks_count, MAX_RESULTS);
			//gpu_find_fx_matches<Tx_Bucketed_Meta1,Tx_Bucketed_Meta2,1><<<(KBC_LOCAL_NUM_BUCKETS-1), 256>>>(KBC_START, KBC_END,
			//		local_kbc_entries, local_kbc_num_entries,
			//		t1_pairing_chunks, t1_pairing_chunks_count, MAX_RESULTS);
			//gpu_find_matches<<<1, 64>>>(1,2, KBC_MAX_ENTRIES_PER_BUCKET, local_kbc_entries, local_kbc_num_entries);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			finish = std::chrono::high_resolution_clock::now();

			total_compute_time_ms += std::chrono::duration_cast<milli>(finish - batch_start).count();
			std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";




			// now copy pair results to CPU memory.
			int num_results = (*pairing_chunks_count);
			total_f2_results_count += num_results;
			std::cout << "   Copying " << num_results << " T1 pairing chunks to CPU...";
			start = std::chrono::high_resolution_clock::now();
			CUDA_CHECK_RETURN(cudaMemcpy(host_t1_pairing_chunks,t1_pairing_chunks,num_results*sizeof(T1_Pairing_Chunk),cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			finish = std::chrono::high_resolution_clock::now();
			total_transfer_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
			std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";


			// and move from CPU memory into reserved criss cross buckets
			std::cout << "    Moving pairing chunks into CPU criss cross storage\n";
			start = std::chrono::high_resolution_clock::now();
			uint32_t batch_id = BATCHES-batches_to_go;
			Tx_Bucketed_Meta2 *host_cast = (Tx_Bucketed_Meta2 *) host_criss_cross_store;
			//cpuT1MoveCopyBufferToCrissCross(batch_id, host_t1_pairing_chunks, num_results, host_cast, &criss_cross_num_entries[batch_id]);
			finish = std::chrono::high_resolution_clock::now();
			total_transfer_time_ms += std::chrono::duration_cast<milli>(finish - start).count();
			std::cout << "   done. " << std::chrono::duration_cast<milli>(finish - start).count() << " ms\n";


			std::cout << "  ** batch finish ** " << std::chrono::duration_cast<milli>(finish - batch_start).count() << " ms\n";
			batches_to_go--;

			KBC_START += KBCS_PER_BATCH;
			//if (BATCHES == 0) {
			//	KBC_END = kBC_LAST_BUCKET_ID;
			//} else {
			KBC_END = KBC_START + KBCS_PER_BATCH;
			//}
			if ((KBC_END - KBC_START + 1) > KBC_LOCAL_NUM_BUCKETS) {
				std::cout << "ERROR: kbc span is more than local buckets allocated!\n" << std::endl;
			}
		}
	}


	finish = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "Total time: " << std::chrono::duration_cast<milli>(finish - total_start).count() << " ms\n";
	std::cout << " w/o alloc: " << std::chrono::duration_cast<milli>(finish - total_start_without_memory).count() << " ms\n";
	std::cout << " gpu compute: " << total_compute_time_ms << " ms\n";
	std::cout << "    transfer: " << total_transfer_time_ms << " ms\n";

	std::cout << "*********************" << std::endl;
	/*uint32_t total_entries = 0;
	for (int bucket_id=0;bucket_id<2;bucket_id++) { // NICK_NUM_BUCKETS;i++) {
		int num = local_kbc_num_entries[bucket_id];
		std::cout << "KBC LOCAL num entries bucket " << bucket_id << " : " << num << std::endl;
		total_entries += num;
		//for (int i=0;i<num;i++) {
		//	Bucketed_kBC_Entry entry = local_kbc_entries[bucket_id*KBC_MAX_ENTRIES_PER_BUCKET + i];
		//	std::cout << " x: " << entry.x << " f(x):" << CALC_Y_BUCKETED_KBC_ENTRY(entry, bucket_id) << std::endl;
		//}
	}
	std::cout << "  total entries: " << total_entries << std::endl;
*/
/*
	CUDA_CHECK_RETURN(cudaFree(local_kbc_num_entries));
	CUDA_CHECK_RETURN(cudaFree(device_bucketed_meta_entries));
	CUDA_CHECK_RETURN(cudaFree(device_pairing_chunks));
}*/


