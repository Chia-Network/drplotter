/*
 * phase2.hpp
 *
 *  Created on: Oct 15, 2021
 *      Author: nick
 */

#ifndef PHASE2_HPP_
#define PHASE2_HPP_

#include "nick_globals.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>


// bladebit
//  phase 1: 209s
//  phase 2: 25s
//  phase 3: 102s
//  phase 4: <1s

const uint64_t PHASE_2_MAX_BYTES_PER_UNIT = 12; // enter max. bytes used per entry for any of the tables
const uint64_t PHASE_2_ALLOCATED_BYTES_PER_TABLE = PHASE_2_MAX_BYTES_PER_UNIT * DEVICE_BUFFER_ALLOCATED_ENTRIES; // enter max. bytes used per entry for any of the tables

uint32_t num_set_t4 = 0;
uint32_t num_same_addresses = 0;
uint32_t num_set_t5 = 0;

void readT2BlockFilesToHostMem(uint32_t batch_id, T2BaseRef *t2_data, uint32_t *num_entries) {
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
		std::string filename = "/mnt/kioxia/tmp/T2-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
		//if (batch_id == 0) {
		//	std::cout << "Reading file [" << filename << "]";
		//} else {
		//	std::cout << " [" << filename << "]";
		//}
		FILE* pFile;
		pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
		if (fread(&num_entries[block_id], sizeof(uint32_t), 1, pFile)) {
			//std::cout << " num_entries: " << num_entries[block_id] << std::endl;
			if (fread(t2_data, sizeof(T2BaseRef), num_entries[block_id], pFile)) {
				//std::cout << "success.";
			} else {
				std::cout << "failed.";
			}
		}
		fclose(pFile);
		//if (batch_id == BATCHES-1) {
		//	std::cout << " done." << std::endl;
		//}
		//for (int i = 0; i < 1; i++) {
		//	std::cout << "Value " << i << " is: " << t2_data[0].Lx1 << std::endl;
		//}
	}
}

void readTxBackRefBlockFilesToHostMem(uint32_t table, uint32_t batch_id, BackRef *tx_data, uint32_t *num_entries) {
	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
		std::string filename = "/mnt/kioxia/tmp/T"+std::to_string(table)+"BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
		if (batch_id == 0) {
			std::cout << "Reading file [" << filename << "]";
		} else {
			std::cout << " [" << filename << "]";
		}
		FILE* pFile;
		pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
		//uint32_t num_entries;
		if (fread(&num_entries[block_id], sizeof(uint32_t), 1, pFile)) {
			std::cout << " num_entries: " << num_entries[block_id] << std::endl;
			if (fread(tx_data, sizeof(BackRef), num_entries[block_id], pFile)) {
				std::cout << "success.";
			} else {
				std::cout << "failed.";
			}
		} else {
			std::cout << "Failed to read count " << std::endl;
		}
		fclose(pFile);
		if (batch_id == BATCHES-1) {
			std::cout << " done." << std::endl;
		}
	}

}

void readT6BackRefBlockFilesToHostMem(uint32_t batch_id, uint32_t block_id, T6BackRef *tx_data, uint32_t &num_entries) {
	std::string filename = "/mnt/kioxia/tmp/T6BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";

	FILE* pFile;
	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	if (fread(&num_entries, sizeof(uint32_t), 1, pFile)) {
		std::cout << "reading..." << num_entries << std::endl;
		if (!fread(&tx_data, sizeof(T6BackRef), num_entries, pFile)) {
			std::cout << "failed.";
		}
	}
	fclose(pFile);
}

void readT2BlockFile(uint32_t batch_id, uint32_t block_id, T2BaseRef *t2_data, uint32_t &num_entries) {
	std::string filename = "/mnt/kioxia/tmp/T2-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	//if (batch_id == 0) {
	//	std::cout << "Reading file [" << filename << "]";
	//} else {
	////	std::cout << " [" << filename << "]";
	//}
	FILE* pFile;
	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	if (fread(&num_entries, sizeof(uint32_t), 1, pFile)) {
		//std::cout << " num_entries: " << num_entries << std::endl;
		if (fread(t2_data, sizeof(T2BaseRef), num_entries, pFile)) {
			//std::cout << "success.";
		} else {
			std::cout << "failed.";
		}
	}
	fclose(pFile);
	//if (batch_id == BATCHES-1) {
	//	std::cout << " done." << std::endl;
	//}
	//for (int i = 0; i < 1; i++) {
	//	std::cout << "Value " << i << " is: " << t2_data[0].Lx1 << std::endl;
	//}
}

void readBackRefBlockFile(uint32_t table, uint32_t batch_id, uint32_t block_id, BackRef *tx_data, uint32_t &num_entries) {
	std::string filename = "/mnt/kioxia/tmp/T"+std::to_string(table)+"BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	FILE* pFile;
	//std::cout << "reading " << filename << std::endl;
	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	if (fread(&num_entries, sizeof(uint32_t), 1, pFile)) {
		if (!fread(tx_data, sizeof(BackRef), num_entries, pFile)) {
			std::cout << "failed reading " << filename;
		}
	}
	fclose(pFile);
}

void readT6BlockFile(uint32_t batch_id, uint32_t block_id, T6BackRef *t6_data, uint32_t &num_entries) {
	std::string filename = "/mnt/kioxia/tmp/T6BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	FILE* pFile;
	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	if (fread(&num_entries, sizeof(uint32_t), 1, pFile)) {
		if (fread(t6_data, sizeof(T6BackRef), num_entries, pFile)) {
			//std::cout << "success.";
		} else {
			std::cout << "failed.";
		}
	}
	fclose(pFile);
}

void readT3BaseRefBlockFile(uint32_t batch_id, uint32_t block_id, T3BaseRef *t3_data, uint32_t &num_entries) {
	std::string filename = "/mnt/kioxia/tmp/T3BaseRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	FILE* pFile;
	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	if (fread(&num_entries, sizeof(uint32_t), 1, pFile)) {
		if (fread(t3_data, sizeof(T3BaseRef), num_entries, pFile)) {
			//std::cout << "success.";
		} else {
			std::cout << "failed.";
		}
	}
	fclose(pFile);
}

// should total around 48GB...so maybe don't have to write to disk...
void writeT6FinalBlockFile(uint32_t batch_id, uint32_t block_id, T6FinalEntry *t6_final_data, uint32_t &num_entries) {
	if (num_entries == 0) {
		return;
	}
	std::string filename = "/mnt/kioxia/tmp/T6Final-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
		//if (batch_id == 0) {
		//	std::cout << "Writing backref to file [" << filename << "]";
		//} else {
		//	std::cout << " [" << filename << "]";
		//}
		FILE* pFile;
		pFile = fopen(filename.c_str(), "wb"); // 41228ms for block level writing, 40912ms for batch writing??
		fwrite(&num_entries, sizeof(uint32_t), 1, pFile); // write the num entries first.
		fwrite(t6_final_data, 1, num_entries * sizeof(T6FinalEntry), pFile);
		fclose(pFile);
		//if (batch_id == BATCHES-1) {
		//	std::cout << " done." << std::endl;
		//}

}

void readT2BlockEntry(uint32_t batch_id, uint32_t block_id, uint32_t idx, T2BaseRef *t2_entry) {
	std::string filename = "/mnt/kioxia/tmp/T2-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	uint32_t seekpos = idx * sizeof(T2BaseRef) + sizeof(uint32_t);
	std::cout << "Reading single entry from " << filename << " pos: " << seekpos << std::endl;
	FILE* pFile;

	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	fseek ( pFile , seekpos , SEEK_SET );
	fread(t2_entry, sizeof(T2BaseRef), 1, pFile);
	fclose(pFile);
}

void readT3BlockEntry(uint32_t batch_id, uint32_t block_id, uint32_t idx, T3BaseRef *t3_entry) {
	std::string filename = "/mnt/kioxia/tmp/T3BaseRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	uint32_t seekpos = idx * sizeof(T3BaseRef) + sizeof(uint32_t);
	std::cout << "Reading single entry from " << filename << " pos: " << seekpos << std::endl;
	FILE* pFile;

	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	fseek ( pFile , seekpos , SEEK_SET );
	fread(t3_entry, sizeof(T3BaseRef), 1, pFile);
	fclose(pFile);
}

void readBackRefBlockEntry(uint32_t table, uint32_t batch_id, uint32_t block_id, uint32_t idx, BackRef *return_data) {
	std::string filename = "/mnt/kioxia/tmp/T" + std::to_string(table) + "BackRef-" + std::to_string(batch_id) + "-" + std::to_string(block_id) + ".tmp";
	uint32_t seekpos = idx * sizeof(BackRef) + sizeof(uint32_t);
	std::cout << "Reading single entry from " << filename << " pos: " << seekpos << std::endl;
	FILE* pFile;

	pFile = fopen(filename.c_str(), "rb"); // 41228ms for block level writing, 40912ms for batch writing??
	fseek ( pFile , seekpos , SEEK_SET );
	fread(return_data, sizeof(BackRef), 1, pFile);
	fclose(pFile);
}

void backPropagate(uint32_t table, uint32_t batch_id, uint32_t block_id, uint32_t idx) {
	std::cout << "Back propagate to table: " << table << " batch_id:" << batch_id << " block_id:" << block_id << " idx:" << idx << std::endl;
	BackRef entry;
	readBackRefBlockEntry(table, batch_id, block_id, idx, &entry);
	//std::cout << "Ready entry L:" << entry_data.prev_block_ref_L << " R:" << entry_data.prev_block_ref_R << std::endl;
	uint32_t prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
	uint32_t prev_idx_L = entry.prev_block_ref_L & 0x3FFFFFF;
	uint32_t prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
	uint32_t prev_idx_R = entry.prev_block_ref_R & 0x3FFFFFF;
	printf("T%uBackRef batch_id:%u block_id:%u! L:%u R:%u L_block_id:%u L_idx:%u R_block_id:%u R_idx:%u y:%u\n",
								table, batch_id, block_id, entry.prev_block_ref_L, entry.prev_block_ref_R,
								prev_block_id_L, prev_idx_L,
								prev_block_id_R, prev_idx_R);
	/*if (table > 3) {
		backPropagate(table-1, prev_block_id_L, batch_id, prev_idx_L);
		backPropagate(table-1, prev_block_id_R, batch_id, prev_idx_R);
	} else if (table == 3) {
		// read T2 entries right?
		T2BaseRef L, R;
		readT2BlockEntry(prev_block_id_L, batch_id, prev_idx_L, &L);
		readT2BlockEntry(prev_block_id_R, batch_id, prev_idx_R, &R);
		printf("T2 L: %u %u\n", L.Lx1, L.Lx2);
		printf("T2 R: %u %u\n", R.Lx1, R.Lx2);
	}*/

	if (table > 4) {
		backPropagate(table-1, prev_block_id_L, batch_id, prev_idx_L);
		backPropagate(table-1, prev_block_id_R, batch_id, prev_idx_R);
	} else if (table == 4) {
		// read T3 entries right?
		T3BaseRef L, R;
		readT3BlockEntry(prev_block_id_L, batch_id, prev_idx_L, &L);
		readT3BlockEntry(prev_block_id_R, batch_id, prev_idx_R, &R);
		printf("T3 pos: %u L: %u %u %u %u\n", prev_idx_L, L.Lx1, L.Lx2, L.Lx3, L.Lx4);
		printf("T3 pos: %u R: %u %u %u %u\n", prev_idx_R, R.Lx1, R.Lx2, R.Lx3, R.Lx4);
	}

}



// try to see if we have correct back propagation values stored.
// y = 573855352
// xs 602009779,2127221679, 3186459061,443532047, 1234434947,1652736830, 396228306,464118917,3981993340,
//    3878862024,1730679522,3234011360,521197720,2635193875,2251292298,608281027,1468569780,2075860307,
//    2880258779,999340005,1240438978,4293399624,4226635802,1031429862,2391120891,3533658526,3823422504,
//    3983813271,4180778279,2403148863,2441456056,319558395,2338010591,196206622,1637393731,853158574,2704638588,
//    2368357012,1703808356,451208700,2145291166,2741727812,3305809226,1748168268,415625277,3051905493,4257489502,
//    1429077635,2438113590,3028543211,3993396297,2678430597,458920999,889121073,3577485087,1822568056,2222781147,
//    1942400192,195608354,1460166215,2544813525,3231425778,2958837604,2710532969


/*
 * this is what a single solution looks like file-wise
 * -rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-10-15.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-11-0.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-13-51.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-15-12.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-17-3.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-19-3.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-25-51.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-26-17.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-30-55.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-31-0.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-33-1.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-34-15.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-4-1.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-43-17.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-53-12.tmp
-rw-rw-r-- 1 nick nick 24 Okt 19 11:16 T2-60-55.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-0-10.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-12-51.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-1-43.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-15-43.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-17-51.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-3-10.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-51-35.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T3BackRef-55-35.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-10-38.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-35-40.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-43-38.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-51-40.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T5BackRef-38-5.tmp
-rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T5BackRef-40-5.tmp
-rw-rw-r-- 1 nick nick 16 Okt 19 11:16 T6BackRef-5-8.tmp
 *
 */

// y = 573855352
// xs 602009779,2127221679,3186459061,443532047,1234434947,1652736830,396228306,464118917,3981993340,3878862024,1730679522,3234011360,521197720,2635193875,2251292298,608281027,1468569780,2075860307,2880258779,999340005,1240438978,4293399624,4226635802,1031429862,2391120891,3533658526,3823422504,3983813271,4180778279,2403148863,2441456056,319558395,2338010591,196206622,1637393731,853158574,2704638588,2368357012,1703808356,451208700,2145291166,2741727812,3305809226,1748168268,415625277,3051905493,4257489502,1429077635,2438113590,3028543211,3993396297,2678430597,458920999,889121073,3577485087,1822568056,2222781147,1942400192,195608354,1460166215,2544813525,3231425778,2958837604,2710532969
void findYsolution(char *memstore) {
	if (memstore == NULL) {
		memstore = (char *) malloc(1738014720);
	}
	uint32_t y = 573855352;
	std::cout << "findYsolution: " << y << std::endl;
	T6BackRef *t6_data = (T6BackRef *) &memstore[0];

	// how to back propagate all?
	// read batch. Sort by all blocks. Then read batch related to sorted blocks.
	// loop
	//uint32_t t6_num;
	//readT6BlockFile(0,0,t6_data, t6_num);

	//for (uint32_t batch_id = 0; batch_id < BATCHES; batch_id++) {
		//std::cout << "Scanning T6 batch " << batch_id << std::endl;

	//	for (uint32_t block_id = 0; block_id < BATCHES; block_id++) {
	uint32_t batch_id = 5;
	uint32_t block_id = 8;
			uint32_t num_entries;
			readT6BlockFile(batch_id,block_id,t6_data, num_entries);
			std::cout << "Scanning T6 batch-block " << batch_id << "-" << block_id << " : " << num_entries << " entries" << std::endl;

			for (int i=0;i<num_entries;i++) {
				T6BackRef entry = t6_data[i];
				if (entry.y == y) {
					uint32_t prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
					uint32_t prev_idx_L = entry.prev_block_ref_L & 0x3FFFFFF;
					uint32_t prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
					uint32_t prev_idx_R = entry.prev_block_ref_R & 0x3FFFFFF;
					printf("T6BackRef Y FOUND! L:%u R:%u L_block_id:%u L_idx:%u R_block_id:%u R_idx:%u y:%u\n",
							entry.prev_block_ref_L, entry.prev_block_ref_R,
							prev_block_id_L, prev_idx_L,
							prev_block_id_R, prev_idx_R,
							entry.y);
					backPropagate(5,prev_block_id_L, batch_id, prev_idx_L );
					backPropagate(5,prev_block_id_R, batch_id, prev_idx_R );
				}
			}
	//	}
	//}

}

__global__
void gpu_set_t6_final_data_and_t4_tags_directly(const uint32_t N, T6BackRef *t6_data, BackRef *t5_data, T6FinalEntry *t6_final_data, uint32_t *t4_tags) {
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		T6BackRef entry = t6_data[i];
		uint64_t t6_prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
		uint64_t t6_prev_idx_L      = entry.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t6_prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
		uint64_t t6_prev_idx_R      = entry.prev_block_ref_R & 0x3FFFFFF;

		// now could back ref t5...
		BackRef t5_L, t5_R;
		uint32_t t5_address_L = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_L + t6_prev_idx_L;
		uint32_t t5_address_R = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_R + t6_prev_idx_R;
		t5_L = t5_data[t5_address_L];
		t5_R = t5_data[t5_address_R];
		uint64_t t5_L_prev_block_id_L = t5_L.prev_block_ref_L >> (32 - 6);
		uint64_t t5_L_prev_idx_L      = t5_L.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t5_L_prev_block_id_R = t5_L.prev_block_ref_R >> (32 - 6);
		uint64_t t5_L_prev_idx_R      = t5_L.prev_block_ref_R & 0x3FFFFFF;
		uint64_t t5_R_prev_block_id_L = t5_R.prev_block_ref_L >> (32 - 6);
		uint64_t t5_R_prev_idx_L      = t5_R.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t5_R_prev_block_id_R = t5_R.prev_block_ref_R >> (32 - 6);
		uint64_t t5_R_prev_idx_R      = t5_R.prev_block_ref_R & 0x3FFFFFF;

		T6FinalEntry final_entry = {};
		final_entry.refL = t5_L_prev_block_id_L + (t5_L_prev_block_id_R << 6) + (t6_prev_block_id_L << 12);
		final_entry.refR = t5_R_prev_block_id_L + (t5_R_prev_block_id_R << 6) + (t6_prev_block_id_R << 12);
		//std::cout << "T6 Final set: [" << t5_L_prev_block_id_L << " | " << t5_L_prev_block_id_R << "] - " << t6_prev_block_id_L << std::endl;
		//std::cout << "              [" << t5_R_prev_block_id_L << " | " << t5_R_prev_block_id_R << "] - " << t6_prev_block_id_R << std::endl;
		final_entry.y = entry.y;
		t6_final_data[i] = final_entry;



		// directly set t4 tags
		if (true) { // w/ this is 571ms, without is 440ms. Max optimization is 8 seconds over 64 batches.
			uint32_t value;
			uint64_t file_batch_id, file_block_id, file_idx;
			uint64_t address;
			uint32_t bits_to_set;
			file_batch_id = t5_L_prev_block_id_L; file_block_id = t6_prev_block_id_L; file_idx = t5_L_prev_idx_L;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);

			file_batch_id = t5_L_prev_block_id_R; file_block_id = t6_prev_block_id_L; file_idx = t5_L_prev_idx_R;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);

			file_batch_id = t5_R_prev_block_id_L; file_block_id = t6_prev_block_id_R; file_idx = t5_R_prev_idx_L;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);

			file_batch_id = t5_R_prev_block_id_R; file_block_id = t6_prev_block_id_R; file_idx = t5_R_prev_idx_R;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);
		}

	}
}

__global__
void gpu_backref_t5_tag(const uint32_t N, T6BackRef *t6_data, BackRef *t5_data, T6FinalEntry *t6_final_data, uint32_t *t5_tags) {
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		T6BackRef entry = t6_data[i];
		uint64_t t6_prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
		uint64_t t6_prev_idx_L      = entry.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t6_prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
		uint64_t t6_prev_idx_R      = entry.prev_block_ref_R & 0x3FFFFFF;

		// now could back ref t5...
		BackRef t5_L, t5_R;
		uint32_t t5_address_L = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_L + t6_prev_idx_L;
		uint32_t t5_address_R = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_R + t6_prev_idx_R;
		t5_L = t5_data[t5_address_L];
		t5_R = t5_data[t5_address_R];
		uint64_t t5_L_prev_block_id_L = t5_L.prev_block_ref_L >> (32 - 6);
		uint64_t t5_L_prev_idx_L      = t5_L.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t5_L_prev_block_id_R = t5_L.prev_block_ref_R >> (32 - 6);
		uint64_t t5_L_prev_idx_R      = t5_L.prev_block_ref_R & 0x3FFFFFF;
		uint64_t t5_R_prev_block_id_L = t5_R.prev_block_ref_L >> (32 - 6);
		uint64_t t5_R_prev_idx_L      = t5_R.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t5_R_prev_block_id_R = t5_R.prev_block_ref_R >> (32 - 6);
		uint64_t t5_R_prev_idx_R      = t5_R.prev_block_ref_R & 0x3FFFFFF;

		// tag addresses that were used here...

		uint32_t bits_to_set;
		bits_to_set = 1 << (t5_address_L % 32);
		atomicOr(&t5_tags[t5_address_L / 32], bits_to_set);

		bits_to_set = 1 << (t5_address_R % 32);
		atomicOr(&t5_tags[t5_address_R / 32], bits_to_set);


		T6FinalEntry final_entry = {};
		final_entry.refL = t5_L_prev_block_id_L + (t5_L_prev_block_id_R << 6) + (t6_prev_block_id_L << 12);
		final_entry.refR = t5_R_prev_block_id_L + (t5_R_prev_block_id_R << 6) + (t6_prev_block_id_R << 12);
		//std::cout << "T6 Final set: [" << t5_L_prev_block_id_L << " | " << t5_L_prev_block_id_R << "] - " << t6_prev_block_id_L << std::endl;
		//std::cout << "              [" << t5_R_prev_block_id_L << " | " << t5_R_prev_block_id_R << "] - " << t6_prev_block_id_R << std::endl;
		final_entry.y = entry.y;
		t6_final_data[i] = final_entry;
	}
}

// t6's map to t4's, t5's map to t3's
__global__
void gpu_backref_t4_tag(const uint32_t N, BackRef *t4_data, T3BaseRef *t3_data, T4FinalEntry *t4_final_data, uint32_t *t4_tags) {
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		BackRef entry = t4_data[i];
		uint64_t t4_prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
		uint64_t t4_prev_idx_L      = entry.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t4_prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
		uint64_t t4_prev_idx_R      = entry.prev_block_ref_R & 0x3FFFFFF;

		// now could back ref t5...
		T3BaseRef t3_L, t3_R;
		uint32_t t3_address_L = HOST_MAX_BLOCK_ENTRIES * t4_prev_block_id_L + t4_prev_idx_L;
		uint32_t t3_address_R = HOST_MAX_BLOCK_ENTRIES * t4_prev_block_id_R + t4_prev_idx_R;
		t3_L = t3_data[t3_address_L];
		t3_R = t3_data[t3_address_R];

		T4FinalEntry finalEntry;
		finalEntry.Lx1 = t3_L.Lx1;
		finalEntry.Lx2 = t3_L.Lx2;
		finalEntry.Lx3 = t3_L.Lx3;
		finalEntry.Lx4 = t3_L.Lx4;
		finalEntry.Lx5 = t3_R.Lx1;
		finalEntry.Lx6 = t3_R.Lx2;
		finalEntry.Lx7 = t3_R.Lx3;
		finalEntry.Lx8 = t3_R.Lx4;

		t4_final_data[i] = finalEntry;
	}
}

__global__
void gpu_backref_t4_lxlists(const uint32_t N, BackRef *t4_data, T3BaseRef *t3_data, uint32_t *t4_lx_list, uint32_t *t4_tags) {
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		BackRef entry = t4_data[i];
		uint64_t t4_prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
		uint64_t t4_prev_idx_L      = entry.prev_block_ref_L & 0x3FFFFFF;
		uint64_t t4_prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
		uint64_t t4_prev_idx_R      = entry.prev_block_ref_R & 0x3FFFFFF;

		// now could back ref t5...
		T3BaseRef t3_L, t3_R;
		uint32_t t3_address_L = HOST_MAX_BLOCK_ENTRIES * t4_prev_block_id_L + t4_prev_idx_L;
		uint32_t t3_address_R = HOST_MAX_BLOCK_ENTRIES * t4_prev_block_id_R + t4_prev_idx_R;
		t3_L = t3_data[t3_address_L];
		t3_R = t3_data[t3_address_R];

		uint32_t base_address = i*8;
		t4_lx_list[base_address+0] = t3_L.Lx1;
		t4_lx_list[base_address+1] = t3_L.Lx2;
		t4_lx_list[base_address+2] = t3_L.Lx3;
		t4_lx_list[base_address+3] = t3_L.Lx4;
		t4_lx_list[base_address+4] = t3_R.Lx1;
		t4_lx_list[base_address+5] = t3_R.Lx2;
		t4_lx_list[base_address+6] = t3_R.Lx3;
		t4_lx_list[base_address+7] = t3_R.Lx4;

	}
}

__global__
void gpu_t5_tag_to_t4(const uint32_t N, const uint32_t t5_block_id, BackRef *t5_data, uint32_t *t5_tags, uint32_t *t4_tags) {
	uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < N) {
		uint32_t t5_address = i;
		uint32_t bits_to_set = 1 << (t5_address % 32);
		uint32_t has_set = t5_tags[t5_address / 32] & bits_to_set;
		if (has_set > 0) {
			BackRef t5_entry = t5_data[t5_address];
			uint64_t t5_L_prev_block_id_L = t5_entry.prev_block_ref_L >> (32 - 6);
			uint64_t t5_L_prev_idx_L      = t5_entry.prev_block_ref_L & 0x3FFFFFF;
			uint64_t t5_L_prev_block_id_R = t5_entry.prev_block_ref_R >> (32 - 6);
			uint64_t t5_L_prev_idx_R      = t5_entry.prev_block_ref_R & 0x3FFFFFF;

			uint64_t file_batch_id, file_block_id, file_idx;
			uint64_t address;
			uint32_t bits_to_set;

			file_batch_id = t5_L_prev_block_id_L; file_block_id = t5_block_id; file_idx = t5_L_prev_idx_L;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);

			file_batch_id = t5_L_prev_block_id_R; file_block_id = t5_block_id; file_idx = t5_L_prev_idx_R;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			atomicOr(&t4_tags[address / 32], bits_to_set);
		}
	}
}

void tagPreviousTable(uint32_t t5_block_id, BackRef *t5_data, uint32_t num_entries, uint32_t *t5_tags, uint32_t *t4_tags) {
	// we have to read all T2 entries and merge into T3 table that then contains 4 Lx entries.
	//std::cout << " doing table block " << t5_block_id << std::endl;
	for (int i=0;i<num_entries;i++) {
		uint32_t t5_address = i;
		uint32_t bits_to_set = 1 << (i % 32);
		uint32_t has_set = t5_tags[i / 32] & bits_to_set;
		if (has_set > 0) {
			//std::cout << "WAS SET: t5 block_id: " << t5_block_id << " entry i: " << i << std::endl;
			BackRef t5_entry = t5_data[t5_address];
			uint64_t t5_L_prev_block_id_L = t5_entry.prev_block_ref_L >> (32 - 6);
			uint64_t t5_L_prev_idx_L      = t5_entry.prev_block_ref_L & 0x3FFFFFF;
			uint64_t t5_L_prev_block_id_R = t5_entry.prev_block_ref_R >> (32 - 6);
			uint64_t t5_L_prev_idx_R      = t5_entry.prev_block_ref_R & 0x3FFFFFF;

			uint64_t file_batch_id, file_block_id, file_idx;
			uint64_t address;
			uint32_t bits_to_set;
			file_batch_id = t5_L_prev_block_id_L; file_block_id = t5_block_id; file_idx = t5_L_prev_idx_L;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			address = address / 32;
			//uint32_t has_set = t4_tags[address] & bits_to_set;
			//if (has_set == 0) printf("error did not set first time some address mistake\n");
			t4_tags[address] |= bits_to_set;

			file_batch_id = t5_L_prev_block_id_R; file_block_id = t5_block_id; file_idx = t5_L_prev_idx_R;
			address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
			bits_to_set = 1 << (address % 32);
			address = address / 32;
			//has_set = t4_tags[address] & bits_to_set;
			//if (has_set == 0) printf("error did not set first time some address mistake\n");

			t4_tags[address] |= bits_to_set;

			num_set_t4 += 2;
		}
	}
	//std::cout << " done table block " << t5_block_id << std::endl;
}

void createT6FinalEntries_oldbenchmarks(char *memstore) {
	// 2) T6 must propagate down to T4 and tag all used entries, and then update T6 references to include T4.
		// 3) T6 reads one block at a time for each batch, small memory print
		//       - then T5 one whole batch, since each block references 0..BATCHES
		//       - T4 tag list can be set (booleans)
		//       - update T6 data to include y, 6,6,6 and 6,6,6 references
	const uint64_t T4_TAG_MEM_BYTES_NEEDED = (HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES * BATCHES)) * sizeof(uint32_t)) / 32;
	const uint64_t T5_TAG_MEM_BYTES_NEEDED = T4_TAG_MEM_BYTES_NEEDED; // (HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES)) * sizeof(uint32_t)) / 32;
	const uint64_t T6_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * sizeof(T6BackRef);
	const uint64_t T6_FINAL_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * sizeof(T6FinalEntry);
	const uint64_t T5_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES)) * sizeof(BackRef);
	const uint64_t TOTAL_MEM_BYTES_NEEDED = T4_TAG_MEM_BYTES_NEEDED + T5_TAG_MEM_BYTES_NEEDED + T6_MEM_BYTES_NEEDED + T6_FINAL_MEM_BYTES_NEEDED + T5_MEM_BYTES_NEEDED;

	T6BackRef *device_t6_data;
	BackRef *device_t5_data;
	T6FinalEntry *device_t6_final_data;
	uint32_t *device_t4_tags;
	uint32_t *device_t5_tags;

	if (memstore==NULL) {
		std::cout << "Allocating memory bytes: " << TOTAL_MEM_BYTES_NEEDED << std::endl;
		//memstore = (char *) malloc(TOTAL_MEM_BYTES_NEEDED);
		CUDA_CHECK_RETURN(cudaMallocHost((void**)&memstore, TOTAL_MEM_BYTES_NEEDED)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();
		std::cout << "    host mem allocated..." << std::endl;
		CUDA_CHECK_RETURN(cudaMalloc(&device_t6_data, T6_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMalloc(&device_t5_data, T5_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMalloc(&device_t6_final_data, T6_FINAL_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMalloc(&device_t4_tags, T4_TAG_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMalloc(&device_t5_tags, T5_TAG_MEM_BYTES_NEEDED));
		// clear bits...
		CUDA_CHECK_RETURN(cudaMemset(device_t4_tags, 0, T4_TAG_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMemset(device_t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED));

		std::cout << "    gpu mem allocated..." << std::endl;

		if (memstore == NULL) {
			exit (1);
		}
	}

	// TODO: THIS IS SUPER SLOW ON HOST CPU! but it only needs 5GB so could load into GPU and set it all there...

	uint64_t NEXT_MEM_BYTES_START = 0;

	const uint64_t T5_DATA_START = NEXT_MEM_BYTES_START;
	BackRef *t5_data = (BackRef *) &memstore[T5_DATA_START];
	uint32_t t5_num_entries[BATCHES];
	NEXT_MEM_BYTES_START += T5_MEM_BYTES_NEEDED;

	const uint64_t T6_DATA_START = NEXT_MEM_BYTES_START;
	T6BackRef *t6_data = (T6BackRef *) &memstore[T6_DATA_START];
	NEXT_MEM_BYTES_START += T6_MEM_BYTES_NEEDED;

	const uint64_t T6_FINAL_DATA_START = NEXT_MEM_BYTES_START;
	T6FinalEntry *t6_final_data = (T6FinalEntry *) &memstore[T6_FINAL_DATA_START];
	NEXT_MEM_BYTES_START += T6_FINAL_MEM_BYTES_NEEDED;

	uint32_t *t4_tags = (uint32_t *) &memstore[NEXT_MEM_BYTES_START]; // needs HOST_MAX_BLOCK_ENTRIES * 64 * 64 bytes
	// will reference this as if file, like memstore[batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + block_id*HOST_MAX_BLOCK_ENTRIES]
	memset(t4_tags, 0, T4_TAG_MEM_BYTES_NEEDED);
	NEXT_MEM_BYTES_START += T4_TAG_MEM_BYTES_NEEDED;

	uint32_t *t5_tags = (uint32_t *) &memstore[NEXT_MEM_BYTES_START]; // needs HOST_MAX_BLOCK_ENTRIES * 64 * 64 bytes
	// will reference this as if file, like memstore[batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + block_id*HOST_MAX_BLOCK_ENTRIES]
	memset(t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED);
	NEXT_MEM_BYTES_START += T5_TAG_MEM_BYTES_NEEDED;

	using milli = std::chrono::milliseconds;

	std::cout << "Starting..." << std::endl;
	uint64_t total_t4_tagged = 0;
	num_set_t4 = 0;
	num_same_addresses = 0;
	num_set_t5 = 0;

	const int doCPUmethod = 0;
	const int doGPUmethod = 2; // 1 is single shot setting, 2 is 2-phase setting
	/*
*******method 1**************
All compute loop time: 30760 ms
*********************
 Tagged T4 entries: 3531979836 should be 114437654 out of max 4563402752
*********************
Total time: 33862 ms

*******method 2**************
All compute loop time: 28907 ms
*********************
All compute loop time: 28753 ms
*********************
 Tagged T4 entries: 3531979836 should be 114437654 out of max 4563402752
*********************
Total time: 32010 ms

*/
	int blockSize = 256;


	auto compute_loop_start = std::chrono::high_resolution_clock::now();
	//for (uint32_t t6_batch_id = 5; t6_batch_id < 6; t6_batch_id++) {
	for (uint32_t t6_batch_id = 0; t6_batch_id < BATCHES; t6_batch_id++) {
		auto batch_start = std::chrono::high_resolution_clock::now();
		if (doCPUmethod > 0) {
			memset(t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED);
		} else {
			CUDA_CHECK_RETURN(cudaMemset(device_t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED));
		}
		for (uint64_t t5_block_id = 0; t5_block_id < BATCHES; t5_block_id++) {
			readBackRefBlockFile(5, t5_block_id, t6_batch_id,
					&t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],
					t5_num_entries[t5_block_id]);
			//std::cout << "Loading T5 batch-block " << t5_block_id << "-" << t6_batch_id << " : " << t5_num_entries[t5_block_id] << " entries" << std::endl;
			if (doGPUmethod > 0)
				CUDA_CHECK_RETURN(cudaMemcpy(&device_t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],&t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],t5_num_entries[t5_block_id]*sizeof(BackRef),cudaMemcpyHostToDevice));
		}
		if (doGPUmethod > 0)
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		// TODO: we need to make sure we are getting the correct values/tags set
		// find the file for the single value y and follow that y back to see if we are doing it right....

		//for (uint32_t t6_block_id = 8; t6_block_id < 9; t6_block_id++) { //  BATCHES; t6_block_id++) {
		for (uint32_t t6_block_id = 0; t6_block_id < BATCHES; t6_block_id++) {
			uint32_t t6_num_entries;
			readT6BlockFile(t6_batch_id,t6_block_id,t6_data, t6_num_entries);
			if (doGPUmethod > 0) {
				CUDA_CHECK_RETURN(cudaMemcpy(device_t6_data, t6_data,t6_num_entries*sizeof(T6BackRef),cudaMemcpyHostToDevice));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				//std::cout << "Scanning T6 batch-block " << t6_batch_id << "-" << t6_block_id << " : " << t6_num_entries << " entries" << std::endl;
				int numBlocks = (t6_num_entries + blockSize - 1) / (blockSize);
				if (doGPUmethod == 1)
					gpu_set_t6_final_data_and_t4_tags_directly<<<numBlocks,blockSize>>>(t6_num_entries,device_t6_data, device_t5_data, device_t6_final_data, device_t4_tags);
				else
					gpu_backref_t5_tag<<<numBlocks,blockSize>>>(t6_num_entries,device_t6_data, device_t5_data, device_t6_final_data, device_t5_tags);

				CUDA_CHECK_RETURN(cudaDeviceSynchronize());

				// now write back results to hostmem
				CUDA_CHECK_RETURN(cudaMemcpy(t6_final_data, device_t6_final_data,t6_num_entries*sizeof(T6FinalEntry),cudaMemcpyDeviceToHost));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			}
			if (doCPUmethod > 0)
			for (int i=0;i<t6_num_entries;i++) {
				T6BackRef entry = t6_data[i];
			//if (entry.y == 573855352) {
				uint64_t t6_prev_block_id_L = entry.prev_block_ref_L >> (32 - 6);
				uint64_t t6_prev_idx_L      = entry.prev_block_ref_L & 0x3FFFFFF;
				uint64_t t6_prev_block_id_R = entry.prev_block_ref_R >> (32 - 6);
				uint64_t t6_prev_idx_R      = entry.prev_block_ref_R & 0x3FFFFFF;

				// now could back ref t5...
				BackRef t5_L, t5_R;
				uint32_t t5_address_L = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_L + t6_prev_idx_L;
				uint32_t t5_address_R = HOST_MAX_BLOCK_ENTRIES * t6_prev_block_id_R + t6_prev_idx_R;
				t5_L = t5_data[t5_address_L];
				t5_R = t5_data[t5_address_R];
				uint64_t t5_L_prev_block_id_L = t5_L.prev_block_ref_L >> (32 - 6);
				uint64_t t5_L_prev_idx_L      = t5_L.prev_block_ref_L & 0x3FFFFFF;
				uint64_t t5_L_prev_block_id_R = t5_L.prev_block_ref_R >> (32 - 6);
				uint64_t t5_L_prev_idx_R      = t5_L.prev_block_ref_R & 0x3FFFFFF;
				uint64_t t5_R_prev_block_id_L = t5_R.prev_block_ref_L >> (32 - 6);
				uint64_t t5_R_prev_idx_L      = t5_R.prev_block_ref_L & 0x3FFFFFF;
				uint64_t t5_R_prev_block_id_R = t5_R.prev_block_ref_R >> (32 - 6);
				uint64_t t5_R_prev_idx_R      = t5_R.prev_block_ref_R & 0x3FFFFFF;

				// tag addresses that were used here...
				if (doCPUmethod == 2) {
					uint32_t bits_to_set;
					bits_to_set = 1 << (t5_address_L % 32);
					uint32_t value = t5_tags[t5_address_L / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t5_tags[t5_address_L / 32] |= bits_to_set;

					bits_to_set = 1 << (t5_address_R % 32);
					value = t5_tags[t5_address_R / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t5_tags[t5_address_R / 32] |= bits_to_set;

					num_set_t5 += 2;
				}

				T6FinalEntry final_entry = {};
				final_entry.refL = t5_L_prev_block_id_L + (t5_L_prev_block_id_R << 6) + (t6_prev_block_id_L << 12);
				final_entry.refR = t5_R_prev_block_id_L + (t5_R_prev_block_id_R << 6) + (t6_prev_block_id_R << 12);
				//std::cout << "T6 Final set: [" << t5_L_prev_block_id_L << " | " << t5_L_prev_block_id_R << "] - " << t6_prev_block_id_L << std::endl;
				//std::cout << "              [" << t5_R_prev_block_id_L << " | " << t5_R_prev_block_id_R << "] - " << t6_prev_block_id_R << std::endl;
				final_entry.y = entry.y;
				t6_final_data[i] = final_entry;



				// directly set t4 tags
				if (doCPUmethod == 1) {
					uint32_t value;
					uint64_t file_batch_id, file_block_id, file_idx;
					uint64_t address;
					uint32_t bits_to_set;
					file_batch_id = t5_L_prev_block_id_L; file_block_id = t6_prev_block_id_L; file_idx = t5_L_prev_idx_L;
					address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
					bits_to_set = 1 << (address % 32);
					value = t4_tags[address / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t4_tags[address / 32] |= bits_to_set;

					file_batch_id = t5_L_prev_block_id_R; file_block_id = t6_prev_block_id_L; file_idx = t5_L_prev_idx_R;
					address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
					bits_to_set = 1 << (address % 32);
					value = t4_tags[address / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t4_tags[address / 32] |= bits_to_set;

					file_batch_id = t5_R_prev_block_id_L; file_block_id = t6_prev_block_id_R; file_idx = t5_R_prev_idx_L;
					address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
					bits_to_set = 1 << (address % 32);
					value = t4_tags[address / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t4_tags[address / 32] |= bits_to_set;

					file_batch_id = t5_R_prev_block_id_R; file_block_id = t6_prev_block_id_R; file_idx = t5_R_prev_idx_R;
					address = file_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + file_block_id*HOST_MAX_BLOCK_ENTRIES + file_idx;
					bits_to_set = 1 << (address % 32);
					value = t4_tags[address / 32] & bits_to_set;
					if (value > 1) { num_same_addresses++; }
					t4_tags[address / 32] |= bits_to_set;

					num_set_t4 += 4;
				}

				// just for benchmarks
				// CPU:
				// 10239 ms with / 32 per batch, writing to t4 tags directly
				//  5057 ms with / 32 per batch but writing to t5 then scan t5 tags and write to t4 tags
				//
				//  1800ms without writing to tags, but fetch t5 data and setting t6 backrefs. -> still 1.9 minutes
				//  1588ms with T6 writing tags for t5 instead of fetching t5, doesn't seem to save much huh?
				//   1479ms without reading t5 at all -- so almost no gain (although to be fair t5 was cached reads).

				// Bladebit is 25s phase 2
				// read t5+t6 data only is 232ms, lowest bound, 15s min.
				// read and transfer to gpu is 324ms - total 20s
				// gpu setting data is 350ms...hallejuya
				//     - 26 seconds total but without tags written
				//     - 28.8 seconds writting back final data T6
				//     - 41 seconds w/ tags written.
				//     even settings tags is 640ms hot god damn I love gpu, vs 6500ms = x10!
				//     but can this be improved so less random writes?
				//     total time is 41s

				//backPropagate(5,prev_block_id_L, batch_id, prev_idx_L );
				//backPropagate(5,prev_block_id_R, batch_id, prev_idx_R );
				//printf("%u %u %u %u\n", t5_L_prev_block_id_L, t5_L_prev_block_id_R, t5_R_prev_block_id_L, t5_R_prev_block_id_R);

			}
			//}// entry.y

			//writeT6FinalBlockFile(t6_batch_id,t6_block_id,t6_data,t6_num_entries);

		}

		// 2067ms w/o any tagging
		// 3865ms w/ tagging but not tagging t4
		// 6299ms w tag on 5 and t4 tags all set
		// 10954ms tagging 4 directly (skipping 5)
		if (doCPUmethod == 2) {
			for (uint64_t t5_block_id = 0; t5_block_id < BATCHES; t5_block_id++) {
				uint32_t num_entries = t5_num_entries[t5_block_id];
				//std::cout << "Doing previous table tag for t6_batch_id: " << t6_batch_id << std::endl;
				tagPreviousTable(t5_block_id,
						&t5_data[HOST_MAX_BLOCK_ENTRIES * t5_block_id], t5_num_entries[t5_block_id],
						&t5_tags[(HOST_MAX_BLOCK_ENTRIES * t5_block_id) / 32], // note /32 since 32 bits
						t4_tags);
			}
		}

		if (doGPUmethod == 2) {
			for (uint64_t t5_block_id = 0; t5_block_id < BATCHES; t5_block_id++) {
				uint32_t num_entries = t5_num_entries[t5_block_id];
				uint32_t t5_address = HOST_MAX_BLOCK_ENTRIES * t5_block_id;
				int numBlocks = (num_entries + blockSize - 1) / (blockSize);
				gpu_t5_tag_to_t4<<<numBlocks,blockSize>>>(num_entries, t5_block_id,
						&device_t5_data[t5_address],
						&device_t5_tags[t5_address/32], device_t4_tags);
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}
		}

		auto batch_end = std::chrono::high_resolution_clock::now();
		//std::cout << "*********************" << std::endl;
		std::cout << "*** Batch " << t6_batch_id << " time: " << std::chrono::duration_cast<milli>(batch_end - batch_start).count() << " ms ***\n";
		//std::cout << "*********************" << std::endl;
	}
	if (doGPUmethod > 0) {
		// technically don't need to do this if stays in device memory...just for verfication purposes.
		CUDA_CHECK_RETURN(cudaMemcpy(t4_tags, device_t4_tags,T4_TAG_MEM_BYTES_NEEDED,cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}
	auto compute_loop_end = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "All compute loop time: " << std::chrono::duration_cast<milli>(compute_loop_end - compute_loop_start).count() << " ms\n";
	std::cout << "*********************" << std::endl;



	/*std::cout << "setting tags..." << std::endl;
		for (uint32_t t4_batch_id = 0; t4_batch_id < BATCHES; t4_batch_id++) {
			//std::cout << "setting batch " << t4_batch_id << std::endl;
			for (uint64_t t4_block_id = 0; t4_block_id < BATCHES; t4_block_id++) {
				//for (uint64_t i=0;i<1;i++) {
				for (uint64_t i=0;i<HOST_MAX_BLOCK_ENTRIES;i++) {
					uint64_t address = t4_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + t4_block_id*HOST_MAX_BLOCK_ENTRIES + i;
					uint32_t bits_to_set = 1 << (address % 32);
					address = address / 32;
					t4_tags[address] |= bits_to_set;
					num_set_t4++;
				}
			}
		}*/

	std::cout << "Counting tags..." << std::endl;
	for (uint32_t t4_batch_id = 0; t4_batch_id < BATCHES; t4_batch_id++) {
		//std::cout << "Counting batch " << t4_batch_id << std::endl;
		for (uint64_t t4_block_id = 0; t4_block_id < BATCHES; t4_block_id++) {
			for (uint64_t i=0;i<HOST_MAX_BLOCK_ENTRIES;i++) {
				uint64_t address = t4_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + t4_block_id*HOST_MAX_BLOCK_ENTRIES + i;
				uint32_t bits_to_set = 1 << (address % 32);
				address = address / 32;
				uint32_t has_set = t4_tags[address] & bits_to_set;
				if (has_set > 0) {
					total_t4_tagged++;
					//std::cout << " Tagged entry t4 batch_id: " << t4_batch_id << " block:" << t4_block_id << std::endl;
				};
			}
		}
		//std::cout << "partial result: " << total_t4_tagged << std::endl;
	}
	std::cout << " Num set t5: " << num_set_t5 << std::endl;
	std::cout << " Num set t4: " << num_set_t4 << std::endl;
	std::cout << " Num same addresses: " << num_same_addresses << std::endl;
	std::cout << " Tagged T4 entries: " << total_t4_tagged << " should be 114437654 out of max 4563402752" << std::endl;

	std::cout << " -rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-10-38.tmp" << std::endl
			  << " -rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-35-40.tmp" << std::endl
			  << " -rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-43-38.tmp" << std::endl
			  << " -rw-rw-r-- 1 nick nick 12 Okt 19 11:16 T4BackRef-51-40.tmp" << std::endl;
}


__global__
void gpu_chacha8_xs_to_kbcs(const uint32_t N,
		const __restrict__ uint32_t *input,
		uint32_t *xs, uint32_t *kbcs)
{
	uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	int index = blockIdx.x * blockDim.x + threadIdx.x; //  + x_start/16;

	if (index < N) {
		uint32_t x = xs[index];
		uint32_t x_group = x / 16;
		uint32_t x_selection = x % 16;
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

		uint32_t result_x;
		if (x_selection == 0) result_x = x0;
		if (x_selection == 1) result_x = x1;
		if (x_selection == 2) result_x = x2;
		if (x_selection == 3) result_x = x3;
		if (x_selection == 4) result_x = x4;
		if (x_selection == 5) result_x = x5;
		if (x_selection == 6) result_x = x6;
		if (x_selection == 7) result_x = x7;
		if (x_selection == 8) result_x = x8;
		if (x_selection == 9) result_x = x9;
		if (x_selection == 10) result_x = x10;
		if (x_selection == 11) result_x = x11;
		if (x_selection == 12) result_x = x12;
		if (x_selection == 13) result_x = x13;
		if (x_selection == 14) result_x = x14;
		if (x_selection == 15) result_x = x15;
		uint64_t y = (((uint64_t) result_x) << 6) + (x >> 26);
		uint32_t kbc_bucket_id = uint32_t (y / kBC);
		//printf("x: %u  y:%llu  kbc:%u\n", x, y, kbc_bucket_id);
		kbcs[index] = kbc_bucket_id;
	}
}


__global__
void showSorted(const uint32_t N, uint32_t *list) {
	for (int i=0;i<N;i++) {
		printf("%u ", list[i]);
	}
	printf("\n");
}

void createT6FinalEntriesGPU(char *memstore) {
	// 2) T6 must propagate down to T4 and tag all used entries, and then update T6 references to include T4.
		// 3) T6 reads one block at a time for each batch, small memory print
		//       - then T5 one whole batch, since each block references 0..BATCHES
		//       - T4 tag list can be set (booleans)
		//       - update T6 data to include y, 6,6,6 and 6,6,6 references
	const uint64_t T4_TAG_MEM_BYTES_NEEDED = (HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES * BATCHES)) * sizeof(uint32_t)) / 32;
	const uint64_t T5_TAG_MEM_BYTES_NEEDED = (HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES)) * sizeof(uint32_t)) / 32;
	const uint64_t T4AND6_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * sizeof(T6BackRef);// sizeof(BackRef)
	const uint64_t T4AND6_FINAL_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * sizeof(T4FinalEntry); // * sizeof(T6FinalEntry);
	const uint64_t T3AND5_MEM_BYTES_NEEDED = HOST_MAX_BLOCK_ENTRIES * ((uint64_t) (BATCHES)) * sizeof(T3BaseRef); // * sizeof(BackRef);

	const uint64_t TOTAL_MEM_BYTES_NEEDED = T4_TAG_MEM_BYTES_NEEDED + T5_TAG_MEM_BYTES_NEEDED + T4AND6_MEM_BYTES_NEEDED + T4AND6_FINAL_MEM_BYTES_NEEDED + T3AND5_MEM_BYTES_NEEDED;

	bool verify_results = true;

	T6BackRef *device_t6_data;
	BackRef *device_t4_data;

	T3BaseRef *device_t3_baseref_data;
	BackRef *device_t5_data;


	T4FinalEntry *device_t4_final_data;
	uint32_t *device_t4_lx_list;
	uint32_t *kbcs;
	T6FinalEntry *device_t6_final_data;
	uint32_t *device_t4_tags;
	uint32_t *device_t5_tags;

	if (memstore==NULL) {
		std::cout << "Allocating memory bytes: " << TOTAL_MEM_BYTES_NEEDED << std::endl;
		//memstore = (char *) malloc(TOTAL_MEM_BYTES_NEEDED);
		CUDA_CHECK_RETURN(cudaMallocHost((void**)&memstore, TOTAL_MEM_BYTES_NEEDED)); // = new F2_Result_Pair[HOST_F2_RESULTS_SPACE]();
		std::cout << "    host mem allocated..." << std::endl;

		CUDA_CHECK_RETURN(cudaMalloc(&device_t6_data, T4AND6_MEM_BYTES_NEEDED));
		device_t4_data = (BackRef *) device_t6_data;

		CUDA_CHECK_RETURN(cudaMalloc(&device_t5_data, T3AND5_MEM_BYTES_NEEDED));
		device_t3_baseref_data = (T3BaseRef *) device_t5_data;

		CUDA_CHECK_RETURN(cudaMalloc(&device_t6_final_data, T4AND6_FINAL_MEM_BYTES_NEEDED));
		device_t4_final_data = (T4FinalEntry *) device_t6_final_data; // shared when t6 is done with it
		device_t4_lx_list = (uint32_t *) device_t4_final_data;
		CUDA_CHECK_RETURN(cudaMalloc(&kbcs, T4AND6_FINAL_MEM_BYTES_NEEDED));

		CUDA_CHECK_RETURN(cudaMalloc(&device_t4_tags, T4_TAG_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMalloc(&device_t5_tags, T5_TAG_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMemset(device_t4_tags, 0, T4_TAG_MEM_BYTES_NEEDED));
		CUDA_CHECK_RETURN(cudaMemset(device_t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED));

		std::cout << "    gpu mem allocated..." << std::endl;

		if (memstore == NULL) {
			exit (1);
		}
	}

	// TODO: THIS IS SUPER SLOW ON HOST CPU! but it only needs 5GB so could load into GPU and set it all there...

	uint64_t NEXT_MEM_BYTES_START = 0;

	const uint64_t T5_DATA_START = NEXT_MEM_BYTES_START;
	BackRef *t5_data = (BackRef *) &memstore[T5_DATA_START];
	T3BaseRef *t3_baseref_data = (T3BaseRef *) t5_data;
	uint32_t t5_num_entries[BATCHES];
	NEXT_MEM_BYTES_START += T3AND5_MEM_BYTES_NEEDED;

	const uint64_t T6_DATA_START = NEXT_MEM_BYTES_START;
	T6BackRef *t6_data = (T6BackRef *) &memstore[T6_DATA_START];
	BackRef *t4_data = (BackRef *) t6_data;
	NEXT_MEM_BYTES_START += T4AND6_MEM_BYTES_NEEDED;

	const uint64_t T6_FINAL_DATA_START = NEXT_MEM_BYTES_START;
	T6FinalEntry *t6_final_data = (T6FinalEntry *) &memstore[T6_FINAL_DATA_START];
	T4FinalEntry *t4_final_data = (T4FinalEntry *) t6_final_data;
	uint32_t *t4_lx_list = (uint32_t *) t6_final_data;
	NEXT_MEM_BYTES_START += T4AND6_FINAL_MEM_BYTES_NEEDED;

	uint32_t *t4_tags = (uint32_t *) &memstore[NEXT_MEM_BYTES_START]; // needs HOST_MAX_BLOCK_ENTRIES * 64 * 64 bytes
	// will reference this as if file, like memstore[batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + block_id*HOST_MAX_BLOCK_ENTRIES]
	memset(t4_tags, 0, T4_TAG_MEM_BYTES_NEEDED);
	NEXT_MEM_BYTES_START += T4_TAG_MEM_BYTES_NEEDED;

	uint32_t *t5_tags = (uint32_t *) &memstore[NEXT_MEM_BYTES_START]; // needs HOST_MAX_BLOCK_ENTRIES * 64 * 64 bytes
	memset(t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED);
	NEXT_MEM_BYTES_START += T5_TAG_MEM_BYTES_NEEDED;

	using milli = std::chrono::milliseconds;

	std::cout << "Starting..." << std::endl;
	uint64_t total_t4_tagged = 0;
	num_set_t4 = 0;
	num_same_addresses = 0;
	num_set_t5 = 0;

	int blockSize = 256;

	auto compute_loop_start = std::chrono::high_resolution_clock::now();
	//for (uint32_t t6_batch_id = 5; t6_batch_id < 6; t6_batch_id++) {
	for (uint32_t t6_batch_id = 0; t6_batch_id < BATCHES; t6_batch_id++) {
		auto batch_start = std::chrono::high_resolution_clock::now();
		CUDA_CHECK_RETURN(cudaMemset(device_t5_tags, 0, T5_TAG_MEM_BYTES_NEEDED));

		for (uint64_t t5_block_id = 0; t5_block_id < BATCHES; t5_block_id++) {
			readBackRefBlockFile(5, t5_block_id, t6_batch_id,
					&t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],
					t5_num_entries[t5_block_id]);
			std::cout << "Loading T5 batch-block " << t5_block_id << "-" << t6_batch_id << " : " << t5_num_entries[t5_block_id] << " entries" << std::endl;
			CUDA_CHECK_RETURN(cudaMemcpy(&device_t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],&t5_data[HOST_MAX_BLOCK_ENTRIES*t5_block_id],t5_num_entries[t5_block_id]*sizeof(BackRef),cudaMemcpyHostToDevice));
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		// TODO: we need to make sure we are getting the correct values/tags set
		// find the file for the single value y and follow that y back to see if we are doing it right....

		//for (uint32_t t6_block_id = 8; t6_block_id < 9; t6_block_id++) { //  BATCHES; t6_block_id++) {
		for (uint32_t t6_block_id = 0; t6_block_id < BATCHES; t6_block_id++) {
			uint32_t t6_num_entries;
			readT6BlockFile(t6_batch_id,t6_block_id,t6_data, t6_num_entries);
			CUDA_CHECK_RETURN(cudaMemcpy(device_t6_data, t6_data,t6_num_entries*sizeof(T6BackRef),cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			std::cout << "Scanning T6 batch-block " << t6_batch_id << "-" << t6_block_id << " : " << t6_num_entries << " entries to tag t5s" << std::endl;
			int numBlocks = (t6_num_entries + blockSize - 1) / (blockSize);
			auto tag_start = std::chrono::high_resolution_clock::now();
			gpu_backref_t5_tag<<<numBlocks,blockSize>>>(t6_num_entries,device_t6_data, device_t5_data, device_t6_final_data, device_t5_tags);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			auto tag_end = std::chrono::high_resolution_clock::now();
			std::cout << "*** gpu tag ms: " << std::chrono::duration_cast<milli>(tag_end - tag_start).count() << " ms ***\n";
			// now write back results to hostmem
			std::cout << "writing results to hostmem" << std::endl;

			CUDA_CHECK_RETURN(cudaMemcpy(t6_final_data, device_t6_final_data,t6_num_entries*sizeof(T6FinalEntry),cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		auto tag_start = std::chrono::high_resolution_clock::now();
		for (uint64_t t5_block_id = 0; t5_block_id < BATCHES; t5_block_id++) {
			std::cout << " gpu t5 tag to t4 t5_block_id:" << std::endl;
			uint32_t num_entries = t5_num_entries[t5_block_id];
			uint32_t t5_address = HOST_MAX_BLOCK_ENTRIES * t5_block_id;
			int numBlocks = (num_entries + blockSize - 1) / (blockSize);
			gpu_t5_tag_to_t4<<<numBlocks,blockSize>>>(num_entries, t5_block_id,
						&device_t5_data[t5_address],
						&device_t5_tags[t5_address/32], device_t4_tags);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		auto tag_end = std::chrono::high_resolution_clock::now();
		std::cout << "*** gpu tag ms: " << std::chrono::duration_cast<milli>(tag_end - tag_start).count() << " ms ***\n";
		auto batch_end = std::chrono::high_resolution_clock::now();
		//std::cout << "*********************" << std::endl;
		std::cout << "*** Batch " << t6_batch_id << " time: " << std::chrono::duration_cast<milli>(batch_end - batch_start).count() << " ms ***\n";
		//std::cout << "*********************" << std::endl;
	}

	auto compute_loop_end = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "All compute loop time: " << std::chrono::duration_cast<milli>(compute_loop_end - compute_loop_start).count() << " ms\n";
	std::cout << "*********************" << std::endl;
	auto t4_final_start = std::chrono::high_resolution_clock::now();
	// TODO: free gpu mem and setup t3 and t4 mem
	std::cout << "Doing T4->T3 tags" << std::endl;

	uint32_t t3_num_entries[BATCHES];

	return;

	for (uint32_t t4_batch_id = 0; t4_batch_id < BATCHES; t4_batch_id++) {
		auto batch_start = std::chrono::high_resolution_clock::now();
		std::cout << "Loading T3BaseRef [0-63]-batch " << t4_batch_id << std::endl;

		for (uint64_t t3_block_id = 0; t3_block_id < BATCHES; t3_block_id++) {
			readT3BaseRefBlockFile(t3_block_id, t4_batch_id,
					&t3_baseref_data[HOST_MAX_BLOCK_ENTRIES*t3_block_id],
					t3_num_entries[t3_block_id]);

			CUDA_CHECK_RETURN(cudaMemcpy(&device_t3_baseref_data[HOST_MAX_BLOCK_ENTRIES*t3_block_id],&t3_baseref_data[HOST_MAX_BLOCK_ENTRIES*t3_block_id],
					t3_num_entries[t3_block_id]*sizeof(T3BaseRef), // note T3BaseRef
					cudaMemcpyHostToDevice));
		}
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		// now we have all t3 entries in block row for back referencing from t4 blocks.
		// the t4 blocks now just need to fetch the t3 entries and get the Lx1,Lx2,Lx3,Lx4 * 2 = 8 Lx entries.

		for (uint32_t t4_block_id = 0; t4_block_id < BATCHES; t4_block_id++) {
			uint32_t t4_num_entries;
			readBackRefBlockFile(4, t4_batch_id,t4_block_id,t4_data, t4_num_entries);

			CUDA_CHECK_RETURN(cudaMemcpy(device_t4_data, t4_data,t4_num_entries*sizeof(BackRef),cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			//std::cout << "Scanning T6 batch-block " << t6_batch_id << "-" << t6_block_id << " : " << t6_num_entries << " entries" << std::endl;
			int numBlocks = (t4_num_entries + blockSize - 1) / (blockSize);
			// gpu_backref_t4_tag<<<numBlocks,blockSize>>>(t4_num_entries,device_t4_data, device_t3_baseref_data, device_t4_final_data, device_t4_tags);
			gpu_backref_t4_lxlists<<<numBlocks,blockSize>>>(t4_num_entries,device_t4_data, device_t3_baseref_data, device_t4_lx_list, device_t4_tags);
			gpu_chacha8_xs_to_kbcs<<<numBlocks,blockSize>>>(t4_num_entries*8, chacha_input, device_t4_lx_list, kbcs);
			// wrap raw pointer with a device_ptr
			thrust::device_ptr<uint32_t> device_t4_lx_list_ptr(device_t4_lx_list);
			thrust::sort(device_t4_lx_list_ptr, device_t4_lx_list_ptr + t4_num_entries*8);        // modify your sort line
			showSorted<<<1,1>>>(30,device_t4_lx_list);

			thrust::device_ptr<uint32_t> device_kbcs_ptr(kbcs);
			thrust::sort(device_kbcs_ptr, device_kbcs_ptr + t4_num_entries*8);        // modify your sort line
			showSorted<<<1,1>>>(30,kbcs);

			//thrust::sort(device_t4_lx_list_ptr.begin(), device_t4_lx_list_ptr.end() + t4_num_entries*4);
			//uint32_t new_end = thrust::unique(device_t4_lx_list_ptr, device_t4_lx_list_ptr + t4_num_entries*4);
			//std::cout << "Thrust sorted " << (t4_num_entries*4) << " down to " << new_end << std::endl;

			 // T4 final time: 61754 ms (57808ms without backref t4 tag)
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			// now write back results to hostmem
			// erm...t4_num_entries can change, ay? Since output will be pruned somewhat.
			CUDA_CHECK_RETURN(cudaMemcpy(t4_final_data, device_t4_final_data,t4_num_entries*sizeof(T4FinalEntry),cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			// todo: get t4_final_data into a unique sorted list and compressed...of course could do this in GPU mem
		}
	}

	auto t4_final_end = std::chrono::high_resolution_clock::now();
	std::cout << "*********************" << std::endl;
	std::cout << "T4 final time: " << std::chrono::duration_cast<milli>(t4_final_end - t4_final_start).count() << " ms\n";
	std::cout << "*********************" << std::endl;



	if (verify_results) {
		// technically don't need to copy t4_tags if stays in device memory...just for verification purposes.
		CUDA_CHECK_RETURN(cudaMemcpy(t4_tags, device_t4_tags,T4_TAG_MEM_BYTES_NEEDED,cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		std::cout << "Counting tags..." << std::endl;
		for (uint32_t t4_batch_id = 0; t4_batch_id < BATCHES; t4_batch_id++) {
			//std::cout << "Counting batch " << t4_batch_id << std::endl;
			for (uint64_t t4_block_id = 0; t4_block_id < BATCHES; t4_block_id++) {
				for (uint64_t i=0;i<HOST_MAX_BLOCK_ENTRIES;i++) {
					uint64_t address = t4_batch_id*HOST_MAX_BLOCK_ENTRIES*BATCHES + t4_block_id*HOST_MAX_BLOCK_ENTRIES + i;
					uint32_t bits_to_set = 1 << (address % 32);
					address = address / 32;
					uint32_t has_set = t4_tags[address] & bits_to_set;
					if (has_set > 0) {
						total_t4_tagged++;
						//std::cout << " Tagged entry t4 batch_id: " << t4_batch_id << " block:" << t4_block_id << std::endl;
					};
				}
			}
			//std::cout << "partial result: " << total_t4_tagged << std::endl;
		}
		std::cout << " Tagged T4 entries: " << total_t4_tagged << " should be 114437654 out of max 4563402752" << std::endl;
	}
}


void doPhase2Pruning() {
	char *memstore;

	/*if (true) {
		// test xs'...
		uint32_t *xs;
		uint32_t *kbcs;
		CUDA_CHECK_RETURN(cudaMallocManaged(&xs, 256));
		CUDA_CHECK_RETURN(cudaMallocManaged(&kbcs, 256));
		for (int i=0;i<256;i++) {
			xs[i] = i;
		}
		std::cout << "Doing chacha single xs" << std::endl;
		gpu_chacha8_xs_to_kbcs<<<1,256>>>(256, chacha_input, xs, kbcs);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		for (int i=0;i<256;i++) {
			std::cout << " kbc " << i << " = " << kbcs[i] << std::endl;
		}

	}*/

	if (true) {
		using milli = std::chrono::milliseconds;
		auto total_start = std::chrono::high_resolution_clock::now();
		createT6FinalEntriesGPU(memstore);
		auto total_end = std::chrono::high_resolution_clock::now();
		std::cout << "*********************" << std::endl;
		std::cout << "Total time: " << std::chrono::duration_cast<milli>(total_end - total_start).count() << " ms\n";
		std::cout << "*********************" << std::endl;
	}

	//batch_id:30 block_id: 55dx:6169
	//findYsolution(memstore);

	//std::cout << "Phase 2 Pruning" << std::endl;
	//for (uint32_t batch_id = 0; batch_id < 1; batch_id++) {
	//	readPruneToT2(batch_id,memstore);
		// TODO: now that we have all data in mem for a batch, test whether getting the y will get the actual lx pairs!
	//}
	std::cout << "Done doPhase2Pruning." << std::endl;
}


#endif /* PHASE2_HPP_ */
