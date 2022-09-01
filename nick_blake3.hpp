/*
 * nick_blake3.hpp
 *
 *  Created on: Oct 26, 2021
 *      Author: nick
 */

#ifndef NICK_BLAKE3_HPP_
#define NICK_BLAKE3_HPP_

#define CALC_Y_BUCKETED_KBC_ENTRY(entry, bucket_id) \
	(((uint64_t) bucket_id) * ((uint64_t) 15113) + (uint64_t) entry.y)

#define BSWAP32(i) (__byte_perm ((i), 0, 0x0123))

#define NICK_ROTR32(w,c) \
  (((w) >> (c)) | ((w) << (32 - (c))))

// rotate32 by 8 * c bits (1 byte)
#define NICK_ROTR32_BYTE8(w,c) __byte_perm (w, w, 0x3210 + 0x1111 * c);

// optimized for cuda instructions with rotate by multiples of 8 bites
#define NICK_G(a,b,c,d,x,y) \
  state[a] = state[a] + state[b] + x; \
  state[d] = NICK_ROTR32_BYTE8(state[d] ^ state[a], 2); \
  state[c] = state[c] + state[d]; \
  state[b] = NICK_ROTR32(state[b] ^ state[c], 12); \
  state[a] = state[a] + state[b] + y; \
  state[d] = NICK_ROTR32_BYTE8(state[d] ^ state[a], 1); \
  state[c] = state[c] + state[d]; \
  state[b] = NICK_ROTR32(state[b] ^ state[c], 7); \



__device__
void nick_blake3(const uint32_t* meta, int meta_len, const uint64_t y,
		uint64_t *y_result, uint8_t c_len, uint32_t *c_results) {
	uint32_t state[16];
	uint32_t block_words[16];// = {0};
	size_t input_len = 21;

    block_words[0] = BSWAP32(y >> 6);
	block_words[1] = BSWAP32(__funnelshift_l ( meta[0], y, 26));
	block_words[2] = BSWAP32(__funnelshift_l ( meta[1], meta[0], 26));
	if (meta_len == 2) {
		// [32][6-26][6-26][6-]
		block_words[3] = BSWAP32(meta[1] << 26);
		input_len = 13;
	}
	else if (meta_len == 3) {
		// [32][6-26][6-26][6-26][6-26][6-]
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(meta[2] << 26);
		input_len = 17;
	}
	else if (meta_len == 4) {
		// [32][6-26][6-26][6-26][6-26][6-26][6-]
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(meta[3] << 26);
		input_len = 21;
	}
	else if (meta_len == 6) {
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(__funnelshift_l ( meta[4], meta[3], 26));
		block_words[6] = BSWAP32(__funnelshift_l ( meta[5], meta[4], 26));
		block_words[7] = BSWAP32(meta[5] << 26);
		input_len = 29;
	}
	else if (meta_len == 8) {
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(__funnelshift_l ( meta[4], meta[3], 26));
		block_words[6] = BSWAP32(__funnelshift_l ( meta[5], meta[4], 26));
		block_words[7] = BSWAP32(__funnelshift_l ( meta[6], meta[5], 26));
		block_words[8] = BSWAP32(__funnelshift_l ( meta[7], meta[6], 26));
		block_words[9] = BSWAP32(meta[7] << 26);
		input_len = 37;
	}

	for (int i=meta_len+2;i<16;i++) block_words[i]=0;


	state[0] = 0x6A09E667UL;
	state[1] = 0xBB67AE85UL;
	state[2] = 0x3C6EF372UL;
	state[3] = 0xA54FF53AUL;
	state[4] = 0x510E527FUL;
	state[5] = 0x9B05688CUL;
	state[6] = 0x1F83D9ABUL;
	state[7] = 0x5BE0CD19UL;
	state[8] = 0x6A09E667UL;
	state[9] = 0xBB67AE85UL;
	state[10] = 0x3C6EF372UL;
	state[11] = 0xA54FF53AUL;
	state[12] = 0; // counter_low(0);
	state[13] = 0; // counter_high(0);
	state[14] = (uint32_t) input_len; // take;// (uint32_t)output.block_len;
	state[15] = (uint32_t) (1 | 2 | 8);// (output.flags | ROOT);

	NICK_G(0,4,8,12,block_words[0],block_words[1]);
	NICK_G(1,5,9,13,block_words[2],block_words[3]);
	NICK_G(2,6,10,14,block_words[4],block_words[5]);
	NICK_G(3,7,11,15,block_words[6],block_words[7]);
	NICK_G(0,5,10,15,block_words[8],block_words[9]);
	NICK_G(1,6,11,12,block_words[10],block_words[11]);
	NICK_G(2,7,8,13,block_words[12],block_words[13]);
	NICK_G(3,4,9,14,block_words[14],block_words[15]);
	NICK_G(0,4,8,12,block_words[2],block_words[6]);
	NICK_G(1,5,9,13,block_words[3],block_words[10]);
	NICK_G(2,6,10,14,block_words[7],block_words[0]);
	NICK_G(3,7,11,15,block_words[4],block_words[13]);
	NICK_G(0,5,10,15,block_words[1],block_words[11]);
	NICK_G(1,6,11,12,block_words[12],block_words[5]);
	NICK_G(2,7,8,13,block_words[9],block_words[14]);
	NICK_G(3,4,9,14,block_words[15],block_words[8]);
	NICK_G(0,4,8,12,block_words[3],block_words[4]);
	NICK_G(1,5,9,13,block_words[10],block_words[12]);
	NICK_G(2,6,10,14,block_words[13],block_words[2]);
	NICK_G(3,7,11,15,block_words[7],block_words[14]);
	NICK_G(0,5,10,15,block_words[6],block_words[5]);
	NICK_G(1,6,11,12,block_words[9],block_words[0]);
	NICK_G(2,7,8,13,block_words[11],block_words[15]);
	NICK_G(3,4,9,14,block_words[8],block_words[1]);
	NICK_G(0,4,8,12,block_words[10],block_words[7]);
	NICK_G(1,5,9,13,block_words[12],block_words[9]);
	NICK_G(2,6,10,14,block_words[14],block_words[3]);
	NICK_G(3,7,11,15,block_words[13],block_words[15]);
	NICK_G(0,5,10,15,block_words[4],block_words[0]);
	NICK_G(1,6,11,12,block_words[11],block_words[2]);
	NICK_G(2,7,8,13,block_words[5],block_words[8]);
	NICK_G(3,4,9,14,block_words[1],block_words[6]);
	NICK_G(0,4,8,12,block_words[12],block_words[13]);
	NICK_G(1,5,9,13,block_words[9],block_words[11]);
	NICK_G(2,6,10,14,block_words[15],block_words[10]);
	NICK_G(3,7,11,15,block_words[14],block_words[8]);
	NICK_G(0,5,10,15,block_words[7],block_words[2]);
	NICK_G(1,6,11,12,block_words[5],block_words[3]);
	NICK_G(2,7,8,13,block_words[0],block_words[1]);
	NICK_G(3,4,9,14,block_words[6],block_words[4]);
	NICK_G(0,4,8,12,block_words[9],block_words[14]);
	NICK_G(1,5,9,13,block_words[11],block_words[5]);
	NICK_G(2,6,10,14,block_words[8],block_words[12]);
	NICK_G(3,7,11,15,block_words[15],block_words[1]);
	NICK_G(0,5,10,15,block_words[13],block_words[3]);
	NICK_G(1,6,11,12,block_words[0],block_words[10]);
	NICK_G(2,7,8,13,block_words[2],block_words[6]);
	NICK_G(3,4,9,14,block_words[4],block_words[7]);
	NICK_G(0,4,8,12,block_words[11],block_words[15]);
	NICK_G(1,5,9,13,block_words[5],block_words[0]);
	NICK_G(2,6,10,14,block_words[1],block_words[9]);
	NICK_G(3,7,11,15,block_words[8],block_words[6]);
	NICK_G(0,5,10,15,block_words[14],block_words[10]);
	NICK_G(1,6,11,12,block_words[2],block_words[12]);
	NICK_G(2,7,8,13,block_words[3],block_words[4]);
	NICK_G(3,4,9,14,block_words[7],block_words[13]);


	uint32_t r0 = BSWAP32(state[0] ^ state[8]);
	uint32_t r1 = BSWAP32(state[1] ^ state[9]); // y_result is 38 bits of [a][6-]
	uint32_t r2 = BSWAP32(state[2] ^ state[10]);
	uint32_t r3 = BSWAP32(state[3] ^ state[11]);
	uint32_t r4 = BSWAP32(state[4] ^ state[12]);
	uint32_t r5 = BSWAP32(state[5] ^ state[13]);

	// MINOR OPTIMIZATION: on last table could just return top 32 bits instead of the 38 bits.
	uint64_t y_hi = __funnelshift_l ( r0, 0, 6); // shift 6 of top bits of r0 into y_hi
	uint32_t y_lo = __funnelshift_l ( r1, r0, 6);
	if (c_len > 0) {
		c_results[0] = __funnelshift_l ( r2, r1, 6);
		c_results[1] = __funnelshift_l ( r3, r2, 6);
	}
	if (c_len > 2) {
		c_results[2] = __funnelshift_l ( r4, r3, 6);
	}
	if (c_len > 3) {
		c_results[3] = __funnelshift_l ( r5, r4, 6);
	}

	(*y_result) = (y_hi << 32) + y_lo;

}

__device__
void nick_blake_k29(const uint32_t* meta, int meta_len, const uint64_t y,
		uint64_t *y_result, uint8_t c_len, uint32_t *c_results) {
	uint32_t state[16];
	uint32_t block_words[16];// = {0};
	size_t input_len = 21;

	block_words[0] = BSWAP32(y >> 6);
	block_words[1] = BSWAP32(__funnelshift_l ( meta[0], y, 26));
	block_words[2] = BSWAP32(__funnelshift_l ( meta[1], meta[0], 26));
	if (meta_len == 2) {
		// [32][6-26][6-26][6-]
		block_words[3] = BSWAP32(meta[1] << 26);
		input_len = 13;
	}
	else if (meta_len == 3) {
		// [32][6-26][6-26][6-26][6-26][6-]
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(meta[2] << 26);
		input_len = 17;
	}
	else if (meta_len == 4) {
		// [32][6-26][6-26][6-26][6-26][6-26][6-]
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(meta[3] << 26);
		input_len = 21;
	}
	else if (meta_len == 6) {
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(__funnelshift_l ( meta[4], meta[3], 26));
		block_words[6] = BSWAP32(__funnelshift_l ( meta[5], meta[4], 26));
		block_words[7] = BSWAP32(meta[5] << 26);
		input_len = 29;
	}
	else if (meta_len == 8) {
		block_words[3] = BSWAP32(__funnelshift_l ( meta[2], meta[1], 26));
		block_words[4] = BSWAP32(__funnelshift_l ( meta[3], meta[2], 26));
		block_words[5] = BSWAP32(__funnelshift_l ( meta[4], meta[3], 26));
		block_words[6] = BSWAP32(__funnelshift_l ( meta[5], meta[4], 26));
		block_words[7] = BSWAP32(__funnelshift_l ( meta[6], meta[5], 26));
		block_words[8] = BSWAP32(__funnelshift_l ( meta[7], meta[6], 26));
		block_words[9] = BSWAP32(meta[7] << 26);
		input_len = 37;
	}

	for (int i=meta_len+2;i<16;i++) block_words[i]=0;


	state[0] = 0x6A09E667UL;
	state[1] = 0xBB67AE85UL;
	state[2] = 0x3C6EF372UL;
	state[3] = 0xA54FF53AUL;
	state[4] = 0x510E527FUL;
	state[5] = 0x9B05688CUL;
	state[6] = 0x1F83D9ABUL;
	state[7] = 0x5BE0CD19UL;
	state[8] = 0x6A09E667UL;
	state[9] = 0xBB67AE85UL;
	state[10] = 0x3C6EF372UL;
	state[11] = 0xA54FF53AUL;
	state[12] = 0; // counter_low(0);
	state[13] = 0; // counter_high(0);
	state[14] = (uint32_t) input_len; // take;// (uint32_t)output.block_len;
	state[15] = (uint32_t) (1 | 2 | 8);// (output.flags | ROOT);

	NICK_G(0,4,8,12,block_words[0],block_words[1]);
	NICK_G(1,5,9,13,block_words[2],block_words[3]);
	NICK_G(2,6,10,14,block_words[4],block_words[5]);
	NICK_G(3,7,11,15,block_words[6],block_words[7]);
	NICK_G(0,5,10,15,block_words[8],block_words[9]);
	NICK_G(1,6,11,12,block_words[10],block_words[11]);
	NICK_G(2,7,8,13,block_words[12],block_words[13]);
	NICK_G(3,4,9,14,block_words[14],block_words[15]);
	NICK_G(0,4,8,12,block_words[2],block_words[6]);
	NICK_G(1,5,9,13,block_words[3],block_words[10]);
	NICK_G(2,6,10,14,block_words[7],block_words[0]);
	NICK_G(3,7,11,15,block_words[4],block_words[13]);
	NICK_G(0,5,10,15,block_words[1],block_words[11]);
	NICK_G(1,6,11,12,block_words[12],block_words[5]);
	NICK_G(2,7,8,13,block_words[9],block_words[14]);
	NICK_G(3,4,9,14,block_words[15],block_words[8]);
	NICK_G(0,4,8,12,block_words[3],block_words[4]);
	NICK_G(1,5,9,13,block_words[10],block_words[12]);
	NICK_G(2,6,10,14,block_words[13],block_words[2]);
	NICK_G(3,7,11,15,block_words[7],block_words[14]);
	NICK_G(0,5,10,15,block_words[6],block_words[5]);
	NICK_G(1,6,11,12,block_words[9],block_words[0]);
	NICK_G(2,7,8,13,block_words[11],block_words[15]);
	NICK_G(3,4,9,14,block_words[8],block_words[1]);
	NICK_G(0,4,8,12,block_words[10],block_words[7]);
	NICK_G(1,5,9,13,block_words[12],block_words[9]);
	NICK_G(2,6,10,14,block_words[14],block_words[3]);
	NICK_G(3,7,11,15,block_words[13],block_words[15]);
	NICK_G(0,5,10,15,block_words[4],block_words[0]);
	NICK_G(1,6,11,12,block_words[11],block_words[2]);
	NICK_G(2,7,8,13,block_words[5],block_words[8]);
	NICK_G(3,4,9,14,block_words[1],block_words[6]);
	NICK_G(0,4,8,12,block_words[12],block_words[13]);
	NICK_G(1,5,9,13,block_words[9],block_words[11]);
	NICK_G(2,6,10,14,block_words[15],block_words[10]);
	NICK_G(3,7,11,15,block_words[14],block_words[8]);
	NICK_G(0,5,10,15,block_words[7],block_words[2]);
	NICK_G(1,6,11,12,block_words[5],block_words[3]);
	NICK_G(2,7,8,13,block_words[0],block_words[1]);
	NICK_G(3,4,9,14,block_words[6],block_words[4]);
	NICK_G(0,4,8,12,block_words[9],block_words[14]);
	NICK_G(1,5,9,13,block_words[11],block_words[5]);
	NICK_G(2,6,10,14,block_words[8],block_words[12]);
	NICK_G(3,7,11,15,block_words[15],block_words[1]);
	NICK_G(0,5,10,15,block_words[13],block_words[3]);
	NICK_G(1,6,11,12,block_words[0],block_words[10]);
	NICK_G(2,7,8,13,block_words[2],block_words[6]);
	NICK_G(3,4,9,14,block_words[4],block_words[7]);
	NICK_G(0,4,8,12,block_words[11],block_words[15]);
	NICK_G(1,5,9,13,block_words[5],block_words[0]);
	NICK_G(2,6,10,14,block_words[1],block_words[9]);
	NICK_G(3,7,11,15,block_words[8],block_words[6]);
	NICK_G(0,5,10,15,block_words[14],block_words[10]);
	NICK_G(1,6,11,12,block_words[2],block_words[12]);
	NICK_G(2,7,8,13,block_words[3],block_words[4]);
	NICK_G(3,4,9,14,block_words[7],block_words[13]);

	uint32_t r0 = BSWAP32(state[0] ^ state[8]);
	uint32_t r1 = BSWAP32(state[1] ^ state[9]); // y_result is 38 bits of [a][6-]
	uint32_t r2 = BSWAP32(state[2] ^ state[10]);
	uint32_t r3 = BSWAP32(state[3] ^ state[11]);
	uint32_t r4 = BSWAP32(state[4] ^ state[12]);
	uint32_t r5 = BSWAP32(state[5] ^ state[13]);

	// MINOR OPTIMIZATION: on last table could just return top 32 bits instead of the 38 bits.
	uint64_t y_hi = __funnelshift_l ( r0, 0, 6); // shift 6 of top bits of r0 into y_hi
	uint32_t y_lo = __funnelshift_l ( r1, r0, 6);
	if (c_len > 0) {
		c_results[0] = __funnelshift_l ( r2, r1, 6);
		c_results[1] = __funnelshift_l ( r3, r2, 6);
	}
	if (c_len > 2) {
		c_results[2] = __funnelshift_l ( r4, r3, 6);
	}
	if (c_len > 3) {
		c_results[3] = __funnelshift_l ( r5, r4, 6);
	}

	(*y_result) = ((y_hi << 32) + y_lo) >> 3;

}



#endif /* NICK_BLAKE3_HPP_ */
