#ifndef SRC_CHACHA8_H_
#define SRC_CHACHA8_H_

#include <stdint.h>

struct chacha8_ctx {
    uint32_t input[16];
};

#ifdef __cplusplus
extern "C" {
#endif

// blake...
/*#define NICK_ROTR32(w,c) \
  (((w) >> (c)) | ((w) << (32 - (c))))

#define NICK_G(a,b,c,d,x,y) \
  state[a] = state[a] + state[b] + x; \
  state[d] = NICK_ROTR32(state[d] ^ state[a], 16); \
  state[c] = state[c] + state[d]; \
  state[b] = NICK_ROTR32(state[b] ^ state[c], 12); \
  state[a] = state[a] + state[b] + y; \
  state[d] = NICK_ROTR32(state[d] ^ state[a], 8); \
  state[c] = state[c] + state[d]; \
  state[b] = NICK_ROTR32(state[b] ^ state[c], 7); \

#define NICK_LOAD32(block,i) \
  ((uint32_t)(block[i+0]) << 0) | ((uint32_t)(block[i+1]) << 8) | ((uint32_t)(block[i+2]) << 16) | ((uint32_t)(block[i+3]) << 24)
// end blake*/

#define U32TO32_LITTLE(v) (v)
#define U8TO32_LITTLE(p) (*(const uint32_t *)(p))
#define U32TO8_LITTLE(p, v) (((uint32_t *)(p))[0] = U32TO32_LITTLE(v))
#define ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define ROTATE(v, c) (ROTL32(v, c))
#define XOR(v, w) ((v) ^ (w))
#define PLUS(v, w) ((v) + (w))
#define PLUSONE(v) (PLUS((v), 1))

#define QUARTERROUND(a, b, c, d) \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 16);   \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 12);   \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 8);    \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 7)

#define BYTESWAP32(x) \
	x = (x & 0x0000FFFF) << 16 | (x & 0xFFFF0000) >> 16; \
	x = (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8

void chacha8_keysetup_data(uint32_t *input, const uint8_t *k, uint32_t kbits, const uint8_t *iv);
void chacha8_keysetup(struct chacha8_ctx *x, const uint8_t *k, uint32_t kbits, const uint8_t *iv);
void chacha8_get_k32_keystream_data(const uint32_t *input, uint64_t pos, uint32_t n_blocks, uint32_t *c);
void chacha8_get_keystream_data(const uint32_t *input,uint64_t pos,uint32_t n_blocks,uint8_t *c);
void chacha8_get_keystream(
    const struct chacha8_ctx *x,
    uint64_t pos,
    uint32_t n_blocks,
    uint8_t *c);

#ifdef __cplusplus
}
#endif

#endif  // SRC_CHACHA8_H_
