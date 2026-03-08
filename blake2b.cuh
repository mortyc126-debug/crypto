#pragma once
#include <stdint.h>

// Blake2b constants
#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES   64

// IV
__device__ __constant__ uint64_t BLAKE2B_IV[8] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

// Sigma permutation table
__device__ __constant__ uint8_t SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#define G(r,i,a,b,c,d,m)  \
    a = a + b + m[SIGMA[r][2*i+0]]; \
    d = ROTR64(d ^ a, 32); \
    c = c + d; \
    b = ROTR64(b ^ c, 24); \
    a = a + b + m[SIGMA[r][2*i+1]]; \
    d = ROTR64(d ^ a, 16); \
    c = c + d; \
    b = ROTR64(b ^ c, 63);

#define ROUND(r,v,m) \
    G(r,0,v[0],v[4],v[ 8],v[12],m); \
    G(r,1,v[1],v[5],v[ 9],v[13],m); \
    G(r,2,v[2],v[6],v[10],v[14],m); \
    G(r,3,v[3],v[7],v[11],v[15],m); \
    G(r,4,v[0],v[5],v[10],v[15],m); \
    G(r,5,v[1],v[6],v[11],v[12],m); \
    G(r,6,v[2],v[7],v[ 8],v[13],m); \
    G(r,7,v[3],v[4],v[ 9],v[14],m);

// Compress one 128-byte block
__device__ __forceinline__ void blake2b_compress(
    uint64_t h[8], const uint64_t m[16],
    uint64_t t0, uint64_t t1, bool last)
{
    uint64_t v[16];
    v[ 0]=h[0]; v[ 1]=h[1]; v[ 2]=h[2]; v[ 3]=h[3];
    v[ 4]=h[4]; v[ 5]=h[5]; v[ 6]=h[6]; v[ 7]=h[7];
    v[ 8]=0x6A09E667F3BCC908ULL;
    v[ 9]=0xBB67AE8584CAA73BULL;
    v[10]=0x3C6EF372FE94F82BULL;
    v[11]=0xA54FF53A5F1D36F1ULL;
    v[12]=0x510E527FADE682D1ULL ^ t0;
    v[13]=0x9B05688C2B3E6C1FULL ^ t1;
    v[14]=0x1F83D9ABFB41BD6BULL ^ (last ? 0xFFFFFFFFFFFFFFFFULL : 0ULL);
    v[15]=0x5BE0CD19137E2179ULL;

    ROUND( 0,v,m); ROUND( 1,v,m); ROUND( 2,v,m); ROUND( 3,v,m);
    ROUND( 4,v,m); ROUND( 5,v,m); ROUND( 6,v,m); ROUND( 7,v,m);
    ROUND( 8,v,m); ROUND( 9,v,m); ROUND(10,v,m); ROUND(11,v,m);

    for(int i=0;i<8;i++) h[i] ^= v[i] ^ v[i+8];
}

// Hash arbitrary input -> outlen bytes (<=64), result in out[]
__device__ __forceinline__ void blake2b_hash(
    const uint8_t* in, uint32_t inlen,
    uint8_t* out, uint32_t outlen)
{
    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | outlen);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;

    uint64_t m[16];
    uint32_t offset = 0;
    uint64_t counter = 0;

    while(inlen - offset > 128) {
        counter += 128;
        const uint8_t* p = in + offset;
        for(int i=0;i<16;i++){
            m[i] = 0;
            for(int j=0;j<8;j++)
                m[i] |= ((uint64_t)p[i*8+j]) << (j*8);
        }
        blake2b_compress(h, m, counter, 0, false);
        offset += 128;
    }

    uint8_t buf[128] = {};
    uint32_t rem = inlen - offset;
    for(uint32_t i=0;i<rem;i++) buf[i] = in[offset+i];
    counter += rem;
    for(int i=0;i<16;i++){
        m[i] = 0;
        for(int j=0;j<8;j++)
            m[i] |= ((uint64_t)buf[i*8+j]) << (j*8);
    }
    blake2b_compress(h, m, counter, 0, true);

    for(uint32_t i=0;i<outlen;i++)
        out[i] = (h[i/8] >> ((i%8)*8)) & 0xFF;
}

// ============================================================
// FIX v0.7: nonce must be passed as big-endian bytes to match
// Ergo network / pool verification.
//
// Pool reconstructs nonce as: en1_bytes || en2_bytes (big-endian)
// e.g. en1=bf65, en2=00000000 -> nonce bytes = bf 65 00 00 00 00 00 00
// = 0x0000bf6500000000 read as big-endian = stored as uint64 0x0000bf6500000000
//
// blake2b expects input bytes in natural (big-endian network) order.
// So nonce uint64 must be written as BIG-ENDIAN bytes into the message block.
//
// Previous bug: nonce was placed directly into m[4] as a little-endian uint64,
// causing bytes to be fed as: 00 00 00 00 65 bf 00 00 (reversed).
// Fix: bswap64(nonce) before storing in m[4].
// ============================================================

__device__ __forceinline__ uint64_t bswap64_inline(uint64_t x) {
    // byte-reverse a uint64
    return  ((x & 0x00000000000000FFULL) << 56) |
            ((x & 0x000000000000FF00ULL) << 40) |
            ((x & 0x0000000000FF0000ULL) << 24) |
            ((x & 0x00000000FF000000ULL) <<  8) |
            ((x & 0x000000FF00000000ULL) >>  8) |
            ((x & 0x0000FF0000000000ULL) >> 24) |
            ((x & 0x00FF000000000000ULL) >> 40) |
            ((x & 0xFF00000000000000ULL) >> 56);
}

// Hash exactly 32+8=40 bytes (blob_hash[32] || nonce[8]) -> 64 bytes
// FIX: nonce is stored as big-endian bytes (network order) to match pool verification.
__device__ __forceinline__ void blake2b_hash_40(
    const uint64_t* blob_words,  // 4 x uint64 (32 bytes, LE words = natural msg bytes)
    uint64_t nonce,               // nonce as integer; written BE into hash input
    uint64_t out[8])              // 64 bytes output
{
    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 64);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;

    // Single block: 40 bytes padded to 128 with zeros.
    // blob_words[0..3]: msg bytes already in correct LE-word order.
    // nonce: MUST be byte-swapped so it enters blake2b as big-endian bytes.
    uint64_t m[16] = {
        blob_words[0], blob_words[1], blob_words[2], blob_words[3],
        bswap64_inline(nonce),  // FIX: was just `nonce` (wrong LE byte order)
        0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    blake2b_compress(h, m, 40, 0, true);
    for(int i=0;i<8;i++) out[i]=h[i];
}

// Hash exactly 31*32=992 bytes (sum of 32 DAG elements) -> 32 bytes output
__device__ __forceinline__ void blake2b_hash_sum(
    const uint8_t* sum_bytes, uint32_t sum_len,
    uint64_t out[4])
{
    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;

    uint64_t m[16];
    uint32_t offset = 0;
    uint64_t counter = 0;

    while(sum_len - offset > 128) {
        counter += 128;
        const uint8_t* p = sum_bytes + offset;
        for(int i=0;i<16;i++){
            m[i]=0; for(int j=0;j<8;j++) m[i]|=((uint64_t)p[i*8+j])<<(j*8);
        }
        blake2b_compress(h, m, counter, 0, false);
        offset += 128;
    }
    uint8_t buf[128] = {};
    uint32_t rem = sum_len - offset;
    for(uint32_t i=0;i<rem;i++) buf[i]=sum_bytes[offset+i];
    counter += rem;
    for(int i=0;i<16;i++){
        m[i]=0; for(int j=0;j<8;j++) m[i]|=((uint64_t)buf[i*8+j])<<(j*8);
    }
    blake2b_compress(h, m, counter, 0, true);
    for(int i=0;i<4;i++) out[i]=h[i];
}
