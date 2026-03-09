#pragma once
#include <stdint.h>
#include "blake2b.cuh"

struct MineResult {
    uint64_t nonce;
    bool found;
    uint32_t fh_be[8];   // DEBUG: winning hash
};

__device__ __forceinline__ void load_dag_element(
    const uint8_t* __restrict__ dag,
    uint32_t idx,
    uint8_t* elem)
{
    const uint8_t* p = dag + (uint64_t)idx * 32;
    #pragma unroll
    for(int j=0;j<32;j++) elem[j] = p[j];
}

__device__ __forceinline__ void add256(uint8_t* __restrict__ a, const uint8_t* __restrict__ b) {
    uint32_t carry = 0;
    #pragma unroll
    for(int i=31; i>=0; i--){
        uint32_t s = (uint32_t)a[i] + b[i] + carry;
        a[i] = (uint8_t)s;
        carry = s >> 8;
    }
}

__device__ __forceinline__ void blake2b_hash_32(const uint8_t* in, uint64_t out[4]) {
    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;
    uint64_t m[16] = {};
    #pragma unroll
    for(int i=0; i<32; i++)
        m[i/8] |= ((uint64_t)in[i]) << ((i%8)*8);
    blake2b_compress(h, m, 32ULL, 0ULL, true);
    out[0]=h[0]; out[1]=h[1]; out[2]=h[2]; out[3]=h[3];
}

__global__ void mine_kernel(
    const uint8_t* __restrict__ dag,
    uint64_t N,
    const uint64_t* __restrict__ blob_words,
    uint64_t nonce_start,
    uint32_t batch_size,
    const uint32_t* __restrict__ target,
    MineResult* result)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= batch_size) return;
    if(result->found) return;

    uint64_t nonce = nonce_start + (uint64_t)tid;

    // Step 1: seed = Blake2b256(msg[32] || nonce_BE[8]) -> 32 bytes
    // coinbase = msg(32) || en1(2) || en2(6) = msg || nonce_BE (40 bytes)
    uint64_t seed[4];
    blake2b_hash_40(blob_words, nonce, seed);

    // Step 2: i_idx = toBigIntBE(seed[24:32]) % N
    // seed[24:32] = word seed[3] stored as LE; toBigIntBE = bswap64(seed[3])
    uint32_t i_idx = (uint32_t)(bswap64_inline(seed[3]) % N);

    // Step 3: Look up pre-computed DAG element at i_idx
    // e = dag[i_idx][1..31] (31 bytes, skipping the leading zero byte)
    const uint8_t* eptr = dag + (uint64_t)i_idx * 32;

    // Step 4: genIndexes seed = e[31 bytes] || coinbase[40 bytes] = 71 bytes
    // Hash these 71 bytes with blake2b-256 (single block, counter=71)
    //
    // Byte layout:
    //   bytes  0..30: e[1..31] from eptr
    //   bytes 31..62: msg bytes (blob_words as LE uint64)
    //   bytes 63..70: nonce as 8-byte BE
    //
    // Pack into 16 LE uint64 words for blake2b_compress
    uint64_t m_gi[16] = {};

    // e bytes: eptr[1..31] (skip eptr[0] which is always 0)
    #pragma unroll
    for(int j = 0; j < 31; j++)
        m_gi[j/8] |= ((uint64_t)eptr[j+1]) << ((j%8)*8);

    // msg bytes from blob_words (already LE uint64 words)
    #pragma unroll
    for(int j = 0; j < 32; j++) {
        uint8_t b = (uint8_t)((blob_words[j/8] >> ((j%8)*8)) & 0xFF);
        int pos = 31 + j;
        m_gi[pos/8] |= ((uint64_t)b) << ((pos%8)*8);
    }

    // nonce as 8-byte big-endian
    #pragma unroll
    for(int j = 0; j < 8; j++) {
        uint8_t b = (uint8_t)((nonce >> ((7-j)*8)) & 0xFF);
        int pos = 63 + j;
        m_gi[pos/8] |= ((uint64_t)b) << ((pos%8)*8);
    }

    // blake2b-256 of 71 bytes
    uint64_t hh[8];
    hh[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    hh[1] = 0xBB67AE8584CAA73BULL;
    hh[2] = 0x3C6EF372FE94F82BULL;
    hh[3] = 0xA54FF53A5F1D36F1ULL;
    hh[4] = 0x510E527FADE682D1ULL;
    hh[5] = 0x9B05688C2B3E6C1FULL;
    hh[6] = 0x1F83D9ABFB41BD6BULL;
    hh[7] = 0x5BE0CD19137E2179ULL;
    blake2b_compress(hh, m_gi, 71ULL, 0ULL, true);

    // genIndexes hash (32 bytes = first 4 words)
    uint64_t gihash[4] = {hh[0], hh[1], hh[2], hh[3]};

    // Step 5: Extended hash = gihash || gihash (64 bytes), but we only need first 35
    // extended[0..31] = gihash bytes in LE word order
    // extended[32..34] = first 3 bytes again (for wrap-around sliding window)
    uint8_t extended[36];
    #pragma unroll
    for(int i=0; i<4; i++){
        #pragma unroll
        for(int j=0; j<8; j++)
            extended[i*8+j] = (uint8_t)(gihash[i] >> (j*8));
    }
    extended[32] = extended[0];
    extended[33] = extended[1];
    extended[34] = extended[2];

    // Step 6: Generate 32 indices via sliding 4-byte big-endian window
    uint32_t idx[32];
    #pragma unroll
    for(int i=0; i<32; i++){
        uint32_t v = ((uint32_t)extended[i]   << 24) |
                     ((uint32_t)extended[i+1] << 16) |
                     ((uint32_t)extended[i+2] <<  8) |
                      (uint32_t)extended[i+3];
        idx[i] = (uint32_t)((uint64_t)v % N);
    }

    // Step 7: Sum 32 DAG elements (32-byte / 256-bit big-endian addition)
    uint8_t sum[32] = {};
    uint8_t elem[32];
    #pragma unroll 4
    for(int k=0; k<32; k++){
        load_dag_element(dag, idx[k], elem);
        add256(sum, elem);
    }

    // Step 8: Final hash = Blake2b256(sum[32])
    uint64_t final_hash[4];
    blake2b_hash_32(sum, final_hash);

    // Step 9: Convert to big-endian uint32 words for comparison
    uint32_t fh_be[8];
    #pragma unroll
    for(int i=0; i<4; i++){
        uint32_t lo32 = (uint32_t)(final_hash[i] & 0xFFFFFFFF);
        uint32_t hi32 = (uint32_t)(final_hash[i] >> 32);
        fh_be[i*2+0] = __byte_perm(lo32, 0, 0x0123);
        fh_be[i*2+1] = __byte_perm(hi32, 0, 0x0123);
    }

    // Step 10: fh_be < target ?
    bool less = false;
    #pragma unroll
    for(int i=0; i<8; i++){
        if(fh_be[i] < target[i]){ less = true;  break; }
        if(fh_be[i] > target[i]){ less = false; break; }
    }

    if(less){
        if(atomicCAS((int*)&result->found, 0, 1) == 0){
            result->nonce = nonce;
            #pragma unroll
            for(int i=0;i<8;i++) result->fh_be[i] = fh_be[i];
        }
    }
}
