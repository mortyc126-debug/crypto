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
    const uint8_t* p = dag + (uint64_t)idx * 32 + 1;
    #pragma unroll
    for(int j=0;j<31;j++) elem[j] = p[j];
}

__device__ __forceinline__ void add248(uint8_t* __restrict__ a, const uint8_t* __restrict__ b) {
    uint32_t carry = 0;
    #pragma unroll
    for(int i=30; i>=0; i--){
        uint32_t s = (uint32_t)a[i] + b[i] + carry;
        a[i] = (uint8_t)s;
        carry = s >> 8;
    }
}

__device__ __forceinline__ void blake2b_hash_31(const uint8_t* in, uint64_t out[4]) {
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
    for(int i=0; i<31; i++)
        m[i/8] |= ((uint64_t)in[i]) << ((i%8)*8);
    blake2b_compress(h, m, 31ULL, 0ULL, true);
    out[0]=h[0]; out[1]=h[1]; out[2]=h[2]; out[3]=h[3];
}

// Blake2b-256 of 32 bytes (4 uint64 LE words) -> 32 bytes (4 uint64)
// Used for genIndexes second hash step.
__device__ __forceinline__ void blake2b_256_32(const uint64_t in[4], uint64_t out[4]) {
    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;
    uint64_t m[16] = {in[0], in[1], in[2], in[3], 0,0,0,0,0,0,0,0,0,0,0,0};
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
    uint64_t seed[4];
    blake2b_hash_40(blob_words, nonce, seed);

    // Step 2: hash = Blake2b256(seed) -> 32 bytes [genIndexes internal hash]
    // Ergo reference: genIndexes(seed) first hashes the seed again.
    uint64_t hash[4];
    blake2b_256_32(seed, hash);

    // Step 3: Build 35-byte extended hash (32 bytes + wrap first 3 bytes)
    // hash[4] is LE uint64 words -> bytes in memory order.
    uint8_t extended[36];
    #pragma unroll
    for(int i=0; i<4; i++){
        #pragma unroll
        for(int j=0; j<8; j++)
            extended[i*8+j] = (uint8_t)(hash[i] >> (j*8));
    }
    extended[32] = extended[0];
    extended[33] = extended[1];
    extended[34] = extended[2];

    // Step 4: Generate 32 indices via sliding 4-byte big-endian window
    uint32_t idx[32];
    #pragma unroll
    for(int i=0; i<32; i++){
        uint32_t v = ((uint32_t)extended[i]   << 24) |
                     ((uint32_t)extended[i+1] << 16) |
                     ((uint32_t)extended[i+2] <<  8) |
                      (uint32_t)extended[i+3];
        idx[i] = (uint32_t)((uint64_t)v % N);
    }

    // Step 5: Sum 32 DAG elements (31-byte big-endian addition)
    uint8_t sum[31] = {};
    uint8_t elem[31];
    #pragma unroll 4
    for(int k=0; k<32; k++){
        load_dag_element(dag, idx[k], elem);
        add248(sum, elem);
    }

    // Step 6: Final hash = Blake2b256(sum[31])
    uint64_t final_hash[4];
    blake2b_hash_31(sum, final_hash);

    // Step 7: Convert to big-endian uint32 words for comparison
    uint32_t fh_be[8];
    #pragma unroll
    for(int i=0; i<4; i++){
        uint32_t lo32 = (uint32_t)(final_hash[i] & 0xFFFFFFFF);
        uint32_t hi32 = (uint32_t)(final_hash[i] >> 32);
        fh_be[i*2+0] = __byte_perm(lo32, 0, 0x0123);
        fh_be[i*2+1] = __byte_perm(hi32, 0, 0x0123);
    }

    // Step 8: fh_be < target ?
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
