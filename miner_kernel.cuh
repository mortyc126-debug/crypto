#pragma once
#include <stdint.h>
#include "blake2b.cuh"

struct MineResult {
    uint64_t nonce;
    bool found;
    uint32_t fh_be[8];   // DEBUG: winning hash
};

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    return __byte_perm(x>>32, x, 0x0123) | ((uint64_t)__byte_perm(x>>32, x, 0x4567) << 32);
}

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

    // Step 1: seed = Blake2b_64(blob[32] || nonce_BE[8])
    // FIX v0.7: nonce is now hashed as big-endian bytes (see blake2b_hash_40)
    uint64_t seed64[8];
    blake2b_hash_40(blob_words, nonce, seed64);

    // Step 2: extendedHash - take high 32 bits of each bswap'd seed word
    uint32_t ext[9];
    #pragma unroll
    for(int i=0; i<8; i++)
        ext[i] = (uint32_t)(bswap64(seed64[i]) >> 32);
    ext[8] = ext[0];

    // Step 3: 32 indexes
    uint32_t idx[32];
    #pragma unroll
    for(int i=0; i<8; i++){
        uint32_t hi=ext[i], lo=ext[i+1];
        idx[i*4+0] = (uint32_t)((uint64_t)hi % N);
        idx[i*4+1] = (uint32_t)((((uint64_t)hi << 8)  | (lo >> 24)) % N);
        idx[i*4+2] = (uint32_t)((((uint64_t)hi << 16) | (lo >> 16)) % N);
        idx[i*4+3] = (uint32_t)((((uint64_t)hi << 24) | (lo >>  8)) % N);
    }

    // Step 4: sum 32 DAG elements (31-byte bignum addition)
    uint8_t sum[31] = {};
    uint8_t elem[31];
    #pragma unroll 4
    for(int k=0; k<32; k++){
        load_dag_element(dag, idx[k], elem);
        add248(sum, elem);
    }

    // Step 5: final hash
    uint64_t final_hash[4];
    blake2b_hash_31(sum, final_hash);

    // Step 6: convert to BE uint32 for comparison
    uint32_t fh_be[8];
    #pragma unroll
    for(int i=0; i<4; i++){
        uint32_t lo32 = (uint32_t)(final_hash[i] & 0xFFFFFFFF);
        uint32_t hi32 = (uint32_t)(final_hash[i] >> 32);
        fh_be[i*2+0] = __byte_perm(lo32, 0, 0x0123);
        fh_be[i*2+1] = __byte_perm(hi32, 0, 0x0123);
    }

    // Step 7: fh_be < target ?
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
