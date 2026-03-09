#pragma once
#include <stdint.h>
#include "blake2b.cuh"

// FIX v0.21: N formula updated to match ErgoStratumServer pool:
//   - iterationsNumber = floor((h - IncreaseStart) / IncreasePeriodForN) + 1  (+1 vs old code)
//   - Formula: N = N / 100 * 105  (integer division first, then multiply - matches BigInt pool code)
//   - AUTOLYKOS_MAX_N updated to pool value 2147387550
static const uint64_t AUTOLYKOS_N0      = 67108864ULL;
static const uint64_t AUTOLYKOS_MAX_N   = 2147387550ULL;  // pool: 2147387550
static const uint64_t AUTOLYKOS_START   = 614400ULL;
static const uint64_t AUTOLYKOS_PERIOD  = 51200ULL;

inline uint64_t autolykos_n(uint64_t height) {
    if(height < AUTOLYKOS_START) return AUTOLYKOS_N0;
    uint64_t epoch = (height - AUTOLYKOS_START) / AUTOLYKOS_PERIOD;
    uint64_t N = AUTOLYKOS_N0;
    uint64_t iters = epoch + 1;  // FIX v0.21: pool applies epoch+1 iterations
    for(uint64_t e = 0; e < iters; e++) {
        N = N / 100 * 105;  // FIX v0.21: match pool's BigInt: floor(N/100)*105
        if(N >= AUTOLYKOS_MAX_N) return AUTOLYKOS_MAX_N;
    }
    return N;
}

// FIX v0.21: DAG element formula corrected to match pool (ErgoStratumServer):
//   Input: i_4BE (4 bytes) || h_4BE (4 bytes) || M_8192 (8192 bytes) = 8200 bytes total
//   where M = [uint64_be(0), uint64_be(1), ..., uint64_be(1023)] (sequential, NOT zeros)
//   Output: blake2b-256(input).slice(1, 32) — drops byte[0], keeps 31 bytes
//   Stored as: out[0]=0, out[1..31]=the 31 element bytes
//
// Previous bug (v0.20 and before): used 8-byte i/h and M=zeros (8192 zero bytes).
// Pool uses 4-byte i/h and M=sequential uint64 BE values.
__global__ void dag_gen_kernel(
    uint8_t* __restrict__ dag,
    uint64_t start_i,
    uint64_t height,
    uint64_t count)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if(tid >= count) return;
    uint64_t i = start_i + tid;

    uint32_t i32 = (uint32_t)i;
    uint32_t h32 = (uint32_t)height;

    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;

    // Block 0 (bytes 0-127):
    //   bytes 0-3:   i as 4-byte BE -> packed LE = bswap32(i32)
    //   bytes 4-7:   h as 4-byte BE -> packed LE = bswap32(h32)
    //   bytes 8-127: M[0..14] as uint64 BE -> m[k] = bswap64(k-1) for k=1..15
    {
        uint64_t m[16];
        m[0] = (uint64_t)bswap32_inline(i32) | ((uint64_t)bswap32_inline(h32) << 32);
        #pragma unroll
        for(int k = 1; k <= 15; k++)
            m[k] = bswap64_inline((uint64_t)(k - 1));
        blake2b_compress(h, m, 128ULL, 0ULL, false);
    }

    // Blocks 1..63 (16 M words each):
    //   Block n starts at M word index 16n-1
    //   m[k] = bswap64(16n - 1 + k) for k=0..15
    #pragma unroll 1
    for(int blk = 1; blk <= 63; blk++) {
        uint64_t m[16];
        uint64_t base = (uint64_t)(16 * blk - 1);
        #pragma unroll
        for(int k = 0; k < 16; k++)
            m[k] = bswap64_inline(base + (uint64_t)k);
        blake2b_compress(h, m, (uint64_t)(blk + 1) * 128ULL, 0ULL, false);
    }

    // Block 64 (last): M[1023] + zeros, total bytes consumed = 8200
    {
        uint64_t m[16] = {};
        m[0] = bswap64_inline(1023ULL);
        blake2b_compress(h, m, 8200ULL, 0ULL, true);
    }

    // Store: out[0]=0 (discarded byte), out[1..31] = blake2b bytes[1..31]
    uint8_t* out = dag + i * 32;
    out[0] = 0;
    #pragma unroll
    for(int b = 1; b < 32; b++)
        out[b] = (uint8_t)((h[b/8] >> ((b%8)*8)) & 0xFF);
}

cudaError_t build_dag(uint8_t** d_dag, uint64_t height, uint64_t* out_N) {
    uint64_t N = autolykos_n(height);
    *out_N = N;

    size_t dag_bytes = (size_t)N * 32;
    double dag_gb = dag_bytes / 1073741824.0;
    printf("[DAG] height=%llu  N=%llu  size=%.3fGB\n",
           (unsigned long long)height, (unsigned long long)N, dag_gb);
    fflush(stdout);

    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    printf("[DAG] VRAM: free=%.2fGB / total=%.2fGB\n",
           free_b/1073741824.0, total_b/1073741824.0);
    fflush(stdout);

    if(dag_bytes > free_b) {
        printf("[DAG] ERROR: not enough VRAM (need %.2fGB, have %.2fGB free)\n",
               dag_gb, free_b/1073741824.0);
        fflush(stdout);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t err = cudaMalloc(d_dag, dag_bytes);
    if(err != cudaSuccess) {
        printf("[DAG] cudaMalloc FAILED: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        return err;
    }
    printf("[DAG] cudaMalloc OK (%.3fGB allocated)\n", dag_gb);
    fflush(stdout);

    const int      THREADS = 256;
    const uint64_t CHUNK   = 1ULL << 16;  // 65536 elements per kernel launch

    printf("[DAG] Generating %llu elements (async)...\n", (unsigned long long)N);
    fflush(stdout);

    for(uint64_t start=0; start<N; start+=CHUNK) {
        uint64_t count = (start+CHUNK > N) ? (N-start) : CHUNK;
        uint32_t grid  = (uint32_t)((count+THREADS-1)/THREADS);
        dag_gen_kernel<<<grid, THREADS>>>(*d_dag, start, height, count);
    }

    // Single synchronize after all chunks are queued
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("[DAG] ERROR: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(*d_dag); *d_dag=nullptr; return err;
    }
    printf("[DAG] Generation complete.\n");
    fflush(stdout);
    return cudaSuccess;
}
