#pragma once
#include <stdint.h>
#include "blake2b.cuh"

static const uint64_t AUTOLYKOS_N0      = 67108864ULL;
static const uint64_t AUTOLYKOS_MAX_N   = 2143944600ULL;
static const uint64_t AUTOLYKOS_START   = 614400ULL;
static const uint64_t AUTOLYKOS_PERIOD  = 51200ULL;

inline uint64_t autolykos_n(uint64_t height) {
    if(height < AUTOLYKOS_START) return AUTOLYKOS_N0;
    uint64_t epoch = (height - AUTOLYKOS_START) / AUTOLYKOS_PERIOD;
    uint64_t N = AUTOLYKOS_N0;
    for(uint64_t e = 0; e < epoch; e++) {
        N = (uint64_t)((double)N * 1.05);
        if(N >= AUTOLYKOS_MAX_N) return AUTOLYKOS_MAX_N;
    }
    return N;
}

__global__ void dag_gen_kernel(
    uint8_t* __restrict__ dag,
    uint64_t start_i,
    uint64_t height,
    uint64_t count)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if(tid >= count) return;
    uint64_t i = start_i + tid;

    uint64_t h[8];
    h[0] = 0x6A09E667F3BCC908ULL ^ (0x01010000ULL | 32);
    h[1] = 0xBB67AE8584CAA73BULL;
    h[2] = 0x3C6EF372FE94F82BULL;
    h[3] = 0xA54FF53A5F1D36F1ULL;
    h[4] = 0x510E527FADE682D1ULL;
    h[5] = 0x9B05688C2B3E6C1FULL;
    h[6] = 0x1F83D9ABFB41BD6BULL;
    h[7] = 0x5BE0CD19137E2179ULL;

    { uint64_t m[16]={i,height,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      blake2b_compress(h, m, 128ULL, 0ULL, false); }

    { uint64_t mz[16]={};
      #pragma unroll 1
      for(int blk=1; blk<=63; blk++)
          blake2b_compress(h, mz, (uint64_t)(blk+1)*128ULL, 0ULL, false);
      blake2b_compress(h, mz, 8208ULL, 0ULL, true); }

    uint8_t* out = dag + i * 32;
    out[0] = 0;
    #pragma unroll
    for(int b=1; b<32; b++)
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
    const uint64_t CHUNK   = 1ULL << 16;

    uint64_t total_chunks = (N + CHUNK - 1) / CHUNK;
    printf("[DAG] Generating %llu elements in %llu chunks of %llu...\n",
           (unsigned long long)N,
           (unsigned long long)total_chunks,
           (unsigned long long)CHUNK);
    fflush(stdout);

    for(uint64_t start=0; start<N; start+=CHUNK) {
        uint64_t count = (start+CHUNK > N) ? (N-start) : CHUNK;
        uint32_t grid  = (uint32_t)((count+THREADS-1)/THREADS);

        dag_gen_kernel<<<grid, THREADS>>>(*d_dag, start, height, count);

        err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("[DAG] LAUNCH ERROR @%llu: %s\n",
                   (unsigned long long)start, cudaGetErrorString(err));
            fflush(stdout);
            cudaFree(*d_dag); *d_dag=nullptr; return err;
        }

        err = cudaDeviceSynchronize();
        if(err != cudaSuccess) {
            printf("[DAG] SYNC ERROR @%llu (%.1f%%): %s\n",
                   (unsigned long long)start,
                   100.0*start/N,
                   cudaGetErrorString(err));
            fflush(stdout);
            cudaFree(*d_dag); *d_dag=nullptr; return err;
        }

        if((start % (1024*1024)) < CHUNK) {
            printf("[DAG] %.1f%%  (%llu / %llu)\n",
                   100.0*(start+count)/N,
                   (unsigned long long)(start+count),
                   (unsigned long long)N);
            fflush(stdout);
        }
    }

    printf("[DAG] 100.0%%  Generation complete.\n");
    fflush(stdout);
    return cudaSuccess;
}
