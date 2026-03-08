// ErgoMiner v0.9
// Fix: nonce hashed as big-endian bytes to match pool/network verification
// Fix: increased GPU occupancy (BLOCKS 112->448) for ~4x hashrate improvement
#define _WIN32_WINNT 0x0600
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

#define LOG(...) do { printf(__VA_ARGS__); fflush(stdout); } while(0)
#define CUDA_CHECK(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess) LOG("[CUDA ERROR] %s:%d %s => %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_e)); \
} while(0)

// ===== JSON helpers =====
static std::vector<std::string> parse_params_array(const std::string& json) {
    std::vector<std::string> res;
    size_t pos=json.find("\"params\":[");
    if(pos==std::string::npos) return res;
    pos+=10;
    int depth=1; bool in_str=false;
    size_t elem_start=pos;
    for(size_t i=pos;i<json.size();i++){
        char c=json[i];
        if(c=='\\'&&in_str){i++;continue;}
        if(c=='"'){in_str=!in_str;continue;}
        if(in_str) continue;
        if(c=='['||c=='{'){depth++;continue;}
        if(c==']'||c=='}'){
            depth--;
            if(depth==0){
                std::string e=json.substr(elem_start,i-elem_start);
                size_t a=0,b=e.size();
                while(a<b&&(e[a]==' '||e[a]=='\t'||e[a]=='\n'||e[a]=='\r'))a++;
                while(b>a&&(e[b-1]==' '||e[b-1]=='\t'||e[b-1]=='\n'||e[b-1]=='\r'))b--;
                if(a<b&&e[a]=='"'&&e[b-1]=='"'){a++;b--;}
                res.push_back(e.substr(a,b-a));
                break;
            }
            continue;
        }
        if(c==','&&depth==1){
            std::string e=json.substr(elem_start,i-elem_start);
            size_t a=0,b=e.size();
            while(a<b&&(e[a]==' '||e[a]=='\t'||e[a]=='\n'||e[a]=='\r'))a++;
            while(b>a&&(e[b-1]==' '||e[b-1]=='\t'||e[b-1]=='\n'||e[b-1]=='\r'))b--;
            if(a<b&&e[a]=='"'&&e[b-1]=='"'){a++;b--;}
            res.push_back(e.substr(a,b-a));
            elem_start=i+1;
        }
    }
    return res;
}
static std::string json_get_str(const std::string& json,const std::string& key){
    std::string search="\""+key+"\":\"";
    size_t pos=json.find(search);
    if(pos==std::string::npos) return "";
    pos+=search.size();
    size_t end=pos;
    while(end<json.size()){if(json[end]=='\\'){end+=2;continue;}if(json[end]=='"')break;end++;}
    return json.substr(pos,end-pos);
}
static int json_get_id(const std::string& json){
    size_t p=json.find("\"id\":");
    if(p==std::string::npos) return -1;
    p+=5; while(p<json.size()&&json[p]==' ')p++;
    if(json[p]=='n') return -1;
    return atoi(json.c_str()+p);
}
static void hex2bytes(const char* hex,uint8_t* bytes,int len){
    for(int i=0;i<len;i++){unsigned int b=0;sscanf(hex+i*2,"%02x",&b);bytes[i]=(uint8_t)b;}
}
static std::string bytes2hex(const uint8_t* bytes,int len){
    std::string s; char buf[3];
    for(int i=0;i<len;i++){sprintf(buf,"%02x",bytes[i]);s+=buf;}
    return s;
}

#include "dag.cuh"
#include "miner_kernel.cuh"

static uint8_t*    d_dag        = nullptr;
static uint64_t    dag_N        = 0;
static MineResult* d_result     = nullptr;
static uint32_t*   d_target     = nullptr;
static uint64_t*   d_blob_words = nullptr;

// Debug kernel: verbose intermediate value dump to pinpoint CPU/GPU divergence
// Output layout (all uint64, 64 elements):
//   [0..7]   = seed64[0..7]       (blake2b_hash_40 output)
//   [8..15]  = ext[0..7] as u64   (upper 32 bits of bswap64(seed_word))
//   [16..19] = idx[0..3] as u64
//   [20..23] = DAG[idx[0]][0..3] as u64 (raw 32-byte elem, first 4 words)
//   [24..27] = DAG[idx[0]][4..7] as u64
//   [28..31] = sum[0..3] as u64 (31-byte sum, packed as LE)
//   [32..35] = final_hash[0..3] (blake2b-256 of sum)
//   [36..43] = fh_be[0..7] as u64
__global__ void debug_one_hash_kernel(
    const uint8_t* __restrict__ dag,
    uint64_t N,
    const uint64_t* __restrict__ blob_words,
    uint64_t nonce,
    uint64_t* out_debug)  // 64 x uint64
{
    if(threadIdx.x != 0 || blockIdx.x != 0) return;

    // Seed
    uint64_t seed64[8];
    blake2b_hash_40(blob_words, nonce, seed64);
    for(int i=0;i<8;i++) out_debug[i]=seed64[i];

    // ext
    uint32_t ext[9];
    for(int i=0;i<8;i++){
        uint64_t lo=(uint64_t)(uint32_t)seed64[i], hi=(uint64_t)(seed64[i]>>32);
        uint64_t swapped = (uint64_t)__byte_perm((uint32_t)hi,(uint32_t)lo,0x0123) |
                           ((uint64_t)__byte_perm((uint32_t)hi,(uint32_t)lo,0x4567)<<32);
        ext[i]=(uint32_t)(swapped>>32);
        out_debug[8+i]=(uint64_t)ext[i];
    }
    ext[8]=ext[0];

    // idx
    uint32_t idx[32];
    for(int i=0;i<8;i++){
        uint32_t hi=ext[i],lo=ext[i+1];
        idx[i*4+0]=(uint32_t)((uint64_t)hi%N);
        idx[i*4+1]=(uint32_t)((((uint64_t)hi<<8)|(lo>>24))%N);
        idx[i*4+2]=(uint32_t)((((uint64_t)hi<<16)|(lo>>16))%N);
        idx[i*4+3]=(uint32_t)((((uint64_t)hi<<24)|(lo>>8))%N);
    }
    for(int i=0;i<4;i++) out_debug[16+i]=(uint64_t)idx[i];

    // First DAG element raw (32 bytes = 4 uint64)
    {
        const uint8_t* p=dag+(uint64_t)idx[0]*32;
        for(int i=0;i<4;i++){
            uint64_t v=0;
            for(int j=0;j<8;j++) v|=((uint64_t)p[i*8+j])<<(j*8);
            out_debug[20+i]=v;
        }
        // DAG elements are 32 bytes each — no second word group
        for(int i=0;i<4;i++) out_debug[24+i]=0;
    }

    // Sum
    uint8_t sum[31]={};
    for(int k=0;k<32;k++){
        const uint8_t* p=dag+(uint64_t)idx[k]*32+1;
        uint32_t carry=0;
        for(int j=30;j>=0;j--){uint32_t s=(uint32_t)sum[j]+p[j]+carry;sum[j]=(uint8_t)s;carry=s>>8;}
    }
    // Pack sum as LE uint64 words
    {
        uint64_t w=0; int wb=0;
        int wi=28;
        for(int i=0;i<31;i++){
            w|=((uint64_t)sum[i])<<(wb*8); wb++;
            if(wb==8){out_debug[wi++]=w;w=0;wb=0;}
        }
        if(wb) out_debug[wi]=w;
    }

    // Final hash
    uint64_t fh[4];
    blake2b_hash_31(sum, fh);
    for(int i=0;i<4;i++) out_debug[32+i]=fh[i];

    // fh_be
    uint32_t fh_be[8];
    for(int i=0;i<4;i++){
        uint32_t lo32=(uint32_t)(fh[i]&0xFFFFFFFF);
        uint32_t hi32=(uint32_t)(fh[i]>>32);
        fh_be[i*2+0]=__byte_perm(lo32,0,0x0123);
        fh_be[i*2+1]=__byte_perm(hi32,0,0x0123);
    }
    for(int i=0;i<8;i++) out_debug[36+i]=(uint64_t)fh_be[i];
}

struct Job {
    std::string job_id;
    uint64_t    height;
    uint8_t     msg[32];
    uint32_t    target[8];
    uint32_t    nonce_prefix;
    bool        valid=false;
};

static Job              g_job;
static std::mutex       g_job_mutex;
static std::atomic<uint32_t> g_nonce_counter{0};
static std::atomic<uint64_t> g_hash_count{0};
static std::atomic<bool>     g_running{true};
static std::atomic<int>      g_accepted{0};
static std::atomic<int>      g_rejected{0};

static SOCKET     g_sock=INVALID_SOCKET;
static std::mutex g_sock_mutex;
static std::atomic<int> g_msg_id{1};
static const char* POOL_HOST   = "erg.2miners.com";
static const char* POOL_PORT   = "8888";
static const char* WALLET_ADDR = "9gh3EZ1ETq8NWQjBBnxk5vFpYz48mGH46gboa6fxMjqgwKucwL5";
static const char* WORKER_NAME = "rig0";

static std::string g_extranonce1    = "";
static int         g_extranonce2_size = 4;
static int         g_nonce_total_size = 6;

static bool send_line(const std::string& line){
    std::string msg=line+"\n";
    std::lock_guard<std::mutex> lock(g_sock_mutex);
    int r=send(g_sock,msg.c_str(),(int)msg.size(),0);
    if(r!=(int)msg.size()){ LOG("[NET] send failed (sent %d/%d, line=%s)\n",r,(int)msg.size(),line.substr(0,60).c_str()); return false; }
    LOG("[SEND] %s\n",line.c_str());
    return true;
}
static bool recv_line(std::string& line){
    line.clear(); char c;
    while(true){int r=recv(g_sock,&c,1,0);if(r<=0)return false;if(c=='\n')return true;if(c!='\r')line+=c;}
}
static bool stratum_connect(){
    struct addrinfo hints={},*res=nullptr;
    hints.ai_family=AF_INET; hints.ai_socktype=SOCK_STREAM;
    if(getaddrinfo(POOL_HOST,POOL_PORT,&hints,&res)!=0) return false;
    g_sock=socket(res->ai_family,res->ai_socktype,res->ai_protocol);
    if(g_sock==INVALID_SOCKET){freeaddrinfo(res);return false;}
    DWORD to=120000; setsockopt(g_sock,SOL_SOCKET,SO_RCVTIMEO,(char*)&to,sizeof(to));
    if(connect(g_sock,res->ai_addr,(int)res->ai_addrlen)!=0){
        closesocket(g_sock); g_sock=INVALID_SOCKET;
        freeaddrinfo(res); return false;
    }
    freeaddrinfo(res);
    LOG("[NET] Connected to %s:%s\n",POOL_HOST,POOL_PORT);
    return true;
}

static void parse_subscribe_response(const std::string& line) {
    size_t result_pos = line.find("\"result\":");
    if(result_pos == std::string::npos) return;
    std::string sub = line.substr(result_pos);
    size_t search = 9;
    while(search < sub.size()) {
        if(sub[search] == '"') {
            size_t end = search+1;
            while(end < sub.size() && sub[end] != '"') end++;
            std::string candidate = sub.substr(search+1, end-search-1);
            if(candidate.size() >= 2 && candidate.size() <= 16 && candidate.size()%2==0) {
                bool all_hex = true;
                for(char c : candidate) if(!isxdigit((unsigned char)c)){all_hex=false;break;}
                if(all_hex) {
                    g_extranonce1 = candidate;
                    int en1_bytes = (int)candidate.size() / 2;
                    size_t after = end+1;
                    while(after < sub.size() && (sub[after]==',' || sub[after]==' ')) after++;
                    if(after < sub.size() && isdigit(sub[after])) {
                        char* endp;
                        long val = strtol(sub.c_str()+after, &endp, 10);
                        if(endp == sub.c_str()+after || val < 2 || val > 16) {
                            LOG("[STRATUM] Invalid nonce_total_size from pool, using defaults\n");
                        } else {
                            g_nonce_total_size = (int)val;
                            g_extranonce2_size = g_nonce_total_size - en1_bytes;
                            if(g_extranonce2_size < 1) g_extranonce2_size = 4;
                        }
                        LOG("[STRATUM] extranonce1=%s (%d bytes), nonce_total=%d, extranonce2_size=%d\n",
                            g_extranonce1.c_str(), en1_bytes, g_nonce_total_size, g_extranonce2_size);
                    }
                    break;
                }
            }
            search = end+1;
        } else {
            search++;
        }
    }
}

static uint32_t g_share_target[8]={0x00200000,0,0,0,0,0,0,0};
static std::mutex g_target_mutex;
static uint32_t g_network_b[8]={0x00000000,0x3f000000,0x03f00000,0x003f0000,
                                  0x0003f000,0x00003f00,0x000003f0,0x0000003f};
static bool   g_network_b_set = false;
static double g_current_diff  = 1.0;
static std::atomic<int> g_submit_count{0};
static DWORD g_submit_window_start = 0;
static const int MAX_SUBMITS_PER_SEC = 4;

static void parse_decimal_bignum(const char* s, uint32_t out[8]){
    memset(out, 0, 32);
    for(; *s; s++){
        if(*s < '0' || *s > '9') break;
        uint32_t digit = *s - '0';
        uint64_t carry = digit;
        for(int i = 7; i >= 0; i--){
            uint64_t cur = (uint64_t)out[i] * 10 + carry;
            out[i] = (uint32_t)(cur & 0xFFFFFFFF);
            carry = cur >> 32;
        }
    }
}

static void set_difficulty(double diff){
    if(diff<=0) diff=1.0;
    g_current_diff = diff;
    uint64_t d = (uint64_t)(diff + 0.5);
    if(d < 1) d = 1;
    uint32_t tmp[8];
    uint64_t rem = 0;
    for(int i = 0; i < 8; i++){
        uint64_t cur = (rem << 32) | (uint64_t)g_network_b[i];
        tmp[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    { std::lock_guard<std::mutex> lk(g_target_mutex); memcpy(g_share_target, tmp, 32); }
    LOG("[DIFF] difficulty=%.4f -> share_target=%08x %08x %08x\n",
        diff, tmp[0], tmp[1], tmp[2]);
}

static void parse_notify(const std::string& line){
    auto p=parse_params_array(line);
    LOG("[NOTIFY] %zu params\n",p.size());
    for(size_t i=0;i<p.size();i++) LOG("  p[%zu]='%s'\n",i,p[i].substr(0,80).c_str());
    if(p.size()<3){LOG("[NOTIFY] Too few params\n");return;}

    Job job; job.job_id=p[0]; job.height=0; job.nonce_prefix=0;
    memset(job.msg,0,32);
    { std::lock_guard<std::mutex> lk(g_target_mutex); memcpy(job.target, g_share_target, 32); }

    bool got_msg=false, clean=false;

    for(size_t i=1;i<p.size();i++){
        const std::string& s=p[i];
        if(s=="true"){clean=true;continue;}
        if(s=="false"){clean=false;continue;}
        if(s.empty()) continue;

        bool all_hex=true; for(char c:s) if(!isxdigit((unsigned char)c)){all_hex=false;break;}
        bool all_dec=true; for(char c:s) if(!isdigit((unsigned char)c)){all_dec=false;break;}

        if(all_hex && s.size()==64 && !got_msg){
            hex2bytes(s.c_str(), job.msg, 32);
            got_msg=true;
            continue;
        }
        if(all_hex && s.size()==8){
            unsigned int v=0; sscanf(s.c_str(),"%08x",&v);
            job.nonce_prefix=(uint32_t)v;
            continue;
        }
        if(all_dec && s.size()>=7 && s.size()<=8){
            uint64_t v=strtoull(s.c_str(),nullptr,10);
            if(v>100000 && v<10000000){job.height=v; continue;}
        }
        if(all_dec && s.size() > 20) {
            uint32_t parsed_b[8];
            parse_decimal_bignum(s.c_str(), parsed_b);
            bool nonzero = false;
            for(int w=0;w<8;w++) if(parsed_b[w]){nonzero=true;break;}
            if(nonzero){
                memcpy(g_network_b, parsed_b, 32);
                g_network_b_set = true;
                LOG("[NOTIFY] b=%08x %08x %08x (parsed from p[6])\n",
                    g_network_b[0], g_network_b[1], g_network_b[2]);
                double d_cur = g_current_diff > 0 ? g_current_diff : 1.0;
                uint64_t d = (uint64_t)(d_cur + 0.5); if(d<1) d=1;
                uint32_t tmp2[8]; uint64_t rem = 0;
                for(int w=0;w<8;w++){
                    uint64_t cur=(rem<<32)|(uint64_t)g_network_b[w];
                    tmp2[w]=(uint32_t)(cur/d);
                    rem=cur%d;
                }
                { std::lock_guard<std::mutex> lk(g_target_mutex); memcpy(g_share_target, tmp2, 32); }
                LOG("[DIFF] recalc after b update -> share_target=%08x %08x %08x\n",
                    tmp2[0],tmp2[1],tmp2[2]);
            }
            continue;
        }
    }

    if(!got_msg){LOG("[NOTIFY] No msg hash!\n");return;}
    job.valid=true;

    LOG("[JOB] id=%s height=%llu clean=%d prefix=%08x msg=%s\n",
        job.job_id.c_str(),(unsigned long long)job.height,(int)clean,
        job.nonce_prefix,bytes2hex(job.msg,8).c_str());

    std::lock_guard<std::mutex> lk(g_job_mutex);
    g_job=job;
    if(clean) g_nonce_counter.store(0);
}

static void stratum_recv_thread(){
    std::string line;
    while(g_running){
        if(!recv_line(line)){LOG("[NET] Disconnected\n");g_running=false;break;}
        if(line.empty()) continue;
        LOG("[RECV] %s\n",line.substr(0,300).c_str());
        std::string method=json_get_str(line,"method");
        int id=json_get_id(line);
        if(method=="mining.notify") parse_notify(line);
        else if(method=="mining.set_difficulty"){
            auto pp=parse_params_array(line);
            if(!pp.empty()) set_difficulty(atof(pp[0].c_str()));
        }
        else if(id>0){
            if(id==1) {
                bool has_result = line.find("\"result\":")!=std::string::npos;
                bool error_ok   = line.find("\"error\":null")!=std::string::npos
                               || line.find("\"error\"")==std::string::npos;
                if(has_result && error_ok) parse_subscribe_response(line);
            }
            if(line.find("\"result\":true")!=std::string::npos){
                if(id<=2) LOG("[STRATUM] OK\n");
                else LOG("[POOL] ACCEPTED (+%d)\n",++g_accepted);
            }
            else if(line.find("\"result\":false")!=std::string::npos ||
                    line.find("\"error\":[")!=std::string::npos){
                if(id<=2) LOG("[STRATUM] FAIL: %s\n",line.c_str());
                else{g_rejected++;LOG("[POOL] REJECTED: %s\n",line.c_str());}
            }
        }
    }
}

static void stratum_subscribe(){
    char buf[256];
    snprintf(buf,sizeof(buf),"{\"id\":%d,\"method\":\"mining.subscribe\",\"params\":[\"ergominer/0.9\"]}",g_msg_id.fetch_add(1));
    send_line(buf);
}
static void stratum_authorize(){
    char buf[768];
    std::string u=std::string(WALLET_ADDR)+"."+WORKER_NAME;
    snprintf(buf,sizeof(buf),"{\"id\":%d,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}",g_msg_id.fetch_add(1),u.c_str());
    send_line(buf);
}

static void stratum_submit(const std::string& job_id, uint64_t nonce_found){
    // FIX v0.8: 2miners Ergo stratum expects the FULL nonce (en1+en2) as a single hex string,
    // not just en2. nonce_total_size=6 bytes = 12 hex chars.
    // e.g. en1=d628, en2=4b549d40 -> submit "d6284b549d40"
    uint64_t en2_mask = (g_extranonce2_size >= 8) ? 0xFFFFFFFFFFFFFFFFULL
                                                   : ((1ULL << (g_extranonce2_size*8)) - 1ULL);
    uint64_t en2_val = nonce_found & en2_mask;

    char en2_hex[32] = {};
    char fmt[16]; sprintf(fmt, "%%0%dllx", g_extranonce2_size*2);
    sprintf(en2_hex, fmt, (unsigned long long)en2_val);

    // Full nonce = en1 + en2 (total nonce_total_size bytes = nonce_total_size*2 hex chars)
    std::string full_nonce = g_extranonce1 + std::string(en2_hex);

    char buf[1024];
    std::string u = std::string(WALLET_ADDR) + "." + WORKER_NAME;
    snprintf(buf, sizeof(buf), "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\"]}",
        g_msg_id.fetch_add(1), u.c_str(), job_id.c_str(), full_nonce.c_str());
    send_line(buf);

    LOG("[SUBMIT] job=%s nonce=%016llx en1=%s en2=%s full=%s\n",
        job_id.c_str(), (unsigned long long)nonce_found,
        g_extranonce1.c_str(), en2_hex, full_nonce.c_str());
}

// ===== DAG Self-Test =====
__global__ void dag_selftest_kernel(const uint8_t* __restrict__ dag, uint8_t* out) {
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    for(int i=0; i<64; i++) out[i] = dag[i];
}

static void dag_selftest(const uint8_t* d_dag, uint64_t height) {
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 64));
    dag_selftest_kernel<<<1,1>>>(d_dag, d_out);
    cudaDeviceSynchronize();
    uint8_t out[64];
    cudaMemcpy(out, d_out, 64, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    LOG("[DAG-VERIFY] height=%llu  DAG[0]: ", (unsigned long long)height);
    for(int i=0;i<32;i++) LOG("%02x",out[i]);
    LOG("\n[DAG-VERIFY] DAG[1]: ");
    for(int i=32;i<64;i++) LOG("%02x",out[i]);
    LOG("\n");
    LOG("[DAG-VERIFY] Python check: import hashlib,struct; "
        "e=lambda i,h: (b'\\x00'+hashlib.blake2b(struct.pack('<QQ',i,h)+bytes(8192),digest_size=32).digest()[1:32]).hex(); "
        "print(e(0,%llu)); print(e(1,%llu))\n",
        (unsigned long long)height, (unsigned long long)height);
}

static void gpu_mine_loop(){
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,4096));

    // FIX v0.7: increased BLOCKS from 112 to 448 (28 SM * 16 blocks/SM)
    // This gives ~4x more GPU threads per batch -> ~4x hashrate improvement.
    // RTX 3060 has 28 SMs; 16 blocks/SM * 256 threads = 4096 threads/SM = good occupancy.
    const int THREADS = 256;
    const int BLOCKS  = 28 * 16;   // 448 blocks (was 28*4=112)
    const uint32_t BATCH = THREADS * BLOCKS;  // 114688 (was 14336)

    CUDA_CHECK(cudaMalloc(&d_result,    sizeof(MineResult)));
    CUDA_CHECK(cudaMalloc(&d_target,    8*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blob_words,4*sizeof(uint64_t)));

    Job cur; bool have_job=false;
    int iter=0;

    while(g_running){
        {
            std::lock_guard<std::mutex> lk(g_job_mutex);
            if(g_job.valid&&g_job.job_id!=cur.job_id){
                cur=g_job; have_job=true;
                LOG("[GPU] New job %s height=%llu prefix=%08x\n",
                    cur.job_id.c_str(),(unsigned long long)cur.height,cur.nonce_prefix);
            }
        }
        if(!have_job){Sleep(100);continue;}

        // DAG rebuild if needed
        {
            uint64_t needed_N=autolykos_n(cur.height);
            if(needed_N!=dag_N||d_dag==nullptr){
                if(d_dag){cudaFree(d_dag);d_dag=nullptr;dag_N=0;}
                DWORD t0=GetTickCount();
                if(build_dag(&d_dag,cur.height,&dag_N)!=cudaSuccess){
                    Sleep(5000);continue;
                }
                size_t fb,tb; cudaMemGetInfo(&fb,&tb);
                LOG("[GPU] DAG OK N=%llu time=%.1fs free=%.0fMB\n",
                    (unsigned long long)dag_N,(GetTickCount()-t0)/1000.0,fb/1048576.0);
                dag_selftest(d_dag, cur.height);
            }
        }

        iter++;
        uint32_t ctr = g_nonce_counter.fetch_add(BATCH);

        // Nonce = extranonce1 (high bytes) || extranonce2 counter (low bytes)
        uint64_t en1_val = 0;
        for(int i = 0; i < (int)g_extranonce1.size(); i += 2) {
            unsigned int b = 0;
            sscanf(g_extranonce1.c_str()+i, "%02x", &b);
            en1_val = (en1_val << 8) | b;
        }
        int shift_bits = g_extranonce2_size * 8;
        uint64_t nonce_start = (shift_bits > 0 && shift_bits < 64)
            ? ((en1_val << shift_bits) | (uint64_t)ctr)
            : (uint64_t)ctr;

        uint64_t bw[4]; memcpy(bw,cur.msg,32);
        CUDA_CHECK(cudaMemcpy(d_blob_words,bw,32,cudaMemcpyHostToDevice));
        { std::lock_guard<std::mutex> lk(g_target_mutex);
          CUDA_CHECK(cudaMemcpy(d_target,g_share_target,32,cudaMemcpyHostToDevice)); }

        // First iteration: verbose debug to compare CPU vs GPU intermediate values
        if(iter==1){
            uint64_t* d_dbg=nullptr;
            CUDA_CHECK(cudaMalloc(&d_dbg, 64*sizeof(uint64_t)));
            cudaMemset(d_dbg, 0, 64*sizeof(uint64_t));

            LOG("[DEBUG] Verbose GPU debug for nonce=%016llx\n",(unsigned long long)nonce_start);
            debug_one_hash_kernel<<<1,1>>>(d_dag,dag_N,d_blob_words,nonce_start,d_dbg);
            cudaDeviceSynchronize();
            uint64_t dbg[64]={};
            cudaMemcpy(dbg,d_dbg,64*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            cudaFree(d_dbg);

            LOG("[DEBUG] GPU seed[0:4]:   %016llx %016llx %016llx %016llx\n",
                (unsigned long long)dbg[0],(unsigned long long)dbg[1],
                (unsigned long long)dbg[2],(unsigned long long)dbg[3]);
            LOG("[DEBUG] GPU ext[0:4]:    %08x %08x %08x %08x\n",
                (unsigned)(dbg[8]),(unsigned)(dbg[9]),
                (unsigned)(dbg[10]),(unsigned)(dbg[11]));
            LOG("[DEBUG] GPU idx[0:4]:    %u %u %u %u\n",
                (unsigned)(dbg[16]),(unsigned)(dbg[17]),
                (unsigned)(dbg[18]),(unsigned)(dbg[19]));
            LOG("[DEBUG] GPU DAG[idx0]:   %016llx %016llx %016llx %016llx\n",
                (unsigned long long)dbg[20],(unsigned long long)dbg[21],
                (unsigned long long)dbg[22],(unsigned long long)dbg[23]);
            LOG("[DEBUG] GPU sum[0:4]:    %016llx %016llx %016llx %016llx\n",
                (unsigned long long)dbg[28],(unsigned long long)dbg[29],
                (unsigned long long)dbg[30],(unsigned long long)dbg[31]);
            LOG("[DEBUG] GPU final_hash:  %016llx %016llx %016llx %016llx\n",
                (unsigned long long)dbg[32],(unsigned long long)dbg[33],
                (unsigned long long)dbg[34],(unsigned long long)dbg[35]);
            LOG("[DEBUG] GPU fh_be:       %08x%08x%08x%08x%08x%08x%08x%08x\n",
                (unsigned)(dbg[36]),(unsigned)(dbg[37]),(unsigned)(dbg[38]),(unsigned)(dbg[39]),
                (unsigned)(dbg[40]),(unsigned)(dbg[41]),(unsigned)(dbg[42]),(unsigned)(dbg[43]));
            LOG("[DEBUG] share_target:    %08x%08x%08x%08x%08x%08x%08x%08x\n",
                g_share_target[0],g_share_target[1],g_share_target[2],g_share_target[3],
                g_share_target[4],g_share_target[5],g_share_target[6],g_share_target[7]);
        }

        MineResult zero={0,false,{}};
        CUDA_CHECK(cudaMemcpy(d_result,&zero,sizeof(zero),cudaMemcpyHostToDevice));

        mine_kernel<<<BLOCKS,THREADS>>>(d_dag,dag_N,d_blob_words,nonce_start,BATCH,d_target,d_result);

        if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess){
            LOG("[GPU] Kernel error iter=%d\n",iter); Sleep(1000); continue;
        }

        g_hash_count.fetch_add(BATCH);

        MineResult res;
        CUDA_CHECK(cudaMemcpy(&res,d_result,sizeof(res),cudaMemcpyDeviceToHost));
        if(res.found){
            LOG("[FOUND] nonce=%016llx  hash=%08x%08x%08x%08x...\n",
                (unsigned long long)res.nonce,res.fh_be[0],res.fh_be[1],res.fh_be[2],res.fh_be[3]);
            DWORD now = GetTickCount();
            if(now - g_submit_window_start >= 1000) {
                g_submit_window_start = now;
                g_submit_count.store(0);
            }
            if(g_submit_count.fetch_add(1) < MAX_SUBMITS_PER_SEC) {
                std::string jid;{std::lock_guard<std::mutex> lk(g_job_mutex);jid=cur.job_id;}
                stratum_submit(jid, res.nonce);
            } else {
                LOG("[FLOOD] Skipping submit (too many shares/sec)\n");
            }
        }

        if(iter%1000==0){
            LOG("[ITER %d] share_target[0]=%08x  hash_count=%llu MH=%.2f\n",
                iter,g_share_target[0],(unsigned long long)g_hash_count.load(),
                g_hash_count.load()/1e6);
        }
    }
}

static void keepalive_thread(){
    // Send a ping every 45s to prevent pool from dropping the connection.
    // 2miners drops miners after ~120s of silence.
    while(g_running){
        for(int i=0;i<45&&g_running;i++) Sleep(1000);
        if(!g_running) break;
        if(g_sock==INVALID_SOCKET) continue;
        // Send keepalive as mining.extranonce.subscribe (harmless no-op)
        char buf[256];
        snprintf(buf,sizeof(buf),"{\"id\":%d,\"method\":\"mining.extranonce.subscribe\",\"params\":[]}",g_msg_id.fetch_add(1));
        send_line(buf);
    }
}

static void hashrate_thread(){
    DWORD tl=GetTickCount(); uint64_t hl=0;
    while(g_running){
        Sleep(15000);
        DWORD tn=GetTickCount(); uint64_t hn=g_hash_count.load();
        double dt=(tn-tl)/1000.0; if(dt<1)dt=1;
        LOG("[RATE] %.2f MH/s  A=%d R=%d\n",(hn-hl)/dt/1e6,g_accepted.load(),g_rejected.load());
        tl=tn; hl=hn;
    }
}

int main(){
    srand((unsigned)time(nullptr));
    LOG("=== ErgoMiner v0.9 ===\n");
    LOG("[FIX] nonce hashed as big-endian bytes (matches pool verification)\n");
    LOG("[FIX] increased GPU batch size for better hashrate\n");
    WSADATA wsa; WSAStartup(MAKEWORD(2,2),&wsa);
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    LOG("[GPU] %s  SM=%d  VRAM=%.0fMB\n",prop.name,prop.multiProcessorCount,prop.totalGlobalMem/1048576.0);
    LOG("[GPU] Batch size: %d threads (%d blocks x %d threads)\n", 256*28*16, 28*16, 256);

    std::thread(hashrate_thread).detach();
    std::thread(keepalive_thread).detach();

    while(true){
        if(g_sock!=INVALID_SOCKET){ closesocket(g_sock); g_sock=INVALID_SOCKET; }

        LOG("[NET] Connecting to %s:%s...\n", POOL_HOST, POOL_PORT);
        if(!stratum_connect()){
            LOG("[NET] Connection failed, retrying in 10s...\n");
            Sleep(10000);
            continue;
        }

        g_running.store(true);
        g_msg_id = 1;
        g_extranonce1 = "";
        g_extranonce2_size = 4;
        g_nonce_total_size = 6;
        { std::lock_guard<std::mutex> lk(g_job_mutex); g_job = Job{}; }

        stratum_subscribe(); Sleep(300); stratum_authorize();
        std::thread(stratum_recv_thread).detach();

        for(int i=0;i<150&&g_running;i++){
            { std::lock_guard<std::mutex> lk(g_job_mutex); if(g_job.valid) break; }
            Sleep(100);
        }
        bool have_initial_job;
        { std::lock_guard<std::mutex> lk(g_job_mutex); have_initial_job=g_job.valid; }
        if(!have_initial_job){
            LOG("[MAIN] No job received, reconnecting...\n");
            g_running.store(false);
            Sleep(5000);
            continue;
        }

        gpu_mine_loop();

        g_running.store(false);
        LOG("[NET] Reconnecting in 5s...\n");
        Sleep(5000);
    }

    WSACleanup();
    return 0;
}
