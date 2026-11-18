// 草稿扩展变体 pa_ext：v5 内核 + 每序列变长 q（1..SQ=4 行，按 cu_q 取/写，位置 seqlen−qlen+r）。
// 动机：草稿扩展曾走 prefill 核（grid 只有 bs 个 block），128k 一次 17 ms；split-KV 后预期 ~0.35 ms。
// v5：v4 + split 按页交错分配（split sp 取页 sp, sp+N, …，并发 block 的访存合成一条连续流，
//   避开顺序页表下等距起址撞同一显存通道——探针：顺序页 128 block 只有 205 GB/s）+ V 整行读（64 B/线程，探针整页读比半页读高 40%）。
// v4：C 接口与 pa_lib.cpp 完全一致（pa_decode / pa_combine / pa_pick_nsplit / ...），只换主内核。
// 旧内核（pa_v3）每页「装 V→sync→算」无预取、1 wave/SIMD，131072×1 发射账只需 38 µs 却实测 176 µs（78% 在等显存）。
// v4：4 个 wave 沿 head_dim 分工（QK 各算 64 维部分点积经 LDS 求和；PV 各算 64 个输出维），共享同一半页 K/V，
//     装载时解包一次成 fp16 存 LDS，软件流水（V 读在 sync 后发、下半页 K 在 PV 期间飞），每 wave 只需 acc 16 + Q 8 个 VGPR。
#include <hip/hip_runtime.h>
#include <cmath>
using u8 = unsigned char; using u16 = unsigned short;
typedef _Float16 half_t;
typedef half_t half4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ u16 f2bf(float f){ unsigned u=__float_as_uint(f); if(!(f==f)) return 0x7fc0; return (u16)((u + 0x7fffu + ((u>>16)&1u))>>16); }
__device__ __forceinline__ float fexp(float x){ return __builtin_amdgcn_exp2f(x * __uint_as_float(0x3fb8aa3bu)); }
__device__ __forceinline__ unsigned shfl_lane(unsigned v, int src){ return __builtin_amdgcn_ds_bpermute(src << 2, v); }
__device__ __forceinline__ float bf2fl(u16 h) { return __uint_as_float((unsigned)h << 16); }

#define HD   256
#define PAGE 64
#define HP   32             // 半页粒度
#define HQ   4
#ifndef PA_SQ
#define PA_SQ 4             // 4=验证步（q=4）；1=草稿步（q=1）
#endif
#define SQ   PA_SQ
#define NQ   (HQ * SQ)      // 16 或 4
#define NQMAX 16
#define NW   4
#ifndef QFMT
#define QFMT 4              // 4=e4m3  5=e5m2
#endif
#define PA_MAX_SPLIT 512

#if QFMT == 5
__device__ __forceinline__ u32x2 dq2(unsigned w) {
    u32x2 r;
    r[0] = __builtin_amdgcn_perm(0u, w, 0x010C000Cu);
    r[1] = __builtin_amdgcn_perm(0u, w, 0x030C020Cu);
    return r;
}
#define QCORR 1.0f
#else
// e4m3 → fp16 位重排，值差 2^8；K 侧由 scale 吸收、V 侧由输出乘回（与 pa_lib.cpp 相同约定）
__device__ __forceinline__ unsigned e4m3pair(unsigned x) { return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u); }
__device__ __forceinline__ u32x2 dq2(unsigned w) {
    u32x2 r;
    r[0] = e4m3pair(__builtin_amdgcn_perm(0u, w, 0x010C000Cu));
    r[1] = e4m3pair(__builtin_amdgcn_perm(0u, w, 0x030C020Cu));
    return r;
}
#define QCORR 256.0f
#endif

#define KSTR 130      // K 半页 [32 token][256 dim fp16 = 128 dword + 2 pad]
#define VSTR 18       // V 半页 [256 dim][32 token fp16 = 16 dword + 2 pad]
#define S16_BYTES (HD * VSTR * 4)          // 18432
#define RSTR 36       // s_red 每行 36 dword：[L][tb 2][H 4][j 4] = 32 + 4 pad（b128 读 4 拍最优）

extern "C" __global__ __launch_bounds__(256)
void pa_ext(float* __restrict__ o_part,   // [bs, nsplit, NQ, HD] f32 部分输出（已乘 QCORR，未归一化）
           float* __restrict__ ml,       // [bs, nsplit, NQ, 2]
           const u16* __restrict__ q,    // [bs, SQ, HQ, HD] bf16
           const u8* __restrict__ kc, const u8* __restrict__ vc,
           const int* __restrict__ bt, const int* __restrict__ slen,
           const int* __restrict__ cu_q,                  // [bs+1] 各序列 q 行起点（token 数）
           int n_per, float scale, int nsplit, int G) {
    const int b = blockIdx.x, sp = blockIdx.y;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;      // 本序列新 token 数（1..SQ）
    const int tid = threadIdx.x;
    const int lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int seqlen = slen[b];
    const int npages = (seqlen + PAGE - 1) / PAGE;
    // 本 split 处理页 sp, sp+nsplit, sp+2·nsplit, …（交错）；每页两个半页背靠背。空 split 不 return（写 m=-inf/l=0/acc=0 供 combine 忽略）

    const bool qvalid = (L < NQ) && ((L / HQ) < qlen);          // 行 L = (新 token L/HQ, 头 L%HQ)，超出 qlen 的行无效
    const int qpos = seqlen - qlen + (qvalid ? (L / HQ) : 0);   // 绝对位置（因果）：新 token 位于序列末尾 qlen 个
    half4 qreg[4];
    {
        const u16* qp = q + ((size_t)(q0 + (qvalid ? (L / HQ) : 0)) * HQ + (L % HQ)) * HD + 64 * w;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const u32x2 rq = *(const u32x2*)&qp[kk * 16 + 4 * H];
            half4 t;
            t[0] = (half_t)__uint_as_float(rq[0] << 16); t[1] = (half_t)__uint_as_float(rq[0] & 0xFFFF0000u);
            t[2] = (half_t)__uint_as_float(rq[1] << 16); t[3] = (half_t)__uint_as_float(rq[1] & 0xFFFF0000u);
            qreg[kk] = qvalid ? t : (half4){(half_t)0.f, (half_t)0.f, (half_t)0.f, (half_t)0.f};
        }
    }

    float m_run = -INFINITY, l_run = 0.f;
    f32x4 acc[4];                                             // acc[jj][j] = O[row L][dim 64w + 16jj + H + 4j]
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) acc[jj] = (f32x4){0.f, 0.f, 0.f, 0.f};

    __shared__ __align__(16) u8 s16[S16_BYTES];
    __shared__ __align__(16) float s_red[NW][16][RSTR];       // QK 部分和归约：[w][L][tb*16 + H*4 + j]
    __shared__ half_t s_p[NW][16][18];
    unsigned* s16d = (unsigned*)s16;

    const int ktk = tid >> 3, ksg = tid & 7;                  // K 半页装载：token ktk，dim 段 ksg·32
    u32x4 pre[2];                                             // K 预取（本半页）
    u32x4 vrow[4];                                            // V 整行 64 B（本页两个半页共用）
    auto kaddr = [&](int p, int half) -> const u8* {
        return kc + (size_t)bt[b * n_per + p] * PAGE * HD + (size_t)(half * HP + ktk) * HD + ksg * 32;
    };
    // 页序列：p = (k·nsplit + sp)·G + g，g∈[0,G)；下一页 = p+1（组内）或 p + (nsplit−1)·G + 1（跳组）
    auto next_page = [&](int p) -> int { return ((p % G) == G - 1) ? p + (nsplit - 1) * G + 1 : p + 1; };
    const int p0 = sp * G;
    if (p0 < npages) {
        const u8* kp = kaddr(p0, 0);
#pragma unroll
        for (int c = 0; c < 2; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
    }

    for (int p = p0; p < npages; p = next_page(p))
    for (int half = 0; half < 2; ++half) {
        const int hp = p * 2 + half;
        const int ntok = min(HP, seqlen - hp * HP);
        if (ntok <= 0) break;                                  // 尾页只有前半有效时跳过后半（整 block 一致）
        const u8* vp = vc + (size_t)bt[b * n_per + p] * HD * PAGE + (size_t)tid * PAGE;

        // ── ① K 解包 → s16[token][dim]
        __syncthreads();
        {
            unsigned* dst = s16d + ktk * KSTR + ksg * 16;
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int e = 0; e < 4; ++e) { const u32x2 d = dq2(pre[c][e]); *(u32x2*)&dst[c * 8 + e * 2] = d; }
        }
        __syncthreads();
        // ── ② 本页 V 整行读（只在前半页发出，64 B/线程；后半页直接用 vrow[2..3]）。在 sync 之后发出，QK 期间飞行
        if (half == 0) {
#pragma unroll
            for (int c = 0; c < 4; ++c) vrow[c] = *(const u32x4*)(vp + c * 16);
        }

        // ── ③ QK 部分点积（本 wave 64 维）：B[4H+i][L] = K[16tb+L][64w + 16kk + 4H + i]
        f32x4 s[2];
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
            f32x4 a = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const half4 bfrag = *(const half4*)&s16d[(16 * tb + L) * KSTR + 32 * w + 8 * kk + 2 * H];
                a = __builtin_amdgcn_mmac_f32_16x16x16f16(qreg[kk], bfrag, a);
            }
            s[tb] = a;
        }
        // 4 个 wave 的部分和经 LDS 求和：写 s_red[w][L][tb*16 + H*4 + j]（每 lane 4 个 float 连续 → b128）
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) *(f32x4*)&s_red[w][L][tb * 16 + H * 4] = s[tb];
        __syncthreads();
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
            f32x4 t = *(const f32x4*)&s_red[0][L][tb * 16 + H * 4];
#pragma unroll
            for (int k = 1; k < NW; ++k) t += *(const f32x4*)&s_red[k][L][tb * 16 + H * 4];
            s[tb] = t;
        }
        // s_red 在下一半页 QK 后才会再写，中间隔 ① 的 sync，此处不必再 sync

        // ── ④ 在线 softmax（4 个 wave 各算一份，数据相同结果相同）
        float mb = -INFINITY;
#pragma unroll
        for (int tb = 0; tb < 2; ++tb)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tok = 16 * tb + H + 4 * j;
                const int gtok = hp * HP + tok;
                float v = (tok < ntok && gtok <= qpos) ? s[tb][j] * scale : -INFINITY;
                s[tb][j] = v; mb = fmaxf(mb, v);
            }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            mb = fmaxf(mb, __uint_as_float(shfl_lane(__float_as_uint(mb), lane ^ d)));
        const bool need = mb > m_run + 8.0f;
        const float m_new = need ? mb : m_run;
        if (__ballot(need)) {
            const float corr = (m_run == -INFINITY) ? 0.f : fexp(m_run - m_new);
            l_run *= corr;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[jj][j] *= corr;
        }
        m_run = m_new;
        float lb = 0.f;
#pragma unroll
        for (int tb = 0; tb < 2; ++tb)
#pragma unroll
            for (int j = 0; j < 4; ++j) { s[tb][j] = (s[tb][j] == -INFINITY) ? 0.f : fexp(s[tb][j] - m_new); lb += s[tb][j]; }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            lb += __uint_as_float(shfl_lane(__float_as_uint(lb), lane ^ d));
        l_run += lb;

        // ── ⑤ V 解包 → s16[dim][token]（QK 读 s16 已在 ③ 的 sync 前完成）
        __syncthreads();
        {
            unsigned* dst = s16d + tid * VSTR;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const u32x4 src = (half == 0) ? vrow[c] : vrow[2 + c];
#pragma unroll
                for (int e = 0; e < 4; ++e) { const u32x2 d = dq2(src[e]); *(u32x2*)&dst[c * 8 + e * 2] = d; }
            }
        }
        __syncthreads();
        // ── ⑥ 下一半页 K 读（PV 期间飞行）：同页后半，或下一页（p+nsplit）前半
        {
            const int np_ = (half == 0) ? p : next_page(p);
            const int nh_ = (half == 0) ? 1 : 0;
            if (np_ < npages && (np_ * 2 + nh_) * HP < seqlen) {
                const u8* kp = kaddr(np_, nh_);
#pragma unroll
                for (int c = 0; c < 2; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
            }
        }

        // ── ⑦ PV（本 wave 输出维 64w..64w+63 = jj 4w..4w+3）
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s_p[w][L][H + 4 * j] = (half_t)s[tb][j];
            __builtin_amdgcn_s_waitcnt(0xC07F);
            half4 pa;
#pragma unroll
            for (int i = 0; i < 4; ++i) pa[i] = s_p[w][L][4 * H + i];
            __builtin_amdgcn_s_waitcnt(0xC07F);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                // B[4H+i][L] = V[token 16tb+4H+i][dim 64w + 16jj + L]
                const half4 bfrag = *(const half4*)&s16d[(64 * w + 16 * jj + L) * VSTR + 8 * tb + 2 * H];
                acc[jj] = __builtin_amdgcn_mmac_f32_16x16x16f16(pa, bfrag, acc[jj]);
            }
        }
    }

    // ── 写本 split 的部分结果（不归一化，留给 combine；乘 QCORR 与旧内核一致）
    if (w == 0 && H == 0 && qvalid) {                          // 守卫 L<NQ：NQ<16 时每 split 只有 NQ*2 个 float
        ml[((size_t)b * nsplit + sp) * NQ * 2 + L * 2 + 0] = m_run;
        ml[((size_t)b * nsplit + sp) * NQ * 2 + L * 2 + 1] = l_run;
    }
    if (qvalid) {
        float* op = o_part + ((size_t)b * nsplit + sp) * NQ * HD + (size_t)L * HD + 64 * w;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
            for (int j = 0; j < 4; ++j) op[16 * jj + H + 4 * j] = acc[jj][j] * QCORR;
    }
}

// ── 合并各 split（与 pa_lib.cpp 的 pa_combine 逐字一致）：grid=(bs, NQ)，256 线程分摊 HD
extern "C" __global__ __launch_bounds__(256)
void pa_combine(u16* __restrict__ out, const float* __restrict__ o_part, const float* __restrict__ ml,
                const int* __restrict__ cu_q, int nsplit_used, int nsplit) {
    const int b = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;
    if ((r / HQ) >= qlen) return;                             // 无效行不写（整 block 一致，无 sync 悬挂）
    __shared__ float s_m[PA_MAX_SPLIT], s_l[PA_MAX_SPLIT], s_sc[PA_MAX_SPLIT];
    __shared__ float s_red[256];
    float mloc = -INFINITY;
    for (int k = tid; k < nsplit_used; k += 256) {
        const float mk = ml[((size_t)b * nsplit + k) * NQ * 2 + (size_t)r * 2];
        s_m[k] = mk; s_l[k] = ml[((size_t)b * nsplit + k) * NQ * 2 + (size_t)r * 2 + 1];
        mloc = fmaxf(mloc, mk);
    }
    s_red[tid] = mloc; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s_red[tid] = fmaxf(s_red[tid], s_red[tid + st]); __syncthreads(); }
    const float mg = s_red[0];
    float lloc = 0.f;
    for (int k = tid; k < nsplit_used; k += 256) {
        const float sc = (s_m[k] == -INFINITY) ? 0.f : fexp(s_m[k] - mg);
        s_sc[k] = sc; lloc += s_l[k] * sc;
    }
    __syncthreads();
    s_red[tid] = lloc; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) s_red[tid] += s_red[tid + st]; __syncthreads(); }
    const float inv = (s_red[0] > 0.f) ? 1.f / s_red[0] : 0.f;
    float v = 0.f;
    for (int k = 0; k < nsplit_used; ++k)
        v += o_part[(((size_t)b * nsplit + k) * NQ + r) * HD + tid] * s_sc[k];
    out[((size_t)(q0 + r / HQ) * HQ + (r % HQ)) * HD + tid] = f2bf(v * inv);   // [total_q, HQ, HD]
}

extern "C" {
int pa_workspace_bytes(int bs, int nsplit, int* o_part_bytes, int* ml_bytes) {
    const size_t op = (size_t)bs * nsplit * NQ * HD * sizeof(float);
    const size_t mlb = (size_t)bs * nsplit * NQ * 2 * sizeof(float);
    if (o_part_bytes) *o_part_bytes = (int)op;
    if (ml_bytes)     *ml_bytes     = (int)mlb;
    return (int)(op + mlb);
}
// 交错粒度 G 与 nsplit 的启发式：按顺序页表扫描（2026-09-22，pa_seq.py）拟合
//   总页 ≤512：G=1；bs≥16：G=8；其余 G=4。目标 block 数：总页 ≤256→64、≤4096→256、否则 1024（可用 SGLANG_PA_NSPLIT / SGLANG_PA_G 覆盖）
#include <cstdlib>
int pa_pick_g(int npages, int bs) {
    const char* e = getenv("SGLANG_PA_G"); if (e && *e) return atoi(e);
    const long total = (long)npages * bs;
    return (total <= 512) ? 1 : (bs >= 16) ? 8 : 4;
}
int pa_pick_nsplit(int npages, int bs) {
    const long total = (long)npages * bs;
    int target_blocks = (total <= 256) ? 64 : (total <= 4096) ? 256 : 1024;
    int ns = target_blocks / (bs > 0 ? bs : 1);
    if (ns < 8) ns = 8;
    int p = 8; while (p < ns && p < PA_MAX_SPLIT) p <<= 1;
    return p;
}
void pa_decode_ext(void* out, const void* q, const void* kc, const void* vc,
                   const int* bt, const int* slen, const int* cu_q, float* o_part, float* ml,
                   int bs, int n_per, int npages, int nsplit, float scale_in, void* stream) {
    const float scale = scale_in * QCORR;
    const int G = pa_pick_g(npages, bs);
    const int ngroups = (npages + G - 1) / G;
    const int nsplit_used = (ngroups < nsplit) ? ngroups : nsplit;
    hipStream_t st = (hipStream_t)stream;
    hipLaunchKernelGGL(pa_ext, dim3(bs, nsplit), dim3(256), 0, st,
                       o_part, ml, (const u16*)q, (const u8*)kc, (const u8*)vc, bt, slen, cu_q, n_per, scale, nsplit, G);
    hipLaunchKernelGGL(pa_combine, dim3(bs, NQ), dim3(256), 0, st,
                       (u16*)out, o_part, ml, cu_q, nsplit_used, nsplit);
}
int pa_qfmt() { return QFMT; }
int pa_sq()   { return SQ; }
}
