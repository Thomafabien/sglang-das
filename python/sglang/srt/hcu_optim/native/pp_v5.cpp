// 自写分页 prefill 注意力 v5：v4 + 三处减非 mmac 指令——延迟缩放（只在 m 增长超阈值时缩放 acc）、
//   e4m3 解包用 v_perm（4 条/输出 dword）、因果掩码只在尾半页计算（主循环标量分支跳过）。
//   依据（项目实测）：矩阵单元延迟 42 周期、1 wave/SIMD 只有峰值 77%，需有效并发 ≥3 wave/SIMD；
//   v3（39 KB LDS、1 wave/SIMD）每 wave-页 13275 周期 = mmac 4147 + 其余 9600 完全相加，零重叠。
//   v1/v2 教训：4 个 wave 各自把同一页 fp8 解包 4 遍，VALU 是厂商 4.6 倍（PMC 实测），LDS 读本身已是 1 条/mmac。
//   书三 23.4：矩阵加载/软件流水/减少同步；厂商 bf16 内核本身就是 1 block/CU、1 wave/SIMD 靠 ILP。
// grid = (chunk/16 个 Q 块, bs)，block 256 线程 = 4 wave（各 4 token × 4 头 = 16 行 Q）。
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
#define HQ   4
#define NW   4
#define TPW  4
#define TPB  (NW * TPW)
#ifndef QFMT
#define QFMT 4
#endif

// fp8 dword（4 个元素）→ 两个 dword 的 4 个 fp16（低 dword: 元素 0,1；高 dword: 元素 2,3）
#if QFMT == 5
// e5m2：与 fp16 偏置同为 15，字节左移 8 即得。v_perm 一条完成：r0 = b0<<8 | b1<<24，r1 = b2<<8 | b3<<24
__device__ __forceinline__ u32x2 dq2(unsigned w) {
    u32x2 r;
    r[0] = __builtin_amdgcn_perm(0u, w, 0x010C000Cu);
    r[1] = __builtin_amdgcn_perm(0u, w, 0x030C020Cu);
    return r;
}
#define QCORR 1.0f
#else
// e4m3：[S EEEE MMM] → 符号到 bit15、[EEEE MMM] 左移 7 当 fp16 读，值统一差 2^8（规格/次规格同因子，见 pa_lib.cpp）。
// 先 v_perm 把字节铺到 x = b<<8 | b'<<24，再 (x & 0x80008000) | ((x>>1) & 0x3F803F80)：4 条/输出 dword（原写法编译出 ~11 条）
__device__ __forceinline__ unsigned e4m3pair(unsigned x) {
    return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u);
}
__device__ __forceinline__ u32x2 dq2(unsigned w) {
    u32x2 r;
    r[0] = e4m3pair(__builtin_amdgcn_perm(0u, w, 0x010C000Cu));
    r[1] = e4m3pair(__builtin_amdgcn_perm(0u, w, 0x030C020Cu));
    return r;
}
#define QCORR 256.0f
#endif

// LDS 行 stride（dword）：≡ 4 (mod 32) 且 16 B 对齐 ⇒ 读侧 b64（每 lane 2 dword、半 wave 16 行×2）两拍铺满 32 bank，写侧可 b128
// 半页粒度：HP=32 token。stride 取偶数且 mod 32 后 4-bank 组交错铺满（读侧 b64 两拍无额外冲突），8 B 对齐（写侧 b64）
#define HP   32
#define KSTR 130      // K 阶段 [32 token][256 dim fp16 = 128 dword + 2 pad]  = 16640 B
#define VSTR 18       // V 阶段 [256 dim][32 token fp16 = 16 dword + 2 pad]   = 18432 B
#define S16_BYTES (HD * VSTR * 4)

extern "C" __global__ __launch_bounds__(256)
void pp_v5(u16* __restrict__ out, const u16* __restrict__ q,
           const u8* __restrict__ kc, const u8* __restrict__ vc,
           const int* __restrict__ bt, const int* __restrict__ cu_q, const int* __restrict__ slen,
           int n_per, float scale) {
    const int qb = blockIdx.x, b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 63, w = tid >> 6;
    const int L = lane & 15, H = lane >> 4;
    const int q0 = cu_q[b], qlen = cu_q[b + 1] - q0;
    if (qb * TPB >= qlen) return;
    const int seqlen = slen[b];
    const int prefix = seqlen - qlen;
    const int tok_local = qb * TPB + w * TPW + (L / HQ);
    const bool qvalid = tok_local < qlen;
    const int qpos = prefix + (qvalid ? tok_local : 0);
    const int qmax = prefix + min(qb * TPB + TPB - 1, qlen - 1);
    const int nhalf = qmax / HP + 1;                         // 因果：只需扫到含 qmax 的半页
    // 本 block 最小 query 位置 qmin；半页 hp 满足 32hp+31 ≤ qmin 时对全部 query 可见且整半页有效 ⇒ 不必算掩码
    const int qmin = prefix + qb * TPB;
    const int hp_full = (qmin >= HP - 1) ? ((qmin - (HP - 1)) / HP + 1) : 0;

    half4 qreg[16];
    {
        const u16* qp = q + ((size_t)(q0 + (qvalid ? tok_local : 0)) * HQ + (L % HQ)) * HD;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {                     // 一次 8 B（4 个 bf16），不逐元素读
            const u32x2 rq = *(const u32x2*)&qp[kk * 16 + 4 * H];
            half4 t;
            t[0] = (half_t)__uint_as_float(rq[0] << 16); t[1] = (half_t)__uint_as_float(rq[0] & 0xFFFF0000u);
            t[2] = (half_t)__uint_as_float(rq[1] << 16); t[3] = (half_t)__uint_as_float(rq[1] & 0xFFFF0000u);
            qreg[kk] = qvalid ? t : (half4){(half_t)0.f, (half_t)0.f, (half_t)0.f, (half_t)0.f};
        }
    }

    float m_run = -INFINITY, l_run = 0.f;
    f32x4 acc[16];
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) acc[jj] = (f32x4){0.f, 0.f, 0.f, 0.f};

    __shared__ __align__(16) u8 s16[S16_BYTES];
    __shared__ half_t s_p[NW][16][18];
    unsigned* s16d = (unsigned*)s16;

    // 装载分工（半页 32 token）：K 线程 t 读 token t/8 的 dim 段 (t%8)·32（32 B fp8，8 线程拼一行 256 B）；
    //                          V 线程 t 读 dim t 的 32 个 token（32 B，在 64 B 页行内偏移 32·半）
    const int ktk = tid >> 3, ksg = tid & 7;
    u32x4 pre[2];                                         // 预取缓冲：V 读在 QK 期间活跃、K 预取在 PV 期间活跃，生命期不重叠故共用
    {   // 首个半页 K 预取
        const u8* kp = kc + (size_t)bt[b * n_per] * PAGE * HD + (size_t)ktk * HD + ksg * 32;
#pragma unroll
        for (int c = 0; c < 2; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
    }

    for (int hp = 0; hp < nhalf; ++hp) {
        const int pg = bt[b * n_per + (hp >> 1)];
        const int hoff = (hp & 1) * HP;                       // 半页在页内的 token 偏移
        const int ntok = min(HP, seqlen - hp * HP);
        const u8* vp = vc + (size_t)pg * HD * PAGE + (size_t)tid * PAGE + hoff;

        // ── ① K：解包 kreg → s16[token][dim]（fp16）
        __syncthreads();                                      // 上一页 PV 读完 s16
        {
            unsigned* dst = s16d + ktk * KSTR + ksg * 16;      // 本线程 32 个 fp16 = 16 dword，8 B 对齐
#pragma unroll
            for (int c = 0; c < 2; ++c) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const u32x2 d = dq2(pre[c][e]);
                    *(u32x2*)&dst[c * 8 + e * 2] = d;
                }
            }
        }
        __syncthreads();
        // ── ② 发出本页 V 的全局读（延迟由 QK 盖住）。必须在 __syncthreads 之后：
        //    __syncthreads 展开为 s_waitcnt vmcnt(0) lgkmcnt(0) + s_barrier，放在前面会把这批读等完
#pragma unroll
        for (int c = 0; c < 2; ++c) pre[c] = *(const u32x4*)(vp + c * 16);

        // ── ③ S = Q·Kᵀ：B[4H+i][L] = K[token 16tb+L][dim 64r4+16kk+4H+i] → s16 行 (16tb+L)、dword 32r4+8kk+2H（2 dword）
        f32x4 s[2];
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) s[tb] = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r4 = 0; r4 < 4; ++r4)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int tb = 0; tb < 2; ++tb) {
                    const half4 bfrag = *(const half4*)&s16d[(16 * tb + L) * KSTR + 32 * r4 + 8 * kk + 2 * H];
                    s[tb] = __builtin_amdgcn_mmac_f32_16x16x16f16(qreg[r4 * 4 + kk], bfrag, s[tb]);
                }
        // （试过按 kk 奇偶拆成 4 条累加链：63.5 vs 62.2 ms 反而慢 2%，矩阵单元延迟不是此处主因，已回滚）

        // ── ④ 在线 softmax（每半页一次）。主循环（hp < hp_full）不算掩码；标量分支，无发散
        float mb = -INFINITY;
        if (hp < hp_full) {
#pragma unroll
            for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                for (int j = 0; j < 4; ++j) { s[tb][j] *= scale; mb = fmaxf(mb, s[tb][j]); }
        } else {
#pragma unroll
            for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int tok = 16 * tb + H + 4 * j;             // 半页内序号
                    const int gtok = hp * HP + tok;
                    float v = (tok < ntok && gtok <= qpos) ? s[tb][j] * scale : -INFINITY;
                    s[tb][j] = v; mb = fmaxf(mb, v);
                }
        }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            mb = fmaxf(mb, __uint_as_float(shfl_lane(__float_as_uint(mb), lane ^ d)));
        // 延迟缩放：m 只在增长超过阈值 8 时更新（P 最大 e^8≈2981，half 可表示；最终除以 l 后数值等价），
        // 于是绝大多数半页跳过 64 条 acc*corr。need=false 的 lane 有 m_new==m_run ⇒ corr=1。
        const bool need = mb > m_run + 8.0f;                  // m_run=-inf 时只要 mb 有限即为真
        const float m_new = need ? mb : m_run;
        if (__ballot(need)) {                                  // wave 内任一 lane 需要 ⇒ 整 wave 做一次缩放
            const float corr = (m_run == -INFINITY) ? 0.f : fexp(m_run - m_new);
            l_run *= corr;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[jj][j] *= corr;
        }
        m_run = m_new;
        float lb = 0.f;
#pragma unroll
        for (int tb = 0; tb < 2; ++tb)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s[tb][j] = (s[tb][j] == -INFINITY) ? 0.f : fexp(s[tb][j] - m_new);
                lb += s[tb][j];
            }
#pragma unroll
        for (int d = 16; d <= 32; d <<= 1)
            lb += __uint_as_float(shfl_lane(__float_as_uint(lb), lane ^ d));
        l_run += lb;

        // ── ⑤ V：解包 vreg → s16[dim][token]（fp16）
        __syncthreads();                                      // QK 读完 s16
        {
            unsigned* dst = s16d + tid * VSTR;                 // 本线程 32 个 token 的 fp16 = 16 dword
#pragma unroll
            for (int c = 0; c < 2; ++c) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const u32x2 d = dq2(pre[c][e]);
                    *(u32x2*)&dst[c * 8 + e * 2] = d;
                }
            }
        }
        __syncthreads();
        // ── ⑥ 发出下一页 K 的全局读（延迟由 PV 盖住；同样必须在 __syncthreads 之后）
        if (hp + 1 < nhalf) {
            const int hn = hp + 1;
            const u8* kp = kc + (size_t)bt[b * n_per + (hn >> 1)] * PAGE * HD
                              + (size_t)((hn & 1) * HP + ktk) * HD + ksg * 32;
#pragma unroll
            for (int c = 0; c < 2; ++c) pre[c] = *(const u32x4*)(kp + c * 16);
        }

        // ── ⑦ O += P·V：B[4H+i][L] = V[半页内 token 16tb+4H+i][dim 16jj+L] → s16 行 (16jj+L)、dword 8tb+2H
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s_p[w][L][H + 4 * j] = (half_t)s[tb][j];
            // s_p 是本 wave 私有，lane 锁步，只需等 LDS 写完成；用 0xC07F（lgkmcnt(0)，不等 vmcnt）
            // 否则 waitcnt(0) 会把 PV 期间飞行中的 K 预取一起等完（v3 首版反汇编实证）
            __builtin_amdgcn_s_waitcnt(0xC07F);
            half4 pa;
#pragma unroll
            for (int i = 0; i < 4; ++i) pa[i] = s_p[w][L][4 * H + i];
            __builtin_amdgcn_s_waitcnt(0xC07F);
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const half4 bfrag = *(const half4*)&s16d[(16 * jj + L) * VSTR + 8 * tb + 2 * H];
                acc[jj] = __builtin_amdgcn_mmac_f32_16x16x16f16(pa, bfrag, acc[jj]);
            }
        }
    }

    // ── 归一化 + 经 LDS 转置成行连续后合并写出（复用 s16；每 wave 16 行 × 128 dim × 2 B = 4 KB，分 2 批）
    const float inv = (l_run > 0.f) ? (QCORR / l_run) : 0.f;
    u16* op = out + ((size_t)(q0 + (qvalid ? tok_local : 0)) * HQ + (L % HQ)) * HD;
    for (int batch = 0; batch < 2; ++batch) {
        __syncthreads();
        u16* so = (u16*)s16 + w * (16 * 136);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int jj = batch * 8 + t;
#pragma unroll
            for (int j = 0; j < 4; ++j) so[L * 136 + H + 4 * j + 16 * t] = f2bf(acc[jj][j] * inv);
        }
        __syncthreads();
        const unsigned op_lo = (unsigned)((size_t)op & 0xffffffffu), op_hi = (unsigned)((size_t)op >> 32);
        const unsigned qv = qvalid ? 1u : 0u;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const unsigned lo = shfl_lane(op_lo, r), hi = shfl_lane(op_hi, r), v = shfl_lane(qv, r);
            if (v) {
                u16* dst = (u16*)(((size_t)hi << 32) | lo) + batch * 128 + lane * 2;
                *(unsigned*)dst = *(const unsigned*)&so[r * 136 + lane * 2];
            }
        }
    }
}

extern "C" {
void pp_prefill(void* out, const void* q, const void* kc, const void* vc,
                const void* bt, const void* cu_q, const void* slen,
                int bs, int n_per, int max_qlen, float scale, void* stream) {
    hipStream_t st = (hipStream_t)stream;
    const int nqb = (max_qlen + TPB - 1) / TPB;
    const float sc = scale * QCORR;
    hipLaunchKernelGGL(pp_v5, dim3(nqb, bs), dim3(256), 0, st,
        (u16*)out, (const u16*)q, (const u8*)kc, (const u8*)vc,
        (const int*)bt, (const int*)cu_q, (const int*)slen, n_per, sc);
}
int pp_qfmt() { return QFMT; }
}
