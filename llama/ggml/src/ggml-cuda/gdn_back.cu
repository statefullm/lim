#include "gdn_back.cuh"
#include "ggml-cuda/common.cuh"

// GDN backward CUDA kernel for non-KDA, arbitrary n_tokens.
//
// Strategy: each thread block owns one (head, sequence). Each warp owns columns.
// Phase 1 (forward sweep): replay forward pass per column, store per-token scalars
//   (g_exp_t, kv_t, delta_t) in shared memory — only 3 floats per token per block.
//   At the end, s_reg holds S_T (final state after all tokens).
// Phase 2 (backward sweep): walk tokens in reverse. Maintain s_reg = S_t by
//   undoing the forward update: S_{t-1}[i] = (S_t[i] - k_t[i]*delta_t) / g_exp_t.
//   This is O(T) total, no recomputation needed.
//
// d_v is written directly to global memory per column (no conflicts).
// d_g and d_beta are per-head scalars — only col==0 writes them.
// d_q and d_k use atomicAdd for head expansion handling.

template <int S_v>
__global__ void gdn_back_cuda(const float * grad,   // d_output: [S_v, H, n_tokens, n_seqs]
              const float * q,       // [S_k, H_k, n_tokens, ne3_q]
              const float * k,       // [S_k, H_k, n_tokens, ne3_q]
              const float * v,       // [S_v, H, n_tokens, n_seqs]
              const float * g,       // [1, H, n_tokens, n_seqs]
              const float * beta,    // [1, H, n_tokens, n_seqs]
              const float * state,   // [S_v, S_v, H, n_seqs]  (initial state s0)
              float *       dst,     // flat output: [d_q | d_k | d_v | d_g | d_beta | d_state]
              int64_t       H,
              int64_t       n_tokens,
              int64_t       n_seqs,
              int64_t       sq1,     // q stride in dim 1 (floats)
              int64_t       sq2,     // q stride in dim 2 = tokens (floats)
              int64_t       sq3,     // q stride in dim 3 (floats)
              int64_t       sv1,     // v stride in dim 1 (floats)
              int64_t       sv2,     // v stride in dim 2 = tokens (floats)
              int64_t       sv3,     // v stride in dim 3 (floats)
              int64_t       sb1,     // beta stride in dim 1 (floats)
              int64_t       sb2,     // beta stride in dim 2 = tokens (floats)
              int64_t       sb3,     // beta stride in dim 3 (floats)
              const uint3   neqk1_magic,
              const uint3   rq3_magic,
              float         scale,
              int64_t       S_k,
              int64_t       H_k,
              int64_t       nq,      // total elements in q
              int64_t       nk,      // total elements in k
              int64_t       nv,      // total elements in v
              int64_t       ng,      // total elements in g
              int64_t       nb) {    // total elements in beta

    extern __shared__ float shared[];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    if (col >= S_v) return;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    // Shared memory: only per-token scalars. 3*n_tokens floats — tiny even for n_tokens=256.
    float * s_g_exp = shared;
    float * s_kv    = shared + n_tokens;
    float * s_delta = shared + 2 * n_tokens;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // Base pointers for this (head, seq).
    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;

    const int64_t gb_base = sequence * sb3 + h_idx * sb1;
    const float * g_ptr   = g    + gb_base;
    const float * b_ptr   = beta + gb_base;

    // grad[d_output]: [S_v, H, n_tokens, n_seqs]
    const int64_t grad_base_offs = sequence * S_v * H * n_tokens + h_idx * S_v;
    const float * grad_base = grad + grad_base_offs;

    // Load initial state s0 for this column.
    const float * s0_col = state + col * S_v * H * n_seqs + h_idx * S_v * n_seqs + sequence * S_v;

    // Cache s0 in registers (will become running state S_t).
    float s_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_reg[r] = s0_col[i];
    }

    // ===== PHASE 1: FORWARD SWEEP =====
    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q_base + t * sq2;
        const float * k_t = k_base + t * sq2;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            } else {
                k_reg[r] = 0.0f;
                q_reg[r] = 0.0f;
            }
        }

        const float g_val_raw = g_ptr[t * sb2];
        const float g_exp_val = expf(g_val_raw);
        const float beta_val  = b_ptr[t * sb2];

        // kv = sum_i(S[i] * k[i])
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_reg[r] * k_reg[r];
        }
        float kv_val = warp_reduce_sum<warp_size>(kv_shard);

        const float * v_t = v_base + t * sv2;
        float delta_val = (v_t[col] - g_exp_val * kv_val) * beta_val;

        // Update state: S[i] = g_exp * S[i] + k[i] * delta
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                s_reg[r] = g_exp_val * s_reg[r] + k_reg[r] * delta_val;
            } else {
                s_reg[r] = g_exp_val * s_reg[r];
            }
        }

        // Store scalars to shared memory (lane 0 only).
        if (lane == 0) {
            s_g_exp[t] = g_exp_val;
            s_kv[t]    = kv_val;
            s_delta[t] = delta_val;
        }
    }
    __syncthreads();

    // After forward sweep, s_reg holds S_T (state after all tokens).
    // For backward we need to undo: S_{t-1}[i] = (S_t[i] - k_t[i]*delta_t) / g_exp_t

    // ===== PHASE 2: BACKWARD SWEEP =====
    // dS[i] accumulates gradient flowing backward through the recurrent state.
    float dS[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        dS[r] = 0.0f;
    }

    const int64_t dq_h_offs = (iq3 * H_k + iq1) * S_k;
    const int64_t dk_h_offs = dq_h_offs;

    // Pre-compute output base offsets for d_v, d_g, d_beta.
    const int64_t dv_base = nq + nk + (sequence * H + h_idx) * S_v * n_tokens;
    const int64_t dg_base = nq + nk + nv + sequence * H * n_tokens + h_idx * n_tokens;
    const int64_t db_base = nq + nk + nv + ng + sequence * H * n_tokens + h_idx * n_tokens;

    for (int t = n_tokens - 1; t >= 0; t--) {
        const float * q_t = q_base + t * sq2;
        const float * k_t = k_base + t * sq2;
        const float * v_t = v_base + t * sv2;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            } else {
                k_reg[r] = 0.0f;
                q_reg[r] = 0.0f;
            }
        }

        const float g_exp_val = s_g_exp[t];
        const float kv_val    = s_kv[t];
        const float delta_val = s_delta[t];
        const float beta_val  = b_ptr[t * sb2];

        // At this point s_reg holds S_t (state after processing token t).
        const float d_out = grad_base[col + t * S_v * H];

        // --- Backward through output: o_t[c] = scale * sum_i(S_t[i] * q_t[i]) ---
        float d_sn_o[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            d_sn_o[r] = (i < S_k) ? d_out * scale * q_reg[r] : 0.0f;
        }

        // --- Backward through S_t = S_d + k*delta: ---
        float d_delta_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            d_delta_shard += d_sn_o[r] * k_reg[r];
        }
        float d_delta_from_kd = warp_reduce_sum<warp_size>(d_delta_shard);

        // --- Backward through delta = (v - kv)*beta: ---
        float d_v_val     = d_delta_from_kd * beta_val;
        float d_beta_val  = d_delta_from_kd * (v_t[col] - g_exp_val * kv_val);
        float d_kv        = -d_delta_from_kd * beta_val;

        // --- d_S_t_total[i] = d_sn_o[i] + d_kv * k[i] ---
        float d_st_total[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                d_st_total[r] = d_sn_o[r] + d_kv * k_reg[r];
            } else {
                d_st_total[r] = d_sn_o[r];
            }
        }

        // Add accumulated dS from later tokens.
        float d_sd_total[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            d_sd_total[r] = d_st_total[r] + dS[r];
        }

        // --- Backward through S_d = g_exp * S_{t-1}: ---
        float d_g_exp_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (t > 0) {
                float s_prev_i = (s_reg[r] - ((i < S_k) ? k_reg[r] * delta_val : 0.0f)) / g_exp_val;
                d_g_exp_shard += d_sd_total[r] * s_prev_i;
            } else {
                d_g_exp_shard += d_sd_total[r] * s_reg[r];
            }
        }

        float d_ge = warp_reduce_sum<warp_size>(d_g_exp_shard);
        const float d_g_val = g_exp_val * d_ge;

        // === WRITE GRADIENTS TO GLOBAL MEMORY ===

        // d_v: each column writes its own unique location — no conflict.
        {
            const int64_t dv_idx = dv_base + col + t * S_v;
            dst[dv_idx] = d_v_val;
        }

        // d_g and d_beta: per-head scalars, only col==0 writes.
        if (col == 0 && lane == 0) {
            dst[dg_base + t] = d_g_val;
            dst[db_base + t] = d_beta_val;
        }

        // d_q and d_k: atomicAdd for head expansion.
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                atomicAdd(dst + dq_h_offs + t * S_k * H_k + i, d_out * scale * s_reg[r]);
                float s_prev_i = (s_reg[r] - k_reg[r] * delta_val) / g_exp_val;
                float dk_val = d_sn_o[r] * delta_val + d_kv * g_exp_val * s_prev_i;
                atomicAdd(dst + nq + dk_h_offs + t * S_k * H_k + i, dk_val);
            }
        }

        // d_state for t==0: each column writes its own rows.
        if (t == 0) {
            const int64_t ds_base = nq + nk + nv + ng + nb +
                (sequence * H + h_idx) * S_v * S_v + col * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                dst[ds_base + i] = g_exp_val * d_sd_total[r];
            }
        }

        // Undo forward update: S_{t-1}[i] = (S_t[i] - k_t[i]*delta_t) / g_exp_t
        // Propagate dS: d_S_{t-1}[i] += g_exp_t * d_sd_total[i]
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_k) {
                s_reg[r] = (s_reg[r] - k_reg[r] * delta_val) / g_exp_val;
            } else {
                s_reg[r] = s_reg[r] / g_exp_val;
            }
            dS[r] = g_exp_val * d_sd_total[r];
        }
    }
}

template <int S_v>
static void launch_gdn_back(
        const float * grad_d, const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * beta_d, const float * state_d,
        float * dst_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t S_k, int64_t H_k,
        int64_t nq, int64_t nk, int64_t nv, int64_t ng, int64_t nb,
        cudaStream_t stream) {

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    // Shared memory: only 3 * n_tokens floats for scalars. For n_tokens=256: ~3KB.
    size_t smem_bytes = 3 * n_tokens * sizeof(float);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, smem_bytes, stream);
    ggml_cuda_kernel_launch(gdn_back_cuda<S_v>, launch_params,
        grad_d, q_d, k_d, v_d, g_d, beta_d, state_d, dst_d,
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        neqk1_magic, rq3_magic, scale, S_k, H_k, nq, nk, nv, ng, nb);
}

void ggml_cuda_op_gdn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_grad  = dst->src[0];  // d_output: [S_v, H, n_tokens, n_seqs]
    ggml_tensor * src_q     = dst->src[1];  // [S_k, H_k, n_tokens, ne3_q]
    ggml_tensor * src_k     = dst->src[2];  // [S_k, H_k, n_tokens, ne3_q]
    ggml_tensor * src_v     = dst->src[3];  // [S_v, H, n_tokens, n_seqs]
    ggml_tensor * src_g     = dst->src[4];  // [1, H, n_tokens, n_seqs]
    ggml_tensor * src_beta  = dst->src[5];  // [1, H, n_tokens, n_seqs]
    ggml_tensor * src_state = dst->src[6];  // [S_v, S_v, H, n_seqs]

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t , nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t , nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    GGML_ASSERT(src_g->ne[0] == 1); // non-KDA only

    const int64_t S_k = neq0;
    const int64_t H_k = neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * grad_d  = (const float *) src_grad->data;
    const float * q_d     = (const float *) src_q->data;
    const float * k_d     = (const float *) src_k->data;
    const float * v_d     = (const float *) src_v->data;
    const float * g_d     = (const float *) src_g->data;
    const float * beta_d  = (const float *) src_beta->data;
    const float * state_d = (const float *) src_state->data;
    float *       dst_d   = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    const int64_t nq = ggml_nelements(src_q);
    const int64_t nk = ggml_nelements(src_k);
    const int64_t nv = ggml_nelements(src_v);
    const int64_t ng = ggml_nelements(src_g);
    const int64_t nb = ggml_nelements(src_beta);

    // Zero the output buffer (atomicAdd for d_q/d_k needs clean state).
    cudaMemsetAsync(dst_d, 0, ggml_nelements(dst) * sizeof(float), stream);

    const uint3 neqk1_magic = init_fastdiv_values(neq1 ? neq1 : 1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            launch_gdn_back<16>(grad_d, q_d, k_d, v_d, g_d, beta_d, state_d, dst_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, S_k, H_k, nq, nk, nv, ng, nb, stream);
            break;
        case 32:
            launch_gdn_back<32>(grad_d, q_d, k_d, v_d, g_d, beta_d, state_d, dst_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, S_k, H_k, nq, nk, nv, ng, nb, stream);
            break;
        case 64:
            launch_gdn_back<64>(grad_d, q_d, k_d, v_d, g_d, beta_d, state_d, dst_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, S_k, H_k, nq, nk, nv, ng, nb, stream);
            break;
        case 128:
            launch_gdn_back<128>(grad_d, q_d, k_d, v_d, g_d, beta_d, state_d, dst_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, S_k, H_k, nq, nk, nv, ng, nb, stream);
            break;
        default:
            GGML_ABORT("unsupported S_v for GDN backward");
            break;
    }
}

