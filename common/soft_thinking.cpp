#include "soft_thinking.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

// Internal header for direct tensor access
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

void soft_thinking_init(
    soft_thinking_context & ctx,
    const llama_model     * model,
    soft_thinking_params  & /* params */
) {
    const int n_embd = llama_model_n_embd(model);
    ctx.embedding_buffer.resize(n_embd, 0.0f);
    ctx.state = soft_thinking_state::THINKING;
    ctx.low_entropy_count = 0;

    // End-of-thinking detection is now model-agnostic via llama_vocab_is_control()
}

static float compute_full_entropy(const std::vector<float> & probs) {
    float entropy = 0.0f;
    for (float p : probs) {
        if (p > 1e-12f) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

static soft_token sample_top_k(
    const std::vector<std::pair<float, llama_token>> & candidates,
    int top_k
) {
    int k = std::min(top_k, (int)candidates.size());

    soft_token result;
    result.indices.reserve(k);
    result.probs.reserve(k);

    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        result.indices.push_back(candidates[i].second);
        result.probs.push_back(candidates[i].first);
        sum_p += candidates[i].first;
    }

    result.top_id = candidates[0].second;

    // Renormalize probabilities and compute entropy
    result.entropy = 0.0f;
    for (float & p : result.probs) {
        p /= sum_p;
        if (p > 1e-12f) {
            result.entropy -= p * std::log(p);
        }
    }

    return result;
}

// Find minimum k that captures target fraction of full distribution entropy
static soft_token sample_entropy_preserving(
    const std::vector<std::pair<float, llama_token>> & candidates,
    float full_entropy,
    float entropy_frac,
    int min_k,
    int max_k
) {
    float target_entropy = entropy_frac * full_entropy;

    // Find minimum k that captures target entropy
    // Entropy contribution from each token: -p * log(p)
    float captured_entropy = 0.0f;
    int k = 0;

    for (const auto & [prob, token] : candidates) {
        if (prob > 1e-12f) {
            captured_entropy -= prob * std::log(prob);
        }
        k++;

        // Stop when we've captured enough entropy AND met minimum
        if (captured_entropy >= target_entropy && k >= min_k) {
            break;
        }
        if (k >= max_k) {
            break;
        }
    }

    // Ensure we have at least min_k tokens
    k = std::max(k, std::min(min_k, (int)candidates.size()));

    soft_token result;
    result.indices.reserve(k);
    result.probs.reserve(k);

    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        result.indices.push_back(candidates[i].second);
        result.probs.push_back(candidates[i].first);
        sum_p += candidates[i].first;
    }

    result.top_id = candidates[0].second;

    // Renormalize probabilities and compute entropy over the concept
    result.entropy = 0.0f;
    for (float & p : result.probs) {
        p /= sum_p;
        if (p > 1e-12f) {
            result.entropy -= p * std::log(p);
        }
    }

    return result;
}

soft_token soft_thinking_sample(llama_context * ctx, const soft_thinking_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Get logits from last token position
    float * logits = llama_get_logits(ctx);

    // Apply softmax to convert logits to probabilities
    // First find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // Compute exp(logits - max) and sum
    std::vector<float> probs(n_vocab);
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] /= sum_exp;
    }

    // Build candidates array with probabilities
    std::vector<std::pair<float, llama_token>> candidates;
    candidates.reserve(n_vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        candidates.push_back({probs[id], id});
    }

    // Determine how many tokens we need to sort based on sampler type
    int sort_k;
    switch (params.sampler) {
        case SOFT_THINKING_SAMPLER_TOP_K:
            sort_k = params.top_k;
            break;
        case SOFT_THINKING_SAMPLER_ENTROPY_PRESERVING:
            sort_k = params.max_k;
            break;
        default:
            sort_k = params.max_k;
            break;
    }

    // Partial sort to get top candidates (more efficient than full sort)
    sort_k = std::min(sort_k, (int)candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + sort_k, candidates.end(),
        [](const std::pair<float, llama_token> & a, const std::pair<float, llama_token> & b) {
            return a.first > b.first;  // Descending by probability
        }
    );

    // Sample based on sampler type
    switch (params.sampler) {
        case SOFT_THINKING_SAMPLER_TOP_K:
            return sample_top_k(candidates, params.top_k);

        case SOFT_THINKING_SAMPLER_ENTROPY_PRESERVING: {
            float full_entropy = compute_full_entropy(probs);
            return sample_entropy_preserving(
                candidates,
                full_entropy,
                params.entropy_frac,
                params.min_k,
                params.max_k
            );
        }

        default:
            // Fallback to entropy-preserving
            float full_entropy = compute_full_entropy(probs);
            return sample_entropy_preserving(
                candidates,
                full_entropy,
                params.entropy_frac,
                params.min_k,
                params.max_k
            );
    }
}

// Get dequantization function for a GGML type
typedef void (*dequantize_fn)(const void * src, float * dst, int64_t k);

static dequantize_fn get_dequantize_fn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return [](const void * src, float * dst, int64_t k) {
                memcpy(dst, src, k * sizeof(float));
            };
        case GGML_TYPE_F16:
            return [](const void * src, float * dst, int64_t k) {
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)src;
                for (int64_t i = 0; i < k; ++i) {
                    dst[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            };
        case GGML_TYPE_BF16:
            return [](const void * src, float * dst, int64_t k) {
                const ggml_bf16_t * src_bf16 = (const ggml_bf16_t *)src;
                for (int64_t i = 0; i < k; ++i) {
                    dst[i] = ggml_bf16_to_fp32(src_bf16[i]);
                }
            };
        default:
            // For quantized types, use the type traits
            const ggml_type_traits * traits = ggml_get_type_traits(type);
            if (traits && traits->to_float) {
                return (dequantize_fn)traits->to_float;
            }
            return nullptr;
    }
}

float * soft_thinking_compute_embedding(
    soft_thinking_context       & st_ctx,
    const llama_model           * model,
    const soft_token            & token,
    const soft_thinking_params  & params
) {
    const int n_embd = llama_model_n_embd(model);

    // Clear buffer
    std::fill(st_ctx.embedding_buffer.begin(), st_ctx.embedding_buffer.end(), 0.0f);

    // Get token embedding tensor directly from the model
    // The model struct is defined in llama-model.h
    ggml_tensor * embd_tensor = model->tok_embd;

    if (embd_tensor == nullptr) {
        throw std::runtime_error("Could not find token embedding tensor in model");
    }

    // Get dequantization function
    dequantize_fn dequantize = get_dequantize_fn(embd_tensor->type);
    if (dequantize == nullptr) {
        throw std::runtime_error("Unsupported embedding tensor type for dequantization");
    }

    // Calculate row size in bytes
    const size_t row_size = ggml_row_size(embd_tensor->type, n_embd);

    // Temporary buffer for one dequantized row
    std::vector<float> row_buf(n_embd);

    // Compute mixture weights with sharpening (Fix 3)
    std::vector<float> weights = token.probs;

    float alpha;
    if (params.adaptive_alpha && !token.probs.empty()) {
        // Adaptive mode (default): interpolate between alpha_low and alpha_high based on entropy
        // Compute normalized entropy (0 = certain, 1 = max uncertainty)
        float max_entropy = std::log((float)token.probs.size());
        float normalized_entropy = (max_entropy > 0.0f) ? (token.entropy / max_entropy) : 0.0f;
        // High entropy -> use alpha_low (softer), low entropy -> use alpha_high (sharper)
        alpha = params.alpha_low + (1.0f - normalized_entropy) * (params.alpha_high - params.alpha_low);
    } else {
        // Static mode: use alpha_high as the fixed exponent
        alpha = params.alpha_high;
    }

    if (alpha != 1.0f && !weights.empty()) {
        float sum_w = 0.0f;
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = std::pow(weights[i], alpha);
            sum_w += weights[i];
        }
        // Renormalize
        if (sum_w > 0.0f) {
            for (float & w : weights) {
                w /= sum_w;
            }
        }
    }

    // Accumulate weighted embeddings with (optionally sharpened) weights
    for (size_t i = 0; i < token.indices.size(); ++i) {
        llama_token id = token.indices[i];
        float weight = weights[i];

        // Calculate pointer to this token's embedding row
        const void * row_ptr = (const char *)embd_tensor->data + (id * row_size);

        // Dequantize to float
        dequantize(row_ptr, row_buf.data(), n_embd);

        // Weighted accumulation
        for (int j = 0; j < n_embd; ++j) {
            st_ctx.embedding_buffer[j] += row_buf[j] * weight;
        }
    }

    return st_ctx.embedding_buffer.data();
}

// Legacy overload without params (uses default alpha=1.0, no sharpening)
float * soft_thinking_compute_embedding(
    soft_thinking_context & st_ctx,
    const llama_model     * model,
    const soft_token      & token
) {
    soft_thinking_params default_params;
    return soft_thinking_compute_embedding(st_ctx, model, token, default_params);
}


bool soft_thinking_check_cold_stop(
    soft_thinking_context       & st_ctx,
    const soft_thinking_params  & params,
    float                         entropy
) {
    if (entropy < params.entropy_threshold) {
        st_ctx.low_entropy_count++;
    } else {
        st_ctx.low_entropy_count = 0;
    }

    if (st_ctx.low_entropy_count >= params.cold_stop_steps) {
        st_ctx.state = soft_thinking_state::ANSWERING;
        st_ctx.low_entropy_count = 0;
        return true;  // Should stop thinking
    }

    return false;
}


void soft_thinking_reset(soft_thinking_context & ctx) {
    ctx.state = soft_thinking_state::THINKING;
    ctx.low_entropy_count = 0;
    std::fill(ctx.embedding_buffer.begin(), ctx.embedding_buffer.end(), 0.0f);
}


llama_batch soft_thinking_prepare_batch(
    const float * embd,
    int           n_embd,
    llama_pos     pos,
    llama_seq_id  seq_id
) {
    // Allocate batch with embedding support (embd != 0)
    llama_batch batch = llama_batch_init(1, n_embd, 1);

    // Copy embedding data
    memcpy(batch.embd, embd, n_embd * sizeof(float));

    // Set batch parameters
    batch.n_tokens = 1;
    batch.token = nullptr;  // Signal that we're using embeddings, not tokens
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0] = 1;  // We want logits for this position

    return batch;
}


soft_token_class soft_thinking_classify_token(const char * text, llama_token_attr attr) {
    if (text == nullptr || text[0] == '\0') {
        return TOKEN_CLASS_UNKNOWN;
    }

    // Check for control tokens via attribute
    if (attr & LLAMA_TOKEN_ATTR_CONTROL) {
        return TOKEN_CLASS_CONTROL;
    }

    // Analyze first character for primary classification
    unsigned char c = (unsigned char)text[0];

    // Whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        return TOKEN_CLASS_WHITESPACE;
    }

    // Digits (including tokens starting with digits)
    if (c >= '0' && c <= '9') {
        return TOKEN_CLASS_DIGIT;
    }

    // Brackets
    if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}') {
        return TOKEN_CLASS_BRACKET;
    }

    // Operators
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '=' ||
        c == '<' || c == '>' || c == '!' || c == '&' || c == '|' ||
        c == '^' || c == '%' || c == '~') {
        return TOKEN_CLASS_OPERATOR;
    }

    // Punctuation
    if (c == ',' || c == '.' || c == ';' || c == ':' || c == '?' ||
        c == '\'' || c == '"' || c == '`') {
        return TOKEN_CLASS_PUNCT;
    }

    // Alpha (letters, underscore, or extended Unicode likely to be text)
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 128) {
        return TOKEN_CLASS_ALPHA;
    }

    return TOKEN_CLASS_UNKNOWN;
}

bool soft_thinking_classes_compatible(soft_token_class c1, soft_token_class c2) {
    if (c1 == c2) return true;
    if (c1 == TOKEN_CLASS_UNKNOWN || c2 == TOKEN_CLASS_UNKNOWN) return true;

    // Specific compatibility rules for code/math:
    // - Digits can mix with punctuation (for decimals like "3.14")
    if ((c1 == TOKEN_CLASS_DIGIT && c2 == TOKEN_CLASS_PUNCT) ||
        (c1 == TOKEN_CLASS_PUNCT && c2 == TOKEN_CLASS_DIGIT)) {
        return true;
    }

    return false;
}


soft_token soft_thinking_sample_from_sampler(
    common_sampler             * smpl,
    llama_context              * ctx,
    const soft_thinking_params & params,
    int                          tok_idx
) {
    // CRITICAL: common_sampler_sample() can advance RNG / mutate
    // sampler-internal state even if we don't call common_sampler_accept().
    // During soft thinking we want a read-only view of the policy-processed
    // candidate distribution (penalties, temperature, grammar, etc.) without
    // perturbing the real sampler used for ANSWERING.
    //
    // To make this non-invasive, we sample on a cloned sampler and read the
    // processed candidates from the clone.
    common_sampler_ptr smpl_tmp(common_sampler_clone(smpl));

    const llama_token sampled_id = common_sampler_sample(smpl_tmp.get(), ctx, tok_idx, /* grammar_first= */ false);

    // Get the processed candidate array (sorted by probability)
    llama_token_data_array * candidates = common_sampler_get_candidates(smpl_tmp.get(), /* do_sort= */ true);

    if (!candidates || candidates->size == 0) {
        // Fallback: return single-token result
        soft_token result;
        result.indices.push_back(sampled_id);
        result.probs.push_back(1.0f);
        result.entropy = 0.0f;
        result.top_id = sampled_id;
        return result;
    }

    // Get vocab for token classification if needed
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Apply token class filtering if enabled (Fix 4)
    std::vector<llama_token_data> filtered_candidates;
    if (params.class_filtering && candidates->size > 0) {
        // Classify the top token
        const char * top_text = llama_vocab_get_text(vocab, candidates->data[0].id);
        llama_token_attr top_attr = llama_vocab_get_attr(vocab, candidates->data[0].id);
        soft_token_class top_class = soft_thinking_classify_token(top_text, top_attr);

        // Filter to compatible classes
        filtered_candidates.reserve(candidates->size);
        for (size_t i = 0; i < candidates->size; ++i) {
            const char * text = llama_vocab_get_text(vocab, candidates->data[i].id);
            llama_token_attr attr = llama_vocab_get_attr(vocab, candidates->data[i].id);
            soft_token_class cls = soft_thinking_classify_token(text, attr);

            if (soft_thinking_classes_compatible(top_class, cls)) {
                filtered_candidates.push_back(candidates->data[i]);
            }
        }

        // Fallback if too few candidates after filtering
        if ((int)filtered_candidates.size() < params.class_min_k) {
            // Use original candidates without filtering
            filtered_candidates.clear();
            filtered_candidates.assign(candidates->data, candidates->data + candidates->size);
        } else {
            // Renormalize probabilities after filtering
            float sum_p = 0.0f;
            for (auto & c : filtered_candidates) {
                sum_p += c.p;
            }
            if (sum_p > 0.0f) {
                for (auto & c : filtered_candidates) {
                    c.p /= sum_p;
                }
            }
        }
    } else {
        filtered_candidates.assign(candidates->data, candidates->data + candidates->size);
    }

    // Compute full entropy from filtered candidates
    float full_entropy = 0.0f;
    for (const auto & c : filtered_candidates) {
        if (c.p > 1e-12f) {
            full_entropy -= c.p * std::log(c.p);
        }
    }

    // Determine concept width based on sampler type
    int k = 0;
    switch (params.sampler) {
        case SOFT_THINKING_SAMPLER_TOP_K:
            k = std::min(params.top_k, (int)filtered_candidates.size());
            break;

        case SOFT_THINKING_SAMPLER_ENTROPY_PRESERVING: {
            float target_entropy = params.entropy_frac * full_entropy;
            float captured_entropy = 0.0f;

            for (size_t i = 0; i < filtered_candidates.size() && k < params.max_k; ++i) {
                float p = filtered_candidates[i].p;
                if (p > 1e-12f) {
                    captured_entropy -= p * std::log(p);
                }
                k++;
                if (captured_entropy >= target_entropy && k >= params.min_k) {
                    break;
                }
            }
            k = std::max(k, std::min(params.min_k, (int)filtered_candidates.size()));
            break;
        }

        default:
            k = std::min(params.top_k, (int)filtered_candidates.size());
            break;
    }

    // Build soft_token from top-k candidates
    soft_token result;
    result.indices.reserve(k);
    result.probs.reserve(k);

    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        result.indices.push_back(filtered_candidates[i].id);
        result.probs.push_back(filtered_candidates[i].p);
        sum_p += filtered_candidates[i].p;
    }

    result.top_id = filtered_candidates[0].id;

    // Renormalize and compute concept entropy
    result.entropy = 0.0f;
    if (sum_p > 0.0f) {
        for (float & p : result.probs) {
            p /= sum_p;
            if (p > 1e-12f) {
                result.entropy -= p * std::log(p);
            }
        }
    }

    return result;
}

// ============================================================================
// Adaptive Cointegration Sampling (Entropy Feedback Control)
// ============================================================================

// compute normalized entropy from candidate array
static float compute_normalized_entropy(const llama_token_data * data, size_t size) {
    if (size <= 1) return 0.0f;

    float entropy = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float p = data[i].p;
        if (p > 1e-12f) {
            entropy -= p * std::log(p);
        }
    }
    // Normalize by max possible entropy for this support size
    float max_entropy = std::log((float)size);
    return (max_entropy > 0.0f) ? (entropy / max_entropy) : 0.0f;
}

// compute normalized entropy from soft_token (after sharpening)
static float compute_injected_entropy(const soft_token & st) {
    if (st.probs.size() <= 1) return 0.0f;

    float entropy = 0.0f;
    for (float p : st.probs) {
        if (p > 1e-12f) {
            entropy -= p * std::log(p);
        }
    }
    float max_entropy = std::log((float)st.probs.size());
    return (max_entropy > 0.0f) ? (entropy / max_entropy) : 0.0f;
}

soft_token soft_thinking_sample_cointegrated(
    common_sampler             * smpl,
    llama_context              * ctx,
    soft_thinking_context      & st_ctx,
    const soft_thinking_params & params,
    int                          tok_idx
) {
    // =========================================================================
    // Step 0: Observe model distribution (response to previous injected concept)
    // =========================================================================
    // Clone sampler to avoid mutating RNG state
    common_sampler_ptr smpl_tmp(common_sampler_clone(smpl));
    const llama_token sampled_id = common_sampler_sample(smpl_tmp.get(), ctx, tok_idx, false);
    llama_token_data_array * candidates = common_sampler_get_candidates(smpl_tmp.get(), true);

    if (!candidates || candidates->size == 0) {
        soft_token result;
        result.indices.push_back(sampled_id);
        result.probs.push_back(1.0f);
        result.entropy = 0.0f;
        result.top_id = sampled_id;
        return result;
    }

    // =========================================================================
    // Step 1: Compute H_out on FIXED support size (stability)
    // =========================================================================
    // Use a fixed support size to avoid jitter from variable candidate set sizes
    const int N_out = std::min((int)candidates->size, params.coint_entropy_support);
    float H_out = compute_normalized_entropy(candidates->data, N_out);

    // =========================================================================
    // Step 2: Entropy feedback control
    // =========================================================================
    // Only apply feedback after warmup AND when we have a valid previous H_in
    if (params.cointegration) {
        if (st_ctx.warmup_steps < params.coint_warmup_steps || !std::isfinite(st_ctx.prev_injected_entropy)) {
            // Warmup: just increment counter, initialize adaptive params if first step
            if (st_ctx.warmup_steps == 0) {
                st_ctx.adaptive_k = (float)(params.min_k + params.max_k) / 2.0f;
                st_ctx.adaptive_alpha = (params.alpha_low + params.alpha_high) / 2.0f;
            }
            st_ctx.warmup_steps++;
        } else {
            // Error signal: e = H_out(t) - H_in(t-1)
            // H_out is model's response to the concept we injected last step
            float error = H_out - st_ctx.prev_injected_entropy;

            // EMA smoothing
            st_ctx.ema_error = params.coint_ema_beta * error +
                               (1.0f - params.coint_ema_beta) * st_ctx.ema_error;

            // Compute adjustments (negative feedback)
            // e > 0 (model more uncertain than input): tighten (k↓, α↑)
            // e < 0 (model more certain than input): loosen (k↑, α↓)
            float dk = -params.coint_k_gain * st_ctx.ema_error;
            float da = params.coint_alpha_gain * st_ctx.ema_error;

            // Rate limiting to prevent chatter
            dk = std::clamp(dk, -params.coint_rate_limit, params.coint_rate_limit);
            da = std::clamp(da, -params.coint_alpha_rate_limit, params.coint_alpha_rate_limit);

            // Apply adjustments
            st_ctx.adaptive_k += dk;
            st_ctx.adaptive_alpha += da;

            // Clamp to bounds
            st_ctx.adaptive_k = std::clamp(st_ctx.adaptive_k, (float)params.min_k, (float)params.max_k);
            st_ctx.adaptive_alpha = std::clamp(st_ctx.adaptive_alpha, params.alpha_low, params.alpha_high);
        }
    }

    // =========================================================================
    // Step 3: Build concept token with adaptive k
    // =========================================================================
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Apply token class filtering if enabled
    std::vector<llama_token_data> filtered_candidates;
    if (params.class_filtering && candidates->size > 0) {
        const char * top_text = llama_vocab_get_text(vocab, candidates->data[0].id);
        llama_token_attr top_attr = llama_vocab_get_attr(vocab, candidates->data[0].id);
        soft_token_class top_class = soft_thinking_classify_token(top_text, top_attr);

        filtered_candidates.reserve(candidates->size);
        for (size_t i = 0; i < candidates->size; ++i) {
            const char * text = llama_vocab_get_text(vocab, candidates->data[i].id);
            llama_token_attr attr = llama_vocab_get_attr(vocab, candidates->data[i].id);
            soft_token_class cls = soft_thinking_classify_token(text, attr);

            if (soft_thinking_classes_compatible(top_class, cls)) {
                filtered_candidates.push_back(candidates->data[i]);
            }
        }

        if ((int)filtered_candidates.size() < params.class_min_k) {
            filtered_candidates.assign(candidates->data, candidates->data + candidates->size);
        } else {
            float sum_p = 0.0f;
            for (auto & c : filtered_candidates) sum_p += c.p;
            if (sum_p > 0.0f) {
                for (auto & c : filtered_candidates) c.p /= sum_p;
            }
        }
    } else {
        filtered_candidates.assign(candidates->data, candidates->data + candidates->size);
    }

    // Use adaptive k (floored to int)
    int k = (int)st_ctx.adaptive_k;
    k = std::min(k, (int)filtered_candidates.size());
    k = std::max(k, 1);

    // Build soft_token
    soft_token result;
    result.indices.reserve(k);
    result.probs.reserve(k);

    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        result.indices.push_back(filtered_candidates[i].id);
        result.probs.push_back(filtered_candidates[i].p);
        sum_p += filtered_candidates[i].p;
    }
    result.top_id = filtered_candidates[0].id;

    // Renormalize
    if (sum_p > 0.0f) {
        for (float & p : result.probs) p /= sum_p;
    }

    // =========================================================================
    // Step 4: Apply adaptive alpha sharpening and compute H_in for next step
    // =========================================================================
    float alpha = st_ctx.adaptive_alpha;
    if (alpha != 1.0f && result.probs.size() > 1) {
        float sum_w = 0.0f;
        for (float & p : result.probs) {
            p = std::pow(p, alpha);
            sum_w += p;
        }
        if (sum_w > 0.0f) {
            for (float & p : result.probs) p /= sum_w;
        }
    }

    // Compute entropy of the sharpened concept (this is H_in for next step)
    result.entropy = 0.0f;
    for (float p : result.probs) {
        if (p > 1e-12f) {
            result.entropy -= p * std::log(p);
        }
    }

    // =========================================================================
    // Step 5: Store H_in(t) for feedback control on next step
    // =========================================================================
    // This is the normalized entropy of the concept we're about to inject
    st_ctx.prev_injected_entropy = compute_injected_entropy(result);

    return result;
}
