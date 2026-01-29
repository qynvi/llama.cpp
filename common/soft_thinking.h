#ifndef SOFT_THINKING_H
#define SOFT_THINKING_H

#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <cmath>

// Forward declaration for sampler integration
struct common_sampler;

// Token classes for class-aware soft mixing (Fix 4)
enum soft_token_class {
    TOKEN_CLASS_UNKNOWN     = 0,
    TOKEN_CLASS_ALPHA       = 1,  // Letters/identifiers
    TOKEN_CLASS_DIGIT       = 2,  // Numbers
    TOKEN_CLASS_OPERATOR    = 3,  // + - * / = < > etc
    TOKEN_CLASS_BRACKET     = 4,  // () [] {}
    TOKEN_CLASS_PUNCT       = 5,  // , . ; :
    TOKEN_CLASS_WHITESPACE  = 6,  // Space, tab, newline
    TOKEN_CLASS_CONTROL     = 7,  // Special/control tokens
};

// Result of soft sampling - represents a "concept token"
struct soft_token {
    std::vector<llama_token> indices;  // Top-K token IDs
    std::vector<float>       probs;    // Normalized probabilities
    float                    entropy;  // Shannon entropy for Cold Stop
    llama_token              top_id;   // Highest probability token (for display)
};

// Configuration for soft thinking
struct soft_thinking_params {
    bool  enabled                = false;
    soft_thinking_sampler_type sampler = SOFT_THINKING_SAMPLER_ENTROPY_PRESERVING;

    // top-k sampler params
    int   top_k                  = 15;           // Number of tokens in concept (paper: 10-15)

    // entropy-preserving sampler params
    float entropy_frac           = 0.90f;        // Fraction of entropy to preserve (0.0-1.0)
    int   min_k                  = 3;            // Minimum tokens in concept
    int   max_k                  = 50;           // Maximum tokens in concept

    // cold stop params
    float entropy_threshold      = 0.1f;         // Cold Stop entropy threshold tau
    int   cold_stop_steps        = 256;          // Consecutive low-entropy steps before stopping
    // Note: End-of-thinking detection is now model-agnostic via llama_vocab_is_control()

    // Mixture sharpening params (Fix 3)
    // Default: adaptive alpha (varies based on entropy)
    // Use --soft-thinking-static-alpha to override with fixed values
    bool  adaptive_alpha         = true;         // If true, vary alpha based on entropy (default: on)
    float alpha_low              = 1.0f;         // Alpha when uncertainty is high
    float alpha_high             = 2.5f;         // Alpha when uncertainty is low

    // Token class filtering params (Fix 4)
    // Default: enabled (avoids mixing incompatible token classes)
    // Use --soft-thinking-smudge-tokens to disable
    bool  class_filtering        = true;         // Enable token class filtering (default: on)
    int   class_min_k            = 3;            // Minimum tokens after filtering (fallback if too few)

    // Adaptive cointegration params (entropy feedback control)
    // Negative feedback: e = H_out - H_in
    //   e > 0 (model more uncertain than input): tighten (k↓, α↑)
    //   e < 0 (model more certain than input): loosen (k↑, α↓)
    bool  cointegration          = true;         // Enable adaptive width/alpha via entropy feedback
    float coint_k_gain           = 2.0f;         // Gain for k adjustment
    float coint_alpha_gain       = 0.3f;         // Gain for alpha adjustment
    float coint_ema_beta         = 0.3f;         // EMA smoothing factor for error signal
    float coint_rate_limit       = 3.0f;         // Max change in k per step (rate limiter)
    float coint_alpha_rate_limit = 0.5f;         // Max change in alpha per step
    int   coint_warmup_steps     = 3;            // Steps before enabling feedback
    int   coint_entropy_support  = 64;           // Fixed support size for H_out measurement (stability)
};

// State machine for inference loop
enum class soft_thinking_state {
    THINKING,    // Using soft concept tokens
    ANSWERING    // Using standard discrete tokens
};

// Runtime state
struct soft_thinking_context {
    soft_thinking_state state              = soft_thinking_state::THINKING;
    int                 low_entropy_count  = 0;
    std::vector<float>  embedding_buffer;  // Persistent buffer for weighted embedding

    // Adaptive cointegration state (entropy feedback control)
    float adaptive_k            = 10.0f;   // Continuous concept width
    float adaptive_alpha        = 1.5f;    // Continuous sharpening exponent
    float ema_error             = 0.0f;    // Smoothed entropy error (H_out - H_in)
    float prev_injected_entropy = NAN;     // H_in(t-1): entropy of previous injected concept (NAN until first injection)
    int   warmup_steps          = 0;       // Steps before enabling feedback (warm start)
};

// ============================================================================
// Core Functions
// ============================================================================

// Initialize context (call once before inference loop)
void soft_thinking_init(
    soft_thinking_context & ctx,
    const llama_model     * model,
    soft_thinking_params  & params
);

// Perform soft sampling: get concept token based on sampler type
soft_token soft_thinking_sample(
    llama_context              * ctx,
    const soft_thinking_params & params
);

// Compute weighted embedding from soft token with optional sharpening (Fix 3)
// Returns pointer to internal buffer (valid until next call)
float * soft_thinking_compute_embedding(
    soft_thinking_context       & st_ctx,
    const llama_model           * model,
    const soft_token            & token,
    const soft_thinking_params  & params
);

// Legacy overload without params (uses default alpha=1.0, no sharpening)
float * soft_thinking_compute_embedding(
    soft_thinking_context & st_ctx,
    const llama_model     * model,
    const soft_token      & token
);

// Check Cold Stop condition, update state
// Returns true if should stop thinking
bool soft_thinking_check_cold_stop(
    soft_thinking_context       & st_ctx,
    const soft_thinking_params  & params,
    float                         entropy
);

// Reset state (for new conversation/prompt)
void soft_thinking_reset(soft_thinking_context & ctx);

// Prepare a batch with embedding instead of tokens
// Returns an initialized batch with the embedding set
// Caller is responsible for calling llama_batch_free() on the returned batch
llama_batch soft_thinking_prepare_batch(
    const float * embd,
    int           n_embd,
    llama_pos     pos,
    llama_seq_id  seq_id
);

// ============================================================================
// Policy-Consistent Sampling (Fix 2)
// ============================================================================

// Perform soft sampling using the sampler's processed candidates
// This ensures the concept distribution respects penalties, temperature, grammar, etc.
// NOTE: Does NOT call common_sampler_accept() - caller handles that appropriately
soft_token soft_thinking_sample_from_sampler(
    common_sampler             * smpl,
    llama_context              * ctx,
    const soft_thinking_params & params,
    int                          tok_idx
);

// Adaptive cointegration sampling with entropy feedback control
// Uses negative feedback: e = H_out - H_in
//   e > 0 (model more uncertain than input): tighten (k↓, α↑)
//   e < 0 (model more certain than input): loosen (k↑, α↓)
// This couples concept width with the model's internal certainty trajectory
soft_token soft_thinking_sample_cointegrated(
    common_sampler             * smpl,
    llama_context              * ctx,
    soft_thinking_context      & st_ctx,
    const soft_thinking_params & params,
    int                          tok_idx
);

// ============================================================================
// Token Class Classification (Fix 4)
// ============================================================================

// Classify a token by its text for class-aware soft mixing
soft_token_class soft_thinking_classify_token(const char * text, llama_token_attr attr);

// Check if two token classes are compatible for mixing
bool soft_thinking_classes_compatible(soft_token_class c1, soft_token_class c2);

#endif // SOFT_THINKING_H
