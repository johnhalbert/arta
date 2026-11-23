use anyhow::{Result, ensure};
use llama_cpp_4::{
    context::params::{LlamaAttentionType, LlamaFlashAttnType, RopeScalingType},
    prelude::{GgmlType, LlamaContext, LlamaContextParams},
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextConfig {
    pub n_batch: Option<NonZeroU32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub n_seq_max: Option<NonZeroU32>,
    pub n_rs_seq: Option<u32>,
    pub n_outputs_max: Option<u32>,
    pub n_outputs_max_per_seq: Option<u32>,
    pub attention_type: Option<AttentionType>,
    pub flash_attention: Option<FlashAttention>,
    pub cache_type_k: Option<CacheType>,
    pub cache_type_v: Option<CacheType>,
    pub rope_scaling_type: Option<RopeScaling>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub yarn_ext_factor: Option<f32>,
    pub yarn_attn_factor: Option<f32>,
    pub yarn_beta_fast: Option<f32>,
    pub yarn_beta_slow: Option<f32>,
    pub yarn_orig_ctx: Option<u32>,
    pub offload_kqv: Option<bool>,
    pub op_offload: Option<bool>,
    pub swa_full: Option<bool>,
    pub kv_unified: Option<bool>,
    pub moe_cache_size: Option<usize>,
    pub no_perf: Option<bool>,
}

impl ContextConfig {
    pub(in crate::provider::llama_cpp) fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("n_threads", self.n_threads),
            ("n_threads_batch", self.n_threads_batch),
        ] {
            ensure!(
                value.is_none_or(|value| value > 0),
                "context.{name} must be positive"
            );
        }
        for (name, value) in [
            ("rope_freq_base", self.rope_freq_base),
            ("rope_freq_scale", self.rope_freq_scale),
            ("yarn_ext_factor", self.yarn_ext_factor),
            ("yarn_attn_factor", self.yarn_attn_factor),
            ("yarn_beta_fast", self.yarn_beta_fast),
            ("yarn_beta_slow", self.yarn_beta_slow),
        ] {
            ensure!(
                value.is_none_or(f32::is_finite),
                "context.{name} must be finite"
            );
        }
        ensure!(
            !matches!(self.flash_attention, Some(FlashAttention::Disabled))
                || self.cache_type_v.is_none_or(|cache| !cache.is_quantized()),
            "context.cache_type_v needs Flash Attention when quantized"
        );
        Ok(())
    }

    pub(in crate::provider::llama_cpp) fn params(&self, length: NonZeroU32) -> LlamaContextParams {
        let mut params = LlamaContextParams::default().with_n_ctx(Some(length));
        macro_rules! apply {
            ($($field:ident => $setter:ident $(, $convert:expr)?;)*) => {
                $(
                    if let Some(value) = self.$field {
                        $(let value = $convert(value);)?
                        params = params.$setter(value);
                    }
                )*
            };
        }
        apply! {
            n_batch => with_n_batch, NonZeroU32::get;
            n_ubatch => with_n_ubatch;
            n_threads => with_n_threads;
            n_threads_batch => with_n_threads_batch;
            n_seq_max => with_n_seq_max, NonZeroU32::get;
            n_rs_seq => with_n_rs_seq;
            n_outputs_max => with_n_outputs_max;
            n_outputs_max_per_seq => with_n_outputs_max_per_seq;
            attention_type => with_attention_type, LlamaAttentionType::from;
            flash_attention => with_flash_attn_type, LlamaFlashAttnType::from;
            cache_type_k => with_cache_type_k, GgmlType::from;
            cache_type_v => with_cache_type_v, GgmlType::from;
            rope_scaling_type => with_rope_scaling_type, RopeScalingType::from;
            rope_freq_base => with_rope_freq_base;
            rope_freq_scale => with_rope_freq_scale;
            yarn_ext_factor => with_yarn_ext_factor;
            yarn_attn_factor => with_yarn_attn_factor;
            yarn_beta_fast => with_yarn_beta_fast;
            yarn_beta_slow => with_yarn_beta_slow;
            yarn_orig_ctx => with_yarn_orig_ctx;
            offload_kqv => with_offload_kqv;
            op_offload => with_op_offload;
            swa_full => with_swa_full;
            kv_unified => with_kv_unified;
            moe_cache_size => with_moe_cache_size;
            no_perf => with_no_perf;
        }
        params
    }

    pub(in crate::provider::llama_cpp) fn prompt_batch_size(
        &self,
        context: &LlamaContext<'_>,
        token_count: usize,
    ) -> usize {
        let batch = context.n_batch();
        let batch = if matches!(self.attention_type, Some(AttentionType::NonCausal)) {
            batch.min(context.n_ubatch())
        } else {
            batch
        };
        token_count.min(batch as usize)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    Unspecified,
    Causal,
    NonCausal,
}

impl From<AttentionType> for LlamaAttentionType {
    fn from(value: AttentionType) -> Self {
        match value {
            AttentionType::Unspecified => Self::Unspecified,
            AttentionType::Causal => Self::Causal,
            AttentionType::NonCausal => Self::NonCausal,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttention {
    Auto,
    Disabled,
    Enabled,
}

impl From<FlashAttention> for LlamaFlashAttnType {
    fn from(value: FlashAttention) -> Self {
        match value {
            FlashAttention::Auto => Self::Auto,
            FlashAttention::Disabled => Self::Disabled,
            FlashAttention::Enabled => Self::Enabled,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScaling {
    Unspecified,
    None,
    Linear,
    Yarn,
}

impl From<RopeScaling> for RopeScalingType {
    fn from(value: RopeScaling) -> Self {
        match value {
            RopeScaling::Unspecified => Self::Unspecified,
            RopeScaling::None => Self::None,
            RopeScaling::Linear => Self::Linear,
            RopeScaling::Yarn => Self::Yarn,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
    Q5_0,
    Q5_1,
}

impl CacheType {
    fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }
}

impl From<CacheType> for GgmlType {
    fn from(value: CacheType) -> Self {
        match value {
            CacheType::F32 => Self::F32,
            CacheType::F16 => Self::F16,
            CacheType::Bf16 => Self::BF16,
            CacheType::Q8_0 => Self::Q8_0,
            CacheType::Q4_0 => Self::Q4_0,
            CacheType::Q4_1 => Self::Q4_1,
            CacheType::Iq4Nl => Self::IQ4NL,
            CacheType::Q5_0 => Self::Q5_0,
            CacheType::Q5_1 => Self::Q5_1,
        }
    }
}
