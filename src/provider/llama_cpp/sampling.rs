//! Token selection shared by text and tool inference. An absent sampling table chooses a
//! greedy sampler. A present table configures the native common sampler and seeds its history
//! with the prompt tokens.

use super::Sampling;
use anyhow::{Context, Result, ensure};
use llama_cpp_4::{
    chat::json_schema_to_grammar,
    common_sampler::{CommonSampler, CommonSamplerParams, CommonSamplerType, GrammarSource},
    prelude::{AddBos, ChatParams, LlamaContext, LlamaModel, LlamaSampler, LlamaToken, Special},
};
use serde::{Deserialize, Serialize};

/// Optional native sampler settings. Omitted scalar fields inherit common-sampler defaults.
/// Grammar selection depends on whether the request exposes tools.
#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingConfig {
    /// Random number generator seed for reproducible sampling under the same backend
    /// conditions.
    pub seed: Option<u32>,
    /// Ordered native stages. Omission preserves the default chain.
    pub samplers: Option<Vec<SamplerStage>>,
    /// Minimum number of candidates retained by filters that support it.
    pub min_keep: Option<i32>,
    /// Retain up to this many highest-scoring candidates in the top-k stage.
    pub top_k: Option<i32>,
    /// Nucleus sampling probability mass cutoff.
    pub top_p: Option<f32>,
    /// Minimum probability relative to the most probable candidate.
    pub min_p: Option<f32>,
    /// Chance of applying the exclude-top-choices stage.
    pub xtc_probability: Option<f32>,
    /// Probability threshold used to identify candidates for exclusion by XTC.
    pub xtc_threshold: Option<f32>,
    /// Probability mass retained by locally typical sampling.
    pub typ_p: Option<f32>,
    /// Temperature applied to logits. Zero selects greedy decoding with the default stage
    /// order.
    pub temp: Option<f32>,
    /// Range over which the dynamic temperature stage can vary the base temperature.
    pub dynatemp_range: Option<f32>,
    /// Exponent mapping entropy to a dynamic temperature adjustment.
    pub dynatemp_exponent: Option<f32>,
    /// Recent token window for repetition, frequency, and presence penalties.
    pub penalty_last_n: Option<i32>,
    /// Repetition penalty factor. One is neutral.
    pub penalty_repeat: Option<f32>,
    /// Penalty proportional to token frequency in the penalty window.
    pub penalty_freq: Option<f32>,
    /// Penalty for a token already present in the window.
    pub penalty_present: Option<f32>,
    /// Strength of DRY sequence-repetition penalties.
    pub dry_multiplier: Option<f32>,
    /// Base of the exponential penalty for extending repeated sequences.
    pub dry_base: Option<f32>,
    /// Length of a repeated sequence allowed before DRY applies a penalty.
    pub dry_allowed_length: Option<i32>,
    /// Number of recent tokens scanned for repeated sequences.
    pub dry_penalty_last_n: Option<i32>,
    /// String boundaries that break sequences considered by DRY.
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Target token probability for the adaptive-p stage.
    pub adaptive_target: Option<f32>,
    /// Decay of the moving average used for adaptive sampling.
    pub adaptive_decay: Option<f32>,
    /// Native Mirostat mode. Zero disables it, one and two select its two algorithms.
    pub mirostat: Option<i32>,
    /// Standard-deviation threshold used by the top-n-sigma stage.
    pub top_n_sigma: Option<f32>,
    /// Target entropy for Mirostat.
    pub mirostat_tau: Option<f32>,
    /// Learning rate for Mirostat adaptation.
    pub mirostat_eta: Option<f32>,
    /// Disable native sampler performance measurement.
    pub no_perf: Option<bool>,
    /// Suppress end-of-generation candidates with negative-infinity bias when no grammar
    /// constrains the step.
    pub ignore_eos: bool,
    /// User grammar for ordinary text. Mutually exclusive with JSON Schema.
    pub grammar: Option<String>,
    /// JSON Schema source converted to a grammar for ordinary text generation.
    pub json_schema: Option<String>,
    /// Apply grammar filtering first when sampling with the common sampler. Defaults to true
    /// in Arta.
    pub grammar_first: Option<bool>,
    /// Nonnegative values enable a thinking-token budget and require template reasoning tags.
    pub reasoning_budget_tokens: Option<i32>,
    /// Text included in the forced end sequence before the first closing reasoning tag.
    pub reasoning_budget_message: Option<String>,
    /// Per-token score adjustments applied after validating IDs against the loaded
    /// vocabulary.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub logit_bias: Vec<LogitBias>,
}

impl SamplingConfig {
    /// Check probability ranges, finite parameters, repetition penalty bounds, Mirostat
    /// modes, and grammar exclusivity before constructing the provider.
    pub(super) fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("top_p", self.top_p),
            ("min_p", self.min_p),
            ("typ_p", self.typ_p),
            ("xtc_probability", self.xtc_probability),
            ("xtc_threshold", self.xtc_threshold),
            ("adaptive_decay", self.adaptive_decay),
        ] {
            ensure!(
                value.is_none_or(|value| (0.0..=1.0).contains(&value)),
                "sampling.{name} must be between 0 and 1"
            );
        }
        for (name, value) in [
            ("temp", self.temp),
            ("dynatemp_range", self.dynatemp_range),
            ("dynatemp_exponent", self.dynatemp_exponent),
            ("penalty_repeat", self.penalty_repeat),
            ("penalty_freq", self.penalty_freq),
            ("penalty_present", self.penalty_present),
            ("dry_multiplier", self.dry_multiplier),
            ("dry_base", self.dry_base),
            ("adaptive_target", self.adaptive_target),
            ("top_n_sigma", self.top_n_sigma),
            ("mirostat_tau", self.mirostat_tau),
            ("mirostat_eta", self.mirostat_eta),
        ] {
            ensure!(
                value.is_none_or(f32::is_finite),
                "sampling.{name} must be finite"
            );
        }
        ensure!(
            self.mirostat.is_none_or(|mode| (0..=2).contains(&mode)),
            "sampling.mirostat must be 0, 1, or 2"
        );
        ensure!(
            self.min_keep.is_none_or(|count| count >= 0),
            "sampling.min_keep must be nonnegative"
        );
        ensure!(
            self.penalty_repeat
                .is_none_or(|value| value > 0.0 && value.recip().is_finite()),
            "sampling.penalty_repeat must be positive with a finite reciprocal"
        );
        ensure!(
            self.grammar.is_none() || self.json_schema.is_none(),
            "set only one of sampling.grammar and sampling.json_schema"
        );
        Ok(())
    }

    /// Overlay supplied parameters on native defaults. Use user grammar or converted schema
    /// for text and the template's grammar and triggers for tools. Validate
    /// vocabulary-dependent biases here.
    fn params(&self, model: &LlamaModel, sampling: &Sampling<'_>) -> Result<CommonSamplerParams> {
        let mut params = CommonSamplerParams::new();
        let mut scalars = params.scalars();
        macro_rules! override_scalars {
            ($($field:ident),* $(,)?) => {
                $(if let Some(value) = self.$field {
                    scalars.$field = value;
                })*
            };
        }
        override_scalars!(
            seed,
            min_keep,
            top_k,
            top_p,
            min_p,
            xtc_probability,
            xtc_threshold,
            typ_p,
            temp,
            dynatemp_range,
            dynatemp_exponent,
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            adaptive_target,
            adaptive_decay,
            mirostat,
            top_n_sigma,
            mirostat_tau,
            mirostat_eta,
            no_perf,
            reasoning_budget_tokens,
        );
        params.set_scalars(&scalars);

        if let Some(order) = &self.samplers {
            let order: Vec<_> = order.iter().copied().map(CommonSamplerType::from).collect();
            params.set_samplers(&order)?;
        }
        if let Some(breakers) = &self.dry_sequence_breakers {
            let breakers: Vec<_> = breakers.iter().map(String::as_str).collect();
            params.set_dry_breakers(&breakers)?;
        }
        let (chat, constrained) = match sampling {
            Sampling::Text(chat) => {
                let json_grammar = self
                    .json_schema
                    .as_deref()
                    .map(|schema| json_schema_to_grammar(schema, true))
                    .transpose()
                    .context("invalid sampling.json_schema")?;
                let grammar = self.grammar.as_deref().or(json_grammar.as_deref());
                if let Some(grammar) = grammar {
                    params.set_grammar(grammar, GrammarSource::User, false)?;
                }
                (*chat, grammar.is_some())
            }
            Sampling::Tools { chat, .. } => {
                if !chat.grammar.is_empty() {
                    params.set_grammar(
                        &chat.grammar,
                        GrammarSource::ToolCalls,
                        chat.grammar_lazy,
                    )?;
                    for trigger in &chat.grammar_triggers {
                        params.add_grammar_trigger(
                            &trigger.kind,
                            &trigger.value,
                            LlamaToken(trigger.token),
                        )?;
                    }
                }
                (Some(*chat), !chat.grammar.is_empty())
            }
        };
        if let Some(chat) = chat {
            params.set_generation_prompt(chat.generation_prompt())?;
        }
        if self
            .reasoning_budget_tokens
            .is_some_and(|budget| budget >= 0)
        {
            let chat = chat.context("a sampling reasoning budget requires a chat template")?;
            self.apply_reasoning_budget(&mut params, model, chat)?;
        }
        let vocabulary_size = model.n_vocab();
        for bias in &self.logit_bias {
            ensure!(
                (0..vocabulary_size).contains(&bias.token),
                "sampling.logit_bias token {} is outside the model vocabulary",
                bias.token
            );
            ensure!(
                bias.bias.is_finite() || bias.bias == f32::NEG_INFINITY,
                "sampling.logit_bias must be finite or negative infinity"
            );
            params.add_logit_bias(LlamaToken(bias.token), bias.bias)?;
        }
        if self.ignore_eos && !constrained {
            for token in (0..vocabulary_size)
                .map(LlamaToken)
                .filter(|&token| model.is_eog_token(token))
            {
                params.add_logit_bias(token, f32::NEG_INFINITY)?;
            }
        }
        Ok(params)
    }

    /// Tokenize the template's opening and closing reasoning markers without BOS. Configure
    /// the forced exit from reasoning using the supplied message and first closing marker.
    fn apply_reasoning_budget(
        &self,
        params: &mut CommonSamplerParams,
        model: &LlamaModel,
        chat: &ChatParams,
    ) -> Result<()> {
        let end_tags: Vec<String> = serde_json::from_str(&chat.thinking_end_tags_json)
            .context("failed to read chat template reasoning tags")?;
        ensure!(
            !chat.thinking_start_tag.is_empty()
                && end_tags.first().is_some_and(|tag| !tag.is_empty()),
            "the chat template does not supply reasoning tags for sampling.reasoning_budget_tokens"
        );
        let tokenize =
            |text: &str| model.str_to_token_with_special(text, AddBos::Never, Special::Tokenize);
        let start = tokenize(&chat.thinking_start_tag)?;
        let ends = end_tags
            .iter()
            .map(|tag| tokenize(tag))
            .collect::<Result<Vec<_>, _>>()?;
        let message = self.reasoning_budget_message.as_deref().unwrap_or_default();
        let forced = tokenize(&format!("{message}{}", end_tags[0]))?;
        params.set_reasoning_budget(&start, &ends, &forced, message)?;
        Ok(())
    }
}

/// Adjustment to one vocabulary token's score before selection.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogitBias {
    /// Token ID within the loaded model's vocabulary.
    pub token: i32,
    /// Finite score adjustment or negative infinity to suppress the token.
    pub bias: f32,
}

/// Selectable stages in the order passed to the native common sampler.
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerStage {
    /// Apply repetition, frequency, and presence penalties.
    Penalties,
    /// Penalize extensions of repeated sequences.
    Dry,
    /// Filter candidates using the configured score deviation cutoff.
    TopNSigma,
    /// Keep the highest-scoring candidates by count.
    TopK,
    /// Filter candidates by typicality.
    TypicalP,
    /// Filter candidates by cumulative probability.
    TopP,
    /// Filter by probability relative to the best candidate.
    MinP,
    /// Probabilistically exclude top choices.
    Xtc,
    /// Rescale logits by temperature.
    Temperature,
    /// Apply the native infilling sampler.
    Infill,
    /// Adapt sampling toward the configured probability target.
    AdaptiveP,
}

impl From<SamplerStage> for CommonSamplerType {
    fn from(value: SamplerStage) -> Self {
        match value {
            SamplerStage::Penalties => Self::Penalties,
            SamplerStage::Dry => Self::Dry,
            SamplerStage::TopNSigma => Self::TopNSigma,
            SamplerStage::TopK => Self::TopK,
            SamplerStage::TypicalP => Self::TypicalP,
            SamplerStage::TopP => Self::TopP,
            SamplerStage::MinP => Self::MinP,
            SamplerStage::Xtc => Self::Xtc,
            SamplerStage::Temperature => Self::Temperature,
            SamplerStage::Infill => Self::Infill,
            SamplerStage::AdaptiveP => Self::AdaptiveP,
        }
    }
}

/// Greedy sampler (optionally chained with tool grammar) or common sampler with its
/// grammar-first flag.
pub(super) enum Sampler {
    /// Highest-score sampler, optionally preceded by a native tool grammar filter.
    Greedy(LlamaSampler),
    /// Stateful common sampler paired with the grammar-first flag for each sampling call.
    Configured(CommonSampler, bool),
}

impl Sampler {
    /// Construct the configured chain and feed it prompt tokens with grammar acceptance
    /// disabled. In greedy tool mode, prepend the template grammar sampler when present.
    pub(super) fn new(
        model: &LlamaModel,
        config: Option<&SamplingConfig>,
        sampling: &Sampling<'_>,
        prompt: &[LlamaToken],
    ) -> Result<Self> {
        if let Some(config) = config {
            let mut params = config.params(model, sampling)?;
            let mut sampler = CommonSampler::new(model, &mut params)
                .context("failed to initialize configured llama.cpp sampler")?;
            for &token in prompt {
                sampler.accept(token, false);
            }
            return Ok(Self::Configured(
                sampler,
                config.grammar_first.unwrap_or(true),
            ));
        }

        let greedy = LlamaSampler::greedy();
        if let Sampling::Tools { chat, .. } = sampling
            && let Some(grammar) = chat.grammar_sampler(model)
        {
            return Ok(Self::Greedy(LlamaSampler::chain_simple([grammar, greedy])));
        }
        Ok(Self::Greedy(greedy))
    }

    /// Read the last logits row. The common sampler explicitly accepts the selected token to
    /// advance its state and grammar.
    pub(super) fn sample(&mut self, context: &mut LlamaContext<'_>) -> Result<LlamaToken> {
        match self {
            Self::Greedy(sampler) => Ok(sampler.try_sample(context, -1)?),
            Self::Configured(sampler, grammar_first) => {
                let token = sampler.sample(context, -1, *grammar_first)?;
                sampler.accept(token, true);
                Ok(token)
            }
        }
    }
}
