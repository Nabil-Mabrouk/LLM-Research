## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
  Encoders and Large Language Models

**Published Date:** 2023-01-30T00:56:51Z

**Link:** http://arxiv.org/pdf/2301.12597v3

**Abstract:**

  The cost of vision-and-language pre-training has become increasingly
prohibitive due to end-to-end training of large-scale models. This paper
proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps
vision-language pre-training from off-the-shelf frozen pre-trained image
encoders and frozen large language models. BLIP-2 bridges the modality gap with
a lightweight Querying Transformer, which is pre-trained in two stages. The
first stage bootstraps vision-language representation learning from a frozen
image encoder. The second stage bootstraps vision-to-language generative
learning from a frozen language model. BLIP-2 achieves state-of-the-art
performance on various vision-language tasks, despite having significantly
fewer trainable parameters than existing methods. For example, our model
outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable
parameters. We also demonstrate the model's emerging capabilities of zero-shot
image-to-text generation that can follow natural language instructions.


---

## Improving Cross-lingual Information Retrieval on Low-Resource Languages
  via Optimal Transport Distillation

**Published Date:** 2023-01-29T22:30:36Z

**Link:** http://arxiv.org/pdf/2301.12566v1

**Abstract:**

  Benefiting from transformer-based pre-trained language models, neural ranking
models have made significant progress. More recently, the advent of
multilingual pre-trained language models provides great support for designing
neural cross-lingual retrieval models. However, due to unbalanced pre-training
data in different languages, multilingual language models have already shown a
performance gap between high and low-resource languages in many downstream
tasks. And cross-lingual retrieval models built on such pre-trained models can
inherit language bias, leading to suboptimal result for low-resource languages.
Moreover, unlike the English-to-English retrieval task, where large-scale
training collections for document ranking such as MS MARCO are available, the
lack of cross-lingual retrieval data for low-resource language makes it more
challenging for training cross-lingual retrieval models. In this work, we
propose OPTICAL: Optimal Transport distillation for low-resource Cross-lingual
information retrieval. To transfer a model from high to low resource languages,
OPTICAL forms the cross-lingual token alignment task as an optimal transport
problem to learn from a well-trained monolingual retrieval model. By separating
the cross-lingual knowledge from knowledge of query document matching, OPTICAL
only needs bitext data for distillation training, which is more feasible for
low-resource languages. Experimental results show that, with minimal training
data, OPTICAL significantly outperforms strong baselines on low-resource
languages, including neural machine translation.


---

## Language Models are Drummers: Drum Composition with Natural Language
  Pre-Training

**Published Date:** 2023-01-03T15:47:53Z

**Link:** http://arxiv.org/pdf/2301.01162v1

**Abstract:**

  Automatic music generation with artificial intelligence typically requires a
large amount of data which is hard to obtain for many less common genres and
musical instruments. To tackle this issue, we present ongoing work and
preliminary findings on the possibility for deep models to transfer knowledge
from language to music, by finetuning large language models pre-trained on a
massive text corpus on only hundreds of MIDI files of drum performances. We
show that by doing so, one of the largest, state-of-the-art models (GPT3) is
capable of generating reasonable drum grooves, while models that are not
pre-trained (Transformer) shows no such ability beyond naive repetition.
Evaluating generated music is a challenging task, more so is evaluating drum
grooves with little precedence in literature. Hence, we propose a tailored
structural evaluation method and analyze drum grooves produced by GPT3 compared
to those played by human professionals, exposing the strengths and weaknesses
of such generation by language-to-music transfer. Our findings suggest that
language-to-music transfer learning with large language models is viable and
promising.


---

## Large Language Models Can Be Easily Distracted by Irrelevant Context

**Published Date:** 2023-01-31T20:48:57Z

**Link:** http://arxiv.org/pdf/2302.00093v3

**Abstract:**

  Large language models have achieved impressive performance on various natural
language processing tasks. However, so far they have been evaluated primarily
on benchmarks where all information in the input context is relevant for
solving the task. In this work, we investigate the distractibility of large
language models, i.e., how the model problem-solving accuracy can be influenced
by irrelevant context. In particular, we introduce Grade-School Math with
Irrelevant Context (GSM-IC), an arithmetic reasoning dataset with irrelevant
information in the problem description. We use this benchmark to measure the
distractibility of cutting-edge prompting techniques for large language models,
and find that the model performance is dramatically decreased when irrelevant
information is included. We also identify several approaches for mitigating
this deficiency, such as decoding with self-consistency and adding to the
prompt an instruction that tells the language model to ignore the irrelevant
information.


---

## XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked
  Language Models

**Published Date:** 2023-01-25T09:15:17Z

**Link:** http://arxiv.org/pdf/2301.10472v1

**Abstract:**

  Large multilingual language models typically rely on a single vocabulary
shared across 100+ languages. As these models have increased in parameter count
and depth, vocabulary size has remained largely unchanged. This vocabulary
bottleneck limits the representational capabilities of multilingual models like
XLM-R. In this paper, we introduce a new approach for scaling to very large
multilingual vocabularies by de-emphasizing token sharing between languages
with little lexical overlap and assigning vocabulary capacity to achieve
sufficient coverage for each individual language. Tokenizations using our
vocabulary are typically more semantically meaningful and shorter compared to
XLM-R. Leveraging this improved vocabulary, we train XLM-V, a multilingual
language model with a one million token vocabulary. XLM-V outperforms XLM-R on
every task we tested on ranging from natural language inference (XNLI),
question answering (MLQA, XQuAD, TyDiQA), and named entity recognition
(WikiAnn) to low-resource tasks (Americas NLI, MasakhaNER).


---

## Memory Augmented Large Language Models are Computationally Universal

**Published Date:** 2023-01-10T02:37:44Z

**Link:** http://arxiv.org/pdf/2301.04589v1

**Abstract:**

  We show that transformer-based large language models are computationally
universal when augmented with an external memory. Any deterministic language
model that conditions on strings of bounded length is equivalent to a finite
automaton, hence computationally limited. However, augmenting such models with
a read-write memory creates the possibility of processing arbitrarily large
inputs and, potentially, simulating any algorithm. We establish that an
existing large language model, Flan-U-PaLM 540B, can be combined with an
associative read-write memory to exactly simulate the execution of a universal
Turing machine, $U_{15,2}$. A key aspect of the finding is that it does not
require any modification of the language model weights. Instead, the
construction relies solely on designing a form of stored instruction computer
that can subsequently be programmed with a specific set of prompts.


---

## ViDeBERTa: A powerful pre-trained language model for Vietnamese

**Published Date:** 2023-01-25T07:26:54Z

**Link:** http://arxiv.org/pdf/2301.10439v2

**Abstract:**

  This paper presents ViDeBERTa, a new pre-trained monolingual language model
for Vietnamese, with three versions - ViDeBERTa_xsmall, ViDeBERTa_base, and
ViDeBERTa_large, which are pre-trained on a large-scale corpus of high-quality
and diverse Vietnamese texts using DeBERTa architecture. Although many
successful pre-trained language models based on Transformer have been widely
proposed for the English language, there are still few pre-trained models for
Vietnamese, a low-resource language, that perform good results on downstream
tasks, especially Question answering. We fine-tune and evaluate our model on
three important natural language downstream tasks, Part-of-speech tagging,
Named-entity recognition, and Question answering. The empirical results
demonstrate that ViDeBERTa with far fewer parameters surpasses the previous
state-of-the-art models on multiple Vietnamese-specific natural language
understanding tasks. Notably, ViDeBERTa_base with 86M parameters, which is only
about 23% of PhoBERT_large with 370M parameters, still performs the same or
better results than the previous state-of-the-art model. Our ViDeBERTa models
are available at: https://github.com/HySonLab/ViDeBERTa.


---

## On Robustness of Prompt-based Semantic Parsing with Large Pre-trained
  Language Model: An Empirical Study on Codex

**Published Date:** 2023-01-30T13:21:00Z

**Link:** http://arxiv.org/pdf/2301.12868v3

**Abstract:**

  Semantic parsing is a technique aimed at constructing a structured
representation of the meaning of a natural-language question. Recent
advancements in few-shot language models trained on code have demonstrated
superior performance in generating these representations compared to
traditional unimodal language models, which are trained on downstream tasks.
Despite these advancements, existing fine-tuned neural semantic parsers are
susceptible to adversarial attacks on natural-language inputs. While it has
been established that the robustness of smaller semantic parsers can be
enhanced through adversarial training, this approach is not feasible for large
language models in real-world scenarios, as it requires both substantial
computational resources and expensive human annotation on in-domain semantic
parsing data. This paper presents the first empirical study on the adversarial
robustness of a large prompt-based language model of code, \codex. Our results
demonstrate that the state-of-the-art (SOTA) code-language models are
vulnerable to carefully crafted adversarial examples. To address this
challenge, we propose methods for improving robustness without the need for
significant amounts of labeled data or heavy computational resources.


---

## Improving Cross-lingual Information Retrieval on Low-Resource Languages
  via Optimal Transport Distillation

**Published Date:** 2023-01-29T22:30:36Z

**Link:** http://arxiv.org/pdf/2301.12566v1

**Abstract:**

  Benefiting from transformer-based pre-trained language models, neural ranking
models have made significant progress. More recently, the advent of
multilingual pre-trained language models provides great support for designing
neural cross-lingual retrieval models. However, due to unbalanced pre-training
data in different languages, multilingual language models have already shown a
performance gap between high and low-resource languages in many downstream
tasks. And cross-lingual retrieval models built on such pre-trained models can
inherit language bias, leading to suboptimal result for low-resource languages.
Moreover, unlike the English-to-English retrieval task, where large-scale
training collections for document ranking such as MS MARCO are available, the
lack of cross-lingual retrieval data for low-resource language makes it more
challenging for training cross-lingual retrieval models. In this work, we
propose OPTICAL: Optimal Transport distillation for low-resource Cross-lingual
information retrieval. To transfer a model from high to low resource languages,
OPTICAL forms the cross-lingual token alignment task as an optimal transport
problem to learn from a well-trained monolingual retrieval model. By separating
the cross-lingual knowledge from knowledge of query document matching, OPTICAL
only needs bitext data for distillation training, which is more feasible for
low-resource languages. Experimental results show that, with minimal training
data, OPTICAL significantly outperforms strong baselines on low-resource
languages, including neural machine translation.


---

## A Watermark for Large Language Models

**Published Date:** 2023-01-24T18:52:59Z

**Link:** http://arxiv.org/pdf/2301.10226v3

**Abstract:**

  Potential harms of large language models can be mitigated by watermarking
model output, i.e., embedding signals into generated text that are invisible to
humans but algorithmically detectable from a short span of tokens. We propose a
watermarking framework for proprietary language models. The watermark can be
embedded with negligible impact on text quality, and can be detected using an
efficient open-source algorithm without access to the language model API or
parameters. The watermark works by selecting a randomized set of "green" tokens
before a word is generated, and then softly promoting use of green tokens
during sampling. We propose a statistical test for detecting the watermark with
interpretable p-values, and derive an information-theoretic framework for
analyzing the sensitivity of the watermark. We test the watermark using a
multi-billion parameter model from the Open Pretrained Transformer (OPT)
family, and discuss robustness and security.


---

## Unifying Molecular and Textual Representations via Multi-task Language
  Modelling

**Published Date:** 2023-01-29T23:56:45Z

**Link:** http://arxiv.org/pdf/2301.12586v2

**Abstract:**

  The recent advances in neural language models have also been successfully
applied to the field of chemistry, offering generative solutions for classical
problems in molecular design and synthesis planning. These new methods have the
potential to fuel a new era of data-driven automation in scientific discovery.
However, specialized models are still typically required for each task, leading
to the need for problem-specific fine-tuning and neglecting task
interrelations. The main obstacle in this field is the lack of a unified
representation between natural language and chemical representations,
complicating and limiting human-machine interaction. Here, we propose the first
multi-domain, multi-task language model that can solve a wide range of tasks in
both the chemical and natural language domains. Our model can handle chemical
and natural language concurrently, without requiring expensive pre-training on
single domains or task-specific models. Interestingly, sharing weights across
domains remarkably improves our model when benchmarked against state-of-the-art
baselines on single-domain and cross-domain tasks. In particular, sharing
information across domains and tasks gives rise to large improvements in
cross-domain tasks, the magnitude of which increase with scale, as measured by
more than a dozen of relevant metrics. Our work suggests that such models can
robustly and efficiently accelerate discovery in physical sciences by
superseding problem-specific fine-tuning and enhancing human-model
interactions.


---

## Improving Cross-lingual Information Retrieval on Low-Resource Languages
  via Optimal Transport Distillation

**first_author:** Zhiqi Huang et al.

**Published Date:** 2023-01-29T22:30:36Z

**Link:** http://arxiv.org/pdf/2301.12566v1

**Abstract:**

  Benefiting from transformer-based pre-trained language models, neural ranking
models have made significant progress. More recently, the advent of
multilingual pre-trained language models provides great support for designing
neural cross-lingual retrieval models. However, due to unbalanced pre-training
data in different languages, multilingual language models have already shown a
performance gap between high and low-resource languages in many downstream
tasks. And cross-lingual retrieval models built on such pre-trained models can
inherit language bias, leading to suboptimal result for low-resource languages.
Moreover, unlike the English-to-English retrieval task, where large-scale
training collections for document ranking such as MS MARCO are available, the
lack of cross-lingual retrieval data for low-resource language makes it more
challenging for training cross-lingual retrieval models. In this work, we
propose OPTICAL: Optimal Transport distillation for low-resource Cross-lingual
information retrieval. To transfer a model from high to low resource languages,
OPTICAL forms the cross-lingual token alignment task as an optimal transport
problem to learn from a well-trained monolingual retrieval model. By separating
the cross-lingual knowledge from knowledge of query document matching, OPTICAL
only needs bitext data for distillation training, which is more feasible for
low-resource languages. Experimental results show that, with minimal training
data, OPTICAL significantly outperforms strong baselines on low-resource
languages, including neural machine translation.


---

