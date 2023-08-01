## RobBERT-2022: Updating a Dutch Language Model to Account for Evolving
  Language Use

**Published Date:** 2022-11-15T14:55:53Z

**Link:** http://arxiv.org/pdf/2211.08192v1

**Abstract:**

  Large transformer-based language models, e.g. BERT and GPT-3, outperform
previous architectures on most natural language processing tasks. Such language
models are first pre-trained on gigantic corpora of text and later used as
base-model for finetuning on a particular task. Since the pre-training step is
usually not repeated, base models are not up-to-date with the latest
information. In this paper, we update RobBERT, a RoBERTa-based state-of-the-art
Dutch language model, which was trained in 2019. First, the tokenizer of
RobBERT is updated to include new high-frequent tokens present in the latest
Dutch OSCAR corpus, e.g. corona-related words. Then we further pre-train the
RobBERT model using this dataset. To evaluate if our new model is a plug-in
replacement for RobBERT, we introduce two additional criteria based on concept
drift of existing tokens and alignment for novel tokens.We found that for
certain language tasks this update results in a significant performance
increase. These results highlight the benefit of continually updating a
language model to account for evolving language use.


---

## LMentry: A Language Model Benchmark of Elementary Language Tasks

**Published Date:** 2022-11-03T18:01:12Z

**Link:** http://arxiv.org/pdf/2211.02069v2

**Abstract:**

  As the performance of large language models rapidly improves, benchmarks are
getting larger and more complex as well. We present LMentry, a benchmark that
avoids this "arms race" by focusing on a compact set of tasks that are trivial
to humans, e.g. writing a sentence containing a specific word, identifying
which words in a list belong to a specific category, or choosing which of two
words is longer. LMentry is specifically designed to provide quick and
interpretable insights into the capabilities and robustness of large language
models. Our experiments reveal a wide variety of failure cases that, while
immediately obvious to humans, pose a considerable challenge for large language
models, including OpenAI's latest 175B-parameter instruction-tuned model,
TextDavinci002. LMentry complements contemporary evaluation approaches of large
language models, providing a quick, automatic, and easy-to-run "unit test",
without resorting to large benchmark suites of complex tasks.


---

## X$^2$-VLM: All-In-One Pre-trained Model For Vision-Language Tasks

**Published Date:** 2022-11-22T16:48:01Z

**Link:** http://arxiv.org/pdf/2211.12402v1

**Abstract:**

  Vision language pre-training aims to learn alignments between vision and
language from a large amount of data. We proposed multi-grained vision language
pre-training, a unified approach which can learn vision language alignments in
multiple granularity. This paper advances the proposed method by unifying image
and video encoding in one model and scaling up the model with large-scale data.
We present X$^2$-VLM, a pre-trained VLM with a modular architecture for both
image-text tasks and video-text tasks. Experiment results show that X$^2$-VLM
performs the best on base and large scale for both image-text and video-text
tasks, making a good trade-off between performance and model scale. Moreover,
we show that the modular design of X$^2$-VLM results in high transferability
for X$^2$-VLM to be utilized in any language or domain. For example, by simply
replacing the text encoder with XLM-R, X$^2$-VLM outperforms state-of-the-art
multilingual multi-modal pre-trained models without any multilingual
pre-training. The code and pre-trained models will be available at
github.com/zengyan-97/X2-VLM.


---

## HyperTuning: Toward Adapting Large Language Models without
  Back-propagation

**Published Date:** 2022-11-22T18:52:25Z

**Link:** http://arxiv.org/pdf/2211.12485v1

**Abstract:**

  Fine-tuning large language models for different tasks can be costly and
inefficient, and even methods that reduce the number of tuned parameters still
require full gradient-based optimization. We propose HyperTuning, a novel
approach to model adaptation that uses a hypermodel to generate task-specific
parameters for a fixed downstream model. We demonstrate a simple setup for
hypertuning with HyperT5, a T5-based hypermodel that produces soft prefixes or
LoRA parameters for a frozen T5 model from few-shot examples. We train HyperT5
in two stages: first, hyperpretraining with a modified conditional language
modeling objective that trains a hypermodel to generate parameters; second,
multi-task fine-tuning (MTF) on a large number of diverse language tasks. We
evaluate HyperT5 on P3, MetaICL and Super-NaturalInstructions datasets, and
show that it can effectively generate parameters for unseen tasks. Moreover, we
show that using hypermodel-generated parameters as initializations for further
parameter-efficient fine-tuning improves performance. HyperTuning can thus be a
flexible and efficient way to leverage large language models for diverse
downstream applications.


---

## Grafting Pre-trained Models for Multimodal Headline Generation

**Published Date:** 2022-11-14T08:59:59Z

**Link:** http://arxiv.org/pdf/2211.07210v1

**Abstract:**

  Multimodal headline utilizes both video frames and transcripts to generate
the natural language title of the videos. Due to a lack of large-scale,
manually annotated data, the task of annotating grounded headlines for video is
labor intensive and impractical. Previous researches on pre-trained language
models and video-language models have achieved significant progress in related
downstream tasks. However, none of them can be directly applied to multimodal
headline architecture where we need both multimodal encoder and sentence
decoder. A major challenge in simply gluing language model and video-language
model is the modality balance, which is aimed at combining visual-language
complementary abilities. In this paper, we propose a novel approach to graft
the video encoder from the pre-trained video-language model on the generative
pre-trained language model. We also present a consensus fusion mechanism for
the integration of different components, via inter/intra modality relation.
Empirically, experiments show that the grafted model achieves strong results on
a brand-new dataset collected from real-world applications.


---

## Towards a Mathematics Formalisation Assistant using Large Language
  Models

**Published Date:** 2022-11-14T16:52:32Z

**Link:** http://arxiv.org/pdf/2211.07524v1

**Abstract:**

  Mathematics formalisation is the task of writing mathematics (i.e.,
definitions, theorem statements, proofs) in natural language, as found in books
and papers, into a formal language that can then be checked for correctness by
a program. It is a thriving activity today, however formalisation remains
cumbersome. In this paper, we explore the abilities of a large language model
(Codex) to help with formalisation in the Lean theorem prover. We find that
with careful input-dependent prompt selection and postprocessing, Codex is able
to formalise short mathematical statements at undergrad level with nearly 75\%
accuracy for $120$ theorem statements. For proofs quantitative analysis is
infeasible and we undertake a detailed case study. We choose a diverse set of
$13$ theorems at undergrad level with proofs that fit in two-three paragraphs.
We show that with a new prompting strategy Codex can formalise these proofs in
natural language with at least one out of twelve Codex completion being easy to
repair into a complete proof. This is surprising as essentially no aligned data
exists for formalised mathematics, particularly for proofs. These results
suggest that large language models are a promising avenue towards fully or
partially automating formalisation.


---

## Learning an Artificial Language for Knowledge-Sharing in Multilingual
  Translation

**Published Date:** 2022-11-02T17:14:42Z

**Link:** http://arxiv.org/pdf/2211.01292v2

**Abstract:**

  The cornerstone of multilingual neural translation is shared representations
across languages. Given the theoretically infinite representation power of
neural networks, semantically identical sentences are likely represented
differently. While representing sentences in the continuous latent space
ensures expressiveness, it introduces the risk of capturing of irrelevant
features which hinders the learning of a common representation. In this work,
we discretize the encoder output latent space of multilingual models by
assigning encoder states to entries in a codebook, which in effect represents
source sentences in a new artificial language. This discretization process not
only offers a new way to interpret the otherwise black-box model
representations, but, more importantly, gives potential for increasing
robustness in unseen testing conditions. We validate our approach on
large-scale experiments with realistic data volumes and domains. When tested in
zero-shot conditions, our approach is competitive with two strong alternatives
from the literature. We also use the learned artificial language to analyze
model behavior, and discover that using a similar bridge language increases
knowledge-sharing among the remaining languages.


---

## Massively Multilingual ASR on 70 Languages: Tokenization, Architecture,
  and Generalization Capabilities

**Published Date:** 2022-11-10T18:43:42Z

**Link:** http://arxiv.org/pdf/2211.05756v1

**Abstract:**

  End-to-end multilingual ASR has become more appealing because of several
reasons such as simplifying the training and deployment process and positive
performance transfer from high-resource to low-resource languages. However,
scaling up the number of languages, total hours, and number of unique tokens is
not a trivial task. This paper explores large-scale multilingual ASR models on
70 languages. We inspect two architectures: (1) Shared embedding and output and
(2) Multiple embedding and output model. In the shared model experiments, we
show the importance of tokenization strategy across different languages. Later,
we use our optimal tokenization strategy to train multiple embedding and output
model to further improve our result. Our multilingual ASR achieves 13.9%-15.6%
average WER relative improvement compared to monolingual models. We show that
our multilingual ASR generalizes well on an unseen dataset and domain,
achieving 9.5% and 7.5% WER on Multilingual Librispeech (MLS) with zero-shot
and finetuning, respectively.


---

## An Analysis of Social Biases Present in BERT Variants Across Multiple
  Languages

**Published Date:** 2022-11-25T23:38:08Z

**Link:** http://arxiv.org/pdf/2211.14402v1

**Abstract:**

  Although large pre-trained language models have achieved great success in
many NLP tasks, it has been shown that they reflect human biases from their
pre-training corpora. This bias may lead to undesirable outcomes when these
models are applied in real-world settings. In this paper, we investigate the
bias present in monolingual BERT models across a diverse set of languages
(English, Greek, and Persian). While recent research has mostly focused on
gender-related biases, we analyze religious and ethnic biases as well and
propose a template-based method to measure any kind of bias, based on sentence
pseudo-likelihood, that can handle morphologically complex languages with
gender-based adjective declensions. We analyze each monolingual model via this
method and visualize cultural similarities and differences across different
dimensions of bias. Ultimately, we conclude that current methods of probing for
bias are highly language-dependent, necessitating cultural insights regarding
the unique ways bias is expressed in each language and culture (e.g. through
coded language, synecdoche, and other similar linguistic concepts). We also
hypothesize that higher measured social biases in the non-English BERT models
correlate with user-generated content in their training.


---

