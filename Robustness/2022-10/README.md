## Benchmarking Language Models for Code Syntax Understanding

**Published Date:** 2022-10-26T04:47:18Z

**Link:** http://arxiv.org/pdf/2210.14473v1

**Abstract:**

  Pre-trained language models have demonstrated impressive performance in both
natural language processing and program understanding, which represent the
input as a token sequence without explicitly modeling its structure. Some prior
works show that pre-trained language models can capture the syntactic rules of
natural languages without finetuning on syntax understanding tasks. However,
there is limited understanding of how well pre-trained models understand the
code structure so far. In this work, we perform the first thorough benchmarking
of the state-of-the-art pre-trained models for identifying the syntactic
structures of programs. Specifically, we introduce CodeSyntax, a large-scale
dataset of programs annotated with the syntactic relationships in their
corresponding abstract syntax trees. Our key observation is that existing
language models pretrained on code still lack the understanding of code syntax.
In fact, these pre-trained programming language models fail to match the
performance of simple baselines based on positional offsets and keywords. We
also present a natural language benchmark to highlight the differences between
natural languages and programming languages in terms of syntactic structure
understanding. Our findings point out key limitations of existing pre-training
methods for programming languages, and suggest the importance of modeling code
syntactic structures.


---

## TabLLM: Few-shot Classification of Tabular Data with Large Language
  Models

**Published Date:** 2022-10-19T17:08:13Z

**Link:** http://arxiv.org/pdf/2210.10723v2

**Abstract:**

  We study the application of large language models to zero-shot and few-shot
classification of tabular data. We prompt the large language model with a
serialization of the tabular data to a natural-language string, together with a
short description of the classification problem. In the few-shot setting, we
fine-tune the large language model using some labeled examples. We evaluate
several serialization methods including templates, table-to-text models, and
large language models. Despite its simplicity, we find that this technique
outperforms prior deep-learning-based tabular classification methods on several
benchmark datasets. In most cases, even zero-shot classification obtains
non-trivial performance, illustrating the method's ability to exploit prior
knowledge encoded in large language models. Unlike many deep learning methods
for tabular datasets, this approach is also competitive with strong traditional
baselines like gradient-boosted trees, especially in the very-few-shot setting.


---

## Subword Segmental Language Modelling for Nguni Languages

**Published Date:** 2022-10-12T18:41:00Z

**Link:** http://arxiv.org/pdf/2210.06525v1

**Abstract:**

  Subwords have become the standard units of text in NLP, enabling efficient
open-vocabulary models. With algorithms like byte-pair encoding (BPE), subword
segmentation is viewed as a preprocessing step applied to the corpus before
training. This can lead to sub-optimal segmentations for low-resource languages
with complex morphologies. We propose a subword segmental language model (SSLM)
that learns how to segment words while being trained for autoregressive
language modelling. By unifying subword segmentation and language modelling,
our model learns subwords that optimise LM performance. We train our model on
the 4 Nguni languages of South Africa. These are low-resource agglutinative
languages, so subword information is critical. As an LM, SSLM outperforms
existing approaches such as BPE-based models on average across the 4 languages.
Furthermore, it outperforms standard subword segmenters on unsupervised
morphological segmentation. We also train our model as a word-level sequence
model, resulting in an unsupervised morphological segmenter that outperforms
existing methods by a large margin for all 4 languages. Our results show that
learning subword segmentation is an effective alternative to existing subword
segmenters, enabling the model to discover morpheme-like subwords that improve
its LM capabilities.


---

## Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination

**Published Date:** 2022-10-21T21:33:10Z

**Link:** http://arxiv.org/pdf/2210.12261v1

**Abstract:**

  Large-scale pretrained language models have made significant advances in
solving downstream language understanding tasks. However, they generally suffer
from reporting bias, the phenomenon describing the lack of explicit commonsense
knowledge in written text, e.g., ''an orange is orange''. To overcome this
limitation, we develop a novel approach, Z-LaVI, to endow language models with
visual imagination capabilities. Specifically, we leverage two complementary
types of ''imaginations'': (i) recalling existing images through retrieval and
(ii) synthesizing nonexistent images via text-to-image generation. Jointly
exploiting the language inputs and the imagination, a pretrained
vision-language model (e.g., CLIP) eventually composes a zero-shot solution to
the original language tasks. Notably, fueling language models with imagination
can effectively leverage visual knowledge to solve plain language tasks. In
consequence, Z-LaVI consistently improves the zero-shot performance of existing
language models across a diverse set of language tasks.


---

## InforMask: Unsupervised Informative Masking for Language Model
  Pretraining

**Published Date:** 2022-10-21T07:10:56Z

**Link:** http://arxiv.org/pdf/2210.11771v1

**Abstract:**

  Masked language modeling is widely used for pretraining large language models
for natural language understanding (NLU). However, random masking is
suboptimal, allocating an equal masking rate for all tokens. In this paper, we
propose InforMask, a new unsupervised masking strategy for training masked
language models. InforMask exploits Pointwise Mutual Information (PMI) to
select the most informative tokens to mask. We further propose two
optimizations for InforMask to improve its efficiency. With a one-off
preprocessing step, InforMask outperforms random masking and previously
proposed masking strategies on the factual recall benchmark LAMA and the
question answering benchmark SQuAD v1 and v2.


---

## Spontaneous Emerging Preference in Two-tower Language Model

**Published Date:** 2022-10-13T13:55:19Z

**Link:** http://arxiv.org/pdf/2210.07041v1

**Abstract:**

  The ever-growing size of the foundation language model has brought
significant performance gains in various types of downstream tasks. With the
existence of side-effects brought about by the large size of the foundation
language model such as deployment cost, availability issues, and environmental
cost, there is some interest in exploring other possible directions, such as a
divide-and-conquer scheme. In this paper, we are asking a basic question: are
language processes naturally dividable? We study this problem with a simple
two-tower language model setting, where two language models with identical
configurations are trained side-by-side cooperatively. With this setting, we
discover the spontaneous emerging preference phenomenon, where some of the
tokens are consistently better predicted by one tower while others by another
tower. This phenomenon is qualitatively stable, regardless of model
configuration and type, suggesting this as an intrinsic property of natural
language. This study suggests that interesting properties of natural language
are still waiting to be discovered, which may aid the future development of
natural language processing techniques.


---

## Data-Efficient Cross-Lingual Transfer with Language-Specific Subnetworks

**Published Date:** 2022-10-31T19:23:33Z

**Link:** http://arxiv.org/pdf/2211.00106v1

**Abstract:**

  Large multilingual language models typically share their parameters across
all languages, which enables cross-lingual task transfer, but learning can also
be hindered when training updates from different languages are in conflict. In
this paper, we propose novel methods for using language-specific subnetworks,
which control cross-lingual parameter sharing, to reduce conflicts and increase
positive transfer during fine-tuning. We introduce dynamic subnetworks, which
are jointly updated with the model, and we combine our methods with
meta-learning, an established, but complementary, technique for improving
cross-lingual transfer. Finally, we provide extensive analyses of how each of
our methods affects the models.


---

## Adapters for Enhanced Modeling of Multilingual Knowledge and Text

**Published Date:** 2022-10-24T21:33:42Z

**Link:** http://arxiv.org/pdf/2210.13617v2

**Abstract:**

  Large language models appear to learn facts from the large text corpora they
are trained on. Such facts are encoded implicitly within their many parameters,
making it difficult to verify or manipulate what knowledge has been learned.
Language models have recently been extended to multilingual language models
(MLLMs), enabling knowledge to be learned across hundreds of languages.
Meanwhile, knowledge graphs contain facts in an explicit triple format, which
require careful and costly curation and are only available in a few
high-resource languages, restricting their research and application. To address
these issues, we propose to enhance MLLMs with knowledge from multilingual
knowledge graphs (MLKGs) so as to tackle language and knowledge graph tasks
across many languages, including low-resource ones. Specifically, we introduce
a lightweight adapter set to enhance MLLMs with cross-lingual entity alignment
and facts from MLKGs for many languages. Experiments on common benchmarks show
that such enhancement benefits both MLLMs and MLKGs, achieving: (1) comparable
or improved performance for knowledge graph completion and entity alignment
relative to baselines, especially for low-resource languages (for which
knowledge graphs are unavailable); and (2) improved MLLM performance on
language understanding tasks that require multilingual factual knowledge; all
while maintaining performance on other general language tasks.


---

## Scaling Back-Translation with Domain Text Generation for Sign Language
  Gloss Translation

**Published Date:** 2022-10-13T14:25:08Z

**Link:** http://arxiv.org/pdf/2210.07054v2

**Abstract:**

  Sign language gloss translation aims to translate the sign glosses into
spoken language texts, which is challenging due to the scarcity of labeled
gloss-text parallel data. Back translation (BT), which generates
pseudo-parallel data by translating in-domain spoken language texts into sign
glosses, has been applied to alleviate the data scarcity problem. However, the
lack of large-scale high-quality domain spoken language text data limits the
effect of BT. In this paper, to overcome the limitation, we propose a Prompt
based domain text Generation (PGEN) approach to produce the large-scale
in-domain spoken language text data. Specifically, PGEN randomly concatenates
sentences from the original in-domain spoken language text data as prompts to
induce a pre-trained language model (i.e., GPT-2) to generate spoken language
texts in a similar style. Experimental results on three benchmarks of sign
language gloss translation in varied languages demonstrate that BT with spoken
language texts generated by PGEN significantly outperforms the compared
methods. In addition, as the scale of spoken language texts generated by PGEN
increases, the BT technique can achieve further improvements, demonstrating the
effectiveness of our approach. We release the code and data for facilitating
future research in this field.


---

## EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge
  Distillation and Modal-adaptive Pruning

**Published Date:** 2022-10-14T13:26:41Z

**Link:** http://arxiv.org/pdf/2210.07795v1

**Abstract:**

  Pre-trained vision-language models (VLMs) have achieved impressive results in
a range of vision-language tasks. However, popular VLMs usually consist of
hundreds of millions of parameters which brings challenges for fine-tuning and
deployment in real-world applications due to space, memory, and latency
constraints. In this work, we introduce a distilling then pruning framework to
compress large vision-language models into smaller, faster, and more accurate
ones. We first shrink the size of a pre-trained large VLM and apply knowledge
distillation in the vision-language pre-training stage to obtain a
task-agnostic compact VLM. Then we propose a modal-adaptive pruning algorithm
to automatically infer the importance of vision and language modalities for
different downstream tasks and adaptively remove redundant structures and
neurons in different encoders with controllable target sparsity. We apply our
framework to train EfficientVLM, a fast and accurate vision-language model
consisting of 6 vision layers, 3 text layers, and 3 cross-modal fusion layers,
accounting for only 93 million parameters in total, which is 44.3% of the
teacher model. EfficientVLM retains 98.4% performance of the teacher model and
accelerates its inference speed by 2.2x. EfficientVLM achieves a large absolute
improvement over previous SoTA efficient VLMs of similar sizes by a large
margin on various vision-language tasks, including VQAv2 (+4.9%), NLVR2
(+5.6%), ITR (R@1 on TR +17.2%, on IR + 15.6% ) and COCO caption generation
(CIDEr +6.5), demonstrating a large potential on training lightweight VLMs.


---

## Bootstrapping Multilingual Semantic Parsers using Large Language Models

**Published Date:** 2022-10-13T19:34:14Z

**Link:** http://arxiv.org/pdf/2210.07313v2

**Abstract:**

  Despite cross-lingual generalization demonstrated by pre-trained multilingual
models, the translate-train paradigm of transferring English datasets across
multiple languages remains to be a key mechanism for training task-specific
multilingual models. However, for many low-resource languages, the availability
of a reliable translation service entails significant amounts of costly
human-annotated translation pairs. Further, translation services may continue
to be brittle due to domain mismatch between task-specific input text and
general-purpose text used for training translation models. For multilingual
semantic parsing, we demonstrate the effectiveness and flexibility offered by
large language models (LLMs) for translating English datasets into several
languages via few-shot prompting. Through extensive comparisons on two public
datasets, MTOP and MASSIVE, spanning 50 languages and several domains, we show
that our method of translating data using LLMs outperforms a strong
translate-train baseline on 41 out of 50 languages. We study the key design
choices that enable more effective multilingual data translation via prompted
LLMs.


---

## Shapley Head Pruning: Identifying and Removing Interference in
  Multilingual Transformers

**Published Date:** 2022-10-11T18:11:37Z

**Link:** http://arxiv.org/pdf/2210.05709v1

**Abstract:**

  Multilingual transformer-based models demonstrate remarkable zero and
few-shot transfer across languages by learning and reusing language-agnostic
features. However, as a fixed-size model acquires more languages, its
performance across all languages degrades, a phenomenon termed interference.
Often attributed to limited model capacity, interference is commonly addressed
by adding additional parameters despite evidence that transformer-based models
are overparameterized. In this work, we show that it is possible to reduce
interference by instead identifying and pruning language-specific parameters.
First, we use Shapley Values, a credit allocation metric from coalitional game
theory, to identify attention heads that introduce interference. Then, we show
that removing identified attention heads from a fixed model improves
performance for a target language on both sentence classification and
structural prediction, seeing gains as large as 24.7\%. Finally, we provide
insights on language-agnostic and language-specific attention heads using
attention visualization.


---

## Developing a general-purpose clinical language inference model from a
  large corpus of clinical notes

**Published Date:** 2022-10-12T20:08:45Z

**Link:** http://arxiv.org/pdf/2210.06566v1

**Abstract:**

  Several biomedical language models have already been developed for clinical
language inference. However, these models typically utilize general
vocabularies and are trained on relatively small clinical corpora. We sought to
evaluate the impact of using a domain-specific vocabulary and a large clinical
training corpus on the performance of these language models in clinical
language inference. We trained a Bidirectional Encoder Decoder from
Transformers (BERT) model using a diverse, deidentified corpus of 75 million
deidentified clinical notes authored at the University of California, San
Francisco (UCSF). We evaluated this model on several clinical language
inference benchmark tasks: clinical and temporal concept recognition, relation
extraction and medical language inference. We also evaluated our model on two
tasks using discharge summaries from UCSF: diagnostic code assignment and
therapeutic class inference. Our model performs at par with the best publicly
available biomedical language models of comparable sizes on the public
benchmark tasks, and is significantly better than these models in a
within-system evaluation on the two tasks using UCSF data. The use of in-domain
vocabulary appears to improve the encoding of longer documents. The use of
large clinical corpora appears to enhance document encoding and inferential
accuracy. However, further research is needed to improve abbreviation
resolution, and numerical, temporal, and implicitly causal inference.


---

## Modeling the Graphotactics of Low-Resource Languages Using Sequential
  GANs

**Published Date:** 2022-10-26T01:21:00Z

**Link:** http://arxiv.org/pdf/2210.14409v1

**Abstract:**

  Generative Adversarial Networks (GANs) have been shown to aid in the creation
of artificial data in situations where large amounts of real data are difficult
to come by. This issue is especially salient in the computational linguistics
space, where researchers are often tasked with modeling the complex morphologic
and grammatical processes of low-resource languages. This paper will discuss
the implementation and testing of a GAN that attempts to model and reproduce
the graphotactics of a language using only 100 example strings. These
artificial, yet graphotactically compliant, strings are meant to aid in
modeling the morphological inflection of low-resource languages.


---

## MALM: Mixing Augmented Language Modeling for Zero-Shot Machine
  Translation

**Published Date:** 2022-10-01T17:01:30Z

**Link:** http://arxiv.org/pdf/2210.00320v1

**Abstract:**

  Large pre-trained language models have brought remarkable progress in NLP.
Pre-training and Fine-tuning have given state-of-art performance across tasks
in text processing. Data Augmentation techniques have also helped build
state-of-art models on low or zero resource tasks. Many works in the past have
attempted at learning a single massively-multilingual machine translation model
for zero-shot translation. Although those translation models are producing
correct translations, the main challenge is those models are producing the
wrong languages for zero-shot translation. This work and its results indicate
that prompt conditioned large models do not suffer from off-target language
errors i.e. errors arising due to translation to wrong languages. We
empirically demonstrate the effectiveness of self-supervised pre-training and
data augmentation for zero-shot multi-lingual machine translation.


---

## The Effectiveness of Masked Language Modeling and Adapters for Factual
  Knowledge Injection

**Published Date:** 2022-10-03T13:08:09Z

**Link:** http://arxiv.org/pdf/2210.00907v1

**Abstract:**

  This paper studies the problem of injecting factual knowledge into large
pre-trained language models. We train adapter modules on parts of the
ConceptNet knowledge graph using the masked language modeling objective and
evaluate the success of the method by a series of probing experiments on the
LAMA probe. Mean P@K curves for different configurations indicate that the
technique is effective, increasing the performance on subsets of the LAMA probe
for large values of k by adding as little as 2.1% additional parameters to the
original models.


---

## Zemi: Learning Zero-Shot Semi-Parametric Language Models from Multiple
  Tasks

**Published Date:** 2022-10-01T04:08:50Z

**Link:** http://arxiv.org/pdf/2210.00185v2

**Abstract:**

  Although large language models have achieved impressive zero-shot ability,
the huge model size generally incurs high cost. Recently, semi-parametric
language models, which augment a smaller language model with an external
retriever, have demonstrated promising language modeling capabilities. However,
it remains unclear whether such semi-parametric language models can perform
competitively well as their fully-parametric counterparts on zero-shot
generalization to downstream tasks. In this work, we introduce $\text{Zemi}$, a
zero-shot semi-parametric language model. To our best knowledge, this is the
first semi-parametric language model that can demonstrate strong zero-shot
performance on a wide range of held-out unseen tasks. We train $\text{Zemi}$
with a novel semi-parametric multitask prompted training paradigm, which shows
significant improvement compared with the parametric multitask training as
proposed by T0. Specifically, we augment the multitask training and zero-shot
evaluation with retrieval from a large-scale task-agnostic unlabeled corpus. In
order to incorporate multiple potentially noisy retrieved augmentations, we
further propose a novel $\text{augmentation fusion}$ module leveraging
perceiver resampler and gated cross-attention. Notably, our proposed
$\text{Zemi}_\text{LARGE}$ outperforms T0-3B by 16% on all seven evaluation
tasks while being 3.9x smaller in model size.


---

## Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination

**Published Date:** 2022-10-21T21:33:10Z

**Link:** http://arxiv.org/pdf/2210.12261v1

**Abstract:**

  Large-scale pretrained language models have made significant advances in
solving downstream language understanding tasks. However, they generally suffer
from reporting bias, the phenomenon describing the lack of explicit commonsense
knowledge in written text, e.g., ''an orange is orange''. To overcome this
limitation, we develop a novel approach, Z-LaVI, to endow language models with
visual imagination capabilities. Specifically, we leverage two complementary
types of ''imaginations'': (i) recalling existing images through retrieval and
(ii) synthesizing nonexistent images via text-to-image generation. Jointly
exploiting the language inputs and the imagination, a pretrained
vision-language model (e.g., CLIP) eventually composes a zero-shot solution to
the original language tasks. Notably, fueling language models with imagination
can effectively leverage visual knowledge to solve plain language tasks. In
consequence, Z-LaVI consistently improves the zero-shot performance of existing
language models across a diverse set of language tasks.


---

## InforMask: Unsupervised Informative Masking for Language Model
  Pretraining

**Published Date:** 2022-10-21T07:10:56Z

**Link:** http://arxiv.org/pdf/2210.11771v1

**Abstract:**

  Masked language modeling is widely used for pretraining large language models
for natural language understanding (NLU). However, random masking is
suboptimal, allocating an equal masking rate for all tokens. In this paper, we
propose InforMask, a new unsupervised masking strategy for training masked
language models. InforMask exploits Pointwise Mutual Information (PMI) to
select the most informative tokens to mask. We further propose two
optimizations for InforMask to improve its efficiency. With a one-off
preprocessing step, InforMask outperforms random masking and previously
proposed masking strategies on the factual recall benchmark LAMA and the
question answering benchmark SQuAD v1 and v2.


---

## Spontaneous Emerging Preference in Two-tower Language Model

**Published Date:** 2022-10-13T13:55:19Z

**Link:** http://arxiv.org/pdf/2210.07041v1

**Abstract:**

  The ever-growing size of the foundation language model has brought
significant performance gains in various types of downstream tasks. With the
existence of side-effects brought about by the large size of the foundation
language model such as deployment cost, availability issues, and environmental
cost, there is some interest in exploring other possible directions, such as a
divide-and-conquer scheme. In this paper, we are asking a basic question: are
language processes naturally dividable? We study this problem with a simple
two-tower language model setting, where two language models with identical
configurations are trained side-by-side cooperatively. With this setting, we
discover the spontaneous emerging preference phenomenon, where some of the
tokens are consistently better predicted by one tower while others by another
tower. This phenomenon is qualitatively stable, regardless of model
configuration and type, suggesting this as an intrinsic property of natural
language. This study suggests that interesting properties of natural language
are still waiting to be discovered, which may aid the future development of
natural language processing techniques.


---

## EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge
  Distillation and Modal-adaptive Pruning

**Published Date:** 2022-10-14T13:26:41Z

**Link:** http://arxiv.org/pdf/2210.07795v1

**Abstract:**

  Pre-trained vision-language models (VLMs) have achieved impressive results in
a range of vision-language tasks. However, popular VLMs usually consist of
hundreds of millions of parameters which brings challenges for fine-tuning and
deployment in real-world applications due to space, memory, and latency
constraints. In this work, we introduce a distilling then pruning framework to
compress large vision-language models into smaller, faster, and more accurate
ones. We first shrink the size of a pre-trained large VLM and apply knowledge
distillation in the vision-language pre-training stage to obtain a
task-agnostic compact VLM. Then we propose a modal-adaptive pruning algorithm
to automatically infer the importance of vision and language modalities for
different downstream tasks and adaptively remove redundant structures and
neurons in different encoders with controllable target sparsity. We apply our
framework to train EfficientVLM, a fast and accurate vision-language model
consisting of 6 vision layers, 3 text layers, and 3 cross-modal fusion layers,
accounting for only 93 million parameters in total, which is 44.3% of the
teacher model. EfficientVLM retains 98.4% performance of the teacher model and
accelerates its inference speed by 2.2x. EfficientVLM achieves a large absolute
improvement over previous SoTA efficient VLMs of similar sizes by a large
margin on various vision-language tasks, including VQAv2 (+4.9%), NLVR2
(+5.6%), ITR (R@1 on TR +17.2%, on IR + 15.6% ) and COCO caption generation
(CIDEr +6.5), demonstrating a large potential on training lightweight VLMs.


---

## Developing a general-purpose clinical language inference model from a
  large corpus of clinical notes

**Published Date:** 2022-10-12T20:08:45Z

**Link:** http://arxiv.org/pdf/2210.06566v1

**Abstract:**

  Several biomedical language models have already been developed for clinical
language inference. However, these models typically utilize general
vocabularies and are trained on relatively small clinical corpora. We sought to
evaluate the impact of using a domain-specific vocabulary and a large clinical
training corpus on the performance of these language models in clinical
language inference. We trained a Bidirectional Encoder Decoder from
Transformers (BERT) model using a diverse, deidentified corpus of 75 million
deidentified clinical notes authored at the University of California, San
Francisco (UCSF). We evaluated this model on several clinical language
inference benchmark tasks: clinical and temporal concept recognition, relation
extraction and medical language inference. We also evaluated our model on two
tasks using discharge summaries from UCSF: diagnostic code assignment and
therapeutic class inference. Our model performs at par with the best publicly
available biomedical language models of comparable sizes on the public
benchmark tasks, and is significantly better than these models in a
within-system evaluation on the two tasks using UCSF data. The use of in-domain
vocabulary appears to improve the encoding of longer documents. The use of
large clinical corpora appears to enhance document encoding and inferential
accuracy. However, further research is needed to improve abbreviation
resolution, and numerical, temporal, and implicitly causal inference.


---

## Language-free Training for Zero-shot Video Grounding

**Published Date:** 2022-10-24T06:55:29Z

**Link:** http://arxiv.org/pdf/2210.12977v1

**Abstract:**

  Given an untrimmed video and a language query depicting a specific temporal
moment in the video, video grounding aims to localize the time interval by
understanding the text and video simultaneously. One of the most challenging
issues is an extremely time- and cost-consuming annotation collection,
including video captions in a natural language form and their corresponding
temporal regions. In this paper, we present a simple yet novel training
framework for video grounding in the zero-shot setting, which learns a network
with only video data without any annotation. Inspired by the recent
language-free paradigm, i.e. training without language data, we train the
network without compelling the generation of fake (pseudo) text queries into a
natural language form. Specifically, we propose a method for learning a video
grounding model by selecting a temporal interval as a hypothetical correct
answer and considering the visual feature selected by our method in the
interval as a language feature, with the help of the well-aligned
visual-language space of CLIP. Extensive experiments demonstrate the prominence
of our language-free training framework, outperforming the existing zero-shot
video grounding method and even several weakly-supervised approaches with large
margins on two standard datasets.


---

## WHEN FLUE MEETS FLANG: Benchmarks and Large Pre-trained Language Model
  for Financial Domain

**Published Date:** 2022-10-31T18:35:18Z

**Link:** http://arxiv.org/pdf/2211.00083v1

**Abstract:**

  Pre-trained language models have shown impressive performance on a variety of
tasks and domains. Previous research on financial language models usually
employs a generic training scheme to train standard model architectures,
without completely leveraging the richness of the financial data. We propose a
novel domain specific Financial LANGuage model (FLANG) which uses financial
keywords and phrases for better masking, together with span boundary objective
and in-filing objective. Additionally, the evaluation benchmarks in the field
have been limited. To this end, we contribute the Financial Language
Understanding Evaluation (FLUE), an open-source comprehensive suite of
benchmarks for the financial domain. These include new benchmarks across 5 NLP
tasks in financial domain as well as common benchmarks used in the previous
research. Experiments on these benchmarks suggest that our model outperforms
those in prior literature on a variety of NLP tasks. Our models, code and
benchmark data are publicly available on Github and Huggingface.


---

## Language Generation Models Can Cause Harm: So What Can We Do About It?
  An Actionable Survey

**Published Date:** 2022-10-14T10:43:39Z

**Link:** http://arxiv.org/pdf/2210.07700v2

**Abstract:**

  Recent advances in the capacity of large language models to generate
human-like text have resulted in their increased adoption in user-facing
settings. In parallel, these improvements have prompted a heated discourse
around the risks of societal harms they introduce, whether inadvertent or
malicious. Several studies have explored these harms and called for their
mitigation via development of safer, fairer models. Going beyond enumerating
the risks of harms, this work provides a survey of practical methods for
addressing potential threats and societal harms from language generation
models. We draw on several prior works' taxonomies of language model risks to
present a structured overview of strategies for detecting and ameliorating
different kinds of risks/harms of language generators. Bridging diverse strands
of research, this survey aims to serve as a practical guide for both LM
researchers and practitioners, with explanations of different mitigation
strategies' motivations, their limitations, and open problems for future
research.


---

## Forging Multiple Training Objectives for Pre-trained Language Models via
  Meta-Learning

**Published Date:** 2022-10-19T04:38:26Z

**Link:** http://arxiv.org/pdf/2210.10293v1

**Abstract:**

  Multiple pre-training objectives fill the vacancy of the understanding
capability of single-objective language modeling, which serves the ultimate
purpose of pre-trained language models (PrLMs), generalizing well on a mass of
scenarios. However, learning multiple training objectives in a single model is
challenging due to the unknown relative significance as well as the potential
contrariety between them. Empirical studies have shown that the current
objective sampling in an ad-hoc manual setting makes the learned language
representation barely converge to the desired optimum. Thus, we propose
\textit{MOMETAS}, a novel adaptive sampler based on meta-learning, which learns
the latent sampling pattern on arbitrary pre-training objectives. Such a design
is lightweight with negligible additional training overhead. To validate our
approach, we adopt five objectives and conduct continual pre-training with
BERT-base and BERT-large models, where MOMETAS demonstrates universal
performance gain over other rule-based sampling strategies on 14 natural
language processing tasks.


---

## Retrieval Oriented Masking Pre-training Language Model for Dense Passage
  Retrieval

**Published Date:** 2022-10-27T02:43:48Z

**Link:** http://arxiv.org/pdf/2210.15133v1

**Abstract:**

  Pre-trained language model (PTM) has been shown to yield powerful text
representations for dense passage retrieval task. The Masked Language Modeling
(MLM) is a major sub-task of the pre-training process. However, we found that
the conventional random masking strategy tend to select a large number of
tokens that have limited effect on the passage retrieval task (e,g. stop-words
and punctuation). By noticing the term importance weight can provide valuable
information for passage retrieval, we hereby propose alternative retrieval
oriented masking (dubbed as ROM) strategy where more important tokens will have
a higher probability of being masked out, to capture this straightforward yet
essential information to facilitate the language model pre-training process.
Notably, the proposed new token masking method will not change the architecture
and learning objective of original PTM. Our experiments verify that the
proposed ROM enables term importance information to help language model
pre-training thus achieving better performance on multiple passage retrieval
benchmarks.


---

## Robustification of Multilingual Language Models to Real-world Noise in
  Crosslingual Zero-shot Settings with Robust Contrastive Pretraining

**Published Date:** 2022-10-10T15:40:43Z

**Link:** http://arxiv.org/pdf/2210.04782v2

**Abstract:**

  Advances in neural modeling have achieved state-of-the-art (SOTA) results on
public natural language processing (NLP) benchmarks, at times surpassing human
performance. However, there is a gap between public benchmarks and real-world
applications where noise, such as typographical or grammatical mistakes, is
abundant and can result in degraded performance. Unfortunately, works which
evaluate the robustness of neural models on noisy data and propose
improvements, are limited to the English language. Upon analyzing noise in
different languages, we observe that noise types vary greatly across languages.
Thus, existing investigations do not generalize trivially to multilingual
settings. To benchmark the performance of pretrained multilingual language
models, we construct noisy datasets covering five languages and four NLP tasks
and observe a clear gap in the performance between clean and noisy data in the
zero-shot cross-lingual setting. After investigating several ways to boost the
robustness of multilingual models in this setting, we propose Robust
Contrastive Pretraining (RCP). RCP combines data augmentation with a
contrastive loss term at the pretraining stage and achieves large improvements
on noisy (and original test data) across two sentence-level (+3.2%) and two
sequence-labeling (+10 F1-score) multilingual classification tasks.


---

## Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination

**first_author:** Yue Yang et al.

**Published Date:** 2022-10-21T21:33:10Z

**Link:** http://arxiv.org/pdf/2210.12261v1

**Abstract:**

  Large-scale pretrained language models have made significant advances in
solving downstream language understanding tasks. However, they generally suffer
from reporting bias, the phenomenon describing the lack of explicit commonsense
knowledge in written text, e.g., ''an orange is orange''. To overcome this
limitation, we develop a novel approach, Z-LaVI, to endow language models with
visual imagination capabilities. Specifically, we leverage two complementary
types of ''imaginations'': (i) recalling existing images through retrieval and
(ii) synthesizing nonexistent images via text-to-image generation. Jointly
exploiting the language inputs and the imagination, a pretrained
vision-language model (e.g., CLIP) eventually composes a zero-shot solution to
the original language tasks. Notably, fueling language models with imagination
can effectively leverage visual knowledge to solve plain language tasks. In
consequence, Z-LaVI consistently improves the zero-shot performance of existing
language models across a diverse set of language tasks.


---

## InforMask: Unsupervised Informative Masking for Language Model
  Pretraining

**first_author:** Nafis Sadeq et al.

**Published Date:** 2022-10-21T07:10:56Z

**Link:** http://arxiv.org/pdf/2210.11771v1

**Abstract:**

  Masked language modeling is widely used for pretraining large language models
for natural language understanding (NLU). However, random masking is
suboptimal, allocating an equal masking rate for all tokens. In this paper, we
propose InforMask, a new unsupervised masking strategy for training masked
language models. InforMask exploits Pointwise Mutual Information (PMI) to
select the most informative tokens to mask. We further propose two
optimizations for InforMask to improve its efficiency. With a one-off
preprocessing step, InforMask outperforms random masking and previously
proposed masking strategies on the factual recall benchmark LAMA and the
question answering benchmark SQuAD v1 and v2.


---

## Spontaneous Emerging Preference in Two-tower Language Model

**first_author:** Zhengqi He et al.

**Published Date:** 2022-10-13T13:55:19Z

**Link:** http://arxiv.org/pdf/2210.07041v1

**Abstract:**

  The ever-growing size of the foundation language model has brought
significant performance gains in various types of downstream tasks. With the
existence of side-effects brought about by the large size of the foundation
language model such as deployment cost, availability issues, and environmental
cost, there is some interest in exploring other possible directions, such as a
divide-and-conquer scheme. In this paper, we are asking a basic question: are
language processes naturally dividable? We study this problem with a simple
two-tower language model setting, where two language models with identical
configurations are trained side-by-side cooperatively. With this setting, we
discover the spontaneous emerging preference phenomenon, where some of the
tokens are consistently better predicted by one tower while others by another
tower. This phenomenon is qualitatively stable, regardless of model
configuration and type, suggesting this as an intrinsic property of natural
language. This study suggests that interesting properties of natural language
are still waiting to be discovered, which may aid the future development of
natural language processing techniques.


---

## EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge
  Distillation and Modal-adaptive Pruning

**first_author:** Tiannan Wang et al.

**Published Date:** 2022-10-14T13:26:41Z

**Link:** http://arxiv.org/pdf/2210.07795v1

**Abstract:**

  Pre-trained vision-language models (VLMs) have achieved impressive results in
a range of vision-language tasks. However, popular VLMs usually consist of
hundreds of millions of parameters which brings challenges for fine-tuning and
deployment in real-world applications due to space, memory, and latency
constraints. In this work, we introduce a distilling then pruning framework to
compress large vision-language models into smaller, faster, and more accurate
ones. We first shrink the size of a pre-trained large VLM and apply knowledge
distillation in the vision-language pre-training stage to obtain a
task-agnostic compact VLM. Then we propose a modal-adaptive pruning algorithm
to automatically infer the importance of vision and language modalities for
different downstream tasks and adaptively remove redundant structures and
neurons in different encoders with controllable target sparsity. We apply our
framework to train EfficientVLM, a fast and accurate vision-language model
consisting of 6 vision layers, 3 text layers, and 3 cross-modal fusion layers,
accounting for only 93 million parameters in total, which is 44.3% of the
teacher model. EfficientVLM retains 98.4% performance of the teacher model and
accelerates its inference speed by 2.2x. EfficientVLM achieves a large absolute
improvement over previous SoTA efficient VLMs of similar sizes by a large
margin on various vision-language tasks, including VQAv2 (+4.9%), NLVR2
(+5.6%), ITR (R@1 on TR +17.2%, on IR + 15.6% ) and COCO caption generation
(CIDEr +6.5), demonstrating a large potential on training lightweight VLMs.


---

