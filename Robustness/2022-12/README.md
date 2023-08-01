## JamPatoisNLI: A Jamaican Patois Natural Language Inference Dataset

**Published Date:** 2022-12-07T03:07:02Z

**Link:** http://arxiv.org/pdf/2212.03419v1

**Abstract:**

  JamPatoisNLI provides the first dataset for natural language inference in a
creole language, Jamaican Patois. Many of the most-spoken low-resource
languages are creoles. These languages commonly have a lexicon derived from a
major world language and a distinctive grammar reflecting the languages of the
original speakers and the process of language birth by creolization. This gives
them a distinctive place in exploring the effectiveness of transfer from large
monolingual or multilingual pretrained models. While our work, along with
previous work, shows that transfer from these models to low-resource languages
that are unrelated to languages in their training set is not very effective, we
would expect stronger results from transfer to creoles. Indeed, our experiments
show considerably better results from few-shot learning of JamPatoisNLI than
for such unrelated languages, and help us begin to understand how the unique
relationship between creoles and their high-resource base languages affect
cross-lingual transfer. JamPatoisNLI, which consists of naturally-occurring
premises and expert-written hypotheses, is a step towards steering research
into a traditionally underserved language and a useful benchmark for
understanding cross-lingual NLP.


---

## MiLMo:Minority Multilingual Pre-trained Language Model

**Published Date:** 2022-12-04T09:28:17Z

**Link:** http://arxiv.org/pdf/2212.01779v2

**Abstract:**

  Pre-trained language models are trained on large-scale unsupervised data, and
they can fine-turn the model only on small-scale labeled datasets, and achieve
good results. Multilingual pre-trained language models can be trained on
multiple languages, and the model can understand multiple languages at the same
time. At present, the search on pre-trained models mainly focuses on rich
resources, while there is relatively little research on low-resource languages
such as minority languages, and the public multilingual pre-trained language
model can not work well for minority languages. Therefore, this paper
constructs a multilingual pre-trained model named MiLMo that performs better on
minority language tasks, including Mongolian, Tibetan, Uyghur, Kazakh and
Korean. To solve the problem of scarcity of datasets on minority languages and
verify the effectiveness of the MiLMo model, this paper constructs a minority
multilingual text classification dataset named MiTC, and trains a word2vec
model for each language. By comparing the word2vec model and the pre-trained
model in the text classification task, this paper provides an optimal scheme
for the downstream task research of minority languages. The final experimental
results show that the performance of the pre-trained model is better than that
of the word2vec model, and it has achieved the best results in minority
multilingual text classification. The multilingual pre-trained model MiLMo,
multilingual word2vec model and multilingual text classification dataset MiTC
are published on http://milmo.cmli-nlp.com/.


---

## Lessons learned from the evaluation of Spanish Language Models

**Published Date:** 2022-12-16T10:33:38Z

**Link:** http://arxiv.org/pdf/2212.08390v1

**Abstract:**

  Given the impact of language models on the field of Natural Language
Processing, a number of Spanish encoder-only masked language models (aka BERTs)
have been trained and released. These models were developed either within large
projects using very large private corpora or by means of smaller scale academic
efforts leveraging freely available data. In this paper we present a
comprehensive head-to-head comparison of language models for Spanish with the
following results: (i) Previously ignored multilingual models from large
companies fare better than monolingual models, substantially changing the
evaluation landscape of language models in Spanish; (ii) Results across the
monolingual models are not conclusive, with supposedly smaller and inferior
models performing competitively. Based on these empirical results, we argue for
the need of more research to understand the factors underlying them. In this
sense, the effect of corpus size, quality and pre-training techniques need to
be further investigated to be able to obtain Spanish monolingual models
significantly better than the multilingual ones released by large private
companies, specially in the face of rapid ongoing progress in the field. The
recent activity in the development of language technology for Spanish is to be
welcomed, but our results show that building language models remains an open,
resource-heavy problem which requires to marry resources (monetary and/or
computational) with the best research expertise and practice.


---

## Prompting Is Programming: A Query Language for Large Language Models

**Published Date:** 2022-12-12T18:09:09Z

**Link:** http://arxiv.org/pdf/2212.06094v3

**Abstract:**

  Large language models have demonstrated outstanding performance on a wide
range of tasks such as question answering and code generation. On a high level,
given an input, a language model can be used to automatically complete the
sequence in a statistically-likely way. Based on this, users prompt these
models with language instructions or examples, to implement a variety of
downstream tasks. Advanced prompting methods can even imply interaction between
the language model, a user, and external tools such as calculators. However, to
obtain state-of-the-art performance or adapt language models for specific
tasks, complex task- and model-specific programs have to be implemented, which
may still require ad-hoc interaction.
  Based on this, we present the novel idea of Language Model Programming (LMP).
LMP generalizes language model prompting from pure text prompts to an intuitive
combination of text prompting and scripting. Additionally, LMP allows
constraints to be specified over the language model output. This enables easy
adaption to many tasks while abstracting language model internals and providing
high-level semantics.
  To enable LMP, we implement LMQL(short for Language Model Query Language),
which leverages the constraints and control flow from an LMP prompt to generate
an efficient inference procedure that minimizes the number of expensive calls
to the underlying language model.
  We show that LMQL can capture a wide range of state-of-the-art prompting
methods in an intuitive way, especially facilitating interactive flows that are
challenging to implement with existing high-level APIs. Our evaluation shows
that we retain or increase the accuracy on several downstream tasks, while also
significantly reducing the required amount of computation or cost in the case
of pay-to-use APIs (26-85% cost savings).


---

## Large Language Models Meet NL2Code: A Survey

**Published Date:** 2022-12-19T12:55:32Z

**Link:** http://arxiv.org/pdf/2212.09420v2

**Abstract:**

  The task of generating code from a natural language description, or NL2Code,
is considered a pressing and significant challenge in code intelligence. Thanks
to the rapid development of pre-training techniques, surging large language
models are being proposed for code, sparking the advances in NL2Code. To
facilitate further research and applications in this field, in this paper, we
present a comprehensive survey of 27 existing large language models for
NL2Code, and also review benchmarks and metrics. We provide an intuitive
comparison of all existing models on the HumanEval benchmark. Through in-depth
observation and analysis, we provide some insights and conclude that the key
factors contributing to the success of large language models for NL2Code are
"Large Size, Premium Data, Expert Tuning". In addition, we discuss challenges
and opportunities regarding the gap between models and humans. We also create a
website https://nl2code.github.io to track the latest progress through
crowd-sourcing. To the best of our knowledge, this is the first survey of large
language models for NL2Code, and we believe it will contribute to the ongoing
development of the field.


---

## Go-tuning: Improving Zero-shot Learning Abilities of Smaller Language
  Models

**Published Date:** 2022-12-20T17:36:49Z

**Link:** http://arxiv.org/pdf/2212.10461v1

**Abstract:**

  With increasing scale, large language models demonstrate both quantitative
improvement and new qualitative capabilities, especially as zero-shot learners,
like GPT-3. However, these results rely heavily on delicate prompt design and
large computation. In this work, we explore whether the strong zero-shot
ability could be achieved at a smaller model scale without any external
supervised data. To achieve this goal, we revisit masked language modeling and
present a geometry-guided self-supervised learning method (Go-tuningfor short)
by taking a small number of task-aware self-supervised data to update language
models further. Experiments show that Go-tuning can enable T5-small (80M)
competitive zero-shot results compared with large language models, such as
T5-XL (3B). We also apply Go-tuning on multi-task settings and develop a
multi-task model, mgo-T5 (250M). It can reach the average performance of OPT
(175B) on 9 datasets.


---

## ALERT: Adapting Language Models to Reasoning Tasks

**Published Date:** 2022-12-16T05:15:41Z

**Link:** http://arxiv.org/pdf/2212.08286v2

**Abstract:**

  Current large language models can perform reasonably well on complex tasks
that require step-by-step reasoning with few-shot learning. Are these models
applying reasoning skills they have learnt during pre-training and reason
outside of their training context, or are they simply memorizing their training
corpus at finer granularity and have learnt to better understand their context?
To tease apart these possibilities, we introduce ALERT, a benchmark and suite
of analyses for assessing language models' reasoning ability comparing
pre-trained and finetuned models on complex tasks that require reasoning skills
to solve. ALERT provides a test bed to asses any language model on fine-grained
reasoning skills, which spans over 20 datasets and covers 10 different
reasoning skills. We leverage ALERT to further investigate the role of
finetuning. With extensive empirical analysis we find that language models
learn more reasoning skills such as textual entailment, abductive reasoning,
and analogical reasoning during finetuning stage compared to pretraining state.
We also find that when language models are finetuned they tend to overfit to
the prompt template, which hurts the robustness of models causing
generalization problems.


---

## MiLMo:Minority Multilingual Pre-trained Language Model

**Published Date:** 2022-12-04T09:28:17Z

**Link:** http://arxiv.org/pdf/2212.01779v2

**Abstract:**

  Pre-trained language models are trained on large-scale unsupervised data, and
they can fine-turn the model only on small-scale labeled datasets, and achieve
good results. Multilingual pre-trained language models can be trained on
multiple languages, and the model can understand multiple languages at the same
time. At present, the search on pre-trained models mainly focuses on rich
resources, while there is relatively little research on low-resource languages
such as minority languages, and the public multilingual pre-trained language
model can not work well for minority languages. Therefore, this paper
constructs a multilingual pre-trained model named MiLMo that performs better on
minority language tasks, including Mongolian, Tibetan, Uyghur, Kazakh and
Korean. To solve the problem of scarcity of datasets on minority languages and
verify the effectiveness of the MiLMo model, this paper constructs a minority
multilingual text classification dataset named MiTC, and trains a word2vec
model for each language. By comparing the word2vec model and the pre-trained
model in the text classification task, this paper provides an optimal scheme
for the downstream task research of minority languages. The final experimental
results show that the performance of the pre-trained model is better than that
of the word2vec model, and it has achieved the best results in minority
multilingual text classification. The multilingual pre-trained model MiLMo,
multilingual word2vec model and multilingual text classification dataset MiTC
are published on http://milmo.cmli-nlp.com/.


---

## Go-tuning: Improving Zero-shot Learning Abilities of Smaller Language
  Models

**Published Date:** 2022-12-20T17:36:49Z

**Link:** http://arxiv.org/pdf/2212.10461v1

**Abstract:**

  With increasing scale, large language models demonstrate both quantitative
improvement and new qualitative capabilities, especially as zero-shot learners,
like GPT-3. However, these results rely heavily on delicate prompt design and
large computation. In this work, we explore whether the strong zero-shot
ability could be achieved at a smaller model scale without any external
supervised data. To achieve this goal, we revisit masked language modeling and
present a geometry-guided self-supervised learning method (Go-tuningfor short)
by taking a small number of task-aware self-supervised data to update language
models further. Experiments show that Go-tuning can enable T5-small (80M)
competitive zero-shot results compared with large language models, such as
T5-XL (3B). We also apply Go-tuning on multi-task settings and develop a
multi-task model, mgo-T5 (250M). It can reach the average performance of OPT
(175B) on 9 datasets.


---

## ZEROTOP: Zero-Shot Task-Oriented Semantic Parsing using Large Language
  Models

**Published Date:** 2022-12-21T07:06:55Z

**Link:** http://arxiv.org/pdf/2212.10815v1

**Abstract:**

  We explore the use of large language models (LLMs) for zero-shot semantic
parsing. Semantic parsing involves mapping natural language utterances to
task-specific meaning representations. Language models are generally trained on
the publicly available text and code and cannot be expected to directly
generalize to domain-specific parsing tasks in a zero-shot setting. In this
work, we propose ZEROTOP, a zero-shot task-oriented parsing method that
decomposes a semantic parsing problem into a set of abstractive and extractive
question-answering (QA) problems, enabling us to leverage the ability of LLMs
to zero-shot answer reading comprehension questions. For each utterance, we
prompt the LLM with questions corresponding to its top-level intent and a set
of slots and use the LLM generations to construct the target meaning
representation. We observe that current LLMs fail to detect unanswerable
questions; and as a result, cannot handle questions corresponding to missing
slots. To address this problem, we fine-tune a language model on public QA
datasets using synthetic negative samples. Experimental results show that our
QA-based decomposition paired with the fine-tuned LLM can correctly parse ~16%
of utterances in the MTOP dataset without requiring any annotated data.


---

## LegalRelectra: Mixed-domain Language Modeling for Long-range Legal Text
  Comprehension

**Published Date:** 2022-12-16T00:15:14Z

**Link:** http://arxiv.org/pdf/2212.08204v1

**Abstract:**

  The application of Natural Language Processing (NLP) to specialized domains,
such as the law, has recently received a surge of interest. As many legal
services rely on processing and analyzing large collections of documents,
automating such tasks with NLP tools emerges as a key challenge. Many popular
language models, such as BERT or RoBERTa, are general-purpose models, which
have limitations on processing specialized legal terminology and syntax. In
addition, legal documents may contain specialized vocabulary from other
domains, such as medical terminology in personal injury text. Here, we propose
LegalRelectra, a legal-domain language model that is trained on mixed-domain
legal and medical corpora. We show that our model improves over general-domain
and single-domain medical and legal language models when processing
mixed-domain (personal injury) text. Our training architecture implements the
Electra framework, but utilizes Reformer instead of BERT for its generator and
discriminator. We show that this improves the model's performance on processing
long passages and results in better long-range text comprehension.


---

## Improved Long-Form Spoken Language Translation with Large Language
  Models

**Published Date:** 2022-12-19T22:36:53Z

**Link:** http://arxiv.org/pdf/2212.09895v1

**Abstract:**

  A challenge in spoken language translation is that plenty of spoken content
is long-form, but short units are necessary for obtaining high-quality
translations. To address this mismatch, we fine-tune a general-purpose, large
language model to split long ASR transcripts into segments that can be
independently translated so as to maximize the overall translation quality. We
compare to several segmentation strategies and find that our approach improves
BLEU score on three languages by an average of 2.7 BLEU overall compared to an
automatic punctuation baseline. Further, we demonstrate the effectiveness of
two constrained decoding strategies to improve well-formedness of the model
output from above 99% to 100%.


---

## Cramming: Training a Language Model on a Single GPU in One Day

**Published Date:** 2022-12-28T18:59:28Z

**Link:** http://arxiv.org/pdf/2212.14034v1

**Abstract:**

  Recent trends in language modeling have focused on increasing performance
through scaling, and have resulted in an environment where training language
models is out of reach for most researchers and practitioners. While most in
the community are asking how to push the limits of extreme computation, we ask
the opposite question: How far can we get with a single GPU in just one day?
  We investigate the downstream performance achievable with a transformer-based
language model trained completely from scratch with masked language modeling
for a single day on a single consumer GPU. Aside from re-analyzing nearly all
components of the pretraining pipeline for this scenario and providing a
modified pipeline with performance close to BERT, we investigate why scaling
down is hard, and which modifications actually improve performance in this
scenario. We provide evidence that even in this constrained setting,
performance closely follows scaling laws observed in large-compute settings.
Through the lens of scaling laws, we categorize a range of recent improvements
to training and architecture and discuss their merit and practical
applicability (or lack thereof) for the limited compute setting.


---

## Understanding Stereotypes in Language Models: Towards Robust Measurement
  and Zero-Shot Debiasing

**Published Date:** 2022-12-20T22:41:24Z

**Link:** http://arxiv.org/pdf/2212.10678v1

**Abstract:**

  Generated texts from large pretrained language models have been shown to
exhibit a variety of harmful, human-like biases about various demographics.
These findings prompted large efforts aiming to understand and measure such
effects, with the goal of providing benchmarks that can guide the development
of techniques mitigating these stereotypical associations. However, as recent
research has pointed out, the current benchmarks lack a robust experimental
setup, consequently hindering the inference of meaningful conclusions from
their evaluation metrics. In this paper, we extend these arguments and
demonstrate that existing techniques and benchmarks aiming to measure
stereotypes tend to be inaccurate and consist of a high degree of experimental
noise that severely limits the knowledge we can gain from benchmarking language
models based on them. Accordingly, we propose a new framework for robustly
measuring and quantifying biases exhibited by generative language models.
Finally, we use this framework to investigate GPT-3's occupational gender bias
and propose prompting techniques for mitigating these biases without the need
for fine-tuning.


---

## MiLMo:Minority Multilingual Pre-trained Language Model

**first_author:** Junjie Deng et al.

**Published Date:** 2022-12-04T09:28:17Z

**Link:** http://arxiv.org/pdf/2212.01779v2

**Abstract:**

  Pre-trained language models are trained on large-scale unsupervised data, and
they can fine-turn the model only on small-scale labeled datasets, and achieve
good results. Multilingual pre-trained language models can be trained on
multiple languages, and the model can understand multiple languages at the same
time. At present, the search on pre-trained models mainly focuses on rich
resources, while there is relatively little research on low-resource languages
such as minority languages, and the public multilingual pre-trained language
model can not work well for minority languages. Therefore, this paper
constructs a multilingual pre-trained model named MiLMo that performs better on
minority language tasks, including Mongolian, Tibetan, Uyghur, Kazakh and
Korean. To solve the problem of scarcity of datasets on minority languages and
verify the effectiveness of the MiLMo model, this paper constructs a minority
multilingual text classification dataset named MiTC, and trains a word2vec
model for each language. By comparing the word2vec model and the pre-trained
model in the text classification task, this paper provides an optimal scheme
for the downstream task research of minority languages. The final experimental
results show that the performance of the pre-trained model is better than that
of the word2vec model, and it has achieved the best results in minority
multilingual text classification. The multilingual pre-trained model MiLMo,
multilingual word2vec model and multilingual text classification dataset MiTC
are published on http://milmo.cmli-nlp.com/.


---

## ALERT: Adapting Language Models to Reasoning Tasks

**first_author:** Ping Yu et al.

**Published Date:** 2022-12-16T05:15:41Z

**Link:** http://arxiv.org/pdf/2212.08286v2

**Abstract:**

  Current large language models can perform reasonably well on complex tasks
that require step-by-step reasoning with few-shot learning. Are these models
applying reasoning skills they have learnt during pre-training and reason
outside of their training context, or are they simply memorizing their training
corpus at finer granularity and have learnt to better understand their context?
To tease apart these possibilities, we introduce ALERT, a benchmark and suite
of analyses for assessing language models' reasoning ability comparing
pre-trained and finetuned models on complex tasks that require reasoning skills
to solve. ALERT provides a test bed to asses any language model on fine-grained
reasoning skills, which spans over 20 datasets and covers 10 different
reasoning skills. We leverage ALERT to further investigate the role of
finetuning. With extensive empirical analysis we find that language models
learn more reasoning skills such as textual entailment, abductive reasoning,
and analogical reasoning during finetuning stage compared to pretraining state.
We also find that when language models are finetuned they tend to overfit to
the prompt template, which hurts the robustness of models causing
generalization problems.


---

