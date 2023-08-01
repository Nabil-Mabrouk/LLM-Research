## Sequence to sequence pretraining for a less-resourced Slovenian language

**Published Date:** 2022-07-28T10:08:50Z

**Link:** http://arxiv.org/pdf/2207.13988v2

**Abstract:**

  Large pretrained language models have recently conquered the area of natural
language processing. As an alternative to predominant masked language modelling
introduced in BERT, the T5 model has introduced a more general training
objective, namely sequence to sequence transformation, which includes masked
language model but more naturally fits text generation tasks such as machine
translation, summarization, question answering, text simplification, dialogue
systems, etc. The monolingual variants of T5 models have been limited to
well-resourced languages, while the massively multilingual T5 model supports
101 languages. In contrast, we trained two different sized T5-type sequence to
sequence models for morphologically rich Slovene language with much less
resources and analyzed their behavior on 11 tasks. Concerning classification
tasks, the SloT5 models mostly lag behind the monolingual Slovene SloBERTa
model but are useful for the generative tasks.


---

## Multilinguals at SemEval-2022 Task 11: Complex NER in Semantically
  Ambiguous Settings for Low Resource Languages

**Published Date:** 2022-07-14T13:00:41Z

**Link:** http://arxiv.org/pdf/2207.06882v1

**Abstract:**

  We leverage pre-trained language models to solve the task of complex NER for
two low-resource languages: Chinese and Spanish. We use the technique of Whole
Word Masking(WWM) to boost the performance of masked language modeling
objective on large and unsupervised corpora. We experiment with multiple neural
network architectures, incorporating CRF, BiLSTMs, and Linear Classifiers on
top of a fine-tuned BERT layer. All our models outperform the baseline by a
significant margin and our best performing model obtains a competitive position
on the evaluation leaderboard for the blind test set.


---

## Neural Data-to-Text Generation Based on Small Datasets: Comparing the
  Added Value of Two Semi-Supervised Learning Approaches on Top of a Large
  Language Model

**Published Date:** 2022-07-14T11:53:04Z

**Link:** http://arxiv.org/pdf/2207.06839v1

**Abstract:**

  This study discusses the effect of semi-supervised learning in combination
with pretrained language models for data-to-text generation. It is not known
whether semi-supervised learning is still helpful when a large-scale language
model is also supplemented. This study aims to answer this question by
comparing a data-to-text system only supplemented with a language model, to two
data-to-text systems that are additionally enriched by a data augmentation or a
pseudo-labeling semi-supervised learning approach.
  Results show that semi-supervised learning results in higher scores on
diversity metrics. In terms of output quality, extending the training set of a
data-to-text system with a language model using the pseudo-labeling approach
did increase text quality scores, but the data augmentation approach yielded
similar scores to the system without training set extension. These results
indicate that semi-supervised learning approaches can bolster output quality
and diversity, even when a language model is also present.


---

## PLM-ICD: Automatic ICD Coding with Pretrained Language Models

**Published Date:** 2022-07-12T03:56:28Z

**Link:** http://arxiv.org/pdf/2207.05289v1

**Abstract:**

  Automatically classifying electronic health records (EHRs) into diagnostic
codes has been challenging to the NLP community. State-of-the-art methods
treated this problem as a multilabel classification problem and proposed
various architectures to model this problem. However, these systems did not
leverage the superb performance of pretrained language models, which achieved
superb performance on natural language understanding tasks. Prior work has
shown that pretrained language models underperformed on this task with the
regular finetuning scheme. Therefore, this paper aims at analyzing the causes
of the underperformance and developing a framework for automatic ICD coding
with pretrained language models. We spotted three main issues through the
experiments: 1) large label space, 2) long input sequences, and 3) domain
mismatch between pretraining and fine-tuning. We propose PLMICD, a framework
that tackles the challenges with various strategies. The experimental results
show that our proposed framework can overcome the challenges and achieves
state-of-the-art performance in terms of multiple metrics on the benchmark
MIMIC data. The source code is available at https://github.com/MiuLab/PLM-ICD


---

