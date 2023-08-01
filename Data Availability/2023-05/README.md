## Instruction Tuning with Lexicons for Zero-Shot Style Classification

**Published Date:** 2023-05-24T00:17:36Z

**Link:** http://arxiv.org/pdf/2305.14592v1

**Abstract:**

  Style is used to convey authors' intentions and attitudes. Despite the
success of large pre-trained language models on style classification, prior
work relies on fine-tuning with labeled examples. Prompting large language
models to classify style without fine-tuning is challenging because language
styles can be difficult to define. In this study, we investigate the
effectiveness of style lexicons as a means for instructing language models how
to identify new styles that are unseen during training. Our experiments show
that lexicon-based instructions improve transfer zero-shot performance
significantly. We will release our code and data.


---

## Gender Lost In Translation: How Bridging The Gap Between Languages
  Affects Gender Bias in Zero-Shot Multilingual Translation

**Published Date:** 2023-05-26T13:51:50Z

**Link:** http://arxiv.org/pdf/2305.16935v1

**Abstract:**

  Neural machine translation (NMT) models often suffer from gender biases that
harm users and society at large. In this work, we explore how bridging the gap
between languages for which parallel data is not available affects gender bias
in multilingual NMT, specifically for zero-shot directions. We evaluate
translation between grammatical gender languages which requires preserving the
inherent gender information from the source in the target language. We study
the effect of encouraging language-agnostic hidden representations on models'
ability to preserve gender and compare pivot-based and zero-shot translation
regarding the influence of the bridge language (participating in all language
pairs during training) on gender preservation. We find that language-agnostic
representations mitigate zero-shot models' masculine bias, and with increased
levels of gender inflection in the bridge language, pivoting surpasses
zero-shot translation regarding fairer gender preservation for speaker-related
gender agreement.


---

## Cross-Lingual Supervision improves Large Language Models Pre-training

**Published Date:** 2023-05-19T16:14:07Z

**Link:** http://arxiv.org/pdf/2305.11778v1

**Abstract:**

  The recent rapid progress in pre-training Large Language Models has relied on
using self-supervised language modeling objectives like next token prediction
or span corruption. On the other hand, Machine Translation Systems are mostly
trained using cross-lingual supervision that requires aligned data between
source and target languages. We demonstrate that pre-training Large Language
Models on a mixture of a self-supervised Language Modeling objective and the
supervised Machine Translation objective, therefore including cross-lingual
parallel data during pre-training, yields models with better in-context
learning abilities. As pre-training is a very resource-intensive process and a
grid search on the best mixing ratio between the two objectives is
prohibitively expensive, we propose a simple yet effective strategy to learn it
during pre-training.


---

## Parameter-Efficient Cross-lingual Transfer of Vision and Language Models
  via Translation-based Alignment

**Published Date:** 2023-05-02T14:09:02Z

**Link:** http://arxiv.org/pdf/2305.03510v1

**Abstract:**

  Pre-trained vision and language models such as CLIP have witnessed remarkable
success in connecting images and texts with a primary focus on English texts.
Despite recent efforts to extend CLIP to support other languages, disparities
in performance among different languages have been observed due to uneven
resource availability. Additionally, current cross-lingual transfer methods of
those pre-trained models would consume excessive resources for a large number
of languages. Therefore, we propose a new parameter-efficient cross-lingual
transfer learning framework that utilizes a translation-based alignment method
to mitigate multilingual disparities and explores parameter-efficient
fine-tuning methods for parameter-efficient cross-lingual transfer. Extensive
experiments on XTD and Multi30K datasets, covering 11 languages under
zero-shot, few-shot, and full-dataset learning scenarios, show that our
framework significantly reduces the multilingual disparities among languages
and improves cross-lingual transfer results, especially in low-resource
scenarios, while only keeping and fine-tuning an extremely small number of
parameters compared to the full model (e.g., Our framework only requires 0.16\%
additional parameters of a full-model for each language in the few-shot
learning scenario).


---

## Panda LLM: Training Data and Evaluation for Open-Sourced Chinese
  Instruction-Following Large Language Models

**Published Date:** 2023-05-04T17:49:09Z

**Link:** http://arxiv.org/pdf/2305.03025v1

**Abstract:**

  This project focuses on enhancing open-source large language models through
instruction-tuning and providing comprehensive evaluations of their
performance. We explore how various training data factors, such as quantity,
quality, and linguistic distribution, influence the performance of
instruction-tuned models trained on publicly accessible high-quality
instruction datasets for both English and Chinese languages. Our goal is to
supplement evaluation with quantitative analyses, providing valuable insights
for the continued advancement of open-source chat models. Our model, data, and
code are publicly available for others to use and build upon.


---

## Taxi1500: A Multilingual Dataset for Text Classification in 1500
  Languages

**Published Date:** 2023-05-15T09:43:32Z

**Link:** http://arxiv.org/pdf/2305.08487v1

**Abstract:**

  While natural language processing tools have been developed extensively for
some of the world's languages, a significant portion of the world's over 7000
languages are still neglected. One reason for this is that evaluation datasets
do not yet cover a wide range of languages, including low-resource and
endangered ones. We aim to address this issue by creating a text classification
dataset encompassing a large number of languages, many of which currently have
little to no annotated data available. We leverage parallel translations of the
Bible to construct such a dataset by first developing applicable topics and
employing a crowdsourcing tool to collect annotated data. By annotating the
English side of the data and projecting the labels onto other languages through
aligned verses, we generate text classification datasets for more than 1500
languages. We extensively benchmark several existing multilingual language
models using our dataset. To facilitate the advancement of research in this
area, we will release our dataset and code.


---

## Instruction Tuning with Lexicons for Zero-Shot Style Classification

**Published Date:** 2023-05-24T00:17:36Z

**Link:** http://arxiv.org/pdf/2305.14592v1

**Abstract:**

  Style is used to convey authors' intentions and attitudes. Despite the
success of large pre-trained language models on style classification, prior
work relies on fine-tuning with labeled examples. Prompting large language
models to classify style without fine-tuning is challenging because language
styles can be difficult to define. In this study, we investigate the
effectiveness of style lexicons as a means for instructing language models how
to identify new styles that are unseen during training. Our experiments show
that lexicon-based instructions improve transfer zero-shot performance
significantly. We will release our code and data.


---

## Panda LLM: Training Data and Evaluation for Open-Sourced Chinese
  Instruction-Following Large Language Models

**Published Date:** 2023-05-04T17:49:09Z

**Link:** http://arxiv.org/pdf/2305.03025v1

**Abstract:**

  This project focuses on enhancing open-source large language models through
instruction-tuning and providing comprehensive evaluations of their
performance. We explore how various training data factors, such as quantity,
quality, and linguistic distribution, influence the performance of
instruction-tuned models trained on publicly accessible high-quality
instruction datasets for both English and Chinese languages. Our goal is to
supplement evaluation with quantitative analyses, providing valuable insights
for the continued advancement of open-source chat models. Our model, data, and
code are publicly available for others to use and build upon.


---

## Benchmarking Arabic AI with Large Language Models

**Published Date:** 2023-05-24T10:16:16Z

**Link:** http://arxiv.org/pdf/2305.14982v1

**Abstract:**

  With large Foundation Models (FMs), language technologies (AI in general) are
entering a new paradigm: eliminating the need for developing large-scale
task-specific datasets and supporting a variety of tasks through set-ups
ranging from zero-shot to few-shot learning. However, understanding FMs
capabilities requires a systematic benchmarking effort by comparing FMs
performance with the state-of-the-art (SOTA) task-specific models. With that
goal, past work focused on the English language and included a few efforts with
multiple languages. Our study contributes to ongoing research by evaluating FMs
performance for standard Arabic NLP and Speech processing, including a range of
tasks from sequence tagging to content classification across diverse domains.
We start with zero-shot learning using GPT-3.5-turbo, Whisper, and USM,
addressing 33 unique tasks using 59 publicly available datasets resulting in 96
test setups. For a few tasks, FMs performs on par or exceeds the performance of
the SOTA models but for the majority it under-performs. Given the importance of
prompt for the FMs performance, we discuss our prompt strategies in detail and
elaborate on our findings. Our future work on Arabic AI will explore few-shot
prompting, expand the range of tasks, and investigate additional open-source
models.


---

## Instruction Tuning with Lexicons for Zero-Shot Style Classification

**first_author:** Ruohao Guo et al.

**Published Date:** 2023-05-24T00:17:36Z

**Link:** http://arxiv.org/pdf/2305.14592v1

**Abstract:**

  Style is used to convey authors' intentions and attitudes. Despite the
success of large pre-trained language models on style classification, prior
work relies on fine-tuning with labeled examples. Prompting large language
models to classify style without fine-tuning is challenging because language
styles can be difficult to define. In this study, we investigate the
effectiveness of style lexicons as a means for instructing language models how
to identify new styles that are unseen during training. Our experiments show
that lexicon-based instructions improve transfer zero-shot performance
significantly. We will release our code and data.


---

## Gender Lost In Translation: How Bridging The Gap Between Languages
  Affects Gender Bias in Zero-Shot Multilingual Translation

**first_author:** Lena Cabrera et al.

**Published Date:** 2023-05-26T13:51:50Z

**Link:** http://arxiv.org/pdf/2305.16935v1

**Abstract:**

  Neural machine translation (NMT) models often suffer from gender biases that
harm users and society at large. In this work, we explore how bridging the gap
between languages for which parallel data is not available affects gender bias
in multilingual NMT, specifically for zero-shot directions. We evaluate
translation between grammatical gender languages which requires preserving the
inherent gender information from the source in the target language. We study
the effect of encouraging language-agnostic hidden representations on models'
ability to preserve gender and compare pivot-based and zero-shot translation
regarding the influence of the bridge language (participating in all language
pairs during training) on gender preservation. We find that language-agnostic
representations mitigate zero-shot models' masculine bias, and with increased
levels of gender inflection in the bridge language, pivoting surpasses
zero-shot translation regarding fairer gender preservation for speaker-related
gender agreement.


---

