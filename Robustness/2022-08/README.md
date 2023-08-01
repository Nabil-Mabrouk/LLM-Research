## MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural
  Code Generation

**Published Date:** 2022-08-17T11:16:52Z

**Link:** http://arxiv.org/pdf/2208.08227v4

**Abstract:**

  Large language models have demonstrated the ability to generate both natural
language and programming language text. Such models open up the possibility of
multi-language code generation: could code generation models generalize
knowledge from one language to another? Although contemporary code generation
models can generate semantically correct Python code, little is known about
their abilities with other languages. We propose MultiPL-E, a system for
translating unit test-driven code generation benchmarks to new languages. We
create the first massively multilingual code generation benchmark by using
MultiPL-E to translate two popular Python code generation benchmarks to 18
additional programming languages.
  We use MultiPL-E to extend the HumanEval benchmark and MBPP benchmark to 18
languages that encompass a range of programming paradigms and popularity. Using
these new parallel benchmarks, we evaluate the multi-language performance of
three state-of-the-art code generation models: Codex, CodeGen, and InCoder. We
find that Codex matches or even exceeds its performance on Python for several
other languages. The range of programming languages represented in MultiPL-E
allow us to explore the impact of language frequency and language features on
model performance. Finally, the MultiPL-E approach of compiling code generation
benchmarks to new programming languages is both scalable and extensible, making
it straightforward to evaluate new models, benchmarks, and languages.


---

## IndicSUPERB: A Speech Processing Universal Performance Benchmark for
  Indian languages

**Published Date:** 2022-08-24T20:14:52Z

**Link:** http://arxiv.org/pdf/2208.11761v2

**Abstract:**

  A cornerstone in AI research has been the creation and adoption of
standardized training and test datasets to earmark the progress of
state-of-the-art models. A particularly successful example is the GLUE dataset
for training and evaluating Natural Language Understanding (NLU) models for
English. The large body of research around self-supervised BERT-based language
models revolved around performance improvements on NLU tasks in GLUE. To
evaluate language models in other languages, several language-specific GLUE
datasets were created. The area of speech language understanding (SLU) has
followed a similar trajectory. The success of large self-supervised models such
as wav2vec2 enable creation of speech models with relatively easy to access
unlabelled data. These models can then be evaluated on SLU tasks, such as the
SUPERB benchmark. In this work, we extend this to Indic languages by releasing
the IndicSUPERB benchmark. Specifically, we make the following three
contributions. (i) We collect Kathbath containing 1,684 hours of labelled
speech data across 12 Indian languages from 1,218 contributors located in 203
districts in India. (ii) Using Kathbath, we create benchmarks across 6 speech
tasks: Automatic Speech Recognition, Speaker Verification, Speaker
Identification (mono/multi), Language Identification, Query By Example, and
Keyword Spotting for 12 languages. (iii) On the released benchmarks, we train
and evaluate different self-supervised models alongside a commonly used
baseline FBANK. We show that language-specific fine-tuned models are more
accurate than baseline on most of the tasks, including a large gap of 76\% for
the Language Identification task. However, for speaker identification,
self-supervised models trained on large datasets demonstrate an advantage. We
hope IndicSUPERB contributes to the progress of developing speech language
understanding models for Indian languages.


---

## Training a T5 Using Lab-sized Resources

**Published Date:** 2022-08-25T13:55:16Z

**Link:** http://arxiv.org/pdf/2208.12097v1

**Abstract:**

  Training large neural language models on large datasets is resource- and
time-intensive. These requirements create a barrier to entry, where those with
fewer resources cannot build competitive models. This paper presents various
techniques for making it possible to (a) train a large language model using
resources that a modest research lab might have, and (b) train it in a
reasonable amount of time. We provide concrete recommendations for
practitioners, which we illustrate with a case study: a T5 model for Danish,
the first for this language.


---

## MulZDG: Multilingual Code-Switching Framework for Zero-shot Dialogue
  Generation

**Published Date:** 2022-08-18T04:28:20Z

**Link:** http://arxiv.org/pdf/2208.08629v1

**Abstract:**

  Building dialogue generation systems in a zero-shot scenario remains a huge
challenge, since the typical zero-shot approaches in dialogue generation rely
heavily on large-scale pre-trained language generation models such as GPT-3 and
T5. The research on zero-shot dialogue generation without cumbersome language
models is limited due to lacking corresponding parallel dialogue corpora. In
this paper, we propose a simple but effective Multilingual learning framework
for Zero-shot Dialogue Generation (dubbed as MulZDG) that can effectively
transfer knowledge from an English corpus with large-scale training samples to
a non-English corpus with zero samples. Besides, MulZDG can be viewed as a
multilingual data augmentation method to improve the performance of the
resource-rich language. First, we construct multilingual code-switching
dialogue datasets via translation utterances randomly selected from monolingual
English datasets. Then we employ MulZDG to train a unified multilingual
dialogue model based on the code-switching datasets. The MulZDG can conduct
implicit semantic alignment between different languages. Experiments on
DailyDialog and DSTC7 datasets demonstrate that MulZDG not only achieve
competitive performance under zero-shot case compared to training with
sufficient examples but also greatly improve the performance of the source
language.


---

