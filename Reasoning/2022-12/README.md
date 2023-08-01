## Can Retriever-Augmented Language Models Reason? The Blame Game Between
  the Retriever and the Language Model

**Published Date:** 2022-12-18T19:27:41Z

**Link:** http://arxiv.org/pdf/2212.09146v2

**Abstract:**

  Augmenting pretrained language models with retrievers to select the
supporting documents has shown promise in effectively solving common NLP
problems, including language modeling and question answering, in an
interpretable way. In this paper, we first study the strengths and weaknesses
of different retriever-augmented language models (REALM, $k$NN-LM, FiD coupled
with DPR, and ATLAS and Flan-T5 coupled with Contriever) in reasoning over the
retrieved statements in different tasks. We show how the retrieve-then-read
models' limitations in reasoning are rooted both in the retriever module as
well as the language model. Our experimental results demonstrate that the
similarity metric used by the retrievers is generally insufficient for
reasoning tasks. Additionally, we show that the language models in
retriever-augmented models do not take the complicated relations between the
statements into account, which leads to poor reasoning performance even when
using the larger models. Moreover, we analyze the reasoning performance of
large language models using multihop retrieval but we only observe minor
improvements. Overall, this shows great room for further research in this area.


---

## INCLUSIFY: A benchmark and a model for gender-inclusive German

**Published Date:** 2022-12-05T19:37:48Z

**Link:** http://arxiv.org/pdf/2212.02564v1

**Abstract:**

  Gender-inclusive language is important for achieving gender equality in
languages with gender inflections, such as German. While stirring some
controversy, it is increasingly adopted by companies and political
institutions. A handful of tools have been developed to help people use
gender-inclusive language by identifying instances of the generic masculine and
providing suggestions for more inclusive reformulations. In this report, we
define the underlying tasks in terms of natural language processing, and
present a dataset and measures for benchmarking them. We also present a model
that implements these tasks, by combining an inclusive language database with
an elaborate sequence of processing steps via standard pre-trained models. Our
model achieves a recall of 0.89 and a precision of 0.82 in our benchmark for
identifying exclusive language; and one of its top five suggestions is chosen
in real-world texts in 44% of cases. We sketch how the area could be further
advanced by training end-to-end models and using large language models; and we
urge the community to include more gender-inclusive texts in their training
data in order to not present an obstacle to the adoption of gender-inclusive
language. Through these efforts, we hope to contribute to restoring justice in
language and, to a small extent, in reality.


---

## LUNA: Language Understanding with Number Augmentations on Transformers
  via Number Plugins and Pre-training

**Published Date:** 2022-12-06T01:31:37Z

**Link:** http://arxiv.org/pdf/2212.02691v2

**Abstract:**

  Transformers are widely used in NLP tasks. However, current approaches to
leveraging transformers to understand language expose one weak spot: Number
understanding. In some scenarios, numbers frequently occur, especially in
semi-structured data like tables. But current approaches to rich-number tasks
with transformer-based language models abandon or lose some of the numeracy
information - e.g., breaking numbers into sub-word tokens - which leads to many
number-related errors. In this paper, we propose the LUNA framework which
improves the numerical reasoning and calculation capabilities of
transformer-based language models. With the number plugin of NumTok and NumBed,
LUNA represents each number as a whole to model input. With number
pre-training, including regression loss and model distillation, LUNA bridges
the gap between number and vocabulary embeddings. To the best of our knowledge,
this is the first work that explicitly injects numeracy capability into
language models using Number Plugins. Besides evaluating toy models on toy
tasks, we evaluate LUNA on three large-scale transformer models (RoBERTa, BERT,
TabBERT) over three different downstream tasks (TATQA, TabFact, CrediTrans),
and observe the performances of language models are constantly improved by
LUNA. The augmented models also improve the official baseline of TAT-QA (EM:
50.15 -> 59.58) and achieve SOTA performance on CrediTrans (F1 = 86.17).


---

