## Are Multilingual Models the Best Choice for Moderately Under-resourced
  Languages? A Comprehensive Assessment for Catalan

**Published Date:** 2021-07-16T13:52:01Z

**Link:** http://arxiv.org/pdf/2107.07903v1

**Abstract:**

  Multilingual language models have been a crucial breakthrough as they
considerably reduce the need of data for under-resourced languages.
Nevertheless, the superiority of language-specific models has already been
proven for languages having access to large amounts of data. In this work, we
focus on Catalan with the aim to explore to what extent a medium-sized
monolingual language model is competitive with state-of-the-art large
multilingual models. For this, we: (1) build a clean, high-quality textual
Catalan corpus (CaText), the largest to date (but only a fraction of the usual
size of the previous work in monolingual language models), (2) train a
Transformer-based language model for Catalan (BERTa), and (3) devise a thorough
evaluation in a diversity of settings, comprising a complete array of
downstream tasks, namely, Part of Speech Tagging, Named Entity Recognition and
Classification, Text Classification, Question Answering, and Semantic Textual
Similarity, with most of the corresponding datasets being created ex novo. The
result is a new benchmark, the Catalan Language Understanding Benchmark (CLUB),
which we publish as an open resource, together with the clean textual corpus,
the language model, and the cleaning pipeline. Using state-of-the-art
multilingual models and a monolingual model trained only on Wikipedia as
baselines, we consistently observe the superiority of our model across tasks
and settings.


---

## Structural Guidance for Transformer Language Models

**Published Date:** 2021-07-30T23:14:51Z

**Link:** http://arxiv.org/pdf/2108.00104v1

**Abstract:**

  Transformer-based language models pre-trained on large amounts of text data
have proven remarkably successful in learning generic transferable linguistic
representations. Here we study whether structural guidance leads to more
human-like systematic linguistic generalization in Transformer language models
without resorting to pre-training on very large amounts of data. We explore two
general ideas. The "Generative Parsing" idea jointly models the incremental
parse and word sequence as part of the same sequence modeling task. The
"Structural Scaffold" idea guides the language model's representation via
additional structure loss that separately predicts the incremental constituency
parse. We train the proposed models along with a vanilla Transformer language
model baseline on a 14 million-token and a 46 million-token subset of the BLLIP
dataset, and evaluate models' syntactic generalization performances on SG Test
Suites and sized BLiMP. Experiment results across two benchmarks suggest
converging evidence that generative structural supervisions can induce more
robust and humanlike linguistic generalization in Transformer language models
without the need for data intensive pre-training.


---

## A Primer on Pretrained Multilingual Language Models

**Published Date:** 2021-07-01T18:01:46Z

**Link:** http://arxiv.org/pdf/2107.00676v2

**Abstract:**

  Multilingual Language Models (\MLLMs) such as mBERT, XLM, XLM-R,
\textit{etc.} have emerged as a viable option for bringing the power of
pretraining to a large number of languages. Given their success in zero-shot
transfer learning, there has emerged a large body of work in (i) building
bigger \MLLMs~covering a large number of languages (ii) creating exhaustive
benchmarks covering a wider variety of tasks and languages for evaluating
\MLLMs~ (iii) analysing the performance of \MLLMs~on monolingual, zero-shot
cross-lingual and bilingual tasks (iv) understanding the universal language
patterns (if any) learnt by \MLLMs~ and (v) augmenting the (often) limited
capacity of \MLLMs~ to improve their performance on seen or even unseen
languages. In this survey, we review the existing literature covering the above
broad areas of research pertaining to \MLLMs. Based on our survey, we recommend
some promising directions of future research.


---

