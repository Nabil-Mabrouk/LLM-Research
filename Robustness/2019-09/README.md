## Dynamic Fusion: Attentional Language Model for Neural Machine
  Translation

**Published Date:** 2019-09-11T07:14:58Z

**Link:** http://arxiv.org/pdf/1909.04879v1

**Abstract:**

  Neural Machine Translation (NMT) can be used to generate fluent output. As
such, language models have been investigated for incorporation with NMT. In
prior investigations, two models have been used: a translation model and a
language model. The translation model's predictions are weighted by the
language model with a hand-crafted ratio in advance. However, these approaches
fail to adopt the language model weighting with regard to the translation
history. In another line of approach, language model prediction is incorporated
into the translation model by jointly considering source and target
information. However, this line of approach is limited because it largely
ignores the adequacy of the translation output.
  Accordingly, this work employs two mechanisms, the translation model and the
language model, with an attentive architecture to the language model as an
auxiliary element of the translation model. Compared with previous work in
English--Japanese machine translation using a language model, the experimental
results obtained with the proposed Dynamic Fusion mechanism improve BLEU and
Rank-based Intuitive Bilingual Evaluation Scores (RIBES) scores. Additionally,
in the analyses of the attention and predictivity of the language model, the
Dynamic Fusion mechanism allows predictive language modeling that conforms to
the appropriate grammatical structure.


---

## Reweighted Proximal Pruning for Large-Scale Language Representation

**Published Date:** 2019-09-27T04:10:10Z

**Link:** http://arxiv.org/pdf/1909.12486v2

**Abstract:**

  Recently, pre-trained language representation flourishes as the mainstay of
the natural language understanding community, e.g., BERT. These pre-trained
language representations can create state-of-the-art results on a wide range of
downstream tasks. Along with continuous significant performance improvement,
the size and complexity of these pre-trained neural models continue to increase
rapidly. Is it possible to compress these large-scale language representation
models? How will the pruned language representation affect the downstream
multi-task transfer learning objectives? In this paper, we propose Reweighted
Proximal Pruning (RPP), a new pruning method specifically designed for a
large-scale language representation model. Through experiments on SQuAD and the
GLUE benchmark suite, we show that proximal pruned BERT keeps high accuracy for
both the pre-training task and the downstream multiple fine-tuning tasks at
high prune ratio. RPP provides a new perspective to help us analyze what
large-scale language representation might learn. Additionally, RPP makes it
possible to deploy a large state-of-the-art language representation model such
as BERT on a series of distinct devices (e.g., online servers, mobile phones,
and edge devices).


---

## Ouroboros: On Accelerating Training of Transformer-Based Language Models

**Published Date:** 2019-09-14T23:21:56Z

**Link:** http://arxiv.org/pdf/1909.06695v1

**Abstract:**

  Language models are essential for natural language processing (NLP) tasks,
such as machine translation and text summarization. Remarkable performance has
been demonstrated recently across many NLP domains via a Transformer-based
language model with over a billion parameters, verifying the benefits of model
size. Model parallelism is required if a model is too large to fit in a single
computing device. Current methods for model parallelism either suffer from
backward locking in backpropagation or are not applicable to language models.
We propose the first model-parallel algorithm that speeds the training of
Transformer-based language models. We also prove that our proposed algorithm is
guaranteed to converge to critical points for non-convex problems. Extensive
experiments on Transformer and Transformer-XL language models demonstrate that
the proposed algorithm obtains a much faster speedup beyond data parallelism,
with comparable or better accuracy. Code to reproduce experiments is to be
found at \url{https://github.com/LaraQianYang/Ouroboros}.


---

## Large-Scale Multilingual Speech Recognition with a Streaming End-to-End
  Model

**Published Date:** 2019-09-11T19:46:21Z

**Link:** http://arxiv.org/pdf/1909.05330v1

**Abstract:**

  Multilingual end-to-end (E2E) models have shown great promise in expansion of
automatic speech recognition (ASR) coverage of the world's languages. They have
shown improvement over monolingual systems, and have simplified training and
serving by eliminating language-specific acoustic, pronunciation, and language
models. This work presents an E2E multilingual system which is equipped to
operate in low-latency interactive applications, as well as handle a key
challenge of real world data: the imbalance in training data across languages.
Using nine Indic languages, we compare a variety of techniques, and find that a
combination of conditioning on a language vector and training language-specific
adapter layers produces the best model. The resulting E2E multilingual model
achieves a lower word error rate (WER) than both monolingual E2E models (eight
of nine languages) and monolingual conventional systems (all nine languages).


---

## Improving Pre-Trained Multilingual Models with Vocabulary Expansion

**Published Date:** 2019-09-26T23:56:07Z

**Link:** http://arxiv.org/pdf/1909.12440v1

**Abstract:**

  Recently, pre-trained language models have achieved remarkable success in a
broad range of natural language processing tasks. However, in multilingual
setting, it is extremely resource-consuming to pre-train a deep language model
over large-scale corpora for each language. Instead of exhaustively
pre-training monolingual language models independently, an alternative solution
is to pre-train a powerful multilingual deep language model over large-scale
corpora in hundreds of languages. However, the vocabulary size for each
language in such a model is relatively small, especially for low-resource
languages. This limitation inevitably hinders the performance of these
multilingual models on tasks such as sequence labeling, wherein in-depth
token-level or sentence-level understanding is essential.
  In this paper, inspired by previous methods designed for monolingual
settings, we investigate two approaches (i.e., joint mapping and mixture
mapping) based on a pre-trained multilingual model BERT for addressing the
out-of-vocabulary (OOV) problem on a variety of tasks, including part-of-speech
tagging, named entity recognition, machine translation quality estimation, and
machine reading comprehension. Experimental results show that using mixture
mapping is more promising. To the best of our knowledge, this is the first work
that attempts to address and discuss the OOV issue in multilingual settings.


---

## Multilingual Graphemic Hybrid ASR with Massive Data Augmentation

**Published Date:** 2019-09-14T03:46:49Z

**Link:** http://arxiv.org/pdf/1909.06522v3

**Abstract:**

  Towards developing high-performing ASR for low-resource languages, approaches
to address the lack of resources are to make use of data from multiple
languages, and to augment the training data by creating acoustic variations. In
this work we present a single grapheme-based ASR model learned on 7
geographically proximal languages, using standard hybrid BLSTM-HMM acoustic
models with lattice-free MMI objective. We build the single ASR grapheme set
via taking the union over each language-specific grapheme set, and we find such
multilingual graphemic hybrid ASR model can perform language-independent
recognition on all 7 languages, and substantially outperform each monolingual
ASR model. Secondly, we evaluate the efficacy of multiple data augmentation
alternatives within language, as well as their complementarity with
multilingual modeling. Overall, we show that the proposed multilingual
graphemic hybrid ASR with various data augmentation can not only recognize any
within training set languages, but also provide large ASR performance
improvements.


---

## Low-Resource Parsing with Crosslingual Contextualized Representations

**Published Date:** 2019-09-19T00:23:09Z

**Link:** http://arxiv.org/pdf/1909.08744v1

**Abstract:**

  Despite advances in dependency parsing, languages with small treebanks still
present challenges. We assess recent approaches to multilingual contextual word
representations (CWRs), and compare them for crosslingual transfer from a
language with a large treebank to a language with a small or nonexistent
treebank, by sharing parameters between languages in the parser itself. We
experiment with a diverse selection of languages in both simulated and truly
low-resource scenarios, and show that multilingual CWRs greatly facilitate
low-resource dependency parsing even without crosslingual supervision such as
dictionaries or parallel text. Furthermore, we examine the non-contextual part
of the learned language models (which we call a "decontextual probe") to
demonstrate that polyglot language models better encode crosslingual lexical
correspondence compared to aligned monolingual language models. This analysis
provides further evidence that polyglot training is an effective approach to
crosslingual transfer.


---

## Phrase-Level Class based Language Model for Mandarin Smart Speaker Query
  Recognition

**Published Date:** 2019-09-02T05:55:36Z

**Link:** http://arxiv.org/pdf/1909.00556v1

**Abstract:**

  The success of speech assistants requires precise recognition of a number of
entities on particular contexts. A common solution is to train a class-based
n-gram language model and then expand the classes into specific words or
phrases. However, when the class has a huge list, e.g., more than 20 million
songs, a fully expansion will cause memory explosion. Worse still, the list
items in the class need to be updated frequently, which requires a dynamic
model updating technique. In this work, we propose to train pruned language
models for the word classes to replace the slots in the root n-gram. We further
propose to use a novel technique, named Difference Language Model (DLM), to
correct the bias from the pruned language models. Once the decoding graph is
built, we only need to recalculate the DLM when the entities in word classes
are updated. Results show that the proposed method consistently and
significantly outperforms the conventional approaches on all datasets, esp. for
large lists, which the conventional approaches cannot handle.


---

## Process Query Language: Design, Implementation, and Evaluation

**Published Date:** 2019-09-20T14:50:36Z

**Link:** http://arxiv.org/pdf/1909.09543v1

**Abstract:**

  Organizations can benefit from the use of practices, techniques, and tools
from the area of business process management. Through the focus on processes,
they create process models that require management, including support for
versioning, refactoring and querying. Querying thus far has primarily focused
on structural properties of models rather than on exploiting behavioral
properties capturing aspects of model execution. While the latter is more
challenging, it is also more effective, especially when models are used for
auditing or process automation. The focus of this paper is to overcome the
challenges associated with behavioral querying of process models in order to
unlock its benefits. The first challenge concerns determining decidability of
the building blocks of the query language, which are the possible behavioral
relations between process tasks. The second challenge concerns achieving
acceptable performance of query evaluation. The evaluation of a query may
require expensive checks in all process models, of which there may be
thousands. In light of these challenges, this paper proposes a special-purpose
programming language, namely Process Query Language (PQL) for behavioral
querying of process model collections. The language relies on a set of
behavioral predicates between process tasks, whose usefulness has been
empirically evaluated with a pool of process model stakeholders. This study
resulted in a selection of the predicates to be implemented in PQL, whose
decidability has also been formally proven. The computational performance of
the language has been extensively evaluated through a set of experiments
against two large process model collections.


---

