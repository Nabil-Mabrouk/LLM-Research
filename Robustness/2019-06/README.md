## Cross-Lingual Training for Automatic Question Generation

**Published Date:** 2019-06-06T11:31:24Z

**Link:** http://arxiv.org/pdf/1906.02525v1

**Abstract:**

  Automatic question generation (QG) is a challenging problem in natural
language understanding. QG systems are typically built assuming access to a
large number of training instances where each instance is a question and its
corresponding answer. For a new language, such training instances are hard to
obtain making the QG problem even more challenging. Using this as our
motivation, we study the reuse of an available large QG dataset in a secondary
language (e.g. English) to learn a QG model for a primary language (e.g. Hindi)
of interest. For the primary language, we assume access to a large amount of
monolingual text but only a small QG dataset. We propose a cross-lingual QG
model which uses the following training regime: (i) Unsupervised pretraining of
language models in both primary and secondary languages and (ii) joint
supervised training for QG in both languages. We demonstrate the efficacy of
our proposed approach using two different primary languages, Hindi and Chinese.
We also create and release a new question answering dataset for Hindi
consisting of 6555 sentences.


---

## Barack's Wife Hillary: Using Knowledge-Graphs for Fact-Aware Language
  Modeling

**Published Date:** 2019-06-17T19:48:41Z

**Link:** http://arxiv.org/pdf/1906.07241v2

**Abstract:**

  Modeling human language requires the ability to not only generate fluent text
but also encode factual knowledge. However, traditional language models are
only capable of remembering facts seen at training time, and often have
difficulty recalling them. To address this, we introduce the knowledge graph
language model (KGLM), a neural language model with mechanisms for selecting
and copying facts from a knowledge graph that are relevant to the context.
These mechanisms enable the model to render information it has never seen
before, as well as generate out-of-vocabulary tokens. We also introduce the
Linked WikiText-2 dataset, a corpus of annotated text aligned to the Wikidata
knowledge graph whose contents (roughly) match the popular WikiText-2
benchmark. In experiments, we demonstrate that the KGLM achieves significantly
better performance than a strong baseline language model. We additionally
compare different language model's ability to complete sentences requiring
factual knowledge, showing that the KGLM outperforms even very large language
models in generating facts.


---

## How multilingual is Multilingual BERT?

**Published Date:** 2019-06-04T15:12:47Z

**Link:** http://arxiv.org/pdf/1906.01502v1

**Abstract:**

  In this paper, we show that Multilingual BERT (M-BERT), released by Devlin et
al. (2018) as a single language model pre-trained from monolingual corpora in
104 languages, is surprisingly good at zero-shot cross-lingual model transfer,
in which task-specific annotations in one language are used to fine-tune the
model for evaluation in another language. To understand why, we present a large
number of probing experiments, showing that transfer is possible even to
languages in different scripts, that transfer works best between typologically
similar languages, that monolingual corpora can train models for
code-switching, and that the model can find translation pairs. From these
results, we can conclude that M-BERT does create multilingual representations,
but that these representations exhibit systematic deficiencies affecting
certain language pairs.


---

## A Deep Generative Model for Code-Switched Text

**Published Date:** 2019-06-21T06:27:17Z

**Link:** http://arxiv.org/pdf/1906.08972v1

**Abstract:**

  Code-switching, the interleaving of two or more languages within a sentence
or discourse is pervasive in multilingual societies. Accurate language models
for code-switched text are critical for NLP tasks. State-of-the-art
data-intensive neural language models are difficult to train well from scarce
language-labeled code-switched text. A potential solution is to use deep
generative models to synthesize large volumes of realistic code-switched text.
Although generative adversarial networks and variational autoencoders can
synthesize plausible monolingual text from continuous latent space, they cannot
adequately address code-switched text, owing to their informal style and
complex interplay between the constituent languages. We introduce VACS, a novel
variational autoencoder architecture specifically tailored to code-switching
phenomena. VACS encodes to and decodes from a two-level hierarchical
representation, which models syntactic contextual signals in the lower level,
and language switching signals in the upper layer. Sampling representations
from the prior and decoding them produced well-formed, diverse code-switched
sentences. Extensive experiments show that using synthetic code-switched text
with natural monolingual data results in significant (33.06%) drop in
perplexity.


---

