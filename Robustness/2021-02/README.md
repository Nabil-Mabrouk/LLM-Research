## Prompt Programming for Large Language Models: Beyond the Few-Shot
  Paradigm

**Published Date:** 2021-02-15T05:27:55Z

**Link:** http://arxiv.org/pdf/2102.07350v1

**Abstract:**

  Prevailing methods for mapping large generative language models to supervised
tasks may fail to sufficiently probe models' novel capabilities. Using GPT-3 as
a case study, we show that 0-shot prompts can significantly outperform few-shot
prompts. We suggest that the function of few-shot examples in these cases is
better described as locating an already learned task rather than meta-learning.
This analysis motivates rethinking the role of prompts in controlling and
evaluating powerful language models. In this work, we discuss methods of prompt
programming, emphasizing the usefulness of considering prompts through the lens
of natural language. We explore techniques for exploiting the capacity of
narratives and cultural anchors to encode nuanced intentions and techniques for
encouraging deconstruction of a problem into components before producing a
verdict. Informed by this more encompassing theory of prompt programming, we
also introduce the idea of a metaprompt that seeds the model to generate its
own natural language prompts for a range of tasks. Finally, we discuss how
these more general methods of interacting with language models can be
incorporated into existing and future benchmarks and practical applications.


---

## Does He Wink or Does He Nod? A Challenging Benchmark for Evaluating Word
  Understanding of Language Models

**Published Date:** 2021-02-06T15:15:57Z

**Link:** http://arxiv.org/pdf/2102.03596v1

**Abstract:**

  Recent progress in pretraining language models on large corpora has resulted
in large performance gains on many NLP tasks. These large models acquire
linguistic knowledge during pretraining, which helps to improve performance on
downstream tasks via fine-tuning. To assess what kind of knowledge is acquired,
language models are commonly probed by querying them with `fill in the blank'
style cloze questions. Existing probing datasets mainly focus on knowledge
about relations between words and entities. We introduce WDLMPro (Word
Definition Language Model Probing) to evaluate word understanding directly
using dictionary definitions of words. In our experiments, three popular
pretrained language models struggle to match words and their definitions. This
indicates that they understand many words poorly and that our new probing task
is a difficult challenge that could help guide research on LMs in the future.


---

## Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based
  Bias in NLP

**Published Date:** 2021-02-28T11:07:37Z

**Link:** http://arxiv.org/pdf/2103.00453v2

**Abstract:**

  When trained on large, unfiltered crawls from the internet, language models
pick up and reproduce all kinds of undesirable biases that can be found in the
data: they often generate racist, sexist, violent or otherwise toxic language.
As large models require millions of training examples to achieve good
performance, it is difficult to completely prevent them from being exposed to
such content. In this paper, we first demonstrate a surprising finding:
pretrained language models recognize, to a considerable degree, their
undesirable biases and the toxicity of the content they produce. We refer to
this capability as self-diagnosis. Based on this finding, we then propose a
decoding algorithm that, given only a textual description of the undesired
behavior, reduces the probability of a language model producing problematic
text. We refer to this approach as self-debiasing. Self-debiasing does not rely
on manually curated word lists, nor does it require any training data or
changes to the model's parameters. While we by no means eliminate the issue of
language models generating biased text, we believe our approach to be an
important step in this direction.


---

## Customizing Contextualized Language Models forLegal Document Reviews

**Published Date:** 2021-02-10T22:14:15Z

**Link:** http://arxiv.org/pdf/2102.05757v1

**Abstract:**

  Inspired by the inductive transfer learning on computer vision, many efforts
have been made to train contextualized language models that boost the
performance of natural language processing tasks. These models are mostly
trained on large general-domain corpora such as news, books, or
Wikipedia.Although these pre-trained generic language models well perceive the
semantic and syntactic essence of a language structure, exploiting them in a
real-world domain-specific scenario still needs some practical considerations
to be taken into account such as token distribution shifts, inference time,
memory, and their simultaneous proficiency in multiple tasks. In this paper, we
focus on the legal domain and present how different language model strained on
general-domain corpora can be best customized for multiple legal document
reviewing tasks. We compare their efficiencies with respect to task
performances and present practical considerations.


---

## Evaluating Contextualized Language Models for Hungarian

**Published Date:** 2021-02-22T09:29:01Z

**Link:** http://arxiv.org/pdf/2102.10848v1

**Abstract:**

  We present an extended comparison of contextualized language models for
Hungarian. We compare huBERT, a Hungarian model against 4 multilingual models
including the multilingual BERT model. We evaluate these models through three
tasks, morphological probing, POS tagging and NER. We find that huBERT works
better than the other models, often by a large margin, particularly near the
global optimum (typically at the middle layers). We also find that huBERT tends
to generate fewer subwords for one word and that using the last subword for
token-level tasks is generally a better choice than using the first one.


---

