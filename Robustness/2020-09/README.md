## On the Ability and Limitations of Transformers to Recognize Formal
  Languages

**Published Date:** 2020-09-23T17:21:33Z

**Link:** http://arxiv.org/pdf/2009.11264v2

**Abstract:**

  Transformers have supplanted recurrent models in a large number of NLP tasks.
However, the differences in their abilities to model different syntactic
properties remain largely unknown. Past works suggest that LSTMs generalize
very well on regular languages and have close connections with counter
languages. In this work, we systematically study the ability of Transformers to
model such languages as well as the role of its individual components in doing
so. We first provide a construction of Transformers for a subclass of counter
languages, including well-studied languages such as n-ary Boolean Expressions,
Dyck-1, and its generalizations. In experiments, we find that Transformers do
well on this subclass, and their learned mechanism strongly correlates with our
construction. Perhaps surprisingly, in contrast to LSTMs, Transformers do well
only on a subset of regular languages with degrading performance as we make
languages more complex according to a well-known measure of complexity. Our
analysis also provides insights on the role of self-attention mechanism in
modeling certain behaviors and the influence of positional encoding schemes on
the learning and generalization abilities of the model.


---

## Reusing a Pretrained Language Model on Languages with Limited Corpora
  for Unsupervised NMT

**Published Date:** 2020-09-16T11:37:10Z

**Link:** http://arxiv.org/pdf/2009.07610v3

**Abstract:**

  Using a language model (LM) pretrained on two languages with large
monolingual data in order to initialize an unsupervised neural machine
translation (UNMT) system yields state-of-the-art results. When limited data is
available for one language, however, this method leads to poor translations. We
present an effective approach that reuses an LM that is pretrained only on the
high-resource language. The monolingual LM is fine-tuned on both languages and
is then used to initialize a UNMT model. To reuse the pretrained LM, we have to
modify its predefined vocabulary, to account for the new language. We therefore
propose a novel vocabulary extension method. Our approach, RE-LM, outperforms a
competitive cross-lingual pretraining model (XLM) in English-Macedonian (En-Mk)
and English-Albanian (En-Sq), yielding more than +8.3 BLEU points for all four
translation directions.


---

## Real-Time Execution of Large-scale Language Models on Mobile

**Published Date:** 2020-09-15T01:59:17Z

**Link:** http://arxiv.org/pdf/2009.06823v2

**Abstract:**

  Pre-trained large-scale language models have increasingly demonstrated high
accuracy on many natural language processing (NLP) tasks. However, the limited
weight storage and computational speed on hardware platforms have impeded the
popularity of pre-trained models, especially in the era of edge computing. In
this paper, we seek to find the best model structure of BERT for a given
computation size to match specific devices. We propose the first compiler-aware
neural architecture optimization framework. Our framework can guarantee the
identified model to meet both resource and real-time specifications of mobile
devices, thus achieving real-time execution of large transformer-based models
like BERT variants. We evaluate our model on several NLP tasks, achieving
competitive results on well-known benchmarks with lower latency on mobile
devices. Specifically, our model is 5.2x faster on CPU and 4.1x faster on GPU
with 0.5-2% accuracy loss compared with BERT-base. Our overall framework
achieves up to 7.8x speedup compared with TensorFlow-Lite with only minor
accuracy loss.


---

## Dialogue-adaptive Language Model Pre-training From Quality Estimation

**Published Date:** 2020-09-10T16:46:46Z

**Link:** http://arxiv.org/pdf/2009.04984v2

**Abstract:**

  Pre-trained language models (PrLMs) have achieved great success on a wide
range of natural language processing tasks by virtue of the universal language
representation ability obtained by self-supervised learning on a large corpus.
These models are pre-trained on standard plain texts with general language
model (LM) training objectives, which would be insufficient to model
dialogue-exclusive attributes like specificity and informativeness reflected in
these tasks that are not explicitly captured by the pre-trained universal
language representations. In this work, we propose dialogue-adaptive
pre-training objectives (DAPO) derived from quality estimation to simulate
dialogue-specific features, namely coherence, specificity, and informativeness.
As the foundation for model pre-training, we synthesize a new dialogue corpus
and build our training set with two unsupervised methods: 1) coherence-oriented
context corruption, including utterance ordering, insertion, and replacement,
to help the model capture the coherence inside the dialogue contexts; and 2)
specificity-oriented automatic rescoring, which encourages the model to measure
the quality of the synthesized data for dialogue-adaptive pre-training by
considering specificity and informativeness. Experimental results on widely
used open-domain response selection and quality estimation benchmarks show that
DAPO significantly improves the baseline models and achieves state-of-the-art
performance on the MuTual leaderboard, verifying the effectiveness of
estimating quality evaluation factors into pre-training.


---

