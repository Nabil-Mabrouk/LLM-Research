## Exploring Cross-lingual Textual Style Transfer with Large Multilingual
  Language Models

**Published Date:** 2022-06-05T20:02:30Z

**Link:** http://arxiv.org/pdf/2206.02252v1

**Abstract:**

  Detoxification is a task of generating text in polite style while preserving
meaning and fluency of the original toxic text. Existing detoxification methods
are designed to work in one exact language. This work investigates multilingual
and cross-lingual detoxification and the behavior of large multilingual models
like in this setting. Unlike previous works we aim to make large language
models able to perform detoxification without direct fine-tuning in given
language. Experiments show that multilingual models are capable of performing
multilingual style transfer. However, models are not able to perform
cross-lingual detoxification and direct fine-tuning on exact language is
inevitable.


---

## Distilling a Pretrained Language Model to a Multilingual ASR Model

**Published Date:** 2022-06-25T12:36:11Z

**Link:** http://arxiv.org/pdf/2206.12638v1

**Abstract:**

  Multilingual speech data often suffer from long-tailed language distribution,
resulting in performance degradation. However, multilingual text data is much
easier to obtain, yielding a more useful general language model. Hence, we are
motivated to distill the rich knowledge embedded inside a well-trained teacher
text model to the student speech model. We propose a novel method called the
Distilling a Language model to a Speech model (Distill-L2S), which aligns the
latent representations of two different modalities. The subtle differences are
handled by the shrinking mechanism, nearest-neighbor interpolation, and a
learnable linear projection layer. We demonstrate the effectiveness of our
distillation method by applying it to the multilingual automatic speech
recognition (ASR) task. We distill the transformer-based cross-lingual language
model (InfoXLM) while fine-tuning the large-scale multilingual ASR model
(XLSR-wav2vec 2.0) for each language. We show the superiority of our method on
20 low-resource languages of the CommonVoice dataset with less than 100 hours
of speech data.


---

## Ancestor-to-Creole Transfer is Not a Walk in the Park

**Published Date:** 2022-06-09T09:28:10Z

**Link:** http://arxiv.org/pdf/2206.04371v1

**Abstract:**

  We aim to learn language models for Creole languages for which large volumes
of data are not readily available, and therefore explore the potential transfer
from ancestor languages (the 'Ancestry Transfer Hypothesis'). We find that
standard transfer methods do not facilitate ancestry transfer. Surprisingly,
different from other non-Creole languages, a very distinct two-phase pattern
emerges for Creoles: As our training losses plateau, and language models begin
to overfit on their source languages, perplexity on the Creoles drop. We
explore if this compression phase can lead to practically useful language
models (the 'Ancestry Bottleneck Hypothesis'), but also falsify this. Moreover,
we show that Creoles even exhibit this two-phase pattern even when training on
random, unrelated languages. Thus Creoles seem to be typological outliers and
we speculate whether there is a link between the two observations.


---

## Offline RL for Natural Language Generation with Implicit Language Q
  Learning

**Published Date:** 2022-06-05T18:38:42Z

**Link:** http://arxiv.org/pdf/2206.11871v2

**Abstract:**

  Large language models distill broad knowledge from text corpora. However,
they can be inconsistent when it comes to completing user specified tasks. This
issue can be addressed by finetuning such models via supervised learning on
curated datasets, or via reinforcement learning. In this work, we propose a
novel offline RL method, implicit language Q-learning (ILQL), designed for use
on language models, that combines both the flexible utility maximization
framework of RL algorithms with the ability of supervised learning to leverage
previously collected data, as well as its simplicity and stability. Our method
employs a combination of value conservatism alongside an implicit dataset
support constraint in learning value functions, which are then used to guide
language model generations towards maximizing user-specified utility functions.
In addition to empirically validating ILQL, we present a detailed empirical
analysis of situations where offline RL can be useful in natural language
generation settings, demonstrating how it can be a more effective utility
optimizer than prior approaches for end-to-end dialogue, and how it can
effectively optimize high variance reward functions based on subjective
judgement, such as whether to label a comment as toxic or not.


---

## Offline RL for Natural Language Generation with Implicit Language Q
  Learning

**first_author:** Charlie Snell et al.

**Published Date:** 2022-06-05T18:38:42Z

**Link:** http://arxiv.org/pdf/2206.11871v2

**Abstract:**

  Large language models distill broad knowledge from text corpora. However,
they can be inconsistent when it comes to completing user specified tasks. This
issue can be addressed by finetuning such models via supervised learning on
curated datasets, or via reinforcement learning. In this work, we propose a
novel offline RL method, implicit language Q-learning (ILQL), designed for use
on language models, that combines both the flexible utility maximization
framework of RL algorithms with the ability of supervised learning to leverage
previously collected data, as well as its simplicity and stability. Our method
employs a combination of value conservatism alongside an implicit dataset
support constraint in learning value functions, which are then used to guide
language model generations towards maximizing user-specified utility functions.
In addition to empirically validating ILQL, we present a detailed empirical
analysis of situations where offline RL can be useful in natural language
generation settings, demonstrating how it can be a more effective utility
optimizer than prior approaches for end-to-end dialogue, and how it can
effectively optimize high variance reward functions based on subjective
judgement, such as whether to label a comment as toxic or not.


---

