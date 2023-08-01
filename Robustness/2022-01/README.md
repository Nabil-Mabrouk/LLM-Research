## A Large and Diverse Arabic Corpus for Language Modeling

**Published Date:** 2022-01-23T11:17:53Z

**Link:** http://arxiv.org/pdf/2201.09227v3

**Abstract:**

  Language models (LMs) have introduced a major paradigm shift in Natural
Language Processing (NLP) modeling where large pre-trained LMs became integral
to most of the NLP tasks. The LMs are intelligent enough to find useful and
relevant representations of the language without any supervision. Perhaps,
these models are used to fine-tune typical NLP tasks with significantly high
accuracy as compared to the traditional approaches. Conversely, the training of
these models requires a massively large corpus that is a good representation of
the language. English LMs generally perform better than their other language
counterparts, due to the availability of massive English corpora. This work
elaborates on the design and development of a large Arabic corpus. It consists
of over 500 GB of Arabic cleaned text targeted at improving cross-domain
knowledge and downstream generalization capability of large-scale language
models. Moreover, the corpus is utilized in the training of a large Arabic LM.
In order to evaluate the effectiveness of the LM, a number of typical NLP tasks
are fine-tuned. The tasks demonstrate a significant boost from 4.5 to 8.5% when
compared to tasks fine-tuned on multi-lingual BERT (mBERT). To the best of my
knowledge, this is currently the largest clean and diverse Arabic corpus ever
collected.


---

## Transfer Learning Approaches for Building Cross-Language Dense Retrieval
  Models

**Published Date:** 2022-01-20T22:11:38Z

**Link:** http://arxiv.org/pdf/2201.08471v1

**Abstract:**

  The advent of transformer-based models such as BERT has led to the rise of
neural ranking models. These models have improved the effectiveness of
retrieval systems well beyond that of lexical term matching models such as
BM25. While monolingual retrieval tasks have benefited from large-scale
training collections such as MS MARCO and advances in neural architectures,
cross-language retrieval tasks have fallen behind these advancements. This
paper introduces ColBERT-X, a generalization of the ColBERT
multi-representation dense retrieval model that uses the XLM-RoBERTa (XLM-R)
encoder to support cross-language information retrieval (CLIR). ColBERT-X can
be trained in two ways. In zero-shot training, the system is trained on the
English MS MARCO collection, relying on the XLM-R encoder for cross-language
mappings. In translate-train, the system is trained on the MS MARCO English
queries coupled with machine translations of the associated MS MARCO passages.
Results on ad hoc document ranking tasks in several languages demonstrate
substantial and statistically significant improvements of these trained dense
retrieval models over traditional lexical CLIR baselines.


---

## ScaLA: Accelerating Adaptation of Pre-Trained Transformer-Based Language
  Models via Efficient Large-Batch Adversarial Noise

**Published Date:** 2022-01-29T01:47:01Z

**Link:** http://arxiv.org/pdf/2201.12469v1

**Abstract:**

  In recent years, large pre-trained Transformer-based language models have led
to dramatic improvements in many natural language understanding tasks. To train
these models with increasing sizes, many neural network practitioners attempt
to increase the batch sizes in order to leverage multiple GPUs to improve
training speed. However, increasing the batch size often makes the optimization
more difficult, leading to slow convergence or poor generalization that can
require orders of magnitude more training time to achieve the same model
quality. In this paper, we explore the steepness of the loss landscape of
large-batch optimization for adapting pre-trained Transformer-based language
models to domain-specific tasks and find that it tends to be highly complex and
irregular, posing challenges to generalization on downstream tasks.
  To tackle this challenge, we propose ScaLA, a novel and efficient method to
accelerate the adaptation speed of pre-trained transformer networks. Different
from prior methods, we take a sequential game-theoretic approach by adding
lightweight adversarial noise into large-batch optimization, which
significantly improves adaptation speed while preserving model generalization.
Experiment results show that ScaLA attains 2.7--9.8$\times$ adaptation speedups
over the baseline for GLUE on BERT-base and RoBERTa-large, while achieving
comparable and sometimes higher accuracy than the state-of-the-art large-batch
optimization methods. Finally, we also address the theoretical aspect of
large-batch optimization with adversarial noise and provide a theoretical
convergence rate analysis for ScaLA using techniques for analyzing non-convex
saddle-point problems.


---

## Submix: Practical Private Prediction for Large-Scale Language Models

**Published Date:** 2022-01-04T04:23:38Z

**Link:** http://arxiv.org/pdf/2201.00971v1

**Abstract:**

  Recent data-extraction attacks have exposed that language models can memorize
some training samples verbatim. This is a vulnerability that can compromise the
privacy of the model's training data. In this work, we introduce SubMix: a
practical protocol for private next-token prediction designed to prevent
privacy violations by language models that were fine-tuned on a private corpus
after pre-training on a public corpus. We show that SubMix limits the leakage
of information that is unique to any individual user in the private corpus via
a relaxation of group differentially private prediction. Importantly, SubMix
admits a tight, data-dependent privacy accounting mechanism, which allows it to
thwart existing data-extraction attacks while maintaining the utility of the
language model. SubMix is the first protocol that maintains privacy even when
publicly releasing tens of thousands of next-token predictions made by large
transformer-based models such as GPT-2.


---

