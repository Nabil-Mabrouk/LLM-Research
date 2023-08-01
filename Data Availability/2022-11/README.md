## AfroLM: A Self-Active Learning-based Multilingual Pretrained Language
  Model for 23 African Languages

**Published Date:** 2022-11-07T02:15:25Z

**Link:** http://arxiv.org/pdf/2211.03263v2

**Abstract:**

  In recent years, multilingual pre-trained language models have gained
prominence due to their remarkable performance on numerous downstream Natural
Language Processing tasks (NLP). However, pre-training these large multilingual
language models requires a lot of training data, which is not available for
African Languages. Active learning is a semi-supervised learning algorithm, in
which a model consistently and dynamically learns to identify the most
beneficial samples to train itself on, in order to achieve better optimization
and performance on downstream tasks. Furthermore, active learning effectively
and practically addresses real-world data scarcity. Despite all its benefits,
active learning, in the context of NLP and especially multilingual language
models pretraining, has received little consideration. In this paper, we
present AfroLM, a multilingual language model pretrained from scratch on 23
African languages (the largest effort to date) using our novel self-active
learning framework. Pretrained on a dataset significantly (14x) smaller than
existing baselines, AfroLM outperforms many multilingual pretrained language
models (AfriBERTa, XLMR-base, mBERT) on various NLP downstream tasks (NER, text
classification, and sentiment analysis). Additional out-of-domain sentiment
analysis experiments show that \textbf{AfroLM} is able to generalize well
across various domains. We release the code source, and our datasets used in
our framework at https://github.com/bonaventuredossou/MLM_AL.


---

## GreenPLM: Cross-Lingual Transfer of Monolingual Pre-Trained Language
  Models at Almost No Cost

**Published Date:** 2022-11-13T18:59:15Z

**Link:** http://arxiv.org/pdf/2211.06993v3

**Abstract:**

  Large pre-trained models have revolutionized natural language processing
(NLP) research and applications, but high training costs and limited data
resources have prevented their benefits from being shared equally amongst
speakers of all the world's languages. To address issues of cross-linguistic
access to such models and reduce energy consumption for sustainability during
large-scale model training, this study proposes an effective and
energy-efficient framework called GreenPLM that uses bilingual lexicons to
directly "translate" pre-trained language models of one language into another
at almost no additional cost. We validate this approach in 18 languages' BERT
models and show that this framework is comparable to, if not better than, other
heuristics with high training costs. In addition, given lightweight continued
pre-training on limited data where available, this framework outperforms the
original monolingual language models in six out of seven tested languages with
up to 200x less pre-training efforts. Aiming at the Leave No One Behind
Principle (LNOB), our approach manages to reduce inequalities between languages
and energy consumption greatly. We make our codes and models publicly available
here: \url{https://github.com/qcznlp/GreenPLMs}


---

