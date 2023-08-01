## Larger-Scale Transformers for Multilingual Masked Language Modeling

**Published Date:** 2021-05-02T23:15:02Z

**Link:** http://arxiv.org/pdf/2105.00572v1

**Abstract:**

  Recent work has demonstrated the effectiveness of cross-lingual language
model pretraining for cross-lingual understanding. In this study, we present
the results of two larger multilingual masked language models, with 3.5B and
10.7B parameters. Our two new models dubbed XLM-R XL and XLM-R XXL outperform
XLM-R by 1.8% and 2.4% average accuracy on XNLI. Our model also outperforms the
RoBERTa-Large model on several English tasks of the GLUE benchmark by 0.3% on
average while handling 99 more languages. This suggests pretrained models with
larger capacity may obtain both strong performance on high-resource languages
while greatly improving low-resource languages. We make our code and models
publicly available.


---

## Adapting Monolingual Models: Data can be Scarce when Language Similarity
  is High

**Published Date:** 2021-05-06T17:43:40Z

**Link:** http://arxiv.org/pdf/2105.02855v2

**Abstract:**

  For many (minority) languages, the resources needed to train large models are
not available. We investigate the performance of zero-shot transfer learning
with as little data as possible, and the influence of language similarity in
this process. We retrain the lexical layers of four BERT-based models using
data from two low-resource target language varieties, while the Transformer
layers are independently fine-tuned on a POS-tagging task in the model's source
language. By combining the new lexical layers and fine-tuned Transformer
layers, we achieve high task performance for both target languages. With high
language similarity, 10MB of data appears sufficient to achieve substantial
monolingual transfer performance. Monolingual BERT-based models generally
achieve higher downstream task performance after retraining the lexical layer
than multilingual BERT, even when the target language is included in the
multilingual model.


---

## Neural Morphology Dataset and Models for Multiple Languages, from the
  Large to the Endangered

**Published Date:** 2021-05-26T09:35:38Z

**Link:** http://arxiv.org/pdf/2105.12428v1

**Abstract:**

  We train neural models for morphological analysis, generation and
lemmatization for morphologically rich languages. We present a method for
automatically extracting substantially large amount of training data from FSTs
for 22 languages, out of which 17 are endangered. The neural models follow the
same tagset as the FSTs in order to make it possible to use them as fallback
systems together with the FSTs. The source code, models and datasets have been
released on Zenodo.


---

## A Multilingual Modeling Method for Span-Extraction Reading Comprehension

**Published Date:** 2021-05-31T11:05:30Z

**Link:** http://arxiv.org/pdf/2105.14880v1

**Abstract:**

  Span-extraction reading comprehension models have made tremendous advances
enabled by the availability of large-scale, high-quality training datasets.
Despite such rapid progress and widespread application, extractive reading
comprehension datasets in languages other than English remain scarce, and
creating such a sufficient amount of training data for each language is costly
and even impossible. An alternative to creating large-scale high-quality
monolingual span-extraction training datasets is to develop multilingual
modeling approaches and systems which can transfer to the target language
without requiring training data in that language. In this paper, in order to
solve the scarce availability of extractive reading comprehension training data
in the target language, we propose a multilingual extractive reading
comprehension approach called XLRC by simultaneously modeling the existing
extractive reading comprehension training data in a multilingual environment
using self-adaptive attention and multilingual attention. Specifically, we
firstly construct multilingual parallel corpora by translating the existing
extractive reading comprehension datasets (i.e., CMRC 2018) from the target
language (i.e., Chinese) into different language families (i.e., English).
Secondly, to enhance the final target representation, we adopt self-adaptive
attention (SAA) to combine self-attention and inter-attention to extract the
semantic relations from each pair of the target and source languages.
Furthermore, we propose multilingual attention (MLA) to learn the rich
knowledge from various language families. Experimental results show that our
model outperforms the state-of-the-art baseline (i.e., RoBERTa_Large) on the
CMRC 2018 task, which demonstrate the effectiveness of our proposed
multi-lingual modeling approach and show the potentials in multilingual NLP
tasks.


---

