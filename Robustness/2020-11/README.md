## Unsupervised Domain Adaptation of a Pretrained Cross-Lingual Language
  Model

**Published Date:** 2020-11-23T16:00:42Z

**Link:** http://arxiv.org/pdf/2011.11499v1

**Abstract:**

  Recent research indicates that pretraining cross-lingual language models on
large-scale unlabeled texts yields significant performance improvements over
various cross-lingual and low-resource tasks. Through training on one hundred
languages and terabytes of texts, cross-lingual language models have proven to
be effective in leveraging high-resource languages to enhance low-resource
language processing and outperform monolingual models. In this paper, we
further investigate the cross-lingual and cross-domain (CLCD) setting when a
pretrained cross-lingual language model needs to adapt to new domains.
Specifically, we propose a novel unsupervised feature decomposition method that
can automatically extract domain-specific features and domain-invariant
features from the entangled pretrained cross-lingual representations, given
unlabeled raw texts in the source language. Our proposed model leverages mutual
information estimation to decompose the representations computed by a
cross-lingual model into domain-invariant and domain-specific parts.
Experimental results show that our proposed method achieves significant
performance improvements over the state-of-the-art pretrained cross-lingual
language model in the CLCD setting. The source code of this paper is publicly
available at https://github.com/lijuntaopku/UFD.


---

## EstBERT: A Pretrained Language-Specific BERT for Estonian

**Published Date:** 2020-11-09T21:33:53Z

**Link:** http://arxiv.org/pdf/2011.04784v3

**Abstract:**

  This paper presents EstBERT, a large pretrained transformer-based
language-specific BERT model for Estonian. Recent work has evaluated
multilingual BERT models on Estonian tasks and found them to outperform the
baselines. Still, based on existing studies on other languages, a
language-specific BERT model is expected to improve over the multilingual ones.
We first describe the EstBERT pretraining process and then present the results
of the models based on finetuned EstBERT for multiple NLP tasks, including POS
and morphological tagging, named entity recognition and text classification.
The evaluation results show that the models based on EstBERT outperform
multilingual BERT models on five tasks out of six, providing further evidence
towards a view that training language-specific BERT models are still useful,
even when multilingual models are available.


---

## Entity Linking in 100 Languages

**Published Date:** 2020-11-05T07:28:35Z

**Link:** http://arxiv.org/pdf/2011.02690v1

**Abstract:**

  We propose a new formulation for multilingual entity linking, where
language-specific mentions resolve to a language-agnostic Knowledge Base. We
train a dual encoder in this new setting, building on prior work with improved
feature representation, negative mining, and an auxiliary entity-pairing task,
to obtain a single entity retrieval model that covers 100+ languages and 20
million entities. The model outperforms state-of-the-art results from a far
more limited cross-lingual linking task. Rare entities and low-resource
languages pose challenges at this large-scale, so we advocate for an increased
focus on zero- and few-shot evaluation. To this end, we provide Mewsli-9, a
large new multilingual dataset (http://goo.gle/mewsli-dataset) matched to our
setting, and show how frequency-based analysis provided key insights for our
model and training enhancements.


---

