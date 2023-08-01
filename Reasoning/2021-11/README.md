## Towards Building ASR Systems for the Next Billion Users

**Published Date:** 2021-11-06T19:34:33Z

**Link:** http://arxiv.org/pdf/2111.03945v3

**Abstract:**

  Recent methods in speech and language technology pretrain very LARGE models
which are fine-tuned for specific tasks. However, the benefits of such LARGE
models are often limited to a few resource rich languages of the world. In this
work, we make multiple contributions towards building ASR systems for low
resource languages from the Indian subcontinent. First, we curate 17,000 hours
of raw speech data for 40 Indian languages from a wide variety of domains
including education, news, technology, and finance. Second, using this raw
speech data we pretrain several variants of wav2vec style models for 40 Indian
languages. Third, we analyze the pretrained models to find key features:
codebook vectors of similar sounding phonemes are shared across languages,
representations across layers are discriminative of the language family, and
attention heads often pay attention within small local windows. Fourth, we
fine-tune this model for downstream ASR for 9 languages and obtain
state-of-the-art results on 3 public datasets, including on very low-resource
languages such as Sinhala and Nepali. Our work establishes that multilingual
pretraining is an effective strategy for building ASR systems for the
linguistically diverse speakers of the Indian subcontinent. Our code, data and
models are available publicly at https://indicnlp.ai4bharat.org/indicwav2vec/
and we hope they will help advance research in ASR for Indic languages.


---

## CL-NERIL: A Cross-Lingual Model for NER in Indian Languages

**Published Date:** 2021-11-23T12:09:15Z

**Link:** http://arxiv.org/pdf/2111.11815v1

**Abstract:**

  Developing Named Entity Recognition (NER) systems for Indian languages has
been a long-standing challenge, mainly owing to the requirement of a large
amount of annotated clean training instances. This paper proposes an end-to-end
framework for NER for Indian languages in a low-resource setting by exploiting
parallel corpora of English and Indian languages and an English NER dataset.
The proposed framework includes an annotation projection method that combines
word alignment score and NER tag prediction confidence score on source language
(English) data to generate weakly labeled data in a target Indian language. We
employ a variant of the Teacher-Student model and optimize it jointly on the
pseudo labels of the Teacher model and predictions on the generated weakly
labeled data. We also present manually annotated test sets for three Indian
languages: Hindi, Bengali, and Gujarati. We evaluate the performance of the
proposed framework on the test sets of the three Indian languages. Empirical
results show a minimum 10% performance improvement compared to the zero-shot
transfer learning model on all languages. This indicates that weakly labeled
data generated using the proposed annotation projection method in target Indian
languages can complement well-annotated source language data to enhance
performance. Our code is publicly available at
https://github.com/aksh555/CL-NERIL


---

## Towards Building ASR Systems for the Next Billion Users

**Published Date:** 2021-11-06T19:34:33Z

**Link:** http://arxiv.org/pdf/2111.03945v3

**Abstract:**

  Recent methods in speech and language technology pretrain very LARGE models
which are fine-tuned for specific tasks. However, the benefits of such LARGE
models are often limited to a few resource rich languages of the world. In this
work, we make multiple contributions towards building ASR systems for low
resource languages from the Indian subcontinent. First, we curate 17,000 hours
of raw speech data for 40 Indian languages from a wide variety of domains
including education, news, technology, and finance. Second, using this raw
speech data we pretrain several variants of wav2vec style models for 40 Indian
languages. Third, we analyze the pretrained models to find key features:
codebook vectors of similar sounding phonemes are shared across languages,
representations across layers are discriminative of the language family, and
attention heads often pay attention within small local windows. Fourth, we
fine-tune this model for downstream ASR for 9 languages and obtain
state-of-the-art results on 3 public datasets, including on very low-resource
languages such as Sinhala and Nepali. Our work establishes that multilingual
pretraining is an effective strategy for building ASR systems for the
linguistically diverse speakers of the Indian subcontinent. Our code, data and
models are available publicly at https://indicnlp.ai4bharat.org/indicwav2vec/
and we hope they will help advance research in ASR for Indic languages.


---

