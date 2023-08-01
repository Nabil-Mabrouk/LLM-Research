## HinFlair: pre-trained contextual string embeddings for pos tagging and
  text classification in the Hindi language

**Published Date:** 2021-01-18T09:23:35Z

**Link:** http://arxiv.org/pdf/2101.06949v1

**Abstract:**

  Recent advancements in language models based on recurrent neural networks and
transformers architecture have achieved state-of-the-art results on a wide
range of natural language processing tasks such as pos tagging, named entity
recognition, and text classification. However, most of these language models
are pre-trained in high resource languages like English, German, Spanish.
Multi-lingual language models include Indian languages like Hindi, Telugu,
Bengali in their training corpus, but they often fail to represent the
linguistic features of these languages as they are not the primary language of
the study. We introduce HinFlair, which is a language representation model
(contextual string embeddings) pre-trained on a large monolingual Hindi corpus.
Experiments were conducted on 6 text classification datasets and a Hindi
dependency treebank to analyze the performance of these contextualized string
embeddings for the Hindi language. Results show that HinFlair outperforms
previous state-of-the-art publicly available pre-trained embeddings for
downstream tasks like text classification and pos tagging. Also, HinFlair when
combined with FastText embeddings outperforms many transformers-based language
models trained particularly for the Hindi language.


---

## Training Multilingual Pre-trained Language Model with Byte-level
  Subwords

**Published Date:** 2021-01-23T10:01:28Z

**Link:** http://arxiv.org/pdf/2101.09469v2

**Abstract:**

  The pre-trained language models have achieved great successes in various
natural language understanding (NLU) tasks due to its capacity to capture the
deep contextualized information in text by pre-training on large-scale corpora.
One of the fundamental components in pre-trained language models is the
vocabulary, especially for training multilingual models on many different
languages. In the technical report, we present our practices on training
multilingual pre-trained language models with BBPE: Byte-Level BPE (i.e., Byte
Pair Encoding). In the experiment, we adopted the architecture of NEZHA as the
underlying pre-trained language model and the results show that NEZHA trained
with byte-level subwords consistently outperforms Google multilingual BERT and
vanilla NEZHA by a notable margin in several multilingual NLU tasks. We release
the source code of our byte-level vocabulary building tools and the
multilingual pre-trained language models.


---

