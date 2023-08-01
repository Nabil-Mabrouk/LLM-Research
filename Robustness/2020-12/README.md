## Directed Beam Search: Plug-and-Play Lexically Constrained Language
  Generation

**Published Date:** 2020-12-31T03:05:44Z

**Link:** http://arxiv.org/pdf/2012.15416v1

**Abstract:**

  Large pre-trained language models are capable of generating realistic text.
However, controlling these models so that the generated text satisfies lexical
constraints, i.e., contains specific words, is a challenging problem. Given
that state-of-the-art language models are too large to be trained from scratch
in a manageable time, it is desirable to control these models without
re-training them. Methods capable of doing this are called plug-and-play.
Recent plug-and-play methods have been successful in constraining small
bidirectional language models as well as forward models in tasks with a
restricted search space, e.g., machine translation. However, controlling large
transformer-based models to meet lexical constraints without re-training them
remains a challenge. In this work, we propose Directed Beam Search (DBS), a
plug-and-play method for lexically constrained language generation. Our method
can be applied to any language model, is easy to implement and can be used for
general language generation. In our experiments we use DBS to control GPT-2. We
demonstrate its performance on keyword-to-phrase generation and we obtain
comparable results as a state-of-the-art non-plug-and-play model for lexically
constrained story generation.


---

## Universal Sentence Representation Learning with Conditional Masked
  Language Model

**Published Date:** 2020-12-28T18:06:37Z

**Link:** http://arxiv.org/pdf/2012.14388v3

**Abstract:**

  This paper presents a novel training method, Conditional Masked Language
Modeling (CMLM), to effectively learn sentence representations on large scale
unlabeled corpora. CMLM integrates sentence representation learning into MLM
training by conditioning on the encoded vectors of adjacent sentences. Our
English CMLM model achieves state-of-the-art performance on SentEval, even
outperforming models learned using supervised signals. As a fully unsupervised
learning method, CMLM can be conveniently extended to a broad range of
languages and domains. We find that a multilingual CMLM model co-trained with
bitext retrieval (BR) and natural language inference (NLI) tasks outperforms
the previous state-of-the-art multilingual models by a large margin, e.g. 10%
improvement upon baseline models on cross-lingual semantic search. We explore
the same language bias of the learned representations, and propose a simple,
post-training and model agnostic approach to remove the language identifying
information from the representation while still retaining sentence semantics.


---

## AraGPT2: Pre-Trained Transformer for Arabic Language Generation

**Published Date:** 2020-12-31T09:48:05Z

**Link:** http://arxiv.org/pdf/2012.15520v2

**Abstract:**

  Recently, pre-trained transformer-based architectures have proven to be very
efficient at language modeling and understanding, given that they are trained
on a large enough corpus. Applications in language generation for Arabic are
still lagging in comparison to other NLP advances primarily due to the lack of
advanced Arabic language generation models. In this paper, we develop the first
advanced Arabic language generation model, AraGPT2, trained from scratch on a
large Arabic corpus of internet text and news articles. Our largest model,
AraGPT2-mega, has 1.46 billion parameters, which makes it the largest Arabic
language model available. The Mega model was evaluated and showed success on
different tasks including synthetic news generation, and zero-shot question
answering. For text generation, our best model achieves a perplexity of 29.8 on
held-out Wikipedia articles. A study conducted with human evaluators showed the
significant success of AraGPT2-mega in generating news articles that are
difficult to distinguish from articles written by humans. We thus develop and
release an automatic discriminator model with a 98% percent accuracy in
detecting model-generated text. The models are also publicly available, hoping
to encourage new research directions and applications for Arabic NLP.


---

## Extracting Training Data from Large Language Models

**Published Date:** 2020-12-14T18:39:09Z

**Link:** http://arxiv.org/pdf/2012.07805v2

**Abstract:**

  It has become common to publish large (billion parameter) language models
that have been trained on private datasets. This paper demonstrates that in
such settings, an adversary can perform a training data extraction attack to
recover individual training examples by querying the language model.
  We demonstrate our attack on GPT-2, a language model trained on scrapes of
the public Internet, and are able to extract hundreds of verbatim text
sequences from the model's training data. These extracted examples include
(public) personally identifiable information (names, phone numbers, and email
addresses), IRC conversations, code, and 128-bit UUIDs. Our attack is possible
even though each of the above sequences are included in just one document in
the training data.
  We comprehensively evaluate our extraction attack to understand the factors
that contribute to its success. Worryingly, we find that larger models are more
vulnerable than smaller models. We conclude by drawing lessons and discussing
possible safeguards for training large language models.


---

## Audio Captioning using Pre-Trained Large-Scale Language Model Guided by
  Audio-based Similar Caption Retrieval

**Published Date:** 2020-12-14T08:27:36Z

**Link:** http://arxiv.org/pdf/2012.07331v1

**Abstract:**

  The goal of audio captioning is to translate input audio into its description
using natural language. One of the problems in audio captioning is the lack of
training data due to the difficulty in collecting audio-caption pairs by
crawling the web. In this study, to overcome this problem, we propose to use a
pre-trained large-scale language model. Since an audio input cannot be directly
inputted into such a language model, we utilize guidance captions retrieved
from a training dataset based on similarities that may exist in different
audio. Then, the caption of the audio input is generated by using a pre-trained
language model while referring to the guidance captions. Experimental results
show that (i) the proposed method has succeeded to use a pre-trained language
model for audio captioning, and (ii) the oracle performance of the pre-trained
model-based caption generator was clearly better than that of the conventional
method trained from scratch.


---

## Extracting Training Data from Large Language Models

**Published Date:** 2020-12-14T18:39:09Z

**Link:** http://arxiv.org/pdf/2012.07805v2

**Abstract:**

  It has become common to publish large (billion parameter) language models
that have been trained on private datasets. This paper demonstrates that in
such settings, an adversary can perform a training data extraction attack to
recover individual training examples by querying the language model.
  We demonstrate our attack on GPT-2, a language model trained on scrapes of
the public Internet, and are able to extract hundreds of verbatim text
sequences from the model's training data. These extracted examples include
(public) personally identifiable information (names, phone numbers, and email
addresses), IRC conversations, code, and 128-bit UUIDs. Our attack is possible
even though each of the above sequences are included in just one document in
the training data.
  We comprehensively evaluate our extraction attack to understand the factors
that contribute to its success. Worryingly, we find that larger models are more
vulnerable than smaller models. We conclude by drawing lessons and discussing
possible safeguards for training large language models.


---

## Cross-lingual Transfer of Abstractive Summarizer to Less-resource
  Language

**Published Date:** 2020-12-08T09:30:38Z

**Link:** http://arxiv.org/pdf/2012.04307v2

**Abstract:**

  Automatic text summarization extracts important information from texts and
presents the information in the form of a summary. Abstractive summarization
approaches progressed significantly by switching to deep neural networks, but
results are not yet satisfactory, especially for languages where large training
sets do not exist. In several natural language processing tasks, a
cross-lingual model transfer is successfully applied in less-resource
languages. For summarization, the cross-lingual model transfer was not
attempted due to a non-reusable decoder side of neural models that cannot
correct target language generation. In our work, we use a pre-trained English
summarization model based on deep neural networks and sequence-to-sequence
architecture to summarize Slovene news articles. We address the problem of
inadequate decoder by using an additional language model for the evaluation of
the generated text in target language. We test several cross-lingual
summarization models with different amounts of target data for fine-tuning. We
assess the models with automatic evaluation measures and conduct a small-scale
human evaluation. Automatic evaluation shows that the summaries of our best
cross-lingual model are useful and of quality similar to the model trained only
in the target language. Human evaluation shows that our best model generates
summaries with high accuracy and acceptable readability. However, similar to
other abstractive models, our models are not perfect and may occasionally
produce misleading or absurd content.


---

## CoCoLM: COmplex COmmonsense Enhanced Language Model with Discourse
  Relations

**Published Date:** 2020-12-31T15:05:36Z

**Link:** http://arxiv.org/pdf/2012.15643v2

**Abstract:**

  Large-scale pre-trained language models have demonstrated strong knowledge
representation ability. However, recent studies suggest that even though these
giant models contains rich simple commonsense knowledge (e.g., bird can fly and
fish can swim.), they often struggle with the complex commonsense knowledge
that involves multiple eventualities (verb-centric phrases, e.g., identifying
the relationship between ``Jim yells at Bob'' and ``Bob is upset'').To address
this problem, in this paper, we propose to help pre-trained language models
better incorporate complex commonsense knowledge. Different from existing
fine-tuning approaches, we do not focus on a specific task and propose a
general language model named CoCoLM. Through the careful training over a
large-scale eventuality knowledge graphs ASER, we successfully teach
pre-trained language models (i.e., BERT and RoBERTa) rich complex commonsense
knowledge among eventualities. Experiments on multiple downstream commonsense
tasks that requires the correct understanding of eventualities demonstrate the
effectiveness of CoCoLM.


---

## Extracting Training Data from Large Language Models

**first_author:** Nicholas Carlini et al.

**Published Date:** 2020-12-14T18:39:09Z

**Link:** http://arxiv.org/pdf/2012.07805v2

**Abstract:**

  It has become common to publish large (billion parameter) language models
that have been trained on private datasets. This paper demonstrates that in
such settings, an adversary can perform a training data extraction attack to
recover individual training examples by querying the language model.
  We demonstrate our attack on GPT-2, a language model trained on scrapes of
the public Internet, and are able to extract hundreds of verbatim text
sequences from the model's training data. These extracted examples include
(public) personally identifiable information (names, phone numbers, and email
addresses), IRC conversations, code, and 128-bit UUIDs. Our attack is possible
even though each of the above sequences are included in just one document in
the training data.
  We comprehensively evaluate our extraction attack to understand the factors
that contribute to its success. Worryingly, we find that larger models are more
vulnerable than smaller models. We conclude by drawing lessons and discussing
possible safeguards for training large language models.


---

