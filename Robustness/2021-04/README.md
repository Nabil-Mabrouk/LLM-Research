## Should we Stop Training More Monolingual Models, and Simply Use Machine
  Translation Instead?

**Published Date:** 2021-04-21T10:21:24Z

**Link:** http://arxiv.org/pdf/2104.10441v1

**Abstract:**

  Most work in NLP makes the assumption that it is desirable to develop
solutions in the native language in question. There is consequently a strong
trend towards building native language models even for low-resource languages.
This paper questions this development, and explores the idea of simply
translating the data into English, thereby enabling the use of pretrained, and
large-scale, English language models. We demonstrate empirically that a large
English language model coupled with modern machine translation outperforms
native language models in most Scandinavian languages. The exception to this is
Finnish, which we assume is due to inferior translation quality. Our results
suggest that machine translation is a mature technology, which raises a serious
counter-argument for training native language models for low-resource
languages. This paper therefore strives to make a provocative but important
point. As English language models are improving at an unprecedented pace, which
in turn improves machine translation, it is from an empirical and environmental
stand-point more effective to translate data from low-resource languages into
English, than to build language models for such languages.


---

## Few-Shot Keyword Spotting in Any Language

**Published Date:** 2021-04-03T17:27:37Z

**Link:** http://arxiv.org/pdf/2104.01454v4

**Abstract:**

  We introduce a few-shot transfer learning method for keyword spotting in any
language. Leveraging open speech corpora in nine languages, we automate the
extraction of a large multilingual keyword bank and use it to train an
embedding model. With just five training examples, we fine-tune the embedding
model for keyword spotting and achieve an average F1 score of 0.75 on keyword
classification for 180 new keywords unseen by the embedding model in these nine
languages. This embedding model also generalizes to new languages. We achieve
an average F1 score of 0.65 on 5-shot models for 260 keywords sampled across 13
new languages unseen by the embedding model. We investigate streaming accuracy
for our 5-shot models in two contexts: keyword spotting and keyword search.
Across 440 keywords in 22 languages, we achieve an average streaming keyword
spotting accuracy of 87.4% with a false acceptance rate of 4.3%, and observe
promising initial results on keyword search.


---

## Improving Biomedical Pretrained Language Models with Knowledge

**Published Date:** 2021-04-21T03:57:26Z

**Link:** http://arxiv.org/pdf/2104.10344v1

**Abstract:**

  Pretrained language models have shown success in many natural language
processing tasks. Many works explore incorporating knowledge into language
models. In the biomedical domain, experts have taken decades of effort on
building large-scale knowledge bases. For example, the Unified Medical Language
System (UMLS) contains millions of entities with their synonyms and defines
hundreds of relations among entities. Leveraging this knowledge can benefit a
variety of downstream tasks such as named entity recognition and relation
extraction. To this end, we propose KeBioLM, a biomedical pretrained language
model that explicitly leverages knowledge from the UMLS knowledge bases.
Specifically, we extract entities from PubMed abstracts and link them to UMLS.
We then train a knowledge-aware language model that firstly applies a text-only
encoding layer to learn entity representation and applies a text-entity fusion
encoding to aggregate entity representation. Besides, we add two training
objectives as entity detection and entity linking. Experiments on the named
entity recognition and relation extraction from the BLURB benchmark demonstrate
the effectiveness of our approach. Further analysis on a collected probing
dataset shows that our model has better ability to model medical knowledge.


---

## GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation

**Published Date:** 2021-04-18T11:39:33Z

**Link:** http://arxiv.org/pdf/2104.08826v2

**Abstract:**

  Large-scale language models such as GPT-3 are excellent few-shot learners,
allowing them to be controlled via natural text prompts. Recent studies report
that prompt-based direct classification eliminates the need for fine-tuning but
lacks data and inference scalability. This paper proposes a novel data
augmentation technique that leverages large-scale language models to generate
realistic text samples from a mixture of real samples. We also propose
utilizing soft-labels predicted by the language models, effectively distilling
knowledge from the large-scale language models and creating textual
perturbations simultaneously. We perform data augmentation experiments on
diverse classification tasks and show that our method hugely outperforms
existing text augmentation methods. Ablation studies and a qualitative analysis
provide more insights into our approach.


---

## A Masked Segmental Language Model for Unsupervised Natural Language
  Segmentation

**Published Date:** 2021-04-16T00:00:05Z

**Link:** http://arxiv.org/pdf/2104.07829v2

**Abstract:**

  Segmentation remains an important preprocessing step both in languages where
"words" or other important syntactic/semantic units (like morphemes) are not
clearly delineated by white space, as well as when dealing with continuous
speech data, where there is often no meaningful pause between words.
Near-perfect supervised methods have been developed for use in resource-rich
languages such as Chinese, but many of the world's languages are both
morphologically complex, and have no large dataset of "gold" segmentations into
meaningful units. To solve this problem, we propose a new type of Segmental
Language Model (Sun and Deng, 2018; Kawakami et al., 2019; Wang et al., 2021)
for use in both unsupervised and lightly supervised segmentation tasks. We
introduce a Masked Segmental Language Model (MSLM) built on a span-masking
transformer architecture, harnessing the power of a bi-directional masked
modeling context and attention. In a series of experiments, our model
consistently outperforms Recurrent SLMs on Chinese (PKU Corpus) in segmentation
quality, and performs similarly to the Recurrent model on English (PTB). We
conclude by discussing the different challenges posed in segmenting
phonemic-type writing systems.


---

## Scaling End-to-End Models for Large-Scale Multilingual ASR

**Published Date:** 2021-04-30T08:24:11Z

**Link:** http://arxiv.org/pdf/2104.14830v2

**Abstract:**

  Building ASR models across many languages is a challenging multi-task
learning problem due to large variations and heavily unbalanced data. Existing
work has shown positive transfer from high resource to low resource languages.
However, degradations on high resource languages are commonly observed due to
interference from the heterogeneous multilingual data and reduction in
per-language capacity. We conduct a capacity study on a 15-language task, with
the amount of data per language varying from 7.6K to 53.5K hours. We adopt
GShard [1] to efficiently scale up to 10B parameters. Empirically, we find that
(1) scaling the number of model parameters is an effective way to solve the
capacity bottleneck - our 500M-param model already outperforms monolingual
baselines and scaling it to 1B and 10B brought further quality gains; (2)
larger models are not only more data efficient, but also more efficient in
terms of training cost as measured in TPU days - the 1B-param model reaches the
same accuracy at 34% of training time as the 500M-param model; (3) given a
fixed capacity budget, adding depth works better than width and large encoders
do better than large decoders; (4) with continuous training, they can be
adapted to new languages and domains.


---

## Should we Stop Training More Monolingual Models, and Simply Use Machine
  Translation Instead?

**Published Date:** 2021-04-21T10:21:24Z

**Link:** http://arxiv.org/pdf/2104.10441v1

**Abstract:**

  Most work in NLP makes the assumption that it is desirable to develop
solutions in the native language in question. There is consequently a strong
trend towards building native language models even for low-resource languages.
This paper questions this development, and explores the idea of simply
translating the data into English, thereby enabling the use of pretrained, and
large-scale, English language models. We demonstrate empirically that a large
English language model coupled with modern machine translation outperforms
native language models in most Scandinavian languages. The exception to this is
Finnish, which we assume is due to inferior translation quality. Our results
suggest that machine translation is a mature technology, which raises a serious
counter-argument for training native language models for low-resource
languages. This paper therefore strives to make a provocative but important
point. As English language models are improving at an unprecedented pace, which
in turn improves machine translation, it is from an empirical and environmental
stand-point more effective to translate data from low-resource languages into
English, than to build language models for such languages.


---

## A Masked Segmental Language Model for Unsupervised Natural Language
  Segmentation

**Published Date:** 2021-04-16T00:00:05Z

**Link:** http://arxiv.org/pdf/2104.07829v2

**Abstract:**

  Segmentation remains an important preprocessing step both in languages where
"words" or other important syntactic/semantic units (like morphemes) are not
clearly delineated by white space, as well as when dealing with continuous
speech data, where there is often no meaningful pause between words.
Near-perfect supervised methods have been developed for use in resource-rich
languages such as Chinese, but many of the world's languages are both
morphologically complex, and have no large dataset of "gold" segmentations into
meaningful units. To solve this problem, we propose a new type of Segmental
Language Model (Sun and Deng, 2018; Kawakami et al., 2019; Wang et al., 2021)
for use in both unsupervised and lightly supervised segmentation tasks. We
introduce a Masked Segmental Language Model (MSLM) built on a span-masking
transformer architecture, harnessing the power of a bi-directional masked
modeling context and attention. In a series of experiments, our model
consistently outperforms Recurrent SLMs on Chinese (PKU Corpus) in segmentation
quality, and performs similarly to the Recurrent model on English (PTB). We
conclude by discussing the different challenges posed in segmenting
phonemic-type writing systems.


---

## Understanding Chinese Video and Language via Contrastive Multimodal
  Pre-Training

**Published Date:** 2021-04-19T15:58:45Z

**Link:** http://arxiv.org/pdf/2104.09411v1

**Abstract:**

  The pre-trained neural models have recently achieved impressive performances
in understanding multimodal content. However, it is still very challenging to
pre-train neural models for video and language understanding, especially for
Chinese video-language data, due to the following reasons. Firstly, existing
video-language pre-training algorithms mainly focus on the co-occurrence of
words and video frames, but ignore other valuable semantic and structure
information of video-language content, e.g., sequential order and
spatiotemporal relationships. Secondly, there exist conflicts between video
sentence alignment and other proxy tasks. Thirdly, there is a lack of
large-scale and high-quality Chinese video-language datasets (e.g., including
10 million unique videos), which are the fundamental success conditions for
pre-training techniques.
  In this work, we propose a novel video-language understanding framework named
VICTOR, which stands for VIdeo-language understanding via Contrastive
mulTimOdal pRe-training. Besides general proxy tasks such as masked language
modeling, VICTOR constructs several novel proxy tasks under the contrastive
learning paradigm, making the model be more robust and able to capture more
complex multimodal semantic and structural relationships from different
perspectives. VICTOR is trained on a large-scale Chinese video-language
dataset, including over 10 million complete videos with corresponding
high-quality textual descriptions. We apply the pre-trained VICTOR model to a
series of downstream applications and demonstrate its superior performances,
comparing against the state-of-the-art pre-training methods such as VideoBERT
and UniVL. The codes and trained checkpoints will be publicly available to
nourish further developments of the research community.


---

## Multilingual Language Models Predict Human Reading Behavior

**Published Date:** 2021-04-12T13:03:49Z

**Link:** http://arxiv.org/pdf/2104.05433v1

**Abstract:**

  We analyze if large language models are able to predict patterns of human
reading behavior. We compare the performance of language-specific and
multilingual pretrained transformer models to predict reading time measures
reflecting natural human sentence processing on Dutch, English, German, and
Russian texts. This results in accurate models of human reading behavior, which
indicates that transformer models implicitly encode relative importance in
language in a way that is comparable to human processing mechanisms. We find
that BERT and XLM models successfully predict a range of eye tracking features.
In a series of experiments, we analyze the cross-domain and cross-language
abilities of these models and show how they reflect human sentence processing.


---

## LT-LM: a novel non-autoregressive language model for single-shot lattice
  rescoring

**Published Date:** 2021-04-06T14:06:07Z

**Link:** http://arxiv.org/pdf/2104.02526v1

**Abstract:**

  Neural network-based language models are commonly used in rescoring
approaches to improve the quality of modern automatic speech recognition (ASR)
systems. Most of the existing methods are computationally expensive since they
use autoregressive language models. We propose a novel rescoring approach,
which processes the entire lattice in a single call to the model. The key
feature of our rescoring policy is a novel non-autoregressive Lattice
Transformer Language Model (LT-LM). This model takes the whole lattice as an
input and predicts a new language score for each arc. Additionally, we propose
the artificial lattices generation approach to incorporate a large amount of
text data in the LT-LM training process. Our single-shot rescoring performs
orders of magnitude faster than other rescoring methods in our experiments. It
is more than 300 times faster than pruned RNNLM lattice rescoring and N-best
rescoring while slightly inferior in terms of WER.


---

## Should we Stop Training More Monolingual Models, and Simply Use Machine
  Translation Instead?

**first_author:** Tim Isbister et al.

**Published Date:** 2021-04-21T10:21:24Z

**Link:** http://arxiv.org/pdf/2104.10441v1

**Abstract:**

  Most work in NLP makes the assumption that it is desirable to develop
solutions in the native language in question. There is consequently a strong
trend towards building native language models even for low-resource languages.
This paper questions this development, and explores the idea of simply
translating the data into English, thereby enabling the use of pretrained, and
large-scale, English language models. We demonstrate empirically that a large
English language model coupled with modern machine translation outperforms
native language models in most Scandinavian languages. The exception to this is
Finnish, which we assume is due to inferior translation quality. Our results
suggest that machine translation is a mature technology, which raises a serious
counter-argument for training native language models for low-resource
languages. This paper therefore strives to make a provocative but important
point. As English language models are improving at an unprecedented pace, which
in turn improves machine translation, it is from an empirical and environmental
stand-point more effective to translate data from low-resource languages into
English, than to build language models for such languages.


---

## Few-Shot Keyword Spotting in Any Language

**first_author:** Mark Mazumder et al.

**Published Date:** 2021-04-03T17:27:37Z

**Link:** http://arxiv.org/pdf/2104.01454v4

**Abstract:**

  We introduce a few-shot transfer learning method for keyword spotting in any
language. Leveraging open speech corpora in nine languages, we automate the
extraction of a large multilingual keyword bank and use it to train an
embedding model. With just five training examples, we fine-tune the embedding
model for keyword spotting and achieve an average F1 score of 0.75 on keyword
classification for 180 new keywords unseen by the embedding model in these nine
languages. This embedding model also generalizes to new languages. We achieve
an average F1 score of 0.65 on 5-shot models for 260 keywords sampled across 13
new languages unseen by the embedding model. We investigate streaming accuracy
for our 5-shot models in two contexts: keyword spotting and keyword search.
Across 440 keywords in 22 languages, we achieve an average streaming keyword
spotting accuracy of 87.4% with a false acceptance rate of 4.3%, and observe
promising initial results on keyword search.


---

## Improving Biomedical Pretrained Language Models with Knowledge

**first_author:** Zheng Yuan et al.

**Published Date:** 2021-04-21T03:57:26Z

**Link:** http://arxiv.org/pdf/2104.10344v1

**Abstract:**

  Pretrained language models have shown success in many natural language
processing tasks. Many works explore incorporating knowledge into language
models. In the biomedical domain, experts have taken decades of effort on
building large-scale knowledge bases. For example, the Unified Medical Language
System (UMLS) contains millions of entities with their synonyms and defines
hundreds of relations among entities. Leveraging this knowledge can benefit a
variety of downstream tasks such as named entity recognition and relation
extraction. To this end, we propose KeBioLM, a biomedical pretrained language
model that explicitly leverages knowledge from the UMLS knowledge bases.
Specifically, we extract entities from PubMed abstracts and link them to UMLS.
We then train a knowledge-aware language model that firstly applies a text-only
encoding layer to learn entity representation and applies a text-entity fusion
encoding to aggregate entity representation. Besides, we add two training
objectives as entity detection and entity linking. Experiments on the named
entity recognition and relation extraction from the BLURB benchmark demonstrate
the effectiveness of our approach. Further analysis on a collected probing
dataset shows that our model has better ability to model medical knowledge.


---

## A Masked Segmental Language Model for Unsupervised Natural Language
  Segmentation

**first_author:** C. M. Downey et al.

**Published Date:** 2021-04-16T00:00:05Z

**Link:** http://arxiv.org/pdf/2104.07829v2

**Abstract:**

  Segmentation remains an important preprocessing step both in languages where
"words" or other important syntactic/semantic units (like morphemes) are not
clearly delineated by white space, as well as when dealing with continuous
speech data, where there is often no meaningful pause between words.
Near-perfect supervised methods have been developed for use in resource-rich
languages such as Chinese, but many of the world's languages are both
morphologically complex, and have no large dataset of "gold" segmentations into
meaningful units. To solve this problem, we propose a new type of Segmental
Language Model (Sun and Deng, 2018; Kawakami et al., 2019; Wang et al., 2021)
for use in both unsupervised and lightly supervised segmentation tasks. We
introduce a Masked Segmental Language Model (MSLM) built on a span-masking
transformer architecture, harnessing the power of a bi-directional masked
modeling context and attention. In a series of experiments, our model
consistently outperforms Recurrent SLMs on Chinese (PKU Corpus) in segmentation
quality, and performs similarly to the Recurrent model on English (PTB). We
conclude by discussing the different challenges posed in segmenting
phonemic-type writing systems.


---

## Understanding Chinese Video and Language via Contrastive Multimodal
  Pre-Training

**first_author:** Chenyi Lei et al.

**Published Date:** 2021-04-19T15:58:45Z

**Link:** http://arxiv.org/pdf/2104.09411v1

**Abstract:**

  The pre-trained neural models have recently achieved impressive performances
in understanding multimodal content. However, it is still very challenging to
pre-train neural models for video and language understanding, especially for
Chinese video-language data, due to the following reasons. Firstly, existing
video-language pre-training algorithms mainly focus on the co-occurrence of
words and video frames, but ignore other valuable semantic and structure
information of video-language content, e.g., sequential order and
spatiotemporal relationships. Secondly, there exist conflicts between video
sentence alignment and other proxy tasks. Thirdly, there is a lack of
large-scale and high-quality Chinese video-language datasets (e.g., including
10 million unique videos), which are the fundamental success conditions for
pre-training techniques.
  In this work, we propose a novel video-language understanding framework named
VICTOR, which stands for VIdeo-language understanding via Contrastive
mulTimOdal pRe-training. Besides general proxy tasks such as masked language
modeling, VICTOR constructs several novel proxy tasks under the contrastive
learning paradigm, making the model be more robust and able to capture more
complex multimodal semantic and structural relationships from different
perspectives. VICTOR is trained on a large-scale Chinese video-language
dataset, including over 10 million complete videos with corresponding
high-quality textual descriptions. We apply the pre-trained VICTOR model to a
series of downstream applications and demonstrate its superior performances,
comparing against the state-of-the-art pre-training methods such as VideoBERT
and UniVL. The codes and trained checkpoints will be publicly available to
nourish further developments of the research community.


---

