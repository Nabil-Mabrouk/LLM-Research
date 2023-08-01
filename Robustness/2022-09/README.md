## Language-Family Adapters for Low-Resource Multilingual Neural Machine
  Translation

**Published Date:** 2022-09-30T05:02:42Z

**Link:** http://arxiv.org/pdf/2209.15236v3

**Abstract:**

  Large multilingual models trained with self-supervision achieve
state-of-the-art results in a wide range of natural language processing tasks.
Self-supervised pretrained models are often fine-tuned on parallel data from
one or multiple language pairs for machine translation. Multilingual
fine-tuning improves performance on low-resource languages but requires
modifying the entire model and can be prohibitively expensive. Training a new
adapter on each language pair or training a single adapter on all language
pairs without updating the pretrained model has been proposed as a
parameter-efficient alternative. However, the former does not permit any
sharing between languages, while the latter shares parameters for all languages
and is susceptible to negative interference. In this paper, we propose training
language-family adapters on top of mBART-50 to facilitate cross-lingual
transfer. Our approach outperforms related baselines, yielding higher
translation scores on average when translating from English to 17 different
low-resource languages. We also show that language-family adapters provide an
effective method to translate to languages unseen during pretraining.


---

## ASR2K: Speech Recognition for Around 2000 Languages without Audio

**Published Date:** 2022-09-06T22:48:29Z

**Link:** http://arxiv.org/pdf/2209.02842v1

**Abstract:**

  Most recent speech recognition models rely on large supervised datasets,
which are unavailable for many low-resource languages. In this work, we present
a speech recognition pipeline that does not require any audio for the target
language. The only assumption is that we have access to raw text datasets or a
set of n-gram statistics. Our speech pipeline consists of three components:
acoustic, pronunciation, and language models. Unlike the standard pipeline, our
acoustic and pronunciation models use multilingual models without any
supervision. The language model is built using n-gram statistics or the raw
text dataset. We build speech recognition for 1909 languages by combining it
with Crubadan: a large endangered languages n-gram database. Furthermore, we
test our approach on 129 languages across two datasets: Common Voice and CMU
Wilderness dataset. We achieve 50% CER and 74% WER on the Wilderness dataset
with Crubadan statistics only and improve them to 45% CER and 69% WER when
using 10000 raw text utterances.


---

## Bidirectional Language Models Are Also Few-shot Learners

**Published Date:** 2022-09-29T01:35:57Z

**Link:** http://arxiv.org/pdf/2209.14500v2

**Abstract:**

  Large language models such as GPT-3 (Brown et al., 2020) can perform
arbitrary tasks without undergoing fine-tuning after being prompted with only a
few labeled examples. An arbitrary task can be reformulated as a natural
language prompt, and a language model can be asked to generate the completion,
indirectly performing the task in a paradigm known as prompt-based learning. To
date, emergent prompt-based learning capabilities have mainly been demonstrated
for unidirectional language models. However, bidirectional language models
pre-trained on denoising objectives such as masked language modeling produce
stronger learned representations for transfer learning. This motivates the
possibility of prompting bidirectional models, but their pre-training
objectives have made them largely incompatible with the existing prompting
paradigm. We present SAP (Sequential Autoregressive Prompting), a technique
that enables the prompting of bidirectional models. Utilizing the machine
translation task as a case study, we prompt the bidirectional mT5 model (Xue et
al., 2021) with SAP and demonstrate its few-shot and zero-shot translations
outperform the few-shot translations of unidirectional models like GPT-3 and
XGLM (Lin et al., 2021), despite mT5's approximately 50% fewer parameters. We
further show SAP is effective on question answering and summarization. For the
first time, our results demonstrate prompt-based learning is an emergent
property of a broader class of language models, rather than only unidirectional
models.


---

## Bidirectional Language Models Are Also Few-shot Learners

**Published Date:** 2022-09-29T01:35:57Z

**Link:** http://arxiv.org/pdf/2209.14500v2

**Abstract:**

  Large language models such as GPT-3 (Brown et al., 2020) can perform
arbitrary tasks without undergoing fine-tuning after being prompted with only a
few labeled examples. An arbitrary task can be reformulated as a natural
language prompt, and a language model can be asked to generate the completion,
indirectly performing the task in a paradigm known as prompt-based learning. To
date, emergent prompt-based learning capabilities have mainly been demonstrated
for unidirectional language models. However, bidirectional language models
pre-trained on denoising objectives such as masked language modeling produce
stronger learned representations for transfer learning. This motivates the
possibility of prompting bidirectional models, but their pre-training
objectives have made them largely incompatible with the existing prompting
paradigm. We present SAP (Sequential Autoregressive Prompting), a technique
that enables the prompting of bidirectional models. Utilizing the machine
translation task as a case study, we prompt the bidirectional mT5 model (Xue et
al., 2021) with SAP and demonstrate its few-shot and zero-shot translations
outperform the few-shot translations of unidirectional models like GPT-3 and
XGLM (Lin et al., 2021), despite mT5's approximately 50% fewer parameters. We
further show SAP is effective on question answering and summarization. For the
first time, our results demonstrate prompt-based learning is an emergent
property of a broader class of language models, rather than only unidirectional
models.


---

## PaLI: A Jointly-Scaled Multilingual Language-Image Model

**Published Date:** 2022-09-14T17:24:07Z

**Link:** http://arxiv.org/pdf/2209.06794v4

**Abstract:**

  Effective scaling and a flexible task interface enable large language models
to excel at many tasks. We present PaLI (Pathways Language and Image model), a
model that extends this approach to the joint modeling of language and vision.
PaLI generates text based on visual and textual inputs, and with this interface
performs many vision, language, and multimodal tasks, in many languages. To
train PaLI, we make use of large pre-trained encoder-decoder language models
and Vision Transformers (ViTs). This allows us to capitalize on their existing
capabilities and leverage the substantial cost of training them. We find that
joint scaling of the vision and language components is important. Since
existing Transformers for language are much larger than their vision
counterparts, we train a large, 4-billion parameter ViT (ViT-e) to quantify the
benefits from even larger-capacity vision models. To train PaLI, we create a
large multilingual mix of pretraining tasks, based on a new image-text training
set containing 10B images and texts in over 100 languages. PaLI achieves
state-of-the-art in multiple vision and language tasks (such as captioning,
visual question-answering, scene-text understanding), while retaining a simple,
modular, and scalable design.


---

## LGDN: Language-Guided Denoising Network for Video-Language Modeling

**Published Date:** 2022-09-23T03:35:59Z

**Link:** http://arxiv.org/pdf/2209.11388v3

**Abstract:**

  Video-language modeling has attracted much attention with the rapid growth of
web videos. Most existing methods assume that the video frames and text
description are semantically correlated, and focus on video-language modeling
at video level. However, this hypothesis often fails for two reasons: (1) With
the rich semantics of video contents, it is difficult to cover all frames with
a single video-level description; (2) A raw video typically has
noisy/meaningless information (e.g., scenery shot, transition or teaser).
Although a number of recent works deploy attention mechanism to alleviate this
problem, the irrelevant/noisy information still makes it very difficult to
address. To overcome such challenge, we thus propose an efficient and effective
model, termed Language-Guided Denoising Network (LGDN), for video-language
modeling. Different from most existing methods that utilize all extracted video
frames, LGDN dynamically filters out the misaligned or redundant frames under
the language supervision and obtains only 2--4 salient frames per video for
cross-modal token-level alignment. Extensive experiments on five public
datasets show that our LGDN outperforms the state-of-the-arts by large margins.
We also provide detailed ablation study to reveal the critical importance of
solving the noise issue, in hope of inspiring future video-language work.


---

