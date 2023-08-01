## Towards Developing a Multilingual and Code-Mixed Visual Question
  Answering System by Knowledge Distillation

**Published Date:** 2021-09-10T03:47:29Z

**Link:** http://arxiv.org/pdf/2109.04653v1

**Abstract:**

  Pre-trained language-vision models have shown remarkable performance on the
visual question answering (VQA) task. However, most pre-trained models are
trained by only considering monolingual learning, especially the resource-rich
language like English. Training such models for multilingual setups demand high
computing resources and multilingual language-vision dataset which hinders
their application in practice. To alleviate these challenges, we propose a
knowledge distillation approach to extend an English language-vision model
(teacher) into an equally effective multilingual and code-mixed model
(student). Unlike the existing knowledge distillation methods, which only use
the output from the last layer of the teacher network for distillation, our
student model learns and imitates the teacher from multiple intermediate layers
(language and vision encoders) with appropriately designed distillation
objectives for incremental knowledge extraction. We also create the large-scale
multilingual and code-mixed VQA dataset in eleven different language setups
considering the multiple Indian and European languages. Experimental results
and in-depth analysis show the effectiveness of the proposed VQA model over the
pre-trained language-vision models on eleven diverse language setups.


---

## TEASEL: A Transformer-Based Speech-Prefixed Language Model

**Published Date:** 2021-09-12T14:08:57Z

**Link:** http://arxiv.org/pdf/2109.05522v1

**Abstract:**

  Multimodal language analysis is a burgeoning field of NLP that aims to
simultaneously model a speaker's words, acoustical annotations, and facial
expressions. In this area, lexicon features usually outperform other modalities
because they are pre-trained on large corpora via Transformer-based models.
Despite their strong performance, training a new self-supervised learning (SSL)
Transformer on any modality is not usually attainable due to insufficient data,
which is the case in multimodal language learning. This work proposes a
Transformer-Based Speech-Prefixed Language Model called TEASEL to approach the
mentioned constraints without training a complete Transformer model. TEASEL
model includes speech modality as a dynamic prefix besides the textual modality
compared to a conventional language model. This method exploits a conventional
pre-trained language model as a cross-modal Transformer model. We evaluated
TEASEL for the multimodal sentiment analysis task defined by CMU-MOSI dataset.
Extensive experiments show that our model outperforms unimodal baseline
language models by 4% and outperforms the current multimodal state-of-the-art
(SoTA) model by 1% in F1-score. Additionally, our proposed method is 72%
smaller than the SoTA model.


---

## CTAL: Pre-training Cross-modal Transformer for Audio-and-Language
  Representations

**Published Date:** 2021-09-01T04:18:19Z

**Link:** http://arxiv.org/pdf/2109.00181v1

**Abstract:**

  Existing audio-language task-specific predictive approaches focus on building
complicated late-fusion mechanisms. However, these models are facing challenges
of overfitting with limited labels and low model generalization abilities. In
this paper, we present a Cross-modal Transformer for Audio-and-Language, i.e.,
CTAL, which aims to learn the intra-modality and inter-modality connections
between audio and language through two proxy tasks on a large amount of
audio-and-language pairs: masked language modeling and masked cross-modal
acoustic modeling. After fine-tuning our pre-trained model on multiple
downstream audio-and-language tasks, we observe significant improvements across
various tasks, such as, emotion classification, sentiment analysis, and speaker
verification. On this basis, we further propose a specially-designed fusion
mechanism that can be used in fine-tuning phase, which allows our pre-trained
model to achieve better performance. Lastly, we demonstrate detailed ablation
studies to prove that both our novel cross-modality fusion component and
audio-language pre-training methods significantly contribute to the promising
results.


---

## IndicBART: A Pre-trained Model for Indic Natural Language Generation

**Published Date:** 2021-09-07T07:08:33Z

**Link:** http://arxiv.org/pdf/2109.02903v2

**Abstract:**

  In this paper, we study pre-trained sequence-to-sequence models for a group
of related languages, with a focus on Indic languages. We present IndicBART, a
multilingual, sequence-to-sequence pre-trained model focusing on 11 Indic
languages and English. IndicBART utilizes the orthographic similarity between
Indic scripts to improve transfer learning between similar Indic languages. We
evaluate IndicBART on two NLG tasks: Neural Machine Translation (NMT) and
extreme summarization. Our experiments on NMT and extreme summarization show
that a model specific to related languages like IndicBART is competitive with
large pre-trained models like mBART50 despite being significantly smaller. It
also performs well on very low-resource translation scenarios where languages
are not included in pre-training or fine-tuning. Script sharing, multilingual
training, and better utilization of limited model capacity contribute to the
good performance of the compact IndicBART model.


---

## GPT-3 Models are Poor Few-Shot Learners in the Biomedical Domain

**Published Date:** 2021-09-06T15:50:37Z

**Link:** http://arxiv.org/pdf/2109.02555v2

**Abstract:**

  Deep neural language models have set new breakthroughs in many tasks of
Natural Language Processing (NLP). Recent work has shown that deep transformer
language models (pretrained on large amounts of texts) can achieve high levels
of task-specific few-shot performance comparable to state-of-the-art models.
However, the ability of these large language models in few-shot transfer
learning has not yet been explored in the biomedical domain. We investigated
the performance of two powerful transformer language models, i.e. GPT-3 and
BioBERT, in few-shot settings on various biomedical NLP tasks. The experimental
results showed that, to a great extent, both the models underperform a language
model fine-tuned on the full training data. Although GPT-3 had already achieved
near state-of-the-art results in few-shot knowledge transfer on open-domain NLP
tasks, it could not perform as effectively as BioBERT, which is orders of
magnitude smaller than GPT-3. Regarding that BioBERT was already pretrained on
large biomedical text corpora, our study suggests that language models may
largely benefit from in-domain pretraining in task-specific few-shot learning.
However, in-domain pretraining seems not to be sufficient; novel pretraining
and few-shot learning strategies are required in the biomedical NLP domain.


---

## DziriBERT: a Pre-trained Language Model for the Algerian Dialect

**Published Date:** 2021-09-25T11:51:35Z

**Link:** http://arxiv.org/pdf/2109.12346v3

**Abstract:**

  Pre-trained transformers are now the de facto models in Natural Language
Processing given their state-of-the-art results in many tasks and languages.
However, most of the current models have been trained on languages for which
large text resources are already available (such as English, French, Arabic,
etc.). Therefore, there are still a number of low-resource languages that need
more attention from the community. In this paper, we study the Algerian dialect
which has several specificities that make the use of Arabic or multilingual
models inappropriate. To address this issue, we collected more than one million
Algerian tweets, and pre-trained the first Algerian language model: DziriBERT.
When compared with existing models, DziriBERT achieves better results,
especially when dealing with the Roman script. The obtained results show that
pre-training a dedicated model on a small dataset (150 MB) can outperform
existing models that have been trained on much more data (hundreds of GB).
Finally, our model is publicly available to the community.


---

## IndicBART: A Pre-trained Model for Indic Natural Language Generation

**Published Date:** 2021-09-07T07:08:33Z

**Link:** http://arxiv.org/pdf/2109.02903v2

**Abstract:**

  In this paper, we study pre-trained sequence-to-sequence models for a group
of related languages, with a focus on Indic languages. We present IndicBART, a
multilingual, sequence-to-sequence pre-trained model focusing on 11 Indic
languages and English. IndicBART utilizes the orthographic similarity between
Indic scripts to improve transfer learning between similar Indic languages. We
evaluate IndicBART on two NLG tasks: Neural Machine Translation (NMT) and
extreme summarization. Our experiments on NMT and extreme summarization show
that a model specific to related languages like IndicBART is competitive with
large pre-trained models like mBART50 despite being significantly smaller. It
also performs well on very low-resource translation scenarios where languages
are not included in pre-training or fine-tuning. Script sharing, multilingual
training, and better utilization of limited model capacity contribute to the
good performance of the compact IndicBART model.


---

## Learning Natural Language Generation from Scratch

**Published Date:** 2021-09-20T08:46:51Z

**Link:** http://arxiv.org/pdf/2109.09371v1

**Abstract:**

  This paper introduces TRUncated ReinForcement Learning for Language (TrufLL),
an original ap-proach to train conditional language models from scratch by only
using reinforcement learning (RL). AsRL methods unsuccessfully scale to large
action spaces, we dynamically truncate the vocabulary spaceusing a generic
language model. TrufLL thus enables to train a language agent by solely
interacting withits environment without any task-specific prior knowledge; it
is only guided with a task-agnostic languagemodel. Interestingly, this approach
avoids the dependency to labelled datasets and inherently reduces pre-trained
policy flaws such as language or exposure biases. We evaluate TrufLL on two
visual questiongeneration tasks, for which we report positive results over
performance and language metrics, which wethen corroborate with a human
evaluation. To our knowledge, it is the first approach that successfullylearns
a language generation policy (almost) from scratch.


---

## Allocating Large Vocabulary Capacity for Cross-lingual Language Model
  Pre-training

**Published Date:** 2021-09-15T14:04:16Z

**Link:** http://arxiv.org/pdf/2109.07306v1

**Abstract:**

  Compared to monolingual models, cross-lingual models usually require a more
expressive vocabulary to represent all languages adequately. We find that many
languages are under-represented in recent cross-lingual language models due to
the limited vocabulary capacity. To this end, we propose an algorithm VoCap to
determine the desired vocabulary capacity of each language. However, increasing
the vocabulary size significantly slows down the pre-training speed. In order
to address the issues, we propose k-NN-based target sampling to accelerate the
expensive softmax. Our experiments show that the multilingual vocabulary
learned with VoCap benefits cross-lingual language model pre-training.
Moreover, k-NN-based target sampling mitigates the side-effects of increasing
the vocabulary size while achieving comparable performance and faster
pre-training speed. The code and the pretrained multilingual vocabularies are
available at https://github.com/bozheng-hit/VoCapXLM.


---

## Cross-Lingual Language Model Meta-Pretraining

**Published Date:** 2021-09-23T03:47:44Z

**Link:** http://arxiv.org/pdf/2109.11129v1

**Abstract:**

  The success of pretrained cross-lingual language models relies on two
essential abilities, i.e., generalization ability for learning downstream tasks
in a source language, and cross-lingual transferability for transferring the
task knowledge to other languages. However, current methods jointly learn the
two abilities in a single-phase cross-lingual pretraining process, resulting in
a trade-off between generalization and cross-lingual transfer. In this paper,
we propose cross-lingual language model meta-pretraining, which learns the two
abilities in different training phases. Our method introduces an additional
meta-pretraining phase before cross-lingual pretraining, where the model learns
generalization ability on a large-scale monolingual corpus. Then, the model
focuses on learning cross-lingual transfer on a multilingual corpus.
Experimental results show that our method improves both generalization and
cross-lingual transfer, and produces better-aligned representations across
different languages.


---

## Structural Persistence in Language Models: Priming as a Window into
  Abstract Language Representations

**Published Date:** 2021-09-30T10:38:38Z

**Link:** http://arxiv.org/pdf/2109.14989v2

**Abstract:**

  We investigate the extent to which modern, neural language models are
susceptible to structural priming, the phenomenon whereby the structure of a
sentence makes the same structure more probable in a follow-up sentence. We
explore how priming can be used to study the potential of these models to learn
abstract structural information, which is a prerequisite for good performance
on tasks that require natural language understanding skills. We introduce a
novel metric and release Prime-LM, a large corpus where we control for various
linguistic factors which interact with priming strength. We find that
Transformer models indeed show evidence of structural priming, but also that
the generalisations they learned are to some extent modulated by semantic
information. Our experiments also show that the representations acquired by the
models may not only encode abstract sequential structure but involve certain
level of hierarchical syntactic information. More generally, our study shows
that the priming paradigm is a useful, additional tool for gaining insights
into the capacities of language models and opens the door to future
priming-based investigations that probe the model's internal states.


---

## IndicBART: A Pre-trained Model for Indic Natural Language Generation

**first_author:** Raj Dabre et al.

**Published Date:** 2021-09-07T07:08:33Z

**Link:** http://arxiv.org/pdf/2109.02903v2

**Abstract:**

  In this paper, we study pre-trained sequence-to-sequence models for a group
of related languages, with a focus on Indic languages. We present IndicBART, a
multilingual, sequence-to-sequence pre-trained model focusing on 11 Indic
languages and English. IndicBART utilizes the orthographic similarity between
Indic scripts to improve transfer learning between similar Indic languages. We
evaluate IndicBART on two NLG tasks: Neural Machine Translation (NMT) and
extreme summarization. Our experiments on NMT and extreme summarization show
that a model specific to related languages like IndicBART is competitive with
large pre-trained models like mBART50 despite being significantly smaller. It
also performs well on very low-resource translation scenarios where languages
are not included in pre-training or fine-tuning. Script sharing, multilingual
training, and better utilization of limited model capacity contribute to the
good performance of the compact IndicBART model.


---

## Multimodal Conditionality for Natural Language Generation

**first_author:** Michael Sollami et al.

**Published Date:** 2021-09-02T22:06:07Z

**Link:** http://arxiv.org/pdf/2109.01229v1

**Abstract:**

  Large scale pretrained language models have demonstrated state-of-the-art
performance in language understanding tasks. Their application has recently
expanded into multimodality learning, leading to improved representations
combining vision and language. However, progress in adapting language models
towards conditional Natural Language Generation (NLG) has been limited to a
single modality, generally text. We propose MAnTiS, Multimodal Adaptation for
Text Synthesis, a general approach for multimodal conditionality in
transformer-based NLG models. In this method, we pass inputs from each modality
through modality-specific encoders, project to textual token space, and finally
join to form a conditionality prefix. We fine-tune the pretrained language
model and encoders with the conditionality prefix guiding the generation. We
apply MAnTiS to the task of product description generation, conditioning a
network on both product images and titles to generate descriptive text. We
demonstrate that MAnTiS outperforms strong baseline approaches on standard NLG
scoring metrics. Furthermore, qualitative assessments demonstrate that MAnTiS
can generate human quality descriptions consistent with given multimodal
inputs.


---

