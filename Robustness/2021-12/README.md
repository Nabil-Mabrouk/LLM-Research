## Large Language Models are not Models of Natural Language: they are
  Corpus Models

**Published Date:** 2021-12-13T22:39:46Z

**Link:** http://arxiv.org/pdf/2112.07055v2

**Abstract:**

  Natural Language Processing (NLP) has become one of the leading application
areas in the current Artificial Intelligence boom. Transfer learning has
enabled large deep learning neural networks trained on the language modeling
task to vastly improve performance in almost all downstream language tasks.
Interestingly, when the language models are trained with data that includes
software code, they demonstrate remarkable abilities in generating functioning
computer code from natural language specifications. We argue that this creates
a conundrum for the claim that eliminative neural models are a radical
restructuring in our understanding of cognition in that they eliminate the need
for symbolic abstractions like generative phrase structure grammars. Because
the syntax of programming languages is by design determined by phrase structure
grammars, neural models that produce syntactic code are apparently
uninformative about the theoretical foundations of programming languages. The
demonstration that neural models perform well on tasks that involve clearly
symbolic systems, proves that they cannot be used as an argument that language
and other cognitive systems are not symbolic. Finally, we argue as a corollary
that the term language model is misleading and propose the adoption of the
working term corpus model instead, which better reflects the genesis and
contents of the model.


---

## Multilingual Text Classification for Dravidian Languages

**Published Date:** 2021-12-03T04:26:49Z

**Link:** http://arxiv.org/pdf/2112.01705v1

**Abstract:**

  As the fourth largest language family in the world, the Dravidian languages
have become a research hotspot in natural language processing (NLP). Although
the Dravidian languages contain a large number of languages, there are
relatively few public available resources. Besides, text classification task,
as a basic task of natural language processing, how to combine it to multiple
languages in the Dravidian languages, is still a major difficulty in Dravidian
Natural Language Processing. Hence, to address these problems, we proposed a
multilingual text classification framework for the Dravidian languages. On the
one hand, the framework used the LaBSE pre-trained model as the base model.
Aiming at the problem of text information bias in multi-task learning, we
propose to use the MLM strategy to select language-specific words, and used
adversarial training to perturb them. On the other hand, in view of the problem
that the model cannot well recognize and utilize the correlation among
languages, we further proposed a language-specific representation module to
enrich semantic information for the model. The experimental results
demonstrated that the framework we proposed has a significant performance in
multilingual text classification tasks with each strategy achieving certain
improvements.


---

## Jigsaw: Large Language Models meet Program Synthesis

**Published Date:** 2021-12-06T12:30:22Z

**Link:** http://arxiv.org/pdf/2112.02969v1

**Abstract:**

  Large pre-trained language models such as GPT-3, Codex, and Google's language
model are now capable of generating code from natural language specifications
of programmer intent. We view these developments with a mixture of optimism and
caution. On the optimistic side, such large language models have the potential
to improve productivity by providing an automated AI pair programmer for every
programmer in the world. On the cautionary side, since these large language
models do not understand program semantics, they offer no guarantees about
quality of the suggested code. In this paper, we present an approach to augment
these large language models with post-processing steps based on program
analysis and synthesis techniques, that understand the syntax and semantics of
programs. Further, we show that such techniques can make use of user feedback
and improve with usage. We present our experiences from building and evaluating
such a tool jigsaw, targeted at synthesizing code for using Python Pandas API
using multi-modal inputs. Our experience suggests that as these large language
models evolve for synthesizing code from intent, jigsaw has an important role
to play in improving the accuracy of the systems.


---

## DPRK-BERT: The Supreme Language Model

**Published Date:** 2021-12-01T15:36:13Z

**Link:** http://arxiv.org/pdf/2112.00567v1

**Abstract:**

  Deep language models have achieved remarkable success in the NLP domain. The
standard way to train a deep language model is to employ unsupervised learning
from scratch on a large unlabeled corpus. However, such large corpora are only
available for widely-adopted and high-resource languages and domains. This
study presents the first deep language model, DPRK-BERT, for the DPRK language.
We achieve this by compiling the first unlabeled corpus for the DPRK language
and fine-tuning a preexisting the ROK language model. We compare the proposed
model with existing approaches and show significant improvements on two DPRK
datasets. We also present a cross-lingual version of this model which yields
better generalization across the two Korean languages. Finally, we provide
various NLP tools related to the DPRK language that would foster future
research.


---

## WECHSEL: Effective initialization of subword embeddings for
  cross-lingual transfer of monolingual language models

**Published Date:** 2021-12-13T12:26:02Z

**Link:** http://arxiv.org/pdf/2112.06598v2

**Abstract:**

  Large pretrained language models (LMs) have become the central building block
of many NLP applications. Training these models requires ever more
computational resources and most of the existing models are trained on English
text only. It is exceedingly expensive to train these models in other
languages. To alleviate this problem, we introduce a novel method -- called
WECHSEL -- to efficiently and effectively transfer pretrained LMs to new
languages. WECHSEL can be applied to any model which uses subword-based
tokenization and learns an embedding for each subword. The tokenizer of the
source model (in English) is replaced with a tokenizer in the target language
and token embeddings are initialized such that they are semantically similar to
the English tokens by utilizing multilingual static word embeddings covering
English and the target language. We use WECHSEL to transfer the English RoBERTa
and GPT-2 models to four languages (French, German, Chinese and Swahili). We
also study the benefits of our method on very low-resource languages. WECHSEL
improves over proposed methods for cross-lingual parameter transfer and
outperforms models of comparable size trained from scratch with up to 64x less
training effort. Our method makes training large language models for new
languages more accessible and less damaging to the environment. We make our
code and models publicly available.


---

## An Inference Approach To Question Answering Over Knowledge Graphs

**Published Date:** 2021-12-21T10:07:55Z

**Link:** http://arxiv.org/pdf/2112.11070v1

**Abstract:**

  Knowledge Graphs (KG) act as a great tool for holding distilled information
from large natural language text corpora. The problem of natural language
querying over knowledge graphs is essential for the human consumption of this
information. This problem is typically addressed by converting the natural
language query to a structured query and then firing the structured query on
the KG. Direct answering models over knowledge graphs in literature are very
few. The query conversion models and direct models both require specific
training data pertaining to the domain of the knowledge graph. In this work, we
convert the problem of natural language querying over knowledge graphs to an
inference problem over premise-hypothesis pairs. Using trained deep learning
models for the converted proxy inferencing problem, we provide the solution for
the original natural language querying problem. Our method achieves over 90%
accuracy on MetaQA dataset, beating the existing state-of-the-art. We also
propose a model for inferencing called Hierarchical Recurrent Path
Encoder(HRPE). The inferencing models can be fine-tuned to be used across
domains with less training data. Our approach does not require large
domain-specific training data for querying on new knowledge graphs from
different domains.


---

## Analyzing the Limits of Self-Supervision in Handling Bias in Language

**Published Date:** 2021-12-16T05:36:08Z

**Link:** http://arxiv.org/pdf/2112.08637v2

**Abstract:**

  Prompting inputs with natural language task descriptions has emerged as a
popular mechanism to elicit reasonably accurate outputs from large-scale
generative language models with little to no in-context supervision. This also
helps gain insight into how well language models capture the semantics of a
wide range of downstream tasks purely from self-supervised pre-training on
massive corpora of unlabeled text. Such models have naturally also been exposed
to a lot of undesirable content like racist and sexist language and there is
limited work on awareness of models along these dimensions. In this paper, we
define and comprehensively evaluate how well such language models capture the
semantics of four tasks for bias: diagnosis, identification, extraction and
rephrasing. We define three broad classes of task descriptions for these tasks:
statement, question, and completion, with numerous lexical variants within each
class. We study the efficacy of prompting for each task using these classes and
the null task description across several decoding methods and few-shot
examples. Our analyses indicate that language models are capable of performing
these tasks to widely varying degrees across different bias dimensions, such as
gender and political affiliation. We believe our work is an important step
towards unbiased language models by quantifying the limits of current
self-supervision objectives at accomplishing such sociologically challenging
tasks.


---

## Fine-Tuning Large Neural Language Models for Biomedical Natural Language
  Processing

**Published Date:** 2021-12-15T04:20:35Z

**Link:** http://arxiv.org/pdf/2112.07869v1

**Abstract:**

  Motivation: A perennial challenge for biomedical researchers and clinical
practitioners is to stay abreast with the rapid growth of publications and
medical notes. Natural language processing (NLP) has emerged as a promising
direction for taming information overload. In particular, large neural language
models facilitate transfer learning by pretraining on unlabeled text, as
exemplified by the successes of BERT models in various NLP applications.
However, fine-tuning such models for an end task remains challenging,
especially with small labeled datasets, which are common in biomedical NLP.
  Results: We conduct a systematic study on fine-tuning stability in biomedical
NLP. We show that finetuning performance may be sensitive to pretraining
settings, especially in low-resource domains. Large models have potential to
attain better performance, but increasing model size also exacerbates
finetuning instability. We thus conduct a comprehensive exploration of
techniques for addressing fine-tuning instability. We show that these
techniques can substantially improve fine-tuning performance for lowresource
biomedical NLP applications. Specifically, freezing lower layers is helpful for
standard BERT-BASE models, while layerwise decay is more effective for
BERT-LARGE and ELECTRA models. For low-resource text similarity tasks such as
BIOSSES, reinitializing the top layer is the optimal strategy. Overall,
domainspecific vocabulary and pretraining facilitate more robust models for
fine-tuning. Based on these findings, we establish new state of the art on a
wide range of biomedical NLP applications.
  Availability and implementation: To facilitate progress in biomedical NLP, we
release our state-of-the-art pretrained and fine-tuned models:
https://aka.ms/BLURB.


---

## From Good to Best: Two-Stage Training for Cross-lingual Machine Reading
  Comprehension

**Published Date:** 2021-12-09T07:31:15Z

**Link:** http://arxiv.org/pdf/2112.04735v1

**Abstract:**

  Cross-lingual Machine Reading Comprehension (xMRC) is challenging due to the
lack of training data in low-resource languages. The recent approaches use
training data only in a resource-rich language like English to fine-tune
large-scale cross-lingual pre-trained language models. Due to the big
difference between languages, a model fine-tuned only by a source language may
not perform well for target languages. Interestingly, we observe that while the
top-1 results predicted by the previous approaches may often fail to hit the
ground-truth answers, the correct answers are often contained in the top-k
predicted results. Based on this observation, we develop a two-stage approach
to enhance the model performance. The first stage targets at recall: we design
a hard-learning (HL) algorithm to maximize the likelihood that the top-k
predictions contain the accurate answer. The second stage focuses on precision:
an answer-aware contrastive learning (AA-CL) mechanism is developed to learn
the fine difference between the accurate answer and other candidates. Our
extensive experiments show that our model significantly outperforms a series of
strong baselines on two cross-lingual MRC benchmark datasets.


---

## DOCmT5: Document-Level Pretraining of Multilingual Language Models

**Published Date:** 2021-12-16T08:58:52Z

**Link:** http://arxiv.org/pdf/2112.08709v2

**Abstract:**

  In this paper, we introduce DOCmT5, a multilingual sequence-to-sequence
language model pretrained with large scale parallel documents. While previous
approaches have focused on leveraging sentence-level parallel data, we try to
build a general-purpose pretrained model that can understand and generate long
documents. We propose a simple and effective pretraining objective - Document
reordering Machine Translation (DrMT), in which the input documents that are
shuffled and masked need to be translated. DrMT brings consistent improvements
over strong baselines on a variety of document-level generation tasks,
including over 12 BLEU points for seen-language-pair document-level MT, over 7
BLEU points for unseen-language-pair document-level MT and over 3 ROUGE-1
points for seen-language-pair cross-lingual summarization. We achieve
state-of-the-art (SOTA) on WMT20 De-En and IWSLT15 Zh-En document translation
tasks. We also conduct extensive analysis on various factors for document
pretraining, including (1) The effects of pretraining data quality and (2) The
effects of combining mono-lingual and cross-lingual pretraining. We plan to
make our model checkpoints publicly available.


---

