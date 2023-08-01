## Exploring Neural Net Augmentation to BERT for Question Answering on
  SQUAD 2.0

**Published Date:** 2019-08-04T16:48:24Z

**Link:** http://arxiv.org/pdf/1908.01767v3

**Abstract:**

  Enhancing machine capabilities to answer questions has been a topic of
considerable focus in recent years of NLP research. Language models like
Embeddings from Language Models (ELMo)[1] and Bidirectional Encoder
Representations from Transformers (BERT) [2] have been very successful in
developing general purpose language models that can be optimized for a large
number of downstream language tasks. In this work, we focused on augmenting the
pre-trained BERT language model with different output neural net architectures
and compared their performance on question answering task posed by the Stanford
Question Answering Dataset 2.0 (SQUAD 2.0) [3]. Additionally, we also
fine-tuned the pre-trained BERT model parameters to demonstrate its
effectiveness in adapting to specialized language tasks. Our best output
network, is the contextualized CNN that performs on both the unanswerable and
answerable question answering tasks with F1 scores of 75.32 and 64.85
respectively.


---

## Encoder-Agnostic Adaptation for Conditional Language Generation

**Published Date:** 2019-08-19T17:22:58Z

**Link:** http://arxiv.org/pdf/1908.06938v2

**Abstract:**

  Large pretrained language models have changed the way researchers approach
discriminative natural language understanding tasks, leading to the dominance
of approaches that adapt a pretrained model for arbitrary downstream tasks.
However it is an open-question how to use similar techniques for language
generation. Early results in the encoder-agnostic setting have been mostly
negative. In this work we explore methods for adapting a pretrained language
model to arbitrary conditional input. We observe that pretrained transformer
models are sensitive to large parameter changes during tuning. We therefore
propose an adaptation that directly injects arbitrary conditioning into self
attention, an approach we call pseudo self attention. Through experiments on
four diverse conditional text generation tasks we show that this
encoder-agnostic technique outperforms strong baselines, produces coherent
generations, and is data efficient.


---

## ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for
  Vision-and-Language Tasks

**Published Date:** 2019-08-06T17:33:52Z

**Link:** http://arxiv.org/pdf/1908.02265v1

**Abstract:**

  We present ViLBERT (short for Vision-and-Language BERT), a model for learning
task-agnostic joint representations of image content and natural language. We
extend the popular BERT architecture to a multi-modal two-stream model,
pro-cessing both visual and textual inputs in separate streams that interact
through co-attentional transformer layers. We pretrain our model through two
proxy tasks on the large, automatically collected Conceptual Captions dataset
and then transfer it to multiple established vision-and-language tasks --
visual question answering, visual commonsense reasoning, referring expressions,
and caption-based image retrieval -- by making only minor additions to the base
architecture. We observe significant improvements across tasks compared to
existing task-specific models -- achieving state-of-the-art on all four tasks.
Our work represents a shift away from learning groundings between vision and
language only as part of task training and towards treating visual grounding as
a pretrainable and transferable capability.


---

