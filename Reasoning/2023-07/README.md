## Garbage in, garbage out: Zero-shot detection of crime using Large
  Language Models

**Published Date:** 2023-07-04T01:29:15Z

**Link:** http://arxiv.org/pdf/2307.06844v1

**Abstract:**

  This paper proposes exploiting the common sense knowledge learned by large
language models to perform zero-shot reasoning about crimes given textual
descriptions of surveillance videos. We show that when video is (manually)
converted to high quality textual descriptions, large language models are
capable of detecting and classifying crimes with state-of-the-art performance
using only zero-shot reasoning. However, existing automated video-to-text
approaches are unable to generate video descriptions of sufficient quality to
support reasoning (garbage video descriptions into the large language model,
garbage out).


---

## Leveraging Large Language Models to Generate Answer Set Programs

**Published Date:** 2023-07-15T03:40:55Z

**Link:** http://arxiv.org/pdf/2307.07699v1

**Abstract:**

  Large language models (LLMs), such as GPT-3 and GPT-4, have demonstrated
exceptional performance in various natural language processing tasks and have
shown the ability to solve certain reasoning problems. However, their reasoning
capabilities are limited and relatively shallow, despite the application of
various prompting techniques. In contrast, formal logic is adept at handling
complex reasoning, but translating natural language descriptions into formal
logic is a challenging task that non-experts struggle with. This paper proposes
a neuro-symbolic method that combines the strengths of large language models
and answer set programming. Specifically, we employ an LLM to transform natural
language descriptions of logic puzzles into answer set programs. We carefully
design prompts for an LLM to convert natural language descriptions into answer
set programs in a step by step manner. Surprisingly, with just a few in-context
learning examples, LLMs can generate reasonably complex answer set programs.
The majority of errors made are relatively simple and can be easily corrected
by humans, thus enabling LLMs to effectively assist in the creation of answer
set programs.


---

## Large Language Models Perform Diagnostic Reasoning

**Published Date:** 2023-07-18T01:43:00Z

**Link:** http://arxiv.org/pdf/2307.08922v1

**Abstract:**

  We explore the extension of chain-of-thought (CoT) prompting to medical
reasoning for the task of automatic diagnosis. Motivated by doctors' underlying
reasoning process, we present Diagnostic-Reasoning CoT (DR-CoT). Empirical
results demonstrate that by simply prompting large language models trained only
on general text corpus with two DR-CoT exemplars, the diagnostic accuracy
improves by 15% comparing to standard prompting. Moreover, the gap reaches a
pronounced 18% in out-domain settings. Our findings suggest expert-knowledge
reasoning in large language models can be elicited through proper promptings.


---

## Leveraging Large Language Models to Generate Answer Set Programs

**Published Date:** 2023-07-15T03:40:55Z

**Link:** http://arxiv.org/pdf/2307.07699v1

**Abstract:**

  Large language models (LLMs), such as GPT-3 and GPT-4, have demonstrated
exceptional performance in various natural language processing tasks and have
shown the ability to solve certain reasoning problems. However, their reasoning
capabilities are limited and relatively shallow, despite the application of
various prompting techniques. In contrast, formal logic is adept at handling
complex reasoning, but translating natural language descriptions into formal
logic is a challenging task that non-experts struggle with. This paper proposes
a neuro-symbolic method that combines the strengths of large language models
and answer set programming. Specifically, we employ an LLM to transform natural
language descriptions of logic puzzles into answer set programs. We carefully
design prompts for an LLM to convert natural language descriptions into answer
set programs in a step by step manner. Surprisingly, with just a few in-context
learning examples, LLMs can generate reasonably complex answer set programs.
The majority of errors made are relatively simple and can be easily corrected
by humans, thus enabling LLMs to effectively assist in the creation of answer
set programs.


---

## Cross-Lingual NER for Financial Transaction Data in Low-Resource
  Languages

**Published Date:** 2023-07-16T00:45:42Z

**Link:** http://arxiv.org/pdf/2307.08714v1

**Abstract:**

  We propose an efficient modeling framework for cross-lingual named entity
recognition in semi-structured text data. Our approach relies on both knowledge
distillation and consistency training. The modeling framework leverages
knowledge from a large language model (XLMRoBERTa) pre-trained on the source
language, with a student-teacher relationship (knowledge distillation). The
student model incorporates unsupervised consistency training (with KL
divergence loss) on the low-resource target language.
  We employ two independent datasets of SMSs in English and Arabic, each
carrying semi-structured banking transaction information, and focus on
exhibiting the transfer of knowledge from English to Arabic. With access to
only 30 labeled samples, our model can generalize the recognition of merchants,
amounts, and other fields from English to Arabic. We show that our modeling
approach, while efficient, performs best overall when compared to
state-of-the-art approaches like DistilBERT pre-trained on the target language
or a supervised model directly trained on labeled data in the target language.
  Our experiments show that it is enough to learn to recognize entities in
English to reach reasonable performance in a low-resource language in the
presence of a few labeled samples of semi-structured data. The proposed
framework has implications for developing multi-lingual applications,
especially in geographies where digital endeavors rely on both English and one
or more low-resource language(s), sometimes mixed with English or employed
singly.


---

## GenRec: Large Language Model for Generative Recommendation

**Published Date:** 2023-07-02T02:37:07Z

**Link:** http://arxiv.org/pdf/2307.00457v2

**Abstract:**

  In recent years, large language models (LLM) have emerged as powerful tools
for diverse natural language processing tasks. However, their potential for
recommender systems under the generative recommendation paradigm remains
relatively unexplored. This paper presents an innovative approach to
recommendation systems using large language models (LLMs) based on text data.
In this paper, we present a novel LLM for generative recommendation (GenRec)
that utilized the expressive power of LLM to directly generate the target item
to recommend, rather than calculating ranking score for each candidate item one
by one as in traditional discriminative recommendation. GenRec uses LLM's
understanding ability to interpret context, learn user preferences, and
generate relevant recommendation. Our proposed approach leverages the vast
knowledge encoded in large language models to accomplish recommendation tasks.
We first we formulate specialized prompts to enhance the ability of LLM to
comprehend recommendation tasks. Subsequently, we use these prompts to
fine-tune the LLaMA backbone LLM on a dataset of user-item interactions,
represented by textual data, to capture user preferences and item
characteristics. Our research underscores the potential of LLM-based generative
recommendation in revolutionizing the domain of recommendation systems and
offers a foundational framework for future explorations in this field. We
conduct extensive experiments on benchmark datasets, and the experiments shows
that our GenRec has significant better results on large dataset.


---

## Divert More Attention to Vision-Language Object Tracking

**Published Date:** 2023-07-19T15:22:06Z

**Link:** http://arxiv.org/pdf/2307.10046v1

**Abstract:**

  Multimodal vision-language (VL) learning has noticeably pushed the tendency
toward generic intelligence owing to emerging large foundation models. However,
tracking, as a fundamental vision problem, surprisingly enjoys less bonus from
recent flourishing VL learning. We argue that the reasons are two-fold: the
lack of large-scale vision-language annotated videos and ineffective
vision-language interaction learning of current works. These nuisances motivate
us to design more effective vision-language representation for tracking,
meanwhile constructing a large database with language annotation for model
learning. Particularly, in this paper, we first propose a general attribute
annotation strategy to decorate videos in six popular tracking benchmarks,
which contributes a large-scale vision-language tracking database with more
than 23,000 videos. We then introduce a novel framework to improve tracking by
learning a unified-adaptive VL representation, where the cores are the proposed
asymmetric architecture search and modality mixer (ModaMixer). To further
improve VL representation, we introduce a contrastive loss to align different
modalities. To thoroughly evidence the effectiveness of our method, we
integrate the proposed framework on three tracking methods with different
designs, i.e., the CNN-based SiamCAR, the Transformer-based OSTrack, and the
hybrid structure TransT. The experiments demonstrate that our framework can
significantly improve all baselines on six benchmarks. Besides empirical
results, we theoretically analyze our approach to show its rationality. By
revealing the potential of VL representation, we expect the community to divert
more attention to VL tracking and hope to open more possibilities for future
tracking with diversified multimodal messages.


---

## Leveraging Large Language Models to Generate Answer Set Programs

**first_author:** Adam Ishay et al.

**Published Date:** 2023-07-15T03:40:55Z

**Link:** http://arxiv.org/pdf/2307.07699v1

**Abstract:**

  Large language models (LLMs), such as GPT-3 and GPT-4, have demonstrated
exceptional performance in various natural language processing tasks and have
shown the ability to solve certain reasoning problems. However, their reasoning
capabilities are limited and relatively shallow, despite the application of
various prompting techniques. In contrast, formal logic is adept at handling
complex reasoning, but translating natural language descriptions into formal
logic is a challenging task that non-experts struggle with. This paper proposes
a neuro-symbolic method that combines the strengths of large language models
and answer set programming. Specifically, we employ an LLM to transform natural
language descriptions of logic puzzles into answer set programs. We carefully
design prompts for an LLM to convert natural language descriptions into answer
set programs in a step by step manner. Surprisingly, with just a few in-context
learning examples, LLMs can generate reasonably complex answer set programs.
The majority of errors made are relatively simple and can be easily corrected
by humans, thus enabling LLMs to effectively assist in the creation of answer
set programs.


---

