## Dynamic Large Language Models on Blockchains

**Published Date:** 2023-07-20T03:26:57Z

**Link:** http://arxiv.org/pdf/2307.10549v1

**Abstract:**

  Training and deploying the large language models requires a large mount of
computational resource because the language models contain billions of
parameters and the text has thousands of tokens. Another problem is that the
large language models are static. They are fixed after the training process. To
tackle these issues, in this paper, we propose to train and deploy the dynamic
large language model on blockchains, which have high computation performance
and are distributed across a network of computers. A blockchain is a secure,
decentralized, and transparent system that allows for the creation of a
tamper-proof ledger for transactions without the need for intermediaries. The
dynamic large language models can continuously learn from the user input after
the training process. Our method provides a new way to develop the large
language models and also sheds a light on the next generation artificial
intelligence systems.


---

## Negated Complementary Commonsense using Large Language Models

**Published Date:** 2023-07-13T15:03:48Z

**Link:** http://arxiv.org/pdf/2307.06794v1

**Abstract:**

  Larger language models, such as GPT-3, have shown to be excellent in many
tasks. However, we demonstrate that out-of-ordinary questions can throw the
model off guard. This work focuses on finding answers to negated complementary
questions in commonsense scenarios. We illustrate how such questions adversely
affect the model responses. We propose a model-agnostic methodology to improve
the performance in negated complementary scenarios. Our method outperforms
few-shot generation from GPT-3 (by more than 11 points) and, more importantly,
highlights the significance of studying the response of large language models
in negated complementary questions. The code, data, and experiments are
available under: https://github.com/navidre/negated_complementary_commonsense.


---

## Exploring the In-context Learning Ability of Large Language Model for
  Biomedical Concept Linking

**Published Date:** 2023-07-03T16:19:50Z

**Link:** http://arxiv.org/pdf/2307.01137v1

**Abstract:**

  The biomedical field relies heavily on concept linking in various areas such
as literature mining, graph alignment, information retrieval,
question-answering, data, and knowledge integration. Although large language
models (LLMs) have made significant strides in many natural language processing
tasks, their effectiveness in biomedical concept mapping is yet to be fully
explored. This research investigates a method that exploits the in-context
learning (ICL) capabilities of large models for biomedical concept linking. The
proposed approach adopts a two-stage retrieve-and-rank framework. Initially,
biomedical concepts are embedded using language models, and then embedding
similarity is utilized to retrieve the top candidates. These candidates'
contextual information is subsequently incorporated into the prompt and
processed by a large language model to re-rank the concepts. This approach
achieved an accuracy of 90.% in BC5CDR disease entity normalization and 94.7%
in chemical entity normalization, exhibiting a competitive performance relative
to supervised learning methods. Further, it showed a significant improvement,
with an over 20-point absolute increase in F1 score on an oncology matching
dataset. Extensive qualitative assessments were conducted, and the benefits and
potential shortcomings of using large language models within the biomedical
domain were discussed. were discussed.


---

## Multilevel Large Language Models for Everyone

**Published Date:** 2023-07-25T03:18:04Z

**Link:** http://arxiv.org/pdf/2307.13221v1

**Abstract:**

  Large language models have made significant progress in the past few years.
However, they are either generic {\it or} field specific, splitting the
community into different groups. In this paper, we unify these large language
models into a larger map, where the generic {\it and} specific models are
linked together and can improve each other, based on the user personal input
and information from the internet. The idea of linking several large language
models together is inspired by the functionality of human brain. The specific
regions on the brain cortex are specific for certain low level functionality.
And these regions can jointly work together to achieve more complex high level
functionality. Such behavior on human brain cortex sheds the light to design
the multilevel large language models that contain global level, field level and
user level models. The user level models run on local machines to achieve
efficient response and protect the user's privacy. Such multilevel models
reduce some redundancy and perform better than the single level models. The
proposed multilevel idea can be applied in various applications, such as
natural language processing, computer vision tasks, professional assistant,
business and healthcare.


---

## Vision Language Transformers: A Survey

**Published Date:** 2023-07-06T19:08:56Z

**Link:** http://arxiv.org/pdf/2307.03254v1

**Abstract:**

  Vision language tasks, such as answering questions about or generating
captions that describe an image, are difficult tasks for computers to perform.
A relatively recent body of research has adapted the pretrained transformer
architecture introduced in \citet{vaswani2017attention} to vision language
modeling. Transformer models have greatly improved performance and versatility
over previous vision language models. They do so by pretraining models on a
large generic datasets and transferring their learning to new tasks with minor
changes in architecture and parameter values. This type of transfer learning
has become the standard modeling practice in both natural language processing
and computer vision. Vision language transformers offer the promise of
producing similar advancements in tasks which require both vision and language.
In this paper, we provide a broad synthesis of the currently available research
on vision language transformer models and offer some analysis of their
strengths, limitations and some open questions that remain.


---

## Backdoor Attacks for In-Context Learning with Language Models

**Published Date:** 2023-07-27T08:28:58Z

**Link:** http://arxiv.org/pdf/2307.14692v1

**Abstract:**

  Because state-of-the-art language models are expensive to train, most
practitioners must make use of one of the few publicly available language
models or language model APIs. This consolidation of trust increases the
potency of backdoor attacks, where an adversary tampers with a machine learning
model in order to make it perform some malicious behavior on inputs that
contain a predefined backdoor trigger. We show that the in-context learning
ability of large language models significantly complicates the question of
developing backdoor attacks, as a successful backdoor must work against various
prompting strategies and should not affect the model's general purpose
capabilities. We design a new attack for eliciting targeted misclassification
when language models are prompted to perform a particular target task and
demonstrate the feasibility of this attack by backdooring multiple large
language models ranging in size from 1.3 billion to 6 billion parameters.
Finally we study defenses to mitigate the potential harms of our attack: for
example, while in the white-box setting we show that fine-tuning models for as
few as 500 steps suffices to remove the backdoor behavior, in the black-box
setting we are unable to develop a successful defense that relies on prompt
engineering alone.


---

## MorphPiece : Moving away from Statistical Language Representation

**Published Date:** 2023-07-14T10:35:04Z

**Link:** http://arxiv.org/pdf/2307.07262v1

**Abstract:**

  Tokenization is a critical part of modern NLP pipelines. However,
contemporary tokenizers for Large Language Models are based on statistical
analysis of text corpora, without much consideration to the linguistic
features. We propose a linguistically motivated tokenization scheme,
MorphPiece, which is based partly on morphological segmentation of the
underlying text. A GPT-style causal language model trained on this tokenizer
(called MorphGPT) shows superior convergence compared to the same architecture
trained on a standard BPE tokenizer. Specifically we get Language Modeling
performance comparable to a 6 times larger model. Additionally, we evaluate
MorphGPT on a variety of NLP tasks in supervised and unsupervised settings and
find superior performance across the board, compared to GPT-2 model.


---

## A Systematic Survey of Prompt Engineering on Vision-Language Foundation
  Models

**Published Date:** 2023-07-24T17:58:06Z

**Link:** http://arxiv.org/pdf/2307.12980v1

**Abstract:**

  Prompt engineering is a technique that involves augmenting a large
pre-trained model with task-specific hints, known as prompts, to adapt the
model to new tasks. Prompts can be created manually as natural language
instructions or generated automatically as either natural language instructions
or vector representations. Prompt engineering enables the ability to perform
predictions based solely on prompts without updating model parameters, and the
easier application of large pre-trained models in real-world tasks. In past
years, Prompt engineering has been well-studied in natural language processing.
Recently, it has also been intensively studied in vision-language modeling.
However, there is currently a lack of a systematic overview of prompt
engineering on pre-trained vision-language models. This paper aims to provide a
comprehensive survey of cutting-edge research in prompt engineering on three
types of vision-language models: multimodal-to-text generation models (e.g.
Flamingo), image-text matching models (e.g. CLIP), and text-to-image generation
models (e.g. Stable Diffusion). For each type of model, a brief model summary,
prompting methods, prompting-based applications, and the corresponding
responsibility and integrity issues are summarized and discussed. Furthermore,
the commonalities and differences between prompting on vision-language models,
language models, and vision models are also discussed. The challenges, future
directions, and research opportunities are summarized to foster future research
on this topic.


---

## SuryaKiran at MEDIQA-Sum 2023: Leveraging LoRA for Clinical Dialogue
  Summarization

**Published Date:** 2023-07-11T10:38:58Z

**Link:** http://arxiv.org/pdf/2307.05162v1

**Abstract:**

  Finetuning Large Language Models helps improve the results for
domain-specific use cases. End-to-end finetuning of large language models is
time and resource intensive and has high storage requirements to store the
finetuned version of the large language model. Parameter Efficient Fine Tuning
(PEFT) methods address the time and resource challenges by keeping the large
language model as a fixed base and add additional layers, which the PEFT
methods finetune. This paper demonstrates the evaluation results for one such
PEFT method Low Rank Adaptation (LoRA), for Clinical Dialogue Summarization.
The evaluation results show that LoRA works at par with end-to-end finetuning
for a large language model. The paper presents the evaluations done for solving
both the Subtask A and B from ImageCLEFmedical
{https://www.imageclef.org/2023/medical}


---

## Evaluating the Capability of Large-scale Language Models on Chinese
  Grammatical Error Correction Task

**Published Date:** 2023-07-08T13:10:59Z

**Link:** http://arxiv.org/pdf/2307.03972v1

**Abstract:**

  Large-scale language models (LLMs) has shown remarkable capability in various
of Natural Language Processing (NLP) tasks and attracted lots of attention
recently. However, some studies indicated that large language models fail to
achieve promising result beyond the state-of-the-art models in English
grammatical error correction (GEC) tasks. In this report, we aim to explore the
how large language models perform on Chinese grammatical error correction tasks
and provide guidance for future work. We conduct experiments with 3 different
LLMs of different model scale on 4 Chinese GEC dataset. Our experimental
results indicate that the performances of LLMs on automatic evaluation metrics
falls short of the previous sota models because of the problem of
over-correction. Furthermore, we also discover notable variations in the
performance of LLMs when evaluated on different data distributions. Our
findings demonstrates that further investigation is required for the
application of LLMs on Chinese GEC task.


---

## Gloss Attention for Gloss-free Sign Language Translation

**Published Date:** 2023-07-14T14:07:55Z

**Link:** http://arxiv.org/pdf/2307.07361v1

**Abstract:**

  Most sign language translation (SLT) methods to date require the use of gloss
annotations to provide additional supervision information, however, the
acquisition of gloss is not easy. To solve this problem, we first perform an
analysis of existing models to confirm how gloss annotations make SLT easier.
We find that it can provide two aspects of information for the model, 1) it can
help the model implicitly learn the location of semantic boundaries in
continuous sign language videos, 2) it can help the model understand the sign
language video globally. We then propose \emph{gloss attention}, which enables
the model to keep its attention within video segments that have the same
semantics locally, just as gloss helps existing models do. Furthermore, we
transfer the knowledge of sentence-to-sentence similarity from the natural
language model to our gloss attention SLT network (GASLT) to help it understand
sign language videos at the sentence level. Experimental results on multiple
large-scale sign language datasets show that our proposed GASLT model
significantly outperforms existing methods. Our code is provided in
\url{https://github.com/YinAoXiong/GASLT}.


---

## Several categories of Large Language Models (LLMs): A Short Survey

**Published Date:** 2023-07-05T18:18:23Z

**Link:** http://arxiv.org/pdf/2307.10188v1

**Abstract:**

  Large Language Models(LLMs)have become effective tools for natural language
processing and have been used in many different fields. This essay offers a
succinct summary of various LLM subcategories. The survey emphasizes recent
developments and efforts made for various LLM kinds, including task-based
financial LLMs, multilingual language LLMs, biomedical and clinical LLMs,
vision language LLMs, and code language models. The survey gives a general
summary of the methods, attributes, datasets, transformer models, and
comparison metrics applied in each category of LLMs. Furthermore, it highlights
unresolved problems in the field of developing chatbots and virtual assistants,
such as boosting natural language processing, enhancing chatbot intelligence,
and resolving moral and legal dilemmas. The purpose of this study is to provide
readers, developers, academics, and users interested in LLM-based chatbots and
virtual intelligent assistant technologies with useful information and future
directions.


---

## TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a
  Domain-Specific Expert in Transportation Safety

**Published Date:** 2023-07-28T05:17:11Z

**Link:** http://arxiv.org/pdf/2307.15311v1

**Abstract:**

  Large Language Models (LLMs) have shown remarkable effectiveness in various
general-domain natural language processing (NLP) tasks. However, their
performance in transportation safety domain tasks has been suboptimal,
primarily attributed to the requirement for specialized transportation safety
expertise in generating accurate responses [1]. To address this challenge, we
introduce TrafficSafetyGPT, a novel LLAMA-based model, which has undergone
supervised fine-tuning using TrafficSafety-2K dataset which has human labels
from government produced guiding books and ChatGPT-generated instruction-output
pairs. Our proposed TrafficSafetyGPT model and TrafficSafety-2K train dataset
are accessible at https://github.com/ozheng1993/TrafficSafetyGPT.


---

## PatternGPT :A Pattern-Driven Framework for Large Language Model Text
  Generation

**Published Date:** 2023-07-02T04:32:41Z

**Link:** http://arxiv.org/pdf/2307.00470v4

**Abstract:**

  Large language models(LLMS)have shown excellent text generation capabilities,
capable of generating fluent human-like responses for many downstream tasks.
However, applying large language models to real-world critical tasks remains
challenging due to their susceptibility to hallucinations and inability to
directly use external knowledge. To cope with the above challenges, this paper
proposes PatternGPT, a pattern-driven text generation framework for Large
Language Models. Firstly, the framework utilizes the extraction capability of
Large Language Models to generate rich and diversified structured and
formalized patterns, which facilitates the introduction of external knowledge
to do the computation, and then draws on the idea of federated learning to use
multiple agents to achieve the sharing in order to obtain more diversified
patterns, and finally uses judgment criteria and optimization algorithm to
search for high-quality patterns to guide the generation of models. Finally,
external knowledge such as judgment criteria and optimization algorithms are
used to search for high-quality patterns, and the searched patterns are used to
guide model generation. This framework has the advantages of generating
diversified patterns, protecting data privacy, combining external knowledge,
and improving the quality of generation, which provides an effective method to
optimize the text generation capability of large language models, and make it
better applied to the field of intelligent dialogue and content generation.


---

## On the application of Large Language Models for language teaching and
  assessment technology

**Published Date:** 2023-07-17T11:12:56Z

**Link:** http://arxiv.org/pdf/2307.08393v1

**Abstract:**

  The recent release of very large language models such as PaLM and GPT-4 has
made an unprecedented impact in the popular media and public consciousness,
giving rise to a mixture of excitement and fear as to their capabilities and
potential uses, and shining a light on natural language processing research
which had not previously received so much attention. The developments offer
great promise for education technology, and in this paper we look specifically
at the potential for incorporating large language models in AI-driven language
teaching and assessment systems. We consider several research areas and also
discuss the risks and ethical considerations surrounding generative AI in
education technology for language learners. Overall we find that larger
language models offer improvements over previous models in text generation,
opening up routes toward content generation which had not previously been
plausible. For text generation they must be prompted carefully and their
outputs may need to be reshaped before they are ready for use. For automated
grading and grammatical error correction, tasks whose progress is checked on
well-known benchmarks, early investigations indicate that large language models
on their own do not improve on state-of-the-art results according to standard
evaluation metrics. For grading it appears that linguistic features established
in the literature should still be used for best performance, and for error
correction it may be that the models can offer alternative feedback styles
which are not measured sensitively with existing methods. In all cases, there
is work to be done to experiment with the inclusion of large language models in
education technology for language learners, in order to properly understand and
report on their capacities and limitations, and to ensure that foreseeable
risks such as misinformation and harmful bias are mitigated.


---

## Unveiling Gender Bias in Terms of Profession Across LLMs: Analyzing and
  Addressing Sociological Implications

**Published Date:** 2023-07-18T11:38:45Z

**Link:** http://arxiv.org/pdf/2307.09162v1

**Abstract:**

  Gender bias in artificial intelligence (AI) and natural language processing
has garnered significant attention due to its potential impact on societal
perceptions and biases. This research paper aims to analyze gender bias in
Large Language Models (LLMs) with a focus on multiple comparisons between GPT-2
and GPT-3.5, some prominent language models, to better understand its
implications. Through a comprehensive literature review, the study examines
existing research on gender bias in AI language models and identifies gaps in
the current knowledge. The methodology involves collecting and preprocessing
data from GPT-2 and GPT-3.5, and employing in-depth quantitative analysis
techniques to evaluate gender bias in the generated text. The findings shed
light on gendered word associations, language usage, and biased narratives
present in the outputs of these Large Language Models. The discussion explores
the ethical implications of gender bias and its potential consequences on
social perceptions and marginalized communities. Additionally, the paper
presents strategies for reducing gender bias in LLMs, including algorithmic
approaches and data augmentation techniques. The research highlights the
importance of interdisciplinary collaborations and the role of sociological
studies in mitigating gender bias in AI models. By addressing these issues, we
can pave the way for more inclusive and unbiased AI systems that have a
positive impact on society.


---

## Multilevel Large Language Models for Everyone

**Published Date:** 2023-07-25T03:18:04Z

**Link:** http://arxiv.org/pdf/2307.13221v1

**Abstract:**

  Large language models have made significant progress in the past few years.
However, they are either generic {\it or} field specific, splitting the
community into different groups. In this paper, we unify these large language
models into a larger map, where the generic {\it and} specific models are
linked together and can improve each other, based on the user personal input
and information from the internet. The idea of linking several large language
models together is inspired by the functionality of human brain. The specific
regions on the brain cortex are specific for certain low level functionality.
And these regions can jointly work together to achieve more complex high level
functionality. Such behavior on human brain cortex sheds the light to design
the multilevel large language models that contain global level, field level and
user level models. The user level models run on local machines to achieve
efficient response and protect the user's privacy. Such multilevel models
reduce some redundancy and perform better than the single level models. The
proposed multilevel idea can be applied in various applications, such as
natural language processing, computer vision tasks, professional assistant,
business and healthcare.


---

## MorphPiece : Moving away from Statistical Language Representation

**Published Date:** 2023-07-14T10:35:04Z

**Link:** http://arxiv.org/pdf/2307.07262v1

**Abstract:**

  Tokenization is a critical part of modern NLP pipelines. However,
contemporary tokenizers for Large Language Models are based on statistical
analysis of text corpora, without much consideration to the linguistic
features. We propose a linguistically motivated tokenization scheme,
MorphPiece, which is based partly on morphological segmentation of the
underlying text. A GPT-style causal language model trained on this tokenizer
(called MorphGPT) shows superior convergence compared to the same architecture
trained on a standard BPE tokenizer. Specifically we get Language Modeling
performance comparable to a 6 times larger model. Additionally, we evaluate
MorphGPT on a variety of NLP tasks in supervised and unsupervised settings and
find superior performance across the board, compared to GPT-2 model.


---

## On the application of Large Language Models for language teaching and
  assessment technology

**Published Date:** 2023-07-17T11:12:56Z

**Link:** http://arxiv.org/pdf/2307.08393v1

**Abstract:**

  The recent release of very large language models such as PaLM and GPT-4 has
made an unprecedented impact in the popular media and public consciousness,
giving rise to a mixture of excitement and fear as to their capabilities and
potential uses, and shining a light on natural language processing research
which had not previously received so much attention. The developments offer
great promise for education technology, and in this paper we look specifically
at the potential for incorporating large language models in AI-driven language
teaching and assessment systems. We consider several research areas and also
discuss the risks and ethical considerations surrounding generative AI in
education technology for language learners. Overall we find that larger
language models offer improvements over previous models in text generation,
opening up routes toward content generation which had not previously been
plausible. For text generation they must be prompted carefully and their
outputs may need to be reshaped before they are ready for use. For automated
grading and grammatical error correction, tasks whose progress is checked on
well-known benchmarks, early investigations indicate that large language models
on their own do not improve on state-of-the-art results according to standard
evaluation metrics. For grading it appears that linguistic features established
in the literature should still be used for best performance, and for error
correction it may be that the models can offer alternative feedback styles
which are not measured sensitively with existing methods. In all cases, there
is work to be done to experiment with the inclusion of large language models in
education technology for language learners, in order to properly understand and
report on their capacities and limitations, and to ensure that foreseeable
risks such as misinformation and harmful bias are mitigated.


---

## SINC: Self-Supervised In-Context Learning for Vision-Language Tasks

**Published Date:** 2023-07-15T08:33:08Z

**Link:** http://arxiv.org/pdf/2307.07742v1

**Abstract:**

  Large Pre-trained Transformers exhibit an intriguing capacity for in-context
learning. Without gradient updates, these models can rapidly construct new
predictors from demonstrations presented in the inputs. Recent works promote
this ability in the vision-language domain by incorporating visual information
into large language models that can already make in-context predictions.
However, these methods could inherit issues in the language domain, such as
template sensitivity and hallucination. Also, the scale of these language
models raises a significant demand for computations, making learning and
operating these models resource-intensive. To this end, we raise a question:
``How can we enable in-context learning for general models without being
constrained on large language models?". To answer it, we propose a succinct and
general framework, Self-supervised IN-Context learning (SINC), that introduces
a meta-model to learn on self-supervised prompts consisting of tailored
demonstrations. The learned models can be transferred to downstream tasks for
making in-context predictions on-the-fly. Extensive experiments show that SINC
outperforms gradient-based methods in various vision-language tasks under
few-shot settings. Furthermore, the designs of SINC help us investigate the
benefits of in-context learning across different tasks, and the analysis
further reveals the essential components for the emergence of in-context
learning in the vision-language domain.


---

## RSGPT: A Remote Sensing Vision Language Model and Benchmark

**Published Date:** 2023-07-28T02:23:35Z

**Link:** http://arxiv.org/pdf/2307.15266v1

**Abstract:**

  The emergence of large-scale large language models, with GPT-4 as a prominent
example, has significantly propelled the rapid advancement of artificial
general intelligence and sparked the revolution of Artificial Intelligence 2.0.
In the realm of remote sensing (RS), there is a growing interest in developing
large vision language models (VLMs) specifically tailored for data analysis in
this domain. However, current research predominantly revolves around visual
recognition tasks, lacking comprehensive, large-scale image-text datasets that
are aligned and suitable for training large VLMs, which poses significant
challenges to effectively training such models for RS applications. In computer
vision, recent research has demonstrated that fine-tuning large vision language
models on small-scale, high-quality datasets can yield impressive performance
in visual and language understanding. These results are comparable to
state-of-the-art VLMs trained from scratch on massive amounts of data, such as
GPT-4. Inspired by this captivating idea, in this work, we build a high-quality
Remote Sensing Image Captioning dataset (RSICap) that facilitates the
development of large VLMs in the RS field. Unlike previous RS datasets that
either employ model-generated captions or short descriptions, RSICap comprises
2,585 human-annotated captions with rich and high-quality information. This
dataset offers detailed descriptions for each image, encompassing scene
descriptions (e.g., residential area, airport, or farmland) as well as object
information (e.g., color, shape, quantity, absolute position, etc). To
facilitate the evaluation of VLMs in the field of RS, we also provide a
benchmark evaluation dataset called RSIEval. This dataset consists of
human-annotated captions and visual question-answer pairs, allowing for a
comprehensive assessment of VLMs in the context of RS.


---

## Leveraging Label Variation in Large Language Models for Zero-Shot Text
  Classification

**Published Date:** 2023-07-24T17:49:31Z

**Link:** http://arxiv.org/pdf/2307.12973v1

**Abstract:**

  The zero-shot learning capabilities of large language models (LLMs) make them
ideal for text classification without annotation or supervised training. Many
studies have shown impressive results across multiple tasks. While tasks, data,
and results differ widely, their similarities to human annotation can aid us in
tackling new tasks with minimal expenses. We evaluate using 5 state-of-the-art
LLMs as "annotators" on 5 different tasks (age, gender, topic, sentiment
prediction, and hate speech detection), across 4 languages: English, French,
German, and Spanish. No single model excels at all tasks, across languages, or
across all labels within a task. However, aggregation techniques designed for
human annotators perform substantially better than any one individual model.
Overall, though, LLMs do not rival even simple supervised models, so they do
not (yet) replace the need for human annotation. We also discuss the tradeoffs
between speed, accuracy, cost, and bias when it comes to aggregated model
labeling versus human annotation.


---

## BatGPT: A Bidirectional Autoregessive Talker from Generative Pre-trained
  Transformer

**Published Date:** 2023-07-01T15:10:01Z

**Link:** http://arxiv.org/pdf/2307.00360v1

**Abstract:**

  BatGPT is a large-scale language model designed and trained jointly by Wuhan
University and Shanghai Jiao Tong University. It is capable of generating
highly natural and fluent text in response to various types of input, including
text prompts, images, and audio. In the modeling level, we employ a
bidirectional autoregressive architecture that allows the model to efficiently
capture the complex dependencies of natural language, making it highly
effective in tasks such as language generation, dialog systems, and question
answering. Moreover, the bidirectional autoregressive modeling not only
operates from left to right but also from right to left, effectively reducing
fixed memory effects and alleviating model hallucinations.
  In the training aspect, we propose a novel parameter expansion method for
leveraging the pre-training of smaller models and employ reinforcement learning
from both AI and human feedback, aimed at improving the model's alignment
performance. Overall, these approaches significantly improve the effectiveness
of BatGPT, and the model can be utilized for a wide range of natural language
applications.


---

## Multilevel Large Language Models for Everyone

**first_author:** Yuanhao Gong et al.

**Published Date:** 2023-07-25T03:18:04Z

**Link:** http://arxiv.org/pdf/2307.13221v1

**Abstract:**

  Large language models have made significant progress in the past few years.
However, they are either generic {\it or} field specific, splitting the
community into different groups. In this paper, we unify these large language
models into a larger map, where the generic {\it and} specific models are
linked together and can improve each other, based on the user personal input
and information from the internet. The idea of linking several large language
models together is inspired by the functionality of human brain. The specific
regions on the brain cortex are specific for certain low level functionality.
And these regions can jointly work together to achieve more complex high level
functionality. Such behavior on human brain cortex sheds the light to design
the multilevel large language models that contain global level, field level and
user level models. The user level models run on local machines to achieve
efficient response and protect the user's privacy. Such multilevel models
reduce some redundancy and perform better than the single level models. The
proposed multilevel idea can be applied in various applications, such as
natural language processing, computer vision tasks, professional assistant,
business and healthcare.


---

## MorphPiece : Moving away from Statistical Language Representation

**first_author:** Haris Jabbar et al.

**Published Date:** 2023-07-14T10:35:04Z

**Link:** http://arxiv.org/pdf/2307.07262v1

**Abstract:**

  Tokenization is a critical part of modern NLP pipelines. However,
contemporary tokenizers for Large Language Models are based on statistical
analysis of text corpora, without much consideration to the linguistic
features. We propose a linguistically motivated tokenization scheme,
MorphPiece, which is based partly on morphological segmentation of the
underlying text. A GPT-style causal language model trained on this tokenizer
(called MorphGPT) shows superior convergence compared to the same architecture
trained on a standard BPE tokenizer. Specifically we get Language Modeling
performance comparable to a 6 times larger model. Additionally, we evaluate
MorphGPT on a variety of NLP tasks in supervised and unsupervised settings and
find superior performance across the board, compared to GPT-2 model.


---

## On the application of Large Language Models for language teaching and
  assessment technology

**first_author:** Andrew Caines et al.

**Published Date:** 2023-07-17T11:12:56Z

**Link:** http://arxiv.org/pdf/2307.08393v1

**Abstract:**

  The recent release of very large language models such as PaLM and GPT-4 has
made an unprecedented impact in the popular media and public consciousness,
giving rise to a mixture of excitement and fear as to their capabilities and
potential uses, and shining a light on natural language processing research
which had not previously received so much attention. The developments offer
great promise for education technology, and in this paper we look specifically
at the potential for incorporating large language models in AI-driven language
teaching and assessment systems. We consider several research areas and also
discuss the risks and ethical considerations surrounding generative AI in
education technology for language learners. Overall we find that larger
language models offer improvements over previous models in text generation,
opening up routes toward content generation which had not previously been
plausible. For text generation they must be prompted carefully and their
outputs may need to be reshaped before they are ready for use. For automated
grading and grammatical error correction, tasks whose progress is checked on
well-known benchmarks, early investigations indicate that large language models
on their own do not improve on state-of-the-art results according to standard
evaluation metrics. For grading it appears that linguistic features established
in the literature should still be used for best performance, and for error
correction it may be that the models can offer alternative feedback styles
which are not measured sensitively with existing methods. In all cases, there
is work to be done to experiment with the inclusion of large language models in
education technology for language learners, in order to properly understand and
report on their capacities and limitations, and to ensure that foreseeable
risks such as misinformation and harmful bias are mitigated.


---

