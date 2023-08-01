## Lost in Translation: Large Language Models in Non-English Content
  Analysis

**Published Date:** 2023-06-12T19:10:47Z

**Link:** http://arxiv.org/pdf/2306.07377v1

**Abstract:**

  In recent years, large language models (e.g., Open AI's GPT-4, Meta's LLaMa,
Google's PaLM) have become the dominant approach for building AI systems to
analyze and generate language online. However, the automated systems that
increasingly mediate our interactions online -- such as chatbots, content
moderation systems, and search engines -- are primarily designed for and work
far more effectively in English than in the world's other 7,000 languages.
Recently, researchers and technology companies have attempted to extend the
capabilities of large language models into languages other than English by
building what are called multilingual language models.
  In this paper, we explain how these multilingual language models work and
explore their capabilities and limits. Part I provides a simple technical
explanation of how large language models work, why there is a gap in available
data between English and other languages, and how multilingual language models
attempt to bridge that gap. Part II accounts for the challenges of doing
content analysis with large language models in general and multilingual
language models in particular. Part III offers recommendations for companies,
researchers, and policymakers to keep in mind when considering researching,
developing and deploying large and multilingual language models.


---

## Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages

**Published Date:** 2023-06-29T08:20:57Z

**Link:** http://arxiv.org/pdf/2306.16774v1

**Abstract:**

  Vision-Language Pre-training (VLP) has advanced the performance of many
vision-language tasks, such as image-text retrieval, visual entailment, and
visual reasoning. The pre-training mostly utilizes lexical databases and image
queries in English. Previous work has demonstrated that the pre-training in
English does not transfer well to other languages in a zero-shot setting.
However, multilingual pre-trained language models (MPLM) have excelled at a
variety of single-modal language tasks. In this paper, we propose a simple yet
efficient approach to adapt VLP to unseen languages using MPLM. We utilize a
cross-lingual contextualized token embeddings alignment approach to train text
encoders for non-English languages. Our approach does not require image input
and primarily uses machine translation, eliminating the need for target
language data. Our evaluation across three distinct tasks (image-text
retrieval, visual entailment, and natural language visual reasoning)
demonstrates that this approach outperforms the state-of-the-art multilingual
vision-language models without requiring large parallel corpora. Our code is
available at https://github.com/Yasminekaroui/CliCoTea.


---

## ChatGPT is not Enough: Enhancing Large Language Models with Knowledge
  Graphs for Fact-aware Language Modeling

**Published Date:** 2023-06-20T12:21:06Z

**Link:** http://arxiv.org/pdf/2306.11489v1

**Abstract:**

  Recently, ChatGPT, a representative large language model (LLM), has gained
considerable attention due to its powerful emergent abilities. Some researchers
suggest that LLMs could potentially replace structured knowledge bases like
knowledge graphs (KGs) and function as parameterized knowledge bases. However,
while LLMs are proficient at learning probabilistic language patterns based on
large corpus and engaging in conversations with humans, they, like previous
smaller pre-trained language models (PLMs), still have difficulty in recalling
facts while generating knowledge-grounded contents. To overcome these
limitations, researchers have proposed enhancing data-driven PLMs with
knowledge-based KGs to incorporate explicit factual knowledge into PLMs, thus
improving their performance to generate texts requiring factual knowledge and
providing more informed responses to user queries. This paper reviews the
studies on enhancing PLMs with KGs, detailing existing knowledge graph enhanced
pre-trained language models (KGPLMs) as well as their applications. Inspired by
existing studies on KGPLM, this paper proposes to enhance LLMs with KGs by
developing knowledge graph-enhanced large language models (KGLLMs). KGLLM
provides a solution to enhance LLMs' factual reasoning ability, opening up new
avenues for LLM research.


---

## Opportunities for Large Language Models and Discourse in Engineering
  Design

**Published Date:** 2023-06-15T14:46:44Z

**Link:** http://arxiv.org/pdf/2306.09169v1

**Abstract:**

  In recent years, large language models have achieved breakthroughs on a wide
range of benchmarks in natural language processing and continue to increase in
performance. Recently, the advances of large language models have raised
interest outside the natural language processing community and could have a
large impact on daily life. In this paper, we pose the question: How will large
language models and other foundation models shape the future product
development process? We provide the reader with an overview of the subject by
summarizing both recent advances in natural language processing and the use of
information technology in the engineering design process. We argue that
discourse should be regarded as the core of engineering design processes, and
therefore should be represented in a digital artifact. On this basis, we
describe how foundation models such as large language models could contribute
to the design discourse by automating parts thereof that involve creativity and
reasoning, and were previously reserved for humans. We describe how
simulations, experiments, topology optimizations, and other process steps can
be integrated into a machine-actionable, discourse-centric design process.
Finally, we outline the future research that will be necessary for the
implementation of the conceptualized framework.


---

## AutoScrum: Automating Project Planning Using Large Language Models

**Published Date:** 2023-06-05T19:16:37Z

**Link:** http://arxiv.org/pdf/2306.03197v1

**Abstract:**

  Recent advancements in the field of large language models have made it
possible to use language models for advanced reasoning. In this paper we
leverage this ability for designing complex project plans based only on knowing
the current state and the desired state. Two approaches are demonstrated - a
scrum based approach and a shortcut plan approach. The scrum based approach
executes an automated process of requirements gathering, user story mapping,
feature identification, task decomposition and finally generates questions and
search terms for seeking out domain specific information to assist with task
completion. The shortcut approach looks at most recent snapshot of the current
and desired state and generates the next most reasonable task to do in order to
get to the desired state as quickly as possible. In this paper we automate
everything using a novel concept of "Language Programs". These are programs
written in natural language designed to process input data through the language
model. Guidance language is used for all LLM programs. All demo source code for
this paper is available at https://github.com/autoscrum/autoscrum


---

## Prompting Large Language Models to Reformulate Queries for Moment
  Localization

**Published Date:** 2023-06-06T05:48:09Z

**Link:** http://arxiv.org/pdf/2306.03422v1

**Abstract:**

  The task of moment localization is to localize a temporal moment in an
untrimmed video for a given natural language query. Since untrimmed video
contains highly redundant contents, the quality of the query is crucial for
accurately localizing moments, i.e., the query should provide precise
information about the target moment so that the localization model can
understand what to look for in the videos. However, the natural language
queries in current datasets may not be easy to understand for existing models.
For example, the Ego4D dataset uses question sentences as the query to describe
relatively complex moments. While being natural and straightforward for humans,
understanding such question sentences are challenging for mainstream moment
localization models like 2D-TAN. Inspired by the recent success of large
language models, especially their ability of understanding and generating
complex natural language contents, in this extended abstract, we make early
attempts at reformulating the moment queries into a set of instructions using
large language models and making them more friendly to the localization models.


---

## Tracking public attitudes toward ChatGPT on Twitter using sentiment
  analysis and topic modeling

**Published Date:** 2023-06-22T15:10:18Z

**Link:** http://arxiv.org/pdf/2306.12951v1

**Abstract:**

  ChatGPT sets a new record with the fastest-growing user base, as a chatbot
powered by a large language model (LLM). While it demonstrates state-of-the-art
capabilities in a variety of language-generating tasks, it also raises
widespread public concerns regarding its societal impact. In this paper, we
utilize natural language processing approaches to investigate the public
attitudes towards ChatGPT by applying sentiment analysis and topic modeling
techniques to Twitter data. Our result shows that the overall sentiment is
largely neutral to positive, which also holds true across different occupation
groups. Among a wide range of topics mentioned in tweets, the most popular
topics are Artificial Intelligence, Search Engines, Education, Writing, and
Question Answering.


---

## Lost in Translation: Large Language Models in Non-English Content
  Analysis

**Published Date:** 2023-06-12T19:10:47Z

**Link:** http://arxiv.org/pdf/2306.07377v1

**Abstract:**

  In recent years, large language models (e.g., Open AI's GPT-4, Meta's LLaMa,
Google's PaLM) have become the dominant approach for building AI systems to
analyze and generate language online. However, the automated systems that
increasingly mediate our interactions online -- such as chatbots, content
moderation systems, and search engines -- are primarily designed for and work
far more effectively in English than in the world's other 7,000 languages.
Recently, researchers and technology companies have attempted to extend the
capabilities of large language models into languages other than English by
building what are called multilingual language models.
  In this paper, we explain how these multilingual language models work and
explore their capabilities and limits. Part I provides a simple technical
explanation of how large language models work, why there is a gap in available
data between English and other languages, and how multilingual language models
attempt to bridge that gap. Part II accounts for the challenges of doing
content analysis with large language models in general and multilingual
language models in particular. Part III offers recommendations for companies,
researchers, and policymakers to keep in mind when considering researching,
developing and deploying large and multilingual language models.


---

## Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages

**Published Date:** 2023-06-29T08:20:57Z

**Link:** http://arxiv.org/pdf/2306.16774v1

**Abstract:**

  Vision-Language Pre-training (VLP) has advanced the performance of many
vision-language tasks, such as image-text retrieval, visual entailment, and
visual reasoning. The pre-training mostly utilizes lexical databases and image
queries in English. Previous work has demonstrated that the pre-training in
English does not transfer well to other languages in a zero-shot setting.
However, multilingual pre-trained language models (MPLM) have excelled at a
variety of single-modal language tasks. In this paper, we propose a simple yet
efficient approach to adapt VLP to unseen languages using MPLM. We utilize a
cross-lingual contextualized token embeddings alignment approach to train text
encoders for non-English languages. Our approach does not require image input
and primarily uses machine translation, eliminating the need for target
language data. Our evaluation across three distinct tasks (image-text
retrieval, visual entailment, and natural language visual reasoning)
demonstrates that this approach outperforms the state-of-the-art multilingual
vision-language models without requiring large parallel corpora. Our code is
available at https://github.com/Yasminekaroui/CliCoTea.


---

## Prompting Large Language Models to Reformulate Queries for Moment
  Localization

**Published Date:** 2023-06-06T05:48:09Z

**Link:** http://arxiv.org/pdf/2306.03422v1

**Abstract:**

  The task of moment localization is to localize a temporal moment in an
untrimmed video for a given natural language query. Since untrimmed video
contains highly redundant contents, the quality of the query is crucial for
accurately localizing moments, i.e., the query should provide precise
information about the target moment so that the localization model can
understand what to look for in the videos. However, the natural language
queries in current datasets may not be easy to understand for existing models.
For example, the Ego4D dataset uses question sentences as the query to describe
relatively complex moments. While being natural and straightforward for humans,
understanding such question sentences are challenging for mainstream moment
localization models like 2D-TAN. Inspired by the recent success of large
language models, especially their ability of understanding and generating
complex natural language contents, in this extended abstract, we make early
attempts at reformulating the moment queries into a set of instructions using
large language models and making them more friendly to the localization models.


---

## Using Large Language Models to Provide Explanatory Feedback to Human
  Tutors

**Published Date:** 2023-06-27T14:19:12Z

**Link:** http://arxiv.org/pdf/2306.15498v1

**Abstract:**

  Research demonstrates learners engaging in the process of producing
explanations to support their reasoning, can have a positive impact on
learning. However, providing learners real-time explanatory feedback often
presents challenges related to classification accuracy, particularly in
domain-specific environments, containing situationally complex and nuanced
responses. We present two approaches for supplying tutors real-time feedback
within an online lesson on how to give students effective praise. This
work-in-progress demonstrates considerable accuracy in binary classification
for corrective feedback of effective, or effort-based (F1 score = 0.811), and
ineffective, or outcome-based (F1 score = 0.350), praise responses. More
notably, we introduce progress towards an enhanced approach of providing
explanatory feedback using large language model-facilitated named entity
recognition, which can provide tutors feedback, not only while engaging in
lessons, but can potentially suggest real-time tutor moves. Future work
involves leveraging large language models for data augmentation to improve
accuracy, while also developing an explanatory feedback interface.


---

## Lost in Translation: Large Language Models in Non-English Content
  Analysis

**first_author:** Gabriel Nicholas et al.

**Published Date:** 2023-06-12T19:10:47Z

**Link:** http://arxiv.org/pdf/2306.07377v1

**Abstract:**

  In recent years, large language models (e.g., Open AI's GPT-4, Meta's LLaMa,
Google's PaLM) have become the dominant approach for building AI systems to
analyze and generate language online. However, the automated systems that
increasingly mediate our interactions online -- such as chatbots, content
moderation systems, and search engines -- are primarily designed for and work
far more effectively in English than in the world's other 7,000 languages.
Recently, researchers and technology companies have attempted to extend the
capabilities of large language models into languages other than English by
building what are called multilingual language models.
  In this paper, we explain how these multilingual language models work and
explore their capabilities and limits. Part I provides a simple technical
explanation of how large language models work, why there is a gap in available
data between English and other languages, and how multilingual language models
attempt to bridge that gap. Part II accounts for the challenges of doing
content analysis with large language models in general and multilingual
language models in particular. Part III offers recommendations for companies,
researchers, and policymakers to keep in mind when considering researching,
developing and deploying large and multilingual language models.


---

## Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages

**first_author:** Yasmine Karoui et al.

**Published Date:** 2023-06-29T08:20:57Z

**Link:** http://arxiv.org/pdf/2306.16774v1

**Abstract:**

  Vision-Language Pre-training (VLP) has advanced the performance of many
vision-language tasks, such as image-text retrieval, visual entailment, and
visual reasoning. The pre-training mostly utilizes lexical databases and image
queries in English. Previous work has demonstrated that the pre-training in
English does not transfer well to other languages in a zero-shot setting.
However, multilingual pre-trained language models (MPLM) have excelled at a
variety of single-modal language tasks. In this paper, we propose a simple yet
efficient approach to adapt VLP to unseen languages using MPLM. We utilize a
cross-lingual contextualized token embeddings alignment approach to train text
encoders for non-English languages. Our approach does not require image input
and primarily uses machine translation, eliminating the need for target
language data. Our evaluation across three distinct tasks (image-text
retrieval, visual entailment, and natural language visual reasoning)
demonstrates that this approach outperforms the state-of-the-art multilingual
vision-language models without requiring large parallel corpora. Our code is
available at https://github.com/Yasminekaroui/CliCoTea.


---

