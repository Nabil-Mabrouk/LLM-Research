## Adapting Pre-trained Language Models to African Languages via
  Multilingual Adaptive Fine-Tuning

**Published Date:** 2022-04-13T16:13:49Z

**Link:** http://arxiv.org/pdf/2204.06487v3

**Abstract:**

  Multilingual pre-trained language models (PLMs) have demonstrated impressive
performance on several downstream tasks for both high-resourced and
low-resourced languages. However, there is still a large performance drop for
languages unseen during pre-training, especially African languages. One of the
most effective approaches to adapt to a new language is \textit{language
adaptive fine-tuning} (LAFT) -- fine-tuning a multilingual PLM on monolingual
texts of a language using the pre-training objective. However, adapting to a
target language individually takes a large disk space and limits the
cross-lingual transfer abilities of the resulting models because they have been
specialized for a single language. In this paper, we perform
\textit{multilingual adaptive fine-tuning} on 17 most-resourced African
languages and three other high-resource languages widely spoken on the African
continent to encourage cross-lingual transfer learning. To further specialize
the multilingual PLM, we removed vocabulary tokens from the embedding layer
that corresponds to non-African writing scripts before MAFT, thus reducing the
model size by around 50%. Our evaluation on two multilingual PLMs (AfriBERTa
and XLM-R) and three NLP tasks (NER, news topic classification, and sentiment
classification) shows that our approach is competitive to applying LAFT on
individual languages while requiring significantly less disk space.
Additionally, we show that our adapted PLM also improves the zero-shot
cross-lingual transfer abilities of parameter efficient fine-tuning methods.


---

## Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in
  Natural Language Understanding

**Published Date:** 2022-04-13T10:32:03Z

**Link:** http://arxiv.org/pdf/2204.06283v2

**Abstract:**

  In the age of large transformer language models, linguistic evaluation play
an important role in diagnosing models' abilities and limitations on natural
language understanding. However, current evaluation methods show some
significant shortcomings. In particular, they do not provide insight into how
well a language model captures distinct linguistic skills essential for
language understanding and reasoning. Thus they fail to effectively map out the
aspects of language understanding that remain challenging to existing models,
which makes it hard to discover potential limitations in models and datasets.
In this paper, we introduce Curriculum as a new format of NLI benchmark for
evaluation of broad-coverage linguistic phenomena. Curriculum contains a
collection of datasets that covers 36 types of major linguistic phenomena and
an evaluation procedure for diagnosing how well a language model captures
reasoning skills for distinct types of linguistic phenomena. We show that this
linguistic-phenomena-driven benchmark can serve as an effective tool for
diagnosing model behavior and verifying model learning quality. In addition,
our experiments provide insight into the limitation of existing benchmark
datasets and state-of-the-art models that may encourage future research on
re-designing datasets, model architectures, and learning objectives.


---

## Do Not Fire the Linguist: Grammatical Profiles Help Language Models
  Detect Semantic Change

**Published Date:** 2022-04-12T11:20:42Z

**Link:** http://arxiv.org/pdf/2204.05717v1

**Abstract:**

  Morphological and syntactic changes in word usage (as captured, e.g., by
grammatical profiles) have been shown to be good predictors of a word's meaning
change. In this work, we explore whether large pre-trained contextualised
language models, a common tool for lexical semantic change detection, are
sensitive to such morphosyntactic changes. To this end, we first compare the
performance of grammatical profiles against that of a multilingual neural
language model (XLM-R) on 10 datasets, covering 7 languages, and then combine
the two approaches in ensembles to assess their complementarity. Our results
show that ensembling grammatical profiles with XLM-R improves semantic change
detection performance for most datasets and languages. This indicates that
language models do not fully cover the fine-grained morphological and syntactic
signals that are explicitly represented in grammatical profiles.
  An interesting exception are the test sets where the time spans under
analysis are much longer than the time gap between them (for example,
century-long spans with a one-year gap between them). Morphosyntactic change is
slow so grammatical profiles do not detect in such cases. In contrast, language
models, thanks to their access to lexical information, are able to detect fast
topical changes.


---

## Training Language Models with Language Feedback

**Published Date:** 2022-04-29T15:06:58Z

**Link:** http://arxiv.org/pdf/2204.14146v4

**Abstract:**

  Pretrained language models often do not perform tasks in ways that are in
line with our preferences, e.g., generating offensive text or factually
incorrect summaries. Recent work approaches the above issue by learning from a
simple form of human evaluation: comparisons between pairs of model-generated
task outputs. Comparison feedback conveys limited information about human
preferences per human evaluation. Here, we propose to learn from natural
language feedback, which conveys more information per human evaluation. We
learn from language feedback on model outputs using a three-step learning
algorithm. First, we condition the language model on the initial output and
feedback to generate many refinements. Second, we choose the refinement with
the highest similarity to the feedback. Third, we finetune a language model to
maximize the likelihood of the chosen refinement given the input. In synthetic
experiments, we first evaluate whether language models accurately incorporate
feedback to produce refinements, finding that only large language models (175B
parameters) do so. Using only 100 samples of human-written feedback, our
learning algorithm finetunes a GPT-3 model to roughly human-level summarization
ability.


---

## Transformer-Based Language Models for Software Vulnerability Detection

**Published Date:** 2022-04-07T04:57:42Z

**Link:** http://arxiv.org/pdf/2204.03214v2

**Abstract:**

  The large transformer-based language models demonstrate excellent performance
in natural language processing. By considering the transferability of the
knowledge gained by these models in one domain to other related domains, and
the closeness of natural languages to high-level programming languages, such as
C/C++, this work studies how to leverage (large) transformer-based language
models in detecting software vulnerabilities and how good are these models for
vulnerability detection tasks. In this regard, firstly, a systematic (cohesive)
framework that details source code translation, model preparation, and
inference is presented. Then, an empirical analysis is performed with software
vulnerability datasets with C/C++ source codes having multiple vulnerabilities
corresponding to the library function call, pointer usage, array usage, and
arithmetic expression. Our empirical results demonstrate the good performance
of the language models in vulnerability detection. Moreover, these language
models have better performance metrics, such as F1-score, than the contemporary
models, namely bidirectional long short-term memory and bidirectional gated
recurrent unit. Experimenting with the language models is always challenging
due to the requirement of computing resources, platforms, libraries, and
dependencies. Thus, this paper also analyses the popular platforms to
efficiently fine-tune these models and present recommendations while choosing
the platforms.


---

## Por Qué Não Utiliser Alla Språk? Mixed Training with Gradient
  Optimization in Few-Shot Cross-Lingual Transfer

**Published Date:** 2022-04-29T04:05:02Z

**Link:** http://arxiv.org/pdf/2204.13869v1

**Abstract:**

  The current state-of-the-art for few-shot cross-lingual transfer learning
first trains on abundant labeled data in the source language and then
fine-tunes with a few examples on the target language, termed target-adapting.
Though this has been demonstrated to work on a variety of tasks, in this paper
we show some deficiencies of this approach and propose a one-step mixed
training method that trains on both source and target data with
\textit{stochastic gradient surgery}, a novel gradient-level optimization.
Unlike the previous studies that focus on one language at a time when
target-adapting, we use one model to handle all target languages simultaneously
to avoid excessively language-specific models. Moreover, we discuss the
unreality of utilizing large target development sets for model selection in
previous literature. We further show that our method is both development-free
for target languages, and is also able to escape from overfitting issues. We
conduct a large-scale experiment on 4 diverse NLP tasks across up to 48
languages. Our proposed method achieves state-of-the-art performance on all
tasks and outperforms target-adapting by a large margin, especially for
languages that are linguistically distant from the source language, e.g., 7.36%
F1 absolute gain on average for the NER task, up to 17.60% on Punjabi.


---

## Language Contamination Helps Explain the Cross-lingual Capabilities of
  English Pretrained Models

**Published Date:** 2022-04-17T23:56:54Z

**Link:** http://arxiv.org/pdf/2204.08110v4

**Abstract:**

  English pretrained language models, which make up the backbone of many modern
NLP systems, require huge amounts of unlabeled training data. These models are
generally presented as being trained only on English text but have been found
to transfer surprisingly well to other languages. We investigate this
phenomenon and find that common English pretraining corpora actually contain
significant amounts of non-English text: even when less than 1% of data is not
English (well within the error rate of strong language classifiers), this leads
to hundreds of millions of foreign language tokens in large-scale datasets. We
then demonstrate that even these small percentages of non-English data
facilitate cross-lingual transfer for models trained on them, with target
language performance strongly correlated to the amount of in-language data seen
during pretraining. In light of these findings, we argue that no model is truly
monolingual when pretrained at scale, which should be considered when
evaluating cross-lingual transfer.


---

## Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in
  Natural Language Understanding

**Published Date:** 2022-04-13T10:32:03Z

**Link:** http://arxiv.org/pdf/2204.06283v2

**Abstract:**

  In the age of large transformer language models, linguistic evaluation play
an important role in diagnosing models' abilities and limitations on natural
language understanding. However, current evaluation methods show some
significant shortcomings. In particular, they do not provide insight into how
well a language model captures distinct linguistic skills essential for
language understanding and reasoning. Thus they fail to effectively map out the
aspects of language understanding that remain challenging to existing models,
which makes it hard to discover potential limitations in models and datasets.
In this paper, we introduce Curriculum as a new format of NLI benchmark for
evaluation of broad-coverage linguistic phenomena. Curriculum contains a
collection of datasets that covers 36 types of major linguistic phenomena and
an evaluation procedure for diagnosing how well a language model captures
reasoning skills for distinct types of linguistic phenomena. We show that this
linguistic-phenomena-driven benchmark can serve as an effective tool for
diagnosing model behavior and verifying model learning quality. In addition,
our experiments provide insight into the limitation of existing benchmark
datasets and state-of-the-art models that may encourage future research on
re-designing datasets, model architectures, and learning objectives.


---

## Estimating the Personality of White-Box Language Models

**Published Date:** 2022-04-25T23:53:53Z

**Link:** http://arxiv.org/pdf/2204.12000v2

**Abstract:**

  Technology for open-ended language generation, a key application of
artificial intelligence, has advanced to a great extent in recent years.
Large-scale language models, which are trained on large corpora of text, are
being used in a wide range of applications everywhere, from virtual assistants
to conversational bots. While these language models output fluent text,
existing research shows that these models can and do capture human biases. Many
of these biases, especially those that could potentially cause harm, are being
well-investigated. On the other hand, studies that infer and change human
personality traits inherited by these models have been scarce or non-existent.
Our work seeks to address this gap by exploring the personality traits of
several large-scale language models designed for open-ended text generation and
the datasets used for training them. We build on the popular Big Five factors
and develop robust methods that quantify the personality traits of these models
and their underlying datasets. In particular, we trigger the models with a
questionnaire designed for personality assessment and subsequently classify the
text responses into quantifiable traits using a Zero-shot classifier. Our
estimation scheme sheds light on an important anthropomorphic element found in
such AI models and can help stakeholders decide how they should be applied as
well as how society could perceive them. Additionally, we examined approaches
to alter these personalities, adding to our understanding of how AI models can
be adapted to specific contexts.


---

## Post-Training Dialogue Summarization using Pseudo-Paraphrasing

**Published Date:** 2022-04-28T13:42:19Z

**Link:** http://arxiv.org/pdf/2204.13498v1

**Abstract:**

  Previous dialogue summarization techniques adapt large language models
pretrained on the narrative text by injecting dialogue-specific features into
the models. These features either require additional knowledge to recognize or
make the resulting models harder to tune. To bridge the format gap between
dialogues and narrative summaries in dialogue summarization tasks, we propose
to post-train pretrained language models (PLMs) to rephrase from dialogue to
narratives. After that, the model is fine-tuned for dialogue summarization as
usual. Comprehensive experiments show that our approach significantly improves
vanilla PLMs on dialogue summarization and outperforms other SOTA models by the
summary quality and implementation costs.


---

## You Don't Know My Favorite Color: Preventing Dialogue Representations
  from Revealing Speakers' Private Personas

**Published Date:** 2022-04-26T09:36:18Z

**Link:** http://arxiv.org/pdf/2205.10228v1

**Abstract:**

  Social chatbots, also known as chit-chat chatbots, evolve rapidly with large
pretrained language models. Despite the huge progress, privacy concerns have
arisen recently: training data of large language models can be extracted via
model inversion attacks. On the other hand, the datasets used for training
chatbots contain many private conversations between two individuals. In this
work, we further investigate the privacy leakage of the hidden states of
chatbots trained by language modeling which has not been well studied yet. We
show that speakers' personas can be inferred through a simple neural network
with high accuracy. To this end, we propose effective defense objectives to
protect persona leakage from hidden states. We conduct extensive experiments to
demonstrate that our proposed defense objectives can greatly reduce the attack
accuracy from 37.6% to 0.5%. Meanwhile, the proposed objectives preserve
language models' powerful generation ability.


---

## Breaking Character: Are Subwords Good Enough for MRLs After All?

**Published Date:** 2022-04-10T18:54:43Z

**Link:** http://arxiv.org/pdf/2204.04748v1

**Abstract:**

  Large pretrained language models (PLMs) typically tokenize the input string
into contiguous subwords before any pretraining or inference. However, previous
studies have claimed that this form of subword tokenization is inadequate for
processing morphologically-rich languages (MRLs). We revisit this hypothesis by
pretraining a BERT-style masked language model over character sequences instead
of word-pieces. We compare the resulting model, dubbed TavBERT, against
contemporary PLMs based on subwords for three highly complex and ambiguous MRLs
(Hebrew, Turkish, and Arabic), testing them on both morphological and semantic
tasks. Our results show, for all tested languages, that while TavBERT obtains
mild improvements on surface-level tasks \`a la POS tagging and full
morphological disambiguation, subword-based PLMs achieve significantly higher
performance on semantic tasks, such as named entity recognition and extractive
question answering. These results showcase and (re)confirm the potential of
subword tokenization as a reasonable modeling assumption for many languages,
including MRLs.


---

## Mining Logical Event Schemas From Pre-Trained Language Models

**Published Date:** 2022-04-12T16:41:18Z

**Link:** http://arxiv.org/pdf/2204.05939v1

**Abstract:**

  We present NESL (the Neuro-Episodic Schema Learner), an event schema learning
system that combines large language models, FrameNet parsing, a powerful
logical representation of language, and a set of simple behavioral schemas
meant to bootstrap the learning process. In lieu of a pre-made corpus of
stories, our dataset is a continuous feed of "situation samples" from a
pre-trained language model, which are then parsed into FrameNet frames, mapped
into simple behavioral schemas, and combined and generalized into complex,
hierarchical schemas for a variety of everyday scenarios. We show that careful
sampling from the language model can help emphasize stereotypical properties of
situations and de-emphasize irrelevant details, and that the resulting schemas
specify situations more comprehensively than those learned by other systems.


---

## Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in
  Natural Language Understanding

**first_author:** Zeming Chen et al.

**Published Date:** 2022-04-13T10:32:03Z

**Link:** http://arxiv.org/pdf/2204.06283v2

**Abstract:**

  In the age of large transformer language models, linguistic evaluation play
an important role in diagnosing models' abilities and limitations on natural
language understanding. However, current evaluation methods show some
significant shortcomings. In particular, they do not provide insight into how
well a language model captures distinct linguistic skills essential for
language understanding and reasoning. Thus they fail to effectively map out the
aspects of language understanding that remain challenging to existing models,
which makes it hard to discover potential limitations in models and datasets.
In this paper, we introduce Curriculum as a new format of NLI benchmark for
evaluation of broad-coverage linguistic phenomena. Curriculum contains a
collection of datasets that covers 36 types of major linguistic phenomena and
an evaluation procedure for diagnosing how well a language model captures
reasoning skills for distinct types of linguistic phenomena. We show that this
linguistic-phenomena-driven benchmark can serve as an effective tool for
diagnosing model behavior and verifying model learning quality. In addition,
our experiments provide insight into the limitation of existing benchmark
datasets and state-of-the-art models that may encourage future research on
re-designing datasets, model architectures, and learning objectives.


---

