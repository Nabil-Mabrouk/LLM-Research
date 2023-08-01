## Beyond the limitations of any imaginable mechanism: large language
  models and psycholinguistics

**Published Date:** 2023-02-28T20:49:38Z

**Link:** http://arxiv.org/pdf/2303.00077v1

**Abstract:**

  Large language models are not detailed models of human linguistic processing.
They are, however, extremely successful at their primary task: providing a
model for language. For this reason and because there are no animal models for
language, large language models are important in psycholinguistics: they are
useful as a practical tool, as an illustrative comparative, and
philosophically, as a basis for recasting the relationship between language and
thought.


---

## LEALLA: Learning Lightweight Language-agnostic Sentence Embeddings with
  Knowledge Distillation

**Published Date:** 2023-02-16T16:05:34Z

**Link:** http://arxiv.org/pdf/2302.08387v1

**Abstract:**

  Large-scale language-agnostic sentence embedding models such as LaBSE (Feng
et al., 2022) obtain state-of-the-art performance for parallel sentence
alignment. However, these large-scale models can suffer from inference speed
and computation overhead. This study systematically explores learning
language-agnostic sentence embeddings with lightweight models. We demonstrate
that a thin-deep encoder can construct robust low-dimensional sentence
embeddings for 109 languages. With our proposed distillation methods, we
achieve further improvements by incorporating knowledge from a teacher model.
Empirical results on Tatoeba, United Nations, and BUCC show the effectiveness
of our lightweight models. We release our lightweight language-agnostic
sentence embedding models LEALLA on TensorFlow Hub.


---

## Measuring The Impact Of Programming Language Distribution

**Published Date:** 2023-02-03T19:47:22Z

**Link:** http://arxiv.org/pdf/2302.01973v3

**Abstract:**

  Current benchmarks for evaluating neural code models focus on only a small
subset of programming languages, excluding many popular languages such as Go or
Rust. To ameliorate this issue, we present the BabelCode framework for
execution-based evaluation of any benchmark in any language. BabelCode enables
new investigations into the qualitative performance of models' memory, runtime,
and individual test case results. Additionally, we present a new code
translation dataset called Translating Python Programming Puzzles (TP3) from
the Python Programming Puzzles (Schuster et al. 2021) benchmark that involves
translating expert-level python functions to any language. With both BabelCode
and the TP3 benchmark, we investigate if balancing the distributions of 14
languages in a training dataset improves a large language model's performance
on low-resource languages. Training a model on a balanced corpus results in, on
average, 12.34% higher $pass@k$ across all tasks and languages compared to the
baseline. We find that this strategy achieves 66.48% better $pass@k$ on
low-resource languages at the cost of only a 12.94% decrease to high-resource
languages. In our three translation tasks, this strategy yields, on average,
30.77% better low-resource $pass@k$ while having 19.58% worse high-resource
$pass@k$.


---

## Improving Massively Multilingual ASR With Auxiliary CTC Objectives

**Published Date:** 2023-02-24T18:59:51Z

**Link:** http://arxiv.org/pdf/2302.12829v2

**Abstract:**

  Multilingual Automatic Speech Recognition (ASR) models have extended the
usability of speech technologies to a wide variety of languages. With how many
languages these models have to handle, however, a key to understanding their
imbalanced performance across different languages is to examine if the model
actually knows which language it should transcribe. In this paper, we introduce
our work on improving performance on FLEURS, a 102-language open ASR benchmark,
by conditioning the entire model on language identity (LID). We investigate
techniques inspired from recent Connectionist Temporal Classification (CTC)
studies to help the model handle the large number of languages, conditioning on
the LID predictions of auxiliary tasks. Our experimental results demonstrate
the effectiveness of our technique over standard CTC/Attention-based hybrid
models. Furthermore, our state-of-the-art systems using self-supervised models
with the Conformer architecture improve over the results of prior work on
FLEURS by a relative 28.4% CER. Trained models and reproducible recipes are
available at https://github.com/espnet/espnet/tree/master/egs2/fleurs/asr1 .


---

## Chat2VIS: Generating Data Visualisations via Natural Language using
  ChatGPT, Codex and GPT-3 Large Language Models

**Published Date:** 2023-02-04T05:19:31Z

**Link:** http://arxiv.org/pdf/2302.02094v2

**Abstract:**

  The field of data visualisation has long aimed to devise solutions for
generating visualisations directly from natural language text. Research in
Natural Language Interfaces (NLIs) has contributed towards the development of
such techniques. However, the implementation of workable NLIs has always been
challenging due to the inherent ambiguity of natural language, as well as in
consequence of unclear and poorly written user queries which pose problems for
existing language models in discerning user intent. Instead of pursuing the
usual path of developing new iterations of language models, this study uniquely
proposes leveraging the advancements in pre-trained large language models
(LLMs) such as ChatGPT and GPT-3 to convert free-form natural language directly
into code for appropriate visualisations. This paper presents a novel system,
Chat2VIS, which takes advantage of the capabilities of LLMs and demonstrates
how, with effective prompt engineering, the complex problem of language
understanding can be solved more efficiently, resulting in simpler and more
accurate end-to-end solutions than prior approaches. Chat2VIS shows that LLMs
together with the proposed prompts offer a reliable approach to rendering
visualisations from natural language queries, even when queries are highly
misspecified and underspecified. This solution also presents a significant
reduction in costs for the development of NLI systems, while attaining greater
visualisation inference abilities compared to traditional NLP approaches that
use hand-crafted grammar rules and tailored models. This study also presents
how LLM prompts can be constructed in a way that preserves data security and
privacy while being generalisable to different datasets. This work compares the
performance of GPT-3, Codex and ChatGPT across a number of case studies and
contrasts the performances with prior studies.


---

## BBT-Fin: Comprehensive Construction of Chinese Financial Domain
  Pre-trained Language Model, Corpus and Benchmark

**Published Date:** 2023-02-18T22:20:37Z

**Link:** http://arxiv.org/pdf/2302.09432v2

**Abstract:**

  To advance Chinese financial natural language processing (NLP), we introduce
BBT-FinT5, a new Chinese financial pre-training language model based on the T5
model. To support this effort, we have built BBT-FinCorpus, a large-scale
financial corpus with approximately 300GB of raw text from four different
sources. In general domain NLP, comprehensive benchmarks like GLUE and
SuperGLUE have driven significant advancements in language model pre-training
by enabling head-to-head comparisons among models. Drawing inspiration from
these benchmarks, we propose BBT-CFLEB, a Chinese Financial Language
understanding and generation Evaluation Benchmark, which includes six datasets
covering both understanding and generation tasks. Our aim is to facilitate
research in the development of NLP within the Chinese financial domain. Our
model, corpus and benchmark are released at
https://github.com/ssymmetry/BBT-FinCUGE-Applications. Our work belongs to the
Big Bang Transformer (BBT), a large-scale pre-trained language model project.


---

## $k$NN-Adapter: Efficient Domain Adaptation for Black-Box Language Models

**Published Date:** 2023-02-21T18:54:21Z

**Link:** http://arxiv.org/pdf/2302.10879v1

**Abstract:**

  Fine-tuning a language model on a new domain is standard practice for domain
adaptation. However, it can be infeasible when it comes to modern large-scale
language models such as GPT-3, which can only be accessed through APIs, making
it difficult to access the internal parameters of the model. In this paper, we
propose $k$NN-Adapter, a method to effectively adapt these black-box large
language models (LLMs) to a new domain. The $k$NN-Adapter builds on top of the
retrieval-augmented language model, and adaptively learns to interpolate the
output of the language model with retrieval results from a datastore consisting
of the target domain data. Our experiments on four different domains
demonstrate that $k$NN-Adapter significantly improves perplexity, and works
particularly well in settings with limited access to LLMs. Additionally, we
show that $k$NN-Adapter is more effective than fine-tuning when the amount of
training data is limited. We also release a dataset to encourage further study.


---

## ChatGPT and Other Large Language Models as Evolutionary Engines for
  Online Interactive Collaborative Game Design

**Published Date:** 2023-02-09T15:44:43Z

**Link:** http://arxiv.org/pdf/2303.02155v2

**Abstract:**

  Large language models (LLMs) have taken the scientific world by storm,
changing the landscape of natural language processing and human-computer
interaction. These powerful tools can answer complex questions and,
surprisingly, perform challenging creative tasks (e.g., generate code and
applications to solve problems, write stories, pieces of music, etc.). In this
paper, we present a collaborative game design framework that combines
interactive evolution and large language models to simulate the typical human
design process. We use the former to exploit users' feedback for selecting the
most promising ideas and large language models for a very complex creative task
- the recombination and variation of ideas. In our framework, the process
starts with a brief and a set of candidate designs, either generated using a
language model or proposed by the users. Next, users collaborate on the design
process by providing feedback to an interactive genetic algorithm that selects,
recombines, and mutates the most promising designs. We evaluated our framework
on three game design tasks with human designers who collaborated remotely.


---

## Language Quantized AutoEncoders: Towards Unsupervised Text-Image
  Alignment

**Published Date:** 2023-02-02T06:38:44Z

**Link:** http://arxiv.org/pdf/2302.00902v2

**Abstract:**

  Recent progress in scaling up large language models has shown impressive
capabilities in performing few-shot learning across a wide range of text-based
tasks. However, a key limitation is that these language models fundamentally
lack visual perception - a crucial attribute needed to extend these models to
be able to interact with the real world and solve vision tasks, such as in
visual-question answering and robotics. Prior works have largely connected
image to text through pretraining and/or fine-tuning on curated image-text
datasets, which can be a costly and expensive process. In order to resolve this
limitation, we propose a simple yet effective approach called
Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to
align text-image data in an unsupervised manner by leveraging pretrained
language models (e.g., BERT, RoBERTa). Our main idea is to encode image as
sequences of text tokens by directly quantizing image embeddings using a
pretrained language codebook. We then apply random masking followed by a BERT
model, and have the decoder reconstruct the original image from BERT predicted
text token embeddings. By doing so, LQAE learns to represent similar images
with similar clusters of text tokens, thereby aligning these two modalities
without the use of aligned text-image pairs. This enables few-shot image
classification with large language models (e.g., GPT-3) as well as linear
classification of images based on BERT text features. To the best of our
knowledge, our work is the first work that uses unaligned images for multimodal
tasks by leveraging the power of pretrained language models.


---

## Massively Multilingual Shallow Fusion with Large Language Models

**Published Date:** 2023-02-17T14:46:38Z

**Link:** http://arxiv.org/pdf/2302.08917v1

**Abstract:**

  While large language models (LLM) have made impressive progress in natural
language processing, it remains unclear how to utilize them in improving
automatic speech recognition (ASR). In this work, we propose to train a single
multilingual language model (LM) for shallow fusion in multiple languages. We
push the limits of the multilingual LM to cover up to 84 languages by scaling
up using a mixture-of-experts LLM, i.e., generalist language model (GLaM). When
the number of experts increases, GLaM dynamically selects only two at each
decoding step to keep the inference computation roughly constant. We then apply
GLaM to a multilingual shallow fusion task based on a state-of-the-art
end-to-end model. Compared to a dense LM of similar computation during
inference, GLaM reduces the WER of an English long-tail test set by 4.4%
relative. In a multilingual shallow fusion task, GLaM improves 41 out of 50
languages with an average relative WER reduction of 3.85%, and a maximum
reduction of 10%. Compared to the baseline model, GLaM achieves an average WER
reduction of 5.53% over 43 languages.


---

## Training-free Lexical Backdoor Attacks on Language Models

**Published Date:** 2023-02-08T15:18:51Z

**Link:** http://arxiv.org/pdf/2302.04116v1

**Abstract:**

  Large-scale language models have achieved tremendous success across various
natural language processing (NLP) applications. Nevertheless, language models
are vulnerable to backdoor attacks, which inject stealthy triggers into models
for steering them to undesirable behaviors. Most existing backdoor attacks,
such as data poisoning, require further (re)training or fine-tuning language
models to learn the intended backdoor patterns. The additional training process
however diminishes the stealthiness of the attacks, as training a language
model usually requires long optimization time, a massive amount of data, and
considerable modifications to the model parameters. In this work, we propose
Training-Free Lexical Backdoor Attack (TFLexAttack) as the first training-free
backdoor attack on language models. Our attack is achieved by injecting lexical
triggers into the tokenizer of a language model via manipulating its embedding
dictionary using carefully designed rules. These rules are explainable to human
developers which inspires attacks from a wider range of hackers. The sparse
manipulation of the dictionary also habilitates the stealthiness of our attack.
We conduct extensive experiments on three dominant NLP tasks based on nine
language models to demonstrate the effectiveness and universality of our
attack. The code of this work is available at
https://github.com/Jinxhy/TFLexAttack.


---

## Beyond the limitations of any imaginable mechanism: large language
  models and psycholinguistics

**Published Date:** 2023-02-28T20:49:38Z

**Link:** http://arxiv.org/pdf/2303.00077v1

**Abstract:**

  Large language models are not detailed models of human linguistic processing.
They are, however, extremely successful at their primary task: providing a
model for language. For this reason and because there are no animal models for
language, large language models are important in psycholinguistics: they are
useful as a practical tool, as an illustrative comparative, and
philosophically, as a basis for recasting the relationship between language and
thought.


---

## Measuring The Impact Of Programming Language Distribution

**Published Date:** 2023-02-03T19:47:22Z

**Link:** http://arxiv.org/pdf/2302.01973v3

**Abstract:**

  Current benchmarks for evaluating neural code models focus on only a small
subset of programming languages, excluding many popular languages such as Go or
Rust. To ameliorate this issue, we present the BabelCode framework for
execution-based evaluation of any benchmark in any language. BabelCode enables
new investigations into the qualitative performance of models' memory, runtime,
and individual test case results. Additionally, we present a new code
translation dataset called Translating Python Programming Puzzles (TP3) from
the Python Programming Puzzles (Schuster et al. 2021) benchmark that involves
translating expert-level python functions to any language. With both BabelCode
and the TP3 benchmark, we investigate if balancing the distributions of 14
languages in a training dataset improves a large language model's performance
on low-resource languages. Training a model on a balanced corpus results in, on
average, 12.34% higher $pass@k$ across all tasks and languages compared to the
baseline. We find that this strategy achieves 66.48% better $pass@k$ on
low-resource languages at the cost of only a 12.94% decrease to high-resource
languages. In our three translation tasks, this strategy yields, on average,
30.77% better low-resource $pass@k$ while having 19.58% worse high-resource
$pass@k$.


---

## Language Quantized AutoEncoders: Towards Unsupervised Text-Image
  Alignment

**Published Date:** 2023-02-02T06:38:44Z

**Link:** http://arxiv.org/pdf/2302.00902v2

**Abstract:**

  Recent progress in scaling up large language models has shown impressive
capabilities in performing few-shot learning across a wide range of text-based
tasks. However, a key limitation is that these language models fundamentally
lack visual perception - a crucial attribute needed to extend these models to
be able to interact with the real world and solve vision tasks, such as in
visual-question answering and robotics. Prior works have largely connected
image to text through pretraining and/or fine-tuning on curated image-text
datasets, which can be a costly and expensive process. In order to resolve this
limitation, we propose a simple yet effective approach called
Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to
align text-image data in an unsupervised manner by leveraging pretrained
language models (e.g., BERT, RoBERTa). Our main idea is to encode image as
sequences of text tokens by directly quantizing image embeddings using a
pretrained language codebook. We then apply random masking followed by a BERT
model, and have the decoder reconstruct the original image from BERT predicted
text token embeddings. By doing so, LQAE learns to represent similar images
with similar clusters of text tokens, thereby aligning these two modalities
without the use of aligned text-image pairs. This enables few-shot image
classification with large language models (e.g., GPT-3) as well as linear
classification of images based on BERT text features. To the best of our
knowledge, our work is the first work that uses unaligned images for multimodal
tasks by leveraging the power of pretrained language models.


---

## The Wisdom of Hindsight Makes Language Models Better Instruction
  Followers

**Published Date:** 2023-02-10T12:16:38Z

**Link:** http://arxiv.org/pdf/2302.05206v1

**Abstract:**

  Reinforcement learning has seen wide success in finetuning large language
models to better align with instructions via human feedback. The so-called
algorithm, Reinforcement Learning with Human Feedback (RLHF) demonstrates
impressive performance on the GPT series models. However, the underlying
Reinforcement Learning (RL) algorithm is complex and requires an additional
training pipeline for reward and value networks. In this paper, we consider an
alternative approach: converting feedback to instruction by relabeling the
original one and training the model for better alignment in a supervised
manner. Such an algorithm doesn't require any additional parameters except for
the original language model and maximally reuses the pretraining pipeline. To
achieve this, we formulate instruction alignment problem for language models as
a goal-reaching problem in decision making. We propose Hindsight Instruction
Relabeling (HIR), a novel algorithm for aligning language models with
instructions. The resulting two-stage algorithm shed light to a family of
reward-free approaches that utilize the hindsightly relabeled instructions
based on feedback. We evaluate the performance of HIR extensively on 12
challenging BigBench reasoning tasks and show that HIR outperforms the baseline
algorithms and is comparable to or even surpasses supervised finetuning.


---

## LaMPP: Language Models as Probabilistic Priors for Perception and Action

**Published Date:** 2023-02-03T15:14:04Z

**Link:** http://arxiv.org/pdf/2302.02801v1

**Abstract:**

  Language models trained on large text corpora encode rich distributional
information about real-world environments and action sequences. This
information plays a crucial role in current approaches to language processing
tasks like question answering and instruction generation. We describe how to
leverage language models for *non-linguistic* perception and control tasks. Our
approach casts labeling and decision-making as inference in probabilistic
graphical models in which language models parameterize prior distributions over
labels, decisions and parameters, making it possible to integrate uncertain
observations and incomplete background knowledge in a principled way. Applied
to semantic segmentation, household navigation, and activity recognition tasks,
this approach improves predictions on rare, out-of-distribution, and
structurally novel inputs.


---

## Meeting the Needs of Low-Resource Languages: The Value of Automatic
  Alignments via Pretrained Models

**Published Date:** 2023-02-15T19:06:17Z

**Link:** http://arxiv.org/pdf/2302.07912v1

**Abstract:**

  Large multilingual models have inspired a new class of word alignment
methods, which work well for the model's pretraining languages. However, the
languages most in need of automatic alignment are low-resource and, thus, not
typically included in the pretraining data. In this work, we ask: How do modern
aligners perform on unseen languages, and are they better than traditional
methods? We contribute gold-standard alignments for Bribri--Spanish,
Guarani--Spanish, Quechua--Spanish, and Shipibo-Konibo--Spanish. With these, we
evaluate state-of-the-art aligners with and without model adaptation to the
target language. Finally, we also evaluate the resulting alignments
extrinsically through two downstream tasks: named entity recognition and
part-of-speech tagging. We find that although transformer-based methods
generally outperform traditional models, the two classes of approach remain
competitive with each other.


---

## Creating a Large Language Model of a Philosopher

**Published Date:** 2023-02-02T01:10:26Z

**Link:** http://arxiv.org/pdf/2302.01339v2

**Abstract:**

  Can large language models be trained to produce philosophical texts that are
difficult to distinguish from texts produced by human philosophers? To address
this question, we fine-tuned OpenAI's GPT-3 with the works of philosopher
Daniel C. Dennett as additional training data. To explore the Dennett model, we
asked the real Dennett ten philosophical questions and then posed the same
questions to the language model, collecting four responses for each question
without cherry-picking. We recruited 425 participants to distinguish Dennett's
answer from the four machine-generated answers. Experts on Dennett's work (N =
25) succeeded 51% of the time, above the chance rate of 20% but short of our
hypothesized rate of 80% correct. For two of the ten questions, the language
model produced at least one answer that experts selected more frequently than
Dennett's own answer. Philosophy blog readers (N = 302) performed similarly to
the experts, while ordinary research participants (N = 98) were near chance
distinguishing GPT-3's responses from those of an "actual human philosopher".


---

## Beyond the limitations of any imaginable mechanism: large language
  models and psycholinguistics

**first_author:** Conor Houghton et al.

**Published Date:** 2023-02-28T20:49:38Z

**Link:** http://arxiv.org/pdf/2303.00077v1

**Abstract:**

  Large language models are not detailed models of human linguistic processing.
They are, however, extremely successful at their primary task: providing a
model for language. For this reason and because there are no animal models for
language, large language models are important in psycholinguistics: they are
useful as a practical tool, as an illustrative comparative, and
philosophically, as a basis for recasting the relationship between language and
thought.


---

## Language Quantized AutoEncoders: Towards Unsupervised Text-Image
  Alignment

**first_author:** Hao Liu et al.

**Published Date:** 2023-02-02T06:38:44Z

**Link:** http://arxiv.org/pdf/2302.00902v2

**Abstract:**

  Recent progress in scaling up large language models has shown impressive
capabilities in performing few-shot learning across a wide range of text-based
tasks. However, a key limitation is that these language models fundamentally
lack visual perception - a crucial attribute needed to extend these models to
be able to interact with the real world and solve vision tasks, such as in
visual-question answering and robotics. Prior works have largely connected
image to text through pretraining and/or fine-tuning on curated image-text
datasets, which can be a costly and expensive process. In order to resolve this
limitation, we propose a simple yet effective approach called
Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to
align text-image data in an unsupervised manner by leveraging pretrained
language models (e.g., BERT, RoBERTa). Our main idea is to encode image as
sequences of text tokens by directly quantizing image embeddings using a
pretrained language codebook. We then apply random masking followed by a BERT
model, and have the decoder reconstruct the original image from BERT predicted
text token embeddings. By doing so, LQAE learns to represent similar images
with similar clusters of text tokens, thereby aligning these two modalities
without the use of aligned text-image pairs. This enables few-shot image
classification with large language models (e.g., GPT-3) as well as linear
classification of images based on BERT text features. To the best of our
knowledge, our work is the first work that uses unaligned images for multimodal
tasks by leveraging the power of pretrained language models.


---

