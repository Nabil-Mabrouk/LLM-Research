## Retrieval-based Knowledge Augmented Vision Language Pre-training

**Published Date:** 2023-04-27T02:23:47Z

**Link:** http://arxiv.org/pdf/2304.13923v1

**Abstract:**

  With recent progress in large-scale vision and language representation
learning, Vision Language Pretraining (VLP) models have achieved promising
improvements on various multi-modal downstream tasks. Albeit powerful, these
pre-training models still do not take advantage of world knowledge, which is
implicit in multi-modal data but comprises abundant and complementary
information. In this work, we propose a REtrieval-based knowledge Augmented
Vision Language Pre-training model (REAVL), which retrieves world knowledge
from knowledge graphs (KGs) and incorporates them in vision-language
pre-training. REAVL has two core components: a knowledge retriever that
retrieves knowledge given multi-modal data, and a knowledge-augmented model
that fuses multi-modal data and knowledge. By novelly unifying four
knowledge-aware self-supervised tasks, REAVL promotes the mutual integration of
multi-modal data and knowledge by fusing explicit knowledge with
vision-language pairs for masked multi-modal data modeling and KG relational
reasoning. Empirical experiments show that REAVL achieves new state-of-the-art
performance uniformly on knowledge-based vision-language understanding and
multimodal entity linking tasks, and competitive results on general
vision-language tasks while only using 0.2% pre-training data of the best
models.


---

## A Latent Space Theory for Emergent Abilities in Large Language Models

**Published Date:** 2023-04-19T20:45:01Z

**Link:** http://arxiv.org/pdf/2304.09960v2

**Abstract:**

  Languages are not created randomly but rather to communicate information.
There is a strong association between languages and their underlying meanings,
resulting in a sparse joint distribution that is heavily peaked according to
their correlations. Moreover, these peak values happen to match with the
marginal distribution of languages due to the sparsity. With the advent of LLMs
trained on big data and large models, we can now precisely assess the marginal
distribution of languages, providing a convenient means of exploring the sparse
structures in the joint distribution for effective inferences. In this paper,
we categorize languages as either unambiguous or {\epsilon}-ambiguous and
present quantitative results to demonstrate that the emergent abilities of
LLMs, such as language understanding, in-context learning, chain-of-thought
prompting, and effective instruction fine-tuning, can all be attributed to
Bayesian inference on the sparse joint distribution of languages.


---

## Prompt Engineering and Calibration for Zero-Shot Commonsense Reasoning

**Published Date:** 2023-04-14T07:07:42Z

**Link:** http://arxiv.org/pdf/2304.06962v1

**Abstract:**

  Prompt engineering and calibration make large language models excel at
reasoning tasks, including multiple choice commonsense reasoning. From a
practical perspective, we investigate and evaluate these strategies on smaller
language models. Through experiments on five commonsense reasoning benchmarks,
we find that each strategy favors certain models, but their joint effects are
mostly negative.


---

## Low-resource Bilingual Dialect Lexicon Induction with Large Language
  Models

**Published Date:** 2023-04-19T20:20:41Z

**Link:** http://arxiv.org/pdf/2304.09957v1

**Abstract:**

  Bilingual word lexicons are crucial tools for multilingual natural language
understanding and machine translation tasks, as they facilitate the mapping of
words in one language to their synonyms in another language. To achieve this,
numerous papers have explored bilingual lexicon induction (BLI) in
high-resource scenarios, using a typical pipeline consisting of two
unsupervised steps: bitext mining and word alignment, both of which rely on
pre-trained large language models~(LLMs).
  In this paper, we present an analysis of the BLI pipeline for German and two
of its dialects, Bavarian and Alemannic. This setup poses several unique
challenges, including the scarcity of resources, the relatedness of the
languages, and the lack of standardization in the orthography of dialects. To
evaluate the BLI outputs, we analyze them with respect to word frequency and
pairwise edit distance. Additionally, we release two evaluation datasets
comprising 1,500 bilingual sentence pairs and 1,000 bilingual word pairs. They
were manually judged for their semantic similarity for each Bavarian-German and
Alemannic-German language pair.


---

## CodeKGC: Code Language Model for Generative Knowledge Graph Construction

**Published Date:** 2023-04-18T15:12:34Z

**Link:** http://arxiv.org/pdf/2304.09048v1

**Abstract:**

  Current generative knowledge graph construction approaches usually fail to
capture structural knowledge by simply flattening natural language into
serialized texts or a specification language. However, large generative
language model trained on structured data such as code has demonstrated
impressive capability in understanding natural language for structural
prediction and reasoning tasks. Intuitively, we address the task of generative
knowledge graph construction with code language model: given a code-format
natural language input, the target is to generate triples which can be
represented as code completion tasks. Specifically, we develop schema-aware
prompts that effectively utilize the semantic structure within the knowledge
graph. As code inherently possesses structure, such as class and function
definitions, it serves as a useful model for prior semantic structural
knowledge. Furthermore, we employ a rationale-enhanced generation method to
boost the performance. Rationales provide intermediate steps, thereby improving
knowledge extraction abilities. Experimental results indicate that the proposed
approach can obtain better performance on benchmark datasets compared with
baselines. Code and datasets are available in
https://github.com/zjunlp/DeepKE/tree/main/example/llm.


---

## ERRA: An Embodied Representation and Reasoning Architecture for
  Long-horizon Language-conditioned Manipulation Tasks

**Published Date:** 2023-04-05T06:50:22Z

**Link:** http://arxiv.org/pdf/2304.02251v1

**Abstract:**

  This letter introduces ERRA, an embodied learning architecture that enables
robots to jointly obtain three fundamental capabilities (reasoning, planning,
and interaction) for solving long-horizon language-conditioned manipulation
tasks. ERRA is based on tightly-coupled probabilistic inferences at two
granularity levels. Coarse-resolution inference is formulated as sequence
generation through a large language model, which infers action language from
natural language instruction and environment state. The robot then zooms to the
fine-resolution inference part to perform the concrete action corresponding to
the action language. Fine-resolution inference is constructed as a Markov
decision process, which takes action language and environmental sensing as
observations and outputs the action. The results of action execution in
environments provide feedback for subsequent coarse-resolution reasoning. Such
coarse-to-fine inference allows the robot to decompose and achieve long-horizon
tasks interactively. In extensive experiments, we show that ERRA can complete
various long-horizon manipulation tasks specified by abstract language
instructions. We also demonstrate successful generalization to the novel but
similar natural language instructions.


---

## LaMP: When Large Language Models Meet Personalization

**Published Date:** 2023-04-22T13:42:04Z

**Link:** http://arxiv.org/pdf/2304.11406v2

**Abstract:**

  This paper highlights the importance of personalization in the current state
of natural language understanding and generation and introduces the LaMP
benchmark -- a novel benchmark for training and evaluating language models for
producing personalized outputs. LaMP offers a comprehensive evaluation
framework with diverse language tasks and multiple entries for each user
profile. It consists of seven personalized tasks, spanning three classification
and four text generation tasks. We also propose a retrieval augmentation
approach that retrieves personalized items from user profiles to construct
personalized prompts for large language models. Our baseline zero-shot and
fine-tuned model results indicate that LMs utilizing profile augmentation
outperform their counterparts that do not factor in profile information.


---

## Low-resource Bilingual Dialect Lexicon Induction with Large Language
  Models

**first_author:** Ekaterina Artemova et al.

**Published Date:** 2023-04-19T20:20:41Z

**Link:** http://arxiv.org/pdf/2304.09957v1

**Abstract:**

  Bilingual word lexicons are crucial tools for multilingual natural language
understanding and machine translation tasks, as they facilitate the mapping of
words in one language to their synonyms in another language. To achieve this,
numerous papers have explored bilingual lexicon induction (BLI) in
high-resource scenarios, using a typical pipeline consisting of two
unsupervised steps: bitext mining and word alignment, both of which rely on
pre-trained large language models~(LLMs).
  In this paper, we present an analysis of the BLI pipeline for German and two
of its dialects, Bavarian and Alemannic. This setup poses several unique
challenges, including the scarcity of resources, the relatedness of the
languages, and the lack of standardization in the orthography of dialects. To
evaluate the BLI outputs, we analyze them with respect to word frequency and
pairwise edit distance. Additionally, we release two evaluation datasets
comprising 1,500 bilingual sentence pairs and 1,000 bilingual word pairs. They
were manually judged for their semantic similarity for each Bavarian-German and
Alemannic-German language pair.


---

## CodeKGC: Code Language Model for Generative Knowledge Graph Construction

**first_author:** Zhen Bi et al.

**Published Date:** 2023-04-18T15:12:34Z

**Link:** http://arxiv.org/pdf/2304.09048v1

**Abstract:**

  Current generative knowledge graph construction approaches usually fail to
capture structural knowledge by simply flattening natural language into
serialized texts or a specification language. However, large generative
language model trained on structured data such as code has demonstrated
impressive capability in understanding natural language for structural
prediction and reasoning tasks. Intuitively, we address the task of generative
knowledge graph construction with code language model: given a code-format
natural language input, the target is to generate triples which can be
represented as code completion tasks. Specifically, we develop schema-aware
prompts that effectively utilize the semantic structure within the knowledge
graph. As code inherently possesses structure, such as class and function
definitions, it serves as a useful model for prior semantic structural
knowledge. Furthermore, we employ a rationale-enhanced generation method to
boost the performance. Rationales provide intermediate steps, thereby improving
knowledge extraction abilities. Experimental results indicate that the proposed
approach can obtain better performance on benchmark datasets compared with
baselines. Code and datasets are available in
https://github.com/zjunlp/DeepKE/tree/main/example/llm.


---

