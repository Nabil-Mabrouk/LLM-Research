## Connecting Neural Response measurements & Computational Models of
  language: a non-comprehensive guide

**Published Date:** 2022-03-10T11:24:54Z

**Link:** http://arxiv.org/pdf/2203.05300v1

**Abstract:**

  Understanding the neural basis of language comprehension in the brain has
been a long-standing goal of various scientific research programs. Recent
advances in language modelling and in neuroimaging methodology promise
potential improvements in both the investigation of language's neurobiology and
in the building of better and more human-like language models. This survey
traces a line from early research linking Event Related Potentials and
complexity measures derived from simple language models to contemporary studies
employing Artificial Neural Network models trained on large corpora in
combination with neural response recordings from multiple modalities using
naturalistic stimuli.


---

## Zero-Shot Dependency Parsing with Worst-Case Aware Automated Curriculum
  Learning

**Published Date:** 2022-03-16T11:33:20Z

**Link:** http://arxiv.org/pdf/2203.08555v1

**Abstract:**

  Large multilingual pretrained language models such as mBERT and XLM-RoBERTa
have been found to be surprisingly effective for cross-lingual transfer of
syntactic parsing models (Wu and Dredze 2019), but only between related
languages. However, source and training languages are rarely related, when
parsing truly low-resource languages. To close this gap, we adopt a method from
multi-task learning, which relies on automated curriculum learning, to
dynamically optimize for parsing performance on outlier languages. We show that
this approach is significantly better than uniform and size-proportional
sampling in the zero-shot setting.


---

## Reshaping Robot Trajectories Using Natural Language Commands: A Study of
  Multi-Modal Data Alignment Using Transformers

**Published Date:** 2022-03-25T01:36:56Z

**Link:** http://arxiv.org/pdf/2203.13411v1

**Abstract:**

  Natural language is the most intuitive medium for us to interact with other
people when expressing commands and instructions. However, using language is
seldom an easy task when humans need to express their intent towards robots,
since most of the current language interfaces require rigid templates with a
static set of action targets and commands. In this work, we provide a flexible
language-based interface for human-robot collaboration, which allows a user to
reshape existing trajectories for an autonomous agent. We take advantage of
recent advancements in the field of large language models (BERT and CLIP) to
encode the user command, and then combine these features with trajectory
information using multi-modal attention transformers. We train the model using
imitation learning over a dataset containing robot trajectories modified by
language commands, and treat the trajectory generation process as a sequence
prediction problem, analogously to how language generation architectures
operate. We evaluate the system in multiple simulated trajectory scenarios, and
show a significant performance increase of our model over baseline approaches.
In addition, our real-world experiments with a robot arm show that users
significantly prefer our natural language interface over traditional methods
such as kinesthetic teaching or cost-function programming. Our study shows how
the field of robotics can take advantage of large pre-trained language models
towards creating more intuitive interfaces between robots and machines. Project
webpage: https://arthurfenderbucker.github.io/NL_trajectory_reshaper/


---

## CodeGen: An Open Large Language Model for Code with Multi-Turn Program
  Synthesis

**Published Date:** 2022-03-25T06:55:15Z

**Link:** http://arxiv.org/pdf/2203.13474v5

**Abstract:**

  Program synthesis strives to generate a computer program as a solution to a
given problem specification, expressed with input-output examples or natural
language descriptions. The prevalence of large language models advances the
state-of-the-art for program synthesis, though limited training resources and
data impede open access to such models. To democratize this, we train and
release a family of large language models up to 16.1B parameters, called
CODEGEN, on natural language and programming language data, and open source the
training library JAXFORMER. We show the utility of the trained model by
demonstrating that it is competitive with the previous state-of-the-art on
zero-shot Python code generation on HumanEval. We further investigate the
multi-step paradigm for program synthesis, where a single program is factorized
into multiple prompts specifying subproblems. To this end, we construct an open
benchmark, Multi-Turn Programming Benchmark (MTPB), consisting of 115 diverse
problem sets that are factorized into multi-turn prompts. Our analysis on MTPB
shows that the same intent provided to CODEGEN in multi-turn fashion
significantly improves program synthesis over that provided as a single turn.
We make the training library JAXFORMER and model checkpoints available as open
source contribution: https://github.com/salesforce/CodeGen.


---

## PERT: Pre-training BERT with Permuted Language Model

**Published Date:** 2022-03-14T07:58:34Z

**Link:** http://arxiv.org/pdf/2203.06906v1

**Abstract:**

  Pre-trained Language Models (PLMs) have been widely used in various natural
language processing (NLP) tasks, owing to their powerful text representations
trained on large-scale corpora. In this paper, we propose a new PLM called PERT
for natural language understanding (NLU). PERT is an auto-encoding model (like
BERT) trained with Permuted Language Model (PerLM). The formulation of the
proposed PerLM is straightforward. We permute a proportion of the input text,
and the training objective is to predict the position of the original token.
Moreover, we also apply whole word masking and N-gram masking to improve the
performance of PERT. We carried out extensive experiments on both Chinese and
English NLU benchmarks. The experimental results show that PERT can bring
improvements over various comparable baselines on some of the tasks, while
others are not. These results indicate that developing more diverse
pre-training tasks is possible instead of masked language model variants.
Several quantitative studies are carried out to better understand PERT, which
might help design PLMs in the future. Resources are available:
https://github.com/ymcui/PERT


---

## IT5: Large-scale Text-to-text Pretraining for Italian Language
  Understanding and Generation

**Published Date:** 2022-03-07T22:39:01Z

**Link:** http://arxiv.org/pdf/2203.03759v1

**Abstract:**

  The T5 model and its unified text-to-text paradigm contributed in advancing
the state-of-the-art for many natural language processing tasks. While some
multilingual variants of the T5 model have recently been introduced, their
performances were found to provide suboptimal performances for languages other
than English if compared to monolingual variants. We are motivated by these
findings to introduce IT5, the first family of encoder-decoder transformer
models pretrained specifically on Italian. We perform a thorough cleaning of a
web-crawled Italian corpus including more than 40 billion words and use it to
pretrain three IT5 models of different sizes. The performance of IT5 models and
their multilingual counterparts is then evaluated on a broad range of natural
language understanding and generation benchmarks for Italian. We find the
monolingual IT5 models to provide the best scale-to-performance ratio across
tested models, consistently outperforming their multilingual counterparts and
setting a new state-of-the-art for most Italian conditional language generation
tasks.


---

