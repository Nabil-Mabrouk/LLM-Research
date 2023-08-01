## Data-Efficient Learning of Natural Language to Linear Temporal Logic
  Translators for Robot Task Specification

**Published Date:** 2023-03-09T00:09:58Z

**Link:** http://arxiv.org/pdf/2303.08006v2

**Abstract:**

  To make robots accessible to a broad audience, it is critical to endow them
with the ability to take universal modes of communication, like commands given
in natural language, and extract a concrete desired task specification, defined
using a formal language like linear temporal logic (LTL). In this paper, we
present a learning-based approach for translating from natural language
commands to LTL specifications with very limited human-labeled training data.
This is in stark contrast to existing natural-language to LTL translators,
which require large human-labeled datasets, often in the form of labeled pairs
of LTL formulas and natural language commands, to train the translator. To
reduce reliance on human data, our approach generates a large synthetic
training dataset through algorithmic generation of LTL formulas, conversion to
structured English, and then exploiting the paraphrasing capabilities of modern
large language models (LLMs) to synthesize a diverse corpus of natural language
commands corresponding to the LTL formulas. We use this generated data to
finetune an LLM and apply a constrained decoding procedure at inference time to
ensure the returned LTL formula is syntactically correct. We evaluate our
approach on three existing LTL/natural language datasets and show that we can
translate natural language commands at 75\% accuracy with far less human data
($\le$12 annotations). Moreover, when training on large human-annotated
datasets, our method achieves higher test accuracy (95\% on average) than prior
work. Finally, we show the translated formulas can be used to plan
long-horizon, multi-stage tasks on a 12D quadrotor.


---

## CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive
  Learning

**Published Date:** 2023-03-22T17:59:59Z

**Link:** http://arxiv.org/pdf/2303.12793v1

**Abstract:**

  This work focuses on sign language retrieval-a recently proposed task for
sign language understanding. Sign language retrieval consists of two sub-tasks:
text-to-sign-video (T2V) retrieval and sign-video-to-text (V2T) retrieval.
Different from traditional video-text retrieval, sign language videos, not only
contain visual signals but also carry abundant semantic meanings by themselves
due to the fact that sign languages are also natural languages. Considering
this character, we formulate sign language retrieval as a cross-lingual
retrieval problem as well as a video-text retrieval task. Concretely, we take
into account the linguistic properties of both sign languages and natural
languages, and simultaneously identify the fine-grained cross-lingual (i.e.,
sign-to-word) mappings while contrasting the texts and the sign videos in a
joint embedding space. This process is termed as cross-lingual contrastive
learning. Another challenge is raised by the data scarcity issue-sign language
datasets are orders of magnitude smaller in scale than that of speech
recognition. We alleviate this issue by adopting a domain-agnostic sign encoder
pre-trained on large-scale sign videos into the target domain via
pseudo-labeling. Our framework, termed as domain-aware sign language retrieval
via Cross-lingual Contrastive learning or CiCo for short, outperforms the
pioneering method by large margins on various datasets, e.g., +22.4 T2V and
+28.0 V2T R@1 improvements on How2Sign dataset, and +13.7 T2V and +17.1 V2T R@1
improvements on PHOENIX-2014T dataset. Code and models are available at:
https://github.com/FangyunWei/SLRT.


---

## Data-Efficient Learning of Natural Language to Linear Temporal Logic
  Translators for Robot Task Specification

**Published Date:** 2023-03-09T00:09:58Z

**Link:** http://arxiv.org/pdf/2303.08006v2

**Abstract:**

  To make robots accessible to a broad audience, it is critical to endow them
with the ability to take universal modes of communication, like commands given
in natural language, and extract a concrete desired task specification, defined
using a formal language like linear temporal logic (LTL). In this paper, we
present a learning-based approach for translating from natural language
commands to LTL specifications with very limited human-labeled training data.
This is in stark contrast to existing natural-language to LTL translators,
which require large human-labeled datasets, often in the form of labeled pairs
of LTL formulas and natural language commands, to train the translator. To
reduce reliance on human data, our approach generates a large synthetic
training dataset through algorithmic generation of LTL formulas, conversion to
structured English, and then exploiting the paraphrasing capabilities of modern
large language models (LLMs) to synthesize a diverse corpus of natural language
commands corresponding to the LTL formulas. We use this generated data to
finetune an LLM and apply a constrained decoding procedure at inference time to
ensure the returned LTL formula is syntactically correct. We evaluate our
approach on three existing LTL/natural language datasets and show that we can
translate natural language commands at 75\% accuracy with far less human data
($\le$12 annotations). Moreover, when training on large human-annotated
datasets, our method achieves higher test accuracy (95\% on average) than prior
work. Finally, we show the translated formulas can be used to plan
long-horizon, multi-stage tasks on a 12D quadrotor.


---

## Data-Efficient Learning of Natural Language to Linear Temporal Logic
  Translators for Robot Task Specification

**first_author:** Jiayi Pan et al.

**Published Date:** 2023-03-09T00:09:58Z

**Link:** http://arxiv.org/pdf/2303.08006v2

**Abstract:**

  To make robots accessible to a broad audience, it is critical to endow them
with the ability to take universal modes of communication, like commands given
in natural language, and extract a concrete desired task specification, defined
using a formal language like linear temporal logic (LTL). In this paper, we
present a learning-based approach for translating from natural language
commands to LTL specifications with very limited human-labeled training data.
This is in stark contrast to existing natural-language to LTL translators,
which require large human-labeled datasets, often in the form of labeled pairs
of LTL formulas and natural language commands, to train the translator. To
reduce reliance on human data, our approach generates a large synthetic
training dataset through algorithmic generation of LTL formulas, conversion to
structured English, and then exploiting the paraphrasing capabilities of modern
large language models (LLMs) to synthesize a diverse corpus of natural language
commands corresponding to the LTL formulas. We use this generated data to
finetune an LLM and apply a constrained decoding procedure at inference time to
ensure the returned LTL formula is syntactically correct. We evaluate our
approach on three existing LTL/natural language datasets and show that we can
translate natural language commands at 75\% accuracy with far less human data
($\le$12 annotations). Moreover, when training on large human-annotated
datasets, our method achieves higher test accuracy (95\% on average) than prior
work. Finally, we show the translated formulas can be used to plan
long-horizon, multi-stage tasks on a 12D quadrotor.


---

