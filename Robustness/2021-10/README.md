## Can Character-based Language Models Improve Downstream Task Performance
  in Low-Resource and Noisy Language Scenarios?

**Published Date:** 2021-10-26T14:59:16Z

**Link:** http://arxiv.org/pdf/2110.13658v1

**Abstract:**

  Recent impressive improvements in NLP, largely based on the success of
contextual neural language models, have been mostly demonstrated on at most a
couple dozen high-resource languages. Building language models and, more
generally, NLP systems for non-standardized and low-resource languages remains
a challenging task. In this work, we focus on North-African colloquial
dialectal Arabic written using an extension of the Latin script, called
NArabizi, found mostly on social media and messaging communication. In this
low-resource scenario with data displaying a high level of variability, we
compare the downstream performance of a character-based language model on
part-of-speech tagging and dependency parsing to that of monolingual and
multilingual models. We show that a character-based model trained on only 99k
sentences of NArabizi and fined-tuned on a small treebank of this language
leads to performance close to those obtained with the same architecture
pre-trained on large multilingual and monolingual models. Confirming these
results a on much larger data set of noisy French user-generated content, we
argue that such character-based language models can be an asset for NLP in
low-resource and high language variability set-tings.


---

## Investigating Robustness of Dialog Models to Popular Figurative Language
  Constructs

**Published Date:** 2021-10-01T23:55:16Z

**Link:** http://arxiv.org/pdf/2110.00687v1

**Abstract:**

  Humans often employ figurative language use in communication, including
during interactions with dialog systems. Thus, it is important for real-world
dialog systems to be able to handle popular figurative language constructs like
metaphor and simile. In this work, we analyze the performance of existing
dialog models in situations where the input dialog context exhibits use of
figurative language. We observe large gaps in handling of figurative language
when evaluating the models on two open domain dialog datasets. When faced with
dialog contexts consisting of figurative language, some models show very large
drops in performance compared to contexts without figurative language. We
encourage future research in dialog modeling to separately analyze and report
results on figurative language in order to better test model capabilities
relevant to real-world use. Finally, we propose lightweight solutions to help
existing models become more robust to figurative language by simply using an
external resource to translate figurative language to literal (non-figurative)
forms while preserving the meaning to the best extent possible.


---

## ClimateBert: A Pretrained Language Model for Climate-Related Text

**Published Date:** 2021-10-22T18:47:34Z

**Link:** http://arxiv.org/pdf/2110.12010v3

**Abstract:**

  Over the recent years, large pretrained language models (LM) have
revolutionized the field of natural language processing (NLP). However, while
pretraining on general language has been shown to work very well for common
language, it has been observed that niche language poses problems. In
particular, climate-related texts include specific language that common LMs can
not represent accurately. We argue that this shortcoming of today's LMs limits
the applicability of modern NLP to the broad field of text processing of
climate-related texts. As a remedy, we propose CLIMATEBERT, a transformer-based
language model that is further pretrained on over 2 million paragraphs of
climate-related texts, crawled from various sources such as common news,
research articles, and climate reporting of companies. We find that CLIMATEBERT
leads to a 48% improvement on a masked language model objective which, in turn,
leads to lowering error rates by 3.57% to 35.71% for various climate-related
downstream tasks like text classification, sentiment analysis, and
fact-checking.


---

## LMSOC: An Approach for Socially Sensitive Pretraining

**Published Date:** 2021-10-20T00:10:37Z

**Link:** http://arxiv.org/pdf/2110.10319v1

**Abstract:**

  While large-scale pretrained language models have been shown to learn
effective linguistic representations for many NLP tasks, there remain many
real-world contextual aspects of language that current approaches do not
capture. For instance, consider a cloze-test "I enjoyed the ____ game this
weekend": the correct answer depends heavily on where the speaker is from, when
the utterance occurred, and the speaker's broader social milieu and
preferences. Although language depends heavily on the geographical, temporal,
and other social contexts of the speaker, these elements have not been
incorporated into modern transformer-based language models. We propose a simple
but effective approach to incorporate speaker social context into the learned
representations of large-scale language models. Our method first learns dense
representations of social contexts using graph representation learning
algorithms and then primes language model pretraining with these social context
representations. We evaluate our approach on geographically-sensitive
language-modeling tasks and show a substantial improvement (more than 100%
relative lift on MRR) compared to baselines.


---

## Continual Learning in Multilingual NMT via Language-Specific Embeddings

**Published Date:** 2021-10-20T10:38:57Z

**Link:** http://arxiv.org/pdf/2110.10478v1

**Abstract:**

  This paper proposes a technique for adding a new source or target language to
an existing multilingual NMT model without re-training it on the initial set of
languages. It consists in replacing the shared vocabulary with a small
language-specific vocabulary and fine-tuning the new embeddings on the new
language's parallel data. Some additional language-specific components may be
trained to improve performance (e.g., Transformer layers or adapter modules).
Because the parameters of the original model are not modified, its performance
on the initial languages does not degrade. We show on two sets of experiments
(small-scale on TED Talks, and large-scale on ParaCrawl) that this approach
performs as well or better as the more costly alternatives; and that it has
excellent zero-shot performance: training on English-centric data is enough to
translate between the new language and any of the initial languages.


---

## ClimateBert: A Pretrained Language Model for Climate-Related Text

**Published Date:** 2021-10-22T18:47:34Z

**Link:** http://arxiv.org/pdf/2110.12010v3

**Abstract:**

  Over the recent years, large pretrained language models (LM) have
revolutionized the field of natural language processing (NLP). However, while
pretraining on general language has been shown to work very well for common
language, it has been observed that niche language poses problems. In
particular, climate-related texts include specific language that common LMs can
not represent accurately. We argue that this shortcoming of today's LMs limits
the applicability of modern NLP to the broad field of text processing of
climate-related texts. As a remedy, we propose CLIMATEBERT, a transformer-based
language model that is further pretrained on over 2 million paragraphs of
climate-related texts, crawled from various sources such as common news,
research articles, and climate reporting of companies. We find that CLIMATEBERT
leads to a 48% improvement on a masked language model objective which, in turn,
leads to lowering error rates by 3.57% to 35.71% for various climate-related
downstream tasks like text classification, sentiment analysis, and
fact-checking.


---

## Continual Learning in Multilingual NMT via Language-Specific Embeddings

**Published Date:** 2021-10-20T10:38:57Z

**Link:** http://arxiv.org/pdf/2110.10478v1

**Abstract:**

  This paper proposes a technique for adding a new source or target language to
an existing multilingual NMT model without re-training it on the initial set of
languages. It consists in replacing the shared vocabulary with a small
language-specific vocabulary and fine-tuning the new embeddings on the new
language's parallel data. Some additional language-specific components may be
trained to improve performance (e.g., Transformer layers or adapter modules).
Because the parameters of the original model are not modified, its performance
on the initial languages does not degrade. We show on two sets of experiments
(small-scale on TED Talks, and large-scale on ParaCrawl) that this approach
performs as well or better as the more costly alternatives; and that it has
excellent zero-shot performance: training on English-centric data is enough to
translate between the new language and any of the initial languages.


---

## ClimateBert: A Pretrained Language Model for Climate-Related Text

**first_author:** Nicolas Webersinke et al.

**Published Date:** 2021-10-22T18:47:34Z

**Link:** http://arxiv.org/pdf/2110.12010v3

**Abstract:**

  Over the recent years, large pretrained language models (LM) have
revolutionized the field of natural language processing (NLP). However, while
pretraining on general language has been shown to work very well for common
language, it has been observed that niche language poses problems. In
particular, climate-related texts include specific language that common LMs can
not represent accurately. We argue that this shortcoming of today's LMs limits
the applicability of modern NLP to the broad field of text processing of
climate-related texts. As a remedy, we propose CLIMATEBERT, a transformer-based
language model that is further pretrained on over 2 million paragraphs of
climate-related texts, crawled from various sources such as common news,
research articles, and climate reporting of companies. We find that CLIMATEBERT
leads to a 48% improvement on a masked language model objective which, in turn,
leads to lowering error rates by 3.57% to 35.71% for various climate-related
downstream tasks like text classification, sentiment analysis, and
fact-checking.


---

