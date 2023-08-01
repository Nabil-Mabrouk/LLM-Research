## Vokenization: Improving Language Understanding with Contextualized,
  Visual-Grounded Supervision

**Published Date:** 2020-10-14T02:11:51Z

**Link:** http://arxiv.org/pdf/2010.06775v1

**Abstract:**

  Humans learn language by listening, speaking, writing, reading, and also, via
interaction with the multimodal real world. Existing language pre-training
frameworks show the effectiveness of text-only self-supervision while we
explore the idea of a visually-supervised language model in this paper. We find
that the main reason hindering this exploration is the large divergence in
magnitude and distributions between the visually-grounded language datasets and
pure-language corpora. Therefore, we develop a technique named "vokenization"
that extrapolates multimodal alignments to language-only data by contextually
mapping language tokens to their related images (which we call "vokens"). The
"vokenizer" is trained on relatively small image captioning datasets and we
then apply it to generate vokens for large language corpora. Trained with these
contextually generated vokens, our visually-supervised language models show
consistent improvements over self-supervised alternatives on multiple
pure-language tasks such as GLUE, SQuAD, and SWAG. Code and pre-trained models
publicly available at https://github.com/airsplay/vokenization


---

## Towards Fully Bilingual Deep Language Modeling

**Published Date:** 2020-10-22T12:22:50Z

**Link:** http://arxiv.org/pdf/2010.11639v1

**Abstract:**

  Language models based on deep neural networks have facilitated great advances
in natural language processing and understanding tasks in recent years. While
models covering a large number of languages have been introduced, their
multilinguality has come at a cost in terms of monolingual performance, and the
best-performing models at most tasks not involving cross-lingual transfer
remain monolingual. In this paper, we consider the question of whether it is
possible to pre-train a bilingual model for two remotely related languages
without compromising performance at either language. We collect pre-training
data, create a Finnish-English bilingual BERT model and evaluate its
performance on datasets used to evaluate the corresponding monolingual models.
Our bilingual model performs on par with Google's original English BERT on GLUE
and nearly matches the performance of monolingual Finnish BERT on a range of
Finnish NLP tasks, clearly outperforming multilingual BERT. We find that when
the model vocabulary size is increased, the BERT-Base architecture has
sufficient capacity to learn two remotely related languages to a level where it
achieves comparable performance with monolingual models, demonstrating the
feasibility of training fully bilingual deep language models. The model and all
tools involved in its creation are freely available at
https://github.com/TurkuNLP/biBERT


---

## Cross-lingual Machine Reading Comprehension with Language Branch
  Knowledge Distillation

**Published Date:** 2020-10-27T13:12:17Z

**Link:** http://arxiv.org/pdf/2010.14271v1

**Abstract:**

  Cross-lingual Machine Reading Comprehension (CLMRC) remains a challenging
problem due to the lack of large-scale annotated datasets in low-source
languages, such as Arabic, Hindi, and Vietnamese. Many previous approaches use
translation data by translating from a rich-source language, such as English,
to low-source languages as auxiliary supervision. However, how to effectively
leverage translation data and reduce the impact of noise introduced by
translation remains onerous. In this paper, we tackle this challenge and
enhance the cross-lingual transferring performance by a novel augmentation
approach named Language Branch Machine Reading Comprehension (LBMRC). A
language branch is a group of passages in one single language paired with
questions in all target languages. We train multiple machine reading
comprehension (MRC) models proficient in individual language based on LBMRC.
Then, we devise a multilingual distillation approach to amalgamate knowledge
from multiple language branch models to a single model for all target
languages. Combining the LBMRC and multilingual distillation can be more robust
to the data noises, therefore, improving the model's cross-lingual ability.
Meanwhile, the produced single multilingual model is applicable to all target
languages, which saves the cost of training, inference, and maintenance for
multiple models. Extensive experiments on two CLMRC benchmarks clearly show the
effectiveness of our proposed method.


---

## Language ID in the Wild: Unexpected Challenges on the Path to a
  Thousand-Language Web Text Corpus

**Published Date:** 2020-10-27T19:29:17Z

**Link:** http://arxiv.org/pdf/2010.14571v2

**Abstract:**

  Large text corpora are increasingly important for a wide variety of Natural
Language Processing (NLP) tasks, and automatic language identification (LangID)
is a core technology needed to collect such datasets in a multilingual context.
LangID is largely treated as solved in the literature, with models reported
that achieve over 90% average F1 on as many as 1,366 languages. We train LangID
models on up to 1,629 languages with comparable quality on held-out test sets,
but find that human-judged LangID accuracy for web-crawl text corpora created
using these models is only around 5% for many lower-resource languages,
suggesting a need for more robust evaluation. Further analysis revealed a
variety of error modes, arising from domain mismatch, class imbalance, language
similarity, and insufficiently expressive models. We propose two classes of
techniques to mitigate these errors: wordlist-based tunable-precision filters
(for which we release curated lists in about 500 languages) and
transformer-based semi-supervised LangID models, which increase median dataset
precision from 5.5% to 71.2%. These techniques enable us to create an initial
data set covering 100K or more relatively clean sentences in each of 500+
languages, paving the way towards a 1,000-language web text corpus.


---

## Pretrained Language Models for Dialogue Generation with Multiple Input
  Sources

**Published Date:** 2020-10-15T07:53:28Z

**Link:** http://arxiv.org/pdf/2010.07576v1

**Abstract:**

  Large-scale pretrained language models have achieved outstanding performance
on natural language understanding tasks. However, it is still under
investigating how to apply them to dialogue generation tasks, especially those
with responses conditioned on multiple sources. Previous work simply
concatenates all input sources or averages information from different input
sources. In this work, we study dialogue models with multiple input sources
adapted from the pretrained language model GPT2. We explore various methods to
fuse multiple separate attention information corresponding to different
sources. Our experimental results show that proper fusion methods deliver
higher relevance with dialogue history than simple fusion baselines.


---

## Comparison of Interactive Knowledge Base Spelling Correction Models for
  Low-Resource Languages

**Published Date:** 2020-10-20T17:31:07Z

**Link:** http://arxiv.org/pdf/2010.10472v1

**Abstract:**

  Spelling normalization for low resource languages is a challenging task
because the patterns are hard to predict and large corpora are usually required
to collect enough examples. This work shows a comparison of a neural model and
character language models with varying amounts on target language data. Our
usage scenario is interactive correction with nearly zero amounts of training
examples, improving models as more data is collected, for example within a chat
app. Such models are designed to be incrementally improved as feedback is given
from users. In this work, we design a knowledge-base and prediction model
embedded system for spelling correction in low-resource languages. Experimental
results on multiple languages show that the model could become effective with a
small amount of data. We perform experiments on both natural and synthetic
data, as well as on data from two endangered languages (Ainu and Griko). Last,
we built a prototype system that was used for a small case study on Hinglish,
which further demonstrated the suitability of our approach in real world
scenarios.


---

## CoLAKE: Contextualized Language and Knowledge Embedding

**Published Date:** 2020-10-01T11:39:32Z

**Link:** http://arxiv.org/pdf/2010.00309v1

**Abstract:**

  With the emerging branch of incorporating factual knowledge into pre-trained
language models such as BERT, most existing models consider shallow, static,
and separately pre-trained entity embeddings, which limits the performance
gains of these models. Few works explore the potential of deep contextualized
knowledge representation when injecting knowledge. In this paper, we propose
the Contextualized Language and Knowledge Embedding (CoLAKE), which jointly
learns contextualized representation for both language and knowledge with the
extended MLM objective. Instead of injecting only entity embeddings, CoLAKE
extracts the knowledge context of an entity from large-scale knowledge bases.
To handle the heterogeneity of knowledge context and language context, we
integrate them in a unified data structure, word-knowledge graph (WK graph).
CoLAKE is pre-trained on large-scale WK graphs with the modified Transformer
encoder. We conduct experiments on knowledge-driven tasks, knowledge probing
tasks, and language understanding tasks. Experimental results show that CoLAKE
outperforms previous counterparts on most of the tasks. Besides, CoLAKE
achieves surprisingly high performance on our synthetic task called
word-knowledge graph completion, which shows the superiority of simultaneously
contextualizing language and knowledge representation.


---

## OCNLI: Original Chinese Natural Language Inference

**Published Date:** 2020-10-12T04:25:48Z

**Link:** http://arxiv.org/pdf/2010.05444v1

**Abstract:**

  Despite the tremendous recent progress on natural language inference (NLI),
driven largely by large-scale investment in new datasets (e.g., SNLI, MNLI) and
advances in modeling, most progress has been limited to English due to a lack
of reliable datasets for most of the world's languages. In this paper, we
present the first large-scale NLI dataset (consisting of ~56,000 annotated
sentence pairs) for Chinese called the Original Chinese Natural Language
Inference dataset (OCNLI). Unlike recent attempts at extending NLI to other
languages, our dataset does not rely on any automatic translation or non-expert
annotation. Instead, we elicit annotations from native speakers specializing in
linguistics. We follow closely the annotation protocol used for MNLI, but
create new strategies for eliciting diverse hypotheses. We establish several
baseline results on our dataset using state-of-the-art pre-trained models for
Chinese, and find even the best performing models to be far outpaced by human
performance (~12% absolute performance gap), making it a challenging new
resource that we hope will help to accelerate progress in Chinese NLU. To the
best of our knowledge, this is the first human-elicited MNLI-style corpus for a
non-English language.


---

## Detecting ESG topics using domain-specific language models and data
  augmentation approaches

**Published Date:** 2020-10-16T11:20:07Z

**Link:** http://arxiv.org/pdf/2010.08319v1

**Abstract:**

  Despite recent advances in deep learning-based language modelling, many
natural language processing (NLP) tasks in the financial domain remain
challenging due to the paucity of appropriately labelled data. Other issues
that can limit task performance are differences in word distribution between
the general corpora - typically used to pre-train language models - and
financial corpora, which often exhibit specialized language and symbology.
Here, we investigate two approaches that may help to mitigate these issues.
Firstly, we experiment with further language model pre-training using large
amounts of in-domain data from business and financial news. We then apply
augmentation approaches to increase the size of our dataset for model
fine-tuning. We report our findings on an Environmental, Social and Governance
(ESG) controversies dataset and demonstrate that both approaches are beneficial
to accuracy in classification tasks.


---

## HateBERT: Retraining BERT for Abusive Language Detection in English

**Published Date:** 2020-10-23T15:14:14Z

**Link:** http://arxiv.org/pdf/2010.12472v2

**Abstract:**

  In this paper, we introduce HateBERT, a re-trained BERT model for abusive
language detection in English. The model was trained on RAL-E, a large-scale
dataset of Reddit comments in English from communities banned for being
offensive, abusive, or hateful that we have collected and made available to the
public. We present the results of a detailed comparison between a general
pre-trained language model and the abuse-inclined version obtained by
retraining with posts from the banned communities on three English datasets for
offensive, abusive language and hate speech detection tasks. In all datasets,
HateBERT outperforms the corresponding general BERT model. We also discuss a
battery of experiments comparing the portability of the generic pre-trained
language model and its corresponding abusive language-inclined counterpart
across the datasets, indicating that portability is affected by compatibility
of the annotated phenomena.


---

## SLM: Learning a Discourse Language Representation with Sentence
  Unshuffling

**Published Date:** 2020-10-30T13:33:41Z

**Link:** http://arxiv.org/pdf/2010.16249v1

**Abstract:**

  We introduce Sentence-level Language Modeling, a new pre-training objective
for learning a discourse language representation in a fully self-supervised
manner. Recent pre-training methods in NLP focus on learning either bottom or
top-level language representations: contextualized word representations derived
from language model objectives at one extreme and a whole sequence
representation learned by order classification of two given textual segments at
the other. However, these models are not directly encouraged to capture
representations of intermediate-size structures that exist in natural languages
such as sentences and the relationships among them. To that end, we propose a
new approach to encourage learning of a contextualized sentence-level
representation by shuffling the sequence of input sentences and training a
hierarchical transformer model to reconstruct the original ordering. Through
experiments on downstream tasks such as GLUE, SQuAD, and DiscoEval, we show
that this feature of our model improves the performance of the original BERT by
large margins.


---

## The Turking Test: Can Language Models Understand Instructions?

**Published Date:** 2020-10-22T18:44:16Z

**Link:** http://arxiv.org/pdf/2010.11982v1

**Abstract:**

  Supervised machine learning provides the learner with a set of input-output
examples of the target task. Humans, however, can also learn to perform new
tasks from instructions in natural language. Can machines learn to understand
instructions as well? We present the Turking Test, which examines a model's
ability to follow natural language instructions of varying complexity. These
range from simple tasks, like retrieving the nth word of a sentence, to ones
that require creativity, such as generating examples for SNLI and SQuAD in
place of human intelligence workers ("turkers"). Despite our lenient evaluation
methodology, we observe that a large pretrained language model performs poorly
across all tasks. Analyzing the model's error patterns reveals that the model
tends to ignore explicit instructions and often generates outputs that cannot
be construed as an attempt to solve the task. While it is not yet clear whether
instruction understanding can be captured by traditional language models, the
sheer expressivity of instruction understanding makes it an appealing
alternative to the rising few-shot inference paradigm.


---

## Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced
  Language Model Pre-training

**Published Date:** 2020-10-23T22:14:50Z

**Link:** http://arxiv.org/pdf/2010.12688v2

**Abstract:**

  Prior work on Data-To-Text Generation, the task of converting knowledge graph
(KG) triples into natural text, focused on domain-specific benchmark datasets.
In this paper, however, we verbalize the entire English Wikidata KG, and
discuss the unique challenges associated with a broad, open-domain, large-scale
verbalization. We further show that verbalizing a comprehensive, encyclopedic
KG like Wikidata can be used to integrate structured KGs and natural language
corpora. In contrast to the many architectures that have been developed to
integrate these two sources, our approach converts the KG into natural text,
allowing it to be seamlessly integrated into existing language models. It
carries the further advantages of improved factual accuracy and reduced
toxicity in the resulting language model. We evaluate this approach by
augmenting the retrieval corpus in a retrieval language model and showing
significant improvements on the knowledge intensive tasks of open domain QA and
the LAMA knowledge probe.


---

## Towards Zero-Shot Multilingual Synthetic Question and Answer Generation
  for Cross-Lingual Reading Comprehension

**Published Date:** 2020-10-22T19:59:37Z

**Link:** http://arxiv.org/pdf/2010.12008v3

**Abstract:**

  We propose a simple method to generate multilingual question and answer pairs
on a large scale through the use of a single generative model. These synthetic
samples can be used to improve the zero-shot performance of multilingual QA
models on target languages. Our proposed multi-task training of the generative
model only requires the labeled training samples in English, thus removing the
need for such samples in the target languages, making it applicable to far more
languages than those with labeled data. Human evaluations indicate the majority
of such samples are grammatically correct and sensible. Experimental results
show our proposed approach can achieve large gains on the XQuAD dataset,
reducing the gap between zero-shot and supervised performance of smaller QA
models on various languages.


---

## Comparison of Interactive Knowledge Base Spelling Correction Models for
  Low-Resource Languages

**first_author:** Yiyuan Li et al.

**Published Date:** 2020-10-20T17:31:07Z

**Link:** http://arxiv.org/pdf/2010.10472v1

**Abstract:**

  Spelling normalization for low resource languages is a challenging task
because the patterns are hard to predict and large corpora are usually required
to collect enough examples. This work shows a comparison of a neural model and
character language models with varying amounts on target language data. Our
usage scenario is interactive correction with nearly zero amounts of training
examples, improving models as more data is collected, for example within a chat
app. Such models are designed to be incrementally improved as feedback is given
from users. In this work, we design a knowledge-base and prediction model
embedded system for spelling correction in low-resource languages. Experimental
results on multiple languages show that the model could become effective with a
small amount of data. We perform experiments on both natural and synthetic
data, as well as on data from two endangered languages (Ainu and Griko). Last,
we built a prototype system that was used for a small case study on Hinglish,
which further demonstrated the suitability of our approach in real world
scenarios.


---

