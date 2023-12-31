## Cross-lingual Transfer of Sentiment Classifiers

**Published Date:** 2020-05-15T10:15:27Z

**Link:** http://arxiv.org/pdf/2005.07456v3

**Abstract:**

  Word embeddings represent words in a numeric space so that semantic relations
between words are represented as distances and directions in the vector space.
Cross-lingual word embeddings transform vector spaces of different languages so
that similar words are aligned. This is done by constructing a mapping between
vector spaces of two languages or learning a joint vector space for multiple
languages. Cross-lingual embeddings can be used to transfer machine learning
models between languages, thereby compensating for insufficient data in
less-resourced languages. We use cross-lingual word embeddings to transfer
machine learning prediction models for Twitter sentiment between 13 languages.
We focus on two transfer mechanisms that recently show superior transfer
performance. The first mechanism uses the trained models whose input is the
joint numerical space for many languages as implemented in the LASER library.
The second mechanism uses large pretrained multilingual BERT language models.
Our experiments show that the transfer of models between similar languages is
sensible, even with no target language data. The performance of cross-lingual
models obtained with the multilingual BERT and LASER library is comparable, and
the differences are language-dependent. The transfer with CroSloEngual BERT,
pretrained on only three languages, is superior on these and some closely
related languages.


---

## Can Multilingual Language Models Transfer to an Unseen Dialect? A Case
  Study on North African Arabizi

**Published Date:** 2020-05-01T11:29:23Z

**Link:** http://arxiv.org/pdf/2005.00318v1

**Abstract:**

  Building natural language processing systems for non standardized and low
resource languages is a difficult challenge. The recent success of large-scale
multilingual pretrained language models provides new modeling tools to tackle
this. In this work, we study the ability of multilingual language models to
process an unseen dialect. We take user generated North-African Arabic as our
case study, a resource-poor dialectal variety of Arabic with frequent
code-mixing with French and written in Arabizi, a non-standardized
transliteration of Arabic to Latin script. Focusing on two tasks,
part-of-speech tagging and dependency parsing, we show in zero-shot and
unsupervised adaptation scenarios that multilingual language models are able to
transfer to such an unseen dialect, specifically in two extreme cases: (i)
across scripts, using Modern Standard Arabic as a source language, and (ii)
from a distantly related language, unseen during pretraining, namely Maltese.
Our results constitute the first successful transfer experiments on this
dialect, paving thus the way for the development of an NLP ecosystem for
resource-scarce, non-standardized and highly variable vernacular languages.


---

## Cross-modal Language Generation using Pivot Stabilization for Web-scale
  Language Coverage

**Published Date:** 2020-05-01T06:58:18Z

**Link:** http://arxiv.org/pdf/2005.00246v1

**Abstract:**

  Cross-modal language generation tasks such as image captioning are directly
hurt in their ability to support non-English languages by the trend of
data-hungry models combined with the lack of non-English annotations. We
investigate potential solutions for combining existing language-generation
annotations in English with translation capabilities in order to create
solutions at web-scale in both domain and language coverage. We describe an
approach called Pivot-Language Generation Stabilization (PLuGS), which
leverages directly at training time both existing English annotations (gold
data) as well as their machine-translated versions (silver data); at run-time,
it generates first an English caption and then a corresponding target-language
caption. We show that PLuGS models outperform other candidate solutions in
evaluations performed over 5 different target languages, under a large-domain
testset using images from the Open Images dataset. Furthermore, we find an
interesting effect where the English captions generated by the PLuGS models are
better than the captions generated by the original, monolingual English model.


---

## Language Conditioned Imitation Learning over Unstructured Data

**Published Date:** 2020-05-15T17:08:50Z

**Link:** http://arxiv.org/pdf/2005.07648v2

**Abstract:**

  Natural language is perhaps the most flexible and intuitive way for humans to
communicate tasks to a robot. Prior work in imitation learning typically
requires each task be specified with a task id or goal image -- something that
is often impractical in open-world environments. On the other hand, previous
approaches in instruction following allow agent behavior to be guided by
language, but typically assume structure in the observations, actuators, or
language that limit their applicability to complex settings like robotics. In
this work, we present a method for incorporating free-form natural language
conditioning into imitation learning. Our approach learns perception from
pixels, natural language understanding, and multitask continuous control
end-to-end as a single neural network. Unlike prior work in imitation learning,
our method is able to incorporate unlabeled and unstructured demonstration data
(i.e. no task or language labels). We show this dramatically improves language
conditioned performance, while reducing the cost of language annotation to less
than 1% of total data. At test time, a single language conditioned visuomotor
policy trained with our method can perform a wide variety of robotic
manipulation skills in a 3D environment, specified only with natural language
descriptions of each task (e.g. "open the drawer...now pick up the block...now
press the green button..."). To scale up the number of instructions an agent
can follow, we propose combining text conditioned policies with large
pretrained neural language models. We find this allows a policy to be robust to
many out-of-distribution synonym instructions, without requiring new
demonstrations. See videos of a human typing live text commands to our agent at
language-play.github.io


---

## Efficient Deployment of Conversational Natural Language Interfaces over
  Databases

**Published Date:** 2020-05-31T19:16:27Z

**Link:** http://arxiv.org/pdf/2006.00591v2

**Abstract:**

  Many users communicate with chatbots and AI assistants in order to help them
with various tasks. A key component of the assistant is the ability to
understand and answer a user's natural language questions for
question-answering (QA). Because data can be usually stored in a structured
manner, an essential step involves turning a natural language question into its
corresponding query language. However, in order to train most natural
language-to-query-language state-of-the-art models, a large amount of training
data is needed first. In most domains, this data is not available and
collecting such datasets for various domains can be tedious and time-consuming.
In this work, we propose a novel method for accelerating the training dataset
collection for developing the natural language-to-query-language machine
learning models. Our system allows one to generate conversational multi-term
data, where multiple turns define a dialogue session, enabling one to better
utilize chatbot interfaces. We train two current state-of-the-art NL-to-QL
models, on both an SQL and SPARQL-based datasets in order to showcase the
adaptability and efficacy of our created data.


---

## Cross-modal Language Generation using Pivot Stabilization for Web-scale
  Language Coverage

**Published Date:** 2020-05-01T06:58:18Z

**Link:** http://arxiv.org/pdf/2005.00246v1

**Abstract:**

  Cross-modal language generation tasks such as image captioning are directly
hurt in their ability to support non-English languages by the trend of
data-hungry models combined with the lack of non-English annotations. We
investigate potential solutions for combining existing language-generation
annotations in English with translation capabilities in order to create
solutions at web-scale in both domain and language coverage. We describe an
approach called Pivot-Language Generation Stabilization (PLuGS), which
leverages directly at training time both existing English annotations (gold
data) as well as their machine-translated versions (silver data); at run-time,
it generates first an English caption and then a corresponding target-language
caption. We show that PLuGS models outperform other candidate solutions in
evaluations performed over 5 different target languages, under a large-domain
testset using images from the Open Images dataset. Furthermore, we find an
interesting effect where the English captions generated by the PLuGS models are
better than the captions generated by the original, monolingual English model.


---

## How Context Affects Language Models' Factual Predictions

**Published Date:** 2020-05-10T09:28:12Z

**Link:** http://arxiv.org/pdf/2005.04611v1

**Abstract:**

  When pre-trained on large unsupervised textual corpora, language models are
able to store and retrieve factual knowledge to some extent, making it possible
to use them directly for zero-shot cloze-style question answering. However,
storing factual knowledge in a fixed number of weights of a language model
clearly has limitations. Previous approaches have successfully provided access
to information outside the model weights using supervised architectures that
combine an information retrieval system with a machine reading component. In
this paper, we go a step further and integrate information from a retrieval
system with a pre-trained language model in a purely unsupervised way. We
report that augmenting pre-trained language models in this way dramatically
improves performance and that the resulting system, despite being unsupervised,
is competitive with a supervised machine reading baseline. Furthermore,
processing query and context with different segment tokens allows BERT to
utilize its Next Sentence Prediction pre-trained classifier to determine
whether the context is relevant or not, substantially improving BERT's
zero-shot cloze-style question-answering performance and making its predictions
robust to noisy contexts.


---

## Cross-modal Language Generation using Pivot Stabilization for Web-scale
  Language Coverage

**first_author:** Ashish V. Thapliyal et al.

**Published Date:** 2020-05-01T06:58:18Z

**Link:** http://arxiv.org/pdf/2005.00246v1

**Abstract:**

  Cross-modal language generation tasks such as image captioning are directly
hurt in their ability to support non-English languages by the trend of
data-hungry models combined with the lack of non-English annotations. We
investigate potential solutions for combining existing language-generation
annotations in English with translation capabilities in order to create
solutions at web-scale in both domain and language coverage. We describe an
approach called Pivot-Language Generation Stabilization (PLuGS), which
leverages directly at training time both existing English annotations (gold
data) as well as their machine-translated versions (silver data); at run-time,
it generates first an English caption and then a corresponding target-language
caption. We show that PLuGS models outperform other candidate solutions in
evaluations performed over 5 different target languages, under a large-domain
testset using images from the Open Images dataset. Furthermore, we find an
interesting effect where the English captions generated by the PLuGS models are
better than the captions generated by the original, monolingual English model.


---

