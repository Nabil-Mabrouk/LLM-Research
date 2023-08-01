## On the Multilingual Capabilities of Very Large-Scale English Language
  Models

**Published Date:** 2021-08-30T16:18:50Z

**Link:** http://arxiv.org/pdf/2108.13349v1

**Abstract:**

  Generative Pre-trained Transformers (GPTs) have recently been scaled to
unprecedented sizes in the history of machine learning. These models, solely
trained on the language modeling objective, have been shown to exhibit
outstanding few-shot learning capabilities in a number of different tasks.
Nevertheless, aside from anecdotal experiences, little is known regarding their
multilingual capabilities, given the fact that the pre-training corpus is
almost entirely composed of English text. In this work, we investigate the
multilingual skills of GPT-3, focusing on one language that barely appears in
the pre-training corpus, Catalan, which makes the results especially
meaningful; we assume that our results may be relevant for other languages as
well. We find that the model shows an outstanding performance, particularly in
generative tasks, with predictable limitations mostly in language understanding
tasks but still with remarkable results given the zero-shot scenario. We
investigate its potential and limits in extractive question-answering and
natural language generation, as well as the effect of scale in terms of model
size.


---

## Sentence Bottleneck Autoencoders from Transformer Language Models

**Published Date:** 2021-08-31T19:39:55Z

**Link:** http://arxiv.org/pdf/2109.00055v2

**Abstract:**

  Representation learning for text via pretraining a language model on a large
corpus has become a standard starting point for building NLP systems. This
approach stands in contrast to autoencoders, also trained on raw text, but with
the objective of learning to encode each input as a vector that allows full
reconstruction. Autoencoders are attractive because of their latent space
structure and generative properties. We therefore explore the construction of a
sentence-level autoencoder from a pretrained, frozen transformer language
model. We adapt the masked language modeling objective as a generative,
denoising one, while only training a sentence bottleneck and a single-layer
modified transformer decoder. We demonstrate that the sentence representations
discovered by our model achieve better quality than previous methods that
extract representations from pretrained transformers on text similarity tasks,
style transfer (an example of controlled generation), and single-sentence
classification tasks in the GLUE benchmark, while using fewer parameters than
large pretrained models.


---

