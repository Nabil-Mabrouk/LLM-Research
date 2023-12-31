## Neural Polysynthetic Language Modelling

**Published Date:** 2020-05-11T22:57:04Z

**Link:** http://arxiv.org/pdf/2005.05477v2

**Abstract:**

  Research in natural language processing commonly assumes that approaches that
work well for English and and other widely-used languages are "language
agnostic". In high-resource languages, especially those that are analytic, a
common approach is to treat morphologically-distinct variants of a common root
as completely independent word types. This assumes, that there are limited
morphological inflections per root, and that the majority will appear in a
large enough corpus, so that the model can adequately learn statistics about
each form. Approaches like stemming, lemmatization, or subword segmentation are
often used when either of those assumptions do not hold, particularly in the
case of synthetic languages like Spanish or Russian that have more inflection
than English.
  In the literature, languages like Finnish or Turkish are held up as extreme
examples of complexity that challenge common modelling assumptions. Yet, when
considering all of the world's languages, Finnish and Turkish are closer to the
average case. When we consider polysynthetic languages (those at the extreme of
morphological complexity), approaches like stemming, lemmatization, or subword
modelling may not suffice. These languages have very high numbers of hapax
legomena, showing the need for appropriate morphological handling of words,
without which it is not possible for a model to capture enough word statistics.
  We examine the current state-of-the-art in language modelling, machine
translation, and text prediction for four polysynthetic languages: Guaran\'i,
St. Lawrence Island Yupik, Central Alaskan Yupik, and Inuktitut. We then
propose a novel framework for language modelling that combines knowledge
representations from finite-state morphological analyzers with Tensor Product
Representations in order to enable neural language models capable of handling
the full range of typologically variant languages.


---

