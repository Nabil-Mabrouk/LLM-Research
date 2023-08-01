## Language Model Supervision for Handwriting Recognition Model Adaptation

**Published Date:** 2018-08-04T04:27:05Z

**Link:** http://arxiv.org/pdf/1808.01423v1

**Abstract:**

  Training state-of-the-art offline handwriting recognition (HWR) models
requires large labeled datasets, but unfortunately such datasets are not
available in all languages and domains due to the high cost of manual
labeling.We address this problem by showing how high resource languages can be
leveraged to help train models for low resource languages.We propose a transfer
learning methodology where we adapt HWR models trained on a source language to
a target language that uses the same writing script.This methodology only
requires labeled data in the source language, unlabeled data in the target
language, and a language model of the target language. The language model is
used in a bootstrapping fashion to refine predictions in the target language
for use as ground truth in training the model.Using this approach we
demonstrate improved transferability among French, English, and Spanish
languages using both historical and modern handwriting datasets. In the best
case, transferring with the proposed methodology results in character error
rates nearly as good as full supervised training.


---

