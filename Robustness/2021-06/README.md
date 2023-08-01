## Multilingual transfer of acoustic word embeddings improves when training
  on languages related to the target zero-resource language

**Published Date:** 2021-06-24T08:37:05Z

**Link:** http://arxiv.org/pdf/2106.12834v1

**Abstract:**

  Acoustic word embedding models map variable duration speech segments to fixed
dimensional vectors, enabling efficient speech search and discovery. Previous
work explored how embeddings can be obtained in zero-resource settings where no
labelled data is available in the target language. The current best approach
uses transfer learning: a single supervised multilingual model is trained using
labelled data from multiple well-resourced languages and then applied to a
target zero-resource language (without fine-tuning). However, it is still
unclear how the specific choice of training languages affect downstream
performance. Concretely, here we ask whether it is beneficial to use training
languages related to the target. Using data from eleven languages spoken in
Southern Africa, we experiment with adding data from different language
families while controlling for the amount of data per language. In word
discrimination and query-by-example search evaluations, we show that training
on languages from the same family gives large improvements. Through
finer-grained analysis, we show that training on even just a single related
language gives the largest gain. We also find that adding data from unrelated
languages generally doesn't hurt performance.


---

## GEM: A General Evaluation Benchmark for Multimodal Tasks

**Published Date:** 2021-06-18T03:14:13Z

**Link:** http://arxiv.org/pdf/2106.09889v1

**Abstract:**

  In this paper, we present GEM as a General Evaluation benchmark for
Multimodal tasks. Different from existing datasets such as GLUE, SuperGLUE,
XGLUE and XTREME that mainly focus on natural language tasks, GEM is a
large-scale vision-language benchmark, which consists of GEM-I for
image-language tasks and GEM-V for video-language tasks. Comparing with
existing multimodal datasets such as MSCOCO and Flicker30K for image-language
tasks, YouCook2 and MSR-VTT for video-language tasks, GEM is not only the
largest vision-language dataset covering image-language tasks and
video-language tasks at the same time, but also labeled in multiple languages.
We also provide two baseline models for this benchmark. We will release the
dataset, code and baseline models, aiming to advance the development of
multilingual multimodal research.


---

## Revisiting the Primacy of English in Zero-shot Cross-lingual Transfer

**Published Date:** 2021-06-30T16:05:57Z

**Link:** http://arxiv.org/pdf/2106.16171v1

**Abstract:**

  Despite their success, large pre-trained multilingual models have not
completely alleviated the need for labeled data, which is cumbersome to collect
for all target languages. Zero-shot cross-lingual transfer is emerging as a
practical solution: pre-trained models later fine-tuned on one transfer
language exhibit surprising performance when tested on many target languages.
English is the dominant source language for transfer, as reinforced by popular
zero-shot benchmarks. However, this default choice has not been systematically
vetted. In our study, we compare English against other transfer languages for
fine-tuning, on two pre-trained multilingual models (mBERT and mT5) and
multiple classification and question answering tasks. We find that other
high-resource languages such as German and Russian often transfer more
effectively, especially when the set of target languages is diverse or unknown
a priori. Unexpectedly, this can be true even when the training sets were
automatically translated from English. This finding can have immediate impact
on multilingual zero-shot systems, and should inform future benchmark designs.


---

## Leveraging Pre-trained Language Model for Speech Sentiment Analysis

**Published Date:** 2021-06-11T20:15:21Z

**Link:** http://arxiv.org/pdf/2106.06598v1

**Abstract:**

  In this paper, we explore the use of pre-trained language models to learn
sentiment information of written texts for speech sentiment analysis. First, we
investigate how useful a pre-trained language model would be in a 2-step
pipeline approach employing Automatic Speech Recognition (ASR) and
transcripts-based sentiment analysis separately. Second, we propose a pseudo
label-based semi-supervised training strategy using a language model on an
end-to-end speech sentiment approach to take advantage of a large, but
unlabeled speech dataset for training. Although spoken and written texts have
different linguistic characteristics, they can complement each other in
understanding sentiment. Therefore, the proposed system can not only model
acoustic characteristics to bear sentiment-specific information in speech
signals, but learn latent information to carry sentiments in the text
representation. In these experiments, we demonstrate the proposed approaches
improve F1 scores consistently compared to systems without a language model.
Moreover, we also show that the proposed framework can reduce 65% of human
supervision by leveraging a large amount of data without human sentiment
annotation and boost performance in a low-resource condition where the human
sentiment annotation is not available enough.


---

