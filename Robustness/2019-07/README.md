## Cross-Lingual Transfer for Distantly Supervised and Low-resources
  Indonesian NER

**Published Date:** 2019-07-25T16:04:09Z

**Link:** http://arxiv.org/pdf/1907.11158v1

**Abstract:**

  Manually annotated corpora for low-resource languages are usually small in
quantity (gold), or large but distantly supervised (silver). Inspired by recent
progress of injecting pre-trained language model (LM) on many Natural Language
Processing (NLP) task, we proposed to fine-tune pre-trained language model from
high-resources languages to low-resources languages to improve the performance
of both scenarios. Our empirical experiment demonstrates significant
improvement when fine-tuning pre-trained language model in cross-lingual
transfer scenarios for small gold corpus and competitive results in large
silver compare to supervised cross-lingual transfer, which will be useful when
there is no parallel annotation in the same task to begin. We compare our
proposed method of cross-lingual transfer using pre-trained LM to different
sources of transfer such as mono-lingual LM and Part-of-Speech tagging (POS) in
the downstream task of both large silver and small gold NER dataset by
exploiting character-level input of bi-directional language model task.


---

## Language comparison via network topology

**Published Date:** 2019-07-16T11:33:04Z

**Link:** http://arxiv.org/pdf/1907.06944v2

**Abstract:**

  Modeling relations between languages can offer understanding of language
characteristics and uncover similarities and differences between languages.
Automated methods applied to large textual corpora can be seen as opportunities
for novel statistical studies of language development over time, as well as for
improving cross-lingual natural language processing techniques. In this work,
we first propose how to represent textual data as a directed, weighted network
by the text2net algorithm. We next explore how various fast,
network-topological metrics, such as network community structure, can be used
for cross-lingual comparisons. In our experiments, we employ eight different
network topology metrics, and empirically showcase on a parallel corpus, how
the methods can be used for modeling the relations between nine selected
languages. We demonstrate that the proposed method scales to large corpora
consisting of hundreds of thousands of aligned sentences on an of-the-shelf
laptop. We observe that on the one hand properties such as communities, capture
some of the known differences between the languages, while others can be seen
as novel opportunities for linguistic studies.


---

## Learn Spelling from Teachers: Transferring Knowledge from Language
  Models to Sequence-to-Sequence Speech Recognition

**Published Date:** 2019-07-13T06:27:24Z

**Link:** http://arxiv.org/pdf/1907.06017v1

**Abstract:**

  Integrating an external language model into a sequence-to-sequence speech
recognition system is non-trivial. Previous works utilize linear interpolation
or a fusion network to integrate external language models. However, these
approaches introduce external components, and increase decoding computation. In
this paper, we instead propose a knowledge distillation based training approach
to integrating external language models into a sequence-to-sequence model. A
recurrent neural network language model, which is trained on large scale
external text, generates soft labels to guide the sequence-to-sequence model
training. Thus, the language model plays the role of the teacher. This approach
does not add any external component to the sequence-to-sequence model during
testing. And this approach is flexible to be combined with shallow fusion
technique together for decoding. The experiments are conducted on public
Chinese datasets AISHELL-1 and CLMAD. Our approach achieves a character error
rate of 9.3%, which is relatively reduced by 18.42% compared with the vanilla
sequence-to-sequence model.


---

