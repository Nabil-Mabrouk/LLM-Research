## Large Language Model Distillation Doesn't Need a Teacher

**Published Date:** 2023-05-24T08:18:35Z

**Link:** http://arxiv.org/pdf/2305.14864v1

**Abstract:**

  Knowledge distillation trains a smaller student model to match the output
distribution of a larger teacher to maximize the end-task performance under
computational constraints. However, existing literature on language model
distillation primarily focuses on compressing encoder-only models that are then
specialized by task-specific supervised finetuning. We need to rethink this
setup for more recent large language models with tens to hundreds of billions
of parameters. Task-specific finetuning is impractical at this scale, and model
performance is often measured using zero/few-shot prompting. Thus, in this
work, we advocate for task-agnostic zero-shot evaluated distillation for large
language models without access to end-task finetuning data. We propose a
teacher-free task-agnostic distillation method, which uses a truncated version
of the larger model for initialization, and continues pretraining this model
using a language modeling objective. Our teacher-free method shines in a
distillation regime where it is infeasible to fit both the student and teacher
into the GPU memory. Despite its simplicity, our method can effectively reduce
the model size by 50\%, matching or outperforming the vanilla distillation
method on perplexity and accuracy on 13 zero-shot end-tasks while being 1.5x
computationally efficient.


---

## Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale

**Published Date:** 2023-05-26T21:22:10Z

**Link:** http://arxiv.org/pdf/2305.17266v2

**Abstract:**

  In recent years, language models have drastically grown in size, and the
abilities of these models have been shown to improve with scale. The majority
of recent scaling laws studies focused on high-compute high-parameter count
settings, leaving the question of when these abilities begin to emerge largely
unanswered. In this paper, we investigate whether the effects of pre-training
can be observed when the problem size is reduced, modeling a smaller,
reduced-vocabulary language. We show the benefits of pre-training with masked
language modeling (MLM) objective in models as small as 1.25M parameters, and
establish a strong correlation between pre-training perplexity and downstream
performance (GLUE benchmark). We examine downscaling effects, extending scaling
laws to models as small as ~1M parameters. At this scale, we observe a break of
the power law for compute-optimal models and show that the MLM loss does not
scale smoothly with compute-cost (FLOPs) below $2.2 \times 10^{15}$ FLOPs. We
also find that adding layers does not always benefit downstream performance.


---

