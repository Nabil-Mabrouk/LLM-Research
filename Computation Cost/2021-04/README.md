## Transfer training from smaller language model

**Published Date:** 2021-04-23T02:56:02Z

**Link:** http://arxiv.org/pdf/2104.11390v1

**Abstract:**

  Large language models have led to state-of-the-art accuracies across a range
of tasks. However,training large language model needs massive computing
resource, as more and more open source pre-training models are available, it is
worthy to study how to take full advantage of available model. We find a method
to save training time and resource cost by changing the small well-trained
model to large model. We initialize a larger target model from a smaller source
model by copy weight values from source model and padding with zeros or small
initialization values on it to make the source and target model have
approximate outputs, which is valid due to block matrix multiplication and
residual connection in transformer structure. We test the target model on
several data sets and find it is still comparable with the source model. When
we continue training the target model, the training loss can start from a
smaller value.


---

