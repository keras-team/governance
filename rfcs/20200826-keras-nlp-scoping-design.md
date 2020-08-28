# Keras NLP

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Mark Momernick (momernick@google.com), Francois Chollet (fchollet@google.com) |
| **Updated**   | 2020-08-26                                           |


## Objective

We aim at describing the scope of [keras-nlp](https://github.com/keras-team/keras-nlp), especially:
- What keras-nlp should include
- Boundaries between keras-nlp and [tensorflow addons](https://github.com/tensorflow/addons)
- Boundaries between keras-nlp and [tensorflow model garden](https://github.com/tensorflow/models)
- Boundaries between keras-nlp and [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).
- Boundaries between keras-nlp and [tf.text](https://www.tensorflow.org/tutorials/tensorflow_text/intro).

## Motivation

Natural language processing (nlp) has become an essential component in many machine learning solutions.
However, a Keras-native modeling solution is still lacking for TF2.

## User Benefit

We hope to help machine learning engineer benefit from re-using Keras native, optimized, and well-tested
components to build their models. We hope to help research scientists benefit from subclassing and re-composing
Keras primitive components to test new research ideas. 

## Design Proposal

We propose keras-nlp to include most transformer-based modules, specifically:
- Keras layer components such as Transformer encoder and decoder blocks.
- Keras task components such as masked language, span labeler and named entity recognition.
- Tensorflow operations such as beam search.
- Keras optimizer utilities such as learning rate schedules widely used.
- Data loader and preprocessing for different dataset, such as SQUAD, GLUE.

Criteria for keras-nlp:
- Widely accepted building components that serve various modeling tasks.
- Tasks that improves model training.
- Operations that will work in CPU/GPU/TPU.

Boundaries between keras-nlp and TF.Text:
- TF.Text will contain all pre-processing operations, such as WordPiece Tokenizer, n-grams, that handles strings.
- keras-nlp will contain components after tokenization.

Boundaries between keras-nlp and Tensorflow Addons:
- Highly experimental modeling, layers, losses, etc, live in addons.
- Components from addons will graduate to Model Garden, given it incurs more usage,
 and it works in CPU/GPU/TPU. The API interface will remain experimental after graduation.

Boundaries between keras-nlp and Model Garden:
- End to end modeling workflow and model specific details live in Model Garden
- Model garden will re-use most of the building blocks from keras-nlp
- Components from Model Garden can graduate to keras-nlp, given it is widely accepted, 
 it works performant in CPU/GPU/TPU. The API interface should remain stable after graduation.

Boundaries between keras-nlp and Keras:
- keras-nlp will contain language specific components.
- Components from keras-nlp can graduate to Keras core, given its usage expands beyond
 natural language processing. One example is `tf.keras.layers.MultiHeadAttention`

Dependencies:
- Tensorflow version >= 2.4.
- Tensorflow datasets.

Backwards compatibility:
We propose to guarantee major release backwards compatibility.

Maintenance:
This repository will be maintained by Keras team.

Performance Benchmark:
We will set-up Keras benchmark utilities to help users contribute to this repository.

## Detailed Design
Detailed design will be separate from scoping.

## Questions and Discussion Topics