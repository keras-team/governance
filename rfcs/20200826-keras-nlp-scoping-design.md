# Keras NLP

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Mark Omernick (momernick@google.com), Francois Chollet (fchollet@google.com), Hongkun Yu (hongkuny@google.com)|
| **Updated**   | 2020-09-11                                           |


## Objective

We aim at describing the scope of [keras-nlp](https://github.com/keras-team/keras-nlp), especially:

- What use cases `keras-nlp` should cover
- Boundaries between `keras-nlp` and [tensorflow addons](https://github.com/tensorflow/addons)
- Boundaries between `keras-nlp` and [tensorflow model garden](https://github.com/tensorflow/models)
- Boundaries between `keras-nlp` and [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras).
- Boundaries between `keras-nlp` and [tf.text](https://www.tensorflow.org/tutorials/tensorflow_text/intro).

## Motivation

Natural Language Processing (NLP) is a major application area for our users.
In recent years, Transformer-based models have become the foundation of many NLP workflows.
These workflows tend to reuse similar components, for which in some cases third-party packages
have been developed by the open-source community.

These third-party solutions are not always kept up to date or up to the same quality standards as core Keras.
They also raise the issue of API standardization.

To fix this, we want machine learning engineers to have access to a standard Keras-native,
optimized, and well-tested set of components to build their Transformer-based (and beyond) NLP workflows.

This provides key user benefits:

- The package would be first-party and thus always up to date with modern best practices.
- High code quality and testing standards and strict quality control: same level of trust as core Keras
- A shared API standard across the community
- Ability for the open-source community to build more advanced solutions *on top* of this package instead of reinventing it
- Ability for research scientists to benefit from subclassing and customizing base components to quickly test new research ideas

## Design Proposal

`keras-nlp` will include most standard Transformer-based modules, specifically:

- Keras layer components such as Transformer encoder and decoder blocks.
- Keras task components such as masked language, span labeler and named entity recognition.
- Tensorflow operations such as beam search.
- Keras optimizer utilities such as learning rate schedules widely used.
- Data loader and preprocessing for different dataset, such as SQUAD, GLUE.

### Success criteria for keras-nlp

- Reusable and standardized components that cover the above
- Easy-to-use API
- Models run on CPU/GPU/TPU seamlessly
- State of the art performance
- Models can be readily deployed to production

### Boundaries between keras-nlp and tf.text

- `tf.text` will contain all pre-processing operations, such as WordPiece Tokenizer, n-grams, that handles strings.
- `keras-nlp` will contain modeling components that cover workflows past the tokenization stage.

### Boundaries between `keras-nlp` and TensorFlow Addons:

- Highly experimental modeling, layers, losses, etc, live in Addons (e.g. newly published research code).
- Components from Addons will graduate to Model Garden, given they get sufficient usage,
and given that they work on CPU/GPU/TPU. The API interface will remain experimental for a short time after graduation,
so as to leave us the option to make changes based on user feedback.

### Boundaries between keras-nlp and Model Garden

- End to end modeling workflow and model specific details live in Model Garden
- Model garden will re-use most of the building blocks from keras-nlp
- Components from Model Garden can graduate to keras-nlp, given they get sufficient usage,
and given that they work on CPU/GPU/TPU. The API interface should remain stable after graduation.

### Boundaries between keras-nlp and core Keras

- `keras-nlp` will contain NLP-specific components
(e.g. the `MultiHeadAttention` layer may be used outside of NLP, and thus is shipping in core Keras).
- Components from keras-nlp can graduate to Keras core, given its usage expands beyond
 natural language processing.

## Dependencies

- Tensorflow version >= 2.4
- Tensorflow datasets

## Backwards compatibility

We propose to guarantee major release backwards compatibility.

## Maintenance

The `keras-nlp` codebase will be primarily maintained by the Keras team at Google,
with help and contributions from the community. The codebase will be developed
on GitHub as part of the `keras-team` organization. The same process for tracking
issues and reviewing PRs will be used as for the core Keras repository.

## Performance Benchmark

We will set up Keras benchmark utilities to help users contribute to this repository.

Detailed design will be shared in a separate document (this document only focuses on scope).

## Questions and Discussion Topics

Please share any questions or suggestion.
