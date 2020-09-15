# Keras CV

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com) |
| **Updated**   | 2020-08-27                                           |


## Objective

This document describes the scope of the [keras-cv](https://github.com/keras-team/keras-cv) package, especially:
- What use cases `keras-cv` should cover
- Boundaries between `keras-cv` and [TensorFlow Addons](https://github.com/tensorflow/addons)
- Boundaries between `keras-cv` and [TensorFlow model garden](https://github.com/tensorflow/models)
- Boundaries between `keras-cv` and [tf.keras.applications](https://keras.io/api/applications/)

## Motivation

Computer vision (CV) is a major application area for our users.
Keras on its own provides good support for image classification tasks, in particular via `tf.keras.applications`.
However, a Keras-native modeling solutions for more advanced tasks,
such as object detection, instance segmentation, etc., is still lacking.

As a result, the open-source community has rolled out many different solutions for these use cases,
made available via PyPI and GitHub. These third-party solutions are not always kept up to date, and
many still rely on the legacy multi-backend Keras. They also raise the issue of API standardization.

To fix this, we want machine learning engineers to have access to a standard Keras-native,
optimized, and well-tested set of components to build their advanced computer vision models.

This provides key user benefits:

- The package would be first-party and thus always up to date with modern best practices.
- High code quality and testing standards and strict quality control: same level of trust as core Keras
- A shared API standard across the community
- Ability for the open-source community to build more advanced solutions *on top* of this package instead of reinventing it

## Design Proposal

`keras-cv` will provide components that cover the following areas:

- Object Detection tasks.
- Instance Segmentation tasks.
- Semantic Segmentation tasks.
- Keypoint Detection tasks.
- Video Classification tasks.
- Object Tracking tasks.

Specifically, for Object Detection tasks, `keras-cv` will include most anchor-based modules:

- Common objects such as anchor generator, box matcher.
- Keras layer components such as ROI generator, NMS postprocessor.
- Keras backbone components that fills the gap from keras-applications.
- Keras losses and metrics, such as Focal loss and coco metrics.
- Data loader and preprocessing for different dataset, such as COCO.

For Semantic Segmentation tasks, `keras-cv` will include:

- Keras head components such as Atrous Spatial Pyramid Pooling (ASPP).

### Success criteria for `keras-cv`

- Cover all modeling tasks listed above
- Easy-to-use API
- Models run on CPU/GPU/TPU seamlessly
- State of the art performance
- Models can be readily deployed to production

### Boundaries between keras-cv and keras-applications

- keras-applications will be improved to include basic building blocks such as mobilenet bottleneck, that
 include feature maps
- keras-cv will depend on keras-applications for importing backbones.

### Boundaries between keras-cv and Tensorflow Addons

- Highly experimental modeling, layers, losses, etc, live in addons.
- Components from addons will graduate to keras-cv, given it incurs more usage,
 and it works in CPU/GPU/TPU. The API interface will remain experimental after graduation.

### Boundaries between keras-cv and Model Garden

- End to end modeling workflow and model specific details live in Model Garden
- Model garden will re-use most of the building blocks from keras-cv and Tensorflow Addons.
- Components from Model Garden can graduate to keras-cv, given it is widely accepted, 
 it works performant in CPU/GPU/TPU. The API interface should remain stable after graduation.

## Dependencies

- Tensorflow version >= 2.4
- Tensorflow datasets
- Keras-applications

## Backwards compatibility

We propose to guarantee major release backwards compatibility.

## Maintenance & development process

The `keras-cv` codebase will be primarily maintained by the Keras team at Google,
with help and contributions from the community. The codebase will be developed
on GitHub as part of the `keras-team` organization. The same process for tracking
issues and reviewing PRs will be used as for the core Keras repository.

## Performance benchmark

We will set up Keras benchmark utilities to help users contribute to this repository.

## Detailed Design

Detailed design will be shared in a separate document (this document only focuses on scope).

## Questions and Discussion Topics

Please share any questions or suggestion.
