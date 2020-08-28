# Keras CV

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com) |
| **Updated**   | 2020-08-27                                           |


## Objective

We aim at describing the scope of [keras-cv](https://github.com/keras-team/keras-cv), especially:
- What areas should keras-cv include
- Boundaries between keras-cv and [tensorflow addons](https://github.com/tensorflow/addons)
- Boundaries between keras-cv and [tensorflow model garden](https://github.com/tensorflow/models)
- Boundaries between keras-cv and [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

## Motivation

Computer vision (cv) has been an essential component in many machine learning solutions. Keras has been widely
used in many image classification tasks. However, a Keras-native modeling solutions for more advanced tasks,
such as object detection, instance segmentation, etc., is still lacking. 

## User Benefit

We hope to help machine learning engineers benefit from re-using Keras native, optimized, and well-tested
components to build their models. We hope to help research scientists benefit from subclassing and re-composing
Keras primitive components to test new research ideas. 

## Design Proposal

We propose keras-cv to provide components that cover the following areas:
- Object Detection tasks.
- Instance Segmentation tasks.
- Semantic Segmentation tasks.
- Keypoint Detection tasks.
- Video Classification tasks.
- Object Tracking tasks.

Specifically, for Object Detection tasks, we propose keras-cv to include most anchor-based modules:
- Common objects such as anchor generator, box matcher.
- Keras layer components such as ROI generator, NMS postprocessor.
- Keras backbone components that fills the gap from keras-applications.
- Keras losses and metrics, such as Focal loss and coco metrics.
- Data loader and preprocessing for different dataset, such as COCO.

For Semantic Segmentation tasks, we propose keras-cv to include:
- Keras head components such as Atrous Spetial Pyramid Pooling (ASPP).

Criteria for keras-cv:
- Widely accepted building components that serve various modeling tasks.
- Tasks that improves model training.
- Operations that will work in CPU/GPU/TPU.

Boundaries between keras-cv and keras-applications:
- keras-applications will be improved to include basic building blocks such as mobilenet bottleneck, that
 include feature maps
- keras-cv will depend on keras-applications for importing backbones.

Boundaries between keras-cv and Tensorflow Addons:
- Highly experimental modeling, layers, losses, etc, live in addons.
- Components from addons will graduate to Model Garden, given it incurs more usage,
 and it works in CPU/GPU/TPU. The API interface will remain experimental after graduation.

Boundaries between keras-cv and Model Garden:
- End to end modeling workflow and model specific details live in Model Garden
- Model garden will re-use most of the building blocks from keras-cv
- Components from Model Garden can graduate to keras-cv, given it is widely accepted, 
 it works performant in CPU/GPU/TPU. The API interface should remain stable after graduation.

Dependencies:
- Tensorflow version >= 2.4.
- Tensorflow datasets.
- Keras-applications

Backwards compatibility:
We propose to guarantee major release backwards compatibility.

Maintenance:
This repository will be maintained by Keras team.

Performance Benchmark:
We will set-up Keras benchmark utilities to help users contribute to this repository.

## Detailed Design
Detailed design will be separate from scoping.

## Questions and Discussion Topics