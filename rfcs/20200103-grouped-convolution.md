# Keras Grouped Convolution

| Status        | Proposed                          |
| :------------ | :-------------------------------- |
| **Author(s)** | Lukas Geiger (lukas@plumerai.com) |
| **Updated**   | 2020-01-03                        |

## Objective and Motivation

This proposal aims to add support for grouped convolution to the `keras.layers.Conv{1,2,3}D` API.

Grouped convolutions are a special case of sparsely connected convolutions and have been successfully used in [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf), [ShuffleNet](https://arxiv.org/pdf/1707.01083.pdf), [CondenseNet](https://arxiv.org/pdf/1711.09224.pdf) and many follow up works.

This feature is supported by other frameworks like [Caffe](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html) or [PyTorch](https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d) and has been requested many times in the TensorFlow and Keras community (https://github.com/tensorflow/tensorflow/issues/3332, https://github.com/tensorflow/tensorflow/issues/11662, https://github.com/keras-team/keras/issues/3334, https://github.com/tensorflow/tensorflow/pull/10482).

Grouped convolutions are now supported in TensorFlow core via [CUDNN 7](https://developer.nvidia.com/cudnn) on GPUs (https://github.com/tensorflow/tensorflow/pull/25818) and inside XLA compiled functions on CPU and GPUs, but there is no support via the Keras layers API yet. Adding support for grouped convolutions should make it easier for users to implement above mentioned models with TensorFlow and Keras in a fast way.

## Design Proposal

This proposal aims to add a `groups` argument to `keras.layers.Conv{1,2,3}D` which defines the number of channel groups similar to how other frameworks implement it in their high-level APIs.

Since TensorFlow has native support for group convolutions, there are no changes required to the `keras.layers.Conv{1,2,3}D.call` methods, thus this proposal has no impact on the performance of standard convolutions in the Keras API and only requires small changes to `__init__` and `build` methods.

```diff
--- a/tensorflow/python/keras/layers/convolutional.py
+++ b/tensorflow/python/keras/layers/convolutional.py
@@ -77,6 +77,16 @@ class Conv(Layer):
       the dilation rate to use for dilated convolution.
       Currently, specifying any `dilation_rate` value != 1 is
       incompatible with specifying any `strides` value != 1.
+    groups: Integer, the number of channel groups controlling the connections
+      between inputs and outputs. Input channels and `filters` must both be
+      divisible by `groups`. For example,
+        - At `groups=1`, all inputs are convolved to all outputs.
+        - At `groups=2`, the operation becomes equivalent to having two
+          convolutional layers side by side, each seeing half the input
+          channels, and producing half the output channels, and both
+          subsequently concatenated.
+        - At `groups=input_channels`, each input channel is convolved with its
+          own set of filters, of size `input_channels / filters`
     activation: Activation function to use.
       If you don't specify anything, no activation is applied.
     use_bias: Boolean, whether the layer uses a bias.
@@ -106,6 +116,7 @@ class Conv(Layer):
                padding='valid',
                data_format=None,
                dilation_rate=1,
+               groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
```

## Questions and Discussion Topics

### Is it justified to become part of the core API?

The amount of issues asking for this feature might justify an addition to the core API. An alternative would be to add support for it to [TensorFlow Addons](https://github.com/tensorflow/addons). Since this is an addition do an existing layer that is supported by many other frameworks and not a new layer type I propose to bypass TensorFlow Addons in this case.

### Grouped Convolutions on CPU

Support for grouped convolutions is currently not available on CPUs outside of XLA compiled functions (https://github.com/tensorflow/tensorflow/issues/29005). I personally don't think this is a major problem, since already now not all features of `keras.layers.Conv{1,2,3}D` are supported on CPUs (e.g. `data_format='channels_first'`) and a workaround using `@tf.function(experimental_compile=True)` exists on CPU only machines.
