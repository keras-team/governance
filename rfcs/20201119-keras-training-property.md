# Title of RFC

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Gareth Jones (gareth@arturo.ai)                    |
| **Sponsor**   | Needs Sponsor                      |
| **Updated**   | 2020-11-19                                           |


## Objective

Improve the ability to control the `training` arg by introducing a `training` property on the base layer. 

## Motivation

It is difficult to fine-tune many kinds of models because the training
argument is difficult to access. Adding a property would make it easier to
configure modular models, e.g. segmentation models to better support 
transfer learning applications.

```python
# A keras Model with two sub networks, backbone and head
model = tf.keras.Model(...)
# model.backbone
# model.head

# -- Normal Training

# full training mode
model(x, training=True)

# -- Inference
model(x, training=False)

# -- Transfer learning (Proposed)
# Train head with fixed backbone
model.backbone.training = False
model(x, training=True)

# Finte tune the full model
model.backbone.training = True
model(x, training=True)
```

## User Benefit

Extend the Keras base layer to better support transfer learning across a diverse set of use cases with
an intuitive interface.

## Design Proposal

Introduce a `training` property to take precedence over the 5 existing cases for determining
the state of `training` in a layer.

```python
# -- except from __call__
# Training mode for `Layer.call` is set via (in order of priority):
# (1) The `training` property set on this layer, if it was explicitly set
# (2) The `training` argument passed to this `Layer.call`, if it is not None
# (3) The training mode of an outer `Layer.call`.
# (4) The default mode set by `tf.keras.backend.set_learning_phase` (if set)
# (5) Any non-None default value for `training` specified in the call
#  signature
# (6) False (treating the layer as if it's in inference)
```  

The `self.training` would only be accessed if the layer expects a training arg. To maintain backwards compatibility`
 _training_prop_set` is used to determine whether the user explicitly wanted to use the property. When instantiating a
 layer the following logic would be applied:

```python
# __init__
if self._expects_training_arg():
  self.training = self._default_training_arg() # in setter: _training_prop_set = True
  
self._training_prop_set = False
```

Possible code change to precedence

```python

# def _set_training_mode
if self._expects_training_arg:
  # (1) `training` was set
  if self._training_prop_set:
    training_mode = self.training
  # (2) `training` was passed to this `Layer.call`.
  elif self._call_arg_was_passed('training', args, kwargs):
    training_mode = self._get_call_arg_value('training', args, kwargs)
```

## Questions and Discussion Topics

* Should `_expects_training_arg` be used to determine whether the property is accessed?