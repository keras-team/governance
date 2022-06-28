# Keras Style Guide

This style guide contains information needed to write code in the style of the Keras
codebase.

### Layer-ish Functions

When writing Keras code it is common to create functions that apply a logical block of
deep learning code, such as a ResNet block.  We write our blocks using a closure pattern
that makes the behavior of these blocks mirror Keras layers.

Bad:

```python
def dense_block(x, blocks, name=None):
    """A dense block.
    Args:
      blocks: integer, the number of building blocks.
      name: string, block label.
    Returns:
      a function that takes an input Tensor representing a DenseBlock.
    """
    if name is None:
        name = f"dense_block_{backend.get_uid('dense_block')}"

    for i in range(blocks):
        x = ConvBlock(32, name=f"{name}_block_{i}")(x)
    return x

# Usage:
x = dense_block(x, 32)
```

Good:

```python
def DenseBlock(blocks, name=None):
    """A dense block.
    Args:
      blocks: integer, the number of building blocks.
      name: string, block label.
    Returns:
      a function that takes an input Tensor representing a DenseBlock.
    """
    if name is None:
        name = f"dense_block_{backend.get_uid('dense_block')}"

    def apply(x):
        for i in range(blocks):
            x = ConvBlock(32, name=f"{name}_block_{i}")(x)
        return x

    return apply

# Usage:
x = DenseBlock(32)(x)
```

### Variable names

Make sure to use fully-spelled out variable names. Do not use single-letter variable names.
Do not use abbreviations unless they're completely obvious (e.g. `num_layers` is ok).

This is bad:

```python
m = get_model(u=32, d=0.5)
```

This is good:

```python
model = get_model(units=32, dropout_rate=0.5)
```

### Imports

Import modules, not individual objects. In particular, don't import individual layers. Typically
you should import the following:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Then access objects from these modules:

```python
tf.Variable(...)
tf.reshape(...)
keras.Input(...)
keras.Model(...)
keras.optimizers.Adam(...)
layers.Layer(...)
layers.Conv2D(...)
```

Note: do **not** use `import keras`. Use `from tensorflow import keras` instead.
