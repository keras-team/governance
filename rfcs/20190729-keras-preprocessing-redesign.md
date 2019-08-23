# Keras Preprocessing API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Francois Chollet (fchollet@google.com), Frederic Branchaud-Charron (Frederic.Branchaud-Charron@usherbrooke.ca)|
| **Updated**   | 2019-08-21                                           |


## Context

`tf.data.Dataset` is the main API for data loading and preprocessing in TensorFLow. It has two advantages:

- It supports GPU prefetching
- It supports distribution via the Distribution Strategies API

Meanwhile, `keras.preprocessing` is a major API for data loading and preprocessing in Keras. It is based
on Numpy and Scipy, and it produces instances of the `keras.utils.Sequence` class, which are finite-length,
resettable Python generators that yield batches of data.

Some features of `keras.preprocessing` are highly useful and don't have straightforward equivalents in `tf.data`
(in particular image data augmentation and dynamic time series iteration).

Ideally, the utilities in `keras.preprocessing` should be made compatible with `tf.data`.
This presents the opportunity to improve on the existing API. In particular we don't have good support
for image segmentation use cases today.

Some features are also being supplanted by [preprocessing layers](https://github.com/keras-team/governance/blob/master/rfcs/20190502-preprocessing-layers.md), in particular text processing. 
As a result we may want move the current API to an API similar to Layers.


## Goals

- Unify "keras.preprocessing" and the recently-introduced [Preprocessing Layers API](https://github.com/keras-team/governance/blob/master/rfcs/20190502-preprocessing-layers.md).
- Make all features of `keras.preprocessing` compatible with `tf.data`.
- As a by-product, add required ops to TensorFlow (`tf.image`).


## Proposed changes at a high-level


- Deprecate `ImagePipelineGenerator` in favor of new `ImagePipeline` class similar to a `Sequential` model.
- Inherits from `keras.layers.PreprocessingLayer` for all image transformations.
- Deprecate `Tokenizer` class in favor of `TextVectorization` preprocessing layer.
- Replace `TimeseriesGenerator` with a function-based API.


## Detailed API changes


### ImagePipeline

#### Constructor

`ImagePipeline` inherits from `keras.model.Sequential` and takes a list of layers as inputs. In the future it will inherit from `PreprocessingStage`.

`ImagePipeline` is a preprocessing layer that encapsulate a series of image transformations. Since some of these transformations may be trained (featurewise normalization), it exposes the method `adapt`, like all other preprocessing layers.


```python

class ImagePipeline(Sequential):

    def __init__(self, layers:List[Layer]):
        ...
```

#### Example usage

```python
preprocessor = ImagePipeline([
    RandomFlip(horizontal=True),
    RandomRotation(0.2, fill_mode='constant'),
    RandomZoom(0.2, fill_mode='constant'),
    RandomTranslation(0.2, fill_mode='constant'),
    Normalization(),  # This is the same Normalization introduced in preprocessing layers
])
preprocessor.adapt(sample_data)  # optional step in case the object needs to be trained

dataset = preprocessor.from_directory(dir_name, image_size=(512, 512))
model.fit(dataset, epochs=10)
```

#### Methods

```python
def from_directory(
    directory,
    targets='inferred',
    target_mode='categorical',
    class_names='inferred',
    color_mode='rgb',
    batch_size=32,
    image_size=(255, 255),
    shuffle=True,
    seed=None,
    follow_links=False,
    validation_split=None,
    subset='training',
    subset=None):
    """Generates a Dataset from files in a directory.

    # Arguments:
        directory: Directory where the data is located.
            If `targets` is "inferred", it should contain
            subdirectories, each containing images for a class.
            Otherwise, the directory structure is ignored.
        targets: Either
            "inferred" (targets are generated from the directory structure),
            None (no targets),
            or a list of integer labels of the same size as the number of image
            files found in the directory.
        target_mode:
            - 'categorical' means that the inferred labels are
                encoded as a categorical vector (e.g. for categorical_crossentropy).
            - 'binary' means that the inferred labels (there can be only 2)
                are encoded as binary scalars (e.g. for binary_crossentropy).
        class_names: Only valid if "targets" is "inferred". This is the explict
            list of class names (must match names of subdirectories). Used
            to control the order of the classes (otherwise alphanumerical order is used).
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            Whether the images will be converted to
            have 1, 3, or 4 channels.
        batch_size: Size of the batches of data (default: 32).
        image_size: Size to resize images to after they are read from disk.
          Since the pipeline processes batches of images that must all have the same size,
          this must be provided.
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        follow_links: Whether to follow links inside
            subdirectories (default: False).
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation". Only used if `validation_split` is set.
    """

def from_dataframe(
    dataframe,
    directory=None,
    data_column='filename',
    target_column='class',
    target_mode='categorical',
    weight_column=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(255, 255),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None):
    """Generates a Dataset from a Pandas dataframe.

    # Arguments:
        dataframe: Pandas dataframe instance.
        directory: The directory that image paths refer to.
        data_column: Name of column with the paths for the input images.
        target_column: Name of column with the class information.
        target_mode:
            - 'categorical' means that the inferred labels are
                encoded as a categorical vector (e.g. for categorical_crossentropy).
            - 'binary' means that the inferred labels (there can be only 2)
                are encoded as binary scalars (e.g. for binary_crossentropy).
        weight_column: Name of column with sample weight information.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            Whether the images will be converted to
            have 1, 3, or 4 channels.
        batch_size: Size of the batches of data (default: 32).
        image_size: Size to resize images to after they are read from disk.
          Since the pipeline processes batches of images that must all have the same size,
          this must be provided.
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation". Only used if `validation_split` is set.
    """

def preview(self, data, save_to_directory=None, save_prefix=None, save_format='png'):
    """Enables users to preview the image augmentation configuration.

    # Arguments
        data: Image data. Could be strings (a list of image paths), a list of PIL image instances,
            a list of arrays, or a list of eager tensors.
        save_to_directory: Directory to save transformed images. Mandatory if not in a notebook.
            If in a notebook and this is not specified, images are displayed in-line.
        save_prefix: String, filename prefix for saved images.
        save_format: String, extension for saved images.
    """
```

**Note:** `from_arrays` is not included since it is possible to transform Numpy data simply by calling the `ImagePipeline` object (like a layer).


### Layers

The new data augmentation layers will inherit `keras.layers.Layer` and work in a similar way.

```python
Resizing(height, width)  # Resize while distorting aspect ratio
CenterCrop(height, width)  # Resize without distorting aspect ratio
Rescaling(value)  # Divide by `value`
RandomFlip(horizontal=False, vertical=False, seed=None)
RandomTranslation(amplitude=0., fill_mode='constant', fill_value=0., seed=None)
RandomRotation(amplitude=0., fill_mode='constant', fill_value=0., seed=None)
RandomZoom(amplitude=0., fill_mode='constant', fill_value=0., seed=None)
RandomBrightness(amplitude=0., seed=None)
RandomContrast(amplitude=0., seed=None)
RandomSaturation(amplitude=0., seed=None)
RandomWidth(amplitude=0., seed=None)  # Expand / shrink width while distorting aspect ratio
RandomHeight(amplitude=0., seed=None)  # Expand / shrink height while distorting aspect ratio
```

The `amplitude` argument may be:
- a positive float: it is understood as "fraction of total" (total is the current width, or height, or 180 degrees in the case `RandomRotation`). E.g. `0.2` results in variations in the [-20%, +20%] range. If larger than 1, it is rounded to one for the lower boundary (but not the higher boundary).
- a tuple of 2 positive floats: understood as a fractional range, e.g. `(0.2, 0.4)` is interpreted as the [-20%, +40%] range. The first float may not be larger than 1.

To do a random center crop that zooms in and discards part of the image, you would do:

```python
preprocessor = ImagePipeline([
  RandomZoom([0., 0.2]),
  CenterCrop(height, width),
])
```


#### Notes

- We are dropping support for ZCA whitening as it is no longer popular in the computer vision community.
- We don't have immediate support for random translations along only one axis.
- We only plan on implementing support for `data_format='channels_last'`. As such this argument does not appear in the API.


#### Example implementation

```python
class RandomFlip(PreprocessingLayer):

  def __init__(self, horizontal=False, vertical=False, seed=None):
    self.horizontal = horizontal
    self.vertical = vertical
    self.seed = seed
    self._current_seed = seed or random_value()

  def call(self, inputs, training=None):
    seed = self._current_seed
    self._current_seed += 1
    if training:
      if self.horizontal:
        inputs = tf.image.random_flip_left_right(inputs, seed=seed)
      if self.vertical:
        inputs = tf.image.random_flip_up_down(inputs, seed=seed)
    return inputs
```



#### Question: how to support image segmentation in a simple way?

**Requirements:**
- Image loading and image augmentation should be synced across inputs and targets
- It should be possible to use different standardization preprocessing (outside of augmentation) across inputs and targets

**Proposal:**

```python
# Shared spatial transformations for inputs and targets
augmenter = ImagePipeline([
    RandomRotation(0.5),
    RandomFlip(vertical=True)
])

input_pipeline = ImagePipeline([
    augmenter,
    RandomBrightness(0.2),
    RandomContrast(0.2),
    RandomSaturation(0.2),
])
target_pipeline = ImagePipeline([
    augmenter,
    OneHot(num_classes)
])

input_ds = input_pipeline.from_directory(
    input_dir, targets=None, image_size=(150, 150), batch_size=32,
    seed=123)  # This seed supercedes the per-layer seed in all transformations
target_ds = target_pipeline.from_directory(
    target_dir,  # target_dir should have same structure as input_dir.
    targets=None, image_size=(150, 150), batch_size=32, seed=123)

ds = tf.data.Dataset.zip((input_ds, target_ds))
model.fit(ds)
```


### TimeseriesGenerator

- Deprecate existing `TimeSeriesGenerator` class
- Introduce functional replacement `timeseries_dataset`:

```python
def timeseries_dataset(
      data, targets, length,
      sampling_rate=1,
      stride=1,
      start_index=0,
      end_index=None,
      shuffle=False,
      reverse=False,
      batch_size=128):
      """Utility function for generating batches of temporal data.

      This function takes in a sequence of data-points gathered at
      equal intervals, along with time series parameters such as
      stride, length of history, etc., to produce batches for
      training/validation.

      # Arguments
          data: Indexable generator (such as list or Numpy array)
              containing consecutive data points (timesteps).
              The data should be at 2D, and axis 0 is expected
              to be the time dimension.
          targets: Targets corresponding to timesteps in `data`.
              It should have same length as `data`.
          length: Length of the output sequences (in number of timesteps).
          sampling_rate: Period between successive individual timesteps
              within sequences. For rate `r`, timesteps
              `data[i]`, `data[i-r]`, ... `data[i - length]`
              are used for create a sample sequence.
          stride: Period between successive output sequences.
              For stride `s`, consecutive output samples would
              be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
          start_index: Data points earlier than `start_index` will not be used
              in the output sequences. This is useful to reserve part of the
              data for test or validation.
          end_index: Data points later than `end_index` will not be used
              in the output sequences. This is useful to reserve part of the
              data for test or validation.
          shuffle: Whether to shuffle output samples,
              or instead draw them in chronological order.
          reverse: Boolean: if `true`, timesteps in each output sample will be
              in reverse chronological order.
          batch_size: Number of timeseries samples in each batch
              (except maybe the last one).

      # Returns
          A Dataset instance.
      """
```

