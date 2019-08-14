# Keras Preprocessing API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Francois Chollet (fchollet@google.com)               |
| **Updated**   | 2019-07-29                                           |


## Context

`tf.data.Dataset` is the main API for data loading and preprocessing in TensorFLow. It has two advantages:

- It supports GPU prefetching
- It supports distribution via the Distribution Strategies API

Meanwhile, `keras.preprocessing` in a major API for data loading and preprocessing in Keras. It is based
on Numpy and Scipy, and it produces instances of the `keras.utils.Sequence` class, which are finite-length,
resettable Python generators that yield batches of data.

Some features of `keras.preprocessing` are highly useful and don't have straightforward equivalents in `tf.data`
(in particular image data augmentation and dynamic time series iteration).

Ideally, the utilities in `keras.preprocessing` should be made compatible with `tf.data`.
This presents the opportunity to improve on the existing API. In particular we don't have good support
for image segmentation use cases today.

Some features are also being subplanted by [preprocessing layers](https://github.com/keras-team/governance/blob/master/rfcs/20190502-preprocessing-layers.md), in particular text processing. 
As a result we may want to deprecate them.


## Goals

- Make all features of `keras.preprocessing` compatible with `tf.data`
- As a by-product, add required ops to TensorFlow
- Improve the ergonomy and user-friendliness of the `keras.preprocessing` APIs


## Proposed changes at a high-level

- Remove the submobules `image`, `text`, `timeseries` from the public API and expose their contents as part of the top-level `keras.preprocessing`.
- Make all classes in `keras.preprocessing` JSON-serializable.
- Deprecate `Tokenizer` class in favor of `TextVectorization` preprocessing layer
- Make image-transformation functions in `affine_transformations` submodule compatible with TensorFlow tensor inputs (accepting tensors and returning tensors).
- Improve signature of the above-mentioned functions, by using fully-spelled argument names, and fewer arguments if possible.
- Add `dataset_from_arrays`, `dataset_from_dataframe`, and `dataset_from_directory` methods in `ImageDataGenerator` (note: we could use `*_to_*` as well).
- Rename methods `flow` to `generator_from_arrays`, `flow_from_directory` to `generator_from_directory`, and `flow_from_dataframe` to `generator_to_dataframe`.
- Improve signature of the above-mentioned methods.
- Figure out how to support image segmentation use cases with `ImageDataGenerator` (open question).
- Refactor `TimeseriesGenerator` to follow a similar design as `ImageDataGenerator`. Only configuration arguments should be passed in the constructor. The data should be passed to methods such as `dataset_from_arrays`.


## Detailed API changes


### ImageDataGenerator

#### Constructor

```python
# CURRENT

def __init__(
    self,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    brightness_range=None,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    postprocessing_function=None,
    data_format='channels_last',
    validation_split=0.0,
    interpolation_order=1,
    dtype='float32'):
    """Holds configuration for image data preprocessing and augmentation.

    # Arguments
        featurewise_center: Boolean.
            Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean.
            Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_whitening: Boolean. Apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            - With `width_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `width_shift_range=[-1, 0, +1]`,
                while with `width_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-height_shift_range, +height_shift_range)`
            - With `height_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `height_shift_range=[-1, 0, +1]`,
                while with `height_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.
        shear_range: Float. Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'.
            Points outside the boundaries of the input are filled
            according to the given mode:
            - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 'nearest':  aaaaaaaa|abcd|dddddddd
            - 'reflect':  abcddcba|abcd|dcbaabcd
            - 'wrap':  abcdabcd|abcd|abcdabcd
        cval: Float or Int.
            Value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocessing_function: function that will be applied on each input.
            The function will run before the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        postprocessing_function: function that will be applied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).
        interpolation_order: int, order to use for
            the spline interpolation. Higher is slower.
        dtype: Dtype to use for the generated arrays.
    """
```

```python
# PROPOSED

def __init__(
    self,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    brightness_range=None,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    fill_value=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    resize=None,
    preprocessing_function=None,
    interpolation_order=1,
    data_format='channels_last',
    dtype='float32'):
    """Holds configuration for image data preprocessing and augmentation.

    # Arguments
        featurewise_center: Boolean.
            Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean.
            Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_whitening: Boolean. Apply ZCA whitening.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            - With `width_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `width_shift_range=[-1, 0, +1]`,
                while with `width_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-height_shift_range, +height_shift_range)`
            - With `height_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `height_shift_range=[-1, 0, +1]`,
                while with `height_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.
        shear_range: Float. Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'.
            Points outside the boundaries of the input are filled
            according to the given mode:
            - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 'nearest':  aaaaaaaa|abcd|dddddddd
            - 'reflect':  abcddcba|abcd|dcbaabcd
            - 'wrap':  abcdabcd|abcd|abcdabcd
        fill_value: Float or Int.
            Value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        resize: Tuple of int (height, width). Dimensions to resize
            images to.
        preprocessing_function: function that will be applied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation_order: int, order to use for
            the spline interpolation. Higher is slower.
        dtype: Dtype to use for the generated arrays.
    """
```

Notes:

1) The meaning of existing arguments does not change.

2) The `*_range` arguments are polymorphic (no change from current state):
- Float value should be between 0 and 1 and indicates a fraction (e.g. 0.2 indicates "-20% +20%" variation).
- Integer value N indicates a [-N +N] variation interval, where N is either pixels or degrees depending on use case.
- Tuple of 2 floats or 2 integers indicates 2 different boundaries in the intervals above.

3) Data augmentation happens on actual images, before normalization (this is more correct, and necessary in order to visualize the effect of augmentation).


#### New additions

```python
@classmethod
def from_config(self, config):
    # Returns instance

def get_config(self, config):
    # Returns JSON-serializable config dict

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

def apply(self, data, seed=None):
    # Takes an image or batch of images (PIL instance(s), array, tensor)
    # and returns a Numpy array (or tensor) of standardized and augmented data.
    # Note: at this time we have the methods `standardize` and `random_transform`;
    # this new one does both at the same time.
```

#### Arrays methods

```python
# CURRENT

def flow(
    self,
    x,
    y=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset=None):
    """Returns Sequence instance (generator).

    # Arguments
        x: Input data. Numpy array of rank 4 or a tuple.
            If tuple, the first element
            should contain the images and the second element
            another numpy array or a list of numpy arrays
            that gets passed to the output
            without any modifications.
            Can be used to feed the model miscellaneous data
            along with the images.
            In case of grayscale data, the channels axis of the image array
            should have value 1, in case
            of RGB data, it should have value 3, and in case
            of RGBA data, it should have value 4.
        y: Labels.
        batch_size: Int (default: 32).
        shuffle: Boolean (default: True).
        sample_weight: Sample weights.
        seed: Int (default: None).
        save_to_dir: None or str (default: None).
            This allows you to optionally specify a directory
            to which to save the augmented pictures being generated
            (useful for visualizing what you are doing).
        save_prefix: Str (default: `''`).
            Prefix to use for filenames of saved pictures
            (only relevant if `save_to_dir` is set).
        save_format: one of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
    """
```

```python
# PROPOSED

def array_to_generator(
    self,
    data,
    targets=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    validation_split=None,
    subset='training'):
    """Returns Sequence instance (generator).

    # Arguments
        data: Input data, rank-4 Numpy array.
        targets: Labels.
        batch_size: Int (default: 32).
        shuffle: Boolean (default: True).
        sample_weight: Sample weights.
        seed: Int (default: None).
        validation_split: Float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
    """

def array_to_dataset(
    self,
    data,
    targets=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    validation_split=None,
    subset='training'):
    """Returns Dataset instance.

    # Arguments
        data: Input data, rank-4 Numpy array.
        targets: Labels.
        batch_size: Int (default: 32).
        shuffle: Boolean (default: True).
        sample_weight: Sample weights.
        seed: Int (default: None).
        validation_split: Float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
    """
```

#### Directory methods

```python
# CURRENT

def flow_from_directory(
    self,
    directory,
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    subset=None,
    interpolation='nearest'):
    """Takes the path to a directory & generates batches of augmented data.

    # Arguments
        directory: string, path to the target directory.
            It should contain one subdirectory per class.
            Any PNG, JPG, BMP, PPM or TIF images
            inside each of the subdirectories directory tree
            will be included in the generator.
            See [this script](
            https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
            for more details.
        target_size: Tuple of integers `(height, width)`,
            default: `(256, 256)`.
            The dimensions to which all images found will be resized.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            Whether the images will be converted to
            have 1, 3, or 4 channels.
        classes: Optional list of class subdirectories
            (e.g. `['dogs', 'cats']`). Default: None.
            If not provided, the list of classes will be automatically
            inferred from the subdirectory names/structure
            under `directory`, where each subdirectory will
            be treated as a different class
            (and the order of the classes, which will map to the label
            indices, will be alphanumeric).
            The dictionary containing the mapping from class names to class
            indices can be obtained via the attribute `class_indices`.
        class_mode: One of "categorical", "binary", "sparse",
            "input", or None. Default: "categorical".
            Determines the type of label arrays that are returned:
            - "categorical" will be 2D one-hot encoded labels,
            - "binary" will be 1D binary labels,
                "sparse" will be 1D integer labels,
            - "input" will be images identical
                to input images (mainly used to work with autoencoders).
            - If None, no labels are returned
              (the generator will only yield batches of image data,
              which is useful to use with `model.predict_generator()`).
              Please note that in case of class_mode None,
              the data still needs to reside in a subdirectory
              of `directory` for it to work correctly.
        batch_size: Size of the batches of data (default: 32).
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None).
            This allows you to optionally specify
            a directory to which to save
            the augmented pictures being generated
            (useful for visualizing what you are doing).
        save_prefix: Str. Prefix to use for filenames of saved pictures
            (only relevant if `save_to_dir` is set).
        save_format: One of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        follow_links: Whether to follow symlinks inside
            class subdirectories (default: False).
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
        interpolation: Interpolation method used to
            resample the image if the
            target size is different from that of the loaded image.
            Supported methods are `"nearest"`, `"bilinear"`,
            and `"bicubic"`.
            If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
            supported. If PIL version 3.4.0 or newer is installed,
            `"box"` and `"hamming"` are also supported.
            By default, `"nearest"` is used.
    """
```

```python
# PROPOSED

def generator_from_directory(
    self,
    directory,
    targets='inferred',
    target_mode='categorical',
    class_names='inferred',
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=None,
    follow_links=False,
    validation_split=None,
    subset='training'):
    """Takes a directory and returns a generator of images.

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
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        follow_links: Whether to follow links inside
            subdirectories (default: False).
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation". Only used if `validation_split` is set.
    """

def dataset_from_directory(
    self,
    directory,
    targets='inferred',
    target_mode='categorical',
    class_names='inferred',
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=None,
    follow_links=False,
    validation_split=None,
    subset='training'):
    """Takes a directory and returns a Dataset yielding batches of images.

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
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        follow_links: Whether to follow links inside
            subdirectories (default: False).
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: One of "training" or "validation". Only used if `validation_split` is set.
    """
```

Notes:
- Resizing (`target_size`) is moved to constructor (`resize`).
- `validation_split` is moved from constructor to here.

#### Question: how to support image segmentation in a simple way?

**Requirements:**
- Image loading and image augmentation should be synced across inputs and targets
- It should be possible to use different standardization preprocessing (outside of augmentation) across inputs and targets

**Proposal:** (TBD)

```python
augmenter = ImageDataGenerator(...)
input_preprocessor = ImageDataGenerator(...)  # or write your own function
target_preprocessor = ImageDataGenerator(...)  # or write your own function

input_ds = augmenter.dataset_from_directory(input_dir, targets=None, seed=123)
target_ds = augmenter.dataset_from_directory(target_dir, targets=None, seed=123)  # target_ds should have same structure.
input_ds = input_ds.map(input_preprocessor.standardize)  # `standardize` could work on tensors.
target_ds = input_ds.map(target_preprocessor.standardize)

ds = tf.data.Dataset.zip((input_ds, target_ds))

model.fit(ds)
```



#### Dataframe methods

```python
# CURRENT

def flow_from_dataframe(
    self,
    dataframe,
    directory=None,
    x_col='filename',
    y_col='class',
    weight_col=None,
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset=None,
    interpolation='nearest',
    validate_filenames=True,
    **kwargs):
```

```python
# PROPOSED

def generator_from_dataframe(
    self,
    dataframe,
    directory=None,
    data_column='filename',
    target_column='class',
    weight_column=None,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    subset=None):

def dataset_from_dataframe(
    self,
    dataframe,
    directory=None,
    data_column='filename',
    target_column='class',
    weight_column=None,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    subset=None):
```

### Utility functions

These are functions located at `affine_transformations.py`.

Proposed changes: API simplification & standardization, and extension of supported types.

- All functions should work on either single images or batches of images
- All functions should work on either Numpy arrays or TF tensors
- `row_axis=1, col_axis=2, channel_axis=0` -> `data_format='channels_last`
- `x` -> `images`
- `rg`, `intentity`, etc -> `amplitude`. Same name across all functions. May be a tuple.
- Replace multiple arguments such as `wrg`, `hrg` with a single `amplitude` argument which may be a tuple
- `cval` -> `fill_value`


Examples:

```python

# CURRENT

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0., interpolation_order=1):

# PROPOSED

def random_rotation(images, amplitude, data_format='channels_last',
                    fill_mode='nearest', fill_value=0., interpolation_order=1):
```


### TimeseriesGenerator

Details for this API are TBD.

It isn't entirely clear how to proceed. Maybe we should abandon the class and introduce utility functions with the same features.




