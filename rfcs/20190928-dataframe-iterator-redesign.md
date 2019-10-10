# DataFrameIterator API redesign

| Status        | Proposed                              |
|:--------------|:--------------------------------------|
| **Author** | Rodrigo Agundez (rragundez@gmail.com) |
| **Updated**   | 2019-09-28                            |

## Objective

Change the design of `DataFrameIterator` class to be more intuitive and flexible such that the `flow_from_dataframe` if `ImageDataGenerator` method handles:

- regression,
- binary classification,
- multi-class classification,
- multi-label classification,
- object detection,
- segmentation,
- Autoencoder,
- image noise removal,
- upscaling image to a higher resolution,
- custom semi-supervised learning use case,
- multi-task learning using any combination of the above use cases,
- multi-input with any of the above use cases,
- moonshot: multi-input where each input can have different data augmentation,

where the input data type is image data.

I did an initial refactoring in my [fork's branch df-iterator-redesign](https://github.com/rragundez/keras-preprocessing/tree/df-iterator-redesign/keras_preprocessing/image), with some [notebook examples](https://github.com/rragundez/keras-preprocessing/tree/df-iterator-redesign/keras_preprocessing/image/notebook_examples ) where many but not all use cases and functionality is covered. NOTE: I have no intention of making a PR with this, but it serves as an example of what can be achieved.

## Motivation

Currently the `DataFrameIterator` is limited by its design, but it can potentially be used to address numerous use cases now and new ones in the future.

My proposal is based on two concepts:

1. A single input or output can be linked to a single column.
2. The user should tell us how to transform the data in that column. This is mostly applicable for outputs. For example:

    - For an input column: is it an image file (png, jpg, tiff, etc.)?, is it an image in a numpy array format (.npy)? or another format? (this might be better handled automatically by detecting the file extension though.)
    - For an output column: how do we transform this column (sparse, categorical, no transformation, bounding box, as image, etc.)?

I believe that by adopting these 2 concepts the objectives can be achieved. It does require a great deal of refactoring and change the way the API is used.

This change will be extremely valuable for anyone with a use case with image data and `flow_from_dataframe` can become the to-go method for any image use case.

> Note: I actually think if this is implemented properly `flow_from_directory` should be deprecated/removed, as it only satisfies a subset of use cases that `flow_from_dataframe` addresses.

I identify the following problems with the current API:

- There is no support if images are in any file format other than picture files (png, jpg, tiff, etc.) for example, as Numpy arrays (npy and npz). Not only is there no support for it, but is very difficult to add.
- Setting a seed happens at a global level which I believe is a bad practice. Instead a random state should be created or obtained. Then a random process like data augmentation can be repeated by re-using that random state.
- Because of the previous point, any use case where the input and output are images (segmentation, image noise removal, upscaling image to a higher resolution, etc.) currently there is the unnecessary need to create two identical image generators and link them via `zip`. I show in this document how I think it should work instead.
- There is currently no support for bounding boxes or landmarks. It is also difficult to add.
- There is no general support of all use cases for multi-task learning. Currently there is only the possibility of using the `raw` mode for each task. This means the user has to perform the transformations necessary and then add them to the DataFrame. Example: input the image of a person and output the gender (binary), the race (multi-class) and the age (regression).
- Binary mode does the same as sparse mode. In this case I would keep the sparse mode and remove the binary mode as sparse mode encapsulates the binary mode.

Related work: I've been adding new use cases to the functionality of `DataFrameIterator`, in particular multi-label and multi-output, but I don't see how to add all the other functionality with the current API. I already started working on the refactoring in my fork of the `keras-preprocessing` repository under the branch `df-iterator-redesign`.

## User Benefit

The addition of many use cases and the easy way to extend the API to handle more. Of course the changes proposed here will not be backwards compatible therefore there can be a warning period with a blog post that shows how to take each current use case to the new API and what new use cases are possible. Some of the headlines can be something like: Update your code to the new ImageDataGenerator API, ImageDataGenerator tackles new image use cases with new API, or a headline if we release one use case at a time.

## Design Proposal

### DataFrameIterator

#### Constructor

```python

class DataFrameIterator:

    def __init__(self,
                 dataframe,
                 input_columns,
                 output_columns=None,
                 weight_column=None,
                 output_modes=None,
                 input_image_sizes=(255, 255),
                 output_image_sizes=None,
                 input_color_modes='rgb',
                 output_color_modes=None,
                 image_data_generator=None,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 validation_split=None,
                 subset=None):
        """Generates a Dataset from a Pandas dataframe.

        # Arguments:
            dataframe: Pandas dataframe instance.
            input_columns: String or list of strings. If string, the column name
                containing the absolute paths to the images. If list of strings,
                the column names that contain the absolute paths to the images
                for a multi-input architecture.
            output_columns: None, string or list of strings specifying the column
                or columns that contain the output or outputs information, for
                a single output or multi-output architecture respectively.
                If None, no output is returned which is useful to use in
                `model.predict_generator()`.
                Default: None.
            weight_column: Name of column with sample weight information.
                Default: None.
            output_modes: None, string or dictionary with column names as keys and modes as
            values, specifying the mode to apply to the columns given in output_columns. The following modes are available:
                - "categorical"
                - "sparse"
                - "image"
                - "bounding_box"
                - "landmark"
                - None
                Default: None.
            input_image_sizes: Tuple or dictionary with column names as keys and tuples sizes as values, specifying size to use to resize images coming for the input columns.
                Default: (255, 255).
            output_image_sizes: None, tuple or dictionary with column names as keys and tuples sizes as values, specifying size to use to resize images coming for the output columns. If None and `image_input_shapes` is tuple then it will take the same value as `image_input_shapes`.
                Default: None.
            input_color_modes: String or dictionary with column names as keys and modes as values, specifying the color mode for the input columns. The following modes are available:
                - "grayscale"
                - "rgb"
                - "rgba"
                Default: "rgb".
            output_color_modes: None, string or dictionary with column names as keys and modes as values, specifying the color mode for the output columns. If None and `input_color_modes` is string then it will take thae same value as `input_color_modes`. The following modes are available:
                - "grayscale"
                - "rgb"
                - "rgba"
                Default: None.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data.
            seed: Optional random seed for shuffling and transformations.
            validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.
            subset: One of "training" or "validation". Only used if `validation_split` is set.
        """
```

> Note A: I prefer to use plural, `input_columns` and `output_columns`, instead of `input_column` and `output_column` to keep consistency with the [`Model` class](https://keras.io/models/model/) signature which has `inputs` and `outputs`.

> Note B: I also think we should not use the word `target(s)` in this documentation, but `output(s)`. I believe this is more consistent with the fact that we do not know if the user will actually use the output(s) as targets or if it will apply some extra processing to the output to convert it to targets. I think this should also be applied to the code (naming variables).

#### Example usage

Suppose the table below is stored in a Pandas DataFrame variables call `df`.

| img_path     | img_np_path | regression |  binary | multi_class | multi_label | object_detection   | img_path_extra     | regression_extra |
|--------------|-------------|------------|---------|-------------|-------------|--------------------|--------------------|------------------|
|  /tmp/5.jpg  | /tmp/5.npy  | 5          | cat     | dog         | [cat]       | [[x, y, h, w]]     | /tmp/extra/5.jpg   | 92               |
|  /tmp/13.jpg | /tmp/13.npy | 6          | cat     | cat         | [dog, cat]  | [[x1, y1, h1, w1], [x2, y2, h2, w2]]     | /tmp/extra/13.jpg  | 24               |
|  /tmp/12.jpg | /tmp/12.npy | 7          | dog     | cat         | [dog, cat]  | [[x, y, h, w]]     | /tmp/extra/12.jpg  | 38               |
|  /tmp/10.jpg | /tmp/10.npy | 7          | dog     | dog         | [dog]       | []     | /tmp/extra/10.jpg  | 47               |
|  /tmp/4.jpg  | /tmp/4.npy  | 2          | cat     | cat         | [dog]       | [[x, y, h, w]]     | /tmp/extra/4.jpg   | 74               |


> Note: in most example I ommit using the `image_data_generator` argument for brevity, but the same use case should hold if it is used.

**Inference**

Single input.
```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path'
)
```

Multi-input.
```python
df_iter = DataFrameIterator(
    df,
    input_columns=['img_path', 'img_path_extra']
)
```

**Regression**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='regression'
)
```

Input image as a numpy array file.

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_np_path',
    output_columns='regression'
)
```

**Binary classification**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='binary',
    output_modes={'binary': 'sparse'}
)
```

**Multi-class classification**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='multi_class',
    output_modes={'multi_class': 'sparse'}
)
```

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='multi_class',
    output_modes={'multi_class': 'categorical'}
)
```

**Multi-label classification**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='multi_label',
    output_modes={'multi_label': 'categorical'}
)
```

**Object detection**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='object_detection',
    output_modes={'object_detection': 'bounding_box'}
)
```

**Segmentation | Image noise removal | Upscaling to higher resolution**

No data augmentation, without the need for the user to set a seed for yield correct pairs of images.
```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='img_path_extra',
    output_modes={'img_path_extra': 'image'}
)
```

Same data augmentation to input and output image, without the need for the user to set a seed.
```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='img_path_extra',
    output_modes={'img_path_extra': 'image'},
    image_data_generator=img_generator
)
```

**Autoencoder: output same as input**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='img_path',
    output_modes={'img_path': 'image'}
)
```

**Custom semi-supervised learning use case**
```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns='img_path',
    output_modes={'img_path': custom_target_processing_function}
)
```


**Multi-output: regression & regression**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns=['regression', 'regression_extra']
)
```

**Multi-output: regression & binary classification**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns=['regression', 'binary'],
    output_modes={'binary': 'sparse'}
)
```

**Multi-output: regression & multi-class classification**

```python
df_iter = DataFrameIterator(
    df,
    input_columns='img_path',
    output_columns=['regression', 'multi_class'],
    output_modes={'multi_class': 'categorical'}
)

```

**Multi-output: multi-class & multi-label classification**

```python
df_iter = DataFrameIterator(
    df,
    input_column='img_path',
    output_columns=['multi_class', 'multi_label'],
    output_modes={'multi_class': 'categorical', 'multi_label': 'categorical'}
)
```

**Multi-input & regression**

For both inputs data augmentation is not applied and both input tensors have dimensions (255, 255, 3).
```python
df_iter = DataFrameIterator(
    df,
    input_column=['img_path', 'img_path_extra'],
    output_columns='regression',
    color_modes='rgb',
    input_image_sizes=(255, 255)
)
```

For both inputs data augmentation is applied equally, first input tensor has size (10, 10, 3) and second input tensor has size (20, 20, 1).
```python
df_iter = DataFrameIterator(
    df,
    input_column=['img_path', 'img_path_extra'],
    output_columns='regression',
    color_modes={'img_path': 'rgb', 'img_path_extra': 'grayscale'},
    input_image_sizes={'img_path': (10, 10), 'img_path_extra': (20, 20)},
    image_data_generator=img_generator
)
```

### High-level changes needed to the currrent source code

#### Output transformations

- Current: transformations are embedded directly into the `Iterator` base class and the `DataFrameIterator` class.
- Proposal: a module `data_transformations.py` which handles the logic. Which has almost all private methods except for a functions `transform_output` and `transform_batch`. `transform_output` applies the logic necessary to obtain the information to create a batch (applied in `DataFrameIterator`). `transform_batch` has the logic to actually create the batch (applied in `Iterator`).


#### DataFrameIterator and Iterator

- The logic should be changed to use the `transform_output` and `transform_batch`.
- There should be a loop or similar that accumullates the batches from each input or output.
- Data agumentation should be replicated within the creation of a batch for any input or output of image type.

### Impact

- The UX and usability learning curve will be higher to the current one. Therefore good documentation and examples are extremely important.
- I believe that performance should not be impacted by this change as each use case can be addressed and optimized separately in the `data_transformations.py` module.
- Dependencies should stay the same.
- It won't be backwards compatible at all. A plan to communicate the change needs to be carefully thought with incremental steps: announcement, deprecation warning, change to new API.

## Questions and Discussion Topics

- Should I use names of classes and methods based the current API or based on [20190729-keras-preprocessing-redesign.md](https://github.com/keras-team/governance/blob/master/rfcs/20190729-keras-preprocessing-redesign.md)? I currently base it on the current API.

- Do we need to think about images using separate data augmentation logics, in the case of multi-input or multi-output? I think this is a stretch for this RFC.

- Do we want the `data_transformations.py` array/tensor transformations to be already done in `tensorflow` land or `numpy` land?

- For image classification use cases, we need a clear distinction/definition not only in the documentation but also in the code between `labels`, `classes`, `label_indices`, `class_indices` and `class_to_index`. Where `labels` and `classes` are only strings, and `label_indices` and `class_indices` are defined as integers. `class_to_index` is a dictionary mapping tthe class string to the integer index.

- There is currently functionality, which I'm not sure is wise to include in the redesign as the API changes focus to a more broad use cases. For example:

    - currently the userer can give a directory and provide relative paths. Supporting this feature greatly affects the logic for segmentation, multi-input and multi-output.
    - currently the user can give a list of classes to use. I think it can also be an option that the user filters the df. Is one line in pandas.
    - introduce a warning or exception if validation set has a class which is not in the training set?

- Suppose the user wants the same image as input and output but the wants the color modes and sizes to be different. For example, for different color modes, training a model which takes grayscale images to color images. For different sizes, imagine training a model which takes a low resolution image to a higher resolution. With the current proposal the user will have to create the grayscale image and the down-sampled images and create an extra column with the paths too them. This can be solved by having a `output_color_modes` and `output_image_sizes` parameters, or by allowing the user to optionally specify these parameters when using the output mode `image`.

- I think we should add 2 mote color modes: `grayscale_16bit` and `grayscale_32bit`. I recently added support for them but some parts are a bit hacky as the code has to "guess" the desired format. Currently this happens because internally the arrays are in `float32` such that the original mode information is lost. Should I submit another RFC for this? I think so.

- Do we still want to provide the functionality of saving to disk the augmented images? This can become a bit complex to solve for all types of transformation and image data types, see [PR 244](https://github.com/keras-team/keras-preprocessing/pull/244).
