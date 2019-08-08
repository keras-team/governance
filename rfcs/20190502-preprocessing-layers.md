# Keras Preprocessing Layers

| Status        | Accepted      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Mark Omernick (momernick@google.com), Stan Bileschi (bileschi@google.com), Kester Tong (kestert@google.com), Francois Chollet (fchollet@google.com) |
| **Updated**   | 2019-05-21                                           |


## Objective

We aim at providing additional Keras layers to handle [data preprocessing operations](https://en.wikipedia.org/wiki/Data_pre-processing)
such as text vectorization, data normalization, and data discretization (binning).
These operations are currently handled separately from a Keras model via utilities such
as those from `keras.preprocessing`.

These new layers will allow users to include data preprocessing directly in their Keras model, so as to create models that map raw data (such as uint8 tensors for images, or string tensors for text) to predictions.


## Key benefits

Including preprocessing layers in the Keras model means that the same preprocessing steps will be performed when that model is exported and used in serving.
It also means the steps will be part of the model when the model is saved and loaded as part of another model.

This presents the following advantages:

- Model portability (encapsulation for sharing models). With PreprocessingLayers, your Keras Model contains all the preprocessing it requires. If another user wishes to use your model in a different workflow, there is no risk of incorrect preprocessing. Models will be more end-to-end.
- Serving reliability. The Model object will contain everything you expect to be done at serving time.
- Simpler optimization using tf.data and tf.Transform. By providing simple, well defined building blocks for preprocessing, we simplify the process of using tf.data and tf.Transform to optimize preprocessing steps. Users can offload computation of vocabularies, quantiles and mean and variance, to tf.Transform.  They can also use tf.data to move data preprocessing in training off the critical path. The preprocessing layer API is designed to make both of these easy and simple.

In particular, we expect preprocessing layers to make it easier to serve models in TF.js or in mobile applications. It will also reduce the risk that benchmarks of Keras applications use incorrect preprocessing and subsquently publish invalid findings.


## Design overview

### End-to-end workflow overview

Case where a user has a single preprocessing layer to do image normalization.

```python
normalization = keras.layers.Normalization(axis=-1)
normalization.adapt(data_sample)

model = keras.Sequential([
    normalization,
    keras.applications.ResNet50(weights=None),
])
model.fit(data, targets, epochs=10)
```

Case where a user has a single preprocessing layer to do text vectorization where each input sample is encoded as a sequence of word indices.

```python
vectorization = keras.layers.TextVectorization(mode='int')
vectorization.adapt(data_sample)

model = keras.Sequential([
    vectorization,
    keras.layers.Embedding(128),  # The number of int indices is not specified since it is inferred.
    keras.layers.LSTM(32),
    keras.layers.Dense(10, activation='softmax'),
])
model.fit(data, targets, epochs=10)
```

Case where a user has a single preprocessing layer to do text vectorization where each input sample is encoded as a dense vector of TF-IDF scores.

```python
vectorization = keras.layers.TextVectorization(mode='tfidf')
vectorization.adapt(data_sample)

model = keras.Sequential([
    vectorization,
    keras.layers.Dense(10, activation='softmax'),
])
model.fit(data, targets, epochs=10)
```

Case where a user chains a a normalization step with a discretization step.

```python
normalization = keras.layers.Normalization()
discretization = keras.layers.Discretization()
preprocessing_stage = keras.layers.PreprocessingStage([normalization,
                                                       discretization])
preprocessing_stage.adapt(data_sample)

model = keras.Sequential([
    preprocessing_stage,
    keras.layers.Dense(10, activation='softmax'),
])
model.fit(data, targets, epochs=10)
```


### Base class: `PreprocessingLayer`

All preprocessing layers inherit from a base class: `PreprocessingLayer`, which itself inherits from `Layer`.

This class presents a few key differences compared to regular layers:

**Separate training mechanism**

The internal state of a `PreprocessingLayer` is not affected by backpropagation: all of its weights are non-trainable. A `PreprocessingLayer` has to be trained in a separate step, as follow:

```python
preprocessing_layer.adapt(data_sample)
```

**Possible non-differentiability**

Processing layers extend Keras by allowing preprocessing to be part of the model. Unlike existing layers, these computations are not always differentiable, e.g. both `Discretize` and `VectorizeText` are non-differentiable.

As a result, all preprocessing layers are treated as frozen when used as part of a model. In addition, if a non-differentiable layer is used in the middle of a model (rather than at the start), the model will raise an exception related to differentiability when trying to compute gradients (e.g. as part of `fit`).


### New layers

- `PreprocessingLayer` base class: implements shared logic, in particular the `adapt` method for setting the state of the layer.
- `PreprocessingStage` class: makes it possible to chain multiple preprocessing layers together while training them in one single `adapt` call (by doing cascading training of the underlying layers).
- `Normalization`: normalizes data feature-wise by subtracting the mean of some sample dataset and dividing by the variance.
- `Discretization`: transforms continuous data into one-hot encoded binary vectors representing the different "bins" that the continuous data belongs to.
- `TextVectorization`: transforms string data into either dense vectors (e.g. TF-IDF transform) or sequences of token indices (e.g. to be passed to an `Embedding` layer).


## Design details

### Detailed layer signatures

#### PreprocessingLayer

```python
def adapt(self, data, reset_state=True):
    """Fits the state of the preprocessing layer to the data being passed.

    Arguments:
        data: The data to train on. It can be passed either as a tf.data Dataset,
            or as a numpy array (or a dict or list of arrays in case of multi-input
            preprocessing stages).
        reset_state: Optional argument specifying whether to clear the state of the
            layer at the start of the call to `adapt`, or whether to start from
            the existing state. This argument may not be relevant to all
            preprocessing layers: a subclass of PreprocessingLayer may chose to
            only implement `adapt(self, data)`.
    """
```

#### PrepocessingStage

There are two ways to instantiate a `PrepocessingStage` layer: either `Sequential` style (pass a list of preprocessing layer instances) or Functional style (pass the inputs and outputs of a DAG of preprocessing layers).

If any layer other than `PreprocessingLayer` instances is included in a `PrepocessingStage`, these layers will be treated as frozen both during `adapt` and later during `fit`.


#### Normalization

```python
def __init__(self, axis=-1, **kwargs):
    """Feature-wise normalization of the data.

    Arguments:
        axis: Integer or tuple of integers, the axis or axes
            that should be normalized (typically the features axis).

    Input shape and type:
        dtype: floating point.
        shape: any shape with rank >= 2 is accepted.

    Output shape and type:
        dtype: same as input.
        shape: same as input.

    What happens in `adapt`:
        Compute mean and variance of the data
        and store them as the layer's weights.
    """
```

#### Discretization

```python
def __init__(self, bins=None, strategy='quantiles', sparse=False, **kwargs):
    """Maps continuous data into one-hot binary vectors of bin indicators.

    Each non-overlapping bin covers
    a contiguous portion of the dimension considered.
    Bin boundaries can be provided by the user or learned as quantiles.

    Arguments:
        bins: int | List<float>
            If bins is an int, then bin boundaries are to be learned,
            and the width of the output will be exactly bins.
            For instance, setting bins to 4 implies that
            inputs are to be sorted into quantiles,
            and three boundaries are to be learned,
            corresponding to the 25th, 50th, and 75th percentile value.
            If, instead, bins is a list of floats, then those are
            the bin boundary values and nothing is to be learned.
            The width of the output will in that case be the len(bins) + 1.
        strategy: callable | 'quantiles'
            If strategy is the string 'quantiles' (default),
            then bin boundaries will be learned such that each bin
            receives an approximately equal number of sample input values.
            ‘Strategy’ may also be a callable that takes
            (float value, list[float] boundaries) and returns
            an int bucket_index which represents
            which bucket to map ‘value’ to.
        sparse: If True, the layer will output a SparseTensor.
            Otherwise it will be dense.
            This does not change the shape or structure of the output.
            Specifically tf.sparse.to_dense(output) will be the same for both.

    Input shape and type:
        dtype: floating point.
        shape: [batch_size, ..., features]

    Output shape and type:
        dtype: int
        shape: [batch_size, ..., features, num_bins]
            i.e., the same as the input shape,
            with an additional dimension corresponding to
            the number of bins, which is equal to either
            the bins constructor argument (if it is an integer),
            or the length of the bins constructor argument plus 1,
            if it is a list.

    What happens in `adapt`:
        We use a streaming quantile estimator to update the bin boundaries
        so that statistically an element is about equally likely
        to fall into any bin.
        Multiple calls to update continue to mutate
        the layer based on all data seen so far.
    """
```

#### TextVectorization

This layer has basic options for managing text in the Keras model.
It is expected that more advanced users needing custom control will uses Keras-compatible layers provided by tf.text.

Transform a batch of strings (one sample = one string) into either a list of token indices
(one sample = 1D int tensor), or a dense representation (1 sample = 1D float vector).

The processing of each sample unfolds as:
- Standardize each sample (usually lowercasing + punctuation stripping)
- Split each sample into substrings (usually words)
- Recombine substrings into tokens (usually ngrams)
- Index tokens (associate a unique int value with each token)
- Transform each sample using this index, either into a vector of ints or a dense float vector.


```python
def __init__(self,
             tokens=None,
             standardize='lower_and_strip_punctuation',
             split='whitespace',
             ngrams=1,
             mode='int',
             max_length=None):
    """Transforms text into dense vectors or sequences of word indices.

    Arguments:
        tokens: None (default) | int | list<string>
            If tokens is an int, then this layer will learn
            an internal vocabulary of size (tokens - 2),
            such that each of the most frequent (tokens - 2) words
            is assigned assigned to one of the values in [0, tokens).
            The output will have a total to tokens possible values,
            once the out-of-vocabulary value (1)
            and the reserved masking value (0) is taken into account.
            If tokens is None, the number of tokens is automatically inferred
            from the training data (the output will have a number
            of possible values equal to the total number of unique tokens
            seen in the data, plus 2).
            If, instead, tokens is a list of strings, then it constitutes
            exactly to a map from string to integer,
            and there is nothing to be learned.
            The vocabulary output width will be len(tokens) + 2,
            accounting for the out-of-vocabulary value (1)
            and the reserved masking value (0).
        standardize: 'lower_and_strip_punctuation' (default) | None | callable string -> string
            if standardize is the string "lower_and_strip_punctuation",
            each sample is converted to lowercase
            and the following characters are stripped from each sample
            before splitting: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            if it is a callable, that callable is used
            to preprocess each input string before splitting.
        split: ‘whitespace’ (default) | None | Callable string -> list<string>
            if split is ‘whitespace’, then the string
            will be split on whitespace characters.
            if split is None, then each string is treated as a single token.
            if, instead, split is a function from strings to lists of strings,
            then that function will be applied to each string in the input.
        ngrams: 1 (default) | 2 | 3
            Controls the ngram functionality of this layer.
            This layer performs ngrams by concatenating strings
            with no separator and no begin or end tokens;
            the ngramming algorithm is not configurable.
            if ngrams is an int N = 2 or 3,
            the substrings returned by the split function
            are combined into N-grams before being indexed.
        mode: 'int' (default) | 'count' | 'binary' | 'tfidf'
            controls how the integerized words are
            reduced and packed into an output vector.
            if mode is 'count', then the output vector will be
            of length tokens, and the element at position i will
            summarize how many times the string mapping to
            integer i occurred in the split input.
            If, instead, mode is 'binary',
            then the output vector will be the same as for 'count'
            but will contain a 1 if the count is greater than 0.
            if, instead, mode is 'tfidf',
            then the output vector will be the same as for 'count',
            but instead of counts of tokens, will contain
            the weighted count where weights are determined
            by the ‘tfidf’ algorithm.
            if, instead, mode is 'int', then the output vector is
            an int tensor where each int is the index of one token
            in the input string.
        max_length:  None (default) | int.
            Only used if mode=int. If set to an int,
            the output int tensors are of shape [..., max_length],
            with longer sequences being truncated at the end and
            shorter sequences being right-padded.
            If set to None, output sequences are
            of shape [..., max_length_in_batch],
            where max_length_in_batch is the length
            of the longest sequence in the current batch:
            shorter sequences get right-padded.

    Input shape and type:
        dtype: string.
        shape: (batch_size, ..., 1)

    Output shape and type:
        if `mode='int'`:
            dtype: int
            shape: (batch_size, ..., max_length), where max_length
                is the length of the longest token sequence in the current batch, or
                the value of the argument `max_length` if it was passed.
        else:
            dtype: floating point.
            shape: (batch_size, ..., num_tokens)

    What happens in `adapt`:
        We build an index mapping tokens to token indices,
        and in the case of `mode='count'` and `mode='tfidf`,
        we keep track of how many time each token has appeared.
    """
```

### Writing a subclass of `PreprocessingLayer`

The following 3 methods should be overridden:

- `__init__`: constructor of the layer, used to configure its behavior.
- `build(self, inputs_shape)`: creates the state variables of the layer.
- `call(self, inputs)`: transforms the inputs (should only be called after `adapt` has been called).
- `adapt(self, data, [reset_state=True])`: sets the state of the layer given the data provided (either as a tf.data dataset or numpy array(s)). The `reset_state` argument is optional and may be ignored.


### Handling of async prefetching

Some preprocessing ops are CPU-only and benefit from being executed asynchronously on the accelerator host (as opposed to the accelerator itself, e.g. GPU or TPU),
with a batch of data being prepocessed on the host while the previous batch is being processed by the accelerator. This pattern is known as "async prefetching".

This is normally done as part of a tf.data pipeline. The current proposal implies moving some of that preprocessing to inside the model itself, which is normally
executed end-to-end on an accelerator.

This means that we need a way to lift the preprocessing part of the model in a tf.data pipeline during model training. In `fit`, we can do this automatically.
In custom training loops, we will expect the user to do it manually (see subsection "Custom training loops").

We propose the addition of two new methods on the `Model` class:

```python
def get_preprocessing_stage(self):
    """Retrieves the preprocessing part of the model.

    This is the part of the model that should be executed asynchronously
    on the device host during training.

    Returns:
        Instance of `PreprocessingLayer` or `PreprocessingStage`.
        May be None if the model does not start with preprocessing layers.
    """
    pass

def get_main_stage(self):
    """Retrieves the main processing part of the model.

    This is the part of the model that should be executed
    on the accelator device.

    Returns:
        Model instance.
    """
```

Thus, for any model that starts with preprocessing layers, the following:

```python
outputs = model(inputs)
```

is functionally equivalent to:

```python
preprocessed_inputs = model.get_preprocessing_stage()(inputs)
outputs = model.get_main_stage()(preprocessed_inputs)
```


#### Examples:

Sequential model with a preprocessing layer:

```python
vectorization = keras.layers.TextVectorization()
vectorization.adapt(data_sample)

model = keras.Sequential([
    vectorization,
    keras.layers.Dense(10, activation='softmax'),
])

# This is the `vectorization` layer.
preproc_stage = model.get_preprocessing_stage()
# model containing the `Dense` layer only.
main_stage = model.get_main_stage()
```

Functional model with 2 branches, each with a preprocessing layer:

```python
normalization_a = layers.Normalization()
normalization_b = layers.Normalization()
normalization_a.adapt(data_a)
normalization_b.adapt(data_b)

input_a = Input(shape_a)
input_b = Input(shape_b)
normed_a = normalization_a(input_a)
normed_b = normalization_b(input_b)
a = layers.Dense(32)(normed_a)
b = layers.Dense(32)(normed_b)
c = layers.concatenate([a, b])
outputs = layers.Dense(1, activation='sigmoid')(c)

model = Model([input_a, input_b], outputs)

# `PreprocessingStage` instance
# mapping `[input_a, input_b]` to `[normed_a, normed_b]`
preproc_stage = model.get_preprocessing_stage()

# Model instance mapping `[normed_a, normed_b]` to `outputs`.
main_stage = model.get_main_stage()
```

Subclassed model with a preprocessing layer:

```python
class MyModel(Model):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.preproc_layer = layers.Normalization()
        self.submodel = MySubmodel()

    def call(self, inputs):
        return self.submodel(self.preproc_layer(inputs))

    def get_preprocessing_stage(self):
        return self.preproc_layer

    def get_main_stage(self):
        return self.submodel
```


#### The case of the built-in `fit` loop


When calling `fit` or `evaluate` on a Dataset a model that contains preprocessing layers,
the lifting happens automatically and the user-facing workflow doesn't change.

```python
model.fit(dataset, epochs=10)
```

#### Custom training loops

When writing custom training loops, the user must manually do the lifting of the preprocessing stage
into the data pipeline:

```python
model = Model(...)
preproc_stage = model.get_preprocessing_stage()
main_model = model.get_main_stage()

preproc_dataset = Dataset(...)
preproc_stage.adapt(preproc_dataset)

# Map the preprocessing stage on the dataset.
dataset = Dataset(...)
dataset = dataset.map(preproc_stage)

# Regular training loop (using `main_model`).
for x, y in dataset:
    with GradientTape() as tape:
        y_pred = main_model(x)
        loss = loss_fn(y, y_pred)
        ...
```

In general, you won't have to refer to `get_preprocessing_stage` and `get_main_stage` directly, because you will
probably already have direct handles on your preprocessing layer and the rest of the model:

```python
normalization = layers.Normalization()
normalization.adapt(preproc_dataset)
dataset = dataset.map(normalization)

for x, y in dataset:
    with GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        ...
```


## Questions and Discussion Topics

### Naming Discussion

#### Naming conventions to follow for preprocessing layers

[RESOLUTION: we will use option A]

We have two possible sets of names for the layers:

##### Option A: Normalization, Discretization, TextVectorization

Pros: consistent with most existing layers, in particular BatchNormalization.
Cons: It's longer.

##### Option B: Normalize, Discretize, VectorizeText

Pros: It's shorter.
Cons: Normalize vs BatchNormalization is jarring.


#### Using the name "preprocessing" or "processing"

[RESOLUTION: we will use option A, "preprocessing"]

It has been proposed that we use the name "processing" throughout the API instead of "preprocessing".

##### Option A: "preprocessing".

Pros:
1) The meaning of "preprocessing" is clear to all users ("data normalization and stuff").
2) We need a clear semantic boundary between the main data processing flow of a model and what goes before it (the preprocessing stage).
3) It replaces the functionality of the `keras.preprocessing` module, and should be consistent with this naming convention.

Cons:
The `Normalization` layer, being differentiable, can be used in the middle of a model, rather than at the start.
However, there's nothing weird about keeping the name "preprocessing" in this specific case: it is widely understood that a `Normalization` layer is doing "data preprocessing", independently of where you use it -- in fact, normalization is the first example that shows up in most definitions of "data preprocessing". 


##### Option B: "processing".

Pros: The Normalization layer can be used elsewhere in a model than at the start (although it would have to be trained separately).
Cons: It's very generic, and does not clearly convey the difference between "preprocessing stage" and "main processing stage" required by the async prefetching API.


#### Name to use for `adapt` method

[RESOLUTION: decision delayed until implementation]

We may want to use the name `fit` instead (other suggestions welcome).

Pros of using `fit`: consistency with `model.fit()`, and the `fit` method on `ImageDataGenerator` and `Tokenizer` from the `keras.preprocessing` module.
Cons of using `fit`: It may confuse users, since `preprocessing_layer.fit()` would have a different signature.

---

[OTHER ADDITIONS FROM DESIGN REVIEW]

- We should decouple the user-facing `adapt(data)` method (or `fit(data)`), and the implementer-facing method, so as to make it easier to implement support for different data formats.



