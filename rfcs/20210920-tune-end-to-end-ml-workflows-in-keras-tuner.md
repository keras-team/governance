# Tune end-to-end ML workflows in KerasTuner

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author**    | Haifeng Jin (haifengj@google.com)                    |
| **Updated**   | 2021-09-20                                           |

## Objective

Improving the user experience of KerasTuner to tune end-to-end workflows.
Reduce the learning curve and code hacks for workflows involves hyperparameters
in data preprocessing and model fitting.

## Motivation

Different users prefer different workflows for their tuning process -- like
Keras has different getting-started tutorials for engineers and researchers.
There are users who prefer to learn more about the framework and to implement
everything by overriding class methods, and users who prefer to write
everything from scratch to have a shorter learning curve and better
configurability for the details.  For example, some users would like to
override `Model.train_step()` to make the code cleaner, others like to write
the training loop from scratch.


Currently, KerasTuner has good support for the users who would like to
restructure their code by learning the KerasTuner framework, and for users who
only need to do some light customization of the model building process.
However, the support for users who need to write their model building and
training process from scratch is not adequate.


Moreover, many users use the hyperparameter tuning library as an intermediate
step in their ML process rather than their main API. In their workflow,
implementing and training a model with Keras are usually a separate process
from hyperparameter tuning. They would first write the code using Keras, then
try to put it into KerasTuner to tune, and put the hyperparameter values back
into their Keras model. Therefore, we should maximize the code and model
portability in KerasTuner for these users, and minimize the code changes
required for them to adopt and remove KerasTuner.

### The old workflow

The current workflow for writing their model training process with KerasTuner
is as follows. The user defines the model in the `HyperModel.build()` function.
Defines the data preprocessing and model training by overriding
`Tuner.run_trial()`. The arguments, like the dataset, are passed through the
`Tuner.search()` function, and finally received by `Tuner.run_trial()`.


```py
import keras_tuner as kt

class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    # Model building
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    return model

class MyTuner(kt.Tuner):
  def run_trial(self, trial, *fit_args, **fit_kwargs):
    hp = trial.hyperparameters
  
    # data preprocessing       
    training_data, validation_data = data_preprocessing(
        hp, *fit_args, **fit_kwargs)
    model = self.hypermodel.build(hp)
   
    # model training
    model.fit(
        training_data,
        epochs=hp.Int(...),
        validation_data=validation_data,
        ...)
       
    # evaluation and reporting
    score = model.evaluate(validation_data, ...)
    self.oracle.update_trial(trial.trial_id, {'score': score})
    self.save_model(trial.trial_id, model)

tuner = MyTuner(
    hypermodel=MyHyperModel(),
    objective=kt.Objective('score', 'min'),
    ...)

# Passing in the args
tuner.search(*fit_args, **fit_kwargs)
```

### Problems

The key problem of this workflow is that the code is split in two classes. Any
control flow and data flow between data preprocessing, model building, and
model training would all have to pass through the framework and function calls.
To use the framework, the user would have to understand how these different
functions are called, and wire their data and information properly between
these functions.

### Use cases to improve

The following use cases are not well supported because of the problem above.

#### Configure and jointly tune data preprocessing and model training

For example, writing a custom training loop, or tuning the data preprocessing
steps, or anything in the training loop like whether to shuffle the training
data, they need to override the `Tuner.run_trial()` function, which adds more
to the learning curve.

For example, in natural language processing, tokenization and vectorization may
affect the later model type. They will need to find a way to pass this
information from `Tuner.run_trial()` to HyperModel.build.

#### Tune existing Keras code

If the users have their code for model building and training ready written using
Keras, and they want to tune some of the hyperparameters, they would have to
change the code a lot to separate their code apart and wire the data flow and
control flow between the overridden functions.

#### Retrain the model after tuning

If the user wants to retrain the model using the best hyperparameter values
found, there is not a straight-forward way to do it if they used the
hyperparameter in `Tuner.run_trial()` for data preprocessing and model
training.

## User Benefit

The use cases described above would all have smooth workflows, without much
extra code or learning of the framework.

## Design Proposal

We propose two workflows: the `Tuner` workflow and the `HyperModel` workflow to
solve the problems above.

The `Tuner` workflow is to override `Tuner.run_trial()`. The user can put all the
code for data preprocessing, model building, model training all in one place in
the `Tuner.run_trial()` function. No `HyperModel` is needed. It supports all the
use cases mentioned above by providing the maximum freedom to the user.

The `HyperModel` workflow follows the original `HyperModel` style. It is easier
to learn and needs less code compared to the first workflow, but covers all the
use cases as long as the code for building and training the model are separate.
The user only needs to override the `HyperModel.fit()` for any tuning of the
data preprocessing and model fitting process.

## Detailed Design

### The `Tuner` workflow

Here is an end-to-end code example of the new workflow.

The user only needs to override `Tuner.run_trial()` to put everything together,
including data preprocessing, model building, and model training. It returns
the evaluation results back to the tuner. 

```py
class MyTuner(kt.Tuner):
  def run_trial(self, trial, x, y, callbacks=None, **kwargs):
    hp = trial.hyperparameters
    # Data preprocessing
    num_features = hp.Int("num_features", 10, 15)
    x, y = feature_selection(num_features=num_features, x, y)
    # Model building
    # Input shape depending on data preprocessing.
    inputs = keras.Input(shape=(num_features,))
    outputs = keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='relu')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse',
                  metrics=['mae'])
    # Model training
    history = model.fit(
        x,
        y,
        epochs=100,
        validation_data=validation_data,
        # Tune whether to use shuffle.
        shuffle=hp.Boolean("shuffle"),
        # Tune whether to use sample_weights.
        sample_weight=sample_weight if hp.Boolean("sample_weight") else None,
        # The provided callbacks list contains checkpointing and tensorboard.
        callbacks=callbacks)
    # Save the model to a unique path with `trial_id`.
    model.save(os.path.join(trial.trial_id, 'model'))
    # Returning the evaluation results
    return np.min(history.history["val_mae"])

# When Tuner.run_trial is overridden,
# `hypermodel` and `objective` are optional.
tuner = MyTuner(
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# Anything passed to `search()` will
# go to `**kwargs` for `Tuner.run_trial()`.
tuner.search(x, y)
# Get the best model.
best_model = tuner.get_best_models()[0]
```

There are several important features in this workflow:

* Tune the arguments in `HyperModel.fit()`, like `shuffle` and `sample_weight`.

* Share local variables across the workflow. For example, the model building
  process can access the `num_features`, which is a variable in data
  preprocessing. It solves the problem of joint tuning.

* Use built-in callbacks for convenience. The callbacks argument contains
  callback functions for checkpointing and TensorBoard setup.

* The return value is flexible. It can be a single value, or a list of values,
  or a dictionary of metrics, or even a `History` object returned by
  `model.fit()`.

* The `hypermodel` and `objective` can be optional. The user doesn't need to
  define a `HyperModel`. If the return value is a single value, it will be
  minimized by default. Therefore, objective is also optional.

* The user can build a unique path to save each model with `trial.trial_id`.

For the use case of reusing existing Keras code. The user can use the following
workflow, which calls a function using all the hyperparameters. The user only
needs to write a function to call the existing Keras code and return the
evaluation results.

```py
class MyTuner(kt.Tuner):
 def run_trial(self, trial, **kwargs):
   hp = trial.hyperparameters
   return build_and_evaluate_model(
       hp.Int("num_features", 10, 15),
       hp.Choice('units', [8, 16, 32]),
       ...
       trial.trial_id,
   ))
   # Save model can be handled by the user.
   # `trial_id` is unique for each trial.

tuner = MyTuner(...)
tuner.search()
# Retraining the model
build_and_evaluate_model(**tuner.get_best_hyperparameters()[0])
```
	

In this workflow, the user can easily retrain the model by calling the function again with the best hyperparameters.

### The HyperModel workflow

For users who prefer to follow the old workflow, they can also implement the HyperModel above by overriding the build function and the fit function. The build function builds and returns the model. The fit function does the data preprocessing and model training.

Following is a code example implementing the same functionality of the code example above.

```py
import numpy as np
import keras_tuner as kt
from tensorflow import keras

class MyHyperModel(kt.HyperModel):

  def build(self, hp):
    # Model building
    # Input shape depends on a hyperparameter used by data preprocessing.
    inputs = keras.Input(shape=(hp.Int("num_features", 10, 15),))
    x = keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='relu')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse',
                  metrics=['mae'])
    return model
  
  def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
    # Data preprocessing
    # Get the hyperparameter value used in `build()`.
    x, y = feature_selection(num_features=hp.get("num_features"), x, y)
    # Model training
    # Returning the training history
    # or a similar dictionary if using custom training loop.
    return model.fit(
        x,
        y,
        epochs=100,
        validation_data=validation_data,
        # Tune whether to use shuffle.
        shuffle=hp.Boolean("shuffle"),
        # Tune whether to use sample_weights.
        sample_weight=sample_weight if hp.Boolean("sample_weight") else None,
        # The provided callbacks list contains checkpointing and tensorboard.
        callbacks=callbacks)

tuner = kt.RandomSearch(
    hypermodel=MyHyperModel(),
    objective=kt.Objective('val_mae', 'min'),
    directory='dir',
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# Any arg passed to `search()` would be passed to `fit()`.
tuner.search(x, y)

# Exporting the best models.
models = tuner.get_best_models(num_models=2)

# Retraining the model with the second best hyperparameters.
second_best_hp = tuner.get_best_hyperparameters(num_models=2)[1]
hypermodel = MyHyperModel()
model = hypermodel.build(second_best_hp)
hypermodel.fit(
    hp=second_best_hp, 
    model=model,
    x=new_x,
    y=new_y,
    validation_data=new_validation_data,
    # Save the model at its best epoch to a custom path
    callbacks=[tf.keras.callbacks.ModelCheckpoint(
        filepath="path_to_checkpoint",
        monitor='val_loss',
        save_best_only=True)])
# Save the final model.
model.save("path_to_saved_model")
```

Please take note of the following four points:

* Similar to `Tuner.run_trial()`, the return value of the fit function supports
  all different formats.

* The user can use built-in callbacks just like in `Tuner.run_trial()`.

* `build()` and `fit()` can share hyperparameters. In this example,
  `num_features` is shared between the two functions. In `fit()`, we can use
  `hp.get()` to obtain the value of a hyperparameter used in `build()`.

* We can easily retrain the model with any hyperparameter value set with
  `hypermodel.build()` and `hypermodel.fit()`.

With these proposed workflows, the user now has the maximum flexibility. Any
step in an end-to-end machine learning workflow can be tuned. Moreover, the
changes needed to tune existing Keras code is minimized.

Here we present HyperModel code examples of three important use cases:

* Text tokenization.

* Custom training loop.

* Fine tuning with pretrained weights.

#### Text tokenization

```py
import json

# Save the vocabulary to disk before search.
text_vectorizer = layers.TextVectorization()
text_vectorizer.adapt(dataset.map(lambda x, y: x))
with open('vocab.json', 'w') as f:
  json.dump(text_vectorizer.get_vocabulary(), f)

class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    inputs = keras.Input(shape=(10,))
    outputs = layers.Embedding(
        # max_token is a hyperparameter also used in text vectorization.
        input_dim=hp.Int("max_tokens", 100, 500, step=100),
        output_dim=64)(inputs)
    outputs = layers.LSTM(hp.Int("units", 32, 128, step=32))(outputs)
    outputs = layers.Dense(1, activation='sigmoid')(outputs)
    model = keras.Model(inputs, outputs)
    model.compile(loss='mse')
    return model
  
  def fit(self, hp, model, dataset, validation_data, callbacks, **kwargs):
    # Load the vocabulary from file.
    with open('vocab.json', 'r') as f:
      vocab = json.load(f)

    # Create and adapt the text vectorizer.
    text_vectorizer = layers.TextVectorization(
        # The max_tokens is a hyperparameter created in build().
        vocabulary=vocab[:hp.get("max_tokens")],
        output_mode="int",
        output_sequence_length=10)
  
    return model.fit(
        # Convert x from strings to integer vectors.
        dataset.map(
            lambda x, y: (text_vectorizer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE),
        validation_data=validation_data,
        callbacks=callbacks,
    )
```
	

#### Custom training loop

```py
class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    inputs = keras.Input(shape=(10,))
    outputs = layers.Dense(hp.Int("units", 16, 128), activation='relu')(inputs)
    outputs = layers.Dense(1, activation='sigmoid')
    model = keras.Model(inputs, outputs)
    return model
  
  def fit(self, hp, model, dataset, validation_data, **kwargs):
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_tracker = tf.keras.metrics.Mean()
    # Track the validation loss
    val_loss_tracker = tf.keras.metrics.Mean()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # Record the minimum validation loss during fit.
    min_val_loss = float("inf")
  
     @tf.function
    def run_train_step(data):
      images = tf.dtypes.cast(data[0], "float32") / 255.0
      labels = data[1]
      with tf.GradientTape() as tape:
        logits = model(images)
        loss = loss_fn(labels, logits)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      loss_tracker.update_state(loss)
  
     @tf.function
    def run_val_step(data):
      images = tf.dtypes.cast(data[0], "float32") / 255.0
      labels = data[1]
      logits = model(images)
      loss = loss_fn(labels, logits)
      val_loss_tracker.update_state(loss)
  
    for epoch in range(2):
      for batch, data in enumerate(dataset):
        run_train_step(data)
      print(f"Epoch loss: {loss_tracker.result().numpy()}")
      loss_tracker.reset_states()
      for batch, data in enumerate(validation_data):
        run_val_step(data)
      val_loss = val_loss_tracker.result().numpy()
      min_val_loss = min(min_val_loss, val_loss)
      print(f"Epoch val_loss: {val_loss}")
      val_loss_tracker.reset_states()
  
    return min_val_loss
```

You may also subclass `keras.Model` to override `train_step()`.

#### Fine tuning with pretrained weights

```py
class MyHyperModel(kt.HyperModel):

  def build(self, hp):
    return keras.Sequential([
        keras.applications.ResNet50(
            weights="imagenet",
            input_shape=(32, 32, 3),
            include_top=False,
        ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(hp.Int("units", 32, 128)),
        layers.Dense(1),
    ])
  
  def fit(self, hp, model, dataset, validation_data, callbacks, **kwargs):
    # Fit the model with the `base_model` freezed.
    model.layers[0].trainable = False
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.fit(dataset, epochs=20)
    # Fit the model again with some layers in the `base_model` freezed.
    model.layers[0].trainable = True
    for layer in model.layers[:hp.Int("freeze", 0, 20)]:
      layer.trainable = False
    model.compile(
        # Use a smaller learning rate.
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    return model.fit(
        dataset,
        epochs=20,
        callbacks=callbacks,
        validation_data=validation_data)
```

### API documentation

The APIs in the new `HyperModel` class are as follows.

```py
class HyperModel():
  def fit(self, hp, model, callbacks, **kwargs):
    """Train the model.
   
    Args:
        hp: HyperParameters.
        model: `keras.Model` built in the `build()` function.
        callbacks: A list of prebuild Keras callbacks for model checkpointing
          and tensorboard configuration.
        **kwargs: Anything the user defines. They are passed from
            `Tuner.search()`.
   
    Returns:
        A `History` object, a similar dictionary, or a single value.
    """
    pass

class Tuner():
  def run_trial(self, trial, callbacks, **kwargs):
    """Train the model.
   
    Args:
        trial: Trial. The current Trial object.
        callbacks: A list of prebuild Keras callbacks for model checkpointing
          and tensorboard configuration.
        **kwargs: Anything the user defines. They are passed from Tuner.search().

    Returns:
        A `History` object, a similar dictionary, or a single value.
    """
```

## Questions and Discussion Topics

Does the fit function need `trial_id` in the args to do model saving? The user
may need this arg to build unique saving paths for the models.
