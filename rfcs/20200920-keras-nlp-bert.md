# keras-nlp Transformer Encoder API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com), Hongkun Yu (hongkuny@google.com)|
| **Sponsor(s)** | Mark Omernick (momernick@google.com)|
| **Updated**   | 2020-09-21                                           |


## Objective

We aim at providing a set of Keras layers to handle Transformer-Encoder BERT-style models.

## Key Benefits

BERT-style Transformer-Encoders are a state-of-art technique that powers many NLP tasks:

- Single sentence classification task, e.g., sentiment analysis
- Sentence pair classification task, e.g., next sentence prediction
- Question answering task, e.g., SQuAD
- Single sentence tagging task, e.g., named entity recognition

With this proposal, Keras users will be able to handle the tasks above with a simple API. 

## Design overview

This proposal builds on the assumption that inputs are lookup indices, i.e., `tf.int64` sequences.
Tokenization is not part of this proposal but will be our immediate next step.

### Classification task

Case where a user want to use a pretrained BERT encoder for sentiment analysis:

```python
# Considering a imbd review dataset
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_nlp
import tensorflow_text as tftext

imdb_reviews = tfds.load('imdb_reviews')
train_ds = imdb_reviews['train'].batch(32)
test_ds = imdb_reviews['test'].batch(32)

# Tokenization with BertTokenizer
vocab_path = "gs://<bucket_name>/<file_path>/vocab.txt"
tokenizer = tftext.BertTokenizer(vocab_path, token_out_type=tf.int64, lower_case=False)
SEQUENCE_LENGTH = 128
def preprocess(input_text):
  token_ids = tokenizer.tokenize_with_offsets(input_text)
  segment_ids = tf.concat([tf.zeros_like(cls), tf.ones_like(token_ids), tf.ones_like(sep)], axis=1)
  output_shape = [None, SEQUENCE_LENGTH]
  token_ids = token_ids.merge_dims(-2, -1)
  segment_ids = segment_ids.merge_dims(-2, -1).to_tensor(shape=output_shape)
  input_mask = tf.ones_like(token_ids).to_tensor(shape=output_shape)
  token_ids = token_ids.to_tensor(shape=output_shape)
  return {
      'input_ids': token_ids,
      'input_mask': input_mask,
      'input_type_ids': segment_ids
  }

strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
  encoder = keras_nlp.encoders.BertEncoder(vocab_size=30522, max_sequence_length=512, type_vocab_size=2)
  encoder.load_weights("gs://<bucket_name>/<file_path>")
  token_ids = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='input_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='input_mask', dtype=tf.int32)
  type_ids = tf.keras.layers.Input(shape=(128,), name='input_type_ids', dtype=tf.int32)
  x = encoder([token_ids, input_mask, type_ids])['pooled_output']
  x = tf.keras.layers.Dropout(rate=0.1)(x)
  output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(inputs=[token_ids, input_mask, type_ids], outputs=output)

model.compile('adam', 'binary_crossentropy', ['accuracy'])
model.fit(train_ds, epochs=5, validation_data=test_ds)
```

### Pretraining task

We aim to provide pretrained checkpoints for `BertEncoder` with different datasets and different sizes through TF Hub,
however the user can choose to pretrain a new BertEncoder based on their own dataset.

```python
with strategy.scope():
  encoder = keras_nlp.encoders.BertEncoder(vocab_size, max_sequence_length, type_vocab_size)
  token_ids = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='word_token_ids', dtype=tf.int32)
  input_mask = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='input_mask', dtype=tf.int32)
  type_ids = tf.keras.layers.Input(shape=(128,), name='input_type_ids', dtype=tf.int32)
  masked_lm_positions = tf.keras.layers.Input(shape=(None,), name='masked_lm_positions', dtype=tf.int32)
  x = encoder([token_ids, input_mask, type_ids])['pooled_output']
  cls_output, sequence_output = output['pooled_output'], outputs['sequence_output']
  masked_lm = keras_nlp.layers.MaskedLM(embedding_table=encoder.get_embedding_table())
  lm_output = masked_lm(sequence_output, masked_positions=masked_lm_positions)
  cls_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(cls_output)
  model = tf.keras.Model(inputs=[token_ids, input_mask, type_ids, masked_lm_positions],
                         outputs={'lm_output': masked_lm, 'cls_output': cls_output})

model.compile('adam', {'lm_output': 'sparse_categorical_crossentropy', 'cls_output': 'sparse_categorical_crossentropy'})
model.fit(train_ds, epochs=100)
```

### Other encoder-based networks

`BertEncoder` is the first encoder network we propose in this doc. However other encoder networks can be easily
built on top of the `TransformerEncoder` layer. For example, for a transformer encoder sharing mechanism
with [ALBERT](https://arxiv.org/pdf/1909.11942.pdf), this can be achieved by:

```python
token_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids')
mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask')
type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids')
word_embeddings = keras_nlp.layers.OnDeviceEmbedding(vocab_size, embedding_width)(token_ids)
position_embeddings = keras_nlp.layers.PositionEmbedding(max_sequence_length)(word_embeddings)
type_embeddings = keras_nlp.layers.OnDeviceEmbedding(
  vocab_size=type_vocab_size, embedding_width=embedding_width, use_one_hot=True)(type_ids)
embeddings = tf.keras.layers.Add()([word_embeddings, position_embeddings, type_embeddings])
embeddings = tf.keras.layers.LayerNormalization(axis=-1)(embeddings)
embeddings = tf.keras.layers.Dropout(rate=dropout_rate)(embeddings)
embeddings = tf.keras.layers.experimental.EinsumDense(
  '...x,xy->...y', output_shape=hidden_size, bias_axes='y')(embeddings)
data = emnbeddings
attention_mask = layers.SelfAttentionMask()([data, mask])
shared_layer = keras_nlp.layers.TransformerEncoder(num_attention_heads, inner_dim)
for _ in range(num_layers):
  data = shared_layer([data, attention_mask])
first_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(data)
cls_output = tf.keras.layers.Dense(units=hidden_size, activation='tanh')(first_token_tensor)
outputs = dict(sequence_output=data, pooled_output=cls_output)
model = tf.keras.Model(inputs=[word_ids, mask, type_ids], outputs=outputs)
```

## Detailed Design

### Layers -- TransformerEncoder

This layer encapsulates a single layer of Transformer Encoder.

```python
class TransformerEncoder(tf.keras.layers.Layer):
  """TransformerEncoder layer.

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `tf.keras.layers.MultiHeadAttention` layer with a
  two-layer feedforward network.

  References:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding](https://arxiv.org/abs/1810.04805)
  """

  def __init__(self,
               num_attention_heads,
               inner_dim,
               inner_activation,
               output_range=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               **kwargs):
    """Initializes `TransformerEncoder`.

    Arguments:
      num_attention_heads: Number of attention heads.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network.
      output_range: the sequence output range, [0, output_range) for slicing the
        target sequence. `None` means the target sequence is not sliced.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: Dropout probability for within the attention layer.
      inner_dropout: Dropout probability for the first Dense layer in a
        two-layer feedforward network.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      **kwargs: keyword arguments/
    """
```

### Layers -- SelfAttentionMask

```python
class SelfAttentionMask(tf.keras.layers.Layer):
  """Create 3D attention mask from a 2D tensor mask."""

  def call(self, inputs, to_mask):
  """
  Args:
    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
```

### Layers -- OnDeviceEmbedding
This is the experimental layer that would support either one-hot tf.matmul approach or tf.gather approach.

```python
class OnDeviceEmbedding(tf.keras.layers.Layer):
  """Performs an embedding lookup suitable for accelerator devices.

  This layer uses either tf.gather or tf.one_hot to translate integer indices to
  float embeddings.

  Arguments:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Output size of the embedding layer.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
      lookup. Defaults to False (that is, using tf.gather). Setting this option
      to True may improve performance, especially on small vocabulary sizes, but
      will generally require more memory.
  """

  def __init__(self,
               vocab_size,
               embedding_width,
               initializer="glorot_uniform",
               use_one_hot=False,
               **kwargs):
```

### Layers -- PositionEmbedding

```python
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  Arguments:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".

  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """
```

### Layers -- MaskedLM

```python
class MaskedLM(tf.keras.layers.Layer):
  """Masked language model network head for BERT modeling.

  This layer implements a masked language model based on the provided
  transformer based encoder. It assumes that the encoder network being passed
  has a "get_embedding_table()" method.

  Arguments:
    embedding_table: The embedding table from encoder network.
    activation: The activation, if any, for the dense layer.
    initializer: The initializer for the dense layer. Defaults to a Glorot
      uniform initializer.
    output: The output style for this layer. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               embedding_table,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               name=None,
               **kwargs):
```

### Encoders -- BertEncoder

```python
class BertEncoder(tf.keras.Model):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Arguments:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
        dropout.
    attention_dropout: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yeilds the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
  """

  def __init__(
      self,
      vocab_size,
      hidden_size=768,
      num_layers=12,
      num_attention_heads=12,
      max_sequence_length=512,
      type_vocab_size=16,
      inner_dim=3072,
      inner_activation='gelu',
      output_dropout=0.1,
      attention_dropout=0.1,
      initializer='truncated_normal',
      output_range=None,
      embedding_width=None,
      **kwargs):
```

## Questions and Discussion Topics

Gathering feedbacks on arguments & naming conventions.
