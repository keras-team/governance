# keras-cv Single Stage Two-Dimensional Object Detection API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com)|
| **Contributor(s)** | Pengchong Jin (pengchong@google.com)|
| **Updated**   | 2020-09-28                                           |

## Objective

We aim at providing the core primitive components for training and serving single stage two-dimensional object
detection models, such as Single-Shot MultiBox Detector (SSD), RetinaNet, You-Only-Look-Once (YOLO).

## Key Benefits

Single stage object detection models are a state-of-art technique that powers many computer vision tasks, they provide
faster detection compared to two stage models (such as FasterRCNN), while maintaining comparable performance.

With this proposal, Keras users will be able to build end-to-end models with a much simplified API.

## Design overview

This proposal includes the specific core components for building single stage object detection models. It does not,
include, however, 1) Data augmentation, such as image and groundtruth box preprocessing, 2) Model backbone, such as
DarkNet, or functions to generate feature maps 3) Detection heads, such as Feature Pyramid, 4) metrics utils such as
COCO Evaluator, or visualization utils. Data augmentation will be included as a separate RFC that handles a
broader context than object detection.

Model backbone and detection heads are model specific, we anticipate them to be analyzed and proposed in 
keras-applications for heavily used patterns, however the user can build them easily using keras.

#### Training

Case where a user want to train from scratch:
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv

# Considering a COCO dataset
coco_dataset = tfds.load('coco/2017')
train_ds, eval_ds = coco_dataset['train'], coco_dataset['validation']

def preprocess(features):
  image, gt_boxes, gt_labels = features['image'], features['objects']['bbox'], features['objects']['label']
  # preprocess image, gt_boxes, gt_labels, such as flip, resize, and padding, and reserve 0 for background label.
  return image, gt_boxes, gt_labels

anchor_generator = keras_cv.ops.AnchorGenerator(anchor_sizes, scales, aspect_ratios, strides)
similarity_calculator = keras_cv.ops.IOUSimilarity()
box_matcher = keras_cv.ops.BoxMatcher(positive_threshold, negative_threshold)
target_gather = keras_cv.ops.TargerGather()
box_encoder = keras_cv.ops.BoxCoder(offset='sigmoid')

def encode_label(image, gt_boxes, gt_labels):
  anchor_boxes = anchor_generator(image_size)
  iou = similarity_calculator(gt_boxes, anchor_boxes)
  match_indices, match_indicators = box_matcher(iou)

  mask = tf.less_equal(match_indicators, 0)
  class_mask = tf.expand_dims(mask, -1)
  box_mask = tf.tile(class_mask, [1, 4])

  class_targets = target_gather(gt_labels, match_indices, class_mask, -1)
  box_targets = target_gather(gt_boxes, match_indices, box_mask, 0.0)
  box_targets = box_encoder(box_targets, anchor_boxes)

  weights = tf.squeeze(tf.ones_like(gt_labels), axis=-1)
  ignore_mask = tf.equal(match_indicators, -2)
  class_weights = target_gather(weights, match_indices, ignore_mask, 0.0)
  box_weights = target_gather(weights, match_indices, mask, 0.0)

  return (image, {'classification': class_targets, 'regression': box_targets},
          {'classification': class_weights, 'regression': box_weights})

class RetinaNet(tf.keras.Model):
  # includes backbone and feature pyramid head.
  def __init__(self):
    # self.backbone = Model Backbone that returns dict of feature map
    # self.fpn = Feature Pyramid Heads that
    # self.head = classification and regression heads
  
  def call(self, image, training=None):
    feature_map = self.backbone(image, training)
    feature_map = self.fpn(feature_map, training)
    class_scores, boxes = self.head(feature_map, training)
    return {'classification': class_scores, 'regression': boxes}

transformed_train_ds = train_ds.map(preprocess).map(encode_label).batch(128).shuffle(1024)
transformed_eval_ds = eval_ds.map(preprocess).map(encode_label).batch(128)

strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
    optimizer = tf.keras.optimizers.SGD(lr_scheduler)
    model = RetinaNet()
    model.compile(optimizer, {'classification': keras_cv.losses.Focal(), 'regression': tf.keras.losses.Huber()}, [])

model.fit(transformed_train_ds, epochs=120, validation_data=transformed_eval_ds)
model.save(file_path)
``` 

#### Serving

Case where a user want to serve the trained model for a single image.
```python
loaded_model = tf.keras.models.load(model)
detection_generator = keras_cv.layers.NMSDetectionDecoder()
@tf.function
def serving_fn(image):
  batched_image = tf.expand_dims(image)
  results = loaded_model(batched_image, training=False)
  classes, scores, boxes = detection_generator(results['classification'], results['regression'])
  return classes, scores, boxes
```

## Detailed Design

#### Ops -- AnchorGenerator
```python
class AnchorGenerator:
  """Utility to generate anchors for a multiple feature maps."""

  def __init__(self,
               anchor_sizes,
               scales,
               aspect_ratios,
               strides,
               clip_boxes=False):
    """Constructs multiscale anchors.

    Args:
      anchor_sizes: A list/dict of int represents the anchor size for each scale. The
        anchor height will be `anchor_size / sqrt(aspect_ratio)`, anchor width
        will be `anchor_size * sqrt(aspect_ratio)` for each scale.
      scales: A list/tuple/dict, or a list/tuple/dict of a list/tuple of positive
        floats representing the actual anchor size to the base `anchor_size`.
      aspect_ratios: A list/tuple/dict, or a list/tuple/dict of a list/tuple of positive
        floats representing the ratio of anchor width to anchor height.
      strides: A list/tuple of ints represent the anchor stride size between
        center of anchors at each scale.
      clip_boxes: Boolean to represents whether the anchor coordinates should be
        clipped to the image size. Defaults to `False`. 

    Input shape: the size of the image, `[H, W, C]`
    Output shape: the size of anchors concat on each level, `[(H /
      strides) * (W / strides), K * 4]`
    """
  def __call__(self, image_size):
    """
    Args:
      image_size: a tuple of 2 for image_height and image_width.
    Returns:
      anchors: a dict or single Tensor.
    """
```

#### Ops -- BoxMatcher
```python
class BoxMatcher:
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  positive_threshold (upper threshold) and negative_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= positive_threshold: Highest similarity. Matched/Positive!
  (2) positive_threshold > similarity >= negative_threshold: Medium similarity.
        This is Ignored.
  (3) negative_threshold > similarity: Lowest similarity for Negative Match.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(
      self,
      positive_threshold,
      negative_threshold=None,
      force_match_for_each_col=False,
      positive_value=1,
      negative_value=-1,
      ignore_value=-2):
    """Construct BoxMatcher.

    Args:
      positive_threshold: Threshold for positive matches. Positive if
        sim >= positive_threshold, where sim is the maximum value of the
        similarity matrix for a given column. Set to None for no threshold.
      negative_threshold: Threshold for negative matches. Negative if
        sim < negative_threshold. Defaults to positive_threshold when set to None.
      force_match_for_each_col: If True, ensures that each column is matched to
        at least one row (which is not guaranteed otherwise if the
        positive_threshold is high). Defaults to False.
      positive_value: An integer to fill for positive match indicators.
      negative_value: An integer to fill for negative match indicators.
      ignore_value: An integer to fill for ignored match indicators.

    Raises:
      ValueError: If negative_threshold > positive_threshold.
    """

  def __call__(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: A float tensor of shape [N, M], or [Batch_size, N, M]
        representing any similarity metric.

    Returns:
      A integer tensor of shape [N] with corresponding match indices for each
      of M columns, for positive match, the match result will be the
      corresponding row index, for negative match, the match will be
      `negative_value`, for ignored match, the match result will be
      `ignore_value`.
    """
```

#### Ops -- IOUSimilarity
```python
class IouSimilarity:
  """Class to compute similarity based on Intersection over Union (IOU) metric."""

  def __call__(self, groundtruth_boxes, anchors):
    """Compute pairwise IOU similarity between ground truth boxes and anchors.

    Args:
      groundtruth_boxes: a float Tensor with N boxes.
      anchors: a float Tensor with M boxes.

    Returns:
      A tensor with shape [M, N] or [batch_size, M, N] representing pairwise
        iou scores, anchor per row and groundtruth_box per colulmn.

    Input shape:
      groundtruth_boxes: [N, 4], or [batch_size, N, 4]
      anchors: [M, 4], or [batch_size, M, 4]

    Output shape:
      [M, N], or [batch_size, M, N]
    """
```

#### Ops -- TargerGather
```python
class AnchorLabeler:
  """Labeler for dense object detector."""

  def __init__(self):
    """Constructs Anchor Labeler."""

  def __call__(self, labels, match_indices, mask, mask_val=0.0):
    """Labels anchors with ground truth inputs.

    Args:
      labels: An integer tensor with shape [N, dim], or [B, N, dim] representing
        groundtruth classes.
      match_indices: An integer tensor with shape [N] or [B, N] representing match
        ground truth box index.
      mask: An integer tensor with shape [N] representing match
        labels, e.g., 1 for positive, -1 for negative, -2 for ignore.
      mask_val: An python primitive to fill in places where mask is True.
    """
```

#### Losses -- Focal
```python
class FocalLoss(tf.keras.losses.Loss):
  """Implements a Focal loss for classification problems.

  Reference:
    [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  """

  def __init__(self,
               alpha,
               gamma,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    """Initializes `FocalLoss`.

    Arguments:
      alpha: The `alpha` weight factor for binary class imbalance.
      gamma: The `gamma` focusing parameter to re-weight loss.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op. Defaults to 'retinanet_class_loss'.
    """
  def call(self, y_true, y_pred):
    """Invokes the `FocalLoss`.

    Arguments:
      y_true: A tensor of size [batch, num_anchors, num_classes]
      y_pred: A tensor of size [batch, num_anchors, num_classes]

    Returns:
      Summed loss float `Tensor`.
    """
```

#### Ops -- BoxCoder
```python
class BoxCoder:
  """box coder for RetinaNet, FasterRcnn, SSD, and YOLO."""

  def __init__(self, scale_factors=None):
    """Constructor for BoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw. If
        set to None, does not perform scaling. For Faster RCNN, the open-source
        implementation recommends using [10.0, 10.0, 5.0, 5.0].
      offset: The offset used to code the box coordinates, it can be 'sigmoid',
        i.e., coded_coord = coord + sigmoid(tx) which
        is used for RetinaNet, FasterRcnn, and SSD, or it can be 'linear',
        i.e., encoded_coord = coord + width * tx which is used for YOLO. 
    """
  def encode(self, boxes, anchors):
    """Compute coded_coord from coord."""
  def decode(self, boxes, anchors):
    """Compute coord from coded_coord."""
```

#### Layers -- NMSDetectionDecoder
```python
class NMSDetectionDecoder(tf.keras.layers.Layer):
  """Generates detected boxes with scores and classes for one-stage detector."""

  def __init__(self,
               pre_nms_top_k=5000,
               pre_nms_score_threshold=0.05,
               nms_iou_threshold=0.5,
               max_num_detections=100,
               use_batched_nms=False,
               **kwargs):
    """Initializes a detection generator.

    Args:
      pre_nms_top_k: int, the number of top scores proposals to be kept before
        applying NMS.
      pre_nms_score_threshold: float, the score threshold to apply before
        applying  NMS. Proposals whose scores are below this threshold are
        thrown away.
      nms_iou_threshold: float in [0, 1], the NMS IoU threshold.
      max_num_detections: int, the final number of total detections to generate.
      use_batched_nms: bool, whether or not use
        `tf.image.combined_non_max_suppression`.
      **kwargs: other key word arguments passed to Layer.
    """

  def call(self, raw_boxes, raw_scores, anchor_boxes, image_shape):
    """Generate final detections.

    Args:
      raw_boxes: a dict with keys representing FPN levels and values
        representing box tenors of shape
        [batch, feature_h, feature_w, num_anchors * 4].
      raw_scores: a dict with keys representing FPN levels and values
        representing logit tensors of shape
        [batch, feature_h, feature_w, num_anchors].
      anchor_boxes: a tensor of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: a tensor of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.

    Returns:
    `detection_boxes`: float Tensor of shape [batch, max_num_detections, 4]
      representing top detected boxes in [y1, x1, y2, x2].
    `detection_scores`: float Tensor of shape [batch, max_num_detections]
      representing sorted confidence scores for detected boxes. The values
      are between [0, 1].
    `detection_classes`: int Tensor of shape [batch, max_num_detections]
      representing classes for detected boxes.
    `num_detections`: int Tensor of shape [batch] only the first
      `num_detections` boxes are valid detections
    """
```

## Questions and Discussion Topics
Gathering feedbacks on arguments & naming conventions.