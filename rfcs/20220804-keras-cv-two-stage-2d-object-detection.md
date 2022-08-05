# keras-cv Two Stage Two-Dimensional Object Detection API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com)|
| **Contributor(s)** | Francois Chollet (fchollet@google.com)|
| **Updated**   | 2022-08-04                                           |

## Objective

We aim at providing the core primitive components for training and serving two-stage two-dimensional object
detection models, specifically Faster RCNN.
Pretrained models will also be provided, similar to keras-applications.

## Key Benefits

Two-stage object detection models are state-of-art technique that powers many computer vision tasks, they provide
more accurate detection compared to single-stage models (such as SSD), while maintaining lower inference speed.

With this proposal, Keras users will be able to build end-to-end models with a simple API.

## Design overview

This proposal includes the specific core components for building faster rcnn models. It does not, however, include:

1. Model backbone, such as ResNet, or functions to generate feature maps
2. Detection heads, such as Feature Pyramid
3. metrics utilities such as COCO Evaluator, or visualization utils.
4. primitive components from [single-stage detector]([url](https://github.com/keras-team/governance/blob/master/rfcs/20200928-keras-cv-single-stage-2d-object-detection.md)), we will re-use those components in this design.

Data augmentation with ground truth box processing is currently being developed in KerasCV.

In this document, region of interest (roi) is used interchangeably with region proposal, or simply proposal.

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
  # but a batch of images (typically 2 per GPU) should have same size.
  return image, gt_boxes, gt_labels

anchor_generator = keras_cv.ops.AnchorGenerator(anchor_sizes, scales, aspect_ratios, strides)
similarity_calculator = keras_cv.layers.IOUSimilarity()
# positive anchor with IOU > 0.7, negative anchor with IOU <= 0.3
rpn_box_matcher = keras_cv.ops.BoxMatcher([0.7, 0.3])
# positive ROI with IOU > 0.5, negative ROI with IOU <= 0.5
rcnn_box_mather = keras_cv.ops.BoxMatcher(0.5)
target_gather = keras_cv.ops.TargetGather()
box_coder = keras_cv.ops.BoxCoder(offset='sigmoid')
rpn_= keras_cv.layers.ProposalSampler(positive_fraction=0.5, batch_size=256)
rcnn_sampler = keras_cv.layers.ProposalSampler(positive_fraction=0.25, batch_size=128)
rpn_labeler = keras_cv.ops.AnchorLabeler(rpn_sampler, rpn_box_matcher, similarity_calculator, box_coder)
rcnn_labeler = keras_cv.ops.AnchorLabeler(rcnn_sampler, rcnn_box_matcher, similarity_calculator, box_coder)
roi_filter = keras_cv.layers.ROIFilter(pre_nms_top_k=2000, nms_iou_threshold=0.7, test_pre_nms_top_k=1000)
roi_pooler = keras_cv.layers.ROIPooler(output_size=[7, 7])
# Build RPN and ROI Heads, use Keras backbone
backbone = tf.keras.applications.ResNet50()

def encode_rpn_label(image, gt_boxes, gt_labels):
  anchor_boxes = anchor_generator(image_size)
  cls_targets, box_targets, cls_weights, box_weights = rpn_labeler(anchor_boxes, gt_boxes, gt_labels)
  return (gt_boxes, gt_labels, cls_targets, box_targets), (cls_weights, box_weights)

class FasterRCNN(tf.keras.Model):
  # includes backbone and feature pyramid head.
  def __init__(self, backbone='resnet50_fpn', rpn_head, roi_head, roi_filter, roi_pooler):
    # self.backbone = Model Backbone that returns dict of feature map, or Feature Pyramid Network that wraps it
    # self.rpn_head = Region Proposal Network that provides objectness scores and bbox offset against anchor boxes
    # self.roi_filter = A filter layer that shrinks from a dense predictions to topk sparse predictions based on scores
    # self.roi_head = RCNN detection network that provides softmaxed classification score and bbox offset against rois
    # self.rpn_cls_loss_fn = a Binary CrossEntropy Keras loss 
    # self.rpn_reg_loss_fn = a Regression Keras loss, e.g., Huber loss
    # self.rcnn_cls_loss_fn = a Binary CrossEntropy Keras loss
    # self.rcnn_reg_loss_fn = a Regression Keras loss, e.g., Huber loss
  
  def call(self, image, training=None):
    # returns a single or multi level feature maps
    feature_map = self.backbone(image, training)
    # from the region proposal network, returns the predicted objectness scores
    # and class-agnostic offsets relative to anchor boxes
    rpn_cls_pred, rpn_bbox_pred = self.rpn_head(feature_map)
    # apply offset to anchors and recover proposal in (x1, y1, x2, y2) format
    rpn_rois = box_coder.decode_offset(anchors, rpn_bbox_pred)
    # select top-k proposals according to objectness scores
    rois, cls_pred = self.roi_filter(rpn_rois, rpn_cls_pred)
    # pooling feature map with variable sized rois to fixed size feature map
    feature_map = self.roi_pooler(feature_map, rois)
    # get class independent scores and bounding boxes offsets relative to proposals
    rcnn_cls_pred, rcnn_bbox_pred = self.roi_head(feature_map)
    if not training:
      rcnn_cls_pred, rcnn_bbox_pred = self.nms_detection_decoder(rois, rcnn_cls_pred, rcnn_bbox_pred, image_shape)
      return rcnn_cls_pred, rcnn_bbox_pred
    return {"rpn_cls_pred": rpn_cls_pred, "rpn_bbox_pred": rpn_bbox_pred, "rois": rois,
            "rcnn_cls_pred": rcnn_cls_pred, "rcnn_bbox_pred": rcnn_bbox_pred}
  
  def train_step(self, data):
    image, (gt_labels, gt_boxes, rpn_cls_targets, rpn_box_targets), (rpn_cls_weights, rpn_box_weights) = data
    # Using approximate joint training instead of alternating training
    with tf.GradientTape() as tape:
      outputs = self(x, training=True)
      # Compute RPN losses using targets from input pipeline, this will normalize by N_cls and N_reg as well
      rpn_cls_loss = rpn_cls_loss_fn(rpn_cls_targets, outputs["rpn_cls_pred"], rpn_cls_weights)
      rpn_box_loss = rpn_reg_loss_fn(rpn_box_targets, outputs["rpn_boxes_pred"], rpn_box_weights)
      # Compute RCNN losses which only picks k-th bbox prediction where k is the predicted class
      rois = outputs["rpn_rois"]
      rcnn_cls_true, rcnn_box_true, rcnn_cls_weights, rcnn_box_weights = self.rcnn_labeler(rois, gt_boxes, gt_labels)
      rcnn_cls_loss = rcnn_cls_loss_fn(rcnn_scores_true, outputs["rcnn_cls_scores"], rcnn_cls_weights)
      rcnn_box_loss = rcnn_reg_loss_fn(rcnn_box_true, outputs["rcnn_bbox_offsets"], rcnn_box_weights)
      total_loss = rpn_cls_loss + rpn_box_loss + rcnn_cls_loss + rcnn_box_loss
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    return self.compute_metrics(...)
      

transformed_train_ds = train_ds.map(preprocess).map(encode_rpn_label).batch(128).shuffle(1024)
transformed_eval_ds = eval_ds.map(preprocess).map(encode_rpn_label).batch(128)

strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
    optimizer = tf.keras.optimizers.SGD(lr_scheduler)
    model = RetinaNet()
    model.compile(optimizer=optimizer,
                  loss={'classification': keras_cv.losses.Focal(), 'regression': tf.keras.losses.Huber()},
                  metrics=[])

model.fit(transformed_train_ds, epochs=100, validation_data=transformed_eval_ds)
model.save(file_path)
``` 

#### Serving

Case where a user want to serve the trained model for a single image, this will be identical to single-stage object detector.

## Detailed Design

For the rest of the design, we denote `B` as batch size, `N` as the number of ground truth boxes, and `M` as the number
of anchor boxes.

We propose 3 layers and 1 op in this RFC.

#### Layers -- ProposalSampler
Given a dense anchor/proposal set, we propose ProposalSampler layer to for selecting positive and negative proposals according
to the required batch size and positive : negative ratio
boxes or anchor boxes and pass a mask
 
```python
class ProposalSampler(tf.keras.layers.Layer):
  """Class to select positive and negative proposals."""
 
  def __init__(self, positive_fraction, batch_size, positive_indicator=1, negative_indicator=-1):
    """Initializes ProposalSampler layer.
    Args:
      positive_fraction: A float number between [0, 1], 0.5 means positive:negative ratio is 1:1
      batch_size: the number of samples to generate
      positive_indicator: for the inputs to the layer, value for positive proposal, default to 1
      negative_indicator: for the inputs to the layer, value for negative proposal, default to -1
    """
 
  def call(self, matched_indicators):
    """Get a balanced positive and negative samples.
 
    Args:
      matched_indicators: A int Tensor [N], or [B, N] represent positive or negative values
 
    Returns:
      Int tensors with shape [sample_size] or [B, sample_size] representing the selected indices for propsals.
 
    """
```

#### Layers -- ROIPooler
We propose ROIPooler layer to crop feature maps from proposals
 
```python
class ROIPooler(tf.keras.layers.Layer):
  """Class to compute extract feature maps from region proposals by quantization."""
 
  def __init__(self, output_size=[7, 7]):
    """Initializes ROIPooler layer.
    Args:
      output_size: A tuple representing the output height and width. 
    """
 
  def call(self, feature_maps, rois):
    """Compute pairwise IOU similarity between ground truth boxes and anchors.
 
    Args:
      groundtruth_boxes: A float Tensor [H, W, C] or [B, H, W, C] or dict of multiple levels
      rois: A float or int Tensor [M], or [B, M] represent coordinates within [H, W].
 
    Returns:
      A float tensor with shape [output_size] or [B, output_size] representing cropped feature maps.
    """
```

#### Layers -- ROIFilter
We propose ROIFilter layer to select top-k proposals based on some score
 
```python
class ROIFilter(tf.keras.layers.Layer):
  """Class to select top-k proposals based on some score."""
 
  def __init__(self, 
               pre_nms_top_k: int = 2000,
               pre_nms_score_threshold: float = 0.0,
               pre_nms_min_size_threshold: float = 0.0,
               nms_iou_threshold: float = 0.7,
               num_proposals: int = 1000,
               test_pre_nms_top_k: int = 1000,
               test_pre_nms_score_threshold: float = 0.0,
               test_pre_nms_min_size_threshold: float = 0.0,
               test_nms_iou_threshold: float = 0.7,
               test_num_proposals: int = 1000,
               use_batched_nms: bool = False,):
    """Initializes ROIFilter layer.
    Args:
      pre_nms_top_k: An `int` of the number of top scores proposals to be kept
        before applying NMS.
      pre_nms_score_threshold: A `float` of the score threshold to apply before
        applying NMS. Proposals whose scores are below this threshold are
        thrown away.
      pre_nms_min_size_threshold: A `float` of the threshold of each side of the
        box (w.r.t. the scaled image). Proposals whose sides are below this
        threshold are thrown away.
      nms_iou_threshold: A `float` in [0, 1], the NMS IoU threshold.
      num_proposals: An `int` of the final number of proposals to generate.
      test_pre_nms_top_k: An `int` of the number of top scores proposals to be
        kept before applying NMS in testing.
      test_pre_nms_score_threshold: A `float` of the score threshold to apply
        before applying NMS in testing. Proposals whose scores are below this
        threshold are thrown away.
      test_pre_nms_min_size_threshold: A `float` of the threshold of each side
        of the box (w.r.t. the scaled image) in testing. Proposals whose sides
        are below this threshold are thrown away.
      test_nms_iou_threshold: A `float` in [0, 1] of the NMS IoU threshold in
        testing.
      test_num_proposals: An `int` of the final number of proposals to generate
        in testing.
      use_batched_nms: A `bool` of whether or not use
        `tf.image.combined_non_max_suppression`.
    """
 
  def call(self, self,
           rois: Mapping[str, tf.Tensor],
           raw_scores: Mapping[str, tf.Tensor],
           image_shape: tf.Tensor):
    """.
 
    Args:
      rois: A float Tensor [N], or [B, N] represent region proposals.
      roi_scores: A float Tensor [N], or [B, N] represent scores for each region.
      image_shape: A int tensor [2] or [B, 2] representing image size.
 
    Returns:
      roi: A `tf.Tensor` of shape [B, num_proposals, 4], the proposed
        ROIs in the scaled image coordinate.
      roi_scores: A `tf.Tensor` of shape [B, num_proposals], scores of the
        proposed ROIs.

    """
```

#### Ops -- AnchorLabeler

```python
class AnchorLabeler:
  """Labelers that matches ground truth with anchors and proposals."""

  def __init__(self,
               proposal_sampler,
               proposal_matcher,
               similarity_calculator,
               box_coder):
    """.

    Args:
      proposal_sampler: a ProposalSampler
      proposal_matcher: A BoxMatcher
      similarity_calculator: Such as IOU layer
      box_coder: a BoxCoder that transforms between different formats

    """
  def __call__(self, proposals, gt_boxes, gt_labels):
    """
    Args:
      proposals: a float [N, 4] Tensor represent different proposals.
      gt_boxes: a float [M, 4] Tensor represent ground truth boxes.
      gt_labels: a int [M] Tensor represent ground truth labels.
    Returns:
      cls_targets: a int [K] Tensor represent mapped proposal labels from ground truth labels.
      box_targets: a float [K, 4] Tensor represent mapped proposal boxes from ground truth boxes.
      cls_weights: a float [K] Tensor represent weights for each cls_targets
      box_weights: a float [K] or [K, 4] Tensor represent weights for each box_targets
    """
```

## Questions and Discussion Topics
* Should we provide a meta arch for FasterRCNN.
* SHould we provide some default out-of-box RPN Head and ROI Head.
