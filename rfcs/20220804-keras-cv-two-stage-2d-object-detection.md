# keras-cv Two Stage Two-Dimensional Object Detection API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com), Francois Chollet (fchollet@google.com)|
| **Contributor(s)** | Yeqing Li (yeqing@google.com)|
| **Updated**   | 2022-08-04                                           |

## Objective

We aim at providing the core primitive components for training and serving two-stage two-dimensional object
detection models, specifically Faster RCNN.
Pretrained models will also be provides, similar to keras-applications.

## Key Benefits

Two-stage object detection models are a state-of-art technique that powers many computer vision tasks, they provide
more accurate detection compared to single-stage models (such as SSD), while maintaining comparable performance.

With this proposal, Keras users will be able to build end-to-end models with a simple API.

## Design overview

This proposal includes the specific core components for building faster rcnn models. It does not, however, include:

1. Model backbone, such as ResNet, or functions to generate feature maps
2. Detection heads, such as Feature Pyramid
3. metrics utilities such as COCO Evaluator, or visualization utils.
4. primitive components from [single-stage detector]([url](https://github.com/keras-team/governance/blob/master/rfcs/20200928-keras-cv-single-stage-2d-object-detection.md)).

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
box_matcher = keras_cv.ops.BoxMatcher(positive_threshold, negative_threshold)
target_gather = keras_cv.ops.TargetGather()
box_coder = keras_cv.ops.BoxCoder(offset='sigmoid')
rpn_sampler = keras_cv.layers.ProposalSampler()
rcnn_sampler = keras_cv.layers.ProposalSampler()
rpn_labeler = keras_cv.ops.AnchorLabeler()
rcnn_labeler = keras_cv.ops.AnchorLabeler()

def encode_label(image, gt_boxes, gt_labels):
  anchor_boxes = anchor_generator(image_size)
  iou = similarity_calculator(gt_boxes, anchor_boxes)
  match_indices, match_indicators = box_matcher(iou)

  mask = tf.less_equal(match_indicators, 0)
  class_mask = tf.expand_dims(mask, -1)
  box_mask = tf.tile(class_mask, [1, 4])

  class_targets = target_gather(gt_labels, match_indices, class_mask, -1)
  box_targets = target_gather(gt_boxes, match_indices, box_mask, 0.0)
  box_targets = box_coder.encode(box_targets, anchor_boxes)

  weights = tf.squeeze(tf.ones_like(gt_labels), axis=-1)
  ignore_mask = tf.equal(match_indicators, -2)
  class_weights = target_gather(weights, match_indices, ignore_mask, 0.0)
  box_weights = target_gather(weights, match_indices, mask, 0.0)

  return (image, {'classification': class_targets, 'regression': box_targets},
          {'classification': class_weights, 'regression': box_weights})

class FasterRCNN(tf.keras.Model):
  # includes backbone and feature pyramid head.
  def __init__(self, backbone='resnet50_fpn', roi_head, detection_head, roi_filter, roi_pooler, roi_sampler):
    # self.backbone = Model Backbone that returns dict of feature map, or Feature Pyramid Network that wraps it
    # self.rpn_head = Region Proposal Network that provides objectness scores and bbox offset against anchor boxes
    # self.roi_filter = A filter layer that shrinks from a dense predictions to topk sparse predictions based on scores
    # self.roi_head = RCNN detection network that provides softmaxed classification score and bbox offset against rois
  
  def call(self, image, training=None):
    # returns a single or multi level feature maps
    feature_map = self.backbone(image, training)
    # from the region proposal network, returns the predicted objectness scores
    # and class-agnostic offsets relative to anchor boxes
    rpn_cls_scores, rpn_bbox_offsets = self.rpn_head(feature_map)
    # apply offset to anchors and recover proposal in (x1, y1, x2, y2) format
    rpn_rois = box_coder.decode_offset(anchors, rpn_bbox_offsets)
    # select top-k proposals according to objectness scores
    rois, cls_scores = self.roi_filter(rpn_rois, rpn_cls_scores)
    # pooling feature map with variable sized rois to fixed size feature map
    feature_map = self.roi_pooler(feature_map, rois)
    # get class independent scores and bounding boxes offsets relative to proposals
    rcnn_cls_scores, rcnn_bbox_offsets = self.roi_head(feature_map)
    if not training:
      rcnn_cls_scores, rcnn_bboxes = self.nms_detection_decoder(rois, rcnn_cls_scores, rcnn_bbox_offsets, image_shape)
      return rcnn_cls_scores, rcnn_bboxes
    return {"rpn_binary_scores": rpn_cls_scores, "rpn_bbox_offsets": rpn_bbox_offsets, "rpn_rois": rpn_rois,
            "rcnn_cls_scores": rcnn_cls_scores, "rcnn_bbox_offsets": rcnn_bbox_offsets}
  
  def train_step(self, data):
    image, (gt_labels, gt_boxes, anchors, rpn_scores_true, rpn_box_true), sample_weights = data
    # Using approximate joint training instead of alternating training
    with tf.GradientTape() as tape:
      outputs = self(x, training=True)
      # Compute RPN losses using targets from input pipeline, this will normalize by N_cls and N_reg as well
      rpn_cls_loss = rpn_cls_loss_fn(rpn_scores_true, outputs["rpn_scores"])
      rpn_box_loss = rpn_reg_loss_fn(rpn_box_true, outputs["rpn_boxes_offsets"])
      # Compute RCNN losses which only picks k-th bbox prediction where k is the predicted class
      rois = outputs["rpn_rois"]
      rcnn_cls_true, rcnn_box_true = self.rcnn_labeler(rois, gt_boxes, gt_labels)
      rcnn_cls_loss = rcnn_cls_loss_fn(rcnn_scores_true, outputs["rcnn_cls_scores"])
      rcnn_box_loss = rcnn_reg_loss_fn(rcnn_box_true, outputs["rcnn_bbox_offsets"])
      total_loss = rpn_cls_loss + rpn_box_loss + rcnn_cls_loss + rcnn_box_loss
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    return self.compute_metrics(...)
      

transformed_train_ds = train_ds.map(preprocess).map(encode_label).batch(128).shuffle(1024)
transformed_eval_ds = eval_ds.map(preprocess).map(encode_label).batch(128)

strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
    optimizer = tf.keras.optimizers.SGD(lr_scheduler)
    model = RetinaNet()
    model.compile(optimizer=optimizer,
                  loss={'classification': keras_cv.losses.Focal(), 'regression': tf.keras.losses.Huber()},
                  metrics=[])

model.fit(transformed_train_ds, epochs=120, validation_data=transformed_eval_ds)
model.save(file_path)
``` 

#### Serving

Case where a user want to serve the trained model for a single image.

```python
loaded_model = tf.keras.models.load(file_path)
box_coder = keras_cv.ops.BoxCoder(offset='sigmoid')
anchor_generator = keras_cv.ops.AnchorGenerator()
anchor_boxes = anchor_generator(image_size)
detection_generator = keras_cv.layers.NMSDetectionDecoder()

@tf.function
def serving_fn(image):
  batched_image = tf.expand_dims(image)
  raw_boxes, scores = loaded_model(batched_image, training=False)
  decoded_boxes = box_coder.decode(raw_boxes, anchor_boxes)
  classes, scores, boxes, _ = detection_generator(scores, decoded_boxes)
  return {'classes': classes, 'scores': scores, 'boxes': boxes}
```

## Detailed Design

For the rest of the design, we denote `B` as batch size, `N` as the number of ground truth boxes, and `M` as the number
of anchor boxes.

We propose 2 layers, 1 loss and 4 ops in this RFC.


## Questions and Discussion Topics
