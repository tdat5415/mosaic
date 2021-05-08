import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def swap_xy(boxes):
    boxes = tf.cast(boxes, dtype=tf.float32)
    return tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis=-1) # 쌓음 # 1차원 -> 2차원

def convert_to_xywh(boxes):
    center = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    WnH    = boxes[..., 2:] - boxes[..., :2]
    return tf.concat([center, WnH], axis=-1)

def convert_to_corners(boxes):
    xymin = boxes[..., :2] - boxes[..., 2:] / 2.0
    xymax = boxes[..., :2] + boxes[..., 2:] / 2.0
    return tf.concat([xymin, xymax], axis=-1)

def convert_xmymwh_to_corners(boxes):
    xymin = boxes[..., :2]
    xymax = xymin + boxes[..., 2:]
    return tf.concat([xymin, xymax], axis=-1)

def convert_corners_to_xmymwh(boxes):
  xymin = boxes[..., :2]
  wh = boxes[..., 2:] - xymin
  return tf.concat([xymin, wh], axis=-1)

def resize_and_pad_image(image, # (height, width, channels)
                         min_side=800.0, max_side=1333.0, jitter=[1200,1200], stride=128.0):
    # pad with zeros on right and botton to make the image shape divisible by 'stride'
    """return
    image: resized and padded image.
    image_shape: shape of the image before padding.
    ratio: the scaling factor used to resize the image.
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32) # 채널빼고
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
      tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
      image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

def compute_iou(boxes1, boxes2):
    """
    boxes1 : shape(N, 4), [x,y,width,height]
    boxes2 : shape(M, 4), [x,y,width,height]

    return
    IOU matrix : shape(N,M)
    """
    boxes1_corners = convert_to_corners(boxes1) # [xmin, ymin, xmax, ymax]
    boxes2_corners = convert_to_corners(boxes2) # [xmin, ymin, xmax, ymax]
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) # left up # shape(N, M, 2)
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:]) # right down # shape(N, M, 2)
    intersection = tf.maximum(0.0, rd - lu) # if rd - lu < 0, then 0
    intersection_area = intersection[:,:,0] * intersection[:,:,1] # intersection boxes's  (width * height) # shape(N, M)
    boxes1_area = boxes1[:,2] * boxes1[:,3] # shape(N,)
    boxes2_area = boxes2[:,2] * boxes2[:,3] # shape(M,)
    union_area = tf.maximum(
      boxes1_area[:, None] + boxes2_area - intersection_area, # shape(N, M)
      1e-8) # epsilon : to avoid divided by zero
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0) # 1이상은 1, 0미만은 0으로 # 합집합 / 교집합

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def resize_and_pad_image2(image, default_shape=(768, 1280)):
    original_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    ratio = tf.cast(default_shape[0],dtype=tf.float32) / original_shape[0]
    resize_shape = original_shape * ratio
    if resize_shape[1] > default_shape[1]:
      ratio = default_shape[1] / original_shape[1]
      resize_shape = original_shape * ratio

    image = tf.image.resize(image, tf.cast(resize_shape, dtype=tf.int32))
    image = tf.image.pad_to_bounding_box(
      image, 0, 0, default_shape[0], default_shape[1]
    )
    return image, tf.cast(original_shape, dtype=tf.int32), ratio

def prepare_image2(image, default_shape=(384, 384)):
    image, _, ratio = resize_and_pad_image2(image, default_shape)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def visualize_detections(image, boxes, figsize=(14,14), linewidth=1, color=[0,0,1]):
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    boxes = boxes.numpy()
    ax = plt.gca()
    for box in boxes:
        x1, y1, w, h = box
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
    plt.show()
    return ax

class AnchorBox:
    """Generates anchor boxes.
    for feature maps at strides [8, 16, 32, 64, 128].
    format: [x, y, width, height]
    """
    def __init__(self):
        # self.aspect_ratios = [0.5, 1.0, 2.0] # 가로세로비율
        self.aspect_ratios = [1.0] # 가로세로비율
        # self.scales = [2 ** x for x in [0, 1/3, 2/3]]
        self.scales = [2 ** x for x in [0]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales) # maybe num of anchor in cell
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims() # 3 * [1,9,2]

    def _compute_dims(self):
        anchor_dims_all = [] # 5 * [1,1,9,2]
        for area in self._areas:
            anchor_dims = [] # 3*3 * [1,1,2]
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio) 
                anchor_width = area / anchor_height
                dims = tf.reshape( # shape(2,) -> shape(1,1,2)
                    tf.stack([anchor_width, anchor_height], axis=-1), [1,1,2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all # 5 * [1,1,9,2]
  
    def _get_anchors(self, feature_height, feature_width, level):
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5 # len : num grid width ?
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        #centers = tf.stack(tf.meshgrid(rx,ry), axis=-1) * self._strides[level] # shape(fh, fw, 2) # make real value
        centers = tf.stack(tf.meshgrid(rx,ry), axis=-1) * tf.cast(self._strides[level], dtype=tf.float32) # shape(fh, fw, 2) # make real value
        centers = tf.expand_dims(centers, axis=-2) # shape(fh, fw, 1, 2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1]) # shape(fh, fw, 9, 2)
        dims = tf.tile( # shape(len(fh, fw, 9, 2)
            self._anchor_dims[level], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1) # shape(fh, fw, 9, 4)
        return tf.reshape(anchors, [feature_height * feature_width * self._num_anchors, 4]) # shape(fh * fw * 9, 4)

    def get_anchors(self, image_height, image_width):
        anchors = [
                   self._get_anchors(
                       tf.math.ceil(image_height / v), # ex> 160 / 8 = 20 # num of grid height # 8 is cellsize
                       tf.math.ceil(image_width / v),
                       i
                   )
                  #  for i in range(3,8)
                   for i, v in enumerate(self._strides)
        ]
        return tf.concat(anchors, axis=0) # shape(5 * fh * fw * 9, 4)
