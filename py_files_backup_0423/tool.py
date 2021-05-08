import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt
from py_files import util

def detect_face(image, inference_model):
  image = tf.cast(image, dtype=tf.float32)
  input_image, ratio = util.prepare_image2(image)
  detections = inference_model.predict(input_image)
  num_detections = detections[3][0]
  boxes = detections[0][0, :num_detections] / ratio.numpy()

  image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
  boxes = tf.stack([
                     tf.clip_by_value(boxes[:, 0], 0.0, image_shape[1] - 1.0),
                     tf.clip_by_value(boxes[:, 1], 0.0, image_shape[0] - 1.0),
                     tf.clip_by_value(boxes[:, 2], 0.0, image_shape[1] - 1.0),
                     tf.clip_by_value(boxes[:, 3], 0.0, image_shape[0] - 1.0),
  ], axis=-1)

  boxes = util.convert_corners_to_xmymwh(boxes)
  return boxes

def crop_square_face(image, boxes, shape):
  if len(boxes) == 0:
    return []
  boxes = tf.cast(boxes, dtype=tf.int32)
  face_images = tf.map_fn(lambda x:crop_pad_resize(x, image, shape), boxes)
  return tf.cast(face_images, dtype=tf.uint8)

def crop_pad_resize(box, image, shape):
  x1, y1, w, h = box.numpy()
  croped_image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
  croped_image = tf.image.pad_to_bounding_box(
    croped_image, (max(w,h)-min(w,h))//2*(h<w), (max(w,h)-min(w,h))//2*(h>=w),
    max(w,h), max(w,h)
  )
  croped_image = tf.image.resize(croped_image, shape)
  return tf.cast(croped_image, dtype=tf.int32)

def add_encoding(name, enables, boxes, encodinges):
  if len(boxes) == 0:
    return enables
  big_ind = tf.argmax(boxes[:, 2] * boxes[:, 3])
  if name in enables:
    enables[name] = tf.concat([enables[name], encodinges[big_ind:big_ind+1]], axis=0)
  else:
    enables[name] = encodinges[big_ind:big_ind+1]
  return enables

def encodinges_classify(encodinges, enables):
  if len(encodinges) == 0:
    return []
  enables = list(map(lambda x:tf.reduce_mean(x, axis=0), enables.values()))
  enables = tf.stack(enables, axis=0)
  mae = tf.map_fn(lambda x:tf.keras.metrics.mean_absolute_error(x, enables),
                  encodinges) # shape(len(encodinges), len(enables))
  mae = tf.reduce_min(mae, axis=-1)
  mask = tf.where(mae < 0.45, True, False)
  return mask

def add_mosaic(image, boxes, mask, mode='positive'):
  if len(mask) == 0:
    return image
  boxes = tf.stack(boxes, axis=0)
  mask = tf.stack(mask, axis=0)
  assert tf.rank(boxes) == 2
  assert tf.shape(mask)[0] == tf.shape(boxes)[0]
  assert tf.rank(image) == 3
  assert mode in ['positive', 'negative']
  
  boxes = tf.cast(boxes, dtype=tf.int32)
  image = tf.cast(image, dtype=tf.uint8).numpy()
  
  if mode == 'negative':
    pass
  elif mode == 'positive':
    mask = mask ^ True 

  for box, cover in zip(boxes, mask):
    if cover:
      x1, y1, w, h = box
      croped_image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
      img_shape = tf.convert_to_tensor([h,w])
      temp = tf.image.resize(croped_image, [10,8], method=method)
      mosaic = tf.image.resize(temp, img_shape, method=method)

      image[y1:y1+h, x1:x1+w, :] = mosaic

  return image
