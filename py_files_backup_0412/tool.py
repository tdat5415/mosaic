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
'''
def crop_square_face(image, boxes, shape):
  centers = boxes[:, :2] + boxes[:, 2:] / 2
  # side_len = tf.reduce_max(boxes[:,2:], axis=-1, keepdims=True)
  side_len = tf.reduce_min(boxes[:,2:], axis=-1, keepdims=True)
  left_top = centers - side_len / 2
  squares = tf.cast(tf.concat([left_top, side_len], axis=-1), dtype=tf.int32)
  face_images = []
  for box in squares:
    x1, y1, s = box
    croped_image = tf.image.crop_to_bounding_box(image, y1, x1, s, s)
    croped_image = tf.image.resize(croped_image, shape)
    face_images.append(croped_image)

  return tf.cast(tf.stack(face_images, axis=0), dtype=tf.int32)
'''
def crop_square_face(image, boxes, shape):
  face_images = []
  boxes = tf.cast(boxes, dtype=tf.int32)
  for box in boxes:
    x1, y1, w, h = box.numpy()
    croped_image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
    croped_image = tf.image.pad_to_bounding_box(
      croped_image, (max(w,h)-min(w,h))//2*(h<w), (max(w,h)-min(w,h))//2*(h>=w),
      max(w,h), max(w,h)
    )
    croped_image = tf.image.resize(croped_image, shape)
    face_images.append(croped_image)

  return tf.cast(tf.stack(face_images, axis=0), dtype=tf.int32)

def add_mosaic(image, boxes, mask, mode='positive'):
  assert mask.dtype == tf.bool
  assert tf.rank(boxes) == 2
  assert tf.rank(mask) == 1
  assert tf.shape(mask)[0] == tf.shape(boxes)[0]
  assert tf.rank(image) == 3
  assert mode in ['positive', 'negative']
  
  if mode == 'positive':
    pass
  elif mode == 'negative':
    mask = mask ^ True 

  boxes = tf.cast(boxes, dtype=tf.int32)
  image = tf.cast(image, dtype=tf.int32).numpy()

  for box, cover in zip(boxes, mask):
    if cover:
      x1, y1, w, h = box
      croped_image = tf.image.crop_to_bounding_box(image, y1, x1, h, w)
      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
      img_shape = tf.convert_to_tensor([h,w])
      temp = tf.image.resize(croped_image, [10,8], method=method)
      mosaic = tf.image.resize(temp, img_shape, method=method)

      image[y1:y1+h, x1:x1+w, :] = mosaic

  return tf.convert_to_tensor(image)

def add_encoding(name, enables, boxes, encodinges):
  big_ind = tf.argmax(boxes[:, 2] * boxes[:, 3])

  if name in enables:
    enables[name] = tf.concat([enables[name], encodinges], axis=0)
  else:
    enables[name] = encodinges

  return enables

def encodinges_classify(encodinges, enables):
  mae_list = []
  for one in enables.values():
    one = tf.reduce_mean(one, axis=0)
    mae = tf.keras.metrics.mean_absolute_error(one, encodinges)
    mae_list.append(mae)
    # for e in one:
    #   mae = tf.keras.metrics.mean_absolute_error(e, encodinges)
    #   mae_list.append(mae)
  mae_list = tf.stack(mae_list, axis=0) # shape(len(enables), len(encodinges))
  mae_list = tf.reduce_min(mae_list, axis=0)
  print(mae_list)
  mask = tf.where(mae_list < 0.45, True, False)
  return mask
