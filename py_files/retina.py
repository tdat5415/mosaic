import tensorflow as tf
import numpy as np
from py_files import util

def get_backbone():
    """builds ResNet50 with pre_trained imagenet weights"""
    backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])
    c3_output, c4_output, c5_output = [ backbone.get_layer(layer_name).output
                                        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
    return tf.keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

class FeaturePyramid(tf.keras.layers.Layer):
    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        # self.upsample_2x = tf.keras.layers.UpSampling2D(2)
        self.bn_c3 = tf.keras.layers.BatchNormalization()
        self.bn_c4 = tf.keras.layers.BatchNormalization()
        self.bn_c5 = tf.keras.layers.BatchNormalization()
        self.bn_c6 = tf.keras.layers.BatchNormalization()
        self.bn_c7 = tf.keras.layers.BatchNormalization()

        self.relu_ = tf.keras.layers.Activation('relu')

        self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        self.bias_init = tf.constant_initializer(-np.log((1-0.01) / 0.01))
        
        self.conv_c3_3x3_4 = tf.keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer="zeros")
        self.conv_c4_3x3_4 = tf.keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer="zeros")
        self.conv_c5_3x3_4 = tf.keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer="zeros")
        self.conv_c6_3x3_4 = tf.keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer="zeros")
        self.conv_c7_3x3_4 = tf.keras.layers.Conv2D(4, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer="zeros")

        self.conv_c3_3x3_1 = tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
        self.conv_c4_3x3_1 = tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
        self.conv_c5_3x3_1 = tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
        self.conv_c6_3x3_1 = tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
        self.conv_c7_3x3_1 = tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)

        self.reshape_4 = tf.keras.layers.Reshape([-1, 4])
        self.reshape_1 = tf.keras.layers.Reshape([-1, 1])


    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)

        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

        p3_output = self.relu_(self.bn_c3(p3_output))
        p4_output = self.relu_(self.bn_c4(p4_output))
        p5_output = self.relu_(self.bn_c5(p5_output))
        p6_output = self.relu_(self.bn_c6(p6_output))
        p7_output = self.relu_(self.bn_c7(p7_output))

        p3_box = self.conv_c3_3x3_4(p3_output)
        p4_box = self.conv_c4_3x3_4(p4_output)
        p5_box = self.conv_c5_3x3_4(p5_output)
        p6_box = self.conv_c6_3x3_4(p6_output)
        p7_box = self.conv_c7_3x3_4(p7_output)

        p3_cls = self.conv_c3_3x3_1(p3_output)
        p4_cls = self.conv_c4_3x3_1(p4_output)
        p5_cls = self.conv_c5_3x3_1(p5_output)
        p6_cls = self.conv_c6_3x3_1(p6_output)
        p7_cls = self.conv_c7_3x3_1(p7_output)

        p3_box = self.reshape_4(p3_box)
        p4_box = self.reshape_4(p4_box)
        p5_box = self.reshape_4(p5_box)
        p6_box = self.reshape_4(p6_box)
        p7_box = self.reshape_4(p7_box)

        p3_cls = self.reshape_1(p3_cls)
        p4_cls = self.reshape_1(p4_cls)
        p5_cls = self.reshape_1(p5_cls)
        p6_cls = self.reshape_1(p6_cls)
        p7_cls = self.reshape_1(p7_cls)

        return tf.concat([p3_box, p4_box, p5_box, p6_box, p7_box], axis=1), tf.concat([p3_cls, p4_cls, p5_cls, p6_cls, p7_cls], axis=1)
    
class RetinaNet(tf.keras.Model):
  # num_classes: Number of classes in the dataset.
  # backbone: ResNet50
    def __init__(self, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)

        # prior_probability = tf.constant_initializer(-np.log((1-0.01) / 0.01))
        # self.cls_head = build_head(9 * 1, prior_probability)
        # self.box_head = build_head(9 * 4, "zeros")
        # self.cls_head = build_head(1, prior_probability)
        # self.box_head = build_head(4, "zeros")

    def call(self, image, training=False):
        features_box, features_cls = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        # cls_outputs = [] 
        # box_outputs = []
        # for feature in features:
        #   box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
        #   cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, 1]))
        # cls_outputs = tf.concat(cls_outputs, axis=1)
        # box_outputs = tf.concat(box_outputs, axis=1) # shape(N, -1*5, 4)
        # cls_outputs = tf.reshape(cls_outputs, shape=[N, -1, 1])
        # box_outputs = tf.reshape(box_outputs, shape=[N, -1, 4]) # shape(N, -1*5, 4)
        # return tf.concat([box_outputs, cls_outputs], axis=-1)
        return tf.concat([features_box, features_cls], axis=-1)
    
class DecodePredictions(tf.keras.layers.Layer):
    # num_classes: Number of classes intehr dataset
    # confidence_threshold: Minimum class probability, below which detections are pruned.
    # nms_iou_threshold: IOU threshold for the NMS operation.
    # max_detections_per_class: -
    # max_detections: -
    # box_variance: the scaling factors used to scale the bounding box predictions.
    def __init__(self, 
               confidence_threshold = 0.05,
               nms_iou_threshold = 0.5,
               max_detections_per_class = 50,
               max_detections = 50,
               box_variance = [0.1, 0.1, 0.2, 0.2],
               **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = util.AnchorBox()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions): # LabelEncoder._compute_box_target과 비교 ㄱㄱ
        boxes = box_predictions * self._box_variance
        boxes = tf.concat([
                           boxes[:,:,:2] * anchor_boxes[:,:,2:] + anchor_boxes[:,:,:2],
                           tf.math.exp(boxes[:,:,2:]) * anchor_boxes[:,:,2:]
        ], axis=-1)
        boxes_transformed = util.convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions): # 한장씩 들어옴
        images_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(images_shape[1], images_shape[2]) # shape(5 * fh * fw * 9, 4)
        box_predictions = predictions[:,:,:4]
        cls_predictions = tf.nn.sigmoid(predictions[:,:,4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions) # (N, -1, 4)
        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2), # (N,-1,1,4) ?
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
    
def get_pretrained_model(weights_dir):
    resnet50_backbone = get_backbone()
    model = RetinaNet(resnet50_backbone)
    
    #weights_dir = "./model_ckpt"
    #weights_dir = "/content/drive/MyDrive/Colab Notebooks/얼굴모자이크/model"
    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)
    
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    
    return inference_model
