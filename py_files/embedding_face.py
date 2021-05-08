import tensorflow as tf

class EmbeddingFace():
  def __init__(self, path):
    self.model = tf.keras.models.load_model(path)
    self.input_shape = self.model.input_shape[1:3]
  
  def __call__(self, face_imgs):
    if tf.rank(face_imgs) == 4:
      embedded_face = self.model.predict(face_imgs)
      return embedded_face
    else:
      return []
