#convert_base_code

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


###########
###########
# handrecognition_model_trained.h5 파일 load 하는 코드 작성
# keras.models.load_model 함수 이용

model = tf.keras.models.load_model('D:\\base_code\\1_convert_base_code\\handrecognition_model_trained.h5', compile=False)


###########
###########
# load한 handrecognition_model_trained.h5 모델을 .tflite모델로 변환하는 코드 작성
#
path = 'D:\\base_code\\1_convert_base_code\\'
model.save(path, save_format="tf")

converter = tf.lite.TFLiteConverter.from_saved_model('D:\\base_code\\1_convert_base_code\\')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

open('D:\\base_code\\1_convert_base_code\\model.tflite', 'wb').write(tflite_model)