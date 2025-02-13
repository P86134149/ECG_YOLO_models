import tensorflow as tf

# 加载模型
file_name = 'MIX_LMUEBCnet_1D_normalized.h5'
model = tf.keras.models.load_model(file_name)

# 转换模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

# 保存转换后的模型
with open(f"{file_name.split('.')[0]}.tflite", "wb") as f:
    f.write(tfmodel)

#tflite_convert --keras_model_file=MIX_resnetV1_1D_d32_normalized.h5 --output_file=MIX_resnetV1_1D_d32_normalized.tflite