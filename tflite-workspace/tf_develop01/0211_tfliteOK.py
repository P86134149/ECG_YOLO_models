import tensorflow as tf
import numpy as np
import pandas as pd

# 讀取 CSV 檔案
csv_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\tflite-workspace\tf_develop01\100m2_N.csv"
data = pd.read_csv(csv_path, header=None)
input_data = np.array(data.values, dtype=np.float32)

# 載入 TFLite 模型
# model_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\1D\5\MIX_LMUEBCnet_1D_normalized_converted.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 獲取模型的輸入和輸出張量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 準備輸入數據
input_data = np.resize(input_data, (256, 1))  # 調整數據形狀為 [256, 1]
input_data = np.expand_dims(input_data, axis=0)  # 添加批次維度，最終形狀為 [1, 256, 1]
input_shape = input_details[0]['shape']
print(f"Input shape: {input_shape}")

# 設置模型輸入
interpreter.set_tensor(input_details[0]['index'], input_data)

# 執行推論
interpreter.invoke()

# 獲取模型輸出
output_data = interpreter.get_tensor(output_details[0]['index'])

# 獲取前3個分類結果
top_3_indices = np.argsort(output_data[0])[-3:][::-1]
top_3_scores = output_data[0][top_3_indices]

# 輸出結果
for i, (index, score) in enumerate(zip(top_3_indices, top_3_scores)):
    print(f"Top {i+1} class index: {index}, score: {score}")