import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
import numpy as np
from tensorflow.python.keras.layers.core import Dropout

def load_data():
    # 利用pandas读取数据集
    cv_path = '.\dataset.csv'
    df = pd.read_csv(cv_path)     

    # 转化数据格式及数据整理
    df = np.array(df)
    df = np.random.permutation(df)
    # data = tf.convert_to_tensor(df, tf.float32)
    feather_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']
    feather_num = len(feather_names)

    ratio = 0.85
    offset = int(df.shape[0] * ratio)
    training_data = df[:offset]

    #数据归一化
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = \
                        training_data.max(axis=0), \
                        training_data.min(axis=0), \
                        training_data.sum(axis=0) / df.shape[0]
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feather_num):
        #print(maximums[i], minimums[i], avgs[i])
        df[:, i] = (df[:, i] - minimums[i]) / (maximums[i] - minimums[i])
    # 训练集和测试集的划分比例
    training_data = df[:offset]
    test_data = df[offset:]
    training_data = tf.convert_to_tensor(training_data, tf.float32)
    test_data = tf.convert_to_tensor(test_data, tf.float32)
    return training_data, test_data

# 获取数据
num_classes = 2
input_shape = (13,)
batch_size = 8
epochs = 15
training_data, test_data = load_data()
x_train = training_data[:, :-1]
y_train = training_data[:, -1:]
y_train = keras.utils.to_categorical(y_train, num_classes)

x_test = test_data[:, :-1]
y_test = test_data[:, -1:]
y_test = keras.utils.to_categorical(y_test, num_classes)

#建立模型
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax")
    ]
)
# 查看网络结构
model.summary()

# 配置网络模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 启动训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# 进行模型评估
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# 模型保存
model.save(".\Z_model.h5")
print("max_values: ", max_values)
print("min_values: ", min_values)