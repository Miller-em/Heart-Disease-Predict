import sys 
from GUI import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import tensorflow as tf 
from tensorflow import keras
import numpy as np

"""
下列数据用于做数据的归一化：
max_values:  [ 77.    1.    3.  200.  564.    1.    2.  202.    1.    6.2   2.    4.
   3.    1. ]
min_values:  [ 29.   0.   0.  94. 126.   0.   0.  88.   0.   0.   0.   0.   0.   0.]
avg_values:  [4.62574257e+01 5.67656766e-01 8.58085809e-01 1.11537954e+02
 2.10557756e+02 1.28712871e-01 4.45544554e-01 1.26963696e+02
 2.73927393e-01 8.79537954e-01 1.19471947e+00 6.13861386e-01
 1.94719472e+00 4.75247525e-01]
"""
metrics = {}
max_values = [77, 1, 3, 200, 564, 1, 2, 202, 1, 6.2, 2, 4, 3]
min_values = [29, 0, 0, 94, 126, 0, 0, 88, 0, 0, 0, 0, 0, 0]
avg_values = [4.62574257e+01, 5.67656766e-01, 8.58085809e-01, 1.11537954e+02,
 2.10557756e+02, 1.28712871e-01, 4.45544554e-01, 1.26963696e+02,
 2.73927393e-01, 8.79537954e-01, 1.19471947e+00, 6.13861386e-01,
 1.94719472e+00]
max_values = np.array(max_values,dtype="float32")
min_values = np.array(min_values,dtype="float32")
avg_values = np.array(avg_values,dtype="float32")
feather_num = 13

# 读取模型
model = keras.models.load_model('Z_model.h5')
model.summary()

class MyGUIDemo(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyGUIDemo, self).__init__()
        self.setupUi(self)
        self.signal(self)
        self.WidgetsUi(self)
    
    def WidgetsUi(self, MainWindow):
        # lineEdit
        self.lineEdit_2.setPlaceholderText('Please enter!')
        self.lineEdit_3.setPlaceholderText('Please enter!')
        self.lineEdit_4.setPlaceholderText('Please enter!')
        self.lineEdit_5.setPlaceholderText('Please enter!')
        self.lineEdit_6.setPlaceholderText('Please enter!')
        self.lineEdit.setPlaceholderText('Please enter!')

    def signal(self, MainWindow):
        # Button
        self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.generate_result)
        # lineedit
        self.lineEdit.textChanged.connect(self.check_input_func)
        self.lineEdit_2.textChanged.connect(self.check_input_func)
        self.lineEdit_3.textChanged.connect(self.check_input_func)
        self.lineEdit_4.textChanged.connect(self.check_input_func)
        self.lineEdit_5.textChanged.connect(self.check_input_func)
        self.lineEdit_6.textChanged.connect(self.check_input_func)
        # comboBox
        self.comboBox_2.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_3.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_4.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_5.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_6.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_7.currentIndexChanged.connect(self.check_input_func)
        self.comboBox_8.currentIndexChanged.connect(self.check_input_func)
    
    def generate_result(self):
        mt = [] #用于存各项指标
        metrics['age'] = self.lineEdit.text()
        metrics['sex'] = self.comboBox_2.currentIndex()
        metrics['cp'] = self.comboBox_8.currentIndex()
        metrics['trestbps'] = self.lineEdit_2.text()
        metrics['clol'] = self.lineEdit_4.text()
        metrics['fbs'] = self.comboBox_3.currentIndex()
        metrics['restecg'] = self.comboBox_4.currentIndex()
        metrics['thalach'] = self.lineEdit_3.text()
        metrics['exang'] = self.comboBox_5.currentIndex()
        metrics['oldpeak'] = self.lineEdit_6.text()
        metrics['slope'] = self.comboBox_6.currentIndex()
        metrics['ca'] = self.lineEdit_5.text()
        metrics['thal'] = self.comboBox_7.currentIndex()

        print(metrics)
        # 将字典中的值装入tensor, 同时进行预测
        for value in metrics.values():
            mt.append(value)
        mt = np.array(mt,dtype="float32")
        mt = np.expand_dims(mt, axis=0)
        metrics_str = str(metrics)
        for i in range(feather_num):
            mt[:, i] = (mt[:, i] - min_values[i]) / (max_values[i] - min_values[i])
        mt = tf.convert_to_tensor(mt, dtype="float32")
        predicted = model.predict(mt[:1])
        print(predicted[:1])
        isDisease = np.argmax(predicted[:1])
        if isDisease == 1:
            QMessageBox.warning(self, 
                                '您的初步预测结果', 
                                "您的输入的指标为：\n"+metrics_str+'\n'+"您的初步检测结果为："+'患病, 请及时就医！')
        elif isDisease == 0:
            QMessageBox.information(self, 
                                '您的初步预测结果', 
                                "您的输入的指标为：\n"+metrics_str+'\n'+"您的初步检测结果为："+'不患病')
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.clear()
        self.lineEdit_6.clear()
    
    def check_input_func(self):
        if self.lineEdit.text() and self.lineEdit_2.text() and \
            self.lineEdit_3.text() and self.lineEdit_4.text() and \
            self.lineEdit_5.text() and self.lineEdit_6.text() and \
            self.comboBox_7.currentIndex():
            self.pushButton.setEnabled(True)
        else:
            self.pushButton.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyGUIDemo()
    window.setWindowIcon(QtGui.QIcon("E:\Github代码管理\Keras_tf2\心脏病预测\Heart.ico"))
    window.setWindowTitle("基于神经网络的心脏病检测系统")
    window.show()
    sys.exit(app.exec_())
