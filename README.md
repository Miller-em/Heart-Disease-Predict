###  基于神经网络的心脏病健康系统





#### 导语

这篇文章旨在记录该系统设计的过程，同时指导从零开始搭建本健康系统的环境，并在自己的电脑上把这个心脏病健康系统run起来。

下面是这个文件夹下各个文件的介绍：

```
.\心脏病预测
├─build						---	该系统可执行文件的相关配置文件
  ├─....
├─dist
  ├─Model_Presicted.exe		  --- 基于神经网络的心脏病预测系统的可执行文件，双击可以打开，有点慢，不推荐
├─dataset.csv				 --- 心脏病各项指标的数据集
├─DiseasePredict_GUI.ui		  --- 由Qt designer生成的.ui文件
├─GUI.py					---  由.ui文件转化为的.py文件,作为模块导入
├─Heart.ico					 --- 系统的图标
├─Model_Predict.py			 --- 运行该脚本运行系统
├─README.md					 --- 手册
├─Z_model.h5				 --- 预训练模型
├─Z_nn_keras.py				 --- 用于数据集的训练的网络配置
```



#### 安装Python

因为本系统是由纯python实现，所以需要安装python的环境。

下面有一篇博客指导安装：https://blog.csdn.net/Wang_Jiankun/article/details/80565719

检验是否安装成功-->打开**cmd**-->输入python：

![image-20210129211344787](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722184244.png)



#### 安装tensorflow

因为现在已经用的是**tf2**， 虽然我们用的是keras，但是现在keras和tensorflow已经合并了，所以我们只需要安装**tensorflow**就好了。不懂tensorflow和keras做什么的不要紧，只需要看一下百度百科，知道是是干嘛的。

打开**cmd**-->输入：`pip install tensorflow-cpu -i https://pypi.douban,com/simple`



#### 安装pyqt5

我们这套系统的图形界面是由pyqt5来写的，我们需要安装的命令如下：

```
pip install PyQt5 -i https://pypi.douban.com/simple
pip install PyQt5-tools -i https://pypi.douban.com/simple
```



#### 数据集介绍：

数据源： [UCI开源数据集heart_disease](http://archive.ics.uci.edu/ml/datasets/Heart+Disease)
针对美国某区域的心脏病检查患者的体测数据，共303条数据。具体字段如下表：

| 字段名   | 含义                          | 类型   | 描述                                                     |
| :------- | :---------------------------- | :----- | :------------------------------------------------------- |
| age      | 年龄                          | string | 对象的年龄，数字表示                                     |
| sex      | 性别                          | string | 对象的性别，female和male                                 |
| cp       | 胸部疼痛类型                  | string | 痛感由重到无typical、atypical、non-anginal、asymptomatic |
| trestbps | 血压                          | string | 血压数值                                                 |
| chol     | 胆固醇                        | string | 胆固醇数值                                               |
| fbs      | 空腹血糖                      | string | 血糖含量大于120mg/dl为true，否则为false                  |
| restecg  | 心电图结果                    | string | 是否有T波，由轻到重为norm、hyp                           |
| thalach  | 最大心跳数                    | string | 最大心跳数                                               |
| exang    | 运动时是否心绞痛              | string | 是否有心绞痛，true为是，false为否                        |
| oldpeak  | 运动相对于休息的ST depression | string | st段压数值                                               |
| slop     | 心电图ST segment的倾斜度      | string | ST segment的slope，程度分为down、flat、up                |
| ca       | 透视检查看到的血管数          | string | 透视检查看到的血管数                                     |
| thal     | 缺陷种类                      | string | 并发种类，由轻到重norm、fix、rev                         |
| status   | 是否患病                      | string | 是否患病，buff是健康、sick是患病                         |



#### 算法部分

该系统的算法部分是利用**Keras**搭建的神经网络：

网络结构如下：

![image-20210129213206274](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190338.png)

我们采用了3层的全连接层，分别是算子size:`<7, 128>`, ` <128, 64>` ，`<64, 2>`。为了防止过拟合现象，我还加入了`Dropout层`，减少过拟合线性。你可以根据自己的需求更改网络，不影响后面程序的运行，修改网络只需要修改这里：

![image-20210129213554196](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190353.png)



#### 图形界面部分

该系统的实现是通过**PyQt5**来实现的，首先是利用的Qt Designer设计一个大概的页面：

![image-20210125212744847](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190405.png)

Qt Designer生成的`ui`文件利用下面的命令转化为`python`文件：

```
pyuic5 -o GUI.py disease.ui
```

然后就在这个文件夹下面生成了一个`GUI.py`

![image-20210125213403833](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190421.png)

接下来就是在预测脚本中(`Model_Predict.py`)导入GUI模块：

![image-20210125213644793](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190429.png)

在预测文件中测试一下，看看是不是能够正常的运行，下面是显示的GUI的Code:

```python
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
import tensorflow as tf 
from tensorflow import keras
from keras import layers
import pandas as pd 
import numpy as np
import sys 
import GUI



if __name__ == "__main__":
    # model = keras.models.load_model('E:\Github代码管理\Keras_tf2\心脏病预测\Z_model.h5')  #选取自己的.h模型名称
    # model.summary()
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = GUI.Ui_MainWindow()
    ui.setupUi(window)
    window.show()

    sys.exit(app.exec_())

```

运行效果如下：

![image-20210125213918537](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190438.png)



上面只是演示了**qt designer**设计的GUI的展示过程。下面这个是本系统的界面：

![image-20210129214210257](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190450.png)

**效果：**

输入以下指标，检测为患病。

![image-20210129214931185](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190459.png)



![image-20210129214829265](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190513.png)



检测为未患病：

![image-20210129215126738](https://cdn.jsdelivr.net/gh/Miller-em/IMAGS/img/20210722190525.png)

#### 让系统run起来

上面的环境如果安装好了，就可以成功的运行该系统了，直接在当前的目录下打开cmd，输入：`python Model_Predict.py`就好了。

如果想发给别人，然而别人没有安装环境，那么就直接打开**dist文件夹下的.exe文件**， 然而因为该文件是由python转化为的，所以运行速度较慢，打开也比较慢。



#### 结语

由于本系统的数据集数量较少，拟合程度有限，无法达到很好的数据预测效果，预测时可能会出现一些误判。**所以该系统的检测结果只能作为初步的参考，实际结果还是得医院检查为准。**