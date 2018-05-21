# HandGestureClassify
基于tensorflow的手势识别和分类

博文地址：https://blog.csdn.net/yyyerica/article/details/80151473

原手部检测代码源自： https://github.com/timctho/convolutional-pose-machines-tensorflow 

分类代码参考： https://blog.csdn.net/Enchanted_ZhouH/article/details/74116823


使用说明： 

1.手势识别 

运行 run_demo_hand_with_tracker.py 进行实时手势识别 

修改 config.py 中的 DEMO_TYPE 可更改输出的图像类型 

将 run_demo_hand_with_tracker.py 中的 cv2.imwrite('./storePic/11'+str(i)+'.jpg', local_img.astype(np.uint8),[int(cv2.IMWRITE_JPEG_QUALITY), 90]) 语句解除注释可以保存图片到项目目录下，可以自行修改存储目录


2.手势图像分类 

classmain.py 代码用于训练分类 

用于训练的手势数据集存于 classify -- handGesturePic 中，需要自行运行run_demo_hand_with_tracker.py保存图片作为训练集

训练好的模型存在 classify--modelSave 中 

调用 useClassifyModel.py 进行分类结果验证

其中，手部检测的模型请到https://github.com/timctho/convolutional-pose-machines-tensorflow 下载（tf版本）

手势图像分类的模型参数请自行训练classmain.py保存
