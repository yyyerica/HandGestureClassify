from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "./storePic/5.jpg"
path2 = "./storePic/7.jpg"
path3 = "./storePic/102304.jpg"
path4 = "./storePic/12034.jpg"
path5 = "./storePic/ok.jpg"
path6 = "./storePic/2.jpg"

dict = {0:'5',1:'7',2:'12034',3:'ok',4:'102304',5:'2'}

w=100
h=100
c=3


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data6 = read_one_image(path6)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    saver = tf.train.import_meta_graph('./classify/modelSave/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./classify/modelSave/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第",i+1,"个手势预测:"+dict[output[i]])
