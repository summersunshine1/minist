import tensorflow as tf
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
train_data_path = pardir+'/data/train.csv'
imgsize = 28

epoch = 20
batch_size = 100

def split_train_and_validate(x_train,y_target):
    train_x,test_x,y_train,y_test = cross_validation.train_test_split(x_train,y_target,test_size=0.25)
    return train_x,test_x,y_train,y_test

def encoder(arr,bins):
    ohe = OneHotEncoder(sparse=False,n_values = bins)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr) 

def readData():
    data = pd.read_csv(train_data_path,encoding = 'utf-8')
    labels = data['label']
    columes = list(data.columns.values)
    pic_data = data[columes[1:]]
    length = len(pic_data)
    label_arr = []
    pic_arr = []
    for i in range(length):
        label = labels[i]
        pic = list(pic_data.iloc[i])
        label_arr.append([label])
        pic_arr.append(pic)
    pic_arr = np.array(pic_arr)    
    return label_arr,pic_arr
    
def weight_variable(shape):
    w = tf.Variable(tf.zeros(shape))
    # w = tf.truncated_normal(shape, stddev = 0.1)
    return w
    
def bias_variable(shape):
    b = tf.Variable(tf.zeros(shape))
    # b = tf.Variable(tf.constant(0.1,shape = shape))
    return b

def network():
    x = tf.placeholder(tf.float32,[None,imgsize*imgsize])
    y_ = tf.placeholder(tf.float32,[None, 10])
    w = weight_variable([imgsize*imgsize, 10])
    b = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(x,w)+b)
    # loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    y_input,x_input= readData()
    coder = encoder(y_input,10)
    train_x,test_x,y_train,y_test = split_train_and_validate(x_input,coder)
    length = len(train_x)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    for i in range(epoch):
        j = 0
        while(j<length):
            batch_x=train_x[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            # print(batch_x[0])
            # print(batch_y[0])
            _,loss_= sess.run([train_step,loss],feed_dict={x:batch_x,y_:batch_y})
            j += batch_size
        print("epoch "+ str(i)+" loss: "+str(loss_))
        acc = sess.run(accuracy,feed_dict={x:test_x,y_:y_test})
        print("epoch "+ str(i) + ": "+str(acc))
        
    sess.close()
    
if __name__=="__main__":
    network()
    
    