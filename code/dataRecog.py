import tensorflow as tf
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()
train_data_path = pardir+'/data/train.csv'
test_data_path = pardir + '/data/test.csv'
res_path = pardir + '/data/res.csv'
imgsize = 28

epoch = 20
batch_size = 100
hidden_size = 500
hidden_size2 = 300  

def write_res(arr):
    f = open(res_path,'w',encoding = 'utf-8')
    f.writelines(','.join(['"ImageId"','"Label"'])+'\n')
    for i in range(len(arr)):
        info = [str(i+1)]
        info.append(str(arr[i]))
        outline = ','.join(info)+'\n'
        f.writelines(outline)
    f.close()

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
    
def readTestData():
    data = pd.read_csv(test_data_path,encoding = 'utf-8')
    columes = list(data.columns.values)
    pic_data = data[columes]
    length = len(pic_data)
    label_arr = []
    pic_arr = []
    for i in range(length):
        pic = list(pic_data.iloc[i])
        pic_arr.append(pic)
    pic_arr = np.array(pic_arr)    
    return pic_arr
    
def weight_variable(shape):
    # w = tf.Variable(tf.zeros(shape))
    # w = tf.truncated_normal(shape, stddev = stddev_)
    w = tf.Variable(tf.random_normal(shape,seed=1))
    return w
    
def bias_variable(shape_):
    # b = tf.Variable(tf.zeros(shape))
    # b = tf.Variable(tf.constant(v,shape = shape))
    b = tf.Variable(tf.random_normal(shape = shape_,seed=1))
    return b

def network():
    x = tf.placeholder(tf.float32,[None,imgsize*imgsize],name = 'x')
    y_ = tf.placeholder(tf.float32,[None, 10],name = 'y')
    w = weight_variable(0,[imgsize*imgsize, 10])
    b = bias_variable([10])
    y = tf.matmul(x,w)+b
    y_res = tf.argmax(y,1,name = 'predict')
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
    saver = tf.train.Saver()
    for i in range(epoch):
        j = 0
        while(j<length):
            batch_x=train_x[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            # print(batch_x[0])
            # print(batch_y[0])
            a,_,loss_= sess.run([accuracy, train_step,loss],feed_dict={x:batch_x,y_:batch_y})
            j += batch_size

        print("epoch "+ str(i)+" loss: "+str(loss_))
        acc = sess.run(accuracy,feed_dict={x:test_x,y_:y_test})
        print("epoch "+ str(i) + ": "+str(acc))
        
    save_path = saver.save(sess, pardir+'/model/softmax.ckpt')       
    sess.close()
    
def MLP():
    x = tf.placeholder(tf.float32,[None,imgsize*imgsize],name = 'x')
    y_ = tf.placeholder(tf.float32,[None, 10],name = 'y')
    
    keep_prob = tf.placeholder(tf.float32,name = 'prob')
    w1 = weight_variable(0.1,[imgsize*imgsize, hidden_size])
    b1 = bias_variable(0.1,[hidden_size])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    hidden_dropout = tf.nn.dropout(y1, keep_prob)
    w2 = weight_variable(0.1,[hidden_size,hidden_size2])
    b2 = bias_variable(0.1,[hidden_size2])
    y2 = tf.nn.relu(tf.matmul(hidden_dropout,w2)+b2)
    hidden1_dropout = tf.nn.dropout(y2, keep_prob)
    
    w3 = weight_variable(0.1,[hidden_size2,10])
    b3 = bias_variable(0.1,[10])
    y3 = tf.matmul(hidden1_dropout,w3)+b3
    # y2_output = tf.nn.softmax(y2)
    y_res = tf.argmax(y3,1,name = 'predict')
    # loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y3))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(y_res,tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy', accuracy)
    y_input,x_input= readData()
    coder = encoder(y_input,10)
    train_x,test_x,y_train,y_test = split_train_and_validate(x_input,coder)
    length = len(train_x)
    
    merged = tf.summary.merge_all()
 
    sess = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter(pardir+'/log')
    
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    lastacc = 0
    count = 0
    for i in range(2):
        j = 0
        while(j<length):
            batch_x=train_x[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            # print(batch_x[0])
            # print(batch_y[0])
            summary,a,_,loss_,w_= sess.run([merged,accuracy,train_step,loss,w2],feed_dict={x:batch_x,y_:batch_y,keep_prob:0.75})
            j += batch_size
        train_writer.add_summary(summary,i)
        print("epoch "+ str(i)+" train_accuracy :"+str(a))
        print("epoch "+ str(i)+" loss: "+str(loss_))
        acc = sess.run(accuracy,feed_dict={x:test_x,y_:y_test,keep_prob:1})
        print("epoch "+ str(i) + ": "+str(acc))
        if acc<lastacc:
            count+=1
        else:
            count = 0
        lastacc = acc
        if count==3:
            break
    train_writer.close()
    save_path = saver.save(sess, pardir+'/model/nn.ckpt')       
    sess.close()
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
def bn(input, phase):
    out = tf.contrib.layers.batch_norm(input, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
    return out
    
def conv():
    x = tf.placeholder(tf.float32,[None,imgsize*imgsize],name = 'x')
    y_ = tf.placeholder(tf.float32,[None, 10],name = 'y')
    phase = tf.placeholder(tf.bool, name="phase")
    x_img = tf.reshape(x,[-1,imgsize,imgsize,1])
    w_conv1 = weight_variable([5,5,1,32]) 
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(bn(conv2d(x_img, w_conv1)+b_conv1, phase))
    h_pool1 = max_pool(h_conv1)
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    tf.get_variable_scope().reuse_variables() 
    h_conv2 = tf.nn.relu(bn(conv2d(h_pool1, w_conv2)+b_conv2, phase))
    h_pool2 = max_pool(h_conv2)
    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    h_fc1 = tf.nn.relu(bn(tf.matmul(h_pool2_flat, w_fc1)+b_fc1, phase))
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, 10])
    bias_fc2 = bias_variable([10])
    y_out = tf.matmul(h_fc1_dropout, w_fc2)+bias_fc2
    y_res = tf.argmax(y_out,1,name = 'predict')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)) 
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y_out,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(pardir+'/log')
    y_input,x_input= readData()
    coder = encoder(y_input,10)
    train_x,test_x,y_train,y_test = split_train_and_validate(x_input,coder)
    length = len(train_x)
    
    lastacc = 0
    count = 0
    for i in range(30):
        j = 0
        while(j<length):
            batch_x=train_x[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                summary,a,_,loss_= sess.run([merged,accuracy,train_step,loss],feed_dict={x:batch_x,y_:batch_y,keep_prob:0.75, phase:1})
            j += batch_size
        train_writer.add_summary(summary,i)
        print("epoch "+ str(i)+" train_accuracy :"+str(a))
        print("epoch "+ str(i)+" loss: "+str(loss_))
        acc = sess.run(accuracy,feed_dict={x:test_x,y_:y_test,keep_prob:1,phase:0})
        print("epoch "+ str(i) + ": "+str(acc))
        if acc<lastacc:
            count+=1
        else:
            count = 0
        lastacc = acc
        if count==3:
            break
    train_writer.close()
    save_path = saver.save(sess, pardir+'/model/cnn.ckpt')       
    sess.close()
   
def modle_predict():
    saver = tf.train.import_meta_graph(pardir+'/model/cnn.ckpt.meta')
    sess = tf.Session()
    saver.restore(sess,pardir+'/model/cnn.ckpt')
    graph = tf.get_default_graph()
    predict = graph.get_tensor_by_name("predict:0")
    x = graph.get_tensor_by_name("x:0")
    x_input = readTestData()
    prob = graph.get_tensor_by_name("prob:0")
    predict_ = sess.run(predict, feed_dict={x:x_input,prob:1})
    write_res(predict_)
    
if __name__=="__main__":
    conv()
    # modle_predict()
    