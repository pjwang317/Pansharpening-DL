import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import scipy.misc as smc
import scipy.ndimage
import matplotlib.pyplot as plt
import random


import os
from skimage.external import tifffile
import datetime

from read_image import read_image_ms,read_image_pan

### download data


PAN_train_data,PAN_test_data=read_image_pan('../training/PAN',1)
MS_train_data,MS_test_data=read_image_ms('../training/MS',1)

weightb=loadmat('../networks/params/net64.mat')


#### parameters
ratio=4
L=12
mav_value=2**(np.float32(L))




####  getdata for training()

def getdata(data1,data2):
    size = len(data1)
    MS_ref=np.empty((size,60,60,4))
    input_data=np.empty((size,60,60,7))
           

           
    for i in range(0,size):
        X=data1[i,:,:]
        Y=data2[i,:,:,:]
        ###  normalization                       

        X_=X.astype('float32')/mav_value
        Y_=Y.astype('float32')/mav_value

        ###  downsampling
        X_1=scipy.ndimage.interpolation.zoom(X_,(1./4),order=3,prefilter=False)
                     
        Y_1=scipy.ndimage.interpolation.zoom(Y_[:,:,0],(1./4),order=3,prefilter=False)
                      
        Y_2=scipy.ndimage.interpolation.zoom(Y_[:,:,1],(1./4),order=3,prefilter=False)
        Y_3=scipy.ndimage.interpolation.zoom(Y_[:,:,2],(1./4),order=3,prefilter=False)
        Y_4=scipy.ndimage.interpolation.zoom(Y_[:,:,3],(1./4),order=3,prefilter=False)

        ### compute the indices
        NDVI=(Y_4-Y_3)/(Y_4+Y_3)
        NDWI=-((Y_4-Y_2)/(Y_2+Y_4))

                     
                      
        ###  upsampling
        Y_11=scipy.ndimage.interpolation.zoom(Y_1,4,order=3,prefilter=False)
        Y_22=scipy.ndimage.interpolation.zoom(Y_2,4,order=3,prefilter=False)
        Y_33=scipy.ndimage.interpolation.zoom(Y_3,4,order=3,prefilter=False)
        Y_44=scipy.ndimage.interpolation.zoom(Y_4,4,order=3,prefilter=False)

        NDVI_=scipy.ndimage.interpolation.zoom(NDVI,4,order=3,prefilter=False)
        NDWI_=scipy.ndimage.interpolation.zoom(NDWI,4,order=3,prefilter=False)

                      
                      

        MS_ref[i,:,:,:]=Y_
        band=np.array([Y_11.T,Y_22.T,Y_33.T,Y_44.T,X_1.T,NDVI_.T,NDWI_.T])
        input_data[i,:,:,:]=band.T.astype('single')

    return input_data,MS_ref

#### getdata for the test 

     


### data preprocessing

BATCH_SIZE = 8
LR = 0.001


with tf.name_scope('inputs'):
    tf_x=tf.placeholder(tf.float32, [BATCH_SIZE,76,76,7],name='PAN_input')
    I_X=tf.reshape(tf_x,[-1,76,76,7])

    tf_y=tf.placeholder(tf.float32,[BATCH_SIZE,60,60,4],name='MS_input')





####  CNN layers

with tf.name_scope('weights'):

    weights = {
            'w1': tf.Variable(weightb['w1'], name='w1'),
            'w2': tf.Variable(weightb['w2'], name='w2'),
            'w3': tf.Variable(weightb['w3'], name='w3'),
		    'w4': tf.Variable(tf.random_normal([11,11,7,60],stddev=1e-3), name='w4'),
		    'w5': tf.Variable(tf.random_normal([3,3,60,20],stddev=1e-3), name='w5'),
		    'w6': tf.Variable(tf.random_normal([5,5,60,20],stddev=1e-3), name='w6'),
		    'w7': tf.Variable(tf.random_normal([7,7,60,20],stddev=1e-3), name='w7'),
		    'w8': tf.Variable(tf.random_normal([3,3,60,30],stddev=1e-3), name='w8'),
		    'w9': tf.Variable(tf.random_normal([3,3,30,10],stddev=1e-3), name='w9'),
		    'w10': tf.Variable(tf.random_normal([5,5,30,10],stddev=1e-3), name='w10'),
		    'w11': tf.Variable(tf.random_normal([7,7,30,10],stddev=1e-3), name='w11'),
		    'w12': tf.Variable(tf.random_normal([5,5,30,4],stddev=1e-3), name='w12')
            }
           
with tf.name_scope('biases'):
           
    biases = {
            'b1': tf.Variable(weightb['b1'].reshape(64), name='b1'),
            'b2': tf.Variable(weightb['b2'].reshape(32), name='b2'),
            'b3': tf.Variable(weightb['b3'].reshape(4), name='b3'),
	        'b4': tf.Variable(tf.zeros([60]), name='b4'),
            'b5': tf.Variable(tf.zeros([20]), name='b5'),
            'b6': tf.Variable(tf.zeros([20]), name='b6'),
            'b7': tf.Variable(tf.zeros([20]), name='b7'),
            'b8': tf.Variable(tf.zeros([30]), name='b8'),
            'b9': tf.Variable(tf.zeros([10]), name='b9'),
            'b10': tf.Variable(tf.zeros([10]), name='b10'),
            'b11': tf.Variable(tf.zeros([10]), name='b11'),
            'b12': tf.Variable(tf.zeros([4]), name='b12')
            }
           
    


###  conv layer one
with tf.name_scope('conv1'):

    conv1=tf.nn.relu(tf.nn.conv2d(I_X,weights['w1'],strides=[1,1,1,1],padding='VALID')+biases['b1'],
                            name='conv1')
           
    tf.summary.histogram('conv1+/weights', weights['w1'])
    tf.summary.histogram('conv1+/biases', biases['b1'])

###  batch normalization layer
###  conv1_=tf.layers.batch_normalization(conv1,momentum=0.4)

###  conv layer two
with tf.name_scope('conv2'):
    conv2=tf.nn.relu(tf.nn.conv2d(conv1,weights['w2'],strides=[1,1,1,1],padding='VALID')+biases['b2'],
                            name='conv1')
    tf.summary.histogram('conv2+/weights', weights['w2'])
    tf.summary.histogram('conv2+/biases', biases['b2'])
           
###  conv layer three
with tf.name_scope('conv3'):
    conv3=tf.nn.conv2d(conv2,weights['w3'],strides=[1,1,1,1],padding='VALID')+biases['b3']
    tf.summary.histogram('conv3+/weights', weights['w3'])
    tf.summary.histogram('conv3+/biases', biases['b3'])
    output_cnn_shallow = conv3
    tf.summary.histogram('conv3+/outputs', output_cnn_shallow)
           

		   
#### resnet laybers

    conv4=tf.nn.relu(tf.nn.conv2d(I_X,weights['w4'],strides=[1,1,1,1],padding='VALID')+biases['b4'],name='conv4')
		   
		   
    conv5_1=tf.nn.relu(tf.nn.conv2d(conv4,weights['w5'],strides=[1,1,1,1], padding='SAME')+biases['b5'],name='conv5_1')
    conv5_2=tf.nn.relu(tf.nn.conv2d(conv4,weights['w6'],strides=[1,1,1,1], padding='SAME')+biases['b6'],name='conv5_2')
    conv5_3=tf.nn.relu(tf.nn.conv2d(conv4,weights['w7'],strides=[1,1,1,1], padding='SAME')+biases['b7'],name='conv5_3')
    conv5 = tf.concat([conv5_1,conv5_2,conv5_3],axis=3)+conv4
           ####conv5 = tf.layers.batch_normalization(conv5_4,momentum=0.4)+conv4
		   
    conv6 = tf.nn.relu(tf.nn.conv2d(conv5,weights['w8'],strides=[1,1,1,1],padding='VALID')+biases['b8'],name='conv6')
		   
    conv7_1 =tf.nn.relu(tf.nn.conv2d(conv6,weights['w9'],strides=[1,1,1,1],padding='SAME')+biases['b9'],name='conv7_1')
    conv7_2 =tf.nn.relu(tf.nn.conv2d(conv6,weights['w10'],strides=[1,1,1,1],padding='SAME')+biases['b10'],name='conv7_2')
    conv7_3 =tf.nn.relu(tf.nn.conv2d(conv6,weights['w11'],strides=[1,1,1,1],padding='SAME')+biases['b11'],name='conv7_3')
		   
    conv7 = tf.concat([conv7_1,conv7_2,conv7_3],axis=3)+ conv6
           ####conv7 = tf.layers.batch_normalization(conv7_4,momentum=0.4)+ conv6
		   
    conv8 = tf.nn.conv2d(conv7,weights['w12'],strides=[1,1,1,1],padding='VALID')+biases['b12']

    output = conv8 + output_cnn_shallow

           
		   
		   
### compute the loss function & optimizer
###loss=tf.losses.mean_squared_error(tf_y,output,weights=1.0,scope=None)

with tf.name_scope('loss'):
##  loss= tf.reduce_mean(tf.square(tf_y-output))
           ##         L1 loss 
    loss =tf.reduce_mean( tf.abs(tf_y - output))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(LR)
    train_op=optimizer.minimize(loss=loss)




###  get all the parameters of the training
params=tf.trainable_variables()

print("trainable variables-----")
for idx,v in enumerate(params):
    print("param{:3}:{:15}",idx,v.name)

                     
### create the saver
saver = tf.train.Saver()


### create session
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
           
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:/Tools/Python/logs", tf.get_default_graph())
           
           
    for step in range(300):
        starttime = datetime.datetime.now()

        batch_idx= len(PAN_train_data)//BATCH_SIZE
                      
        for idx in range(0,batch_idx):   
            pan_train = PAN_train_data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            ms_train = MS_train_data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            input_data,MS_ref = getdata(pan_train,ms_train)
            input_data = np.pad(input_data, ((0,0),(8,8),(8,8),(0,0)),mode='edge')
                                 
               
            _, loss_ = sess.run([train_op, loss], {tf_x: input_data, tf_y: MS_ref})
                                 
        endtime = datetime.datetime.now()
        time = (endtime-starttime).seconds
                 
        if step % 1 == 0:
            print('Step:',step,'| train loss: %.6f' % loss_,'| %.2f seconds/step' % time)

            result = sess.run(merged,feed_dict={tf_x: input_data, tf_y: MS_ref})
            writer.add_summary(result, step)

        if (step+1) % 50 ==0:
            save_path=saver.save(sess,"C:/Tools/Python/Python35/RemoteSensingEx/myCodes/training/newdata/parameter/%s" % step +"_MSDCNN_indices_1203.ckpt")
           
            W1=sess.run(params[0])
           
                                 #gamma=sess.run(params[2])
                                 #beta=sess.run(params[3])
            W2=sess.run(params[1])
            W3=sess.run(params[2])
            W4=sess.run(params[3])
            W5=sess.run(params[4])
            W6=sess.run(params[5])
            W7=sess.run(params[6])
            W8=sess.run(params[7])
            W9=sess.run(params[8])
            W10=sess.run(params[9])
            W11=sess.run(params[10])
            W12=sess.run(params[11])


            b1=sess.run(params[12])
            b2=sess.run(params[13])
            b3=sess.run(params[14])
            b4=sess.run(params[15])
            b5=sess.run(params[16])
            b6=sess.run(params[17])
            b7=sess.run(params[18])
            b8=sess.run(params[19])
            b9=sess.run(params[20])
            b10=sess.run(params[21])
            b11=sess.run(params[22])
            b12=sess.run(params[23])

            savemat('%s' % (step+1) + '_resnet_L1_1203.mat',{'W1':W1,'W2':W2,'W3':W3,'W4':W4,'W5':W5,'W6':W6,
                        'W7':W7,'W8':W8,'W9':W9,'W10':W10,'W11':W11,'W12':W12,
                        'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5,'b6':b6,
                        'b7':b7,'b8':b8,'b9':b9,'b10':b10,'b11':b11,'b12':b12})
           


