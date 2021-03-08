import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.io as sio
from downgraded import downgraded





def PNN_test(I_MS_LR,I_PAN,weightb,mode):
           
           #### parameters
           ratio=4
           L=12
           mav_value=2**(np.float32(L))

           weights = {
                      'w1': tf.Variable(weightb['W1'], name='w1'),
                      'w2': tf.Variable(weightb['W2'], name='w2'),
                      'w3': tf.Variable(weightb['W3'], name='w3'),
                      'w4': tf.Variable(weightb['W4'], name='w4'),
                      'w5': tf.Variable(weightb['W5'], name='w5'),
                      'w6': tf.Variable(weightb['W6'], name='w6'),
                      'w7': tf.Variable(weightb['W7'], name='w7'),
                      'w8': tf.Variable(weightb['W8'], name='w8'),
                      'w9': tf.Variable(weightb['W9'], name='w9'),
                      'w10': tf.Variable(weightb['W10'], name='w10'),
                      'w11': tf.Variable(weightb['W11'], name='w11'),
                      'w12': tf.Variable(weightb['W12'], name='w12')
                     }

           
           biases = {
                      'b1': tf.Variable(weightb['b1'].reshape(64), name='b1'),
                      'b2': tf.Variable(weightb['b2'].reshape(32), name='b2'),
                      'b3': tf.Variable(weightb['b3'].reshape(4), name='b3'),
                      'b4': tf.Variable(weightb['b4'].reshape(60), name='b4'),
                      'b5': tf.Variable(weightb['b5'].reshape(20), name='b5'),
                      'b6': tf.Variable(weightb['b6'].reshape(20), name='b6'),
                      'b7': tf.Variable(weightb['b7'].reshape(20), name='b7'),
                      'b8': tf.Variable(weightb['b8'].reshape(30), name='b8'),
                      'b9': tf.Variable(weightb['b9'].reshape(10), name='b9'),
                      'b10': tf.Variable(weightb['b10'].reshape(10), name='b10'),
                      'b11': tf.Variable(weightb['b11'].reshape(10), name='b11'),
                      'b12': tf.Variable(weightb['b12'].reshape(4), name='b12')                      
                    }
           ### choose the mode
           if mode != 'full':
                      print('PNN_TEST_I_MS_LR',I_MS_LR.shape)
                      I_MS_LR,I_PAN = downgraded(I_MS_LR,I_PAN,ratio)
                      
           
           ### input preparation
           
           print('2:',I_MS_LR.shape)
##           NDxI_LR = []

##           NDxI_LR = np.stack(((I_MS_LR[3,:,:]-I_MS_LR[2,:,:])/(I_MS_LR[3,:,:]+I_MS_LR[2,:,:]),
##                               (I_MS_LR[1,:,:]-I_MS_LR[3,:,:])/(I_MS_LR[1,:,:]+I_MS_LR[3,:,:])),axis=0)

                      
           I_MS_1=scipy.ndimage.interpolation.zoom(I_MS_LR[0,:,:],4,order=3,prefilter=False)
           I_MS_2=scipy.ndimage.interpolation.zoom(I_MS_LR[1,:,:],4,order=3,prefilter=False)
           I_MS_3=scipy.ndimage.interpolation.zoom(I_MS_LR[2,:,:],4,order=3,prefilter=False)
           I_MS_4=scipy.ndimage.interpolation.zoom(I_MS_LR[3,:,:],4,order=3,prefilter=False)

##           NDxI_LR_1 =scipy.ndimage.interpolation.zoom(NDxI_LR[0,:,:],4,order=3,prefilter=False)
##           NDxI_LR_2 =scipy.ndimage.interpolation.zoom(NDxI_LR[1,:,:],4,order=3,prefilter=False)

           I_MS_1 = np.expand_dims(I_MS_1,axis=0)
           I_MS_2 = np.expand_dims(I_MS_2,axis=0)
           I_MS_3 = np.expand_dims(I_MS_3,axis=0)
           I_MS_4 = np.expand_dims(I_MS_4,axis=0)
           
##           NDxI_LR_1 = np.expand_dims(NDxI_LR_1,axis=0)
##           NDxI_LR_2 = np.expand_dims(NDxI_LR_2,axis=0)

           I_in = np.stack((I_MS_1,I_MS_2,I_MS_3,I_MS_4,I_PAN),axis=0).astype('single')/mav_value
           I_in=np.squeeze(I_in)
           
##           I_in=np.vstack((I_in,NDxI_LR_1,NDxI_LR_2)).astype('single')

           I_in = np.pad(I_in, ((0,0),(8,8),(8,8)),mode='edge')
           
           I_in=I_in.transpose(1,2,0)
           

           I_in=tf.reshape(I_in,[-1,I_in.shape[0],I_in.shape[1],5])




           
           
           

           with tf.Session() as sess:
                      conv1=tf.nn.relu(tf.nn.conv2d(
                                 I_in,weights['w1'],strides=[1,1,1,1],
                                 padding='VALID')+biases['b1'])
                      conv2=tf.nn.relu(tf.nn.conv2d(
                                 conv1,weights['w2'],strides=[1,1,1,1],
                                 padding='VALID')+biases['b2'])
                      conv3=tf.nn.conv2d(
                                 conv2,weights['w3'],strides=[1,1,1,1],
                                 padding='VALID')+biases['b3']
                      output_cnn_shallow = conv3
                      

                      conv4=tf.nn.relu(tf.nn.conv2d(
                                 I_in,weights['w4'],strides=[1,1,1,1],
                                 padding='VALID')+biases['b4'])
		   
		   
                      conv5_1=tf.nn.relu(tf.nn.conv2d(
                                 conv4,weights['w5'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b5'])
                      conv5_2=tf.nn.relu(tf.nn.conv2d(
                                 conv4,weights['w6'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b6'])
                      conv5_3=tf.nn.relu(tf.nn.conv2d(
                                 conv4,weights['w7'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b7'])
                      conv5 = tf.concat([conv5_1,conv5_2,conv5_3],axis=3)+conv4
                      
		   
                      conv6 = tf.nn.relu(tf.nn.conv2d(
                                 conv5,weights['w8'],
                                 strides=[1,1,1,1],
                                 padding='VALID')+biases['b8'])
		   
                      conv7_1 =tf.nn.relu(tf.nn.conv2d(
                                 conv6,weights['w9'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b9'])
                      conv7_2 =tf.nn.relu(tf.nn.conv2d(
                                 conv6,weights['w10'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b10'])
                      conv7_3 =tf.nn.relu(tf.nn.conv2d(
                                 conv6,weights['w11'],
                                 strides=[1,1,1,1],
                                 padding='SAME')+biases['b11'])
		   
                      conv7 = tf.concat([conv7_1,conv7_2,conv7_3],axis=3)+ conv6
                      
		   
                      conv8 = tf.nn.conv2d(
                                 conv7,weights['w12'],strides=[1,1,1,1],
                                 padding='VALID')+biases['b12']

                      output = conv8 + output_cnn_shallow


                      I_out = output * mav_value

                      
                      init=tf.global_variables_initializer()
                      sess.run(init)
                      I_out = I_out.eval()

                      

                      return I_out


                      




