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
                      'w3': tf.Variable(weightb['W3'], name='w3')
                     }

           
           biases = {
                      'b1': tf.Variable(weightb['b1'].reshape(64), name='b1'),
                      'b2': tf.Variable(weightb['b2'].reshape(32), name='b2'),
                      'b3': tf.Variable(weightb['b3'].reshape(4), name='b3')
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
           #I_in = np.expand_dims(I_in,axis=0)
##           print('I_in:',I_in.shape)


           
           I_in=I_in.transpose(1,2,0)
##           print('I_in222:',I_in.shape)
           

           I_in=tf.reshape(I_in,[-1,I_in.shape[0],I_in.shape[1],5])
##           print('I_in333:',I_in.shape)




           
           
           

           with tf.Session() as sess:
                      conv1=tf.nn.relu(tf.nn.conv2d(I_in,weights['w1'],strides=[1,1,1,1],padding='VALID')+biases['b1'])
                      conv2=tf.nn.relu(tf.nn.conv2d(conv1,weights['w2'],strides=[1,1,1,1],padding='VALID')+biases['b2'])
                      conv3=tf.nn.conv2d(conv2,weights['w3'],strides=[1,1,1,1],padding='VALID')+biases['b3']


                      I_out = conv3 * mav_value

                      
                      init=tf.global_variables_initializer()
                      sess.run(init)
                      I_out = I_out.eval()

                      

                      return I_out


                      




