import os,sys
import tensorflow as tf
import gdal

#  change the data_dir, also the number to your own
data_dir='C:/Users/USER-1/Documents/Peijuan/article/fromHocam/psgan/PSGan-master/data/psgan/data/WV3/test_real/'
testlist=['%s/datalist/%d'%(data_dir,number) for number in range(1,66)]
#testlist=['%s/test_sim/%d'%(data_dir,number) for number in range(1,201)]
#trainfiles=['%s/train/%d'%(data_dir,number) for number in range(16000)]
output_dir="C:/Users/USER-1/Documents/Peijuan/article/fromHocam/psgan/PSGan-master/data/psgan/"

print(output_dir)
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(inputfiles, name):
    num_examples=len(inputfiles)
    print(num_examples)
    filename=os.path.join(output_dir,name+'.tfrecords')
    print ('Writing', filename)

    writer=tf.python_io.TFRecordWriter(filename)
    for (file,i) in zip(inputfiles, range(num_examples)):
        print(file,i)
        
        img_name = '%s_%d' % (name, i)
       
        mul_filename = '%s_mul.tif' % file
        blur_filename = '%s_lr_u.tif' % file
        pan_filename = '%s_pan.tif' % file
        
        
        #  modification here, change the string to bytes *4
        im_mul_raw = bytes(gdal.Open(mul_filename).ReadAsArray().transpose(1, 2, 0).tostring())
        im_blur_raw = bytes(gdal.Open(blur_filename).ReadAsArray().transpose(1, 2, 0).tostring())
        im_pan_raw = bytes(gdal.Open(pan_filename).ReadAsArray().reshape([128, 128, 1]).tostring())

        img_name = bytes(img_name,'utf-8')

        example = tf.train.Example(features=tf.train.Features(feature={
            'im_name': _bytes_feature(img_name),
            'im_mul_raw': _bytes_feature(im_mul_raw),
            'im_blur_raw':_bytes_feature(im_blur_raw),
            'im_pan_raw':_bytes_feature(im_pan_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def convert_to_test(inputfiles, name):
    num_examples=len(inputfiles)
    print(num_examples)
    filename=os.path.join(output_dir,name+'.tfrecords')
    print ('Writing', filename)

    writer=tf.python_io.TFRecordWriter(filename)
    for (file,i) in zip(inputfiles, range(num_examples)):
        
        img_name = '%s_%d' % (name, i)
       
        mul_filename = '%s_mul.tif' % file
        blur_filename = '%s_lr_u.tif' % file
        pan_filename = '%s_pan.tif' % file
        
        
        #  modification here, change the string to bytes *4
        im_mul_raw = bytes(gdal.Open(mul_filename).ReadAsArray().transpose(1, 2, 0).tostring())
        im_blur_raw = bytes(gdal.Open(blur_filename).ReadAsArray().transpose(1, 2, 0).tostring())
        im_pan_raw = bytes(gdal.Open(pan_filename).ReadAsArray().reshape([800, 800, 1]).tostring())

        img_name = bytes(img_name,'utf-8')

        example = tf.train.Example(features=tf.train.Features(feature={
            'im_name': _bytes_feature(img_name),
            'im_mul_raw': _bytes_feature(im_mul_raw),
            'im_blur_raw':_bytes_feature(im_blur_raw),
            'im_pan_raw':_bytes_feature(im_pan_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
#convert_to(trainfiles,'train')
#convert_to_test(testlist,'test_sim')

print('start to converting...')
convert_to_test(testlist,'test_real_WV3')



