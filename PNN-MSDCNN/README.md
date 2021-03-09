# PNN-MSDCNN

The references are from: 

1. Masi, G.; Cozzolino, D.; Verdoliva, L.; Scarpa, G. Pansharpening by Convolutional Neural Networks. Remote Sens. 2016, 8, 594. https://doi.org/10.3390/rs8070594
2. Q. Yuan, Y. Wei, X. Meng, H. Shen and L. Zhang, "A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 11, no. 3, pp. 978-989, March 2018, doi: 10.1109/JSTARS.2018.2794888.



Environment:

tensorflow 1.10
scipy
matplotlib
skimage


For training:

cd ./training

PNN: PNN_train.py
PNN_noindices PNN _noindices_train.py
MSDCNN: MSDCNN_train.py
MSDCNN_noindices: MSDCNN _noindices_train.py
For training, run the above codes separately


For testing:

cd ./test

For full resolution: 
run PAN_testing_full_real_data.py

For resuced resolution:
run PAN_testing_reduced_visualization.py

For different models, we also need to change the test model from PNN_test.py/PNN_test_no_indices.py/PNN_test_resnet.py/PNN_test_resnet_no_indices
