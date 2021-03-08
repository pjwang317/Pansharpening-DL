# PSGan

Environment:
tensorflow 1.10
gdal


pip install GDAL-2.3.3-cp35-cp35m-win_amd64.whl


step1 : python file_rename.py rename all the MS and PAN files, and put them in the same data file
step2 : python gen_mul.py and gen_pan.py both for test_sim and test_real
step3 : python tfrecord.py to generate the 	test_sim_WV2.tfrecord/test_sim_WV3.tfrecord/test_real_WV2.tfrecord/test_resl_WV3.tfrecord
step4 : python psgan_test.py --mode test --checkpoint C:/Users/USER-1/Documents/Peijuan/article/fromHocam/psgan/PSGan-master/data/psgan/output2/ --output_dir C:/Users/USER-1/Documents/Peijuan/article/fromHocam/psgan/PSGan-master/data/psgan/WV2/output2/test_sim/