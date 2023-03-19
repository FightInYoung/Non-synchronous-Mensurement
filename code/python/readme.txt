The provided network can be used to estimate the covariance matrix of NSM. The NSM utilizes a 56-channel spiral array with the shifting interval of 0.9m.

56mic_3_train.py£ºtraining the model
56mic_3_test.py: testing the model
Unet_structure.py: providing the net block for '56mic_3_train'

save_net_train: saving the trained network
save_fig: saving the training MUSIC results
mat: using the trained net to generate the covariance matrix

running env:
	python 3.6.13
	tensorflow 1.14.0
	tensorflow-gpu 1.14.0
	numpy 1.19.5
	matplotlib 3.1.1
	h5py 2.7.0
	cudnn 7.6.5
	cuda 10.0.130
