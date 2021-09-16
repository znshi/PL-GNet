import tensorflow as tf
from tensorflow.python.client import device_lib
#from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.slim as slim
from compute_mcc import *
#import scipy.io as sio
import math
import h5py
#from compute_mcc import compute_mcc,metrics,_fast_hist,label_accuracy_score
from hilbert import hilbertCurve
#from compute_IoU import compute_precision,bb_IoU
# sys.path.append('.')
import os,sys
from scipy import signal
import time
import skimage
import skimage.io, skimage.transform
from skimage.transform import resize
from skimage.util import view_as_windows
import scipy.misc
import scipy.io as sio
from skimage import img_as_uint
import matplotlib.pyplot as plt
# import pandas
import glob 
import datetime
from skimage.color import rgb2ycbcr 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pywt
# import cv2
import re
from PIL import Image
import os
import bisect
import pickle

SRM_Kernel = np.array([
    [[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
    [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
])
SRM_Kernel = np.vstack((SRM_Kernel, SRM_Kernel, SRM_Kernel)).reshape(3, 7, 5, 5).transpose(2,3,0,1)


tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
log_device_placement = True
# Parameters
lr = 0.00003
training_iters = 50000000
batch_size = 30
display_step = 10
nb_nontamp_img=16960
nb_tamp_img=68355
nbFilter=32


# LSTM network parameters
n_input = 240 # data input (img shape: 64x64)
n_steps = 64 # timesteps
nBlock=int(math.sqrt(n_steps))
n_hidden = 64# hidden layer num of features
nStride=int(math.sqrt(n_hidden))
# other parameters
imSize=256
# Network Parameters
n_classes = 2 # manipulated vs unmanipulated
mx=127.0

# tf Graph input
input_layer = tf.placeholder("float", [None, imSize,imSize,3])
y= tf.placeholder("float", [2,None, imSize,imSize])
freqFeat=tf.placeholder("float", [None, 248,248,3])
# freqFeat=tf.placeholder("float", [None, 64,240])
# freqFeat=tf.placeholder("float", [None, 256,256,3])
filter = tf.Variable(tf.random_normal([5,5,3,9]))
ratio=15.0 #tf.placeholder("float",[1])
#out_rnn=tf.placeholder("float", [None, 128,128,3])
# W1 = tf.get_variable('W1', [5,5,3, 10], tf.float32, xavier_initializer())
# b1 = tf.Variable(tf.random_normal([5,5,3]))

############################################################################
#total_layers = 25 #Specify how deep we want our network
units_between_stride = 2
upsample_factor=16
beta=.01
outSize=16
############################################################################
seq = np.linspace(0,63,64).astype(int)
order3 = hilbertCurve(3)
order3 = np.reshape(order3,(64))
print(seq)
print(order3)
hilbert_ind = np.lexsort((seq,order3))
actual_ind=np.lexsort((seq,hilbert_ind))

weights = {
    'out': tf.Variable(tf.random_normal([64,64,nbFilter]))
}
biases = {
    'out': tf.Variable(tf.random_normal([nbFilter]))
}

atrous_fil = tf.Variable(tf.random_normal([3, 3, 256,256]),name = 'atrous_fil')
atrous_fil_1 =tf.Variable(tf.random_normal([3, 3, 256,256]),name ='atrous_fil_1')
atrous_fil_2 =tf.Variable(tf.random_normal([3, 3, 256,256]),name ='atrous_fil_2')


atrous_fil1 = tf.Variable(tf.random_normal([3, 3, 256,256]),name ='atrous_fil1')
atrous_fil2 = tf.Variable(tf.random_normal([3, 3, 256,256]),name ='atrous_fil2')
atrous_fil3 = tf.Variable(tf.random_normal([3, 3, 256,256]),name ='atrous_fil3')




with tf.device('/gpu:1'):

    def conv_mask_gt(z): 
        # Get ones for each class instead of a number -- we need that
        # for cross-entropy loss later on. Sometimes the groundtruth
        # masks have values other than 1 and 0. 
#         class_labels_tensor = (z==1)
#         background_labels_tensor = (z==0)
        
        class_labels_tensor = (z==1)
        background_labels_tensor = (z==0)
        # Convert the boolean values into floats -- so that
        # computations in cross-entropy loss is correct
        bit_mask_class = np.float32(class_labels_tensor)
        bit_mask_background = np.float32(background_labels_tensor)
        combined_mask=[]
        combined_mask.append(bit_mask_background)
        combined_mask.append(bit_mask_class)
        #combined_mask = tf.concat(concat_dim=3, values=[bit_mask_background,bit_mask_class])		

        # Lets reshape our input so that it becomes suitable for 
        # tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
        #flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))	
        return combined_mask#flat_labels

    def get_kernel_size(factor):
        #Find the kernel size given the desired factor of upsampling.
        return 2 * factor - factor % 2

    def upsample_filt(size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(factor, number_of_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """    
        filter_size = get_kernel_size(factor)

        weights = np.zeros((filter_size,filter_size,number_of_classes,number_of_classes), dtype=np.float32)    
        upsample_kernel = upsample_filt(filter_size)    
        for i in range(number_of_classes):        
            weights[:, :, i, i] = upsample_kernel    
        return weights


    def resUnit(input_layer,i,nbF):
        with tf.variable_scope("res_unit"+str(i)):
        #input_layer=tf.reshape(input_layer,[-1,64,64,3])
            part1 = slim.batch_norm(input_layer,activation_fn=None)
            part2 = tf.nn.relu(part1)
            part3 = slim.conv2d(part2,nbF,[3,3],activation_fn=None)
            part4 = slim.batch_norm(part3,activation_fn=None)
            part5 = tf.nn.relu(part4)
            part6 = slim.conv2d(part5,nbF,[3,3],activation_fn=None)	
            output = input_layer + part6
            return output

    #tf.reset_default_graph()

    def segNet(input_layer,bSize,freqFeat,weights,biases):
        
        # layer1: resblock, input size(256,256)
        layer1 = tf.nn.conv2d(input_layer, SRM_Kernel, strides=[1,1,1,1],padding ='SAME',name = 'SRM_out' )
        layer2 = tf.nn.conv2d(input_layer, filter, strides=[1,1,1,1],padding ='SAME',name = 'SRM_out1' )
        concat = tf.concat([layer1, layer2], axis=3, name='concat')
        print('concat:',concat.shape)
        
        Conv_1 = slim.conv2d(concat,nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(1))
        ReLu_1 = tf.nn.relu(Conv_1)
        Pool_1 = slim.max_pool2d(ReLu_1, [2, 2], scope='pool_'+str(1))
        print('Pool_1:',Pool_1.shape)
        
        # layer2: resblock, input size(128,128)   
        Conv_2 = slim.conv2d(Pool_1,2*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(2))
        ReLu_2 = tf.nn.relu(Conv_2)
        Conv_3 = slim.conv2d(ReLu_2,2*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(3))
        ReLu_3 = tf.nn.relu(Conv_3)
        Pool_2 = slim.max_pool2d(ReLu_3, [2, 2], scope='pool_'+str(2))
        print('Pool_2:',Pool_2.shape)
        
        # layer3: resblock, input size(64,64) 
        Conv_4 = slim.conv2d(Pool_2,4*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(4))
        ReLu_4 = tf.nn.relu(Conv_4)
        Conv_5 = slim.conv2d(ReLu_4,4*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(5))
        ReLu_5 = tf.nn.relu(Conv_5)
        Conv_6 = slim.conv2d(ReLu_5,4*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(6))
        ReLu_6 = tf.nn.relu(Conv_6)
        Pool_3 = slim.max_pool2d(ReLu_6, [2, 2], scope='pool_'+str(3))
        # layer4: resblock, input size(32,32) 
        print('Pool_3:',Pool_3.shape)
        
        Conv_7 = slim.conv2d(Pool_3,8*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(7))
        ReLu_7= tf.nn.relu(Conv_7)
        Conv_8 = slim.conv2d(ReLu_7,8*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(8))
        ReLu_8 = tf.nn.relu(Conv_8)
        Conv_9 = slim.conv2d(ReLu_8,8*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(9))
        ReLu_9 = tf.nn.relu(Conv_9)
        print('ReLu_9:',ReLu_9.shape)
    
        Conv_11_AC = tf.nn.atrous_conv2d(ReLu_9,atrous_fil,rate =2,padding = 'SAME',name = 'conv_'+str(11))
        BN_2 = slim.batch_norm(Conv_11_AC,activation_fn=None)
        Tan_2 = tf.nn.relu(BN_2)
        Conv_12_AC = tf.nn.atrous_conv2d(Tan_2,atrous_fil_1,rate =2,padding = 'SAME',name = 'conv_'+str(12))
        BN_3 = slim.batch_norm(Conv_12_AC,activation_fn=None)
        Tan_3 = tf.nn.relu(BN_3)
        Conv_13_AC = tf.nn.atrous_conv2d(Tan_3,atrous_fil_2,rate =2,padding = 'SAME',name = 'conv_'+str(13))
        BN_4 = slim.batch_norm(Conv_13_AC,activation_fn=None)
        Tan_4 = tf.nn.relu(BN_4)
        output = ReLu_9 + Tan_4
#         output = slim.max_pool2d(output, [2, 2], scope='pool_'+str(4))
        
#         layer55 = tf.nn.relu(output)
       
        layer6 = slim.conv2d(output,8*nbFilter,[1,1],normalizer_fn=slim.batch_norm,scope='conv_'+str(14))
        layer6 = tf.nn.relu(layer6)
       
                                      
        layer7 = tf.nn.atrous_conv2d(output,atrous_fil1,rate = 3,padding = 'SAME',name='conv_'+str(15))
        layer7 = tf.nn.relu(layer7)
        
                                      
        layer8 = tf.nn.atrous_conv2d(output,atrous_fil2,rate = 5,padding = 'SAME',name='conv_'+str(16))
        layer8 = tf.nn.relu(layer8)
        
                                      
        layer9 = tf.nn.atrous_conv2d(output,atrous_fil3,rate = 7,padding = 'SAME',name='conv_'+str(17))
        layer9 = tf.nn.relu(layer9)
       
        
        layer10 = layer6+layer7+layer8+layer9
        print('layer10:',layer10.shape)
        
        layer12 = tf.nn.relu(layer10)
        layer13 = slim.max_pool2d(layer12, [2, 2], scope='pool_'+str(4))
        print('layer13:',layer13.shape)
     
        # lstm network 
        layer55 = slim.conv2d(freqFeat,16,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(18))
        layer55 = tf.nn.relu(layer55)
        layer66 = slim.conv2d(layer55,1,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(19))
        layer66 = tf.nn.relu(layer66)
        print('layer6_shape:',layer66.shape)  # ( bs, 256, 256, 1)
        
        layer77=tf.transpose(layer66,[0,3,1,2])
        print('layer7_shape:',layer77.shape)  # ( bs, 256, 256, 1)
        
        y_list = tf.split(layer77,8,axis=3)
        print('y_list_shape2:',len(y_list))
        xy_list = [tf.split(x,8,axis =2) for x in y_list] ##
        print('xy_list_shape2:',len(xy_list))
        xy = [item for items in xy_list for item in items]
        print('xy_shape:',len(xy))
#         xy = torch.cat(xy,1)
        xy = tf.concat(xy,1)
        print('xy_shape2:',xy.shape)
#         patches = xy.view(-1,64,64)#
        patches = tf.reshape(xy,(-1,64,31*31))
        print('patches_shape2:',patches.shape)
        
        
#         patches=tf.transpose(freqFeat,[1,0,2])
#         patches=tf.gather(patches,hilbert_ind)
#         patches=tf.transpose(patches,[1,0,2])
#         print('patches:',patches.shape)
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        xCell=tf.unstack(patches, n_steps, 1)
        # 2 stacked layers
        stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),output_keep_prob=0.9) for _ in range(2)] )
        out, state = rnn.static_rnn(stacked_lstm_cell, xCell, dtype=tf.float32)
        # organizing the lstm output
        out=tf.gather(out,actual_ind)
        # convert to lstm output (64,batchSize,nbFilter)
        lstm_out=tf.matmul(out,weights['out'])+biases['out']
        print('lstm_out1:',lstm_out.shape)
        lstm_out=tf.transpose(lstm_out,[1,0,2])
        print('lstm_out2:',lstm_out.shape)
        
        # convert to size(batchSize, 8,8, nbFilter)
        lstm_out=tf.reshape(lstm_out,[bSize,8,8,nbFilter])
        # perform batch normalization and activiation
        lstm_out=slim.batch_norm(lstm_out,activation_fn=None)
        lstm_out=tf.nn.relu(lstm_out)
        print('lstm_out3:',lstm_out.shape)
        # upsample lstm output to (batchSize, 16,16, nbFilter)
        temp=tf.random_normal([bSize,outSize,outSize,nbFilter])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(2, nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        lstm_out = tf.nn.conv2d_transpose(lstm_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 2, 2, 1])
        print('lstm_out4:',lstm_out.shape)
        # reduce the filter size to nbFilter for layer4
        top = slim.conv2d(layer13,nbFilter,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
        top = tf.nn.relu(top)
        print('top:',top.shape)
        # concatenate both lstm features and image features
        joint_out=tf.concat([top,lstm_out],3)
        print('joint_out:',joint_out.shape)
        # perform upsampling (batchSize, 64,64, 2*nbFilter)
        temp=tf.random_normal([bSize,outSize*4,outSize*4,2*nbFilter])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4, 2*nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer4 = tf.nn.conv2d_transpose(joint_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 	
        # reduce filter sizes	
        upsampled_layer4 = slim.conv2d(upsampled_layer4,2,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_'+str(12))
        upsampled_layer4=slim.batch_norm(upsampled_layer4,activation_fn=None)
        upsampled_layer4=tf.nn.relu(upsampled_layer4)
        # upsampling to (batchSize, 256,256, nbClasses)
        temp=tf.random_normal([bSize,outSize*16,outSize*16,2])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4,2)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer5 = tf.nn.conv2d_transpose(upsampled_layer4, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 
        #upsampled_layer5=slim.batch_norm(upsampled_layer5,activation_fn=None)
        #upsampled_layer5 = slim.conv2d(upsampled_layer5,2,[3,3], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_'+str(5))
        #upsampled_layer5=tf.nn.relu(upsampled_layer5)


        return upsampled_layer5


    y1=tf.transpose(y,[1,2,3,0])
    upsampled_logits=segNet(input_layer,batch_size,freqFeat,weights,biases)
    print('upsampled_logits_shape:',upsampled_logits.shape)

    flat_pred=tf.reshape(upsampled_logits,(-1,n_classes))
    print('flat_pred_shape:',flat_pred.shape)
    
    flat_y=tf.reshape(y1,(-1,n_classes))

    #loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_pred,labels=flat_y))

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(flat_y,flat_pred, 1.0))

    #all_weights  = tf.trainable_variables()
    #regLoss = tf.add_n([ tf.nn.l2_loss(v) for v in all_weights ]) * beta
    #loss = 0.75*loss1+loss2
    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    update = trainer.minimize(loss)
    #update2 = trainer.minimize(loss2)

    probabilities=tf.nn.softmax(flat_pred)
    correct_pred=tf.equal(tf.argmax(probabilities,1),tf.argmax(flat_y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    y_actual=tf.argmax(flat_y,1)
    y_pred=tf.argmax(flat_pred,1)

    mask_actual= tf.argmax(y1,3)
    mask_pred=tf.argmax(upsampled_logits,3)


# Initializing the variables
# init = tf.initialize_all_variables()

saver = tf.train.Saver()

config=tf.ConfigProto()
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

batch_size = 30
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver.restore(sess,'../model_s/modelS.ckpt')
    print ('session starting .................!!!!')
            
    TP = 0; FP = 0;TN = 0; FN = 0 
    #TP1=0;FP1=0
    num_images=batch_size
    

    tx = np.load('../dataset_npy/NC_16/mani/NC16_mani_test_img.npy')
    tx= np.multiply(tx,1.0/mx)
    ty = np.load('../dataset_npy/NC_16/mani/NC16_mani_test_label.npy')
    freq4 = np.load('../dataset_npy/NC_16/mani/NC16_mani_test_imgS_feat.npy')
    
    
    print('ty.shape:',ty.shape)
    nTx=np.zeros((batch_size,256,256,3))
    nTy=np.zeros((batch_size,256,256))
#     nTx1=np.zeros((batch_size,64,240))
    nTx1=np.zeros((batch_size,248,248,3))
    
    n_chunks=np.shape(tx)[0]//batch_size
    tAcc=np.zeros(n_chunks)
    
    n1=0;n2=len(tx)
    pred = np.zeros([150,256,256])
    prob = np.zeros([256*256*150,2])
    for chunk in range(n1,n_chunks):
#         nTx[imNb-n1]=tx[imNb] 
#         nTy[imNb-n1]=ty[imNb]
#         nTx1[imNb-n1]=freq4[imNb]
        print('chunk=',chunk)
        nTx=tx[((chunk)*num_images):((chunk+1)*num_images),...]
        
        nTy=ty[((chunk)*num_images):((chunk+1)*num_images),...]
#         print(nTy.shape)
        nTx1=freq4[((chunk)*num_images):((chunk+1)*num_images),...]
       
        
#         print ('nTx = ',np.shape(nTx))
#         print ('nTy = ',np.shape(nTy))
#         print ('nTx1 = ',np.shape(nTx1))
        ty_prime=conv_mask_gt(nTy)
        final_predictions, final_probabilities,y2=sess.run([mask_pred,probabilities,mask_actual], feed_dict={input_layer: nTx, y:ty_prime, freqFeat: nTx1})
    
        print ('final_predictions_shape:',np.shape(final_predictions))
        print ('final_probabilities_shape;',np.shape(final_probabilities))
        pred[((chunk)*num_images):((chunk+1)*num_images),...] =  final_predictions
        prob[((chunk)*num_images)*256*256:((chunk+1)*num_images)*256*256,...] = final_probabilities
        
        print ('pred_shape:',np.shape(pred))
        print ('prob_shape:',np.shape(prob))
    print('-----------------------------------------------------------')
    print ('pred_shape:',np.shape(pred))
    print ('prob_shape:',np.shape(prob))
    final_predictions1 =pred 
    final_probabilities1 =prob
    print ('final_predictions1_shape:',np.shape(final_predictions1))
    print ('final_probabilities1_shape;',np.shape(final_probabilities1))
    
    #sio.savemat('pred_res.mat',{'img':nTx,'labels':nTy,'pred':final_predictions,'prob':final_probabilities,'gT':y2})
