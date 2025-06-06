import tensorflow as tf
import numpy as np
import cv2
import os

from models import reconstruction
from utils import DataLoader, load, save
import constant



os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-200000'
batch_size = constant.TEST_BATCH_SIZE



# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))


# define testing generator function
with tf.variable_scope('Reconstruction', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    lr_test_stitched, hr_test_stitched = reconstruction(test_inputs)
 


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        
        # Determine the number of test images from the DataLoader
        # Assuming all subfolders (warp1, warp2, etc.) have the same number of images
        if data_loader.images and 'warp1' in data_loader.images and data_loader.images['warp1']['length'] > 0:
            length = data_loader.images['warp1']['length']
            print(f"Found {length} image(s) to process based on DataLoader.")
        else:
            print("Warning: Could not determine number of images from DataLoader or DataLoader is empty. Defaulting to 0.")
            length = 0
            
        if length == 0:
            print("No images to process. Exiting inference.")
            return

        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_image_clips(i), axis=0)
            _, stitch_result = sess.run([lr_test_stitched, hr_test_stitched], feed_dict={test_inputs: input_clip})
            
            stitch_result = (stitch_result+1) * 127.5    
            stitch_result = stitch_result[0]
            # Ensure results directory exists
            results_dir = constant.RESULT_DIR
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, str(i+1).zfill(6) + ".jpg")
            cv2.imwrite(path, stitch_result)
            print('i = {} / {}'.format( i + 1, length))
            
        print("===================DONE!==================")  

    inference_func(snapshot_dir)

    

