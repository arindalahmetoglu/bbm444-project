#training dataset path
TRAIN_FOLDER = '../../Dataset/Stage2/training/'

#testing dataset path
TEST_FOLDER = '../../../temp_sift_subset_udis_work_101121/'

#GPU index
GPU = '0'

#batch size for training
TRAIN_BATCH_SIZE = 1

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 200000

# checkpoints path
SNAPSHOT_DIR = './checkpoints'

#sumary path
SUMMARY_DIR = "./summary"

# the فصلoss weights
RECONSTRUCTION_W = 1
SSIM_W = 1
PERCEPTUAL_W = 0.00001
SMOOTH_W = 0.0000001

#
# dataset
#
#TEST_BATCH_SIZE = 1
TEST_FOLDER = '../../../temp_sift_subset_udis_work_101121/'
RESULT_DIR = r'/home/arin/Projects/bbm444-project/processing_data/global_stitching_results/UDIS'


#
# network
#
LR_SIZE = 256
HR_SIZE = 512
