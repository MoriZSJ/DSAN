no_cuda = False
cuda_id = '0'
seed = 1
log_interval = 10
bottle_neck = True

batch_size = 32
epochs = 1000
lr = 0.01
momentum = 0.9
l2_decay = 5e-4
param = 0.5  # a->d a->w Config.py中param设为0.5，别的任务0.3， d->a w->a 设为0.3，epoch设为800.
# officehome好像是0.5（0.3 0.5 1试试吧，忘记具体了），imageclef好像也是0.5


class_num = 31  # 65 for office-home
tensorboard_path = 'tensorboard_log/office31/'
root_path = "/home/mori/Programming/DatasetCollection/office31/"
source_name = "amazon"         #
target_name = "webcam"         #
