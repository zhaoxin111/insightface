from easydict import EasyDict as edict

config = edict()
config.dataset = "emore"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 60
config.lr = 0.05
config.output = '/data/zhaoxin_data/renren_filtered_data_v3/'

if config.dataset == "emore":
    config.rec = '/data/zhaoxin_data/renren_filtered_data_v3/'
    config.num_classes = 142199
    config.num_epoch = 30


    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [10, 16, 22] if m - 1 <= epoch])


    config.lr_func = lr_step_func
