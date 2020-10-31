from easydict import EasyDict as edict

config = edict()
config.dataset = "emore"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1
config.output_folder = '/home/zhaoxin/exp_results/r100_glint360k'

if config.dataset == "emore":
    config.rec = "/home/zhaoxin/datasets/glint360k/glint360k/"
    config.num_classes = 360232
    config.num_epoch = 20


    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [10, 16, 22] if m - 1 <= epoch])


    config.lr_func = lr_step_func
