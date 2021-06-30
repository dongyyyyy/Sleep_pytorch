from train.single_epoch.train_DeepSleepNet import training_deepsleepnet_dataloader_seoul
from train.single_epoch.train_ResNet import training_resnet_dataloader_seoul,training_resnet50_dataloader_seoul
if __name__ == '__main__':
    # training_deepsleepnet_dataloader_seoul(use_channel=[0,1],classification_mode='5class',gpu_num=0)
    training_resnet_dataloader_seoul(use_channel=[0,1],classification_mode='5class',gpu_num=1)