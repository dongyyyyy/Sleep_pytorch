from include.header import *

class DeepSleepNet_block(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel,conv1,conv2,mp1,mp2):
        super(DeepSleepNet_block, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channel,out_channels=conv1[3],kernel_size=conv1[0],stride=conv1[1],padding=conv1[2],
                               bias=False)

        self.conv2 = nn.Conv1d(in_channels=conv1[3],out_channels=conv2[3],kernel_size=conv2[0],stride=conv2[1],padding=conv2[2],
                               bias=False)

        self.conv3 = nn.Conv1d(in_channels=conv2[3], out_channels=conv2[3], kernel_size=conv2[0], stride=conv2[1],
                               padding=conv2[2],
                               bias=False)
        self.conv4 = nn.Conv1d(in_channels=conv2[3], out_channels=conv2[3], kernel_size=conv2[0], stride=conv2[1],
                               padding=conv2[2],
                               bias=False)

        self.maxpool1 = nn.MaxPool1d(kernel_size=mp1[0],stride=mp1[1],padding=mp1[2])
        self.maxpool2 = nn.MaxPool1d(kernel_size=mp2[0], stride=mp2[1], padding=mp2[2])

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.maxpool2(out)

        return out



class DeepSleepNet_FE(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel=1):
        super(DeepSleepNet_FE, self).__init__()
        self.bigCNN = DeepSleepNet_block(in_channel,conv1=[800,100,400,64],conv2=[6,1,3,128],mp1=[4,4,2],mp2=[2,2,1])
        self.smallCNN = DeepSleepNet_block(in_channel,conv1=[100, 12, 50, 64], conv2=[8, 1, 4, 128], mp1=[8, 8, 4], mp2=[4, 4, 2])

    def forward(self,x):
        feature_big = self.bigCNN(x)
        feature_small = self.smallCNN(x)

        feature_big = torch.flatten(feature_big, 1)
        feature_small = torch.flatten(feature_small, 1)

        out = torch.cat((feature_big,feature_small),dim=1)

        return out



class DeepSleepNet_classification(nn.Module):  # 2D 없이 1D로 통일하기
    def __init__(self,in_channel=1,class_num=5):
        super(DeepSleepNet_classification, self).__init__()
        self.feature_extract = DeepSleepNet_FE(in_channel=in_channel)
        self.dropout = nn.Dropout(p=0.5)
        self.classification = nn.Linear(3456,class_num)

    def forward(self,x):
        out = self.feature_extract(x)
        out = self.classification(out)

        return out
