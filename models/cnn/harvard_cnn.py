from include.header import *
from models.cnn.ResNet_custom import *


class multiFilterBlock(nn.Module):
    def __init__(self,kernel_size_small=[[100,10,0],[50,2,0]],blocks=[16,32],kernel_size_big=[[200,10,0],[100,2,0]],maxpool_size_small=[1,1],maxpool_size_big=[1,1]):
        super().__init__()
        self.conv_big1 = nn.Conv1d(in_channels=1,out_channels=blocks[0],kernel_size=kernel_size_big[0][0],stride=kernel_size_big[0][1],bias=False)
        self.bn_big1 = nn.BatchNorm1d(blocks[0])
        self.relu_big1 = nn.ReLU()

        self.conv_big2 = nn.Conv1d(in_channels=blocks[0],out_channels=blocks[1],kernel_size=kernel_size_big[1][0],stride=kernel_size_big[1][1],bias=False)
        self.bn_big2 = nn.BatchNorm1d(blocks[1])
        self.relu_big2 = nn.ReLU()

        self.maxpool_big = nn.MaxPool1d(maxpool_size_big[0],maxpool_size_big[1])

        self.conv_small1 = nn.Conv1d(in_channels=1,out_channels=blocks[0],kernel_size=kernel_size_small[0][0],stride=kernel_size_small[0][1],bias=False)
        self.bn_small1 = nn.BatchNorm1d(blocks[0])
        self.relu_small1 = nn.ReLU()

        self.conv_small2 = nn.Conv1d(in_channels=blocks[0],out_channels=blocks[1],kernel_size=kernel_size_small[1][0],stride=kernel_size_small[1][1],bias=False)
        self.bn_small2 = nn.BatchNorm1d(blocks[1])
        self.relu_small2 = nn.ReLU()

        self.maxpool_small = nn.MaxPool1d(maxpool_size_small[0],maxpool_size_small[1])

    def forward(self, x):

        out_big = self.conv_big1(x)
        out_big = self.bn_big1(out_big)
        out_big = self.relu_big1(out_big)

        out_big = self.conv_big2(out_big)
        out_big = self.bn_big2(out_big)
        out_big = self.relu_big2(out_big)

        out_big = self.maxpool_big(out_big)

        out_small = self.conv_small1(x)
        out_small = self.bn_small1(out_small)
        out_small = self.relu_small1(out_small)

        out_small = self.conv_small2(out_small)
        out_small = self.bn_small2(out_small)
        out_small = self.relu_small2(out_small)
        out_small = self.maxpool_small(out_small)


        return out

class harvard_resnet_FE(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3,4,6,3], layer_filters=[64, 128, 256, 512], maxpool_size=[2,2,2],in_channel=6,
                 block_kernel_size=3,block_stride_size=1,num_classes=5, use_batchnorm=True, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,dropout_p=0.):
        super().__init__()
        self.inplanes = in_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)

        self.dropout = nn.Identity()
        self.dropout_p = dropout_p
        if self.dropout_p != 0.:
            self.dropout = nn.Dropout(p=self.dropout_p)

        self.block_kernel_size = block_kernel_size
        self.block_stride_size = block_stride_size

        self.padding = self.block_kernel_size // 2

        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3],
                                       dilate=replace_stride_with_dilation[2])

        self.maxpool1 = nn.MaxPool1d(maxpool_size[0],maxpool_size[0])
        self.maxpool2 = nn.MaxPool1d(maxpool_size[1],maxpool_size[1])
        self.maxpool3 = nn.MaxPool1d(maxpool_size[2],maxpool_size[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                '''
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                '''

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # print('drop out : ',self.dropout_p)
        layers.append(
            block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, stride=stride, downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width, dilation=self.padding,
                  norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.padding,
                                norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.layer1(x) # 375
        x = self.layer2(x) # 375 / 4 = 94
        x = self.layer3(x) # 94 / 4 = 24
        x = self.layer4(x) # 24 / 4 = 6

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2
        

        return x

    def forward(self, x):
        return self._forward_impl(x)

class harvard_cnn(nn.Module):
    def __init__(self,in_channels=6,class_num=5):
        super().__init__()
        self.fe1 = multiFilterBlock()
        self.fe2 = multiFilterBlock()
        self.fe3 = multiFilterBlock()
        self.fe4 = multiFilterBlock()
        self.fe5 = multiFilterBlock()
        self.fe6 = multiFilterBlock()

        self.fe_resnet = harvard_resnet_FE(in_channel=32)

        self.fc = nn.Linear(512,class_num)

    def forward(self, x):
        fe1 = self.fe1(x)
        fe2 = self.fe2(x)
        fe3 = self.fe3(x)
        fe4 = self.fe4(x)
        fe5 = self.fe5(x)
        fe6 = self.fe6(x)

        fe = torch.cat((fe1,fe2,fe3,fe4,fe5,fe6),dim=1)

        out = self.fe_resnet(fe)

        out = self.fc(out)

        return out


