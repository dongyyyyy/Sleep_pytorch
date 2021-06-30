from include.header import *
from models.modules.cnn_module import *


class ResNet_200hz(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3,block_stride_size=1,embedding=256, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,dilation=1,
                 norm_layer=None,dropout_p=0.):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = layer_filters[0]
        self.dilation = dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)

        self.dropout = nn.Identity()
        self.dropout_p = dropout_p
        if self.dropout_p != 0.:
            self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.bn1 = nn.Identity()
        if self.use_batchnorm:
            self.bn1 = norm_layer(self.inplanes)
        
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool[0], stride=maxpool[1], padding=maxpool[2])

        self.block_kernel_size = block_kernel_size
        self.block_stride_size = block_stride_size

        self.padding = self.block_kernel_size // 2

        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        self.fc = nn.Linear(self.inplanes, num_classes)

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
        print(f'self.dilation = {self.dilation}')
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
                  base_width=self.base_width, padding=self.padding,dilation=self.dilation,
                  norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, padding=self.padding,dilation=self.dilation,
                                norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x shape : 6000
        # See note [TorchScript super()]
        x = self.conv1_1d(x) # 6000 / 4 = 1500
        x = self.bn1(x) 

        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x) # 1500 / 4 = 375
        # print(f'x1 shape = {x.shape}')
        x = self.layer1(x) # 375
        # print(f'x2 shape = {x.shape}')
        x = self.layer2(x) # 375 / 4 = 94
        x = self.layer3(x) # 94 / 4 = 24
        x = self.layer4(x) # 24 / 4 = 6

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2
        x = self.fc(x)  # -1

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_200hz_FE(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 128, 256, 512], in_channel=1,
                 block_kernel_size=3,block_stride_size=1,embedding=256, num_classes=5, use_batchnorm=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,dilation=1,
                 norm_layer=None,dropout_p=0.):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.use_batchnorm = use_batchnorm
        self.inplanes = layer_filters[0]
        self.dilation = dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_conv[0], stride=first_conv[1],
                                  padding=first_conv[2],
                                  bias=False)
        # self.conv1_1d = nn.Conv1d(in_channel, self.inplanes, kernel_size=200, stride=40, padding=100,
        #                           bias=False)

        self.dropout = nn.Identity()
        self.dropout_p = dropout_p
        if self.dropout_p != 0.:
            self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.bn1 = nn.Identity()
        if self.use_batchnorm:
            self.bn1 = norm_layer(self.inplanes)
        
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool[0], stride=maxpool[1], padding=maxpool[2])

        self.block_kernel_size = block_kernel_size
        self.block_stride_size = block_stride_size

        self.padding = self.block_kernel_size // 2

        self.layer1 = self._make_layer(block, layer_filters[0], layers[0])
        self.layer2 = self._make_layer(block, layer_filters[1], layers[1], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_filters[2], layers[2], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_filters[3], layers[3], stride=self.block_stride_size,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

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
        print(f'self.dilation = {self.dilation}')
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
                  base_width=self.base_width, padding=self.padding,dilation=self.dilation,
                  norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_kernel_size=self.block_kernel_size, groups=self.groups,
                                base_width=self.base_width, padding=self.padding,dilation=self.dilation,
                                norm_layer=norm_layer,dropout_p=self.dropout_p,use_batchnorm=self.use_batchnorm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x shape : 6000
        # See note [TorchScript super()]
        x = self.conv1_1d(x) # 6000 / 4 = 1500
        x = self.bn1(x) 

        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x) # 1500 / 4 = 375

        x = self.layer1(x) # 375
        x = self.layer2(x) # 375 / 4 = 94
        x = self.layer3(x) # 94 / 4 = 24
        x = self.layer4(x) # 24 / 4 = 6

        x = self.avgpool(x)  # -3
        x = torch.flatten(x, 1)  # -2


        return x

    def forward(self, x):
        return self._forward_impl(x)