from include.header import *
from models.modules.transformer_module import *

class featureExtract_window(nn.Module):
    def __init__(self, in_channels=2,first_conv=[11,6,0],use_batchnorm=True,padding=True,weightnorm=False):
        super(featureExtract_window, self).__init__()

        self.conv1 = self._make_layer(inplanes=in_channels, planes=64, kernel_size=first_conv[0],padding=padding,stride=first_conv[1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv2 = self._make_layer(inplanes=64, planes=64, kernel_size=5,padding=padding,stride=3,use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv3 = self._make_layer(inplanes=64, planes=128, kernel_size=3,padding=padding,stride=2,use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv4 = self._make_layer(inplanes=128, planes=128, kernel_size=3,padding=padding,stride=1,use_batchnorm=use_batchnorm,weightnorm=weightnorm)                                            
        self.conv5 = self._make_layer(inplanes=128, planes=256, kernel_size=3,padding=padding,stride=1,use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
    def _make_layer(self, inplanes, planes, kernel_size,padding=True,stride=1,use_batchnorm=True,weightnorm=False):
        
        if use_batchnorm:
            if padding:
                layers = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size//2,
                                        bias=False),
                    nn.BatchNorm1d(planes),
                )
            else:
                layers = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=0,
                                        bias=False),
                    nn.BatchNorm1d(planes),
                )

        else:
            if padding:
                layers = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size//2,
                                        bias=False)
            else:
                layers = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=0,
                                        bias=False)
        
        return layers

    def forward(self,x):
        # print('x shape : ',x.shape)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        
        return out


class featureExtract_window_new(nn.Module):
    def __init__(self, in_channels=2,blocks=[64,64,128,128,256],kernel_size=[[19,4],[9,2],[9,2],[9,1],[9,1]],use_batchnorm=True,padding=True,weightnorm=False):
        super().__init__()

        self.conv1 = self._make_layer(inplanes=in_channels, planes=blocks[0], kernel_size=kernel_size[0][0],
        padding=padding,stride=kernel_size[0][1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv2 = self._make_layer(inplanes=blocks[0], planes=blocks[1], kernel_size=kernel_size[1][0],
        padding=padding,stride=kernel_size[1][1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv3 = self._make_layer(inplanes=blocks[1], planes=blocks[2], kernel_size=kernel_size[2][0],
        padding=padding,stride=kernel_size[2][1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
        self.conv4 = self._make_layer(inplanes=blocks[2], planes=blocks[3], kernel_size=kernel_size[3][0],
        padding=padding,stride=kernel_size[3][1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                                            
        self.conv5 = self._make_layer(inplanes=blocks[3], planes=blocks[4], kernel_size=kernel_size[4][0],
        padding=padding,stride=kernel_size[4][1],use_batchnorm=use_batchnorm,weightnorm=weightnorm)                
                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  #
    def _make_layer(self, inplanes, planes, kernel_size,padding=True,stride=1,use_batchnorm=True,weightnorm=False):
        
        if use_batchnorm:
            if padding:
                layers = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size//2,
                                        bias=False),
                    nn.BatchNorm1d(planes),
                )
            else:
                layers = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=0,
                                        bias=False),
                    nn.BatchNorm1d(planes),
                )

        else:
            if padding:
                layers = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size//2,
                                        bias=False)
            else:
                layers = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=0,
                                        bias=False)
        
        return layers

    def forward(self,x):
        # print('x shape : ',x.shape)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        
        return out
        
class inner_transformer_permute(nn.Module):
    def __init__(self, inner_seq_length=29,in_channels=2,first_conv=[11,6,0],i_hidden=256,d_hidden=256,f_hidden=512,dropout=0.4,
                 n_head=8,num_layers=3,mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,
                 num_classes =5,use_batchnorm=True,padding=True,weightnorm=False,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.featureExtract = featureExtract_window(in_channels=in_channels,first_conv=first_conv,use_batchnorm=use_batchnorm,padding=padding,weightnorm=weightnorm)

        self.inner_transformer = Encoder(seq_length=inner_seq_length,hidden_size=d_hidden,f_hidden=f_hidden,dropout=dropout,n_head=n_head,num_layers=num_layers,
                                        mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,layer_norm=layer_norm,layer_norm_first=layer_norm_first)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.fc = nn.Linear(inner_seq_length,num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(i_hidden,num_classes)
    def forward(self, x):
        # print('inner_Transformer x shape 1 : ',x.shape)
        seq = []
        # print(x.shape)
        # x shape : [batch, sequence,/ channel, window size]
        for sequence in range(x.size(1)):
            x_t = x[:,sequence,:,:]
            # print(f'x_t shape = {x_t.shape}')
            x_t = self.featureExtract(x_t)

            seq.append(x_t)
        
        out = torch.stack(seq,dim=1)
        # [batch, seq, vector]
        # print('feature extract : ',out.shape)
        out,_ = self.inner_transformer(out)
        # [batch, seq, vector]
        # print('inner output : ',out.shape)
        out = out.permute(0,2,1)
        
        out = self.avgpool(out)
        # [batch, seq, 1]
        out = torch.flatten(out,1)
        out = self.fc(out)

        return out



class outer_transformer_permute_withSkip_end(nn.Module):
    def __init__(self, FeatureExtract,outer_seq_length=3,i_hidden=256,d_hidden=256,f_hidden=512,fc_hidden=256,dropout=0.1,
                 n_head=8,num_layers=6,mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,
                 num_classes =5,layer_norm=True,layer_norm_first=False,output_index = -1):
        super().__init__()
        self.featureExtract = FeatureExtract
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.fc = nn.Linear(inner_seq_length,num_classes)
        self.outer_transformer = Encoder(seq_length=outer_seq_length,hidden_size=d_hidden,f_hidden=f_hidden,dropout=dropout,n_head=n_head,num_layers=num_layers,
                                        mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                        layer_norm=layer_norm,layer_norm_first=layer_norm_first)
        self.featureExtract.eval()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(i_hidden,fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.classification = nn.Linear(fc_hidden,num_classes)
        self.output_index = output_index
        print('output index : ',output_index)


    def forward(self, x):
        seq = []
        # x shape : [batch, outer_sequence, inner_sequence, channel, window size]
        with torch.no_grad():
            for outer_sequence in range(x.size(1)):
                x_out = x[:,outer_sequence,:,:]
                seq_inner = self.featureExtract(x_out)

                seq.append(seq_inner)
        
        out = torch.stack(seq,dim=1)
        # [batch, out, vector]
        skip = out[:,self.output_index,:]
        # print('feature extract : ',out.shape)
        out,_ = self.outer_transformer(out)
        # [batch, seq, vector]
        # print('inner output : ',out.shape)
        out = out.permute(0,2,1)
        out = self.avgpool(out)
        # [batch, seq, 1]
        out = torch.flatten(out,1)

        out += skip

        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.classification(out)

        return out

class outer_transformer_permute_end(nn.Module):
    def __init__(self, FeatureExtract,outer_seq_length=3,i_hidden=256,d_hidden=256,f_hidden=512,fc_hidden=256,dropout=0.1,
                 n_head=8,num_layers=6,mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,
                 num_classes =5,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.featureExtract = FeatureExtract
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.fc = nn.Linear(inner_seq_length,num_classes)
        self.outer_transformer = Encoder(seq_length=outer_seq_length,hidden_size=d_hidden,f_hidden=f_hidden,dropout=dropout,n_head=n_head,num_layers=num_layers,
                                        mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,
                                        layer_norm=layer_norm,layer_norm_first=layer_norm_first)
        # self.featureExtract.eval()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(i_hidden,fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.classification = nn.Linear(fc_hidden,num_classes)


    def forward(self, x):
        seq = []
        # x shape : [batch, outer_sequence, inner_sequence, channel, window size]
        # with torch.no_grad():
        for outer_sequence in range(x.size(1)):
            x_out = x[:,outer_sequence,:,:]
            seq_inner = self.featureExtract(x_out)

            seq.append(seq_inner)
        
        out = torch.stack(seq,dim=1)
        # [batch, out, vector]
        # print('feature extract : ',out.shape)
        out,_ = self.outer_transformer(out)
        # [batch, seq, vector]
        # print('inner output : ',out.shape)
        out = out.permute(0,2,1)
        out = self.avgpool(out)
        # [batch, seq, 1]
        out = torch.flatten(out,1)

        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.classification(out)

        return out


class inner_transformer_permute_newFE(nn.Module):
    def __init__(self, inner_seq_length=29,in_channels=2,first_conv=[11,6],i_hidden=256,d_hidden=256,f_hidden=512,dropout=0.4,
                 n_head=8,num_layers=3,mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,
                 num_classes =5,use_batchnorm=True,padding=True,weightnorm=False,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.featureExtract = featureExtract_window_new(in_channels=in_channels,blocks=[64,64,128,128,256],kernel_size=[first_conv,[9,2],[9,2],[9,1],[9,1]],use_batchnorm=True,padding=True,weightnorm=False)

        self.inner_transformer = Encoder(seq_length=inner_seq_length,hidden_size=d_hidden,f_hidden=f_hidden,dropout=dropout,n_head=n_head,num_layers=num_layers,
                                        mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,layer_norm=layer_norm,layer_norm_first=layer_norm_first)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.fc = nn.Linear(inner_seq_length,num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(i_hidden,num_classes)
    def forward(self, x):
        # print('inner_Transformer x shape 1 : ',x.shape)
        seq = []
        # x shape : [batch, sequence, channel, window size]
        for sequence in range(x.size(1)):
            # print(f'x shape = {x.shape}')
            x_t = x[:,sequence,:,:]
            # print(f'x_t shape = {x_t.shape}')
            x_t = self.featureExtract(x_t)

            seq.append(x_t)
        
        out = torch.stack(seq,dim=1)
        # [batch, seq, vector]
        # print('feature extract : ',out.shape)
        out,_ = self.inner_transformer(out)
        # [batch, seq, vector]
        # print('inner output : ',out.shape)
        out = out.permute(0,2,1)
        
        out = self.avgpool(out)
        # [batch, seq, 1]
        out = torch.flatten(out,1)
        out = self.fc(out)

        return out

class inner_transformer_permute_conv1(nn.Module):
    def __init__(self, inner_seq_length=29,in_channels=2,first_conv=[11,6,0],i_hidden=256,d_hidden=256,f_hidden=512,dropout=0.4,
                 n_head=8,num_layers=3,mq_active=None,mv_active=None,f_first_active=F.gelu,f_second_active=None,
                 num_classes =5,use_batchnorm=True,padding=True,weightnorm=False,layer_norm=True,layer_norm_first=False):
        super().__init__()
        self.featureExtract = featureExtract_window(in_channels=in_channels,first_conv=first_conv,use_batchnorm=use_batchnorm,padding=padding,weightnorm=weightnorm)

        self.inner_transformer = Encoder(seq_length=inner_seq_length,hidden_size=d_hidden,f_hidden=f_hidden,dropout=dropout,n_head=n_head,num_layers=num_layers,
                                        mq_active=mq_active,mv_active=mv_active,f_first_active=f_first_active,f_second_active=f_second_active,layer_norm=layer_norm,layer_norm_first=layer_norm_first)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        # self.fc = nn.Linear(inner_seq_length,num_classes)
        self.conv1 = nn.Conv1d(inner_seq_length,1,1,bias=False)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.relu1 = nn.ReLU()

        self.fc = nn.Linear(i_hidden,num_classes)
    def forward(self, x):
        # print('inner_Transformer x shape 1 : ',x.shape)
        seq = []
        # print(f' x shape = {x.shape}')
        # x shape : [batch, sequence, channel, window size]
        for sequence in range(x.size(1)):
            # print(f'x shape = {x.shape}')
            x_t = x[:,sequence,:,:]
            # print(f'x_t shape = {x_t.shape}')
            x_t = self.featureExtract(x_t)

            seq.append(x_t)
        
        out = torch.stack(seq,dim=1)
        # [batch, seq, vector]
        # print('feature extract : ',out.shape)
        out,_ = self.inner_transformer(out)
        # [batch, seq, vector]
        # print('inner output : ',out.shape)
        out = self.conv1(out)
        out = torch.flatten(out,1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc(out)

        return out