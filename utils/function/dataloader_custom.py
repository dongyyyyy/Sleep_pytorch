from utils.function.function import *


def make_weights_for_balanced_classes(data_list, nclasses=5,check_file='.npy'):
    count = [0] * nclasses
    
    for data in data_list:
        count[int(data.split(check_file)[0].split('_')[-1])] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data_list)
    for idx, val in enumerate(data_list):
        weight[idx] = weight_per_class[int(val.split(check_file)[0].split('_')[-1])]
    return weight , count

class Sleep_Dataset_cnn_withPath_5classes_newformat(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []
        # print(self.dataset_list)
        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
                    # print(f'filename = {signals_file} // label = {label}')
        # exit(1)
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1],
                 use_cuda = True,
                 classification_mode='5class'
                 ):
        # use_channel
        # 0 : C3-M2 / 1 : C4-M1 / 2 : F3-M2 / 3 : F4-M1 / 4 : O1-M2 / 5 : O2-M1 / 6 : E1-M2 / 7 : E2-M1
        # 8 : 1-2 / 9 : ECG / 10 : Flow / 11 : Chest / 12 : abdomen
        self.channel_list = ['C3-M2','E1-M2']
        self.channel_200hz_list = ['C3-M2','E1-M2']
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()

        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.classification_mode = classification_mode
        print(f'label len = {len(self.labels)}')
        print(f'signals len = {len(self.signals_files_path)}')
        print(f'total length = {self.length}')
        print('classification_mode : ',classification_mode)


    def __getitem__(self, index):
        loader_signals= None
        labels = int(self.labels[index])
        count = 0
        if self.classification_mode == 'REM-NoneREM':
                if labels == 0: # Wake
                    labels = 0
                elif labels == 4: #REM
                    labels = 2
                else: # None-REM
                    labels = 1
        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2

        # current file index
        for channel_num in self.use_channel:
            channel_path = self.channel_list[channel_num]
            addition_signals_files_path = self.signals_files_path[index].split('/')
            addition_signals_files_path[-3] = channel_path
            addition_signals_files_path = '/'.join(addition_signals_files_path)
            
            # print('current path = ', addition_signals_files_path)
            # exit(1)
            signals = np.load(addition_signals_files_path)

            # for i in range(self.seq_size):
            #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

            if self.use_cuda:
                signals = torch.from_numpy(signals).float()
            # print(f'channel_path = {channel_path}')
            if count == 0:
                loader_signals = signals
                count += 1
            else:
                if self.use_cuda:
                    loader_signals = torch.cat((loader_signals,signals),dim=0)
                else:
                    loader_signals = np.concatenate((loader_signals,signals),axis=0)
        
        return loader_signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_cnn_windows_withPath_5classes_newformat(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1],
                 use_cuda = True,
                 window_size=800,
                 stride=400,
                 sample_rate=200,
                 epoch_size=30,
                 classification_mode='5class'
                 ):
        # use_channel
        # 0 : C3-M2 / 1 : C4-M1 / 2 : F3-M2 / 3 : F4-M1 / 4 : O1-M2 / 5 : O2-M1 / 6 : E1-M2 / 7 : E2-M1
        # 8 : 1-2 / 9 : ECG / 10 : Flow / 11 : Chest / 12 : abdomen
        self.channel_list = ['C3-M2','E1-M2']
        self.channel_200hz_list = ['C3-M2','E1-M2']
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()

        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.classification_mode = classification_mode
        print(f'label len = {len(self.labels)}')
        print(f'signals len = {len(self.signals_files_path)}')
        print(f'total length = {self.length}')
        print('classification_mode : ',classification_mode)

        self.seq_size = ((sample_rate*epoch_size)-window_size)//stride + 1
        

        self.window_size = window_size
        self.stride = stride

    def __getitem__(self, index):
        loader_signals = None
        
        labels = int(self.labels[index])
        if self.classification_mode == 'REM-NoneREM':
                if labels == 0: # Wake
                    labels = 0
                elif labels == 4: #REM
                    labels = 2
                else: # None-REM
                    labels = 1
        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2

        # current file index
        count = 0
        for channel_num in self.use_channel:
            channel_path = self.channel_list[channel_num]
            addition_signals_files_path = self.signals_files_path[index].split('/')
            addition_signals_files_path[-3] = channel_path
            addition_signals_files_path = '/'.join(addition_signals_files_path)
            
            signals = []

            c_signals = np.load(addition_signals_files_path)
            if channel_path in self.channel_200hz_list:
                for inner_i in range(self.seq_size_200hz):
                    temp = c_signals[:,inner_i*self.stride_200hz:(inner_i*self.stride_200hz)+self.window_size_200hz]
                    signals.append(temp)        
            # print(signals)
            signals = np.array(signals)
            # print(signals.shape)
            # print('signals.shape : ',signals.shape)
            if self.use_cuda:
                signals = torch.from_numpy(signals).float()
            # print(f'signals shape : {signals.shape}')
            # print(f'channel_path = {channel_path}')
            if count == 0:
                loader_signals = signals
                count+=1
            else:
                loader_signals = torch.cat((loader_signals,signals),1)


        return loader_signals,labels
        
    def __len__(self):
        return self.length 

