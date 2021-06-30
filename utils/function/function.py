from include.header import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
        
#Standard Scaler torch
def data_preprocessing_torch(signals): # 하나의 데이터셋에 대한 data_preprocessing (using torch)
    signals = (signals - signals.mean(dim=1).unsqueeze(1))/signals.std(dim=1).unsqueeze(1)
    return signals

#Standard Scaler npy
def data_preprocessing_numpy(signals): # zero mean unit variance 한 환자에 대한 signal 전체에 대한 normalize
    signals = (signals - np.expand_dims(signals.mean(axis=1), axis=1)) / np.expand_dims(signals.std(axis=1), axis=1)
    return signals

#MinMax Scaler torch
def data_preprocessing_oneToOne_torch(signals,min,max,max_value):
    signals_std = (signals + max_value) / (2*max_value)
    signals_scaled = signals_std * (max - min) + min
    return signals_scaled

def get_dataset_selectChannel(signals_path,annotations_path,filename,select_channel=[0,1,2],use_noise=False,epsilon=0.5,noise_scale=2e-6,preprocessing=False,norm_methods='Standard',cut_value=200,device='cpu'):
    signals = np.load(signals_path+filename)

    annotations = np.load(annotations_path+filename)
    # print(signals.shape)
    signals = signals[select_channel]

    signals = torch.from_numpy(signals).float().to(device)
    annotations = torch.from_numpy(annotations).long().to(device)

    if preprocessing:
        if norm_methods=='Standard':
            signals = data_preprocessing_torch(signals)
        elif norm_methods=='OneToOne':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,-1,1,cut_value)
        elif norm_methods=='MinMax':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,0,1,cut_value)

    return signals,annotations

def expand_signals_torch(signals,channel_len,sample_rate=200,epoch_sec=30):
    signals = signals.unsqueeze(0)
    #print(signals.shape)
    signals = signals.transpose(1,2)
    #print(batch_signals.shape)
    signals = signals.view(-1,sample_rate*epoch_sec,channel_len)
    #print(batch_signals.shape)
    signals = signals.transpose(1,2)
    return signals

def suffle_dataset_list(dataset_list): # 데이터 셔플
    random.shuffle(dataset_list)
    return dataset_list

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:         # Conv weight init
        torch.nn.init.xavier_uniform_(m.weight.data)
    

def int_to_string(num):
    str_num = str(num).zfill(4)
    return str_num

# lowpass filter
def butter_lowpass_filter(data, cutoff, order=4,nyq=100):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False,output='ba')
    y = filtfilt(b, a, data)
    return y

# highpass filter
def butter_highpass_filter(data, cutoff, order=4,fs=200):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='high', analog=False,output='ba')

    y = filtfilt(b, a, data)
    # b = The numerator coefficient vector of the filter (분자)
    # a = The denominator coefficient vector of the filter (분모)

    return y

def butter_bandpass(lowcut, highcut, fs=200 , order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(N=order,Wn=[low,high],btype='bandpass', analog=False,output='ba')
    return b,a

# bandpass filter
def butter_bandpass_filter(signals, lowcut, highcut, fs , order = 4):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)

    y = lfilter(b,a,signals)
    return y

def butter_filter_sos(signals, lowcut=None, highcut=None, fs=200 , order =4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.butter(N=order,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
        filtered = signal.sosfilt(sos,signals)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.butter(N=order,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None: 
        sos = signal.butter(N=order,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals 
    filtered = signal.sosfilt(sos,signals)
    return filtered

def ellip_filter_sos(signals,rp=6,rs=53, lowcut=None, highcut=None, fs = 200 , order = 4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None: 
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals 
    filtered = signal.sosfilt(sos,signals)
    return filtered

def check_label_info(signals_path, file_list,class_mode='5class',check='All',check_file='.npy'):
    if class_mode =='5class':
        labels = np.zeros(5)
    else:
        labels = np.zeros(3)
    for dataset_folder in file_list:
        osa = int(dataset_folder.split('_')[-2])
        signals_paths = signals_path + dataset_folder+'/'
        signals_list = []
        if check == 'normal':
            if osa == 0:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'mild':
            # print('mild')
            if osa == 1:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'moderate':
            if osa == 2:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'severe':
            if osa == 3:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        else: # All
            signals_list = os.listdir(signals_paths)
            signals_list.sort()
        if len(signals_list) != 0:
            for signals_filename in signals_list:
                if class_mode == '5class':
                    labels[int(signals_filename.split(check_file)[0].split('_')[-1])] += 1
                else:
                    current_label = int(signals_filename.split(check_file)[0].split('_')[-1])
                    if current_label == 4:
                        labels[2] += 1
                    elif current_label == 0:
                        labels[0] += 1
                    else:
                        labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent

def check_label_info_withPath( file_list,class_mode='5class',check='All',check_file='.npy'):
    if class_mode =='5class':
        labels = np.zeros(5)
    else:
        labels = np.zeros(3)
    for signals_paths in file_list:
        # print(signals_paths)
        osa = int(signals_paths.split('/')[-2].split('_')[-2])
        signals_list = []
        if check == 'normal':
            if osa == 0:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'mild':
            # print('mild')
            if osa == 1:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'moderate':
            if osa == 2:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'severe':
            if osa == 3:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        else: # All
            signals_list = os.listdir(signals_paths)
            signals_list.sort()
        if len(signals_list) != 0:
            for signals_filename in signals_list:
                if class_mode == '5class':
                    labels[int(signals_filename.split(check_file)[0].split('_')[-1])] += 1
                else:
                    current_label = int(signals_filename.split(check_file)[0].split('_')[-1])
                    if current_label == 4:
                        labels[2] += 1
                    elif current_label == 0:
                        labels[0] += 1
                    else:
                        labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent

def check_label_info_W_NR_R(signals_path, file_list):
    labels = np.zeros(3)
    for dataset_folder in file_list:
        signals_paths = signals_path + dataset_folder+'/'
        signals_list = os.listdir(signals_paths)
        signals_list.sort()
        for signals_filename in signals_list:
            if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0:
                labels[0] += 1
            elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                labels[2] += 1
            else:
                labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent

def check_label_change_W_NR_R(signals_path, file_list):
    
    total_change = [0 for _ in range(6)] # W -> NR / W -> R / NR -> W / NR -> R / R -> W / R -> NR
    total_count = [0 for _ in range(6)]
    total_num = [[] for _ in range(6)]
    
    for dataset_folder in file_list:
        current_label = 0
        count = 0
        signals_paths = signals_path + dataset_folder+'/'
        signals_list = os.listdir(signals_paths)
        signals_list.sort()
        
        for index,signals_filename in enumerate(signals_list):
            if index == 0:
                if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0:
                    current_label = 0
                elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                    current_label = 2
                else:
                    current_label = 1
                count = 1
            else: # W -> NR / W -> R / NR -> W / NR -> R / R -> W / R -> NR
                if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0: # Wake
                    if current_label == 0: # Wake 그대로 지속
                        count += 1
                    else: # NR 또는 R에서 Wake로 온 경우
                        if current_label == 1: # NR -> W
                            total_change[2] += 1 
                            total_count[2] += count
                            total_num[2].append(count)
                        else:
                            total_change[4] += 1
                            total_count[4] += count
                            total_num[4].append(count)
                        current_label = 0 # label change
                        count = 1 # count init

                elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                    if current_label == 2:
                        count += 1
                    else:
                        if current_label == 0: # W -> R
                            total_change[1] += 1 
                            total_count[1] += count
                            total_num[1].append(count)
                        else: # NR -> R
                            total_change[3] += 1
                            total_count[3] += count
                            total_num[3].append(count)

                        current_label = 2 # label change
                        count = 1 # count init
                else:
                    if current_label == 1:
                        count += 1
                    else:
                        if current_label == 0: # W -> NR
                            total_change[0] += 1 
                            total_count[0] += count
                            total_num[0].append(count)
                        else: # R -> NR
                            total_change[5] += 1
                            total_count[5] += count
                            total_num[5].append(count)

                        current_label = 1 # label change
                        count = 1 # count init
    print(total_change)
    print(total_count)
    total_change = np.array(total_change)
    total_count= np.array(total_count)
    total_num = [np.array(i) for i in total_num]
    
    print(total_count/total_change)
    print(np.sum(total_count)/np.sum(total_change))

    # total_mean = total_num.mean(1)
    for index,i in enumerate(total_num):
        print(i[:100])
        print('mean : ',i.mean())
        print('std : ',i.std())
        plt.hist(i, bins=50)
        plt.savefig('/home/eslab/%d_plot.png'%index)
        plt.cla()
    
    
    # exit(1)


def psd(signals):
    freqs, psd = signal.welch(signals,200,nperseg=200)
    return freqs, psd

def check_change_label_info():
    file_path = '/home/eslab/dataset/seoulDataset/2channel_prefilter_minmax_-1_1/signals_dataloader/'
    file_list = os.listdir(file_path)
    osa_list = [[],[],[],[]]
    for filename in file_list:
        osa_list[int(filename.split('_')[1])].append(filename)
        
    for index in range(len(osa_list)):
        print(len(osa_list[index]))

    normal_label,normal_label_percent = check_label_info_W_NR_R(signals_path = file_path, file_list = osa_list[0])
    mild_label,mild_label_percent = check_label_info_W_NR_R(signals_path = file_path, file_list = osa_list[1])
    moderate_label,moderate_label_percent = check_label_info_W_NR_R(signals_path = file_path, file_list = osa_list[2])
    severe_label,severe_label_percent = check_label_info_W_NR_R(signals_path = file_path, file_list = osa_list[3])

    for i in range(len(normal_label_percent)):
        print('%.3f '%normal_label_percent[i], end='')
    print('')
    for i in range(len(normal_label_percent)):
        print('%.3f '%mild_label_percent[i], end='')
    print('')
    for i in range(len(normal_label_percent)):
        print('%.3f '%moderate_label_percent[i], end='')
    print('')
    for i in range(len(normal_label_percent)):
        print('%.3f '%severe_label_percent[i], end='')
    print('')
    check_label_change_W_NR_R(signals_path = file_path, file_list = osa_list[0])
    check_label_change_W_NR_R(signals_path = file_path, file_list = osa_list[1])
    check_label_change_W_NR_R(signals_path = file_path, file_list = osa_list[2])
    check_label_change_W_NR_R(signals_path = file_path, file_list = osa_list[3])


def check_sleepstaging():
    signals_path = '/home/eslab/dataset/seoulDataset/7channel_prefilter_butter_minmax_-1_1/signals_dataloader/'

    dataset_list = os.listdir(signals_path)
    dataset_list.sort()
    
    random_seed = 2
    
    random.seed(random_seed) # seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    random.shuffle(dataset_list)

    osa_dataset_list = [[],[],[],[]]
    training_fold_list = []
    validation_fold_list = []
    test_fold_list = []
    for dataset in dataset_list:
        osa_dataset_list[int(dataset.split('_')[-2])].append(dataset)
    osa_len = []
    for index in range(len(osa_dataset_list)):
        osa_len.append(len(osa_dataset_list[index]))

    print(osa_len)
    train_len = [int(index * 0.7) for index in osa_len]
    
    val_len = [(osa_len[index] - train_len[index])//2 for index in range(len(osa_len))]
    print('train osa : ',np.array(train_len))
    print('val osa : ', np.array(val_len))
    print('test osa : ',np.array(osa_len)-np.array(train_len)-np.array(val_len))
    training_fold_list = []
    validation_fold_list = []
    test_fold_list = []


    for osa_index in range(len(osa_dataset_list)):
        
        for i in range(0,train_len[osa_index]):
            training_fold_list.append(osa_dataset_list[osa_index][i])
        for i in range(train_len[osa_index],train_len[osa_index]+val_len[osa_index]):
            validation_fold_list.append(osa_dataset_list[osa_index][i])
        for i in range(train_len[osa_index]+val_len[osa_index],len(osa_dataset_list[osa_index])):
            test_fold_list.append(osa_dataset_list[osa_index][i])    
    
    # print(dataset_list[:10])

    print(len(training_fold_list))
    print(len(validation_fold_list))
    print(len(test_fold_list)) 
    check_list = ['normal','mild','moderate','severe']
    for index,check in enumerate(check_list):
        osa_label, osa_label_percent = check_label_info(signals_path = signals_path, file_list = training_fold_list,class_mode='Rem-NoneREM',check=check)
        print(f'train {check} info ==> {osa_label} // {osa_label_percent}')

    for index,check in enumerate(check_list):
        osa_label, osa_label_percent = check_label_info(signals_path = signals_path, file_list = validation_fold_list,class_mode='Rem-NoneREM',check=check)
        print(f'validation {check} info ==> {osa_label} // {osa_label_percent}')

    for index,check in enumerate(check_list):
        osa_label, osa_label_percent = check_label_info(signals_path = signals_path, file_list = test_fold_list,class_mode='Rem-NoneREM',check=check)
        print(f'test {check} info ==> {osa_label} // {osa_label_percent}')

    train_label,train_label_percent = check_label_info(signals_path = signals_path, file_list = training_fold_list,class_mode='Rem-NoneREM',check='All')
    val_label,val_label_percent = check_label_info(signals_path = signals_path, file_list = validation_fold_list,class_mode='Rem-NoneREM',check='All')
    test_label,test_label_percent = check_label_info(signals_path = signals_path, file_list = test_fold_list,class_mode='Rem-NoneREM',check='All')

    print(train_label)
    print(train_label_percent)
    print(val_label)
    print(val_label_percent)
    print(test_label)
    print(test_label_percent)

def make_spectrogram_image(signals_path,save_path):
    signals = np.load(signals_path)
    plt.specgram(signals[0],Fs=200)
    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(save_path)
    plt.cla()

def func_spectrogram(arg_list):
    signals_path = arg_list[0]
    signals_list = os.listdir(signals_path)
    signals_list.sort()

    for filename in signals_list:
        current_signals = signals_path + filename
        patient_paths = signals_path.split('/')
        # print(patient_paths)
        patient_paths[-3] = 'spectrogram_dataloader'
        save_spectrogram = '/'.join(patient_paths) + filename.split('.npy')[0] + '.png'
        make_spectrogram_image(current_signals,save_spectrogram)

    print('finish : ',signals_path)
def make_spectrogram_img():
    signals_path = '/home/eslab/dataset/seoulDataset/1channel_prefilter_butter_standard_notch/signals_dataloader/'
    spectro_path = '/home/eslab/dataset/seoulDataset/1channel_prefilter_butter_standard_notch/spectrogram_dataloader/'

    file_list = os.listdir(signals_path)
    file_list.sort()
    signals_path = [signals_path + filename + '/' for filename in file_list]
    spectro_path = [spectro_path + filename + '/' for filename in file_list]

    for dirpath in spectro_path:
        os.makedirs(dirpath,exist_ok=True)

    # print(file_list)
    cpu_num = multiprocessing.cpu_count()
    print('cpu_num : ',cpu_num)
    arg_list = []
    for i in range(len(signals_path)):
        arg_list.append([signals_path[i],spectro_path[i]])
    # print(arg_list)
    start = time.time()
    pool = Pool(cpu_num)

    pool.map(func_spectrogram,arg_list)
    pool.close()
    pool.join()

def accuracy_curve():
    logging_filename = '/home/eslab/kdy/git/SleepStaging_servey/log/seoulDataset/Two_Transformer_3class/transformer_two_3class_avg_7_0.00010_Cosine_800_400_Asymetric_0_minmax.txt'
    csv_file = '/home/eslab/b.csv'
    f = open(logging_filename,'r')

    lines = f.readlines()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_acc = []
    for line in lines:
        line_split = line.split(' ')
        if line_split[0] == 'train':
            train_loss.append(float(line_split[-6].split('%')[0]))
            train_acc.append(float(line_split[-1].split('%')[0]))
        elif line_split[0] == 'val':
            val_loss.append(float(line_split[-6].split('%')[0])) 
            val_acc.append(float(line_split[-1].split('%')[0])) 
        elif line_split[0] == 'test':
            test_acc.append(float(line_split[-1].split('%')[0]))
    total_len = len(train_loss)
    fs = open(csv_file,'w',encoding='utf-8')
    wr = csv.writer(fs)
    for i in range(total_len):
        wr.writerow([train_acc[i],val_acc[i],test_acc[i]])
    fs.close()

def notch_filter(signals, fs, w0, Q):
    b, a = signal.iirnotch(w0, Q,fs)
    signals = signal.lfilter(b,a,signals)

    return signals


def check_label_info_withPath( file_list,class_mode='5class',check='All',check_file='.npy'):
    if class_mode =='5class':
        labels = np.zeros(5)
    else:
        labels = np.zeros(3)
    for signals_paths in file_list:
        # print(signals_paths)
        osa = int(signals_paths.split('/')[-2].split('_')[-2])
        signals_list = []
        if check == 'normal':
            if osa == 0:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'mild':
            # print('mild')
            if osa == 1:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'moderate':
            if osa == 2:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        elif check == 'severe':
            if osa == 3:
                signals_list = os.listdir(signals_paths)
                signals_list.sort()
        else: # All
            signals_list = os.listdir(signals_paths)
            signals_list.sort()
        if len(signals_list) != 0:
            for signals_filename in signals_list:
                if class_mode == '5class':
                    labels[int(signals_filename.split(check_file)[0].split('_')[-1])] += 1
                else:
                    current_label = int(signals_filename.split(check_file)[0].split('_')[-1])
                    if current_label == 4:
                        labels[2] += 1
                    elif current_label == 0:
                        labels[0] += 1
                    else:
                        labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent



def interp_1d(arr,short_sample=750,long_sample=6000):
      return np.interp(
    np.arange(0,long_sample),
    np.linspace(0,long_sample,num=short_sample),
    arr)


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)