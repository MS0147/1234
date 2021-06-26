import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display 
import cv2
from sklearn.model_selection import train_test_split
import torch

def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y

def data_aug(train_x,train_y,times=2):
    res_train_x=[]
    res_train_y=[]
    for idx in range(train_x.shape[0]):
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug=aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)

    res_train_x=np.array(res_train_x)
    res_train_y=np.array(res_train_y)

    return res_train_x,res_train_y

def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[:, j] =x[:,i]
            j += 1
            x_1[:, j] = (x[:, i] + x[:, i + 1]) / 2
            j += 1
        else:
            x_1[:, j] = x[:, i]
            j += 1
    return x_1


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

def ConvSTFT(data, save_name):
    lst = [] #npy로 저장할 데이터들
    length = len(data) #출력할 데이터 개수

    
    for i in range(length):
                 
        #STFT
        D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
        
        #ampiltude로 변환
        magnitude = np.abs(D_highres)
                 
        #amplitude를 db 스케일로 변환
        log_spectrogram = librosa.amplitude_to_db(magnitude)
                 
        #화이트 노이즈 제거
        log_spectrogram = log_spectrogram[:,10:150]
                 
        #128,128로 resize
        log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
        
        lst.append(log_spectrogram)
        if i%50==0:
            print(i,'/',length)

    #npy로 저장 
    lst = np.array(lst)
    lst = lst.reshape(lst.shape[0],1,lst.shape[1],lst.shape[2])
    output_filename = save_name
    print(lst.shape)
    np.save(output_filename, lst)

n_data = np.load('ano0/N_samples.npy')
test_N,test_N_y, train_N,train_N_y = getFloderK(N_samples,opt.0,0)
train_N, val_N, train_N_y, val_N_y = getPercent(train_N, train_N_y, 0.1, 0)

s_data = np.load('ano0/S_samples.npy') 
v_data = np.load('ano0/V_samples.npy') 
f_data = np.load('ano0/F_samples.npy') 
q_data = np.load('ano0/Q_samples.npy')

n_fft_n= 256
win_length_n=64
hp_length_n=2
sr = 360 





#ConvSTFT(n_data,'n_spectrogram')
ConvSTFT(train_N,'train_N')
ConvSTFT(test_N,'test_N')
ConvSTFT(val_N,'val_N')

ConvSTFT(s_data,'s_spectrogram')
ConvSTFT(v_data,'v_spectrogram')
ConvSTFT(f_data,'f_spectrogram')
ConvSTFT(q_data,'q_spectrogram')

