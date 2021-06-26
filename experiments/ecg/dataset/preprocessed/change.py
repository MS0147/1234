import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display 
import cv2

n_data = np.load('ano0/N_samples.npy') 
s_data = np.load('ano0/S_samples.npy') 
v_data = np.load('ano0/V_samples.npy') 
f_data = np.load('ano0/F_samples.npy') 
q_data = np.load('ano0/Q_samples.npy')

n_fft_n= 256
win_length_n=64
hp_length_n=2
sr = 360

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

    for i in range(data.shape[0]):
        for j in range(1):
            data[i][j]=normalize(data[i][j][:])
    data=data[:,:1,:]
    
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


##########

ConvSTFT(n_data,'n_spectrogram')
ConvSTFT(s_data,'s_spectrogram')
ConvSTFT(v_data,'v_spectrogram')
ConvSTFT(f_data,'f_spectrogram')
ConvSTFT(q_data,'q_spectrogram')

