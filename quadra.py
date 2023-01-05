import math
import os
import numpy as np
import noisereduce 
import librosa
from scipy import signal
import librosa.display 
import matplotlib.pyplot as plt

dataset='test'   # 'train' if in train mode
number = 140  # nums of data in dataset
C0 = 343  # 声速
D = 0.2  # 麦克风间距
NOISE_LEN = 4000  # 噪声样本的长度
SR_MULTIPLIER = 15  # 升采样的倍数

def read_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return sr,y


def freq_plot(y):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    d = np.abs(librosa.stft(y)) ** 2 
    S = librosa.feature.melspectrogram(S=d)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('linear')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('log')

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    
    plt.show()

def wave_plot(f_e,sig):
    t_e = 1 / f_e
    x_tmp = np.arange(0, t_e * len(sig), t_e)
    plt.figure()
    plt.plot(x_tmp, sig)
    plt.axis([0, 2, -0.5, 0.5])
    plt.xlabel('t/s')
    plt.ylabel('amplitude/V')
    plt.show()

def reducenoise(mic1, mic2, noise_len):

    mic1_noise = mic1[0:noise_len]
    mic1_dn = noisereduce.reduce_noise(audio_clip=mic1, noise_clip=mic1_noise)
    mic2_noise = mic2[0:noise_len]
    mic2_dn = noisereduce.reduce_noise(audio_clip=mic2, noise_clip=mic2_noise)
    return mic1_dn, mic2_dn


def upsample(ch1, ch2, orig_sr, target_sr):

    ch1_new = librosa.resample(ch1, orig_sr, target_sr, fix = True, scale = False)
    ch2_new = librosa.resample(ch2, orig_sr, target_sr, fix = True, scale = False)
    return ch1_new, ch2_new


def calcurelevance(ch1, ch2):

    n_sample = len(ch1)
    n_fft = 2 ** math.ceil(math.log2(2 * n_sample - 1))
    CH1 = np.fft.fft(ch1, n_fft)
    CH2 = np.fft.fft(ch2, n_fft)
    G = np.multiply(CH1, np.conj(CH2))
    r = np.fft.fftshift(np.real(np.fft.ifft(G, n_fft)))
    return r


def bandpass_filter(ch1, ch2):

    b,a=signal.butter(8,[0.02,0.3],'bandpass')
    ch1_ft = signal.filtfilt(b,a,ch1)
    ch2_ft = signal.filtfilt(b,a,ch2)
    return ch1_ft, ch2_ft


def estimate(ch1, ch2, sr):

    ch1_ft, ch2_ft = bandpass_filter(ch1, ch2) #带通滤波
    ch1_dn, ch2_dn = reducenoise(ch1_ft, ch2_ft, NOISE_LEN) #降噪
    sr_up = sr * SR_MULTIPLIER 
    ch1_up, ch2_up = upsample(ch1_dn, ch2_dn, sr, sr_up) #升采样
    r = calcurelevance(ch1_up, ch2_up) #计算相关函数

    #计算角度
    n_mid = int(len(r) / 2)
    n_neighbor = int(sr_up * 2 * D / C0)
    delta_n = np.argmax(
        r[n_mid - n_neighbor: n_mid + n_neighbor]) - n_neighbor
    delta_t = delta_n / sr_up
    cos_theta = C0 * delta_t / D

    if cos_theta>1:   # 由于浮点计算精度问题，costheta可能出现大于一的情况
        cos_theta=1
    if cos_theta<-1:
        cos_theta=-1
    theta = math.acos(cos_theta)

    theta_degree = theta / math.pi * 180
    return theta_degree


def main():
    
    print('Processing:\n')
    angles = np.zeros(number)
    angles1 = np.zeros(number)
    angles2 = np.zeros(number)
    for k in range(1, number + 1):

        #read data 
        path1 = os.path.join('./'+dataset+'/', str(k) + '_mic4.wav')
        path2 = os.path.join('./'+dataset+'/', str(k) + '_mic2.wav')
        path3 = os.path.join('./'+dataset+'/', str(k) + '_mic1.wav')
        path4 = os.path.join('./'+dataset+'/', str(k) + '_mic3.wav')

        #sr, ch1= read_audio(path1)
        #wave_plot(sr,ch1)
        
        sr, ch1= read_audio(path1)
        _, ch2= read_audio(path2)
        sr, ch3= read_audio(path3)
        _, ch4= read_audio(path4)

        #calcu angle of two mic duals
        angles1[k - 1] = estimate(ch1, ch2, sr)
        angles2[k - 1] = estimate(ch3, ch4, sr)

        #combine angle of two mic duals
        if(angles1[k - 1]<90):
            if(angles1[k - 1]<45):
                angles[k - 1]=180-angles2[k - 1]
            else:
                if(angles2[k - 1]<90):
                    angles[k - 1]=90+angles1[k - 1]
                else:
                    angles[k - 1]=90-angles1[k - 1]
        else:
            if(angles1[k - 1]>135):
                angles[k - 1]=180+angles2[k - 1]
            else:
                if(angles2[k - 1]<90):
                    angles[k - 1]=90+angles1[k - 1]
                else:
                    angles[k - 1]=450-angles1[k - 1]

        print('{:6d}: Estimated Angle is {:.2f}  degree '.format(
            k, angles[k - 1]))

    #save result
    with open('./'+dataset+'/'+'result.txt','w') as file:
        for k in range(1, number + 1):
            file.write(str(angles[k-1]))
            file.write('\n')
        file.close()


if __name__ == '__main__':
    main()
