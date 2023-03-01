# %%
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# speech-silence and voice-unvoiced
BASE_PATH = None


def read_lab(lab_file_name: str):
    """Read lab file
    lab_file_name: str, name of lab file
    Return: list of lists [start_time, end_time, label]
    """
    data = []
    with open(lab_file_name) as f:
        for line in f.readlines():
            data.append(line.split())
    return data

# %%


def load_data(audio_name: str):
    signal, sr = librosa.load(os.path.join(BASE_PATH, f"{audio_name}.wav"))
    lab_data = read_lab(os.path.join(BASE_PATH, f"{audio_name}.lab"))
    timestamp_label = lab_data[:-2]
    t_i = 0
    t_f = signal.shape[0] / sr
    t = np.linspace(t_i, t_f, num=signal.shape[0])
    return signal, sr, t, timestamp_label


# %%



# %%
# def cost(threshold):
class SpeechSlienceDiscriminator:
    def __init__(self, audio_name, signal, sr, t, timestamp_label):
        self.audio_name = audio_name
        self.signal = signal
        self.signal[self.signal == 0] = 1e-5
        self.sr = sr
        self.t = t
        self. timestamp_label = timestamp_label
        self.calc_log_STE()
        self.calc_silent_frame_idx()
        
    def calc_log_STE(self, frame_length=0.02):
        STE = []
        frame_size = int(self.sr * frame_length)
        frames_count = len(self.signal) // frame_size
        frame_edges = []
        for i in range(frames_count):
            startIdx = i*frame_size
            stop_Idx = startIdx + frame_size
            window = np.zeros(self.signal.shape)
            window[startIdx:stop_Idx] = 1
            value = np.sum(np.square(self.signal) * window)
            STE.append(value)
            frame_edges.append(startIdx)
        STE = np.array(STE)
        STE = STE.reshape(-1)
        frame_edges = np.array(frame_edges)
        frame_edges = frame_edges.reshape(-1)
        log_STE = np.log(STE)
        self.log_STE = log_STE
        self.frame_edges = frame_edges
        return log_STE, frame_edges
    
    def calc_silent_frame_idx(self):
        silent_timestamps = list(filter(lambda x: x[2] == "sil", self.timestamp_label))
        silent_timestamps = list(map(lambda x: x[:2], silent_timestamps))
        silent_idx = []
        for timestamp_pair in silent_timestamps:
            start = float(timestamp_pair[0])
            end = float(timestamp_pair[1])
            start_idx = len(self.t[self.t<start])
            end_idx = len(self.t[self.t<end])
            silent_idx.append((start_idx, end_idx))
        silent_frame_idx = []
        for idx_pair in silent_idx:
            frame_size = self.frame_edges[1] - self.frame_edges[0]
            start_idx = int(idx_pair[0]/frame_size)
            end_idx = int(idx_pair[1]/frame_size)
            silent_frame_idx.append((start_idx, end_idx))
        self.silent_frame_idx = silent_frame_idx
        frame_in_silence = np.full(self.log_STE.shape, 0)
        for idx_pair in self.silent_frame_idx:
            frame_in_silence[idx_pair[0]:idx_pair[1]+1] = 1
        self.frame_in_silence = frame_in_silence
        
    def plot_ste(self):
        plt.plot(self.log_STE, color = "yellow")
        threshold = self.log_STE[self.silent_frame_idx[0][1]]
        plt.axhline(y = threshold, color = "red")
        for idx_pair in self.silent_frame_idx:
            plt.axvline(x = idx_pair[0], color = "red")
            plt.axvline(x = idx_pair[1], color = "red") 
        plt.show()
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
        
    def cross_entropy(self, y_hat):
        y = self.frame_in_silence
        y_hat[y_hat == 1] = 0.99999
        loss = -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        return loss.sum()/len(self.log_STE)
        
    
    def logistic_regression(self):
        # clf = LogisticRegression(random_state=0).fit(self.log_STE.reshape(-1, 1), self.frame_in_silence)
        # print(clf.score(self.log_STE.reshape(-1, 1), self.frame_in_silence))
        epoch = 0
        w = np.random.normal(size = (1, ))
        b = 0
        z = w * self.log_STE + b
        y_hat = self.sigmoid(z)
        loss = self.cross_entropy(y_hat)
        pv_w = [0]
        pv_b = [0]
        while (loss > 0.15) or epoch < 1000:
            y = self.frame_in_silence
            dLoss_dw = ((y_hat - y)*self.log_STE)/len(self.log_STE)
            dLoss_db = (y_hat - y)/len(self.log_STE)
            v_w = 0.9 * pv_w[-1] + 1 * dLoss_dw.sum()
            v_b = 0.9 * pv_b[-1] + 1 * dLoss_db.sum()
            w -= v_w
            b -= v_b
            z = w * self.log_STE + b
            y_hat = self.sigmoid(z)
            loss = self.cross_entropy(y_hat)
            pv_w.append(v_w)
            pv_b.append(v_b)
            print(f"{loss=}, {epoch=}")
            epoch +=1
            # print(loss)
        self.w = w
        self.b = b
        return w, b
    
    def predict(self):
        print(self.frame_in_silence)
        z = self.w * self.log_STE + self.b
        y_hat = self.sigmoid(z)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = y_hat.astype(int)
        print(y_hat)
# %%
BASE_PATH = "TinHieuHuanLuyen"
audio_name_train_list = list(filter(lambda x: x.endswith(".wav"), os.listdir(BASE_PATH)))
audio_name_train_list = list(map(lambda x: x[:-4], audio_name_train_list))
signal_list = [0] * len(audio_name_train_list)
sr_list = [0] * len(audio_name_train_list)
t_list = [0] * len(audio_name_train_list)
timestamp_label_list = [0] * len(audio_name_train_list)
signal_frames_list = [0] * len(audio_name_train_list)
frame_size_list = [0] * len(audio_name_train_list)
frames_count_list = [0] * len(audio_name_train_list)
i = 2
signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i] = load_data(
    audio_name_train_list[i]
)
a = SpeechSlienceDiscriminator(audio_name_train_list[i], signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i])
a.logistic_regression()
a.predict()

# %%
