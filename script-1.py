# %%
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import os

import librosa
import numpy as np

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


def get_closest(arr, values):
    """Get closest value in an sorted array
    arr: np.ndarray
    values: List of values to find
    Return: List of closest values to input values
    """
    arr = np.array(arr)
    values = np.array(values, dtype=np.float64)
    idx = np.searchsorted(arr, values)
    idx = np.array(idx)
    idx[arr[idx] - values > np.diff(arr).mean() * 0.5] -= 1
    return arr[idx]


def get_closest_idx(arr, values):
    """Get closest index in an sorted array
    arr: np.ndarray
    values: List of values to find
    Return: List of closest index to input values
    """
    arr = np.array(arr)
    values = np.array(values, dtype=np.float64)
    idx = np.searchsorted(arr, values, side="left")
    idx = np.array(idx)
    return idx


def array_norm(arr: np.ndarray):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    return (arr - min_arr) / (max_arr - min_arr)


def array_norm_by_threshold(arr: np.ndarray, threshold):
    """Normalize given function by threshold
    arr: np.ndarray
    T: threshold
    """
    min_arr = min(arr)
    max_arr = max(arr)
    return np.where(
        arr >= threshold,
        (arr - threshold) / (max_arr - threshold),
        (arr - threshold) / (threshold - min_arr),
    )


# %%


def load_data(audio_name: str):
    signal, sr = librosa.load(os.path.join(BASE_PATH, f"{audio_name}.wav"))
    lab_data = read_lab(os.path.join(BASE_PATH, f"{audio_name}.lab"))
    timestamp_label = lab_data[:-2]
    t_i = 0
    t_f = signal.shape[0] / sr
    t = np.linspace(t_i, t_f, num=signal.shape[0])
    return signal, sr, t, timestamp_label


# def separate_frames(signal, sr, frame_length=0.02):
#     """Separate signal into frames
#     signal: np.ndarray
#     sr: sampling rate
#     frame_length: length of frame
#     Return: Array of frames
#     """
#     frame_size = int(sr * frame_length)
#     frame_count = len(signal) // frame_size
#     signal_frames = []
#     for i in range(frame_count):
#         startIdx = k*frame_size
#         stopIdx = startIdx + frame_size
#         window = np.zeros(signal.shape)
#         window[]
#     return np.array(signal_frames), frame_size, frame_count


def calc_STE(signal, sr, frame_length=0.02):
    """Calculate STE
    signal_frames: Array of frames
    Return: Array of STE for each frame
    """
    STE = []
    frame_size = int(sr*frame_length)
    frames_count = len(signal) // frame_size
    frame_edges = []
    for i in range(frames_count):
        startIdx = i*frame_size
        stop_Idx = startIdx + frame_size
        window = np.zeros(signal.shape)
        window[startIdx:stop_Idx] = 1
        value = np.sum(np.square(signal) * window)
        STE.append(value)
        frame_edges.append(startIdx)
    STE = np.array(STE)
    STE = STE.reshape(-1)
    frame_edges = np.array(frame_edges)
    frame_edges = frame_edges.reshape(-1)
    return STE, frame_edges

def separate_sp_sl(STE, timestamp_label, t):
    STE_sp = np.array([])
    STE_sl = np.array([])
    for line in timestamp_label:
        if line[2] == "sp":
            try:
                idx1 = int(get_closest_idx(t, line[0]))
                idx2 = int(get_closest_idx(t, line[1]))
                STE_sp = np.append(STE_sp, STE[idx1:idx2])
            except:
                print(line)
        if line[2] == "sl":
            try:
                idx1 = int(get_closest_idx(t, line[0]))
                idx2 = int(get_closest_idx(t, line[1]))
                STE_sl = np.append(STE_sl, STE[idx1:idx2])
            except:
                print(line)
    return np.array(STE_sp), np.array(STE_sl)


# %%
BASE_PATH = "TinHieuHuanLuyen"
audio_name_list = list(filter(lambda x: x.endswith(".wav"), os.listdir(BASE_PATH)))
audio_name_list = list(map(lambda x: x[:-4], audio_name_list))
signal_list = [0] * len(audio_name_list)
sr_list = [0] * len(audio_name_list)
t_list = [0] * len(audio_name_list)
timestamp_label_list = [0] * len(audio_name_list)
signal_frames_list = [0] * len(audio_name_list)
frame_size_list = [0] * len(audio_name_list)
frames_count_list = [0] * len(audio_name_list)
STE_list = [0] * len(audio_name_list)
STE_speech_list = [0] * len(audio_name_list)
STE_silence_list = [0] * len(audio_name_list)
T_STE_list = [0] * len(audio_name_list)
# for audio in audio_name_list:
#     signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i] = load_data(
#             audio_name_list[i])


# %%
i = 0
signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i] = load_data(
    audio_name_list[i]
)

# %%
signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i]

# %%
t_list[0][-1]


# %%
audio_name_list
import matplotlib.pyplot as plt

# %%
plt.plot(t_list[0], signal_list[0])
# %%
ste, frame_edges = calc_STE(signal_list[0], sr_list[0], 0.02)
plt.plot(ste)

# %%
