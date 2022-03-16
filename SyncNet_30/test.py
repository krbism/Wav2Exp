import audio
import os 
from os.path import join, basename 
from hparams import hparams

syncnet_mel_step_size = 16

def get_frame_id(frame):
    return int(basename(frame).split('.')[0])

def crop_audio_window(spec, start_frame):
    # num_frames = (T x hop_size * fps) / sample_rate
    start_frame_num = get_frame_id(start_frame)
    start_idx = int(80. * (start_frame_num / float(hparams.fps)))

    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

img_name = "71.jpg"

wavpath = join("/home/ubuntu/KRB_Projects/wav2exp/SyncNet_30/data/M0002_front_surprised_level_1_017", "audio.wav")
wav = audio.load_wav(wavpath, hparams.sample_rate)

orig_mel = audio.melspectrogram(wav).T

mel = crop_audio_window(orig_mel.copy(), img_name)

print("mel size", mel.shape)

