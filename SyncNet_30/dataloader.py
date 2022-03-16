from os.path import dirname, join, basename, isfile
from tqdm import tqdm

import audio

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torchvision.transforms import transforms as tf
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split

import os, random, cv2, argparse
from hparams import hparams, get_image_list



parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')
# parser1.add_argument("--train_path", help="Root folder of the preprocessed MEAD dataset", required=False, type=str)
# parser1.add_argument("--test_path", help="Root folder of the preprocessed MEAD dataset", required=False, type=str)
parser.add_argument("--data_root", help="Root folder of the preprocessed MEAD dataset", required=True, type=str)
# parser.add_argument("--audio_root", help="Root folder of the preprocessed MEAD audio dataset", required=True, type=str)


args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
#Settings for 30FPS (Refer journal for calculation)
syncnet_T = 6
syncnet_mel_step_size = 16

trans = tf.Compose([tf.ToTensor()])

class Dataset(object):
    def __init__(self, all_files, root_dir):
        vid = []
        #vid = os.listdir(args.data_root)
        for vid_name in all_files:
            vid.append(os.path.join(root_dir, vid_name))
        self.all_videos = vid
        print(len(self.all_videos))

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_exp_vector(self, exp):
        content = exp
        if content == "neutral":
            #Neutral: 1
            exp_vec = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
        
        elif content == "happy":
            #Happiness: 6+12
            exp_vec = np.array([0, 0, 0, 0, 3.0, 0, 0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
            
        elif content == "sad":
            #Sadness: 1+4+15
            exp_vec = np.array([3.0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
            
        elif content == "surprised":
            #Surprise: 1+2+5+26
            exp_vec = np.array([3.0, 3.0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
            
        elif content == "fear":
            #Fear: 1+2+4+5+7+20+26
            exp_vec = np.array([3.0, 3.0, 3.0, 3.0, 0, 3.0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0, 3.0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
            
        elif content == "disgust":
            #Disgust: 9+15+16
            exp_vec = np.array([0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0, 3.0, 1.0, 0, 0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = exp_vec
            
        elif content == "angry":
            #Anger:4+5+7+23
            exp_vec = np.array([0, 0, 3.0, 3.0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = content
            
        else:
            #Contempt: 12+14
            exp_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 3.0, 3.0, 0, 0, 0, 0, 0, 0, 0])
            # data[os.path.basename(filepath[:-4]).split("_")[0]] = content
        return exp_vec

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        # print(start_id)
        window_fnames = []
        #Choose the next 5 frames of the Temporal window
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
            

        return window_fnames


    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)
        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def crop_audio_window_mel(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + 16

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 6
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels
    def get_exp_window(self, fnames):
        exp_win = []
        for vidname in fnames:
            temp_gt = vidname.split("/")[-2].split("_")
            exp_vec = self.get_exp_vector(temp_gt[2])
            exp_win.append(exp_vec)
        return exp_win

    def prepare_window(self, window):
    # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname_gt = self.all_videos[idx]
            # print("GT",vidname_gt)
            img_names = list(glob(join(vidname_gt, '*.jpg')))
            #Store all the images of the randomly chosen video in img_names (One video at a time)
            if len(img_names) <= 3 * syncnet_T:
                #print("0 failed")
                continue
                #Condition for minimum num of images in a video
            img_name = random.choice(img_names) #Ground truth img
            ########    
            video_name_gt = vidname_gt.split(dirname(vidname_gt))[1]
            temp_gt = video_name_gt.split("_")
            
            
            idx_ip = random.randint(0, len(self.all_videos) - 1)
            vidname_ip = self.all_videos[idx_ip]
            video_name_ip = vidname_ip.split(dirname(vidname_ip))[1]
            temp_ip = video_name_ip.split("_")
            while temp_ip[0]!=temp_gt[0] or temp_ip[1]!=temp_gt[1]: #Add the condition for not picking the same video again.
                idx_ip = random.randint(0, len(self.all_videos) - 1)
                vidname_ip = self.all_videos[idx_ip]
                video_name_ip = vidname_ip.split(dirname(vidname_ip))[1]
                temp_ip = video_name_ip.split("_")
                

            img_names_ip = list(glob(join(vidname_ip, '*.jpg')))
            wrong_img_name = random.choice(img_names_ip) #Input img (pick this from another experssion of the same angle and actor)
            if len(wrong_img_name) <= 3 * syncnet_T:
                # print("1* failed")
                continue
            ##Window creation
            window_fnames = self.get_window(img_name) #Pass GT to create a window

            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                # print("1 failed")
                continue
            exp_window = self.get_exp_window(window_fnames)
            wrong_exp_window = self.get_exp_window(wrong_window_fnames)
            

            window = self.read_window(window_fnames)
            if window is None:
                # print("2 failed")
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                # print("3 failed")
                continue

            try:
                wavpath = join(vidname_ip, "audio.wav")
                # temp_path = vidname_ip.split("/")[-1]
                # aud_name = temp_path.split("_")
                # aud_path = aud_name[0]+"/"+"audio"+"/"+aud_name[2]+"/"+aud_name[3]+"_"+aud_name[4]+"/"+aud_name[5]+".m4a"
                #wav_path = join(args.audio_root, aud_path)
                
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                print("AUDIO FAILED!!")
                #print(wav_path)
             
                continue

            #We DON'T need mel at the moment. ONLY indiv_mel is fine.

            mel = self.crop_audio_window_mel(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != 16):
                # print("4 failed")
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            #window[:, :, window.shape[2]//2:] = 0. #Not required for our task (Cropping of lower half and concatenating)

            wrong_window = self.prepare_window(wrong_window)
            x = wrong_window.copy()

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # print(mel.size())
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            # print(indiv_mels.size())
            y = torch.FloatTensor(y)

            exp_y = torch.FloatTensor(exp_window)

            exp_x = torch.FloatTensor(wrong_exp_window)
            #a = y[0]
            #exp_vec_gt = torch.FloatTensor(exp_vec_gt)
            # print("ALL PLASSED")

            return x, indiv_mels, exp_x, exp_y, y, mel

if __name__ == "__main__":
    # Dataset and Dataloader setup
    
    # test_dataset = Dataset('val')
    all_files = os.listdir(args.data_root)
    print(len(all_files))
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print("Len of train files", len(train_files))
    print("Len of test files", len(test_files))

    train_dataset = Dataset(train_files, args.data_root)
    test_dataset = Dataset(test_files, args.data_root)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=25, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1, num_workers = hparams.num_workers)

    for i, (x, indiv_mels, exp_x, exp_y, y, mel) in enumerate(train_data_loader):
        print(i)
        print(mel.size())

        