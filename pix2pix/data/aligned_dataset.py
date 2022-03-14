import os
import torch
from os.path import dirname, join, basename, isfile
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

import data.hparams
from data.hparams import hparams as hparams

import data.audio as audio
import warnings
warnings.filterwarnings("ignore")



class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + 5

        return spec[start_idx : end_idx, :]

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + 1):
            m = self.crop_audio_window(spec, i - 2)
            # print("m shape", m.shape)
            if m.shape[0] != 5:
                return None
            mels.append(m.T)


        mels = np.asarray(mels)
        return mels

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

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        while(1):
            index_gt = random.randint(0, len(self.AB_paths) - 1)
            B_path = self.AB_paths[index_gt]
            B_name = dirname(B_path)
            video_name_gt = B_name.split(dirname(B_name))[1]
            temp_gt = video_name_gt.split("_")
            B = Image.open(B_path).convert('RGB')

            index_ip = random.randint(0, len(self.AB_paths) - 1)
            A_path = self.AB_paths[index_ip]
            A_name = dirname(A_path)
            video_name_ip = A_name.split(dirname(A_name))[1]
            temp_ip = video_name_ip.split("_")
            # print("temp_ip", temp_ip)
            while temp_ip[0]!=temp_gt[0] or temp_ip[1]!=temp_gt[1]: #Add the condition for not picking the same video again.
                index_ip = random.randint(0, len(self.AB_paths) - 1)
                A_path = self.AB_paths[index_ip]
                A_name = dirname(A_path)
                video_name_ip = A_name.split(dirname(A_name))[1]
                temp_ip = video_name_ip.split("_")

            exp_vec_ip = self.get_exp_vector(temp_ip[2])
            exp_vec_gt = self.get_exp_vector(temp_gt[2])
            
            A = Image.open(A_path).convert('RGB')

            wavpath = join(B_name, "audio.wav")
            wav = audio.load_wav(wavpath, hparams.sample_rate)
            orig_mel = audio.melspectrogram(wav).T
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), B_path)
            if indiv_mels is None:
                continue
            # print("indicv", indiv_mels.shape)
            indiv_mels = torch.FloatTensor(indiv_mels)

            # print("indicv", indiv_mels.size())

            # apply the same transform to both A and B
            transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

            A = A_transform(A)
            B = B_transform(B)

            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'audio':indiv_mels, 'exp_x':exp_vec_ip, 'exp_y':exp_vec_gt}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
