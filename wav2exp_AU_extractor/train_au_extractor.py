from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import AU_extractor as AU_model
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

import re

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default = "ckpts", type=str)
# parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 6
syncnet_mel_step_size = 16
digits = re.compile(r'(\d+)')

class Dataset(object):
    def __init__(self, root_dir, all_files, split):
        vid = []
        for vid_name in all_files:
            vid.append(os.path.join(root_dir, vid_name))
        self.all_videos = vid


    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))

            if not isfile(frame):
                print(start_id)
                print("declined frame", frame)
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

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

   
    def tokenize(self, filename):
        return tuple(int(token) if match else token
                     for token, match in
                     ((fragment, digits.search(fragment))
                      for fragment in digits.split(filename)))


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = sorted(list(glob(join(vidname, '*.jpg'))))
            if len(img_names) == 0:
                continue
            img_names.sort(key=self.tokenize)
            img_names_split = img_names[:-7]
            img_name = random.choice(img_names_split)
            img_name_dir = dirname(img_name)
            img_name_dir_1 = img_name_dir.split(dirname(img_name_dir))[1]
            img_exp = img_name_dir_1.split("_")

            exp_vec = self.get_exp_vector(img_exp[2])  

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                print(e)
                print("exception occured")
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                print("audio failed with mel shape", mel.shape[0])
                continue

            # H x W x 3 * T
            exp_vec = exp_vec/3
            exp_vec = torch.FloatTensor(exp_vec)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return mel, exp_vec

l1loss = nn.L1Loss()

def train(device, model, train_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    print("We are training")
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        print("Epoch:", global_epoch)
        print("Global step:", global_step)
        for step, (mel, tar_exp_vec) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            
            mel = mel.to(device)
            tar_exp_vec = tar_exp_vec.to(device)

            pred_exp_vec = model(mel)

            loss = l1loss(pred_exp_vec, tar_exp_vec)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % 1000 == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    for step, (mel, tar_exp_vec) in enumerate(test_data_loader):

        model.eval()

        # Transform data to CUDA device
        mel = mel.to(device)

        tar_exp_vec = tar_exp_vec.to(device)

        pred_exp_vec = model(mel)

        loss = l1loss(pred_exp_vec, tar_exp_vec)
        print(loss)
        losses.append(loss.item())

        if step > eval_steps: break

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)

    return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    # checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    all_files = os.listdir(args.data_root)

    # all_files_test = os.listdir(test_path)

    # Dataset and Dataloader setup
    train_dataset = Dataset(args.data_root, all_files, 'train')
    # test_dataset = Dataset(test_path, all_files, 'val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=2)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = AU_model().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    # if checkpoint_path is not None:
        # load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
