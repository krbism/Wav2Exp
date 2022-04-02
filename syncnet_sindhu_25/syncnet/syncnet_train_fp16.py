from os.path import dirname, join, basename, isfile
from tqdm import tqdm

# from models import SyncNet3D_align as SyncNet
from models.syncnet import SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse, subprocess
from hparams import hparams

import re

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='ckpts', required=False, type=str)
# parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('-g', '--n_gpu', type=int, default=1, required=False, help='No of GPUs')
    
args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 25
syncnet_mel_step_size = 100

digits = re.compile(r'(\d+)')

class Dataset(object):
    def __init__(self, root_dir, all_files):
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
                print(frame)
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        start_idx = 4 * start_frame_num

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

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

            img_names = list(glob(join(vidname, '*.jpg')))
            img_names.sort(key=self.tokenize)
            img_names_split = img_names[:-25]
            # print(img_names)
            # if len(img_names) <= 3 * syncnet_T:
            #     continue
            img_name = random.choice(img_names_split)

            chosen = img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                print("none")
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    print(e)
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            # print("Mel: ", mel.shape)                             # 100x80

            if (mel.shape[0] != syncnet_mel_step_size):
                print("audio failed")
                continue

            # start_frame_num = int(16000/25) * (self.get_frame_id(img_name) + 1)
            # cropped_wav = wav[start_frame_num : start_frame_num + 16000]
            # if len(cropped_wav) != 16000: continue

            # out = cv2.VideoWriter('checkpoints/result_{}.avi'.format(idx), 
            #                         cv2.VideoWriter_fourcc(*'DIVX'), 25, hparams.img_size)
            # for xx in window:
            #     out.write(xx.astype(np.uint8))

            # out.release()
            # audio.save_wav(cropped_wav, 'checkpoints/wav_{}.wav'.format(idx), hparams.sample_rate)

            # command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format('checkpoints/wav_{}.wav'.format(idx), 
            #                                                         'checkpoints/result_{}.avi'.format(idx), 
            #                                                         'checkpoints/result_voice_{}_{}.mp4'.format(idx, y.item()))

            # subprocess.call(command, shell=True)
            # exit(0)

            # T x H x W x 3
            x = np.stack(window, axis=0) / 255.
            x = x.transpose(3, 0, 1, 2)
            x = x[:, :, x.shape[2]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # mel = torch.FloatTensor(mel.T)

            return x, mel


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("Correct: ", correct.shape)
    # print("Pred: ", pred.shape)

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

logloss = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=1)

def cosine_loss(a, v):

    print("A: ", a.size()) # B, 512, T
    print("V: ", v.size()) # B, 512, T


    time_size = a.size(2)
    losses = []
    p1s, p5s = [], []
    label = torch.arange(time_size).to(device)
    for i in range(a.size(0)):
        ft_v = v[[i],:,:].transpose(2,0)
        ft_a = a[[i],:,:].transpose(2,0)
        output = cos(ft_v.expand(-1, -1, time_size), ft_a.expand(-1, -1, time_size).transpose(0, 2)) 

        losses.append(logloss(output, label))
        p1, p5 = accuracy(output, label)
        p1s.append(p1.item())
        p5s.append(p5.item())

    loss = sum(losses) / len(losses)
    p1 = sum(p1s) / len(p1s)
    p5 = sum(p5s) / len(p5s)

    return loss, p1, p5

scaler = torch.cuda.amp.GradScaler()


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        running_p1 = 0.
        running_p5 = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            # print("Mel inp: ", mel.size())                  # Bx80x100
            # print("Image: ", x.size())                      # Bx3x25x48x96
            with torch.cuda.amp.autocast():
                a, v = model(mel, x)
                loss, p1, p5 = cosine_loss(a, v)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            running_p1 += p1
            running_p5 += p5

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}, P1: {}, P5: {}'.format(running_loss / (step + 1),
                                                                        running_p1 / (step + 1),
                                                                        running_p5 / (step + 1)))

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    p1s, p5s = [], []
    while 1:
        for step, (x, mel) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)

            with torch.cuda.amp.autocast():
                a, v = model(mel, x)
                loss, p1, p5 = cosine_loss(a, v)

            losses.append(loss.item())
            p1s.append(p1)
            p5s.append(p5)

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        p1 = sum(p1s) / len(p1s)
        p5 = sum(p5s) / len(p5s)
        print('Loss: {}, P1: {}, P5: {}'.format(averaged_loss, p1, p5))

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
    # model.load_state_dict(checkpoint["state_dict"])

    s = checkpoint["state_dict"]
    new_s = {}
    
    for k, v in s.items():
        if args.n_gpu > 1:
            if not k.startswith('module.'):
                new_s['module.'+k] = v
            else:
                new_s[k] = v
        else:
            new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

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

    hparams.syncnet_batch_size = args.n_gpu * hparams.syncnet_batch_size  

    # test_path = "/media/shankar/05a3ed34-f47f-4e90-b99d-dd973f2b86da/Wav2Expression/test"
    test_path = "/home/ubuntu/KRB_Projects/wav2exp/test"
    all_files = os.listdir(args.data_root)

    test_files = os.listdir(test_path)
    
    # Dataset and Dataloader setup
    train_dataset = Dataset(args.data_root, all_files)
    test_dataset = Dataset(test_path, test_files)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    total_batch = len(train_data_loader)
    print("Total train batch: ", total_batch)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet()
    if args.n_gpu > 1:
        print("Using", args.n_gpu, "GPUs for the model!")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU for the model!")
    model.to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    # if checkpoint_path is not None:
        # load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)