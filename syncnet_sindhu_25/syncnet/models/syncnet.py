import torch
from torch import nn
from torch.nn import functional as F

from .conv import *


class SyncNet_v2(nn.Module):
    def __init__(self):
        super(SyncNet_v2, self).__init__()


        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=2), # T,48,48

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,24,24
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,3,3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0)
        )
        
       
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=1), # (20 x T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),   # (10 x T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(2,1), padding=1), # (5 x T)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # Conv2d(256, 512, kernel_size=5, stride=(3,1), padding=0), # (1 x (T - 4))
            Conv2d(256, 512, kernel_size=5, stride=1, padding=(0,2)), # (1 x T)
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        self.downsample = nn.Sequential(
            Conv1d(512, 512, kernel_size=3, stride=2, padding=1),  # (T/2)
            Conv1d(512, 512, kernel_size=3, stride=2, padding=1)   # (T/4)
        )


    def forward(self, audio_sequences, face_sequences):

        with torch.cuda.amp.autocast():

            # print("Audio seq: ", audio_sequences.size())                            # Bx1x80x100
            # print("Face seq: ", face_sequences.size())                              # Bx3x25x48x96
            
            # --------------------------- Face embedding ------------------------------------#
            face_enc = self.face_encoder(face_sequences)
            # print("Face encoding output: ", face_enc.size())                        # Bx512x25x1x1

            face_embedding = face_enc.squeeze(3).squeeze(3) 
            # print("Face embedding: ", face_embedding.size())                        # Bx512x25

            # -------------------------- Audio embedding -----------------------------------#

            audio_embedding = self.audio_encoder(audio_sequences)
            # print("Audio enc: ", audio_embedding.size())                            # Bx512x1x100

            audio_embedding = audio_embedding.squeeze(2) 

            audio_embedding = self.downsample(audio_embedding)
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x25

            
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
            face_embedding = F.normalize(face_embedding, p=2, dim=1)

            # print("Face embedding: ", face_embedding.size())                        # Bx512x25
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x25


        return audio_embedding, face_embedding


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()


        self.face_encoder = nn.Sequential(
            # Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=(0, 2, 2)), # T - 4,24,48
            Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=2), # T,48,48

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,24,24
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,3,3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0)
        )
        
       
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x 4T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (20 x 2T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),   # (10 x 2T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (5 x T)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            # Conv2d(256, 512, kernel_size=5, stride=(3,1), padding=0), # (1 x (T - 4))
            Conv2d(256, 512, kernel_size=5, stride=1, padding=(0,2)), # (1 x T)
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)


    def forward(self, audio_sequences, face_sequences):

        with torch.cuda.amp.autocast():

            # print("Audio seq: ", audio_sequences.size())                            # Bx1x80x100
            # print("Face seq: ", face_sequences.size())                              # Bx3x25x48x96
            
            # --------------------------- Face embedding ------------------------------------#
            face_enc = self.face_encoder(face_sequences)
            # print("Face encoding output: ", face_enc.size())                        # Bx512x25x1x1

            face_embedding = face_enc.squeeze(3).squeeze(3) 
            # print("Face embedding: ", face_embedding.size())                        # Bx512x25

            # -------------------------- Audio embedding -----------------------------------#

            audio_embedding = self.audio_encoder(audio_sequences)
            # print("Audio enc: ", audio_embedding.size())                            # Bx512x1x25

            audio_embedding = audio_embedding.squeeze(2) 
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x25

            
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
            face_embedding = F.normalize(face_embedding, p=2, dim=1)

            # print("Face embedding: ", face_embedding.size())                        # Bx512x25
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x25


        return audio_embedding, face_embedding

class SyncNet_v0(nn.Module):
    def __init__(self):
        super(SyncNet_v0, self).__init__()

        # self.face_encoder = nn.Sequential(
        #     Conv3d(3, 32, kernel_size=5, stride=(1,1,2), padding=2),             # Bx32x25x48x48
        #     Conv3d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),  

        #     Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),            # Bx64x25x24x24
        #     Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

        #     Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),           # Bx128x25x12x12
        #     Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

        #     Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),          # Bx256x25x6x6
        #     Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

        #     Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1)),          # Bx512x25x3x3
        #     Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            
        #     Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1,3,3), padding=(0, 1, 1)),          # Bx512x25x1x1
        #     Conv3d(512, 512, kernel_size=1, stride=1, padding=0)
        # )

        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=2), # T,48,48

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,24,24
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T,3,3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        
        self.time_upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        # self.audio_encoder = nn.Sequential(
        #     Conv1d(80, 128, kernel_size=3, stride=1, padding=1),
        #     Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        #     Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

        #     Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
        #     Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        #     Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

        #     Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
        #     Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        #     Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

        #     Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
        #     Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1), # (20 x T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),   # (10 x T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1), # (5 x T)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=(5,3), stride=1, padding=(0,1)), # (1 x T)
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, audio_sequences, face_sequences):

        with torch.cuda.amp.autocast():

            # print("Audio seq: ", audio_sequences.size())                            # Bx80x100
            # print("Face seq: ", face_sequences.size())                              # Bx3x25x48x96
            
            # --------------------------- Face embedding ------------------------------------#
            face_enc = self.face_encoder(face_sequences)
            # print("Face encoding output: ", face_enc.size())                        # Bx512x25x1x1

            face_enc = face_enc.squeeze(3).squeeze(3) 
            # print("Face enc reshaped: ", face_enc.size())                           # Bx512x25

            face_embedding = self.time_upsampler(face_enc)
            # print("Face embedding: ", face_embedding.size())                        # Bx512x100

            # -------------------------- Audio embedding -----------------------------------#

            audio_embedding = self.audio_encoder(audio_sequences)
            # print("Audio enc: ", audio_embedding.size())                            # Bx512x1x100

            audio_embedding = audio_embedding.squeeze(2) 
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x100

            
            audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
            face_embedding = F.normalize(face_embedding, p=2, dim=1)

            # print("Face embedding: ", face_embedding.size())                        # Bx512x100
            # print("Audio embedding: ", audio_embedding.size())                      # Bx512x100


        return audio_embedding, face_embedding



####### JOURNAL #####

class SyncNet3D_align(nn.Module):
    def __init__(self):
        super(SyncNet3D_align, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=(0, 2, 2)), # T - 4,24,48

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T - 4,24,24
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T - 4,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T - 4,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T - 4, 3, 3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x 4T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (20 x 2T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),   # (10 x 2T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (5 x T)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=5, stride=1, padding=0), # (1 x (T - 4))
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences):
        # print(face_sequences.size())
        face_embedding = self.face_encoder(face_sequences)  # (B, 512, T - 4, 1, 1)
        audio_embedding = self.audio_encoder(audio_sequences) # (B, 512, 1, T - 4)

        # print(face_embedding.size())
        # print(audio_embedding.size())

        audio_embedding = audio_embedding.squeeze(2) # (B, 512, T - 4)
        face_embedding = face_embedding.squeeze(3).squeeze(3) # (B, 512, T - 4)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

class SyncNet3D_contrastive(nn.Module):
    def __init__(self):
        super(SyncNet3D_contrastive, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(1, 1, 2), padding=(0, 2, 2)), # T - 4,24,48

            Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T - 4,24,24
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T - 4,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T - 4,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T - 4, 3, 3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x 4T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (20 x 2T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),   # (10 x 2T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (5 x T)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=5, stride=1, padding=0), # (1 x (T - 4))
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences):
        # face_sequences: (B, C, T, H, W)
        # audio_sequences: (B, N + 1, 1, H, W)

        assert face_sequences.size(2) == 5

        if len(audio_sequences.size()) == 5:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        # print(face_sequences.size())
        face_embedding = self.face_encoder(face_sequences)  # (B, 512, T - 4, 1, 1)
        audio_embedding = self.audio_encoder(audio_sequences) # (B * (N + 1), 512, 1, T - 4)

        # print(face_embedding.size())
        # print(audio_embedding.size())

        audio_embedding = audio_embedding.squeeze(2).squeeze(2) # (B * (N + 1), 512)
        face_embedding = face_embedding.squeeze(2).squeeze(2).squeeze(2)  # (B, 512)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        if audio_embedding.size(0) != face_embedding.size(0):
            audio_embeddings = torch.split(audio_embedding, face_sequences.size(0), dim=0)
            audio_embedding = torch.stack(audio_embeddings, dim=1)

        return audio_embedding, face_embedding


class SyncNet3D_contrastive_10f(nn.Module):
    def __init__(self):
        super(SyncNet3D_contrastive_10f, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv3d(3, 32, kernel_size=5, stride=(2, 1, 2), padding=(2, 2, 2)), # 5,24,48

            Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)), # 5,24,24
            Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), residual=True),
            Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), residual=True),

            Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),    # T - 4,12,12
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),   # T - 4,6,6
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # T - 4, 3, 3
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=0),
            Conv3d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1), # (40 x 4T)
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (20 x 2T)
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),   # (10 x T)
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (5 x T/2)
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=5, stride=1, padding=0), # (1 x (T - 4))
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences):
        # face_sequences: (B, C, T, H, W)
        # audio_sequences: (B, N + 1, 1, H, W)

        assert face_sequences.size(2) == 10

        if len(audio_sequences.size()) == 5:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        # print(face_sequences.size())
        face_embedding = self.face_encoder(face_sequences)  # (B, 512, T - 4, 1, 1)
        audio_embedding = self.audio_encoder(audio_sequences) # (B * (N + 1), 512, 1, T - 4)

        # print(face_embedding.size())
        # print(audio_embedding.size())

        audio_embedding = audio_embedding.squeeze(2).squeeze(2) # (B * (N + 1), 512)
        face_embedding = face_embedding.squeeze(2).squeeze(2).squeeze(2)  # (B, 512)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        if audio_embedding.size(0) != face_embedding.size(0):
            audio_embeddings = torch.split(audio_embedding, face_sequences.size(0), dim=0)
            audio_embedding = torch.stack(audio_embeddings, dim=1)

        return audio_embedding, face_embedding

class SyncNet_color_2d_contrastive(nn.Module):
    def __init__(self):
        super(SyncNet_color_2d_contrastive, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 2), padding=1),        # 27 x 2T
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(3, 2), padding=1),       # 9 x T
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),      # 3, 3
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        if len(audio_sequences.size()) == 5:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        if audio_embedding.size(0) != face_embedding.size(0):
            audio_embeddings = torch.split(audio_embedding, face_sequences.size(0), dim=0)
            audio_embedding = torch.stack(audio_embeddings, dim=1)

        return audio_embedding, face_embedding
