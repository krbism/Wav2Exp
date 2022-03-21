import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class AU_extractor(nn.Module):
    def __init__(self):
        super(AU_extractor, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(8, 10, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(10, 10, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(10, 10, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(10, 12, kernel_size=3, stride=3, padding=1),
            Conv2d(12, 12, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(12, 12, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(12, 14, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(14, 14, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(14, 14, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(14, 17, kernel_size=3, stride=1, padding=0),
            Conv2d(17, 17, kernel_size=1, stride=1, padding=0),)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_sequences): # audio_sequences := (B, dim, T)
        print("audio input", audio_sequences.size())
        audio_embedding = self.audio_encoder(audio_sequences)
        print("audio", audio_embedding.size())

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        audio_embedding = self.sigmoid(audio_embedding)

        return audio_embedding


# if __name__ == '__main__':
#     aud = torch.randn([2,1,80,16])
#     model = AU_extractor()
#     aud_emb = model(aud)
#     print("Audio embedding output", aud_emb)
