import os
import torchaudio
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


class Extraction:
    def __init__(self, dir, annotation_file, dimension, transform=None):
        self.data_dir = dir
        self.data_label = pd.read_csv(annotation_file)
        self.transform = transform
        self.item = []
        self.path = None
        self.speeches = []
        self.dimension = dimension
        if self.transform in ['spectogram-raw', 'spectogram-power']:
            self.n_fft = 1024
            self.win_length = None
            self.hop_length = 512
        elif self.transform == 'mel-spectrogram':
            self.n_fft = 1024
            self.win_length = None
            self.hop_length = 512
            self.n_mels = 128
        elif self.transform == 'mfcc':
            self.n_fft = 2048
            self.win_length = None
            self.hop_length = 512
            self.n_mels = 256
            self.n_mfcc = 256
        elif self.transform == 'lfcc':
            self.n_fft = 2048
            self.win_length = None
            self.hop_length = 512
            self.n_lfcc = 256

    def __getitem__(self, index):
        """returns feature vector which files names and other information."""
        self.path = os.path.join(self.data_dir, self.data_label.iloc[index, 0])
        speech = torchaudio.load(self.path)
        if self.transform:
            speech = self.extract_feature(speech, self.transform)
        self.speeches.append(speech)
        new_item = {'path': self.path,
                    'speech_waveform': speech[0],
                    'speech_rate': speech[1],
                    'label': self.label,
                    'transform': self.transform}
        self.item.append(new_item)

        return new_item

    def __len__(self):
        """returns the length of the dataset"""
        return len(self.data_label)

    def extract_feature(self, file, feature_type):
        """
        based of parameter feature type will return that feature for the specific file.
        file: path to the file
        feature_type: [waveform, spectogram-raw, spectogram-power, mel-spectogram, mfcc, lfcc]
        :return extracted feature
        """
        sample_waveform, sample_rate = torchaudio.load(file)
        if feature_type == 'waveform':
            return sample_waveform
        elif feature_type == 'spectogram-raw':
            spectogram = T.Spectrogram(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=True,
                pad_mode="reflect",
                power=None)
            return spectogram(sample_waveform)
        elif feature_type == 'spectogram-power':
            spectogram_power = T.Spectrogram(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0)
            return spectogram_power(sample_waveform)
        elif feature_type == 'mel-spectogram':
            mel = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                onesided=True,
                n_mels=self.n_mels,
                mel_scale="htk")
            return mel(sample_waveform)
        elif feature_type == 'mfcc':
            mfcc = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    "n_fft": self.n_fft,
                    "n_mels": self.n_mels,
                    "hop_length": self.hop_length,
                    "mel_scale": "htk",
                })
            return mfcc(sample_waveform)
        elif feature_type == 'lfcc':
            lfcc = T.LFCC(
                sample_rate=sample_rate,
                n_lfcc=self.n_lfcc,
                speckwargs={
                    "n_fft": self.n_fft,
                    "win_length": self.win_length,
                    "hop_length": self.hop_length,
                })
            return lfcc(sample_waveform)
        else:
            print('Feature-type not known or not implemented.')

    def get_file_label(self, index):
        """returns label of the file"""
        return self.data_label.iloc[index, 0]

    def get_file_grade(self):
        """returns the severity level of the speech sample"""
        ...

    def normalizer(self, p=2.0, dim=1, eps=1e-12, out=None):
        """normalizes the feature vector"""
        for n in len(self.item):
            self.item[n]['speech_waveform'] = torch.nn.functional.normalize(self.item[n]['speech_waveform'], p, dim, eps, out)

    def length_fixer(self):
        """Make all files have the same length either by padding for truncating"""
        for n in len(self.speeches):
            if len(self.speeches[n][0]) < self.dimension:
                self.speeches[n][0] = F.pad(self.speeches[n][0],
                                            (self.dimension - self.speeches[n][0].size(1), 0))
            elif len(self.speeches[n][0]) < self.dimension:
                ...