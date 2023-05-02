import os
import torchaudio
import pandas as pd

d = {'name': [],
     'sample_speech': [],
     'sample_rate': []}

directory = '/home/lalex/Munka/PycharmProjects2/BME/thesis/dataset/audio'

for file in os.listdir(directory):
    sample = os.path.join(directory, file)
    SAMPLE_SPEECH, SAMPLE_RATE = torchaudio.load(sample)
    d['name'].append(sample)
    d['sample_speech'].append(SAMPLE_SPEECH)
    d['sample_rate'].append(SAMPLE_RATE)

pd.DataFrame(data=d).to_csv('extraction.csv')
