import argparse

import music21 as m21
import numpy as np

from os import listdir

N_FEATURES = 2
DATA_DIR = 'data'

def load_midi(filename):
    return m21.converter.parse(filename)

def format_data(midi):

    k = midi.analyze('key')
    i = m21.interval.Interval(k.tonic, m21.pitch.Pitch('C'))
    midi = midi.transpose(i)
    notes = midi.parts.flat.notes
    n0 = m21.pitch.Pitch('C')

    for n in notes:
        interval = m21.interval.Interval(n0, n)

        try:
            print(interval)
        except:
            breakpoint()
        #data.append((interval, n.duration.quarterLength))

    #return data

filenames = [DATA_DIR+'/'+f for f in listdir(DATA_DIR) if f.upper().endswith('MID')]

print(len(filenames))
for filename in filenames[3:4]:
    print(filename)
    midi = load_midi(filename)
    data = format_data(midi)

"""
steps = range(0, len(data), 5)

xs = [data[i0: i1] for i0, i1 in zip(steps, steps[1:])]
ys = [data[i] for i in steps[1:]]

xs = np.array(xs).reshape((len(xs), 5, N_FEATURES))
ys = np.array(ys).reshape((len(xs), 1, N_FEATURES))
"""

def train():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed, Dropout
    from keras.utils import to_categorical

    model = Sequential()

    model.add(LSTM(512, return_sequences=True, input_shape=(None, N_FEATURES)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(N_FEATURES, activation='linear'))
    #print(model.summary(90))

    model.compile(
        loss='mse',
        optimizer='rmsprop'
    )

    model.fit(xs, ys, epochs = 50)
    model.save('models/test.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    if args.train: train()
