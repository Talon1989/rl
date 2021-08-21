import nltk
import numpy as np
import os
import random
import sys
import tensorflow
from tensorflow import keras


# https://github.com/ajhalthor/Keras_LSTM_Text_Generator/blob/master/Text%20Generator%20(LSTM%20%2B%20Keras).ipynb


#################  SETUP  #################

# nltk.download('book')
directory = 'C:/Users/Fabio/AppData/Roaming/nltk_data/'
corpora_dir = directory + 'corpora/state_union'
file_list = []
for root, _, files in os.walk(corpora_dir):
    for fnames in files:
        file_list.append(os.path.join(root, fnames))
print('Read %d files ...' % len(file_list))
docs = []
for f in file_list:
    with open(f, 'r') as fin:
        try:
            s_form = fin.read().lower().replace('\n', '')
            docs.append(s_form)
        except UnicodeDecodeError:
            pass
text = ' '.join(docs)
print('corpus length: %d' % len(text))

characters = sorted(list(set(text)))
print('Total number of unique characters: %d' % len(characters))
char_indices = dict((c, i) for i, c in enumerate(characters))  # characters to index
indices_char = dict((i, c) for i, c in enumerate(characters))  # index to characters


#################  LSTM PREPROCESSING  #################

maxlen = 40
step = 3
sentences, next_chars = [], []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences: %d' % len(sentences))

print('Vectorization')
x = np.zeros([len(sentences), maxlen, len(characters)], dtype=np.bool)  # tensor (training data)
y = np.zeros([len(sentences), len(characters)], dtype=np.bool)  # label
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1  # populating tensor input
    y[i, char_indices[next_chars[i]]] = 1  # populating y with character after the sequence

#  helper function to sample an index from a proba array
def sample(preds, temp=1.):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#################  LSTM  #################

def build_model():
    hidden_units = 2**7
    model = keras.models.Sequential([
        keras.layers.LSTM(units=hidden_units, input_shape=(maxlen, len(characters))),
        keras.layers.Dense(units=len(characters), activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.RMSprop()
    )
    return model

model = build_model()

def on_epoch_end(epoch, _):
    print('\n----- Generating text after epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [.2, .5, 1., 1.2]:
        print('----- Diversity: %s' % diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print("----- Generating with seed: ' %s ' " % sentence)
        sys.stdout.write(generated)
        for i in range(400):
            x_pred = np.zeros([1, maxlen, len(characters)])
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    model.save_weights('data/lstm/saved_weights.hdf5', overwrite=True)

print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
checkpointer = keras.callbacks.ModelCheckpoint(
    filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True
)

def on_epoch_end(epoch, _):
    print('\n----- Generating text after epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [.2, .5, 1., 1.2]:
        print('----- Diversity: %s' % diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print("----- Generating with seed: ' %s ' " % sentence)
        sys.stdout.write(generated)
        for i in range(400):
            x_pred = np.zeros([1, maxlen, len(characters)])
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    model.save_weights('data/lstm/saved_weights.hdf5', overwrite=True)

model.fit(x, y, batch_size=2**7, epochs=30, callbacks=[print_callback, checkpointer])












































































































































































































































































































































































































































































