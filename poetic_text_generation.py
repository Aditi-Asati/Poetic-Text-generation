import random
import numpy as np
import tensorflow as tf

# to create a neural network later on
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Activation
#LSTM is the recurrent layer with memory
#dense is the hidden layer
#activation is the output layer
from tensorflow.keras.optimizers import RMSprop

#using tf to directly load the Shakespeare file into our script
filepath = tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# text = text[20000:100000]

character = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(character))
index_to_char = dict((i, c) for i, c in enumerate(character))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
  sentences.append(text[i: i+SEQ_LENGTH])
  next_char.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(character)), dtype = np.bool)
y = np.zeros((len(sentences), len(character)), dtype = np.bool)

for i, sentence in enumerate(sentences):
  for j, char in enumerate(sentence):
    x[i, j, char_to_index[char]] = 1
  y[i, char_to_index[next_char[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(character))))
model.add(Dense(len(character)))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=4)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
  start_index = random.randint(0, len(text) - SEQ_LENGTH)
  sentence = text[start_index: start_index + SEQ_LENGTH]
  generated = ""
  generated += sentence

  for i in range(length):

    x = np.zeros((1, SEQ_LENGTH, len(character)), dtype = np.bool)

    for j, char in enumerate(sentence):
      x[0, j, char_to_index[char]] = 1

    x_predictions = model.predict(x, verbose=0)[0]

    next_index = sample(x_predictions, temperature)
    next_char = index_to_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char

  return generated

print("\n---------Temp 1--------")
print(generate_text(500, 1))
print("\n---------Temp 0.8--------")
print(generate_text(500, 0.8))
print("\n---------Temp 0.6--------")
print(generate_text(500, 0.6))
print("\n---------Temp 0.4--------")
print(generate_text(500, 0.4))