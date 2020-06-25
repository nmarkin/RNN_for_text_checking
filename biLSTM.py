import io
import numpy as np
from pickle import dump
from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras_preprocessing.sequence import pad_sequences

file = open("republic_word_seq_15_masked.txt", 'r')
text = file.read()
text = text.split('\n')
file.close()

tokenizer = Tokenizer(filters='\n')
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
vocab_size = len(tokenizer.word_index) + 1

print(tokenizer.index_word)
print()
print(tokenizer.word_index)

# save the tokenizer
dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequences = pad_sequences(sequences, 15 + 1)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
print(X[0],y[0])
# total sequences: 118671
print("sequences created")

embeddings_dict = {}
with io.open('wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    n, d = map(int, f.readline().split())
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

EMBEDDING_DIM = 300
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    elif word != '[masked]':
        # words not found in embedding index will be treated as names.
        # zero vector is saved for masks
        embedding_vector = embeddings_dict.get('name')
        embedding_matrix[i] = embedding_vector


print("embedding matrix created")

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=15,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300)))
model.add(Dense(300, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('bilstm_15.h5')