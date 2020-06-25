import heapq
import string

import numpy as np
from pickle import load
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

# bilstm = load_model('bilstm_15.h5')
word_based = load_model('wbw_50.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))
# tokenizer1 = load(open('tokenizer1.pkl', 'rb'))
word_index = tokenizer.word_index
# word_index1 = tokenizer1.word_index


letter_based = load_model('lbl_40.h5')
SEQUENCE_LENGTH = 40
text = open("republic.txt", encoding='utf-8').read().lower()
text = text.replace('--', ' ')
text = text.replace('\n', ' ')

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def check_bilstm(texts):
    transformed = []
    masked = []
    for el in texts:
        seq = el.split()
        for i in range(len(seq)):
            tmp = el.split()
            tmp[i] = '[MASKED]'
            masked.append(tmp)
            transformed.append([word_index1[x] for x in tmp])
    transformed = pad_sequences(transformed, 15)
    for i in range(0, len(transformed)):
        el = transformed[i:i+1]
        res = bilstm.predict(el)
        res = res.argsort()[0]
        res = res[-5:]
        tmp = []
        for el in res:
            tmp.append(tokenizer.index_word[el])
        print(masked[i], tmp)

def predict_3_words(texts):
    transformed = []
    for el in texts:
        el = el.lower()
        el = el.split()
        print(el)
        transformed.append([word_index[x] for x in el])
    transformed = pad_sequences(transformed, 50)
    for i in range(0, len(transformed)):
        el = transformed[i:i+1]
        res = word_based.predict(el)
        res = res.argsort()[0]
        res = res[-3:]
        tmp = []
        for el in res:
            tmp.append(tokenizer.index_word[el])
        print(texts[i], tmp)

def predict_words_for_check(texts):
    transformed = []
    transformed.append([word_index[x] for x in texts.lower().split()])
    transformed = pad_sequences(transformed, 50)
    for i in range(0, len(transformed)):
        el = transformed[i:i+1]
        res = word_based.predict(el)
        res = res.argsort()[0]
        res = res[-5:]
        tmp = []
        for el in res:
            tmp.append(tokenizer.index_word[el])
        print(texts, tmp)

def check_words(texts):
    for text in texts:
        # split into tokens by white space
        tokens = text.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        words = []
        for word in tokens:
            words.append(word)
            predict_words_for_check(' '.join(words))


def prepare_input(text):
    if len(text) > SEQUENCE_LENGTH:
        text = text[-40:]
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, (40 - len(text)) + t, char_indices[char]] = 1.
    return x


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completion(text):
    original_text = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = letter_based.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        if len(original_text + completion) + 2 > len(original_text) and\
                (next_char == ' ' or next_char in string.punctuation):
            return completion


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = letter_based.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


def predict_by_letters(text, n=3):
    for el in text:
        print(el)
        print(predict_completions(el, n))
        print()


sentences = [
        "the ",
        "i want to go home",
        "where should we ",
        "he is ",
        "to be ",
        "much aged ",
        "Then it is impossible that God should ever be willing to change; being,\
        as is supposed, the fairest and best that is conceivable, every God\
        remains absolutely and for ever in his own form.",
        "to his house and there we found his brothers lysias and euthydemus and with them thrasymachus\
         the chalcedonian charmantides the paeanian and cleitophon the son of aristonymus there too was \
         cephalus the father of polemarchus whom i had not seen for a long time and i thought him very ",
        "rich and mighty man who had a great opinion of his own power was the first to say that justice\
         is doing good to your friends and harm to your enemies most true he said yes i said but if this\
          definition of justice also breaks down what other can be",
        "let us have no more lies of that sort. Neither must we have mothers\
        under the influence of the poets scaring their children with a bad\
        version of these myths--telling how certain gods, as they say, 'Go about\
        by night in the likeness of so many strangers and in divers forms;' but\
        let them take heed lest they make cowards of their children, and at the\
        same time speak blasphemy against the gods."
    ]

check_words(sentences)
print('///////////////////////////////')
# check_bilstm(sentences)
print('///////////////////////////////')
predict_by_letters(sentences)
