import string
import numpy as np


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


data = open("republic.txt", encoding='utf-8').read()
print(data[:200])
tokens = clean_doc(data)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

file = open("republic_word_seq_15_masked.txt", 'w')

# organize into sequences of tokens
length = 15
sequences = list()
for i in range(length, len(tokens), 3):
    # select sequence of tokens
    masks = np.random.choice(15, 3, replace=False)
    for j in masks:
        seq = tokens[i-length:i]
        seq.append(seq[j])
        seq[j] = '[MASKED]'
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
        file.write(line)
        file.write('\n')
print('Total Sequences: %d' % len(sequences))
file.close()
