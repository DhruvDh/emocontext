# Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, RNN, Lambda
from keras import optimizers
from keras.models import load_model
from keras.metrics import categorical_accuracy
from sklearn.utils import class_weight
import pickle
from tqdm import tqdm
import json
import argparse
import os
import regex as re
import io
import sys
from keras.models import Model
import tensorflow as tf
import tensorflow_hub as hub


def normalize(s):
    """
    Given a text, cleans and normalizes it.
    """

    eyes = r"[8:=;-]"
    nose = r"['`\-._]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, s, flags=FLAGS)

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "{} ".format(hashtag_body)
        else:
            result = " ".join(
                [""] + re.split(r"(?=[A-Z])", hashtag_body, flags=re.MULTILINE | re.DOTALL))
        return result

    FLAGS = re.MULTILINE | re.DOTALL

    smile = [">:]", ":-)", ":)", ":o)", ":]", ":3",
             ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", "^_^", "(^.^)", "^.^", "^ω^", "(^○^", "(^o^", ":)<", ":)‑", "3:)"]
    laugh = [">:D", ":-D", ":D", "8-D", "x-D", "X-D", "=-D",
             "=D", "=-3", "8-)", '8‑d', "=3", ":)o", "(^�^", "(≧∇≦"]
    sad = [">:[", ":-(", ":(",  ":-c", ":c", ":-<", ":-[",
           ":[", ":{", ">.>", "<.<", ">.<"]
    wink = [">;]", ";-)", ";)", "*-)", "*)", ";-]", ";]", ";D", ";^)", ':^*']
    tongue = [">:P", ":-P", ":P", "X-P", "x-p", ":-p",
              ":p", "=p", ":-Þ", ":Þ", ":-b", ":b", "=p", "=P"]
    surprise = [">:o", ">:O", ":-O", ":O", "°o°",
                "°O°", ":O", "o_O", "o.O", "8-0"]
    annoyed = [">:\\", ">:/", ":-/", ":-.", ":\\",
               "=/", "=\\", ":S", '://', ":'", ":x", "t_t", ":/'"]
    cry = [":'(", ";'("]

    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)

    # normalising emojis
    for emoji in smile:
        s = s.replace(emoji.lower(), ':)')

    for emoji in laugh:
        s = s.replace(emoji.lower(), ':d')

    for emoji in sad:
        s = s.replace(emoji.lower(), ':(')

    for emoji in wink:
        s = s.replace(emoji.lower(), ';)')

    for emoji in tongue:
        s = s.replace(emoji.lower(), ':p')

    for emoji in annoyed:
        s = s.replace(emoji.lower(), ':/')

    for emoji in surprise:
        s = s.replace(emoji.lower(), ':o')

    s = s.replace(" :0 ", " :o ")

    for emoji in cry:
        s = s.replace(emoji.lower(), ':(')

    # Isolate punctuation
    # s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,]+)', r' \1 ', s)
    # dont wanna mess with '
    # s = re.sub(r'([\']+)', r' \1 ', s)
    s = re.sub(r'([\"]+)', r' \1 ', s)
    s = re.sub(r'([\.]+)', r' \1 ', s)
    s = re.sub(r'(\(]+)', r' \1 ', s)
    s = re.sub(r'([\)]+)', r' \1 ', s)
    s = re.sub(r'([\!]+)', r' \1 ', s)
    s = re.sub(r'([\?]+)', r' \1 ', s)
    s = re.sub(r'([\-]+)', r' \1 ', s)
    s = re.sub(r'([\\]+)', r' \1 ', s)
    s = re.sub(r'([\/]+)', r' \1 ', s)
    s = re.sub(r'([\,]+)', r' \1 ', s)

    # Isolate emojis
    s = re.sub('([\U00010000-\U0010ffff]+)', r' \1 ', s, flags=re.UNICODE)

    # dealing with hashtags
    s = re_sub(r"#\S+", hashtag)

    # dealing with @user mentions
    s = re_sub(r"@\w+", "<user>")

    # text emojis
    s = re_sub(r"{}[\s]*{}[\s]*[)dD]+|[)dD]+[\s]*{}[\s]*{}".format(eyes,
                                                                   nose, nose, eyes), " :) ")
    s = re_sub(r"{}[\s]*{}[\s]*p+".format(eyes, nose), " :p ")
    s = re_sub(
        r"{}[\s]*{}[\s]*\(+|\)+[\s]*{}[\s]*{}".format(eyes, nose, nose, eyes), " :( ")
    s = re_sub(r"{}[\s]*{}[\s]*[\/|l*]".format(eyes, nose), " :/ ")

    # Remove some special characters
    # s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    # s = s.replace('&', ' and ')
    # s = s.replace('@', ' at ')
    # s = s.replace('0', ' zero ')
    # s = s.replace('1', ' one ')
    # s = s.replace('2', ' two ')
    # s = s.replace('3', ' three ')
    # s = s.replace('4', ' four ')
    # s = s.replace('5', ' five ')
    # s = s.replace('6', ' six ')
    # s = s.replace('7', ' seven ')
    # s = s.replace('8', ' eight ')
    # s = s.replace('9', ' nine ')
    return s.strip()


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    u1 = []
    u2 = []
    u3 = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            # repeatedChars = ['.', '?', '!', ',']
            # for c in repeatedChars:
            #     lineSplit = line.split(c)
            #     while True:
            #         try:
            #             lineSplit.remove('')
            #         except:
            #             break
            #     cSpace = ' ' + c + ' '
            #     line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            line[1] = '<S> ' + normalize(line[1]) + ' </S>'
            line[2] = '<S> ' + normalize(line[2]) + ' </S>'
            line[3] = '<S> ' + normalize(line[3]) + ' </S>'

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' '.join(line[1:4])

            u1.append(line[1])
            u2.append(line[2])
            u3.append(line[3])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv)

    if mode == "train":
        return indices, conversations, labels, u1, u2, u3
    else:
        return indices, conversations, u1, u2, u3


trainDataPath = "data/train.txt"
testDataPath = "data/devwithoutlabels.txt"

solutionPath = "/"

vectorsDir = "vectors/"
vectorName = "vectors.txt"

NUM_CLASSES = 4
MAX_NB_WORDS = None

MAX_SEQUENCE_LENGTH = 24
EMBEDDING_DIM = 300
CUSTOM_EMBEDDING_DIM = 100

BATCH_SIZE = 128
LSTM_DIM = 1024
DROPOUT = 0.5
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

### --- Processing data  --- ###
print("Processing training data...")
trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(
    trainDataPath, mode="train")

class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(labels), labels)

print("Processing test data...")
testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(
    testDataPath, mode="test")

np.random.shuffle(trainIndices)

u1_train = np.array(u1_train)[trainIndices]
u2_train = np.array(u2_train)[trainIndices]
u3_train = np.array(u3_train)[trainIndices]

labels = np.array(labels)[trainIndices]

### --- Making model  --- ###
print("Building Model...")
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


turn1 = Input(shape=(1,), dtype=tf.string, name='main_input1')
turn2 = Input(shape=(1,), dtype=tf.string, name='main_input2')
turn3 = Input(shape=(1,), dtype=tf.string, name='main_input3')

turn1_embeddings = elmo(tf.squeeze(tf.cast(turn1, tf.string)),
                        signature="default", as_dict=True)["elmo"]
turn2_embeddings = elmo(tf.squeeze(tf.cast(turn2, tf.string)),
                        signature="default", as_dict=True)["elmo"]
turn3_embeddings = elmo(tf.squeeze(tf.cast(turn3, tf.string)),
                        signature="default", as_dict=True)["elmo"]

lstm_turn1 = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))(turn1_embeddings)
lstm_turn2 = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))(turn2_embeddings)
lstm_turn3 = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))(turn3_embeddings)

inp = Concatenate(axis=-1)([lstm_turn1, lstm_turn2, lstm_turn3])
inp = Reshape((3, 2*LSTM_DIM, ))(inp)

final = LSTM(LSTM_DIM, dropout=DROPOUT)(inp)

out = Dense(NUM_CLASSES, activation='softmax')(final)

adam = optimizers.adam(lr=LEARNING_RATE)
model = Model([turn1, turn2, turn3], out)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
print(model.summary())


### --- Training Model  --- ###
print("Training Model...")
model.fit([u1_train, u2_train, u3_train], labels, epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE, class_weight=class_weights)

model.save(os.path.join('models', 'EP%d_LR%de-5_LDim%d_BS%d.h5' %
                        (NUM_EPOCHS, int(LEARNING_RATE*(10**4)), LSTM_DIM, BATCH_SIZE)))


print("Creating solution file...")
predictions = model.predict(
    [u1_test, u2_test, u3_test], batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)

with io.open(solutionPath, "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    with io.open(testDataPath, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            out.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(label2emotion[predictions[lineNum]] + '\n')
print("Completed. Model parameters: ")
print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
      % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
