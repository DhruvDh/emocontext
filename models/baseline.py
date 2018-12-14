# Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, RNN, add
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

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = "../data/train.txt"
testDataPath = "../data/devwithoutlabels.txt"
# Output file that will be generated. This file can be directly submitted.
solutionPath = "soln/"
# Path to directory where GloVe file is saved.
gloveDir = "_files"
vectorName = "_files/vectors.txt"

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
# To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_NB_WORDS = None
# All sentences having lesser number of words than this will be padded
MAX_SEQUENCE_LENGTH = None
EMBEDDING_DIM = None               # The dimension of the word embeddings
CUSTOM_EMBEDDING_DIM = None
# The batch size to be chosen for training the model.
BATCH_SIZE = None
# The dimension of the representations learnt by the LSTM model
LSTM_DIM = None
# Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
DROPOUT = None
NUM_EPOCHS = None                  # Number of epochs to train a model for


label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def normalize(s):
    """
    Given a text, cleans and normalizes it.
    """

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

    smile = [">:]", 'B-)', ":-)", ":)", ":o)", ":]", " :3 ", "B)", ':-)', '(:', '(^・^)', '(:'
             ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", "^_^", "(^.^)", "^.^", "^ω^", "(^○^", "(^○^)", "(^o^", "(^o^)", ":)<", ":)‑", "3:)"]
    laugh = [">:D", ":-D", ":D", "8-D", "x-D", "X-D", "=-D",
             "=D", "=-3", "8-)", '8‑d', "=3", ":)o", "(^�^", "(^�^)", "(≧∇≦", '(≧∇≦)']
    sad = [">:[", ":-(", ":(",  ":-c", ":c", ":-<", ":-[",
           ":[", ":{", ">.>", "<.<", ">.<", ';(', '):']
    wink = [">;]", ";-)", ";)", "*-)", "*)", ";-]", ':*', ':-*'
            ";]", ";D", ";^)", ':^*', ';‑ )', ';")', ';")*']
    tongue = [">:P", ":-P", ":P", "X-P", "x-p", ":-p", 'xd', 't.t', "t_t"
              ":p", "=p", ":-Þ", ":Þ", ":-b", ":b", "=p", "=P"]
    surprise = [">:o", ">:O", ":-O", ":O", "°o°",
                "°O°", ":O", "o_O", "o.O", "8-0", 'D:']
    annoyed = [">:\\\\", '>:\\', ">:/", ":-/", ":-.", ':\\\\', ':\\', ':@', '-_-', '//:'
               "=/", "=\\\\", ":S", '://', ":x", ":/'", '=\\']
    cry = [":'(", ";'("]

    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)

    # normalising emojis
    for emoji in sad:
        s = s.replace(emoji.lower(), ' :( ')

    for emoji in smile:
        s = s.replace(emoji.lower(), ' :) ')

    for emoji in laugh:
        s = s.replace(emoji.lower(), ' :d ')

    for emoji in wink:
        s = s.replace(emoji.lower(), ' ;) ')

    for emoji in tongue:
        s = s.replace(emoji.lower(), ' :p ')

    for emoji in annoyed:
        s = s.replace(emoji.lower(), ' :/ ')

    for emoji in surprise:
        s = s.replace(emoji.lower(), ' :o ')

    s = s.replace(" :0 ", " :o ")

    for emoji in cry:
        s = s.replace(emoji.lower(), ':(')

    # Isolate punctuation
    # s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,]+)', r' \1 ', s)
    # dont wanna mess with '
    # s = re.sub(r'([\']+)', r' \1 ', s)
    s = re.sub(r'([\"]+)', r' \1 ', s)
    s = re.sub(r'([\.]+)', r' \1 ', s)
    s = re.sub(r'([\(]+)', r' \1 ', s)
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
    s = re_sub(r"(\s*:\s*)+(\s*\)\s*)+", " :) ")
    s = re_sub(r"(\s*;\s*)+(\s*\)\s*)+", " ;) ")
    s = re_sub(r"(\s*:\s*)+(\s*p\s*)+", " :p ")
    s = re_sub(r"(\s*:\s*)+(\s*\(\s*)+", " :( ")
    s = re_sub(r"(\s*:\s*)+(\s*\/\s*)+", " :/ ")
    s = re_sub(r"(\s*:\s*)+(\s*d\s*)+", " :d ")
    s = re_sub(r"(\s*:\s*)+(\s*o\s*)+", " :o ")

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
                line = line.strip().split('\t')
                line[1] = normalize(line[1])
                line[2] = normalize(line[2])
                line[3] = normalize(line[3])

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


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(
        np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision +
                                         recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" %
              (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision +
                                                    macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
        macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" %
          (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision +
                                                    microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
        accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    # if vectorName.startswith('glove'):
    #     embeddingsIndex = {}
    # #     Load the embedding vectors from ther GloVe file
    #     with io.open(os.path.join(gloveDir, vectorName), encoding="utf8") as f:
    #         for line in f:
    #             values = line.split(' ')
    #            # print(values)
    #             word = values[0]
    #             embeddingVector = np.array([float(val) for val in values[1:]])
    #             embeddingsIndex[word] = embeddingVector

    #     print('Found %s word vectors.' % len(embeddingsIndex))
    #     with open(os.path.join(gloveDir, vectorName+'_index.pickle'), 'wb') as handle:
    #         pickle.dump(embeddingsIndex, handle,
    #                     protocol=pickle.HIGHEST_PROTOCOL)

    #     # Minimum word index of any word is 1.
    #     embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    #     for word, i in wordIndex.items():
    #         embeddingVector = embeddingsIndex.get(word)
    #         if embeddingVector is not None:
    #             # words not found in embedding index will be all-zeros.
    #             embeddingMatrix[i] = embeddingVector

    #     with open(os.path.join(gloveDir, vectorName+'_matrix.pickle'), 'wb') as handle:
    #         pickle.dump(embeddingMatrix, handle,
    #                     protocol=pickle.HIGHEST_PROTOCOL)
    #     return embeddingMatrix
    # else:
    with io.open(os.path.join(gloveDir, vectorName), 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = data.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            print('no vector for ', word)

    return embeddingMatrix


def getCustomEmbeddingMatrix(wordIndex):
    with io.open(os.path.join(gloveDir,'custom_vectors_V2.5_2500.txt'), 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, CUSTOM_EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = data.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            print('no custom vector for ', word)

    return embeddingMatrix


def buildModel(embeddingMatrix, customEmbeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    x2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    x3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=True)

    customEmbeddingLayer = Embedding(customEmbeddingMatrix.shape[0],
                                     CUSTOM_EMBEDDING_DIM,
                                     weights=[customEmbeddingMatrix],
                                     input_length=MAX_SEQUENCE_LENGTH,
                                     trainable=True)

    _emb1 = embeddingLayer(x1)
    _emb2 = embeddingLayer(x2)
    _emb3 = embeddingLayer(x3)

    c_emb1 = customEmbeddingLayer(x1)
    c_emb2 = customEmbeddingLayer(x2)
    c_emb3 = customEmbeddingLayer(x3)

    _lstm1 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True))
    _lstm2 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True))
    _lstm3 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True))

    __lstm1 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT))
    __lstm2 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT))
    __lstm3 = Bidirectional(
        LSTM(LSTM_DIM, dropout=DROPOUT))

    emb1 = Concatenate(axis=-1)([_emb1, c_emb1])
    emb2 = Concatenate(axis=-1)([_emb2, c_emb2])
    emb3 = Concatenate(axis=-1)([_emb3, c_emb3])

    lstm1 = __lstm1(_lstm1(emb1))
    lstm2 = __lstm2(_lstm2(emb2))
    lstm3 = __lstm3(_lstm3(emb3))

    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    # context = Dense(LSTM_DIM)(Concatenate(axis=-1)([lstm1, lstm2]))
    # text = Dense(LSTM_DIM)(lstm3)

    inp = Reshape((3, 2*LSTM_DIM, ))(inp)

    lstm_up = LSTM(LSTM_DIM, dropout=DROPOUT)

    out = lstm_up(inp)

    # out = Dense(NUM_CLASSES)(Concatenate(axis=-1)([text, context]))
    out = Dense(NUM_CLASSES, activation='softmax')(out)

    adam = optimizers.adam(lr=LEARNING_RATE, amsgrad=True, epsilon=1e-8)
    model = Model([x1, x2, x3], out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    print(model.summary())
    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument(
        '-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, CUSTOM_EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    CUSTOM_EMBEDDING_DIM = config["custom_embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(
        trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)

    print(trainTexts[0])

    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(labels), labels)
    print(class_weights)

    print("Processing test data...")
    testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(
        testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='')
    tokenizer.fit_on_texts(u1_train+u2_train+u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(
        u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(
        u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    words = [word for word in wordIndex.keys()]
    with open('vectors\\words.txt', 'w', encoding='utf8') as f:
        for item in words:
            f.write("%s\n" % item)
    print("Wrote words to vectors/words.txt...")

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    customEmbeddingMatrix = getCustomEmbeddingMatrix(wordIndex)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(trainIndices)

    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]

    labels = labels[trainIndices]

    # Perform k-fold cross validation
    # metrics = {"accuracy": [],
    #            "microPrecision": [],
    #            "microRecall": [],
    #            "microF1": []}

    # print("Starting k-fold cross validation...")
    # print('-'*40)
    # print("Building model...")
    # model = buildModel(embeddingMatrix)
    # model.fit([u1_data,u2_data,u3_data], labels,
    #              validation_split=0.1,
    #              epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    # accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
    # metrics["accuracy"].append(accuracy)
    # metrics["microPrecision"].append(microPrecision)
    # metrics["microRecall"].append(microRecall)
    # metrics["microF1"].append(microF1)

    # print("\n============= Metrics =================")
    # print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
    # print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
    # print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
    # print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

    print("\n======================================")

    print("Retraining model on entire data to create solution file")
    model = buildModel(embeddingMatrix, customEmbeddingMatrix)

    model.fit([u1_data, u2_data, u3_data], labels,
              epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights)
    model.save(os.path.join('models', 'EP%d_LR%de-5_LDim%d_BS%d.h5' %
                            (NUM_EPOCHS, int(LEARNING_RATE*(10**4)), LSTM_DIM, BATCH_SIZE)))

    # print('Loading model...')
    # model = load_model('models\\7032.h5')

    print("Creating solution file...")
    u1_testData, u2_testData, u3_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(
        u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(
        [u1_testData, u2_testData, u3_testData], batch_size=128)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write(
            '\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


if __name__ == '__main__':
    main()
