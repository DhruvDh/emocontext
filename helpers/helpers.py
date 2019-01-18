import regex as re
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import torch
import io, sys, os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torchvision


def normalize(s):
    """
    Given a text, cleans and normalizes it.
    """

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
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)

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
    s = re.sub('([\U00010000-\U0010ffff])', r' \1 ', s, flags=re.UNICODE)

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

    return s.strip()

def get_class_weights(series):
    return torch.tensor(class_weight.compute_class_weight(
        'balanced', series.unique(), np.array(series.tolist())
    )).float()

def get_embedding_matrix(wordIndex, file_name, embedding_dimensions):
    with io.open(os.path.join('..', '_files', file_name), 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, embedding_dimensions))
    for word, i in wordIndex.items():
        embeddingVector = data.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            print('no custom vector for ', word)

    return torch.stack([torch.tensor(x) for x in embeddingMatrix])

def make_tensors(dataset):
    tokenizer = Tokenizer(num_words=None, filters='')
    tokenizer.fit_on_texts(dataset['turn1'].tolist() + dataset['turn2'].tolist() + dataset['turn3'].tolist())

    turn1 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn1'].tolist())]
    turn2 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn2'].tolist())]
    turn3 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn3'].tolist())]

    len1 = torch.stack([torch.tensor(x[0]).long() for x in turn1]).unsqueeze(1)
    len2 = torch.stack([torch.tensor(x[0]).long() for x in turn2]).unsqueeze(1)
    len3 = torch.stack([torch.tensor(x[0]).long() for x in turn3]).unsqueeze(1)

    turn1 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn1])])
    turn2 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn2])])
    turn3 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn3])])

    label_tokenizer = Tokenizer(num_words=None, filters='')
    label_tokenizer.fit_on_texts(dataset['label'].tolist())

    labels = [label_tokenizer.word_index[x] - 1 for x in dataset['label'].tolist()]
    labels = torch.tensor(labels).long().unsqueeze(1)

    return (    
        torch.cat(
            (
                len1, len2, len3,
                turn1, turn2, turn3,
                labels
            ),
            dim=1),
        turn1.shape[1],
        turn2.shape[1],
        turn3.shape[1],
        tokenizer,
        label_tokenizer.word_index
    )