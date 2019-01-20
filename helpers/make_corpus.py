import pandas as pd
import regex as re
from helpers import normalize
from keras.preprocessing.text import Tokenizer


# clear file
# file = 'custom_corpus.txt'
file = 'corpus.txt'
open("../_files/" + file, "w").close()

_max = 0
with open('../_files/' + 'corpus.txt', 'w', encoding='utf8') as f:
    print("Normalising train.txt...")
    for index, row in pd.read_csv("../data/train.txt", sep="\t").iterrows():
        f.write("%s\n" % normalize(row['turn1']))
        f.write("%s\n" % normalize(row['turn2']))
        f.write("%s\n" % normalize(row['turn3']))

    print("Normalising dev.txt...")
    for index, row in pd.read_csv('../data/dev.txt', sep='\t').iterrows():
        f.write("%s\n" % normalize(row['turn1']))
        f.write("%s\n" % normalize(row['turn2']))
        f.write("%s\n" % normalize(row['turn3']))

    print("Normalising testwithoutlabels.txt...")
    for index, row in pd.read_csv('../data/testwithoutlabels.txt', sep='\t').iterrows():
        f.write("%s\n" % normalize(row['turn1']))
        f.write("%s\n" % normalize(row['turn2']))
        f.write("%s\n" % normalize(row['turn3']))

    # print("Normalising rdany_conversations_2016-03-01.csv...")
    # for index, row in pd.read_csv("../_files/rdany_conversations_2016-03-01.csv").iterrows():
    #     if not (row['text'].startswith('[') and row['text'].endswith(']')):
    #         f.write("%s\n" % normalize(row['text']))

    # print("Normalising TWITTER CONV CORPUS.txt...")
    # with open("../_files/TWITTER CONV CORPUS.txt", 'r', encoding='utf8') as corp:
    #     for line in corp:
    #         if line.isspace() or line.strip() == '':
    #             continue
    #         else:
    #             f.write("%s\n" % normalize(line.strip()))

print("Done... Corpus written to ../_files/" + file)
print("Building word index...")

with open('../_files/' + file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]

    tokenizer = Tokenizer(num_words=None, filters='')
    tokenizer.fit_on_texts(lines)

    if file != 'corpus.txt':
        with open('../_files/words_' + file, 'w', encoding='utf-8') as fw:
            for word, index in tokenizer.word_index.items():
                fw.write("%s\n" % word)
        print("Wrote " + str(len(tokenizer.word_index)) + " words to ../_files/words_" + file)        
    else:
        with open('../_files/words.txt', 'w', encoding='utf-8') as fw:
            for word, index in tokenizer.word_index.items():
                fw.write("%s\n" % word)
        print("Wrote " + str(len(tokenizer.word_index)) + " words to ../_files/words.txt")
        

