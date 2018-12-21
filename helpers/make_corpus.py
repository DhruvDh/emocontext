import pandas as pd
import regex as re
from normalize import normalize

# clear file
open("_files/corpus.txt", "w").close()

_data = pd.read_csv("../data/train.txt", sep="\t")
_rdany_chats = pd.read_csv("_files/rdany_conversations_2016-03-01.csv")
_max = 0
with open('_files/corpus.txt', 'w', encoding='utf8') as f:
    for index, row in _data.iterrows():

        f.write("%s\n" % normalize(row['turn1']))
        f.write("%s\n" % normalize(row['turn2']))
        f.write("%s\n" % normalize(row['turn3']))

    for index, row in _rdany_chats.iterrows():
        if not (row['text'].startswith('[') and row['text'].endswith(']')):
            f.write("%s\n" % normalize(row['text']))

    with open("_files/TWITTER CONV CORPUS.txt", 'r', encoding='utf8') as corp:
        for line in corp:
            if line.isspace() or line.strip() == '':
                continue
            else:
                f.write("%s\n" % normalize(line.strip()))
