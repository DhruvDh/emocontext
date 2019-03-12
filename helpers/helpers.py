import regex as re
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import torch
import io, sys, os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torchvision
import emoji

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

    smile = [">:]", 'B-)', ":-)", ":)", ":o)", ":]", " :3 ", "B)", ':-)', '(:', '(^・^)', '(:', ':‑)', ': ^ )', ":')", ":'‑)", "(・ω・)",
             ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", "^_^", "(^.^)", "^.^", "^ω^", "(^○^", "(^○^)", "(^o^", "(^o^)", ":)<", ":)‑", "3:)"]
    laugh = [">:D", ":-D", ":D", "8-D", "x-D", "X-D", "=-D",
             "=D", "=-3", "8-)", '8‑d', "=3", ":)o", "(^�^", "(^�^)", "(≧∇≦", '(≧∇≦)']
    sad = [">:[", ":-(", ":(",  ":-c", ":c", ":-<", ":-[",
           ":[", ":{", ">.>", "<.<", ">.<", ';(', '):']
    wink = [">;]", ";-)", ";)", "*-)", "*)", ";-]", ':*', ':-*', ";‑)",
            ";]", ";D", ";^)", ':^*', ';‑)', ';‑ )', ';")', ';")*'] 
    tongue = [">:P", ":-P", ":P", "X-P", "x-p", ":-p", 'xd', 't.t', "t_t"
              ":p", "=p", ":-Þ", ":Þ", ":-b", ":b", "=p", "=P"]
    surprise = [">:o", ">:O", ":-O", ":O", "°o°", ":()",
                "°O°", ":O", "o_O", "o.O", "8-0", 'D:']
    annoyed = [">:\\\\", '>:\\', ">:/", ":-/", ":-.", ':\\\\', ':\\', ':@', '-_-', '//:'
               "=/", "=\\\\", ":S", '://', ":x", ":/'", '=\\']
    cry = [":'(", ";'("]

    s = s.lower()
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)

    s = emoji.demojize(s)
    s = re_sub(r":([^\s])*:", r" \1 ")
    s = " ".join(s.split("_"))

    for emojis in sad:
        s = s.replace(emojis.lower(), ' :( ')

    for emojis in smile:
        s = s.replace(emojis.lower(), ' :) ')

    for emojis in laugh:
        s = s.replace(emojis.lower(), ' :d ')

    for emojis in wink:
        s = s.replace(emojis.lower(), ' ;) ')

    for emojis in tongue:
        s = s.replace(emojis.lower(), ' :p ')

    for emojis in annoyed:
        s = s.replace(emojis.lower(), ' :/ ')

    for emojis in surprise:
        s = s.replace(emojis.lower(), ' :o ')

    for emojis in cry:
        s = s.replace(emojis.lower(), ' :( ')

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

    
    s = s.replace(" </3 ", " 💔 ")  
    s = s.replace(" <\3 ", " 💔 ")
    s = s.replace(" :0 ", " :o ")
    s = s.replace("&apos;", "'")
    s = s.replace("&amp;", "&")
  
    s = s.replace(" youre ", " you're ")
    s = s.replace(" r ", " are ")
    s = s.replace(" ur ", " your ")
    s = s.replace(" u ", " you ")
    s = s.replace(" dont ", " don't ")
    s = s.replace(" cant ", " can't ")
    s = s.replace(" m ", " am ")
    s = s.replace(" im ", " i am ")
    s = s.replace(" y ", " why ")
    s = s.replace(" n ", " and ")
    s = s.replace(" gf ", " girlfriend ")
    s = s.replace(" bf ", " boyfriend ")
    s = s.replace(" k ", " okay ")
    s = s.replace(" gf ", " girlfriend ")
    s = s.replace(" iam ", " i am ")    
    s = s.replace(" plz ", " please ")
    s = s.replace(" pls ", " please ")
    s = s.replace(" lmao ", " lol ")
    s = s.replace(" lets ", " let's ")
    s = s.replace(" havent ", " haven't ")
    s = s.replace(" wht ", " what ")
    s = s.replace(" arent ", " aren't ")
    s = s.replace(" wat ", " what ")
    s = s.replace(" wasnt ", " wasn't ")
    s = s.replace(" was'nt ", " wasn't ")
    s = s.replace(" werent ", " weren't ")
    s = s.replace(" were'nt ", " weren't ")
    s = s.replace(" theres ", " there's ")
    s = s.replace(" cos ", " cause ")
    s = s.replace(" coz ", " cause ")
    s = s.replace(" aint ", " ain't ")
    s = s.replace(" idk ", " i don't know ")
    s = s.replace(" abt ", " about ")
    s = s.replace(" yep ", " yes ")
    s = s.replace(" nah ", " no ")
    s = s.replace(" tho ", " though ")
    s = s.replace(" hehe ", " haha ")
    s = s.replace(" nd ", " and ")
    s = s.replace(" wt ", " what ")
    s = s.replace(" dnt ", " don't ")
    s = s.replace(" knw ", " know ")
    s = s.replace(" shes ", " she's ")
    s = s.replace(" hes ", " he's ")
    s = s.replace(" ohhh ", " ohh ")
    s = s.replace(" ohhhh ", " ohh ")
    s = s.replace(" awww ", " aww ")
    s = s.replace(" awwww ", " aww ")
    s = s.replace(" theyre ", " they're ")
    s = s.replace(" wouldnt ", " wouldn't ")
    s = s.replace(" cuz ", " cause ")
    s = s.replace(" couldnt ", " couldn't ")
    s = s.replace(" ppl ", " people ")
    s = s.replace(" den ", " then ")
    s = s.replace(" yea ", " yeah ")
    s = s.replace(" yaa ", " yeah ")
    s = s.replace(" gud ", " good ")
    s = s.replace(" wouldnt ", " wouldn't ")
    s = s.replace(" nt ", " not ")
    s = s.replace(" whos ", " who's ")
    s = s.replace(" youve ", " you've ")
    s = s.replace(" wont ", " won't ")
    s = s.replace(" msg ", " message ")
    s = s.replace(" hola ", " hi ")
    s = s.replace(" yess ", " yes ")
    s = s.replace(" yesss ", " yes ")
    s = s.replace(" yepp ", " yes ")
    s = s.replace(" yeppp ", " yes ")
    s = s.replace(" bt ", " but ")
    s = s.replace(" ohk ", " oh okay ")
    s = s.replace(" ok ", " okay ")
    s = s.replace(" okk ", " okay ")
    s = s.replace(" rn ", " right now ")
    s = s.replace(" nw ", " now ")
    s = s.replace(" its ", " it's ")
    s = s.replace(" sry ", " sorry ")
    s = s.replace(" luv ", " love ")
    s = s.replace(" tht ", " that ")
    s = s.replace(" frnd ", " friend ")
    s = s.replace(" bout ", " about ")
    s = s.replace(" fav ", " favorite ")
    s = s.replace(" favourite ", " favorite ")
    s = s.replace( " shouldnt ", " shouldn't ")
    s = s.replace(" si ", " yes ")
    s = s.replace(" y'all ", " you all ")
    s = s.replace(" yall ", " you all ")
    s = s.replace(" thnx ", " thanks ")
    s = s.replace(" thnks ", " thanks ")
    s = s.replace(" lil ", " little ")
    s = s.replace(" bcoz ", " because ")
    s = s.replace(" whens ", " when's ")
    s = s.replace(" hw ", " how ")
    s = s.replace(" oooh ", " ooh ")
    s = s.replace(" ooooh ", " ooh ")
    s = s.replace(" ooohh ", " ooh ")
    s = s.replace(" fr ", " for ")
    s = s.replace(" jst ", " just ")
    s = s.replace(" dunno ", " don't know ")
    s = s.replace(" hw ", " how ")
    s = s.replace(" urs ", " your's ")
    s = s.replace(" hv ", " have ")
    s = s.replace(" ahhh ", " ahh ")
    s = s.replace(" soo ", " so ")
    s = s.replace(" sooo ", " so ")
    s = s.replace(" meee ", " me ")
    s = s.replace(" mee ", " me ")
    s = s.replace(" doesnt ", " doesn't ")
    s = s.replace(" youd ", " you'd ")
    s = s.replace(" yah ", " yeah ")
    s = s.replace(" hahaha ", " haha ")
    s = s.replace(" hahahah ", " haha ")
    s = s.replace(" hahah ", " haha ")
    s = s.replace(" hehehe ", " haha ")
    s = s.replace(" plzz ", " please ")
    s = s.replace(" plss ", " please ")
    s = s.replace(" rofl ", " lol ")
    s = s.replace(" lmfao ", " lol ")
    s = s.replace(" tbh ", " to be honest ")
    s = s.replace(" imo ", " in my opinion ")
    s = s.replace(" imho ", " in my honest opinion ")
    s = s.replace(" ofc ", " of course ")
    s = s.replace(" ofcourse ", " of course ")
    s = s.replace(" hav ", " have ")
    s = s.replace(" bcz ", " because ")
    s = s.replace(" dint ", " didn't ")
    s = s.replace(" didnt ", " didn't ")
    s = s.replace(" cnt ", " can't ")
    s = s.replace(" cn ", " can ")
    s = s.replace(" thats ", " that's ")
    s = s.replace(" 'cause' ", " because ")
    s = s.replace(" dat ", " that ")
    s = s.replace(" okkkk ", " okay ")
    s = s.replace(" rofl ", " lol ")
    s = s.replace(" ive ", " i've ")
    s = s.replace(" kno ", " know ")
    s = s.replace(" crzy ", " crazy ")
    s = s.replace(" boob's ", " boobs ")
    s = s.replace(" ans ", " answer ")

    s = re_sub(r"\s+", " ")
    return s.strip()

def get_class_weights(series, labels):
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
            print('no vector for ', word)

    return torch.stack([torch.tensor(x) for x in embeddingMatrix])

def make_tensors(dataset, tokenizer, label_tokenizer):
    turn1 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn1'].tolist())]
    turn2 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn2'].tolist())]
    turn3 = [[len(x)] + x for x in tokenizer.texts_to_sequences(dataset['turn3'].tolist())]

    len1 = torch.stack([torch.tensor(x[0]).long() for x in turn1]).unsqueeze(1)
    len2 = torch.stack([torch.tensor(x[0]).long() for x in turn2]).unsqueeze(1)
    len3 = torch.stack([torch.tensor(x[0]).long() for x in turn3]).unsqueeze(1)

    turn1 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn1])])
    turn2 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn2])])
    turn3 = torch.stack([torch.tensor(x).long() for x in pad_sequences([x[1:] for x in turn3])])

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
    )