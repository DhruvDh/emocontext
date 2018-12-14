import pandas as pd
import regex as re


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
