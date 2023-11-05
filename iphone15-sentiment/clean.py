from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import re

STOP_WORD = list(thai_stopwords()) + [" ", "\n"]
FORMAT = r"[\u0E00-\u0E7Fa-zA-Z'0-9]+"

def tokenize(sentence):
    return word_tokenize(sentence, engine="newmm")

def cleaning_stop_word(tk_list):
    return [word for word in tk_list if word not in STOP_WORD]

def cleaning_symbols_emoji(tk_list):
    return [re.findall(FORMAT, text)[0] for text in tk_list if re.findall(FORMAT, text)]

def big_cleaning(sentence):
    return  cleaning_symbols_emoji( cleaning_stop_word( tokenize(sentence) ) )