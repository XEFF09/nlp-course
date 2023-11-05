import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import re

from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie
import tensorflow as tf
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

custom_dict_thai_word = {
    'อัลตร้า','11PM','720p','คริสตัส','เน็ตฟลิก','p5','ps5','4k',
    'แอนดรอย','ทุกรุ่น','เอเบิลเม็น','ช้ากว่า','ไทเทเนี่ยม','เอไอ','จุงเบย','แหล่ะ',
    'ps4','iphone','A17','ไทเท','s23','แอป','i15','i14',
    'i13','i12','i11','เเมร่ง','ยูทูป','5G','ดรอป','เเละ',
    'พับจี','เเพง','เเยก','เเทบ','ชาจ','ทวิต','ไอจี','แล้ว',
    'เคส','ตาลุกวาว','IP15','IP14','ip15','ip14','เเรง','บอก',
    'ตู้ม','อุส่าห์','ก่อน','ภาพยนต์','แลค','วอท','แม่ง','เกนชิน',
    'แปป','เครม','ปัจใจ','ชิฟ','บลา','อุ่นภูมิ','ไลน์นิ่ง','สแปค',
    'คอเกม','แอนดอย','อินฟลู','โซเชียล','จาก','เกมเมอร์','สน้บสนุน','อีสปอร์ต',
    'ไวเลส','ม้าก','เปลี้ยน','แอนดอย','เกมมิง','มาก','คอนทร่า','เมม',
    'โอ่ว','ออนดรอย','โอเอส','ก่อน','แอนดอรย์','พอซ','เกมมิ่ง','โอเคร้',
    'ไอแลนด์','ลี่นปื้ด','15','14','แม้ก','เหน','โครต','ราคา','คมระเคือง',
    'ภาพ','ไม่ได้','ไม่เคย','ไม่ใช่','ไม่ถูก','ไม่มีปัญหา','ไม่ร้อน'
}

thai_word_set = set(thai_words())
custom_dict_thai_word.update(thai_word_set)
trie = dict_trie(dict_source=custom_dict_thai_word)

STOP_WORD = list(thai_stopwords()) + [" ", "\n", "ๆ"]
STOP_WORD.remove("ไม่")
FORMAT = r"[\u0E00-\u0E7Fa-zA-Z'0-9]+"

def tokenize(sentence):
    return word_tokenize(sentence, engine="newmm", keep_whitespace=False, custom_dict=trie)

def cleaning_stop_word(tk_list):
    return [word.replace("ๆ", "") for word in tk_list if word not in STOP_WORD]

def cleaning_symbols_emoji(tk_list):
    return [re.findall(FORMAT, text)[0] for text in tk_list if re.findall(FORMAT, text)]

def big_cleaning(sentence):
    tokens = cleaning_symbols_emoji(cleaning_stop_word(tokenize(sentence)))
    return tokens

loaded_model = tf.keras.models.load_model('lstm_model.h5')
loaded_model.summary()

THRESHOLD = 0.75

sheet_id = '1s0LdF3Q-XYzB0N-kVHM4XilMl20WpMHojE8XkwdAZa8'

df_0 = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%200')
df_1 = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%201')
frame = [df_0, df_1]
data = pd.concat(frame, ignore_index=True)

x = data["text"].apply(big_cleaning).astype(str)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)

def predict_class(df):
    df = df["message"].apply(big_cleaning).astype(str)
    df = tokenizer.texts_to_sequences(df)
    padded_sequence = pad_sequences(df, maxlen=85)

    lstm_pred = loaded_model.predict(padded_sequence)
    lstm_label = np.argmax(lstm_pred, axis=1)
    lstm_label[lstm_pred.max(axis=1) < THRESHOLD] = -1
    
    check = np.where(lstm_label == 0, "Positive", np.where(lstm_label == 1, "Negative", "Neutral"))
    return check 

df = pd.DataFrame({"message": ["ผมชอบ 15นะ เป็นอนาคตของเกมมิ่ง"]})
predicted_labels = predict_class(df=df)
print(predicted_labels)