#coding:utf-8


import re
import cPickle
from collections import defaultdict
import numpy as np
import pandas as pd
import nltk

top_id_file = r'./data/top_id.txt'
bottom_id_file = r'./data/bottom_id.txt'
top_fea_text_file = r'./data/top_text.txt'
bottom_fea_text_file = r'./data/bottom_text.txt'
top_rule_text_file = r'./data/top_text_for_rule.txt'
bottom_rule_text_file = r'./data/bottom_text_for_rule.txt'


#process textual data
def build_data(data_folder, clean_string=True):
    top_id_f = open(top_id_file, 'w')
    bottom_id_f = open(bottom_id_file, 'w')
    top_fea_text_f = open(top_fea_text_file, 'w')
    bottom_fea_text_f = open(bottom_fea_text_file, 'w')
    top_rule_text_f = open(top_rule_text_file, 'w')
    bottom_rule_text_f = open(bottom_rule_text_file, 'w')

    revs = []
    [top_file,bottom_file] = data_folder
    vocab = defaultdict(float)   #{word:word_num}
    with open(top_file, "r") as f:
        for line in f:
            cloth = {}
            #print(str(line))
            line = line.strip()
            list = line.split('\t')
            #print(list)

            cloth["id"] = list[0]
            cloth["title"] = list[1]
         
            if len(list) == 3:
                category_text = list[2].strip()
                category_text = category_text.split(">")[1:]
                category_text = " ".join(category_text)
                cloth["category"] =category_text
            else:
                cloth["category"] = ""

            orig_text = cloth["title"] + " " + cloth["category"]
            #print(orig_text)

            if clean_string:
                fea_text = clean_str_for_fea(orig_text)
            else:
                fea_text = orig_text.lower()
            words = set(fea_text.split())  

            rule_text = clean_str_for_rule(cloth["title"]) + " " + cloth["category"].lower()
            #print(rule_text)

            for word in words:
                vocab[word] += 1
            datum  = {"cloth":cloth,
                      "orig_text": orig_text,
                      "new_text":fea_text,
                      "num_words": len(fea_text.split()),
                      "split": 0} # 0-top, 1-bottom
            revs.append(datum)

            top_id_f.write(str(float(cloth["id"]))+"\n")
            top_fea_text_f.write(fea_text+"\n")
            top_rule_text_f.write(rule_text+"\n")

    with open(bottom_file, "r") as f:
        for line in f:
            cloth = {}
            #print(str(line))
            line = line.strip()
            list = line.split('\t')
            #print(list)

            cloth["id"] = list[0]
            cloth["title"] = list[1]

            if len(list) == 3:
                category_text = list[2].strip()
                category_text = category_text.split(">")[1:]
                category_text = " ".join(category_text)
                cloth["category"] =category_text
            else:
                cloth["category"] = ""

            orig_text = cloth["title"] + " " + cloth["category"]
            #print(orig_text)

            if clean_string:
                fea_text = clean_str_for_fea(orig_text)
            else:
                fea_text = orig_text.lower()
            words = set(fea_text.split())

            rule_text = clean_str_for_rule(cloth["title"]) + " " + cloth["category"].lower()
            #print(rule_text)

            for word in words:
                vocab[word] += 1
            datum  = {"cloth":cloth,
                      "orig_text": orig_text,
                      "new_text":fea_text,
                      "num_words": len(fea_text.split()),
                      "split": 0} # 0-top, 1-bottom
            revs.append(datum)

            bottom_id_f.write(str(float(cloth["id"])) + "\n")
            bottom_fea_text_f.write(fea_text + "\n")
            bottom_rule_text_f.write(rule_text+"\n")

    top_id_f.close()
    bottom_id_f.close()
    top_fea_text_f.close()
    bottom_fea_text_f.close()
    top_rule_text_f.close()
    bottom_rule_text_f.close()

    return revs, vocab

#to extract feature 
def clean_str_for_fea(string, TREC=True):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r">", " ", string)
    string = re.sub('\d+', '', string)
    #string = re.sub(r"\'", " ", string)

    return string.strip() if TREC else string.strip().lower()

#to extract rule
def clean_str_for_rule(str):
    str = re.sub("\+", " ", str)
    str = re.sub(r"/", " ", str)
    str = re.sub(r"\(", " ", str)
    str = re.sub(r"\)", " ", str)
    str = re.sub(r"\|", " ", str)
    str = re.sub(r",", " ", str)
    str = re.sub(r"!", " ", str)
    str = re.sub(r"\.", " ", str)
    str = re.sub(r":", " ", str)
    str = re.sub(r"\?", " ", str)
    str = re.sub(r"\[", " ", str)
    str = re.sub(r"\]", " ", str)
    str = re.sub(r"\\", " ", str)
    str = re.sub(r"&", " ", str)
    str = re.sub(r"-", " ", str)
    str = re.sub(r"'", " ", str)
    str = re.sub(r"\"", " ", str)
    str = re.sub(r">", " ", str)
    str = re.sub('\d+', '', str)

    return str.strip().lower()

#load word_vec file
def load_bin_vec(fname, vocab):

    word_vecs = {}
    w_f = open("./temporary_data/all_word_in_w2v.txt","w")
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        #print(vocab_size)
        #print(layer1_size)
        binary_len = np.dtype('float32').itemsize * layer1_size 

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    w_f.write(" "+word + " \n")
                    break
                if ch != '\n':
                    word.append(ch)

            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

                #print("*****"+str(word_vecs[word])+"*******")
            else:
                f.read(binary_len)

    w_f.close()
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=3, k=300):
    #print(word_vecs.keys())
    #print(vocab.keys())
    no_exist_word = []
    for word in vocab:
        if  word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            list = (word,vocab[word])
            no_exist_word.append(list)
    return no_exist_word


def get_W(word_vecs, k=300):

    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32') 
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def main_f():

    top_file = r"./data/top.txt"    #raw
    bottom_file = r"./data/bottom.txt"  #raw
    w2v_file = r"./w2v/GoogleNews-vectors-negative300.bin"

    data_folder = [top_file,bottom_file]
    print "loading data...",
    revs, vocab = build_data(data_folder, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])    
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    no_exist_word = add_unknown_words(w2v, vocab)

    no_exist_word_f = open("./temporary_data/no_exist_word.txt", "w")
    for item in no_exist_word:
        no_exist_word_f.write(item[0]+"\t"+str(item[1])+"\n")
    print("no exist word num = "+str(len(no_exist_word)))
    no_exist_word_f.close()


    W, word_idx_map = get_W(w2v)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("./fea/cloth.binary.p", "wb"))
    print "dataset created!"

    word_in_data_f = open("./temporary_data/word_in_data.txt", "w")
    for word in vocab:
        word_in_data_f.write(word+"\t"+str(vocab[word])+"\n")
    word_in_data_f.close()

main_f()