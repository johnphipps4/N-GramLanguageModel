import re
import numpy as np
from collections import Counter
import csv
import gensim
import gensim.models
from scipy import spatial

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def train_read_input(input_file):
    opn = open(input_file.format(0)).read()


def read_input(input_file):
    # model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)
    opn = open(input_file.format(0)).read()
    lines = opn.split("\n")
    opn = []
    for line in lines:
        opn.append(line.split(" "))
    # print opn

    count = 0
    for line in opn:
        result = (model.most_similar(positive=[line[2], line[1]], negative=[line[0]]))
        if result[0][0] == line[3]:
            count += 1
        print count

    # print count
    # print len(opn)



# x = read_input("~/Desktop/GoogleNews-vectors-negative300.bin.cpgz")
# read_input("analogy_test.txt")


def clean(file_path):
    opn = open(file_path.format(0)).read()
    opn = opn.replace("\n"," ")
    opn = opn.replace("' ","'")
    return opn.split(" ")

""" given a relative file path returns a list of paragraphs
 - requires: file_path to be a valid file path """
def sep_paras(file_path):
    lst = []
    opn = open(file_path.format(0)).read()
    paralist = opn.split("\n")
    for para in paralist:
        tknzd = para.replace("'' ","''")
        tknzd = para.split(" ")
        lst.append(tknzd)
    return lst


obama = clean('Assignment1_resources/train/obama.txt')
model = gensim.models.Word2Vec(obama, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
trump = clean('Assignment1_resources/train/trump.txt')
model_trump = gensim.models.Word2Vec(trump, size=100)
w2v_trump = dict(zip(model_trump.wv.index2word, model_trump.wv.syn0))

paras = sep_paras('Assignment1_resources/test/test.txt')

def calVector(lst):
    length = len(lst)
    xarray = []
    for i in range(len(lst[0])):
        xarray.append(0)
    for warray in lst:
        for i in range(len(warray)):
            xarray[i] += warray[i]
    for i in range(len(xarray)):
        xarray[i] /= length
    return xarray



class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X ])

def compare(trump_avg, obama_avg, para_lst):
    lst = []
    for para in para_lst:
        #currmodel = gensim.models.Word2Vec(para, size=100)
        #currw2v = dict(zip(currmodel.wv.index2word, currmodel.wv.syn0))
        para_array = MeanEmbeddingVectorizer(w2v).transform(para)
        para_avg = calVector(para_array)
        tpara_array = MeanEmbeddingVectorizer(w2v_trump).transform(para)
        tpara_avg = calVector(tpara_array)
        trumpresult = 1 - spatial.distance.cosine(trump_avg, tpara_avg)
        obamaresult = 1 - spatial.distance.cosine(obama_avg, para_avg)
        if obamaresult > trumpresult:
            lst.append(0)
        else:
            lst.append(1)
    return lst

def makecsv(filename, lst):
    file = filename + ".csv"
    with open(file, 'wb') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(['Id', 'Prediction'])
        for i in range(len(lst) - 1):
            wr.writerow([i, lst[i]])

obama_array = MeanEmbeddingVectorizer(w2v).transform(obama)
obama_avg = calVector(obama_array)
trump_array = MeanEmbeddingVectorizer(w2v_trump).transform(trump)
trump_avg = calVector(trump_array)
result_lst = compare(trump_avg, obama_avg, paras)
print(result_lst)
makecsv("sec8", result_lst)