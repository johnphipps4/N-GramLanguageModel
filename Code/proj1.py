import re
import numpy as np
from collections import Counter
import csv

""" given a relative file path returns a list of tokens
 - requires: file_path to be a valid file path"""
def clean(file_path):
    opn = open(file_path.format(0)).read()
    opn = opn.replace("\n"," ")
    opn = opn.replace("’ ","’")
    return opn.split(" ")

""" given a relative file path returns a list of paragraphs
 - requires: file_path to be a valid file path """
def sep_paras(file_path):
    lst = []
    opn = open(file_path.format(0)).read()
    paralist = opn.split("\n")
    for para in paralist:
        tknzd = para.replace("’ ","’")
        tknzd = para.split(" ")
        lst.append(tknzd)
    return lst

""" sets all files to be used throughout code"""
testtext = clean('Assignment1_resources/test/test.txt')
obamatrain = clean('Assignment1_resources/train/obama.txt')
trumptrain = clean('Assignment1_resources/train/trump.txt')
obamadev = clean('Assignment1_resources/development/obama.txt')
trumpdev = clean('Assignment1_resources/development/trump.txt')
paras = sep_paras('Assignment1_resources/test/test.txt')

chop = testtext[0:250]

""" receives list of tokens and returns dictionary with unique tokens
    as keys and unigram frequency of tokens throughout the text as values"""
def getUniCount(txt):
    countdict = dict()
    for word in txt:
        if word in countdict: 
            countdict[word] += 1
        else:
            countdict[word] = 1
    return countdict

""" receives list of tokens and returns dictionary with unique tokens
    as keys and unigram probabilities of tokens occuring in whole doc as values"""
def getUniProb(count_dict):
    probdict = dict()
    for k,v in count_dict.items():
        probdict[k] = v/len(testtext)
    return probdict

""" receives list of tokens and returns a 2D dictionary for bigram frequencies"""
def getBiCount(txt):
    countdict = dict()
    for i in range(len(txt)-1):
        curr = txt[i]
        nxt = txt[i+1]
        if curr in countdict.keys():
            if nxt in countdict[curr]:
                countdict[curr][nxt] += 1
            else:
                countdict[curr][nxt] = 1
        else:
            countdict[curr] = {nxt: 1}
    return countdict

""" returns a 2D dictionary for bigram probabilities
 - requires: unicount_dict to be the unigram count dictionary 
             bicount_dict to be the bigram count dictionary """
def getBiProb(unicount_dict,bicount_dict):
    probdict = dict()
    for k1,v1 in bicount_dict.items():
        probdict[k1] = dict()
        for k2,v2 in v1.items():
            probdict[k1][k2] = v2 / unicount_dict[k1]
    return probdict

""" returns an int indicating the (rounded) average length of all
    sentences in the corpus."""
def ave_sentence_length(txt):
    punctuation = filter(lambda x: x[1] in ['.', '?', '!'], enumerate(txt))
    indices = [i[0] for i in punctuation]
    subtract = [0] + indices[:-1]
    difference = np.subtract(indices, subtract)
    avg = np.mean(difference)
    return int(np.rint(avg))

""" receives list of float values and returns new list of float 
    values summing to 1. fixes numpy random.choice glitch to ensure
    weighted probabilities sum to 1"""
def div_sum(vals):
    newlst = []
    sv = sum(vals)
    for v in vals:
        newlst.append(v / sv)
    return newlst

""" Returns a randomly generated sentence made up of unigrams.
        The sentence terminates when either:
                1. The next unigram to add is an ending punctuation
                   term: '.', '!', or '?'
                2. The sentence is as long as the average sentence
                   of the corpus """
def unigram_sentence(txt,seed):
    if seed != "": sentence = ['<s>'] + seed.split(" ")
    else: sentence = ['<s>']
    unigrams = getUniCount(txt)
    while len(sentence) <= ave_sentence_length(txt):
        unigram_val = np.random.choice(list(getUniProb(getUniCount(txt)).keys()), 
                                       p = div_sum(list(getUniProb(getUniCount(txt)).values())))
        sentence.append(unigram_val)
        if (unigram_val in ['.', '!', '?']):
            return sentence + ['</s>']
    return sentence + ['</s>']

""" Returns a randomly generated sentence made up of bigrams.
    The sentence terminates when either:
            1. The next unigram to add is an ending punctuation
               term: '.', '!', or '?'
            2. The sentence is as long as the average sentence of the
               corpus """
def bigram_sentence(txt,seed):
    if seed != "": 
        sentence = ['<s>'] + seed.split(" ")
        bigram_start = sentence[-1]
    else:
        bigram_start = np.random.choice(list(getUniProb(getUniCount(txt)).keys()), 
                                        p = div_sum(list(getUniProb(getUniCount(txt)).values())))
        sentence = ['<s>', bigram_start] 

    # Generate sentences
    while bigram_start not in ['.', '!', '?'] and len(sentence) <= ave_sentence_length(txt):
        start_dict = getBiProb(getUniCount(txt),getBiCount(txt))[bigram_start]
        next_word = np.random.choice(list(start_dict.keys()), p = div_sum(list(start_dict.values())))
        sentence.append(next_word)
        bigram_start = next_word

    return sentence + ['</s>'] 

""" receives a unigram count dictionary and returns a new 
    dictionary with all tokens with only one count substituted to a <unk> tag"""
def uniCountUnk(old_dict):
    new_dict = {}
    new_dict["<unk>"] = 0
    for k,v in old_dict.items():
        if v == 1:
            new_dict.pop(k, None)
            new_dict["<unk>"] += 1 
        else:
            new_dict[k] = v
    return new_dict

""" receives a token list and returns a new 2D bicount dictionary
    with all one-time occuring tokens substituted to a <unk> tag"""
def biCountUnk(txt):
    newlst = []
    newlst = txt
    unicount_old = getUniCount(txt)
    for k,v in unicount_old.items():
        if v == 1:
            newlst[newlst.index(k)] = "<unk>"
    return getBiCount(newlst)

""" SMOOTHED BIGRAM HELPERS
    first chunk of the formula"""
def getMax(first,second,bicount_dict):
    if second not in bicount_dict[first] or bicount_dict[first][second] - 0.75 < 0:
        return 0
    else:
        return (bicount_dict[first][second] - 0.75) / sum(bicount_dict[first].values())
    
""" returns lambda value for given first word"""
def getLambda(first,bicount_dict):
    return 0.75 / sum(bicount_dict[first].values()) * len(bicount_dict[first])

""" returns pkn value for given second word"""
def getPKN(second,bicount_dict):
    counter = 0
    total = 0
    for k,v in bicount_dict.items():
        if second in v:
            counter += 1
        for k2,v2 in v.items():
            total += 1
    return (counter / total)

""" returns a lambda dictionary where the keys are all possible words in the text 
    and each key is mapped to its lambda value"""
def makeLambdaDict(bicount_dict):
    probdict = dict()
    for k in bicount_dict.keys():
        probdict[k] = getLambda(k,bicount_dict)
    return probdict
   
""" returns a pkn dictionary where the keys are all possible words in the text 
    and each key is mapped to its pkn value"""
def makePKNDict(bicount_dict):
    probdict = dict()
    for k in bicount_dict.keys():
        probdict[k] = getPKN(k,bicount_dict)
    return probdict

""" uses the Kneser Ney method to calculate bigram probabilities. 
    returns a 2D dictionary with all possible pairs of words"""
def smoothed_bigram(lambda_dict,pkn_dict,bicount_dict):
    probdict = dict()
    for w1 in bicount_dict.keys():
        probdict[w1] = dict()
        for w2 in bicount_dict.keys():
            probdict[w1][w2] = getMax(w1,w2,bicount_dict) + lambda_dict[w1] * pkn_dict[w2]
    return probdict

""" uses add-1 smoothing to calculate unigram probabilities.
 - requires: unicount_dict to be the unigram count dictionary with unknown words"""
def smoothed_unigram(unicount_dict):
    add1dict = dict()
    for w in unicount_dict.keys():
        add1dict[w] = unicount_dict[w] + 1
    return getUniProb(add1dict)

""" computes maximum entropy of the given development set for unigrams
 - requires: unidict is a smoothed unigram prob dictionary"""
def uniPerplexity(dev_txt, unidict):
    new_sum = 0
    for i in range(len(dev_txt) - 1):
        w = dev_txt[i]
        if w in unidict.keys():
            x = unidict[w]
        else:
            x = unidict["<unk>"]
        new_sum += -1 * np.log(x)
    result = np.exp((1.0/len(dev_txt))*new_sum)
    return result

""" computes maximum entropy of the given development set for bigrams
 - requires: bicount_dict is a smoothed bigram prob dictionary"""
def perplexity(dev_txt,bicount_dict):
    new_sum = 0
    for i in range(len(dev_txt)-1):
        w1 = dev_txt[i]
        w2 = dev_txt[i+1]
        if w1 in bicount_dict.keys():
            if w2 in bicount_dict[w1].keys(): 
                x = bicount_dict[w1][w2]
            else:
                x = bicount_dict[w1]["<unk>"]
        else:
            if w2 in bicount_dict["<unk>"].keys(): 
                x = bicount_dict["<unk>"][w2]
            else:
                x = bicount_dict["<unk>"] ["<unk>"] 
        new_sum += -1 * np.log(x) # where x is the probability of bigram
    result = np.exp((1.0/len(dev_txt))*new_sum)
    return result # for all bigrams in N where N is len of dev_corpus

""" runs speech classification on our bigram models. 
 - returns: a list of 0s and 1s. list[i] will be 0 if the clasification 
            algorithm determines that paragraph[i] in test data should be Obama, 1 if Trump.
 - requires: trump_perp is the perplexity of trump's test/dev data
             obama_perp is the perplexity of obama's test/dev data
             trump_smoo is the smoothed bigram dictionary for trump
             obama_cmoo is the smoothed bigram dictionary for obama"""
def classification(trump_perp,obama_perp,trump_smoo,obama_smoo):
    lst = []
    for para in paras:
        trump_delta = abs(perplexity(para,trump_smoo) - trump_perp)
        obama_delta = abs(perplexity(para,obama_smoo) - obama_perp)
        if trump_delta < obama_delta:
            lst.append(1)
        else:
            lst.append(0)
    return lst

""" runs speech classification on our unigram models. 
 - returns: a list of 0s and 1s. list[i] will be 0 if the clasification 
            algorithm determines that paragraph[i] in test data should be Obama, 1 if Trump.
 - requires: trump_perp is the perplexity of trump's test/dev data
             obama_perp is the perplexity of obama's test/dev data
             trump_smoo is the smoothed unigram dictionary for trump
             obama_cmoo is the smoothed unigram dictionary for obama"""
def uniClassification(trump_perp,obama_perp,trump_smoo,obama_smoo):
    lst = []
    for para in paras:
        trump_delta = abs(uniPerplexity(para,trump_smoo) - trump_perp)
        obama_delta = abs(uniPerplexity(para,obama_smoo) - obama_perp)
        if trump_delta < obama_delta:
            lst.append(1)
        else:
            lst.append(0)
    return lst

""" helper function that makes a csv file from the classification output
 - requires: filename is a string
             lst is the list returned by either the classification function for bigrams, 
             or the uniClassification function for unigrams."""
def makecsv(filename, lst):
    file = filename + ".csv"
    with open(file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(['Id', 'Prediction'])
        for i in range(len(lst) - 1):
            wr.writerow([i, lst[i]])

""" helper function for calculating the average word vector.
 - requires: lst is a list of vectors with the same length """
def calVector(lst):
    length = len(lst)
    xarray = []
    for warray in lst:
        for i in range(len(warray)):
            xarray[i] += warray[i]
    for i in range(len(xarray)):
        xarray[i] /= length
    return xarray
    

## pre-set arguments for smoothed_bigram function
#bicount_obama = biCountUnk(obamatrain)
#bicount_trump = biCountUnk(trumptrain)
#
#lambda_d_trump = makeLambdaDict(bicount_trump)
#lambda_d_obama = makeLambdaDict(bicount_obama)
#
#pkn_d_trump = makePKNDict(bicount_trump)
#pkn_d_obama = makePKNDict(bicount_obama)
#
## create smoothed_bigram dictionary
#smoothed_trump = smoothed_bigram(lambda_d_trump,pkn_d_trump,bicount_trump)
#smoothed_obama = smoothed_bigram(lambda_d_obama,pkn_d_obama,bicount_obama)
#
## set perplexity variables
#trump_perp = perplexity(trumptrain,smoothed_trump)
#obama_perp = perplexity(obamatrain,smoothed_obama)

#print(trump_perp)
#print(obama_perp)

#class_lst = classification(trump_perp,obama_perp,smoothed_trump,smoothed_obama)
#print(class_lst)
#makecsv("classification", class_lst)

##run perplexity on unigrams
#trump_uni_count = getUniCount(trumptrain)
#obama_uni_count = getUniCount(obamatrain)
#
#trump_uni_c_unk = uniCountUnk(trump_uni_count)
#obama_uni_c_unk = uniCountUnk(obama_uni_count)
#
#smoothed_trump_uni = smoothed_unigram(trump_uni_c_unk)
#smoothed_obama_uni = smoothed_unigram(obama_uni_c_unk)
#
#trump_perp_uni_dev = uniPerplexity(trumpdev, smoothed_trump_uni)
#obama_perp_uni_dev = uniPerplexity(obamadev, smoothed_obama_uni)
#print(trump_perp_uni_dev)
#print(obama_perp_uni_dev)
#
#trump_perp_uni_dev = uniPerplexity(trumptrain, smoothed_trump_uni)
#obama_perp_uni_dev = uniPerplexity(obamatrain, smoothed_obama_uni)
#
#class_lst_uni = uniClassification(trump_perp_uni_dev, obama_perp_uni_dev,smoothed_trump_uni, smoothed_obama_uni)
#print(class_lst_uni)
#makecsv("classuni", class_lst_uni)


