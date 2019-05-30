import re
import numpy as np
from collections import Counter
import csv


""" given a file path returns a list of paragraphs"""
def sep_paras(file_path):
    lst = []
    opn = open(file_path.format(0)).read()
    paralist = opn.split("\n")
    for para in paralist:
        tknzd = para.replace("’ ","’")
        tknzd = para.split(" ")
        lst.append(tknzd)
    return lst

paras = sep_paras('Assignment1_resources/test/test.txt')
print(paras)