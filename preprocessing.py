# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:33:40 2019

@author: Rufina
"""

from os import listdir
import random
import pickle

FOLDERS_NUMB = 17
PAIRS_NUMB = 1
PATH_TO_ORIGINAL_DATA = 'data\Image'

add_string = lambda str1, str2: str1+str2

DAY_PATHS = []
NIGHT_PATHS = []
counter = 0
original_data = listdir(PATH_TO_ORIGINAL_DATA)
for i in range(FOLDERS_NUMB):
    day_paths = listdir(f'{PATH_TO_ORIGINAL_DATA}\{original_data[i]}\day')
    night_paths = listdir(f'{PATH_TO_ORIGINAL_DATA}\{original_data[i]}\\night')
    DAY_PATHS += list(map(lambda string: f'{original_data[i]}\day\\'+string, random.sample(day_paths, PAIRS_NUMB)))
    NIGHT_PATHS += list(map(lambda string: f'{original_data[i]}\\night\\'+string, random.sample(night_paths, PAIRS_NUMB)))

with open('day_paths.pickle', 'wb') as f:
     pickle.dump(DAY_PATHS, f)
with open('night_paths.pickle', 'wb') as f:
     pickle.dump(NIGHT_PATHS, f)
    
    
    