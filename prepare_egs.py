import argparse
from helper import *
import numpy as np
import os
import pickle




parser = argparse.ArgumentParser(description="Prepare training examples")
parser.add_argument('directory', help='Name of directory inside data/')
parser.add_argument('n_features', help='Number of features to use')
directory = parser.parse_args().directory.strip()
n_features = parser.parse_args().n_features.strip()



OFF_CENTER = 150
f = open('conf/off_center.txt', 'w')
f.write(str(OFF_CENTER))
f.close()




words_dict = {}
dicts = [f for f in os.listdir('embed_' + directory) if '.dict' in f]
for file in dicts:
    f = open('embed_' + directory + '/' + file, 'rb')
    current = pickle.load(f)
    f.close()
    for k in current:
        words_dict[k] = current[k]
    
    
scps = [file for file in list_non_hidden('embed_' + directory + '/' + n_features) if file.endswith('.scp')]
try:
    scps.remove('embed.scp')
except ValueError:
    pass
scps.sort()



for scp in scps:
    print('Reading data of', scp)
    embed = read_kaldi('embed_' + directory + '/' + n_features + '/' + scp, 'scp')
    uttids = {}
    for i, k in enumerate(embed):
        k_rev = k[::-1]
        last_dash_idx = len(k) - k_rev.find('-') - 1
        uttid = k[:last_dash_idx]
        if uttid not in uttids:
            uttids[uttid] = []
        uttids[uttid].append(k)
    
    # To write .feat file for the current .scp
    for j, uttid in enumerate(uttids):
        embed_keys = uttids[uttid]
        digits = len(str(len(embed_keys)))
        for i in range(len(embed_keys)):
            assert embed_keys[i] == uttid + '-' + str(i).zfill(digits)
        feats = np.vstack([embed[k] for k in embed_keys])
        
        assert len(words_dict[uttid]) == feats.shape[0]
        
        f = open('embed_' + directory + '/' + n_features + '/egs/' + uttid + '.feat', 'wb')
        pickle.dump(feats, f)
        f.close()
   
    egs_f_name = scp.replace('.scp', '.txt')
    print('Writing to', egs_f_name)
    egs_f = open('embed_' + directory + '/' + n_features + '/egs_txt/' + egs_f_name, 'w')
    # Each line in this file is of the format:
    # <utterance-id> <start> <label>
    # <start> indicates the first 1792-vector to be included from the features.
    # Including <start>, 301 1792-vectors will be used in the training example.
    # <start> and <end> are zero indexed
    # <label> indicates the punctuation at the center of start to end
    for uttid in uttids:
        words = words_dict[uttid]
        prev_word_idx = 0
        for i in range(len(words)):
            word_idx = words[i][1]
            if word_idx != prev_word_idx:
                assert i != 0
                egs_f.write(uttid + ' ' + str(i-OFF_CENTER-1) + ' ' + str(words[i-1][2]) + '\n')
            prev_word_idx = word_idx
        egs_f.write(uttid + ' ' + str(i - OFF_CENTER) + ' ' + str(words[-1][2]) + '\n')
        
    egs_f.close()
    
    
    
egs_txt_files = list_non_hidden('embed_' + directory + '/' + n_features + '/egs_txt')
try:
    egs_txt_files.remove('egs.txt')
except ValueError:
    pass
egs_txt_files.sort()

big_egs_txt_f = open('embed_' + directory + '/' + n_features + '/egs_txt/egs.txt', 'w')
for file in egs_txt_files:
    f = open('embed_' + directory + '/' + n_features + '/egs_txt/' + file, 'r')
    lines = f.read().split('\n')
    f.close()
    try:
        while True:
            lines.remove('')
    except ValueError:
        pass
    
    for line in lines:
        big_egs_txt_f.write(line + '\n')
        
big_egs_txt_f.close()