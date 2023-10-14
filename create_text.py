import argparse
from helper import *
from parse_raw_data import remove_special


parser = argparse.ArgumentParser(description="Creating text file in data directory")
parser.add_argument('dataset', help='singapore or name of other dataset')
parser.add_argument('directory', help='Name of directory inside data/')
dataset = parser.parse_args().dataset
directory = parser.parse_args().directory


text_f = open('data/' + directory + '/text', 'w')
if dataset == 'singapore':
    texts = load_pkl('dataset/singapore_scripts.pkl')
    
    f = open('data/' + directory + '/wav.scp', 'r')
    wav_scp = f.read().split('\n')
    f.close()
    try:
        while True:
            wav_scp.remove('')
    except ValueError:
        pass
    uttids = []
    for line in wav_scp:
        space_idx = line.find(' ')
        uttids.append(line[:space_idx])
    uttids.sort()
    
    for uttid in uttids:
        dash_idx = uttid.find('-')
        assert dash_idx != -1
        pkl_id = uttid[dash_idx+1:]
        transcript = texts[pkl_id]
        
        transcript = transcript.lower()
        transcript = remove_special(transcript, [])
        
        text_f.write(uttid + ' ' + transcript + '\n')

else:
    texts = [file for file in list_non_hidden('db/' + directory + '_text') if '.' in file]
    texts.sort()
    
    for t in texts:
        f = open('db/' + directory + '_text/' + t, 'r')
        transcript = f.read().strip()
        f.close()
        
        transcript = transcript.lower()
        transcript = remove_special(transcript, [])
        
        text_f.write(t[:-4] + ' ' + transcript + '\n')
        
text_f.close()