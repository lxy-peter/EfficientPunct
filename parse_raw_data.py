from helper import *
import nltk
from num2words import num2words
import os
import random
import re
import unidecode




considered_punct = ['.', ',', '?']



def remove_special(input_str, ignore):
    """
    Removes non-alphanumeric characters from a given string, but does not remove
    characters in ignore, spaces, and apostrophes/single quotes ['].
    Also cleans up bad formatting like trailing whitespaces, double spaces, etc.

    Parameters
    ----------
    input_str : str
        String from which to remove special characters
    ignore : list
        List of characters to not remove. By default, this function will not remove single spaces.

    Returns
    -------
    edited : str
        String with special characters removed
    """
    pattern = "[^a-zA-Z0-9' "
    for mark in ignore:
        if mark == '.':
            pattern += '\.'
        elif mark == '?':
            pattern += '\?'
        else:
            pattern += mark
    pattern += ']'
    target = list(re.finditer(pattern, input_str))
    target = [item.group() for item in target]
    target = set(target)
    edited = input_str
    for t in target:
        edited = edited.replace(t, ' ')
    
    edited = remove_double_spaces(edited)
    edited = edited.strip()
    
    for mark in ignore:
        while edited.find(' ' + mark) != -1:
            edited = edited.replace(' ' + mark, mark)
    
    pattern = '[0-9],[0-9]'
    target = list(re.finditer(pattern, edited))
    target = [item.group() for item in target]
    target = set(target)
    for t in target:
        replacement = t.replace(',', '')
        edited = edited.replace(t, replacement)
        
    return edited




def numbers2words(s):
    """
    Converts all numbers in a string to words

    Parameters
    ----------
    s : str
        Input string possibly containing numbers

    Returns
    -------
    str
        String with numbers converted to words

    """
    
    if has_numbers(s):
        
        s_list = s.split()
        for i in range(len(s_list)):
            
            word = s_list[i]
            while has_numbers(word):
                
                if not has_letters(word):
                    # Determine characters' index range which are numbers
                    num_started = False
                    decimal_seen = False
                    for j in range(len(word)):
                        if not num_started:
                            if word[j].isdigit():
                                start = j
                                end = j
                                num_started = True
                        else:
                            if word[j].isdigit():
                                end = j
                            elif word[j] == '.' and j != len(word)-1 and not decimal_seen:
                                end = j
                                decimal_seen = True
                            else:
                                break
                    to_convert = word[start:end+1]
                    
                    if '.' in to_convert:
                        num = float(to_convert)
                    else:
                        num = int(to_convert)
                    
                    if type(num) == int and num >= 1100 and num < 3000:
                        repl = num2words(num, to='year')
                    else:
                        repl = num2words(num)
                    repl = repl.replace('-', ' ')
                    
                    word = word[:start] + ' ' + repl + ' ' + word[end+1:]
                
                else:
                    repl = ''
                    for char in word:
                        if has_numbers(char):
                            repl += num2words(char) + ' '
                        else:
                            repl += char + ' '
                    repl = repl.strip()
                    word = repl
            
            s_list[i] = word
        
        s_rec = ''
        for word in s_list:
            s_rec += word + ' '
        s_rec = s_rec.strip()
        
        while s_rec.find('  ') != -1:
            s_rec = s_rec.replace('  ', ' ')
        
        return s_rec
    
    else:
        return s
    



def parse_singapore_text():
    all_singapore_data = ''
    for part_num in [1,2]:
        
        path = 'dataset/singapore_text/PART' + str(part_num) + '/DATA/CHANNEL0/SCRIPT'
        files = list_non_hidden(path)
        files.sort()
        
        for filename in files:
            f = open(path + '/' + filename, 'r')
            text = f.read()
            f.close()
            text = unidecode.unidecode(text)
            text = text.split('\n')
            text = [t.replace('\ufeff', '') for t in text]
            
            try:
                text.remove('')
            except ValueError:
                pass
            
            for i in range(len(text)):
                if i % 2 == 0:
                    line = text[i]
                    int(line[0:10])
                        
                    line = line[line.find('\t')+1:]
                    line = remove_special(line, considered_punct)
                    
                    all_singapore_data += line + '\n'
        
    f = open('dataset/compiled/all_singapore_data.txt', 'w')
    f.write(all_singapore_data)
    f.close()



def parse_gutenberg():
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    all_gutenberg_data = ''
    books_folders = ['ascii', 'utf-8']
    
    for folder in books_folders:
        path = 'dataset/gutenberg/books/' + folder
        subfolders = list_non_hidden(path)
        subfolders.sort()
        
        N = len(subfolders)
        i = 0
        
        for subfolder in subfolders:
            i += 1
            print('Processing ' + folder + ', ' + str(i) + '/' + str(N))
            
            file = list_non_hidden(path + '/' + subfolder)
            assert len(file) == 1
            
            try:
                f = open(path + '/' + subfolder + '/' + file[0], 'r')
                text = f.read()
            except UnicodeDecodeError:
                f = open(path + '/' + subfolder + '/' + file[0], 'r', encoding='latin-1')
                text = f.read()
            f.close()
            text = unidecode.unidecode(text)
            text = nltk_tokenizer.tokenize(text)
            text = [remove_special(t, considered_punct) for t in text]
            
            for t in text:
                all_gutenberg_data += t + '\n'
    
    f = open('dataset/compiled/all_gutenberg_data.txt', 'w')
    f.write(all_gutenberg_data)
    f.close()



def parse_mustc():
    f = open('dataset/mustc2-revised/all_text.txt', 'r')
    text = f.read()
    f.close()
    text = unidecode.unidecode(text)
    text = text.split('\n')
    
    text = [remove_special(t, considered_punct) for t in text]
    
    all_mustc2_data = ''
    for t in text:
        all_mustc2_data += t + '\n'
    all_mustc2_data = all_mustc2_data[:-1]
    
    f = open('dataset/compiled/all_mustc_data.txt', 'w')
    f.write(all_mustc2_data)
    f.close()


def parse_oanc():
    oanc_files = os.listdir('dataset/oanc')
    oanc_files = [f for f in oanc_files if not f.startswith('.')]
    
    sentences = []
    for i, o in enumerate(oanc_files):
        if i % 1000 == 0:
            print('Processing file', i+1, '/', len(oanc_files))
        
        f = open('dataset/oanc/' + o, 'r')
        text = f.read()
        f.close()
        text = unidecode.unidecode(text)
        text = text.replace('\n', ' ')
        while text.find('  ') != -1:
            text = text.replace('  ', ' ')
        text = remove_special(text, considered_punct)
        text += ' X'
        
        fs_pos = re.search('\. [A-Z0-9]', text)
        qm_pos = re.search('\? [A-Z0-9]', text)
        
        while fs_pos is not None or qm_pos is not None:
            if fs_pos is None and qm_pos is not None:
                pos = qm_pos.span()[0]
            elif fs_pos is not None and qm_pos is None:
                pos = fs_pos.span()[0]
            else:
                pos = min(fs_pos.span()[0], qm_pos.span()[0])
            
            sentences.append(text[:pos+1].strip())
            text = text[pos+1:]
            
            fs_pos = re.search('\. [A-Z0-9]', text)
            qm_pos = re.search('\? [A-Z0-9]', text)
    
    sentences = [s for s in sentences if len(s.split()) <= 35]
    
    out_str = ''
    for s in sentences:
        out_str += s + '\n'
    f = open('dataset/compiled/all_oanc_data.txt', 'w')
    f.write(out_str)
    f.close()
    

def clean_text(dataset, train_dev):
    """
    Parameters
    ----------
    dataset : str
        Can be 'gutensing', 'mustc', or 'oanc'
    train_dev : bool
        Specifies whether or not to split resulting data into training and validation sets
    """
    
    print('Reading in data...')
    
    if dataset == 'gutensing':
        f = open('dataset/compiled/all_gutenberg_data.txt', 'r')
        gutenberg = f.read().strip().split('\n')
        f.close
        f = open('dataset/compiled/all_singapore_data.txt', 'r')
        singapore = f.read().strip().split('\n')
        f.close()
        samples = gutenberg + singapore
        
    elif dataset == 'mustc':
        f = open('dataset/compiled/all_mustc_data.txt', 'r')
        samples = f.read().strip().split('\n')
        f.close()
    
    elif dataset == 'oanc':
        f = open('dataset/compiled/all_oanc_data.txt', 'r')
        samples = f.read().strip().split('\n')
        f.close()
    
    else:
        raise RuntimeError("dataset argument must be 'gutensing', 'mustc', or 'oanc'")
    
    
    samples = [s.strip() for s in samples]
    
    new_samples = []
    for s in samples:
        try:
            if s[-1].isalnum():
                new_samples.append(s + '.')
            else:
                new_samples.append(s)
        except IndexError:
            pass
    samples = new_samples

    print('Converting numbers to words...')
    
    new_samples = []
    for s in samples:
        new_samples.append(numbers2words(s))
    samples = new_samples
    new_samples = [s for s in samples]
    
    print('Concatenating samples...')
    
    for i in range(len(samples)):
        n = random.randint(3, 10)
        new_s = ''
        for j in range(i, i+n):
            try:
                new_s += samples[j] + ' '
            except IndexError:
                pass
        new_s = new_s.strip()
        new_samples.append(new_s)
    samples = new_samples
    
    print('Writing to file...')
    
    if train_dev:
        dev_idx = random.sample(range(len(samples)), k=50000)
    else:
        dev_idx = []
    train_idx = set(range(len(samples))) - set(dev_idx)
    
    if train_dev:
        print('Processing validation samples...')
        f = open('dataset/compiled/' + dataset + '_dev.txt', 'w')
        for i in dev_idx:
            f.write(samples[i] + '\n')
        f.close()
    
    if dataset == 'gutensing':
        file_counter = 0
        sample_counter = 0
        f = open('dataset/compiled/' + dataset + '_train' + str(file_counter) + '.txt', 'w')
        for i in train_idx:
            if i % 100000 == 0:
                print('Processing training sample', i+1, '/', len(train_idx))
            
            if sample_counter == 1000000:
                f.close()
                file_counter += 1
                f = open('dataset/compiled/' + dataset + '_train' + str(file_counter) + '.txt', 'w')
                sample_counter = 0
                
            f.write(samples[i] + '\n')
            sample_counter += 1
        f.close()
        
        print('Cleaning up training files...')
        for i in range(file_counter + 1):
            f = open('dataset/compiled/' + dataset + '_train' + str(file_counter) + '.txt', 'r+')
            text = f.read()[:-1]
            f.seek(0)
            f.write(text)
            f.truncate()
            f.close()
    else:
        f = open('dataset/compiled/' + dataset + '_train.txt', 'w')
        for i in train_idx:
            f.write(samples[i])
        f.close()

        
    