import kaldiio
from kaldiio import ReadHelper
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import re
from statistics import median
import struct
import subprocess


# The following function was obtained from
# https://gist.github.com/lukasklein/8c474782ed66c7115e10904fecbed86a#file-flacduration-py
def bytes_to_int(bytes: list) -> int:
        result = 0
        for byte in bytes:
            result = (result << 8) + byte
        return result


def cleanup_save(text, filename):
    while text.find("  ") != -1:
        text = text.replace("  ", " ")
    text = text.strip()
    
    # Write to file
    f = open(filename, "w")
    f.write(text)
    f.close()


def extract_punctuation(text):
    punct = text.lower()
    unpunct = text.lower()
    marks = get_non_alphanum(unpunct)
    for p in marks:
        if p != "'":
            unpunct = unpunct.replace(p, "")
    unpunct_words = unpunct.split()
    
    labels = []
    for word in unpunct_words:
        
        try:
            assert punct.find(word) == 0
        except AssertionError:
            # For debugging
            # print(word)
            # print(punct)
            # raise AssertionError
            return {'Input': [], 'Label': []}
            
        idx = len(word) # index of character immediately after word
        
        if idx == len(punct):
            labels.append(0)
            break
        else:
            next_word = idx
            while is_special_mark(punct[next_word]):
                next_word += 1
                
                if next_word == len(punct):
                    next_word = None
                    break
            
            symbols = list(punct[idx:next_word])
            
            is_space = [symbol == ' ' for symbol in symbols]
            
            # There is no punctuation (NP) after this word
            if all(is_space):
                labels.append(0)
            else:
                is_not_space_idx = is_space.index(False)
                symbol = symbols[is_not_space_idx]
                if symbol == '.':
                    labels.append(1)
                elif symbol == ',':
                    labels.append(2)
                elif symbol == '?':
                    labels.append(3)
                else:
                    raise AssertionError("Unrecognized symbol: " + symbol)
            
            punct = punct[next_word:]
    
    return labels


# The following function was obtained from
# https://gist.github.com/lukasklein/8c474782ed66c7115e10904fecbed86a#file-flacduration-py
def get_flac_dur(filename: str) -> float:
    """
    Returns the duration of a FLAC file in seconds
    https://xiph.org/flac/format.html
    """
    with open(filename, 'rb') as f:
        if f.read(4) != b'fLaC':
            raise ValueError('File is not a flac file')
        header = f.read(4)
        while len(header):
            meta = struct.unpack('4B', header)  # 4 unsigned chars
            block_type = meta[0] & 0x7f  # 0111 1111
            size = bytes_to_int(header[1:4])

            if block_type == 0:  # Metadata Streaminfo
                streaminfo_header = f.read(size)
                unpacked = struct.unpack('2H3p3p8B16p', streaminfo_header)
                """
                https://xiph.org/flac/format.html#metadata_block_streaminfo
                16 (unsigned short)  | The minimum block size (in samples)
                                       used in the stream.
                16 (unsigned short)  | The maximum block size (in samples)
                                       used in the stream. (Minimum blocksize
                                       == maximum blocksize) implies a
                                       fixed-blocksize stream.
                24 (3 char[])        | The minimum frame size (in bytes) used
                                       in the stream. May be 0 to imply the
                                       value is not known.
                24 (3 char[])        | The maximum frame size (in bytes) used
                                       in the stream. May be 0 to imply the
                                       value is not known.
                20 (8 unsigned char) | Sample rate in Hz. Though 20 bits are
                                       available, the maximum sample rate is
                                       limited by the structure of frame
                                       headers to 655350Hz. Also, a value of 0
                                       is invalid.
                3  (^)               | (number of channels)-1. FLAC supports
                                       from 1 to 8 channels
                5  (^)               | (bits per sample)-1. FLAC supports from
                                       4 to 32 bits per sample. Currently the
                                       reference encoder and decoders only
                                       support up to 24 bits per sample.
                36 (^)               | Total samples in stream. 'Samples'
                                       means inter-channel sample, i.e. one
                                       second of 44.1Khz audio will have 44100
                                       samples regardless of the number of
                                       channels. A value of zero here means
                                       the number of total samples is unknown.
                128 (16 char[])      | MD5 signature of the unencoded audio
                                       data. This allows the decoder to
                                       determine if an error exists in the
                                       audio data even when the error does not
                                       result in an invalid bitstream.
                """

                samplerate = bytes_to_int(unpacked[4:7]) >> 4
                sample_bytes = [(unpacked[7] & 0x0F)] + list(unpacked[8:12])
                total_samples = bytes_to_int(sample_bytes)
                duration = float(total_samples) / samplerate

                return duration
            header = f.read(4)




def get_non_alphanum(text):
    text = re.sub('[a-zA-Z0-9]', '', text)
    text = text.replace(' ', '')
    non_alphanum = set(text)
    
    return non_alphanum


def has_letters(in_str):
    return any(char.isalpha() for char in in_str)


def has_numbers(in_str):
    return any(char.isdigit() for char in in_str)


def is_special_mark(s):
    if s.isalnum():
        return False
    elif s == "'":
        return False
    else:
        return True


def list_non_hidden(dir_name):
    """
    Lists all non-hidden files in a directory
    
    Parameters
    ----------
    dir_name : str
        Path to directory containing files
    
    Returns
    -------
    files : list
        List of filenames
    """
    files = os.listdir(dir_name)
    files = [file for file in files if not file.startswith(".")]
    return files


def load_pkl(file):
    f = open(file, 'rb')
    var = pickle.load(f)
    f.close()
    return var


def read_kaldi(filename, filetype):
    contents = {}
    helper = ReadHelper(filetype + ':' + filename)
    for k, v in helper:
        contents[k] = list(v)
    return contents


def read_list_from_file(filename):
    f = open(filename, 'r')
    lst = f.read()
    f.close()
    lst = lst.split()
    try:
        lst.remove('')
    except ValueError:
        pass
    return lst


def read_mat(filename):
    f = open(filename, 'r')
    mat = f.read()
    f.close()
    mat = mat.split('\n')
    mat.remove('')

    lda_mat = np.ndarray((0, 1793))

    for i in range(len(mat)):
        row = mat[i]
        row = row.replace('[', '').replace(']', '')
        row = row.split()
        row = [float(num) for num in row]
        if len(row) != 0:
            lda_mat = np.vstack((lda_mat, np.array(row)))
    
    return lda_mat


def remove_double_spaces(s):
    while s.find("  ") != -1:
        s = s.replace("  ", " ")
    return s


def run_bash(command):
    """
    Executes a bash command
    """
    process = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


def save_pkl(var, file):
    f = open(file, 'wb')
    pickle.dump(var, f)
    f.close()



def split_by_sentence(text):
    sents = []
    while text != '':
        find_min = []
        fs_idx = text.find('.')
        if fs_idx != -1:
            find_min.append(fs_idx)
            
        qm_idx = text.find('?')
        if qm_idx != -1:
            find_min.append(qm_idx)
            
        nl_idx = text.find('\n')
        if nl_idx != -1:
            find_min.append(nl_idx)
        
        assert len(find_min) > 0
        idx = min(find_min)
        
        sent = text[:idx+1].strip()
        sents.append(sent)
        
        text = text[idx+1:]
        
        if len(sents) % 1000 == 0:
            print(len(sents), 'processed...')

    try:
        while True:
            sents.remove('')
    except ValueError:
        pass

    return sents




def write_list_to_file(lst, filename):
    """
    Saves the contents of a list to file, with elements separated by a space

    Parameters
    ----------
    lst : list
        List to save
    filename : str
        Path and filename of where to save
    """
    str_to_write = ''
    for l in lst:
        str_to_write += str(l) + ' '
    str_to_write = str_to_write.strip()
    f = open(filename, 'w')
    f.write(str_to_write)
    f.close()
    
    
    
    

