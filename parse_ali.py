import argparse
from bert_fine_tune_train import BERTFineTuneForPunct
from helper import *
import kaldiio
import numpy as np
import os
from parse_raw_data import remove_special
import pickle
import statistics
from transformers import BertTokenizer
import torch




parser = argparse.ArgumentParser(description="Parse alignments")
parser.add_argument('dataset', help='singapore or name of other dataset')
parser.add_argument('directory', help='Name of directory inside data/')
dataset = parser.parse_args().dataset
directory = parser.parse_args().directory


device = "cuda:0" if torch.cuda.is_available() else "cpu"
utts_processed = 0


# Getting int2word and word2int for Kaldi language
f = open('data/lang/words.txt', 'r')
wordints = f.read().split('\n')
try:
    while True:
        wordints.remove('')
except ValueError:
    pass
wordints = [line.split() for line in wordints]
# int2word = {int(line[1]):line[0] for line in wordints}
word2int = {line[0]:int(line[1]) for line in wordints}
    
    
    

if dataset == 'singapore':
    all_text = load_pkl('dataset/singapore_scripts.pkl')



# Aligning phones with words
def ali_phones_words():
    """
    Performs phone-word-alignment

    Returns
    -------
    phones : dict
        Aligned phones and words
    words : dict
        Transcribed words in each utterance
    """
    
    print('Time-phone-word alignment starting')
    
    f = open('data/' + directory + '/text', 'r')
    lines = f.read().split('\n')
    f.close()
    try:
        while True:
            lines.remove('')
    except ValueError:
        pass
    lines = [line.split() for line in lines]
    words = {line[0] : line[1:] for line in lines}
    for k in words:
        words_list = []
        for word in words[k]:
            try:
                word_int = word2int[word]
            except KeyError:
                word_int = word2int['<unk>']
            words_list.append(word_int)
        words[k] = words_list
    
    # Reading in decode_phones.ctm
    ali_dir = 'exp/chain_cleaned_1d/tdnn1d_sp_' + directory + '_ali'
    ctms = [item for item in list_non_hidden(ali_dir) if item.endswith('.ctm')]
    ctms.sort()
    decode_phones_ctm = []
    for ctm in ctms:
        f = open(ali_dir + '/' + ctm, 'r')
        current = f.read().split('\n')
        try:
            while True:
                current.remove('')
        except ValueError:
            pass
        decode_phones_ctm.extend(current)
    
    # Format of phones[utt_id]:
    # [start_time, duration, phone, word, word_idx]
    phones = {}
    for line in decode_phones_ctm:
        sep = line.find(' ')
        key = line[:sep].strip()
        value = line[sep+1:].strip().split()
        
        if key not in phones:
            phones[key] = [[3*float(value[1]), 3*float(value[2]), int(value[3])]]
        else:
            phones[key].append([3*float(value[1]), 3*float(value[2]), int(value[3])])
    
    
    # Reading in phone sequence for each word
    f = open('data/lang_rescore/phones/align_lexicon.int', 'r')
    align_lexicon = f.read().split('\n')
    f.close()
    try:
        while True:
            align_lexicon.remove('')
    except ValueError:
        pass
    
    word2phones = {}
    for line in align_lexicon:
        line_list = line.split()[1:]
        line_list = [int(l) for l in line_list]
        word = line_list[0]
        phone_seq = line_list[1:]
        if word not in word2phones:
            word2phones[word] = [phone_seq]
        else:
            word2phones[word].append(phone_seq)
    
    
    # Assigning words to each phone for each utterance
    for key in phones:
        
        word_seq = words[key]
        
        i = 0
        
        for word_i in range(len(word_seq)):
            
            word = word_seq[word_i]
            
            phone_seqs = word2phones[word]
            phone_seqs_len = set(len(seq) for seq in phone_seqs)
            seq_found = False
            
            while not seq_found:
                
                for l in phone_seqs_len:
                    for seq in phone_seqs:
                        if len(seq) == l:
                            
                            if [phones[key][j][2] for j in range(i, min(i+l,len(phones[key])))] == seq:
                                for j in range(i, i+l):
                                    phones[key][j] += [word, word_i]
                                seq_found = True
                                i += l
                                
                        if seq_found:
                            break
                    if seq_found:
                        break
                    
                if not seq_found:
                    try:
                        prev_word = phones[key][i-1][3]
                    except IndexError:
                        prev_word = None
                    if prev_word:
                        prev_word_i = phones[key][i-1][4]
                        phones[key][i] += [prev_word, prev_word_i]
                    i += 1
        
        for j in range(i, len(phones[key])):
            try:
                prev_word = phones[key][j-1][3]
            except IndexError:
                prev_word = None
            if prev_word:
                prev_word_i = phones[key][j-1][4]
                phones[key][j] += [prev_word, prev_word_i]
                
    print('Time-phone-word alignment complete')
    return phones, words



def concat_one_ark(words_dict, ali_dict, kaldi_dict, bert_tokenizer, bert_model):
    embeddings_dict = {}
    embeddings_words_dict = {}
    embeddings_labels_dict = {}
    global utts_processed
    total_utts = len(words_dict)
    
    def find_match(l1, l2):
        """
        Returns indices of matched elements between l1 and l2 in a tuple
        """
        for x in range(len(l1)):
            for y in range(len(l2)):
                if l1[x] == l2[y]:
                    return (x, y)
        return None
    
    def first_consec(l):
        """
        Returns index of last consecutively increasing integer in l
        """
        if len(l) == 1:
            return 0
        for i in range(1, len(l)):
            if l[i] - l[i-1] != 1:
                return i - 1
        return i
    
    fail_count = 0
    
    
    for k in kaldi_dict.keys():
        if utts_processed % 100 == 0:
            print('Processing utterance', utts_processed + 1, '/', total_utts)
        
        try:
            ali = ali_dict[k]
            kaldi = kaldi_dict[k]
        except KeyError:
            fail_count += 1
            continue
        
        if dataset == 'singapore':
            actual_k_idx = k.find('-')
            actual_k = k[actual_k_idx+1:]
            text = all_text[actual_k]
        else:
            f = open('db/' + directory + '_text/' + k + '.txt', 'r')
            text = f.read().strip()
            f.close()
        
        labels = extract_punctuation(text)
        text = text.lower()
        text = remove_special(text, [])
        words_list = text.split()
        assert len(words_list) > 0
        
        # print('BERT embeddings computation is starting')
        
        inputs_word_tokens = bert_tokenizer.tokenize(text)
        inputs_num_tokens = bert_tokenizer(text, return_tensors='pt')
        inputs_num_tokens = inputs_num_tokens.to(device)
        N = inputs_num_tokens['input_ids'].shape[1]
        
        if N > 512:
            
            inputs_num_tokens = inputs_num_tokens['input_ids'][:, 1:-1]
            
            # List containing indices of split locations. Each element
            # is index of first token in a group.
            split_loc = [0]
            while inputs_num_tokens.shape[1] - split_loc[-1] > 510:
                for i in range(split_loc[-1] + 510, split_loc[-1], -1):
                    if inputs_word_tokens[i] in words_list:
                        break
                split_loc.append(i)
            assert len(split_loc) >= 2
            
            bert = torch.zeros([N-2, 768], dtype=torch.float32)
            next_empty_row_idx = 0
            
            for i in range(len(split_loc)):
                begin = split_loc[i]
                
                if i != len(split_loc) - 1:
                    end = split_loc[i+1]
                else:
                    end = inputs_num_tokens.shape[1]
                
                current_input_ids = torch.hstack((torch.tensor([[101]]).to(device),
                                                  inputs_num_tokens[:, begin:end],
                                                  torch.tensor([[102]]).to(device)))
                current_inputs_num_tokens = {'input_ids': current_input_ids,
                                             'token_type_ids': torch.zeros((1, current_input_ids.shape[1]), dtype=torch.int64).to(device),
                                             'attention_mask': torch.ones((1, current_input_ids.shape[1]), dtype=torch.int64).to(device)}
                current_bert = bert_model.bert_last_hidden(current_inputs_num_tokens)
                current_bert = torch.squeeze(current_bert, dim=0)
                
                bert[begin:end, :] = current_bert
            
            assert end == bert.shape[0]
            
        else:
            bert = bert_model.bert_last_hidden(inputs_num_tokens)
            bert = torch.squeeze(bert, dim=0)
        # print('BERT embeddings computation is ending')
        
        # print('Token grouping is starting')
        word_token = 0
        token_groups = []
        for word in words_list:
            span = ''
            tokens = []
            while span != word:
                span += inputs_word_tokens[word_token].replace('#', '')
                tokens.append(word_token)
                word_token += 1
            token_groups.append(tokens)
        
        assert len(token_groups) == len(words_list)
        assert token_groups[-1][-1] == bert.shape[0] - 1
        
        
        # Elements of bert_words contains BERT embeddings for each word
        bert_words = []
        for group in token_groups:
            if len(group) == 1:
                bert_words.append(bert[group[0], :].detach().cpu().numpy())
            elif len(group) > 1:
                assert max(group) == group[-1]
                bert_words.append(bert[group[-1], :].detach().cpu().numpy())
            else:
                raise RuntimeError('Length of token group for a word is less than 1')
        # print('Token grouping is ending')   
        
        # print('Kaldi-BERT words matching is starting')
        kaldi_words = words_dict[k]
        
        words_list_tmp = []
        for word in words_list:
            try:
                word_int = word2int[word]
            except KeyError:
                word_int = word2int['<unk>']
            words_list_tmp.append(word_int)
        words_list = words_list_tmp
        
        # Dictionary mapping indices in kaldi_words to indices in words_list
        kaldi_words_tmp = [word for word in kaldi_words]
        words_list_tmp = [word for word in words_list]
        
        kaldi2words = {}
        
        kaldi_first = 0
        words_first = 0
        i = 1
        while i <= max(len(kaldi_words_tmp), len(words_list_tmp)):
            kaldi_check = kaldi_words_tmp[:i]
            words_check = words_list_tmp[:i]
            match = find_match(kaldi_check, words_check)
            if match:
                kaldi2words[kaldi_first + match[0]] = words_first + match[1]
                kaldi_first += match[0] + 1
                kaldi_words_tmp = kaldi_words_tmp[match[0]+1:]
                words_first += match[1] + 1
                words_list_tmp = words_list_tmp[match[1]+1:]
                i = 1
            else:
                i += 1
        
        missing = [i for i in range(len(kaldi_words)) if i not in kaldi2words]
        assert len(missing) == 0
        
        # while len(missing) > 0:
        #     seq_last = first_consec(missing)
        #     current = missing[:seq_last+1]
        #     # lower indicates the lowest index in words_list that can be used in missing
        #     if current[0] - 1 >= 0:
        #         lower = kaldi2words[current[0] - 1] + 1
        #         lower = min(lower, len(words_list)-1)
        #     else:
        #         lower = 0
            
        #     # upper indicates the highest index in words_list that can be used in missing
        #     if current[-1] + 1 < len(kaldi_words):
        #         upper = kaldi2words[current[-1] + 1] - 1
        #         upper = max(upper, 0)
        #     else:
        #         upper = len(words_list) - 1
                
        #     upper = max(upper, lower)
            
        #     range_allowed = list(range(lower, upper+1))
        #     if len(range_allowed) == 0:
        #         lower = max(0, lower - 1)
        #         upper = min(len(words_list) - 1, upper + 1)
                
            
        #     if len(range_allowed) == 1:
        #         for c in current:
        #             kaldi2words[c] = range_allowed[0]
            
        #     elif len(range_allowed) > 1:
                
        #         if len(current) == 1:
        #             kaldi2words[current[0]] = int(statistics.median(range_allowed))
        #         elif len(current) < 1:
        #             raise RuntimeError('len(current) < 1')
        #         else:
        #             range_mapped = [r - min(range_allowed) for r in range_allowed]
        #             range_mapped = [r / max(range_mapped) for r in range_mapped]
                    
        #             current_mapped = [c - min(current) for c in current]
        #             current_mapped = [c / max(current_mapped) for c in current_mapped]
                    
        #             for i in range(len(current)):
        #                 c = current_mapped[i]
                        
        #                 min_diff = 99999
        #                 min_j = None
                        
        #                 for j in range(len(range_allowed)):
        #                     to = range_mapped[j]
                            
        #                     if abs(c - to) < min_diff:
        #                         min_diff = abs(c - to)
        #                         min_j = j
                                
        #                 kaldi2words[current[i]] = range_allowed[min_j]
            
        #     else:
        #         raise RuntimeError('len(ranged_allowed) < 1')
            
        #     missing = [i for i in range(len(kaldi_words)) if i not in kaldi2words]
        
        # print('Kaldi-BERT words matching is ending')
        
        # print('Embedding concatenation is starting')
        # This loop determines the size of embeddings to allocate
        total_rows = 0
        for seg in ali:
            if len(seg) == 5:
                start = round(seg[0] * 100)
                end = round(seg[1] * 100) + start
                end = min(end, kaldi.shape[0])
                current_rows = end - start
                if current_rows <= 0:
                    continue
            elif len(seg) == 3:
                pass
                current_rows = 0
            else:
                raise RuntimeError('Length of seg in ali is neither 3 nor 5, which is an error')
            total_rows += current_rows
        
        if total_rows == 0:
            continue
        
        embeddings = np.ndarray((total_rows, 1792))
        embeddings_words = []
        embeddings_labels = []
        
        next_empty_row_idx = 0
        
        fail = False
        for i, seg in enumerate(ali):
            
            if len(seg) == 5:
                start = round(seg[0] * 100)
                end = round(seg[1] * 100) + start
                end = min(end, kaldi.shape[0])
                current_rows = end - start
                if current_rows <= 0:
                    continue
                
                word_idx = kaldi2words[seg[4]]
                bert_embed = bert_words[word_idx]
                bert_embed = np.reshape(bert_embed, (1, 768))
                kaldi_embed = kaldi[start:end, :]
                bert_embed = np.repeat(bert_embed, kaldi_embed.shape[0], axis=0)
                embed = np.hstack((bert_embed, kaldi_embed))
                
                embeddings[next_empty_row_idx : next_empty_row_idx + embed.shape[0]] = embed
                next_empty_row_idx += embed.shape[0]
                
                try:
                    embeddings_words += [(words_list[word_idx], seg[4], labels[word_idx]) for _ in range(embed.shape[0])]
                except KeyError:
                    fail = True
                    break
                embeddings_labels += [labels[word_idx] for _ in range(embed.shape[0])]
                
            elif len(seg) == 3:
                pass
            
            else:
                raise RuntimeError('Length of seg in ali is neither 3 nor 5, which is an error')
        
        if fail:
            fail_count += 1
            continue
        
        assert embeddings.shape[0] <= kaldi.shape[0]
        assert embeddings.shape[0] == len(embeddings_words)
        
        # print('Embedding concatenation is ending')
        
        # Each value in embeddings_words_dict is a tuple (word, index, label)
        embeddings_dict[k] = embeddings
        embeddings_words_dict[k] = embeddings_words
        embeddings_labels_dict[k] = embeddings_labels
        
        utts_processed += 1
        
    # FOR TESTING BEGIN
    # associations = []
    # keys = list(kaldi2words.keys())
    # keys.sort()
    # for i in range(len(keys)):
    #     k = keys[i]
    #     associations.append((kaldi_words[k], words_list[kaldi2words[k]], i))
    
    # associations = [(int2word[a[0]], int2word[a[1]], a[2]) for a in associations]
    # FOR TESTING END
    
    print('Successfully aligned', len(embeddings_dict), 'utterances, failed', fail_count)
        
    return embeddings_dict, embeddings_words_dict, embeddings_labels_dict



def concat_embed():
    
    # Getting alignments and words in each utterance
    ali_dict, words_dict = ali_phones_words()
    
    # Getting BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    # Getting BERT model
    bert_model_path = 'bert/bert.pt'
    bert_model = BERTFineTuneForPunct()
    if torch.cuda.is_available() and 'cuda' in device:
        bert_model.load_state_dict(torch.load(bert_model_path))
    else:
        bert_model.load_state_dict(torch.load(bert_model_path, map_location=torch.device('cpu')))
    bert_model = bert_model.to(device)
    bert_model.eval()
    
    
    # Creating utt2spk file (contains punctuation labels for each word)
    f = open('embed_' + directory + '/utt2spk', 'w')
    f.close()
    
    # Getting Kaldi embeddings
    output12_dir = 'exp/chain_cleaned_1d/tdnn1d_sp/output12_' + directory + '/split_scp'
    scps = os.listdir(output12_dir)
    scps = [s for s in scps if '.scp' in s]
    try:
        scps.remove('output.scp')
    except ValueError:
        pass
    scps.sort()
    
    for a in scps:
        print('Processing', a)
        
        ark = read_kaldi(output12_dir + '/' + a, 'scp')
        kaldi_dict = {}
        
        for i, k in enumerate(ark):
            if i % 1000 == 0:
                print('Processing', a, '| stacking', i+1, '/', len(ark))
            kaldi_dict[k] = np.vstack(ark[k])
        del ark
        
        # Each value in embeddings_words_dict is a tuple (word, index, label)
        embeddings_dict, embeddings_words_dict, embeddings_labels_dict = \
            concat_one_ark(words_dict, ali_dict, kaldi_dict, bert_tokenizer, bert_model)
        
        f = open('embed_' + directory + '/utt2spk', 'a')
        out_ark = {}
        for i, k in enumerate(embeddings_dict):
            if i % 1000 == 0:
                print('Processing', a, '| utterance', i+1, '/', len(embeddings_dict))
            embeddings = embeddings_dict[k]
            labels = embeddings_labels_dict[k]
            
            digits = len(str(embeddings.shape[0]))
            
            for i in range(embeddings.shape[0]):
                utt_id = k + '-' + str(i).zfill(digits)
                out_ark[utt_id] = embeddings[i,:]
    
                f.write(utt_id + ' ' + str(labels[i]) + '\n')
        f.close()
        
        print('Saving embeddings for', a)
        
        new_name = a.replace('output', 'embed')
        ark_name = new_name.replace('.scp', '.ark')
        kaldiio.save_ark('embed_' + directory + '/1792/' + ark_name, out_ark, scp='embed_'+directory+'/1792/'+new_name)
    
        new_name = a.replace('output', 'words').replace('scp', 'dict')
        f = open('embed_' + directory + '/' + new_name, 'wb')
        pickle.dump(embeddings_words_dict, f)
        f.close()
    
    del out_ark
    
    # Combines contents of each separate embed file into embed.scp
    scps = os.listdir('embed_' + directory + '/1792')
    scps = [file for file in scps if '.scp' in file]
    combined = ''
    for scp in scps:
        f = open('embed_' + directory + '/1792/' + scp, 'r')
        combined += f.read()
        f.close()
    f = open('embed_' + directory + '/1792/embed.scp', 'w')
    f.write(combined)
    f.close()
    
    

concat_embed()


