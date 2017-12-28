# -*- coding:utf-8 -*-
"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
sys.path.append('/home/hirose/git/deepfix')
import numpy as np
import os, random
from util.helpers import get_lines
from util.tokenizer import Java_Tokenizer, C_Tokenizer, EmptyProgramException, UnexpectedTokenException

class FixIDNotFoundInSource(Exception):
    pass

def rename_ids(corrupted_program, fix):
    corrupted_program_new = ''
    fix_new = ''
    
    name_dictionary = {}

    for token in corrupted_program.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                name_dictionary[token] = '_<id>_' + str(len(name_dictionary) + 1) + '@'
    
    for token in fix.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                raise FixIDNotFoundInSource
    
    # Rename
    for token in corrupted_program.split():
        if '_<id>_' in token:
            corrupted_program_new += name_dictionary[token] + " "
        else:
            corrupted_program_new += token + " "
    
    for token in fix.split():
        if '_<id>_' in token:
            fix_new += name_dictionary[token] + " "
        else:
            fix_new += token + " "
    
    return corrupted_program_new, fix_new
    
def generate_binned_training_data(max_program_length, min_program_length, max_fix_length, kind_mutations,\
                                  max_mutations, max_variants, mutations_series):    
    # FIXME tokenizerをjava用に変換
    tokenizer = Java_Tokenizer()
    #tokenizer = C_Tokenizer()
    tokenize = tokenizer.tokenize
    
    if kind_mutations == 'typo':
        keep_names = False
        from typo_mutator import LoopCountThresholdExceededException, FailedToMutateException,\
                                 token_mutate, token_mutate_series, token_mutate_series_any_fix, \
                                 get_mutation_distribution
    elif kind_mutations == 'ids':
        keep_names = True
        from undeclared_mutator import LoopCountThresholdExceededException, FailedToMutateException, \
                                       token_mutate, token_mutate_series, get_mutation_distribution
    else:
        raise Exception
    
    # Token strings
    token_strings = {'train': {}, 'validation': {}}
    program_lengths = []
    fix_lengths = []
    program_counts = {'train'   : {fold: 0 for fold in range(0, 5)},\
                      'validation' : {fold: 0 for fold in range(0, 5)}}
    
    program_lists  = {'train'   : {fold: [] for fold in range(0, 5)},\
                      'validation' : {fold: [] for fold in range(0, 5)}}
    
    exceptions_in_mutate_call = 0
    total_mutate_calls = 0

    for fold in range(0, 5):
        token_strings['train'][fold] = {'programs': [], 'fixes': []}
        token_strings['validation'][fold] = {'programs': [], 'fixes': []}

    for type_ in ['train', 'validation']:
        for fold in range(0, 5):
            basedir = os.path.join('/home/hirose/git/deepfix', 'data', type_, 'fold_%d' % fold)
            print basedir

            for source_file in os.listdir(basedir):
                print source_file
                with open(os.path.join(basedir, source_file), 'r+') as f:
                    source_code = f.read()
                    # FIXME tokenizeの実装をjava用に
                    tokenized_program, _, _, _ = tokenize(source_code, keep_literals=False, keep_names=keep_names)

                    # Correct pairs
                    program_length = len(tokenized_program.split())
                    program_lengths.append(program_length)
                    fix_length = 0
                    fix_lengths.append(fix_length)

                    # トークン列のidをrenameする
                    if program_length >= min_program_length and program_length <= max_program_length:
                        # FIXME tokenized_programはマスクされたものではダメでは?
                        # 正解なのでfixは無い
                        # TODO answerのコードはfixなしのデータとして使えるかも
                        cc_pair_prog, cc_pair_fix = rename_ids(tokenized_program, '')
                        print "cc_pair_prog"
                        print cc_pair_prog
                        print "cc_pair_fix"
                        print cc_pair_fix

                    token_strings[type_][fold]['programs'].append(cc_pair_prog)
                    token_strings[type_][fold]['fixes'].append(cc_pair_fix)
                    program_counts[type_][fold] += 1
                    program_lists[type_][fold].append(source_file)

                    total_mutate_calls += 1

                    # FIXME mutateをスキップする
                    class SkipException(Exception):
                        pass
                    try:
                        raise SkipException
                        # mutateする(データの増量化)
                        if mutations_series:
                            iterator = token_mutate_series(tokenized_program, max_mutations, max_variants)    
                        else:
                            iterator = token_mutate(tokenized_program, max_mutations, max_variants)
                         
                    except SkipException:
                        print "mutate step is skipped"
                        iterator = []
                    except FailedToMutateException:
                        print 'Failed to mutate', user_id, code_id
                        exceptions_in_mutate_call += 1
                    except LoopCountThresholdExceededException:
                        print 'Loop count threshold exceeded', user_id, code_id
                        exceptions_in_mutate_call += 1
                    except FixIDNotFoundInSource:
                        print 'Fix id not found in source', user_id, code_id
                        exceptions_in_mutate_call += 1
                    except ValueError: #### FOR NOW ####
                        exceptions_in_mutate_call += 1
                        if kind_mutations != 'ids':
                            raise
                    except AssertionError: #### FOR NOW ####
                        exceptions_in_mutate_call += 1
                        if kind_mutations != 'ids':
                            raise
                    except Exception:
                        print 'File:', source_file
                        print 'Source:', source_code

                        if kind_mutations != 'ids': #### FOR NOW ####
                            raise

                    for corrupted_program, fix in iterator:
                        try:
                            corrupted_program_new, fix_new = rename_ids(corrupted_program, fix)
                        except Exception:
                            if kind_mutations == 'ids': #### FOR NOW ####
                                continue
                        
                        corrupted_program_length = len(corrupted_program_new.split())
                        fix_length               = len(fix_new.split())
                        fix_lengths.append(fix_length)

                        if corrupted_program_length >= min_program_length and \
                           corrupted_program_length <= max_program_length and fix_length <= max_fix_length:
                            token_strings[type_][fold]['programs'].append(corrupted_program_new)
                            token_strings[type_][fold]['fixes'].append(fix_new)
                            program_counts[type_][fold] += 1
                            program_lists[type_][fold].append(source_file)

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print program_lengths
    print fix_lengths
    
    for fold in range(0, 5):
        assert(len(token_strings['validation'][fold]['programs']) == len(token_strings['validation'][fold]['fixes']))
        assert(len(token_strings['train'][fold]['programs']) == len(token_strings['train'][fold]['fixes']))
    
    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))]
    print 'Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))]
    print 'Total mutate calls:', total_mutate_calls
    print 'Exceptions in mutate() call:', exceptions_in_mutate_call, '\n'

    tokenizer.shutdown()

    return token_strings, get_mutation_distribution(), program_lists

def build_dictionary(token_strings, drop_ids_in_fix=False):
    tl_dict = {}

    for key in token_strings:
        for key2 in token_strings[key]:
            for tokens in token_strings[key][key2]['programs'] + token_strings[key][key2]['fixes']:
                length = 0

                for token in tokens.split():
                    length += 1
                    token = token.strip()

                    # tokenにIDを割り当てる
                    if token not in tl_dict:
                        tl_dict[token] = len(tl_dict) + 1

    if drop_ids_in_fix and '_<id>_@' not in tl_dict:
        tl_dict['_<id>_@'] = len(tl_dict) + 1
    
    # 予約語の追加
    tl_dict['_eos_'] = len(tl_dict) + 1
    tl_dict['_pad_'] = 0
    
    print 'Dictionary size:', len(tl_dict)
    
    rev_tl_dict = {}

    # tl_dictをコピーしてるだけ?
    for key, value in tl_dict.iteritems():
        rev_tl_dict[value] = key
    
    return tl_dict, rev_tl_dict

def load_dictionary(directory):
    tl_dict = np.load(os.path.join(directory, 'translate_dict.npy')).item()
    rev_tl_dict = np.load(os.path.join(directory, 'rev_translate_dict.npy')).item()
    
    return tl_dict, rev_tl_dict

def vectorize(tokens, vector_length, tl_dict, max_program_length, max_fix_length, drop_ids=False, reverse=False, vecFor='encoder'):    
    vec_tokens = []

    for token in tokens.split():
        if drop_ids and '_<id>_' == token[:6] and token[-1] == '@':
            vec_tokens.append(tl_dict['_<id>_@'])
        else:
            vec_tokens.append(tl_dict[token])        
    
    if vecFor == 'encoder':
        if len(vec_tokens) > max_program_length:
            return None
        
        # No _go_ or _eos_ required
        padding_length = vector_length - len(vec_tokens)
        padding = [tl_dict['_pad_']] * padding_length
    
        if reverse:
            vec_tokens = padding + vec_tokens[::-1]
        else:
            vec_tokens = padding + vec_tokens
            
    elif vecFor == 'decoder':    
        if len(vec_tokens) + 1 > max_fix_length:
            return None
        
        # For the _eos_ symbol (_go_ provided by network automagically)
        vector_length = vector_length + 1
        padding_length = vector_length - (len(vec_tokens) + 1)        
        padding = [tl_dict['_pad_']] * padding_length
        vec_tokens = vec_tokens + [tl_dict['_eos_']] + padding        
    
    return np.asarray(vec_tokens)

def vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids_in_fix=False):
    token_vectors = {}

    for key in token_strings.keys():
        token_vectors[key] = {}

        for key2 in token_strings[key].keys():
            token_vectors[key][key2] = {'programs': [], 'fixes': []}

    for key in token_strings.keys():
        for key2 in token_strings[key].keys():
            for prog_tokens, fix_tokens in zip(token_strings[key][key2]['programs'], token_strings[key][key2]['fixes']):
                prog_vector = vectorize(prog_tokens, max_program_length, tl_dict, max_program_length, max_fix_length, reverse=True, vecFor='encoder')
                fix_vector  = vectorize(fix_tokens, max_fix_length, tl_dict, max_program_length, max_fix_length, reverse=False, vecFor='decoder')

                if (prog_vector is not None) and (fix_vector is not None):
                    token_vectors[key][key2]['programs'].append(prog_vector)
                    token_vectors[key][key2]['fixes'].append(fix_vector)
        
            assert(len(token_vectors[key][key2]['programs']) == len(token_vectors[key][key2]['fixes']))
    
    print token_vectors
    return token_vectors

def save_pairs(destination, token_vectors):
    # Save pairs
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('mutated-%s.npy' % key)), token_vectors[key]['programs'])
        np.save(os.path.join(destination, ('fixes-%s.npy' % key)),   token_vectors[key]['fixes'])

def save_folds(destination, token_vectors):
    full_list = []

    for i in range(0, 5):
        token_vectors_this_fold = {'train': {'programs': [], 'fixes': []},
                                   'validation': {'programs': [], 'fixes': []},
                                   'test': {'programs': [], 'fixes': []}}

        token_vectors_this_fold['train']['programs']      += token_vectors['train'][i]['programs']
        token_vectors_this_fold['train']['fixes']         += token_vectors['train'][i]['fixes']
        token_vectors_this_fold['validation']['programs'] += token_vectors['validation'][i]['programs']
        token_vectors_this_fold['validation']['fixes']    += token_vectors['validation'][i]['fixes']

        for j in range(0, 5):
            if i != j:
                token_vectors_this_fold['test']['programs'] += token_vectors['validation'][j]['programs']
                token_vectors_this_fold['test']['fixes']    += token_vectors['validation'][j]['fixes']

        try:
            os.makedirs(os.path.join(destination, 'fold_%d' % i))
        except OSError:
            pass

        print 'Fold %d: %d train, %d validation and %d test' % (i, len(token_vectors_this_fold['train']['programs']), len(token_vectors_this_fold['validation']['programs']), len(token_vectors_this_fold['test']['programs']))
        save_pairs(os.path.join(destination, 'fold_%d' % i), token_vectors_this_fold)

    print ''
    
def save_dictionaries(destination, tl_dict, rev_tl_dict):
    np.save(os.path.join(destination, 'translate_dict.npy'), tl_dict)
    np.save(os.path.join(destination, 'rev_translate_dict.npy'), rev_tl_dict)

def load_validation_users(destination):
    return np.load(os.path.join(destination, 'validation_users.npy')).item()
    
def save_validation_users(destination, validation_users):
    np.save(os.path.join(destination, 'validation_users.npy'), validation_users)

def save_test_problems(destination, test_problems):
    np.save(os.path.join(destination, 'test_problems.npy'), test_problems)    

def DEBUG_STOP():
    print "For debugging program stop"
    exit()
