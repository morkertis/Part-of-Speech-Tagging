"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import itertools, operator
from collections import defaultdict
from math import log, isfinite
from collections import Counter
import sys, os, time, platform, nltk
from nltk.util import flatten
from random import choices 
import numpy as np

#global param
epsilon = sys.float_info.epsilon
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()

# use Counters inside these
perWordTagCounts = defaultdict(lambda: defaultdict(float))#{}
transitionCounts = defaultdict(lambda: defaultdict(float))#{}
emissionCounts = defaultdict(lambda: defaultdict(float))#{}

# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = defaultdict(lambda: defaultdict(float))#{} #transisions probabilities
B = defaultdict(lambda: defaultdict(float))#{} #emmissions probabilities



def read_tagged_sentence(f):
    '''
    Return list in format [(w0,t0)....(wi,ti)] where w: word as string, t: tag as string.
    read line from text file and sperate by tab.
    Args:
        f(file) file that open and read from hin line
    Return sentence(list) in the format above
    '''
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence


def read_tagged_corpus(filename):
    '''
    Returns list of list where each inner list is a sentence in format [(w0,t0)....(wi,ti)]
    w: word as string, t: tag as string
    Args:
        filename(str): text file where the data located
    Return sentences (list). each sentence is the foramt as describe 
    '''
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


def learn(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
      and emissionCounts data-structures.
      allTagCounts and perWordTagCounts should be used for baseline tagging and
      should not include pseudocounts, dummy tags and unknowns.
      The transisionCounts and emissionCounts
      should be computed with pseudo tags and shoud be smoothed.
      A and B should be the log-probability of the normalized counts, based on
      transisionCounts and  emissionCounts
    
      Args:
        tagged_sentences: a list of tagged sentences, each tagged sentence is a
         list of pairs (w,t), as retunred by read_tagged_corpus().
    
     Returns:
        [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    #added global statement for uniform values for all the program
    global allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B
    
    #initial the main param
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = init_main_param(allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B)

    for sentence in tagged_sentences:
        
        word_first, tag_first = sentence[0]
        allTagCounts[tag_first] += 1
        perWordTagCounts[word_first][tag_first] += 1
        transitionCounts[START][tag_first] += 1
        emissionCounts[tag_first][word_first] += 1
        
        add_unknown(word_first,tag_first)#add unknown for word and tag counts
        
        tag_last=tag_first
        
        for word, tag in sentence[1:]:
           allTagCounts[tag] += 1
           perWordTagCounts[word][tag] += 1
           emissionCounts[tag][word] += 1
           transitionCounts[tag_last][tag] += 1
           add_unknown(word,tag)#add unknown for word and tag counts
           tag_last = tag
              
        transitionCounts[tag_last][END] += 1
    A = transitionCounts.copy()
    B = emissionCounts.copy()
    A,B = normilize_A_B(A,B)
    
    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]


def init_main_param(allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B):
    '''
    return the initial main program params that in the start of the code.
    Args:
        All the main params as describe above
    Returns:
        the params initialize
    '''
    allTagCounts = Counter()
    
    # use Counters inside these
    perWordTagCounts = defaultdict(lambda: defaultdict(float))#{}
    transitionCounts = defaultdict(lambda: defaultdict(float))#{}
    emissionCounts = defaultdict(lambda: defaultdict(float))#{}
    
    # log probability distributions: do NOT use Counters inside these because
    # missing Counter entries default to 0, not log(0)
    A = defaultdict(lambda: defaultdict(float))#{} #transisions probabilities
    B = defaultdict(lambda: defaultdict(float))#{} #emmissions probabilities
    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]
    
def normilize_A_B(A,B):
    '''
    Returns A transisions log-probabilities "matrix" and B emmissions log-probabilities "matrix".
    A in format of {'tag_previous':{'tag_current':log-prob}}. B in format {'tag':{'word':log-prob}}
    Args:
        A,B (ditonary in dictionary): in the format above
    Returns A,B
    '''
    #A normilize
    for key in A.keys():
        amount = sum(A[key].values())
        for key2 in A[key].keys():
            A[key][key2] = log(A[key][key2]/amount)
    
    #B normilize
    for key in B.keys():
        amount = sum(B[key].values())
        for key2 in B[key].keys():
            B[key][key2] = log(B[key][key2]/amount)
    return A,B
    

def add_unknown(word,tag):
    '''
    Update the global main param counter with UNK (unknown) tag/word with there contexts.
    context can be other word or other tag.
    Args:
        word(str): context of the emissionCounts param
        tag(str): context of the transitionCounts param
    '''
    global transitionCounts,emissionCounts, START, END, UNK
    
    if not emissionCounts.get(UNK,{}).get(UNK,0):
        emissionCounts[UNK][UNK] = 1 #conditional count word unknown given tag unknown     
        transitionCounts[UNK][UNK] = 1 #tag unknown to tag unknown
    
    if not emissionCounts.get(tag,{}).get(UNK,0):     
        emissionCounts[tag][UNK] = 1 #conditional count word unknown given tag vocab
    
    if not emissionCounts.get(UNK,{}).get(word,0):     
        emissionCounts[UNK][word] = 1 #conditional count word vocab given tag unknown
    
    if not transitionCounts.get(tag,{}).get(UNK,0):
        transitionCounts[tag][UNK] = 1 #tag vocab to tag unknown
    
    if not transitionCounts.get(UNK,{}).get(tag,0):
        transitionCounts[UNK][tag] = 1 #tag unknown to tag vocab
    
    #start and end
    if not transitionCounts.get(START,{}).get(UNK,0):
        transitionCounts[START][UNK] = 1 #tag start to tag unknown
    
    if not transitionCounts.get(UNK,{}).get(END,0):
        transitionCounts[UNK][END] = 1 #tag unknown to tag start


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn()
        allTagCounts (Counter): tag counts, as specified in learn()

    Return:
        list: list of pairs

    """
    keys , values = zip(*allTagCounts.items())
    tup_sentence=[]
    for w in sentence:
        if not w in perWordTagCounts.keys():
            t = choices(keys,values)[0]        
        else:
            t = max(perWordTagCounts[w].items() , key=operator.itemgetter(1))[0]
        
        tup_sentence.append((w,t))
    
    return tup_sentence


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    vet_matrix = viterbi(sentence, A, B) #run viterbi and get matrix
    tags = retrace(vet_matrix[-1][0]) #run retrace on viterbi matrix and get list of tags
    return list(zip(sentence,tags)) #return the describe above format
    
#will check
def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

    """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    
    global START,END,UNK,perWordTagCounts
    
    index_tag = dict(enumerate(B.keys()))
    vet_matrix = []
    
    vet_matrix.append([(START,None,A[START][UNK])])#start col
    
    #for each word
    for col in range(len(sentence)+1):
        col_i=[] 
        
        #<end of sentence>
        if col > len(sentence)-1:
            index_tag = {0:END}
            w=None 
        
        # word
        else:
            w = sentence[col]     
            #if not exists change to unknown
            if not w in perWordTagCounts:
                w = UNK
        
        #for each tag
        for t in index_tag.values():
            if w and B[t][w] == 0:# check if tag exist
                continue
           
            next_best = predict_next_best(w,t,vet_matrix[-1],A,B)#predict next best item
            col_i.append(next_best)
            
        vet_matrix.append(col_i)
              
    return vet_matrix


#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    Arg:
        end item of viterbi "matrix"
    Return list of tags. (tag string)
    """
    global START,END
    
    tags=[]
    while end_item[1]:
        if end_item[0] not in [START,END]:
            tags.append(end_item[0])
        end_item=end_item[1]
    
    return tags[::-1]


#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list, A, B):
    """Returns a new item (tuple) in format (t,r,p) for the next best connection for the new tag
    the format explain in viterbi function
    Args:
        word(str)
        tag(str)
        predecessor_list(list): previous list with the items
        A,B: transition and emmission
    Return tuple: (t,r,p)
    """
    best_item = None
    best_p = 0
    for item in predecessor_list:
        previous_tag = item[0]
        p = A[previous_tag][tag] if A[previous_tag][tag] else A.get(previous_tag,{}).get(UNK,0)
       
        if word and B[tag][word] != 0:
            p += B[tag][word]  
       
        p += item[2]
        if not best_item or best_p < p:
            best_item = item
            best_p = p
    
    return (tag,best_item,best_p)    


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    global START,END,perWordTagCounts,allTagCounts
    p = 0   # joint log prob. of words and tags
    
    start = True
    for w,t in sentence:
        #if not exists change to unknown
        if not allTagCounts.get(t,0):
            t = UNK
        if not perWordTagCounts.get(w,0):
            w = UNK
        
        if start:
            start = False
            p+=A[START][t]
        p+=B[t][w]
        last_t = t
    p+=A[last_t][END]
    
    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p
    
 
def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger
    Returns correct (int), correctOOV (int), OOV (int). counters.
    """
    #needed global for checking OOV
    global perWordTagCounts
    assert len(gold_sentence)==len(pred_sentence)
    
    correct = correctOOV = OOV = 0
    
    for (word_gold, tag_gold) , (word_pred , tag_pred) in zip(gold_sentence,pred_sentence):
        OOV_flag = False
        
        if not perWordTagCounts.get(word_gold,0):
            OOV+=1
            OOV_flag=True
        
        if tag_gold == tag_pred:
            correct+=1
            if OOV_flag:
                correctOOV+=1
    return correct, correctOOV, OOV


def split_tuple_list(tup_list):
    '''
    Returns two lists from single tuple(pair) list. 
        [(word,tag),...,(word,tag)]-> [word,....,word],[tag,....,tag]
    Args:
        tup_list(list): list contain tuple of pair
    Return words,tags lists
    '''
    words,tags = zip(*tup_list)
    return words,tags


