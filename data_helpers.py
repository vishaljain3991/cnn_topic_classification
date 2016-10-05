import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels_another():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_text = []
    y = []
    one_hot_vector = [0,0,0,0,0]
    labels = {}
    topics = ['Technology' , 'Business', 'Food', 'Design', 'Books']
    for idx, topic in enumerate(topics):
        clean_questions = list(open(topic + "clean_question.txt", mode = 'rb').readlines())
        clean_questions = [s.strip() for s in clean_questions]
        x_text = x_text + clean_questions
        if topic == 'Technology':
            y = y + [[1,0,0,0,0] for _ in clean_questions]
        elif topic == 'Business':
            y = y + [[0,1,0,0,0] for _ in clean_questions]
        elif topic == 'Food':
            y = y + [[0,0,1,0,0] for _ in clean_questions]
        elif topic == 'Design':
            y = y + [[0,0,0,1,0] for _ in clean_questions]
        elif topic == 'Books':
            y = y + [[0,0,0,0,1] for _ in clean_questions]        # print labels

        one_hot_vector[idx] = 0

    # print y
    y = np.array(y)
    #print labels['Business']
    #y = np.concatenate([labels[0], labels[1], labels[2], labels[3], labels[4]], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
