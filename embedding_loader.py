import os
import time
import pickle
import sys
import numpy as np
import argparse

def main(data_dir):
    f = open(data_dir + '/glove.6B.300d.txt', 'rb')
    g = open(data_dir + '/glove.6B.300d_pickle', 'wb')
    word_dict = {}
    wordvec = []
    for idx, line in enumerate(f.readlines()):
        word_split = line.split(' ')
        word = word_split[0]
        word_dict[word] = idx
        d = word_split[1:]
        d[-1] = d[-1][:-1]
        d = [float(e) for e in d]
        wordvec.append(d)

    embedding = np.array(wordvec)
    pickling = {}
    pickling = {'embedding' : embedding, 'word_dict': word_dict}
    pickle.dump(pickling, g)
    f.close()
    g.close()

#def load_pickle():
#    g = open('/Users/vj/Downloads/glove.6B/glove.6B.300d_pickle', 'rb')
#    pickling = pickle.load(g)
#    print pickling['embedding']

def word_id_convert(data_dir):
    g = open(data_dir + '/data_pickle', 'rb')
    pickling = pickle.load(g)
    x_text = pickling['x']
    y = pickling['y']
    max_document_length = max([len(x.split(" ")) for x in x_text])

    h = open(data_dir + '/glove.6B.300d_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']
    # print len(word_dict)
    # sys.exit()
    splitter = [x.split(" ") for x in x_text]
    word_indices = []
    for sentence in splitter:
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence]
        padding = max_document_length -  len(word_index)
        padder = [2 for i in xrange(padding)]
        word_index = word_index + padder
        word_indices.append(word_index)
        # print word_index
    # print splitter
    print type(x_text)
    word_indices = np.array(word_indices)
    print word_indices
    word_index_pickle = open(data_dir + '/word_index_pickle', 'wb')
    pickling = {'word_indices': word_indices, 'y': y}
    pickle.dump(pickling, word_index_pickle)
    # for x in x_text:
    #     print x

#def load_data():
#    word_index_pickle = open('/Users/vj/Downloads/glove.6B/word_index_pickle', 'rb')
#    pickling = pickle.load(word_index_pickle)
#    word_indices = pickling['word_indices']
#    y = pickling['y']
#    print word_indices.shape
    # print word_indices

def write_concat_vec(data_dir):
    word_index_pickle = open(data_dir + '/word_index_pickle', 'rb')
    pickling = pickle.load(word_index_pickle)
    x = pickling['word_indices']
    y = pickling['y']
    g = open(data_dir + '/glove.6B.300d_pickle', 'rb')
    pickling = pickle.load(g)
    embedding = pickling['embedding']
    
    l = []
    m = []
    arr = np.array([0,1,2,3,4])
    for idx, sentence in enumerate(x):
        concater = np.array([])
        sentence_vec = embedding[sentence].flatten()
        sentence_vec = np.reshape(sentence_vec, (1, sentence_vec.shape[0]))
        # print sentence_vec.shape
        l.append(sentence_vec)
        label = np.sum(y[idx] * arr)
        m.append(label)

    concater = np.squeeze(np.array(l))
    m = np.array(m)

    h = open(data_dir + '/concat_vec_labels', 'wb')
    pickling = {'x' : concater, 'y': m}
    pickle.dump(pickling, h)
    h.close()
    g.close()
    word_index_pickle.close()
    # print concater.shape






if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/home/hw/data',
		           help='data directory containing glove vectors')
	args = parser.parse_args()
	data_dir = args.data_dir
	
	main(data_dir)
	#oad_pickle()
	word_id_convert(data_dir)
	#load_data()
	write_concat_vec(data_dir)
