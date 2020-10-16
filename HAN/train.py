import os
import pandas as pd 
import numpy as np
import re
import pdb
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import model as m

############ Parameters
train_set_percent = 0.8

lr = 0.0005

batch_size = 64
epochs = 100
embed_size = 50

hidden_dim = 20
num_classes = 2

data = './train_flat_file_fiscal.txt'

trained_models_path = './trained_models/'

path2embeddings = '/path/to/glove/file/glove.6B/'
embedfile = 'glove.6B.' + str(embed_size) + 'd'

np.random.seed(0)

# model_type = 'vanila_bilstm'
model_type = 'heirarchical'
max_words_for_vanila_lstm = 1000


############# Loading Pretrained Glove Embeddings
if os.path.isfile(path2embeddings + embedfile + '_w2v.txt'):
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)
else:
    glove2word2vec(glove_input_file=path2embeddings + embedfile + '.txt', \
        word2vec_output_file=path2embeddings + embedfile + '_w2v.txt')
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)


############ Model related
# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

sent_encoder_model = m.sent_encoder(embed_size, hidden_dim)
sent_encoder_model = sent_encoder_model.to(device)

doc_encoder_model = m.doc_encoder(2 * hidden_dim, 2 * hidden_dim)
doc_encoder_model = doc_encoder_model.to(device)

doc_classifier_model = m.doc_classifier(4 * hidden_dim, num_classes)
doc_classifier_model = doc_classifier_model.to(device)


# vanila_bilstm = m.vanila_lstm_classifier(embed_size, hidden_dim, num_classes)
# vanila_bilstm = vanila_bilstm.to(device)

# Defining loss function
criterion = nn.CrossEntropyLoss()

# Defining optimizer with all parameters
if model_type == 'heirarchical':
	total_parameters = list(sent_encoder_model.parameters()) + list(doc_encoder_model.parameters()) \
						+ list(doc_classifier_model.parameters())

	optimizer = torch.optim.Adam(total_parameters, lr = lr)
elif model_type == 'vanila_bilstm':
	optimizer = torch.optim.Adam(vanila_bilstm.parameters(), lr = lr)


############ Helper functions
def get_embed(word):
    # Case folding
    word = word.lower()
    try:
        return (glove_model.get_vector(word))
    except:
        return 0


def single_epoch_heirarchical(data, train_flag):
	golden_labels = []
	predicted_labels = []
	total_loss = 0
	batch_loss = 0
	count = -1

	# Make gradients 0 before starting
	optimizer.zero_grad()

	for (mdna_section, buy_class) in data:
		count = count + 1

		# if (count%2000 == 0):
		# 	print(count)

		mdna_section = [word_tokenize(s) for s in sent_tokenize(mdna_section)]
		buy_class = int(buy_class)

		# Initialize empty tensor for doc embeding. We append sent embeds in it.
		doc_tensor = torch.empty(size=(len(mdna_section), 2 * hidden_dim)).to(device)

		for sent_id in range(len(mdna_section)):
			sent = mdna_section[sent_id]
			word_embed = [get_embed(i) for i in sent if type(get_embed(i)) is not int]
			word_embed = np.array(word_embed)
			word_embed = torch.from_numpy(word_embed).to(device)

			# get sent embed through attention model
			sent_embed, word_alphas = sent_encoder_model(word_embed)

			# append all sent embeds
			doc_tensor[sent_id] = sent_embed

		# pass all sent embeds through an attention layer to get doc embed
		doc_embed, sent_alphas = doc_encoder_model(doc_tensor)

		# use doc embed for classification
		prediction_class = doc_classifier_model(doc_embed)

		# Getting true label
		true_class =  torch.tensor([buy_class]).to(device)

		# Loss
		loss = criterion(prediction_class, true_class)
		# Backpropagation (gradients are accumulated in parameters)
		loss.backward()

		# Accumulating the total loss
		loss_data = float(loss.data.cpu().numpy())
		total_loss += loss_data

		if train_flag and ((count + 1) % batch_size == 0):
			# Gradient Descent after all the batch gradients are accumulated
			optimizer.step()
			optimizer.zero_grad()


		# For getting accuracy 
		golden_labels.append(buy_class)
		predicted_labels.append(torch.argmax(prediction_class, dim=1).data.cpu().numpy()[0])

	# Final update for remaining datapoints not included in any batch
	if train_flag:
		optimizer.step()
		optimizer.zero_grad()

	avg_loss = total_loss/len(data)
	
	# print(len(golden_labels))
	print(Counter(golden_labels))
	# print(len(predicted_labels))
	print(Counter(predicted_labels))
	return avg_loss, golden_labels, predicted_labels




## MAIN

# Splitting trian validation and test
data = open(data).readlines()
data = [(d.strip().split('\t')[0], d.strip().split('\t')[1]) for d in data]

train_data = data[:int(len(data)*train_set_percent)]
val_test = data[int(len(data)*train_set_percent):]
val_data = val_test[:int(len(val_test)*0.5)]
test_data = val_test[int(len(val_test)*0.5):]

del data
del val_test

print('Size of train set: ' + str(len(train_data)))
print('Size of val set: ' + str(len(val_data)))
print('Size of test set: ' + str(len(test_data)))

# pdb.set_trace()
prev_val_loss = 10000

for e in range(epochs):
	# One Epoch
	avg_train_loss, golden_labels, predicted_labels = single_epoch_heirarchical(train_data, train_flag = True)
	print('Avg. train loss per Epoch: ' + str(avg_train_loss) + ' For Epoch: ' + str(e))
	print('Train acc : ' + str(accuracy_score(golden_labels, predicted_labels)) + '|| Train F1: ' + str(f1_score(golden_labels, predicted_labels)))

	avg_val_loss, golden_labels, predicted_labels = single_epoch_heirarchical(val_data, train_flag = False)
	print('Avg. validation loss per Epoch: ' + str(avg_val_loss) + ' For Epoch: ' + str(e))
	print('Val acc : ' + str(accuracy_score(golden_labels, predicted_labels)) + '|| Val F1: ' + str(f1_score(golden_labels, predicted_labels)))
	print('=' * 20)

	loss_folder = 'train_' + str(avg_train_loss) + '_val_' + str(avg_val_loss) + '/'

	full_path = trained_models_path + loss_folder
	if not os.path.exists(full_path):
		os.makedirs(full_path)

	torch.save(sent_encoder_model.state_dict(), full_path + 'sent_encoder_model.pkl')
	torch.save(doc_encoder_model.state_dict(), full_path + 'doc_encoder_model.pkl')
	torch.save(doc_classifier_model.state_dict(), full_path + 'doc_classifier_model.pkl')

	if float(avg_val_loss) <= prev_val_loss:
		prev_val_loss = float(avg_val_loss)
		_, golden_labels, predicted_labels = single_epoch_heirarchical(test_data, train_flag = False)
		report = open(trained_models_path + 'lr0005_report.txt', 'w')
		report.write('Epoch: ' + str(e) + '\n')
		report.write('Accuracy: ' + str(accuracy_score(golden_labels, predicted_labels)) + '\n')
		report.write('F1: ' + str(f1_score(golden_labels, predicted_labels)) + '\n')
		report.write('MCC: ' + str(matthews_corrcoef(golden_labels, predicted_labels)) + '\n')
		report.close()


	
	
	

	
