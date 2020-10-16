import os
import pandas as pd 
import numpy as np
import pdb

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import classification_report

import model as m


############ Parameters
summarization_percentage = 0.2


embed_size = 50
hidden_dim = 20
num_classes = 2


path2embeddings = '/path/to/Glove/Embeddings/glove.6B/'
embedfile = 'glove.6B.' + str(embed_size) + 'd'

trained_model_path = './trained_models/lr0005/epoch_5/'

eval_stats = '../data/eval_flat_file_fiscal_stats.txt'
eval_data = '../data/eval_flat_file_fiscal.txt'

np.random.seed(0)

############# Loading Pretrained Glove Embeddings
if os.path.isfile(path2embeddings + embedfile + '_w2v.txt'):
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)
else:
    glove2word2vec(glove_input_file=path2embeddings + embedfile + '.txt', \
    	word2vec_output_file=path2embeddings + embedfile + '_w2v.txt')
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)

############ Model (Loading the model)
# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

sent_encoder_model = m.sent_encoder(embed_size, hidden_dim)
sent_encoder_model.load_state_dict(torch.load(trained_model_path + 'sent_encoder_model.pkl'))
sent_encoder_model.eval()
sent_encoder_model = sent_encoder_model.to(device)

doc_encoder_model = m.doc_encoder(2 * hidden_dim, 2 * hidden_dim)
doc_encoder_model.load_state_dict(torch.load(trained_model_path + 'doc_encoder_model.pkl'))
doc_encoder_model.eval()
doc_encoder_model = doc_encoder_model.to(device)

doc_classifier_model = m.doc_classifier(4 * hidden_dim, num_classes)
doc_classifier_model.load_state_dict(torch.load(trained_model_path + 'doc_classifier_model.pkl'))
doc_classifier_model.eval()
doc_classifier_model = doc_classifier_model.to(device)

############ Helper functions
def get_embed(word):
    # Case folding
    word = word.lower()
    try:
        return (glove_model.get_vector(word))
    except:
        return 0


def process_mdna(mdna_inp):
	mdna_sent = sent_tokenize(mdna_inp)
	mdna_words = [word_tokenize(i) for i in mdna_sent]
	return mdna_words


def get_summary(mdna_inp):
	processed_mdna = process_mdna(mdna_inp)

	# Initialize empty tensor for doc embeding. We append sent embeds in it.
	doc_tensor = torch.empty(size=(len(processed_mdna), 2 * hidden_dim)).to(device)

	for sent_id in range(len(processed_mdna)):
		sent = processed_mdna[sent_id]
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
	prediction_prob = doc_classifier_model(doc_embed)
	sell_prob = prediction_prob.data.cpu().numpy()[0][0]
	buy_prob = prediction_prob.data.cpu().numpy()[0][1]
	pred_class = torch.argmax(prediction_prob, dim=1).data.cpu().numpy()[0]

	# Getting summary form most attented sentences
	sent_alpha_list = sent_alphas.data.cpu().view(-1).tolist()
	sentid2alpha = {i:sent_alpha_list[i] for i in range(len(sent_alpha_list))}
	sorted_sentid2alpha = sorted(sentid2alpha.items(), key=lambda kv: kv[1], reverse = True)
	if int(len(sent_alpha_list)*summarization_percentage) > 0 :
		selected_sentids = sorted([i[0] for i in sorted_sentid2alpha[ : int(len(sent_alpha_list)*summarization_percentage) ]])
	else:
		selected_sentids = sorted([i[0] for i in sorted_sentid2alpha[ : 1 ]])
	# selected_sentids.sort()
	summary_sents = [' '.join(processed_mdna[i]) for i in selected_sentids]
	generated_summary = ' '.join(summary_sents)

	return generated_summary, buy_prob, sell_prob, pred_class



############ Loading input Data
eval_stats = open(eval_stats).readlines()
eval_data = open(eval_data).readlines()


# MAIN
fout = open('./generated_summaries_' + str(int(summarization_percentage*100)) + 'p.txt', 'w')
fout_stats = open('./generated_summaries_stats_' + str(int(summarization_percentage*100)) + 'p.txt', 'w')
fout_stats.write('cik\tyear\tfiscal_date\tprice_change\tbuy_prob\tsell_prob\tgolden_label\tpred_label\n')

for i in range(len(eval_stats)):
	stats = eval_stats[i].strip()
	mdna_inp = eval_data[i].strip()

	cik = int(stats.split('\t')[0])
	year = int(stats.split('\t')[1])
	fiscal_date = stats.split('\t')[2]
	price_change = stats.split('\t')[3]
	golden_class = stats.split('\t')[4]

	summary, buy_prob, sell_prob, pred_class = get_summary(mdna_inp)

	stat_line = str(cik) + '\t' + str(year) + '\t' + str(fiscal_date) + '\t' + str(float(price_change)) + '\t' + str(buy_prob) +  \
				 '\t' + str(sell_prob) + '\t' + str(golden_class) + '\t' + str(pred_class) + '\n'

	print(stat_line)

	fout.write(summary.strip() + '\n')
	fout_stats.write(stat_line)



	# pdb.set_trace()


