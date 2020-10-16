import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb


class sent_encoder(nn.Module):
	def __init__(self, embed_size, nHidden):
		super(sent_encoder, self).__init__()

		self.embed_size = embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)

		self.uw =  nn.Parameter(torch.randn(2*nHidden, 1), requires_grad = True)
		self.hidden2context = nn.Linear(2*nHidden, 2*nHidden)

	def forward(self, in_seq):
		in_seq = in_seq.view(-1, 1, self.embed_size)
		recurrent, (hidden, c) = self.lstm(in_seq)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		return context, alphas


# For debugging
# sent_encoder = sent_encoder(4, 3)
# inp = torch.rand(2, 4)
# output = sent_encoder(inp)

class doc_encoder(nn.Module):
	def __init__(self, sent_embed_size, nHidden):
		super(doc_encoder, self).__init__()
		self.sent_embed_size = sent_embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(sent_embed_size, nHidden, bidirectional = True)

		self.uw =  nn.Parameter(torch.randn(2*nHidden, 1), requires_grad = True)
		self.hidden2context = nn.Linear(2*nHidden, 2*nHidden)

	def forward(self, in_seq):
		in_seq = in_seq.view(-1, 1, self.sent_embed_size)
		recurrent, (hidden, c) = self.lstm(in_seq)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		return context, alphas



class doc_classifier(nn.Module):
	def __init__(self, doc_embed_size, nClasses):
		super(doc_classifier, self).__init__()
		self.out_linear = nn.Linear(doc_embed_size, nClasses)

	def forward(self, doc_embed):
		out = self.out_linear(doc_embed)
		out = out.view(1,-1)
		return out



class doc_encoder_for_bert(nn.Module):
	def __init__(self, sent_embed_size, nHidden):
		super(doc_encoder_for_bert, self).__init__()
		self.sent_embed_size = sent_embed_size
		self.nHidden = nHidden

		self.projection_layer = nn.Linear(sent_embed_size, nHidden)
		self.lstm = nn.LSTM(nHidden, nHidden, bidirectional = True)

		self.uw =  nn.Parameter(torch.randn(2*nHidden, 1), requires_grad = True)
		self.hidden2context = nn.Linear(2*nHidden, 2*nHidden)

	def forward(self, in_seq):
		projected_in_seq = self.projection_layer(in_seq)

		projected_in_seq = projected_in_seq.view(-1, 1, self.nHidden)
		recurrent, (hidden, c) = self.lstm(projected_in_seq)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		return context, alphas


class vanila_lstm_classifier(nn.Module):
	def __init__(self, embed_size, nHidden, nClasses):
		super(vanila_lstm_classifier, self).__init__()
		self.embed_size = embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)
		self.out_linear = nn.Linear(2 * nHidden, nClasses)


	def forward(self, in_seq):
		in_seq = in_seq.view(-1, 1, self.embed_size)
		recurrent, (hidden, c) = self.lstm(in_seq)
		hidden = hidden.view(-1, 2*self.nHidden)

		out = self.out_linear(hidden)
		out = out.view(1,-1)

		return out