# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30 11  19:53 2018
practical-pytorch/seq2seq-translation/seq2seq-translation.ipynb
https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
@author: GLN
"""
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


USE_CUDA = True

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s.txt' % (lang1, lang2), 'r', encoding='UTF-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

# Print an example pair
print(random.choice(pairs))

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size             # 输入维度，最开始是整个输入词典的大小
        self.hidden_size = hidden_size           # 隐层大小，在这里就是希望的词向量的大小
        self.n_layers = n_layers                 # 隐层个数

        self.embedding = nn.Embedding(input_size, hidden_size)
        # nn.Embedding()有两个输入，第一个参数是词典的大小，第二个参数是词向量的维度（在这里词向量的维度和隐层大小相等）
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        # nn.GRU()第一个参数是输入的特征大小，第二个参数是隐层状态h的特征数，第三个参数是隐层个数

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        # 这里word_inputs即为句子中词的个数，hidden为隐层大小
        seq_len = len(word_inputs)
        print(seq_len)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        # seq_len, batch, emb_dim 在本例中，emb_dim = hidden_size
        # nn.Embedding()的输出，输入为两个维度(每个batch的单词个数，batch的大小)，输出则在两个维度上加上词向量的大小。
        output, hidden = self.gru(embedded, hidden)
        # output [seq_len, batch_size, hidden_size * num_directions]
        # hidden [num_layers * num_directions, batch_size, hidden_size]
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

'''
The general form of the attention calculation relies on the target (decoder) side 
hidden state (s_i) and corresponding source (encoder) side state (h_*), normalized over all states 
to get values summing to 1: 返回注意力权重a_ij
'''
class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        # method参数确定选择哪一种计算注意力energy的方法，hidden_size为隐层大小，max_length在本例中设置为10（即句子最大长度）
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            # 第一个参数为输入的维度，第二个参数为输出的维度，在本例中这两个相等

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        # encoder_output的size为[seq_len, batch_size, hidden_size * num_directions]
        # seq_len即为encoder_outputs的第一个维度的大小（seq_len）

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))
        # attn_energies的size即[seq_len]
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
            # 这里atten_energies[i]里面保存的是一个标量。

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
            return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        # 选择三种方式中的一种（dot, general, concat）
        self.hidden_size = hidden_size
        # 隐层大小，在本例中被设置为500
        self.output_size = output_size
        # 输出层大小，在本例中为目标端词向量大小
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        # 因为在注意力机制中，decoder的隐层状态s和decoder的输出y的embedding，两个输入放入GRU中
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # word_input 就是每次输入的单个单词，它是一维的。这里最开始有测试，所以最开始的三个word_input维度是10，其后的都是维度500
        # last_context 这里最开始有测试，所以最开始的三个last_context是[1,10]，其后的都是[1, 500]
        # last_hidden为encoder层[num_layers * num_directions, batch_size, hidden_size]
        # encoder_outputs为encoder层[seq_len, batch_size, hidden_size * num_directions]

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        # seq_len=1, batch_size=1, emb_dim
        # 在本例中word_embedded [1, 1, emb_dim=500]

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # 将word_embedded和last_context在第三维上进行连接
        # word_embedded [1, 1, emb_dim=500]
        # last_context [1, hidden_size] 在train中可以看到它的设置（330行）。
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # rnn_input [1, 1, emb_dim*2=1000]
        # last_hidden [num_layers * num_directions=2, batch_size=1, hidden_size=500]
        # rnn_output [1, 1, emb_dim=500]
        # hidden [num_layers * num_directions=2, batch_size=1, hidden_size=500]

        # Calculate attention from current RNN state and all encoder outputs: apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        # encoder_outputs [seq_len, batch_size=1, emb_dim=500]
        # rnn_output [1, 1, emb_dim=500] ==> rnn_output.squeeze(0) [1, emb_dim=500]将第0维的1被移除
        # squeeze(0)表示如果第一维度值为1，则去掉，否则不变。
        # attn_weights [1, 1, seq_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # encoder_outputs [seq_len, batch_size=1, emb_dim=500]
        # ==> encoder_outputs.transpose(0,1) [batch_size=1, seq_len, emb_dim=500]
        # context [batch_size=1, 1, emb_dim=500]
        # [1, 1, seq_len].bmm([1, seq_len, emb_dim]) ==> [1, 1, 500]
        # bmm意思就是批量的矩阵乘法，第一个表示批量的个数，后面两个是矩阵，本例中[1, seq_len]x[seq_len, emb_dim] = [1, emb_dim]

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        # rnn_output [1, batch_size=1, emb_dim=500] ==> [batch_size=1, emb_dim=500]
        context = context.squeeze(1)
        # context [batch_size=1, 1, emb_dim=500] ==> [batch_size=1, emb_dim=500]
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        # torch.cat((rnn_output, context), 1) [batch_size=1, emb_dim*2=1000]
        # self.out(torch.cat((rnn_output, context), 1)) [batch_size=1, 2696]
        # 此处2696为目标语言词表大小

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

# 以下部分为测试
encoder_test = EncoderRNN(10, 10, 2)
decoder_test = AttnDecoderRNN('general', 10, 10, 2)
print(encoder_test)
print(decoder_test)


encoder_hidden = encoder_test.init_hidden()
word_input = Variable(torch.LongTensor([1, 2, 3]))
if USE_CUDA:
    encoder_test.cuda()
    word_input = word_input.cuda()
encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

word_inputs = Variable(torch.LongTensor([1, 2, 3]))
decoder_attns = torch.zeros(1, 3, 3)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

if USE_CUDA:
    decoder_test.cuda()
    word_inputs = word_inputs.cuda()
    decoder_context = decoder_context.cuda()

for i in range(3):
    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
    print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
    decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data
# 以上部分为测试

teacher_forcing_ratio = 0.5
clip = 5.0


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    # input_variable.size() [seq_len=6, 1]
    target_length = target_variable.size()[0]
    # target_variable.size() [seq_len=6, 1]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    # encoder_hidden [num_layers * num_directions=2, batch_size=1, hidden_size=500]
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # encoder_outputs [seq_len=6, 1, hidden_size=500]

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    # decoder_input [1, 1]
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    # decoder_context [1, hidden_size=500]
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
    # decoder_hidden [num_layers * num_directions=2, batch_size=1, hidden_size=500]
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            # decoder_output [1, target_word_vocab_size=2696]
            # decoder_context [1, hidden_size=500]
            # decoder_hidden [num_layers * num_directions=2, batch_size=1, hidden_size=500]
            # decoder_attention [1, 1, seq_len]
            loss += criterion(decoder_output, target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            # ni是对应最高值的index值

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    # pytorch梯度裁剪（Clipping Gradient）：torch.nn.utils.clip_grad_norm
    # 既然在BP过程中会产生梯度消失/爆炸（就是偏导无限接近0，导致长时记忆无法更新），
    # 那么最简单粗暴的方法，设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值，
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0].item() / target_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
# input_lang.n_words表示源端词表大小
# hidden_size 500
# n_layers 2
# encoder如下
'''
EncoderRNN(
  (embedding): Embedding(3925, 500)
  (gru): GRU(500, 500, num_layers=2)
)
'''
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
# decoder如下
'''
AttnDecoderRNN(
  (embedding): Embedding(2696, 500)
  (gru): GRU(1000, 500, num_layers=2, dropout=0.05)
  (out): Linear(in_features=1000, out_features=2696, bias=True)
  (attn): Attn(
    (attn): Linear(in_features=500, out_features=500, bias=True)
  )
)
'''

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 1000

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin!
for epoch in range(1, n_epochs + 1):

    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (
        time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0



def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)


def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                     decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni].item())

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly():
    pair = random.choice(pairs)

    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)

    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

evaluate_randomly()

output_words, attentions = evaluate("je suis trop froid .")
plt.matshow(attentions.numpy())

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

evaluate_and_show_attention("elle a cinq ans de moins que moi .")

evaluate_and_show_attention("elle est trop petit .")

evaluate_and_show_attention("je ne crains pas de mourir .")

evaluate_and_show_attention("c est un jeune directeur plein de talent .")


