# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from grammar import Rule, Grammar
import torch.nn.functional as F
from collections import defaultdict


class GeneratingGrammar(Grammar):
    """docstring for GeneratingGrammar"""

    MAX_LENGTH = 20

    def __init__(self, rules):
        self.gen_rules = defaultdict(list)
        self.input_vocabulary = ['EOS', 'SOS']
        self.output_vocabulary = ['EOS', 'SOS', 'PUSH']
        self.input_lang_map = None
        self.output_lang_map = None
        super(GeneratingGrammar, self).__init__(rules)

    def n_input_words(self):
        return len(self.input_vocabulary)

    def n_output_words(self):
        return len(self.output_vocabulary)

    def input_lang(self):
        if self.input_lang_map is None:
            self.input_lang_map = { word : self.input_vocabulary.index(word) for word in self.input_vocabulary }
        return self.input_lang_map

    def output_lang(self):
        if self.output_lang_map is None:
            self.output_lang_map = { word : self.output_vocabulary.index(word) for word in self.output_vocabulary }
        return self.output_lang_map

    def add_rule(self, rule):
        for rhsi in rule.rhs:
            if not rhsi in self.input_vocabulary:
                self.input_vocabulary.append(rhsi)
        if not rule.lhs in self.output_vocabulary:
            self.output_vocabulary.append(rule.lhs)
        self.gen_rules[rule.lhs].append(rule.rhs)
        super(GeneratingGrammar, self).add_rule(rule)

    def gen_example(self, symbol, depth):
        terminal = []
        non_terminal = []
        for rhs in self.gen_rules[symbol]:
            if (any([self.is_cat(sym) for sym in rhs])):
                non_terminal.append(rhs)
            else:
                terminal.append(rhs)
        if 0 < len(non_terminal):
            if 0 < depth or 0 == len(terminal):
                rhss = random.choice(non_terminal)
                strings = []
                symbols = []
                for rhs in rhss:
                    if self.is_cat(rhs):
                        substring, subsymbols = self.gen_example(rhs, depth - 1)
                        strings += substring
                        symbols += subsymbols
                    else:
                        strings.append(rhs[0])
                symbols.append(symbol)
                return strings, symbols
            else:
                return [random.choice(terminal)[0]], ["PUSH", symbol]
        else:
            return [random.choice(terminal)[0]], ["PUSH", symbol]



class InputVocabulary(object):
    """docstring for Vocabulary"""
    def __init__(self):
        super(InputVocabulary, self).__init__()
        self.vocabulary = [
            'SOS',
            'EOS',
            'one',
            'two',
            'three',
            'four',
            'plus',
            'minus',
            'times'
        ]
        self.word2index = { word : self.vocabulary.index(word) for word in self.vocabulary }


class OutputVocabulary(object):
    """docstring for OutputVocabulary"""
    def __init__(self):
        super(OutputVocabulary, self).__init__()
        self.vocabulary = [
            'SOS',
            'EOS',
            'PUSH',
            '$E',
            '$UnOp',
            '$BinOp',
            '$EBO'
        ]
        self.word2index = { word : self.vocabulary.index(word) for word in self.vocabulary }


class DataSet(object):
    """docstring for DataSet"""
    def __init__(self):
        super(DataSet, self).__init__()
        self.training_pairs = [
            ["one plus one",   'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['one plus two',   'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['one plus three', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['two plus two',   'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['two plus three', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['three plus one', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['three plus minus two', 'PUSH $E PUSH $BinOp $EBO PUSH $UnOp PUSH $E $E $E'],
            ['three minus two', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['two times two', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['two times three', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['minus three minus two', 'PUSH $UnOp PUSH $E $E PUSH $BinOp $EBO PUSH $E $E'],
            ['minus three minus two', 'PUSH $UnOp PUSH $E PUSH $BinOp $EBO PUSH $E $E $E'],
            ['three plus three minus two', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E PUSH $BinOp $EBO PUSH $E $E'],
            ['three plus three minus two', 'PUSH $E PUSH $BinOp $EBO PUSH $E PUSH $BinOp $EBO PUSH $E $E $E'],
            ['two times two plus three', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E PUSH $BinOp $EBO PUSH $E $E'],
            ['two times two plus three', 'PUSH $E PUSH $BinOp $EBO PUSH $E PUSH $BinOp $EBO PUSH $E $E $E']
        ]

        self.testing_pairs = [
            ['minus three', 'PUSH $UnOp PUSH $E $E'],
            ['three plus two', 'PUSH $E PUSH $BinOp $EBO PUSH $E $E'],
            ['minus four', 'PUSH $UnOp PUSH $E $E'],
        ]

        def maxpair0length(current, pair):
            pair0length = len(pair[0].split())
            if current > pair0length:
                return current
            else:
                return pair0length

        self.MAX_LENGTH = reduce(maxpair0length, self.training_pairs, 0) + 1
        self.input_lang = InputVocabulary()
        self.output_lang = OutputVocabulary()
        self.n_input_words = len(self.input_lang.vocabulary)
        self.n_output_words = len(self.output_lang.vocabulary)


class EncoderRNN(nn.Module):
    """
    The Encoder
    -----------

    The encoder of a seq2seq network is a RNN that outputs some value for
    every word from the input sentence. For every input word the encoder
    outputs a vector and a hidden state, and uses the hidden state for the
    next input word.

    .. figure:: /_static/img/seq-seq-images/encoder-network.png
       :alt:
    """
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        return self.gru(self.embedding(input).view(1, 1, -1), hidden)

    def initHiddenVariable(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

    def initOutputVariables(self, max_length):
        return Variable(torch.zeros(max_length, self.hidden_size))

    def function(self, inputs, max_length):
        """
        The encoder function accepts each input sequentially, using the same
        hidden variables. The output variables are collected and returned
        along with the hidden variables.
        """

        encoder_outputs = self.initOutputVariables(max_length)
        encoder_hidden = self.initHiddenVariable()

        for ei, inp in enumerate(inputs):
            # calls forward()
            encoder_output, encoder_hidden = self(
                inp, 
                encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
        return encoder_outputs, encoder_hidden


class DecoderRNN(nn.Module):
    """
    Simple Decoder
    ^^^^^^^^^^^^^^

    In the simplest seq2seq decoder we use only last output of the encoder.
    This last output is sometimes called the *context vector* as it encodes
    context from the entire sequence. This context vector is used as the
    initial hidden state of the decoder.

    At every step of decoding, the decoder is given an input token and
    hidden state. The initial input token is the start-of-string ``<SOS>``
    token, and the first hidden state is the context vector (the encoder's
    last hidden state).

    .. figure:: /_static/img/seq-seq-images/decoder-network.png
       :alt:
    """
    def __init__(self, hidden_size, output_size, learning_rate=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        output, hidden = self.gru(
            F.relu(self.embedding(input).view(1, 1, -1)), 
            hidden)
        return self.softmax(self.out(output[0])), hidden

    def initHiddenVariable(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

    def initInputVariable(self, token):
        return Variable(torch.LongTensor([[token]]))

    def function(self, _, targets, hidden, criterion, SOS_token, EOS_token):
        """
        The decoder function accepts inputs using hidden variables supplied.
        The loss is accumulated by comparing the outputs against the targets.
        The loss is returned.
        """
        decoder_input = self.initInputVariable(SOS_token)
        loss = 0
        # No teacher forcing: use its own predictions as the next input
        for di, tgt in enumerate(targets):
            decoder_output, hidden = self(decoder_input, hidden)
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            loss += criterion(decoder_output, tgt)
            decoder_input = self.initInputVariable(ni)
            if ni == EOS_token:
                break
        return loss

    def infer(self, _, hidden, SOS_token, EOS_token):
        decoder_input = self.initInputVariable(SOS_token)
        retval = []
        while True:
            decoder_output, hidden = self(decoder_input, hidden)
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = self.initInputVariable(ni)
            if ni == EOS_token:
                break
            retval.append(ni)
        return retval


class AttnDecoderRNN(nn.Module):
    """
    Attention Decoder
    ^^^^^^^^^^^^^^^^^

    If only the context vector is passed betweeen the encoder and decoder,
    that single vector carries the burden of encoding the entire sentence.

    Attention allows the decoder network to "focus" on a different part of
    the encoder's outputs for every step of the decoder's own outputs. First
    we calculate a set of *attention weights*. These will be multiplied by
    the encoder output vectors to create a weighted combination. The result
    (called ``attn_applied`` in the code) should contain information about
    that specific part of the input sequence, and thus help the decoder
    choose the right output words.

    .. figure:: https://i.imgur.com/1152PYf.png
       :alt:

    Calculating the attention weights is done with another feed-forward
    layer ``attn``, using the decoder's input and hidden state as inputs.
    Because there are sentences of all sizes in the training data, to
    actually create and train this layer we have to choose a maximum
    sentence length (input length, for encoder outputs) that it can apply
    to. Sentences of the maximum length will use all the attention weights,
    while shorter sentences will only use the first few.

    .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
       :alt:
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=20, learning_rate=0.01):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        output, hidden = self.gru(
            F.relu(
                self.attn_combine(
                    torch.cat((
                        embedded[0], 
                        torch.bmm(attn_weights.unsqueeze(0),
                        encoder_outputs.unsqueeze(0))[0]), 
                    1)).unsqueeze(0)), 
            hidden)
        return F.log_softmax(self.out(output[0]), dim=1), hidden

    def initHiddenVariable(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

    def initInputVariable(self, token):
        return Variable(torch.LongTensor([[token]]))

    def function(self, outputs, targets, hidden, criterion, SOS_token, EOS_token):
        """
        The decoder function accepts inputs using hidden variables supplied.
        The loss is accumulated by comparing the outputs against the targets.
        The loss is returned.
        """
        decoder_input = self.initInputVariable(SOS_token)
        loss = 0
        # Without teacher forcing: use its own predictions as the next input
        for di, tgt in enumerate(targets):
            decoder_output, hidden = self(
                decoder_input, hidden, outputs)
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            loss += criterion(decoder_output, tgt)
            decoder_input = self.initInputVariable(ni)
            if ni == EOS_token:
                break
        return loss

    def infer(self, encoder_outputs, hidden, SOS_token, EOS_token):
        decoder_input = self.initInputVariable(SOS_token)
        retval = []
        while True:
            decoder_output, hidden = self(decoder_input, hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = self.initInputVariable(ni)
            if ni == EOS_token:
                break
            retval.append(ni)
        return retval


class Model(object):
    def __init__(self, hidden_size, grammar):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.grammar = grammar
        self.encoder = EncoderRNN(grammar.n_input_words(), hidden_size)
        # self.decoder = DecoderRNN(hidden_size, grammar.n_output_words())
        self.decoder = AttnDecoderRNN(hidden_size, grammar.n_output_words(), dropout_p=0.1, max_length=self.grammar.MAX_LENGTH)

    def variableFromSentence(self, lang, sentence):
        def indexesFromSentence(lang, sentence):
            return [lang[word] for word in sentence.split(' ')]

        indexes = indexesFromSentence(lang, sentence)
        indexes.append(lang['EOS'])
        return Variable(torch.LongTensor(indexes).view(-1, 1))

    def variablesFromPair(self, pair):
        return (
            self.variableFromSentence(self.grammar.input_lang(), pair[0]), 
            self.variableFromSentence(self.grammar.output_lang(), pair[1]))

    def train(self):
        while True:
            strings, symbols = self.grammar.gen_example("$E", round(random.random() * 2))
            # print(' '.join(strings))
            # print(' '.join(symbols))
            training_pair = self.variablesFromPair([' '.join(strings), ' '.join(symbols)])
            # clear gradients
            self.encoder.optimizer.zero_grad()
            self.decoder.optimizer.zero_grad()
    
            encoder_outputs, hidden = self.encoder.function(training_pair[0], self.grammar.MAX_LENGTH)
                    
            loss = self.decoder.function(
                encoder_outputs,
                training_pair[1], 
                hidden, 
                nn.NLLLoss(), 
                self.grammar.input_lang()['SOS'],
                self.grammar.input_lang()['EOS'])

            loss.backward() # compute gradients

            # update weights
            self.encoder.optimizer.step()
            self.decoder.optimizer.step()

            yield loss.data.item() / training_pair[1].size()[0]

    def infer(self, inputs):
        encoder_outputs, hidden = self.encoder.function(
            self.variableFromSentence(self.grammar.input_lang(), inputs), 
            self.grammar.MAX_LENGTH)
        inferences = self.decoder.infer(
            encoder_outputs,
            hidden, 
            self.grammar.input_lang()['SOS'],
            self.grammar.output_lang()['EOS'])
        return [self.grammar.output_vocabulary[t] for t in inferences]

import unittest
import random
random.seed(1234)
torch.manual_seed(1234)

class TestMethods(unittest.TestCase):
    numeral_rules = [
        Rule('$E', 'one'),
        Rule('$E', 'two'),
        Rule('$E', 'three'),
        Rule('$E', 'four'),
    ]

    operator_rules = [
        Rule('$UnOp', 'minus'),
        Rule('$BinOp', 'plus'),
        Rule('$BinOp', 'minus'),
        Rule('$BinOp', 'times'),
    ]

    compositional_rules = [
        Rule('$E', '$UnOp $E'),
        Rule('$EBO', '$E $BinOp'),
        Rule('$E', '$EBO $E')
    ]

    arithmetic_rules = numeral_rules + operator_rules + compositional_rules

    model = None

    @classmethod
    def setUpClass(cls):
        cls.model = Model(48, GeneratingGrammar(TestMethods.arithmetic_rules))
        loss = cls.model.train()
        current_loss = 1
        iter = 0
        while current_loss > 1e-3:
            current_loss = next(loss)
            if (0 == iter % 1000):
                print(current_loss)
            iter += 1
        
    def test1(self):
        goods = 0
        bads = 0
        for sentence, target in DataSet().training_pairs + DataSet().testing_pairs:
            if target.split() == self.model.infer(sentence):
                goods += 1
            else:
                bads += 1
                print("sentence: %s\ntarget: %s\nactual: %s" % (sentence, target, ' '.join(self.model.infer(sentence))))
        print("%s good, %s bad" % (goods, bads))

    def test2(self):
        arithmetic_grammar = GeneratingGrammar(TestMethods.arithmetic_rules)
        self.assertEqual(3, len(arithmetic_grammar.binary_rules))
        self.assertEqual(7, len(arithmetic_grammar.lexical_rules))
        self.assertEqual(4, len(arithmetic_grammar.gen_rules))
        self.assertEqual(6, len(arithmetic_grammar.gen_rules['$E']))
        

if __name__ == "__main__":
    unittest.main()
