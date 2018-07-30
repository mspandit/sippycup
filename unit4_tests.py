# -*- coding: utf-8 -*-
from grammar import Rule, Grammar
from collections import defaultdict
from torch_fw import Module, Embedding, GRU, SGD, Linear, Dropout, Variable
from torch_fw import LongTensor, zeros, NLLLoss, log_softmax, softmax, relu, manual_seed
from torch_fw import cat, bmm, LogSoftmax
import copy

class GeneratingGrammar(Grammar):
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


class EncoderRNN(Module):
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = Embedding(input_size, hidden_size)
        self.gru = GRU(hidden_size, hidden_size)
        self.optimizer = SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        return self.gru(self.embedding(input).view(1, 1, -1), hidden)

    def initHiddenVariable(self):
        return Variable(zeros(1, 1, self.hidden_size))

    def initOutputVariables(self, max_length):
        return Variable(zeros(max_length, self.hidden_size))

    def function(self, inputs, max_length):
        encoder_outputs = self.initOutputVariables(max_length)
        encoder_hidden = self.initHiddenVariable()

        for ei, inp in enumerate(inputs):
            # calls forward()
            encoder_output, encoder_hidden = self(
                inp, 
                encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
        return encoder_outputs, encoder_hidden


class DecoderRNN(Module):
    def __init__(self, hidden_size, output_size, learning_rate=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = Embedding(output_size, hidden_size)
        self.gru = GRU(hidden_size, hidden_size)
        self.out = Linear(hidden_size, output_size)
        self.softmax = LogSoftmax(dim=1)
        self.optimizer = SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        output, hidden = self.gru(
            relu(self.embedding(input).view(1, 1, -1)), 
            hidden)
        return self.softmax(self.out(output[0])), hidden

    def initHiddenVariable(self):
        return Variable(zeros(1, 1, self.hidden_size))

    def initInputVariable(self, token):
        return Variable(LongTensor([[token]]))

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


class BeamSearchState(object):
    """docstring for BeamSearchState"""
    def __init__(self, token_list, finished, decoder_input, hidden, probability=0.0):
        super(BeamSearchState, self).__init__()
        self.token_list = token_list
        self.finished = finished
        self.decoder_input = decoder_input
        self.hidden = hidden
        self.probability = probability

    def initInputVariable(self, token):
        return Variable(LongTensor([[token]]))

    def advance(self, decoder, encoder_outputs, EOS_token):
        """docstring for advance"""
        decoder_output, hidden = decoder(self.decoder_input, self.hidden, encoder_outputs)
        sorted_probabilities, sorted_predictions = decoder_output.data.topk(2)
        retval = []
        for np, ni in zip(sorted_probabilities[0], sorted_predictions[0]):
            if ni == EOS_token:
                retval.append(BeamSearchState(self.token_list, True, self.initInputVariable(ni), hidden, self.probability + np))
            else:
                retval.append(BeamSearchState(self.token_list + [ni], False, self.initInputVariable(ni), hidden, self.probability + np))
        return retval

class AttnDecoderRNN(Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=20, learning_rate=0.01):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = Embedding(self.output_size, self.hidden_size)
        self.attn = Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = Dropout(self.dropout_p)
        self.gru = GRU(self.hidden_size, self.hidden_size)
        self.out = Linear(self.hidden_size, self.output_size)
        self.optimizer = SGD(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))

        attn_weights = softmax(
            self.attn(cat((embedded[0], hidden[0]), 1)), dim=1)

        output, hidden = self.gru(
            relu(
                self.attn_combine(
                    cat((
                        embedded[0], 
                        bmm(attn_weights.unsqueeze(0),
                        encoder_outputs.unsqueeze(0))[0]), 
                    1)).unsqueeze(0)), 
            hidden)
        return log_softmax(self.out(output[0]), dim=1), hidden

    def initHiddenVariable(self):
        return Variable(zeros(1, 1, self.hidden_size))

    def initInputVariable(self, token):
        return Variable(LongTensor([[token]]))

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

    def infer(self, encoder_outputs, hidden, SOS_token, EOS_token, max_length, search_states=None, beam_width=20):
        if search_states is None:
            return self.infer(
                encoder_outputs,
                hidden,
                SOS_token,
                EOS_token,
                max_length,
                [BeamSearchState([], False, self.initInputVariable(SOS_token), hidden)],
                beam_width)
        elif any([not s.finished for s in search_states]):
            new_search_states = []
            for state in search_states:
                if state.finished:
                    new_search_states += [state]
                elif len(state.token_list) < max_length:
                    new_search_states += state.advance(
                        self,
                        encoder_outputs,
                        EOS_token)
            new_search_states.sort(
                key=lambda state: state.probability, reverse=True)
            new_search_states = new_search_states[:beam_width]
            return self.infer(
                encoder_outputs,
                hidden,
                SOS_token,
                EOS_token,
                max_length,
                new_search_states,
                beam_width)
        else:
            return search_states


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
        return Variable(LongTensor(indexes).view(-1, 1))

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
                NLLLoss(), 
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
        inference_states = self.decoder.infer(
            encoder_outputs,
            hidden, 
            self.grammar.input_lang()['SOS'],
            self.grammar.output_lang()['EOS'],
            2 * self.grammar.MAX_LENGTH)
        return [[self.grammar.output_vocabulary[t] for t in state.token_list] for state in inference_states]

import unittest
import random
random.seed(1234)
manual_seed(1234)

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
        cls.model = Model(128, GeneratingGrammar(TestMethods.arithmetic_rules))
        loss = cls.model.train()
        current_loss = 1
        iter = 0
        while current_loss > 5e-4:
            current_loss = next(loss)
            if (0 == iter % 1000):
                print(current_loss)
            iter += 1
        
    def test1(self):
        goods = 0
        bads = 0
        for sentence, target in DataSet().training_pairs + DataSet().testing_pairs:
            inferences = self.model.infer(sentence)
            if target.split() in inferences:
                goods += 1
            else:
                bads += 1
                print("sentence: %s\ntarget: %s" % (sentence, target))
                for inference in inferences:
                    print("actual: %s" % (' '.join(inference)))
        print("%s good, %s bad" % (goods, bads))

    def test2(self):
        arithmetic_grammar = GeneratingGrammar(TestMethods.arithmetic_rules)
        self.assertEqual(3, len(arithmetic_grammar.binary_rules))
        self.assertEqual(7, len(arithmetic_grammar.lexical_rules))
        self.assertEqual(4, len(arithmetic_grammar.gen_rules))
        self.assertEqual(6, len(arithmetic_grammar.gen_rules['$E']))
        

if __name__ == "__main__":
    unittest.main()
