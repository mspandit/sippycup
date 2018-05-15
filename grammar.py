from collections import defaultdict
from itertools import product

from scoring import Model

class Parse(object):
    def __init__(self, rule, children):
        super(Parse, self).__init__()
        self.rule = rule
        self.children = tuple(children[:])
        # Look at __str__()
        self.semantics = self.compute_semantics()
        self.denotation = None

    def __str__(self):
        return '(%s %s)' % (
            self.rule.lhs, 
            ' '.join([str(c) for c in self.children]))
        # Go back to Grammar.parse()
        
    def compute_semantics(self):
        from types import FunctionType

        if Grammar.is_lexical(self.rule) or not isinstance(self.rule.sem, FunctionType):
            return self.rule.sem
        else:
            return self.rule.sem([child.semantics for child in self.children])
        # Go back to unit1_tests.py

    def rule_features(self):
        """
        Returns a map from (string representation of) a rule to its count of 
        usage in the parse. In other words, returns the results of multiple
        feature functions on the parse, one feature function for each rule,
        resulting in the count of that rule's usage in the parse.
        """
        def collect_rule_features(parse, features):
            feature = str(parse.rule)
            features[feature] += 1.0
            for child in parse.children:
                if isinstance(child, Parse):
                    collect_rule_features(child, features)
        features = defaultdict(float)
        collect_rule_features(self, features)
        return features

    def operator_precedence_features(self):
        """
        Traverses the arithmetic expression tree which forms the semantics of
        the parse and adds a feature (op1, op2) whenever op1 appears
        lower in the tree than (i.e. with higher precedence than) than op2.
        In other words, returns the results of multiple feature functions on
        the parse, one feature function for each (op1, op2) pair, resulting
        in a count of when op1 has higher precedence than op2.
        """
        def collect_features(semantics, features):
            if isinstance(semantics, tuple):
                for child in semantics[1:]:
                    collect_features(child, features)
                    if isinstance(child, tuple) and child[0] != semantics[0]:
                        features[(child[0], semantics[0])] += 1.0
        features = defaultdict(float)
        collect_features(self.semantics, features)
        return features

    def score(self, feature_fn, weights):
        """Returns the inner product of feature_fn(self) and weights."""
        addends = []
        for feature, value in feature_fn(self).items():
            addends.append(weights[feature] * value)
        return sum(addends)


class Rule(object):
    def __init__(self, lhs, rhs, sem=None):
        super(Rule, self).__init__()
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem


class Grammar(object):
    def __init__(self, rules=[]):
        super(Grammar, self).__init__()
		# values are instances of list
        self.lexical_rules = defaultdict(list) 
        self.binary_rules = defaultdict(list)
        for rule in rules:
            self.add_rule(rule)

    @staticmethod
    def is_cat(label):
        return label.startswith('$')
            
    @staticmethod
    def is_lexical(rule):
        return all([not Grammar.is_cat(rhsi) for rhsi in rule.rhs])
    
    @staticmethod
    def is_binary(rule):
        return len(rule.rhs) == 2 and Grammar.is_cat(rule.rhs[0]) and Grammar.is_cat(rule.rhs[1])
    
    def add_rule(self, rule):
        if self.is_lexical(rule):
            self.lexical_rules[rule.rhs].append(rule)
        elif self.is_binary(rule):
            self.binary_rules[rule.rhs].append(rule)
        else:
            raise Exception('Cannot accept rule: %s', rule)

    def apply_lexical_rules(self, chart, tokens, start, end):
        """
        Add parses to span (i, j) in chart by applying lexical rules to 
        tokens.
        """
        for rule in self.lexical_rules[tuple(tokens[start:end])]:
            chart[(start, end)].append(Parse(rule, tokens[start:end]))
            # Look at beginning of class Parse

    def apply_binary_rules(self, chart, start, end):
        """Add parses to span (i, j) in chart by applying binary rules."""
        # All ways of splitting the span into two subspans---why we require
        # the grammar to be in Chomsky normal form.
        for split in range(start + 1, end): 
            for parse_left, parse_right in product(
                chart[(start, split)], 
                chart[(split, end)]):
                for rule in self.binary_rules[
                    (parse_left.rule.lhs, parse_right.rule.lhs)]:
                    chart[(start, end)].append(
                        Parse(rule, [parse_left, parse_right]))

    def parse(self, input):
        """Returns a list of all parses for input using the grammar."""
        tokens = input.split()
        chart = defaultdict(list) # map from span (i, j) to list of parses
        for end in range(1, len(tokens) + 1):
            for start in range(end - 1, -1, -1):
                self.apply_lexical_rules(chart, tokens, start, end)
                self.apply_binary_rules(chart, start, end)
        return chart[(0, len(tokens))] # return all parses for full span
    
    from metrics import standard_metrics

    def evaluate(
        self,
        executor=None,
        examples=[],
        examples_label=None,
        metrics=standard_metrics(),
        print_examples=False):
        return Model(grammar=self, executor=executor).evaluate(
            examples=examples,
            metrics=metrics,
            print_examples=print_examples)


