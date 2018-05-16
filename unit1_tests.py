import unittest
from collections import defaultdict
from grammar import Grammar, Rule, Parse
from scoring import Model
from example import Example


class TestMethodsUnit1(unittest.TestCase):
    def test_grammar_creation(self):
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

        arithmetic_grammar = Grammar(arithmetic_rules)
        self.assertEqual(3, len(arithmetic_grammar.binary_rules))
        self.assertEqual(7, len(arithmetic_grammar.lexical_rules))
		
    # Chart Parsing Algorithm: Grammar.parse() produces a list of Parse objects
    
    # 14 examples that produce a single parse
    # Ignore semantics and denotation initially
    one_parse_examples = [
        Example(input="one plus one", semantics=('+', 1, 1), denotation=2),
        Example(input="one plus two", semantics=('+', 1, 2), denotation=3),
        Example(input="one plus three", semantics=('+', 1, 3), denotation=4),
        Example(input="two plus two", semantics=('+', 2, 2), denotation=4),
        Example(input="two plus three", semantics=('+', 2, 3), denotation=5),
        Example(input="three plus one", semantics=('+', 3, 1), denotation=4),
        Example(
			input="three plus minus two", 
			semantics=('+', 3, ('~', 2)), 
			denotation=1),
        Example(input="two plus two", semantics=('+', 2, 2), denotation=4),
        Example(input="three minus two", semantics=('-', 3, 2), denotation=1),
        Example(input="two times two", semantics=('*', 2, 2), denotation=4),
        Example(input="two times three", semantics=('*', 2, 3), denotation=6),
        Example(input="minus three", semantics=('~', 3), denotation=-3),
        Example(input="three plus two", semantics=('+', 3, 2), denotation=5),
        Example(input="minus four", semantics=('~', 4), denotation=-4),
    ]
    
    # 3 examples that produce two parses each
    # Ignore semantics and denotation initially
    two_parse_examples = [
        Example(
			input="minus three minus two", 
			semantics=('-', ('~', 3), 2), 
			denotation=-5),
        Example(
			input="three plus three minus two", 
			semantics=('-', ('+', 3, 3), 2), 
			denotation=4),
        Example(
			input="two times two plus three", 
			semantics=('+', ('*', 2, 2), 3), 
			denotation=7),
    ]

    def test_parsing(self):
        # Ignoring semantics for now...
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

        arithmetic_grammar = Grammar(arithmetic_rules)
        for example in self.one_parse_examples:
            self.assertEqual(1, len(arithmetic_grammar.parse(example.input)), example)
            # print(arithmetic_grammar.parse(example.input)[0])
        for example in self.two_parse_examples:
            self.assertEqual(2, len(arithmetic_grammar.parse(example.input)), example)
            # print(arithmetic_grammar.parse(example.input)[0])
            # print(arithmetic_grammar.parse(example.input)[1])
            
            # SLIDES
            

    # The semantic attachment to a rule specifies how to construct the semantics
    # for the parent (LHS) category.
    numeral_rules = [
        Rule('$E', 'one', 1),
        Rule('$E', 'two', 2),
        Rule('$E', 'three', 3),
        Rule('$E', 'four', 4),
    ]

    operator_rules = [
        Rule('$UnOp', 'minus', '~'),
        Rule('$BinOp', 'plus', '+'),
        Rule('$BinOp', 'minus', '-'),
        Rule('$BinOp', 'times', '*'),
    ]

    # For compositional rules, the semantic attachments are functions which
    # specify how to construct the semantics of the parent from the semantics of
    # the children. 
    compositional_rules = [
        Rule('$E', '$UnOp $E', lambda sems: (sems[0], sems[1])),
        Rule('$EBO', '$E $BinOp', lambda sems: (sems[1], sems[0])),
        Rule('$E', '$EBO $E', lambda sems: (sems[0][0], sems[0][1], sems[1]))
    ]

    arithmetic_rules = numeral_rules + operator_rules + compositional_rules
    
    # grammar.py:Parse.compute_semantics() adds semantics to each parse object

    def test_semantics(self):
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        parses = arithmetic_grammar.parse("two times two plus three")
        self.assertEqual(2, len(parses))
        self.assertEqual(('*', 2, ('+', 2, 3)), parses[0].semantics)
        self.assertEqual(('+', ('*', 2, 2), 3), parses[1].semantics)
        
    def test_evaluation(self):
        """
        Evaluate the grammar on all examples, collecting metrics:
        
        semantics oracle accuracy: # of examples where one parse or the other was
        correct.

        semantics accuracy: # of examples where parse at position 0 was correct.
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        
        from executor import Executor

        examples = self.one_parse_examples + self.two_parse_examples
        self.assertEqual(17, len(examples))

        metrics = arithmetic_grammar.evaluate(
            executor=Executor.execute, 
            examples=examples,
            print_examples=False)

        # three examples where the parse at position 0 was not correct
        self.assertEqual(metrics['semantics accuracy'], 14) 
        # in every example we produced some correct parse.
        self.assertEqual(metrics['semantics oracle accuracy'], 17)
        
        # SLIDES

    def test_rule_features(self):
        """
        See if a count of occurrence of a rule is a good feature for
        ranking parses.
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        parses = arithmetic_grammar.parse("two times two")
        self.assertEqual(1, len(parses))
        # Look at Parse.rule_features()
        rule_features = parses[0].rule_features()
        self.assertEqual(1, rule_features[str(self.compositional_rules[2])])
        self.assertEqual(1, rule_features[str(self.compositional_rules[1])])
        self.assertEqual(1, rule_features[str(self.operator_rules[3])])
        self.assertEqual(2, rule_features[str(self.numeral_rules[1])])

        parses = arithmetic_grammar.parse("two times two plus three")
        self.assertEqual(2, len(parses))
        # Parse.rule_features() is not good at distinguishing parses
        self.assertEqual(
            parses[0].rule_features(), 
            parses[1].rule_features())

    weights = defaultdict(float)
    weights[('*', '+')] = 1.0
    weights[('*', '-')] = 1.0
    weights[('~', '+')] = 1.0
    weights[('~', '-')] = 1.0
    weights[('+', '*')] = -1.0
    weights[('-', '*')] = -1.0
    weights[('+', '~')] = -1.0
    weights[('-', '~')] = -1.0

    def test_operator_precedence_features(self):
        """
        See if a count of operator precedence patterns is a good feature for 
        ranking parses.
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        parses = arithmetic_grammar.parse("two times two plus three")
        self.assertEqual(2, len(parses))
        # Look at Parse.operator_precedence_features(). It generates different
        # results for the two parses
        parse0_features = parses[0].operator_precedence_features()
        parse1_features = parses[1].operator_precedence_features()
        # In the first parse, + precedes * once
        self.assertEqual(parse0_features, {('+', '*'): 1.0})
        # In the second parse, * precedes + once
        self.assertEqual(parse1_features, {('*', '+'): 1.0})
        
        # Look at Parse.score()
        parse0_score = parses[0].score(
            Parse.operator_precedence_features, 
            self.weights)
        parse1_score = parses[1].score(
            Parse.operator_precedence_features, 
            self.weights)
        # Parse.operator_precedence_features() is good at distinguishing parses
        self.assertEqual(-1.0, parse0_score)
        self.assertEqual(1.0, parse1_score)

    def test_evaluation_with_scoring(self):
        """
        Evaluate the grammar on all examples, collecting metrics:
        
        semantics oracle accuracy: # of examples where one parse or the other was
        correct.

        semantics accuracy: # of examples where parse at position 0 was correct.
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        
        from executor import Executor

        arithmetic_model = Model(
            grammar=arithmetic_grammar,
            feature_fn=Parse.operator_precedence_features,
            weights=self.weights,
            executor=Executor.execute)

        from experiment import evaluate_model

        metrics = evaluate_model(
            model=arithmetic_model,
            examples=self.one_parse_examples + self.two_parse_examples
        )
        self.assertEqual(metrics['semantics accuracy'], 16) # Improvement
        self.assertEqual(metrics['semantics oracle accuracy'], 17)
        # Exercise: introduce new features to address "three plus three minus two"
        
        # SLIDES

    def test_learning_from_semantics(self):
        """
        First 13 examples are used for training.
        Last 4 examples are used for testing.
        b_trn: performance metrics on training set before training
        b_tst: performance metrics on test set before training
        a_trn: performance metrics on training set after training
        a_tst: performance metrics on test set after training

        semantics accuracy: # of examples where parse at position 0 was correct.
        denotation accuracy: # of examples where denotation of parse at position 
        0 was correct
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        arithmetic_examples = self.two_parse_examples + self.one_parse_examples
        
        from executor import Executor

        arithmetic_model = Model(
            grammar=arithmetic_grammar,
            feature_fn=Parse.operator_precedence_features,
            weights=defaultdict(float), # Initialize with all weights at zero
            executor=Executor.execute)
            
        # Train based on correct/incorrect semantics
        from metrics import SemanticsAccuracyMetric
        
        b_trn, b_tst, a_trn, a_tst = arithmetic_model.train_test(
            train_examples=arithmetic_examples[:13],
            test_examples=arithmetic_examples[13:],
            training_metric=SemanticsAccuracyMetric(),
            seed=1)

        # BEFORE SGD
        self.assertEqual(b_trn['semantics accuracy'], 10)
        self.assertEqual(b_trn['denotation accuracy'], 11)
        self.assertEqual(b_tst['semantics accuracy'], 4)
        self.assertEqual(b_tst['denotation accuracy'], 4)
        
        # AFTER SGD
        self.assertEqual(a_trn['semantics accuracy'], 13) # Improvement
        self.assertEqual(a_trn['denotation accuracy'], 13) # Improvement
        self.assertEqual(a_tst['semantics accuracy'], 4)
        self.assertEqual(a_tst['denotation accuracy'], 4)

    def test_learning_from_denotation(self):
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        arithmetic_examples = self.two_parse_examples + self.one_parse_examples
        
        from executor import Executor

        arithmetic_model = Model(
            grammar=arithmetic_grammar,
            feature_fn=Parse.operator_precedence_features,
            weights=defaultdict(float), # Initialize with all weights at zero
            executor=Executor.execute)
            
        # Train based on correct/incorrect denotation
        from metrics import DenotationAccuracyMetric
        
        b_trn, b_tst, a_trn, a_tst = arithmetic_model.train_test(
            train_examples=arithmetic_examples[:13],
            test_examples=arithmetic_examples[13:],
            training_metric=DenotationAccuracyMetric(),
            seed=1)

        # BEFORE SGD
        self.assertEqual(b_trn['semantics accuracy'], 10)
        self.assertEqual(b_tst['denotation accuracy'], 4)
        
        # AFTER SGD
        self.assertEqual(a_trn['semantics accuracy'], 12) # Improvement
        self.assertEqual(a_trn['denotation accuracy'], 13) # Improvement

    def test_learning_from_many_denotations(self):
        """
        Large number of examples are used for training.
        Last 4 arithmetic_examples are used for testing.
        b_trn: performance metrics on training set before training
        a_trn: performance metrics on training set after training

        denotation accuracy: # of examples where denotation of parse at position 
        0 was correct
        """
        arithmetic_grammar = Grammar(self.arithmetic_rules)
        arithmetic_examples = self.two_parse_examples + self.one_parse_examples
        
        from executor import Executor

        arithmetic_model = Model(
            grammar=arithmetic_grammar,
            feature_fn=Parse.operator_precedence_features,
            weights=defaultdict(float), # Initialize with all weights at zero
            executor=Executor.execute)
            
        from metrics import DenotationAccuracyMetric
        from arithmetic import arithmetic_dev_examples
        
        b_trn, b_tst, a_trn, a_tst = arithmetic_model.train_test(
            train_examples=arithmetic_dev_examples,
            test_examples=arithmetic_examples[13:],
            training_metric=DenotationAccuracyMetric(),
            seed=1)

        # BEFORE SGD
        self.assertEqual(b_trn['denotation accuracy'], 64)
        
        # AFTER SGD
        self.assertEqual(a_trn['denotation accuracy'], 92) # Improvement
        
        # SLIDES
    
if __name__ == "__main__":
    unittest.main()