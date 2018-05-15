__author__ = "Bill MacCartney"
__copyright__ = "Copyright 2015, Bill MacCartney"
__credits__ = []
__license__ = "GNU General Public License, version 2.0"
__version__ = "0.9"
__maintainer__ = "Bill MacCartney"
__email__ = "See the author's website"

from collections import defaultdict

from parsing import Parse

# TODO: annotations are generating rule features -- they shouldn't.
def rule_features(parse):
    """
    Returns a map from (string representations of) rules to how often they were
    used in the given parse.
    """
    def collect_rule_features(parse, features):
        feature = str(parse.rule)
        features[feature] += 1.0
        for child in parse.children:
            if isinstance(child, Parse):
                collect_rule_features(child, features)
    features = defaultdict(float)
    collect_rule_features(parse, features)
    return features

def score(parse=None, feature_fn=None, weights=None):
    """Returns the inner product of feature_fn(parse) and weights."""
    assert parse and feature_fn and weights != None
    return sum(weights[feature] * value for feature, value in list(feature_fn(parse).items()))

class Model:
    def __init__(self,
                 grammar=None,
                 feature_fn=lambda parse: defaultdict(float),
                 weights=defaultdict(float),
                 executor=None):
        assert grammar
        self.grammar = grammar
        self.feature_fn = feature_fn
        self.weights = weights
        self.executor = executor

    # TODO: Should this become a static function, to match style of parsing.py?
    def parse_input(self, input):
        parses = self.grammar.parse(input)
        for parse in parses:
            if self.executor:
                parse.denotation = self.executor(parse.semantics)
            parse.score = score(parse, self.feature_fn, self.weights)
        return sorted(parses, key=lambda parse: parse.score, reverse=True)

    from metrics import standard_metrics

    def evaluate(self, examples=[], examples_label=None, metrics=standard_metrics(), print_examples=False):
        metric_values = defaultdict(int)
        for example in examples:
            parses = self.parse_input(example.input)
            for metric in metrics:
                metric_value = metric.evaluate(example, parses)
                metric_values[metric.name()] += metric_value
            if print_examples:
                print_parses(example, parses, metrics=metrics)
        return metric_values

    def clone(self):
        return Model(
            grammar=self.grammar,
            feature_fn=self.feature_fn,
            weights=defaultdict(float),  # Zero the weights.
            executor=self.executor)

    def update_weights(self, target_parse, predicted_parse, eta):
        target_features = self.feature_fn(target_parse)
        predicted_features = self.feature_fn(predicted_parse)
        for f in target_features.keys():
            update = target_features[f] - predicted_features[f]
            if update != 0.0:
                self.weights[f] += eta * update
        for f in predicted_features.keys():
            update = target_features[f] - predicted_features[f]
            if update != 0.0:
                self.weights[f] += eta * update
        
    def train(self, examples=[], training_metric=None, epochs=10, eta=0.1, seed=None):
        """
        Run Stochastic Gradient Descent (SGD) with a training metric on a clone of the model. Return the
        clone.
        """

        def print_weights(weights, n=20):
            pairs = [(value, str(key)) for key, value in list(weights.items()) if value != 0]
            pairs = sorted(pairs, reverse=True)
            print()
            if len(pairs) < n * 2:
                print('Feature weights:')
                for value, key in pairs:
                    print('%8.1f\t%s' % (value, key))
            else:
                print('Top %d and bottom %d feature weights:' % (n, n))
                for value, key in pairs[:n]:
                    print('%8.1f\t%s' % (value, key))
                print('%8s\t%s' % ('...', '...'))
                for value, key in pairs[-n:]:
                    print('%8.1f\t%s' % (value, key))
            print()

        def cost(parse_1, parse_2):
            return 0.0 if parse_1 == parse_2 else 1.0

        import random

        if seed:
            random.seed(seed)
        model = self.clone()
        for epoch_num in range(epochs):
            random.shuffle(examples)
            num_correct = 0
            for example in examples:
                # Parse input with current weights.
                parses = model.parse_input(example.input)
                # Get the highest-scoring "good" parse among the candidate parses.
                good_parses = []
                for p in parses:
                    if training_metric.evaluate(example, [p]):
                        good_parses.append(p)
                if good_parses:
                    target_parse = good_parses[0]
                    # Get all (score, parse) pairs.
                    scores = [(p.score + cost(target_parse, p), p) for p in parses]
                    # Get the maximal score.
                    max_score = sorted(scores)[-1][0]
                    # Get all the candidates with the max score and choose one randomly.
                    predicted_parse = random.choice([p for s, p in scores if s == max_score])
                    if training_metric.evaluate(example, [predicted_parse]):
                        num_correct += 1
                    model.update_weights(target_parse, predicted_parse, eta)
            # print('SGD iteration %d: train accuracy: %.3f' % (epoch_num, 1.0 * num_correct / len(examples)))
        # print_weights(model.weights)
        return model
        
    from metrics import SemanticsAccuracyMetric

    def train_test(self,
                   train_examples=[],
                   test_examples=[],
                   metrics=standard_metrics(),
                   training_metric=SemanticsAccuracyMetric(),
                   seed=None,
                   print_examples=False):

        # 'Before' test
        self.weights = defaultdict(float)  # no weights
        before_train_metrics = self.evaluate(
            examples=train_examples,
            examples_label='train',
            metrics=metrics,
            print_examples=print_examples)
        before_test_metrics = self.evaluate(
            examples=test_examples,
            examples_label='test',
            metrics=metrics,
            print_examples=print_examples)

        # Train
        model = self.train(train_examples, training_metric=training_metric, seed=seed)

        # 'After' test
        after_train_metrics = model.evaluate(
            examples=train_examples,
            examples_label='train',
            metrics=metrics,
            print_examples=print_examples)
        after_test_metrics = model.evaluate(
            examples=test_examples,
            examples_label='test',
            metrics=metrics,
            print_examples=print_examples)
        return before_train_metrics, before_test_metrics, after_train_metrics, after_test_metrics