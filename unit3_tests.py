from collections import defaultdict
from annotator import Annotator, NumberAnnotator
from grammar import Rule
from unit2_tests import Unit2Grammar
from geobase import GeobaseReader
from graph_kb import GraphKB
from geoquery import GeoQueryDomain

import unittest


class GeobaseAnnotator(Annotator):
    def __init__(self, geobase):
        self.geobase = geobase

    def annotate(self, tokens):
        phrase = ' '.join(tokens)
        places = self.geobase.binaries_rev['name'][phrase]
        return [('$Entity', place) for place in places]


class Unit3Grammar(Unit2Grammar):
    def __init__(self, rules=[], annotators=[]):
        super(Unit3Grammar, self).__init__(rules, annotators)

    @staticmethod
    def reverse(relation_sem):
        """TODO"""
        # relation_sem is a lambda function which takes an arg and forms a pair,
        # either (rel, arg) or (arg, rel).  We want to swap the order of the pair.
        def apply_and_swap(arg):
            pair = relation_sem(arg)
            return (pair[1], pair[0])
        return apply_and_swap        


class TestMethods(unittest.TestCase):
    optional_words = [
        'the', '?', 'what', 'is', 'in', 'of', 'how', 'many', 'are', 'which', 'that',
        'with', 'has', 'major', 'does', 'have', 'where', 'me', 'there', 'give',
        'name', 'all', 'a', 'by', 'you', 'to', 'tell', 'other', 'it', 'do', 'whose',
        'show', 'one', 'on', 'for', 'can', 'whats', 'urban', 'them', 'list',
        'exist', 'each', 'could', 'about'
    ]

    rules_optionals = [
        Rule('$ROOT', '?$Optionals $Query ?$Optionals', lambda sems: sems[1]),
        Rule('$Optionals', '$Optional ?$Optionals'),
    ] + [Rule('$Optional', word) for word in optional_words]

    rules_collection_entity = [
        Rule('$Query', '$Collection', lambda sems: sems[0]),
        Rule('$Collection', '$Entity', lambda sems: sems[0]),
    ]

    reader = GeobaseReader()
    geobase = GraphKB(reader.tuples)
    annotators = [NumberAnnotator(), GeobaseAnnotator(geobase)]

    def test_simple_grammar(self):
        rules = self.rules_optionals + self.rules_collection_entity
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('what is utah')
        self.assertEqual('/state/utah', parses[0].semantics)
        self.assertEqual(
            ('/state/utah',), 
            self.geobase.executor().execute(parses[0].semantics))

    domain = GeoQueryDomain()

    def test_evaluate_simple_grammar(self):
        from experiment import sample_wins_and_losses
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = self.rules_optionals + self.rules_collection_entity
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(17, metric_values['number of parses'])

    rules_types = [
        Rule('$Collection', '$Type', lambda sems: sems[0]),

        Rule('$Type', 'state', 'state'),
        Rule('$Type', 'states', 'state'),
        Rule('$Type', 'city', 'city'),
        Rule('$Type', 'cities', 'city'),
        Rule('$Type', 'big cities', 'city'),
        Rule('$Type', 'towns', 'city'),
        Rule('$Type', 'river', 'river'),
        Rule('$Type', 'rivers', 'river'),
        Rule('$Type', 'mountain', 'mountain'),
        Rule('$Type', 'mountains', 'mountain'),
        Rule('$Type', 'mount', 'mountain'),
        Rule('$Type', 'peak', 'mountain'),
        Rule('$Type', 'road', 'road'),
        Rule('$Type', 'roads', 'road'),
        Rule('$Type', 'lake', 'lake'),
        Rule('$Type', 'lakes', 'lake'),
        Rule('$Type', 'country', 'country'),
        Rule('$Type', 'countries', 'country'),
    ]
    
    def test_grammar_with_types(self):
        rules = self.rules_optionals + self.rules_collection_entity + self.rules_types
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('name the lakes')
        self.assertEqual(
            ('/lake/becharof', '/lake/champlain', '/lake/erie', '/lake/flathead', '/lake/great_salt_lake', '/lake/huron', '/lake/iliamna', '/lake/lake_of_the_woods', '/lake/michigan', '/lake/mille_lacs', '/lake/naknek', '/lake/okeechobee', '/lake/ontario', '/lake/pontchartrain', '/lake/rainy', '/lake/red', '/lake/salton_sea', '/lake/st._clair', '/lake/superior', '/lake/tahoe', '/lake/teshekpuk', '/lake/winnebago'),
            self.geobase.executor().execute(parses[0].semantics))

    def test_evaluate_grammar_with_types(self):
        from experiment import sample_wins_and_losses
        from geoquery import GeoQueryDomain
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = self.rules_optionals + self.rules_collection_entity + self.rules_types
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(20, metric_values['number of parses'])

    rules_relations = [
        Rule('$Collection', '$Relation ?$Optionals $Collection', lambda sems: sems[0](sems[2])),

        Rule('$Relation', '$FwdRelation', lambda sems: (lambda arg: (sems[0], arg))),
        Rule('$Relation', '$RevRelation', lambda sems: (lambda arg: (arg, sems[0]))),

        Rule('$FwdRelation', '$FwdBordersRelation', 'borders'),
        Rule('$FwdBordersRelation', 'border'),
        Rule('$FwdBordersRelation', 'bordering'),
        Rule('$FwdBordersRelation', 'borders'),
        Rule('$FwdBordersRelation', 'neighbor'),
        Rule('$FwdBordersRelation', 'neighboring'),
        Rule('$FwdBordersRelation', 'surrounding'),
        Rule('$FwdBordersRelation', 'next to'),

        Rule('$FwdRelation', '$FwdTraversesRelation', 'traverses'),
        Rule('$FwdTraversesRelation', 'cross ?over'),
        Rule('$FwdTraversesRelation', 'flow through'),
        Rule('$FwdTraversesRelation', 'flowing through'),
        Rule('$FwdTraversesRelation', 'flows through'),
        Rule('$FwdTraversesRelation', 'go through'),
        Rule('$FwdTraversesRelation', 'goes through'),
        Rule('$FwdTraversesRelation', 'in'),
        Rule('$FwdTraversesRelation', 'pass through'),
        Rule('$FwdTraversesRelation', 'passes through'),
        Rule('$FwdTraversesRelation', 'run through'),
        Rule('$FwdTraversesRelation', 'running through'),
        Rule('$FwdTraversesRelation', 'runs through'),
        Rule('$FwdTraversesRelation', 'traverse'),
        Rule('$FwdTraversesRelation', 'traverses'),

        Rule('$RevRelation', '$RevTraversesRelation', 'traverses'),
        Rule('$RevTraversesRelation', 'has'),
        Rule('$RevTraversesRelation', 'have'),  # 'how many states have major rivers'
        Rule('$RevTraversesRelation', 'lie on'),
        Rule('$RevTraversesRelation', 'next to'),
        Rule('$RevTraversesRelation', 'traversed by'),
        Rule('$RevTraversesRelation', 'washed by'),

        Rule('$FwdRelation', '$FwdContainsRelation', 'contains'),
        # 'how many states have a city named springfield'
        Rule('$FwdContainsRelation', 'has'),
        Rule('$FwdContainsRelation', 'have'),

        Rule('$RevRelation', '$RevContainsRelation', 'contains'),
        Rule('$RevContainsRelation', 'contained by'),
        Rule('$RevContainsRelation', 'in'),
        Rule('$RevContainsRelation', 'found in'),
        Rule('$RevContainsRelation', 'located in'),
        Rule('$RevContainsRelation', 'of'),

        Rule('$RevRelation', '$RevCapitalRelation', 'capital'),
        Rule('$RevCapitalRelation', 'capital'),
        Rule('$RevCapitalRelation', 'capitals'),

        Rule('$RevRelation', '$RevHighestPointRelation', 'highest_point'),
        Rule('$RevHighestPointRelation', 'high point'),
        Rule('$RevHighestPointRelation', 'high points'),
        Rule('$RevHighestPointRelation', 'highest point'),
        Rule('$RevHighestPointRelation', 'highest points'),

        Rule('$RevRelation', '$RevLowestPointRelation', 'lowest_point'),
        Rule('$RevLowestPointRelation', 'low point'),
        Rule('$RevLowestPointRelation', 'low points'),
        Rule('$RevLowestPointRelation', 'lowest point'),
        Rule('$RevLowestPointRelation', 'lowest points'),
        Rule('$RevLowestPointRelation', 'lowest spot'),

        Rule('$RevRelation', '$RevHighestElevationRelation', 'highest_elevation'),
        Rule('$RevHighestElevationRelation', '?highest elevation'),

        Rule('$RevRelation', '$RevHeightRelation', 'height'),
        Rule('$RevHeightRelation', 'elevation'),
        Rule('$RevHeightRelation', 'height'),
        Rule('$RevHeightRelation', 'high'),
        Rule('$RevHeightRelation', 'tall'),

        Rule('$RevRelation', '$RevAreaRelation', 'area'),
        Rule('$RevAreaRelation', 'area'),
        Rule('$RevAreaRelation', 'big'),
        Rule('$RevAreaRelation', 'large'),
        Rule('$RevAreaRelation', 'size'),

        Rule('$RevRelation', '$RevPopulationRelation', 'population'),
        Rule('$RevPopulationRelation', 'big'),
        Rule('$RevPopulationRelation', 'large'),
        Rule('$RevPopulationRelation', 'populated'),
        Rule('$RevPopulationRelation', 'population'),
        Rule('$RevPopulationRelation', 'populations'),
        Rule('$RevPopulationRelation', 'populous'),
        Rule('$RevPopulationRelation', 'size'),

        Rule('$RevRelation', '$RevLengthRelation', 'length'),
        Rule('$RevLengthRelation', 'length'),
        Rule('$RevLengthRelation', 'long'),
    ]
    
    def test_grammar_with_relations(self):
        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types
            + self.rules_relations)
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('what is the capital of vermont ?')
        self.assertEqual(('/state/vermont', 'capital'), parses[0].semantics)
        self.assertEqual(
            ('/city/montpelier_vt',), 
            self.geobase.executor().execute(parses[0].semantics))
    
    def test_evaluate_grammar_with_relations(self):
        from experiment import sample_wins_and_losses
        from geoquery import GeoQueryDomain
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations)
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(256, metric_values['number of parses'])

    rules_intersection = [
        Rule('$Collection', '$Collection $Collection',
             lambda sems: ('.and', sems[0], sems[1])),
        Rule('$Collection', '$Collection $Optional $Collection',
             lambda sems: ('.and', sems[0], sems[2])),
        Rule('$Collection', '$Collection $Optional $Optional $Collection',
             lambda sems: ('.and', sems[0], sems[3])),
    ]

    def test_grammar_with_intersections(self):
        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types
            + self.rules_relations
            + self.rules_intersection)
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('states bordering california')
        self.assertEqual(('.and', 'state', ('borders', '/state/california')), parses[0].semantics)
        self.assertEqual(
            ('/state/arizona', '/state/nevada', '/state/oregon'), 
            self.geobase.executor().execute(parses[0].semantics))

    def test_evaluate_grammar_with_intersections(self):
        from experiment import sample_wins_and_losses
        from geoquery import GeoQueryDomain
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations
            + self.rules_intersection)
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(1177, metric_values['number of parses'])

    rules_superlatives = [
        Rule('$Collection', '$Superlative ?$Optionals $Collection', lambda sems: sems[0] + (sems[2],)),
        Rule('$Collection', '$Collection ?$Optionals $Superlative', lambda sems: sems[2] + (sems[0],)),

        Rule('$Superlative', 'largest', ('.argmax', 'area')),
        Rule('$Superlative', 'largest', ('.argmax', 'population')),
        Rule('$Superlative', 'biggest', ('.argmax', 'area')),
        Rule('$Superlative', 'biggest', ('.argmax', 'population')),
        Rule('$Superlative', 'smallest', ('.argmin', 'area')),
        Rule('$Superlative', 'smallest', ('.argmin', 'population')),
        Rule('$Superlative', 'longest', ('.argmax', 'length')),
        Rule('$Superlative', 'shortest', ('.argmin', 'length')),
        Rule('$Superlative', 'tallest', ('.argmax', 'height')),
        Rule('$Superlative', 'highest', ('.argmax', 'height')),

        Rule('$Superlative', '$MostLeast $RevRelation', lambda sems: (sems[0], sems[1])),
        Rule('$MostLeast', 'most', '.argmax'),
        Rule('$MostLeast', 'least', '.argmin'),
        Rule('$MostLeast', 'lowest', '.argmin'),
        Rule('$MostLeast', 'greatest', '.argmax'),
        Rule('$MostLeast', 'highest', '.argmax'),
    ]
    
    def test_grammar_with_superlatives(self):
        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives)
        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('tallest mountain')
        self.assertEqual(('.argmax', 'height', 'mountain'), parses[0].semantics)
        self.assertEqual(
            ('/mountain/mckinley',), 
            self.geobase.executor().execute(parses[0].semantics))

    def test_evaluate_grammar_with_superlatives(self):
        from experiment import sample_wins_and_losses
        from geoquery import GeoQueryDomain
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives)

        grammar = Unit2Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(2658, metric_values['number of parses'])
    
    rules_reverse_joins = [
        Rule('$Collection', '$Collection ?$Optionals $Relation',
                 lambda sems: Unit3Grammar.reverse(sems[2])(sems[0])),
    ]
    
    def test_grammar_with_reverse_joins(self):
        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives
            + self.rules_reverse_joins)
        grammar = Unit3Grammar(rules=rules, annotators=self.annotators)

        parses = grammar.parse('which states does the rio grande cross')
        self.assertEqual(('.and', 'state', ('/river/rio_grande', 'traverses')), parses[0].semantics)
        self.assertEqual(
            ('/state/colorado', '/state/new_mexico', '/state/texas'), 
            self.geobase.executor().execute(parses[0].semantics))

    def test_evaluate_grammar_with_reverse_joins(self):
        from experiment import sample_wins_and_losses
        from geoquery import GeoQueryDomain
        from metrics import DenotationOracleAccuracyMetric
        from scoring import Model

        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives
            + self.rules_reverse_joins)

        grammar = Unit3Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        metric = DenotationOracleAccuracyMetric()

        # If printing=True, prints a sampling of wins (correct semantics in 
        # first parse) and losses on the dataset.
        metric_values = sample_wins_and_losses(domain=self.domain, model=model, metric=metric, seed=1, printing=False)
        self.assertEqual(11562, metric_values['number of parses'])
        self.assertEqual(152, metric_values['denotation accuracy'])

    def test_evaluate_model(self):
        from experiment import evaluate_model
        from metrics import denotation_match_metrics
        from scoring import Model
        from geo880 import geo880_train_examples
        
        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives
            + self.rules_reverse_joins)

        grammar = Unit3Grammar(rules=rules, annotators=self.annotators)
        model = Model(grammar=grammar, executor=self.geobase.executor().execute)
        # Set print_examples=True and look for 'what state has the shortest
        # river?' and 
        evaluate_model(model=model,
                       examples=geo880_train_examples[:10],
                       metrics=denotation_match_metrics(),
                       print_examples=False)
        # SLIDES

    def test_feature_function(self):
        from experiment import evaluate_model
        from metrics import denotation_match_metrics
        from scoring import Model
        from geo880 import geo880_train_examples

        rules = (
            self.rules_optionals 
            + self.rules_collection_entity 
            + self.rules_types 
            + self.rules_relations
            + self.rules_intersection
            + self.rules_superlatives
            + self.rules_reverse_joins)

        grammar = Unit3Grammar(rules=rules, annotators=self.annotators)

        def empty_denotation_feature(parse):
            features = defaultdict(float)
            if parse.denotation == ():
                features['empty_denotation'] += 1.0
            return features

        weights = {'empty_denotation': -1.0}

        model = Model(grammar=grammar,
                      feature_fn=empty_denotation_feature,
                      weights=weights,
                      executor=self.geobase.executor().execute)
        metric_values = evaluate_model(model=model,
                       examples=geo880_train_examples,
                       metrics=denotation_match_metrics(),
                       print_examples=False)
        self.assertEqual(235, metric_values['denotation accuracy'])

if "__main__" == __name__:
    unittest.main()
        