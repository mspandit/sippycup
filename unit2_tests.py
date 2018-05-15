import unittest
from collections import defaultdict
from grammar import Grammar, Rule, Parse


from types import FunctionType

class Unit2Grammar(Grammar):
    def __init__(self, rules=[], annotators=[], start_symbol='$ROOT'):
        self.unary_rules = defaultdict(list)
        self.categories = set()
        super(Unit2Grammar, self).__init__(rules)
        self.annotators = annotators
        self.start_symbol = start_symbol

    @staticmethod
    def is_unary(rule):
        """
        Returns true iff rule is a unary compasitional rule, i.e. it contains 
        only a single category (non-terminal) on the RHS.
        """
        return 1 == len(rule.rhs) and Unit2Grammar.is_cat(rule.rhs[0])

    def add_rule(self, rule):
        """
        Knows what to do when it encounters unary or n-ary compositional rules, 
        or rules containing optionals.
        """
        if self.contains_optionals(rule):
            self.add_rule_containing_optional(rule)
        elif self.is_lexical(rule):
            self.lexical_rules[rule.rhs].append(rule)
        elif self.is_unary(rule):
            self.unary_rules[rule.rhs].append(rule)
        elif self.is_binary(rule):
            self.binary_rules[rule.rhs].append(rule)
        elif all([self.is_cat(rhsi) for rhsi in rule.rhs]):
            self.add_n_ary_rule(rule)
        else:
            raise Exception('RHS mixes terminals and non-terminals: %s' % rule)

    def parse(self, input):
        """Returns a list of all parses for input using the grammar."""
        tokens = input.split()
        chart = defaultdict(list) # map from span (i, j) to list of parses
        for end in range(1, len(tokens) + 1):
            for start in range(end - 1, -1, -1):
                self.apply_annotators(chart, tokens, start, end)
                self.apply_lexical_rules(chart, tokens, start, end)
                self.apply_binary_rules(chart, start, end)
                self.apply_unary_rules(chart, start, end)
        parses0 = chart[(0, len(tokens))]
        if hasattr(self, 'start_symbol') and self.start_symbol:
            parses = []
            for parse in parses0:
                if parse.rule.lhs == self.start_symbol:
                    parses.append(parse)
            return parses
        else:
            return parses0

    def apply_annotators(self, chart, tokens, start, end):
        """Add parses to chart cell (start, end) by applying annotators."""
        if hasattr(self, 'annotators'):
            for annotator in self.annotators:
                for category, semantics in annotator.annotate(tokens[start:end]):
                    if not self.check_capacity(chart, start, end):
                        return
                    rule = Rule(category, tuple(tokens[start:end]), semantics)
                    chart[(start, end)].append(Parse(rule, tokens[start:end]))

    def apply_unary_rules(self, chart, start, end):
        """
        Add parses to chart cell (start, end) by applying unary rules.
        Note that the last line of this method can add new parses to 
        chart[(start, end)], the list over which we are iterating. Because of 
        this, we essentially get unary closure "for free." (However, if the
        grammar contains unary cycles, we'll get stuck in a loop, which is one
        reason for check_capacity().)
        """
        if hasattr(self, 'unary_rules'):
            for parse in chart[(start, end)]:
                for rule in self.unary_rules[(parse.rule.lhs,)]:
                    if not self.check_capacity(chart, start, end):
                        return
                    chart[(start, end)].append(Parse(rule, [parse]))
    
    MAX_CELL_CAPACITY = 10000
    
    @staticmethod
    def check_capacity(chart, start, end):
        if len(chart[(start, end)]) > Unit2Grammar.MAX_CELL_CAPACITY:
            print('Cell (%d, %d) has reached capacity %d' % (start, end, Unit2Grammar.MAX_CELL_CAPACITY))
            return False
        return True

    def add_n_ary_rule(self, rule):
		"""
		Handles adding a rule with three or more non-terminals on the RHS.
		We introduce a new category which covers all elements on the RHS except
		the first, and then generate two variants of the rule: one which
		consumes those elements to produce the new category, and another which
		combines the new category which the first element to produce the
		original LHS category.  We add these variants in place of the
		original rule.  (If the new rules still contain more than two elements
		on the RHS, we'll wind up recursing.)

		For example, if the original rule is:

		    Rule('$Z', '$A $B $C $D')

		then we create a new category '$Z_$A' (roughly, "$Z missing $A to the left"),
		and add these rules instead:

		    Rule('$Z_$A', '$B $C $D')
		    Rule('$Z', '$A $Z_$A')
		"""
		def add_category(base_name):
			assert self.is_cat(base_name)
			name = base_name
			while name in self.categories:
				name = name + '_'
			self.categories.add(name)
			return name

		def apply_semantics(rule, sems):
		    # Note that this function would not be needed if we required that semantics
		    # always be functions, never bare values.  That is, if instead of
		    # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
		    # But that would be cumbersome.
		    if isinstance(rule.sem, FunctionType):
		        return rule.sem(sems)
		    else:
		        return rule.sem

		category = add_category('%s_%s' % (rule.lhs, rule.rhs[0]))
		self.add_rule(Rule(category, rule.rhs[1:], lambda sems: sems))
		self.add_rule(Rule(rule.lhs, (rule.rhs[0], category),
		                       lambda sems: apply_semantics(rule, [sems[0]] + sems[1])))
    
    @staticmethod
    def is_optional(label):
        """
        Returns true iff the given RHS item is optional, i.e., is marked with an
        initial '?'.
        """
        return label.startswith('?') and len(label) > 1

    @staticmethod
    def contains_optionals(rule):
        """Returns true iff the given Rule contains any optional items on the RHS."""
        return any([Unit2Grammar.is_optional(rhsi) for rhsi in rule.rhs])

    def add_rule_containing_optional(self, rule):
        """
        Handles adding a rule which contains an optional element on the RHS.
        We find the leftmost optional element on the RHS, and then generate
        two variants of the rule: one in which that element is required, and
        one in which it is removed.  We add these variants in place of the
        original rule.  (If there are more optional elements further to the
        right, we'll wind up recursing.)

        For example, if the original rule is:

            Rule('$Z', '$A ?$B ?$C $D')

        then we add these rules instead:

            Rule('$Z', '$A $B ?$C $D')
            Rule('$Z', '$A ?$C $D')
        """
        # Find index of the first optional element on the RHS.
        first = next((idx for idx, elt in enumerate(rule.rhs) if Unit2Grammar.is_optional(elt)), -1)
        assert first >= 0
        assert len(rule.rhs) > 1, 'Entire RHS is optional: %s' % rule
        prefix = rule.rhs[:first]
        suffix = rule.rhs[(first + 1):]
        # First variant: the first optional element gets deoptionalized.
        deoptionalized = (rule.rhs[first][1:],)
        self.add_rule(Rule(rule.lhs, prefix + deoptionalized + suffix, rule.sem))
        # Second variant: the first optional element gets removed.
        # If the semantics is a value, just keep it as is.
        sem = rule.sem
        # But if it's a function, we need to supply a dummy argument for the removed element.
        if isinstance(rule.sem, FunctionType):
            sem = lambda sems: rule.sem(sems[:first] + [None] + sems[first:])
        self.add_rule(Rule(rule.lhs, prefix + suffix, sem))

    @staticmethod
    def sems_0(sems):
        return sems[0]

    @staticmethod
    def sems_1(sems):
        return sems[1]

    @staticmethod
    def merge_dicts(d1, d2):
        if not d2:
            return d1
        result = d1.copy()
        result.update(d2)
        return result


class Annotator(object):
    """Base class for annotators"""
    def annotate(self, tokens):
        """
        Returns a list of pairs, each a category and a semantic representation.
        """
        return []


class NumberAnnotator(Annotator):
    """
    Annotate any string representing a number with category $Number and 
    semantics equal to its numeric value.
    """
    def annotate(self, tokens):
        if 1 == len(tokens):
            try:
                value = float(tokens[0])
                if int(value) == value:
                    value = int(value)
                return [('$Number', value)]
            except ValueError:
                pass
        return []


class TokenAnnotator(Annotator):
    """
    Annotate any single token with category $Token and semantics equal to the
    token itself
    """
    def annotate(self, tokens):
        if 1 == len(tokens):
            return[('$Token', tokens[0])]
        else:
            return []


class TestMethodsUnit2(unittest.TestCase):
    def test_number_annotator(self):
        self.assertEqual(
			[('$Number', 16)], 
			NumberAnnotator().annotate(['16']))

    def test_token_annotator(self):
        self.assertEqual(
			[('$Token', 'foo')], 
			TokenAnnotator().annotate(['foo']))
    
    rules_travel = [
        Rule('$ROOT', '$TravelQuery', Unit2Grammar.sems_0),
        # Add a key-value pair to the semantics of the child.
        Rule(
			'$TravelQuery', 
			'$TravelQueryElements',
            lambda sems: Unit2Grammar.merge_dicts(
				{'domain': 'travel'}, 
				sems[0])),
        # Merge the semantics of the children
        Rule(
			'$TravelQueryElements', 
			'$TravelQueryElement ?$TravelQueryElements',
            lambda sems: Unit2Grammar.merge_dicts(sems[0], sems[1])),
        # Propagate the semantics of the child unchanged.
        Rule('$TravelQueryElement', '$TravelLocation', Unit2Grammar.sems_0),
        Rule('$TravelQueryElement', '$TravelArgument', Unit2Grammar.sems_0),
    ]
    
    rules_travel_locations = [
        Rule('$TravelLocation', '$ToLocation', Unit2Grammar.sems_0),
        Rule('$TravelLocation', '$FromLocation', Unit2Grammar.sems_0),
        Rule(
			'$ToLocation', 
			'$To $Location', 
			lambda sems: {'destination': sems[1]}),
        Rule(
			'$FromLocation', 
			'$From $Location', 
			lambda sems: {'origin': sems[1]}),
        Rule('$To', 'to'),
        Rule('$From', 'from'),
    ]
    
    def test_geonames_annotator(self):
        """
        Maps a phrase like 'boston' to a semantic representation like 
        {id: 4930956, name: 'Boston, MA, US'} using GeoNames API.
        """
        from geonames import GeoNamesAnnotator
        geonames_annotator = GeoNamesAnnotator()
        self.assertEqual(
            [('$Location', {'id': 4930956, 'name': 'Boston, MA, US'})], 
            geonames_annotator.annotate(['boston']))

    def test_locations_grammar(self):
        from geonames import GeoNamesAnnotator

        rules = self.rules_travel + self.rules_travel_locations
        grammar = Unit2Grammar(
			rules=rules, 
			annotators=[GeoNamesAnnotator(live_requests=False)])
        parses = grammar.parse('from boston to austin')
        self.assertEqual(
            {
                'domain': 'travel', 
                'origin': {
                    'name': 'Boston, MA, US', 
                    'id': 4930956 }, 
                'destination': {
                    'name': 'Austin, TX, US', 
                    'id': 4671654 } },
            parses[0].semantics)

    # Handle the most common ways of referring to each travel mode.
    rules_travel_modes = [
        Rule('$TravelArgument', '$TravelMode', Unit2Grammar.sems_0),

        Rule('$TravelMode', '$AirMode', {'mode': 'air'}),
        Rule('$TravelMode', '$BikeMode', {'mode': 'bike'}),
        Rule('$TravelMode', '$BoatMode', {'mode': 'boat'}),
        Rule('$TravelMode', '$BusMode', {'mode': 'bus'}),
        Rule('$TravelMode', '$CarMode', {'mode': 'car'}),
        Rule('$TravelMode', '$TaxiMode', {'mode': 'taxi'}),
        Rule('$TravelMode', '$TrainMode', {'mode': 'train'}),
        Rule('$TravelMode', '$TransitMode', {'mode': 'transit'}),

        Rule('$AirMode', 'air fare'),
        Rule('$AirMode', 'air fares'),
        Rule('$AirMode', 'airbus'),
        Rule('$AirMode', 'airfare'),
        Rule('$AirMode', 'airfares'),
        Rule('$AirMode', 'airline'),
        Rule('$AirMode', 'airlines'),
        Rule('$AirMode', '?by air'),
        Rule('$AirMode', 'flight'),
        Rule('$AirMode', 'flights'),
        Rule('$AirMode', 'fly'),

        Rule('$BikeMode', '?by bike'),
        Rule('$BikeMode', 'bike riding'),

        Rule('$BoatMode', '?by boat'),
        Rule('$BoatMode', 'cruise'),
        Rule('$BoatMode', 'cruises'),
        Rule('$BoatMode', 'norwegian cruise lines'),

        Rule('$BusMode', '?by bus'),
        Rule('$BusMode', 'bus tours'),
        Rule('$BusMode', 'buses'),
        Rule('$BusMode', 'shutle'),
        Rule('$BusMode', 'shuttle'),

        Rule('$CarMode', '?by car'),
        Rule('$CarMode', 'drive'),
        Rule('$CarMode', 'driving'),
        Rule('$CarMode', 'gas'),

        Rule('$TaxiMode', 'cab'),
        Rule('$TaxiMode', 'car service'),
        Rule('$TaxiMode', 'taxi'),

        Rule('$TrainMode', '?by train'),
        Rule('$TrainMode', 'trains'),
        Rule('$TrainMode', 'amtrak'),

        Rule('$TransitMode', '?by public transportation'),
        Rule('$TransitMode', '?by ?public transit'),
    ]

    def test_modes_grammar(self):
        from geonames import GeoNamesAnnotator

        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes
        grammar = Unit2Grammar(rules=rules, annotators=[GeoNamesAnnotator(live_requests=False)])
        parses = grammar.parse('from boston to austin by train')
        self.assertEqual(
            {
                'domain': 'travel',
                'mode': 'train',
                'origin': {
                    'name': 'Boston, MA, US', 
                    'id': 4930956 }, 
                'destination': {
                    'name': 'Austin, TX, US', 
                    'id': 4671654 } },
            parses[0].semantics)

    def test_training_data1(self):
        from experiment import sample_wins_and_losses
        from metrics import SemanticsOracleAccuracyMetric
        from scoring import Model
        from travel import TravelDomain
        from geonames import GeoNamesAnnotator

        domain = TravelDomain()
        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes
        grammar = Unit2Grammar(rules=rules, annotators=[GeoNamesAnnotator(live_requests=False)])
        model = Model(grammar=grammar)
        metric = SemanticsOracleAccuracyMetric()

        metric_values = sample_wins_and_losses(domain=domain, model=model, metric=metric, seed=31, printing=False)

    rules_travel_triggers = [
        Rule('$TravelArgument', '$TravelTrigger', {}),

        Rule('$TravelTrigger', 'tickets'),
        Rule('$TravelTrigger', 'transportation'),
        Rule('$TravelTrigger', 'travel'),
        Rule('$TravelTrigger', 'travel packages'),
        Rule('$TravelTrigger', 'trip'),
    ]
    
    def test_training_data2(self):
        from experiment import sample_wins_and_losses
        from metrics import SemanticsOracleAccuracyMetric
        from scoring import Model
        from travel import TravelDomain
        from geonames import GeoNamesAnnotator

        domain = TravelDomain()
        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes + self.rules_travel_triggers
        grammar = Unit2Grammar(rules=rules, annotators=[GeoNamesAnnotator(live_requests=False)])
        model = Model(grammar=grammar)
        metric = SemanticsOracleAccuracyMetric()

        metric_values = sample_wins_and_losses(domain=domain, model=model, metric=metric, seed=31, printing=False)

    rules_request_types = [
        Rule('$TravelArgument', '$RequestType', Unit2Grammar.sems_0),

        Rule('$RequestType', '$DirectionsRequest', {'type': 'directions'}),
        Rule('$RequestType', '$DistanceRequest', {'type': 'distance'}),
        Rule('$RequestType', '$ScheduleRequest', {'type': 'schedule'}),
        Rule('$RequestType', '$CostRequest', {'type': 'cost'}),

        Rule('$DirectionsRequest', 'directions'),
        Rule('$DirectionsRequest', 'how do i get'),
        Rule('$DistanceRequest', 'distance'),
        Rule('$ScheduleRequest', 'schedule'),
        Rule('$CostRequest', 'cost'),
    ]
    
    def test_training_data3(self):
        from experiment import sample_wins_and_losses
        from metrics import SemanticsOracleAccuracyMetric
        from scoring import Model
        from travel import TravelDomain
        from geonames import GeoNamesAnnotator

        domain = TravelDomain()
        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes + self.rules_travel_triggers + self.rules_request_types
        grammar = Unit2Grammar(rules=rules, annotators=[GeoNamesAnnotator(live_requests=False)])
        model = Model(grammar=grammar)
        metric = SemanticsOracleAccuracyMetric()

        metric_values = sample_wins_and_losses(domain=domain, model=model, metric=metric, seed=31, printing=False)

    rules_optionals = [
        Rule('$TravelQueryElement', '$TravelQueryElement $Optionals', Unit2Grammar.sems_0),
        Rule('$TravelQueryElement', '$Optionals $TravelQueryElement', Unit2Grammar.sems_1),

        Rule('$Optionals', '$Optional ?$Optionals'),

        Rule('$Optional', '$Show'),
        Rule('$Optional', '$Modifier'),
        Rule('$Optional', '$Carrier'),
        Rule('$Optional', '$Stopword'),
        Rule('$Optional', '$Determiner'),

        Rule('$Show', 'book'),
        Rule('$Show', 'give ?me'),
        Rule('$Show', 'show ?me'),

        Rule('$Modifier', 'cheap'),
        Rule('$Modifier', 'cheapest'),
        Rule('$Modifier', 'discount'),
        Rule('$Modifier', 'honeymoon'),
        Rule('$Modifier', 'one way'),
        Rule('$Modifier', 'direct'),
        Rule('$Modifier', 'scenic'),
        Rule('$Modifier', 'transatlantic'),
        Rule('$Modifier', 'one day'),
        Rule('$Modifier', 'last minute'),

        Rule('$Carrier', 'delta'),
        Rule('$Carrier', 'jet blue'),
        Rule('$Carrier', 'spirit airlines'),
        Rule('$Carrier', 'amtrak'),

        Rule('$Stopword', 'all'),
        Rule('$Stopword', 'of'),
        Rule('$Stopword', 'what'),
        Rule('$Stopword', 'will'),
        Rule('$Stopword', 'it'),
        Rule('$Stopword', 'to'),

        Rule('$Determiner', 'a'),
        Rule('$Determiner', 'an'),
        Rule('$Determiner', 'the'),
    ]
    
    def test_training_data4(self):
        from experiment import sample_wins_and_losses
        from metrics import SemanticsOracleAccuracyMetric
        from scoring import Model
        from travel import TravelDomain
        from geonames import GeoNamesAnnotator

        domain = TravelDomain()
        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes + self.rules_travel_triggers + self.rules_request_types + self.rules_optionals
        grammar = Unit2Grammar(rules=rules, annotators=[GeoNamesAnnotator(live_requests=False)])
        model = Model(grammar=grammar)
        metric = SemanticsOracleAccuracyMetric()

        metric_values = sample_wins_and_losses(domain=domain, model=model, metric=metric, seed=31, printing=False)

    rules_not_travel = [
        Rule('$ROOT', '$NotTravelQuery', Unit2Grammar.sems_0),
        Rule('$NotTravelQuery', '$Text', {'domain': 'other'}),
        Rule('$Text', '$Token ?$Text'),
    ]

    def test_training_data5(self):
        from experiment import sample_wins_and_losses
        from metrics import SemanticsOracleAccuracyMetric
        from scoring import Model
        from travel import TravelDomain
        from geonames import GeoNamesAnnotator

        domain = TravelDomain()
        rules = self.rules_travel + self.rules_travel_locations + self.rules_travel_modes + self.rules_travel_triggers + self.rules_request_types + self.rules_optionals + self.rules_not_travel
        grammar = Unit2Grammar(rules=rules, annotators=[
            GeoNamesAnnotator(live_requests=False),
            TokenAnnotator()
        ])
        model = Model(grammar=grammar)
        metric = SemanticsOracleAccuracyMetric()

        metric_values = sample_wins_and_losses(domain=domain, model=model, metric=metric, seed=31, printing=False)
    
    
if __name__ == "__main__":
    unittest.main()