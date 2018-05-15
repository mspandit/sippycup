import unittest


class Executor(object):
    """
	Performs the arithmetic calculations described by semantic representations
	to return a denotation.
	"""
    def __init__(self):
        super(Executor, self).__init__()
    
    ops = {
        '~': lambda x: -x,
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
    }

    @staticmethod
    def execute(sem):
        if isinstance(sem, tuple):
            op = Executor.ops[sem[0]]
            args = []
            for arg in sem[1:]:
                args.append(Executor.execute(arg))
            return op(*args)
        else:
            return sem


class TestMethods(unittest.TestCase):
    def test_one_plus_one(self):
        self.assertEqual(2, Executor.execute(('+', 1, 1)))

    def test_minus_three_minus_two(self):
        self.assertEqual(-5, Executor.execute(('-', ('~', 3), 2)))

    def test_three_plus_three_minus_two(self):
        self.assertEqual(4, Executor.execute(('-', ('+', 3, 3), 2)))

    def test_two_times_two_plus_three(self):
        self.assertEqual(7, Executor.execute(('+', ('*', 2, 2), 3)))

    # SLIDES

if __name__ == "__main__":
    unittest.main()