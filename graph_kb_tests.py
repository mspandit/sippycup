from graph_kb import GraphKB
from geobase import GeobaseReader
import unittest


class TestMethods(unittest.TestCase):
    def test0(self):
        geobase = GraphKB(GeobaseReader().tuples)
        executor = geobase.executor()
        self.assertEqual(
            ('/city/austin_tx',),
            # capital of texas
            executor.execute(('/state/texas', 'capital')) 
        )
        self.assertEqual(
            ('/river/colorado', '/river/green', '/river/san_juan'),
            # rivers that traverse utah
            executor.execute(('.and', 'river', ('traverses', '/state/utah')))
        )
        self.assertEqual(
            ('/mountain/mckinley',),
            # tallest mountain
            executor.execute(('.argmax', 'height', 'mountain'))
        )


if "__main__" == __name__:
    unittest.main()