import unittest

from src.sabre_encapsulated import init as initSabreNew
from src.sabre import init as initSabre

# /Users/prabu/git/sabre/.venv/bin/python -m tests.sabreTest

class TestMainFunction(unittest.TestCase):
    def test_checkAllAbrOptions(self):        
        abrList = ['bola', 'bolae', 'throughput', 'dynamic' , 'dynamicdash']
        movingAverageList = ['ewma', 'sliding']

        for abrA in abrList:
            for movingAvg in movingAverageList:
                result_sabre_encapsulated = initSabreNew(moving_average = movingAvg, abrInput = abrA)
                result_sabre = initSabre(moving_average = movingAvg, abrInput = abrA)
                self.assertDictEqual(result_sabre_encapsulated, result_sabre)

if __name__ == '__main__':
    unittest.main()
