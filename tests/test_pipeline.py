import unittest

def get_recommended_model(income, fitness):
    """Reflected logic: Recommends model based on demographic baseline."""
    if fitness >= 4 and income >= 60000:
        return "KP781"
    elif 3 <= fitness <= 4 and 40000 <= income < 60000:
        return "KP481"
    return "KP281"

class TestAerofitPipeline(unittest.TestCase):
    def test_model_recommendation(self):
        self.assertEqual(get_recommended_model(70000, 5), "KP781")
        self.assertEqual(get_recommended_model(45000, 3), "KP481")
        self.assertEqual(get_recommended_model(30000, 2), "KP281")

if __name__ == '__main__':
    unittest.main()
