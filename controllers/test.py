from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
from utils import validate_gradient

class TestController:
    @staticmethod
    def test(tests):
        for test in tests:
            test()
