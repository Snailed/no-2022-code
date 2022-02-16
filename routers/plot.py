from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
from controllers.plot import PlotController


class PlotRouter:
    @staticmethod
    def route(argv):
        fs = [
            ellipsoid.Ellipsoid.f,
            rosenbrock.Rosenbrock.f,
            log_ellipsoid.LogEllipsoid.f,
            attractive_sector.AttractiveSector.f,
            sum_of_different_powers.SumOfDifferentPowers.f
        ]
        if len(argv) == 2 or len(argv) > 3:
            print('Usage: python main.py plot <FUNCTION>\t Plots <FUNCTION> in a new window. <FUNCTION> must be one of (ellipsoid, rosenbrock, log-ellipsoid, attractive-sector, different-powers)')
            return
        elif argv[2] == 'ellipsoid':
            f = fs[0]
            label = 'f1(x)'
        elif argv[2] == 'rosenbrock':
            f = fs[1]
            label = 'f2(x)'
        elif argv[2] == 'log-ellipsoid':
            f = fs[2]
            label = 'f3(x)'
        elif argv[2] == 'attractive-sector':
            f = fs[3]
            label = 'f4(x)'
        elif argv[2] == 'different-powers':
            f = fs[4]
            label = 'f5(x)'
        else:
            print(
                'Unknown function (ellipsoid, rosenbrock, log-ellipsoid, attractive-sector, different-powers)')
            return
        PlotController.plot(f, label)
