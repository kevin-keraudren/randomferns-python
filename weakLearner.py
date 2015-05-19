import numpy as np

__all__ = [ "AxisAligned",
            "Linear",
            "Conic",
            "Parabola" ]

class WeakLearner:
    def generate_all(self, points, count ):
        return None

    def __str__(self):
        return None

    def run(self, point, test):
        return None

class AxisAligned(WeakLearner):
    """Axis aligned"""
    def __str__(self):
        return "AxisAligned"

    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend( zip(np.zeros(count/2,dtype=int), np.random.uniform(x_min,x_max,count/2)))
        tests.extend( zip(np.ones(count/2,dtype=int), np.random.uniform(y_min,y_max,count/2)))
        return np.array(tests)


    def run(self, points, tests):
        return map( lambda test: points[:,test[0]] > test[1], tests )

class Linear(WeakLearner):
    """Linear"""
    def __str__(self):
        return "Linear"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend( zip(np.random.uniform(x_min,x_max,count),
                          np.random.uniform(y_min,y_max,count),
                          np.random.uniform(0,360,count)))
        return tests

    def run(self, points, tests):
        def _run( test ):
            theta = test[2]*np.pi/180
            return ( np.cos(theta)*(points[:,0]-test[0]) +
                     np.sin(theta)*(points[:,1]-test[1]) ) > 0
        return map( _run, tests )

class Conic(WeakLearner):
    """Non-linear: conic"""
    def __str__(self):
        return "Conic"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = max( points.max(),abs(points.min()) )
        tests = []
        tests.extend( zip( np.random.uniform(x_min,x_max,count),
                           np.random.uniform(y_min,y_max,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count)
                           )
                      )
        
        return tests

    def run( self, points, tests ):
        def _run( test ):
            x = (points[:,0]-test[0])
            y = (points[:,1]-test[1])
            A,B,C,D,E,F = test[2:]
            return ( A*x*x + B*y*y + C*x*x + D*x + E*y + F) > 0
        return map( _run, tests )

class Parabola(WeakLearner):
    """Non-linear: parabola"""
    def __str__(self):
        return "Parabola"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = abs( points.max()-points.min() )
        tests = []
        tests.extend( zip( np.random.uniform(2*x_min,2*x_max,count),
                           np.random.uniform(2*y_min,2*y_max,count),
                           np.random.uniform(-scale,scale,count),
                           np.random.random_integers(0,1,count)
                           )
                      )
        
        return tests

    def run(self, points, tests):
        def _run(test):
            x = (points[:,0]-test[0])
            y = (points[:,1]-test[1])
            p,axis = test[2:]
            if axis == 0:
                return x*x < p*y
            else:
                return y*y < p*x
        return map( _run, tests )
