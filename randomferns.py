import numpy as np
from weakLearner import WeakLearner, AxisAligned

class Fern(object):
    def __init__( self, depth=10, test_class=AxisAligned() ):
        self.depth = depth
        self.test_class = test_class

    def apply_tests( self, points ):
        return reduce( lambda x,y: 2*x+y, self.test_class.run( points, self.tests ) )

    def fit( self, points, responses ):
        self.n_classes = responses.max() + 1 
        self.scores = np.ones( (2**self.depth, self.n_classes), dtype='float64' )
        self.tests = np.array( self.test_class.generate_all( points, self.depth ) )
        self.scores[self.apply_tests(points),responses.astype('int32')] += 1
        self.scores /= points.shape[0] + self.n_classes
        # maximising the product is the same as
        # maximising the sum of logarithms
        self.scores = np.log( self.scores )
        
    def _predict( self, points ):
        return self.scores[self.apply_tests(points)]

    def predict( self, points ):
        return np.argmax( self._predict(points), axis=1 )


class RandomFerns(object):
    def __init__( self,
                  depth=10,
                  n_estimators=50,
                  bootstrap=0.7,
                  test_class=AxisAligned() ):
        self.depth = depth
        self.test_class = test_class
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap

    def fit( self, points, responses ):
        self.ferns = []
        for i in range(self.n_estimators):
            subset = np.random.randint( 0, points.shape[0], int(self.bootstrap*points.shape[0]) )
            self.ferns.append( Fern( self.depth, self.test_class ) )
            self.ferns[-1].fit( points[subset], responses[subset] )

    def _predict( self, points ):
        return np.sum( map( lambda x : x._predict(points), self.ferns ), axis=0 )

    def predict( self, points ):
        return np.argmax( self._predict(points), axis=1 )

    def predict_proba( self, points ):
        proba = np.exp( self._predict(points) )
        return proba / proba.sum(axis=1)[...,np.newaxis]
        
