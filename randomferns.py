import numpy as np
from functools import reduce
from weakLearner import WeakLearner, AxisAligned

class Fern(object):
    def __init__( self, depth=10, test_class=AxisAligned(), regression=False ):
        self.depth = depth
        self.test_class = test_class
        self.regression = regression

    def apply_tests( self, points ):
        return reduce( lambda x,y: 2*x+y, self.test_class.run( points, self.tests ) )

    def fit( self, points, responses ):
        self.tests = np.array( self.test_class.generate_all( points, self.depth ) )
        if self.regression:
            self.target_dim = responses.shape[1]
            self.data = np.zeros( (2**self.depth, self.target_dim), dtype='float64' )
            bins = self.apply_tests(points)
            bincount = np.bincount(bins, minlength=self.data.shape[0])
            for dim in range(self.target_dim):
                self.data[:,dim] += np.bincount(bins, weights=responses[:,dim], minlength=self.data.shape[0])
            self.data[bincount>0] /= bincount[bincount>0][...,np.newaxis]
        else:
            self.n_classes = responses.max() + 1
            self.data = np.ones( (2**self.depth, self.n_classes), dtype='float64' )
            self.data[self.apply_tests(points), responses.astype('int32')] += 1
            self.data /= points.shape[0] + self.n_classes
            # maximising the product is the same as
            # maximising the sum of logarithms
            self.data = np.log( self.data )
        
    def _predict( self, points ):
        return self.data[self.apply_tests(points)]

    def predict( self, points ):
        if self.regression:
            return self._predict(points)
        else:
            return np.argmax( self._predict(points), axis=1 )


class RandomFerns(object):
    def __init__( self,
                  depth=10,
                  n_estimators=50,
                  bootstrap=0.7,
                  test_class=AxisAligned(),
                  regression=False ):
        self.depth = depth
        self.test_class = test_class
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.regression = regression

    def fit( self, points, responses ):
        self.ferns = []
        for i in range(self.n_estimators):
            subset = np.random.randint( 0, points.shape[0], int(self.bootstrap*points.shape[0]) )
            self.ferns.append( Fern( self.depth, self.test_class, regression=self.regression ) )
            self.ferns[-1].fit( points[subset], responses[subset] )

    def _predict( self, points ):
        return np.sum( list(map( lambda x : x._predict(points), self.ferns )), axis=0 )

    def predict( self, points ):
        if self.regression:
            return np.array(list(map( lambda x : x._predict(points), self.ferns )))
        else:
            return np.argmax( self._predict(points), axis=1 )

    def predict_proba( self, points ):
        if self.regression:
            raise NotImplemented("predict_proba is not implemented for regression")
        proba = np.exp( self._predict(points) )
        return proba / proba.sum(axis=1)[...,np.newaxis]
        
