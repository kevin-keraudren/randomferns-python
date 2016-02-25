#!/usr/bin/python

import numpy as np
import cv2

from randomferns import *
import weakLearner
import itertools

def img_test( tree, points, colors, filename, size=512, radius=3, soft=False ):
    img = np.zeros((size,size,3))
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min)/img.shape[0]
    grid = np.arange( v_min, v_max, step )

    xy = np.array( list(itertools.product( grid, grid )) )

    if soft:
        r = tree.predict_proba(xy)
        for c in range(3):
            img[((xy[:,1]-v_min)/step).astype('int32'),
                ((xy[:,0]-v_min)/step).astype('int32')] += r[:,c][...,np.newaxis]*colors[c][np.newaxis,...]
    else:
        labels = tree.predict(xy)
        img[((xy[:,1]-v_min)/step).astype('int32'),
            ((xy[:,0]-v_min)/step).astype('int32'),:] = colors[labels.astype('int32')]

    points = ((points - v_min)/step).astype('int')
    for p,r in zip(points,responses):
        cv2.circle(img, tuple(p), radius+1, (0,0,0), thickness=-1 )
        cv2.circle(img, tuple(p), radius, colors[int(r)].tolist(), thickness=-1 )

    cv2.imwrite(filename,img)

        
t = np.arange(0,10,0.1)

theta = [0,30,60]
colors = np.array( [[255,0,0],
                    [0,255,0],
                    [0,0,255]] )

points = np.zeros((len(t)*len(theta),2))
responses = np.zeros(len(t)*len(theta))
for c in range(len(theta)):
    points[c*len(t):(c+1)*len(t),0] = t**2*np.cos(t+theta[c]) # x
    points[c*len(t):(c+1)*len(t),1] = t**2*np.sin(t+theta[c]) # y
    responses[c*len(t):(c+1)*len(t)] = c

for learner in weakLearner.__all__:
    print(learner)
    fern = Fern( depth=10, test_class=getattr( weakLearner, learner)() )
    fern.fit( points, responses )
    
    img_test( fern, points, colors, 'img/fern_'+str(learner)+'.png' )
    
    randomferns = RandomFerns( depth=10, n_estimators=50,
                               test_class=getattr( weakLearner, learner)() )
    randomferns.fit( points, responses )

    img_test( randomferns, points, colors, 'img/randomferns_'+str(learner)+'.png' )
    img_test( randomferns, points, colors, 'img/randomferns_'+str(learner)+'_soft.png', soft=True )

            
