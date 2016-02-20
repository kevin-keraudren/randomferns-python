Random Ferns in Python
======================

This module is a basic implementation of Random Ferns that allows users to
define their own weak learners (the tests performed at each node).

Classification example
----------------------

These examples train on three spiral (without noise) and predict the whole
plane. They try 4 different weak learners: axis aligned, linear, conic and parabolas.

``python example_ferns.py``

Using one single fern:    

Axis aligned:    
<img src="img/fern_AxisAligned.png" width="200">

Linear:       
<img src="img/fern_Linear.png" width="200">

Conic:          
<img src="img/fern_Conic.png" width="200">

Parabola:           
<img src="img/fern_Parabola.png" width="200">


Using 50 ferns, with soft or hard decision boundaries:       

Axis aligned:       
<img src="img/randomferns_AxisAligned_soft.png" width="200"> &nbsp; <img src="img/randomferns_AxisAligned.png" width="200">

Linear:       
<img src="img/randomferns_Linear_soft.png" width="200"> &nbsp; <img src="img/randomferns_Linear.png" width="200">

Conic:         
<img src="img/randomferns_Conic_soft.png" width="200"> &nbsp; <img src="img/randomferns_Conic.png" width="200">

Parabola:         
<img src="img/randomferns_Parabola_soft.png" width="200"> &nbsp; <img src="img/randomferns_Parabola.png" width="200">

Regression example
------------------

These examples train on two circles and predict the center of the bottom right quadrant
(predicting the center of the image would be too easy!).
They try 4 different weak learners: axis aligned, linear, conic and parabolas.

``python example_ferns_regression.py``

Using one single fern:    

Axis aligned:    
<img src="img/fern_AxisAligned_regression.png" width="200">

Linear:       
<img src="img/fern_Linear_regression.png" width="200">

Conic:          
<img src="img/fern_Conic_regression.png" width="200">

Parabola:           
<img src="img/fern_Parabola_regression.png" width="200">


Using 50 ferns:       

Axis aligned:       
<img src="img/randomferns_AxisAligned_regression.png" width="200">

Linear:       
<img src="img/randomferns_Linear_regression.png" width="200">

Conic:         
<img src="img/randomferns_Conic_regression.png" width="200">

Parabola:         
<img src="img/randomferns_Parabola_regression.png" width="200">

References
----------

[1] M. Ozuysal, P. Fua and V. Lepetit, "Fast Keypoint Recognition in Ten Lines of Code",        
Conference on Computer Vision and Pattern Recognition, Minneapolis, MI, June 2007.        
http://cvlab.epfl.ch/software/ferns

[2] https://github.com/rened/RandomFerns.jl (Note that R. Donner uses averaging for regression)
