## EQL-Div
Paper: __Learning equations for extrapolation and control__
 *by S. S. Sahoo, C. H. Lampert, and G. Martius* in ICML 2018

This is the source code using Python Theano. We are also working on a tensorflow implementation.

### You need
- Python 2.7
- Theano
- graphviz

### Files to use:
* mlfg_final.py
* use createjobs.py to perform hyper-param scan (here for a formula called F0)
    for more complicated problems, use a larger number epochs, e.g. 10000 or 20000
    it will create lots of files. Do this on the cluster or somehow run all the ..sh files on your machine
    - use finish...sh to collect results
    - you also see a typical command line call. The -extrapol flags are used to specify additional data files for testing. 
    The first one is used here for the interpolation test set. 
* use Evaluation.ipynb to perform model selection and look at the result

### Dataset generation
* see ICML-Datasets.ipynb

### For visualisation file extensions to look out for:
* .trainloss for MSE + L1 loss
* .L1 for L1 loss i.e sum of magnitude of all the weights
* .MSE for mean squared error loss
* .extrapoltrainloss mean square error when the magnitude of predicted value goes beyond 10
