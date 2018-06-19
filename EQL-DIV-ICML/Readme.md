## EQL-Div

Paper


### You need
- Python 2.7
- Theano
- graphviz

### Files to use:
* mlfg_final.py
* use createjob.py to perform hyper-param scan
    for more complicated problems, use a larger number epochs, e.g. 10000 or 20000
    it will create lots of files. Do this on the cluster or somehow run all the ..sh files on your machine
    use finish...sh to collect results
* use Evaluation.ipynb to perform model selection and look at the result

### Dataset generation
* see Kinematics-dataset.ipynb


### For visualisation file extensions to look out for:
* .trainloss for MSE + L1 loss
* .L1 for L1 loss i.e sum of magnitude of all the weights
* .MSE for mean squared error loss
* .extrapoltrainloss mean square error when the magnitude of predicted value goes beyond 10
