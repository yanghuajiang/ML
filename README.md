# ML
ML.py is a standalone python pipeline for our machine learning classifier
# Qiuck Start (With examples)
## combo
~~~~~~~~~~~~~~
$ python MLv2.py combo -m examples/146_115_koSC_nmp_genus_core50.matrix -t KO_SC -p examples/146_115 -c randomF -s 1
OUTPUT: examples/146_115_combo_accuracy.tsv
~~~~~~~~~~~~~~
## Select 8 features for training based on combo's results
~~~~~~~~~~~~~~
$ python MLv2.py learn -m examples/146_115_koSC_nmp_genus_core50.matrix -t KO_SC -n 8 -o -p examples/146_115 -c randomF
OUTPUT: examples/
146_115.importance
146_115.model.pkl
146_115_ROC_curve.tsv
146_115.selected_14.distribution
146_115.selected_14.importance
~~~~~~~~~~~~~~
## Predict using established model
~~~~~~~~~~~~~~
$ python MLv2.py randomF -m examples/46_wuhan_Nmp_genus.txt --model examples/146_115.model.pkl -p examples/46_wuhan_Nmp
OUTPUT: examples/46_wuhan_Nmp.predictions
~~~~~~~~~~~~~~
# USAGE
~~~~~~~~~~~~~~
usage: MLv2.py [-h] {learn,combo,randomF,SVM} ...

Machine Learning model with feature importance and prediction.

positional arguments:
  {learn,combo,randomF,SVM}
                        commands
    learn               Train the model
    combo               Evaluate accuracy based on feature selection steps
    randomF             Predict using an existing RandomForest model
    SVM                 Predict using an existing SVM model

optional arguments:
  -h, --help            show this help message and exit
~~~~~~~~~~~~~~
## First Step
Use "combo" function to obtain the best number of feartures for second selection
~~~~~~~~~~~~~~
usage: MLv2.py combo [-h] -m MATRIX -t TARGET -p PREFIX -c {randomF,SVM} [-b] -s STEPWISE

optional arguments:
  -h, --help            show this help message and exit
  -m MATRIX, --matrix MATRIX
                        Tab-delimited matrix containing samples and features.
  -t TARGET, --target TARGET
                        Target group for classification.
  -p PREFIX, --prefix PREFIX
                        Prefix for output files.
  -c {randomF,SVM}, --classifier {randomF,SVM}
                        Type of classifier to use.
  -b, --boole           Flag to transfer input matrix into 1/0 format.
  -s STEPWISE, --stepwise STEPWISE
                        Step size or ratio for feature selection.
~~~~~~~~~~~~~~
## Second Step
Use "learn" function to train the model
~~~~~~~~~~~~~~
usage: MLv2.py learn [-h] -m MATRIX -t TARGET [-n NUM_TOPS] [--min_samples_split MIN_SAMPLES_SPLIT] [-o] [--RFE] -p PREFIX -c {randomF,SVM}
                   [-e NUM_ESTIMATOR] [--min_impurity_decrease MIN_IMPURITY_DECREASE]

optional arguments:
  -h, --help            show this help message and exit
  -m MATRIX, --matrix MATRIX
                        Tab-delimited matrix containing samples and features.
  -t TARGET, --target TARGET
                        Target group for classification.
  -n NUM_TOPS, --num_tops NUM_TOPS
                        Number of top features for second run.
  --min_samples_split MIN_SAMPLES_SPLIT
                        Minimum number of samples required to split an internal node.
  -o, --output_model    Flag to save the trained model.
  --RFE                 Flag to use RFE to downsample features.
  -p PREFIX, --prefix PREFIX
                        Prefix for output files.
  -c {randomF,SVM}, --classifier {randomF,SVM}
                        Type of classifier to use.
  -e NUM_ESTIMATOR, --num_estimator NUM_ESTIMATOR
                        Number of estimators.
  --min_impurity_decrease MIN_IMPURITY_DECREASE
~~~~~~~~~~~~~~
## Third Step
Use the model to predict
~~~~~~~~~~~~~~
usage: MLv2.py randomF [-h] -m MATRIX --model MODEL -p PREFIX

optional arguments:
  -h, --help            show this help message and exit
  -m MATRIX, --matrix MATRIX
                        Tab-delimited matrix containing samples for prediction.
  --model MODEL         Path to the trained model.
  -p PREFIX, --prefix PREFIX
                        Prefix for prediction outputs.
~~~~~~~~~~~~~~
