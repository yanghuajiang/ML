# ML
ML.py is a standalone python pipeline for our machine learning classifier
# Qiuck Start (With examples)
~~~~~~~~~~~~~~
$ python ML.py combo -m examples/115_ko_sc_Nmp_36_genus.abundance -t KO_SC -p examples/115_ko_sc_Nmp_36_genus -c randomF -s 1
OUTPUT: examples/
115_ko_sc_Nmp_36_genus_combo_combo_accuracy.tsv

$ python ML.py learn -m examples/115_ko_sc_Nmp_36_genus.abundance -t KO_SC -n 14 -o -p examples/ko_sc_14_model -c randomF
OUTPUT: examples/
ko_sc_14_model.importance
ko_sc_14_model.model.pkl
ko_sc_14_model_ROC_curve.tsv
ko_sc_14_model.selected_14.distribution
ko_sc_14_model.selected_14.importance

$ python ML.py randomF -m examples/46_wuhan_14_genus.ab --model example/ko_sc_14_model.model.pkl -p example/46_wuhan_14_genus
OUTPUT: examples/
46_wuhan_14_genus.predictions
~~~~~~~~~~~~~~
# USAGE
~~~~~~~~~~~~~~
usage: ML.py {learn,combo,randomF,SVM} ...

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
usage: ML.py combo [-h] -m MATRIX -t TARGET -p PREFIX -c {randomF,SVM} -s STEPWISE

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
  -s STEPWISE, --stepwise STEPWISE
                        Step size or ratio for feature selection.
~~~~~~~~~~~~~~
## Second Step
Use "learn" function to train the model
~~~~~~~~~~~~~~
usage: ML.py learn [-h] -m MATRIX -t TARGET [-n NUM_TOPS] [--min_samples_split MIN_SAMPLES_SPLIT] [-o] [--RFE] -p PREFIX -c {randomF,SVM}
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
usage: ML.py randomF [-h] -m MATRIX --model MODEL -p PREFIX

optional arguments:
  -h, --help            show this help message and exit
  -m MATRIX, --matrix MATRIX
                        Tab-delimited matrix containing samples for prediction.
  --model MODEL         Path to the trained model.
  -p PREFIX, --prefix PREFIX
                        Prefix for prediction outputs.
~~~~~~~~~~~~~~
