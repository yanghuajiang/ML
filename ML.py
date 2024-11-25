import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score
import joblib
import argparse
import datetime, logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def combo_evaluate(args):
    logging.info("Starting combo evaluation.")
    start_time = datetime.datetime.now()
    
    try:
        df = pd.read_csv(args.matrix, sep='\t', index_col=0)
    except Exception as e:
        logging.error(f"Error reading file {args.matrix}: {e}")
        return
    
    X = df.drop(args.target, axis=1)
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if args.classifier == 'randomF':
        model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    elif args.classifier == 'SVM':
        model = SVC(kernel='linear', probability=True, random_state=42)
    
    model.fit(X_train, y_train.to_numpy())
    importances = model.feature_importances_ if args.classifier == 'randomF' else abs(model.coef_[0])
    
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    total_features = len(feature_importance_df)
    steps = list(range(1, total_features + 1, int(args.stepwise) if args.stepwise >= 1 else max(1, int(total_features * args.stepwise))))
    
    accuracy_results = []
    for i, num_features in enumerate(steps):
        top_features = feature_importance_df.head(num_features)['Feature'].values
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        
        model.fit(X_train_selected, y_train.to_numpy())
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        result = {'num_features': num_features, 'accuracy': accuracy, 'f1_score': f1}
        if len(y.unique()) == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_selected)[:, 1]
            result['auc'] = roc_auc_score(y_test, y_proba)
        
        accuracy_results.append(result)
        logging.info(f"Features: {num_features}, Accuracy: {round(accuracy,4)}, F1 Score: {round(f1,4)}")

    accuracy_df = pd.DataFrame(accuracy_results)
    accuracy_df.to_csv(f"{args.prefix}_combo_accuracy.tsv", sep='\t', index=False)
    logging.info(f"Combo evaluation completed in {datetime.datetime.now() - start_time}")

def learn(args):
    try:
        df = pd.read_csv(args.matrix, sep='\t', index_col=0)
    except Exception as e:
        logging.error(f"Error reading file {args.matrix}: {e}")
        return

    X = df.drop(args.target, axis=1)
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if args.classifier == 'randomF':
        model = RandomForestClassifier(n_estimators=args.num_estimator, random_state=42, min_samples_split=args.min_samples_split or 2, min_impurity_decrease=args.min_impurity_decrease)
    elif args.classifier == 'SVM':
        model = SVC(kernel='linear', probability=True, random_state=42)

    if args.RFE:
        logging.info(f"Use RFE to downsample features. Might take a long time")
        selector = RFE(estimator=model, n_features_to_select=args.num_tops, step=1)
        selector = selector.fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        print("Selected Features: ", selector.support_)
        print("Feature Ranking: ", selector.ranking_)

    model.fit(X_train, y_train.to_numpy())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logging.info(f"Accuracy: {round(accuracy,4)}")
    logging.info(f"F1 Score: {round(f1,4)}")
    
    if len(y.unique()) == 2:  # Check if it's a binary classification
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logging.info(f"AUC: {auc}")
    else:
        logging.info("AUC is not calculated for multi-class classification.")

    if args.classifier == 'randomF':
        importances = model.feature_importances_
    elif args.classifier == 'SVM':
        importances = abs(model.coef_[0])

    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    logging.info(importance_df)
    
    importance_df.to_csv(f"{args.prefix}.importance", sep='\t', index=False)

    top_features = importance_df.head(args.num_tops)['Feature'].values if len(importance_df) > args.num_tops else importance_df['Feature'].values
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    model.fit(X_train_selected, y_train.to_numpy())
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logging.info(f"Accuracy after feature selection: {round(accuracy,4)}")
    logging.info(f"F1 Score after feature selection: {round(f1,4)}")
    
    if len(y.unique()) == 2:
        y_proba = model.predict_proba(X_test_selected)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logging.info(f"AUC after feature selection: {round(auc,4)}")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=y_test.unique()[1])
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
        roc_df.to_csv(f"{args.prefix}_ROC_curve.tsv", sep='\t', index=False)
        logging.info(f"Curated ROC curve data saved to {args.prefix}_ROC_curve.tsv")

    if args.classifier == 'randomF':
        importances = model.feature_importances_
    elif args.classifier == 'SVM':
        importances = abs(model.coef_[0])

    features = X_train_selected.columns
    importance_df_selected = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    logging.info(importance_df_selected)
    
    importance_df_selected.to_csv(f"{args.prefix}.selected_{args.num_tops}.importance", sep='\t', index=False)

    grouped_stats_selected = X_train_selected.join(y_train).groupby(args.target).agg(['mean', 'std', 'min', 'max'])
    logging.info(grouped_stats_selected.T)
    grouped_stats_selected.T.to_csv(f"{args.prefix}.selected_{args.num_tops}.distribution", sep='\t')
                    

    if args.output_model:
        joblib.dump(model, f"{args.prefix}.model.pkl")
        logging.info(f'Model saved in {args.prefix}.model.pkl')

def predict(args):
    logging.info(f'Loading {args.model}.')
    try:
        model = joblib.load(args.model)
    except FileNotFoundError:
        logging.error(f"The model file {args.model} was not found.")
        return
    except Exception as e:
        logging.error(f"Error loading the model {args.model}: {e}")
        return
    
    logging.info(f'Loading {args.matrix}.')
    try:
        new_data = pd.read_csv(args.matrix, sep='\t', index_col=0)
    except FileNotFoundError:
        logging.error(f"The data file {args.matrix} was not found.")
        return
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file {args.matrix}. Please check the file format.")
        return

    model_features = model.feature_names_in_
    missing_features = [feat for feat in model_features if feat not in new_data.columns]

    if missing_features:
        raise ValueError(f"The following features are missing in the input data: {', '.join(missing_features)}")
    else:
        new_data_ordered = new_data[model_features]

        with open(args.prefix + '.predictions', 'w') as pout:
            logging.info(f'Prediction saved in {args.prefix}.predictions')
            if hasattr(model, "predict_proba"):
                predictions = model.predict(new_data_ordered)
                probabilities = model.predict_proba(new_data_ordered)
                class_labels = model.classes_
                print("Sample\tPredicted_class\tProbability", file=pout)
                
                for i, sample in enumerate(new_data.index):
                    pred_class = predictions[i]
                    class_index = list(class_labels).index(pred_class)
                    prob = probabilities[i][class_index]
                    print(f"{sample}\t{pred_class}\t{prob:.4f}", file=pout)
            else:
                predictions = model.predict(new_data_ordered)
                print("Sample\tPredicted_class", file=pout)
                for i, sample in enumerate(new_data.index):
                    print(f"{sample}\t{predictions[i]}", file=pout)

def main():
    parser = argparse.ArgumentParser(description='Machine Learning model with feature importance and prediction.')
    subparsers = parser.add_subparsers(help='commands', dest='command')
    subparsers.required = True

    parser_learn = subparsers.add_parser('learn', help='Train the model')
    parser_learn.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples and features.')
    parser_learn.add_argument('-t', '--target', required=True, type=str, help='Target group for classification.')
    parser_learn.add_argument('-n', '--num_tops', type=int, default=20, help='Number of top features for second run.')
    parser_learn.add_argument('--min_samples_split', type=int, help='Minimum number of samples required to split an internal node.')
    parser_learn.add_argument('-o', '--output_model', action='store_true', help='Flag to save the trained model.')
    parser_learn.add_argument('--RFE', action='store_true', help='Flag to use RFE to downsample features.')
    parser_learn.add_argument('-p', '--prefix', required=True, help='Prefix for output files.')
    parser_learn.add_argument('-c', '--classifier', required=True, choices=['randomF', 'SVM'], help='Type of classifier to use.')
    parser_learn.add_argument('-e', '--num_estimator', type=int, default=1000, help='Number of estimators.')
    parser_learn.add_argument('--min_impurity_decrease', type=float, default=0.0, help='0-1, you can set a high value to keep all important splits; Or set a low value to keep all splits.')

    parser_combo = subparsers.add_parser('combo', help='Evaluate accuracy based on feature selection steps')
    parser_combo.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples and features.')
    parser_combo.add_argument('-t', '--target', required=True, type=str, help='Target group for classification.')
    parser_combo.add_argument('-p', '--prefix', required=True, help='Prefix for output files.')
    parser_combo.add_argument('-c', '--classifier', required=True, choices=['randomF', 'SVM'], help='Type of classifier to use.')
    parser_combo.add_argument('-s', '--stepwise', required=True, type=float, help='Step size or ratio for feature selection.')

    parser_predict_rf = subparsers.add_parser('randomF', help='Predict using an existing RandomForest model')
    parser_predict_rf.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples for prediction.')
    parser_predict_rf.add_argument('--model', required=True, help='Path to the trained model.')
    parser_predict_rf.add_argument('-p', '--prefix', required=True, help='Prefix for prediction outputs.')

    parser_predict_svm = subparsers.add_parser('SVM', help='Predict using an existing SVM model')
    parser_predict_svm.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples for prediction.')
    parser_predict_svm.add_argument('--model', required=True, help='Path to the trained model.')
    parser_predict_svm.add_argument('-p', '--prefix', required=True, help='Prefix for prediction outputs.')

    args = parser.parse_args()
    if args.command == 'learn':
        learn(args)
    elif args.command == 'randomF' or args.command == 'SVM':
        predict(args)
    elif args.command == 'combo':
        combo_evaluate(args)

if __name__ == '__main__':
    main()