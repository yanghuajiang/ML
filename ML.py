import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import argparse


def learn(args):
    # Load the dataset
    df = pd.read_csv(args.matrix, sep='\t', index_col=0)
    X = df.drop(args.target, axis=1)
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model based on the chosen classifier
    if args.classifier == 'randomF':
        if args.min_samples_split:
            model = RandomForestClassifier(n_estimators=1000, random_state=42, min_samples_split=args.min_samples_split)
        else:
            model = RandomForestClassifier(n_estimators=1000, random_state=42)
    elif args.classifier == 'SVM':
        model = SVC(kernel='linear', random_state=42)

    # Train the model
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Calculate feature importances or coefficients
    if args.classifier == 'randomF':
        importances = model.feature_importances_
    elif args.classifier == 'SVM':
        importances = abs(model.coef_[0])

    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print(importance_df)

    # Save the feature importances
    importance_df.to_csv(f"{args.prefix}.importance", sep='\t', index=False)

    # Select top features based on importance
    if len(importance_df['Feature'].values) > args.num_tops:
        top_features = importance_df.head(args.num_tops)['Feature'].values
    else:
        top_features = importance_df['Feature'].values

    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    # Retrain the model with selected top features
    model.fit(X_train_selected, y_train.ravel())
    y_pred = model.predict(X_test_selected)
    print(f"Accuracy after feature selection: {accuracy_score(y_test, y_pred)}")

    # Calculate importances or coefficients for the selected features
    if args.classifier == 'randomF':
        importances = model.feature_importances_
    elif args.classifier == 'SVM':
        importances = abs(model.coef_[0])

    features = X_train_selected.columns
    importance_df_selected = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print(importance_df_selected)

    # Save the selected feature importances
    importance_df_selected.to_csv(f"{args.prefix}.selected_{args.num_tops}.importance", sep='\t', index=False)

    # Calculate and print distribution statistics for selected features in each class
    grouped_stats_selected = X_train_selected.join(y_train).groupby(args.target).agg(['mean', 'std', 'min', 'max'])
    print(grouped_stats_selected.T)

    # Save the distribution statistics
    grouped_stats_selected.T.to_csv(f"{args.prefix}.selected_{args.num_tops}.distribution", sep='\t')

    # Save the trained model if the output flag is set
    if args.output_model:
        joblib.dump(model, f"{args.prefix}.model.pkl")


def predict(args):
    # Load the trained model and new data
    model = joblib.load(args.model)
    new_data = pd.read_csv(args.matrix, sep='\t', index_col=0)
    predictions = model.predict(new_data)
    print("Predicted classes:", predictions)


def main():
    parser = argparse.ArgumentParser(description='Machine Learning model with feature importance and prediction.')
    subparsers = parser.add_subparsers(help='commands', dest='command')
    subparsers.required = True

    # Subcommand for learning models
    parser_learn = subparsers.add_parser('learn', help='Train the model')
    parser_learn.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples and features.')
    parser_learn.add_argument('-t', '--target', required=True, type=str, help='Target group for classification.')
    parser_learn.add_argument('-n', '--num_tops', type=int, default=20, help='Number of top features for second run.')
    parser_learn.add_argument('--min_samples_split', type=int, help='Minimum number of samples required to split an internal node.')
    parser_learn.add_argument('-o', '--output_model', action='store_true', help='Flag to save the trained model.')
    parser_learn.add_argument('-p', '--prefix', required=True, help='Prefix for output files.')
    parser_learn.add_argument('-c', '--classifier', required=True, choices=['randomF', 'SVM'], help='Type of classifier to use.')

    # Subcommand for prediction with RandomForest
    parser_predict_rf = subparsers.add_parser('randomF', help='Predict using an existing RandomForest model')
    parser_predict_rf.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples for prediction.')
    parser_predict_rf.add_argument('--model', required=True, help='Path to the trained model.')

    # Subcommand for prediction with SVM
    parser_predict_svm = subparsers.add_parser('SVM', help='Predict using an existing SVM model')
    parser_predict_svm.add_argument('-m', '--matrix', required=True, help='Tab-delimited matrix containing samples for prediction.')
    parser_predict_svm.add_argument('--model', required=True, help='Path to the trained model.')

    args = parser.parse_args()
    if args.command == 'learn':
        learn(args)
    elif args.command == 'randomF' or args.command == 'SVM':
        predict(args)

if __name__ == '__main__':
    main()
