import os
import pickle
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML HomeWork"
    )
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--data", type=str, default='hiv')
    parser.add_argument("--model-type", type=str,
                        choices=['RandomForest', 'SVM'])
    parser.add_argument("--dim", type=int, default=-1)

    args = parser.parse_args()
    return args


def prepare_data(path, dim=None):

    with open(path, 'rb') as file:
        data = pickle.load(file)
    X_list = []
    y_list = []
    for i in tqdm.tqdm(data):
        i['feature'] = i['feature'].mean(axis=0).numpy()
        X_list.append(i['feature'])
        if i['label'].dim() == 0:
            y_list.append(i['label'].item())
        else:
            y_list.append(i['label'][dim].item())
    X = np.array(X_list)
    y = np.array(y_list)
    mask = y != -1
    X = X[mask]
    y = y[mask]
    return X, y


def main(args):
    data = args.data
    model_type = args.model_type
    train_path = os.path.join(args.data_root, data, "train.pkl")
    test_path = os.path.join(args.data_root, data, "test.pkl")
    val_path = os.path.join(args.data_root, data, "valid.pkl")
    X_train, y_train = prepare_data(train_path, args.dim)
    X_test, y_test = prepare_data(test_path, args.dim)
    X_val, y_val = prepare_data(val_path, args.dim)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    data_size = X_train.shape[0]
    if data_size > 50000 and model_type == "SVM":
        np.random.shuffle(X_train)
        np.random.shuffle(y_train)
        X_train = X_train[:50000]
        y_train = y_train[:50000]
    X_test = scaler.transform(X_test)
    if model_type == "SVM":
        model = SVC(kernel='rbf', C=1.0, gamma='scale',
                    random_state=42, verbose=True, class_weight='balanced')
    elif model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, verbose=True, n_jobs=-1, class_weight='balanced')
    else:
        raise ValueError(f"no model type: {model_type}")
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_type}_{data}.pkl')
    model = joblib.load(f'{model_type}_{data}.pkl')
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    test_acc = accuracy_score(y_test, y_test_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    info = f"model_type: {model_type}; data: {data}; dim: {args.dim}\ntest_acc: {round(test_acc,4)}; test_auc: {round(test_auc,4)};"
    with open("train_info.txt", 'a') as f:
        f.write(info)
        f.write("\n")
    print(info)


if __name__ == "__main__":
    args = parse_args()
    main(args)
