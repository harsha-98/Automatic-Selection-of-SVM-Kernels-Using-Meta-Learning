import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics, tree
from pymfe.mfe import MFE
import skgstat as skg
import os

svc_linear = SVC(kernel='linear', C=0.1, gamma="auto")
svc_poly = SVC(kernel='poly', C=0.1, gamma="auto")
svc_rbf = SVC(kernel='rbf', C=0.1, gamma="auto")

models = [svc_linear, svc_poly, svc_rbf]
metadataset = pd.DataFrame(columns=['dataset', 'range', 'sill'])
linear_score = []
poly_score = []
rbf_score = []
best_kernel = []


def read_data():
    """ Function to read datasets from the /data folder """
    fileDir = r"C:\Users\mslme\Documents\MSL\UH\SEM 2\Advanced ML\project\data"
    fileExtList = [".data", ".txt", ".csv"]
    all_files = []
    for fileExt in fileExtList:
        filelist = [_ for _ in os.listdir(fileDir) if _.endswith(fileExt)]
        all_files.append(filelist)
    file_list = [item for sublist in all_files for item in sublist]
    target_cols_dict = {'iris.data': 4, 'dataR2.csv': 9, 'transfusion.data': 4,
                        'data_banknote_authentication.txt': 4, 'Cryotherapy.csv': 6, 'caesarian.csv': 5,
                        'glass.data': 10}

    return file_list, target_cols_dict


def preprocess_data(file, target_columns):
    """Function to fetch source data and perform train test split of the dataset """
    dataset = pd.read_csv(f"data/{file}", sep=',', header=None)
    target_column_id = target_columns[file]
    data = dataset.loc[:, dataset.columns != target_column_id]
    target = dataset[[target_column_id]]
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=7, test_size=0.3)
    file_name = file.split('.')[0]

    return file_name, X_train, X_test, y_train, y_test


def metric_calculation(dataset_name, data, target):
    """ Function to calculate the IQR, Median of attributes of a dataset and to create a variogram model based on these values """
    column_length = data.shape[1]
    attr_metrics = pd.DataFrame(columns=['IQR', 'Median'])

    # for every attribute in a dataset calculate its IQR and Median values
    for i in range(column_length):
        mfe = MFE(features=["iq_range", "median"], summary='mean')
        mfe.fit(data[:, i], target)
        ft = mfe.extract()
        attr_metrics.loc[attr_metrics.shape[0]] = [ft[1][0], ft[1][1]]
    # Calculate the Variogram for a dataset with IQR and Median as its inputs
    V = skg.Variogram(attr_metrics['IQR'], attr_metrics['Median'])
    # Fetch the parameters range, sill of the variogram and store in metadata
    metadataset.loc[metadataset.shape[0]] = [dataset_name, round(V.parameters[0], 3), round(V.parameters[1], 3)]


def kernel_selection(X_train, X_test, y_train, y_test):
    """ Function to predict the best SVM kernel for a given dataset by calculating its accuracy scores"""
    global linear_score, poly_score, rbf_score, best_kernel
    ctr = 1
    for m in models:
        m.fit(X_train, y_train.ravel())
        y_pred = m.predict(X_test)
        if ctr == 1:
            l = round(metrics.accuracy_score(y_test, y_pred), 3)
            linear_score.append(l)
        if ctr == 2:
            p = round(metrics.accuracy_score(y_test, y_pred), 3)
            poly_score.append(p)
        if ctr == 3:
            r = round(metrics.accuracy_score(y_test, y_pred), 3)
            rbf_score.append(r)
        ctr = ctr + 1
    if l >= p and l >= r:
        best_kernel.append('linear')
    if p > l and p > r:
        best_kernel.append('polynomial')
    if r > l and r >= p:
        best_kernel.append('rbf')


def rule_generation():
    """ Function to create the meta-learning rule using the meta-learner (Decision tree)"""
    clf = DecisionTreeClassifier(max_depth=3, random_state=3)
    clf.fit(metadataset.iloc[:, 1:3], metadataset.iloc[:, 6])
    text_representation = tree.export_text(clf, feature_names=['range', 'sill'])
    print(text_representation)


def main():
    file_list, target_cols_dict = read_data()
    # Perform the pre-processing, scaling, metric calculation and kernel selection for every dataset
    for file in file_list:
        file_name, X_train, X_test, y_train, y_test = preprocess_data(file, target_cols_dict)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_trans = sc.transform(X_train)
        X_test_trans = sc.transform(X_test)
        metric_calculation(file_name, np.array(X_train_trans), np.array(y_train))
        kernel_selection(np.array(X_train_trans), np.array(X_test_trans), np.array(y_train), np.array(y_test))
    metadataset['linear_scores'] = linear_score
    metadataset['poly_scores'] = poly_score
    metadataset['rbf_scores'] = rbf_score
    metadataset['best_kernel'] = best_kernel
    rule_generation()
    print(metadataset)


if __name__ == '__main__':
    main()
