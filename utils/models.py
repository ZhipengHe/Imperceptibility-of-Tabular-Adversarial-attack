import pickle

import tensorflow as tf
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42


import os

def train_models(X_train, y_train):
    '''
    Construct and train ['dt', 'rfc', 'svc', 'lr', 'nn']

    nn - one output unit for binary classification (sigmoid)
    nn2 - two output units for binary classification (softmax)

    ---
    Return -> A dictionary container three trained models.
    '''
    nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Activation(tf.nn.sigmoid),
            ]
        )
    nn.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

    nn_2 = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(2),
                tf.keras.layers.Activation(tf.nn.softmax),
            ]
        )
    nn_2.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn_2.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

    models = {
        "dt": DecisionTreeClassifier(random_state=SEED).fit(X_train,y_train),
        "rfc": RandomForestClassifier(random_state=SEED).fit(X_train,y_train),
        "svc": LinearSVC(random_state=SEED).fit(X_train,y_train),
        "lr": LogisticRegression(random_state=SEED).fit(X_train,y_train),
        "gbc": GradientBoostingClassifier(random_state=SEED).fit(X_train,y_train),
        "nn": nn,
        "nn_2": nn_2,
    }

    return models


def evaluation_test(models, X_test, y_test):
    '''
    Evaluation the trained models.
    '''

    if 'dt' in models.keys():
        dt_pred = models['dt'].predict(X_test)
    if 'rfc' in models.keys():
        rfc_pred = models['rfc'].predict(X_test)
    if 'svc' in models.keys():
        svc_pred = models['svc'].predict(X_test)
    if 'lr' in models.keys():
        lr_pred = models['lr'].predict(X_test)
    if 'gbc' in models.keys():
        gbc_pred = models['gbc'].predict(X_test)
    if 'nn' in models.keys():
        nn_pred = (models['nn'].predict(X_test) > 0.5).flatten().astype(int)
    if 'nn_2' in models.keys():
        nn_2_pred = models['nn_2'].predict(X_test).argmax(axis=1).flatten().astype(int)

    # dt_acc = (models['dt'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    # rfc_acc = (models['rfc'].predict(X_test) == y_test).astype(int).sum() / X_test.shape[0]
    # nn_acc = ((models['nn'].predict(X_test) > 0.5).flatten().astype(int) == y_test).astype(int).sum() / X_test.shape[0]


    #### DT model 
    if 'dt' in models.keys():
        print_eval_states(y_test, dt_pred, name="Decision Tree")
    if 'rfc' in models.keys():
        print_eval_states(y_test, rfc_pred, name="Random Forest")
    if 'svc' in models.keys():
        print_eval_states(y_test, svc_pred, name="Linear Support Vector Classification")
    if 'lr' in models.keys():
        print_eval_states(y_test, lr_pred, name="Logistic Regression")
    if 'gbc' in models.keys():
        print_eval_states(y_test, gbc_pred, name="Gradient Boosting")
    if 'nn' in models.keys():
        print_eval_states(y_test, nn_pred, name="Neural Network (single output unit)")
    if 'nn_2' in models.keys():
        print_eval_states(y_test, nn_2_pred, name="Neural Network (two output units)")


def print_eval_states(y_test, y_pred, name=None):

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    recall_score, precision_score, accuracy_score, f1_score
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model: [{name}] | Accuracy: [{accuracy:.4f}] | Precision: [{precision:.4f} | Recall: [{recall:.4f}] | F1: [{f1:.4f}]")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Confusion Matrix ({name})', fontsize=18)
    plt.show()

def save_model_performance(models, dataset_name, X_test, y_test):

    # Create empty lists to store the accuracy, precision, recall, and F1 score for each model
    dataset_names = []
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Iterate over the models and compute the accuracy, precision, recall, and F1 score for each one
    for model, classifier in models.items():

        if model == 'dt':
            predictions = models['dt'].predict(X_test)
        if model == 'rfc':
            predictions = models['rfc'].predict(X_test)
        if model == 'svc':
            predictions = models['svc'].predict(X_test)
        if model == 'lr':
            predictions = models['lr'].predict(X_test)
        if model == 'gbc':
            predictions = models['gbc'].predict(X_test)
        if model == 'nn':
            predictions = (models['nn'].predict(X_test) > 0.5).flatten().astype(int)
        if model == 'nn_2':
            predictions = models['nn_2'].predict(X_test).argmax(axis=1).flatten().astype(int)
        
        # Calculate the accuracy, precision, recall, and F1 score of the model
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # Append the accuracy, precision, recall, and F1 score to the appropriate lists
        dataset_names.append(dataset_name)
        model_names.append(model)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    table = {
            'Dataset Name': dataset_names,
            'Model Name': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores
        }
    
    return table

def save_models(models, dataset_name, path='./saved_models'):
    '''
    Save trained models to desired `path`.
    '''
    storing_folder= f'{path}/{dataset_name}'
    os.makedirs(storing_folder, exist_ok=True)

    if 'dt' in models.keys():
        pickle.dump(models['dt'], open(f'{storing_folder}/dt.p', 'wb'))
    if 'rfc' in models.keys():
        pickle.dump(models['rfc'], open(f'{storing_folder}/rfc.p', 'wb'))
    if 'svc' in models.keys():
        pickle.dump(models['svc'], open(f'{storing_folder}/svc.p', 'wb'))
    if 'lr' in models.keys():
        pickle.dump(models['lr'], open(f'{storing_folder}/lr.p', 'wb'))
    if 'gbc' in models.keys():
        pickle.dump(models['gbc'], open(f'{storing_folder}/gbc.p', 'wb'))
    if 'nn' in models.keys():
        models['nn'].save(f'{storing_folder}/nn.h5',overwrite=True)
    if 'nn_2' in models.keys():
        models['nn_2'].save(f'{storing_folder}/nn_2.h5',overwrite=True)


def load_models(num_features, dataset_name, path='./saved_models'):
    '''
    Load pre-trained model from the `path`.  Will be saved in `./saved_models` by default
    '''

    storing_folder= f'{path}/{dataset_name}'

    ### Load
    models = {}
    models['dt'] = pickle.load(open(f'{storing_folder}/dt.p', 'rb'))
    models['rfc'] = pickle.load(open(f'{storing_folder}/rfc.p', 'rb'))
    models['svc'] = pickle.load(open(f'{storing_folder}/svc.p', 'rb'))
    models['lr'] = pickle.load(open(f'{storing_folder}/lr.p', 'rb'))
    models['gbc'] = pickle.load(open(f'{storing_folder}/gbc.p', 'rb'))
    models['nn'] = tf.keras.models.load_model(f'{storing_folder}/nn.h5')
    models['nn_2'] = tf.keras.models.load_model(f'{storing_folder}/nn_2.h5')

    ## Initialise NN output shape as (None, 1) for tensorflow.v1
    models['nn'].predict(np.zeros((2, num_features)))
    models['nn_2'].predict(np.zeros((2, num_features)))


    return models


