import numpy as np
import pandas as pd

from time import time
from utils.preprocessing import DfInfo
from utils.preprocessing import inverse_dummy
from utils.exceptions import UnsupportedNorm, UnspportedNum
from scipy.stats import pearsonr

from art.attacks.evasion import LowProFool
from art.estimators.classification import SklearnClassifier, KerasClassifier
# from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
# from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnSVC

'''
LowProFool Attack
https://arxiv.org/pdf/1911.03274.pdf

Only works on numerical features and continous features, 
all non-ordered categorical features need to be dropped

'''


# Set the desired parameters for the attack
lowprofool_params = {
    'verbose': True,
    }
BATCH_SIZE = 64


def art_wrap_models(models, feature_range):
    '''
    Wrap the model to meet the requirements to art.attacks.evasion.LowProFool
    '''

    return {
        'lr': ScikitlearnLogisticRegression(models['lr'], clip_values=feature_range),
        'svc': ScikitlearnSVC(models['svc'], clip_values=feature_range),
        'nn_2': KerasClassifier(models['nn_2'], clip_values=feature_range),
    }

def get_lowprofool_instance(wrapped_models, norm):
    '''
    '''
    adv_instance = {}

    for k in wrapped_models.keys():

        if norm != "inf" and isinstance(norm, str):
            raise UnsupportedNorm()
        elif norm == None:
            raise UnsupportedNorm()
        else:
            adv_instance[k] = LowProFool(classifier=wrapped_models[k], norm=norm, **lowprofool_params)

    
    return adv_instance


def calculate_feature_importances(x,y):

    pearson_correlations = [pearsonr(x[:, col], y)[0] for col in range(x.shape[1])]
    absolutes = np.abs(np.array(pearson_correlations))
    importance_vec = absolutes / np.power(np.sum(absolutes ** 2), 0.5)
    importance_vec = np.array(importance_vec) / np.sum(importance_vec)
    
    return importance_vec.astype(np.float64)

def generate_lowprofool_result(
        df_info: DfInfo,
        models,
        num_instances,
        X_train, y_train,
        X_test, y_test,
        norm=None,
        models_to_run=['svc', 'lr', 'nn_2'],
):
    
    feature_range=(0,1)

    print("Feature range:" )
    print(feature_range)

    wrapped_models = art_wrap_models(models, feature_range)

    # importance_vec = calculate_feature_importances(X_train, y_train)

    # Get adversarial examples generator instance.
    adv_instance = get_lowprofool_instance(wrapped_models, norm=norm)

    # Initialise the result dictionary.(It will be the return value.)
    results = {}

    if isinstance(num_instances, int) and num_instances % BATCH_SIZE == 0:

        X_test_re=X_test[0:num_instances]
        y_test_re=y_test[0:num_instances]

    
    elif isinstance(num_instances, str) and num_instances == 'all':
        
        X_test_num = len(X_test) - (len(X_test)%BATCH_SIZE)
        X_test_re=X_test[0:X_test_num]
        y_test_num = len(y_test) - (len(y_test)%BATCH_SIZE)
        y_test_re=y_test[0:y_test_num]

    else:
        raise UnspportedNum()

    # y_test_re_ohe = np.zeros((y_test_re.size, 2))
    # y_test_re_ohe[np.arange(y_test_re.size), y_test_re] = 1

    # Loop through every models (svc, lr, nn_2)
    for k in models_to_run:
        # Intialise the result for the classifier (predicting model).
        results[k] = []

        print(f"Finding adversarial examples for {k}")

        adv_instance[k].fit_importances(x=X_train, y=y_train)

        # Get the prediction from original predictive model in a human-understandable format.
        if k == 'nn_2':
            prediction = np.argmax(models[k].predict(X_test_re), axis=1).astype(int) 
        else:
            prediction = models[k].predict(X_test_re)

        target = 1-prediction
        target_ohe = np.zeros((target.size, 2))
        target_ohe[np.arange(target.size), target] = 1
        

        start_t = time()
        adv = adv_instance[k].generate(X_test_re, target_ohe) 
        end_t = time()

        # Calculate the running time.
        running_time = end_t - start_t

        # Get the prediction from original predictive model in a human-understandable format.
        if k == 'nn_2':
            # nn return float [0, 1], so we need to define a threshold for it. (It's usually 0.5 for most of the classifier).
            prediction = np.argmax(models[k].predict(X_test_re), axis=1).astype(int)
            adv_prediction = np.argmax(models[k].predict(adv), axis=1).astype(int)
            
        else:
            # dt and rfc return int {1, 0}, so we don't need to define a threshold to get the final prediction.
            prediction = models[k].predict(X_test_re)
            adv_prediction = models[k].predict(adv)

        # Looping throguh first `num_instances` in the test set.
        for idx, instance in enumerate(X_test_re):
            example = instance.reshape(1, -1)
            adv_example = adv[idx].reshape(1,-1)

            adv_example_df = inverse_dummy(pd.DataFrame(adv_example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)

            # Change the found input from ohe format to original format.
            input_df = inverse_dummy(pd.DataFrame(example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)
            input_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([prediction[idx]])[0]

            results[k].append({
                "input": example,
                "input_df": input_df,
                "adv_example": adv_example,
                "adv_example_df": adv_example_df,
                "running_time": running_time,
                "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                "prediction": df_info.target_label_encoder.inverse_transform([prediction[idx]])[0],
                "adv_prediction": df_info.target_label_encoder.inverse_transform([adv_prediction[idx]])[0],
            })
    
    return results
