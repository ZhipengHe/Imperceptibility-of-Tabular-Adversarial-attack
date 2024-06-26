import numpy as np
import pandas as pd
from enum import Enum
from typing import List
from utils.preprocessing import DfInfo
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


class InstanceType(Enum):
    ScaledInput = "scaled_input_"
    ScaledAdv = "scaled_adv_"
    OriginInput = "origin_input_"
    OriginAdv = "origin_adv_"

'''
Evaluation Functions.
'''

def get_Linf(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=np.inf)

def get_L2(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=2)


def get_L1(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=1)


def get_sparsity(**kwargs):

    # should remove the target column first.
    
    input_df = kwargs['not_dummy_input']
    adv_df = kwargs['not_dummy_adv']

    input_array = np.array(input_df)
    adv_array = np.array(adv_df)

    # return np.equal(input_array, adv_array).astype(int).sum(axis=1)
    return (input_array != adv_array).astype(int).sum(axis=1)




def get_realisitic(**kwargs):
    '''
    Checking if the numerical columns are in the range of [0, 1].
    '''
    df_info: DfInfo = kwargs['df_info']
    adv_num_array = np.array(kwargs['adv'][df_info.numerical_cols])
    return np.all(np.logical_and(adv_num_array >= 0, adv_num_array <= 1 ), axis=1)

def get_sensitivity(**kwargs,):
    '''
    Get Sensitivity between input and adv. 
    '''

    eps = 1e-8

    input_df = kwargs['input']
    adv_df = kwargs['adv']
    df_info = kwargs['df_info']

    ohe_cat_cols = df_info.get_ohe_cat_cols()
    ohe_num_cols = df_info.get_ohe_num_cols()

    numerical_stds = df_info.get_numerical_stds()

    sen_df = pd.DataFrame({}, columns= df_info.ohe_feature_names)
    sen_df[ohe_cat_cols] = (input_df[ohe_cat_cols] != adv_df[ohe_cat_cols]).astype(int)
    for num_col in ohe_num_cols: 
        sen_df[num_col] = abs(adv_df[num_col] - input_df[num_col]) / (numerical_stds[num_col] + eps)

    if len(ohe_cat_cols) > 0 and len(ohe_num_cols) > 0:
        return (sen_df[ohe_num_cols].mean(axis=1) + sen_df[ohe_cat_cols].mean(axis=1)).tolist()
    elif len(ohe_num_cols) > 0:
        return sen_df[ohe_num_cols].mean(axis=1).tolist()
    elif len(ohe_cat_cols) > 0:
        return sen_df[ohe_cat_cols].mean(axis=1).tolist()
    else:
        raise Exception("No columns provided for MAD.")



def get_mahalanobis(**kwargs,):
    '''
    Get Mahalanobis distance between input and adv.
    '''
    input_df = kwargs['input']
    adv_df = kwargs['adv']
    df_info = kwargs['df_info']

    VI_m = df_info.dummy_df[df_info.ohe_feature_names].cov().to_numpy()

    return [distance.mahalanobis(input_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                adv_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                VI_m) for i in range(len(input_df))]


def get_neighbour_distance(**kwargs,):

    adv_df = kwargs['adv']
    df_info = kwargs['df_info']
    
    adv_arr = adv_df[df_info.ohe_feature_names].to_numpy()
    dataset = df_info.dummy_df[df_info.ohe_feature_names].to_numpy()

    return distance.cdist(adv_arr, dataset, 'minkowski', p=2).min(axis=1).tolist()




class EvaluationMatrix(Enum):
    '''
    All evaluation function should be registed here.
    '''
    L1 = "eval_L1"
    L2 = "eval_L2"
    Linf = "eval_Linf"
    Sparsity = "eval_Sparsity"
    Realistic = "eval_Realistic"
    Sen = "eval_Sen"
    Mahalanobis = "eval_Mahalanobis"
    # Perturbation_Sensitivity = "eval_Perturbation_Sensitivity"
    # Neighbour_Distance = "eval_Neighbour_Distance"

evaluation_name_to_func = {
    # All evaluation function should be registed here as well
    EvaluationMatrix.L1: get_L1,
    EvaluationMatrix.L2: get_L2,
    EvaluationMatrix.Linf: get_Linf,
    EvaluationMatrix.Sparsity: get_sparsity,
    EvaluationMatrix.Realistic: get_realisitic,
    EvaluationMatrix.Sen: get_sensitivity,
    EvaluationMatrix.Mahalanobis: get_mahalanobis,
    # EvaluationMatrix.Perturbation_Sensitivity: get_perturbation_sensitivity,
    # EvaluationMatrix.Neighbour_Distance: get_neighbour_distance,
}


'''
Util functions.
'''

def get_dummy_version(input_df: pd.DataFrame, df_info: DfInfo):
    '''
    Transform the categorical data to ohe format. (Better for calculating the distance)
    '''

    def get_string_dummy_value(x):
        if isinstance(x, float) and x==x:
            x = int(x)

        return str(x)

    number_of_instances = len(input_df)

    init_row = {}
    for k in df_info.ohe_feature_names:
        init_row[k] = 0

    init_df = pd.DataFrame([init_row]*number_of_instances,
                           columns=df_info.ohe_feature_names)

    for k, v in df_info.cat_to_ohe_cat.items():
        for ohe_f in v:
            init_df[ohe_f] = input_df[k].apply(
                lambda x: 1 if ohe_f.endswith(get_string_dummy_value(x)) else 0).tolist()

    for col in df_info.numerical_cols:
        init_df[col] = input_df[col].tolist()

    return init_df


def get_type_instance(df: pd.DataFrame, instance_type: InstanceType, with_original_name: bool = True):
    '''
    Get certain type of instance in the result data frame. Check `InstanceType` to know all types.
    '''

    df = df.copy(deep=True)
    return_df = df[[
        col for col in df.columns if col.startswith(instance_type.value)]]

    if with_original_name:
        return_df.columns = [col.replace(
            instance_type.value, "") for col in return_df.columns]

    return return_df


def prepare_evaluation_dict(result_df: pd.DataFrame, df_info: DfInfo):
    '''
    Prepare the information needed to perform evaluation.
    '''

    return {
        "input": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledInput), df_info),
        "adv": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledAdv), df_info),
        "not_dummy_input": get_type_instance(result_df, InstanceType.ScaledInput).drop(labels=df_info.target_name, axis=1), # .drop(df_info.target_name, axis=1)
        "not_dummy_adv": get_type_instance(result_df, InstanceType.ScaledAdv), #.drop(df_info.target_name, axis=1)
        "df_info": df_info,
    }


def get_evaluations(result_df: pd.DataFrame, df_info: DfInfo, matrix: List[EvaluationMatrix], models=None, model_name=None):
    '''
    Perform evaluation on the result dataframe according to the matrix given.

    [result_df] -> data frame containing input query and its counterfactaul.
    [df_info] -> DfInfo instance containing all data information.
    [matrix] -> The evaluation matrix to perform on `result_df`.
    '''

    evaluation_df = result_df.copy(deep=True)

    adv_found_eaval_df = evaluation_df.copy(deep=True)

    if len(adv_found_eaval_df) < 1:
        raise Exception("No adversarial example found, can't provide any evaluation.")

    input_and_adv = prepare_evaluation_dict(adv_found_eaval_df, df_info)


    metric = {}

    for m in matrix:
        adv_metric = evaluation_name_to_func[m](**input_and_adv)
        adv_found_eaval_df[m.value] = adv_metric
        metric[m.value]=np.array(adv_metric).mean().astype(np.float32)

    evaluation_df.loc[:, adv_found_eaval_df.columns] = adv_found_eaval_df

    return evaluation_df, metric



