import tensorflow as tf
import pandas as pd
import numpy as np

from utils.df_loader import (
    load_adult_df,
    load_compas_df,
    load_german_df,
    load_diabetes_df,
    load_breast_cancer_df,
)
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_df
from utils.models import load_models
from utils.print import print_block
from utils.exceptions import maybe_str_or_int, str2bool

import utils.deepfool as util_deepfool
import utils.carlini as util_carlini
import utils.lowprofool as util_lowprofool
import utils.fgsm as util_fgsm
import utils.bim as util_bim
import utils.mim as util_mim
import utils.pgd as util_pgd
import utils.boundary as util_boundary
import utils.hopskipjump as util_hopskipjump


from utils.save import save_result_as_csv, save_datapoints_as_npy, process_result, process_datapoints

seed = 42

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None # suppress "SettingWithCopyWarning" warning

tf.compat.v1.disable_eager_execution()

print("TF version: ", tf.__version__)
print("Eager execution enabled: ", tf.executing_eagerly())  # False    


import argparse

parser = argparse.ArgumentParser(prog="Generate Adversarial Examples")

# Add an argument
parser.add_argument('-n','--number', 
        type=maybe_str_or_int, 
        default='all', 
        help='Determine the number of adversarial examples generated in experiments. \
            Please use "all" or a postive int which is the multiple of batch_size.',
    )
parser.add_argument('-d', '--deepfool', 
        type=str2bool, default='False',
        help='Run DeepFool attack or not')

parser.add_argument('-c', '--carlini', 
        type=str2bool, default='False',
        help='Run Carlini attack or not')

parser.add_argument('-l', '--lowprofool', 
        type=str2bool, default='False',
        help='Run LowProFool attack or not')

parser.add_argument('-f', '--fgsm', 
        type=str2bool, default='False',
        help='Run FGSM attack or not')

parser.add_argument('-bi', '--bim', 
        type=str2bool, default='False',
        help='Run BIM attack or not')

parser.add_argument('-m', '--mim', 
        type=str2bool, default='False',
        help='Run MIM attack or not')

parser.add_argument('-p', '--pgd', 
        type=str2bool, default='False',
        help='Run PGD attack or not')

parser.add_argument('-b', '--boundary',
        type=str2bool, default='False',
        help='Run Boundary attack or not')

parser.add_argument('-ho', '--hopskipjump',
        type=str2bool, default='False',
        help='Run HopSkipJump attack or not')

parser.add_argument('-t', '--times',
        type=int, required=True, 
        help='Running times')


args = parser.parse_args()


num_instances = args.number # (1) 64 - 1 batch & (2) "all"
running_times = args.times

RUN_DEEPFOOL = args.deepfool
RUN_CARLINI = args.carlini
RUN_LOWPROFOOL = args.lowprofool
RUN_BOUNDARY = args.boundary
RUN_HOPSKIPJUMP = args.hopskipjump
RUN_FGSM = args.fgsm
RUN_BIM = args.bim
RUN_MIM = args.mim
RUN_PGD = args.pgd

models_list_original = ["lr","svc","nn_2"]
# models_list_extended = ["lr","svc","nn_2"]



def run_experiment(data_type_mixed: bool, running_times: int):
    """
    data_type_mix: True - mixed, False - Numerical only
    """
    if data_type_mixed:
        dataset = [
        "adult",
        "german",
        "compas",
        ]
    else:
        dataset = [
        "diabetes",
        "breast_cancer",
        ]

    #### Select dataset ####
    for dataset_name in dataset: 
        print(f"Dataset Name: [{dataset_name}]")
        if dataset_name == "adult":
            dataset_loading_fn = load_adult_df
        elif dataset_name == "german":
            dataset_loading_fn = load_german_df
        elif dataset_name == "compas":
            dataset_loading_fn = load_compas_df
        elif dataset_name == "diabetes":
            dataset_loading_fn = load_diabetes_df
        elif dataset_name == "breast_cancer":
            dataset_loading_fn = load_breast_cancer_df
        else:
            raise Exception("Unsupported dataset")

        df_info = preprocess_df(dataset_loading_fn)

        train_df, test_df = train_test_split(
            df_info.dummy_df, train_size=0.8, random_state=seed, shuffle=True
        )
        X_train = np.array(train_df[df_info.ohe_feature_names])
        y_train = np.array(train_df[df_info.target_name])
        X_test = np.array(test_df[df_info.ohe_feature_names])
        y_test = np.array(test_df[df_info.target_name])

        ### Load models
        models = load_models(X_train.shape[-1], dataset_name)

        # DeepFool attack
        if RUN_DEEPFOOL:
            deepfool_results = util_deepfool.generate_deepfool_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    models_to_run=["lr","svc","nn_2"],
                )
            deepfool_datapoints = process_datapoints(deepfool_results)
            save_datapoints_as_npy("deepfool", dataset_name, deepfool_datapoints, running_times)
            deepfool_result_dfs = process_result(deepfool_results, df_info)
            save_result_as_csv("deepfool", dataset_name, deepfool_result_dfs, running_times)

        if RUN_CARLINI:

            carlini_l_2_results = util_carlini.generate_carlini_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm="l_2", #["l_2", "l_inf"]
                    models_to_run=["lr","svc","nn_2"],
                )
            carlini_l_2_datapoints = process_datapoints(carlini_l_2_results)
            save_datapoints_as_npy("carlini_l_2", dataset_name, carlini_l_2_datapoints, running_times)
            carlini_l_2_result_dfs = process_result(carlini_l_2_results, df_info)
            save_result_as_csv("carlini_l_2", dataset_name, carlini_l_2_result_dfs, running_times)

            carlini_l_inf_results = util_carlini.generate_carlini_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm="l_inf", #["l_2", "l_inf"]
                    models_to_run=["lr","svc","nn_2"],
                )
            carlini_l_inf_datapoints = process_datapoints(carlini_l_inf_results)
            save_datapoints_as_npy("carlini_l_inf", dataset_name, carlini_l_inf_datapoints, running_times)
            carlini_l_inf_result_dfs = process_result(carlini_l_inf_results, df_info)
            save_result_as_csv("carlini_l_inf", dataset_name, carlini_l_inf_result_dfs, running_times)

        if not data_type_mixed and RUN_LOWPROFOOL:

            lowprofool_l_2_results = util_lowprofool.generate_lowprofool_result(
                    df_info,
                    models,
                    num_instances,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    norm=2, #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            lowprofool_l_2_datapoints = process_datapoints(lowprofool_l_2_results)
            save_datapoints_as_npy("lowprofool_l_2", dataset_name, lowprofool_l_2_datapoints, running_times)
            lowprofool_l_2_result_dfs = process_result(lowprofool_l_2_results, df_info)
            save_result_as_csv("lowprofool_l_2", dataset_name, lowprofool_l_2_result_dfs, running_times)

            lowprofool_l_inf_results = util_lowprofool.generate_lowprofool_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm='inf', #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            lowprofool_l_inf_datapoints = process_datapoints(lowprofool_l_inf_results)
            save_datapoints_as_npy("lowprofool_l_inf", dataset_name, lowprofool_l_inf_datapoints, running_times)
            lowprofool_l_inf_result_dfs = process_result(lowprofool_l_inf_results, df_info)
            save_result_as_csv("lowprofool_l_inf", dataset_name, lowprofool_l_inf_result_dfs, running_times)

        if RUN_FGSM:

            fgsm_l_1_results = util_fgsm.generate_fgsm_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm=1, #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            fgsm_l_1_datapoints = process_datapoints(fgsm_l_1_results)
            save_datapoints_as_npy("fgsm_l_1", dataset_name, fgsm_l_1_datapoints, running_times)
            fgsm_l_1_result_dfs = process_result(fgsm_l_1_results, df_info)
            save_result_as_csv("fgsm_l_1", dataset_name, fgsm_l_1_result_dfs, running_times)

            fgsm_l_2_results = util_fgsm.generate_fgsm_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm=2, #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            fgsm_l_2_datapoints = process_datapoints(fgsm_l_2_results)
            save_datapoints_as_npy("fgsm_l_2", dataset_name, fgsm_l_2_datapoints, running_times)
            fgsm_l_2_result_dfs = process_result(fgsm_l_2_results, df_info)
            save_result_as_csv("fgsm_l_2", dataset_name, fgsm_l_2_result_dfs, running_times)

            fgsm_l_inf_results = util_fgsm.generate_fgsm_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm='inf', #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            fgsm_l_inf_datapoints = process_datapoints(fgsm_l_inf_results)
            save_datapoints_as_npy("fgsm_l_inf", dataset_name, fgsm_l_inf_datapoints, running_times)
            fgsm_l_inf_result_dfs = process_result(fgsm_l_inf_results, df_info)
            save_result_as_csv("fgsm_l_inf", dataset_name, fgsm_l_inf_result_dfs, running_times)

        
        if RUN_BIM:

            bim_results = util_bim.generate_bim_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    models_to_run=["lr","svc","nn_2"],
                )
            bim_datapoints = process_datapoints(bim_results)
            save_datapoints_as_npy("bim", dataset_name, bim_datapoints, running_times)
            bim_result_dfs = process_result(bim_results, df_info)
            save_result_as_csv("bim", dataset_name, bim_result_dfs, running_times)

        
        if RUN_MIM:

            mim_results = util_mim.generate_mim_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    models_to_run=["lr","svc","nn_2"],
                )
            mim_datapoints = process_datapoints(mim_results)
            save_datapoints_as_npy("mim", dataset_name, mim_datapoints, running_times)
            mim_result_dfs = process_result(mim_results, df_info)
            save_result_as_csv("mim", dataset_name, mim_result_dfs, running_times)

        if RUN_PGD:

            pgd_l_1_results = util_pgd.generate_pgd_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm=1, #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            pgd_l_1_datapoints = process_datapoints(pgd_l_1_results)
            save_datapoints_as_npy("pgd_l_1", dataset_name, pgd_l_1_datapoints, running_times)
            pgd_l_1_result_dfs = process_result(pgd_l_1_results, df_info)
            save_result_as_csv("pgd_l_1", dataset_name, pgd_l_1_result_dfs, running_times)

            pgd_l_2_results = util_pgd.generate_pgd_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm=2, #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            pgd_l_2_datapoints = process_datapoints(pgd_l_2_results)
            save_datapoints_as_npy("pgd_l_2", dataset_name, pgd_l_2_datapoints, running_times)
            pgd_l_2_result_dfs = process_result(pgd_l_2_results, df_info)
            save_result_as_csv("pgd_l_2", dataset_name, pgd_l_2_result_dfs, running_times)

            pgd_l_inf_results = util_pgd.generate_pgd_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm='inf', #[int, float, 'inf']
                    models_to_run=["lr","svc","nn_2"],
                )
            pgd_l_inf_datapoints = process_datapoints(pgd_l_inf_results)
            save_datapoints_as_npy("pgd_l_inf", dataset_name, pgd_l_inf_datapoints, running_times)
            pgd_l_inf_result_dfs = process_result(pgd_l_inf_results, df_info)
            save_result_as_csv("pgd_l_inf", dataset_name, pgd_l_inf_result_dfs, running_times)


        # Black Box attack
        
        if RUN_BOUNDARY:
            boundary_results = util_boundary.generate_boundary_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    models_to_run=["dt","gbc","lr","svc","nn_2"], # 
                )
            boundary_datapoints = process_datapoints(boundary_results)
            save_datapoints_as_npy("boundary", dataset_name, boundary_datapoints, running_times)
            boundary_result_dfs = process_result(boundary_results, df_info)
            save_result_as_csv("boundary", dataset_name, boundary_result_dfs, running_times)
        
        if RUN_HOPSKIPJUMP:

            hopskipjump_l_2_results = util_hopskipjump.generate_hopskipjump_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm=2,
                    models_to_run=["dt","lr","svc","gbc","nn_2"],
            )
            hopskipjump_l_2_datapoints = process_datapoints(hopskipjump_l_2_results)
            save_datapoints_as_npy("hopskipjump_l_2", dataset_name, hopskipjump_l_2_datapoints, running_times)
            hopskipjump_l_2_result_dfs = process_result(hopskipjump_l_2_results, df_info)
            save_result_as_csv("hopskipjump_l_2", dataset_name, hopskipjump_l_2_result_dfs, running_times)


            hopskipjump_l_inf_results = util_hopskipjump.generate_hopskipjump_result(
                    df_info,
                    models,
                    num_instances,
                    X_test,
                    y_test,
                    norm="inf",
                    models_to_run=["dt","lr","svc","gbc","nn_2"],
            )
            hopskipjump_l_inf_datapoints = process_datapoints(hopskipjump_l_inf_results)
            save_datapoints_as_npy("hopskipjump_l_inf", dataset_name, hopskipjump_l_inf_datapoints, running_times)
            hopskipjump_l_inf_result_dfs = process_result(hopskipjump_l_inf_results, df_info)
            save_result_as_csv("hopskipjump_l_inf", dataset_name, hopskipjump_l_inf_result_dfs, running_times)



print_block("Experiment " + str(running_times))

run_experiment(data_type_mixed=True, running_times=running_times)
run_experiment(data_type_mixed=False, running_times=running_times)

