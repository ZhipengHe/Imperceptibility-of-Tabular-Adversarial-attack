import os
import pandas as pd
import numpy as np

def save_result_as_csv(alg_name, dataset_name, results_df, running_times = None):
    '''
    Save the result dataframe as csv file. It will creat the folder for you and print out the destination. 
    '''
    path = f"./results/{alg_name}_{dataset_name}"
    os.makedirs(path, exist_ok=True)
    for df_k in results_df.keys():
        if running_times is None:
            results_df[df_k].to_csv(f"{path}/{alg_name}_{dataset_name}_{df_k}_result.csv")
        else:
            results_df[df_k].to_csv(f"{path}/{alg_name}_{dataset_name}_{df_k}_result_{running_times}.csv")

    print(f"Result has been saved to {path}")

def save_datapoints_as_npy(alg_name, dataset_name, datapoints, running_times = None):

    path = f"./datapoints/{alg_name}_{dataset_name}"
    os.makedirs(path, exist_ok=True)
    for k in datapoints.keys():
        if running_times is None:
            with open(f"{path}/{alg_name}_{dataset_name}_{k}_arr.npy", 'wb') as f:
                np.save(f, datapoints[k]["arr"])
            with open(f"{path}/{alg_name}_{dataset_name}_{k}_arr_adv.npy", 'wb') as f:
                np.save(f, datapoints[k]["arr_adv"])
        else:
            with open(f"{path}/{alg_name}_{dataset_name}_{k}_arr_{running_times}.npy", 'wb') as f:
                np.save(f, datapoints[k]["arr"])
            with open(f"{path}/{alg_name}_{dataset_name}_{k}_arr_adv_{running_times}.npy", 'wb') as f:
                np.save(f, datapoints[k]["arr_adv"])

    
    print(f"Original data points and adversarial examples have been saved to {path}")



def process_result(results, df_info):
    '''
    Process the result dictionary to construct data frames for each predictive models.
    '''

    results_df = {}

    # Loop through all models
    for k in results.keys():

        all_data = []
        for i in range(len(results[k])):

            final_df = pd.DataFrame([{}])

            # Inverse the scaling process to get the original data for input.
            scaled_input_df = results[k][i]['input_df'].copy(deep=True)
            origin_columns = [
                f"origin_input_{col}" for col in scaled_input_df.columns]
            origin_input_df = scaled_input_df.copy(deep=True)
            scaled_input_df.columns = [
                f"scaled_input_{col}" for col in scaled_input_df.columns]

            origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                origin_input_df[df_info.numerical_cols])
            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            # If counterfactaul found, inverse the scaling process to get the original data for cf.
            if not results[k][i]['adv_example_df'] is None:
                scaled_ae_df = results[k][i]['adv_example_df'].copy(deep=True)
                # Comment this
                # scaled_cf_df.loc[0, target_name] = target_label_encoder.inverse_transform([scaled_cf_df.loc[0, target_name]])[0]
                origin_ae_columns = [
                    f"origin_adv_{col}" for col in scaled_ae_df.columns]
                origin_ae_df = scaled_ae_df.copy(deep=True)
                scaled_ae_df.columns = [
                    f"scaled_adv_{col}" for col in scaled_ae_df.columns]

                origin_ae_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    origin_ae_df[df_info.numerical_cols])
                origin_ae_df.columns = origin_ae_columns

                final_df = final_df.join([scaled_ae_df, origin_ae_df])

            # Record additional information.
            final_df['running_time'] = results[k][i]['running_time']
            final_df['Predict_Success'] = "Y" if results[k][i]['ground_truth'] == results[k][i]['prediction'] else "N"
            final_df['Attack_Success?'] = "Y" if not results[k][i]['ground_truth'] == results[k][i]['adv_prediction'] else "N"
            final_df['ground_truth'] = results[k][i]['ground_truth']
            final_df['prediction'] = results[k][i]['prediction']
            final_df['adv_prediction'] = results[k][i]['adv_prediction']

            all_data.append(final_df)

        results_df[k] = pd.concat(all_data)

    return results_df


def process_datapoints(results):
    '''
    '''

    datapoints = {}

    for k in results.keys():
        arr_list=[]
        arr_list_adv=[]

        for i in range(len(results[k])):
            arr_list.append(results[k][i]['input'])

            if not results[k][i]['adv_example'] is None:
                arr_list_adv.append(results[k][i]['adv_example'])
        
        arr = np.concatenate(arr_list, axis=0)
        arr_adv = np.concatenate(arr_list_adv, axis=0)
    
        datapoints[k] = {"arr": arr, "arr_adv": arr_adv}

    return datapoints



            
    
    

    