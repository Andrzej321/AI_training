import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    #Reading in the num of models and model locations
    model_folder_loc = "../2_trained_models/GRU/trained_models/i7/it_4_norm/state_models/lon/"
    pt_files = [f for f in os.listdir(model_folder_loc) if f.endswith(".pt")]

    num_of_models = len(pt_files)

    #Reading in the result files
    result_folder_loc = "../2_trained_models/GRU/trained_models/i7/it_4_norm/results/"
    meas_files = [f for f in os.listdir(result_folder_loc) if f.endswith(".csv")]

    num_of_meas_files = len(meas_files)

    #Location to save the eval results
    eval_folder_loc = "../2_trained_models/GRU/trained_models/i7/it_4_norm/eval/"

    rows_spec = []
    rows_sum = []

    #Threshold values
    tolerance_1 = 0.3
    tolerance_2 = 0.4

    #For the model-specific results
    for model_it in range(num_of_models):

        rmse_sum = 0
        mae_sum = 0
        pwt_1_abs_sum = 0
        pwt_2_abs_sum = 0
        pwt_1_rel_sum = 0
        pwt_2_rel_sum = 0
        max_err_sum = 0

        for meas_it in range(num_of_meas_files):

            df = pd.read_csv(result_folder_loc + meas_files[meas_it])

            err = df[pt_files[model_it]] - df['veh_u']

            #Calculating the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))

            rmse_sum += rmse
            mae_sum += mae

            #Determining Maximum Absolute Error (MaxAE)
            max_err = np.max(np.abs(err))

            if max_err > max_err_sum:
                max_err_sum = max_err

            # Absolute tolerances
            pwt_1_abs = (np.abs(err) <= tolerance_1).mean()
            pwt_2_abs = (np.abs(err) <= tolerance_2).mean()

            # If you intended relative tolerances:
            pwt_1_rel = (np.abs(err) <= tolerance_1 * np.abs(df['veh_u'])).mean()
            pwt_2_rel = (np.abs(err) <= tolerance_2 * np.abs(df['veh_u'])).mean()

            pwt_1_abs *= 100
            pwt_2_abs *= 100
            pwt_1_rel *= 100
            pwt_2_rel *= 100

            pwt_1_abs_sum += pwt_1_abs
            pwt_2_abs_sum += pwt_2_abs
            pwt_1_rel_sum += pwt_1_rel
            pwt_2_rel_sum += pwt_2_rel

            rows_spec.append({
                'Model': pt_files[model_it],
                'Meas': meas_files[meas_it],
                'RMSE': rmse,
                'MAE': mae,
                'MaxAE': max_err,
                'PwT_1_abs': pwt_1_abs,
                'PwT_2_abs': pwt_2_abs,
                'PwT_1_rel': pwt_1_rel,
                'PwT_2_rel': pwt_2_rel,
            })

        rows_sum.append({
            'Model': pt_files[model_it],
            'RMSE': rmse_sum / num_of_meas_files,
            'MAE': mae_sum / num_of_meas_files,
            'MaxAE': max_err_sum,
            'PwT_1_abs': pwt_1_abs_sum / num_of_meas_files,
            'PwT_2_abs': pwt_2_abs_sum / num_of_meas_files,
            'PwT_1_rel': pwt_1_rel_sum / num_of_meas_files,
            'PwT_2_rel': pwt_2_rel_sum / num_of_meas_files,
        })


    eval_model_spec_df = pd.DataFrame(rows_spec)
    eval_model_sum_df = pd.DataFrame(rows_sum)

    eval_model_spec_df.to_csv(eval_folder_loc + "eval_model_spec.csv")
    eval_model_sum_df.to_csv(eval_folder_loc + "eval_model_sum.csv")