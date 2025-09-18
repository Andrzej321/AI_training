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
    csv_files = [f for f in os.listdir(result_folder_loc) if f.endswith(".csv")]

    num_of_csv_files = len(csv_files)

    #Creating the dataframe storing the results of the evaluation
    evaluation_model_specific_df = pd.DataFrame(columns = ['Measurement', 'RMSE', 'Maximum Absolute Error', 'Percentage within tolerance '])
    evaluation_summed_df = pd.DataFrame(columns = ['model_type', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'percentage_within_tolerance'])

    for model_it in range(num_of_models):

        for csv_it in range(num_of_csv_files):

            df = pd.read_csv(result_folder_loc + csv_files[csv_it])



    #Calculating the RMSE error


    #Calculating the Maximum Absolute Error


    #Calculating the percentage within tolerance limit
    tolerance_lower = 0.1
    tolerance_upper = 0.2
