import statistics_functions as fn
import missing_values as ms
import imputation_algs as imp

POLLUTANTS = ['pm10', 'pm25','o3', 'no2'] 
WEATHER = ['precipitation', 'temp_max', 'temp_min', 'wind']
DATA_SETS = {
    'seattle-weather.csv': WEATHER,
    'south-korean-pollution-data.csv': POLLUTANTS
}

scenario_1 = {'percent_range': (10, 15), 'missing_type': 'random'}
scenario_2 = {'percent_range': (10, 15), 'missing_type': 'burst'}
scenario_3 = {'percent_range': (20, 25), 'missing_type': 'random'}
scenario_4 = {'percent_range': (20, 25), 'missing_type': 'burst'}

SCENARIOS = [scenario_1, scenario_2, scenario_3, scenario_4]

def main():
# Loop through the dataset columns and apply analytics + imputation techniques
    for file_path, columns in DATA_SETS.items():
        for scenario in SCENARIOS:
            df = fn.load_data(file_path)
            fn.apply_analitics(df, columns)

            # Introduce missing values
            df_with_missing = ms.introduce_missing_values(
                df, columns, 
                percent_range=scenario['percent_range'], 
                seed=42, 
                missing_type = scenario['missing_type'])
            fn.apply_analitics(df_with_missing, columns)

            # # Check for missing values before imputation
            print(f"Missing values in {file_path} before imputation:")
            fn.check_missing_values(df_with_missing)

            # Apply each imputation method and check results
            print(f"--- Applying Simple Imputation (Mean) ---")
            df_imputed_simple = imp.simple_imputation(df_with_missing, columns, strategy='mean')
            fn.check_missing_values(df_imputed_simple)
            fn.compute_mae(df, df_imputed_simple, columns)
            fn.compute_rmse(df, df_imputed_simple, columns)

            print(f"--- Applying Multiple Imputation ---")
            df_imputed_multiple = imp.multiple_imputation(df_with_missing, columns)
            fn.check_missing_values(df_imputed_multiple)
            fn.compute_mae(df, df_imputed_multiple, columns)
            fn.compute_rmse(df, df_imputed_multiple, columns)

            print(f"--- Applying EM Imputation ---")
            df_imputed_em = imp.em_imputation(df_with_missing, columns)
            fn.check_missing_values(df_imputed_em)
            fn.compute_mae(df, df_imputed_em, columns)
            fn.compute_rmse(df, df_imputed_em, columns)

            input("Press Enter to continue...")


if __name__ == "__main__":
    main()