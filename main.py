from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values


def main():
    df_train, df_test = load_data(LESCORTS_ST)
    # initial_data_stats(df_train, df_test, POBLESEC_ST)
    handle_missing_values(df_train, df_test)



if __name__ == '__main__':
    main()