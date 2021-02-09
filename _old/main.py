import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Service.utils as util_service
import Service.models as model_service

if __name__ == "__main__":
    threshold = 0
    ## Uploading data
    # train_reg, test_reg = util_service.load_new_data(0,10000)

    train_reg = pd.read_csv('data/postprocess_data/train_reg_non_zero.csv')
    test_reg = pd.read_csv('data/postprocess_data/test_reg_non_zero.csv')
    print(f'TRAIN over {threshold}\n-------\n',train_reg['y'].describe(),
          f'\nTEST over {threshold}\n-------\n',test_reg['y'].describe())

    X_train, y_train, X_test, y_test = util_service.data_scaling(train_reg,test_reg)
    prediction = model_service.linear_regression_model(X_train, y_train, X_test, y_test)

    util_service.model_evaluation(y_test, prediction)
    # model = model_service.tf_linear_model(X_train, y_train, X_test, y_test)






