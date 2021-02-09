import pandas as pd

import Service.utils as util_service
import Service.models as model_service

if __name__ == "__main__":
    ## Uploading data -
    # train_reg, test_reg = util_service.load_new_data(min=0)

    train_reg = pd.read_csv('data/postprocess_data/train_reg_by_category_scaled_from_100_to_inf.csv')
    test_reg = pd.read_csv('data/postprocess_data/test_reg_by_category_scaled_from_100_to_inf.csv')

    print('TRAIN over \n-------\n',train_reg['y'].describe(),
          '\nTEST over \n-------\n',test_reg['y'].describe())

    X_train, y_train, X_test, y_test = util_service.data_scaling(train_reg,test_reg)
    prediction = model_service.linear_regression_model(X_train, y_train, X_test, y_test, columns = train_reg.columns[:-1])
    # prediction = model_service.tf_linear_model(X_train, y_train, X_test, y_test)


    util_service.model_evaluation(y_test, prediction, test_reg[['category', 'Year','Month']])






