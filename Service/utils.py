import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score


def set_category_reg(df):
    """
    :param df:
    :return: pd.DataFrame with 3 col of marker regression by category
    Exmp.
    categort | marker | ---> marker-2 | marker-1 | marker | marker+!
    """
    full_data_reg = pd.DataFrame(columns=['category','Year', 'Month', 'n-3', 'n-2', 'n-1', 'y'])
    all_marker = (list(df['Marker'].unique()))
    all_marker.sort()
    """array of all date - uses for the regression by category"""
    all_category = df['category'].unique()

    for cat in all_category:
        data_cat = pd.DataFrame(columns=['category','Year', 'Month', 'n-3', 'n-2', 'n-1', 'y'])
        data_by_category = df[df['category'] == cat].sort_values(by=['Marker']).set_index('Marker')

        if len(data_by_category) < 4:
            continue
        else:
            for i in range(2, len(data_by_category.index)):
                first_marker_index = all_marker.index(data_by_category.index[i])
                try:
                    # Checking if the category's datafarme has 4 consecutive markers.
                    # if it has creating regression columns
                    if ((data_by_category.index[i] == all_marker[first_marker_index]) &
                            (data_by_category.index[i - 1] == all_marker[first_marker_index - 1]) &
                            (data_by_category.index[i - 2] == all_marker[first_marker_index - 2])  &
                            (data_by_category.index[i - 3] == all_marker[first_marker_index - 3])):
                        data_cat = data_cat.append({
                            'category': cat,
                            'Year': int(data_by_category.index[i][:4]),
                            'Month': int(data_by_category.index[i][5:7]),
                            # 'n-3': data_by_category.loc[data_by_category.index[i - 3]]['Label'],
                            'n-2': data_by_category.loc[data_by_category.index[i - 2]]['Label'],
                            'n-1': data_by_category.loc[data_by_category.index[i - 1]]['Label'],
                            'y': data_by_category.loc[data_by_category.index[i]]['Label']},
                            ignore_index=True)

                except:
                    continue

        full_data_reg = full_data_reg.append(data_cat, ignore_index=True)

    return full_data_reg


def save_df(df,dir_name, file_name):
    file_name = f'data/{dir_name}/{file_name}.csv'
    df_csv = df.to_csv(index=False)
    my_file = open(file_name, 'w')
    my_file.write(df_csv)
    my_file.close()


def load_new_data(min=0,max=np.inf):
    """
     :param: min, max values for excluding data
    :return: Train and Test datasets as a regression datasets, ready for the model
    """

    train_df = pd.read_csv('data/original_data/train.csv')
    val_df = pd.read_csv('data/original_data/val.csv')
    test_df = pd.read_csv('data/original_data/test.csv')
    # Concat the training sets
    full_train_df = pd.concat([train_df, val_df])

    #
    # # Dropping data
    full_train_df = full_train_df[(full_train_df['Label'] > min) & (full_train_df['Label'] < max)]
    test_df = test_df[(test_df['Label'] > min) & (test_df['Label'] < max)]

    train_reg = set_category_reg(full_train_df)
    test_reg = set_category_reg(test_df)

    save_df(train_reg,'postprocess_data', f'train_reg_by_category_from_{min}_to_{max}_n_2')
    save_df(test_reg,'postprocess_data', f'test_reg_by_category_from_{min}_to_{max}_n_2')

    return train_reg, test_reg


def model_evaluation(true_val, prediction, test_columns):

    relative_err = np.abs((true_val - prediction) / true_val) * 100

    eval = pd.DataFrame({'category': test_columns['category'],
                         'Year': test_columns['Year'],
                         'Month': test_columns['Month'],
                         'true_val': true_val,
                         'pred': prediction,
                         'relative_err': relative_err})


    save_df(eval,'evaluations','eval_scaled_from_0_n_2')

    plt.scatter(true_val, prediction)
    plt.plot(true_val, true_val, c='red')
    plt.title('from 250')
    plt.xlabel('True Val')
    plt.legend(['True','Pred'])
    plt.show()

    print('R^2:', r2_score(true_val, prediction))
    print('MAE:', mean_absolute_error(true_val, prediction))
    print('MSE:', mean_squared_error(true_val, prediction))
    print('RMSE:', np.sqrt(mean_squared_error(true_val, prediction)))
    print('MAPE:', np.mean(np.abs((true_val - prediction) / true_val)) * 100)

    sns.distplot((true_val - prediction))
    plt.xlabel('Residuals')
    plt.ylabel('Dist')
    plt.show()


def data_scaling(train, test):

    X_train = train.drop(['y'], axis=1)
    y_train = train['y']
    X_test = test.drop(['y'], axis=1)
    y_test = test['y']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test