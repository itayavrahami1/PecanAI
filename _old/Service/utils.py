import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score


def fix_date(df):
    """
    :param df: Data frame with Marker (date) as a string
    :return: Date frame with Marker (date) as a datetime obj + adding year and month columns
    """
    df['date'] = pd.to_datetime(df['Marker'])

    df['Year'] = df['date'].apply(lambda date: date.year)
    df['Month'] = df['date'].apply(lambda date: date.month)

    return df


def fill_category_label(df):
    mean_label_per_date = df.groupby('Marker').mean()['Label']
    mean_label_per_category = df.groupby('category').mean()['Label']

    all_category = df['category'].unique()
    all_markers = df['Marker'].unique()
    print(len(all_category))

    for marker in all_markers:
        category_diff = set(all_category).difference(set(df[df['Marker'] == marker]['category']))
        print(len(set(df[df['Marker'] == marker]['category'])), 'LEN DIFF', len(category_diff))
        for category in category_diff:
            # print(np.mean([mean_label_per_category[category],mean_label_per_date[marker]]))
            # label_val = np.mean([mean_label_per_category[category], mean_label_per_date[marker]])
            new_row = {'Marker': marker, 'pecan_id': 0, 'category': category, 'Label': 0}
            df = df.append(new_row, ignore_index=True)

    # df_csv = df.to_csv(index=False)
    #
    # my_file = open('data/full_train.csv', 'w')
    # my_file.write(df_csv)
    #
    # my_file.close()

    return df

def set_reg_all_markers(df):
    all_marker = (list(df['Marker'].unique()))
    all_marker.sort()

    df_reg = pd.DataFrame(columns=all_marker)


def set_category_reg(df):
    """
    :param df:
    :return: pd.DataFrame with 3 col of marker regression by category
    Exmp.
    categort | marker | ---> marker-2 | marker-1 | marker | marker+!
    """
    full_data_reg = pd.DataFrame(columns=['Year', 'Month', 'n-2', 'n-1', 'n', 'y'])
    all_marker = (list(df['Marker'].unique()))
    all_marker.sort()
    """array of all date - uses for the regression by category"""
    all_category = df['category'].unique()

    for cat in all_category:
        print(cat)
        data_cat = pd.DataFrame(columns=['Year', 'Month', 'n-2', 'n-1', 'n', 'y'])
        data_by_category = df[df['category'] == cat].sort_values(by=['Marker']).set_index('Marker')

        if len(data_by_category) < 5:
            continue
        else:
            for i in range(2, len(data_by_category.index)):
                first_marker_index = all_marker.index(data_by_category.index[i])
                try:
                    # Checking if the category's datafarme has 4 consecutive markers.
                    # if it has creating regression columns
                    if ((data_by_category.index[i] == all_marker[first_marker_index]) &
                            (data_by_category.index[i - 1] == all_marker[first_marker_index - 1]) &
                            (data_by_category.index[i - 2] == all_marker[first_marker_index - 2]) &
                            (data_by_category.index[i + 1] == all_marker[first_marker_index + 1])):
                        data_cat = data_cat.append({
                            'Year': int(data_by_category.index[i][:4]),
                            'Month': int(data_by_category.index[i][5:7]),
                            'n-2': data_by_category.loc[data_by_category.index[i - 2]]['Label'],
                            'n-1': data_by_category.loc[data_by_category.index[i - 1]]['Label'],
                            'n': data_by_category.loc[data_by_category.index[i]]['Label'],
                            'y': data_by_category.loc[data_by_category.index[i + 1]]['Label']},
                            ignore_index=True)

                except:
                    continue

        full_data_reg = full_data_reg.append(data_cat, ignore_index=True)

    return full_data_reg


def save_df(df, file_name):
    file_name = f'data/postprocess_data/{file_name}.csv'
    df_csv = df.to_csv(index=False)
    my_file = open(file_name, 'w')
    my_file.write(df_csv)
    my_file.close()


def load_new_data(min=0,max=np.inf):
    """
    :return: Train and Test datasets as a regression datasets, ready for the model
    """

    train_df = pd.read_csv('data/original_data/train.csv')
    val_df = pd.read_csv('data/original_data/val.csv')
    test_df = pd.read_csv('data/original_data/test.csv')
    # Concat the training sets
    full_train_df = pd.concat([train_df, val_df])

    print('MIN\n--------\n', min,'\nMAX\n------\n',max)
    # Date Formatting
    full_train_df = fix_date(full_train_df)
    test_df = fix_date(test_df)
    #
    # # Dropping data
    full_train_df = full_train_df[(full_train_df['Label'] > min) & (full_train_df['Label'] < max)]
    test_df = test_df[(test_df['Label'] > min) & (test_df['Label'] < max)]

    train_reg = set_category_reg(full_train_df)
    test_reg = set_category_reg(test_df)

    save_df(train_reg, 'train_reg_non_zero_up_to_10000')
    save_df(test_reg, 'test_reg_non_zero_up_to_10000')

    return train_reg, test_reg


def model_evaluation(true_val, prediction):

    plt.scatter(true_val, prediction)
    plt.plot(true_val, true_val, c='red')
    plt.xlabel('True Val')
    plt.legend(['True','Pred'])
    plt.show()

    epsilon = np.finfo(float).eps
    true_test = true_val.apply(lambda value: np.max([epsilon, value]))

    relative_err = np.abs((true_val - prediction) / true_val)
    plt.scatter(true_val, relative_err)
    plt.xlabel('True Val')
    plt.ylabel('Error [%]')
    plt.xlim([0,500])
    plt.ylim([0,100])
    plt.show()

    sns.distplot(relative_err)
    plt.xlabel('relative_err')
    plt.ylabel('Dist')
    plt.xlim([0, 40])
    # plt.ylim([0, 100])
    plt.show()

    print('R^2:', r2_score(true_val, prediction))
    print('MAE:', mean_absolute_error(true_val, prediction))
    print('MSE:', mean_squared_error(true_val, prediction))
    print('RMSE:', np.sqrt(mean_squared_error(true_val, prediction)))
    print('MAPE:', np.mean(np.abs((true_val - prediction) / true_val)) * 100)
    print('EXPLAINED: ', explained_variance_score(true_val, prediction))
    # print('MAPE:', np.mean(np.abs((true_val - prediction) / true_test) * 100))
    print('%Err < 5%\n------\n', len(relative_err[relative_err<5])/len(relative_err),
          '\n%Err >20%\n------\n' ,len(relative_err[relative_err>10])/len(relative_err))
    sns.distplot((true_val - prediction))

    plt.xlim([-1000, 1000])
    plt.xlabel
    plt.show()


def data_scaling(train, test):

    X_train = train.drop('y', axis=1)
    y_train = train['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test