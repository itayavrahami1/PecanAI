import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score


def linear_regression_model(X_train, y_train, X_test, y_test):
    """
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: predictions after creating a model
    """
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    return lm.predict(X_test)


def tf_linear_model(X_train, y_train, X_test, y_test):

    model = Sequential()

    model.add(Dense(76, activation='relu'))
    model.add(Dense(38, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(x=X_train, y=y_train.values,validation_data=(X_test, y_test.values),batch_size=128, epochs=50)

    losses = pd.DataFrame(model.history.history)
    losses.plot()
    plt.show()

    pred = model.predict(X_test)

    print('R^2:', r2_score(y_test, pred))
    print('MAE: ',mean_absolute_error(y_test, pred))
    print('RMSE',np.sqrt(mean_squared_error(y_test, pred)))
    print('EXPLAINED: ',explained_variance_score(y_test, pred))
    # print('MAPE:', np.mean(np.abs((y_test - pred) / y_test)) * 100)
    # Our predictions
    plt.scatter(y_test, pred)
    plt.plot(y_test, y_test, 'r')
    plt.show()
