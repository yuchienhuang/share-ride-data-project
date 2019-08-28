import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


df = pd.DataFrame(np.load('day_1.npy'), columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
       'passenger_count', 'fare_amount', 'trip_distance', 'day'],
      dtype='object')
df_new = df[['trip_distance']]

squared = ((df['dropoff_longitude'] - df['pickup_longitude'])**2 + (df['dropoff_latitude'] - df['pickup_latitude'])**2 ).values
squared = np.array(list(squared))

df_new['coord_segment_distance'] = pd.Series(np.sqrt(squared), index=df_new.index)
X_train = df_new['coord_segment_distance'].values.reshape(-1,1)
y_train = df_new['trip_distance'].values


poly_features = PolynomialFeatures(degree=4)

X_train_poly = poly_features.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

def distance_predict(X_test):
    return poly_model.predict(poly_features.fit_transform(X_test))
