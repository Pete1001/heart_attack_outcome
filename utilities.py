from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

def createNN(input_nodes):
    nn_model = tf.keras.models.Sequential()
    # Set the input nodes to the number of features
    nn_model.add(tf.keras.layers.Dense(units=10, activation="relu", input_dim=input_nodes))
    nn_model.add(tf.keras.layers.Dense(units=10, activation="relu"))
    nn_model.add(tf.keras.layers.Dense(units=10, activation="relu"))
    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    return nn_model

def trainNN(model, X_train, y_train, epochs):
    # Compile the model and train over more than ### epochs
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    fit_model = model.fit(X_train, y_train, epochs= epochs)

#     nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)
    return fit_model

def splitData(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

def scaleData(X_train, X_test, numerical_columns):
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_columns])
    X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    return scaler, X_train, X_test

def getResults(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def get_feature_importance(model, X_test, y_test):
    def score(model, X_test, y_test):
        y_pred = model.predict(X_test, verbose=0)
        return mean_squared_error(y_test, y_pred)

    # Calculate permutation importance
    results = permutation_importance(model, X_test, y_test, scoring=lambda model, X, y: -score(model, X, y), n_repeats=10, random_state=42)

    importances = []
    # Print feature importance scores
    for i, importance in enumerate(results.importances_mean):
        importances.append({'feature': X_test.columns[i], 'importance' : importance})
    importances.sort(key= lambda x: abs(x['importance']), reverse=True)
    for i in importances:
        print(f"{i['feature']} : {i['importance']}")
    return importances

def encodeOrdinal(df, ordinal_data, ordinal_categories):
    for i in range(len(ordinal_data)):
        encoder = OrdinalEncoder(categories=[ordinal_categories[i]])
        df[ordinal_data[i]] = encoder.fit_transform(df[[ordinal_data[i]]])
    return df

def encodeBinary(df, binary_data):
    # Label encode Binary_Data:
    label_encoder = LabelEncoder()
    for col in binary_data:
        df[col] = label_encoder.fit_transform(df[col])
    return df, label_encoder

def encodeNominal(df, nominal_data):
    if len(nominal_data) == 0:
        return df
    df= pd.get_dummies(df, columns= nominal_data, drop_first=True)
    return df

def cleanData(df, target_column, binary_data = [], ordinal_data = [], nominal_data = [], ordinal_categories = []):
    print('Cleaning Data...')
    numerical_columns = df.select_dtypes(include=[np.number]).drop(columns = target_column).columns.tolist()
    df = encodeOrdinal(df, ordinal_data, ordinal_categories)
    df, LabelEncoder = encodeBinary(df, binary_data)
    df = encodeNominal(df, nominal_data)
    X_train, X_test, y_train, y_test = splitData(df, target_column)
    scaler, X_train_scaled, X_test_scaled = scaleData(X_train,X_test, numerical_columns)
    return scaler, X_train_scaled, X_test_scaled, y_train, y_test, LabelEncoder

def makePrediction(x, nn_model, scaler, numerical_columns):
    x[numerical_columns] = scaler.transform(x[numerical_columns])
    return nn_model.predict(x)[0][0]

def trainNeuralNetwork(X_train, y_train, X_test, y_test, epochs):
    print('Making Neural Network...')
    nn_model = createNN(len(X_train.columns))
    trainNN(nn_model, X_train, y_train, epochs)
    getResults(nn_model, X_test, y_test)
    return nn_model
