import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def Law(norm=False):
    df = pd.read_csv("data/law_data.csv", index_col=0)
    df = pd.get_dummies(df, columns=["race"], prefix="", prefix_sep="")

    df["male"] = df["sex"].map(lambda x: 1 if x == 2 else 0)
    df["female"] = df["sex"].map(lambda x: 1 if x == 1 else 0)
    df = df.drop(axis=1, columns=["sex"])
    df["LSAT"] = df["LSAT"].astype(int)

    df_train, df_test = train_test_split(df, random_state=123, test_size=0.2)
    A = [
        "Amerindian",
        "Asian",
        "Black",
        "Hispanic",
        "Mexican",
        "Other",
        "Puertorican",
        "White",
       
    ]
    X_train = np.hstack(
        (
            df_train[A],
            np.array(df_train["UGPA"]).reshape(-1, 1),
            np.array(df_train["LSAT"]).reshape(-1, 1),
            np.array(df_train["male"]).reshape(-1, 1),
            np.array(df_train["female"]).reshape(-1, 1),
        ))

    y_train = df_train["ZFYA"]
    y_train = pd.Series.to_numpy(y_train)
    X_test = np.hstack(
        (
            df_test[A],
            np.array(df_test["UGPA"]).reshape(-1, 1),
            np.array(df_test["LSAT"]).reshape(-1, 1),
            np.array(df_test["male"]).reshape(-1, 1),
            np.array(df_test["female"]).reshape(-1, 1),
        ))
    norm_fac = []
    y_test = df_test["ZFYA"]
    y_test = pd.Series.to_numpy(y_test)
    norm_y = max(abs(y_train))
    y_train = y_train / max(abs(y_train))
    y_test = y_test / norm_y
    norm_fac.append(norm_y)
    if norm:
        X_mean = np.mean(X_train[:, 8:10], axis=0)
        norm_fac.append(X_mean)
        X_train[:, 8:10] = (X_train[:, 8:10] - X_mean)
        fac = np.abs(X_train[:, 8:10]).max(axis=0)
        norm_fac.append(fac)
        X_train[:, 8:10] = X_train[:, 8:10] / fac
        X_test[:, 8:10] = (X_test[:, 8:10] - X_mean)
        X_test[:, 8:10] = X_test[:, 8:10] / fac

    return X_train, y_train, X_test, y_test, df_train, A, norm_fac
