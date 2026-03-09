import pandas as pd
from sklearn.preprocessing import RobustScaler


def load_and_preprocess_data(filepath="data/creditcard.csv"):
    """
    Loads data, scales features robustly, and splits chronologically.
    """

    df = pd.read_csv(filepath)
    df = df.sort_values('Time')

    # We use RobustScaler because transaction amounts have extreme outliers
    scaler = RobustScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop the original unscaled columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Chronological Split (70% Train, 15% Validation, 15% Test)
    train_idx = int(len(df) * 0.7)
    val_idx = int(len(df) * 0.85)

    train = df.iloc[:train_idx]
    val = df.iloc[train_idx:val_idx]
    test = df.iloc[val_idx:]

    # Separate features (X) and target labels (y)
    X_train, y_train = train.drop('Class', axis=1), train['Class']
    X_val, y_val = val.drop('Class', axis=1), val['Class']
    X_test, y_test = test.drop('Class', axis=1), test['Class']

    return X_train, y_train, X_val, y_val, X_test, y_test
