import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
import openpyxl

wb = openpyxl.load_workbook('reg1.xlsx')
ws = wb.active

data = []
for row in ws.iter_rows(values_only=True):
    csv_string = row[0]
    values = [v.strip().strip('"') for v in csv_string.split(',')]
    data.append(values)

df = pd.DataFrame(data[1:], columns=data[0])
df = df.iloc[:, 1:]

categorical_cols = []
for col in df.columns:
    test_convert = pd.to_numeric(df[col], errors='coerce')
    if test_convert.isna().sum() > len(df) * 0.5:
        categorical_cols.append(col)

print(f"Categorical columns detected: {categorical_cols}")

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna()

print(f"\nOriginal dataset: {len(df)} rows")
print(f"After dropping NaN: {len(df_clean)} rows")
print(f"\nColumn names: {df_clean.columns.tolist()}")
print(f"Dataset shape: {df_clean.shape}\n")

y = df_clean['y'].values
X = df_clean.drop(['y'], axis=1)

print(f"Response variable: y (n={len(y)})")
print(f"Number of predictors: {X.shape[1]}\n")


def best_subset_selection(X, y):
    feature_names = X.columns.tolist()
    X_array = X.values
    n = len(y)
    n_features = X_array.shape[1]
    best_models = {}

    y_mean = np.mean(y)
    rss_null = np.sum((y - y_mean) ** 2)
    best_models[0] = {
        'features': [],
        'RSS': rss_null,
        'R2': 0,
        'AIC': n * np.log(rss_null / n) + 2,
        'BIC': n * np.log(rss_null / n) + np.log(n)
    }

    print(f"Total number of models to evaluate: {2 ** n_features - 1}")
    print("This may take several minutes...\n")

    for k in range(1, n_features + 1):
        print(f"Evaluating models with {k} predictor(s)...")
        best_rss = np.inf
        best_metrics = None

        for combo in combinations(range(n_features), k):
            X_subset = X_array[:, combo]
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            rss = np.sum((y - y_pred) ** 2)

            if rss < best_rss:
                best_rss = rss
                r2 = 1 - (rss / rss_null)
                aic = n * np.log(rss / n) + 2 * (k + 1)
                bic = n * np.log(rss / n) + (k + 1) * np.log(n)

                best_metrics = {
                    'features': [feature_names[i] for i in combo],
                    'RSS': rss,
                    'R2': r2,
                    'AIC': aic,
                    'BIC': bic
                }

        best_models[k] = best_metrics

    return best_models


def forward_stepwise_selection(X, y):
    feature_names = X.columns.tolist()
    X_array = X.values
    n = len(y)
    n_features = X_array.shape[1]

    y_mean = np.mean(y)
    rss_null = np.sum((y - y_mean) ** 2)

    remaining = list(range(n_features))
    selected = []
    best_models = {0: {
        'features': [],
        'RSS': rss_null,
        'R2': 0,
        'AIC': n * np.log(rss_null / n) + 2,
        'BIC': n * np.log(rss_null / n) + np.log(n)
    }}

    print("Performing forward stepwise selection...\n")

    for i in range(n_features):
        best_rss = np.inf
        best_feature = None
        best_metrics = None

        for feature in remaining:
            candidate = selected + [feature]
            model = LinearRegression()
            model.fit(X_array[:, candidate], y)
            y_pred = model.predict(X_array[:, candidate])
            rss = np.sum((y - y_pred) ** 2)

            if rss < best_rss:
                best_rss = rss
                best_feature = feature

                k = len(candidate)
                r2 = 1 - (rss / rss_null)
                aic = n * np.log(rss / n) + 2 * (k + 1)
                bic = n * np.log(rss / n) + (k + 1) * np.log(n)

                best_metrics = {
                    'features': [feature_names[j] for j in candidate],
                    'RSS': rss,
                    'R2': r2,
                    'AIC': aic,
                    'BIC': bic
                }

        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_models[len(selected)] = best_metrics
            print(f"Step {len(selected)}: Added feature {feature_names[best_feature]}")

    return best_models


def backward_stepwise_selection(X, y):
    feature_names = X.columns.tolist()
    X_array = X.values
    n = len(y)
    n_features = X_array.shape[1]

    y_mean = np.mean(y)
    rss_null = np.sum((y - y_mean) ** 2)

    selected = list(range(n_features))
    best_models = {}

    model = LinearRegression()
    model.fit(X_array, y)
    y_pred = model.predict(X_array)
    rss = np.sum((y - y_pred) ** 2)
    r2 = 1 - (rss / rss_null)
    aic = n * np.log(rss / n) + 2 * (n_features + 1)
    bic = n * np.log(rss / n) + (n_features + 1) * np.log(n)

    best_models[n_features] = {
        'features': feature_names.copy(),
        'RSS': rss,
        'R2': r2,
        'AIC': aic,
        'BIC': bic
    }

    print("Performing backward stepwise selection...\n")

    while len(selected) > 1:
        best_rss = np.inf
        worst_feature = None
        best_metrics = None

        for feature in selected:
            candidate = [f for f in selected if f != feature]
            model = LinearRegression()
            model.fit(X_array[:, candidate], y)
            y_pred = model.predict(X_array[:, candidate])
            rss = np.sum((y - y_pred) ** 2)

            if rss < best_rss:
                best_rss = rss
                worst_feature = feature

                k = len(candidate)
                r2 = 1 - (rss / rss_null)
                aic = n * np.log(rss / n) + 2 * (k + 1)
                bic = n * np.log(rss / n) + (k + 1) * np.log(n)

                best_metrics = {
                    'features': [feature_names[j] for j in candidate],
                    'RSS': rss,
                    'R2': r2,
                    'AIC': aic,
                    'BIC': bic
                }

        selected.remove(worst_feature)
        best_models[len(selected)] = best_metrics
        print(f"Step {n_features - len(selected)}: Removed feature {feature_names[worst_feature]}")

    best_models[0] = {
        'features': [],
        'RSS': rss_null,
        'R2': 0,
        'AIC': n * np.log(rss_null / n) + 2,
        'BIC': n * np.log(rss_null / n) + np.log(n)
    }

    return best_models


def print_results(method_name, results):
    print("\n" + "=" * 80)
    print(f"{method_name.upper()} RESULTS:")
    print("=" * 80)

    results_list = []
    for k in sorted(results.keys()):
        metrics = results[k]
        results_list.append({
            'n_predictors': k,
            'predictors': ', '.join(metrics['features']) if metrics['features'] else 'None (Intercept only)',
            'RSS': metrics['RSS'],
            'R2': metrics['R2'],
            'AIC': metrics['AIC'],
            'BIC': metrics['BIC']
        })

    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("MODEL SELECTION CRITERIA:")
    print("=" * 80)

    best_r2 = results_df.loc[results_df['R2'].idxmax()]
    print(f"\nBest by R²: {best_r2['n_predictors']} predictors")
    print(f"  R² = {best_r2['R2']:.6f}")
    print(f"  Predictors: {best_r2['predictors']}")

    best_aic = results_df.loc[results_df['AIC'].idxmin()]
    print(f"\nBest by AIC: {best_aic['n_predictors']} predictors")
    print(f"  AIC = {best_aic['AIC']:.2f}")
    print(f"  Predictors: {best_aic['predictors']}")

    best_bic = results_df.loc[results_df['BIC'].idxmin()]
    print(f"\nBest by BIC: {best_bic['n_predictors']} predictors")
    print(f"  BIC = {best_bic['BIC']:.2f}")
    print(f"  Predictors: {best_bic['predictors']}")

    filename = f"{method_name.lower().replace(' ', '_')}_results.xlsx"
    results_df.to_excel(filename, index=False)
    print(f"\nResults saved to '{filename}'")


print("\n" + "=" * 80)
print("SUBSET SELECTION METHODS")
print("=" * 80)

best_subset_results = best_subset_selection(X, y)
print_results("Best Subset Selection", best_subset_results)

forward_results = forward_stepwise_selection(X, y)
print_results("Forward Stepwise Selection", forward_results)

backward_results = backward_stepwise_selection(X, y)
print_results("Backward Stepwise Selection", backward_results)