import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv('reg_6_complete.csv')
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

    filename = f"{method_name.lower().replace(' ', '_')}_results.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to '{filename}'")


def calculate_cv_mse_curve(X, y, results, method_name, n_folds=10):
    cv_mse_values = []
    n_predictors_list = []

    print(f"\nCalculating CV MSE for {method_name}...")

    for k in sorted(results.keys()):
        features = results[k]['features']

        if not features:
            continue

        X_subset = X[features].values
        model = LinearRegression()

        cv_scores = cross_val_score(model, X_subset, y, cv=n_folds,
                                    scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        cv_std = cv_scores.std()

        cv_mse_values.append(cv_mse)
        n_predictors_list.append(k)

        print(f"  {k} predictors: CV MSE = {cv_mse:.2f} ± {cv_std:.2f}")

    return n_predictors_list, cv_mse_values


def plot_cv_mse_curves(best_subset_data, forward_data, backward_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax1 = axes[0, 0]
    ax1.plot(best_subset_data[0], best_subset_data[1], 'o-',
             label='Best Subset', linewidth=2, markersize=6)
    ax1.plot(forward_data[0], forward_data[1], 's-',
             label='Forward Stepwise', linewidth=2, markersize=6)
    ax1.plot(backward_data[0], backward_data[1], '^-',
             label='Backward Stepwise', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Predictors', fontsize=12)
    ax1.set_ylabel('Cross-Validation MSE', fontsize=12)
    ax1.set_title('CV MSE Comparison: All Methods', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    for data, marker, color in [(best_subset_data, 'o', 'C0'),
                                (forward_data, 's', 'C1'),
                                (backward_data, '^', 'C2')]:
        min_idx = np.argmin(data[1])
        ax1.plot(data[0][min_idx], data[1][min_idx], marker,
                 markersize=12, markerfacecolor='none',
                 markeredgewidth=2, markeredgecolor=color)

    ax2 = axes[0, 1]
    ax2.plot(best_subset_data[0], best_subset_data[1], 'o-',
             linewidth=2, markersize=8, color='C0')
    min_idx = np.argmin(best_subset_data[1])
    ax2.axvline(x=best_subset_data[0][min_idx], color='red',
                linestyle='--', alpha=0.7, label=f'Min at {best_subset_data[0][min_idx]} predictors')
    ax2.plot(best_subset_data[0][min_idx], best_subset_data[1][min_idx],
             'r*', markersize=15, label=f'Min MSE = {best_subset_data[1][min_idx]:.2f}')
    ax2.set_xlabel('Number of Predictors', fontsize=12)
    ax2.set_ylabel('Cross-Validation MSE', fontsize=12)
    ax2.set_title('Best Subset Selection', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(forward_data[0], forward_data[1], 's-',
             linewidth=2, markersize=8, color='C1')
    min_idx = np.argmin(forward_data[1])
    ax3.axvline(x=forward_data[0][min_idx], color='red',
                linestyle='--', alpha=0.7, label=f'Min at {forward_data[0][min_idx]} predictors')
    ax3.plot(forward_data[0][min_idx], forward_data[1][min_idx],
             'r*', markersize=15, label=f'Min MSE = {forward_data[1][min_idx]:.2f}')
    ax3.set_xlabel('Number of Predictors', fontsize=12)
    ax3.set_ylabel('Cross-Validation MSE', fontsize=12)
    ax3.set_title('Forward Stepwise Selection', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(backward_data[0], backward_data[1], '^-',
             linewidth=2, markersize=8, color='C2')
    min_idx = np.argmin(backward_data[1])
    ax4.axvline(x=backward_data[0][min_idx], color='red',
                linestyle='--', alpha=0.7, label=f'Min at {backward_data[0][min_idx]} predictors')
    ax4.plot(backward_data[0][min_idx], backward_data[1][min_idx],
             'r*', markersize=15, label=f'Min MSE = {backward_data[1][min_idx]:.2f}')
    ax4.set_xlabel('Number of Predictors', fontsize=12)
    ax4.set_ylabel('Cross-Validation MSE', fontsize=12)
    ax4.set_title('Backward Stepwise Selection', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cv_mse_curves.png', dpi=300, bbox_inches='tight')
    print("\nCV MSE curves saved to 'cv_mse_curves.png'")
    plt.show()


def create_cv_summary_table(best_subset_data, forward_data, backward_data):
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION MSE SUMMARY")
    print("=" * 80)

    summary_data = []

    for method_name, data in [('Best Subset', best_subset_data),
                              ('Forward Stepwise', forward_data),
                              ('Backward Stepwise', backward_data)]:
        min_idx = np.argmin(data[1])
        min_mse = data[1][min_idx]
        optimal_k = data[0][min_idx]

        summary_data.append({
            'Method': method_name,
            'Optimal k': optimal_k,
            'Min CV MSE': min_mse,
            'CV RMSE': np.sqrt(min_mse)
        })

        print(f"\n{method_name}:")
        print(f"  Optimal number of predictors: {optimal_k}")
        print(f"  Minimum CV MSE: {min_mse:.4f}")
        print(f"  CV RMSE: {np.sqrt(min_mse):.4f}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('cv_mse_summary.xlsx', index=False)
    print("\nCV MSE summary saved to 'cv_mse_summary.xlsx'")

    return summary_df


def evaluate_optimal_models(X, y, best_subset_results, forward_results, backward_results,
                            best_subset_data, forward_data, backward_data):
    print("\n" + "=" * 80)
    print("OPTIMAL MODEL EVALUATION (Selected by CV MSE)")
    print("=" * 80)

    detailed_results = []

    for method_name, results, data in [('Best Subset', best_subset_results, best_subset_data),
                                       ('Forward Stepwise', forward_results, forward_data),
                                       ('Backward Stepwise', backward_results, backward_data)]:
        min_idx = np.argmin(data[1])
        optimal_k = data[0][min_idx]
        features = results[optimal_k]['features']

        print(f"\n{method_name} - Optimal Model ({optimal_k} predictors):")
        print(f"Features: {', '.join(features)}")

        X_subset = X[features].values
        model = LinearRegression()

        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        cv_scores = cross_val_score(model, X_subset, y, cv=10,
                                    scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        cv_rmse = np.sqrt(cv_mse)

        detailed_results.append({
            'Method': method_name,
            'k': optimal_k,
            'Training MSE': mse,
            'Training RMSE': rmse,
            'Training MAE': mae,
            'Training R²': r2,
            'CV MSE': cv_mse,
            'CV RMSE': cv_rmse
        })

        print(f"  Training MSE: {mse:.4f}")
        print(f"  Training RMSE: {rmse:.4f}")
        print(f"  Training MAE: {mae:.4f}")
        print(f"  Training R²: {r2:.4f}")
        print(f"  CV MSE: {cv_mse:.4f}")
        print(f"  CV RMSE: {cv_rmse:.4f}")

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_excel('optimal_models_detailed.xlsx', index=False)
    print("\nDetailed results saved to 'optimal_models_detailed.xlsx'")

    return detailed_df


print("\n" + "=" * 80)
print("CROSS-VALIDATION MSE ANALYSIS")
print("=" * 80)


print("\n" + "=" * 80)
print("SUBSET SELECTION METHODS")
print("=" * 80)

best_subset_results = best_subset_selection(X, y)
print_results("Best Subset Selection", best_subset_results)

forward_results = forward_stepwise_selection(X, y)
print_results("Forward Stepwise Selection", forward_results)

backward_results = backward_stepwise_selection(X, y)
print_results("Backward Stepwise Selection", backward_results)

best_subset_cv_data = calculate_cv_mse_curve(X, y, best_subset_results, "Best Subset Selection")
forward_cv_data = calculate_cv_mse_curve(X, y, forward_results, "Forward Stepwise Selection")
backward_cv_data = calculate_cv_mse_curve(X, y, backward_results, "Backward Stepwise Selection")

plot_cv_mse_curves(best_subset_cv_data, forward_cv_data, backward_cv_data)
cv_summary = create_cv_summary_table(best_subset_cv_data, forward_cv_data, backward_cv_data)
detailed_evaluation = evaluate_optimal_models(X, y, best_subset_results, forward_results,
                                              backward_results, best_subset_cv_data,
                                              forward_cv_data, backward_cv_data)