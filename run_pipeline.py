import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Step 1 & 2: Loading California Housing Dataset...", flush=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    california = fetch_california_housing(as_frame=True)
    df = pd.concat([california.data, california.target.rename("HousePrice")], axis=1)

    X = df.drop("HousePrice", axis=1)
    y = df["HousePrice"]

    print(f"Dataset shape: {df.shape}", flush=True)
    print(f"Features: {list(X.columns)}", flush=True)

    # Step 3: Feature Scaling
    print("\nStep 3: Feature Scaling with StandardScaler...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Train-Test Split
    print("\nStep 4: Train-Test Split (80/20, random_state=42)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}", flush=True)

    # Baseline 1: Linear Regression
    print("\n--- Baseline 1: Linear Regression ---", flush=True)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_train_pred = lr.predict(X_train)
    lr_test_pred = lr.predict(X_test)
    lr_train_rmse = float(np.sqrt(mean_squared_error(y_train, lr_train_pred)))
    lr_test_rmse = float(np.sqrt(mean_squared_error(y_test, lr_test_pred)))
    lr_train_r2 = float(r2_score(y_train, lr_train_pred))
    lr_test_r2 = float(r2_score(y_test, lr_test_pred))
    lr_cv_scores = cross_val_score(lr, X_scaled, y, scoring="neg_root_mean_squared_error", cv=5)
    lr_cv_rmse = float(-lr_cv_scores.mean())
    lr_cv_std = float(lr_cv_scores.std())

    print(f"Linear Regression - Train RMSE: {lr_train_rmse:.4f}, Test RMSE: {lr_test_rmse:.4f}, Test R2: {lr_test_r2:.4f}, CV RMSE: {lr_cv_rmse:.4f} (+/- {lr_cv_std:.4f})", flush=True)

    # Baseline 2: Ridge Regression
    print("\n--- Baseline 2: Ridge Regression (Tuning alpha) ---", flush=True)
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring="neg_root_mean_squared_error", cv=5)
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    ridge_train_pred = best_ridge.predict(X_train)
    ridge_test_pred = best_ridge.predict(X_test)
    ridge_train_rmse = float(np.sqrt(mean_squared_error(y_train, ridge_train_pred)))
    ridge_test_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_test_pred)))
    ridge_train_r2 = float(r2_score(y_train, ridge_train_pred))
    ridge_test_r2 = float(r2_score(y_test, ridge_test_pred))
    ridge_cv_scores = cross_val_score(best_ridge, X_scaled, y, scoring="neg_root_mean_squared_error", cv=5)
    ridge_cv_rmse = float(-ridge_cv_scores.mean())
    ridge_cv_std = float(ridge_cv_scores.std())

    print(f"Ridge Best Alpha: {ridge_grid.best_params_['alpha']}", flush=True)
    print(f"Ridge Regression - Train RMSE: {ridge_train_rmse:.4f}, Test RMSE: {ridge_test_rmse:.4f}, Test R2: {ridge_test_r2:.4f}, CV RMSE: {ridge_cv_rmse:.4f} (+/- {ridge_cv_std:.4f})", flush=True)

    # Step 5: Detect Overfitting with Unconstrained Decision Tree
    print("\n--- Step 5: Detect Overfitting (Unconstrained Decision Tree) ---", flush=True)
    tree_unconstrained = DecisionTreeRegressor(random_state=42)
    tree_unconstrained.fit(X_train, y_train)

    tree_train_pred = tree_unconstrained.predict(X_train)
    tree_test_pred = tree_unconstrained.predict(X_test)

    tree_train_rmse = float(np.sqrt(mean_squared_error(y_train, tree_train_pred)))
    tree_test_rmse = float(np.sqrt(mean_squared_error(y_test, tree_test_pred)))
    tree_train_r2 = float(r2_score(y_train, tree_train_pred))
    tree_test_r2 = float(r2_score(y_test, tree_test_pred))

    print(f"Unconstrained Tree - Train RMSE: {tree_train_rmse:.6f}, Test RMSE: {tree_test_rmse:.4f}", flush=True)
    print(f"Unconstrained Tree - Train R2: {tree_train_r2:.4f}, Test R2: {tree_test_r2:.4f}", flush=True)
    print(f"Train/Test RMSE Gap: {tree_test_rmse - tree_train_rmse:.4f} (Severe Overfitting / High Variance)", flush=True)
    print(f"Tree Max Depth: {tree_unconstrained.get_depth()}, Leaf Nodes: {tree_unconstrained.get_n_leaves()}", flush=True)

    # Step 6: Cross-Validation on Unconstrained Tree
    print("\n--- Step 6: Cross-Validation (5-Fold) ---", flush=True)
    cv_scores = cross_val_score(tree_unconstrained, X_scaled, y, scoring="neg_root_mean_squared_error", cv=5)
    cv_rmse = float(-cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"5-Fold CV Scores (RMSE): {[-round(s, 4) for s in cv_scores]}", flush=True)
    print(f"Mean CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})", flush=True)

    # Step 7: Hyperparameter Tuning Using GridSearchCV
    print("\n--- Step 7: Hyperparameter Tuning Using GridSearchCV ---", flush=True)
    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5
    )
    grid.fit(X_train, y_train)

    print(f"Best Hyperparameters: {grid.best_params_}", flush=True)
    print(f"Best CV Mean RMSE: {-grid.best_score_:.4f}", flush=True)

    # Step 8: Evaluate Optimized Model
    print("\n--- Step 8: Evaluate Optimized Decision Tree ---", flush=True)
    best_tree = grid.best_estimator_
    best_tree_train_pred = best_tree.predict(X_train)
    best_tree_test_pred = best_tree.predict(X_test)

    best_tree_train_rmse = float(np.sqrt(mean_squared_error(y_train, best_tree_train_pred)))
    best_tree_test_rmse = float(np.sqrt(mean_squared_error(y_test, best_tree_test_pred)))
    best_tree_train_r2 = float(r2_score(y_train, best_tree_train_pred))
    best_tree_test_r2 = float(r2_score(y_test, best_tree_test_pred))

    best_tree_cv_scores = cross_val_score(best_tree, X_scaled, y, scoring="neg_root_mean_squared_error", cv=5)
    best_tree_cv_rmse = float(-best_tree_cv_scores.mean())
    best_tree_cv_std = float(best_tree_cv_scores.std())

    print(f"Tuned Tree - Train RMSE: {best_tree_train_rmse:.4f}, Test RMSE: {best_tree_test_rmse:.4f}", flush=True)
    print(f"Tuned Tree - Train R2: {best_tree_train_r2:.4f}, Test R2: {best_tree_test_r2:.4f}", flush=True)
    print(f"Tuned Tree - 5-Fold CV RMSE: {best_tree_cv_rmse:.4f} (+/- {best_tree_cv_std:.4f})", flush=True)
    print(f"Tuned Tree Depth: {best_tree.get_depth()}, Leaf Nodes: {best_tree.get_n_leaves()}", flush=True)

    # Step 9: Model Comparison Summary Table
    print("\n--- Step 9: Model Comparison Summary Table ---", flush=True)
    comparison_df = pd.DataFrame({
        "Model": [
            "Linear Regression (Baseline)",
            "Ridge Regression (Task-2)",
            "Decision Tree (Unconstrained)",
            "Decision Tree (Tuned GridSearchCV)"
        ],
        "Train RMSE": [
            round(lr_train_rmse, 4),
            round(ridge_train_rmse, 4),
            round(tree_train_rmse, 4),
            round(best_tree_train_rmse, 4)
        ],
        "Test RMSE": [
            round(lr_test_rmse, 4),
            round(ridge_test_rmse, 4),
            round(tree_test_rmse, 4),
            round(best_tree_test_rmse, 4)
        ],
        "RMSE Gap (|Test - Train|)": [
            round(abs(lr_test_rmse - lr_train_rmse), 4),
            round(abs(ridge_test_rmse - ridge_train_rmse), 4),
            round(abs(tree_test_rmse - tree_train_rmse), 4),
            round(abs(best_tree_test_rmse - best_tree_train_rmse), 4)
        ],
        "5-Fold CV Mean RMSE": [
            round(lr_cv_rmse, 4),
            round(ridge_cv_rmse, 4),
            round(cv_rmse, 4),
            round(best_tree_cv_rmse, 4)
        ],
        "CV RMSE Std Dev": [
            round(lr_cv_std, 4),
            round(ridge_cv_std, 4),
            round(cv_std, 4),
            round(best_tree_cv_std, 4)
        ],
        "Test R2 Score": [
            round(lr_test_r2, 4),
            round(ridge_test_r2, 4),
            round(tree_test_r2, 4),
            round(best_tree_test_r2, 4)
        ],
        "Generalization Diagnosis": [
            "High Bias (Linear Underfitting)",
            "High Bias (Linear Underfitting)",
            "High Variance (Severe Overfitting)",
            "Well-Generalized (Optimal Trade-off)"
        ]
    })

    print(comparison_df.to_string(index=False), flush=True)
    comparison_df.to_csv("model_comparison_results.csv", index=False)

    # Save Models and Scaler using joblib
    print("\nSaving Models and Preprocessors...", flush=True)
    joblib.dump(best_tree, "models/best_tuned_decision_tree.joblib")
    joblib.dump(scaler, "models/california_housing_scaler.joblib")
    joblib.dump(best_ridge, "models/best_ridge_model.joblib")
    joblib.dump(lr, "models/linear_regression_baseline.joblib")
    joblib.dump(tree_unconstrained, "models/unconstrained_tree.joblib")

    # Generate Publication-Quality Plots
    print("\nGenerating Visualizations...", flush=True)

    # 1. Train vs Test RMSE Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    models = comparison_df["Model"]
    x_pos = np.arange(len(models))
    width = 0.35

    plt.bar(x_pos - width/2, comparison_df["Train RMSE"], width, label='Train RMSE', color='#2563eb', alpha=0.9, edgecolor='#1d4ed8')
    plt.bar(x_pos + width/2, comparison_df["Test RMSE"], width, label='Test RMSE', color='#ef4444', alpha=0.9, edgecolor='#b91c1c')

    plt.xlabel('Model Architecture', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=11, fontweight='bold')
    plt.title('Train vs Test RMSE: Overfitting Detection & Regularization Control', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x_pos, [m.replace(" (", "\n(") for m in models], fontsize=9)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cbd5e1')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(models)):
        plt.text(x_pos[i] - width/2, comparison_df["Train RMSE"][i] + 0.02, f'{comparison_df["Train RMSE"][i]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(x_pos[i] + width/2, comparison_df["Test RMSE"][i] + 0.02, f'{comparison_df["Test RMSE"][i]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig1_train_vs_test_rmse.png")
    plt.close()

    # 2. Validation Curve for max_depth
    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    train_scores_depth = []
    test_scores_depth = []

    for d in depths:
        clf = DecisionTreeRegressor(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, clf.predict(X_train)))
        te_rmse = np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))
        train_scores_depth.append(tr_rmse)
        test_scores_depth.append(te_rmse)

    plt.figure(figsize=(10, 4.8), dpi=300)
    plt.plot(depths, train_scores_depth, 'o-', color='#2563eb', linewidth=2.5, label='Training RMSE')
    plt.plot(depths, test_scores_depth, 's--', color='#dc2626', linewidth=2.5, label='Test RMSE')
    best_d = grid.best_params_['max_depth']
    plt.axvline(x=best_d, color='#16a34a', linestyle=':', linewidth=2, label=f"Selected max_depth = {best_d}")

    plt.ylim(0.0, 1.08)
    plt.annotate('High Bias\n(Underfitting)', xy=(2.5, 0.85), xytext=(2.5, 0.96),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=1.5),
                 ha='center', fontsize=8.5, fontweight='bold')
    plt.annotate('Optimal Balance\n(Best Generalization)', xy=(best_d, test_scores_depth[depths.index(best_d)]), 
                 xytext=(best_d - 1.5, 0.40),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#16a34a", lw=1.5),
                 arrowprops=dict(facecolor='#16a34a', shrink=0.08, width=1.2, headwidth=5),
                 ha='center', fontsize=8.5, fontweight='bold')
    plt.annotate('High Variance\n(Severe Overfitting)', xy=(16, 0.72), xytext=(15, 0.88),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#ef4444", lw=1.5),
                 ha='center', fontsize=8.5, fontweight='bold')

    plt.title('Validation Curve: Effect of Maximum Tree Depth on Generalization Error', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Tree Maximum Depth (max_depth)', fontsize=10.5, fontweight='bold')
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=10.5, fontweight='bold')
    plt.xticks(depths)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='center right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/fig2_validation_curve_depth.png")
    plt.close()

    # 3. 5-Fold Cross-Validation Boxplot
    plt.figure(figsize=(9, 5), dpi=300)
    cv_data = [
        -lr_cv_scores,
        -ridge_cv_scores,
        -cv_scores,
        -best_tree_cv_scores
    ]
    labels = ["Linear Reg", "Ridge Reg", "Unconstrained Tree", "Tuned Tree"]

    box = plt.boxplot(cv_data, labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    colors = ['#bfdbfe', '#bbf7d0', '#fecaca', '#86efac']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#334155')

    plt.title('5-Fold Cross-Validation RMSE Distribution Across Models', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Fold RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/fig3_cv_score_distribution.png")
    plt.close()

    # 4. Actual vs Predicted & Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Actual vs Predicted
    axes[0].scatter(y_test, best_tree_test_pred, alpha=0.25, color='#0284c7', edgecolors='none', s=25)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[0].set_title(f'Tuned Tree: Actual vs Predicted ($R^2$ = {best_tree_test_r2:.3f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual House Price ($100k)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Predicted House Price ($100k)', fontsize=10, fontweight='bold')
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Feature Importances
    feat_importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=True)
    axes[1].barh(feat_importances.index, feat_importances.values, color='#0d9488', edgecolor='#134e4a', alpha=0.85)
    axes[1].set_title('Feature Importance in Tuned Decision Tree', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Relative Importance (Variance Reduction)', fontsize=10, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    for i, v in enumerate(feat_importances.values):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig4_actual_vs_pred_and_importance.png")
    plt.close()

    print("All tasks completed successfully!", flush=True)

if __name__ == '__main__':
    main()
