import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


REGRESSION_FEATURES = [
    "Tmp_SummerMean", "Tmp_WinterMean",
    "Pre_SummerMean", "Pre_WinterMean",
    "Tswrf_SummerMean", "Tswrf_WinterMean",
    "Lon", "Lat",
]


def run_regression(df_full, target_col):
    train_df = df_full[df_full["Continent"] == "Europe"].copy()
    test_df = df_full[df_full["Continent"] == "North America"].copy()

    X_train = train_df[REGRESSION_FEATURES]
    y_train = train_df[target_col]

    X_test = test_df[REGRESSION_FEATURES]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRegression results for {target_col}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R2   = {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=10, alpha=0.5)
    plt.xlabel(f"Observed {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Observed vs Predicted {target_col}")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        color="black"
    )
    plt.tight_layout()
    plt.savefig(f"figures/{target_col.lower()}_observed_vs_predicted.png", dpi = 300)
    plt.show()

    importances = pd.Series(
        model.feature_importances_,
        index=REGRESSION_FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"Feature Importances - {target_col}")
    plt.tight_layout()
    plt.savefig(f"figures/{target_col.lower()}_feature_importance.png", dpi = 300)
    plt.show()

    plot_df = test_df.copy()
    plot_df["Prediction"] = y_pred
    plot_df["Absolute_error"] = np.abs(y_test - y_pred)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        plot_df["Lon"],
        plot_df["Lat"],
        c=plot_df["Absolute_error"],
        s=10
    )
    plt.colorbar(label="Absolute error")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Geographic Prediction Error - {target_col}")
    plt.tight_layout()
    plt.savefig(f"figures/{target_col.lower()}_geographic_error.png", dpi = 300)
    plt.show()

    return {
        f"{target_col}_mse": mse,
        f"{target_col}_rmse": rmse,
        f"{target_col}_mae": mae,
        f"{target_col}_r2": r2,
    }