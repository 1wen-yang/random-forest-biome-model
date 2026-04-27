import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV


BINARY_FEATURES = [
    "Lon", "Lat",
    "Tmp_SummerMean", "Tmp_WinterMean",
    "Pre_SummerMean", "Pre_WinterMean",
    "Tswrf_SummerMean", "Tswrf_WinterMean",
]


MULTICLASS_FEATURES = [
    "Tmp_SummerMean", "Tmp_WinterMean",
    "Pre_SummerMean", "Pre_WinterMean",
    "Tswrf_SummerMean", "Tswrf_WinterMean",
    "Lon", "Lat",
    "NPP", "SoilR", "MaxBiomeCmax", "MaxBiomeLAI",
    "VegC", "LitterC", "SoilC",
]


def run_binary_classification(df_full):
    df_full = df_full.copy()
    df_full["y"] = (df_full["Biome_obs"] == 2).astype(int)

    train_df = df_full[df_full["CountryCode"] == "CHN"].copy()
    test_df = df_full[df_full["CountryCode"] == "USA"].copy()

    X_train = train_df[BINARY_FEATURES]
    y_train = train_df["y"]

    X_test = test_df[BINARY_FEATURES]
    y_test = test_df["y"]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nBinary classification")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    test_df["Prediction"] = y_pred

    correct = test_df[test_df["y"] == test_df["Prediction"]]
    wrong = test_df[test_df["y"] != test_df["Prediction"]]

    plt.figure(figsize=(9, 6))
    plt.scatter(correct["Lon"], correct["Lat"], color="green", s=10, alpha=0.7, label="Correct prediction")
    plt.scatter(wrong["Lon"], wrong["Lat"], color="red", s=10, alpha=0.8, label="Misclassified")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Binary Classification Result")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/binary_geographic_predictions.png", dpi = 300)
    plt.show()


def run_multiclass_classification(df_full):
    train_df = df_full[df_full["Continent"] == "Europe"].copy()
    test_df = df_full[df_full["Continent"] == "North America"].copy()

    X_train = train_df[MULTICLASS_FEATURES]
    y_train = train_df["Biome_obs"]

    X_test = test_df[MULTICLASS_FEATURES]
    y_test = test_df["Biome_obs"]

    print("\nMulticlass training size:", X_train.shape)
    print("Multiclass testing size:", X_test.shape)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 5],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nMulticlass classification")
    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", round(acc, 3))
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix - Multiclass Classification")
    plt.xlabel("Predicted biome")
    plt.ylabel("True biome")
    plt.tight_layout()
    plt.savefig("figures/multiclass_confusion_matrix.png", dpi = 300)
    plt.show()

    importances = pd.Series(
        best_model.feature_importances_,
        index=MULTICLASS_FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances - Multiclass Model")
    plt.tight_layout()
    plt.savefig("figures/multiclass_feature_importance.png", dpi = 300)
    plt.show()

    test_df["Prediction"] = y_pred
    correct_mask = test_df["Biome_obs"] == test_df["Prediction"]

    plt.figure(figsize=(9, 6))
    plt.scatter(
        test_df.loc[correct_mask, "Lon"],
        test_df.loc[correct_mask, "Lat"],
        color="green",
        s=10,
        alpha=0.7,
        label="Correct prediction"
    )
    plt.scatter(
        test_df.loc[~correct_mask, "Lon"],
        test_df.loc[~correct_mask, "Lat"],
        color="red",
        s=10,
        alpha=0.8,
        label="Misclassified"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Distribution of Multiclass Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/multiclass_geographic_errors.png", dpi = 300)
    plt.show()

    top5_features = importances.head(5).index.tolist()

    top5_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    top5_model.fit(train_df[top5_features], y_train)
    top5_pred = top5_model.predict(test_df[top5_features])
    top5_acc = accuracy_score(y_test, top5_pred)

    print("Full feature accuracy:", round(acc, 3))
    print("Top 5 feature accuracy:", round(top5_acc, 3))

    return {
        "multiclass_accuracy": acc,
        "top5_accuracy": top5_acc,
        "best_params": grid_search.best_params_,
    }