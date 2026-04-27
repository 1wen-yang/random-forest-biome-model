from src.data_loader import (
    load_biome_legend,
    load_climate_data,
    load_full_dataset,
)

from src.visualization import (
    plot_global_biomes,
    plot_biome_counts,
    plot_climate_correlations,
)

from src.classification import (
    run_binary_classification,
    run_multiclass_classification,
)

from src.regression import run_regression


def save_metrics(metrics, path="results/model_metrics.txt"):
    with open(path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    biome_names, biome_rgbs = load_biome_legend()
    climate_df = load_climate_data()
    df, df_full = load_full_dataset()

    plot_global_biomes(df, biome_names, biome_rgbs)
    plot_biome_counts(df)
    plot_climate_correlations(climate_df)

    run_binary_classification(df_full)
    classification_metrics = run_multiclass_classification(df_full)

    npp_metrics = run_regression(df_full, "NPP")
    vegc_metrics = run_regression(df_full, "VegC")

    metrics = {}
    metrics.update(classification_metrics)
    metrics.update(npp_metrics)
    metrics.update(vegc_metrics)

    save_metrics(metrics)


if __name__ == "__main__":
    main()