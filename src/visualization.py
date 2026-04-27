import matplotlib.pyplot as plt
import seaborn as sns


def save_or_show(filename):
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", bbox_inches="tight")
    plt.show()


def plot_global_biomes(df, biome_names, biome_rgbs):
    biome_data = df[["Lon", "Lat", "Biome_obs"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(biome_names):
        biome = biome_data[biome_data["Biome_obs"] == i + 1]
        ax.scatter(
            biome["Lon"],
            biome["Lat"],
            c=[biome_rgbs[i]],
            s=1,
            label=name
        )

    ax.legend(markerscale=5, loc="center right", bbox_to_anchor=(1.3, 0.5))
    ax.set_title("Global Biome Distribution")

    save_or_show("biome_distribution.png")


def plot_biome_counts(df):
    biome_counts = df["Biome_obs"].value_counts()

    print("=== Biome_obs distribution ===")
    print(biome_counts)

    biome_counts.plot(kind="bar", figsize=(12, 5))
    plt.title("Biome Class Distribution")
    plt.xlabel("Biome class")
    plt.ylabel("Count")

    save_or_show("biome_class_counts.png")


def plot_climate_correlations(climate_df):
    seasons = ["Spring", "Summer", "Fall", "Winter"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, season in enumerate(seasons):
        row, col = divmod(i, 2)

        vars_season = [
            f"Tmp_{season}",
            f"Tmax_{season}",
            f"Tmin_{season}",
            f"Pre_{season}",
            f"Tswrf_{season}",
        ]

        corr = climate_df[vars_season].corr()

        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[row][col])
        axes[row][col].set_title(f"{season} correlation")

    save_or_show("climate_correlation_heatmap.png")