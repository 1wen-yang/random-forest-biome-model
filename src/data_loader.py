import pandas as pd
from functools import reduce
import pycountry_convert as pc


def format_rgb_string(rgbstring):
    rgbs = rgbstring.split(" ")
    return tuple(float(rgb) for rgb in rgbs)


def load_biome_legend(path="data/legend of biomes.txt"):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    biome_names = lines[0::2]
    biome_names = [" ".join(l.split(" ")[1:]) for l in biome_names]

    biome_rgbs = lines[1::2]
    biome_rgbs = [format_rgb_string(rgbstr) for rgbstr in biome_rgbs]

    return biome_names, biome_rgbs


def load_climate_seasonal(filepath, varname):
    data = pd.read_csv(filepath)
    return data[["Lon", "Lat", "SpringMean", "SummerMean", "FallMean", "WinterMean"]].rename(
        columns={
            "SpringMean": f"{varname}_Spring",
            "SummerMean": f"{varname}_Summer",
            "FallMean": f"{varname}_Fall",
            "WinterMean": f"{varname}_Winter",
        }
    )


def load_climate_data():
    tmp = load_climate_seasonal("data/Tmpdaymean1961_1990_stats.csv", "Tmp")
    tmax = load_climate_seasonal("data/Tmaxdaymean1961_1990_stats.csv", "Tmax")
    tmin = load_climate_seasonal("data/Tmindaymean1961_1990_stats.csv", "Tmin")
    pre = load_climate_seasonal("data/Predaymean1961_1990_stats.csv", "Pre")
    tswrf = load_climate_seasonal("data/Tswrfdaymean1961_1990_stats.csv", "Tswrf")

    climate_df = tmp.merge(tmax, on=["Lon", "Lat"]) \
                    .merge(tmin, on=["Lon", "Lat"]) \
                    .merge(pre, on=["Lon", "Lat"]) \
                    .merge(tswrf, on=["Lon", "Lat"])

    return climate_df


def iso3_to_continent(iso3):
    try:
        iso2 = pc.country_alpha3_to_country_alpha2(iso3)
        continent_code = pc.country_alpha2_to_continent_code(iso2)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except Exception:
        return None


def load_full_dataset():
    df = pd.read_csv("data/LPJ-GUESS_output_BERN1.csv")

    climate_files = {
        "Tmp": "data/Tmpdaymean1961_1990_stats.csv",
        "Tmax": "data/Tmaxdaymean1961_1990_stats.csv",
        "Tmin": "data/Tmindaymean1961_1990_stats.csv",
        "Pre": "data/Predaymean1961_1990_stats.csv",
        "Tswrf": "data/Tswrfdaymean1961_1990_stats.csv",
    }

    climate_dfs = []
    for prefix, path in climate_files.items():
        stats = pd.read_csv(path)
        rename_dict = {
            col: f"{prefix}_{col}"
            for col in stats.columns
            if col not in ["Lon", "Lat"]
        }
        stats = stats.rename(columns=rename_dict)
        climate_dfs.append(stats)

    df_full = reduce(
        lambda left, right: pd.merge(left, right, on=["Lon", "Lat"]),
        [df] + climate_dfs
    )

    countries = pd.read_csv("data/gridlist_pan_gfed_ISO3_UN.txt", sep=r"\s+")
    df_full = df_full.merge(
        countries[["Lon", "Lat", "ISO3"]],
        on=["Lon", "Lat"],
        how="left"
    )

    df_full.rename(columns={"ISO3": "CountryCode"}, inplace=True)
    df_full["Continent"] = df_full["CountryCode"].apply(iso3_to_continent)

    return df, df_full