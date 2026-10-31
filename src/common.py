"""Shared constants and helpers for the DS203 nation-clustering project.

Every notebook imports from here so that the feature list, transformations and
colours stay identical across the whole pipeline. If you change a transformation,
change it HERE (once) and re-run the notebooks.
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. The 11 WDI indicators (WDI code -> short column name)
# --------------------------------------------------------------------------
INDICATORS = {
    "NY.GDP.PCAP.CD":    "GDP_PC",               # GDP per capita (current US$)
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",           # GDP growth (annual %)
    "FP.CPI.TOTL.ZG":    "Inflation",            # Inflation, consumer prices (annual %)
    "SL.UEM.TOTL.ZS":    "Unemployment",         # Unemployment (% of labour force)
    "NE.TRD.GNFS.ZS":    "Trade",                # Trade (% of GDP)
    "NE.GDI.TOTL.ZS":    "Capital_Formation",    # Gross capital formation (% of GDP)
    "SP.DYN.LE00.IN":    "Life_Expectancy",      # Life expectancy at birth (years)
    "SP.URB.TOTL.IN.ZS": "Urban_Pop",            # Urban population (% of total)
    "IT.NET.USER.ZS":    "Internet_Users",       # Individuals using the Internet (% of pop.)
    "EG.ELC.ACCS.ZS":    "Electricity_Access",   # Access to electricity (% of pop.)
    "SE.SEC.ENRR":       "Secondary_Enrollment", # School enrolment, secondary (% gross)
}
FEATURES = list(INDICATORS.values())

# Known categorical labels: used ONLY to interpret / evaluate clusters, never as features.
LABEL_COLS = ["name", "region", "income_group", "lending_type"]

PRETTY = {
    "GDP_PC": "GDP per capita (US$)",
    "GDP_Growth": "GDP growth (%)",
    "Inflation": "Inflation (%)",
    "Unemployment": "Unemployment (%)",
    "Trade": "Trade (% of GDP)",
    "Capital_Formation": "Gross capital formation (% of GDP)",
    "Life_Expectancy": "Life expectancy (years)",
    "Urban_Pop": "Urban population (%)",
    "Internet_Users": "Internet users (% of pop.)",
    "Electricity_Access": "Access to electricity (% of pop.)",
    "Secondary_Enrollment": "Secondary enrolment (% gross)",
}

# World Bank income groups, ordered low -> high (strings exactly as the WB API returns them)
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
INCOME_COLORS = dict(zip(INCOME_ORDER, ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]))

# --------------------------------------------------------------------------
# 2. Transformations (documented design choice - justify in the EDA notebook)
# --------------------------------------------------------------------------
LOG10_COLS = ["GDP_PC"]                    # right-skewed, strictly positive -> log10
SIGNED_LOG_COLS = ["Inflation"]            # heavy right tail, can be negative -> sign(x)*ln(1+|x|)
WINSOR_COLS = ["Trade", "GDP_Growth"]      # a few extreme values -> clip to 1st/99th percentile
WINSOR_Q = (0.01, 0.99)


def signed_log1p(s):
    """sign(x) * ln(1 + |x|): behaves like x near 0, compresses large |x|, keeps the sign."""
    return np.sign(s) * np.log1p(s.abs())


def winsorize(s, q=WINSOR_Q):
    """Clip a Series to its [q_low, q_high] quantiles (NaNs are ignored)."""
    lo, hi = s.quantile(q[0]), s.quantile(q[1])
    return s.clip(lo, hi)


def apply_transforms(df):
    """Return a copy of the 11 feature columns with the documented transformations.

    Renamed columns make the transformation visible in later plots / PCA loadings:
        GDP_PC    -> log10_GDP_PC
        Inflation -> slog1p_Inflation
        Trade, GDP_Growth keep their names but are winsorised.
    """
    out = df[FEATURES].astype(float).copy()
    for c in WINSOR_COLS:
        out[c] = winsorize(out[c])
    for c in SIGNED_LOG_COLS:
        out[c] = signed_log1p(out[c])
        out = out.rename(columns={c: f"slog1p_{c}"})
    for c in LOG10_COLS:
        if (out[c] <= 0).any():
            raise ValueError(f"{c} has non-positive values; log10 is undefined.")
        out[c] = np.log10(out[c])
        out = out.rename(columns={c: f"log10_{c}"})
    return out
