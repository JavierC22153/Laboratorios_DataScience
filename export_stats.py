# export_stats.py
import os, glob, re
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio

def compute_stats_for_folder(folder_path, lake_name, transform="Percentile stretch", p_low=1.0, p_high=99.0, gain=1.0):
    rows = []
    pattern = os.path.join(folder_path, "*.tif")
    for fp in sorted(glob.glob(pattern)):
        try:
            with rasterio.open(fp) as src:
                band = src.read(1, masked=True)
                if hasattr(band, "mask"):
                    valid_mask = ~band.mask
                    valid_count = int(valid_mask.sum())
                    total_count = int(band.size)
                else:
                    valid_count = int(np.count_nonzero(~np.isnan(band)))
                    total_count = int(band.size)
                valid_pct = 0.0 if total_count == 0 else float(valid_count) / float(total_count) * 100.0

                if valid_count == 0:
                    rows.append({'lago': lake_name, 'filepath': fp, 'fecha': None, 'mes': None,
                                 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                                 'valid_pct': valid_pct})
                    continue

                vals = (band.compressed() if hasattr(band, "compressed") else band[~np.isnan(band)]).astype(float)

                if transform == "Percentile stretch":
                    lo = np.percentile(vals, p_low)
                    hi = np.percentile(vals, p_high)
                    if hi - lo <= 0:
                        vals = vals - lo
                    else:
                        vals = np.clip((vals - lo) / (hi - lo), 0, 1)
                elif transform == "Log":
                    if np.any(vals < 0):
                        vals = vals - vals.min() + 1e-6
                    vals = np.log1p(vals)
                elif transform == "Gain":
                    vals = vals * float(gain)

                rows.append({
                    'lago': lake_name,
                    'filepath': fp,
                    'fecha': (lambda m: datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else datetime.fromtimestamp(os.path.getmtime(fp)).date())(re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(fp))),
                    'mes': None,  # la llenamos después
                    'mean': float(np.mean(vals)),
                    'median': float(np.median(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'valid_pct': valid_pct
                })
        except Exception as e:
            rows.append({'lago': lake_name, 'filepath': fp, 'fecha': None, 'mes': None,
                         'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                         'valid_pct': 0.0})
            continue
    return pd.DataFrame(rows)

if __name__ == "__main__":
    path_ami = "NCDI_AMATITLAN"
    path_ati = "NCDI_ATITLAN"
    df1 = compute_stats_for_folder(path_ami, "Amatitlán") if os.path.isdir(path_ami) else pd.DataFrame()
    df2 = compute_stats_for_folder(path_ati, "Atitlán") if os.path.isdir(path_ati) else pd.DataFrame()
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=['fecha']).copy()
    if not df.empty:
        df['mes'] = df['fecha'].apply(lambda d: d.month)
    out = "estadisticos_ncdi.csv"
    df.to_csv(out, index=False)
    print("Exportado:", out, "Filas:", len(df))

