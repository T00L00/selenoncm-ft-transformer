import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from pathlib import Path

DATA_DIR = Path("./data")
BATCH1 = "2024_07_19_Batch1_normalized_feature_select_negcon_batch.gct"
BATCH2 = "2025_01_31_Batch2_normalized_feature_select_negcon_batch.gct"
BATCH3 = "2025_06_10_Batch3_normalized_feature_select_negcon_batch.gct"
BATCH4 = "2025_09_25_Batch4_normalized_feature_select_negcon_batch.gct"
BATCH5 = "2025_09_25_Batch5_normalized_feature_select_negcon_batch.gct"

def extract(path: Path, batch_prefix: str, ko_labels: list[str]):
    gct = parse(str(path))
    df = gct.data_df.T
    df["label"] = gct.col_metadata_df["perturbation"].apply(lambda x: 1 if x in ko_labels else 0)
    df.index = batch_prefix + "_" + df.index.astype(str)
    return df

if __name__ == "__main__":

    # batch 1
    batch1_df = extract(DATA_DIR / BATCH1, "B1", ["3F9KO", "KO"])
    batch1_df.to_pickle(DATA_DIR / "batch1.pkl")
    print(f"Batch 1 {batch1_df.shape}:")
    print(batch1_df.head())

    # batch 2 - has a "nan" perturbation, not sure what this is
    # batch2_df = extract(DATA_DIR / BATCH2, "B2", [""])
    # batch2_df.to_pickle(DATA_DIR / "batch2.pkl")

    # batch 3
    batch3_df = extract(DATA_DIR / BATCH3, "B3", ["SELENON KO"])
    batch3_df.to_pickle(DATA_DIR / "batch3.pkl")
    print(f"Batch 3 {batch3_df.shape}:")
    print(batch3_df.head())

    # batch 4
    batch4_df = extract(DATA_DIR / BATCH4, "B4", ["SELENON KO"])
    batch4_df.to_pickle(DATA_DIR / "batch4.pkl")
    print(f"Batch 4 {batch4_df.shape}:")
    print(batch4_df.head())

    # batch 5
    batch5_df = extract(DATA_DIR / BATCH5, "B5", ["SELENON KO-Exon3", "SELENON KO-Exon5"])
    batch5_df.to_pickle(DATA_DIR / "batch5.pkl")
    print(f"Batch 5 {batch5_df.shape}:")
    print(batch5_df.head())

    # combine batch 1 and batch 5
    b1b5_df = pd.concat([batch1_df, batch5_df], axis=0, ignore_index=False, sort=False)
    b1b5_df.to_pickle(DATA_DIR / "b1b5.pkl")
    print(f"Batch 1 + Batch 5 {b1b5_df.shape}:")
    print(b1b5_df.head())

    # combine all batches
    all_batches = [batch1_df, batch3_df, batch4_df, batch5_df]
    all_df = pd.concat(all_batches, axis=0, ignore_index=False, sort=False)

    assert all_df.shape[0] == batch1_df.shape[0] + batch3_df.shape[0] + batch4_df.shape[0] + batch5_df.shape[0], "Combined dataset sample count inconsistent with individual batches..."

    all_df.to_pickle(DATA_DIR / "all_batches.pkl")
    print(f"All batches combined {all_df.shape}:")
    print(all_df)

