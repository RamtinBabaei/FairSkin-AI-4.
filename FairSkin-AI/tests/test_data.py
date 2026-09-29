import pandas as pd

from fairskin_ai.data import add_skin_group, make_label_map, split_metadata


def _frame(n=120):
    labels = ["benign", "malignant", "non-neoplastic"]
    return pd.DataFrame({
        "image_path": [f"x{i}.png" for i in range(n)],
        "target": [labels[i%3] for i in range(n)],
        "fitzpatrick": [(i%6)+1 for i in range(n)],
    })


def test_skin_groups_and_split():
    df = add_skin_group(_frame())
    assert set(df.skin_group.unique()) == {"I-II", "III-IV", "V-VI"}
    tr, va, te = split_metadata(df, 0.15, 0.15, 42)
    assert len(tr) + len(va) + len(te) == len(df)
    lm = make_label_map(tr, va, te)
    assert len(lm.class_to_idx) == 3
