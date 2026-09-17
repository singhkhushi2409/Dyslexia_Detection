import os

import pandas as pd

from api import extract_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLUMNS = [
    "ink_density", "connected_components", "mean_height", "height_variation",
    "mean_width", "width_variation", "baseline_drift", "mean_spacing",
    "spacing_variation", "estimated_stroke_width", "slant_variation",
    "margin_alignment", "overlap_ratio",
]


def build_feature_table():
    rows = []
    skipped = []
    for directory_name, label in (("non_dyslexic", 0), ("dyslexic", 1)):
        directory = os.path.join(BASE_DIR, "data", directory_name)
        for filename in sorted(os.listdir(directory)):
            path = os.path.join(directory, filename)
            if not os.path.isfile(path):
                continue
            try:
                rows.append(extract_features(path) + [label])
            except (OSError, ValueError) as error:
                skipped.append((path, str(error)))

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label"])
    frame.to_csv(os.path.join(BASE_DIR, "handwriting_features.csv"), index=False)
    print(f"Generated {len(frame)} feature rows.")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for path, error in skipped:
            print(f"  {path}: {error}")


if __name__ == "__main__":
    build_feature_table()