import re


def normalize_label(label):
    """
    Normalizes a class label to avoid mismatches due to:
        - uppercase/lowercase differences  (Iris-Setosa  -> iris-setosa)
        - leading/trailing spaces          ( iris-setosa -> iris-setosa)
        - underscore vs dash               (iris_setosa  -> iris-setosa)
        - multiple spaces                  (Iris  Setosa -> iris-setosa)
    """
    return re.sub(r"[\s_]+", "-", label.strip().lower())


def load_data(filepath, has_label=True):
    """
    Reads a data file and returns a list of samples.

    Parameters:
        filepath  : path to the data file
        has_label : True  -> last column is the class label (default)
                    False -> no label column, only attributes

    Each sample is a dict with:
        'attributes' -> list of floats
        'label'      -> normalized string OR None if has_label=False
    """
    samples = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("\t") if p.strip()]

            if has_label:
                # Last column is the label
                label = normalize_label(parts[-1])
                attributes = [float(p.replace(",", ".")) for p in parts[:-1]]
            else:
                # No label — all columns are attributes
                label = None
                attributes = [float(p.replace(",", ".")) for p in parts]

            samples.append({
                "attributes": attributes,
                "label": label
            })

    return samples


def get_num_attributes(samples):
    if not samples:
        return 0
    return len(samples[0]["attributes"])