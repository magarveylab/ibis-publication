import pandas as pd


def get_class_dict(class_dict_fp: str):
    df = pd.read_csv(class_dict_fp)
    return dict(zip(df.label_id, df.label))
