import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# читаем датасет
df = pd.read_csv("data/complaints_nonempty.csv")

# complaint_types -> список id
def extract_ids(s):
    if pd.isna(s) or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
        return [d["id"] for d in parsed]
    except Exception:
        return []

df["labels"] = df["complaint_types"].apply(extract_ids)

# теперь делаем one-hot (multi-label binarizer)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

# получаем новый DataFrame с one-hot метками
y_df = pd.DataFrame(y, columns=[f"id_{cls}" for cls in mlb.classes_])
dataset = pd.concat([df[["text"]], y_df], axis=1)

print(dataset.head())
dataset.to_csv("data/multilabel", index=False)

