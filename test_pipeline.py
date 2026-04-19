from src.ingestion import ingest
from src.preprocessing import preprocess

df = ingest()
df, encoders = preprocess(df, save_encoders_path="models/encoders.joblib")
print(df.shape)
print(df.head(2))
