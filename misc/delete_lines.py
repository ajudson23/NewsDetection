import pandas as pd

# Load the full dataset
df = pd.read_csv('WELFake_Dataset.csv')

# Check how many real/fake samples we have
print("Label distribution:\n", df['label'].value_counts())

# Filter real (label = 1) and fake (label = 0)
df_real = df[df['label'] == 1].sample(n=100, random_state=42)
df_fake = df[df['label'] == 0].sample(n=100, random_state=42)

# Concatenate the balanced sample
df_sample = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a smaller CSV
df_sample.to_csv('tiny_fakenews_sample.csv', index=False)

print("Saved 200-row balanced dataset to 'tiny_fakenews_sample.csv'")
