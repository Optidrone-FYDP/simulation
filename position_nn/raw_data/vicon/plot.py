import pandas as pd
import matplotlib.pyplot as plt

file_path = "70_up_then_down.csv"
start_frame = 0
line_interval = 30

df = pd.read_csv(file_path, skiprows=[0, 1, 2, 4])

df_filtered = df[df['Frame'] >= start_frame]

plt.figure(figsize=(10, 6))
plt.plot(df_filtered['Frame'], df_filtered['TX'], label='TX', marker='o')
plt.plot(df_filtered['Frame'], df_filtered['TY'], label='TY', marker='s')
plt.plot(df_filtered['Frame'], df_filtered['TZ'], label='TZ', marker='^')

for frame in range(start_frame, df_filtered['Frame'].max(), line_interval):
    plt.axvline(x=frame, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Frame')
plt.ylabel('Translation (mm)')
plt.title(f'Translation Over Time (Starting from Frame {start_frame})')
plt.legend()
plt.grid(True)

plt.savefig("temp.png")
