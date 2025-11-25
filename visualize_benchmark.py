import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 8)

data = {
    'RL': 0.944,
    'GPT-5': 0.799,
    'Qwen3-235B Thinking': 0.781,
    'Claude-Sonnet 4.5': 0.557,
    'Gemini-2.5 Pro': 0.435,
    'Kimi-K2 Thinking': 0.221,
    'GPT-4o': 0.074,
    'Gemini-2.5 Flash': 0.016,
    'GPT-4o mini': 0.010,
}

sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
sorted_items.reverse()
models = [item[0] for item in sorted_items]
percentages = [item[1] * 100 for item in sorted_items]

color_list = [
    '#6B9BD1', '#5FA8C4', '#4FB3B3', '#3DB8A8', '#2DBD9D',
    '#1DC292', '#0DC787', '#00BC7C', '#00B171',
]

if len(models) > len(color_list):
    cmap = LinearSegmentedColormap.from_list('blue_green', color_list, N=len(models))
    colors = [cmap(i) for i in np.linspace(0, 1, len(models))]
else:
    colors = [mcolors.to_rgba(color_list[i]) for i in range(len(models))]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.grid(False)

bars = ax.barh(models, percentages, color=colors, edgecolor='none', height=0.8)

for i, (bar, pct) in enumerate(zip(bars, percentages)):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{pct:.2f}%', 
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Performance (% relative to Expert Policy)', fontsize=12, fontweight='bold')
ax.set_title('Fruit Box Bench', fontsize=20, fontweight='bold', pad=20)
ax.set_xlim(0, 105)

ax.tick_params(axis='y', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.subplots_adjust(top=0.93, bottom=0.12, left=0.15, right=0.95, hspace=0)

output_path = 'out_data/analysis/fruit_box_benchmark.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()