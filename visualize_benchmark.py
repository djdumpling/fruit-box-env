import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 8)

data = {
    'RL': 0.944,
    'gpt-5': 0.799,
    'qwen3-235B-thinking': 0.781,
    'gemini-3-pro-preview': 0.668,
    'gpt-oss-120b': 0.636,
    'claude-sonnet-4.5': 0.557,
    'grok-4-fast': 0.482,
    'gemini-2.5-pro': 0.435,
    'kimi-K2-thinking': 0.221,
    'gpt-4o': 0.074,
    'deepseek-v3.1-terminus': 0.018,
    'gemini-2.5-flash': 0.016,
    'gpt-4o mini': 0.010,
}

sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
sorted_items.reverse()
models = [item[0] for item in sorted_items]
percentages = [item[1] * 100 for item in sorted_items]

color_list = [
    '#7BA3D8', '#6B9BD1', '#5FA8C4', '#54A8B7', '#4FB3B3',
    '#45B5A9', '#3DB8A8', '#35BA9F', '#2DBD9D', '#25BE93',
    '#1DC292', '#15C489', '#0DC787', '#07C97F', '#00BC7C',
    '#00B875', '#00B171',
]

if len(models) > len(color_list):
    cmap = LinearSegmentedColormap.from_list('blue_green', color_list, N=len(models))
    colors = [cmap(i) for i in np.linspace(0, 1, len(models))]
else:
    colors = [mcolors.to_rgba(color_list[i]) for i in range(len(models))]

fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.grid(False)

bars = ax.barh(models, percentages, color=colors, edgecolor='none', height=0.8)

for i, (bar, pct) in enumerate(zip(bars, percentages)):
    width = bar.get_width()
    # Format percentage to remove trailing zeros (e.g., 94.4% instead of 94.40%)
    pct_str = f'{pct:.1f}%'.replace('.0%', '%')
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
            pct_str, 
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Performance (% relative to Expert Policy)', fontsize=12, fontweight='bold')
ax.set_title('Fruit Box Bench', fontsize=20, fontweight='bold', pad=20)
ax.set_xlim(0, 100)

ax.tick_params(axis='y', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.subplots_adjust(top=0.93, bottom=0.12, left=0.15, right=0.95, hspace=0)

output_path = 'out_data/analysis/fruit_box_benchmark.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()