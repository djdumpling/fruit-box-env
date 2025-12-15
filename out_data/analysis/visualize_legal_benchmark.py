import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 8)

data = {
    "anthropic/claude-opus-4.1": 0.19926581271333335,
    "anthropic/claude-sonnet-4.5": 0.060462537756666664,
    "deepseek/deepseek-v3.1-terminus": 0.12068557137333333,
    "google/gemini-2.5-flash": 0.28376953914666664,
    "google/gemini-2.5-pro": 0.4550561,
    "google/gemini-3-pro-preview": 0.8764633424599999,
    "openai/gpt-4o": 0.01681577999333333,
    "openai/gpt-4o-mini": 0.009970565186666667,
    "openai/gpt-5": 0.91413270964,
    "openai/gpt-5.1": 0.9963411896733334,
    "openai/gpt-oss-120b": 0.9075005377666667,
    "openai/gpt-5.2": 0.983,
    "qwen/qwen3-235b-thinking": 0.37477857020666666,
    "x-ai/grok-4": 0.9916256908666666,
    "x-ai/grok-4-fast": 0.95262895631875,
    "prime-intellect/intellect-3": 0.570,
    "z-ai/glm-4.5": 0.595,

}

def clean_model_name(name):
    parts = name.split('/')
    if len(parts) > 1:
        return parts[1]
    return name

sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
sorted_items.reverse()
models = [clean_model_name(item[0]) for item in sorted_items]
scores = [item[1] * 100 for item in sorted_items]

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

fig, ax = plt.subplots(figsize=(10.5, 5))
ax.grid(False)

bars = ax.barh(models, scores, color=colors, edgecolor='none', height=0.8, alpha=0.7)

for bar, score, model in zip(bars, scores, models):
    bar_center_y = bar.get_y() + bar.get_height()/2
    score_str = f'{score:.1f}%'.replace('.0%', '%')
    ax.text(score + 0.5, bar_center_y,
            score_str, ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Fruit Box Legal Bench', fontsize=20, fontweight='bold', pad=20)
ax.set_xlim(0, 107)

ax.tick_params(axis='y', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.subplots_adjust(top=0.93, bottom=0.12, left=0.2, right=0.88, hspace=0)

output_path = Path(__file__).parent / 'fruit_box_legal_benchmark.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

