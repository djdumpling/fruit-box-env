import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 8)

data_avg = {
    'RL': 0.944,
    'gpt-5.2': 0.810,
    'gpt-5': 0.799,
    'qwen3-235B-thinking': 0.781,
    'gemini-3-pro-preview': 0.668,
    'gpt-oss-120b': 0.636,
    'claude-sonnet-4.5': 0.557,
    'grok-4-fast': 0.482,
    'gemini-2.5-pro': 0.435,
    'intellect-3': 0.292,
    'glm-4.5': 0.255,
    'kimi-K2-thinking': 0.221,
    'gpt-4.1': 0.176,
    'gpt-4o': 0.074,
    'deepseek-v3.1-terminus': 0.018,
    'gemini-2.5-flash': 0.016,
    'gpt-4o mini': 0.001,
}

data_best3 = {
    'gpt-5.2': 0.868,
    'qwen3-235B-thinking': 0.846418,
    'gpt-5': 0.834605,
    'gemini-3-pro-preview': 0.792880,
    'gpt-oss-120b': 0.723704,
    'claude-sonnet-4.5': 0.693418,
    'grok-4-fast': 0.646795,
    'gemini-2.5-pro': 0.589298,
    'intellect-3': 0.452,
    'glm-4.5': 0.346,
    'kimi-K2-thinking': 0.392752,
    'gpt-4o': 0.101990,
    'deepseek-v3.1-terminus': 0.031216,
    'gemini-2.5-flash': 0.024781,
    'gpt-4o-mini': 0.003252,
}

sorted_items = sorted(data_avg.items(), key=lambda x: x[1], reverse=True)
sorted_items.reverse()
models = [item[0] for item in sorted_items]
avg_percentages = [item[1] * 100 for item in sorted_items]

best3_percentages = []
for model in models:
    model_key = model.replace(' ', '-')
    if model_key in data_best3:
        best3_percentages.append(data_best3[model_key] * 100)
    else:
        best3_percentages.append(data_avg[model] * 100)

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

models_with_extensions = ['gpt-5', 'qwen3-235B-thinking', 'gemini-3-pro-preview', 'gpt-oss-120b', 
                         'claude-sonnet-4.5', 'grok-4-fast', 'gemini-2.5-pro', 'kimi-K2-thinking']

extension_model_indices = [i for i, model in enumerate(models) if model in models_with_extensions]
base_extension_colors = [colors[i] for i in extension_model_indices]

def increase_saturation_and_darken(rgb, sat_factor=1.3, darken_factor=0.85):
    """Increase saturation and darken slightly while keeping pastel feel"""
    r, g, b = rgb[:3]
    hsv = mcolors.rgb_to_hsv([r, g, b])
    hsv[1] = min(hsv[1] * sat_factor, 0.6)
    hsv[2] = hsv[2] * darken_factor
    rgb_new = mcolors.hsv_to_rgb(hsv)
    return tuple(rgb_new) + (1.0,) if len(rgb) == 4 else tuple(rgb_new)

extension_colors_base = [increase_saturation_and_darken(mcolors.to_rgb(color)) for color in base_extension_colors]

if len(models) > len(extension_colors_base):
    extension_cmap = LinearSegmentedColormap.from_list('extension_gradient', extension_colors_base, N=len(models))
    extension_colors = [extension_cmap(i) for i in np.linspace(0, 1, len(models))]
else:
    extension_colors = []
    for i, model in enumerate(models):
        if model in models_with_extensions:
            idx = models_with_extensions.index(model)
            extension_colors.append(extension_colors_base[idx])
        else:
            extension_colors.append(extension_colors_base[0] if extension_colors_base else '#B0E0E6')

extension_colors = extension_colors[::-1]

fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.grid(False)

bars_avg = ax.barh(models, avg_percentages, color=colors, edgecolor='none', height=0.8, alpha=0.7)

for model, avg_pct, best3_pct, ext_color in zip(models, avg_percentages, best3_percentages, extension_colors):
    if best3_pct > avg_pct:
        extension = best3_pct - avg_pct
        ax.barh(model, extension, left=avg_pct, color=ext_color, edgecolor='none', height=0.8, alpha=1.0)

for bar_avg, avg_pct, best3_pct, model in zip(bars_avg, avg_percentages, best3_percentages, models):
    bar_center_y = bar_avg.get_y() + bar_avg.get_height()/2
    avg_str = f'{avg_pct:.1f}%'.replace('.0%', '%')
    best3_str = f'{best3_pct:.1f}%'.replace('.0%', '%')
    
    bottom_models = ['gpt-4o', 'deepseek-v3.1-terminus', 'gemini-2.5-flash', 'gpt-4o mini']
    
    if model in bottom_models:
        combined_str = f'{avg_str}, {best3_str}'
        ax.text(best3_pct + 0.5, bar_center_y,
                combined_str, ha='left', va='center', fontsize=9, fontweight='bold')
    elif model in ['gpt-5', 'qwen3-235B-thinking', 'gemini-3-pro-preview', 'gpt-oss-120b', 
                   'claude-sonnet-4.5', 'grok-4-fast', 'gemini-2.5-pro', 'intellect-3', 
                   'kimi-K2-thinking', 'glm-4.5']:
        ax.text(avg_pct - 0.3, bar_center_y,
                avg_str, ha='right', va='center', fontsize=9, fontweight='bold')
        if best3_pct > avg_pct:
            ax.text(best3_pct + 0.5, bar_center_y,
                    best3_str, ha='left', va='center', fontsize=10, fontweight='bold')
    elif model == 'gpt-4.1':
        ax.text(avg_pct + 0.5, bar_center_y,
                avg_str, ha='left', va='center', fontsize=9, fontweight='bold')
        if best3_pct > avg_pct:
            ax.text(best3_pct + 0.5, bar_center_y,
                    best3_str, ha='left', va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(avg_pct + 0.5, bar_center_y,
                avg_str, ha='left', va='center', fontsize=9, fontweight='bold')
        if best3_pct > avg_pct:
            ax.text(best3_pct + 0.5, bar_center_y,
                    best3_str, ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Performance (% relative to Expert Policy)', fontsize=12, fontweight='bold')
ax.set_title('Fruit Box Bench (avg@15 and best@3)', fontsize=20, fontweight='bold', pad=20)
ax.set_xlim(0, 100)

ax.tick_params(axis='y', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.subplots_adjust(top=0.93, bottom=0.12, left=0.15, right=0.95, hspace=0)

output_path = Path(__file__).parent / 'fruit_box_benchmark.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()