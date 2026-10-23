"""make_figure.py - defense-family figure, THREE models, corrected numbers.
Generates figures/defense_family_figure.pdf (and .png).
Run:  python make_figure.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

families = ['PromptShield', 'TaskTracker', 'Spotlight', 'LLM-judge', 'All-comb.']

# Corrected population-weighted ASR (%) by family: (no_defense, +separation)
data = {
    'Llama 3.3 70B':     [(89.7,87.6),(90.7,89.6),(97.2,97.0),(89.3,89.5),(88.5,88.5)],
    'GPT-4o-mini':       [(37.8, 8.4),(37.8, 8.7),(74.3,52.8),(33.9,17.2),(33.3,10.8)],
    'Claude Sonnet 4.6': [( 1.6, 0.2),( 1.4, 0.0),( 5.8, 0.0),( 5.7, 0.0),( 2.6, 0.0)],
}

# three panels in one row
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
x = np.arange(len(families))
w = 0.38

for ax, (model, vals) in zip(axes, data.items()):
    nd = [v[0] for v in vals]
    sp = [v[1] for v in vals]
    b1 = ax.bar(x - w/2, nd, w, label='No defense',
                color='#4d4d4d', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, sp, w, label='+ Separation',
                color='#cccccc', edgecolor='black', linewidth=0.5, hatch='////')
    ax.set_title(model, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    # value labels
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(f'{h:.0f}' if h >= 1 else f'{h:.1f}',
                    (b.get_x() + b.get_width()/2, h),
                    xytext=(0, 1.5), textcoords='offset points',
                    ha='center', fontsize=6)

axes[0].set_ylabel('Attack Success Rate (%)', fontsize=9)
# single shared legend at top
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 1.06))

plt.tight_layout()
Path('figures').mkdir(exist_ok=True)
plt.savefig('figures/defense_family_figure.pdf', bbox_inches='tight')
plt.savefig('figures/defense_family_figure.png', dpi=200, bbox_inches='tight')
print("wrote figures/defense_family_figure.pdf and .png")