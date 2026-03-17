"""
Phase 5, Analysis 6: Publication-Ready Figures
===============================================
Regenerates all figures with publication-quality formatting.
Uses seaborn-v0_8-paper style, ColorBrewer palettes, 10pt+ labels,
PDF output, white backgrounds.

No GPU needed.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from Levenshtein import ratio as lev_ratio
import os

TRAIN = {l1: f'data/train/{l1}/kvl_shared_task_{l1}_train.csv' for l1 in ['es','de','cn']}
DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}
COLORS = {'es': '#e41a1c', 'de': '#377eb8', 'cn': '#4daf4a'}
L1_LABELS = {'es': 'Spanish', 'de': 'German', 'cn': 'Mandarin'}


def setup_style():
    """Apply publication-quality matplotlib settings."""
    plt.style.use('seaborn-v0_8-paper')
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def fig1_results_comparison():
    """Bar chart comparing all systems."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    systems = ['Official\nBaseline', 'Our Exp 0\n(base)', 'Hybrid\nEnsemble', 'Hybrid +\nXGBoost']
    es_vals = [1.357, 1.338, 1.021, 0.997]
    de_vals = [1.328, 1.299, 1.013, 1.002]
    cn_vals = [1.175, 1.222, 0.940, 0.932]

    x = np.arange(len(systems))
    w = 0.25
    ax.bar(x - w, es_vals, w, label='Spanish', color=COLORS['es'], alpha=0.85)
    ax.bar(x, de_vals, w, label='German', color=COLORS['de'], alpha=0.85)
    ax.bar(x + w, cn_vals, w, label='Mandarin', color=COLORS['cn'], alpha=0.85)
    ax.set_ylabel('RMSE (↓ better)')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_ylim(0.8, 1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/fig1_results_comparison.pdf')
    plt.show()
    print("Fig 1 saved")


def fig2_cross_l1_correlation(train):
    """Cross-L1 GLMM score correlation heatmap."""
    merged = train['es'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'es'})
    merged = merged.merge(train['de'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'de'}), on='item_id')
    merged = merged.merge(train['cn'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'cn'}), on='item_id')

    fig, ax = plt.subplots(figsize=(4, 3.5))
    corr = merged[['es', 'de', 'cn']].corr()
    corr.index = ['Spanish', 'German', 'Mandarin']
    corr.columns = ['Spanish', 'German', 'Mandarin']
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0.5, vmax=1.0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Cross-L1 GLMM Score Correlation')
    plt.tight_layout()
    plt.savefig('results/figures/fig2_cross_l1_correlation.pdf')
    plt.show()
    print("Fig 2 saved")


def fig3_cognate_scatter(train):
    """Cognate edit distance vs GLMM score for es and de."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for i, l1 in enumerate(['es', 'de']):
        df = train[l1].copy()
        df['edit_dist'] = df.apply(
            lambda r: 1 - lev_ratio(str(r['en_target_word']).lower(), str(r['L1_source_word']).lower()), axis=1
        )
        r, p = stats.pearsonr(df['edit_dist'], df['GLMM_score'])
        axes[i].scatter(df['edit_dist'], df['GLMM_score'], alpha=0.15, s=8, color=COLORS[l1])
        axes[i].set_xlabel('Edit Distance (en ↔ L1)')
        axes[i].set_ylabel('GLMM Score')
        axes[i].set_title(f'{L1_LABELS[l1]} (r={r:.3f})')
        z = np.polyfit(df['edit_dist'], df['GLMM_score'], 1)
        x_line = np.linspace(0, 1, 100)
        axes[i].plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/figures/fig3_cognate_scatter.pdf')
    plt.show()
    print("Fig 3 saved")


def fig4_transfer_heatmap():
    """Cross-L1 transfer RMSE heatmap."""
    transfer = pd.DataFrame({
        'Spanish': [1.550, 1.623, 1.794],
        'German': [1.568, 1.487, 1.827],
        'Mandarin': [1.466, 1.554, 1.278],
    }, index=['Train: Spanish', 'Train: German', 'Train: Mandarin'])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(transfer, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'RMSE'})
    ax.set_title('Cross-L1 Transfer (XGBoost, shared features)')
    plt.tight_layout()
    plt.savefig('results/figures/fig4_transfer_heatmap.pdf')
    plt.show()
    print("Fig 4 saved")


def fig5_glmm_distribution(train):
    """GLMM score distribution by L1."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for l1 in ['es', 'de', 'cn']:
        ax.hist(train[l1]['GLMM_score'], bins=50, alpha=0.4, color=COLORS[l1],
                label=L1_LABELS[l1], density=True)
        train[l1]['GLMM_score'].plot.kde(ax=ax, color=COLORS[l1], linewidth=2)
    ax.set_xlabel('GLMM Score (← harder | easier →)')
    ax.set_ylabel('Density')
    ax.set_title('GLMM Score Distribution by L1')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/figures/fig5_glmm_distribution.pdf')
    plt.show()
    print("Fig 5 saved")


def fig6_difficulty_clusters(dev):
    """3D scatter of difficulty clusters (k=3)."""
    merged = dev['es'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_es'})
    merged = merged.merge(dev['de'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_de'}), on='item_id')
    merged = merged.merge(dev['cn'][['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_cn'}), on='item_id')

    X = merged[['GLMM_es', 'GLMM_de', 'GLMM_cn']].values
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    cluster_names = ['Moderate', 'Easy', 'Hard']

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    for c in range(3):
        mask = labels == c
        ax.scatter(merged.loc[mask, 'GLMM_es'], merged.loc[mask, 'GLMM_de'],
                   merged.loc[mask, 'GLMM_cn'], c=cluster_colors[c], alpha=0.5, s=12,
                   label=cluster_names[c])
    ax.set_xlabel('Spanish', fontsize=10)
    ax.set_ylabel('German', fontsize=10)
    ax.set_zlabel('Mandarin', fontsize=10)
    ax.legend(title='Difficulty Tier', loc='upper left')
    ax.set_title('Word Difficulty Clusters (k=3)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/fig6_difficulty_clusters.pdf')
    plt.show()
    print("Fig 6 saved")


def main():
    setup_style()
    os.makedirs('results/figures', exist_ok=True)

    # Load data
    train = {l1: pd.read_csv(TRAIN[l1]) for l1 in ['es', 'de', 'cn']}
    dev = {l1: pd.read_csv(DEV[l1]) for l1 in ['es', 'de', 'cn']}

    fig1_results_comparison()
    fig2_cross_l1_correlation(train)
    fig3_cognate_scatter(train)
    fig4_transfer_heatmap()
    fig5_glmm_distribution(train)
    fig6_difficulty_clusters(dev)

    # List all figures
    print("\nFigures generated:")
    for f in sorted(os.listdir('results/figures')):
        if f.endswith('.pdf'):
            print(f"  results/figures/{f}")


if __name__ == '__main__':
    main()
