"""
Phase 5, Analysis 5: Difficulty Clustering
===========================================
Joins dev scores across all 3 L1s per item_id, creating a 3D vector
(GLMM_es, GLMM_de, GLMM_cn). Applies K-means clustering to identify
universally easy/hard vs L1-dependent words.

No GPU needed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import os

DEV = {l1: f'data/dev/{l1}/kvl_shared_task_{l1}_dev.csv' for l1 in ['es','de','cn']}


def main():
    os.makedirs('results/figures', exist_ok=True)

    # Load and merge dev data on item_id
    es_dev = pd.read_csv(DEV['es'])[['item_id', 'en_target_word', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_es'})
    de_dev = pd.read_csv(DEV['de'])[['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_de'})
    cn_dev = pd.read_csv(DEV['cn'])[['item_id', 'GLMM_score']].rename(columns={'GLMM_score': 'GLMM_cn'})

    merged = es_dev.merge(de_dev, on='item_id').merge(cn_dev, on='item_id')
    print(f"Merged: {len(merged)} items with scores across all 3 L1s")

    # Elbow method
    X = merged[['GLMM_es', 'GLMM_de', 'GLMM_cn']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 3))
    plt.plot(range(2, 8), inertias, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.tight_layout()
    plt.savefig('results/figures/elbow.pdf')
    plt.show()
    print("Elbow plot saved")

    # K-means with k=3 and k=4
    for k in [3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        merged[f'cluster_k{k}'] = km.fit_predict(X_scaled)

        print(f"\n{'='*60}")
        print(f"  K-MEANS k={k}")
        print(f"{'='*60}")

        for c in range(k):
            subset = merged[merged[f'cluster_k{k}'] == c]
            mean_es = subset['GLMM_es'].mean()
            mean_de = subset['GLMM_de'].mean()
            mean_cn = subset['GLMM_cn'].mean()
            mean_all = (mean_es + mean_de + mean_cn) / 3

            if mean_all > 0.5:
                label = "UNIVERSALLY EASY"
            elif mean_all < -0.5:
                label = "UNIVERSALLY HARD"
            elif abs(mean_es - mean_cn) > 1.0 or abs(mean_de - mean_cn) > 1.0:
                label = "L1-DEPENDENT"
            else:
                label = "MODERATE"

            print(f"\n  Cluster {c}: n={len(subset)}, {label}")
            print(f"    mean GLMM:  es={mean_es:+.3f}  de={mean_de:+.3f}  cn={mean_cn:+.3f}")
            print(f"    word_length: {subset['en_target_word'].str.len().mean():.1f}")
            examples = subset.sample(min(5, len(subset)), random_state=42)['en_target_word'].tolist()
            print(f"    examples: {', '.join(examples)}")

    # 3D scatter for k=3
    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    cluster_names = ['Moderate', 'Easy', 'Hard']

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    for c in range(3):
        mask = merged['cluster_k3'] == c
        ax.scatter(merged.loc[mask, 'GLMM_es'], merged.loc[mask, 'GLMM_de'],
                   merged.loc[mask, 'GLMM_cn'], c=cluster_colors[c], alpha=0.5, s=12,
                   label=cluster_names[c])
    ax.set_xlabel('Spanish', fontsize=10)
    ax.set_ylabel('German', fontsize=10)
    ax.set_zlabel('Mandarin', fontsize=10)
    ax.legend(title='Difficulty Tier', loc='upper left')
    ax.set_title('Word Difficulty Clusters (k=3)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/difficulty_clusters.pdf')
    plt.show()
    print("Cluster figure saved!")

    # Save results
    merged.to_csv('results/clustering_results.csv', index=False)
    print("Done!")


if __name__ == '__main__':
    main()
