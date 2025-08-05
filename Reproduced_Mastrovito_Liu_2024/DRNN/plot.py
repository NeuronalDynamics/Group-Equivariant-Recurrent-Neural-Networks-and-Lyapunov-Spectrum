import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_json('fig6_results.json')

# panel A
sns.lineplot(df, x='g', y='lam0', hue='model', errorbar='sd')
plt.axhline(0, ls='--', c='k'); plt.title('Fig 6A – λ pre-training'); plt.show()

# panel E
sns.lineplot(df, x='g', y='RA', hue='model', errorbar='sd')
plt.title('Fig 6E – Representation alignment'); plt.show()

# panel G (hidden weight change)
subset = df[df['model'].isin(['Gaussian198','Meso198','Gaussian28','MesoSparse28'])]
sns.lineplot(subset, x='g', y='dH', hue='model', errorbar='sd')
plt.title('Fig 6G – ‖ΔH‖'); plt.show()

# --- Fig. 6C: accuracy vs gain ---------------------------------------
sns.lineplot(df, x='g', y='acc', hue='model', errorbar='sd')
plt.ylabel('test accuracy')
plt.ylim(0, 1)
plt.title('Fig 6C – accuracy after training')
plt.show()