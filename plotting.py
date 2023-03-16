# -----------------------------------------------------------------------------
#           PLOT THE RESULTS FROM THE RANDOM STATE ANALYSES
# -----------------------------------------------------------------------------

# (C) Nicolas Rost, 2023


# ------------------------------- Packages ------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------- Load Data --------------------------------------
res = pd.read_csv('classification_results_bac.csv')
res['informativeness'] = res['informativeness'].replace({2:0})


# ------------------------------ Plot BACs ------------------------------------
sns.set(style = 'whitegrid', font_scale = 1.25)
sns.boxplot(data = res, 
            x = 'informativeness', 
            y = 'BAC', 
            hue = 'informativeness', 
            palette = 'colorblind', 
            dodge = False, 
            width = 0.5)
plt.ylim([0.4, 1.0])
plt.xlabel('Number of informative features')
plt.legend([],[], frameon = False)
plt.savefig('BACs_by_info.png', dpi = 300, bbox_inches = 'tight')
plt.close()


# ---------------------------- Plot Variances ---------------------------------
var = res\
    .groupby(['informativeness'])\
    .agg({'BAC': ['mean', 'std']})
var.columns = ['Mean BAC', 'Std. BAC']
sns.pointplot(data = var,
              x = var.index,
              y = 'Std. BAC',
              color = '#2a9d8f')
plt.xlabel('Number of informative features')
plt.savefig('STDs_by_info.png', dpi = 300, bbox_inches = 'tight')
plt.close()
