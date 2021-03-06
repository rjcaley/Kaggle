import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy.cluster import hierarchy
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import dataset and check all of it for missing data and data types
raw_data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print("Missing Data Proportion")
print(pd.isnull(raw_data).mean())
print("Dataset Info")
print(raw_data.info())

# Set aside a test set.
# Remove the 'time' variable since we cannot know in advance when the next follow-up will be.
data = raw_data.copy()
data.drop('time', axis=1, inplace=True)
X, y = data.loc[:, data.columns != 'DEATH_EVENT'], data['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# EDA SECTION
training_data = pd.concat([x_train, y_train], axis=1)
eda_scaler = StandardScaler()
scaled_training_data = eda_scaler.fit_transform(training_data)

# Binary variables: anaemia, diabetes, high_blood_pressure, sex, smoking, DEATH_EVENT
# Continuous variables: age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine,
# Continuous variables (cont'd): serum_sodium

# Crosstab
x_tab_anaemia = pd.crosstab(training_data['anaemia'], training_data['DEATH_EVENT'], normalize='index')
print(x_tab_anaemia)

x_tab_diabetes = pd.crosstab(training_data['diabetes'], training_data['DEATH_EVENT'], normalize='index')
print(x_tab_diabetes)

x_tab_hbp = pd.crosstab(training_data['high_blood_pressure'], training_data['DEATH_EVENT'], normalize='index')
print(x_tab_hbp)
# Note: Not having high blood was a decent indicator of survival

x_tab_sex = pd.crosstab(training_data['sex'], training_data['DEATH_EVENT'], normalize='index')
print(x_tab_sex)

x_tab_smoking = pd.crosstab(training_data['smoking'], training_data['DEATH_EVENT'], normalize='index')
print(x_tab_smoking)

# Correlation Matrix
scatter_matrix = px.scatter_matrix(training_data)
#scatter_matrix.show()

# Plots
fig_sex = px.histogram(training_data, x='sex', color='DEATH_EVENT')
#fig_sex.show()
fig_age = px.histogram(training_data, x='age', color='DEATH_EVENT')
fig_serum = px.histogram(training_data, x='serum_creatinine', color='DEATH_EVENT')
fig_eject = px.histogram(training_data, x='ejection_fraction', color='DEATH_EVENT')
#fig_eject.show()
fig_serum_eject = px.scatter(training_data, x='serum_creatinine', y='ejection_fraction', color='DEATH_EVENT')
fig_serum_eject.update_traces(marker=dict(size=12,
                                          line=dict(width=2,
                                                    color='DarkSlateGrey')),
                              selector=dict(mode='markers'))
# fig_serum_eject.show()

# PCA (without label)
pca_x = training_data.drop('DEATH_EVENT', axis=1)
pca_scaler = StandardScaler()
pca_x = pca_scaler.fit_transform(pca_x)
pca_list = []
for x in range(1, (pca_x.shape[1]) + 1):
    pca = PCA(n_components=x)
    pca.fit(pca_x)
    pca_list.append(pca.explained_variance_ratio_.sum())

pca_plot = pd.DataFrame({'PCA Components': range(1, (pca_x.shape[1]) + 1),
                         'Explained Variance': pca_list})
pca_fig = px.line(pca_plot, x='PCA Components', y='Explained Variance')
# pca_fig.show()


# MODELS
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

# Review Hierarchical Clustering & Correlation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(x_train_std).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=x_test.columns.tolist(), ax=ax1, leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()


def grid_search(model, params, x_data=x_train_std, y_data=y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gcv = GridSearchCV(estimator=model,
                       param_grid=params,
                       scoring=['accuracy', 'f1'],
                       cv=skf,
                       refit='accuracy')
    gcv.fit(x_data, y_data)
    gcv_ser = pd.DataFrame(gcv.cv_results_).loc[gcv.best_index_, :]

    return gcv, gcv_ser


# Create Focus Features DataFrame
focus_features = ['ejection_fraction', 'serum_creatinine']
focus_x_train = x_train.copy()[focus_features]
focus_x_test = x_test.copy()[focus_features]

focus_scaler = StandardScaler()
focus_x_train = focus_scaler.fit_transform(focus_x_train)
focus_x_test = focus_scaler.transform(focus_x_test)

# Logistic Regression - All features
log_reg = LogisticRegression(random_state=42)
log_param_grid = dict(C=np.arange(0.005, 0.1, 0.001), class_weight=[None, 'Balanced'])
log_param_grid_best = dict(C=[0.088])

log_gcv, log_gcv_ser = grid_search(log_reg, log_param_grid_best)

# Logistic Regression Sub-section - plotting results in 3d
log_preds = log_gcv.predict(x_train_std).tolist()
log_pred_probs = log_gcv.predict_proba(x_train_std).tolist()
winning_probs = [p[i] for p, i in zip(log_pred_probs, log_preds)]
log_pca = PCA(n_components=3)
pca_x = log_pca.fit_transform(pca_x)
exp_var = log_pca.explained_variance_ratio_.sum()
log_df = pd.DataFrame(pca_x, columns=[1, 2, 3])
log_df['Prediction'] = log_preds
log_df['Probability'] = winning_probs
log_df = pd.concat([log_df, training_data['DEATH_EVENT'].reset_index()], axis=1)
log_df['Prediction Check'] = ['correct' if actual == pred else 'incorrect' for
                              actual, pred in zip(log_df['DEATH_EVENT'], log_df['Prediction'])]

log_fig = px.scatter_3d(log_df, x=1, y=2, z=3,
                        color='Probability',
                        symbol='Prediction Check')
log_fig.update_layout(title_text='PCA Axes Explain ' + str(round(exp_var * 100, 1)) + '% of Total Variance',
                      legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
#log_fig.show()


# Logistic Regression - Focus Features
log2_reg = LogisticRegression(random_state=42)
log2_param_grid = dict(C=np.arange(0.1, 1.6, 0.1), class_weight=[None, 'Balanced'])
log2_param_grid_best = dict(C=[0.6])

log2_gcv, log2_gcv_ser = grid_search(log2_reg, log2_param_grid_best, focus_x_train)

# SVC - All Features
svc = SVC(probability=True, random_state=42)
svc_param_grid = dict(C=np.arange(0.5, 2.1, 0.1), kernel=['rbf', 'linear', 'poly', 'sigmoid'], degree=np.arange(1, 5),
                      class_weight=[None, 'balanced'])

svc_param_grid_best = dict(C=[1.4], class_weight=[None], degree=[1], kernel=['sigmoid'])
svc_gcv, svc_gcv_ser = grid_search(svc, svc_param_grid_best)

# SVC - Focus Features
svc2 = SVC(probability=True, random_state=42)
svc2_param_grid = dict(C=np.arange(0.05, 0.16, 0.01), kernel=['rbf', 'linear', 'poly', 'sigmoid'],
                       degree=np.arange(1, 5), class_weight=[None, 'balanced'])

svc2_param_grid_best = dict(C=[0.09], class_weight=['balanced'], degree=[1], kernel=['linear'])

svc2_gcv, svc2_gcv_ser = grid_search(svc2, svc2_param_grid_best, focus_x_train)

# KNN - All Features
knn = KNeighborsClassifier()
knn_param_grid = dict(n_neighbors=np.arange(3, 23, 2), weights=['uniform', 'distance'],
                      algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], p=[1, 2])

knn_param_grid_best = dict(algorithm=['auto'], n_neighbors=[11], p=[1], weights=['uniform'])

knn_gcv, knn_gcv_ser = grid_search(knn, knn_param_grid_best)

# KNN - Focus Features
knn2 = KNeighborsClassifier()
knn2_param_grid = dict(n_neighbors=np.arange(3, 23, 2), weights=['uniform', 'distance'],
                       algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], p=[1, 2])

knn2_param_grid_best = dict(algorithm=['auto'], n_neighbors=[13], p=[1], weights=['distance'])

knn2_gcv, knn2_gcv_ser = grid_search(knn2, knn2_param_grid_best, focus_x_train)

# Random Forest - All Features
rf = RandomForestClassifier(n_jobs=-1, random_state=921)
rf_param_grid = dict(n_estimators=[10, 15, 25, 50, 75, 100], max_depth=[None, 1, 2, 3, 4, 5],
                     max_features=['auto', 'log2', 'sqrt'], min_samples_leaf=[2, 3, 4, 5],
                     class_weight=[None, 'balanced', 'balanced_subsample'])

rf_param_grid_best = dict(class_weight=['balanced'], max_depth=[3], max_features=['auto'], min_samples_leaf=[3],
                          n_estimators=[15])

rf_gcv, rf_gcv_ser = grid_search(rf, rf_param_grid_best)

# Random Forest - Focus Features
rf2 = RandomForestClassifier(n_jobs=-1, random_state=921)
rf2_param_grid = dict(n_estimators=[10, 15, 20, 25, 30], max_depth=[None, 5, 6, 7, 10, 15],
                      max_features=['auto', 'log2', 'sqrt'],
                      min_samples_leaf=[2, 3, 4], class_weight=[None, 'balanced', 'balanced_subsample'])

rf2_param_grid_best = dict(class_weight=[None], max_depth=[7], max_features=['auto'], min_samples_leaf=[3],
                           n_estimators=[15])

rf2_gcv, rf2_gcv_ser = grid_search(rf2, rf2_param_grid_best, focus_x_train)

# Gradient Boosting Classifier - All Features
gbc = GradientBoostingClassifier(random_state=42)

gbc_param_grid = dict(learning_rate=[.05, .06, .08, .09, .1, .25], n_estimators=[50, 75, 100, 125, 150, 200],
                      subsample=[0.5, 0.7, 0.8, 0.9, 1.0], min_samples_leaf=[6, 8, 10, 12, 15],
                      max_depth=[1, 2, 3, 4, 5, 6, 7], max_features=['auto', 'sqrt'], n_iter_no_change=[None, 5])

gbc_param_grid_best = dict(learning_rate=[0.08], max_depth=[4], max_features=['sqrt'], min_samples_leaf=[10],
                           n_estimators=[100], subsample=[1], n_iter_no_change=[None])

gbc_gcv, gbc_gcv_ser = grid_search(gbc, gbc_param_grid_best)

# Gradient Boosting Classifier - Focus Features
gbc2 = GradientBoostingClassifier(random_state=42)
gbc2_param_grid = dict(learning_rate=np.arange(0.2, 0.8, 0.1), n_estimators=[15, 25, 35, 50, 75],
                       subsample=np.arange(0.2, 1.0, 0.1), min_samples_leaf=np.arange(3, 10, 1),
                       max_depth=[1, 2, 3, 4, 5, 6, 7, 10, 12], max_features=['auto', 'sqrt'],
                       n_iter_no_change=[None, 5])

gbc2_param_grid_best = dict(learning_rate=[0.4], max_depth=[5], max_features=['auto'], min_samples_leaf=[6],
                            n_estimators=[35], subsample=[0.7], n_iter_no_change=[None])

gbc2_gcv, gbc2_gcv_ser = grid_search(gbc, gbc2_param_grid_best, focus_x_train)

# Build a consolidated results table
res_df = pd.DataFrame(dict(log_all=log_gcv_ser, log_foc=log2_gcv_ser,
                           svc_all=svc_gcv_ser, svc_foc=svc2_gcv_ser,
                           knn_all=knn_gcv_ser, knn_foc=knn2_gcv_ser,
                           rf_all=rf_gcv_ser, rf_foc=rf2_gcv_ser,
                           gbc_all=gbc_gcv_ser, gbc_foc=gbc2_gcv_ser)).T

# Create Box Plots for F1 and Accuracy Scores of Models in Cross Validation
f1_res_cols = res_df.columns[(res_df.columns.str.startswith('split')) & (res_df.columns.str.endswith('f1'))]
acc_res_cols = res_df.columns[(res_df.columns.str.startswith('split')) & (res_df.columns.str.endswith('accuracy'))]

# F1 Box Plots
f1_res = res_df.copy()[f1_res_cols].reset_index()
f1_res_melted = pd.melt(f1_res, id_vars=['index']).drop('variable', axis=1)
f1_res_melted['Model'], f1_res_melted['Variable Set'] = list(zip(*[x.split("_") for x in f1_res_melted['index']]))
fig = px.box(f1_res_melted, x='Model', y='value', color='Variable Set')
fig.update_layout(title_text='F1 Scores', title_x=0.5)
fig.show()

# Accuracy Box Plots
acc_res = res_df.copy()[acc_res_cols].reset_index()
acc_res_melted = pd.melt(acc_res, id_vars=['index']).drop('variable', axis=1)
acc_res_melted['Model'], acc_res_melted['Variable Set'] = list(zip(*[x.split("_") for x in acc_res_melted['index']]))
fig = px.box(acc_res_melted, x='Model', y='value', color='Variable Set')
fig.update_layout(title_text='Accuracy Scores', title_x=0.5)
fig.show()

# Probability Investigation
test_preds_all = []
test_probs_all = []
test_pred_checks_all = []

for mod in [log_gcv, svc_gcv, knn_gcv, rf_gcv, gbc_gcv]:
    preds = mod.predict(x_test_std).tolist()
    probs = mod.predict_proba(x_test_std)
    probs = [p[i] for p, i in zip(probs, preds)]
    pred_checks = [1 if actual == pred else 0 for actual, pred in zip(y_test, preds)]

    test_preds_all = test_preds_all + preds
    test_probs_all = test_probs_all + probs
    test_pred_checks_all = test_pred_checks_all + pred_checks

test_preds_foc = []
test_probs_foc = []
test_pred_checks_foc = []

for mod in [log2_gcv, svc2_gcv, knn2_gcv, rf2_gcv, gbc2_gcv]:
    preds = mod.predict(focus_x_test).tolist()
    probs = mod.predict_proba(focus_x_test)
    probs = [p[i] for p, i in zip(probs, preds)]
    pred_checks = [1 if actual == pred else 0 for actual, pred in zip(y_test, preds)]

    test_preds_foc = test_preds_foc + preds
    test_probs_foc = test_probs_foc + probs
    test_pred_checks_foc = test_pred_checks_foc + pred_checks

mod_list = ['All' for _ in test_preds_all] + ['Focus' for _ in test_preds_foc]

test_prob_df = pd.DataFrame(dict(Prediction=test_preds_all + test_preds_foc,
                                 Probability=test_probs_all + test_probs_foc,
                                 Check=test_pred_checks_all + test_pred_checks_foc,
                                 Variable_Selection=mod_list))

test_prob_df['Binned Probabilities'] = pd.cut(test_prob_df['Probability'], bins=12)
test_prob_grouped = test_prob_df.groupby(['Binned Probabilities', 'Variable_Selection']).mean()['Check'].reset_index()
test_prob_grouped.rename(columns={'Check': 'Accuracy'}, inplace=True)
test_prob_grouped['Expected'] = [x.mid for x in test_prob_grouped['Binned Probabilities']]
test_prob_grouped['Binned Probabilities'] = [str(x) for x in test_prob_grouped['Binned Probabilities']]

fig = px.line(test_prob_grouped, x='Binned Probabilities', y='Accuracy', color='Variable_Selection')
fig.add_trace(go.Scatter(x=test_prob_grouped['Binned Probabilities'],
                         y=test_prob_grouped['Expected'],
                         mode='lines',
                         line=go.scatter.Line(color='gray'),
                         name='Expected',
                         opacity=0.4))
fig.update_traces(mode='markers+lines')
fig.show()


# Statistical Significance - Monte Carlo
def monte_carlo_sim(y_true, y_preds):
    rg = Generator(PCG64(42))
    mod_acc = accuracy_score(y_true, y_preds)
    mod_f1 = f1_score(y_true, y_preds)
    shuffles = [rg.permutation(y_preds) for _ in range(5000)]
    shuffles_acc = [accuracy_score(y_true, y_shuff) for y_shuff in shuffles]
    shuffles_f1 = [f1_score(y_true, y_shuff) for y_shuff in shuffles]
    perct_acc = np.percentile(shuffles_acc, 95)
    perct_f1 = np.percentile(shuffles_f1, 95)
    return dict(model_accuracy=mod_acc, mc_accuracy=perct_acc, model_f1=mod_f1, mc_f1=perct_f1)


log_mc = monte_carlo_sim(y_test, log_gcv.predict(x_test_std))
svc_mc = monte_carlo_sim(y_test, svc_gcv.predict(x_test_std))
knn_mc = monte_carlo_sim(y_test, knn_gcv.predict(x_test_std))
rf_mc = monte_carlo_sim(y_test, rf_gcv.predict(x_test_std))
gbc_mc = monte_carlo_sim(y_test, gbc_gcv.predict(x_test_std))

log2_mc = monte_carlo_sim(y_test, log2_gcv.predict(focus_x_test))
svc2_mc = monte_carlo_sim(y_test, svc2_gcv.predict(focus_x_test))
knn2_mc = monte_carlo_sim(y_test, knn2_gcv.predict(focus_x_test))
rf2_mc = monte_carlo_sim(y_test, rf2_gcv.predict(focus_x_test))
gbc2_mc = monte_carlo_sim(y_test, gbc2_gcv.predict(focus_x_test))

mc_all_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Accuracy MC 95th', 'F1', 'F1 MC 95th'])
for i, (s, mod) in enumerate([('log_all', log_mc), ('svc_all', svc_mc), ('knn_all', knn_mc),
                              ('rf_all', rf_mc), ('gbc_all', gbc_mc)]):
    mc_all_df.loc[i, :] = [s, mod['model_accuracy'], mod['mc_accuracy'], mod['model_f1'], mod['mc_f1']]

mc_foc_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Accuracy MC 95th', 'F1', 'F1 MC 95th'])
for i, (s, mod) in enumerate([('log_foc', log2_mc), ('svc_foc', svc2_mc), ('knn_foc', knn2_mc),
                              ('rf_foc', rf2_mc), ('gbc_foc', gbc2_mc)]):
    mc_foc_df.loc[i, :] = [s, mod['model_accuracy'], mod['mc_accuracy'], mod['model_f1'], mod['mc_f1']]

