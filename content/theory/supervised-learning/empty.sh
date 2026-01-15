#!/bin/bash

set -e

BASE="machine-learning-supervisionato"

mkdir -p "$BASE"

# Funzione helper
create() {
  mkdir -p "$(dirname "$1")"
  touch "$1"
}

# 01 - Fondamenti
create "$BASE/01-fondamenti/definizione-e-concetti.md"
create "$BASE/01-fondamenti/bias-variance-tradeoff.md"
create "$BASE/01-fondamenti/overfitting-underfitting.md"
create "$BASE/01-fondamenti/train-validation-test-split.md"
create "$BASE/01-fondamenti/cross-validation.md"

create "$BASE/01-fondamenti/metriche-valutazione/classificazione/accuracy.md"
create "$BASE/01-fondamenti/metriche-valutazione/classificazione/precision-recall-f1.md"
create "$BASE/01-fondamenti/metriche-valutazione/classificazione/confusion-matrix.md"
create "$BASE/01-fondamenti/metriche-valutazione/classificazione/roc-auc.md"
create "$BASE/01-fondamenti/metriche-valutazione/classificazione/pr-curve.md"
create "$BASE/01-fondamenti/metriche-valutazione/classificazione/metriche-multiclasse.md"

create "$BASE/01-fondamenti/metriche-valutazione/regressione/mse-rmse-mae.md"
create "$BASE/01-fondamenti/metriche-valutazione/regressione/r-squared.md"
create "$BASE/01-fondamenti/metriche-valutazione/regressione/mape.md"
create "$BASE/01-fondamenti/metriche-valutazione/regressione/residual-analysis.md"

# 02 - Preprocessing
create "$BASE/02-preprocessing/feature-scaling/standardization.md"
create "$BASE/02-preprocessing/feature-scaling/normalization.md"
create "$BASE/02-preprocessing/feature-scaling/robust-scaling.md"
create "$BASE/02-preprocessing/feature-scaling/quando-usare-quale.md"

create "$BASE/02-preprocessing/encoding/one-hot-encoding.md"
create "$BASE/02-preprocessing/encoding/label-encoding.md"
create "$BASE/02-preprocessing/encoding/ordinal-encoding.md"
create "$BASE/02-preprocessing/encoding/target-encoding.md"
create "$BASE/02-preprocessing/encoding/frequency-encoding.md"

create "$BASE/02-preprocessing/gestione-missing-values/deletion-methods.md"
create "$BASE/02-preprocessing/gestione-missing-values/imputation-methods.md"
create "$BASE/02-preprocessing/gestione-missing-values/knn-imputation.md"
create "$BASE/02-preprocessing/gestione-missing-values/advanced-imputation.md"

create "$BASE/02-preprocessing/outlier-detection/z-score-method.md"
create "$BASE/02-preprocessing/outlier-detection/iqr-method.md"
create "$BASE/02-preprocessing/outlier-detection/isolation-forest.md"
create "$BASE/02-preprocessing/outlier-detection/local-outlier-factor.md"

create "$BASE/02-preprocessing/feature-engineering/polynomial-features.md"
create "$BASE/02-preprocessing/feature-engineering/interaction-features.md"
create "$BASE/02-preprocessing/feature-engineering/binning-discretization.md"
create "$BASE/02-preprocessing/feature-engineering/log-transforms.md"
create "$BASE/02-preprocessing/feature-engineering/domain-specific-features.md"

# 03 - Feature Selection
create "$BASE/03-feature-selection/filter-methods/correlation-analysis.md"
create "$BASE/03-feature-selection/filter-methods/chi-squared-test.md"
create "$BASE/03-feature-selection/filter-methods/anova-f-test.md"
create "$BASE/03-feature-selection/filter-methods/mutual-information.md"
create "$BASE/03-feature-selection/filter-methods/variance-threshold.md"

create "$BASE/03-feature-selection/wrapper-methods/forward-selection.md"
create "$BASE/03-feature-selection/wrapper-methods/backward-elimination.md"
create "$BASE/03-feature-selection/wrapper-methods/recursive-feature-elimination.md"
create "$BASE/03-feature-selection/wrapper-methods/exhaustive-search.md"

create "$BASE/03-feature-selection/embedded-methods/lasso-regularization.md"
create "$BASE/03-feature-selection/embedded-methods/ridge-regularization.md"
create "$BASE/03-feature-selection/embedded-methods/elastic-net.md"
create "$BASE/03-feature-selection/embedded-methods/tree-based-importance.md"

create "$BASE/03-feature-selection/dimensionality-reduction/pca.md"
create "$BASE/03-feature-selection/dimensionality-reduction/lda.md"
create "$BASE/03-feature-selection/dimensionality-reduction/factor-analysis.md"
create "$BASE/03-feature-selection/dimensionality-reduction/feature-agglomeration.md"

# 04 - Algoritmi Lineari
create "$BASE/04-algoritmi-lineari/regressione-lineare/simple-linear-regression.md"
create "$BASE/04-algoritmi-lineari/regressione-lineare/multiple-linear-regression.md"
create "$BASE/04-algoritmi-lineari/regressione-lineare/assumptions.md"
create "$BASE/04-algoritmi-lineari/regressione-lineare/ols-method.md"
create "$BASE/04-algoritmi-lineari/regressione-lineare/gradient-descent.md"
create "$BASE/04-algoritmi-lineari/regressione-lineare/polynomial-regression.md"

create "$BASE/04-algoritmi-lineari/regressione-logistica/binary-logistic-regression.md"
create "$BASE/04-algoritmi-lineari/regressione-logistica/multinomial-logistic-regression.md"
create "$BASE/04-algoritmi-lineari/regressione-logistica/sigmoid-function.md"
create "$BASE/04-algoritmi-lineari/regressione-logistica/log-loss.md"
create "$BASE/04-algoritmi-lineari/regressione-logistica/decision-boundary.md"

create "$BASE/04-algoritmi-lineari/regularizzazione/ridge-l2.md"
create "$BASE/04-algoritmi-lineari/regularizzazione/lasso-l1.md"
create "$BASE/04-algoritmi-lineari/regularizzazione/elastic-net.md"
create "$BASE/04-algoritmi-lineari/regularizzazione/hyperparameter-tuning.md"

create "$BASE/04-algoritmi-lineari/variants/sgd-classifier.md"
create "$BASE/04-algoritmi-lineari/variants/perceptron.md"
create "$BASE/04-algoritmi-lineari/variants/passive-aggressive.md"

################################
# 05 - Support Vector Machines
################################
create "$BASE/05-support-vector-machines/concetti-base/hyperplane-separation.md"
create "$BASE/05-support-vector-machines/concetti-base/maximum-margin.md"
create "$BASE/05-support-vector-machines/concetti-base/support-vectors.md"
create "$BASE/05-support-vector-machines/concetti-base/soft-margin.md"

create "$BASE/05-support-vector-machines/kernel-methods/kernel-trick.md"
create "$BASE/05-support-vector-machines/kernel-methods/linear-kernel.md"
create "$BASE/05-support-vector-machines/kernel-methods/polynomial-kernel.md"
create "$BASE/05-support-vector-machines/kernel-methods/rbf-gaussian-kernel.md"
create "$BASE/05-support-vector-machines/kernel-methods/sigmoid-kernel.md"

create "$BASE/05-support-vector-machines/svm-variants/c-svc.md"
create "$BASE/05-support-vector-machines/svm-variants/nu-svc.md"
create "$BASE/05-support-vector-machines/svm-variants/svr-regression.md"
create "$BASE/05-support-vector-machines/svm-variants/one-class-svm.md"

create "$BASE/05-support-vector-machines/optimization/smo-algorithm.md"
create "$BASE/05-support-vector-machines/optimization/hyperparameter-c.md"
create "$BASE/05-support-vector-machines/optimization/hyperparameter-gamma.md"
create "$BASE/05-support-vector-machines/optimization/class-weight-balancing.md"

################################
# 06 - Decision Trees
################################
create "$BASE/06-decision-trees/fondamenti/tree-structure.md"
create "$BASE/06-decision-trees/fondamenti/splitting-criteria.md"
create "$BASE/06-decision-trees/fondamenti/gini-impurity.md"
create "$BASE/06-decision-trees/fondamenti/entropy-information-gain.md"
create "$BASE/06-decision-trees/fondamenti/variance-reduction.md"

create "$BASE/06-decision-trees/algoritmi/id3.md"
create "$BASE/06-decision-trees/algoritmi/c45.md"
create "$BASE/06-decision-trees/algoritmi/cart.md"
create "$BASE/06-decision-trees/algoritmi/chaid.md"

create "$BASE/06-decision-trees/pruning/pre-pruning.md"
create "$BASE/06-decision-trees/pruning/post-pruning.md"
create "$BASE/06-decision-trees/pruning/cost-complexity-pruning.md"
create "$BASE/06-decision-trees/pruning/reduced-error-pruning.md"

create "$BASE/06-decision-trees/hyperparameters/max-depth.md"
create "$BASE/06-decision-trees/hyperparameters/min-samples-split.md"
create "$BASE/06-decision-trees/hyperparameters/min-samples-leaf.md"
create "$BASE/06-decision-trees/hyperparameters/max-features.md"
create "$BASE/06-decision-trees/hyperparameters/criterion.md"

create "$BASE/06-decision-trees/vantaggi-svantaggi/interpretability.md"
create "$BASE/06-decision-trees/vantaggi-svantaggi/instability.md"
create "$BASE/06-decision-trees/vantaggi-svantaggi/quando-usarli.md"

################################
# 07 - Ensemble Methods
################################
create "$BASE/07-ensemble-methods/bagging/bootstrap-aggregating.md"
create "$BASE/07-ensemble-methods/bagging/random-forest/algorithm.md"
create "$BASE/07-ensemble-methods/bagging/random-forest/feature-importance.md"
create "$BASE/07-ensemble-methods/bagging/random-forest/oob-error.md"
create "$BASE/07-ensemble-methods/bagging/random-forest/hyperparameters.md"
create "$BASE/07-ensemble-methods/bagging/random-forest/extremely-randomized-trees.md"
create "$BASE/07-ensemble-methods/bagging/bagged-decision-trees.md"

create "$BASE/07-ensemble-methods/boosting/adaboost/algorithm.md"
create "$BASE/07-ensemble-methods/boosting/adaboost/weak-learners.md"
create "$BASE/07-ensemble-methods/boosting/adaboost/weight-updates.md"
create "$BASE/07-ensemble-methods/boosting/adaboost/stagewise-additive.md"

create "$BASE/07-ensemble-methods/boosting/gradient-boosting/gbm-algorithm.md"
create "$BASE/07-ensemble-methods/boosting/gradient-boosting/loss-functions.md"
create "$BASE/07-ensemble-methods/boosting/gradient-boosting/learning-rate.md"
create "$BASE/07-ensemble-methods/boosting/gradient-boosting/subsample.md"
create "$BASE/07-ensemble-methods/boosting/gradient-boosting/early-stopping.md"

create "$BASE/07-ensemble-methods/boosting/xgboost/regularization.md"
create "$BASE/07-ensemble-methods/boosting/xgboost/tree-pruning.md"
create "$BASE/07-ensemble-methods/boosting/xgboost/parallel-processing.md"
create "$BASE/07-ensemble-methods/boosting/xgboost/handling-missing-values.md"
create "$BASE/07-ensemble-methods/boosting/xgboost/custom-objectives.md"

create "$BASE/07-ensemble-methods/boosting/lightgbm/histogram-based-learning.md"
create "$BASE/07-ensemble-methods/boosting/lightgbm/goss.md"
create "$BASE/07-ensemble-methods/boosting/lightgbm/efb.md"
create "$BASE/07-ensemble-methods/boosting/lightgbm/leaf-wise-growth.md"

create "$BASE/07-ensemble-methods/boosting/catboost/ordered-boosting.md"
create "$BASE/07-ensemble-methods/boosting/catboost/categorical-features.md"
create "$BASE/07-ensemble-methods/boosting/catboost/prediction-shift.md"
create "$BASE/07-ensemble-methods/boosting/catboost/symmetric-trees.md"

create "$BASE/07-ensemble-methods/boosting/histogram-gradient-boosting.md"

create "$BASE/07-ensemble-methods/stacking/multi-level-stacking.md"
create "$BASE/07-ensemble-methods/stacking/meta-learner.md"
create "$BASE/07-ensemble-methods/stacking/blending.md"
create "$BASE/07-ensemble-methods/stacking/feature-weighted-stacking.md"

create "$BASE/07-ensemble-methods/voting/hard-voting.md"
create "$BASE/07-ensemble-methods/voting/soft-voting.md"
create "$BASE/07-ensemble-methods/voting/weighted-voting.md"

create "$BASE/07-ensemble-methods/teoria/bias-variance-ensemble.md"
create "$BASE/07-ensemble-methods/teoria/diversity-importance.md"
create "$BASE/07-ensemble-methods/teoria/ensemble-size.md"

################################
# 08 - Nearest Neighbors
################################
create "$BASE/08-nearest-neighbors/knn/algorithm.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/euclidean.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/manhattan.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/minkowski.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/chebyshev.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/hamming.md"
create "$BASE/08-nearest-neighbors/knn/distance-metrics/cosine-similarity.md"
create "$BASE/08-nearest-neighbors/knn/k-selection.md"
create "$BASE/08-nearest-neighbors/knn/weighted-voting.md"
create "$BASE/08-nearest-neighbors/knn/curse-of-dimensionality.md"

create "$BASE/08-nearest-neighbors/optimization/kd-trees.md"
create "$BASE/08-nearest-neighbors/optimization/ball-trees.md"
create "$BASE/08-nearest-neighbors/optimization/locality-sensitive-hashing.md"
create "$BASE/08-nearest-neighbors/optimization/approximate-neighbors.md"

create "$BASE/08-nearest-neighbors/variants/radius-neighbors.md"
create "$BASE/08-nearest-neighbors/variants/nearest-centroid.md"
create "$BASE/08-nearest-neighbors/variants/large-margin-knn.md"

create "$BASE/08-nearest-neighbors/applications/classification.md"
create "$BASE/08-nearest-neighbors/applications/regression.md"
create "$BASE/08-nearest-neighbors/applications/anomaly-detection.md"

################################
# 09 - Naive Bayes
################################
create "$BASE/09-naive-bayes/teoria/bayes-theorem.md"
create "$BASE/09-naive-bayes/teoria/independence-assumption.md"
create "$BASE/09-naive-bayes/teoria/prior-likelihood-posterior.md"
create "$BASE/09-naive-bayes/teoria/laplace-smoothing.md"

create "$BASE/09-naive-bayes/variants/gaussian-naive-bayes.md"
create "$BASE/09-naive-bayes/variants/multinomial-naive-bayes.md"
create "$BASE/09-naive-bayes/variants/bernoulli-naive-bayes.md"
create "$BASE/09-naive-bayes/variants/complement-naive-bayes.md"

create "$BASE/09-naive-bayes/applications/text-classification.md"
create "$BASE/09-naive-bayes/applications/spam-detection.md"
create "$BASE/09-naive-bayes/applications/sentiment-analysis.md"

create "$BASE/09-naive-bayes/limitations/feature-correlation.md"
create "$BASE/09-naive-bayes/limitations/zero-frequency-problem.md"

################################
# 10 - Perceptron
################################
create "$BASE/10-perceptron/single-layer-perceptron.md"
create "$BASE/10-perceptron/activation-functions-basic.md"
create "$BASE/10-perceptron/limitations.md"

################################
# 11 - Class Imbalance
################################
create "$BASE/11-class-imbalance/detection/class-distribution.md"
create "$BASE/11-class-imbalance/detection/imbalance-ratio.md"

create "$BASE/11-class-imbalance/resampling/oversampling/random-oversampling.md"
create "$BASE/11-class-imbalance/resampling/oversampling/smote.md"
create "$BASE/11-class-imbalance/resampling/oversampling/adasyn.md"
create "$BASE/11-class-imbalance/resampling/oversampling/borderline-smote.md"
create "$BASE/11-class-imbalance/resampling/oversampling/smote-variants.md"

create "$BASE/11-class-imbalance/resampling/undersampling/random-undersampling.md"
create "$BASE/11-class-imbalance/resampling/undersampling/tomek-links.md"
create "$BASE/11-class-imbalance/resampling/undersampling/enn.md"
create "$BASE/11-class-imbalance/resampling/undersampling/ncr.md"
create "$BASE/11-class-imbalance/resampling/undersampling/cluster-centroids.md"

create "$BASE/11-class-imbalance/resampling/combined-methods/smote-tomek.md"
create "$BASE/11-class-imbalance/resampling/combined-methods/smote-enn.md"

create "$BASE/11-class-imbalance/algorithmic-approaches/class-weights.md"
create "$BASE/11-class-imbalance/algorithmic-approaches/cost-sensitive-learning.md"
create "$BASE/11-class-imbalance/algorithmic-approaches/threshold-moving.md"
create "$BASE/11-class-imbalance/algorithmic-approaches/ensemble-balancing.md"

create "$BASE/11-class-imbalance/evaluation/stratified-sampling.md"
create "$BASE/11-class-imbalance/evaluation/appropriate-metrics.md"
create "$BASE/11-class-imbalance/evaluation/confusion-matrix-analysis.md"

################################
# 12 - Hyperparameter Tuning
################################
create "$BASE/12-hyperparameter-tuning/search-strategies/grid-search.md"
create "$BASE/12-hyperparameter-tuning/search-strategies/random-search.md"
create "$BASE/12-hyperparameter-tuning/search-strategies/bayesian-optimization.md"
create "$BASE/12-hyperparameter-tuning/search-strategies/genetic-algorithms.md"
create "$BASE/12-hyperparameter-tuning/search-strategies/hyperband.md"
create "$BASE/12-hyperparameter-tuning/search-strategies/optuna-frameworks.md"

create "$BASE/12-hyperparameter-tuning/validation-strategies/k-fold-cv.md"
create "$BASE/12-hyperparameter-tuning/validation-strategies/stratified-k-fold.md"
create "$BASE/12-hyperparameter-tuning/validation-strategies/time-series-cv.md"
create "$BASE/12-hyperparameter-tuning/validation-strategies/nested-cv.md"

create "$BASE/12-hyperparameter-tuning/best-practices/search-space-definition.md"
create "$BASE/12-hyperparameter-tuning/best-practices/computational-budget.md"
create "$BASE/12-hyperparameter-tuning/best-practices/early-stopping-strategies.md"
create "$BASE/12-hyperparameter-tuning/best-practices/parallel-search.md"

create "$BASE/12-hyperparameter-tuning/tools/sklearn-gridsearch.md"
create "$BASE/12-hyperparameter-tuning/tools/optuna.md"
create "$BASE/12-hyperparameter-tuning/tools/hyperopt.md"
create "$BASE/12-hyperparameter-tuning/tools/ray-tune.md"

################################
# 13 - Interpretability
################################
create "$BASE/13-interpretability/model-agnostic/feature-importance/permutation-importance.md"
create "$BASE/13-interpretability/model-agnostic/feature-importance/drop-column-importance.md"
create "$BASE/13-interpretability/model-agnostic/partial-dependence-plots.md"
create "$BASE/13-interpretability/model-agnostic/ice-plots.md"
create "$BASE/13-interpretability/model-agnostic/lime.md"

create "$BASE/13-interpretability/model-agnostic/shap/shapley-values.md"
create "$BASE/13-interpretability/model-agnostic/shap/kernel-shap.md"
create "$BASE/13-interpretability/model-agnostic/shap/tree-shap.md"
create "$BASE/13-interpretability/model-agnostic/shap/deep-shap.md"

create "$BASE/13-interpretability/model-agnostic/anchor-explanations.md"

create "$BASE/13-interpretability/model-specific/linear-coefficients.md"
create "$BASE/13-interpretability/model-specific/tree-visualization.md"
create "$BASE/13-interpretability/model-specific/rule-extraction.md"
create "$BASE/13-interpretability/model-specific/attention-mechanisms.md"

create "$BASE/13-interpretability/global-interpretability/feature-importance-ranking.md"
create "$BASE/13-interpretability/global-interpretability/interaction-effects.md"
create "$BASE/13-interpretability/global-interpretability/surrogate-models.md"

create "$BASE/13-interpretability/local-interpretability/instance-explanations.md"
create "$BASE/13-interpretability/local-interpretability/counterfactual-explanations.md"
create "$BASE/13-interpretability/local-interpretability/prototypes-criticisms.md"

################################
# 14 - Advanced Topics
################################
create "$BASE/14-advanced-topics/active-learning/uncertainty-sampling.md"
create "$BASE/14-advanced-topics/active-learning/query-by-committee.md"
create "$BASE/14-advanced-topics/active-learning/expected-model-change.md"

create "$BASE/14-advanced-topics/semi-supervised-learning/self-training.md"
create "$BASE/14-advanced-topics/semi-supervised-learning/co-training.md"
create "$BASE/14-advanced-topics/semi-supervised-learning/label-propagation.md"

create "$BASE/14-advanced-topics/multi-task-learning/hard-parameter-sharing.md"
create "$BASE/14-advanced-topics/multi-task-learning/soft-parameter-sharing.md"

create "$BASE/14-advanced-topics/online-learning/incremental-learning.md"
create "$BASE/14-advanced-topics/online-learning/streaming-data.md"
create "$BASE/14-advanced-topics/online-learning/concept-drift-handling.md"

create "$BASE/14-advanced-topics/fairness-ethics/algorithmic-bias.md"
create "$BASE/14-advanced-topics/fairness-ethics/fairness-metrics.md"
create "$BASE/14-advanced-topics/fairness-ethics/debiasing-techniques.md"
create "$BASE/14-advanced-topics/fairness-ethics/privacy-preservation.md"

################################
# Workflow Orchestration
################################
create "$BASE/14-advanced-topics/workflow-orchestration/airflow.md"
create "$BASE/14-advanced-topics/workflow-orchestration/kubeflow.md"
create "$BASE/14-advanced-topics/workflow-orchestration/prefect.md"

echo "✅ TUTTA la struttura è stata creata correttamente in '$BASE'"

