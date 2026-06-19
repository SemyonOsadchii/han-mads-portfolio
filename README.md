# Semyon Osadchii - HAN MADS Portfolio

This repository contains my Semester 3 portfolio work for the HAN Master of Applied Data Science. The chapters document practical machine-learning experiments: tuning dense networks, tracking runs with MLflow, comparing sequence models, testing transfer learning, publishing the work with GitHub Pages, and reflecting on responsible reporting.

The live portfolio is published with GitHub Pages:

[https://semyonosadchii.github.io/han-mads-portfolio/](https://semyonosadchii.github.io/han-mads-portfolio/)

## Portfolio Contents

1. **[Week 1 - Hyperparameter Grid Search](./1-hypertuning-gridsearch/summary.md)**
   Dense neural-network tuning on Fashion-MNIST. The original grid tested `23` configurations and reached `82.24%` final validation accuracy; the broader overnight search tested `180` configurations and improved the best final result to `87.66%`.

2. **[Week 2 - MLflow Experiment Tracking](./2-hypertuning-mlflow/summary.md)**
   MLflow-tracked dense-versus-CNN comparison on Fashion-MNIST. The overnight sweep tracked `288` runs, with the best final CNN reaching `87.86%` validation accuracy.

3. **[Week 3 - Recurrent Models for Gesture Classification](./3-hypertuning-rnn/summary.md)**
   Time-series classification on the SmartWatch Gestures dataset. After rerunning the full search with a user-held-out split, plain RNNs stayed far below the target, while the final Conv1D model reached `96.50%` validation accuracy and `92.27%` test accuracy on an unseen user.

4. **[Week 4 - Transfer Learning and Tuning](./4-hypertuning-ray/summary.md)**
   Transfer-learning comparison on the ants/bees dataset. The final overnight tuning run completed `58` trials, and the best ResNet18 configuration reached `96.08%` validation accuracy, clearly ahead of the compact transformer branch.

5. **[Week 5 - Deployment](./5-deployment/summary.md)**
   GitHub Pages deployment for the static portfolio.

6. **[Ethical Reflection](./7-ethics/summary.md)**
   Reflection on responsible model reporting, reproducibility, compute use, and deployment boundaries.

## Repository Notes

Each chapter keeps its summary close to the notebook, scripts, and selected result files that support it. The repository is intended to be readable as a portfolio first: final reports and evidence are kept visible, while local experiment by-products stay out of the way.
