# Simon Osadchii - HAN MADS Portfolio

This portfolio collects my Semester 3 work for the HAN Master of Applied Data Science programme. The project is built around practical machine-learning experiments: tuning models, tracking runs, comparing architectures, reporting results, and publishing the work as a small static website.

The live portfolio is published with GitHub Pages: [simonosadchii.github.io/han-mads-portfolio](https://simonosadchii.github.io/han-mads-portfolio/).

## Portfolio Contents

1. **[Week 1 - Hyperparameter Grid Search](./1-hypertuning-gridsearch/summary.md)**
   A Fashion-MNIST dense-network tuning study. The first structured grid reached `82.24%` final validation accuracy, and the exploratory overnight run improved the best final result to `87.66%`.

1. **[Week 2 - MLflow Experiment Tracking](./2-hypertuning-mlflow/summary.md)**
   A reproducible dense-versus-CNN comparison on Fashion-MNIST using MLflow. The overnight sweep tracked `288` runs, with the best final CNN reaching `87.86%` validation accuracy.

1. **[Week 3 - Recurrent Models for Gesture Classification](./3-hypertuning-rnn/summary.md)**
   A time-series classification experiment on SmartWatch Gestures. Plain RNNs failed to reach the target, while GRU and LSTM models passed it by a wide margin, with the best GRU finishing at `99.84%` validation accuracy.

1. **[Week 4 - Ray Tune and Transfer Learning](./4-hypertuning-ray/summary.md)**
   A Ray Tune comparison between pretrained ResNet18 and a compact transformer on the ants/bees image dataset. ResNet18 was the clear accuracy winner with `93.46%` best validation accuracy, while the transformer branch was smaller and faster but much less accurate.

1. **[Week 5 - Deployment](./5-deployment/summary.md)**
   A static deployment of this portfolio with GitHub Pages. The deployment focuses on making the reports, notebooks, results, and supporting files easy to navigate without running a local server.

1. **[Hackathon - Project Cuddlefish](./6-hackathon/project_cuddlefish.md)**
   A playful image-to-image machine-learning concept using a U-Net style model to generate camouflage patterns from environmental textures.

1. **[Ethical Reflection](./7-ethics/summary.md)**
   A reflection on responsible AI, using the Cuddlefish concept to discuss unintended consequences, responsibility, evaluation, and deployment boundaries.

## Repository Notes

The repository contains the source notebooks, result CSV files, experiment summaries, report artifacts, and deployment configuration used for the portfolio. The website itself is static: GitHub Pages renders the Markdown files from this repository, while the code and experiment outputs remain available for review in the linked chapter folders.

