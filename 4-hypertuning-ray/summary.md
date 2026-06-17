# Week 4 Summary

This chapter tests image-classification hyperparameter tuning on the PyTorch ants/bees dataset. The dataset is small, with `244` training images and `153` validation images, so I framed the work around a theory-based hypothesis: a pretrained convolutional backbone should outperform a compact transformer trained from scratch.

The final experiment is based on the completed overnight tuning run, not the earlier partial Ray result. The comparison contains `58` completed trials: `32` ResNet18 transfer-learning configurations and `26` compact transformer configurations. Each run used `224x224` images and up to `20` epochs. The search covered architecture choices, classifier head size/depth, dropout, activation, optimizer, learning rate, batch size, weight decay, label smoothing, and unfreezing depth.

The main result is clear. The best ResNet18 run, `local_resnet18_031`, reached `96.08%` validation accuracy with validation loss `0.2536`. The best transformer, `local_transformer_024`, reached `77.78%`. The family-level averages tell the same story: ResNet18 averaged `90.44%` validation accuracy across `32` trials, while the transformer family averaged `72.17%` across `26` trials.

The winning ResNet18 configuration used `SGD`, learning rate `0.0005`, batch size `8`, a two-layer `768`-unit head, `GELU`, no dropout, three unfrozen backbone blocks, weight decay `0.0001`, and label smoothing `0.05`. I interpret that as evidence for partial adaptation: the pretrained backbone remains valuable, but the best run still benefits from a wider head and some unfreezing.

The compact transformer branch was still useful as a contrast. Its best model was much smaller, about `5.42 MB` versus about `46.47 MB` for the best ResNet18. However, the accuracy loss is too large for this assignment, so compactness does not compensate for the weaker validation result.

There were also practical engineering lessons. The first Ray-based attempt was useful for shaping the search space, but the final run needed a more controlled local backend with resume support. Saving progress after every trial made the experiment easier to recover after interruptions and reduced the risk of losing a long run.

Overall, the final conclusion is stronger than the earlier draft: ResNet18 transfer learning is the right default model family for this dataset. The full report keeps the main figures and the final CUDA result files close to the implementation, while generated databases and temporary tuning outputs remain local artefacts.

Find the full [report](./report.md), the final CUDA overnight [results](./results_cuda_overnight.csv), the [best configuration](./best_config_cuda_overnight.json), the [search script](./hypertune.py), the shared [tuning helpers](./tuning_common.py), and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)
