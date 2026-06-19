# Week 3 Report

This chapter studies gesture classification on the SmartWatch Gestures dataset. Each sample is a short 3-axis accelerometer sequence, so the input is a time series rather than a fixed vector. The dataloader pads sequences inside each batch, while the feature dimension stays fixed at `3`. The task has `20` gesture classes, so the models use a `20`-unit output layer with cross entropy loss.

The assignment asks whether recurrent models can be improved beyond a baseline RNN, and specifically mentions GRU, LSTM, and Conv1D. I first ran the overnight comparison on the default random file-level split. That run was useful for tuning, but it was too optimistic for the final conclusion because the same users appeared in both training and validation. I therefore reran the same `268` model configurations with a stricter user-held-out split: `U01-U06` for training, `U07` for validation, and `U08` for the final test set.

My hypothesis was that a plain RNN would struggle with this sequence task, while gated recurrent cells would solve it reliably. I expected GRU to be the safest default because it is simpler than LSTM, but I also expected LSTM to be competitive once hidden size and learning rate were tuned.

The final comparison uses `results_user_split_gpu.csv`. The validation user is used for model selection; the test user is reported only after that choice. This matters because some models had a higher test score than the selected model, but they were not the best choice by the validation rule.

| Family | Runs used | Best validation-selected run | Best val. acc. | Test acc. | Parameters | Size | GPU inference / batch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| RNN | `27` | `user-split-rnn-64-1layer-drop0.0-lr0.0005` | `27.25%` | `10.22%` | 5,716 | 24.8 KB | 0.21 ms |
| GRU | `144` | `user-split-gru-256-2layer-drop0.1-lr0.002` | `93.50%` | `92.77%` | 600,340 | 2,347.9 KB | 0.52 ms |
| LSTM | `72` | `user-split-lstm-384-3layer-drop0.0-lr0.001` | `88.25%` | `92.77%` | 2,970,644 | 11,607.2 KB | 1.38 ms |
| Conv1D | `25` | `user-split-conv1d-64-3layer-drop0.2-lr0.001` | `96.50%` | `92.27%` | 27,028 | 115.4 KB | 0.45 ms |

The plain RNN result is the clearest failure case. Even after the full sweep, the best RNN reached only `27.25%` validation accuracy and `10.22%` test accuracy. That is only slightly above random guessing in a 20-class task and nowhere near the assignment target of `90%`. The result supports the idea that ungated recurrent cells are too weak or too hard to optimise for this dataset.

GRU still solved the task much better than a plain RNN. The best GRU by validation reached `93.50%` validation accuracy and `92.77%` test accuracy. This is lower than the earlier random file-level validation result, but it is the more honest number because the test user was not seen during training or validation.

LSTM was competitive on the final test user, but it was weaker for model selection. The best LSTM by validation reached `88.25%` validation accuracy and `92.77%` test accuracy. I would not choose it as the final model because the validation score is lower than GRU and Conv1D, and the selected LSTM is also much larger at about `2.97M` parameters.

The Conv1D branch became the best final choice after the stricter split. The selected Conv1D model reached `96.50%` validation accuracy and `92.27%` test accuracy with only `27,028` parameters. It is also fast in the local GPU batch benchmark. There were Conv1D models with a higher observed test score, including one at `98.50%`, but those were not selected because their validation scores were lower. I keep the validation-selected model as the fair final result.

The old `100%` validation checkpoints are still useful as a warning. They were not caused by a direct train-validation file overlap, but the random file-level split did allow the same participants to appear in both sets. For gesture data, that makes validation easier because user-specific movement patterns can be shared across splits. The user-held-out split is therefore a better estimate of generalisation to a new person.

Overall, the chapter still meets the target, but the conclusion is more careful now. Plain RNNs are not suitable for this dataset. GRU is a strong recurrent baseline on the stricter split, but the final recommendation is the compact Conv1D model because it has the best validation score, passes the `90%` target on the unseen test user, and stays small enough to justify practically.

Find the [notebook](./notebook.ipynb), the short-run [results](./results.csv), the original random-split overnight [results](./results_overnight.csv), the final user-split GPU [results](./results_user_split_gpu.csv), the [reflection](./reflection.md), and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)
