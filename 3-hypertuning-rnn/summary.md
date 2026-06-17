# Week 3 Report

This chapter studies gesture classification on the SmartWatch Gestures dataset. Each sample is a short 3-axis accelerometer sequence, so the input is a time series rather than a fixed vector. The dataloader pads sequences inside each batch, while the feature dimension stays fixed at `3`. The task has `20` gesture classes, so the models use a `20`-unit output layer with cross entropy loss.

The assignment asks whether recurrent models can be improved beyond a baseline RNN, and specifically mentions GRU, LSTM, and Conv1D. I ran the final overnight comparison across these model families. The saved overnight file contains `268` completed runs in total: `27` RNN, `144` GRU, `72` LSTM, and `25` Conv1D runs. The scientific comparison below uses the aggregate results from `results_overnight.csv`.

My hypothesis was that a plain RNN would struggle with this sequence task, while gated recurrent cells would solve it reliably. I expected GRU to be the safest default because it is simpler than LSTM, but I also expected LSTM to be competitive once hidden size and learning rate were tuned.

| Family | Runs used | Best run | Best val. acc. | Final val. acc. | Parameters | Size | CPU inference / batch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| RNN | `27` | `overnight-rnn-64-1layer-drop0.0-lr0.001` | `27.50%` | `25.62%` | 5,716 | 25.3 KB | 0.83 ms |
| GRU | `144` | `overnight-gru-384-2layer-drop0.1-lr0.001` | `100.00%` | `99.84%` | 1,342,868 | 5,249.6 KB | 19.93 ms |
| LSTM | `72` | `overnight-lstm-384-1layer-drop0.0-lr0.0005` | `99.84%` | `99.69%` | 605,204 | 2,367.1 KB | 2.01 ms |
| Conv1D | `25` | `overnight-conv1d-64-4layer-drop0.0-lr0.0005` | `100.00%` | `100.00%` | 39,508 | 166.4 KB | 0.94 ms |

The plain RNN result is the clearest failure case. Even after the overnight sweep, the best RNN reached only `27.50%` best validation accuracy, and the family mean was only about `16.26%`. That is better than random guessing in a 20-class task, but still nowhere near the assignment target of `90%`. The result supports the idea that ungated recurrent cells are too weak or too hard to optimise for this dataset.

GRU solved the task most robustly. The best GRU reached a perfect `100.00%` validation checkpoint and finished at `99.84%`. More importantly, high performance was not isolated to one lucky configuration. Many GRU settings with hidden sizes `128` or `256`, two or three layers, dropout between `0.0` and `0.35`, and learning rates between `0.0003` and `0.002` reached at least `99%` final validation accuracy. That makes GRU the most reliable family in this experiment.

LSTM was also strong. The best LSTM reached `99.84%` best validation accuracy and `99.69%` final validation accuracy. It did not beat the strongest GRU checkpoint, but it came very close while using fewer parameters and much lower CPU latency than the best GRU. The best GRU used about `1.34M` parameters and `19.93 ms` per CPU batch, while the best LSTM used about `605K` parameters and `2.01 ms` per CPU batch. In this local CPU benchmark, LSTM is therefore the better efficiency trade-off among the top recurrent models.

A compact GRU is still a practical option. For example, `overnight-gru-128-1layer-drop0.0-lr0.001` reached `99.38%` final validation accuracy with only `53,652` parameters and about `2.43 ms` per CPU batch. If the goal is a small model rather than the absolute best validation checkpoint, that configuration is easier to justify than the very large `384`-hidden-unit GRU.

The Conv1D branch is worth reporting, but with a caveat. It recorded only `25` configurations, so it is a smaller sweep than the GRU and LSTM comparisons. Still, the best Conv1D run reached `100.00%` final validation accuracy with far fewer parameters and lower CPU latency than the largest GRU. I therefore treat Conv1D as a strong supplementary result, while keeping GRU and LSTM as the main recurrent-model comparison.

Overall, the chapter meets the target. Plain RNNs are not suitable for this dataset, while GRU, LSTM, and the smaller Conv1D sweep all pass the `90%` threshold by a wide margin. My final recommendation is to report GRU as the most robust recurrent model family, LSTM as the strongest recurrent accuracy-efficiency trade-off, and Conv1D as a compact supplementary result that deserves a larger follow-up sweep.

Find the [notebook](./notebook.ipynb), the short-run [results](./results.csv), the overnight [results](./results_overnight.csv), the [reflection](./reflection.md), and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)
