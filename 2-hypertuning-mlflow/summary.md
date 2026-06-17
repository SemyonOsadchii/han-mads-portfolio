# Week 2 Report

This chapter uses MLflow to compare dense and convolutional neural-network configurations on Fashion-MNIST. The assignment was not just to train one model, but to track several hypotheses and make the comparison reproducible. I kept the first short run in `results.csv` as a baseline, then added a larger overnight sweep in `results_overnight.csv`.

My starting hypothesis was that batch normalisation would make training more stable, that dropout would help only when the model had enough capacity, and that a CNN should eventually have an advantage because the input is image data. The short run did not prove that yet. After two epochs, the best model was still the dense `dense_batchnorm` baseline with `80.16%` validation accuracy, while the first CNN attempts were clearly undertrained.

The overnight run gives a more useful picture. It contains `288` tracked runs: `160` dense configurations and `128` CNN configurations. This sweep varied architecture, dropout, batch normalisation, and learning rate. The best final validation score came from a CNN with three convolution blocks, batch normalisation, dropout `0.1`, and learning rate `0.001`. It reached `87.86%` final validation accuracy and `88.53%` as its best validation accuracy during training.

There is one small nuance in the numbers: the highest single validation point was `88.62%`, also from a batch-normalised CNN, but that run finished lower at `86.70%`. For that reason I treat `87.86%` as the main result, because it was the strongest final model rather than only the highest temporary checkpoint.

| Question | Result | Interpretation |
| --- | --- | --- |
| Best short-run baseline | `80.16%` validation accuracy | Useful starting point, but too small for a final architecture conclusion. |
| Best final overnight model | `87.86%` final validation accuracy | A tuned CNN became the strongest final model. |
| Best dense fallback | `87.50%` final validation accuracy | Dense models stayed competitive and were more stable across the grid. |
| Batch normalisation effect | `80.17%` average final accuracy with batch norm vs. `76.98%` without it | This was the most reliable improvement in the sweep. |

The main conclusion changed, but not dramatically. CNNs did not fail in the first run because they were the wrong model family; they needed a better search over channels, learning rate, and regularisation. At the same time, the dense models should not be dismissed. Across the full overnight grid, dense runs averaged `84.44%` final validation accuracy, while CNN runs averaged `71.24%`. That tells me the CNN had the higher ceiling, but it was also easier to configure badly.

Dropout was less clear than batch normalisation. The final winning CNN used dropout `0.1`, but the highest temporary validation score came from a similar CNN without dropout. I would therefore not describe dropout as automatically good or bad here. It helped in some combinations, but it depended on the rest of the configuration.

Overall, the best choice for this experiment is the tuned CNN, with the tuned dense model as a credible fallback. The margin over the best dense run is only about `0.36` percentage points in final validation accuracy, so I would not overstate the result. What I can say is that the larger MLflow sweep turned the first rough comparison into a cleaner story: batch normalisation was consistently useful, CNNs needed more careful tuning, and MLflow made it possible to compare those decisions without losing track of the evidence.

Find the [notebook](./notebook.ipynb), the short-run [results](./results.csv), the overnight [results](./results_overnight.csv), the [reflection](./reflection.md), and the [instructions](./instructions.md).

[Go back to Homepage](../README.md)
