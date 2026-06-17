# Week 1 Report

This chapter studies basic hyperparameter tuning on Fashion-MNIST with a dense neural network. The task was to move beyond one fixed model and test how hidden-layer width, depth, optimizer choice, learning rate, batch size, and training length affect validation accuracy.

The first version of the chapter used a compact grid and targeted follow-up experiments. That run tested `23` configurations and showed that a wider dense model was clearly better than the smaller baselines. The best original result came from a wider two-layer dense network trained with `Adam` for `7` epochs. It reached `82.24%` final validation accuracy, with a best intermediate validation accuracy of `83.44%`.

After that, I added an overnight search mode so the experiment could test a broader space without overwriting the original results. The exploratory overnight run completed `180` configurations and saved them separately in `results_overnight.csv`. This expanded search improved the best final validation accuracy to `87.66%`. Because the run still used short training budgets of `5`, `8`, or `12` epochs, I treat it as a stronger comparison than the first grid, but not as proof that the absolute best architecture has been found.

| Result set | Runs | Best configuration | Final validation accuracy | Best validation accuracy |
| --- | ---: | --- | ---: | ---: |
| Original grid and follow-up | `23` | wider two-layer dense model, `Adam`, `lr=0.001`, `7` epochs | `82.24%` | `83.44%` |
| Overnight grid | `180` | wider four-layer dense model, `AdamW`, `lr=0.0015`, `12` epochs | `87.66%` | `87.66%` |

The comparison changes the conclusion slightly. The original report already showed that wider dense networks were promising. The overnight run supports that direction again, but with a deeper network, more epochs, a larger batch size, and `AdamW` instead of `Adam`.

The improvement from `82.24%` to `87.66%` is about `5.42` percentage points in final validation accuracy. That is large enough to justify the broader search. At the same time, the stronger model is also heavier because it uses depth `4` and trains for `12` epochs, so the best choice depends on whether the goal is maximum accuracy or a smaller, faster experiment.

The main lesson from the saved results is that a small, structured grid was useful for finding the promising region, while the overnight search was useful for refining that region. For this chapter, the best reported model is the stronger overnight configuration, with the caveat that the experiment still compares relatively short training runs.

Find the [notebook](./notebook.ipynb), the original [results](./results.csv), the exploratory overnight [results](./results_overnight.csv), and the [reflection](./reflection.md).

[Go back to Homepage](../README.md)
