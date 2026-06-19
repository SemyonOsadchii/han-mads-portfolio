# Reflection

For this chapter I wanted to understand whether a simple dense network on Fashion-MNIST would benefit more from extra width, extra depth, or longer training. My first hypothesis was that increasing the number of hidden units would help at the start, but that after a certain point the gains would become small. I also expected that training for more epochs would be more useful than just stacking another hidden layer, because this is still a relatively simple model on image data that has already been flattened.

To test that idea, I started with a structured grid search over `units1` and `units2`. I kept the optimizer, learning rate, depth, train steps, and validation steps fixed so that the comparison stayed fair. The first pass used a compact set of widths, which made the heatmap easier to read and helped show whether performance was mainly driven by the first or second hidden layer.

The best short grid result came from a wide two-layer setup with Adam, which reached about `81.15%` validation accuracy after three epochs. That suggested that width was helping, but it was not enough evidence to stop there. After that, I ran targeted follow-up experiments around the best region: more epochs, a deeper model, AdamW, RMSprop, and different batch sizes.

The strongest follow-up kept the same wide two-layer direction, used Adam, and trained for `7` epochs. It reached about `82.24%` final validation accuracy, with the best intermediate validation accuracy going higher during training. This supports the idea that once a good configuration is found, training a little longer can be more effective than changing the architecture again. In contrast, the deeper model reached a strong intermediate score but finished lower, which suggests that the extra layer did not give a stable improvement in this setup.

The broader overnight pass made the conclusion cleaner. It used a more regular search space and a fixed `30`-epoch budget, and the best final validation accuracy improved to `87.66%`. That result is stronger than the early three-epoch grid, but it is still bounded by the model family and search budget. I should read it as the best dense-network result found in this notebook, not as a general Fashion-MNIST benchmark.

One lesson from this chapter is that structured search matters. Keeping most settings fixed made it easier to explain what changed and why the result improved. The follow-up runs were useful because they tested specific questions instead of expanding the search randomly.

If I were to continue this experiment, I would keep the best width setting and then study learning rate and epoch scheduling in more detail. I would not immediately expand to a huge search space, because this chapter already showed that a small, well-chosen set of experiments can answer useful questions. For me, the main takeaway is that hyperparameter tuning works best when the grid is narrow enough to compare fairly and the follow-up runs answer a clear question.
