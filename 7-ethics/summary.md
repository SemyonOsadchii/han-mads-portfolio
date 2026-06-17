# Ethical Reflection

This portfolio is built around hyperparameter tuning and model comparison. The ethical question is therefore not only whether the models reached high validation accuracy, but whether I can explain the evidence behind those scores without making the results look more certain than they are.

In Week 1, the Fashion-MNIST dense-network experiment showed how easy it is to focus too much on the winning configuration. The overnight search improved the best final validation accuracy to `87.66%`, but that number only has meaning together with the search design: `180` configurations, short training budgets, and a limited dense-network family. I should report it as the best result inside that experiment, not as a final statement about Fashion-MNIST models in general.

Week 2 made the need for traceability clearer. The first MLflow run suggested that the dense batch-normalised model was strongest, but the larger `288`-run sweep changed the interpretation and showed that a tuned CNN could reach `87.86%` final validation accuracy. Without MLflow logs and CSV exports, that change would be harder to justify. For me, the ethical lesson is that reproducibility is part of honesty: if I claim that one model family is better, the repository should contain enough evidence to check how I reached that conclusion.

Week 3 added the question of model cost. On the SmartWatch Gestures dataset, GRU, LSTM, and Conv1D all passed the `90%` target, while the plain RNN failed clearly. If I reported only the highest validation score, I would hide an important practical difference. The best GRU was very accurate, but it was also much larger and slower than some alternatives. LSTM and Conv1D made the comparison more realistic because they showed that accuracy, parameter count, and inference time can lead to different recommendations.

Week 4 showed why conclusions need to stay close to the dataset. ResNet18 transfer learning reached `96.08%` validation accuracy on the ants/bees dataset, while the compact transformer stayed far behind. That supports the hypothesis that pretrained convolutional features are a better fit for a small image dataset. It does not prove that transformers are generally worse, and I should not present it that way. The fair conclusion is narrower: for this dataset and search budget, ResNet18 was the right default.

There is also a compute trade-off across the whole portfolio. Larger searches improved the reports, but they also used more time and resources. I do not think the answer is to avoid running experiments, because tuning is the point of the assignment. The responsibility is to make the search deliberate: define the question, save the outputs, separate exploratory runs from final evidence, and stop when the added compute is no longer changing the conclusion.

Publishing the portfolio adds one more responsibility. A public repository should make it clear which files support the conclusions and which outputs are local by-products. That is why the final version keeps summaries, notebooks, selected CSV exports, scripts, and the Week 4 figures visible, while local databases and temporary run folders are kept out of the public site. The goal is not to make the work look perfect; it is to make the work reviewable.

My main takeaway is that responsible machine-learning reporting is about restraint. I should show strong results when the evidence supports them, but I should also show the search budget, validation-only limitations, model cost, and dataset boundaries. In this portfolio, the strongest result is not just one validation score. It is the connection between the experiments, the saved evidence, and the conclusions I am willing to defend.

[Go back to Homepage](../README.md)
