# Ethical Reflection

This reflection focuses on the ethical side of the modelling work in this portfolio. The technical chapters mostly measure accuracy, loss, runtime, model size, and reproducibility. Those metrics matter, but they are not enough on their own. A model can perform well in an experiment and still be a poor choice if the data, use case, or deployment context is not handled responsibly.

One issue is how easy it is to overstate results. In several chapters the best number came from a larger search or from a single strong checkpoint. Reporting only that number would make the work look cleaner than it really is. A more honest report needs to show the search space, the number of runs, the difference between best and final validation accuracy, and the limitations of the experiment. That is especially important when hyperparameter tuning is involved, because repeated trials can make a result look more certain than it is.

Another issue is resource use. Larger searches, deeper models, and longer training runs can improve results, but they also cost more compute time and energy. In this portfolio I tried to separate exploratory runs from the evidence I would actually report. That distinction matters: not every experiment that can be run is worth running, and not every larger model is justified by a small accuracy gain.

There is also a responsibility to keep the work reproducible. If notebooks, result files, and reports disagree, the project becomes hard to audit. For that reason, cleaning the repository is not just cosmetic. It helps someone else understand which files are evidence, which files are temporary artefacts, and how the conclusions were reached.

Deployment adds another layer. Publishing a portfolio makes the work easier to review, but it also means unfinished or misleading material can become part of the public record. The public version should therefore avoid placeholder text, unrelated example projects, and claims that are not backed by the repository.

The main takeaway is that responsible machine-learning work is not only about finding the best model. It is also about reporting uncertainty, keeping evidence traceable, using compute deliberately, and making sure the published version represents the work honestly.

[Go back to Homepage](../README.md)
