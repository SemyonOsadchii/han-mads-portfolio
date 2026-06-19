# Reflection

This chapter changed my view of sequence models more clearly than I expected. Before running the experiments, I assumed that GRU and LSTM would both improve on a plain RNN, but I did not expect the gap to be as large as it was. The plain RNN never came close to the target accuracy, while the gated models and Conv1D handled the gesture sequences much better.

The main reason is probably that the SmartWatch Gestures dataset is a genuine sequence problem. Each gesture is not only a set of accelerometer values, but a movement pattern over time. A plain RNN has to carry useful information through the sequence with a much weaker memory mechanism. GRU and LSTM are better suited to that because their gates can decide what to keep, update, or forget. The stricter split supports that theory: the best RNN stayed at `27.25%` validation accuracy, while the best GRU reached `93.50%` and the best Conv1D reached `96.50%`.

The most important correction came after checking the split more carefully. The first overnight run used a random file-level split, and that made the `100%` validation results look cleaner than they really were. There was no direct file overlap, but the same users appeared in both training and validation. For gesture recognition, that is a real limitation because people move in recognisable ways.

The user-held-out rerun gave a more believable result. I trained on `U01-U06`, selected the model on `U07`, and evaluated the final model on `U08`. With that setup, the best validation-selected model was a compact Conv1D: `96.50%` validation accuracy and `92.27%` test accuracy. That is lower than the earlier random-split result, but I trust it more.

The GRU result is still strong. The best GRU reached `93.50%` validation accuracy and `92.77%` test accuracy on the stricter split, so recurrent models clearly can work here. LSTM was less convincing as a selected model because its validation score was lower and the selected configuration was much larger.

One thing I learned from this chapter is that the best model is not only the one with the highest number in a table. The final model should be chosen before looking at the test set. In this run there were models with higher observed test accuracy than the selected Conv1D model, but choosing them after seeing the test result would be a form of test-set tuning.

If I continued this experiment, I would inspect confusion between gesture classes and test another subject split to see how stable the result is. The current report is already stronger than the first version because it separates tuning performance from final generalisation.

[Go back to Homepage](../README.md)
