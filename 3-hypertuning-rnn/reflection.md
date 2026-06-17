# Reflection

This chapter changed my view of sequence models more clearly than I expected. Before running the experiments, I assumed that GRU and LSTM would both improve on a plain RNN, but I did not expect the gap to be as large as it was. The plain RNN never came close to the target accuracy, while the gated models solved the task reliably once hidden size, number of layers, dropout, and learning rate were tuned.

The main reason is probably that the SmartWatch Gestures dataset is a genuine sequence problem. Each gesture is not only a set of accelerometer values, but a movement pattern over time. A plain RNN has to carry useful information through the sequence with a much weaker memory mechanism. GRU and LSTM are better suited to that because their gates can decide what to keep, update, or forget. The results support that theory: the best RNN stayed around `27.50%` best validation accuracy, while GRU and LSTM reached around `100%`.

The GRU result was the strongest from an accuracy perspective. The best GRU reached a perfect validation checkpoint and finished at `99.84%`, and many other GRU configurations also performed very well. That made GRU feel like the most robust family in the search. It was not just one isolated lucky run.

At the same time, the best GRU was not the most efficient model. It used about `1.34M` parameters and had much higher CPU latency than the best LSTM. The best LSTM reached almost the same validation accuracy with fewer parameters and much lower measured CPU inference time. That made the final decision less obvious: if I only care about the highest validation checkpoint, GRU wins; if I care about a practical accuracy-efficiency trade-off, LSTM is easier to justify.

The Conv1D result was a useful surprise. I originally treated it as a smaller supplementary branch because it had fewer completed configurations than GRU and LSTM. Still, the best Conv1D run reached `100.00%` final validation accuracy with a compact model and low CPU latency. I would not overstate it because the sweep was smaller, but it is strong enough that I would include Conv1D in a larger follow-up comparison.

One thing I learned from this chapter is that the best model is not only the one with the highest score. Reporting model size and inference time changed the interpretation. It made the comparison more honest, because a very large model and a compact model can both reach high accuracy but have very different practical costs.

If I continued this experiment, I would run a balanced follow-up where GRU, LSTM, and Conv1D each receive the same search budget. I would also add a final test-set evaluation and inspect confusion between gesture classes. The validation results are strong, but a final report should still separate tuning performance from final generalisation.

[Go back to Homepage](../README.md)
