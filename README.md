# Random States Classifier Performance

This repo investigates the influence of different random states (seeds) on classification results of a logstic regression classifier with varying performance levels.

## Main script

- Creates simulated classification data (2 outcome classes), 5000 samples & 50 features
- Logistic regression is used as a classifier
- Different numbers of informative features (0, 10, 20, 30, and 40) lead to varying classification accuracies
- 500 different random state initializations (train-test-split and initial parameters) are run per number of informative features
- Results are saved in csv file

## Plotting script

...creates:
- Boxplots of classification accuracies across all 500 runs per informativeness of features
- Lineplot with variance of classification accuracies across all 500 runs per informativeness of features
