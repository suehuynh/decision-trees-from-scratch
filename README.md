# A From-Scratch Decision Tree Engine

### The Project
This repository contains a modular, from-scratch implementation of a Decision Tree algorithm for both Classification and Regression. While modern libraries like Scikit-Learn offer efficient "out-of-the-box" solutions, this project is my personal attempt to build this algorithm from sratch to fully understand its recursive mechanics and statistical impurity.

### Features:
I will try to implement (replicate) these features of the scikit-learn libraries:
- Multiple Impurity Metrics: Supports Training Error, Gini Impurity, Cross Entropy, and Mean Squared Error (MSE) for regression tasks.
- Customizable Splitting Logic: Implements greedy search for optimal feature selection across binary and continuous variables.
- Pruning & Constraints: Integrated support for max_depth and min_samples_split to manage the bias-variance tradeoff.
- Visual Debugging: Built-in tools for tracking loss convergence during tree expansion.

## Algorithm Overview
> A Decision Tree automates this process by looking at your data and finding the "most informative" questions to ask first. It repeats this process until it can classify the data with high confidence.
### Technical Deep-Dive
The engine uses Information Gain to decide where to split. At each node, the algorithm calculates how much "disorder" (Entropy) or "misclassification" (Gini) exists. It then selects the feature split that results in the largest reduction of that disorder.
$Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$

### Datasets for Benchmarking
To test the engine's performance across different domains, the following datasets are included/referenced:
- Marketing Analytics: Predicting ad conversion based on user demographics.
- Biological Data (Palmer Penguins): Classifying species based on physical measurements.
- Synthetic XOR: Visualizing non-linear decision boundaries.

### Future Roadmap

[ ] Gradient Boosting Extension: Implementing a Gradient Boosted Machine (GBM) using these trees as base learners.

[ ] Soft Decision Trees: Experimenting with differentiable "soft" splits using Stochastic Gradient Descent (SGD).

[ ] Polars Integration: Leveraging Polars for faster data manipulation and vectorization.
