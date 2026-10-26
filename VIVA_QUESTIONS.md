# Machine Learning Practical: Viva Questions & Answers

Short answers you can say out loud. **Bold** = keywords examiners listen for.

---

## General ML questions (asked in almost every viva)

**Q1. What is Machine Learning?**
A branch of AI where a computer **learns patterns from data** and makes predictions without being explicitly programmed with rules.

**Q2. Types of Machine Learning?**
- **Supervised**: data has labels (Linear/Logistic Regression, Decision Tree, SVM, AdaBoost)
- **Unsupervised**: no labels, find structure (DBSCAN, K-Means, PCA)
- **Reinforcement**: agent learns by **reward and punishment** (games, robots)

**Q3. Classification vs Regression?**
Classification predicts a **category** (pass/fail). Regression predicts a **continuous number** (salary).

**Q4. What is overfitting and underfitting?**
- **Overfitting**: model memorises training data, performs badly on new data (high variance).
- **Underfitting**: model too simple, bad even on training data (high bias).

**Q5. What is the bias-variance trade-off?**
Simple models have high bias, complex models have high variance. We want a balance where the **total error is lowest**.

**Q6. Why split data into train and test?**
To check how the model does on **unseen data**. Testing on training data gives a falsely high score.

**Q7. What is a feature and a label?**
Feature = input column (hours studied). Label = the output we want to predict (pass/fail).

**Q8. What is a hyperparameter?**
A setting we choose **before** training, not learned from data: learning rate, k in KNN, eps in DBSCAN, number of rounds in AdaBoost.

**Q9. What is gradient descent?**
An optimisation method that updates parameters **in the opposite direction of the gradient** of the cost, step by step, to reach minimum cost.
`parameter = parameter - learning_rate * gradient`

**Q10. What if the learning rate is too big or too small?**
Too big: cost jumps around or **diverges**. Too small: learning is **very slow**.

**Q11. Why did you not use libraries like sklearn/numpy?**
To implement the algorithms **from scratch** and understand the maths behind them.

---

## Experiment 1: Various aspects of ML

**Q1. What aspects did you cover?**
Dataset, statistics (mean, variance, std), train/test split, normalisation, KNN model, evaluation metrics and graphs.

**Q2. What is normalisation? Why is it needed?**
Scaling features to a common range. Min-Max formula: `(x - min) / (max - min)` gives values between 0 and 1. Needed so that a feature with big numbers **doesn't dominate** distance calculations.

**Q3. Normalisation vs Standardisation?**
Normalisation: range 0 to 1. Standardisation (Z-score): `(x - mean) / std`, gives mean 0 and std 1.

**Q4. Why fit the scaler only on training data?**
To avoid **data leakage**: test data must stay "unseen".

**Q5. How does KNN work?**
Find the **k nearest training points** (Euclidean distance) to the new point, and take a **majority vote** of their labels.

**Q6. Why is k usually odd?**
To **avoid ties** in voting for 2 classes.

**Q7. Is KNN a lazy learner?**
Yes. It does **no training**, it just stores data and does the work at prediction time.

**Q8. What is a confusion matrix?**
A table of TP, TN, FP, FN that compares actual vs predicted labels.

**Q9. Define accuracy, precision, recall, F1.**
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP): of predicted positives, how many are correct
- Recall = TP / (TP + FN): of actual positives, how many we found
- F1 = 2PR / (P + R): **harmonic mean** of precision and recall

**Q10. When is accuracy misleading?**
On **imbalanced data**. For example, 95% "no disease" means a model that always says "no" gets 95% accuracy but is useless.

---

## Experiment 2: Linear Regression

**Q1. What is linear regression?**
A supervised **regression** algorithm that fits a straight line `y = mx + c` to predict a continuous value.

**Q2. What are m and c?**
m = **slope** (change in y for 1 unit of x). c = **intercept** (value of y when x = 0).

**Q3. What is the least squares method?**
It finds m and c that **minimise the sum of squared errors**.
`m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²`, `c = ȳ - m·x̄`

**Q4. What cost function is used?**
**Mean Squared Error**: `MSE = (1/n) Σ(y - ŷ)²`

**Q5. Why square the errors?**
So positive and negative errors **don't cancel out**, and big errors are punished more.

**Q6. What is R²?**
Coefficient of determination: `R² = 1 - SS_res / SS_tot`. It shows how much of the variation in y the model explains. **1 = perfect**, 0 = no better than predicting the mean.

**Q7. Why do least squares and gradient descent give the same answer?**
Both minimise the **same MSE cost**. Least squares solves it directly; gradient descent reaches it step by step.

**Q8. Assumptions of linear regression?**
Linear relationship, independent errors, constant variance of errors (**homoscedasticity**), normally distributed errors, no multicollinearity.

**Q9. Simple vs multiple linear regression?**
Simple has one input x. Multiple has many inputs: `y = w1x1 + w2x2 + ... + b`.

**Q10. What does the cost graph show?**
Cost **decreasing** every epoch, which means the model is learning. It flattens when it has converged.

---

## Experiment 3: Logistic Regression

**Q1. What is logistic regression?**
A **classification** algorithm (despite the name) that predicts the **probability** of class 1 using the sigmoid function.

**Q2. What is the sigmoid function?**
`σ(z) = 1 / (1 + e^-z)`. It converts any number into the range **0 to 1**. It is an S-shaped curve, and σ(0) = 0.5.

**Q3. Why not use linear regression for classification?**
Linear output can be **below 0 or above 1**, which is not a valid probability, and it is very sensitive to outliers.

**Q4. What loss function is used?**
**Log-loss / Binary Cross-Entropy**: `L = -(1/n) Σ[y·log(p) + (1-y)·log(1-p)]`

**Q5. Why not MSE for logistic regression?**
MSE with sigmoid gives a **non-convex** cost (many local minima). Log-loss is **convex**.

**Q6. What is the decision boundary?**
The line where p = 0.5, i.e. `w·x + b = 0`. One side is class 1, the other side is class 0.

**Q7. What is the threshold?**
The cut-off probability, normally **0.5**. It can be changed, for example lowered in medical tests to catch more positives.

**Q8. Gradient formula?**
`dw = (1/n) Σ(p - y)·x`, `db = (1/n) Σ(p - y)`, the same form as linear regression.

**Q9. Types of logistic regression?**
Binary (2 classes), Multinomial (more than 2 classes, uses softmax), Ordinal (ordered classes).

**Q10. Is logistic regression linear or non-linear?**
Its **decision boundary is linear**. The sigmoid output is non-linear.

---

## Experiment 4: Decision Tree (ID3)

**Q1. What is a decision tree?**
A tree-shaped model where each **internal node tests an attribute**, each **branch is a value**, and each **leaf is a class label**.

**Q2. What is entropy?**
A measure of **impurity / randomness**: `E = -Σ p·log2(p)`. 0 = pure (all same class). 1 = perfectly mixed (50/50, for 2 classes).

**Q3. What is information gain?**
The **reduction in entropy** after splitting on an attribute: `Gain = E(parent) - Σ (|Sv|/|S|)·E(Sv)`. ID3 picks the attribute with the **highest gain**.

**Q4. Why is Outlook the root in Play Tennis?**
It has the **highest information gain (0.247)**.

**Q5. Entropy of the Play Tennis dataset?**
9 Yes, 5 No, so **0.940**.

**Q6. ID3 vs C4.5 vs CART?**
- ID3: information gain, categorical data
- C4.5: **gain ratio**, handles continuous data and missing values
- CART: **Gini index**, makes binary trees, works for classification and regression

**Q7. What is the Gini index?**
`Gini = 1 - Σ p²`. Another impurity measure (used in CART), and faster since there is no log.

**Q8. When does the tree stop splitting?**
When a node is **pure**, when no attributes are left, or when a max depth or min samples limit is reached.

**Q9. How do you avoid overfitting in trees?**
**Pruning** (pre-pruning: limit depth; post-pruning: cut branches after growing), or use ensembles like Random Forest.

**Q10. Advantages and disadvantages?**
Pros: easy to understand and visualise, no scaling needed. Cons: **overfits easily**, unstable (small data change gives a different tree).

---

## Experiment 5: Bagging, Boosting & AdaBoost

**Q1. What is ensemble learning?**
Combining **many weak models** to make one strong model.

**Q2. What is bagging?**
**Bootstrap Aggregating**: train many models in **parallel** on random samples taken **with replacement**, then take a **majority vote** (or average). It **reduces variance**. Example: **Random Forest**.

**Q3. What is boosting?**
Train models **sequentially**; each new model focuses on the **mistakes** of the previous ones. It **reduces bias**. Examples: AdaBoost, Gradient Boosting, XGBoost.

**Q4. Bagging vs Boosting?**

| Bagging | Boosting |
|---|---|
| Parallel | Sequential |
| Equal weight to every model | Weighted by performance (alpha) |
| Reduces variance | Reduces bias |
| Random Forest | AdaBoost, XGBoost |

**Q5. What is a bootstrap sample?**
A random sample of the same size as the data, **drawn with replacement**, so some rows repeat and some are left out.

**Q6. What is a decision stump?**
A decision tree with **only one split** (depth 1). It is the weak learner in AdaBoost.

**Q7. What is alpha in AdaBoost?**
The **importance (say)** of a stump: `α = ½·ln((1 - error) / error)`. Low error gives big alpha.

**Q8. How are sample weights updated?**
`w = w·e^(-α·y·h(x))`, then normalise. **Wrong samples get larger weights**, so the next stump focuses on them.

**Q9. Final prediction in AdaBoost?**
`sign(Σ α·h(x))`, a **weighted vote** of all stumps.

**Q10. What does the final weights graph show?**
The points with big bars are the **hardest samples** (outliers or ones near the boundary) that AdaBoost kept focusing on.

**Q11. What is the disadvantage of AdaBoost?**
It is **sensitive to noise and outliers**, because it keeps increasing their weight.

---

## Experiment 6: Support Vector Machine

**Q1. What is SVM?**
A supervised algorithm that finds the **hyperplane that separates classes with the maximum margin**.

**Q2. What is a hyperplane?**
The decision boundary `w·x + b = 0`. In 2D it is a line, in 3D a plane.

**Q3. What is the margin?**
The distance between the hyperplane and the nearest points of each class. Width = **2 / ||w||**. SVM **maximises** it.

**Q4. What are support vectors?**
The data points **closest to the hyperplane** (on the margin lines). Only they decide the boundary; removing other points changes nothing.

**Q5. What is hinge loss?**
`max(0, 1 - y·(w·x + b))`. It is zero if a point is correctly classified **and outside the margin**, otherwise it is positive.

**Q6. Hard margin vs soft margin?**
Hard margin: **no misclassification allowed** (only for perfectly separable data). Soft margin: allows some errors, controlled by **C / lambda**.

**Q7. What is the role of lambda (or C)?**
It controls the trade-off between a **wide margin** and **few training errors**. Big C (small lambda) means fewer errors but a narrower margin, with a risk of overfitting.

**Q8. What is the kernel trick?**
It maps data into a **higher dimension** where it becomes linearly separable, without actually computing that mapping.

**Q9. Common kernels?**
Linear, **Polynomial**, **RBF (Gaussian)**, Sigmoid.

**Q10. Why must labels be +1 / -1?**
So that `y·(w·x + b) ≥ 1` works as one condition for both classes.

---

## Experiment 7: DBSCAN

**Q1. What is DBSCAN?**
**Density-Based Spatial Clustering of Applications with Noise**: an **unsupervised** algorithm that groups densely packed points and marks lonely points as **noise**.

**Q2. What are eps and min_pts?**
- **eps**: radius of the neighbourhood around a point
- **min_pts**: minimum points inside eps (including the point itself) to call it a core point

**Q3. Core, border and noise points?**
- **Core**: has ≥ min_pts neighbours within eps
- **Border**: fewer neighbours, but is inside eps of a core point
- **Noise**: neither of the above (an outlier)

**Q4. Density-reachable vs density-connected?**
Reachable: you can reach a point by a **chain of core points**. Connected: two points are both reachable from a **common core point**.

**Q5. Advantages over K-Means?**
- No need to give the **number of clusters**
- Finds clusters of **any shape** (K-Means finds only round ones)
- **Detects outliers / noise**

**Q6. Disadvantages?**
Struggles when clusters have **different densities**, and it is sensitive to the choice of eps.

**Q7. What if eps is too small or too large?**
Too small: most points become **noise**. Too large: all clusters **merge into one**.

**Q8. How to choose eps?**
Use a **k-distance graph**: sort each point's distance to its k-th nearest neighbour and pick the "elbow" (knee) point.

**Q9. Rule of thumb for min_pts?**
min_pts ≥ dimensions + 1, commonly **2 × dimensions**.

**Q10. Time complexity?**
O(n²) with plain distance checks (as in this code), about O(n log n) with a spatial index (KD-tree).

---

## Experiment 8: PCA

**Q1. What is PCA?**
**Principal Component Analysis**: an **unsupervised dimensionality reduction** technique that turns correlated features into a smaller set of uncorrelated **principal components** while keeping maximum variance.

**Q2. Steps of PCA?**
1. Centre (or standardise) the data
2. Compute the **covariance matrix**
3. Find **eigenvalues and eigenvectors**
4. Sort by eigenvalue (largest first)
5. **Project** the data onto the top k eigenvectors

**Q3. What do the eigenvector and eigenvalue mean here?**
Eigenvector = **direction** of a principal component. Eigenvalue = **amount of variance** along that direction.

**Q4. What is the covariance matrix?**
A square matrix showing how each pair of features **varies together**. Diagonal values = variances.

**Q5. Why centre the data first?**
PCA measures variance around the mean. Without centring, the first component would just point at the mean.

**Q6. What is "variance explained"?**
`eigenvalue / sum of eigenvalues × 100`. In our data PC1 explains **96.3%**, so we can drop PC2 and lose very little.

**Q7. How do you choose k?**
Keep enough components to explain about **90–95% of the variance**, or use a **scree plot** (elbow).

**Q8. Are principal components correlated?**
No. They are **orthogonal (perpendicular)** and uncorrelated.

**Q9. What is power iteration?**
A method to find the largest eigenvector: repeatedly do `v = M·v` and normalise. **Deflation** (`M - λ·v·vᵀ`) removes it so we can find the next one.

**Q10. Uses of PCA?**
Reducing features before training, **visualising** high-dimensional data in 2D, noise reduction, image compression.

**Q11. PCA vs LDA?**
PCA is **unsupervised** (maximises variance). LDA is **supervised** (maximises class separation).

---

## Quick formula sheet

| Topic | Formula |
|---|---|
| Euclidean distance | √Σ(aᵢ - bᵢ)² |
| Min-Max scaling | (x - min) / (max - min) |
| Linear regression | y = mx + c |
| MSE | (1/n) Σ(y - ŷ)² |
| R² | 1 - SS_res / SS_tot |
| Sigmoid | 1 / (1 + e⁻ᶻ) |
| Log-loss | -(1/n) Σ[y log p + (1-y) log(1-p)] |
| Entropy | -Σ p log₂ p |
| Gini | 1 - Σ p² |
| Information gain | E(parent) - Σ(\|Sv\|/\|S\|)·E(Sv) |
| AdaBoost alpha | ½ ln((1 - err) / err) |
| AdaBoost weight update | w·e^(-α·y·h(x)) |
| SVM margin | 2 / \|\|w\|\| |
| Hinge loss | max(0, 1 - y(w·x + b)) |
| Covariance | Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1) |
