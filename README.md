# Model Evaluation with Sampling Techniques

This project evaluates the performance of different machine learning models using various sampling techniques on a credit card fraud detection dataset. The models are assessed using accuracy scores across five different sampling techniques:

1. **Simple Random Sampling** - Randomly selects a fraction of the data.
2. **Stratified Sampling** - Samples 80% of each class to maintain class distribution.
3. **Systematic Sampling** - Selects every 5th sample from the data.
4. **Cluster Sampling** - Selects 50% of the data randomly.
5. **Bootstrap Sampling** - Randomly samples the data with replacement.

## Models Evaluated
The following models are evaluated:
- **Decision Tree**
- **K-Nearest Neighbors**
- **Logistic Regression**
- **Naive Bayes**
- **Random Forest**

## Results

### Accuracy Score Results Table:

| Model                        | Bootstrap Sampling | Cluster Sampling | Simple Random Sampling | Stratified Sampling | Systematic Sampling |
|------------------------------|--------------------|------------------|------------------------|----------------------|---------------------|
| Decision Tree                 | 0.990170           | 0.968533         | 0.982297               | 0.982787             | 0.934691            |
| K-Nearest Neighbors           | 0.885966           | 0.803414         | 0.851902               | 0.845902             | 0.702591            |
| Logistic Regression           | 0.946924           | 0.904352         | 0.926611               | 0.923770             | 0.878953            |
| Naive Bayes                   | 0.839486           | 0.876832         | 0.855168               | 0.836885             | 0.836330            |
| Random Forest                 | 0.998689           | 0.992140         | 0.995412               | 0.995902             | 0.993443            |

## Key Findings

- **Bootstrap Sampling** yields the best accuracy scores, particularly with models like Random Forest and Decision Tree.
- **Stratified Sampling** is ideal for datasets with imbalanced classes, ensuring that each class is proportionally represented.
- **Systematic Sampling** tends to produce lower accuracy scores, especially for models like K-Nearest Neighbors.
- **Cluster Sampling** performs well with Decision Tree and Random Forest, though not as well with K-Nearest Neighbors.

These results suggest that **Bootstrap Sampling** is the most effective sampling technique for improving model performance across various machine learning models.

## Visualizations

### Heatmap of Model Accuracy Scores Across Sampling Techniques

Below is the heatmap showing the accuracy of different models with respect to the sampling techniques:

![image](https://github.com/user-attachments/assets/b0a331f9-0fa5-4840-9237-ee9bbeeb2e9b)


This heatmap provides a visual comparison of model performance for each sampling technique. You can clearly see that **Random Forest** and **Decision Tree** outperform other models across most sampling techniques.

### Bar Chart of Model Performance

A bar chart comparing the accuracy scores of different models across all sampling techniques is also included. This chart helps to visualize the differences in performance in a more intuitive way.

## Project Insights

- **Bootstrap Sampling** is the most effective for improving accuracy, particularly for models like Random Forest, which performs best overall.
- **Stratified Sampling** is critical for handling class imbalances, ensuring that both the majority and minority classes are properly represented.
- **Systematic Sampling** showed less promising results, especially for K-Nearest Neighbors, indicating that this sampling technique may not be the best for certain models.
- **Cluster Sampling** had a mixed performance but worked well with models like Decision Tree and Random Forest.
