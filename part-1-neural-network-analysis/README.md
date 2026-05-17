
# Customer Churn Prediction using Neural Networks

## Objective
Build a feed‑forward neural network to predict whether a customer will churn (binary classification).  
The project demonstrates data preprocessing, model training, hyperparameter tuning, and key deep‑learning concepts.

## Dataset
- 2000 rows, 17 columns (customer_id, 16 features + target `churn`).
- Target: 0 = retained, 1 = churned (≈6% churn rate).
- Mixed data types: numerical (tenure, charges, etc.) and categorical (region, plan_type, etc.).

## Files
- `notebook.ipynb` – Full analysis, code, and outputs.
- `requirements.txt` – Python dependencies.
- `results/` – Contains model comparison table and evaluation plots.

## How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Place `customer_churn_nn.csv` in the same folder as the notebook.
3. Run all cells in `notebook.ipynb`.

## Key Findings
- Baseline neural network (2 hidden layers: 64→32 neurons) achieves ~94% test accuracy.
- Class imbalance reduces recall for the churn class – using class weights or SMOTE would improve.
- Hyperparameter experiments show that a larger learning rate (0.01) hurts performance, while reducing neurons or increasing batch size has minor effects.
- Early stopping prevents overfitting; no underfitting observed.

## Reflection (Task 6)
- **Weights & biases**: Weights control the strength of connections; biases allow the activation to shift, enabling the model to fit data that does not pass through the origin.
- **Activation functions**: Introduce non‑linearity, allowing the network to learn complex patterns. Without them, the whole network would be equivalent to a linear model.
- **Learning rate**: Too high → loss oscillates or diverges; too low → very slow convergence.
- **Overfitting/underfitting**: Our model shows no underfitting (high training accuracy). Overfitting is controlled via early stopping and validation split.