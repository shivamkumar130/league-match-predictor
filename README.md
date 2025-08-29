# League of Legends Match Predictor

This project aims to build a machine learning model to predict the outcomes of League of Legends matches using various in-game statistics. The model is implemented using PyTorch and follows a structured approach to data loading, model training, evaluation, visualization, and feature importance analysis.

## Project Structure

```
league-match-predictor
├── data
│   └── league_of_legends_data_large.csv
├── src
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── feature_importance.py
│   └── utils.py
├── notebooks
│   └── league_match_predictor.ipynb
├── requirements.txt
└── README.md
```

## Components

- **data/league_of_legends_data_large.csv**: Contains the dataset used for training and evaluating the model.

- **src/data_loader.py**: Functions for loading and preprocessing the dataset, including reading the CSV file and splitting the data into features and target variables.

- **src/model.py**: Defines the logistic regression model architecture using PyTorch.

- **src/train.py**: Contains the training loop for the model, managing the training process, loss calculation, backpropagation, and parameter updates.

- **src/evaluate.py**: Functions for evaluating the model's performance on the test dataset, including accuracy calculation and generating classification reports.

- **src/visualize.py**: Functions for visualizing model performance, including confusion matrices and ROC curves.

- **src/feature_importance.py**: Evaluates and visualizes the importance of each feature based on the model's weights.

- **src/utils.py**: Utility functions for data transformations and metric calculations.

- **notebooks/league_match_predictor.ipynb**: Jupyter notebook for exploring the dataset, training the model, and visualizing results interactively.

- **requirements.txt**: Lists the dependencies required for the project, including libraries such as pandas, scikit-learn, torch, and matplotlib.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd league-match-predictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data` directory.

## Usage

- To load and preprocess the data, run:
  ```python
  from src.data_loader import load_data
  ```

- To train the model, execute:
  ```python
  from src.train import train_model
  ```

- To evaluate the model, use:
  ```python
  from src.evaluate import evaluate_model
  ```

- For visualizations, call functions from:
  ```python
  from src.visualize import plot_confusion_matrix, plot_roc_curve
  ```

- To analyze feature importance, run:
  ```python
  from src.feature_importance import plot_feature_importance
  ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.