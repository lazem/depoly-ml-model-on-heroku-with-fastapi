# Model Card

This model is developed to predict salary category (whether income exceeds $50K/yr) based on census data
## Model Details
- DecisionTreeClassifier based on `entropy` criterion
- Other default parameterless are used

## Intended Use
- Intended to be used as an exemplary model, part of `Deploying a Machine Learning Model on Heroku with FastAPI` project
- Given a list of categories such as "workclass", "education", "marital-status"..etc predict the income category
## Training & Evaluation Data
- 80% of the [Census Income](https://archive.ics.uci.edu/ml/datasets/census+income) Dataset is used for training.
- 20% is used for evaluation using sklearn `train_test_split` method.

## Metrics
The model is validated against `precision`, `recall`, and `F1` metrics. 

## Ethical Considerations
The census dataset is from 1994 US Census database. Any extrapolation from this dataset on the current date or a different country may be incorrect. 

## Caveats and Recommendations
Dataset source needs some cleaning before it can be converted into pandas dataframe
