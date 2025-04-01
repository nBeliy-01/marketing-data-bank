# Marketing Bank Data
Abstract
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

# Information
The data is related to direct marketing campaign direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
The smallest datasets are provided to test more computationally demanding machine learning algorithms. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# ML Pipeline for Classification Tasks
This decision provides an end-to-end machine learning pipeline for classification problems. It includes data preprocessing, feature engineering, and model training with hyperparameter tuning using Grid Search, Randomized Search, and Hyperopt.

### Features:
* <b>Data Preprocessing:</b> Encoding categorical variables, handling missing values, and feature scaling.
* <b>Feature Engineering:</b> Creating new features based on exploratory data analysis.
* <b>Model Training:</b> Supports XGBoost and other classifiers with SMOTE/ADASYN for imbalanced data handling.
* <b>Hyperparameter Tuning:</b> Uses Grid Search, Randomized Search, and Hyperopt for optimal model selection.
* <b>Pipeline Integration:</b> Built with sklearn.pipeline for seamless integration.

### Usage:
Simply provide your dataset and configure the model parameters to train and evaluate the pipeline.

### Model Comparison results
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Model Params</th>
      <th>Mean Train Score</th>
      <th>Mean Test Score</th>
      <th>Params Tuning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>XGBClassifier</td>
      <td>XGBClassifier(colsample_bytree=0.47296310202684444, learning_rate=0.056639455691759745, max_depth=5, min_child_weight=4.0, reg_alpha=0.9204355789669567, reg_lambda=0.3391291474222857, subsample=0.9068144056978242)</td>
      <td>0.966797</td>
      <td>0.955869</td>
      <td>Hyperport</td>
    </tr>
    <tr>
      <td>XGBClassifier</td>
      <td>XGBClassifier(subsample=0.6072301454462083, max_depth=5, colsample_bytree=0.8010123167692438, learning_rate=0.04216161028349973, reg_alpha=0.44842414298624733, reg_lambda=0.9944574626108207, scale_pos_weight=1.7592525267734538)</td>
      <td>0.996610</td>
      <td>0.945598</td>
      <td>Randomized Search</td>
    </tr>
    <tr>
      <td>LogisticRegression</td>
      <td>LogisticRegression(class_weight='balanced', max_iter=300, solver='liblinear')</td>
      <td>0.937959</td>
      <td>0.938947</td>
      <td>Grid Search</td>
    </tr>
    <tr>
      <td>DecisionTree</td>
  <td>DecisionTreeClassifier(class_weight='balanced', criterion='gini', min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=5, max_depth=5, min_weight_fraction_leaf=0.0, random_state=42, splitter='best', max_leaf_nodes=60, max_features='log2')</td>
      <td>0.840334</td>
      <td>0.837042</td>
      <td>Grid Search</td>
    </tr>
    <tr>
      <td>KNeighbors</td>
      <td>KNeighborsClassifier(leaf_size=30, n_neighbors=5, algorithm='auto', metric='minkowski', p=2, weights='uniform')</td>
      <td>0.893703</td>
      <td>0.796736</td>
      <td>Grid Search</td>
    </tr>
  </tbody>
</table>
