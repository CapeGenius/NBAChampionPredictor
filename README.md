By Rohan Bendapudi, Sam Wisnoski, Bill Le:

We created an NBA Championship predictor utilizing principal component analysis (PCA), support vector machines (linear SVC), and hyperparameter tuning (using GridSearchCV). With PCA, we reduced data dimensionality
to k eigenvectors (using a covariance matrix of the data) and fed those eigenvectors into the SVMs to make predictions on NBA champions. Our model was 95% accurate, with 80-100% recall, 
~50% precision, 0.6 f-score for NBA champion predictions. We also oversampled data using SMOTE and upweighted champion data. 

Credit:
Pandas, SK-Learn, Numpy, SMOTE, MATPLOTLIB, GridSearchCV
