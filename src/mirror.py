import data_preprocess


# 调参 MNB
param_grid ={}
param_grid['alpha'] = [0.001,0.01,0.1,1.5]
model = MultinomialNB()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator= model, param_grid = param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X = X_train_counts_tf, y = dataset_train.target)
print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))
