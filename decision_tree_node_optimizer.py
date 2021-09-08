import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def best_leaf_finder(X, y, bot=2, top=1000, n=6)
    '''
    Optimizes a decision tree node count for lowest mean absolute error.
    
    X, a number of predictor variables (pandas dataframe or series)
    y, a target variable (pandas dataframe or series)
    bot, (optional) a proposed minimum number of tree nodes (int), default is 2 nodes
    top, (optional) a proposed maximum number of tree nodes (int), default is 1000 nodes
    n, (optional) interrogation span  (int)
    
    Returns the optimal tree node count as an integer.
    This value can be used as scikit-learn's DecisionTreeRegressor's max_leaf_nodes argument.
    '''
    
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return(mae)
    
        #linspace function to select node search
    def linarly(bot, top, n):
        return np.linspace(bot, top, num=n).astype(int).tolist()
    
    #starting node search
    max_leaf_nodes = linarly(bot, top, n)

    while max_leaf_nodes[0] + n  < max_leaf_nodes[n-1]:

        #initialize list of MAE values
        maes = []

        for node_size in max_leaf_nodes:
            my_mae = get_mae(node_size, train_X, val_X, train_y, val_y)
            maes.append(my_mae)
            print('Node size: %d \t\t\t Mean absolute error: %d' %(node_size, my_mae))
        print('\n')

        mini = maes.index(min(maes))
        print('Lowest MAE for:', max_leaf_nodes[mini])

        #account for off-linspace top hits
        if mini == 0:
            bot = 0
        else:
            bot = mini-1

        if mini == n:
            top = n
        else:
            top = mini + 1

        print('Searching between %d and %d.' %(max_leaf_nodes[bot], max_leaf_nodes[top]))
        max_leaf_nodes = linarly(max_leaf_nodes[bot],max_leaf_nodes[top], n)

    #remove duplicate node values
    max_leaf_nodes = list( dict.fromkeys(max_leaf_nodes))
    maes = []

    for node_size in max_leaf_nodes:
        my_mae = get_mae(node_size, train_X, val_X, train_y, val_y)
        maes.append(my_mae)
        print('Node size: %d \t\t\t Mean absolute error: %d' %(node_size, my_mae))

    mini = maes.index(min(maes))

    print('\nFinal iteration')
    print('Optimal node size: %d \t\t Optimal Mean absolute error %d' %(max_leaf_nodes[mini],maes[mini]))

    return best_tree_size = max_leaf_nodes[mini]


def optimized_decisiontree(X, y, bot=2, top=1000, n=6)
    '''
    Optimizes a decision tree node count for lowest mean absolute error.
    
    X, a number of predictor variables (pandas dataframe or series)
    y, a target variable (pandas dataframe or series)
    bot, (optional) a proposed minimum number of tree nodes (int), default is 2 nodes
    top, (optional) a proposed maximum number of tree nodes (int), default is 1000 nodes
    n, (optional) interrogation span  (int)
    
    Returns an optimized decision tree model with minimum MAE.
    This model can be treated as an already fit DecisionTreeRegressor.
    
    '''
    
    best_leaf = best_leaf_finder(X, y, bot=2, top=1000, n=6)

    final_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes = best_leaf)

    final_model.fit(X, y)
    
    return final_model