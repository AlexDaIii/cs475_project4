from data import load_data
import numpy as np
import models

DATASETS = ['datasets/easy.train'] #, 'datasets/hard.train', 'datasets/bio.train', 'datasets/finance.train',
            #'datasets/iris.train', 'datasets/speech.train', 'datasets/vision.train']

for filename in DATASETS:
    X, y = load_data(filename)
    lmd = models.LambdaMeans()
    stc = models.StochasticKMeans()
    X = X.todense()
    y = y.reshape(len(y), 1)
    #X#  = np.array([[4,3,7],[0,1,-1],[9,14,2]])
    # lmd.fit(X, y, lambda0=4, iterations=1)
    stc.fit(X, y, iterations=1, num_clusters=2)
