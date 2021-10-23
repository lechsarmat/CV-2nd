import numpy as np
import pandas as pd
import func_RadLearner as rl

train_df = pd.read_csv( "data/data_h2_2.csv" )
S0, S1 = rl.DataTransformation(train_df)

S2 = np.array([[0.,-0.5], [-0.3,0.4], [0.5,-0.5], [0.3,0.8], [0.5,-0.3]])

print('First zone: ' + np.array_str(rl.Classifier(S0, S1, S2).T[0]))