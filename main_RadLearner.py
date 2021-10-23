import numpy as np
import pandas as pd
import func_RadLearner as rl

train_df = pd.read_csv( "data/data_h2_2.csv" )
S0, S1 = rl.DataTransformation(train_df)

print('W: ' + np.array_str(rl.LearningAlg(S0, S1)))