import numpy as np
import pandas as pd

import correlations

if __name__=='__main__':
    #replace this with the actual data frame
    variable_one=np.random.rand(1000)
    variable_two=np.random.rand(1000)
    variable_three=np.random.rand(1000)
    data_frame=pd.DataFrame({'variable_one':variable_one,'variable_two':variable_two,'variable_three':variable_three})

    correlation_matrix=correlations.calculate_correlation_matrix(data_frame)
    print(correlation_matrix)

