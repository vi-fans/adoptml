import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from core.structured_data_insights import correlations

def seasonal_bundling(data_frame,threshold):
    potential_bundles=[]
    sales_correlations=correlations.calculate_correlation_matrix(data_frame)
    print('raw analysis:')
    print(sales_correlations)
    print('******************')
    keys=sales_correlations.keys()
    print('potential bundling:')
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            item=keys[i]
            potential_bundle_item=keys[j]
            if sales_correlations[item][potential_bundle_item]>threshold:
                print('positively correlated:',item,potential_bundle_item)
                potential_bundles.append([item,potential_bundle_item])
            if sales_correlations[item][potential_bundle_item]<-1*threshold:
                print('negatively correlated:',item,potential_bundle_item)
    return potential_bundles

if __name__=='__main__':
    #replace this part with the actual data frame
    daily_sales_item_x=np.random.rand(1000)
    daily_sales_item_y=np.random.rand(1000)
    daily_sales_item_z=daily_sales_item_x+0.1
    data_frame=pd.DataFrame({'daily_sales_item_x':daily_sales_item_x,'daily_sales_item_y':daily_sales_item_y,'daily_sales_item_z':daily_sales_item_z})

    #actual analysis
    seasonal_bundling(data_frame,0.7)

