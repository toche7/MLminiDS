def homework():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("https://raw.githubusercontent.com/toche7/DataSets/main/sale.txt", header = None)
    df.columns = ['population','sale']
    X = df[['population']]
    y = df.sale

    # create a linear regression model
    regEx1 = None
    from sklearn.linear_model import LinearRegression
    # your code here


    # fit the model
    # your code here



    # determine the performance of the model
    R2 = None
    # your code here
    
    # plt.scatter(X,y)
    # plt.plot(X,yhat,'r')
    # plt.show()

    return  regEx1, R2


if __name__ == '__main__':
    homework()
