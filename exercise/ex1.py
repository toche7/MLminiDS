def homework():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("https://raw.githubusercontent.com/toche7/DataSets/main/sale.txt", header = None)
    df.columns = ['population','sale']
    X = df[['population']]
    y = df.sale

    regEx1 = None
    from sklearn.linear_model import LinearRegression
    regEx1 = LinearRegression()
    regEx1.fit(X,y)

    #yhat = regEx1.predict(X)
    
    R2 = None
    R2 = regEx1.score(X,y)
    
    # plt.scatter(X,y)
    # plt.plot(X,yhat,'r')
    # plt.show()

    return  regEx1, R2


if __name__ == '__main__':
    homework()
