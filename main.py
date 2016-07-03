import numpy as np
from linear_regression import LinearRegression
file = '/Users/ruivaz/workspace/git/c_apps/data.csv'

if __name__ == "__main__":
    data_table = np.loadtxt(open(file,"rb"), delimiter=",", skiprows=1)
    lr = LinearRegression(data_table)
    lr.run_gradient_descent()
    lr.plot_solution()



