"""
In linear regression, we approximate y using a straight line. y=mx+b. Then we minimise the squared error
and differentiate with respect to m and ab to get the values of m and b

m = x'y' - (xy)' / (x')^2 - (x^2)'   Here x' is the mean of x and y' is mean of y
for b, put this m in y'=  mx' + b
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)  # it is acutally the default data type
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = ((mean(xs) * mean(ys)) - (mean(xs * ys))) / (mean(xs) ** 2 - mean(xs ** 2))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coeff_of_determination(ys_orig, ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, ys_mean_line)
    return 1 - (squared_error_regression / squared_error_mean)


m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]

coeff_of_deter = coeff_of_determination(ys,regression_line)
print(coeff_of_deter)


plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()

# getting the accuracy of our linear fit line. We use doing squared error. We use square not mod becuase we want to
# penalise the far of points from the line heavily

'''
R squared theory -> coefficient of determination
r^2 = 1 - (squared error of ys) / (squared error of mean of ys)
More the value of R squared better the fit
'''
