import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.000000001
ITERATIONS = 100


class LinearRegression(object):

    def __init__(self, data):
        self.original_data = data
        self.data = np.c_[np.ones(len(data)), data]
        self.number_of_attributes = len(data[0])
        self.number_of_examples = len(data)
        self.theta_array = [1] * (self.number_of_attributes)
        self.best_fit = []

    def calculate_cost(self):
        errors = []

        theta_0 = self.theta_array[0]
        theta_1 = self.theta_array[1]
        factor_1 = 1/float(2*self.number_of_examples)

        for i, example in enumerate(self.data):
            cost_example = ((theta_0 + theta_1 * example[1]) - example[2]) ** 2
            errors.append(cost_example)

        total_cost = factor_1 * sum(errors)
        return total_cost

    def calculate_cost_matrix(self):
        pass

    def gradient_descent(self):
        theta_0 = self.theta_array[0]
        theta_1 = self.theta_array[1]
        sum_errors_theta_0 = 0
        sum_errors_theta_1 = 0
        for i, example in enumerate(self.data):
            h = theta_0*example[0] + theta_1 * example[1]
            sum_errors_theta_0 = sum_errors_theta_0 + (h-example[2])
            sum_errors_theta_1 = sum_errors_theta_1 + (h-example[2]) * example[1]

        new_theta_0 = theta_0 - (ALPHA * (1/float(self.number_of_examples)) * sum_errors_theta_0)
        new_theta_1 = theta_1 - (ALPHA * (1/float(self.number_of_examples)) * sum_errors_theta_1)

        # print 'New Theta 0: %s' % new_theta_0
        # print 'New Theta 1: %s' % new_theta_1
        self.theta_array = [new_theta_0, new_theta_1]

    def run_gradient_descent(self):
        for i in xrange(ITERATIONS):
            cost = self.calculate_cost()
            print "Iteration: %s, Cost: %s" % (i, cost)
            self.gradient_descent()

        for example in self.data:
            h = self.theta_array[0] + (self.theta_array[1] * example[1])
            self.best_fit.append([example[1], h])

    def plot_solution(self):
        print self.original_data
        print self.best_fit
        plt.scatter(*zip(*self.original_data), marker='o')
        plt.plot(*zip(*self.best_fit))
        plt.show()
