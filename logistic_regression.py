import numpy as np
import matplotlib.pyplot as plt
import math
import os
ALPHA = 0.3
ITERATIONS = 1000


class LogisticRegression(object):

    def __init__(self, data):
        self.original_data = data
        self.data = np.c_[np.ones(len(data)), data]
        self.number_of_attributes = len(data[0])
        self.number_of_examples = len(data)
        self.theta_array = [1, 1, 1] #np.asarray([1] * (self.number_of_attributes))
        self.best_fit = []

    def calculate_cost(self):
        factor_1 = 1/float(self.number_of_examples)
        total_cost = 0

        for i, example in enumerate(self.data):
            y_i = example[3]
            cost_example = (-y_i * math.log(self.calculate_example_sigmoid(example[:3]))) - ((1-y_i)*math.log(1-self.calculate_example_sigmoid(example[:3])))
            total_cost += cost_example

        total_cost *= factor_1
        return total_cost

    def calculate_example_sigmoid(self, example):
        h = np.asarray(example).dot(self.theta_array)
        return self.sigmoid(h)

    def calculate_sigmoid(self):
        h = np.asarray(self.data[:,:3]).dot(self.theta_array)
        return self.sigmoid(h)

    def sigmoid(self, value):
        sigmoid_value = 1 / (1 + np.exp(-value))
        return sigmoid_value

    def gradient_descent(self):
        theta_0 = self.theta_array[0]
        theta_1 = self.theta_array[1]
        theta_2 = self.theta_array[1]

        sum_errors_theta_0 = 0
        sum_errors_theta_1 = 0
        sum_errors_theta_2 = 0

        for i, example in enumerate(self.data):
            h = self.calculate_example_sigmoid(example[:3])
            sum_errors_theta_0 = sum_errors_theta_0 + (h-example[3])
            sum_errors_theta_1 = sum_errors_theta_1 + (h-example[3]) * example[1]
            sum_errors_theta_2 = sum_errors_theta_2 + (h-example[3]) * example[2]

        new_theta_0 = theta_0 - (ALPHA * (1/float(self.number_of_examples)) * sum_errors_theta_0)
        new_theta_1 = theta_1 - (ALPHA * (1/float(self.number_of_examples)) * sum_errors_theta_1)
        new_theta_2 = theta_2 - (ALPHA * (1/float(self.number_of_examples)) * sum_errors_theta_2)

        self.theta_array = [new_theta_0, new_theta_1, new_theta_2]

    def run_gradient_descent(self):
        min_cost=100000000
        best_thetas = []
        for i in xrange(ITERATIONS):
            self.gradient_descent()
            cost = self.calculate_cost()
            print "Iteration: %s, Cost: %s" % (i, cost)
            if cost < min_cost:
                min_cost=cost
                best_thetas = self.theta_array

        if best_thetas:
            self.theta_array = best_thetas
            print self.theta_array
            print self.calculate_cost()

        for example in self.data:
            x2 = (-self.theta_array[0]/float(self.theta_array[2])) + ((-self.theta_array[1]/self.theta_array[2]) * example[1])
            self.best_fit.append([example[1], x2])

    def plot_solution(self):
        plt.scatter(self.original_data[:,:1], self.original_data[:,1:2], c=self.original_data[:,2:3], cmap=plt.cm.autumn)
        plt.plot(*zip(*self.best_fit))
        plt.show()


if __name__ == '__main__':

    example_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data_logreg.csv')
    data_table = np.loadtxt(open(example_file_path, "rb"), delimiter=",", skiprows=1)
    lgr = LogisticRegression(data_table)
    lgr.run_gradient_descent()
    lgr.plot_solution()
