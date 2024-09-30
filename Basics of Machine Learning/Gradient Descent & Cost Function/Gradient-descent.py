#Use the following link for the explaination of this code
#https://www.youtube.com/watch?v=vsWrXfO3wWw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=4

import numpy as np
import math
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08
    prev_cost = float('inf')

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) #Determing the cost or error called Mean Squared Error (MSE)
        md = -(2/n)*sum(x*(y - y_predicted)) #Change in coefficient slope
        bd = -(2/n)*sum(y - y_predicted)     #Change in intercept slope
        m_curr = m_curr - learning_rate*md   #Changing the m value based on learning rate and change in m slope
        b_curr = b_curr - learning_rate*bd   #Changing the b value based on learning rate and change in b slope

        if math.isclose(cost, prev_cost, rel_tol=1e-20): #Stopping the loop when the cost isn't changing than the required threshold
            print(f"Convergence reached at iteration {i}")
            break
        
        prev_cost = cost

        print("m {} , b {} , cost {} , iteration {}".format(m_curr,b_curr,cost,i))



#  Y = mX + b
#  5 = m1 + b --> we need to predict what our m and b are by using
#  gradient descent and cost function
#  We know that m=2 and b=3
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)
