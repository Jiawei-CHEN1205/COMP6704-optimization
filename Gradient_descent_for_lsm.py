# Author: Jiawei CHEN
# Date: 2022/10/19
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10) # seed 保证每次随机生成数相同
alpha = 0.005 # 学习率0.01,0.005,0.001
eps = 1e-6  # 误差范围eps 对于满足线性关系的一组(x,y)，其之后的拟合效果可以使得误差不超过这个阈值;
# 但是在本实验中，发现规定的迭代次数内不能达到这个条件，因此这个限制条件失去作用,
# 所以每次设置的迭代次数就是最后实际的运行次数，而没有在这个迭代次数内达到eps收敛。
  
# 100个测试点集（在线性函数的基础上，对y坐标施加一定的正态分布的误差范围）   
testx = list(np.linspace(1,50, num=100, endpoint=True))   # x
mylist = []
for item in testx:
    mylist.append(item*2 + 3 + 5*np.random.randn())
# testy = np.array(mylist)
testy = mylist # y
 
# 对测试点集绘制散点图
plt.scatter(testx, testy, s = 13)
plt.xlabel('x',fontsize = 15,fontweight = 'semibold')
plt.ylabel('y',fontsize = 15,fontweight = 'semibold')
plt.title('Comparison of the scatter plot and the fitting results of the two methods',fontsize = 15,fontweight = 'semibold')
# plt.show()


# 定义梯度下降函数
'''def my_descent_gradient():
    # m is sample num
    n = len(testx)
    w1, w0= 0, 0 # 斜率和截距的初始值设置为0
    sse, sse_new = 0, 0
    grad_w1, grad_w0 = 0, 0 # 初始梯度设置为0
    count = 0
    for step in range(20000): # iteration number can be modified(e.g.,500,1000,5000,10000 or 20000 times)
        count += 1
        for i in range(n):
            base = w1 * testx[i] + w0 - testy[i] # 预测值和实际值的误差
            grad_w1 += testx[i] * base
            grad_w0 += base
            # 计算参数对应梯度
            grad_w1 = grad_w1 / n
            grad_w0 = grad_w0 / n
            # 根据梯度更新参数
            w1 -= alpha * grad_w1
            w0 -= alpha * grad_w0

            # loss function: Mean Squared Error, MSE
            # because 2n is a const, so 1/2n can be ignored
            for j in range(n):
                sse_new += (w1 * testx[j] + w0 - testy[j]) ** 2
        # 如果在迭代次数以内就达到误差停止准则，则迭代停止，否则迭代次数即为设置的step次数
        if abs(sse_new - sse) < eps: 
            break
        else:
            eps1 = abs(sse_new - sse)/n
            sse = sse_new  
    return w1, w0, count, eps1 # a,b are coefficient, and count is the iteration number(initially the number of times needed to reach convergence).

# my_descent_gradient()
descent_w1 , descent_w0, count ,eps1 = my_descent_gradient()
print('Gradient descent algorithm: y = {0} * x + {1}'.format(descent_w1, descent_w0))
print ("The iteration number is: " , count)
print('The final MSE is: {} '.format(eps1))'''



def my_descent_gradient():
    # m is sample num
    n = len(testx)
    w1, w0= 0, 0 # 斜率和截距的初始值设置为0
    # sse, sse_new = 0, 0
    grad_w1, grad_w0 = 0, 0 # 初始梯度设置为0
    count = 0
    for step in range(50000): # iteration number can be modified(e.g.,500,1000,5000,10000 or 20000 times)
        count += 1
        for i in range(n):
            base = w1 * testx[i] + w0 - testy[i] # 预测值和实际值的误差
            grad_w1 += testx[i] * base
            grad_w0 += base
            # 计算参数对应梯度
            grad_w1 = grad_w1 / n
            grad_w0 = grad_w0 / n
            if abs(grad_w1) < eps and abs(grad_w0) < eps: 
            # 如果在迭代次数以内就达到误差停止准则，则迭代停止，否则迭代次数即为设置的step次数
                break
            else:
                # 根据梯度更新参数
                w1 -= alpha * grad_w1
                w0 -= alpha * grad_w0
            final_gradw1 = abs(grad_w1)
            final_gradw0 = abs(grad_w0)          
    return w1, w0, count, final_gradw1,final_gradw0 # a,b are coefficient, and count is the iteration number(initially the number of times needed to reach convergence).

# my_descent_gradient()
descent_w1 , descent_w0, count ,gradw1,gradw0 = my_descent_gradient()
print('Gradient descent algorithm: y = {0} * x + {1}'.format(descent_w1, descent_w0))
print ("The iteration number is: " , count)
print('The final gradient: grad(w1) = {0}, grad(w0) = {1} '.format(gradw1,gradw0))

# 对于梯度下降法得到的参数进行画图l2
descent_x = np.linspace(1,50, num=100, endpoint=True)
descent_y = descent_w1*descent_x + descent_w0 
l2 = plt.plot(descent_x,descent_y,linewidth = 2,color = 'g',label = 'Gradient descent')


# 计算方法拟合 least squares method
def lsm_algebra(x, y):
    n = len(x)
    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))  # w1:斜率
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x)) # w0:截距  
    return w1, w0

w1, w0 = lsm_algebra(np.array(testx),np.array(testy))
print('The algebraic solution to it: y = {0} * x +{1}'.format(w1, w0))

# 将拟合后的直线结果画图l1
formula_x = np.linspace(1,50, num=100, endpoint=True)
formula_y = w1*formula_x + w0 
l1 = plt.plot(formula_x,formula_y,color = 'r',label='Algebraic solution')

plt.legend(fontsize = 13)
plt.show()





