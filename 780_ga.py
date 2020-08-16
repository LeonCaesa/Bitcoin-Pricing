#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Fri Apr 10 16:19:38 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
import geatpy as ea
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from sklearn import linear_model
from scipy.special import spence as Li2
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')

# data=pd.read_csv('bitcoin_option_data.csv')
data = pd.read_csv('bitcoin_data_cleaned.csv')
i_ = complex(0, 1)

warnings.filterwarnings("ignore")


# data cleaning
# only the out of money option

data = data[0.5 * data['Bid_price'] + 0.5 * data['Ask_price'] <= 1200]
data = data[data['T'] < 360]
data = data[data['S0'] < data['K']]  # out of money
data = data.sort_values(by='K')

# plt.scatter(data['K'].values,0.5*data['Ask_price']+0.5*data['Bid_price'])
# plt.xlabel('K')
#plt.ylabel('Option Price')
"""
call_data=data[data['is_call']==1]
put_data=data[data['is_call']==0]

K_list=list(set(data['K'].values))
T_list=list(set(data['T'].values))

for k in K_list:
    for t in T_list:
        flag_c=(call_data['T']==t) & (call_data['K']==k)
        flag_p=(put_data['T']==t) & (put_data['K']==k)

        if (sum(flag_c)!=0)&(sum(flag_p!=0)):
            
            left=call_data[flag_c]['Bid_price'].values-put_data[flag_p]['Ask_price'].values

            middle=call_data[flag_c]['S0'].values-k
            
            right=call_data[flag_c]['Ask_price'].values-put_data[flag_p]['Bid_price'].values
        
            if not (left<=middle)&(right>=middle):
                print(k)
                print(t)
                call_data=call_data[flag_c==False]
                put_data=put_data[flag_p==False]
   """


def phi_tau(u, lambda_, b, m, a, t, A0=0.5):
    """
    characteristic function where tau t following an OU process with gamma innovation


    param: lambda_, mean level in OU process
    param: a, scale parameter in gamma innovation    
    param: b, shape parameter in gamma innovation    
    param: A0, initial state A0        
    param: t, time step
    """
    constant = np.power(lambda_ * b / (lambda_ * b - i_ * u), a * t)

    second = i_ * (m * t + (A0 - m) * (1 - np.exp(-lambda_ * t)) / lambda_) * u

    third = a / lambda_ * (Li2(1 - i_ * u / (i_ * u - lambda_ * b)) -
                           Li2(1 - i_ * np.exp(-lambda_ * t) * u / (i_ * u - lambda_ * b)))

    return constant * np.exp(second + third)

# phi_tau(1,1,1,1,1,1)


def phi_xt(u, alpha, theta, mu, sigma, lambda_, b, m, a, t, A0=0.5):
    """
    characteristic function of Log St -S0
    param: u, point of evaluation for characteristic function

    # para inherited from return innovation
    param: alpha, gamma shape and scale parameter
    param: theta, Brownian location parameter
    param: sigma, Brownian scale parameter
    param: mu, drift parameter


    # para inherited from phi_tau
    param: lambda_, mean level in OU process
    param: a, scale parameter in gamma innovation    
    param: b, shape parameter in gamma innovation    
    param: A0, initial state A0        
    param: t, time step
    """
    first = mu * u

    second = (2 * alpha - 2 * i_ * theta * u + sigma**2 * u**2) / (2 * alpha)

    return phi_tau(first + i_ * alpha * np.log(second), lambda_, b, m, a, t, A0)


def call_price(S0, K, r, T, alpha, theta, mu, sigma, lambda_, b, m, a, t, A0=0.5):
    """
    function to calculate european option price under stochastic volatility with stoachstic time shift

    """
    def phi_xt_temp(u): return phi_xt(u, alpha, theta,
                                      mu, sigma, lambda_, b, m, a, T, A0)

    def constant(u): return phi_xt_temp(u - i_) / (phi_xt_temp(-i_) * i_ * u)
    def inside_exp(u): return -i_ * u * np.log((K / S0) * phi_xt_temp(-i_))
    def inte_mu(u): return np.real(constant(u) * np.exp(inside_exp(u)))

    C1 = S0 * (0.5 + 1 / np.pi * quad(inte_mu, 0, np.inf)[0])
    def inte_mu2(u): return np.real(phi_xt_temp(u) * np.exp(-i_ *
                                                            u * np.log(K / S0 * phi_xt_temp(-i_))) / (i_ * u))

    C2 = K * np.exp(-r * T) * (0.5 + 1 / np.pi * quad(inte_mu2, 0, np.inf)[0])

    return C1 - C2


def evaluate_price(params, predictor=False):
    # def evaluate_price(alpha,theta,mu,sigma,lambda_,b,m,a,A0,predictor=False):
    """
    Function to evaluate price given parameter params
    """
    alpha = params[0]

    theta = params[1]

    mu = params[2]

    sigma = params[3]

    lambda_ = params[4]

    b = params[5]
    m = params[6]
    a = params[7]
    A0 = params[8]

    call_data = data[data['is_call'] == 1]
    r = 0
    target_list = (call_data['Ask_price'].values +
                   call_data['Bid_price'].values) / 2

    def eval_row(row): return call_price(
        row['S0'], row['K'], r, row['T'] / 360, alpha, theta, mu, sigma, lambda_, b, m, a, row['T'] / 360, A0=A0)
    pred_list = call_data.apply(eval_row, axis=1).values

    rmse = np.sqrt(np.mean((target_list - pred_list)**2))

    print(rmse)
    # plt.plot(target_list)
    # plt.plot(pred_list)
    plt.scatter(call_data['K'].values, 0.5 * call_data['Ask_price'] +
                0.5 * call_data['Bid_price'], label='Real Data')
    plt.scatter(call_data['K'].values, pred_list,
                label='SV Time change', marker='+')
    plt.xlabel('K')
    plt.ylabel('Option Price')
    plt.legend()
    plt.show()
    if predictor:
        return rmse, pred_list
    return rmse

# plt.plot(target_list)
# plt.plot(pred_list)
# call_price(S0=500,K=400,r=0,T=1,alpha=1,theta=1,mu=1,sigma=1,lambda_=1,b=1,m=1,a=1,t=1,A0=0.5)


alpha = 8.5
mu = 0.23
theta = -0.17
sigma = 0.56
lambda_ = 2
m = 0.1
A0 = 0.5
a = 2.72
b = 1.44
S0 = 100
K = 21
r = 0
T = 1
t = T
# call_price(S0,K,r,T,alpha,theta,mu,sigma,lambda_,b,m,a,t,A0=0.5)

params = [alpha, theta, mu, sigma, lambda_, b, m, a]

# error=evaluate_price(params)


#bnds = ((0, None), (-1, 1),(-1,1),(0,1),(0,None),(0,None),(0,1),(0,None))
#bnds = ((0, None), (None, None),(None,None),(0,None),(0,None),(0,None),(0,None),(0,None))
bnds = ((0, None), (None, None), (None, None), (0, 1),
        (0, None), (0, None), (0, 2), (0, None), (0, None))

up = [i[0] for i in bnds]


#9.56643287, -0.13491088,  0.1918937 ,  0.79558653,  1.85475686,
#       1.26232333,  0.09530501,  3.21454544

# array([ 9.5566267 , -0.13498061,  0.19209815,  0.79572496,  1.85627467,
#       1.26279405,  0.09570665,  3.21255027])
# x0=np.array([ 9.56643287, -0.13491088,  0.1918937 ,  0.79558653,  1.85475686,
#        1.26232333,  0.09530501,  3.21454544])

x0 = [1 / 0.11, -0.39, 0, 0.73, 0.01, 0.01, 0.01, 0.01, 1]

#res=minimize(evaluate_price,x0=x0,method='Nelder-Mead',options={"maxiter" : 100,'disp': True}, bounds=bnds)

#res=minimize(evaluate_price,x0=x0,method='Nelder-Mead',options={"maxiter" : 1000,'disp': True}, bounds=bnds,callback=make_minimize_cb(path_))


#df=pd.DataFrame([[path_[i],cost_values[i]] for i in range(len(path_))])
# df.to_csv('result_reasonable.csv')


#x0=[1/0.11,-0.39,0,0.73, 0.01,0.01,  0.01,  0.01]
# evaluate_price(x0,A0=1)

x0 = [9.9999,
      -0.200711814,
      9.9999,
      0.46618614,
      4.579633717,
      9.405713272,
      0.551850043,
      7.667517907,
      2.302655249]
evaluate_price(x0, predictor=True)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1, PoolType='Process'):
        name = 'Option Pricing'  # 初始化name（函数名称，可以随意设置）
        Dim = 9  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）

        lb = [0, -10, -10, 0, 0, 0, 0, 0, 0]  # 决策变量下界
        ub = [10, 10, 10, 1, 10, 10, 2, 10, 10]  # 决策变量上界

        lbin = [1, 0, 0, 1, 1, 1, 1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0, 0, 0, 1, 0, 0, 1, 0, 0]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

        self.data = data[data['is_call'] == 1]

        self.dataTarget = (
            self.data['Ask_price'].values + self.data['Bid_price'].values) / 2

    def aimFunc(self, pop):  # objective function

        Vars = pop.Phen  # return the decision matrix
        args = list(zip(list(range(pop.sizes)), [
                    Vars] * pop.sizes, [self.data] * pop.sizes, [self.dataTarget] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())


def subAimFunc(args):

    i = args[0]
    Vars = args[1]
    call_data = args[2]
    target_list = args[3]

    alpha = Vars[i, 0]

    theta = Vars[i, 1]
    mu = Vars[i, 2]
    sigma = Vars[i, 3]
    lambda_ = Vars[i, 4]

    b = Vars[i, 5]

    m = Vars[i, 6]

    a = Vars[i, 7]
    A0 = Vars[i, 8]
    r = 0

    def eval_row(row): return call_price(
        row['S0'], row['K'], r, row['T'] / 360, alpha, theta, mu, sigma, lambda_, b, m, a, row['T'] / 360, A0=A0)
    pred_list = call_data.apply(eval_row, axis=1).values

    rmse = np.sqrt(np.mean((target_list - pred_list)**2))

    ObjV_i = [rmse]

    print(rmse)
    return ObjV_i


error_result = pd.read_csv('error 2.csv')
error_result.iloc[:, 1:].plot()


x0 = [8.554, -0.4, -0., 0.727, 0.01, 0.01, 0.01, 0.01, 0.975]
evaluate_price(x0, predictor=True)


if __name__ == '__main__':
    PoolType = 'Process'  # 设置采用多进程，若修改为: PoolType = 'Thread'，则表示用多线程
    problem = MyProblem()  # 生成问题对象
    """=================================Population Setup==============================="""
    Encoding = 'RI'       # 编码方式
    NIND = 30             # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes,
                      problem.ranges, problem.borders)  # 创建区域描述器
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population = ea.Population(Encoding, Field, NIND)
    """===============================Algorithm Setup============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(
        problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500  # 最大进化代数
    myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
    # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.maxTrappedCount = 50
    """==========================Optimization======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
   # population.save() # 把最后一代种群的信息保存到文件中
    problem.pool.close()  # 及时关闭问题类中的池，否则在采用多进程运算后内存得不到释放
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])  # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]

    variables = pd.DataFrame(var_trace)
    error = pd.DataFrame(obj_trace, columns=['Average', 'Best'])
    variables.to_csv('variable.csv')
    error.to_csv('error.csv')
   # print('最优的目标函数值为：%s'%(best_ObjV))
   # print('最优的控制变量值为：')
   # for i in range(var_trace.shape[1]):
    #    print(var_trace[best_gen, i])
   # print('有效进化代数：%s'%(obj_trace.shape[0]))
   # print('最优的一代是第 %s 代'%(best_gen + 1))
   # print('评价次数：%s'%(myAlgorithm.evalsNum))
   # print('时间已过 %s 秒'%(myAlgorithm.passTime))
    """=================================Result==============================="""
    #problem.test(C = var_trace[best_gen, 0], G = var_trace[best_gen, 1])
