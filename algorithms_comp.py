import numpy as np
import time
import pickle
from scipy.stats import norm
from copy import deepcopy
from copy import copy
from scipy.stats import multivariate_normal as mltnorm
from scipy.stats import randint
import scipy
from tqdm import tqdm

def zo_accsgd_stoch(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)

    alpha, gamma, a = 0, 0, 0
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(yk,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

    rho = max(1, 4 * n * 125 / m)
    stepsize = 1/L * rho 
    print(rho, stepsize)
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    # for k in range(int(N*1.0/bs)):

    
    for k in range(int(N*m*1.0/bs)):

        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)

        directions_counter += 1

        xi_1 = np.random.normal(0, delta, n)
        xi_2 = np.random.normal(0, delta, n)
        # e = np.random.randn(yk.shape[0])
        # e /= np.linalg.norm(e)
        r = np.random.uniform(-1,1)
        ker = ((195*r) / (64)) * (99 * r**4 - 126 * r**2 + 35)

        # grad = 0
        # for _ in range(m):
        #     # xi_1 = np.random.normal(0, delta, n)
        #     # xi_2 = np.random.normal(0, delta, n)
        #     e = np.random.randn(yk.shape[0])
        #     e /= np.linalg.norm(e)
        #     r = np.random.uniform(-1,1)
        #     ker = ((195*r) / (64)) * (99 * r**4 - 126 * r**2 + 35)
        #     grad += (1/yk.shape[0]) * e * (f(yk+t*r*e, [A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - f(yk-t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]))*ker/(2 * t)
        # grad /= m
            
        grad = (1/n)*e * (f(yk+t*r*e, [A_for_batch[batch_ind], y[batch_ind], l2, sparse]) - f(yk-t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]))*ker/(2 * t)
        

        x = yk - stepsize*grad
        z = z - gamma * stepsize * grad  

        a = stepsize * np.sqrt(stepsize * rho)
        gamma = (rho**(-1) + np.sqrt(rho**(-2) + 4 * gamma**2)) / 2
        alpha = (gamma * stepsize) / (gamma * stepsize + a**2)
         
        yk = alpha * z + (1 - alpha) * x
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_AccSGD_E_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res

# def zo_accsgd_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
#     n = len(x_init)
    
#     dumping_constant = np.max([int(N/(10000)), 1])
    
#     if f_star == None:
#         f_star = 0
    
#     f = args[0]
#     A = args[1]
#     y = args[2]
#     l2 = args[3]
#     sparse = args[4]
#     sparse_full = args[5]
#     L = args[6]
#     delta = args[7]
#     t = args[-1]
    
#     m, n = A.shape
    
#     yk = deepcopy(x_init)
#     x = deepcopy(x_init)
#     z = deepcopy(x_init)
    
#     conv_f = np.array([])
#     iters = np.array([])
#     tim = np.array([])
#     sample_complexity = np.array([])
    
#     number_of_directions = 1000
#     number_of_samples = np.min([N, number_of_directions])*n
#     temp_arr = norm().rvs(size=number_of_samples)
    
#     directions_counter = 0
    
#     t_start = time.time()
#     tim = np.append(tim, time.time() - t_start)
#     iters = np.append(iters, 0)
#     conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
#     sample_complexity = np.append(sample_complexity, 0)
    
#     alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
#     if sparse:
#         A_for_batch = A
#     else:
#         A_for_batch = A.toarray()
    
#     for k in range(N):
#         tau = 2.0 / (k+2)
#         alpha = (k+2) * alpha_coeff
#         if directions_counter == number_of_directions-1:
#             temp_arr = norm().rvs(size=number_of_samples)
#             directions_counter = 0
        
#         e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
#         e = e_unnormalized/np.linalg.norm(e_unnormalized)
#         directions_counter += 1
        
#         x = tau*z + (1-tau)*yk
#         xi_1 = np.random.normal(0, delta, n)
#         xi_2 = np.random.normal(0, delta, n)
#         tnabla = e * ((f(x+t*e,[A_for_batch, y, l2, sparse]) + xi_1) - (f(x,[A_for_batch, y, l2, sparse]) + xi_2)) * 1.0/t
#         yk = x - tnabla * 0.5 / L
#         z = z - tnabla * n * alpha
        
#         if ((k+1) % dumping_constant == 0):
#             iters = np.append(iters, k+1)
#             tim = np.append(tim, time.time() - t_start)
#             conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
#             sample_complexity = np.append(sample_complexity, (k+1)*2*m)

#     return res


def ardfds_e_noise_logreg_full(filename, x_init, args, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    dumping_constant = np.max([int(N/(10000)), 1])
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([N, number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(N):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        x = tau*z + (1-tau)*yk
        xi_1 = np.random.normal(0, delta, n)
        xi_2 = np.random.normal(0, delta, n)
        tnabla = e * ((f(x+t*e,[A_for_batch, y, l2, sparse]) + xi_1) - (f(x,[A_for_batch, y, l2, sparse]) + xi_2)) * 1.0/t
        yk = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*m)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    # with open("dump/"+filename+"_ARDFDS_E_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
    #           "_delta_"+str(delta)+".txt", 'wb') as file:
    #     pickle.dump(res, file)

    return res


def ardfds_e_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)
    
    alpha_coeff = (1.0 / (96*(n**2)*L)) * tuning_stepsize_param
    
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):
        tau = 2.0 / (k+2)
        alpha = (k+2) * alpha_coeff
        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        e = e_unnormalized/np.linalg.norm(e_unnormalized)
        directions_counter += 1
        
        # xi_1 = np.random.normal(0, delta, n)
        # xi_2 = np.random.normal(0, delta, n)
        x = tau*z + (1-tau)*yk
        tnabla = e * ((f(x+t*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]) ) - (f(x,[A_for_batch[batch_ind], y[batch_ind], l2, sparse]))) * 1.0/t
        yk = x - tnabla * 0.5 / L
        z = z - tnabla * n * alpha
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_ARDFDS_E_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res


def zo_varag(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
    n = len(x_init)
    
    if f_star == None:
        f_star = 0
    
    f = args[0]
    A = args[1]
    y = args[2]
    l2 = args[3]
    sparse = args[4]
    sparse_full = args[5]
    L = args[6]
    delta = args[7]
    t = args[-1]
    
    m, n = A.shape
    
    dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    
    yk = deepcopy(x_init)
    x = deepcopy(x_init)
    z = deepcopy(x_init)

    s_0 = np.round(np.log(m*(n + 4)) + 1)
    alpha_s = 0.5
    p_s = 0.5
    gamma_s = 1/(12*(n + 4) * L * alpha_s)
    T_s = bs
    
    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])
    
    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)
    
    directions_counter = 0
    
    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0
    
    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(yk,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

        
    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    for k in range(int(N*m*1.0/bs)):

        if directions_counter == number_of_directions-1:
            temp_arr = norm().rvs(size=number_of_samples)
            directions_counter = 0
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs
        
        # e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        # e = e_unnormalized/np.linalg.norm(e_unnormalized)

        directions_counter += 1

        z_inner = z
        x_tilde = yk
        x_inner = x_tilde

        gamma_s = 1/(12*(n + 4) * L * alpha_s)
        p_s = 0.5
        if k <= s_0:
            alpha_s = 0.5
        else:
            alpha_s = 2/(k - s_0 + 4)

        grad_full = (f(x_inner+t*np.zeros(n), [A_for_batch, y, l2, sparse]) - f(x_inner-t*+t*np.zeros(n),[A_for_batch, y, l2, sparse]))/(2 * t)
 
 
        for t in range(T_s):
            param_theta = 0
            param_thetax = np.zeros_like(x_inner)
            # param update
            if t != T_s - 1:
                theta_t = gamma_s/alpha_s * (alpha_s + p_s)
            else:
                theta_t = gamma_s/alpha_s
        
            w = alpha_s * x_inner + p_s * x_tilde

            index = np.random.choice(range(n))
            u = np.random.randn(n, 1)

            print((w + t*u).shape)


            g_x = (f(w+t*u, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])[index] - f(w, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])[index])/t*u

            g_y = (f(x_tilde+t*u, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])[index] - f(x_tilde, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])[index])/t*u
            grad = g_x - g_y + grad_full
            # grad = grad_full
            z_inner = gamma_s * grad + z_inner
            x_inner = (1 - alpha_s - p_s) * x_inner + alpha_s * z_inner + p_s * x_tilde 
            param_theta += theta_t
            param_thetax += theta_t * x_inner

        z = z_inner
        x = x_inner
        yk = param_thetax / param_theta
        
        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(yk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2)
            
    res = {'last_iter'   : yk, 
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_AccSGD_E_logreg_full_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+".txt", 'wb') as file:
        pickle.dump(res, file)
            
    return res