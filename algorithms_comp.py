import numpy as np
import time
import pickle
from scipy.stats import norm
from copy import deepcopy
import scipy.stats as st
from scipy.stats import multivariate_normal as mltnorm
from scipy.stats import randint
import scipy
from tqdm import tqdm



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
    stepsize = tuning_stepsize_param

    alpha_coeff = (1.0 / (96*(n**2)*L))

    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    for k in tqdm(range(int(N*m*1.0/bs))):
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

        # Start of param args
        x = tau*z + (1-tau)*yk

        # Compute grad
        f_1 = f(x+t*e, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])
        f_2 = f(x,     [A_for_batch[batch_ind], y[batch_ind], l2, sparse])

        xi_1 =  delta*np.sin(1.0/(np.linalg.norm(x + t*e - x_star)**2))
        xi_2 =  delta*np.sin(1.0/(np.linalg.norm(x - x_star)**2))

        tnabla = e * ((f_1 + xi_1)-(f_2 + xi_2)) * 1.0/t

        # Continue updating args
        yk = x - tnabla * stepsize
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



def rdfds_e_noise_logreg(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
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

    x = deepcopy(x_init)

    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])

    number_of_directions = 1000
    number_of_samples = np.min([int(N*m*1.0/bs), number_of_directions])*n
    temp_arr = norm().rvs(size=number_of_samples)

    directions_counter = 0

    indices = randint.rvs(low=0, high=m, size=min(int(N*m*1.0/bs), int(1e10/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(x,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

    stepsize = tuning_stepsize_param / n

    alpha = (1.0 / (48*n*L))

    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    for k in tqdm(range(int(N*m/bs))):
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

        # Compute grad
        f_1 = f(x+t*e, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])
        f_2 = f(x,     [A_for_batch[batch_ind], y[batch_ind], l2, sparse])

        xi_1 =  delta*np.sin(1.0/(np.linalg.norm(x + t*e - x_star)**2))
        xi_2 =  delta*np.sin(1.0/(np.linalg.norm(x - x_star)**2))

        tnabla = e * ((f_1 + xi_1) -(f_2 + xi_2)) * 1.0/t

        # Arg update
        x = x - tnabla  * stepsize

        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)

    res = {'last_iter'   : x,
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    with open("dump/"+filename+"_RDFDS_E_logreg_noise_steps_const_"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)

    return res


def rand_sample_on_dsphere(d):
    vec = np.random.randn(d)
    vec /= np.linalg.norm(vec)
    return vec

# Proposed in the paper
def ZO_AccSGD(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
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
    rho = args[8]

    print(rho)

    theoretical = False

    t = args[-1]

    m, n = A.shape

    # Smoothness constant 
    beta = 5

    def compute_Kernal(beta, r):
        if (beta == 1 or beta == 2):
            Kernal = 3*r
        elif (beta == 3 or beta == 4):
            Kernal = ((15*r) / 4) * (5 - 7 * r**3)
        elif (beta == 5 or beta == 6):
            Kernal = ((195*r) / 64) * (99 * r**4 - 126 * r**2 + 35)
        return Kernal
    
    # Parameters setting
    a = 0
    gamma = 0
    alpha = 0

    # if theoretical:
    #     rho = max(1, 4 * n * (3 * beta**3) / bs) * tuning_stepsize_param
    #     stepsize = 1. / (L*rho)
    # else:
    #     rho = bs
    #     stepsize = (1 / (bs * L))

    stepsize = 1/(L * rho)

    # oracle calls
    K = int(N*m*1.0/bs)

    # dumping_constant = np.max([int(N*m/(bs*10000)), 1])
    dumping_constant = 1

    yk = deepcopy(x_init)
    xk = deepcopy(x_init)
    zk = deepcopy(x_init)

    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])

    indices = randint.rvs(low=0, high=m, size=min(int(N*m/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0

    number_of_samples = (N*bs)*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(xk,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()


    for k in tqdm(range(N)):

        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)

        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs

        # Params update
        a = gamma * np.sqrt(stepsize * rho)
        gamma = (rho**(-1) + np.sqrt(rho**(-2) + 4 * gamma**2)) / 2
        alpha = (gamma * stepsize) / (gamma * stepsize + a**2)

        # Args update
        yk = alpha * zk + (1 - alpha) * xk

        grad_estim = 0

        for _ in range(bs):

            if directions_counter == K-1:
                temp_arr = norm().rvs(size=number_of_samples)
                directions_counter = 0

            e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
            e = e_unnormalized/np.linalg.norm(e_unnormalized)

            directions_counter += 1

            r = np.random.uniform(-1,1)
            Kernal = compute_Kernal(beta, r)
            
            f_1 = f(yk + t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
            f_2 = f(yk - t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
            xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)
            grad_estim += (((f_1 + xi_1) - (f_2 + xi_2)) /(2*t)) * Kernal * e

        grad_estim /= bs

        xk = yk - grad_estim * stepsize
        zk = zk - gamma * stepsize * grad_estim
        zk = zk - gamma * stepsize * grad_estim

        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(xk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)



    # for k in tqdm(range(int(N*m*1.0/bs))):
        # if directions_counter == K-1:
        #     temp_arr = norm().rvs(size=number_of_samples)
        #     directions_counter = 0
    #     if indices_counter == indices_size:
    #         indices_counter = 0
    #         indices = randint.rvs(low=0, high=m, size=indices_size)

    #     batch_ind = indices[indices_counter:(indices_counter+bs)]
    #     indices_counter += bs

    #     # Params update
    #     a = gamma * np.sqrt(stepsize * rho)
    #     gamma = (rho**(-1) + np.sqrt(rho**(-2) + 4 * gamma**2)) / 2
    #     alpha = (gamma * stepsize) / (gamma * stepsize + a**2)

    #     # Args update
    #     yk = alpha * zk + (1 - alpha) * xk

        # e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
        # e = e_unnormalized/np.linalg.norm(e_unnormalized)
    #     directions_counter += 1
    #     r = np.random.uniform(-1,1)
    #     Kernal = compute_Kernal(beta, r)

    #     f_1 = f(yk + t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
    #     f_2 = f(yk - t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
    #     xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)

    #     grad_estim = (((f_1 + xi_1) - (f_2 + xi_2)) /(2*t)) * Kernal * e

    #     # grad_estim = 0
    #     # for _ in range(B):

    #     #     # Calculate grad
    #     #     # f_1 = f(yk + t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
    #     #     # f_2 = f(yk - t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])

    #     #     e = rand_sample_on_dsphere(n)
    #     #     r = np.random.uniform(-1,1)

    #     #     Kernal = compute_Kernal(beta, r)

    #     #     f_1 = f(yk + t*r*e,[A_for_batch, y, l2, sparse])
    #     #     f_2 = f(yk - t*r*e,[A_for_batch, y, l2, sparse])

    #     #     if theoretical:
    #     #         eps = 1e-10
    #     #         if bs > 1 and bs <= 15000:
    #     #             delta = np.sqrt(eps**3 / n)
    #     #         else:
    #     #             delta = np.sqrt(np.pow(eps, (3.0 * beta + 1)/(4.0 * (beta - 1))) * np.sqrt(bs) / n)
    #     #         xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)
    #     #     else:
    #     #         xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)

    #     #     # TODO: откуда вообще берется * 1.0? 
    #     #     grad_estim += (((f_1 + xi_1) - (f_2 + xi_2)) /(2*t)) * Kernal * e

    #     # grad_estim /= B

    #     # # Calculate grad
    #     # f_1 = f(yk + t*r*e,[A_for_batch[  batch_ind], y[batch_ind], l2, sparse])
    #     # f_2 = f(yk - t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])

    #     # if theoretical:
    #     #     eps = 1e-10
    #     #     if bs > 1 and bs <= 15000:
    #     #         delta = np.sqrt(eps**3 / n)
    #     #     else:
    #     #         delta = np.sqrt(np.pow(eps, (3.0 * beta + 1)/(4.0 * (beta - 1))) * np.sqrt(bs) / n)
    #     #     xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)
    #     # else:
    #     #     xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)

    #     # # TODO: откуда вообще берется * 1.0? 
    #     # grad_estim = (1/B) * (((f_1 + xi_1) - (f_2 + xi_2)) /(2*t)) * Kernal * e

    #     xk = yk - grad_estim * stepsize
    #     zk = zk - gamma * stepsize * grad_estim

    #     if ((k+1) % dumping_constant == 0):
    #         iters = np.append(iters, k+1)
    #         tim = np.append(tim, time.time() - t_start)
    #         conv_f = np.append(conv_f, f(xk, [A, y, l2, sparse_full]) - f_star)
    #         sample_complexity = np.append(sample_complexity, (k+1)*2*bs)

    res = {'last_iter'   : xk,
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    
    with open("dump/"+filename+"Our_algorithm"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+"_rho_"+str(rho)+".txt", 'wb') as file:
        pickle.dump(res, file)

    return res

def ZO_AccSGD_nllloss(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
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
    rho = args[8]

    theoretical = False

    t = args[-1]

    m, n = A.shape

    # Smoothness constant 
    beta = 5

    # Overbatching constant 
    # B = 100000

    def compute_Kernal(beta, r):
        if (beta == 1 or beta == 2):
            Kernal = 3*r
        elif (beta == 3 or beta == 4):
            Kernal = ((15*r) / 4) * (5 - 7 * r**3)
        elif (beta == 5 or beta == 6):
            Kernal = ((195*r) / 64) * (99 * r**4 - 126 * r**2 + 35)
        return Kernal
    
    # Parameters setting
    a = 0
    gamma = 0
    alpha = 0

    # if theoretical:
    #     rho = max(1, 4 * n * (3 * beta**3) / bs) * tuning_stepsize_param
    #     stepsize = 1. / (L*rho)
    # else:
    #     rho = bs
    #     stepsize = (1 / (bs * L))

    stepsize = 1/(L * rho)

    # oracle calls
    K = int(N*m*1.0/bs)

    dumping_constant = np.max([int(N*m/(bs*10000)), 1])

    yk = deepcopy(x_init)
    xk = deepcopy(x_init)
    zk = deepcopy(x_init)

    conv_f = np.array([])
    iters = np.array([])
    tim = np.array([])
    sample_complexity = np.array([])

    indices = randint.rvs(low=0, high=m, size=min(int(N*m/bs), int(100000/bs))*bs)
    indices_size = len(indices)
    indices_counter = 0

    number_of_samples = K*n
    temp_arr = norm().rvs(size=number_of_samples)
    directions_counter = 0

    t_start = time.time()
    tim = np.append(tim, time.time() - t_start)
    iters = np.append(iters, 0)
    conv_f = np.append(conv_f, f(xk,[A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()


    for k in tqdm(range(N)):

        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)

        batch_ind = indices[indices_counter:(indices_counter+bs)]
        indices_counter += bs

        # Params update
        a = gamma * np.sqrt(stepsize * rho)
        gamma = (rho**(-1) + np.sqrt(rho**(-2) + 4 * gamma**2)) / 2
        alpha = (gamma * stepsize) / (gamma * stepsize + a**2)

        # Args update
        yk = alpha * zk + (1 - alpha) * xk

        grad_estim = 0

        for _ in range(bs):

            if directions_counter == K-1:
                temp_arr = norm().rvs(size=number_of_samples)
                directions_counter = 0

            e_unnormalized = temp_arr[directions_counter*n:(directions_counter+1)*n]
            e = e_unnormalized/np.linalg.norm(e_unnormalized)

            directions_counter += 1

            r = np.random.uniform(-1,1)
            Kernal = compute_Kernal(beta, r)
            
            f_1 = f(yk + t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
            f_2 = f(yk - t*r*e,[A_for_batch[batch_ind], y[batch_ind], l2, sparse])
            xi_1, xi_2 = st.uniform(loc=-delta, scale=2*delta).rvs(size=2)
            grad_estim += (((f_1 + xi_1) - (f_2 + xi_2)) /(2*t)) * Kernal * e

        grad_estim /= bs

        xk = yk - grad_estim * stepsize
        zk = zk - gamma * stepsize * grad_estim
        zk = zk - gamma * stepsize * grad_estim

        if ((k+1) % dumping_constant == 0):
            iters = np.append(iters, k+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(xk, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (k+1)*2*bs)

    res = {'last_iter'   : xk,
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    
    with open("dump/"+filename+"Our_algorithm_nllloss"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)

    return res



def ZO_VARAG(filename, x_init, args, bs=1, N=100, f_star=None, x_star=None, tuning_stepsize_param=1.0):
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

    x = x_init
    x_overscore = deepcopy(x_init)
    x_tilde = deepcopy(x_init)

    s_0 = np.round(np.log((n + 4) * bs) + 1)

    sum_thetas = 0
    sum_thetas_n_xs = 0

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
    conv_f = np.append(conv_f, f(x_tilde, [A, y, l2, sparse_full]) - f_star)
    sample_complexity = np.append(sample_complexity, 0)

    stepsize = tuning_stepsize_param

    if sparse:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()

    for s in tqdm(range(int(N*m*1.0/bs))):

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

        # Param tau for non-strongly convex problems equals 0
        tau = 0

        # Max inner steps is set as bs -> will not been updted
        if s <= s_0:
            alpha = 0.5
        else:
            alpha = 2 / (s - s_0 + 4)
        
        p = 0.5
        
        gamma = stepsize / (12 * (n + 4) * alpha * L)
        theta = (gamma / alpha)

        # Preparations before starting inner loop 

        # \tile{x} = \tilde{x}^{s - 1}
        x_tilde_ = deepcopy(x_tilde)

        # Pivotal gradient
        f_1 = f(x_tilde_ + t * e, [A, y, l2, sparse_full])
        f_2 = f(x_tilde_ - t * e, [A, y, l2, sparse_full])

        xi_1 =  delta*np.sin(1.0/(np.linalg.norm(x_tilde_ + t * e - x_star)**2))
        xi_2 =  delta*np.sin(1.0/(np.linalg.norm(x_tilde_ - t * e - x_star)**2))

        grad_full = ((f_1 + xi_1) - (f_2 + xi_2)) * 1. / (2 * t) * e
        # Starting inner loop

        x_underscore = ((1 + tau * gamma) * (1 - alpha - p) * x_overscore +
                         alpha * x + 
                        (1 + tau * gamma) * p * x_tilde_) / (1 + tau * gamma * (1- alpha))
        
        # g(\undersore{x}_t, u_t, i_t)        
        batch_ind = [np.random.randint(0, bs)]

        f_1 = f(x_underscore + t * e_unnormalized, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])
        f_2 = f(x_underscore, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])

        xi_1 = delta*np.sin(1.0/(np.linalg.norm(x_underscore + t * e_unnormalized - x_star)**2))
        xi_2 = delta*np.sin(1.0/(np.linalg.norm(x_underscore - x_star)**2))
                                    
        grad_1 = e_unnormalized * ((f_1 + xi_1) - (f_2 + xi_2)) / t

        # g(\tilde{x}_y, u_t, i_t)
        f_1 = f(x_tilde_ + t * e_unnormalized, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])
        f_2 = f(x_tilde_, [A_for_batch[batch_ind], y[batch_ind], l2, sparse])

        xi_1 = delta*np.sin(1.0/(np.linalg.norm(x_tilde_ + t * e_unnormalized - x_star)**2))
        xi_2 = delta*np.sin(1.0/(np.linalg.norm(x_tilde_ - x_star)**2))
                                    
        grad_2 = e_unnormalized * ((f_1 + xi_1) - (f_2 + xi_2)) / t 

        # Total gradient approximation G
        G = grad_1 - grad_2 + grad_full

        x = (gamma * G + gamma * tau * x_underscore + x)/(1 + gamma)
        x_overscore = (1 - alpha - p) * x_overscore + alpha * x + p * x_tilde_

        sum_thetas += theta
        sum_thetas_n_xs += theta * x_overscore

        x_tilde = sum_thetas_n_xs / sum_thetas

        if ((s+1) % dumping_constant == 0):
            iters = np.append(iters, s+1)
            tim = np.append(tim, time.time() - t_start)
            conv_f = np.append(conv_f, f(x_tilde, [A, y, l2, sparse_full]) - f_star)
            sample_complexity = np.append(sample_complexity, (s+1)*2*bs)

    res = {'last_iter'   : x_tilde,
           'func_vals'   : conv_f,
           'iters'       : iters,
           'time'        : tim,
           'oracle_calls': sample_complexity}
    
    with open("dump/"+filename+"ZO_VARAG"+str(tuning_stepsize_param)+"_epochs_"+str(N)+
              "_delta_"+str(delta)+"_batch_"+str(bs)+".txt", 'wb') as file:
        pickle.dump(res, file)

    return res