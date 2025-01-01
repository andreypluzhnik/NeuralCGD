import torch
from torch import nn
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
from train import *
import time


def icholesky(A):
    n = A.shape[0]

    for i in range(n):
        A[i,i] = np.sqrt(A[i,i])
        nz, = A[i + 1:, i].nonzero()
        if(len(nz) > 0):
            nz = nz + (i + 1)
            A[nz, i] = A[nz,i]/A[i,i]
        
        for j in nz:
            k, = A[j:n,j].nonzero()
            if(len(k) > 0):
                k = k + j
                A[k, j] = A[k, j] - A[k, i] * A[j, i]
        
    return np.tril(A)



def next_direction(X):
    model.eval() # set model to evaluate mode


    with torch.no_grad():
    #    res = res.to(DEVICE)
        X = X.view((1, 1, dim_x, dim_y, dim_z))
        X = model(X).squeeze().flatten()  
        return X




def pgd_backsolve(A:scipy.sparse.csr_matrix, R:scipy.sparse.csr_matrix,b:np.ndarray,tol = 1e-10, max_it = 3e5):
    # solve P^T A P (iPx) = P^T b
    # R^-1 = P * P^T
    
    x = np.zeros(len(b))
    r = b.copy()
    z = scipy.sparse.linalg.spsolve_triangular(R, r, lower = True) # stores iM * res
    z = scipy.sparse.linalg.spsolve_triangular(R.transpose(), z, lower = False)
    r_old = r.copy()
    z_old = z.copy() # stores iM * res_old
    d = z.copy()

    count = 0
    res_norm = np.linalg.norm(r)
    res_arr = [res_norm]

    while(res_norm > tol and count < max_it):
        if(count % 2 == 0):
            print(f"count: {count}    residual: {res_norm}")
        # print(res_arr[-1])
        alpha = r.dot(z)/d.dot(A.dot(d))
        x = x + alpha * d
        
        r_old = r.copy()
        z_old = z.copy()
        r = r_old - alpha * A.dot(d) # A.dot gives a matrix-vector product in this case
        # R backsolve
        z = scipy.sparse.linalg.spsolve_triangular(R, r, lower = True) 
        z = scipy.sparse.linalg.spsolve_triangular(R.transpose(), z, lower = False)

        beta = r.dot(z) / r_old.dot(z_old)
        d = z + beta * d
        res_norm = scipy.linalg.norm(r, 2)
        res_arr = res_arr + [res_norm]
        
        count+=1

    print(f"Finished in {count} iterations with residual {res_arr[-1]}")
    return res_arr, x, count



def neural_pgd(A, b:torch.Tensor, tol = 1e-16, max_it = 100):
    x = torch.zeros(b.shape[0]).to(DEVICE)
    dk = torch.zeros(x.shape[0]).to(DEVICE)
    dkk = torch.zeros(x.shape[0]).to(DEVICE)
    ak = 1
    akk = 1
   
    d = torch.zeros(x.shape[0])
    res = b.clone()
    res_norm = torch.linalg.norm(res).item()
    # res_old = res.clone()
    res_arr = [res_norm]
    count = 0

    while(res_norm > tol and count < max_it):
        if(count % 2 == 0):
            print(f"count: {count}    residual: {res_norm}")
        d = next_direction(torch.div(res, res_norm))
        d_temp = d.clone()
        d = torch.sub(input = d, alpha = 1, other = torch.dot(d_temp, torch.mv(A, dk))/ak * dk)
        d = torch.sub(input = d, alpha = 1, other = torch.dot(d_temp, torch.mv(A, dkk))/akk * dkk)
        # d -= dk * torch.dot(d, torch.mv(A, dk))/ak - dkk * torch.dot(d, torch.mv(A, dkk))/akk

        akk = ak
        ak = torch.dot(d, torch.mv(A, d))
        alpha = torch.dot(res, d)/ak
        # x = x + alpha * d
        x = torch.add(input = x, alpha = alpha, other = d)

        dkk = dk.clone()
        dk = d.clone()

        res = torch.sub(input = b, alpha = 1, other = torch.mv(A, x))
        res_norm = torch.norm(res,2) 
        res_arr = res_arr + [res_norm.item()]
        count += 1
    
    print(f"Finished in {count} iterations with residual {res_arr[-1]}")
    return res_arr, x, count


if __name__ == "__main__":
    dim_x, dim_y, dim_z = 8, 8, 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.double)

    domain_type = "random"
    filling_fraction = 0.35
    
    fn_map = {"cube" : {"data suffix" : "cube", 
                        "plot title":"PCG vs Neural Preconditioner for 8x8x8 Domain with Cube",
                        "save name":"cube_plot.png"},
                        
            "sphere" : {"data suffix" : "sphere", 
                        "plot title":"PCG vs Neural Preconditioner for 8x8x8 Domain with Sphere",
                        "save name":"sphere_plot.png"},

            "empty" : {"data suffix" : "", 
                        "plot title":"PCG vs Neural Preconditioner for Empty 8x8x8 Domain",
                        "save name":"empty_plot.png"},
            
            "torus" : {"data suffix" : "torus", 
                "plot title":"PCG vs Neural Preconditioner for 8x8x8 Domain with Torus",
                "save name":"torus_plot.png"},
            
            "random" : {"data suffix" : f"random_ff_{filling_fraction}", 
                "plot title":f"PCG vs Neural Preconditioner for 8x8x8 Domain with {filling_fraction * 100}% Random Fill",
                "save name":f"random_{filling_fraction}_plot.png"},
            }


    data_suffix = fn_map[domain_type]["data suffix"]
    plot_title = fn_map[domain_type]["plot title"]
    save_fn = fn_map[domain_type]["save name"]

    models_dir = f"./models/"
    model_fn = "8_8_8_grid_state_16_mod.pth"
    
        
    model = CNN(dim_x, dim_y, dim_z).to(DEVICE)
    model.load_state_dict(torch.load(models_dir + model_fn))
    model.eval()
    matrix_fn = f"A_matrix_8_8_8_{data_suffix}"
    # load A matrix, its stored as COO
    A = scipy.sparse.load_npz(f"./data/{matrix_fn}.npz")
    
    # generate a random test vector
    vectors = np.load(f"./data/{matrix_fn}_data.npy")
    idx = random.randint(0,vectors.shape[0] - 1)
    b = vectors[idx,:]
    # find solution to desired tolerance
    tol = 1e-8
    

    

    ################ Benchmark PCGD
    # create a preconditioner
    dA = A.toarray()
    dA = dA[A.getnnz(0) > 0,:]
    dA = dA[:,A.getnnz(1)>0]
    db = b[A.getnnz(1) > 0]
    R = np.sqrt(dA.diagonal())
    R = scipy.sparse.diags(R).tocsc()

    start_time = time.time()
    res_arr, x, its = pgd_backsolve(dA.copy(), R, db, tol = tol)
    finish_time = time.time()
    time_taken = finish_time - start_time

    
    err = np.linalg.norm(db - dA.dot(x))
    pgd_result_str = f"PCGD finished in {its} iterations with residual {err}. It took {time_taken/its} seconds per it."
    
    ################ Benchmark ICPCGD
    # create a preconditioner
    R = icholesky(dA.copy())

    start_time = time.time()
    ic_res_arr, x, its = pgd_backsolve(dA, R, db, tol = tol)
    finish_time = time.time()
    time_taken = finish_time - start_time

    err = np.linalg.norm(db - dA.dot(x))
    icpgd_result_str = f"PCGD finished in {its} iterations with residual {err}. It took {time_taken/its} seconds per it."



    ################ Benchmark Neural PCGD
    # convert matrix to pytorch sparse format
    A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data, size = [dim_x * dim_y * dim_z, dim_x * dim_y * dim_z])
    A = A.to(DEVICE, dtype=torch.double)
    b = torch.from_numpy(b).to(DEVICE, dtype = torch.double).squeeze()
    start_time = time.time()
    
    
    start_time = time.time()
    n_res_arr, x, its = neural_pgd(A, b, tol = tol)
    finish_time = time.time()
    time_taken = finish_time - start_time


    err = torch.linalg.norm(torch.sub(input = b, other = torch.mv(A, x)), 2)
    npgd_result_str = f"Neural PCGD finished in {its} iterations with residual {err}. It took {time_taken/its} seconds per it."

    print("\n" + pgd_result_str)
    print(npgd_result_str)
    print(icpgd_result_str)
    plt.plot(np.arange(0, len(res_arr)), res_arr, label = "PGD")
    plt.plot(np.arange(0, len(n_res_arr)), n_res_arr, label = "NPGD")
    plt.plot(np.arange(0, len(ic_res_arr)), ic_res_arr, label = "ICPGD")
    plt.title(plot_title)
    plt.yscale('log')
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig(f"./plots/{save_fn}")
    plt.show()
