import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy

lx, ly, lz = None, None, None
domain = None
matrix_fn = ""
training = True

# helpers
def grid_to_row_col(i:int,j:int,k:int) -> int:
    return i + lx * j + lx * ly * k

def is_boundary(i, j, k): # check if cell belongs to the boundary
    if(i == -1 or i == lx or j == -1 or j == ly or k == -1 or k == lz):
        return True
    elif(domain[i, j, k] == True):
        return True
    
    return False


def init_domain(boundary_type:str = "", filling_fraction:float = 0):
    global domain

    filling_fraction = np.clip(filling_fraction, 0 ,1)
    x, y, z = np.indices((lx, ly, lz))
    
    fill_value = True # Value to fill domain with


    cube_small = (x > int(lx/4)) & (x < int(lx * 3/4) + 1) & (y > int(ly/4)) & (y < int(ly * 3/4) + 1) & (z > int(lz/4)) & (z < int(lz * 3/4) + 1)
    cube = (x > int(lx/8)) & (x < int(lx * 6/8)) & (y > int(ly/8)) & (y < int(ly * 6/8)) & (z > int(lz/8)) & (z < int(lz * 6/8))
    sphere = np.square(x - int(lx/2) + 1) + np.square(y - int(ly/2) + 1) + np.square(z - int(lz/2) + 1) < 3/4 * np.sqrt(np.sum(np.square([lx,ly,lz])))
    torus = np.square(int(lx/2) - 1 - np.sqrt(np.square(x - int(lx/2)) + 
            np.square(y - int(ly/2) + 1)))+ np.square(z - int(lz/2) + 1) <= 0.1 * np.sqrt(np.sum(np.square([lx,ly,lz])))

    # combine the objects into a single boolean array
    if(boundary_type == "torus"):
        domain = torus
    elif(boundary_type == "sphere"):
        domain = sphere
    elif(boundary_type == "cube"):
        domain = cube
    elif(boundary_type == "cube_small"):
        domain = cube_small
    elif(boundary_type == "random"):
        domain = np.zeros(lx * ly * lz)

        # Calculate the number of elements to fill
        num_elements = int(domain.shape[0] * filling_fraction)

        # Generate random indices to fill
        idx = np.random.choice(domain.shape[0], num_elements,replace=False) 
        domain[idx] = fill_value
        print(domain.shape)
        domain = domain.reshape((lx, ly, lz)).astype(bool)
        
    else:
        domain = np.zeros((lx, ly, lz)).astype(bool)



    colors = np.empty(domain.shape, dtype=object)
    colors[domain] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(domain, facecolors=colors, edgecolor='k')

    plt.show()
    if(boundary_type != ""):
        ax.get_figure().savefig(f"../geometries/{boundary_type}_boundary_{lx}_{ly}_{lz}.png")


def make_linear_system():

    # initialize 'A' matrix: cells within bulk that are solid reduce the 'effective' 
    # domain on which the Pressure Poisson Equation is solved
    A = np.zeros((lx*ly*lz,lx*ly*lz))


    # iterate over all points in 3d grid
    for k in range(0, lz):
        for j in range(0, ly):
            for i in range(0, lx): 

                row = grid_to_row_col(i, j, k)
                col = row
                
                if(is_boundary(i, j, k)):
                    A[row, col] = 0 
                    continue
                A[row, col] = 6
                
                # impose neumann boundary conditions
                if(not is_boundary(i + 1, j, k)):
                    A[row, grid_to_row_col(i + 1, j, k)] = -1
                else:
                    A[row,col] -= 1

                if(not is_boundary(i - 1, j, k)):
                    A[row, grid_to_row_col(i - 1, j, k)] = -1
                else:
                    A[row,col] -= 1

                if(not is_boundary(i, j + 1, k)):
                    A[row, grid_to_row_col(i, j + 1, k)] = -1
                else:
                    A[row,col] -= 1

                if(not is_boundary(i, j - 1, k)):
                    A[row, grid_to_row_col(i, j - 1, k)] = -1
                else:
                    A[row,col] -= 1
                
                if(not is_boundary(i, j, k + 1)):
                    A[row, grid_to_row_col(i, j, k + 1)] = -1
                else:
                    A[row,col] -= 1
                
                if(not is_boundary(i, j, k - 1)):
                    A[row, grid_to_row_col(i, j, k - 1)] = -1
                else:
                    A[row,col] -= 1

                if(A[row, col] == 0):
                    A[row, col] = 1

            
    print("Poisson Equation Matrix Complete") 
    pA = np.eye(lx * ly * lz, lx * ly * lz)
    pA[:A.shape[0], :A.shape[1]] = A
    pA = scipy.sparse.coo_matrix(pA)
    
    # save compressed version of matrix
    scipy.sparse.save_npz(f"../data/{matrix_fn}.npz", pA)
    
    generate_dataset(A)

def generate_dataset(A:np.ndarray):
    print("Generating dataset...")
    num_vectors = 20000 # dataset vectors
    theta = 25

    ####### Obtain all eigenvectors with non-negative eigenvalues #######
    Ae, Av = map(np.real, scipy.linalg.eigh(A))



    # reorder eigen values 
    ind = np.argsort(Ae)
    Ae, Av = Ae[ind], Av.T[ind, :] # the rows are now eigenvectors
    ev_threshold = 1e-14
    condition = Ae >= ev_threshold
    first_index = np.argmax(condition) if np.any(condition) else -1

    Ae, Av = Ae[first_index:], Av[first_index:, :]

    ####### Generate dataset from linear combinations of eigenvectors #######
    m = Ae.shape[0] # total number of coefficients

    coeff_set = np.zeros((num_vectors, m))

    cutoff = int(m * 0.5 + theta)
    loc = 0
    scale = 1

    for i in range(num_vectors):
    

        # coeffecients pulled from normal distribution
        x_large = 9 * scipy.stats.norm.rvs(loc, scale, cutoff) 
        x = scipy.stats.norm.rvs(loc, scale, int(m - cutoff)) 
         
        coeff = np.concatenate((x_large, x))
        coeff_set[i, :] = coeff    
        if(i % 100 == 0):
            print(f"{i} vectors generated")

    dataset = coeff_set @ Av  # generate vectors from coefficients, shape [num_samples, dim(x)]

    # normalize vectors
    dataset = np.diag(1 / scipy.linalg.norm(dataset, ord = 2, axis = 1)) @ dataset

    if(training):
        dataset = dataset.astype(np.float64)
        np.save(f"../data/{matrix_fn}_training.npy", dataset)
    else:
        dataset = dataset.astype(np.float64)
        np.save(f"../data/{matrix_fn}_data.npy", dataset)

    print("Done")

def main():
    global matrix_fn, training
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test", dest="train", action="store_false", default = True)
    parser.add_argument("-f","--fillfrac", dest="filling_fraction", type = float, default = 0)
    parser.add_argument("-b", "--boundary", dest = "boundary_type", type = str, default = "")

    FLAGS, unparsed = parser.parse_known_args()
    training = FLAGS.train
    if(FLAGS.boundary_type == "random"):
        FLAGS.boundary_type = FLAGS.boundary_type + "_ff_" + str(FLAGS.filling_fraction)
    matrix_fn = f"A_matrix_{lx}_{ly}_{lz}_{FLAGS.boundary_type}"
    init_domain(boundary_type = FLAGS.boundary_type, filling_fraction = FLAGS.filling_fraction)
    make_linear_system()


if __name__ == "__main__":
    lx, ly, lz = 8,8,8
    rho = 1
    dt = 1
    dx = 1 
    main()