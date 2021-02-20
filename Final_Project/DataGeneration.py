#Reference https://github.com/jollybao/LCS
import pylab as plt
import numpy as np
import matplotlib.animation as animation
from scipy.integrate import odeint
from itertools import product
from tqdm import tqdm

def datageneration():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg'
    # mywriter = animation.FFMpegWriter()

    # constants
    p = np.pi
    A = 0.25
    epsilon = 0.25 #Alpha
    w = 2*p
    delta = 0.0001
    dt = 0.1
    partition = 100


    # wave function that defines the characteristics of
    # double gyre
    def phi(x, y, t):
        temp = A * np.sin(p * f(x, t)) * np.sin(p * y)
        return temp


    def f(x, t):
        temp = epsilon * np.sin(w * t) * x**2 + (1 - 2 * epsilon * np.sin(w * t)) * x
        return temp


    # def velocity(x, y, t):
    #     vx = (phi(x, y + delta, t) - phi(x, y - delta, t)) / (2 * delta)
    #     vy = (phi(x - delta, y, t) - phi(x + delta, y, t)) / (2 * delta)
    #     return -1 * vx, -1 * vy


    def velocity(x,y,t):
        vx = -p * A * np.sin(p * f(x,t)) * np.cos(p * y)
        vy = p * A * np.cos(p * f(x,t)) * np.sin(p * y) * derivative(x,t,delta=delta)
        return vx,vy

    # function that computes velocity of particle at each point
    # def update(r, t):
    #     x = r[0]
    #     y = r[1]
    #     vx = (phi(x, y + delta, t) - phi(x, y - delta, t)) / (2 * delta)
    #     vy = (phi(x - delta, y, t) - phi(x + delta, y, t)) / (2 * delta)
    #     return np.array([-1 * vx, -1 * vy], float)

    def update(x,y,t):
        vx, vy = velocity(x,y,t)
        return phi(x+vx,y+vy,t)

    # https://www.math.ubc.ca/~pwalls/math-python/differentiation/differentiation/
    def derivative(x, t, method='central', delta=0.01):
        '''Compute the difference formula for f'(a) with step size h.

        Parameters
        ----------
        f : function
            Vectorized function of one variable
        a : number
            Compute derivative at x = a
        method : string
            Difference formula: 'forward', 'backward' or 'central'
        h : number
            Step size in difference formula

        Returns
        -------
        float
            Difference formula:
                central: f(a+h) - f(a-h))/2h
                forward: f(a+h) - f(a))/h
                backward: f(a) - f(a-h))/h
        '''
        if method == 'central':
            return (f(x + delta, t) - f(x - delta, t)) / (2 * delta)
        # elif method == 'forward':
        #     return (f(a + h) - f(a))/h
        # elif method == 'backward':
        #     return (f(a) - f(a - h))/h
        # else:
        #     raise ValueError("Method must be 'central', 'forward' or 'backward'.")


    # make a 2D mesh grid of size 40*20
    X, Y = plt.meshgrid(np.arange(0, 2, 1 / partition), np.arange(0, 1, 1 / partition))
    Vx, Vy = velocity(X, Y, 0.1)

    plt.figure()
    init_state = phi(X,Y,0)
    plt.imshow(init_state)

    plt.figure()
    nd_step = update(X,Y,0.1)
    plt.imshow(nd_step)

    # vector arrows
    Q = ax.quiver(X, Y, Vx, Vy, scale=10)

    # initialize array of particles
    C = np.empty([N], plt.Circle)
    for i in range(0, N):
        C[i] = plt.Circle((-1, -1), radius=0.03, fc=col[i])

    R = np.empty([N, 2], float)
    for i in range(0, N):
        print("Enter x and y coordinates of the circle ", i + 1)
        R[i][0] = float(input())
        R[i][1] = float(input())
        C[i].center = (R[i][0], R[i][1])
        ax.add_patch(C[i])


    # animation for particle moving along the vector field
    def animate(num, Q, X, Y, C, R, N):
        t = num / 1
        dt = 1 / 10
        Vx, Vy = velocity(X, Y, t)
        Q.set_UVC(Vx, Vy)

        # update particles' positions
        for i in range(0, N):
            for j in range(0, 10):
                r = R[i][:]
                k1 = dt * update(r, t)
                k2 = dt * update(r + 0.5 * k1, t + 0.5 * dt)
                k3 = dt * update(r + 0.5 * k2, t + 0.5 * dt)
                k4 = dt * update(r + k3, t + dt)
                R[i][:] += (k1 + 2 * k2 + 2 * k3 + k4) / 6

            C[i].center = (R[i][0], R[i][1])
        return Q, C


    ani = animation.FuncAnimation(fig, animate,
                                  fargs=(Q, X, Y, C, R, N),
                                  interval=100, blit=False)

    # ani.save('VF_demo.mp4',writer = mywriter)

    plt.show()



def nd_ap_gendata(file_name,load = False):
    # constants
    p = np.pi
    A = 0.25
    epsilon = 0.25  # Alpha
    w = 2 * p
    delta = 0.0001
    dt = 0.1
    partition = 20
    T = 201

    # time points
    t = np.linspace(0, 20, T)

    # X, Y = plt.meshgrid(np.arange(0, 2, 1 / partition), np.arange(0, 1, 1 / partition))
    X = np.arange(0, 2, 1 / partition)
    Y = np.arange(0, 1, 1 / partition)
    Comb_cor = product(X, Y) #Creating all combination of X and Y

    def model(z,t):
        x,y = z
        dxdt = -p * A * np.sin(p * f(x, t)) * np.cos(p * y)
        dydt = p * A * np.cos(p * f(x,t)) * np.sin(p * y) * (np.sin(w * t)*(2*epsilon*x-2) + 1) #df/dx
        return [dxdt, dydt]

    def f(x, t):
        temp = epsilon * np.sin(w * t) * x**2 + (1 - 2 * epsilon * np.sin(w * t)) * x
        return temp

    # y = 0.3
    # z0 = (0.2,0.3)
    if load:
        Diff_Space_time = np.load('{}.npy'.format(file_name))
    else:
        Diff_Space_time = np.zeros([T,X.shape[0] * Y.shape[0],2])
        for idx, i in tqdm(enumerate(Comb_cor)):
            Diff_Space_time[:,idx,:] = odeint(model,i,t)

        np.save(file_name, Diff_Space_time.astype('float32'))
        print('Data Generation Done, Array saved')


    return Diff_Space_time

if __name__ == '__main__':
    nd_ap_gendata()

    print('Finish')