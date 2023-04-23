import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
import torch as torch
import sys as sys
import scipy.io


class PINN():
    def __init__(self):
        self.a = 0.01
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1  # cell length [mm]
        self.D = 0.1  # diffusion coefficient [mm^2/AU]

    def pde2d(self, x, y):
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        # Coupled PDE+ODE Equations
        eq_a = dv_dt - self.D*(dv_dxx + dv_dyy) + \
            self.k*V*(V-self.a)*(V-1) + W*V
        eq_b = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def get_data(self, filename):
        mat = scipy.io.loadmat(filename)
        t, x, y, V, W = mat['t'], mat['x'], mat['y'], mat['Vsav'], mat['Wsav']
        self.min_x, self.max_x = np.min(x), np.max(x)
        self.min_y, self.max_y = np.min(y), np.max(y)
        self.min_t, self.max_t = np.min(t), np.max(t)
        X, T, Y = np.meshgrid(x, t, y)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = V.reshape(-1, 1)
        W = W.reshape(-1, 1)
        return np.hstack((X, Y, T)), V, W

    def BC(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros(
            (len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        return bc

    def IC(self, observe_train, v_train):

        T_ic = observe_train[:, -1].reshape(-1, 1)
        idx_init = np.where(np.isclose(T_ic, 1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init, v_init, component=0)

    def geotime(self):
        geom = dde.geometry.Rectangle([self.min_x, self.min_y], [
                                      self.max_x, self.max_y])
        timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime


def main():
    pinn = PINN()
    X, v, w = pinn.get_data('./data/data_2d_vertical_planar.mat')
    X_train, X_test, v_train, v_test, w_train, w_test = train_test_split(
        X, v, w, test_size=0.9)

    data_list = [X, X_train, v_train, v]

    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, ic, observe_v]
    data = dde.data.TimePDE(geomtime,
                            pinn.pde2d,
                            input_data,
                            num_domain=40000,
                            num_boundary=4000,
                            num_test=1000,
                            anchors=X_train)

    n_neurons = 60
    n_layers = 5
    n_epochs = 10000

    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], "tanh", "Glorot normal")

# Set model_save_path to save the model
    import os
    save_path = os.getcwd()+f"/models/AP2D_model_{n_neurons}x{n_layers}_"

    if sys.argv[1] == "train":

        checker = dde.callbacks.ModelCheckpoint(
            save_path, save_better_only=True, period=2000)
        print("net generated")

        model = dde.Model(data, net)
        model.compile("adam", lr=0.0001)
        print("model compiled")

        """ 
        # Stabalize initialization process by capping the losses
        losshistory, _ = model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        num_init = 1
        while initial_loss > 0.1 or np.isnan(initial_loss):
            num_init += 1
            model = dde.Model(data, net)
            model.compile("adam", lr=0.005)
            losshistory, _ = model.train(epochs=1)
            initial_loss = max(losshistory.loss_train[0])
        """
        losshistory, train_state = model.train(
            epochs=n_epochs, model_save_path=save_path, callbacks=[checker])
        # model.compile("L-BFGS-B")

        # losshistory, train_state = model.train(model_save_path=save_path,callbacks=[checker])

        # Save and plot the loss history
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # Save the model
        model.save(save_path)

    if sys.argv[1] == "predict":

        model = dde.Model(data, net)
        model.compile("adam", lr=0.0001)
        model.restore(save_path+"-"+input("Enter model checkpoint: ")+".pt")
        from plot2d import generate_2D_animation
        generate_2D_animation(pinn, model, "planar_wave")


if __name__ == "__main__":
    # dde.config.set_default_float("float16")

    main()
