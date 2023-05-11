import gc
import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch as torch
import sys as sys
import scipy.io
import os
torch.cuda.empty_cache()

# gc.collect()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1024"
# torch.cuda.memory_summary(device=None, abbreviated=False)


class CustomPointCloud(dde.geometry.PointCloud):
    def __init__(self, points, boundary_points, boundary_normals):
        super(CustomPointCloud, self).__init__(
            points, boundary_points, boundary_normals)

    def compute_k_nearest_neighbors(self, x, k=3):
        # Compute the k-nearest neighbors for each boundary point
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='auto').fit(self.boundary_points)
        distances, indices = nbrs.kneighbors(x)
        return indices

    def boundary_normal(self, x):
        k = 3  # number of neighbors
        indices = self.compute_k_nearest_neighbors(x, k)

        normals = np.zeros_like(x)
        for i, idx in enumerate(indices):

            normal = self.boundary_normals[idx[0]]

            normals[i] = normal

        return normals


class PINN():
    def __init__(self):
        self.a = 0.15
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1
        self.D = 1

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

    def get_data(self):
        from utils import get_data, get_boundary
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        vertices, triangles, vm = get_data()
        # normalize features
        x_mean = 0  # np.mean(vertices[:, 0])
        x_std = 1  # np.std(vertices[:, 0])
        y_mean = 0  # np.mean(vertices[:, 1])
        y_std = 1  # np.std(vertices[:, 1])
        vertices[:, 0] = (vertices[:, 0] - x_mean) / x_std
        vertices[:, 1] = (vertices[:, 1] - y_mean) / y_std

        self.vertices = vertices
        self.triangles = triangles

        print("self.triangles shape:", self.triangles.shape)
        self.vm = vm[:20, :]
        x = vertices[:, 0]
        y = vertices[:, 1]
        t = np.linspace(0, 600, 121)[:20]
        t_mean = 0  # np.mean(t)
        t_std = 1  # np.std(t)
        t = (t - t_mean) / t_std

        X, T = np.meshgrid(x, t)
        Y, T = np.meshgrid(y, t)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = self.vm.reshape(-1, 1)
        vertices_boundary, triangles_boundary = get_boundary()

        self.vertices_boundary = vertices_boundary
        self.triangles_boundary = triangles_boundary

        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)
        print("self.vertices shape:", self.vertices.shape)
        print("self.vertices_boundary shape:", self.vertices_boundary.shape)
        """ 
        X= scaler.transform(X)
        Y = scaler.transform(Y)
        T = scaler.transform(T)
        X_boundary = scaler.transform(X_boundary)
        Y_boundary = scaler.transform(Y_boundary)
        T_boundary = scaler.transform(T_boundary)
        V = scaler.transform(V)
        """
        return np.hstack((X, Y, T)), np.hstack((X_boundary, Y_boundary, T_boundary)), V

    def BC(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros(
            (len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        return bc

    def IC(self, observe_train, v_train):

        T_ic = observe_train[:, -1].reshape(-1, 1)

        idx_init = np.where(np.isclose(T_ic, 5, atol=5))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]

        return dde.PointSetBC(observe_init, v_init, component=0)

    def geotime(self):

        self.boundary_normals = np.load("normals.npy")
        # remove points from vertices that are in the boundary
        vertices_expanded = self.vertices[:, np.newaxis]
        boundary_vertices_expanded = self.vertices_boundary[np.newaxis, :]

        is_vertex_on_boundary = np.any(
            np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        unique_vertices = self.vertices[~is_vertex_on_boundary]
        print("self.boundary_normals shape:", self.boundary_normals.shape)
        geom = CustomPointCloud(
            self.vertices, self.vertices_boundary, self.boundary_normals)
        timedomain = dde.geometry.TimeDomain(0, 600)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime


def main():
    pinn = PINN()

    X, X_boundary, v = pinn.get_data()
    X_train, X_test, v_train, v_test = train_test_split(
        X, v, test_size=0.1)

    data_list = [X, X_train, v_train, v]

    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, observe_v]
    data = dde.data.TimePDE(geomtime,
                            pinn.pde2d,
                            input_data,
                            num_domain=40000,
                            num_boundary=100)
    """

    from sklearn.preprocessing import StandardScaler

    # Get the input data
    X, X_boundary, v = pinn.get_data()
    # Normalize the input data (This had a huge impact on performance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_boundary_scaled = scaler.transform(X_boundary)
    v_scaled = scaler.fit_transform(v)

    # Train test split
    X_train, X_test, v_train, v_test = train_test_split(
        X_scaled, v_scaled, test_size=0.9)

    data_list = [X_scaled, X_train, v_train, v_scaled]

    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, observe_v]

    data = dde.data.TimePDE(geomtime,
                            pinn.pde2d,
                            input_data,
                            anchors=X_train)
    """
    n_neurons = 121
    n_layers = 6
    n_epochs = 6000
    activations = ["sinc", "sin", "sin",
                   "sin", "sin", "sin", "sin"]

    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], activations, "He normal")

    # silu: not too bad, but not great

# Set model_save_path to save the model
    import os
    save_path = os.getcwd()+f"/models/heart_model_{n_neurons}x{n_layers}"

    if sys.argv[1] == "train":
        init_weights = [0, 0, 0, 1]
        checker = dde.callbacks.ModelCheckpoint(
            save_path, save_better_only=True, period=2000)

        model = dde.Model(data, net)

        """ 
        # Stabalize initialization process by capping the losses
        model.compile("adam", lr=0.05)
        losshistory, _ = model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        num_init = 1
        while initial_loss > 10. or np.isnan(initial_loss):
            num_init += 1
            model = dde.Model(data, net)
            model.compile("adam", lr=0.05)
            losshistory, _ = model.train(epochs=1)
            initial_loss = max(losshistory.loss_train[0])
        """
        # Phase1
        model.compile("adam", lr=0.0005, loss_weights=init_weights)
        losshistory, train_state = model.train(
            iterations=10000, model_save_path=save_path)

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        """ 

        model.compile("adam", lr=0.005, loss_weights=init_weights)
        losshistory, train_state = model.train(
            iterations=3000, model_save_path=save_path)

        model.compile("adam", lr=0.00001, loss_weights=init_weights)
        losshistory, train_state = model.train(
            iterations=3000, model_save_path=save_path)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        """
        # Phase 2

        model.compile("adam", lr=0.0005)
        losshistory, train_state = model.train(
            iterations=50000, model_save_path=save_path)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        # Phase 3
        model.compile("L-BFGS-B")

        losshistory, train_state = model.train(
            epochs=n_epochs, model_save_path=save_path, callbacks=[checker])

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # model.compile("L-BFGS-B")

        # losshistory, train_state = model.train(model_save_path=save_path,callbacks=[checker])

        # Save and plot the loss history
        # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # Save the model
        model.save(save_path)

    if sys.argv[1] == "predict":
        torch.device("cpu")
        dde.config.default_device = "cpu"

        model = dde.Model(data, net)
        model.compile("adam", lr=0.0005)
        # model.compile("adam", lr=0.0001)
        model.restore(save_path+"-"+input("Enter model checkpoint: ")+".pt")
        from plot import generate_2D_animation, plot_2D
        plot_2D(pinn, model)
        generate_2D_animation(pinn, model)
        # plot_2D_grid(data_list, pinn, model, "planar_wave")


if __name__ == "__main__":

    torch.cuda.set_per_process_memory_fraction(0.9)
    dde.config.set_default_float("float16")

    main()
