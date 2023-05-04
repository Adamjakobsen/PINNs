import h5py

import numpy as np


def get_data():
    scar_filename = "../meshes/scarmesh_1.h5"

    v_filename = "../meshes/vm.h5"

    with h5py.File(scar_filename, "r") as scar_file:
        coordinates = scar_file["coordinates"][:]
        triangles = scar_file["topology"][:]

    t = np.arange(0, 605, 5)

    vm = np.zeros((len(t), 71257))
    i = 0

    with h5py.File(v_filename, "r") as v_file:
        group = v_file["vm"]
        datasets = list(group.keys())
        float_numbers = [float(number) for number in datasets]
        sorted_float_numbers = sorted(float_numbers)
        # sort datasets starts at t=0 end t=600 with length datasets
        datasets = [str(number) for number in sorted_float_numbers]
        for dataset in datasets:
            data = group[dataset]
            print(dataset)

            vm[i, :] = data[:]
            i += 1

    return coordinates, triangles, vm


def get_boundary():
    boundary_filename = "../meshes//boundary_mesh.h5"

    with h5py.File(boundary_filename, "r") as boundary_file:
        group1 = boundary_file["Mesh"]
        group2 = group1["mesh"]
        coordinates = group2["geometry"][:]
        triangles = group2["topology"][:]

    return coordinates, triangles
