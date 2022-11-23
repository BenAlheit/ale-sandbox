print("start")
import pyvista as pv

print("import pv")
import numpy as np

print("import np")
import matplotlib.pyplot as plt

print("import plt")

stress_comp = 4
E_comp = 4

E_name = "E11__E12__E13__E21__E22__E23__E31__E32__E33"
S_name = "S11__S12__S13__S21__S22__S23__S31__S32__S33"


def plot(file_name, label):
    x = []
    y = []
    # file_name = f'viscoplasticity-{name}/viscoplasticity-{name}.pvd'
    print(file_name)
    reader = pv.get_reader(file_name)
    for i_time in range(len(reader.time_values) - 1):
        print(f"{i_time + 1} / {len(reader.time_values) - 1}")
        reader.set_active_time_point(i_time)
        mesh = reader.read()[0]
        stress = np.mean(mesh[S_name], axis=0)
        E = np.mean(mesh[E_name], axis=0)
        x.append(E[E_comp])
        y.append(stress[stress_comp])
    plt.plot(x, y, marker='s', mfc='none', label=label)


def plot_f(file_name, label):
    x = []
    y = []
    # file_name = f'viscoplasticity-{name}/viscoplasticity-{name}.pvd'
    print(file_name)
    reader = pv.get_reader(file_name)
    for i_time in range(len(reader.time_values) - 1):
        print(f"{i_time + 1} / {len(reader.time_values) - 1}")
        reader.set_active_time_point(i_time)
        mesh = reader.read()[0]
        stress = np.mean(mesh["f"], axis=0)
        E = np.mean(mesh[E_name], axis=0)
        x.append(E[E_comp])
        y.append(stress)
    plt.plot(x, y, marker='s', mfc='none', label=label)


# plot("./cmake-build-debug-wsl/ale-elastoplastic-ut.pvd", "Voce")
# plot_f("./cmake-build-debug-wsl/ale-elastoplastic-ut.pvd", "f")

plot("./cmake-build-debug-remote-host/ale-elastoplastic-ut.pvd", "Voce")
plot_f("./cmake-build-debug-remote-host/ale-elastoplastic-ut.pvd", "f")

plt.xlabel('$E_{22}$')
plt.ylabel(r'$\tau_{22}$')
plt.grid()
plt.legend()

plt.show()
