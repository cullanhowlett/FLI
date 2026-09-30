import numpy as np
import jax_cosmo as jc
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
from jaxpm.painting import cic_paint
jconfig.update("jax_enable_x64", True)

from flbench.model import FieldLevelModel, default_config
from flbench.bricks import get_cosmology
from flbench.nbody import a2g, a2f
from flbench.plot import plot_mesh, plot_pow, mean_slice
from flbench.utils import Path


def run_truth(model, params, seed, frac):
    """N-body truth: linear initial field (a=1), z=0 matter field (1+delta), and a random particle subset (Mpc/h, km/s)."""
    truth = model.predict(rng=seed, samples=params, hide_base=False, hide_det=False, hide_samp=False, frombase=True)
    mesh_shape, cell = np.array(model.mesh_shape), np.array(model.box_shape) / np.array(model.mesh_shape)
    pos = truth['nbody_pos'][-1] % mesh_shape  # final real-space positions, mesh units
    # Peculiar velocities: nbody_vel is dq/dg in mesh units, so v = a H(a) f(a) g(a) vel * cell_size
    a, cosmo = model.a_obs, get_cosmology(**params)
    vel = truth['nbody_vel'][-1] * cell * a * jc.background.H(cosmo, jnp.atleast_1d(a)) * a2f(cosmo, a) * a2g(cosmo, a)
    sub = jr.choice(jr.key(seed), len(pos), (int(frac * len(pos)),), replace=False)
    return dict(config=str(model), seed=seed, frac=frac, **{k: np.asarray(truth[k]) for k in params},
                init_field=np.float32(jnp.fft.irfftn(truth['init_mesh'], s=model.mesh_shape)),
                final_field=np.float32(cic_paint(jnp.zeros(model.mesh_shape), pos)),
                pos=np.float32(pos[sub] * cell), vel=np.float32(vel[sub]))


def load_or_run(path, model, params, seed, frac, remake=False):
    """Reuse a saved truth if it matches the current model config, parameters, seed and fraction; otherwise rerun and save."""
    if path.exists() and not remake:
        t = dict(np.load(path))
        if str(t.get('config')) == str(model) and (t['seed'], t['frac']) == (seed, frac) and all(t[k] == v for k, v in params.items()):
            print(f"Loaded existing truth from {path}")
            return t
        print("Saved truth does not match current settings, rerunning")
    t = run_truth(model, params, seed, frac)
    model.save(path.parent / "model.yaml")
    np.savez(path, **t)
    return t


def slab_velocity(pos, vel, mesh_shape, cell, sli, axis, n_arrow=16):
    """Mass-weighted in-plane velocity in a slab, block-averaged onto a coarse arrow grid. pos in mesh units."""
    plane = [i for i in range(3) if i != axis]
    nb = max(1, mesh_shape[plane[0]] // n_arrow)
    block = lambda m: np.asarray(mean_slice(m, sli, axis)).reshape(-1, nb, mesh_shape[plane[1]] // nb, nb).sum((1, 3))
    mass = block(cic_paint(jnp.zeros(mesh_shape), pos))
    u, v = [block(cic_paint(jnp.zeros(mesh_shape), pos, vel[:, i])) / mass for i in plane]
    centres = [(np.arange(len(x)) + 0.5) * nb * c for x, c in zip((u, u.T), cell[plane])]
    return *np.meshgrid(*centres, indexing='ij'), u, v


def plot_truth(model, t, params, path, slab=20., axis=1, z_init=8., kmax=0.3, n_arrow=16):
    """Slices of the linear (z_init) and z=0 fields with z=0 velocity arrows, P(k) vs linear/halofit, velocity histograms."""
    mesh_shape, cell = np.array(model.mesh_shape), np.array(model.box_shape) / np.array(model.mesh_shape)
    cosmo = get_cosmology(**params)
    init_field, final_field, vel = t['init_field'], t['final_field'], t['vel']
    init_z = init_field * a2g(cosmo, 1 / (1 + z_init)) / a2g(cosmo, 1.)
    shot = np.prod(model.box_shape) / np.prod(mesh_shape)  # Poisson shot noise of all painted particles; linear fields have none

    sli = max(1, round(slab / cell[axis]))  # slice from 0 along the projected mesh axis, in cells
    labels = [f'${c}$ [Mpc/$h$]' for i, c in enumerate('xyz') if i != axis]
    vlim = tuple(np.quantile(final_field - 1, [0.005, 0.995]))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), layout='constrained')
    for ax, (name, delta) in zip(axes, {f'linear, z={z_init:g}': init_z, 'N-body, z=0': final_field - 1}.items()):
        plt.sca(ax)
        plot_mesh(delta, model.box_shape, sli, axis=axis, vlim=vlim)
        ax.set(xlabel=labels[0], ylabel=labels[1], title=f"{name}, ${'xyz'[axis]}<{sli * cell[axis]:g}$ Mpc/$h$")

    # Velocity arrows from the saved particle subset
    X, Y, U, V = slab_velocity(t['pos'] / cell, vel, model.mesh_shape, cell, sli, axis, n_arrow)
    vscale = np.quantile(np.hypot(U, V), 0.95)
    q = axes[1].quiver(X, Y, U, V, color='w', angles='xy', scale_units='xy', scale=vscale / (0.9 * (X[1, 0] - X[0, 0])), width=0.004)
    axes[1].quiverkey(q, 0.78, -0.12, round(vscale, -2), f'{round(vscale, -2):.0f} km/s', labelpos='E', coordinates='axes', color='k')

    plt.sca(axes[2])
    ks, pk = model.spectrum(init_field)
    plot_pow(ks, pk, 'k--', label='linear, z=0')
    plot_pow(ks, jc.power.nonlinear_matter_power(cosmo, jnp.asarray(ks), a=1.), 'k:', label='halofit, z=0')
    ks, pk = model.spectrum(final_field - 1, comp=2)
    plot_pow(ks, pk - shot, 'C1', label='N-body, z=0')
    axes[2].set_xlim(0.0, kmax)
    kp = np.concatenate([np.asarray(l.get_ydata())[np.asarray(l.get_xdata()) <= kmax] for l in axes[2].get_lines()])
    axes[2].set_ylim(0.9 * kp.min(), 1.1 * kp.max())
    axes[2].legend()

    [axes[3].hist(vel[:, j], bins=40, histtype='step', label=f'$v_{c}$') for j, c in enumerate('xyz')]
    axes[3].set(xlabel='peculiar velocity [km/s]', ylabel='N', title=f'rms = {np.std(vel):.0f} km/s')
    axes[3].legend()
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    print(jdevices())
    save_dir = Path(__file__).resolve().parent / "save_dir"
    save_dir.mkdir(parents=True, exist_ok=True)

    seed, frac = 42, 0.1  # simulation seed, fraction of particles saved
    params = {'Omega_m': 0.3, 'sigma8': 0.8, 'b1': 1., 'b2': 0., 'bs2': 0., 'bn2': 0.}
    model = FieldLevelModel(**default_config | {'box_shape': 3 * (512,), 'mesh_shape': 3 * (256,), 'a_obs': 1., 'evolution': 'nbody'})
    print(model)

    truth = load_or_run(save_dir / "truth.npz", model, params, seed, frac)
    plot_truth(model, truth, params, save_dir / "truth_fields.png", slab=50., axis=1)
