import numpy as np
import jax_cosmo as jc
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
from jaxpm.painting import cic_paint
jconfig.update("jax_enable_x64", True)

from flbench.model import FieldLevelModel, default_config
from flbench.bricks import get_cosmology
from flbench.nbody import a2g, a2f
from flbench.plot import plot_mesh, plot_pow
from flbench.utils import Path

print(jdevices())
save_dir = Path(__file__).resolve().parent / "save_dir"
save_dir.mkdir(parents=True, exist_ok=True)

seed, frac = 42, 0.1  # simulation seed, fraction of particles saved
model = FieldLevelModel(**default_config | {'box_shape': 3 * (512,), 'mesh_shape': 3 * (256,), 'a_obs': 1., 'evolution': 'nbody'})
print(model)

# Run the truth simulation, keeping intermediate (deterministic) sites
params = {'Omega_m': 0.3, 'sigma8': 0.8, 'b1': 1., 'b2': 0., 'bs2': 0., 'bn2': 0.}
truth = model.predict(rng=seed, samples=params, hide_base=False, hide_det=False, hide_samp=False, frombase=True)

mesh_shape, cell = np.array(model.mesh_shape), np.array(model.box_shape) / np.array(model.mesh_shape)
pos = truth['nbody_pos'][-1] % mesh_shape  # final real-space positions, mesh units
# Peculiar velocities in km/s: nbody_vel is dq/dg in mesh units, so v = a H(a) f(a) g(a) vel * cell_size
a, cosmo = model.a_obs, get_cosmology(**params)
vel = truth['nbody_vel'][-1] * cell * a * jc.background.H(cosmo, jnp.atleast_1d(a)) * a2f(cosmo, a) * a2g(cosmo, a)

# Linear initial field (normalised to a=1), z=0 matter field (1+delta, all particles), and a random particle subset (Mpc/h, km/s)
init_field = jnp.fft.irfftn(truth['init_mesh'], s=model.mesh_shape)
final_field = cic_paint(jnp.zeros(model.mesh_shape), pos)
sub = jr.choice(jr.key(seed), len(pos), (int(frac * len(pos)),), replace=False)

model.save(save_dir / "model.yaml")
np.savez(save_dir / "truth.npz", seed=seed, frac=frac, **{k: np.asarray(truth[k]) for k in params},
         init_field=np.float32(init_field), final_field=np.float32(final_field), pos=np.float32(pos[sub] * cell),
         vel=np.float32(vel[sub]))

# Plot the linear field at z=8 and the z=0 field on a shared colour scale, their spectra, and subset velocities
z_init = 8.
init_z = init_field * a2g(cosmo, 1 / (1 + z_init)) / a2g(cosmo, 1.)
shot = np.prod(model.box_shape) / len(pos)  # Poisson shot noise of all painted particles; linear fields have none
fields = {f'linear, z={z_init:g}': (init_z, 0, 0.), 'N-body, z=0': (final_field - 1, 2, shot)}  # (delta, CIC deconv order, shot noise)
vlim = tuple(np.quantile(final_field - 1, [0.005, 0.995]))
fig, axes = plt.subplots(1, 4, figsize=(17, 4), layout='constrained')
for ax, (name, (delta, comp, _)) in zip(axes, fields.items()):
    plt.sca(ax)
    plot_mesh(delta, model.box_shape, 0.1, axis=-2, vlim=vlim)
    plt.title(name)
plt.sca(axes[2])
ks, pk = model.spectrum(init_field)
plot_pow(ks, pk, 'k--', label='linear, z=0')
plot_pow(ks, jc.power.nonlinear_matter_power(cosmo, jnp.asarray(ks), a=1.), 'k:', label='halofit, z=0')
ks, pk = model.spectrum(final_field - 1, comp=2)
plot_pow(ks, pk - shot, 'C1', label='N-body, z=0')
kmax = 0.3
axes[2].set_xlim(0.0, kmax)
kp = np.concatenate([np.asarray(l.get_ydata())[np.asarray(l.get_xdata()) <= kmax] for l in axes[2].get_lines()])
axes[2].set_ylim(0.9 * kp.min(), 1.1 * kp.max())

axes[2].legend()
[axes[3].hist(np.asarray(vel[sub, j]), bins=40, histtype='step', label=f'$v_{c}$') for j, c in enumerate('xyz')]
axes[3].set(xlabel='peculiar velocity [km/s]', ylabel='N', title=f'rms = {np.std(vel[sub]):.0f} km/s')
axes[3].legend()
plt.savefig(save_dir / "truth_fields.png", dpi=150)
plt.show()
