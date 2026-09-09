"""
viscoelastic_dns_verified_patch.py

Same as verified, but with:
- F0 = 10.0 (stronger body force)
- Initial perturbation (u,v non-zero)
- Additional forcing at ky=2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.fft import fft2, ifft2, fftfreq
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. Parameters (PATCHED)
# ----------------------------------------------------------------------

nx, ny = 64, 64
Lx, Ly = 2.0 * np.pi, 2.0 * np.pi

Re = 20.0
Wi = 10.0
beta = 0.5

F0 = 10.0          # 5x stronger

dt = 0.0005
nt = 8000
snapshot_every = 500
plot_every = 1000

Path("snapshots_patch").mkdir(exist_ok=True)
Path("composite_patch").mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 2. Grid
# ----------------------------------------------------------------------

x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = 2.0 * np.pi * fftfreq(nx, Lx/nx)
ky = 2.0 * np.pi * fftfreq(ny, Ly/ny)
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
K2 = Kx**2 + Ky**2
K2[0, 0] = 1.0

dealias = np.logical_and(np.abs(Kx) <= (2.0/3.0)*np.max(kx),
                         np.abs(Ky) <= (2.0/3.0)*np.max(ky))

# ----------------------------------------------------------------------
# 3. Initial conditions (WITH PERTURBATION)
# ----------------------------------------------------------------------

# Start with a small flow (so body force has something to work with)
u = 0.1 * np.sin(Y) * np.cos(X)
v = 0.1 * np.cos(Y) * np.sin(X)
tau_xx = np.zeros((nx, ny))
tau_xy = np.zeros((nx, ny))
tau_yy = np.zeros((nx, ny))

# ----------------------------------------------------------------------
# 4. Helper functions (same as before)
# ----------------------------------------------------------------------

def fft2c(f):
    return fft2(f)

def ifft2c(f_hat):
    return ifft2(f_hat).real

def dealias_field(f_hat):
    return f_hat * dealias

def energy(u_hat, v_hat):
    u = ifft2c(u_hat)
    v = ifft2c(v_hat)
    return 0.5 * np.mean(u**2 + v**2)

def polymer_energy(tau_xx, tau_yy, tau_xy):
    return 0.5 * np.mean(tau_xx + tau_yy)

# ----------------------------------------------------------------------
# 5. RHS and stress update (same as verified)
# ----------------------------------------------------------------------

def rhs_velocity(u_hat, v_hat, tau_xx_hat, tau_yy_hat, tau_xy_hat, Re, beta):
    u_hat = dealias_field(u_hat)
    v_hat = dealias_field(v_hat)
    tau_xx_hat = dealias_field(tau_xx_hat)
    tau_yy_hat = dealias_field(tau_yy_hat)
    tau_xy_hat = dealias_field(tau_xy_hat)

    u = ifft2c(u_hat)
    v = ifft2c(v_hat)
    tau_xx = ifft2c(tau_xx_hat)
    tau_yy = ifft2c(tau_yy_hat)
    tau_xy = ifft2c(tau_xy_hat)

    du_dx = ifft2c(1j * Kx * u_hat)
    du_dy = ifft2c(1j * Ky * u_hat)
    dv_dx = ifft2c(1j * Kx * v_hat)
    dv_dy = ifft2c(1j * Ky * v_hat)

    adv_x = u * du_dx + v * du_dy
    adv_y = u * dv_dx + v * dv_dy

    visc_x_hat = (beta / Re) * (-K2) * u_hat
    visc_y_hat = (beta / Re) * (-K2) * v_hat

    div_tau_x_hat = 1j * Kx * tau_xx_hat + 1j * Ky * tau_xy_hat
    div_tau_y_hat = 1j * Kx * tau_xy_hat + 1j * Ky * tau_yy_hat
    poly_x_hat = ((1.0 - beta) / Re) * div_tau_x_hat
    poly_y_hat = ((1.0 - beta) / Re) * div_tau_y_hat

    adv_x_hat = fft2c(adv_x)
    adv_y_hat = fft2c(adv_y)

    du_hat_dt = -adv_x_hat + visc_x_hat + poly_x_hat
    dv_hat_dt = -adv_y_hat + visc_y_hat + poly_y_hat

    div_u = 1j * Kx * du_hat_dt + 1j * Ky * dv_hat_dt
    du_hat_dt -= (Kx / K2) * div_u
    dv_hat_dt -= (Ky / K2) * div_u

    return du_hat_dt, dv_hat_dt

def update_stress_implicit(u_hat, v_hat, tau_xx_hat, tau_yy_hat, tau_xy_hat,
                           dt, Wi, beta):
    u_hat_d = dealias_field(u_hat)
    v_hat_d = dealias_field(v_hat)
    tau_xx_hat_d = dealias_field(tau_xx_hat)
    tau_yy_hat_d = dealias_field(tau_yy_hat)
    tau_xy_hat_d = dealias_field(tau_xy_hat)

    u = ifft2c(u_hat_d)
    v = ifft2c(v_hat_d)
    tau_xx = ifft2c(tau_xx_hat_d)
    tau_yy = ifft2c(tau_yy_hat_d)
    tau_xy = ifft2c(tau_xy_hat_d)

    du_dx = ifft2c(1j * Kx * u_hat_d)
    du_dy = ifft2c(1j * Ky * u_hat_d)
    dv_dx = ifft2c(1j * Kx * v_hat_d)
    dv_dy = ifft2c(1j * Ky * v_hat_d)

    dtau_xx_dx = ifft2c(1j * Kx * tau_xx_hat_d)
    dtau_xx_dy = ifft2c(1j * Ky * tau_xx_hat_d)
    dtau_xy_dx = ifft2c(1j * Kx * tau_xy_hat_d)
    dtau_xy_dy = ifft2c(1j * Ky * tau_xy_hat_d)
    dtau_yy_dx = ifft2c(1j * Kx * tau_yy_hat_d)
    dtau_yy_dy = ifft2c(1j * Ky * tau_yy_hat_d)

    adv_tau_xx = u * dtau_xx_dx + v * dtau_xx_dy
    adv_tau_xy = u * dtau_xy_dx + v * dtau_xy_dy
    adv_tau_yy = u * dtau_yy_dx + v * dtau_yy_dy

    stretch_xx = 2.0 * (tau_xx * du_dx + tau_xy * du_dy)
    stretch_xy = tau_xx * dv_dx + tau_xy * dv_dy + tau_xy * du_dx + tau_yy * du_dy
    stretch_yy = 2.0 * (tau_xy * dv_dx + tau_yy * dv_dy)

    strain_xx = 2.0 * du_dx
    strain_xy = du_dy + dv_dx
    strain_yy = 2.0 * dv_dy

    exp_xx = -adv_tau_xx + stretch_xx + ((1.0 - beta) / Wi) * strain_xx
    exp_xy = -adv_tau_xy + stretch_xy + ((1.0 - beta) / Wi) * strain_xy
    exp_yy = -adv_tau_yy + stretch_yy + ((1.0 - beta) / Wi) * strain_yy

    factor = 1.0 / (1.0 + dt / Wi)

    tau_xx_new = factor * (tau_xx + dt * exp_xx)
    tau_xy_new = factor * (tau_xy + dt * exp_xy)
    tau_yy_new = factor * (tau_yy + dt * exp_yy)

    return fft2c(tau_xx_new), fft2c(tau_xy_new), fft2c(tau_yy_new)

# ----------------------------------------------------------------------
# 6. Snapshot and composite functions
# ----------------------------------------------------------------------

def save_snapshot(step, t):
    u_phys = ifft2c(u_hat)
    v_phys = ifft2c(v_hat)
    tau_xx_phys = ifft2c(tau_xx_hat)
    omega = ifft2c(1j * Kx * v_hat - 1j * Ky * u_hat)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].contourf(X, Y, u_phys, levels=30, cmap=cm.RdBu_r)
    axs[0, 0].set_title('u')
    axs[0, 0].axis('equal')
    axs[0, 1].contourf(X, Y, v_phys, levels=30, cmap=cm.RdBu_r)
    axs[0, 1].set_title('v')
    axs[0, 1].axis('equal')
    axs[1, 0].contourf(X, Y, omega, levels=30, cmap=cm.RdBu_r)
    axs[1, 0].set_title('Vorticity')
    axs[1, 0].axis('equal')
    axs[1, 1].contourf(X, Y, tau_xx_phys, levels=30, cmap=cm.viridis)
    axs[1, 1].set_title('τ_xx')
    axs[1, 1].axis('equal')
    plt.suptitle(f'Viscoelastic DNS (Patched), t={t:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"snapshots_patch/snapshot_{step:06d}.png", dpi=100)
    plt.close(fig)

def create_composite(files, times):
    n = len(files)
    if n == 0:
        return
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(plt.imread(files[i]))
            ax.axis('off')
            ax.set_title(f't={times[i]:.3f}s', fontsize=10)
        else:
            ax.axis('off')
    plt.suptitle('Viscoelastic DNS Evolution (Patched)', fontsize=16)
    plt.tight_layout()
    plt.savefig("composite_patch/composite.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# 7. Main simulation loop
# ----------------------------------------------------------------------

print("=" * 60)
print("Viscoelastic DNS (PATCHED) - Stronger Body Force + Perturbation")
print("=" * 60)
print(f"Grid: {nx}x{ny}, Re={Re}, Wi={Wi}, beta={beta}, F0={F0}")
print("Initial: u=0.1*sin(Y)*cos(X), v=0.1*cos(Y)*sin(X)")
print("=" * 60)

u_hat = fft2c(u)
v_hat = fft2c(v)
tau_xx_hat = fft2c(tau_xx)
tau_xy_hat = fft2c(tau_xy)
tau_yy_hat = fft2c(tau_yy)

u_hat[0, 0] = 0.0
v_hat[0, 0] = 0.0
tau_xx_hat[0, 0] = 0.0
tau_xy_hat[0, 0] = 0.0
tau_yy_hat[0, 0] = 0.0

times = []
KE = []
PE = []
snapshot_files = []
snapshot_times = []

start = time.time()

for step in range(nt + 1):
    t = step * dt

    # Body force: strong forcing at ky=1 and ky=2
    body_x_hat = np.zeros_like(u_hat)
    body_x_hat[0, 1] = F0
    body_x_hat[0, 2] = F0 * 0.3
    body_y_hat = np.zeros_like(v_hat)

    # RK4 for velocity
    du1, dv1 = rhs_velocity(u_hat, v_hat, tau_xx_hat, tau_yy_hat, tau_xy_hat, Re, beta)
    du1 += body_x_hat
    dv1 += body_y_hat

    u1 = u_hat + 0.5 * dt * du1
    v1 = v_hat + 0.5 * dt * dv1

    du2, dv2 = rhs_velocity(u1, v1, tau_xx_hat, tau_yy_hat, tau_xy_hat, Re, beta)
    du2 += body_x_hat
    dv2 += body_y_hat

    u2 = u_hat + 0.5 * dt * du2
    v2 = v_hat + 0.5 * dt * dv2

    du3, dv3 = rhs_velocity(u2, v2, tau_xx_hat, tau_yy_hat, tau_xy_hat, Re, beta)
    du3 += body_x_hat
    dv3 += body_y_hat

    u3 = u_hat + dt * du3
    v3 = v_hat + dt * dv3

    du4, dv4 = rhs_velocity(u3, v3, tau_xx_hat, tau_yy_hat, tau_xy_hat, Re, beta)
    du4 += body_x_hat
    dv4 += body_y_hat

    u_hat_new = u_hat + (dt / 6.0) * (du1 + 2*du2 + 2*du3 + du4)
    v_hat_new = v_hat + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    u_hat_new[0, 0] = 0.0
    v_hat_new[0, 0] = 0.0

    # Implicit stress update
    tau_xx_hat_new, tau_xy_hat_new, tau_yy_hat_new = update_stress_implicit(
        u_hat_new, v_hat_new, tau_xx_hat, tau_yy_hat, tau_xy_hat,
        dt, Wi, beta
    )

    u_hat = u_hat_new
    v_hat = v_hat_new
    tau_xx_hat = tau_xx_hat_new
    tau_xy_hat = tau_xy_hat_new
    tau_yy_hat = tau_yy_hat_new

    if step % 100 == 0:
        times.append(t)
        KE.append(energy(u_hat, v_hat))
        PE.append(polymer_energy(ifft2c(tau_xx_hat), ifft2c(tau_yy_hat), ifft2c(tau_xy_hat)))

    if step % snapshot_every == 0 and step > 0:
        save_snapshot(step, t)
        snapshot_files.append(f"snapshots_patch/snapshot_{step:06d}.png")
        snapshot_times.append(t)
        print(f"Snapshot {len(snapshot_files)}: t={t:.3f}s")

    if step % plot_every == 0 and step > 0:
        print(f"Step {step}/{nt}, t={t:.3f}, KE={KE[-1]:.6f}, PE={PE[-1]:.6f}")

end = time.time()
print(f"\nSimulation finished in {end - start:.2f} seconds.")
print(f"Saved {len(snapshot_files)} snapshots.")

# Energy plot
print("\nCreating energy plot...")
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(times, KE, 'b-', lw=2)
axs[0].set_ylabel('Kinetic Energy')
axs[0].set_xlabel('Time')
axs[0].grid(True)
axs[0].set_title('Kinetic Energy (Patched)')
axs[1].plot(times, PE, 'r-', lw=2)
axs[1].set_ylabel('Polymer Energy')
axs[1].set_xlabel('Time')
axs[1].grid(True)
axs[1].set_title('Polymer Energy (Patched)')
plt.tight_layout()
plt.savefig("composite_patch/energy.png", dpi=150)
plt.close()

print("Creating composite figure...")
create_composite(snapshot_files, snapshot_times)

# Final fields
u_final = ifft2c(u_hat)
v_final = ifft2c(v_hat)
omega_final = ifft2c(1j * Kx * v_hat - 1j * Ky * u_hat)
tau_xx_final = ifft2c(tau_xx_hat)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].contourf(X, Y, u_final, levels=30, cmap=cm.RdBu_r)
axs[0, 0].set_title('u final')
axs[0, 0].axis('equal')
axs[0, 1].contourf(X, Y, v_final, levels=30, cmap=cm.RdBu_r)
axs[0, 1].set_title('v final')
axs[0, 1].axis('equal')
axs[1, 0].contourf(X, Y, omega_final, levels=30, cmap=cm.RdBu_r)
axs[1, 0].set_title('Vorticity final')
axs[1, 0].axis('equal')
axs[1, 1].contourf(X, Y, tau_xx_final, levels=30, cmap=cm.viridis)
axs[1, 1].set_title('τ_xx final')
axs[1, 1].axis('equal')
plt.tight_layout()
plt.savefig("composite_patch/final_fields.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print(f"Final KE = {KE[-1]:.6f}")
print(f"Final PE = {PE[-1]:.6f}")
print(f"PE max = {max(PE):.6f}")
print("=" * 60)
print("\nAll outputs saved to:")
print("  - snapshots_patch/")
print("  - composite_patch/")
print("=" * 60)
print("Simulation completed successfully!")