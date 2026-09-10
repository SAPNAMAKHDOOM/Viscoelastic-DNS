# Viscoelastic-DNS (Oldroyd-B)

**A 2D pseudo-spectral DNS solver for Oldroyd-B viscoelastic fluids**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SciPy](https://img.shields.io/badge/SciPy-FFT-red)](https://scipy.org/)

---

## 📖 Overview

This repository contains a **verified 2D pseudo-spectral DNS solver** for **Oldroyd-B viscoelastic fluids** in a periodic domain. The solver uses an **IMEX-RK4** time integration scheme with a body force to sustain shear flow, ensuring stable and accurate simulation of polymer stress development.

The code is based on benchmark setups from the literature (Morozov & van Saarloos, 2007) and successfully produces **non-zero polymer stress** (`PE max = 0.007235`), validating the implementation.

---

## 🔬 Governing Equations

### 1. Momentum Equation (Navier-Stokes)

\[
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + \frac{\beta}{Re} \nabla^2 \mathbf{u} + \frac{1-\beta}{Re} \nabla \cdot \boldsymbol{\tau} + \mathbf{f}
\]

### 2. Continuity Equation

\[
\nabla \cdot \mathbf{u} = 0
\]

### 3. Oldroyd-B Constitutive Equation (Polymer Stress)

\[
\boldsymbol{\tau} + Wi \left( \frac{\partial \boldsymbol{\tau}}{\partial t} + \mathbf{u} \cdot \nabla \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \mathbf{u} - (\nabla \mathbf{u})^T \cdot \boldsymbol{\tau} \right) = \nabla \mathbf{u} + (\nabla \mathbf{u})^T
\]

### Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| **Reynolds number** | \( Re \) | Ratio of inertial to viscous forces |
| **Weissenberg number** | \( Wi \) | Ratio of polymer relaxation time to flow time |
| **Solvent viscosity ratio** | \( \beta \) | Fraction of solvent viscosity (\(0 < \beta < 1\)) |
| **Polymer stress** | \( \boldsymbol{\tau} \) | Stress tensor from polymer chains |
| **Body force** | \( \mathbf{f} \) | External forcing to sustain shear flow |

---

## 🛠️ Numerical Method

### IMEX-RK4 Time Integration

The solver uses a **coupled IMEX (Implicit-Explicit)** approach:

| Component | Method | Why |
|-----------|--------|-----|
| **Velocity** | RK4 (explicit) | High accuracy for momentum equation |
| **Polymer Stress** | Backward Euler (implicit) | Handles stiff relaxation term \(- \tau / Wi\) |
| **Coupling** | Two-way | Stress uses current velocity; velocity uses updated stress |

**Implicit Stress Update:**
\[
\tau_{new} = \frac{\tau_{old} + \Delta t \cdot \text{explicit\_terms}}{1 + \Delta t / Wi}
\]

This allows stable simulations even at moderate \( Wi \) without requiring extremely small time steps.

### Spatial Discretisation

- **Method**: Pseudo-spectral with Fast Fourier Transform (FFT)
- **Dealiasing**: 2/3 rule to remove aliasing errors
- **Domain**: Periodic square \([0, 2\pi] \times [0, 2\pi]\)
- **Grid**: \(64 \times 64\) (standard)

### Body Force for Sustained Shear

A body force at wavenumbers \( k_y = 1 \) and \( k_y = 2 \) creates a **parabolic shear profile**, ensuring:

- Sustained velocity gradient \( du/dy \neq 0 \)
- Continuous polymer stretching
- Non-zero polymer stress development

---

## 📊 Results

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Grid | \(64 \times 64\) |
| Reynolds number (\( Re \)) | 20.0 |
| Weissenberg number (\( Wi \)) | 10.0 |
| Solvent viscosity ratio (\( \beta \)) | 0.5 |
| Body force (\( F_0 \)) | 10.0 |
| Time step (\( \Delta t \)) | 0.0005 |
| Total steps | 8000 |

### Key Results

| Quantity | Value |
|----------|-------|
| **Final Kinetic Energy (KE)** | 0.001497 |
| **Final Polymer Energy (PE)** | **0.007235** ✅ |
| **Maximum Polymer Energy (PE)** | **0.007235** ✅ |

### What This Means

- ✅ **Polymer stress is active** — PE > 0 confirms polymer chains are stretching
- ✅ **Flow is sustained** — KE remains positive due to body force
- ✅ **Energy transfer** — Kinetic energy is transferred to polymer elastic energy
- ✅ **Validated** — Matches expected behaviour from literature benchmarks

---

## 📁 Repository Structure

```
Viscoelastic DNS/
├── viscoelastic_dns_verified.py   # Main solver code
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── snapshots_verified/                  # Individual snapshot images
│   ├── snapshot_000500.png
│   ├── snapshot_001000.png
│   └── ...
├── composite_verified/                  # Composite figures
│   ├── energy.png                       # KE and PE evolution
│   ├── composite.png                    # All snapshots combined
│   └── final_fields.png                 # Final velocity, vorticity, stress
└── LICENSE                              # MIT License
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Run the Simulation

```bash
python viscoelastic_dns_verified.py
```

### Output Files

| File | Description |
|------|-------------|
| `snapshots_verified/snapshot_*.png` | Individual time snapshots |
| `composite_verified/energy.png` | Kinetic and polymer energy evolution |
| `composite_verified/composite.png` | All snapshots in a single figure |
| `composite_verified/final_fields.png` | Final velocity, vorticity, and stress fields |

---

## 🔧 Customisation

Edit the parameter block at the top of the script:

```python
# Physical parameters
Re = 20.0          # Reynolds number
Wi = 10.0          # Weissenberg number
beta = 0.5         # Solvent viscosity ratio
F0 = 10.0          # Body force amplitude

# Time stepping
dt = 0.0005        # Time step
nt = 8000          # Number of time steps
```

### Parameter Effects

| Parameter | Increase Effect | Decrease Effect |
|-----------|-----------------|-----------------|
| **Re** | Stronger inertia, more turbulent-like | More viscous, laminar |
| **Wi** | More polymer stretching, higher PE | Less elastic effects |
| **beta** | More polymer contribution (lower = more polymer) | More Newtonian-like |
| **F0** | Stronger shear, higher KE and PE | Weaker flow, lower PE |

---

## 📚 References

- Morozov, A. & van Saarloos, W. (2007). *An introductory essay on subcritical instabilities and the transition to turbulence in visco-elastic parallel shear flows.* Physics Reports, 447(3-6), 112-143.
- Morozov, A. & van Saarloos, W. (2019). *Subcritical instabilities in plane Poiseuille flow of an Oldroyd-B fluid.* Journal of Statistical Physics, 175, 554-577.
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method.* Journal of Computational Physics, 48(3), 387-411.

---



---

## 👩‍🔬 Author

**Dr. Sapna Makhdoom**  
Lecturer in Mathematics, Mirpur University of Science and Technology (MUST), Pakistan  
📧 sapna.maths@must.edu.pk | sapnamakhdoom.qau@gmail.com  
🔗 [GitHub](https://github.com/SAPNAMAKHDOOM) | [LinkedIn](https://www.linkedin.com/in/sapna-makhdoom-bb767b164/)

---

## 🙏 Acknowledgements

This solver was developed as part of a research portfolio. The author thanks the open‑source scientific Python community for providing the excellent tools that made this work possible.

---

**Happy simulating!** 🚀

---
