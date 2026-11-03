# Taylor Flow Mass Transfer Libraries

Two complementary Python modules for modelling gas–liquid mass transfer in **Taylor flow** (slug flow) inside capillaries / microchannels.

| File | Purpose |
|------|---------|
| `pyTaylor3.py` | Physical mass transfer (no reaction). Computes film thickness, bubble velocity, concentration profiles and the volumetric mass-transfer coefficient \(k_La\). |
| `Taylor_with_reaction.py` | Extension that includes a first-order homogeneous chemical reaction. Calculates local and overall mass-transfer rates on the bubble surface (caps + cylindrical body). |

Both modules share the same hydrodynamic foundation (Han & Shikazono film-thickness correlation + Bretherton-type film velocity) and are released under a permissive MIT-style licence.

**Author:** Sergio Bordel  
**Contact:** sergio_bordel@hotmail.com

---

## Dependencies

```bash
numpy
scipy
matplotlib   # only needed if you want to plot
```

Python ≥ 3.7 is recommended.

---

## 1. `pyTaylor3.py` – Physical mass transfer

### Physical constants (water @ 25 °C)
```python
rho   = 997.05      # kg m⁻³
mu    = 0.00089     # Pa·s
g     = 9.8         # m s⁻²
D     = 1.88e-9     # m² s⁻¹  (typical liquid diffusivity)
sigma = 0.07199     # N m⁻¹
```

### Main public functions

| Function | Description |
|----------|-------------|
| `delta(mu, U, sigma, rho, D)` | Dimensionless film thickness (Han–Shikazono correlation). |
| `delta2(mu, Us, sigma, rho, Rc, g)` | Returns film thickness \(\delta\) and bubble velocity \(U_b\) for a given superficial liquid velocity \(U_s\). |
| `Evolvedprofiles(D, rho, g, mu, sigma, Us, Rc, Lb, Ls)` | Iteratively solves the concentration profiles in the liquid film around the bubble and in the liquid slug. Returns the saturation factor and the Fourier coefficients. |
| `Kla(Us, Rc, D, Ls, Lb, rho, g, mu, sigma)` | **Key result.** Volumetric mass-transfer coefficient \(k_La\) (s⁻¹) of a unit cell (one bubble + one slug). Also returns the separate contributions of the film and of the two caps. |

### Typical usage

```python
from pyTaylor3 import Kla, delta2

Rc  = 0.001      # capillary radius (m)
Us  = 0.1        # superficial velocity (m/s)
Ls  = 0.11       # liquid slug length (m)
Lb  = 0.11       # bubble length (m)

kla, kla_film, kla_caps = Kla(Us, Rc, D, Ls, Lb, rho, g, mu, sigma)
print(f"kLa = {kla:.4e} s⁻¹")
```

---

## 2. `Taylor_with_reaction.py` – Mass transfer with reaction

This module extends the previous one by solving the convection–diffusion–reaction equation on the bubble surface.

### Temperature-dependent properties
```python
mu(T)     # viscosity of water (Pa·s)
sigma(T)  # surface tension of water (N m⁻¹)
```

### Hydrodynamics (identical to the first module)
- `delta(...)`
- `delta2(...)`  (note: gravity is hard-coded inside)

### Reaction-enhanced mass transfer

The bubble is divided into five regions:
1. Nose (upper cap) – two sub-regions  
2. Cylindrical body  
3. Tail (lower cap) – two sub-regions  

For each region the code evaluates the local mass-transfer rate \(J\) by solving a boundary-value problem that includes the Hatta number (reaction rate constant \(k\)).

| Function | Description |
|----------|-------------|
| `kla(k, Rb, Rc, Us, Ub, alpha, beta, D, rho, mu_v, z_max)` | Local \(k_L\) for a given geometry and reaction rate. |
| `kla_op(k, Rc, Us, D, mu_v, sigma_v, rho, z_max)` | **Convenience wrapper.** Computes film thickness, bubble velocity and shape factors automatically, then returns the overall mass-transfer coefficient \(k_L\). |

### Typical usage

```python
from Taylor_with_reaction import kla_op, mu, sigma

T      = 298.15          # K
Rc     = 0.001           # m
Us     = 0.05            # m/s
D      = 1.5e-9          # m²/s
z_max  = 5.0             # dimensionless cylindrical length
k      = 10.0            # first-order reaction rate constant (s⁻¹)

mu_v    = mu(T)
sigma_v = sigma(T)
rho     = 997.0

kL = kla_op(k, Rc, Us, D, mu_v, sigma_v, rho, z_max)
print(f"kL (with reaction) = {kL:.4e} m/s")
```

---

## Theoretical background (brief)

- **Film thickness** – Han & Shikazono correlation (valid for a wide range of Capillary and Reynolds numbers).  
- **Bubble velocity** – obtained by solving the film-flow balance including gravity.  
- **Concentration field (no reaction)** – Fourier-series solution of the unsteady diffusion equation in a moving frame; iterative matching between bubble and slug regions.  
- **Concentration field (with reaction)** – numerical solution of the parabolic PDE  
  \[
  \frac{\partial c}{\partial t} = D\nabla^2c - kc
  \]
  on the bubble surface, followed by surface integration of the local flux.

---

## Licence

```
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software...

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
```

Full licence text is included at the top of both source files.

---

## Citation

If you use these modules in academic work, please cite the original publications of Sergio Bordel (or contact the author for the preferred reference).
```
