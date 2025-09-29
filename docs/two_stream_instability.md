---
title: Two‑stream instability
---

# Two‑stream instability

The two‑stream instability is a kinetic plasma instability where two beams of charged particles moving at different velocities interact via self‑consistent electric fields.  flexipde includes a simple 1D Vlasov–Poisson solver to simulate this phenomenon.

## Equations

We consider a distribution function $f(x,v,t)$ of electrons in one spatial dimension and one velocity dimension.  The Vlasov equation reads

$$
\partial_t f + v\,\partial_x f + E(x,t)\,\partial_v f = 0,
$$

where $E(x,t)$ is the electric field determined by Poisson’s equation

$$
\partial_x E = 1 - \int f\,\mathrm{d}v.
$$

The model in flexipde discretises velocity space on a regular grid $v_j\in[v_{\min},v_{\max}]$ and uses a spectral method in the spatial direction.  The electric field is obtained by solving Poisson’s equation in Fourier space.  Boundary conditions are periodic in $x$.

## Usage

To simulate the two‑stream instability, create a TOML configuration with a Vlasov model:

```toml
[grid]
dimensions = [[0.0, 2*pi]]
resolution = [64]
periodic = [true]

[model]
type = "vlasov"
nv = 64
v_min = -5.0
v_max = 5.0

[initial_conditions]
amplitude = 0.05
thermal_velocity = 0.5
drift_velocity = 2.0
background_density = 0.5
```

Run from the command line:

```bash
flexipde examples/vlasov_two_stream
```

or from a Python script using :mod:`flexipde.io.build_simulation`.

## Manufactured solution test

The manufactured solution for this model uses a static distribution that is independent of time with zero electric field.  If the perturbation amplitude and drift velocity are set to zero, the distribution should remain constant.  The test `test_vlasov_constant_distribution_is_time_invariant` verifies this behaviour.
