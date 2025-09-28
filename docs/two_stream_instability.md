% The two–stream instability and the Vlasov–Poisson solver
% =======================================================

One of the canonical problems in plasma physics is the **two–stream instability**,
where two counter–propagating beams of charged particles interact via
electrostatic fields.  A simplified description neglects collisions and
collective electromagnetic effects, leading to the **Vlasov–Poisson**
system.  In this document we describe the equations solved by the
``VlasovTwoStream`` model in ``flexipde``, outline how to configure and
run a two–stream simulation, and illustrate how to optimise the
instability growth rate.

## Vlasov–Poisson equations

In a one–dimensional, electrostatic approximation, the evolution of the
distribution function $f(x,v,t)$ is governed by the **Vlasov equation**

$$
\frac{\partial f}{\partial t}
  + v \frac{\partial f}{\partial x}
  + E(x,t) \frac{\partial f}{\partial v}
 = 0,
$$

with a self–consistent electric field $E$ obtained from **Poisson’s
equation**

$$
\frac{\partial E}{\partial x} = \rho(x,t) - \rho_0,
\qquad \rho(x,t) = \int f(x,v,t)\,\mathrm{d}v,
$$

where a neutralising background $\rho_0$ ensures that the zero mode of
the field vanishes.  We normalise physical constants so that the
electron charge and mass are unity.  Periodic boundary conditions are
assumed in both $x$ and $v$.  The numerical solver implemented in
``flexipde`` uses a Fourier spectral method in $x$ and a centred
finite difference in $v$ to compute spatial and velocity gradients.

### Initial conditions for two beams

A typical initial state for the two–stream instability consists of two
Maxwellian beams centred at velocities $\pm v_0$ and broadened by a
thermal width $v_t$:

$$
f_0(v) = \mathrm{e}^{-\left(\frac{v-v_0}{v_t}\right)^2}
         + \mathrm{e}^{-\left(\frac{v+v_0}{v_t}\right)^2}.
$$

This is normalised such that the total number density of the beams is
unity.  A small sinusoidal density perturbation proportional to
$\cos(2\pi m x / L)$ is then applied to seed the instability:

$$
f(x,v,0) = [1 + A \cos(2\pi m x/L)]\, f_0(v),
$$

where $A \ll 1$ is the perturbation amplitude and $m$ is the mode
number.  The parameters $A$, $m$, $v_0$, $v_t$ and the domain length
$L$ can be specified in the `initial_conditions` section of a
configuration file (see below).  More details about the implementation
can be found in the docstring of
``flexipde.models.vlasov.VlasovTwoStream.initial_state``.

### Solving the system in flexipde

To run a two–stream simulation with ``flexipde``, create a TOML
configuration file like the following:

```toml
[grid]
domain = [[0.0, 2*pi]]      # spatial domain of length 2π
shape = [128]               # number of spatial grid points
periodic = [true]

[discretisation]
scheme = "spectral"         # use Fourier derivatives in x
backend = "jax"            # activate JAX for JIT and auto‑diff

[model]
type = "vlasov"            # select the Vlasov–Poisson solver
nv = 64                    # velocity grid resolution
v_min = -5.0               # lower bound of velocity domain
v_max = 5.0                # upper bound of velocity domain

[initial_conditions]
amplitude = 1e-3           # density perturbation amplitude
mode = 1                   # spatial mode number m
drift_velocity = 1.0       # v0: drift speed of each beam
thermal_velocity = 1.0     # vt: thermal width of the beams
background_density = 1.0   # overall normalisation

[solver]
t0 = 0.0
t1 = 15.0
dt0 = 0.1
solver = "Dopri5"          # fifth order Runge–Kutta from Diffrax
```

Run the simulation with

```bash
python -m flexipde.run path/to/vlasov.toml --output two_stream.pkl.gz
```

The solver will produce a compressed pickled result file containing the
times and phase–space distribution at the start and end of the
simulation.  You can load this file with
``SimulationResult.load()`` for further analysis.

### Optimising the growth rate

Because the solver is built on top of JAX and Diffrax, it is
differentiable with respect to its inputs.  As highlighted by
JAX‑Fluids【889881592370340†L35-L41】 and JAX‑MD【925522637824375†L27-L34】, fully
differentiable solvers enable new research directions such as
gradient‑based optimisation and machine learning closures.  The
``flexipde.optim`` module provides a helper function
``simulate_and_grad`` that computes the gradient of a scalar
objective with respect to simulation parameters, and
``optimize_params`` wraps this in a simple training loop using
Optax.

In the ``examples`` directory there is a script
``optimise_vlasov_growth.py`` that tunes the **thermal velocity
ratio** $v_t$ to maximise the growth rate of the two–stream
instability.  The objective is defined as the negative of the mean
electric field energy at the final time; minimising this negative
value maximises the instability growth.  Running the script

```bash
python examples/optimise_vlasov_growth.py
```

will print the optimal thermal velocity found by a gradient ascent
routine.  Feel free to modify the objective function or optimise
additional parameters.

### Further reading

* Consult the [User guide](user_guide.md) for general instructions on
  configuring and running simulations.
* The [Models](models.md) page lists all built‑in models and their
  parameters.
* For a deeper dive into the numerical methods, see the
  [Discretisation](discretization.md) page.