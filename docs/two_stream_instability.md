# Two–stream instability

The **two–stream instability** is a fundamental kinetic phenomenon in plasma
physics: when two counter–streaming beams of charged particles interact,
small perturbations can grow exponentially due to resonant wave–particle
interactions.  In the Vlasov–Poisson model, the distribution function
$f(x,v,t)$ evolves according to

\[\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E(x,t) \frac{\partial f}{\partial v} = 0,\]

where the self–consistent electric field $E(x,t)$ is obtained from Poisson’s
equation

\[\frac{\partial E}{\partial x} = 1 - \int f(x,v,t) \, dv.\]

In FlexiPDE, the two–stream instability is implemented as the
`VlasovTwoStream` model.  It discretises the spatial derivative with either
spectral or finite differences and the velocity derivative with a simple
finite difference.  The initial condition is typically a superposition of
two drifting Maxwellians plus a small sinusoidal perturbation.  You can
configure the thermal velocity, drift velocity, perturbation amplitude and
wavenumber via the TOML configuration file or directly in Python.

See `examples/vlasov_two_stream.toml` and `examples/run_vlasov_two_stream.py` for a
complete working example.  The growth rate of the instability can be
optimised by varying the temperature ratio using the optimisation utilities in
`flexipde.optim`.