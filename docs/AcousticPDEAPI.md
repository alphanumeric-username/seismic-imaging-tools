# Acoustic PDE API spec

Version: 0.1

```python
PARAMETERS = 'vp', 'damp'  # List of parameters

def forward(u: TimeFunction, model: SeismicModel, src: PointSource, dt: TimeDimension):
    pde = model.vp * u.dt2 - u.laplacian + model.damp * u.dt  # Homogeneous PDE expression
    src_expr = src * dt**2 * model.vp**2 # Source injection term

    return pde, src_expr


def adjoint(v: TimeFunction, model: SeismicModel, rec: PointSource, dt: TimeDimension):
    pde = model.vp * u.dt2 - u.laplacian + model.damp * u.dt.T  # Homogeneous PDE expression
    rec_expr = rec * dt**2 * model.vp**2 # Source injection term

    return pde, rec_expr
```