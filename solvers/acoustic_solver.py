from devito import grad, div

PARAMETERS = 'vp', 'rho'  # List of parameters


def forward(u, model, src, dt):
    vp = model.vp
    rho = model.rho

    pde = 1/vp**2 * u.dt2 - rho * div(1/rho * grad(u, .5), -.5) + model.damp * u.dt  # Homogeneous PDE expression
    src_expr = src * dt**2 * vp**2 # Source injection term

    return pde, src_expr


def adjoint(v, model, rec, dt):
    vp = model.vp
    rho = model.rho

    pde = 1/vp**2 * v.dt2 - div(1/rho * grad(v * rho, .5), -.5) + model.damp * v.dt.T  # Homogeneous PDE expression
    rec_expr = rec * dt**2 * vp**2 # Source injection term

    return pde, rec_expr


def gradient(u, v, model):
    grad_eqns = {
        'vp': -u.dt * v.dt
    }

    return grad_eqns