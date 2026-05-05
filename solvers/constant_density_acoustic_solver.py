PARAMETERS = 'vp',  # List of parameters


def forward(u, model, src, dt):
    vp = model.vp

    pde = 1/vp**2 * u.dt2 - u.laplace + model.damp * u.dt  # Homogeneous PDE expression
    src_expr = src * dt**2 * vp**2 # Source injection term

    return pde, src_expr


def adjoint(v, model, rec, dt):
    vp = model.vp

    pde = 1/vp**2 * v.dt2 - v.laplace + model.damp * v.dt.T  # Homogeneous PDE expression
    rec_expr = rec * dt**2 * vp**2 # Source injection term

    return pde, rec_expr