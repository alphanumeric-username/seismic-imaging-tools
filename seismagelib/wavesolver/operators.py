from devito import TimeFunction, Operator, Eq, solve

def create_operators(forward_pde, adjoint_pde):
    def ForwardOperator(model, geometry, space_order=4,
                        save=False, **kwargs):
        # Create symbols for forward wavefield, source and receivers
        u = TimeFunction(name='u', grid=model.grid,
                        save=geometry.nt if save else None,
                        time_order=2, space_order=space_order)
        src = geometry.src
        rec = geometry.rec


        # Create the stencil

        # pde = m * u.dt2 - rho * div(b * grad(u, .5), -.5) + model.damp * u.dt
        # pde = m * b *u.dt2 - div(b * grad(u, .5), -.5) + model.damp * u.dt
        s = model.grid.stepping_dim.spacing

        pde, src_expr = forward_pde(u, model, src, s)

        stencil = Eq(u.forward, solve(pde, u.forward))


        # Create the equation
        # src_term = src.inject(u.forward, expr = src* s**2 / m)
        src_term = src.inject(u.forward, expr = src_expr)
        # src_term = src.inject(u.forward, expr = src* s**2 / (m * b))
        rec_term = rec.interpolate(expr=u)

        equation = [stencil] + src_term + rec_term

        return Operator(equation, subs=model.spacing_map, name='Forward', **kwargs)

    def AdjointOperator(model, geometry, space_order=4,
                        kernel='OT2', save=False, **kwargs):
        v = TimeFunction(name='v', grid=model.grid, 
                        save=geometry.nt if save else None,
                        time_order=2, space_order=space_order)

        srca = geometry.new_src(name='srca', src_type=None)
        rec = geometry.rec

        s = model.grid.stepping_dim.spacing

        # eqn = m*v.dt2 - div(b * grad(rho * v, .5), -.5) + model.damp * v.dt.T
        # eqn = m * b * v.dt2 - div(b * grad(v, .5), -.5) + model.damp * v.dt.T
        pde, rec_expr = adjoint_pde(v, model, rec, s)
    
        stencil = Eq(v.backward, solve(pde, v.backward))

        # Construct expression to inject receiver values
        # receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)
        receivers = rec.inject(field=v.backward, expr=rec_expr)
        # receivers = rec.inject(field=v.backward, expr=rec * s**2 / (m * b))

        # Create interpolation expression for the adjoint-source
        
        source_a = srca.interpolate(expr=v)

        # Substitute spacing terms to reduce flops
        return Operator([stencil] + receivers + source_a, subs=model.spacing_map,
                        name='Adjoint', **kwargs)


    return ForwardOperator, AdjointOperator