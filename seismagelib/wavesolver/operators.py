from devito import TimeFunction, Function, Operator, Eq, solve
from examples.seismic import SeismicModel

def create_operators(forward_pde, adjoint_pde, parameters, gradient_pde=None):
    for pname in parameters:
        if not(pname in SeismicModel._known_parameters):
            SeismicModel._known_parameters.append(pname)
            

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


    if not(gradient_pde is None):
        def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
            """
            Construct a gradient operator in an acoustic media.

            Parameters
            ----------
            model : Model
                Object containing the physical parameters.
            geometry : AcquisitionGeometry
                Geometry object that contains the source (SparseTimeFunction) and
                receivers (SparseTimeFunction) and their position.
            space_order : int, optional
                Space discretization order.
            save : int or Buffer, optional
                Option to store the entire (unrolled) wavefield.
            kernel : str, optional
                Type of discretization, centered or shifted.
            """
            m = model.m

            # Gradient symbol and wavefield symbols
            # grad = Function(name='grad', grid=model.grid)
            grads = {
                pname: Function(name='grad' + str.capitalize(pname), grid=model.grid) for pname in parameters
            }
            u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save else None, 
                             time_order=2, space_order=space_order)
            v = TimeFunction(name='v', grid=model.grid, save=None,
                            time_order=2, space_order=space_order)
            dummy_func = Function(name='dummy', grid=model.grid)
            rec = geometry.rec
            

            s = model.grid.stepping_dim.spacing

            adj_eqn, rec_expr = adjoint_pde(v, model, rec, s)
            grad_eqns = gradient_pde(u, v, model)

            gradient_update = [
                Eq(grads[pname], grads[pname] + grad_eqns.get(pname,dummy_func)) for pname in parameters
            ]
            # Add expression for receiver injection
            receivers = rec.inject(field=v.backward, expr=rec_expr)

            # Substitute spacing terms to reduce flops
            # print([adj_eqn] + receivers + gradient_update)
            return Operator([Eq(v.backward, solve(adj_eqn, v.backward))] + receivers + gradient_update, subs=model.spacing_map,
                            name='Gradient', **kwargs)
    else:
        GradientOperator = None


    return ForwardOperator, AdjointOperator, GradientOperator