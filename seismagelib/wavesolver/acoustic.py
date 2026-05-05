from devito import Function, TimeFunction
from devito.tools import memoized_meth

from examples.seismic import Model, AcquisitionGeometry

from seismagelib.wavesolver.operators import create_operators


def create_solver_class(forward_pde, ajoint_pde, parameters):

    ForwardOperator, AdjointOperator = create_operators(forward_pde, ajoint_pde)

    class _ParametricAcousticWaveSolver():
        def __init__(self, model: Model, geometry: AcquisitionGeometry, kernel='OT2', space_order=4, **kwargs):
            self.model = model
            self.model._initialize_bcs(bcs="damp")
            self.geometry: AcquisitionGeometry = geometry

            assert self.model.grid == geometry.grid

            self.space_order = space_order
            self.kernel = kernel

            # Cache compiler options
            self._kwargs = kwargs


        @property
        def dt(self):
            # Time step can be \sqrt{3}=1.73 bigger with 4th order
            if self.kernel == 'OT4':
                return self.model.dtype(1.73 * self.model.critical_dt)
            return self.model.critical_dt


        @memoized_meth
        def op_fwd(self, save=None):
            """Cached operator for forward runs with buffered wavefield"""
            return ForwardOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)


        @memoized_meth
        def op_adj(self, save=None):
            """Cached operator for adjoint runs"""
            return AdjointOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)

        def forward(self, src=None, rec=None, u=None, model: Model =None, save=None, **kwargs):
            # Source term is read-only, so re-use the default
            src = src or self.geometry.src
            # Create a new receiver object to store the result
            rec = rec or self.geometry.rec

            # Create the forward wavefield if not provided
            u = u or TimeFunction(name='u', grid=self.model.grid,
                                save=self.geometry.nt if save else None,
                                time_order=2, space_order=self.space_order)

            model = model or self.model
            # Pick vp from model unless explicitly provided
            kwargs.update(model.physical_params(**kwargs))
            kwargs2 = {}
            for k, v in kwargs.items():
                if k == 'damp' or k in parameters:
                    kwargs2[k] = v
            kwargs = kwargs2

            summary = self.op_fwd(save).apply(
                src = src, rec = rec, u = u, dt = kwargs.pop('dt', self.dt), **kwargs
            )

            return rec, u, summary
        

        def adjoint(self, rec, srca=None, v=None, model=None, save=None, **kwargs):
            """
            Adjoint modelling function that creates the necessary
            data objects for running an adjoint modelling operator.

            Parameters
            ----------
            rec : SparseTimeFunction or array-like
                The receiver data. Please note that
                these act as the source term in the adjoint run.
            srca : SparseTimeFunction or array-like
                The resulting data for the interpolated at the
                original source location.
            v: TimeFunction, optional
                The computed wavefield.
            model : Model, optional
                Object containing the physical parameters.
            vp : Function or float, optional
                The time-constant velocity.

            Returns
            -------
            Adjoint source, wavefield and performance summary.
            """
            # Create a new adjoint source and receiver symbol
            srca = srca or self.geometry.new_src(name='srca', src_type=None)

            # Create the adjoint wavefield if not provided
            v = v or TimeFunction(name='v', grid=self.model.grid,
                                save=self.geometry.nt if save else None,
                                time_order=2, space_order=self.space_order)

            model = model or self.model
            # Pick vp from model unless explicitly provided
            kwargs.update(model.physical_params(**kwargs))
            kwargs2 = {}
            for k, v in kwargs.items():
                if k == 'damp' or k in parameters:
                    kwargs2[k] = v
            kwargs = kwargs2

            # Execute operator and return wavefield and receiver data
            summary = self.op_adj(save).apply(srca=srca, rec=rec, v=v,
                                        dt=kwargs.pop('dt', self.dt), **kwargs)
            return srca, v, summary
    

    return _ParametricAcousticWaveSolver;