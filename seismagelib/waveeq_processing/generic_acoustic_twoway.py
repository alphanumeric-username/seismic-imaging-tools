__all__ = [
    "GenericAcousticWave2D",
    "GenericAcousticWave3D",
]

import gc
from typing import Any, List, NewType, Tuple, Union, TypeVar

import numpy as np

from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from pylops.utils.twowaympi import MPIShotsController
from pylops.waveeqprocessing.segy import ReadSEGY2D, count_segy_shots

devito_message = deps.devito_import("the vector reflectivity twoway module")

if devito_message is None:
    from devito import (
        DiskSwapConfig,
        Eq,
        Function,
        Operator,
        TensorTimeFunction,
        VectorFunction,
        VectorTimeFunction,
    )
    from devito.builtins import initialize_function

    from examples.seismic import AcquisitionGeometry, Model
    from examples.seismic.acoustic import AcousticWaveSolver
    from seismagelib.data_structures import NDArrayStructIntf


    from pylops.waveeqprocessing._twoway import _CustomSource
    from pylops.waveeqprocessing.twoway import _Wave
else:
    AcousticWaveSolver = Any

MPIComm = TypeVar("mpi4py.MPI.Comm")
AcousticWaveSolverType = NewType("AcousticWaveSolver", AcousticWaveSolver)

class _GenericAcousticWave(_Wave):
    """Devito Acoustic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time in ms
    tn : :obj:`int`
        Final time in ms
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    space_order : :obj:`int`
        Spatial ordering of FD stencil.
    model : :obj:`examples.seismic.Model`
        Devito model object.
    geometry : :obj:`examples.seismic.AcquisitionGeometry`
        Devito acquisition geometry object.
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape.

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        params: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        solver_cls,
        parameter_names: List[str],
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
        op_name: str = "fwd",
        src_y: NDArray = None,
        rec_y: NDArray = None,
        dt: int = None,
        segy_path: str = None,
        segy_mpi: MPIComm = None,
        segy_sample: Union[int, float] = None,
        mpi_instant_reduce: bool = False,
        dswap: bool = False,
        dswap_disks: int = 1,
        dswap_folder: str = None,
        dswap_path: str = None,
        dswap_compression: str = None,
        dswap_compression_value: float | int = None,
        dswap_verbose: bool = False,

    ) -> None:
        self.solver_cls = solver_cls
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        is_2d = len(shape) == 2
        is_3d = len(shape) == 3

        if is_2d and (rec_y is not None or src_y is not None):
            raise Exception(
                "Attempting to create a 3D operator using a 2D intended class!"
            )

        if is_3d and (rec_y is None or src_y is None):
            raise Exception(
                "Attempting to create a 2D operator using a 3D intended class!"
            )
        nparams = len(parameter_names)
        self.parameter_names = parameter_names
        # create model
        self._create_model(shape, origin, spacing, params, nparams, space_order, nbl, dt)
        self._create_geometry(
            src_x, src_y, src_z, rec_x, rec_y, rec_z, t0, tn, src_type, f0=f0
        )
        self.checkpointing = checkpointing
        self.karguments = {}

        if (segy_path):
            if is_3d:
                raise Exception("3D segy reader not available yet")

            nshots, shot_ids = count_segy_shots(segy_path)
            nsy = 1  # 2D

            sample = segy_sample or nshots
            if sample <= 0 or sample > nshots:
                raise Exception("segy sample must be between (0," + str(nshots) + "]")
            elif sample >= 1:
                # Straight number of samples
                sample = int(sample)
            else:
                # Percentage
                sample = int(nshots * sample)

            idxs = np.linspace(0, nshots - 1, num=sample, dtype=int)
            sampled_sids = [shot_ids[i] for i in idxs]

            if segy_mpi:
                controller = MPIShotsController(shape, sample, nsy, nbl, segy_mpi, shot_ids=sampled_sids)
                self.mpi_controller = controller

            self.segyReader = ReadSEGY2D(segy_path, mpi=getattr(self, "mpi_controller", None), shot_ids=sampled_sids)

        self.instant_reduce = mpi_instant_reduce

        self._dswap_opt = {
            "dswap": dswap,
            "dswap_disks": dswap_disks,
            "dswap_folder": dswap_folder,
            "dswap_path": dswap_path,
            "dswap_compression": dswap_compression,
            "dswap_compression_value": dswap_compression_value,
            "dswap_verbose": dswap_verbose,
        }

        super().__init__(
            dtype=np.dtype(dtype),
            dims=(nparams, *shape),
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )
        self._register_multiplications(op_name)

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        params: NDArray,
        nparams: int,
        space_order: int = 6,
        nbl: int = 20,
        dt: int = None,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            dt=dt,
            **{ self.parameter_names[i]: params[i] for i in range(nparams)}
        )

    def updatesrc(self, wav):
        """Update source wavelet

        This routines is used to allow users to pass a custom source
        wavelet to replace the source wavelet generated when the
        object is initialized

        Parameters
        ----------
        wav : :obj:`numpy.ndarray`
            Wavelet

        """
        cmp = self.geometry.nt - len(wav)
        if cmp > 0:
            wav = np.pad(wav, (0, self.geometry.nt - len(wav)))
        elif cmp < 0:
            wav = wav[:self.geometry.nt]


        self.wav = _CustomSource(
            name="src",
            grid=self.model.grid,
            wav=wav,
            time_range=self.geometry.time_axis,
        )
        
        # print('wav shape: ', self.wav.shape, self.geometry.nt)

    # # @abc.abstractmethod
    # def _srcillumination_oneshot(
    #     self, solver: AcousticWaveSolverType, isrc: int
    # ) -> Tuple[NDArray, NDArray]:
    #     """Source wavefield and illumination for one shot

    #     Parameters
    #     ----------
    #     solver : :obj:`AcousticWaveSolver`
    #         Devito's solver object.
    #     isrc : :obj:`int`
    #         Index of source to model

    #     Returns
    #     -------
    #     u0 : :obj:`numpy.ndarray`
    #         Source wavefield
    #     src_ill : :obj:`numpy.ndarray`
    #         Source illumination

    #     """
    #     pass

    # def srcillumination_allshots(self) -> NDArrayStructIntf:
    #     """Source wavefield and illumination for all shots

    #     Parameters
    #     ----------
    #     savewav : :obj:`bool`, optional
    #         Save source wavefield (``True``) or not (``False``)

    #     """
    #     # create geometry for single source
    #     geometry = AcquisitionGeometry(
    #         self.model,
    #         self.geometry.rec_positions,
    #         self.geometry.src_positions[0, :],
    #         self.geometry.t0,
    #         self.geometry.tn,
    #         f0=self.geometry.f0,
    #         src_type=self.geometry.src_type,
    #     )

    #     nsrc = self.geometry.src_positions.shape[0]
    #     # self.src_illumination = np.zeros(self.model.shape)

    #     mtot = np.zeros((len(self.parameter_names), *self.model.shape), dtype=self.dtype)

    #     solver = self.solver_cls(self.model, geometry, space_order=self.space_order)

    #     for isrc in range(nsrc):
    #         solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
    #         m = self._srcillumination_oneshot(solver)
    #         for i, pname in enumerate(self.parameter_names):
    #             mtot[i] += self._crop_model(m[pname].data, self.model.nbl)
    #         del m
    #         gc.collect()

    #     return mtot

    # def _born_oneshot(self, solver: AcousticWaveSolverType, dm: NDArray) -> NDArray:
    #     """Born modelling for one shot

    #     Parameters
    #     ----------
    #     solver : :obj:`AcousticWaveSolver`
    #         Devito's solver object.
    #     dm : :obj:`numpy.ndarray`
    #         Model perturbation

    #     Returns
    #     -------
    #     d : :obj:`numpy.ndarray`
    #         Data

    #     """

    #     # set perturbation
    #     dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
    #     if dmext.ndim == 2:
    #         dmext[
    #             self.model.nbl : -self.model.nbl,
    #             self.model.nbl : -self.model.nbl,
    #         ] = dm
    #     else:
    #         dmext[
    #             self.model.nbl : -self.model.nbl,
    #             self.model.nbl : -self.model.nbl,
    #             self.model.nbl : -self.model.nbl,
    #         ] = dm

    #     # assign source location to source object with custom wavelet
    #     if hasattr(self, "wav"):
    #         self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]

    #     d = solver.jacobian(dmext, src=None if not hasattr(self, "wav") else self.wav)[
    #         0
    #     ]
    #     d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
    #     return d

    # def _born_allshots(self, dm: NDArray) -> NDArray:
    #     """Born modelling for all shots

    #     Parameters
    #     -----------
    #     dm : :obj:`numpy.ndarray`
    #         Model perturbation

    #     Returns
    #     -------
    #     dtot : :obj:`numpy.ndarray`
    #         Data for all shots

    #     """
    #     # create geometry for single source
    #     geometry = AcquisitionGeometry(
    #         self.model,
    #         self.geometry.rec_positions,
    #         self.geometry.src_positions[0, :],
    #         self.geometry.t0,
    #         self.geometry.tn,
    #         f0=self.geometry.f0,
    #         src_type=self.geometry.src_type,
    #     )

    #     # solve
    #     solver = AcousticWaveSolver(
    #         self.model, geometry, space_order=self.space_order, **self._dswap_opt
    #     )

    #     nsrc = self.geometry.src_positions.shape[0]
    #     dtot = []

    #     for isrc in range(nsrc):
    #         solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
    #         d = self._born_oneshot(solver, dm)
    #         dtot.append(d)
    #     dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
    #     return dtot

    # @abc.abstractmethod
    # def _bornadj_oneshot(self, solver: AcousticWaveSolverType, isrc, dobs):
    #     pass

    # def _bornadj_allshots(self, dobs: NDArray) -> NDArray:
        """Adjoint Born modelling for all shots

        Parameters
        ----------
        dobs : :obj:`numpy.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`numpy.ndarray`
            Model

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        nsrc = self.geometry.src_positions.shape[0]
        
        mtot = np.zeros((len(self.parameter_names), *self.model.shape), dtype=self.dtype)

        solver = self.solver_cls(self.model, geometry, space_order=self.space_order)

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            m = self._bornadj_oneshot(solver, isrc, dobs[isrc])
            for i, pname in enumerate(self.parameter_names):
                mtot[i] += self._crop_model(m[pname].data, self.model.nbl)
            del m
            gc.collect()

        controller = getattr(self, "mpi_controller", None)
        if (controller and self.instant_reduce):
            return controller.build_result([mtot])[0]
        else:
            return mtot

    def _fwd_oneshot(self, solver: AcousticWaveSolverType, params: NDArray) -> NDArray:
        """Forward modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # create function representing the physical parameter received as parameter

        for i, pname in enumerate(self.parameter_names):
            pdata = params[i]
            function = Function(
                name=pname,
                grid=self.model.grid,
                space_order=self.model.space_order,
                parameter=True,
            )

            # Assignment of values to physical parameters functions based on the values in 'v'
            initialize_function(function, pdata, self.model.padsizes)

            # add vp to karguments to be used inside devito's solver
            self.karguments.update({pname: function})
        
        if hasattr(self, 'wav'):
            self.wav.coordinates.data[0, :] = solver.geometry.src_positions[:]
            self.karguments['src'] = self.wav
        
        nsrc = self.geometry.src_positions.shape[0]

        d = solver.forward(**self.karguments)[0]
        # print(np.linalg.norm(d.data[:]))
        d = d.resample(solver.geometry.dt).data[:][: solver.geometry.nt].T
        return d

    def _fwd_allshots(self, params: NDArray) -> NDArray:
        """Forward modelling for all shots

        Parameters
        -----------
        v : :obj:`np.ndarray`
            Velocity Model

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[0, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # solve
        solver = self.solver_cls(
            self.model,
            geometry,
            space_order=self.space_order,
        )

        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            solver.geometry.src_positions = self.geometry.src_positions[isrc, :]
            d = self._fwd_oneshot(solver, params)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot
    

    def _register_multiplications(self, op_name: str) -> None:
        if op_name == "born":
            self._acoustic_matvec = self._born_allshots
        if op_name == "fwd":
            self._acoustic_matvec = self._fwd_allshots
        # self._acoustic_rmatvec = self._bornadj_allshots
        self._acoustic_rmatvec = self._fwd_allshots

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_matvec(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._acoustic_rmatvec(x)
        return y


GenericAcousticWave2D = _GenericAcousticWave
GenericAcousticWave3D = _GenericAcousticWave
