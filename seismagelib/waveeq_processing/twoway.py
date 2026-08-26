import numpy as np

from pylops import LinearOperator
from pylops.utils.typing import NDArray
from seismagelib.waveeq_processing._propertiesmixin import PhysicalPropertiesMixin


from examples.seismic import AcquisitionGeometry, Receiver
from examples.seismic.source import TimeAxis
from examples.seismic.utils import sources


from pylops.waveeqprocessing._twoway import _CustomSource


class _Wave(LinearOperator, PhysicalPropertiesMixin):
    def _create_geometry(
        self,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str,
        f0: float = 20.0,
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
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
            Initial time
        tn : :obj:`int`
            Final time in ms
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, self.model.dim))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, -1] = src_z
        if self.model.dim == 3:
            src_coordinates[:, 1] = src_y

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, -1] = rec_z
        if self.model.dim == 3:
            rec_coordinates[:, 1] = rec_y

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
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
        wav_padded = np.pad(wav, (0, self.geometry.nt - len(wav)))

        self.wav = _CustomSource(
            name="src",
            grid=self.model.grid,
            wav=wav_padded,
            time_range=self.geometry.time_axis,
        )

    def create_receiver(
        self, name, rx=None, ry=None, rz=None, t0=None, tn=None, dt=None
    ):
        if self.model.dim == 2 and ry is not None:
            raise Exception("Attempting to create 3D receiver for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt

        rx = rx if rx is not None else self.geometry.rec_positions[:, 0]
        rz = rz if rz is not None else self.geometry.rec_positions[:, -1]
        if self.model.dim == 3:
            ry = ry if ry is not None else self.geometry.rec_positions[:, 1]

        nrec = len(rx)

        rec_coordinates = np.empty((nrec, self.model.dim))
        rec_coordinates[:, 0] = rx
        rec_coordinates[:, -1] = rz
        if self.model.dim == 3:
            rec_coordinates[:, 1] = ry

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)
        return Receiver(
            name=name,
            grid=self.geometry.grid,
            time_range=time_axis,
            npoint=nrec,
            coordinates=rec_coordinates,
        )

    def create_source(
        self,
        name,
        sx=None,
        sy=None,
        sz=None,
        t0=None,
        tn=None,
        dt=None,
        f0=None,
        src_type=None,
    ):

        if self.model.dim == 2 and sy is not None:
            raise Exception("Attempting to create 3D source for a 2D operator!")

        tn = tn or self.geometry.tn
        t0 = t0 or self.geometry.t0
        dt = dt or self.model.critical_dt
        f0 = f0 or self.geometry.f0

        src_type = src_type or self.geometry.src_type

        sx = sx or self.geometry.src_positions[:, 0]
        sz = sz or self.geometry.src_positions[:, -1]
        if self.model.dim == 3:
            sy = sy or self.geometry.src_positions[:, 1]

        nsrc = len(sx)

        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = sx
        src_coordinates[:, -1] = sz
        if self.model.dim == 3:
            src_coordinates[:, 1] = sy

        time_axis = TimeAxis(start=t0, stop=tn, step=self.geometry.dt)

        return sources[src_type](
            name=name,
            grid=self.geometry.grid,
            f0=f0,
            time_range=time_axis,
            npoint=nsrc,
            coordinates=src_coordinates,
            t0=t0,
        )

    def _update_dimensions(self, new_dims, new_dimsd):
        """
        Update the dimensions and shape of the object.

        Parameters:
        -----------
        new_dims : tuple
            The new value of dims.
        new_dimsd : tuple
            The new value of dimsd.

        """

        del self.dims
        del self.dimsd

        self.shape = (np.prod(new_dimsd), np.prod(new_dims))

        self.dims = new_dims
        self.dimsd = new_dimsd

    def _update_geometry(self, rx, rz, sx, sz, nrecs):
        """
        Update the geometry with new receiver and source positions.

        Parameters
        ----------
        rx : array-like
            Array containing the x-coordinates of the receivers.
        rz : array-like
            Array containing the z-coordinates of the receivers.
        sx : float
            x-coordinate of the source.
        sz : float
            z-coordinate of the source.
        nrecs : int
            Number of receivers.
        tn : float
            Final recording time.

        Notes
        -----
        This method updates the `geometry` attribute of the object by creating
        a new `AcquisitionGeometry` instance with the provided receiver and
        source positions, as well as the updated final recording time.

        For now, it only works for 2D operatores.
        """

        new_rec_positions = np.zeros((nrecs, 2))
        new_rec_positions[:, 0] = rx
        new_rec_positions[:, -1] = rz

        new_src_positions = np.zeros((1, 2))
        new_src_positions[:, 0] = sx
        new_src_positions[:, -1] = sz

        self.geometry = AcquisitionGeometry(
            self.model,
            new_rec_positions,
            new_src_positions,
            self.geometry.t0,
            self.geometry.tn,
            src_type=self.geometry.src_type,
            f0=self.geometry.f0,
        )

    def _update_op_coords(self, id_src, relative_coords=False):
        """
        Update operator coordinates and dimensions based on SEGY file data.

        This method retrieves the source and receiver coordinates, time sampling
        information, and other relevant parameters from the SEGY file associated
        with the operator. It updates the internal model time step, geometry, and
        dimensions as needed.

        Raises:
            Exception: If the SEGY file is used but the shot index (`id_src`) is not defined.
        """
        if relative_coords:
            src_coords, rec_coords = self.segyReader.getRelativeCoords(id_src)
        else:
            src_coords, rec_coords = self.segyReader.getCoords(id_src)
        rx, rz = rec_coords
        sx, sz = src_coords

        nrec = len(rx)

        # Check if the number of receivers is variable and differs from the current geometry.
        # If so, update the dimensions to match the new number of receivers for the current shot.
        if nrec != self.geometry.nrec:
            self._update_dimensions(
                new_dims=self.dims, new_dimsd=(1, nrec, self.geometry.nt)
            )

        self._update_geometry(rx, rz, sx, sz, nrec)

    def resample(self, data, num):
        nshots, ntraces, nsteps = data.shape

        time_range = TimeAxis(
            start=self.geometry.time_axis.start,
            stop=self.geometry.time_axis.stop,
            num=nsteps,
        )
        new_data = np.zeros((nshots, ntraces, num), dtype=np.float32)
        for shot_id in range(nshots):

            rec = Receiver(
                name="rec", grid=self.model.grid, npoint=ntraces, time_range=time_range
            )
            rec.data[:] = data[shot_id].T
            rec = rec.resample(num=num)

            new_data[shot_id] = rec.data.T
        return new_data

    def add_args(self, **kwargs):
        self.karguments = kwargs

    def update_args(self, **kwargs):
        self.karguments.update(kwargs)

    def set_shotID(self, id_src, relative_coords=False):
        """
        Set the ID for the shot that will be executed by the operator.

        This method updates the operator's internal state to process a specific
            raise Exception("Can not set shot ID for an operator that doesn't have segyReader")
        for the given shot ID and updates the geometry accordingly.

        Parameters
        ----------
        id_src : int
            The ID of the shot to be executed. This corresponds to the shot index
            in the SEGY file.
        relative_coords : bool, optional
            If True, the coordinates will be treated as relative to the model's
            origin. If False, the coordinates are treated as absolute. Default is False.

        Raises
        ------
        Exception
            If the operator does not have a SEGY reader or if the shot ID is invalid.
        """
        segyReader = getattr(self, "segyReader", None)

        if not segyReader:
            raise Exception(
                "Can not set shot ID for a operator that doesn't have segyReader"
            )

        self._update_op_coords(id_src, relative_coords=relative_coords)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        if len(m.shape) == 2:
            return m[nbl:-nbl, nbl:-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]

