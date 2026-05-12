# Futures
from __future__ import annotations  # forward declaration, required <3.9

# Standard Library
from functools import partialmethod
from math import inf, nan, pi
from numbers import Real
from sys import stderr
from typing import Callable, Iterable, NamedTuple, Optional, Tuple, Union

# PyPI
import numpy as np
import torch

# Local
from .tools_mini import decorator_dict
from .tools_torch import ShapedTensor as Tensor


class PointData(NamedTuple):
	t: float
	pos: Tensor['P','D':2]
	vel: Tensor['P','D':2]
	
	@classmethod
	def acc(cls, items: Iterable[PointData], t=None) -> PointData:
		ps,vs,ts = zip(*items)
		pos,vel = (torch.cat(x) for x in (ps,vs))
		t = ts[-1] if t is None else t
		return PointData(t, pos, vel)
	
	@classmethod
	def acc_gen(cls, items: Iterable[PointData], period: Real) -> Iterable[Tuple[PointData, int]]:
		acc = []
		end_time = 0
		
		def flush():
			nonlocal cls, acc, end_time, period
			if acc:
				yield cls.acc(acc, end_time), len(acc)
				acc.clear()
				end_time += period
		
		for item in items:
			t = item.t
			if t > end_time:
				yield from flush()
				if t > end_time:
					# start new sequence
					end_time = t + period
					if acc:
						print("Starting new sequence", file=stderr)
						# TODO make broken sequences cleaner
			acc.append(item)
		yield from flush()


def clip_inf(value):
	""" This is usually a bad idea. It didn't work for what I wanted."""
	finfo = torch.finfo(value.dtype)
	value.data[value == -inf] = finfo.min
	value.data[value == +inf] = finfo.max
	return value


class GridData(torch.Tensor):
	""" Tensor grid data with a specific channel dimension of "density", "vel_mean", and "vel_var".
		
		Channel (4D) = density (1D) + velocity mean (2D) + velocity (isotropic) variance (1D)
		TODO NOTE: vel_var is isotropic (Sigma = vel_var * eye(2)), not "scalar variance" (= trace(Sigma) = 2*vel_var)
		Tensor shape is [...,4,H,W]
	"""
	
	@staticmethod
	def __new__(cls,
		tensor=None, *,
		density=None, logdensity=None,
		vel_mean=None,
		vel_var=None, vel_logvar=None,
	):
		if tensor is None:
			# todo remove asserts
			assert 1 == sum(x is not None for x in (density, logdensity))
			assert 1 == sum(x is not None for x in (vel_mean,))
			assert 1 == sum(x is not None for x in (vel_var, vel_logvar))
			*sB,_,H,W = (density if density is not None else logdensity).shape
			tensor = torch.full((*sB,4,H,W), nan)
		
		else: assert 0 == sum(x is not None for x in (density,logdensity,vel_mean,vel_var,vel_logvar))
		
		valid_shape = tensor.ndim >= 3 and tensor.shape[-3] == 4
		
		if isinstance(tensor, GridData):
			assert valid_shape
			self = tensor
		elif not valid_shape:
			self = tensor
		else:
			self = tensor.as_subclass(cls)
		
		return self
	
	def __init__(self, tensor=None, **kwargs):
		super().__init__()
		for k in ('density','logdensity','vel_mean','vel_var','vel_logvar'):
			if (v:=kwargs.get(k)) is not None:
				setattr(self, k, v)
	
	@classmethod
	def from_tensor(cls, tensor, format):
		gd = GridData(tensor)
		if format == cls._raw_format:
			return gd
		else:
			return GridData(**{k:v for k,v in zip(format,(gd._raw_density, gd._raw_vel_est, gd._raw_vel_unc))})
	
	def as_tensor(self, *format):
		if format == self._raw_format:
			return self
		else:
			return torch.cat([getattr(self,k) for k in format], dim=-3)
	
	
	#%% These properties simply reference the raw stored values, without definition of scale
	@property
	def _raw_density(self): return self[...,0:1,:,:].as_subclass(self.__class__.__base__)
	@_raw_density.setter
	def _raw_density(self, value): self[...,0:1,:,:] = value
	
	@property
	def _raw_vel_est(self): return self[..., 1:3, :, :].as_subclass(self.__class__.__base__)
	@_raw_vel_est.setter
	def _raw_vel_est(self, value): self[..., 1:3, :, :] = value
	
	@property
	def _raw_vel_unc(self): return self[..., 3:4, :, :].as_subclass(self.__class__.__base__)
	@_raw_vel_unc.setter
	def _raw_vel_unc(self, value): self[..., 3:4, :, :] = value
	
	
	#%% these properties reference the raw stored values, scaled as appropriate for the definition
	#   changing these definitions changes the scale of the stored values
	_raw_format = ('density','vel_mean','vel_var')
	
	@property
	def density(self): return self._raw_density
	@density.setter
	def density(self, value): self._raw_density = value
	
	@property
	def logdensity(self): return self._raw_density.log()
	@logdensity.setter
	def logdensity(self, value): self._raw_density = value.exp()
	
	@property
	def vel_mean(self): return self._raw_vel_est
	@vel_mean.setter
	def vel_mean(self, value): self._raw_vel_est = value
	
	@property
	def vel_var(self): return self._raw_vel_unc
	@vel_var.setter
	def vel_var(self, value): self._raw_vel_unc = value
	
	@property
	def vel_logvar(self): return self._raw_vel_unc.log()
	@vel_logvar.setter
	def vel_logvar(self, value): self._raw_vel_unc = value.exp()
	
	@property
	def vel_std(self): return self._raw_vel_unc.sqrt()
	@vel_std.setter
	def vel_std(self, value): self._raw_vel_unc = value.square()
	
	
	@property
	def count(self): return self.density.sum(dim=(-1,-2), keepdims=True)
	
	# @property
	# def parts(self): return self.density, self.vel_mean, self.vel_var
	
	
	_torch_functions = decorator_dict()
	
	@classmethod
	def __torch_function__(cls, func, types, args=(), kwargs=None):
		if kwargs is None:
			kwargs = {}
		if func in cls._torch_functions:
			return cls._torch_functions[func](*args, **kwargs)
		else:
			return super().__torch_function__(func, types, args, kwargs)
	
	
	def __getitem__(self, item):
		if not isinstance(item, tuple):
			item = (item,)
		# ellips = [i for i,e in enumerate(item) if e is Ellipsis]
		
		ret = super().__getitem__(item)
		try:
			ret.point_pos = self.point_pos[item]
			ret.point_vel = self.point_vel[item]
			ret.point_time = self.point_time[item]
		except Exception: pass
		return ret
	
	
	@staticmethod
	@_torch_functions.add(torch.stack)
	def stack(gds, dim=0, out=None):
		base = [
			gd.as_subclass(torch.Tensor) if isinstance(gd, GridData) else torch.as_tensor(gd)
			for gd in gds
		]
		with torch._C.DisableTorchFunction():
			gd = GridData(torch.stack(base, dim=dim, out=out))
		try:
			if pos:=[gd.point_pos for gd in gds if hasattr(gd,'point_pos')]: gd.point_pos = np.stack(pos, axis=dim)
			if vel:=[gd.point_vel for gd in gds if hasattr(gd,'point_vel')]: gd.point_vel = np.stack(vel, axis=dim)
			if t :=[gd.point_time for gd in gds if hasattr(gd,'point_time')]: gd.point_time = np.stack(t, axis=dim)
		except Exception: pass
		return gd
	
	
	@staticmethod
	@_torch_functions.add(torch.cat)
	def cat(gds, dim=0, out=None):
		base = [
			gd.as_subclass(torch.Tensor) if isinstance(gd, GridData) else torch.as_tensor(gd)
			for gd in gds
		]
		with torch._C.DisableTorchFunction():
			gd = GridData(torch.cat(base, dim=dim, out=out))
		try:
			if pos:=[gd.point_pos for gd in gds if hasattr(gd,'point_pos')]: gd.point_pos = np.concatenate(pos, axis=dim)
			if vel:=[gd.point_vel for gd in gds if hasattr(gd,'point_vel')]: gd.point_vel = np.concatenate(vel, axis=dim)
			if t :=[gd.point_time for gd in gds if hasattr(gd,'point_time')]: gd.point_time = np.concatenate(t, axis=dim)
		except Exception: pass
		return gd
	
	
	def __repr__(self):
		return (self.__class__.__name__+'(\n'+',\n'.join([
			super().__repr__(),
			*[f"{k!s}={v!r}" for (k,v) in self.__dict__.items()]
		])+')')
	
	
	#%% visualisation
	def plot(self, fig=None, axs=None, *, cax=None, show_colorbar=True, plot_points = False):
		# debug: point data if available
		if hasattr(self,'point_pos'):
			point_pos = self.point_pos
			point_vel = self.point_vel
		else: point_pos = None
		
		if self.ndim == 3: self = self[None,None,...]
		if self.ndim == 4: self = self[None,...]
		if self.ndim == 5: pass
		if self.ndim >  5: self = GridData(self.reshape(-1,*self.shape[-4:]))
		
		from matplotlib import pyplot as plt
		from mpl_toolkits.axes_grid1 import ImageGrid
		import numpy as np
		from copy import copy
		from .tools.mpl import circles
		
		from matplotlib.axes import Axes
		
		B,T,_,H,W = self.shape
		
		if fig is None:
			fig = plt.figure(figsize=((T+0.3)*W/4+0.1, B*H/4+0.2), tight_layout=True)
		
		if axs is None:
			axs = ImageGrid(fig, 111, nrows_ncols=(B,T), axes_pad=(0.05,0.10), share_all=True, label_mode='1', cbar_mode='single')
			cax=axs.cbar_axes[0]
		axs = np.array(axs).reshape(B,T)
		
		if cax is None:
			cb_kwargs = dict(ax=axs.flatten())
		else:
			cb_kwargs = dict(cax=cax)
		
		# visualisation params
		# density colormap
		cm = copy(plt.get_cmap('Blues'))
		cm.colorbar_extend = 'max'
		cm.set_over('k')
		cm.set_under('r')
		# velocity
		vel_color = 'g'
		vel_scale = 1  # smaller makes arrows smaller
		
		x,y = np.meshgrid(range(H), range(W), indexing='ij')
		for b in range(B):
			for t in range(T):

				ax: Axes = axs[b][t]
				idx = b * T + t
				label = f"({chr(97 + idx)})"  # chr(97) = 'a'
				ax.text(
					0.5, -0.03, label,
					transform=ax.transAxes,
					ha='center', va='top',
					fontsize=10
				)
				d = self.density[b,t,0].detach().cpu().numpy()
				u, v = self.vel_mean[b,t,:].detach().cpu().numpy()
				s = self.vel_std[b,t,0].detach().cpu().numpy()
				
				# density image
				pd = ax.imshow(d,
					cmap=cm,
					vmin=0, vmax=1,
					# interpolation='hanning',  # default 'nearest'
				)
				ax.autoscale(False)
				
				# velocity mean arrows
				pvm = ax.quiver(y,x, v,u, color=vel_color, angles='xy', scale_units='xy', scale=1/vel_scale)
				
				# velocity variance circles
				pvv = ax.add_collection(circles(y + v*vel_scale, x + u*vel_scale, s*vel_scale, color=vel_color, fc='none'))
				
				# debug: point data if available
				if plot_points:
					if point_pos is not None:
						px, py = (point_pos[t]-0.5).unbind(dim=-1)
						ax.scatter(py, px, 3, 'r')
						pu, pv = point_vel[t].unbind(dim=-1)
						ax.quiver(py,px, pv,pu, color='red', angles='xy', scale_units='xy', scale=1/vel_scale)
				
				ax.set_xticks([])
				ax.set_yticks([])

		if show_colorbar:
			cb = fig.colorbar(pd, **cb_kwargs, label="density")
		return fig
	
	
	def plot_anim(self, period, ret='clip', *, gif=None, video=None, logger=None, robot=None):
		assert 1 <= sum(x is not None for x in (ret, gif, video))
		from moviepy.video.io.bindings import mplfig_to_npimage as mplfig_to_npimage_orig
		from matplotlib import pyplot as plt
		def mplfig_to_npimage(fig, *args, **kwargs):
			""" A version of mplfig_to_npimage that cleans up afterwards. """
			image = mplfig_to_npimage_orig(fig, *args, **kwargs)
			plt.close(fig)
			return image
		
		# plots = list(mplfig_to_npimage(GridData(t.unsqueeze(0)).plot()) for t in self)
		
		def plot_one(i: int):
			frame = GridData(self[i].unsqueeze(dim=0).unsqueeze(dim=0))
			if hasattr(self, 'point_pos'):
				frame.point_pos = [self.point_pos[i]]
				frame.point_vel = [self.point_vel[i]]
			fig = frame.plot()
			if robot:
				ax = fig.get_axes()[0]
				ax.scatter(robot[1][i]-0.5, robot[0][i]-0.5,
					color='orange',
					# edgecolors='white',
				)
			return fig
		
		plots = list(mplfig_to_npimage(plot_one(i)) for i in range(len(self)))
		
		fps = 1/period
		
		if ret == 'clip' or gif or video:
			from moviepy.editor import ImageSequenceClip
			clip = ImageSequenceClip(list(plots), fps=fps)
			if gif:
				clip.write_gif(gif, logger=logger)
			if video:
				clip.write_videofile(video, logger=logger)
		
		if ret == 'clip':
			return clip
		
		if ret == 'tensorboard':
			import numpy as np
			tensor = np.stack(plots)
			iT,iH,iW,iC = range(4)
			tensor = np.expand_dims(tensor.transpose((iT,iC,iH,iW)), axis=0)  # N,T,C,H,W
			return dict(vid_tensor=tensor, fps=fps)
	
	
	def plot_3D(self, period=1, fig=None, ax=None):
		from matplotlib import pyplot as plt
		if fig is None:
			fig = plt.figure()
		if ax is None:
			ax = fig.add_subplot(111, projection='3d')
		# ax.autoscale(False)
		vel_color = 'g'
		vel_scale = 1
		
		# pos = torch.stack(torch.meshgrid(*(torch.arange(s+1) for s in [3,4])), dim=-1)
		import numpy as np
		from .tools.mpl import circles
		T,_,H,W = self.shape
		xx,yy = np.meshgrid(range(H+1), range(W+1), indexing='ij')
		x ,y  = np.meshgrid(range(H), range(W), indexing='ij')
		x = x + 0.5; y = y + 0.5
		
		for t in range(T):
			zi = t*period
			d = self.density[t,0].detach().cpu().numpy()
			u, v = self.vel_mean[t,:].detach().cpu().numpy()
			s = self.vel_std[t,0].detach().cpu().numpy()
			
			# density image
			# pd = ax.imshow(d,
			# 	cmap='Blues',
			# 	vmin=0, vmax=1,
			# 	# interpolation='hanning',  # default 'nearest'
			# 	aspect='auto',
			# 	# zs=z,
			# 	# zdir='z',
			# )
			zz = np.ones_like(xx)*zi
			ax.plot_surface(yy,xx,zz, rstride=1, cstride=1, facecolors=plt.cm.Blues(d), shade=False)
			
			# velocity mean arrows
			z = np.ones_like(x)*zi
			pvm = ax.quiver(y,x,z, v,u,0, color=vel_color, angles='xy', scale_units='xy', scale=1/vel_scale)
			
			# velocity variance circles
			cs = circles(y + v*vel_scale, x + u*vel_scale, s*vel_scale, color=vel_color, fc='none')
			pvv = ax.add_collection3d(zs=[zi]*len(cs), zdir='z')
			
		
		return fig


def rotation_matrix(theta: Real) -> Tensor[2,2]:
	""" Rotation matrix
		Left-multiply column-vectors from local-to-global.
		Right-multiply row-vectors for global-to-local.
	"""
	theta = torch.as_tensor(theta)
	c,s = theta.cos(), theta.sin()
	R = torch.as_tensor([[c,-s], [s, c]])
	return R


class Grid:
	def __init__(self, origin, theta, shape, resolution):
		""" Grid represents the coordinate frame of an image.
			
			:param origin: (m,m), top-left corner
			:param theta: radians, anti-clockwise, of height-axis from x-axis.
			:param shape: (H,W)
			:param resolution: cells/m
		"""
		self.origin = torch.as_tensor(origin)
		self.theta = torch.scalar_tensor(theta)
		self.shape = torch.as_tensor(shape, dtype=torch.int)
		self.resolution = torch.scalar_tensor(resolution)
		assert self.origin.shape == (2,)
		assert self.shape.shape == (2,)
	
	@property
	def rotation(self) -> Tensor[2,2]:
		return rotation_matrix(self.theta)
	
	def plot(self, ax=None, show=None):
		from matplotlib import pyplot as plt
		ax = plt.axes(ax)
		x,y = self.global_grid().unbind(dim=-1)
		ax.scatter(x,y)
		ax.axis('equal')
		if show:
			plt.show()
	
	def rotate_around_centre(self, alpha, inplace=False):
		R_a = rotation_matrix(alpha)
		centre = self.local2global(self.shape/2.)
		origin = (self.origin - centre) @ R_a.T + centre
		theta = self.theta + alpha
		if inplace:
			self.origin = origin
			self.theta = theta
		else:
			return Grid(origin, theta, self.shape, self.resolution)
	rotate_around_centre_ = partialmethod(rotate_around_centre, inplace=True)
	
	
	def local_grid(self) -> Tensor['H','W','D']:
		""" Get the local coordinates of the image pixel centers. """
		return torch.stack(torch.meshgrid(*(torch.arange(s) for s in self.shape), indexing='ij'), dim=-1) + 0.5
	
	def global_grid(self) -> Tensor['H','W','D']:
		""" Get the global coordinates of the image pixel centers. """
		return self.local2global(self.local_grid())
	
	
	PosOrPosAndVel = Union[Tensor[...,'D'], Tuple[Tensor[...,'D'], Tensor[...,'D']]]
	
	def global2local(self, pos: Tensor[...,'D'], vel: Optional[Tensor[...,'D']]=None) -> PosOrPosAndVel:
		""" Global coordinates to local (image) coordinates.
			
			:param pos: position
			:param vel: velocity (optional)
			:return: same as provided input
		"""
		R = self.rotation
		pos = (pos - self.origin) @ R / self.resolution
		if vel is not None:
			vel = vel @ R / self.resolution
			return pos, vel
		return pos
	
	def local2global(self, pos: Tensor[...,'D'], vel: Optional[Tensor[...,'D']]=None) -> PosOrPosAndVel:
		""" Local (image) coordinates to global coordinates.
			
			:param pos: position
			:param vel: velocity (optional)
			:return: same as provided input
		"""
		R = self.rotation.T
		pos = pos * self.resolution @ R + self.origin
		if vel is not None:
			vel = vel * self.resolution @ R
			return pos, vel
		return pos
	
	def points2grid(self,
		pos: Tensor[...,'P','D'],
		vel: Optional[Tensor[...,'P','D']]=None,
		*,
		kernel: Union[str,Callable]=None, kernel_scale: int=1,
		normalise_count: int=1,
		normalise_resolution: Union[False,True,Real]=False,
		normalise_period: Union[False,Real]=False,
		attach_point_data=False,
	):
		""" Convert point data to dense gridded data
			
			:param pos: (m,m)
			:param vel: (m/s,m/s) (optional)
			:param kernel: 'rect', 'tri', or 'hann'
			:param kernel_scale: kernel scale
			:param normalise_count: pos and vel represent observations from count time steps, divide density by count.
			:param normalise_resolution: (meters) normalise spatial resolution to grid cell. True indicates grid resolution.
			:param normalise_period: (seconds) normalise temporal resolution to period.
			:param attach_point_data: (for debugging) save point data in GridData object for plotting purposes.
			:return:
				density. If vel given, returns a GridData with density, vel_mean, and vel_var.
			
			Units:
				- pos: meters
				- vel: meters/second
				- density: 1/spatial^2
				- vel_mean: spatial/temporal
				- vel_var: (spatial/temporal)^2
				normalise_resolution defines `spatial` resolution; True gives grid cell, otherwise meters.
				normalise_period defines `temporal` resolution, otherwise seconds.
		"""
		if kernel is None:
			kernel = self.rect_kernel
		elif isinstance(kernel, str):
			kernel = getattr(self, f'{kernel}_kernel')
		
		if not normalise_period:
			normalise_period = 1
		elif normalise_period is True:
			raise TypeError('period must be provided')
		
		if not normalise_resolution:
			normalise_resolution = 1
		elif normalise_resolution is True:
			normalise_resolution = self.resolution
		
		grid = self.local_grid()
		pos, vel = self.global2local(pos, vel)
		
		# [...,P,H,W,D]
		sH,sW = self.shape
		*sB,sP,sD = pos.shape
		nB = len(sB)
		grid = grid.reshape(*[1]*nB,1,sH,sW,sD)
		pos = pos.reshape(*sB,sP,1,1,sD)
		weights = kernel(pos-grid, scale=kernel_scale)  # [...,P,H,W]
		weights = weights * ((normalise_resolution/self.resolution)**2 / normalise_count)
		iP,iH,iW = range(-3,0)
		density = weights.sum(dim=iP, keepdim=True)  # [...,1,H,W]
		
		if vel is not None:
			# [...,P,D,H,W], NOTE: different from above
			iP,iD,iH,iW = range(-4,0)
			weights = weights.unsqueeze(dim=iD)
			vel = vel.reshape(*sB,sP,sD,1,1)
			vel = vel * (normalise_period * self.resolution / normalise_resolution)
			vel_mean = (weights * vel).sum(dim=iP) / density  # [...,D,H,W]
			
			deviation2 = (vel - vel_mean.unsqueeze(iP)).square().sum(dim=iD, keepdim=True)
			# nnz = weights.count_nonzero(dim=iP)  # not in stable yet
			nnz = (weights != 0).sum(dim=iP)
			bias_correction = nnz / (nnz - 1).float()
			vel_var = bias_correction * (weights * deviation2).sum(dim=iP) / density  # [...,D,H,W]
			
			# [...,D,H,W]
			undefined = (density == 0)
			# todo: use linear interpolation (delaunay tri) where no density
			# todo: could also fill with white noise to train model to ignore these measurements
			vel_mean[undefined.expand_as(vel_mean)] = 0
			vel_var[undefined] = 0
			vel_var[nnz == 1] = 0  # sample variance undefined for one sample
			
			gd = GridData(density=density, vel_mean=vel_mean, vel_var=vel_var)
			if attach_point_data:
				assert not sB  # unimplemented at this stage
				gd.point_pos = np.empty((), dtype=object)
				gd.point_pos[()] = pos.reshape(*sB, sP, sD)
				gd.point_vel = np.empty((), dtype=object)
				gd.point_vel[()] = vel.reshape(*sB, sP, sD)
			return gd
		return density
	
	
	""" Kernels
		:param pos: Tensor[...,D]
		:param scale: Length scale, should be >=1 and integer to maintain density.
			Integral is always one.
		:return: Tensor[...]
	"""
	
	@staticmethod
	def rect_kernel(pos: Tensor, scale=1) -> Tensor:
		mask = ((-scale/2 <= pos) & (pos < scale/2)).all(dim=-1)
		win = mask.to(pos.dtype).div(scale**2)
		return win
	
	@staticmethod
	def tri_kernel(pos: Tensor, scale=1) -> Tensor:
		mask = (pos.abs() < scale).all(dim=-1)
		win = torch.zeros(mask.shape)
		win[mask] = ((1 - abs(pos[mask]/scale))/scale).prod(dim=-1)
		# win[mask] = pos[mask].div(scale).abs().sub(1).neg().div(scale).prod(dim=-1)
		return win
	
	@staticmethod
	def hann_kernel(pos: Tensor, scale=1) -> Tensor:
		mask = (pos.abs() < scale).all(dim=-1)
		win = torch.zeros(mask.shape)
		win[mask] = pos[mask].mul(pi/scale).cos().add(1).div(2*scale).prod(dim=-1)
		return win
	
	@classmethod
	def _check_kernels(cls, kernel_names=('rect','tri','hann')):
		from matplotlib import pyplot as plt
		for scale in (1,2):
			n = 101
			x = torch.linspace(-2*scale, 2*scale, n)
			y = torch.zeros_like(x)
			pos = torch.stack((x,y), dim=-1)  # cross section through zero
			y2 = torch.ones_like(x)*scale/2
			pos2 = torch.stack((x,y2), dim=-1)  # cross section through y=scale/2
			y3 = x.unsqueeze(dim=-1)
			pos3 = torch.stack(torch.broadcast_tensors(x,y3), dim=-1)  # area sampled
			dx = 4*scale/(n-1)
			for kernel_name, color in zip(kernel_names,'rgb'):
				kernel = getattr(cls, f'{kernel_name}_kernel')
				w = kernel(pos, scale)
				plt.plot(x,w, color=color, label=kernel_name)
				w2 = kernel(pos2, scale)
				plt.plot(x,w2, color=color, alpha=0.5)
				w3 = kernel(pos3, scale)
				# integral = w3.sum() * dx**2
				integral = torch.trapz(torch.trapz(w3,dx=dx),dx=dx)
				print(f"{kernel_name}_kernel(scale={scale}) integral = {integral}")
			plt.legend()
			plt.title(f"Scale={scale}")
			plt.show()


def test_check_vel_var():
	g = Grid((0,0),0,(1,3),1)
	gd = g.points2grid(torch.zeros(2,2), torch.as_tensor([[0,1],[0,-1]]).float())
	gd.plot().show()
	print(gd.vel_var)
	torch.var(torch.as_tensor([1.,-1.]))

def test_grid_and_point():
	from .dataset import GridFromH5Dataset
	r = 1
	ds = GridFromH5Dataset('../data/alex1.h5',
	                       Grid((-6,-6),0,(12/r,12/r),r), period=r,
	                       random_rotation=False,
	                       attach_point_data=True,
	                       )  # [T][4,H,W]
	gd = ds[:10]
	fig = gd.plot()
	fig.savefig(f'grid_and_point_{r}.png')
	fig.show()
	a = gd.plot_anim(period=r, gif=f'grid_and_point_{r}.gif')

def test_plot3D():
	import matplotlib as mpl
	mpl.use('tkagg')
	from matplotlib import pyplot as plt
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	vel_color = 'g'
	vel_scale = 1
	from pedpred.grid import Grid
	from pedpred.dataset import GridFromH5Dataset
	r: int = 1  # cells per meter
	grid = Grid((-6,-6), 0, (12*r,12*r), 1/r)
	period = 0.5
	dataset = 'data/alex1'
	step = 2
	data = GridFromH5Dataset(f'{dataset}_valid.h5', grid, period, random_rotation=False, normalise_period=period, attach_point_data=True)
	d = data[0:10]
	self = d
	
	import numpy as np
	T,_,H,W = self.shape
	xx,yy = np.meshgrid(range(H+1), range(W+1), indexing='ij')
	x ,y  = np.meshgrid(range(H), range(W), indexing='ij')
	x = x + 0.5; y = y + 0.5
	
	for t in range(0,T,2):
		zi = t*period
		d = self.density[t,0].detach().cpu() #.numpy()
		u, v = self.vel_mean[t,:].detach().cpu().numpy()
		s = self.vel_std[t,0].detach().cpu().numpy()
		
		z = zi*np.ones_like(x)
		zz = zi*np.ones_like(xx)
		fc = plt.cm.Blues(d.numpy())
		fc[..., -1] = d.clamp(0,1) # change the alpha value
		ax.plot_surface(yy,xx,zz, rstride=1, cstride=1, facecolors=fc, shade=False)
		# ax.quiver(y,x,z,v,u,0, color=vel_color, length=vel_scale, arrow_length_ratio=0.1)
		
		# col = circles(y + v*vel_scale, x + u*vel_scale, s*vel_scale, color=vel_color, fc='none')
		# zs = 0
		# pvv = ax.add_collection3d(col, zs=zs, zdir='z')
	
	return fig


def test_plot_animation():
	from pedpred.grid import Grid, GridData
	from pedpred.dataset import GridFromH5Dataset
	r: int = 1  # cells per meter
	grid = Grid((-6, -6), 0, (12 * r, 12 * r), 1 / r)
	period = 0.5
	dataset = 'data/alex1'
	N = 1372
	data = GridFromH5Dataset(f'{dataset}_valid.h5', grid, 0, random_rotation=False, normalise_period=period, attach_point_data=True)
	gd = GridData(data[:N])
	gd.plot_anim(period=60/N, gif='crowd_0.gif', video='crowd_0.mp4')


def test_pred_anim():
	...


if __name__ == '__main__':
	# import matplotlib as mpl
	# from matplotlib import pyplot as plt
	# mpl.use('tkagg')
	# test_check_vel_var()
	# test_grid_and_point()
	# test_plot3D()
	test_plot_animation()
