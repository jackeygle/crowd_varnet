from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Optional, Tuple, Union

from math import prod

import numpy as np
import torch


@contextmanager
def cuda_context(cuda=None):
	if cuda is None:
		cuda = torch.cuda.is_available()
	old_tensor_type = torch.cuda.FloatTensor if torch.tensor(0).is_cuda else torch.FloatTensor
	old_generator = torch.default_generator
	torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
	torch.default_generator = torch.Generator('cuda' if cuda else 'cpu')
	yield
	torch.set_default_tensor_type(old_tensor_type)
	torch.default_generator = old_generator


class ShapedTensorMeta(type(torch.Tensor)):
	""" Shaped Tensor (class) factory (metaclass).
		
		Mainly for shape hinting:
		>>> Tensor = ShapedTensorMeta.make_shaped_type()
		>>> def my_funk(x: Tensor['H','W'], y: Tensor[...,2]) -> Tensor[:,'H','W']: pass
		Strings represent named dimensions.
		Integers represent dimension size (length).
		Ellipsis (...) can be used (once per call) to represent zero or more unnamed and unsized dimentions.
		Empty slices (:) or None represent unnamed and unsized dimentions.
		Slices with start and stop values represent both named and sized dimentions.
		>>> Tensor['B':100,...,'D':2]
	"""
	def __new__(mcs, name, bases, dct, *args, **kwargs):
		return super().__new__(mcs, name, bases, dct)
	
	def __init__(cls, name, bases, dct, names=(...,), sizes=(...,)):
		super().__init__(name, bases, dct)
		cls.__names = names
		cls.__sizes = sizes
	
	def __instancecheck__(cls, instance):
		# names check
		try:
			instance.refine_names(cls.__names)
		except RuntimeError:
			return False
		# sizes check
		if not all (s is None or s is ... for s in cls.__sizes):
			from torch._namedtensor_internals import resolve_ellipsis
			sizes = resolve_ellipsis(cls.__sizes, instance.shape, 'ShapedTensorMeta.__instancecheck__')
			# if len(sizes) != len(cls.__sizes):
			# 	return False
			# they must be the same size if the name check passed.
			for i,(s,ts) in enumerate(zip(sizes, cls.__sizes)):
				if s is not None:
					if s != ts:
						return False
		return True
	
	DimSpec = Union[slice,str,int,type(...),None]
	
	@classmethod  # meta-class method
	def repr_dimspec(mcs, name: Optional[str], size: Optional[int]) -> str:
		if name is ...:  # and size is ...:
			return '...'
		has_name, has_size = (x is not None for x in (name, size))
		has_even = not (has_name ^ has_size)
		repr_str = ''
		if has_name: repr_str += repr(name)
		if has_even: repr_str += ':'
		if has_size: repr_str += str(size)
		return repr_str
	
	@classmethod  # meta-class method
	def repr_shape(mcs, names: Tuple[str], sizes: Tuple[int]) -> str:
		reprs = (mcs.repr_dimspec(name, size) for name, size in zip(names, sizes))
		return f"[{','.join(reprs)}]"
	
	@property
	def shape_repr(cls) -> str:
		return cls.repr_shape(cls.__names, cls.__sizes)
	
	@classmethod  # meta-class method
	def parse_dimspec(mcs, dimspec: DimSpec) -> Tuple[Optional[str], Optional[int]]:
		error = f"Invalid dimspec {dimspec!r}."
		if isinstance(dimspec, slice):
			assert dimspec.step is None, error
			name, size = dimspec.start, dimspec.stop
			return name, size
		elif isinstance(dimspec, str)   : return mcs.parse_dimspec(slice(dimspec,None))
		elif isinstance(dimspec, int)   : return mcs.parse_dimspec(slice(None   ,dimspec))
		elif dimspec is ...             : return mcs.parse_dimspec(slice(...    ,...))
		elif dimspec is None            : return mcs.parse_dimspec(slice(None   ,None))
		else:
			raise RuntimeError(error)
	
	def __getitem__(cls, item: Tuple[DimSpec,...]):
		if not isinstance(item, Tuple):
			item = (item,)
		names, sizes = (tuple(x) for x in zip(*(cls.parse_dimspec(dimspec) for dimspec in item)))
		return cls.make_shaped_type(cls.__bases__[0], names, sizes)
	
	@classmethod  # meta-class method
	@lru_cache(maxsize=None)
	def make_shaped_type(mcs, tensor_cls, names=(...,), sizes=(...,)):
		shape_repr = mcs.repr_shape(names, sizes)
		return ShapedTensorMeta(tensor_cls.__name__+shape_repr, (tensor_cls,), {}, names, sizes)


ShapedTensor = ShapedTensorMeta.make_shaped_type(torch.Tensor)

## Sparse tensor things

def ravel_idx(indices, shape):
	return torch.as_tensor(np.ravel_multi_index(tuple(indices), shape))

def unravel_idx(indices, shape):
	return torch.stack(tuple(torch.as_tensor(x) for x in np.unravel_index(indices, shape)))

def sparse_eye(n, rank=2, **kwargs):
	i = torch.arange(n)
	i = torch.stack([i]*rank)
	v = torch.ones(n)
	return torch.sparse_coo_tensor(i, v, **kwargs)

def sparse_kron(A, B, **kwargs):
	A = A.to_sparse_coo().coalesce()
	B = B.to_sparse_coo().coalesce()
	A_shape = torch.as_tensor(A.shape)[:,None,None]
	B_shape = torch.as_tensor(B.shape)[:,None,None]
	A_indices = A.indices()[:,:,None]
	B_indices = B.indices()[:,None,:]
	A_values = A.values()[:,None]
	B_values = B.values()[None,:]
	C_shape = A_shape * B_shape
	C_indices = A_indices * B_shape + B_indices
	C_values = A_values * B_values
	C = torch.sparse_coo_tensor(C_indices.flatten(1), C_values.flatten(), C_shape.flatten().tolist(), **kwargs)
	return C


def gen_sat2img_mat(*img_shape):
	""" Generate a Summed-Area-Table (integral-image) to image transformation matrix.
		If img is an image of size sz, and sat is the corresponding integral:
		>>> img.flatten() == M @ sat.flatten()
	"""
	D = len(img_shape)
	sat_shape = tuple(d+1 for d in img_shape)
	p = torch.stack(torch.meshgrid(*D*[torch.arange(2)], indexing='ij')).reshape(D, 1, 2**D)  # [D, 1, 2**D]
	img_idx = torch.stack(torch.meshgrid(*(torch.arange(s) for s in img_shape), indexing='ij')).reshape(D, -1, 1)  # [D, prod(img_shape), 1]
	sat_idx = img_idx + p  # [D, prod(img_shape), 2**D]
	indices = (img_idx, sat_idx)
	indices = tuple(ravel_idx(*args) for args in zip(indices, (img_shape, sat_shape)))
	indices = torch.broadcast_tensors(*indices)
	indices = torch.stack(indices)  # [2, prod(img_shape), 2**D]
	values = (-1)**(D-p.sum(0))  # [1, 2**D]
	values = values.expand(prod(img_shape), 2**D)
	M = torch.sparse_coo_tensor(indices.flatten(1), values.flatten())
	assert M.shape == (prod(img_shape), prod(sat_shape))
	return M


def tup_to_class(f):
	@wraps(f)
	def wrapper(self, *args, **kwargs):
		ret = f(self, *args, **kwargs)
		return self.__class__(*ret)
	return wrapper


class ValGrad:
	""" Basically just a (value, gradient) tuple with a bunch of methods."""
	def __init__(self, value, gradient):
		self.v = value
		self.g = gradient
	
	@tup_to_class
	def neg(self): return -self.v, -self.g
	@tup_to_class
	def add(self, othr): return self.v + othr.v, self.g + othr.g
	@tup_to_class
	def sub(self, othr): return self.v - othr.v, self.g - othr.g
	@tup_to_class
	def mul(self, othr): return self.v * othr.v, self.g * othr.v + self.v * othr.g
	@tup_to_class
	def div(self, othr): return self.v / othr.v, (self.g * othr.v - self.v * othr.g) / othr.v**2
	@tup_to_class
	def pow(self, r): return self.v**r, r * self.g * self.v**(r-1)
	
	@tup_to_class
	def square(self): return self.v**2, 2 * self.g * self.v
	@tup_to_class
	def sum(self, dim=None, keepdim=False): return (x.sum(dim=dim, keepdim=keepdim) for x in self)
	@tup_to_class
	def mean(self, dim=None, keepdim=False): return (x.mean(dim=dim, keepdim=keepdim) for x in self)
	@tup_to_class
	def where(self, condition, othr): return self.v.where(condition, othr.v), self.g.where(condition, othr.g)
	
	def squeeze_(self, dim=None): [x.squeeze_(dim=dim) for x in self]
	
	def __iter__(self):
		yield self.v
		yield self.g
	
	__neg__ = neg
	__add__ = add
	__sub__ = sub
	__mul__ = mul
	__truediv__ = div
	__pow__ = pow


def interp(dy, qx, dx=None, dim=-1, kind='linear', out=None):
	order = dict(previous=0, linear=1)[kind]
	if dx is None:
		qi = qx.floor()
		qi = qi.clamp(min=0, max=dy.shape[dim]-order-1)  # implicit extrapolation
		qt = qx - qi
		qt = qt[:,None,None]  # SUPER shitty :'(
		qi = qi.long()
	else:
		raise NotImplementedError("Input dx not implemented.")
	
	if   kind == 'previous':
		qy = dy.index_select(dim=dim, index=qi)
	elif kind == 'linear':
		qy_l = dy.index_select(dim=dim, index=qi)
		qy_u = dy.index_select(dim=dim, index=qi+1)
		qy = torch.lerp(qy_l, qy_u, qt, out=out)
	else:
		raise NotImplementedError(f"Interpolation {kind=} not implemented.")
	
	return qy

torch.interp = interp
torch.Tensor.interp = interp
