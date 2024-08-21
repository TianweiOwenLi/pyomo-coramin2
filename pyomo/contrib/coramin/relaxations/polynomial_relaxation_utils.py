
from typing import Callable, Optional
from copy import deepcopy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from bisect import bisect_left

from pyomo.core.expr.numeric_expr import (
  NumericExpression,
  UnaryFunctionExpression,
  SumExpression,
  ProductExpression
)

import pyomo.environ as pe

_eps = 1.0e-8

_newton_max_iter = 30

Segment = tuple[tuple[float, float], tuple[float, float]]

def _interval(seg: Segment) -> tuple[float, float]:
  (x1,_),(x2,_) = seg
  assert(x1 < x2)
  return (x1,x2)

def dbg_print(msg: str) -> None:
  if DBG:
    print(msg)

DBG = False


class NonDecreasingLineSegmentStack(deque):
  """
  A stack of line segments of nondecreasing slopes. Helps construct convex 
  envelope via a graham-scan-esque method. 
  """

  def __init__(self):
    self.top_slope = float('-inf')
    super().__init__()

  def try_push(self, seg: Segment, center: float) -> bool:
    seg_slope = _slope(seg)
    if seg_slope >= self.top_slope:
      self.top_slope = _slope(seg)
      super().append((seg, center, self.top_slope))
      dbg_print(f"[NDLSS] pushed {seg}")
      return True
    else:
      dbg_print(f"[NDLSS] failed to push {seg}")
      return False

  def pop(self) -> tuple[Segment, float, float]:
    ret = super().pop()
    if self:
      self.top_slope = self[-1][2]
    else:
      self.top_slope = float('-inf')
    dbg_print(f"[NDLSS] poped {ret}")
    return ret

  def append(self, *args, **kwds):
    raise NotImplementedError('[NDLSS] Please use legal methods')

  def appendleft(self, *args, **kwds):
    raise NotImplementedError('[NDLSS] Please use legal methods')

  def popleft(self, *args, **kwds):
    raise NotImplementedError('[NDLSS] Please use legal methods')

  def export(self) -> list[Segment]:
    return [s for (s, _, _) in self]


def richardson(f: Callable[[float], float], x: float) -> float:
  """
  Numerical differentiation using Richardson Extrapolation.
  """
  n = 6
  h, d = 0.01, np.zeros((n+1, n+1))
  for i in range(n+1):
    d[i,0] = (f(x+h)-f(x-h))/(2*h)
    for j in range(1, i+1):
      d[i,j] = d[i, j-1] + (d[i,j-1]-d[i-1,j-1])/(4**j-1)
    h /= 2.0
  return d[-1,-1]


def diff(f: Callable[[float], float], x: float) -> float:
  """
  Numerical differentiation. 

  TODO Use Richardson Extrapolation for Unknown Property Function
  """
  if hasattr(f, 'diff'):
    return f.diff(x)
  return richardson(f, x)


def tangent_transformation(
  f: Callable[[float], float], 
  r: float
) -> Callable[[float], float]:
  """
  Tangent Transformation: f, r -> T_r f . See paper.
  """

  # staging
  fr = f(r)
  return lambda x : diff(f, x) - (f(x)-fr)/(x-r)
  

# TODO implement Laguerre's Method


def newton_method(f: Callable[[float], float], x: float) -> float:
  x_var, iter = x, 0
  while abs(f(x_var)) > _eps:
    x_var -= f(x_var)/diff(f, x_var)
    iter += 1
    if iter == _newton_max_iter: # only print once
      print("WARNING: Newton's Method Max Iter Reached")
  return x_var


## IDEA: graham scan

def compute_tangent(f, x1, x2):
  for _ in range(30):
    dbg_print(f'[compute_tangent] {x1},{x2}')
    x3 = newton_method(tangent_transformation(f, x2),x1)
    x1,x2=x2,x3
  return (x1,x2)


def compute_tangent_fast(f, ddf, x1, x2) -> Optional[tuple[float, float]]:
  """
  Fast version of tangent computation "through" two local mins. See paper.
  """
  for _ in range(16):
    # dbg_print(f'[compute_tangent_fast] {x1},{x2}')
    Tf = tangent_transformation(f, x2)
    Tf_diff_x1 = ddf(x1) - (-diff(f, x1)*(x2-x1) + (f(x2)-f(x1)))/((x2-x1)**2)
    x3 = x1 - Tf(x1) / Tf_diff_x1
    x1,x2 = x2,x3

  if abs(x1-x2) > 1e-4:
    return (x1,x2)
  return None


def compute_secant_tangent(f, b, x):
  soln = newton_method(tangent_transformation(f, b),x)
  if abs(soln - b) > 1e-4:
    return (min(soln, b), max(soln, b))
  print(f"[compute_sec_tan] WARNING: Newton did not converge! {abs(soln-b)}")
  return None


def _compute_closest_secant_tangent(f, b, centers: list[float]):
  assert(centers)
  _centers = sorted(centers) # TODO remove if known to be sorted

  Tbf = tangent_transformation(f,b)
  if b <= _centers[0]:
    pass
  elif b >= _centers[-1]:
    _centers.reverse()
  else:
    raise ValueError('Centers must lie on either side of tangent transf point')
  for c in centers:
    soln = newton_method(Tbf, c)
    if abs(soln - b) > 1e-4:
      return soln
    
  return None


# def wrong_find_all_roots(f, lb, ub) -> list[float]:
#   """
#   Wrong algorithm for finding roots of f between lb and ub. 

#   TODO replace it with something correct. 
#   """
#   eps = (ub - lb) / 20.0
#   pts = np.arange(lb + (eps/2), ub - (eps/2), eps)
#   roots = []
#   for pt in pts:
#     root = newton_method(f, pt)
#     if lb <= root <= ub:
#       roots.append(root)
  
#   roots.sort()

#   # filter out duplicates
#   new_roots, top = [], float('-inf')
#   for root in roots:
#     if root - top > 1e-4:
#       new_roots.append(root)
#       top = root
  
#   return new_roots


def find_all_real_roots_via_power_coeffs(f, lb, ub) -> list[float]:
  """
  Finds all real roots of a polynomial `f` between `lb` and `ub`.  
  
  ### Args: 
    `f`: a polynomial (python3 `Callable` object with attr `power_coeffs`)

  ### Returns: 
    the list of real roots, sorted from small to large, of `f` between `lb`
    and `ub`. 

  """
  power_coeffs = list(f.power_coeffs)
  power_coeffs.reverse()  # numpy.roots args uses different ordering than ours

  rs = np.roots(power_coeffs)
  rs = [r for r in rs if r.imag == 0.0 and lb <= r.real <= ub]
  rs.sort()
  return rs


def _interleave(l):
  return zip(l, l[1:])


def compute_positive_segments(f, lb, ub) -> list[tuple[float, float]]:
  """
  Finds all intervals of f between lb, ub where f is positive.
  """
  roots = find_all_real_roots_via_power_coeffs(f, lb, ub)

  # no roots, either all or nothing.
  if not roots:
    if f((lb+ub)/2) > 0:
      return [(lb, ub)]
    return []

  positive_segments = []

  if roots[0] - lb > 1e-4 and f(lb) >= 0:
    positive_segments.append((lb, roots[0]))

  for (r1, r2) in _interleave(roots):
    if f((r1+r2)/2) > 0:
      positive_segments.append((r1,r2))

  if ub - roots[-1] > 1e-4 and f(ub) >= 0:
    positive_segments.append((roots[-1], ub))

  return positive_segments


def _center(tup: tuple[float, float]) -> float:
  return (tup[0] + tup[1]) / 2


def _segment(f, iv: tuple[float, float]) -> Segment:
  if iv == None:
    return None
  a,b = iv
  assert(a != b)
  
  if a > b:
    a,b = b,a
  return ((a, f(a)), (b, f(b)))

def _slope(seg: Segment):
  # if seg == None:
  #   return None
  ((x1,y1),(x2,y2)) = seg
  assert(x1 != x2)
  
  if abs(x2-x1) < _eps:
    dbg_print("WARNING: slope computation may be inaccurate")

  return (y2-y1) / (x2-x1)


def cvx_tangent_segment(f, ddf, lb: float, ub: float) -> list[Segment]:
  """
  Naively compute the line segments for convexifying f
  """

  def _constrained_graph_tangent(a, b):
    """
    Computes some 2-point tangent line to the graph of f between (a,b)
    """
    dbg_print(f'[constrained graph tangent]: {a}, {b}')
    assert(a != b)
    if a > b:
      a,b=b,a

    if (a,b) == (lb,ub):
      return (lb,ub)
    if a == lb:
      return compute_secant_tangent(f, lb, b)
    if b == ub:
      return compute_secant_tangent(f, ub, a)
    return compute_tangent_fast(f, ddf, a, b)


  # we first need to identify parts of the functions that are convex, since 
  # the straight line segments of 1D convex envelope would typically be between 
  # convex segments of the function. 
  cvx_segments = compute_positive_segments(ddf, lb, ub)
  dbg_print(f'segs: {cvx_segments}')
  cvx_centers = [_center(seg) for seg in cvx_segments]

  if not cvx_centers: # no convex centers, secant line is convex envelope. 
    return [_segment(f, (lb, ub))]
  
  # if the leftmost convex segment does not include lowerbound lb, then the 
  # function is concave between lb and the lower bound of first convex segment; 
  # in this case, we add lb to list of convex centers, because the function's 
  # convex envelope will contain some line segment that passes through lb. 
  # Same story holds for rightmost convex segment and ub. 
  if cvx_segments[0][0] != lb:
    cvx_centers = [lb] + cvx_centers
  if cvx_segments[-1][-1] != ub:
    cvx_centers = cvx_centers + [ub]
  dbg_print(f'centers: {cvx_centers}')
  
  i, j, n = 0, 1, len(cvx_centers)
  line_segments = NonDecreasingLineSegmentStack()

  while j < n:
    dbg_print(f'[loop]: {i}, {j}')
    tangent_seg = _segment(
      f, 
      _constrained_graph_tangent(cvx_centers[i], cvx_centers[j])
    )
    while tangent_seg and not line_segments.try_push(tangent_seg, cvx_centers[i]):
      _ = line_segments.pop()

      # if failed to push, then we encountered a local issue where two adjacent 
      # graph tangent line segments do not form a convex under-estimation of 
      # the polynomial; in this case, we drop the left tangent line (ie. stack 
      # top), and decrease the left convex-center index of right tangent line. 
      # 
      # Again, think about how Graham Scan works. 
      i -= 1 
      tangent_seg = _segment(
        f, 
        _constrained_graph_tangent(cvx_centers[i], cvx_centers[j])
      )
    i += 1
    j += 1

  return line_segments.export()


def convex_envelope(f, ddf, lb, ub) -> Callable[[float], float]:
  segs = cvx_tangent_segment(f,ddf,lb,ub)
  segs.sort(key=lambda seg : seg[0][0])

  ivx = [(x1,x2) for ((x1,_),(x2,_)) in segs] # requires to be disjoint.
  ivy = [(y1,y2) for ((_,y1),(_,y2)) in segs]

  xs = []
  for iv in ivx:
    xs += [iv[0], iv[1]]

  ys = []
  for iv in ivy:
    ys += [iv[0], iv[1]]

  def g(x):
    assert(lb <= x <= ub)
    idx = bisect_left(xs, x)
    if idx % 2: # in tangent line segment interval
      x0, x1, y0, y1 = xs[idx-1], xs[idx], ys[idx-1], ys[idx]
      return y0 + (y1-y0) * (x - x0) / (x1 - x0)
    return f(x)
  
  return g


def _construct_polynomial(power_coeffs):
  """
  Given a list of coefficients from 0th power and up, returns polynomial fn. 

  eg. [1,2,3] => 3x^2 + 2x + 1
  """
  def f(x):
    ret = 0
    for c in power_coeffs[::-1]:
      ret *= x
      ret += c
    return ret
  
  f.power_coeffs = power_coeffs

  return f


def _df(power_coeff):
  """
  Given a list of coefficients from 0th power and up, returns the coeffs 
  of its derivatives. 

  eg. [1,2,3] => [2,6,0]
  """
  return np.hstack((
    (
      power_coeff * np.array(range(len(power_coeff)))
    )[1:], 
    np.array([0])
  ))


def construct_polynomial_double_deriv_pair(power_coeff):
  """
  Given a list of coefficients from 0th power and up, returns polynomial fn, 
  and the fn of its second derivative.

  eg. [1,2,3] => (3x^2 + 2x + 1, 6)
  """
  return (
    _construct_polynomial(power_coeff), 
    _construct_polynomial(_df(_df(power_coeff)))
  )


def semiauto_test(dim = 6, T = 20, seed = 10725):
  np.random.seed(seed)
  for _ in range(T):
    coeffs = np.random.rand(dim)
    print(coeffs)
    f, ddf = construct_polynomial_double_deriv_pair(coeffs)
    lb, ub = -1, 1 # up to scaling, this is fine

    step = (ub - lb) / 1000
    xs = np.arange(lb + step / 2, ub - step / 2, step)
    plt.plot(xs, np.vectorize(f)(xs))

    g = convex_envelope(f, ddf, lb, ub)
    plt.plot(xs, np.vectorize(g)(xs))
    plt.show()


### Experimental Feature: converts this into pyomo expr


def add_basis_to_model(m: pe.ConcreteModel, B: np.ndarray):
  """
  B is a basis where each column is an element. 

  Returns a function `mk_polynomial`, which converts 
  """

  dim1, dim2 = B.shape
  assert(dim1 == dim2)

  m.basis = deepcopy(B)
  m.basis_fns = [_construct_polynomial(B[:,idx]) for idx in range(dim1)]

  def mk_monomial(var, basis_elt_idx: int) -> UnaryFunctionExpression:
    return UnaryFunctionExpression(
      (var,), 
      f"b_{basis_elt_idx}", 
      m.basis_fns[basis_elt_idx]
    )
  
  def mk_polynomial(
    var : NumericExpression, 
    basis_coeffs: np.ndarray
  ) -> SumExpression:
    return SumExpression([
      c * mk_monomial(var, i) for (i,c) in enumerate(basis_coeffs)
    ])
  
  m.mk_monomial = mk_monomial
  m.mk_polynomial = mk_polynomial
  
  return mk_polynomial

