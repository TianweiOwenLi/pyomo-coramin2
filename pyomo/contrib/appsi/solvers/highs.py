from collections.abc import Iterable
import enum
import logging
import math
from typing import List, Dict, Optional

from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigValue
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import Var, _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import (
    value, is_constant, is_fixed, native_numeric_types,
)
from pyomo.repn import generate_standard_repn
from pyomo.core.base.set import (Reals, NonNegativeReals, NonPositiveReals,
                                 Integers, NonNegativeIntegers, NonPositiveIntegers,
                                 Binary, PercentFraction, UnitInterval)
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
    PersistentSolver, Results, TerminationCondition, MIPSolverConfig,
    PersistentBase, PersistentSolutionLoader
)
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.appsi.highs_bindings import pyhighs, pyhighs_available
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.core.expr.visitor import replace_expressions

logger = logging.getLogger(__name__)


class DegreeError(PyomoException):
    pass


class HighsConfig(MIPSolverConfig):
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super(HighsConfig, self).__init__(description=description,
                                          doc=doc,
                                          implicit=implicit,
                                          implicit_domain=implicit_domain,
                                          visibility=visibility)

        self.logfile: str = self.declare('logfile', ConfigValue(domain=str, default=''))


class HighsResults(Results):
    def __init__(self, solver):
        super(HighsResults, self).__init__()
        self.wallclock_time = None
        self.solution_loader = PersistentSolutionLoader(solver=solver)


class _MutableVarBounds(object):
    def __init__(self, lower_expr, upper_expr, pyomo_var_id, var_map, highs):
        self.pyomo_var_id = pyomo_var_id
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.var_map = var_map
        self.highs = highs

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeColBounds(col_ndx, lb, ub)


class _MutableLinearCoefficient(object):
    def __init__(self, pyomo_con, pyomo_var_id, con_map, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.pyomo_con = pyomo_con
        self.con_map = con_map
        self.var_map = var_map

    def update(self):
        row_ndx = self.con_map[self.pyomo_con]
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeCoeff(row_ndx, col_ndx, value(self.expr))


class _MutableObjectiveCoefficient(object):
    def __init__(self, pyomo_var_id, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.var_map = var_map

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeColCost(col_ndx, value(self.expr))


class _MutableObjectiveOffset(object):
    def __init__(self, expr, highs):
        self.expr = expr
        self.highs = highs

    def update(self):
        self.highs.changeObjectiveOffset(value(self.expr))


class _MutableConstraintBounds(object):
    def __init__(self, lower_expr, upper_expr, pyomo_con, con_map, highs):
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.con = pyomo_con
        self.con_map = con_map
        self.highs = highs

    def update(self):
        row_ndx = self.con_map[self.con]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeRowBounds(row_ndx, lb, ub)


class Highs(PersistentBase, PersistentSolver):
    """
    Interface to HiGHS
    """
    _available = None

    def __init__(self):
        super(Highs, self).__init__()
        self._config = HighsConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._mutable_helpers = dict()
        self._mutable_bounds = dict()
        self._objective_helpers = list()
        self._last_results_object: Optional[HighsResults] = None
        self._domain_to_vtype_map = dict()
        self._sol = None
        if pyhighs_available:
            self._domain_to_vtype_map[id(Reals)] = (-pyhighs.kHighsInf, pyhighs.kHighsInf, pyhighs.HighsVarType.kContinuous)
            self._domain_to_vtype_map[id(NonNegativeReals)] = (0, pyhighs.kHighsInf, pyhighs.HighsVarType.kContinuous)
            self._domain_to_vtype_map[id(NonPositiveReals)] = (-pyhighs.kHighsInf, 0, pyhighs.HighsVarType.kContinuous)
            self._domain_to_vtype_map[id(Integers)] = (-pyhighs.kHighsInf, pyhighs.kHighsInf, pyhighs.HighsVarType.kInteger)
            self._domain_to_vtype_map[id(NonNegativeIntegers)] = (0, pyhighs.kHighsInf, pyhighs.HighsVarType.kInteger)
            self._domain_to_vtype_map[id(NonPositiveIntegers)] = (-pyhighs.kHighsInf, 0, pyhighs.HighsVarType.kInteger)
            self._domain_to_vtype_map[id(Binary)] = (0, 1, pyhighs.HighsVarType.kInteger)
            self._domain_to_vtype_map[id(PercentFraction)] = (0, 1, pyhighs.HighsVarType.kContinuous)
            self._domain_to_vtype_map[id(UnitInterval)] = (0, 1, pyhighs.HighsVarType.kContinuous)

    def available(self):
        if pyhighs_available:
            return self.Availability.FullLicense
        elif cmodel_available:
            from ctypes.util import find_library
            if find_library('highs') is None:
                return self.Availability.NotFound
            else:
                return self.Availability.NeedsCompiledExtension
        else:
            return self.Availability.NeedsCompiledExtension

    def version(self):
        version = (pyhighs.HIGHS_VERSION_MAJOR,
                   pyhighs.HIGHS_VERSION_MINOR,
                   pyhighs.HIGHS_VERSION_PATCH)
        return version

    @property
    def config(self) -> HighsConfig:
        return self._config

    @config.setter
    def config(self, val: HighsConfig):
        self._config = val

    @property
    def highs_options(self):
        """
        Returns
        -------
        highs_options: dict
            A dictionary mapping solver options to values for those options. These
            are solver specific.
        """
        return self._solver_options

    @highs_options.setter
    def highs_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        raise NotImplementedError('Highs does not use a symbol map')

    def _solve(self, timer: HierarchicalTimer):
        config = self.config
        options = self.highs_options
        if config.stream_solver:
            self._solver_model.setOptionValue('log_to_console', True)
        else:
            self._solver_model.setOptionValue('log_to_console', False)
        self._solver_model.setOptionValue('log_file', config.logfile)

        if config.time_limit is not None:
            self._solver_model.setOptionValue('time_limit', config.time_limit)
        if config.mip_gap is not None:
            raise NotImplementedError('Cannot set mip gap.')

        for key, option in options.items():
            self._solver_model.setOptionValue(key, option)
        timer.start('optimize')
        self._solver_model.run()
        timer.stop('optimize')
        return self._postsolve(timer)

    def solve(self, model, timer: HierarchicalTimer = None) -> Results:
        avail = self.available()
        if not avail:
            raise PyomoException(f'Solver {self.__class__} is not available ({avail}).')
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if timer is None:
            timer = HierarchicalTimer()
        if model is not self._model:
            timer.start('set_instance')
            self.set_instance(model)
            timer.stop('set_instance')
        else:
            timer.start('update')
            self.update(timer=timer)
            timer.stop('update')
        res = self._solve(timer)
        self._last_results_object = res
        if self.config.report_timing:
            logger.info('\n' + str(timer))
        return res

    def _add_variables(self, variables: List[_GeneralVarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        lbs = list()
        ubs = list()
        indices = list()
        vtypes = list()

        current_num_vars = len(self._pyomo_var_to_solver_var_map)
        for v in variables:
            v_id = id(v)
            _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
            lb, ub, vtype = self._domain_to_vtype_map[id(_domain)]
            if _fixed:
                lb = _value
                ub = _value
            else:
                if _lb is not None or _ub is not None:
                    if not is_constant(_lb) or not is_constant(_ub):
                        mutable_bound = _MutableVarBounds(lower_expr=NPV_MaxExpression((_lb, lb)),
                                                          upper_expr=NPV_MinExpression((_ub, ub)),
                                                          pyomo_var_id=v_id,
                                                          var_map=self._pyomo_var_to_solver_var_map,
                                                          highs=self._solver_model)
                        self._mutable_bounds[v_id] = (v, mutable_bound)
                if _lb is not None:
                    lb = max(value(_lb), lb)
                if _ub is not None:
                    ub = min(value(_ub), ub)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(current_num_vars)
            self._pyomo_var_to_solver_var_map[v_id] = current_num_vars
            current_num_vars += 1

        pyhighs.highs_addVars(self._solver_model, len(lbs),
                              np.array(lbs, dtype=np.double),
                              np.array(ubs, dtype=np.double))
        pyhighs.highs_changeColIntegrality(self._solver_model,
                                           len(vtypes),
                                           np.array(indices),
                                           np.array(vtypes))

    def _add_params(self, params: List[_ParamData]):
        pass

    def set_instance(self, model):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            raise ImportError(f'Solver not available; Availability: {self.available()}')
        saved_config = self.config
        saved_options = self.highs_options
        saved_update_config = self.update_config
        self.__init__()
        self.config = saved_config
        self.highs_options = saved_options
        self.update_config = saved_update_config
        self._model = model

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        current_num_cons = len(self._pyomo_con_to_solver_con_map)
        lbs = list()
        ubs = list()
        starts = list()
        var_indices = list()
        coef_values = list()

        for con in cons:
            repn = generate_standard_repn(con.body, quadratic=False, compute_values=False)
            if repn.nonlinear_expr is not None:
                raise DegreeError(f'Highs interface does not support expressions of degree {repn.polynomial_degree()}')

            starts.append(len(coef_values))
            for ndx, coef in enumerate(repn.linear_coefs):
                v = repn.linear_vars[ndx]
                v_id = id(v)
                if not is_constant(coef):
                    mutable_linear_coefficient = _MutableLinearCoefficient(pyomo_con=con, pyomo_var_id=v_id,
                                                                           con_map=self._pyomo_con_to_solver_con_map,
                                                                           var_map=self._pyomo_var_to_solver_var_map,
                                                                           expr=coef,
                                                                           highs=self._solver_model)
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = list()
                    self._mutable_helpers[con].append(mutable_linear_coefficient)
                var_indices.append(self._pyomo_var_to_solver_var_map[v_id])
                coef_values.append(value(coef))

            if con.has_lb():
                lb = con.lower - repn.constant
            else:
                lb = -pyhighs.kHighsInf
            if con.has_ub():
                ub = con.upper - repn.constant
            else:
                ub = pyhighs.kHighsInf

            if not is_constant(lb) or not is_constant(ub):
                mutable_con_bounds = _MutableConstraintBounds(lower_expr=lb,
                                                              upper_expr=ub,
                                                              pyomo_con=con,
                                                              con_map=self._pyomo_con_to_solver_con_map,
                                                              highs=self._solver_model)
                if con not in self._mutable_helpers:
                    self._mutable_helpers[con] = [mutable_con_bounds]
                else:
                    self._mutable_helpers[con].append(mutable_con_bounds)

            lbs.append(value(lb))
            ubs.append(value(ub))
            self._pyomo_con_to_solver_con_map[con] = current_num_cons
            self._solver_con_to_pyomo_con_map[current_num_cons] = con
            current_num_cons += 1

        pyhighs.highs_addRows(self._solver_model,
                              np.array(lbs, dtype=np.double),
                              np.array(ubs, dtype=np.double),
                              len(coef_values),
                              np.array(starts),
                              np.array(var_indices),
                              np.array(coef_values, dtype=np.double))

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        raise NotImplementedError('Highs interface does not support SOS constraints')

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = list()
        for con in cons:
            con_ndx = self._pyomo_con_to_solver_con_map.pop(con)
            del self._solver_con_to_pyomo_con_map[con_ndx]
            indices_to_remove.append(con_ndx)
            self._mutable_helpers.pop(con, None)
        pyhighs.highs_deleteRows(self._solver_model, len(indices_to_remove), np.array(indices_to_remove))
        con_ndx = 0
        new_con_map = dict()
        for c in self._pyomo_con_to_solver_con_map.keys():
            new_con_map[c] = con_ndx
            con_ndx += 1
        self._pyomo_con_to_solver_con_map = new_con_map
        self._solver_con_to_pyomo_con_map = {v: k for k, v in self._pyomo_con_to_solver_con_map.items()}

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        raise NotImplementedError('Highs interface does not support SOS constraints')

    def _remove_variables(self, variables: List[_GeneralVarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = list()
        for v in variables:
            v_id = id(v)
            v_ndx = self._pyomo_var_to_solver_var_map.pop(v_id)
            indices_to_remove.append(v_ndx)
            self._mutable_bounds.pop(v_id, None)
        pyhighs.highs_deleteVars(self._solver_model, len(indices_to_remove), np.array(indices_to_remove))
        v_ndx = 0
        new_var_map = dict()
        for v_id in self._pyomo_var_to_solver_var_map.keys():
            new_var_map[v_id] = v_ndx
            v_ndx += 1
        self._pyomo_var_to_solver_var_map = new_var_map

    def _remove_params(self, params: List[_ParamData]):
        pass

    def _update_variables(self, variables: List[_GeneralVarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices = list()
        lbs = list()
        ubs = list()
        vtypes = list()

        for v in variables:
            v_id = id(v)
            self._mutable_bounds.pop(v_id, None)
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            _v, _lb, _ub, _fixed, _domain, _value = self._vars[v_id]
            lb, ub, vtype = self._domain_to_vtype_map[id(_domain)]
            if _fixed:
                lb = _value
                ub = _value
            else:
                if _lb is not None or _ub is not None:
                    if not is_constant(_lb) or not is_constant(_ub):
                        mutable_bound = _MutableVarBounds(lower_expr=NPV_MaxExpression((_lb, lb)),
                                                          upper_expr=NPV_MinExpression((_ub, ub)),
                                                          pyomo_var_id=v_id,
                                                          var_map=self._pyomo_var_to_solver_var_map,
                                                          highs=self._solver_model)
                        self._mutable_bounds[v_id] = (v, mutable_bound)
                if _lb is not None:
                    lb = max(value(_lb), lb)
                if _ub is not None:
                    ub = min(value(_ub), ub)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(v_ndx)

        pyhighs.highs_changeColsBounds(self._solver_model,
                                       len(indices),
                                       np.array(indices),
                                       np.array(lbs, dtype=np.double),
                                       np.array(ubs, dtype=np.double))
        pyhighs.highs_changeColsIntegrality(self._solver_model,
                                            len(indices),
                                            np.array(indices),
                                            np.array(vtypes))

    def update_params(self):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        for con, helpers in self._mutable_helpers.items():
            for helper in helpers:
                helper.update()
        for k, (v, helper) in self._mutable_bounds.items():
            helper.update()
        for helper in self._objective_helpers:
            helper.update()

    def _set_objective(self, obj):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        n = len(self._pyomo_var_to_solver_var_map)
        indices = np.arange(n)
        costs = np.zeros(n, dtype=np.double)
        self._objective_helpers = list()
        if obj is None:
            sense = pyhighs.ObjSense.kMinimize
            self._solver_model.changeObjectiveOffset(0)
        else:
            if obj.sense == minimize:
                sense = pyhighs.ObjSense.kMinimize
            elif obj.sense == maximize:
                sense = pyhighs.ObjSense.kMaximize
            else:
                raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

            repn = generate_standard_repn(obj.expr, quadratic=False, compute_values=False)
            if repn.nonlinear_expr is not None:
                raise DegreeError(f'Highs interface does not support expressions of degree {repn.polynomial_degree()}')

            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                v_id = id(v)
                v_ndx = self._pyomo_var_to_solver_var_map[v_id]
                costs[v_ndx] = value(coef)
                if not is_constant(coef):
                    mutable_objective_coef = _MutableObjectiveCoefficient(pyomo_var_id=v_id,
                                                                          var_map=self._pyomo_var_to_solver_var_map,
                                                                          expr=coef,
                                                                          highs=self._solver_model)
                    self._objective_helpers.append(mutable_objective_coef)

            self._solver_model.changeObjectiveOffset(value(repn.constant))
            if not is_constant(repn.constant):
                mutable_objective_offset = _MutableObjectiveOffset(expr=repn.constant, highs=self._solver_model)
                self._objective_helpers.append(mutable_objective_offset)

        self._solver_model.changeObjectiveSense(sense)
        pyhighs.highs_changeColsCost(self._solver_model, n, indices, costs)

    def _postsolve(self, timer: HierarchicalTimer):
        config = self.config

        highs = self._solver_model
        status = highs.getModelStatus()

        results = HighsResults(self)
        results.wallclock_time = highs.getRunTime()

        if status == pyhighs.HighsModelStatus.kNotset:
            results.termination_condition = TerminationCondition.unknown
        elif status == pyhighs.HighsModelStatus.kLoadError:
            results.termination_condition = TerminationCondition.error
        elif status == pyhighs.HighsModelStatus.kModelError:
            results.termination_condition = TerminationCondition.error
        elif status == pyhighs.HighsModelStatus.kPresolveError:
            results.termination_condition = TerminationCondition.error
        elif status == pyhighs.HighsModelStatus.kSolveError:
            results.termination_condition = TerminationCondition.error
        elif status == pyhighs.HighsModelStatus.kPostsolveError:
            results.termination_condition = TerminationCondition.error
        elif status == pyhighs.HighsModelStatus.kModelEmpty:
            results.termination_condition = TerminationCondition.unknown
        elif status == pyhighs.HighsModelStatus.kOptimal:
            results.termination_condition = TerminationCondition.optimal
        elif status == pyhighs.HighsModelStatus.kInfeasible:
            results.termination_condition = TerminationCondition.infeasible
        elif status == pyhighs.HighsModelStatus.kUnboundedOrInfeasible:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == pyhighs.HighsModelStatus.kUnbounded:
            results.termination_condition = TerminationCondition.unbounded
        elif status == pyhighs.HighsModelStatus.kObjectiveBound:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == pyhighs.HighsModelStatus.kObjectiveTarget:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == pyhighs.HighsModelStatus.kTimeLimit:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == pyhighs.HighsModelStatus.kIterationLimit:
            results.termination_condition = TerminationCondition.maxIterations
        elif status == pyhighs.HighsModelStatus.kUnknown:
            results.termination_condition = TerminationCondition.unknown
        else:
            results.termination_condition = TerminationCondition.unknown

        timer.start('load solution')
        self._sol = highs.getSolution()
        has_feasible_solution = False
        if results.termination_condition == TerminationCondition.optimal:
            has_feasible_solution = True
        elif results.termination_condition in {TerminationCondition.objectiveLimit,
                                               TerminationCondition.maxIterations,
                                               TerminationCondition.maxTimeLimit}:
            if self._sol.value_valid:
                has_feasible_solution = True

        if config.load_solution:
            if has_feasible_solution:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning('Loading a feasible but suboptimal solution. '
                                   'Please set load_solution=False and check '
                                   'results.termination_condition and '
                                   'resutls.found_feasible_solution() before loading a solution.')
                self.load_vars()
            else:
                raise RuntimeError('A feasible solution was not found, so no solution can be loaded.'
                                   'Please set opt.config.load_solution=False and check '
                                   'results.termination_condition and '
                                   'resutls.best_feasible_objective before loading a solution.')
        timer.stop('load solution')

        results.best_objective_bound = None
        results.best_feasible_objective = None
        if self._objective is not None:
            if results.termination_condition == TerminationCondition.optimal:
                results.best_feasible_objective = highs.getObjectiveValue()
            elif has_feasible_solution:
                sub_map = dict()
                for v_id, v_ndx in self._pyomo_var_to_solver_var_map.items():
                    v_val = self._sol.col_value[v_ndx]
                    sub_map[v_id] = v_val
                results.best_feasible_objective = replace_expressions(self._objective.expr,
                                                                      substitution_map=sub_map,
                                                                      descend_into_named_expressions=True,
                                                                      remove_named_expressions=True)

        return results

    def load_vars(self, vars_to_load=None):
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        if self._sol is None or not self._sol.value_valid:
            raise RuntimeError('We dont currently have a valid solution - '
                               'This could be because the termination condition was not optimal or because the '
                               'model was modified since the last solve')
        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = list(self._vars.keys())
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_value

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def get_reduced_costs(self, vars_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise RuntimeError('We dont currently have a valid duals - '
                               'This could be because the termination condition was not optimal or because the '
                               'model was modified since the last solve or the model was a MIP')
        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = list(self._vars.keys())
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_dual

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def get_duals(self, cons_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise RuntimeError('We dont currently have a valid duals - '
                               'This could be because the termination condition was not optimal or because the '
                               'model was modified since the last solve or the model was a MIP')

        res = dict()
        if cons_to_load is None:
            cons_to_load = list(self._pyomo_con_to_solver_con_map.keys())

        duals = self._sol.row_dual

        for c in cons_to_load:
            c_ndx = self._pyomo_con_to_solver_con_map[c]
            res[c] = duals[c_ndx]

        return res

    def get_slacks(self, cons_to_load=None):
        if self._sol is None or not self._sol.value_valid:
            raise RuntimeError('We dont currently have a valid solution - '
                               'This could be because the termination condition was not optimal or because the '
                               'model was modified since the last solve')

        res = dict()
        if cons_to_load is None:
            cons_to_load = list(self._pyomo_con_to_solver_con_map.keys())

        slacks = self._sol.row_value

        for c in cons_to_load:
            c_ndx = self._pyomo_con_to_solver_con_map[c]
            res[c] = slacks[c_ndx]

        return res