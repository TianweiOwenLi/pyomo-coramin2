from pyutilib.services import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import Solver, Results, TerminationCondition, SolverConfig
from pyomo.contrib.appsi.writers import LPWriter
from pyomo.contrib.appsi.utils import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import TeeStream
import sys


logger = logging.getLogger(__name__)


class CbcConfig(SolverConfig):
    def __init__(self):
        super(CbcConfig, self).__init__()
        self.executable = Executable('cbc')
        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class Cbc(Solver):
    def __init__(self):
        self._config = CbcConfig()
        self._solver_options = dict()
        self._writer = LPWriter()
        self._filename = None
        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

    def available(self, exception_flag=False):
        if self.config.executable.path() is None:
            if exception_flag:
                raise RuntimeError('Cbc is not available')
            return False
        return True

    def lp_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.lp'

    def log_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.log'

    def soln_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.soln'

    @property
    def config(self):
        return self._config

    @property
    def solver_options(self):
        return self._solver_options

    @property
    def update_config(self):
        return self._writer.update_config

    def set_instance(self, model):
        self._writer.set_instance(model)

    def add_variables(self, variables: List[_GeneralVarData]):
        self._writer.add_variables(variables)

    def add_params(self, params: List[_ParamData]):
        self._writer.add_params(params)

    def add_constraints(self, cons: List[_GeneralConstraintData]):
        self._writer.add_constraints(cons)

    def add_block(self, block: _BlockData):
        self._writer.add_block(block)

    def remove_variables(self, variables: List[_GeneralVarData]):
        self._writer.remove_variables(variables)

    def remove_params(self, params: List[_ParamData]):
        self._writer.remove_params(params)

    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._writer.remove_constraints(cons)

    def remove_block(self, block: _BlockData):
        self._writer.remove_block(block)

    def set_objective(self, obj: _GeneralObjectiveData):
        self._writer.set_objective(obj)

    def update_variables(self, variables: List[_GeneralVarData]):
        self._writer.update_variables(variables)

    def update_params(self):
        self._writer.update_params()

    def solve(self, model, timer: HierarchicalTimer = None):
        if not self.available():
            raise RuntimeError('Could not find CBC executable')
        if timer is None:
            timer = HierarchicalTimer()
        try:
            TempfileManager.push()
            if self.config.filename is None:
                self._filename = TempfileManager.create_tempfile()
            else:
                self._filename = self.config.filename
            TempfileManager.add_tempfile(self._filename + '.lp', exists=False)
            TempfileManager.add_tempfile(self._filename + '.soln', exists=False)
            TempfileManager.add_tempfile(self._filename + '.log', exists=False)
            timer.start('write lp file')
            self._writer.write(model, self._filename+'.lp', timer=timer)
            timer.stop('write lp file')
            res = self._apply_solver(timer)
            if self.config.report_timing:
                logger.info('\n' + str(timer))
            return res
        finally:
            # finally, clean any temporary files registered with the
            # temp file manager, created/populated *directly* by this
            # plugin.
            TempfileManager.pop(remove=not self.config.keepfiles)
            if not self.config.keepfiles:
                self._filename = None

    def _parse_soln(self):
        results = Results()

        f = open(self._filename + '.soln', 'r')
        all_lines = list(f.readlines())
        f.close()

        termination_line = all_lines[0].lower()
        obj_val = None
        if termination_line.startswith('optimal'):
            results.termination_condition = TerminationCondition.optimal
            obj_val = float(termination_line.split()[-1])
        elif 'infeasible' in termination_line:
            results.termination_condition = TerminationCondition.infeasible
        elif 'unbounded' in termination_line:
            results.termination_condition = TerminationCondition.unbounded
        elif termination_line.startswith('stopped on time'):
            results.termination_condition = TerminationCondition.maxTimeLimit
            obj_val = float(termination_line.split()[-1])
        elif termination_line.startswith('stopped on iterations'):
            results.termination_condition = TerminationCondition.maxIterations
            obj_val = float(termination_line.split()[-1])
        else:
            results.termination_condition = TerminationCondition.unknown

        first_con_line = None
        last_con_line = None
        first_var_line = None
        last_var_line = None

        for ndx, line in enumerate(all_lines):
            if line.strip('*').strip().startswith('0'):
                if first_con_line is None:
                    first_con_line = ndx
                else:
                    last_con_line = ndx - 1
                    first_var_line = ndx
        last_var_line = len(all_lines) - 1

        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

        symbol_map = self._writer.symbol_map

        for line in all_lines[first_con_line:last_con_line+1]:
            split_line = line.strip('*')
            split_line = split_line.split()
            name = split_line[1]
            orig_name = name[:-3]
            if orig_name == 'obj_const_con':
                continue
            con = symbol_map.bySymbol[orig_name]()
            dual_val = float(split_line[-1])
            if con in self._dual_sol:
                if abs(dual_val) > abs(self._dual_sol[con]):
                    self._dual_sol[con] = dual_val
            else:
                self._dual_sol[con] = dual_val

        for line in all_lines[first_var_line:last_var_line+1]:
            split_line = line.split()
            name = split_line[1]
            if name == 'obj_const':
                continue
            val = float(split_line[2])
            rc = float(split_line[3])
            var = symbol_map.bySymbol[name]()
            self._primal_sol[var] = val
            self._reduced_costs[var] = rc

        if results.termination_condition == TerminationCondition.optimal and self.config.load_solution:
            for v, val in self._primal_sol.items():
                v.value = val
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                results.best_feasible_objective = obj_val
        elif results.termination_condition == TerminationCondition.optimal:
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                results.best_feasible_objective = obj_val
        elif self.config.load_solution:
            raise RuntimeError('A feasible solution was not found, so no solution can be loaded.'
                               'Please set opt.config.load_solution=False and check '
                               'results.termination_condition and '
                               'resutls.best_feasible_objective before loading a solution.')

        return results

    def _apply_solver(self, timer: HierarchicalTimer):
        config = self.config

        if config.time_limit is not None:
            timeout = config.time_limit + min(max(1, 0.01 * config.time_limit), 100)
        else:
            timeout = None

        def _check_and_escape_options():
            for key, val in self.solver_options.items():
                tmp_k = str(key)
                _bad = ' ' in tmp_k

                tmp_v = str(val)
                if ' ' in tmp_v:
                    if '"' in tmp_v:
                        if "'" in tmp_v:
                            _bad = True
                        else:
                            tmp_v = "'" + tmp_v + "'"
                    else:
                        tmp_v = '"' + tmp_v + '"'

                if _bad:
                    raise ValueError("Unable to properly escape solver option:"
                                     "\n\t%s=%s" % (key, val) )
                yield tmp_k, tmp_v

        cmd = [str(config.executable)]
        action_options = list()
        if config.time_limit is not None:
            cmd.extend(['-sec', str(config.time_limit)])
            cmd.extend(['-timeMode', 'elapsed'])
        for key, val in _check_and_escape_options():
            if val.strip() != '':
                cmd.append('-'+key, val)
            else:
                action_options.append('-'+key)
        cmd.extend(['-printingOptions', 'all'])
        cmd.extend(['-import', self._filename + '.lp'])
        cmd.extend(action_options)
        cmd.extend(['-stat=1'])
        cmd.extend(['-solve'])
        cmd.extend(['-solu', self._filename + '.soln'])

        ostreams = [LogStream(level=self.config.log_level, logger=self.config.solver_output_logger)]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        with TeeStream(*ostreams) as t:
            timer.start('subprocess')
            cp = subprocess.run(cmd,
                                timeout=timeout,
                                stdout=t.STDOUT,
                                stderr=t.STDERR,
                                universal_newlines=True)
            timer.stop('subprocess')

        if cp.returncode != 0:
            if self.config.load_solution:
                raise RuntimeError('A feasible solution was not found, so no solution can be loaded.'
                                   'Please set opt.config.load_solution=False and check '
                                   'results.termination_condition and '
                                   'results.best_feasible_objective before loading a solution.')
            results = Results()
            results.termination_condition = TerminationCondition.error
            results.best_feasible_objective = None
            self._primal_sol = None
            self._dual_sol = None
            self._reduced_costs = None
        else:
            timer.start('parse solution')
            results = self._parse_soln()
            timer.stop('parse solution')

        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
            results.best_objective_bound = None
        else:
            if self._writer.get_active_objective().sense == minimize:
                results.best_objective_bound = -math.inf
            else:
                results.best_objective_bound = math.inf

        return results

    def load_vars(self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None) -> NoReturn:
        if vars_to_load is None:
            for v, val in self._primal_sol.items():
                v.value = val
        else:
            for v in vars_to_load:
                v.value = self._primal_sol[v]

    def get_duals(self, cons_to_load = None):
        if cons_to_load is None:
            return {k: v for k, v in self._dual_sol.items()}
        else:
            return {c: self._dual_sol[c] for c in cons_to_load}

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None) -> Mapping[_GeneralVarData, float]:
        if vars_to_load is None:
            return ComponentMap((k, v) for k, v in self._reduced_costs.items())
        else:
            return ComponentMap((v, self._reduced_costs[v]) for v in vars_to_load)