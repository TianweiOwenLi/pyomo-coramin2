#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
import logging
from six import iteritems

from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import (
    NumericValue, native_numeric_types, as_numeric, value )
import pyomo.core.base
from pyomo.core.expr.expr_errors import TemplateExpressionError

class _NotSpecified(object): pass

class IndexTemplate(NumericValue):
    """A "placeholder" for an index value in template expressions.

    This class is a placeholder for an index value within a template
    expression.  That is, given the expression template for "m.x[i]",
    where `m.z` is indexed by `m.I`, the expression tree becomes:

    _GetItem:
       - m.x
       - IndexTemplate(_set=m.I, _value=None)

    Constructor Arguments:
       _set: the Set from which this IndexTemplate can take values
    """

    __slots__ = ('_set', '_value', '_index', '_id')

    def __init__(self, _set, index=0, _id=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(IndexTemplate, self).__getstate__()
        for i in IndexTemplate.__slots__:
            state[i] = getattr(self, i)
        return state

    def __deepcopy__(self, memo):
        # Because we leverage deepcopy for expression cloning, we need
        # to see if this is a clone operation and *not* copy the
        # template.
        #
        # TODO: JDS: We should consider converting the IndexTemplate to
        # a proper Component: that way it could leverage the normal
        # logic of using the parent_block scope to dictate the behavior
        # of deepcopy.
        if '__block_scope__' in memo:
            memo[id(self)] = self
            return self
        #
        # "Normal" deepcopying outside the context of pyomo.
        #
        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is _NotSpecified:
            if exception:
                raise TemplateExpressionError(self)
            return None
        else:
            return self._value

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_constant(self):
        """
        Returns False because this cannot immediately be simplified.
        """
        return False

    def is_potentially_variable(self):
        """Returns False because index values cannot be variables.

        The IndexTemplate represents a placeholder for an index value
        for an IndexedComponent, and at the moment, Pyomo does not
        support variable indirection.
        """
        return False

    def __str__(self):
        return self.getname()

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        if self._id is not None:
            return "_%s" % (self._id,)

        _set_name = self._set.getname(fully_qualified, name_buffer, relative_to)
        if self._index is not None and self._set.dimen != 1:
            _set_name += "(%s)" % (self._index,)
        return "{"+_set_name+"}"

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        return self.name

    def set_value(self, *values):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimention
        # 1.  So, for the time being, we will just "trust" the user.
        # After all, the actual Set will raise exceptions if the value
        # is not present.
        if not values:
            self._value = _NotSpecified
        elif self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                raise ValueError("Passed multiple values %s to a scalar "
                                 "IndexTemplate %s" % (values, self))
        else:
            self._value = values

class ReplaceTemplateExpression(EXPR.ExpressionReplacementVisitor):

    def __init__(self, substituter, *args):
        super(ReplaceTemplateExpression, self).__init__()
        self.substituter = substituter
        self.substituter_args = args

    def visiting_potential_leaf(self, node):
        if type(node) is EXPR.GetItemExpression or type(node) is IndexTemplate:
            return True, self.substituter(node, *self.substituter_args)

        return super(
            ReplaceTemplateExpression, self).visiting_potential_leaf(node)


def substitute_template_expression(expr, substituter, *args):
    """Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    and subtituting all occurances of IndexTemplate and
    _GetItemExpression nodes.

    Args:
        substituter: method taking (expression, *args) and returning
           the new object
        *args: these are passed directly to the substituter

    Returns:
        a new expression tree with all substitutions done
    """
    visitor = ReplaceTemplateExpression(substituter, *args)
    return visitor.dfs_postorder_stack(expr)


class _GetItemIndexer(object):
    # Note that this class makes the assumption that only one template
    # ever appears in an expression for a single index

    def __init__(self, expr):
        self._base = expr._base
        self._args = []
        _hash = [ id(self._base) ]
        for x in expr.args:
            try:
                logging.disable(logging.CRITICAL)
                val = value(x)
                self._args.append(val)
                _hash.append(val)
            except TemplateExpressionError as e:
                if x is not e.template:
                    raise TypeError(
                        "Cannot use the param substituter with expression "
                        "templates\nwhere the component index has the "
                        "IndexTemplate in an expression.\n\tFound in %s"
                        % ( expr, ))
                self._args.append(e.template)
                _hash.append(id(e.template._set))
            finally:
                logging.disable(logging.NOTSET)

        self._hash = tuple(_hash)

    def nargs(self):
        return len(self._args)

    def arg(self, i):
        return self._args[i]

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is _GetItemIndexer:
            return self._hash == other._hash
        else:
            return False

    def __str__(self):
        return "%s[%s]" % (
            self._base.name, ','.join(str(x) for x in self._args) )


def substitute_getitem_with_param(expr, _map):
    """A simple substituter to replace _GetItem nodes with mutable Params.

    This substituter will replace all _GetItemExpression nodes with a
    new Param.  For example, this method will create expressions
    suitable for passing to DAE integrators
    """
    if type(expr) is IndexTemplate:
        return expr

    _id = _GetItemIndexer(expr)
    if _id not in _map:
        _map[_id] = pyomo.core.base.param.Param(mutable=True)
        _map[_id].construct()
        _args = []
        _map[_id]._name = "%s[%s]" % (
            expr._base.name, ','.join(str(x) for x in _id._args) )
    return _map[_id]


def substitute_template_with_value(expr):
    """A simple substituter to expand expression for current template

    This substituter will replace all _GetItemExpression / IndexTemplate
    nodes with the actual _ComponentData based on the current value of
    the IndexTamplate(s)

    """

    if type(expr) is IndexTemplate:
        return as_numeric(expr())
    else:
        return expr.resolve_template()



class mock_globals(object):
    """Implement custom context for a user-specified function.

    This class implements a custom context that injects user-specified
    attributes into the globals() context before calling a function (and
    then cleans up the global context after the function returns).

    Parameters
    ----------
        fcn : function
            The function whose globals context will be overridden
        overrides : dict
            A dict mapping {name: object} that will be injected into the
            `fcn` globals() context.
    """
    __slots__ = ('_data',)

    def __init__(self, fcn, overrides):
        self._data = fcn, overrides

    def __call__(self, *args, **kwds):
        fcn, overrides = self._data
        _old = {}
        try:
            for name, val in iteritems(overrides):
                if name in fcn.__globals__:
                    _old[name] = fcn.__globals__[name]
            fcn.__globals__[name] = val

            return fcn(*args, **kwds)
        finally:
            for name, val in iteritems(overrides):
                if name in _old:
                    fcn.__globals__[name] = _old[name]
                else:
                    del fcn.__globals__[name]


class _set_iterator_template_generator(object):
    """Replacement iterator that returns IndexTemplates

    In order to generate template expressions, we hijack the normal Set
    iteration mechanisms so that this iterator is returned instead of
    the usual iterator.  This iterator will return IndexTemplate
    object(s) instead of the actual Set items the first time next() is
    called.
    """
    def __init__(self, _set, context):
        self._set = _set
        self.context = context

    def __iter__(self):
        return self

    def __next__(self):
        # Prevent context from ever being called more than once
        if self.context is None:
            raise StopIteration()

        context, self.context = self.context, None
        _set = self._set
        d = _set.dimen
        if d is None:
            idx = (IndexTemplate(_set, None, context.next_id()),)
        else:
            idx = tuple(
                IndexTemplate(_set, i, context.next_id()) for i in range(d)
            )
        context.cache.append(idx)
        if len(idx) == 1:
            return idx[0]
        else:
            return idx

    next = __next__

class _template_iter_context(object):
    """Manage the iteration context when generating templatized rules

    This class manages the context tracking when generating templatized
    rules.  It has two methods (`sum_template` and `get_iter`) that
    replace standard functions / methods (`sum` and
    :py:meth:`_FiniteSetMixin.__iter__`, respectively).  It also tracks
    unique identifiers for IndexTemplate objects and their groupings
    within `sum()` generators.
    """
    def __init__(self):
        self.cache = []
        self._id = 0

    def get_iter(self, _set):
        return _set_iterator_template_generator(_set, self)

    def npop_cache(self, n):
        result = self.cache[-n:]
        self.cache[-n:] = []
        return result

    def next_id(self):
        self._id += 1
        return self._id

    def sum_template(self, generator):
        init_cache = len(self.cache)
        expr = next(generator)
        final_cache = len(self.cache)
        return EXPR.TemplateSumExpression(
            (expr,), self.npop_cache(final_cache-init_cache)
        )

def templatize_rule(block, rule, index_set):
    context = _template_iter_context()
    #rule = mock_globals(rule, {'sum': context.sum_template})
    try:
        # Override Set iteration to return IndexTemplates
        _old_iter = pyomo.core.base.set._FiniteSetMixin.__iter__
        pyomo.core.base.set._FiniteSetMixin.__iter__ = \
            lambda x: context.get_iter(x)
        # Override sum with our sum
        _old_sum = __builtins__['sum']
        __builtins__['sum'] = context.sum_template
        # Get the index templates needed for calling the rule
        if index_set is not None:
            if not index_set.isfinite():
                raise TemplateExpressionError(
                    None,
                    "Cannot templatize rule with non-finite indexing set")
            indices = iter(index_set).next()
            context.cache.pop()
        else:
            indices = ()
        if type(indices) is not tuple:
            indices = (indices,)
        # Call the rule, returning the template expression and the
        # top-level IndexTemplaed generated when calling the rule.
        #
        # TBD: Should this just return a "FORALL()" expression node that
        # behaves similarly to the GetItemExpression node?
        return rule(block, *indices), indices
    finally:
        pyomo.core.base.set._FiniteSetMixin.__iter__ = _old_iter
        __builtins__['sum'] = _old_sum
        if len(context.cache):
            raise TemplateExpressionError(
                None,
                "Explicit iteration (for loops) over Sets is not supported by "
                "template expressions.  Encountered loop over %s"
                % (context.cache[-1][0]._set,))
