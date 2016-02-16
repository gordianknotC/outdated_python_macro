import sys;sys.path.append("third_parties")
#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, ast, importlib, sys, traceback
import importlib.abc, imp
from types import ModuleType
from copy import deepcopy
from astutil.codegen import to_source
from astutil.ast_tool import *
from astutil import codegen

__version__ = 0.1

IS              =isinstance
MACROS          =[]
macro         = None
macro_modules = []
code_storage  = {}
redirector    = {}
preprocess_calls = ['redirect']
cur_path = os.path.abspath(os.path.curdir)
hook_mode = None
macro_forced = False



class MacroExecutionError(Exception):
    def __init__(self, msg, pragname, node, kargs):
        funcname = '<Unknown_function>' if not kargs.get('context_node') else kargs.get('context_node').name
        print('  File "{}", line {}, in {}\n'.format(kargs['filename'], node.lineno, funcname), file=sys.stderr, end="")
        pragname = '\nsome error occured when executing {} pragma'.format(pragname)
        super(MacroExecutionError, self).__init__(msg)
        self.errors = pragname


def singleton(args=None, kargs=None, target=None):
    args = args or []
    kargs = kargs or {}
    def real(cls):
        o = cls(*args, **kargs)
        if hasattr(o, '__name__'): o.__name__ = cls.__name__
        setattr(sys.modules[__name__], target, o)
        return o
    return real

@singleton(args=['expr','block','astblock','astexpr'], target='macro')
class MacroClass(object):
    def __init__(self, *args):
        MACROS.extend(args)
    def expr(self, fn):
        def call_macro_def(*args, **kargs):
            try:                     return fn(*args, **kargs)
            except Exception as err:
                raise MacroExecutionError(err, fn.__name__, args[0], kargs)
        return call_macro_def

    def test(self, item, **kargs):
        # test for BlockMacroDef and ExprMacroDef
        if isinstance(item, BlockMacroDef):
            decor    = item.decorator_id.lower()
            is_ast   = 'ast' in decor
            is_expr  = 'expr' in decor
            is_block = 'block' in decor
            if 'ast' in kargs:
                return True if is_ast else False
            if is_expr:  return False
            if is_block: return True

        elif isinstance(item, ExprMacroDef):
            decor    = item.decorator_id.lower()
            is_ast   = 'ast' in decor
            is_expr  = 'expr' in decor
            is_block = 'block' in decor
            if 'ast' in kargs:
                return True if is_ast else False
            if is_expr: return True
            if is_block: return False
        #------------------------------------------
        # test for ast nodes

        if 'is_macro' in kargs:
            if not isinstance(item, ast.FunctionDef): return False
            if not (len(item.decorator_list) == 1 and isinstance(item.decorator_list[0], ast.Attribute)):
                return False
            if item.decorator_list[0].value.id == 'macro': return True

        else:
            if not kargs: raise Exception('invalid test usage! should be macro.test({}=True)'.format(k))
            for k,v in kargs.items():
                if not v: raise Exception('invalid test usage! should be macro.test({}=True)'.format(k))

            decor_cls        = item.decorator_list[0].value.id
            decor_id         = item.decorator_list[0].attr.lower()
            if not decor_cls == 'macro': return False
            if not decor_id  in MACROS:  return False
            is_ast           = 'ast'   in decor_id
            is_expr          = 'expr'  in decor_id
            is_block         = 'block' in decor_id

            #print('test:{}'.format(item.name),'cls:', decor_cls, 'id:', decor_id,is_ast, is_expr, is_block)

            if not is_ast:
                if item.args.vararg or item.args.kwarg or item.args.defaults:
                    raise Exception('macro %s has an unsupported signature' % item.name)

            if 'expr'  in kargs and is_expr:  return True
            if 'block' in kargs and is_block: return True
            if 'ast'   in kargs and is_ast:   return True
            else:                             return False

    astBlock = astExpr = multitpleExpr = block = expr

class MacroDefError(Exception):
    """Raised when an invalid macro definition is encountered."""

def getargs(n):
    if isinstance(n, ast.FunctionDef):
        args      = [i for i in n.args.args]
        vararg    = n.args.vararg
        konly     = n.args.kwonlyargs
        kdefaults = n.args.kw_defaults
        karg      = n.args.kwarg
        #defaults  = n.args.defaults or {}
        if args   : args   = dict(args= [i.arg for i in args],  annotations=[i.annotation for i in args],  values='')
        if vararg : vararg = dict(arg = [vararg.arg],           annotation =[vararg.annotation])
        if konly  : konly  = dict(args= [i.arg for i in konly], annotations=[i.annotation for i in konly], values= [i.value for i in kdefaults])
        if karg   : karg   = dict(arg = [karg.arg],             annotation =[karg.annotation])

        ret = dict(args=args, vararg=vararg, kargs=konly, kvararg=karg)
        return ret
    else:
        if isinstance(n, ast.Module):
            #print()
            #print('--------------- getargs ----------------------')
            #print(ast.dump(n))
            #print(to_source(n))
            if len(n.body) > 1: raise Exception('getvars only support for FunctionDef, With and Module encapsulate with one FunctionDef/With')
            return getargs(node.body[0])

def saver(key,body):
    code_storage[key] = body

def getSavedCode(key):
    if key in code_storage: return code_storage[key]
    else:                   return None

def parse_macros(code, filename, module):
    """Find and parse all macros in *code*.  Return a dictionary mapping macro
    names to MacroDefs.
    """
    code   = ast.parse(code, filename)
    macros = {}
    for item in code.body:
        if not macro.test(item, is_macro=True):
            continue

        is_expr  = macro.test(item, expr=True)
        is_block = macro.test(item, block=True)
        is_ast   = macro.test(item, ast=True)

        name = item.name
        args = getargs(item)

        # catch usage for @macro.block and @macro.expr
        if not is_ast and (is_block or is_expr):
            decor = item.decorator_list[0].attr.lower()
            if is_expr:
                if isinstance(item.body[0], ast.Assign):
                        userdef_macro = ExprMacroDef(args,  item.body[0],       decor, name, item, filename, module )
                else:   userdef_macro = ExprMacroDef(args,  item.body[0].value, decor, name, item, filename, module )
            else:       userdef_macro = BlockMacroDef(args, item.body,          decor, name, item, filename, module )

            # catch usage for @macro.astBlock, @macro.astExpr
        elif is_ast and (is_block or is_expr):
            decor = item.decorator_list[0].attr.lower()
            # read user def macro (macro callee function)
            # -------------------------------------------
            if IS(item.body[0], ast.Pass): continue
            #print('instantiate BlockMacroDef for:', item.name, 'decorated from:',decor,name,  code)
            if   is_block: userdef_macro = BlockMacroDef(args, item.body, decor, name, item, filename, module )
            else:          userdef_macro = ExprMacroDef(args, item.body,  decor, name, item, filename, module )
        else:
            #print (to_source(item))
            #print(ast.dump(item))
            if isinstance(item.decorator_list[0], ast.Attribute):
                if item.decorator_list[0].value.id == 'macro':
                    #print(to_source(item))
                    if item.decorator_id[0].attr.lower() in MACROS:
                        raise Exception('uncaught exception, decor:{}'.format(decor))
            else:
                # no macro
                continue

        macros[name] = userdef_macro

    #print('macros:', name)
    #print('macros:', userdef_macro)
    return macros

def import_libs(module, names, dict):
    """Import macros given in *names* from *module*, from a module with the
    given globals *dict*.
    """
    #print(module, names, dict)
    try:   mod = __import__(module, dict, None, ['*'])
    except Exception as err:
        raise MacroDefError('module %s not found: %s' % (module, err))

    filename = mod.__file__
    if filename.lower().endswith(('c', 'o')):
        filename = filename[:-1]
    with open(filename, 'U') as f:
        code = f.read()

def import_macros(module, names, dict):
    """Import macros given in *names* from *module*, from a module with the
    given globals *dict*.    """
    #print(module, names, dict)
    try:   mod = __import__(module, dict, None, ['*'])
    except Exception as err:
        raise MacroDefError('macro module %s not found: %s' % (module, err))

    filename = mod.__file__
    if filename.lower().endswith(('c', 'o')):
        filename = filename[:-1]
    print('open macro file:', filename)
    with open(filename, 'U') as f:
        code = f.read()
    all_macros = parse_macros(code, filename, module)
    macros = {}
    for name, asname in names.items():
        if name == '*':
            macros.update(all_macros)
            break
        try:
            macros[asname] = all_macros[name]
            #print('===========================')
            #print('   macros information ')
            #print ('--------------------------')
            #print('all macros', all_macros)
        except KeyError:
            raise MacroDefError('macro %s not found in module %s' %
                                (name, module))

    #print('import macros:', macros)
    return macros

recur = 0
def fix_locations(node, old_node):
    """Replace all code locations (lineno and col_offset) with the one from
    *old_node* (we cannot preserve original location information for code from
    macros since the compiler cannot know that it's from different files.
    """
    if node == old_node: return node
    is_n_list = isinstance(node, list)
    is_o_list = isinstance(old_node, list)
    condition = lambda x: dict(list=isinstance(x,list),body=hasattr(x,'body'),If=isinstance(x,ast.If), orelse=hasattr(x,'orelse')  )
    global recur
    recur = 0
    def flatten_list(lst, ret=None, con = condition):
        global recur
        for i in lst:
            if   con(i)['list']:
                flatten_list(i, ret)
                continue

            elif con(i)['If']:
                ret.append(i)
                flatten_list(i.body, ret)
                flatten_list(i.orelse, ret)
                continue

            elif con(i)['body']:
                ret.append(i)
                flatten_list(i.body, ret)
                if lst == i.body:
                    recur += 1
                    if recur > 5:
                        # print('============== fix location BUG ====================')
                        # print(lst, i.body, i)
                        # print(lst == i.body)
                        raise Exception('walking into and inftiniy loop')
                continue
            ret.append(i)
        return ret

    def forcePair(a,b):
        reta = []
        retb = []
        a = flatten_list(a, reta)
        b = flatten_list(b, retb)
        padding = [b[-1]] *(len(a) - len(b))
        b = b + padding
        return zip(a,b)

    def _fix(node, lineno, col_offset):
        node.lineno     = lineno
        node.col_offset = col_offset
        for child in ast.iter_child_nodes(node):
            _fix(child, lineno, col_offset)
            #print(child.lineno,lineno, col_offset, child)

    if is_n_list and is_o_list:
        if old_node and node:
            pair = list(forcePair(node, old_node))
            try:
                print('fixloc:', node, old_node)
                print("pair:")
                print([[i[0].lineno, i[1].lineno] for i in pair])
                print('_____')
            except: pass
            for a,b in pair:
                if a and b:
                    #print(a.lineno, a,b.lineno, b)
                    _fix(a, b.lineno,b.col_offset)
                    print(a.lineno, a,b.lineno, b)
            print('_____')
            print([[i[0].lineno, i[1].lineno] for i in pair])
    elif is_n_list and not is_o_list:
        if old_node and node:
            return fix_locations(node, [old_node])
    else:   _fix(node, old_node.lineno, old_node.col_offset)
    return node

def import_macro_module(lib:str):
    global macro_forced
    print('\n macroforced to true')
    macro_forced = True
    importlib.import_module(lib)
    macro_forced = False
    print('\n macroforced to flase')
    raise Exception()

class TranslateError(Exception):
    def __init__(self, msg, filename=None, function_node=None, node=None):
        print('  File "{}", line {}, in {}\n'.format(filename, node.lineno, function_node.name), file=sys.stderr, end="")
        raise Exception(msg)


def parse_ast(s:str, filename='<unknown>', mode='exec')->ast.AST:
        ast.parse(s, filename,mode)


class MacroCallError(Exception):
    """Raised when an invalid macro call is encountered."""

    def __init__(self, node, message):
        Exception.__init__(self, '%s: %s' % (node.lineno, message))

    def add_filename(self, filename):
        self.args = [filename + ':' + self.args[0]]


class ContextChanger(ast.NodeVisitor):
    """
    AST visitor that updates the "context" on nodes that can occur on the LHS or
    RHS in an assignment.  This is needed because on a macro call, arguments
    always have Load context, while in the expansion, they can also have Store
    or other contexts.
    """

    def __init__(self, context):
        self.context = context

    def visit_Name(self, node):
        node.ctx = self.context
        self.generic_visit(node)  # visit children

    visit_Attribute = visit_Subscript = visit_List = visit_Tuple = visit_Name


class CallTransformer(ast.NodeTransformer):
    """
    AST visitor that expands uses of macro arguments and __body__ inside a macro
    definition.
    """

    def __init__(self, args, body=None):
        self.args = args
        self.body = body
        print('instantiate CallTransformer body:',body, 'args:', args)

    def visit_Name(self, node):
        print('callTransformer visti:', ast.dump(node), 'args:', self.args)
        if node.id in self.args:
            if not isinstance(node.ctx, ast.Load):
                new_node = deepcopy(self.args[node.id])
                ContextChanger(node.ctx).visit(new_node)
            else:
                new_node = self.args[node.id]
            print('new node:', ast.dump(new_node))
            return new_node
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        body = self.body
        print('CallTransformer visitExpr')
        print('node:', ast.dump(node))
        print('body:', body)
        if body and isinstance(node.value, ast.Name) and node.value.id == '__body__':
            print('found __body__')
            return fix_locations(ast.If(ast.Num(1), body, []), node)
        return self.generic_visit(node)


class BodyVisitor(ast.NodeVisitor):
    """
    AST visitor that checks for use of __body__, to determine if a block macro
    has a body.
    """

    def __init__(self):
        self.found_body = False

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == '__body__':
            self.found_body = True


class BaseMacro(object):
    def __init__(self, args, expr, decorator_id, name, code, filename, module):
        vararg  = args['vararg'] if args['vararg']  else {'arg':[], 'annotation':[]}
        kargs   = args['kargs']  if args['kargs']   else {'args':[], 'annotations':[], 'values':[]}
        kvararg = args['kvararg']if args['kvararg'] else {'arg':[], 'annotation':[]}

        self.args          = args['args']['args']
        self.args_ann      = args['args']['annotations']
        self.vararg        = vararg['arg']
        self.vararg_ann    = vararg['annotation']
        self.kargs         = kargs['args']
        self.kargs_values  = kargs['values']
        self.kargs_ann     = kargs['annotations']
        self.kvararg       = kvararg['arg']
        self.kvararg_ann   = kvararg['annotation']

        self.filename     = filename
        self.expr         = expr
        self.has_body     = False
        self.decorator_id = decorator_id
        self.func_name    = name
        self.code         = code
        self.module_name       = module

    def __call__(self, args=None, kargs=None ):
        pass


class ExprMacroDef(BaseMacro):
    """
    Definition of an expression macro.
    """

    def __init__(self, args, expr, decorator_id, name, code, filename, module):
        super(ExprMacroDef, self).__init__(args,expr, decorator_id, name, code, filename, module)
        #print('instantiate ExprMacroDef for:', code.name, 'decorated from:',decorator_id,name)
        #print('expr == ', ast.dump(self.expr))
    def expand(self, node, call_args, body=None):
        #print('ExprMacroDef.expand:', ast.dump(self.expr))
        call_args = list(call_args)
        if len(call_args) != len(self.args):
            raise MacroCallError(node, 'invalid number of arguments')
        expr = deepcopy(self.expr)
        argdict = dict(zip(self.args, call_args))
        if not body:
            print('\n\ntransformer visit expr:', ast.dump(expr))
            return CallTransformer(argdict, body).visit(expr)

        else:
            expr = fix_locations(ast.If(ast.Num(1), expr, []), node)
            print('\n\ntransformer visit block expr:', to_source(expr))
            result =  CallTransformer(argdict, body).visit(expr)
            print('translated astExpr:')
            print(to_source(result))
            return result



class BlockMacroDef(BaseMacro):
    """
    Definition of a block macro, with or without body.
    """
    def __init__(self, args, stmts, decorator_id, name,  code, filename, module):
        super(BlockMacroDef, self).__init__(args,stmts, decorator_id, name, code, filename, module)
        #print('instantiate BlockMacroDef for:', code.name, 'decorated from:',decorator_id,name)
        visitor = BodyVisitor()
        visitor.visit(ast.Module(stmts))
        self.stmts = stmts
        self.has_body = visitor.found_body

    def expand(self, node, call_args, body=None):
        # body: macro caller source
        # print('BlockMacroDef.expand')
        call_args = list(call_args)
        if len(call_args) != len(self.args):
            raise MacroCallError(node, 'invalid number of arguments')
        stmts = deepcopy(self.stmts)
        # further check!!!!
        argdict = dict(zip(self.args, call_args))

        #print('fix location:')
        #print(to_source(node))
        new_node = fix_locations(ast.If(ast.Num(1), stmts, []), node)
        return CallTransformer(argdict, body).visit(new_node)


class BaseExpander(ast.NodeTransformer):
    def __init__(self, module, macro_definitions=None, debug=False, filename='', tree=None):
        #print('InitExpander:', module)
        self.module = module
        self.debug=debug
        self.defs = macro_definitions or {}
        self.dependencies = []
        self.filename = filename
        self.tree = tree
        self.under_call = False
        self.argstorage = {}

    def after_transformed(self):
        raise NotImplemented()

    def gen_call(self,funcname:str, args:list, keywrods:list, starargs, kwargs):
        return ast.Call(func=Name(id=funcname, ctx=Load()),args=args, keywrods=keywrods, starargs=starargs, kwargs=kwargs )

    def gen_assign(self, targets, value):
        return ast.Assign(targets=targets, value = value)


    def is_macro_xxx(self, node, condition):
        if node.decorator_list:
            decors = list(filter(condition, node.decorator_list))
            if decors: return decors
            else:      return []
        else:          return []

    # call from FunctionDef
    def is_macro_decorator(self,node):
        def condition(x):
            if isinstance(x, ast.Name):
                if x.id in self.defs: return True
            return False
        return self.is_macro_xxx(node, condition)

    def is_macro_decorator_call(self, node):
        def condition(x):
            if hasattr(x, 'func'):
                if hasattr(x.func,'id'):
                    if x.func.id in self.defs: return True
            return False
        return self.is_macro_xxx(node, condition)

    def is_macro_decorator_attribute(self, node):      raise NotImplemented()

    def is_macro_decorator_attribute_call(self, node): raise NotImplemented()

    def is_macro_expr(self, node):
        value = node.value
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                if value.func.id in self.defs: return True
        return False

    def is_macro_block(self, node): pass

    def is_macro_ast_block(self, node): pass

    def _handle_ast_decorator_macro(self,node,macro_def, decor_call=False):
        #return macro_def.code.body[0]
        print()
        print(' ================ AST Decorator macro ==================', node.lineno)
        print(to_source(node))
        print(ast.dump(node))
        caller_body     = deepcopy(node.body)
        caller_args     = node.args.args

        macro_func     = sys.modules[macro_def.__dict__['module_name']].__dict__[macro_def.__dict__['func_name']]
        caller_func     = sys.modules[ self.module.__name__].__dict__[node.body.name] \
                                       if self.module.__name__ in sys.modules else macro_func

        caller_node     = node
        caller_vararg  = node.args.vararg     if node.args.vararg else ''
        caller_kargs   = node.args.kwonlyargs if node.args.kwonlyargs else []
        caller_kvararg = node.args.kwarg      if node.args.kwarg else ''

        if decor_call:
            caller_args.extend ( self.argstorage[macro_def.func_name]['args'] )
            caller_kargs.extend( self.argstorage[macro_def.func_name]['kargs'] )
            caller_kargs = {i.arg:i.value for i in caller_kargs}


        if caller_vararg:  print('caller_vararg::',  caller_vararg)
        if caller_kargs:   print('caller_kargs::',   caller_kargs)
        if caller_kvararg: print('caller_kvararg::', caller_kvararg)
        nbody_gen_from_macro = execMacro(
             macro_func,           caller_body, caller_args,
             filename=macro_def.filename, caller_func=caller_func,
             module=self.module.__name__, caller_node=caller_node,
             kargs=caller_kargs)

        ret = fix_locations(nbody_gen_from_macro, node.body)
        node.body = ret
        node.decorator_list = []
        node.lineno += 1   # clear decorator
        print('_(_#$%_$%_$_%(_$(_%($#_)$#(')
        print(ast.dump(node))
        return node

    def _handle_ast_block_macro(self,node,macro_def, optional_vars=None, optional_vars_args=None):
        #return macro_def.code.body[0]
        print()
        print(' ================ AST Block macro ==================')
        macro_node     = macro_def.code
        caller_body     = deepcopy(node.body)
        macro_func     = sys.modules[ macro_def.__dict__['module_name'] ].__dict__[macro_def.__dict__['func_name']]
        caller_func     = sys.modules[ self.module.__name__].__dict__[node.body.name] if self.module.__name__ in sys.modules else macro_func
        caller_node     = getSavedCode(node.items[0].context_expr.func.id) or node
        caller_args        = node.items[0].context_expr.args
        optional_subscript = node.items[0].context_expr.args
        kargs              = node.items[0].context_expr.keywords

        if optional_vars:
            if IS(optional_vars, ast.Subscript):
                optional_subscript = node.items[0].context_expr.args

        nbody_gen_from_macro_caller = execMacro(
             macro_func, caller_body, caller_args,
             filename=macro_def.filename,           optional_vars=optional_vars,
             optional_subscript=optional_subscript, optional_vars_args=optional_vars_args,
             kargs=kargs,                           caller_func=caller_func,
             module=self.module.__name__,           caller_node=caller_node)

        ret = fix_locations(nbody_gen_from_macro_caller, node.body)
        return ret

    def _call_check(self, node, macrotype, id = None):
        if node.keywords or node.starargs or node.kwargs:
            raise MacroCallError(node, 'macro call with kwargs or star syntax')
        if not id:
            id = node.func.id
        macro_def = self.defs[id]
        if not isinstance(macro_def, macrotype):
            raise MacroCallError(node, 'macro is not a %s' % macrotype)
        if macro_def.has_body:
            raise MacroCallError(node, 'macro has a __body__ substitution')
        return macro_def

    def _handle_astExpr(self, caller, macro_def):
        caller_args    = caller.args
        macro_func     = sys.modules[macro_def.__dict__['module_name']].__dict__[macro_def.__dict__['func_name']]
        caller_node    = None

        try:
            nbody_gen_from_macro = execMacro(
                 macro_func,            None, caller_args,
                 filename=macro_def.filename, caller_func=None,
                 module=self.module.__name__, caller_node=caller_node,
                 kargs=None)
        except Exception as err:
            raise Exception(err)

        fix_locations(nbody_gen_from_macro, caller)
        return nbody_gen_from_macro

    def _handle_call(self, node, macrotype, id = None):
        macro_def = self._call_check(node, macrotype, id = id)
        if macro_def:
            is_ast = macro.test(macro_def, ast=True)
            expanded_args = map(self.visit, node.args)
            if not is_ast:
                return macro_def.expand(node, expanded_args)
            else:
                if self.under_call: raise Exception('invalid usage of astExpr, not allowed for call arguments or value assignment')
                return self._handle_astExpr(node, macro_def)

    def is_import_macro_stmt(self,node:ast.AST) ->bool:
        return node.module and node.module.endswith('.__macros__')

    def import_macro(self, node):
        modname = node.module[:-11]
        names   = dict((alias.name, alias.asname or alias.name)
                     for alias in node.names)
        self.defs.update(import_macros(
            modname, names, self.module and self.module.__dict__))
        macro_modules.append(modname)
        print('add macro modul:', macro_modules)
    def visit_Import(self, node):
        #print('try visit import')
        for n in node.names:
            print(to_source(n))
        self.dependencies.append(node)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if self.is_import_macro_stmt(node):
            self.import_macro(node)
            return None

        self.generic_visit(node)
        return node

class Expander(BaseExpander):
    """
    AST visitor that expands macros.
    """

    def __init__(self, *args, **kargs):
        super(Expander, self).__init__(*args, **kargs)

    def after_transformed(self):
        pass




    def visit_With(self, node):
        print('visit_With, name:',node.items[0].context_expr)
        # With(                context_expr=Call(func=Name(id='custom_loop', ctx=Load()), args=[Num(n=10)], keywords=[], starargs=None, kwargs=None), optional_vars=None,   body=[Print(dest=None, values=[Str(s='loop continues...')], nl=True)])
        # With(items=[withitem(context_expr=Call(func=Name(id='custom_loop', ctx=Load()), args=[Num(n=10)], keywords=[], starargs=None, kwargs=None), optional_vars=None)], body=[Expr(value=Call(func=Name(id='print', ctx=Load()), args=[Str(s='loop continues...')], keywords=[], starargs=None, kwargs=None))])
        # print(ast.dump(node))
        expr               = node.items[0].context_expr
        optional_vars      = node.items[0].optional_vars
        optional_vars_args = None
        macro_def          = None
        if IS(expr, ast.Call) and IS(expr.func, ast.Name):
            if   IS(optional_vars, ast.Subscript):
                optional_vars_args = [i.id for i in optional_vars.slice.value.elts]
                optional_vars      = optional_vars.value.id
            elif IS(optional_vars, ast.Name):
                optional_vars      = optional_vars.id

            if   expr.func.id in self.defs:  macro_def = self.defs[expr.func.id]
            elif optional_vars in self.defs: macro_def = self.defs[optional_vars]

            if macro_def:
                expanded_body = list(map(self.visit, deepcopy(node.body)))
                is_ast        = macro.test(macro_def, ast = True)
                if not is_ast:
                    if optional_vars: raise MacroCallError(node, 'only astmacro supported for optional_vars')
                    if expr.keywords or expr.starargs or expr.kwargs: raise MacroCallError(node, 'macro call with kwargs or star syntax')
                    if not isinstance(macro_def, BlockMacroDef):      raise MacroCallError(node, 'not a block macro')
                    if not macro_def.has_body:                        raise MacroCallError(node, 'macro has no __body__ substitution')

                #-------------------------------------------------------------
                #       Fetch Macro Caller Source code here.........
                #------------------------------------------------------------
                print()
                print()
                print ('======== MACRO CALLER {}============'.format(node.items[0].context_expr.func.id))
                print('Expander visit With:') # BlockMacroDef
                print('decorated form:', macro_def.decorator_id)
                print('node: macro caller node:', to_source(node))
                print('node ast:', ast.dump(deepcopy(node)))
                print('expanded_body == macro caller body', ast.dump(node.body[0]) )
                print('--------- macro callee {}-----------'.format(macro_def.func_name))
                print('callee ast:', ast.dump( deepcopy(macro_def.stmts[0])))
                print('callee args:',  macro_def.args         )
                print('callee source:', to_source(deepcopy(macro_def.stmts[0])))

                #is_ast = macro.test(macro_def, ast=True)
                if not is_ast: ret = macro_def.expand(node, expr.args, expanded_body)
                else:          return self._handle_ast_block_macro(node, macro_def, optional_vars=optional_vars, optional_vars_args=optional_vars_args)

                print('result:')
                print(ret, ret.body[0].lineno)
                print(ast.dump(ret))
                print(to_source(ret))

                return ret

        #new_node = ast.With(node.items[0].context_expr, node.items[0].optional_vars, expanded_body)
        new_node = self.generic_visit(node)
        new_node.lineno, new_node.col_offset = node.lineno, node.col_offset
        return new_node


    def visit_Expr(self, node):
        value = node.value
        # if isinstance(value, ast.Call) :
        #     if IS(value.func, ast.Attribute):  print('visit_Expr:', value.func.value.id)
        #     else:                              print('visit_Expr:', value.func.id)
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name) :
                if value.func.id in self.defs:
                    ret = self._handle_call(value, (ExprMacroDef, BlockMacroDef))
                    if isinstance(ret, ast.expr):
                        ret = fix_locations(ast.Expr(ret), node)
                    return ret
                elif value.func.id in preprocess_calls:
                    #filename = os.path.basename(self.filename).split('.',1)[0]
                    redirect_func     = eval(to_source(node))
                    redirect_func(self.filename, self.module.__name__)
                    is_export = redirector[self.module.__name__]['export']
                    base_path = os.path.dirname(self.filename)
                    target_path  = os.path.join(base_path,redirector[self.module.__name__]['target'] )
                    redirector[self.module.__name__]['target'] = target_path

                    if is_export: pass
                    else:         self.tree.body = [ast.Pass()]
                    return        ast.Pass()
                return self.generic_visit(node) # walk further to see whether it's an nested expression
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if  node.func.id in self.defs:
                print('call', node.func.id)
                return self._handle_call(node, ExprMacroDef)

        self.under_call=True # mark for it's a call context
        args           = list(map(self.visit, node.args))
        node.args      = args
        self.under_call=False
        return node

    def visit_FunctionDef(self,node):
        decors = self.is_macro_decorator(node)
        if decors:
            print('==============================================')
            print('---------PROCESSING decorator macro-----------')
            if len(decors) > 1: raise Exception('only support one macro decorator for one function')
            node.decorator_list = []
            return self._handle_ast_decorator_macro(node, self.defs[decors[0].id])

        else:
            macro_decor_calls = self.is_macro_decorator_call(node)
            if macro_decor_calls:
                if len(macro_decor_calls) > 1: raise Exception('only support one macro decorator for one function')
                argstorage = self.argstorage
                for d in macro_decor_calls:
                    argstorage[d.func.id] = {'args':d.args, 'kargs':d.keywords, 'varargs':d.starargs,'kvarargs':d.kwargs}
                macro_def = self.defs[macro_decor_calls[0].func.id]
                return self._handle_ast_decorator_macro(node, macro_def, decor_call=True)
        return self.generic_visit(node)

class MacroImporter(object):
    """
    Import hook for use on `sys.meta_path`, to expand macros on import.  Quite a
    pain without having importlib.
    """
    def __init__(self):
        self._cache = {}
        self.loaded_modules = []


    def find_module(self, name, path=None):
        print('find module: ', name)
        try:
            lastname = name.split('.')[-1]
            self._cache[name] = imp.find_module(lastname, path), path
        except ImportError:  return None
        return self

    def load_module(self, name):
        print('load module:',name, name in sys.modules)
        try:
            (fd, fn, info), path = self._cache[name]
            #print(name, '------------------')
            #print('fd:',fd,'fn:',fn,'info:',info,'path:',path)
        except KeyError:
            # can that happen?
            #print(name, '------------------EXCEPTION')
            #print('fd:',fd,'fn:',fn,'info:',info,'path:',path)
            raise ImportError(name)
        if info[2] == imp.PY_SOURCE:
            newpath = None
            filename = fn
            with fd:
                code = fd.read()
        elif info[2] == imp.PY_COMPILED:
            newpath = None
            filename = fn[:-1]
            with open(filename, 'U') as f:
                code = f.read()
        elif info[2] == imp.PKG_DIRECTORY:
            filename = os.path.join(fn, '__init__.py')
            newpath = [fn]
            with open(filename, 'U') as f:
                code = f.read()
        else:
            return imp.load_module(name, fd, fn, info)

        try:
            module = ModuleType(name)
            module.__file__ = filename

            #print('load file:', filename)
            if newpath:
                module.__path__ = newpath

            try:
                tree = ast.parse(code, filename)
                #print('---------- call Expander for code transformation -----------------')
                #print('visit tree:')
                transformed = Expander(module, debug=name=='domination.gameengine', filename=filename, tree=tree).visit(tree)

                # if transformed.dependencies:
                #     module = ModuleType(name)
                #     module.__file__ = filename
                #     module.__path__ = newpath
                #     tree = ast.parse(code)
                #     exec (code, module.__dict__)

            except MacroCallError as err:
                err.add_filename(filename)
                raise

            except ImportError as err:
                raise "invalid syntax: {}".format(filename)

            #print('transformed code:',filename)
            #print(transformed)
            #print(ast.dump(transformed))
            #print(to_source(transformed))

            #if hook_mode == 'macro':
            ast.fix_missing_locations(transformed)
            code = compile(transformed, filename, 'exec')
            sys.modules[name] = module

            #print('before exec=========', name)
            if hook_mode != 'rapydscript' or macro_forced:
                if macro_forced:
                    print('_____macro_______')
                    print(to_source(transformed))
                exec (code, module.__dict__)
                #print('after exec',name)
                #print('redirector:',redirector)
                #print('\n\nmodule.__dict__ for{}:\n'.format(name) , module.__dict__)
            print('module ',name, 'loaded', 'macro_forced:', macro_forced)

            if name in redirector:
                print('________ FOUND IN REDIRECTOR _______', name)
                print(name in sys.modules)
                rec = redirector[name]
                target = rec['target']
                export = rec['export']
                source = to_source(transformed)
                if export:
                    print('______ export -___________', target)
                    with open(target, 'w') as f:
                        f.write(source)
                else:
                    pth = self._cache[name][1]
                    print('path = ', pth)
                    n_name = os.path.basename(target).split('.',1)[0]
                    self.find_module(n_name, path=pth)
                    module =  self.load_module(n_name)
                    sys.modules[name] = module
            else:
                print(name)
                print(redirector)
                print('___ redirector not found:', name, hook_mode)
            return module
        except Exception as err:
            #print('-----------------------------------------')
            # print('---------------------------------------')
            # print('---------------------------------------')
            # print(to_source(transformed))
            # print(len(transformed.body))
            # for b in transformed.body:
            #     print(b.lineno, to_source(b))
            #     for c in ast.iter_child_nodes(b):
            #         lineno = 'n/a' if not hasattr(c, 'lineno') else c.lineno
            #         print(lineno, to_source(c))
            file = open(filename)

            print('cannot import %s: %s' % (name, err), file=sys.stderr, end="")
            raise ImportError('cannot import %s: %s' % (name, err))





def install_hook(mode='macro', transformer = None):
    """Install the import hook that allows to import modules using macros."""
    global hook_mode, to_source, Expander
    hook_mode = mode
    if mode == 'rapydscript':
        def _to_source(*args):
            return codegen.to_source(*args, strip_annotation=True)
        to_source = _to_source

    if transformer: Expander = transformer
    importer = MacroImporter()
    sys.meta_path.insert(3, importer)
    return importer


def remove_hook():
    """Remove any MacroImporter from `sys.meta_path`."""
    sys.meta_path[:] = [importer for importer in sys.meta_path if
                        not isinstance(importer, MacroImporter)]


def is_outdated(from_file, to_file, outdated=None, duration=100):
    from_file, from_ext = from_file.rsplit('.',1)
    to_file = os.path.abspath(os.path.join(os.path.dirname(from_file), to_file))
    to_file,   to_ext   = to_file.rsplit('.',1)

    stat1 = div = stat2 = 0
    if outdated is None:
        try:
            stat1 = os.stat(from_file+ '.' + from_ext).st_mtime
        except:
            raise FileNotFoundError(from_file+ '.' + from_ext)
        #---------------------------------------------------
        if os.path.exists(to_file + '.' + to_ext ):
            stat2 = os.stat(to_file + '.' + to_ext).st_mtime
        else:
            stat2 = stat1 - duration*2
        #----------------------------------------------------
        div = stat1 - stat2
        if div < duration: outdated = False
        else:              outdated = True
    return outdated


def redirect(to_file=None, export=None):
    def real(from_file, module_name, to_file = to_file, export=export):
        print('redirect')
        print(from_file, to_file, module_name)
        if to_file == 'pyj': to_file = from_file.replace('.py', '.pyj')
        # from_file, from_ext = from_file.rsplit('.',1)
        # to_file = os.path.abspath(os.path.join(os.path.dirname(from_file), to_file))
        # to_file,   to_ext   = to_file.rsplit('.',1)
        #
        # stat1 = div = stat2 = 0
        # if export is None:
        #     duration = 100
        #     try:
        #         stat1 = os.stat(from_file+ '.' + from_ext).st_mtime
        #     except:
        #         raise FileNotFoundError(from_file+ '.' + from_ext)
        #     #---------------------------------------------------
        #     if os.path.exists(to_file + '.' + to_ext ):
        #         stat2 = os.stat(to_file + '.' + to_ext).st_mtime
        #     else:
        #         stat2 = stat1 - duration*2
        #     #----------------------------------------------------
        #     div = stat1 - stat2
        #     if div < duration: export = False
        #     else:              export = True
        export = is_outdated(from_file, to_file, outdated=export, duration=100)
        to_file,   to_ext   = to_file.rsplit('.',1)
        redirector[module_name] = {'target':to_file + '.' + to_ext, 'export': export}
        print(redirector)
        raise Exception
    return real


def redirect2(from_file=None, to_file=None, export=None):
    #print('redirect from {} to {}'.format(from_module, to_module))

    from_file, from_ext = from_file.rsplit('.',1)
    to_file,   to_ext   = to_file.rsplit('.',1)
    stat1 = div = stat2 = 0
    if export is None:
        duration = 100
        try:    stat1 = os.stat(from_file+ '.' + from_ext).st_mtime
        except: raise FileNotFoundError(from_file+ '.' + from_ext)
        #---------------------------------------------------
        if os.path.exists(to_file + '.' + to_ext ):
            stat2 = os.stat(to_file + '.' + to_ext).st_mtime
        else:
            stat2 = stat1 - duration*2
        #----------------------------------------------------
        div = stat1 - stat2
        if div < duration: export = False
        else:              export = True

    from_file = from_file.replace(r'/','.')
    to_file = to_file+'.' + to_ext
    print(from_file, tofi)
    key = from_file if from_file.split('.')[-2] != '__init__' else from_file.split('.',1)[-2]
    redirector[key] = {'target':to_file, 'export': export}

    print('\n'*10)
    print('from:', from_file, 'to:', to_file)
    print('export = ', export, 'div = ', div, 'stat1 = ', stat1, 'stat2 = ', stat2)
    print('\n'*10)
