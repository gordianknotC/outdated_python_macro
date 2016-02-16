import sys, ast
from ast import *
from random import random
from copy import deepcopy
from python_macro.astutil.codegen import to_source
from python_macro.utils import Transformer, UncaughtFieldException, Tbase, Tnode, Tvalue, Tfield, Object, Seq, EnumElem, Conditions, EnumElts
from python_macro import macro, saver
from typing import overload
del globals()['walk']

TSTRUCT = 'Struct'
TOBJECT = 'Obj'
TUNION = 'Union'
TDICT = 'Dict'
TLIST = 'List'
TFUNC = 'FuncDef'
IS = isinstance

code_registry = {}
inline_conditions = Conditions()
enum_conditions = Conditions()

binop = {
    Add      : "+",
    Sub      : "-",
    Mult     : "*",
    Div      : "/",
    Mod      : "%",
    LShift   : "<<",
    RShift   : ">>",
    BitOr    : "|",
    BitXor   : "^",
    BitAnd   : "&",
    FloorDiv : "//",
    Pow      : "**"
}
unop = {
    Invert   : "~",
    Not      : "not",
    UAdd     : "+",
    USub     : "-"
}
cmpops = {
    Eq       : "==",
    NotEq    : "!=",
    Lt       : "<",
    LtE      : "<=",
    Gt       : ">",
    GtE      : ">=",
    Is       : "is",
    IsNot    : "is not",
    In       : "in",
    NotIn    : "not in"
}
boolops = {
    And      : 'and',
    Or       : 'or'
}

ast_map = dict(
    name      = Name,
    assign    = Assign,
    call      = Call,
    if_       = If,
    attribute = Attribute,
    expr      = Expr
)

objectPtn = '''
class {name}({inher}):
    __slots__ = {slots}
    def __init__(self,**kwargs):
        defaults = {defaults}
        defaults.update(kwargs)
        for k,v in defaults.items():
            setattr(self, k, v)
        
    def __repr__(self):
        ret = {{i:getattr(self,i) for i in self.__slots__}}
        return "{name}({{}})".format(ret)'''

object_ptn = '''
class {{name}}({inher}):
    __slots__ = {slots}
    def __init__(self, {kargs_def}):
        defaults = {defaults}
        for k,v in defaults.items():
            setattr(self, k, v)

    def __repr__(self):
        ret = {{L}}i:getattr(self,i) for i in self.__slots__{{R}}
        return "{{name}}( {{L}}{{R}} )".format(ret)'''

lst_ptn = '''
class {{name}}(list):
    def __init__(self, {arg_def}):
        super().__init__({arg})

    def insert_bulks(self, index, content):
        init = self[:index]
        tail = self[index+1:]
        init.extend(content)
        init.extend(tail)
        self[:] = init
'''

struct_ptn = '''
__struct = cython.struct({args})
{{name}} = cython.declare(__struct)
'''

funcdef_ptn = '''
def {name}({kargs_def}) -> {rtype}: pass
'''

ptns = dict(List=lst_ptn, Obj=object_ptn, Struct=struct_ptn, FuncDef=funcdef_ptn)




class ContextualTransformer(NodeTransformer):
    def __init__(self, transformer):
        self.contextual_stack = [None,None]
        self.context = []
        self.result_stack = []
        self.transformer = transformer.__class__

    def get_line(self, n:int):
        # fetch current line: get_line(-1)
        # fetch last line:    get_line(-2)
        index = self.contextual_stack.index(self.contextual_stack[n])
        return self.contextual_stack[n], index

    def sequence_node(self, current_line_node, node):
        ret = []
        def walk_field(child, match_node):
            for i,rec in enumerate(iter_fields(child)):
                fieldname, value = rec
                if not value == match_node:
                    if isinstance(value, list):
                        for v in value:
                            ret.extend(walk_field(v, match_node))
                    else: ret.append(value)
                else: return ret # found match node
            return ret # feed ret back to recursive call
        node_stack = walk_field(current_line_node, node)
        return node_stack

    def contextual_visit(self,node):
        print()
        print('contextual visit::')
        self.context = node
        self.before_result = []
        self.result_stack = []
        transformer = self.transformer
        ret = []
        if isinstance(node, If):
            node.test = self.generic_visit(node.test)
            body = transformer().contextual_visit(node.body)
            if node.orelse:
                orelse = transformer().contextual_visit(node.orelse[0])
                if isinstance(node.orelse[0], If): node.orelse = [orelse]
                else:                              node.orelse = orelse
            body = self.result_stack + body
            node.body = body
            return node
        elif isinstance(node, list): body = node
        elif hasattr(node, 'body'): body = node.body
        else:body = [node]

        for bd in body:
            if not bd: continue
            self.contextual_stack.append( bd )
            if isinstance(bd, If):
                bd      = transformer().contextual_visit(bd)
                ret.append(bd)

            elif hasattr(bd, 'body'):
                bd = transformer().contextual_visit(bd)
                ret.append( bd )
                ret.extend(self.result_stack)
                self.result_stack = []

            else:
                self.contextual_stack.append(bd )
                if not isinstance(bd, Return):
                    ret.append(self.generic_visit(bd))
                    ret.extend(self.result_stack)
                else:
                    bd = self.visit_Return(bd)
                    ret.extend(self.result_stack)
                    ret.append(bd)

                self.result_stack = []

        if hasattr(node, 'body'):
            node.body = ret
            return node
        else: return ret



#------------------------------------------
#       AST Utilities
#------------------------------------------
def singleton(args=None, kargs=None, target=None):
    args = args or []
    kargs = kargs or {}
    def real(cls):
        o = cls(*args, **kargs)
        if hasattr(o, '__name__'): o.__name__ = cls.__name__
        setattr(sys.modules[cls.__module__ ], target, o)
        return o
    return real

def globAst(body, conditions=None, do=None):
    if IS(body, list):
        for i in body:
            if conditions(i): 
                print('match condition', i)
                do(i)
            else:
                print('condition mismatched', i)
                globAst(i, conditions, do)
    else:
        if hasattr(body, '__dict__'):
            for k,v in body.__dict__.items():
                if not k in ['lineno', 'col_offset']:
                    globAst(v, conditions, do)


def getArgs(n):
    if isinstance(n, ast.FunctionDef):
        args      = [i for i in n.args.args]
        vararg    = n.args.vararg 
        konly     = n.args.kwonlyargs 
        kdefaults = n.args.kw_defaults 
        karg      = n.args.kwarg 
        if args   : args   = dict(args= [i.arg for i in args],  annotations=[i.annotation for i in args],  values='')
        if vararg : vararg = dict(arg = [vararg.arg],           annotation =[vararg.annotation])
        if konly  : konly  = dict(args= [i.arg for i in konly], annotations=[i.annotation for i in konly], values= [i.value for i in kdefaults])
        if karg   : karg   = dict(arg = [karg.arg],             annotation =[karg.annotation])
        
        ret = dict(args=args, vararg=vararg, kargs=konly, kvararg=karg)
        return ret

    elif isinstance(n, ast.Call):
        args     = [i for i in n.args]
        vararg   = n.starargs
        keywords = [i for i in n.keywords]
        kvararg  = n.kwargs
        if args     : args     = [ getValue(i) for i in args]
        if keywords : keywords = [ [i.arg, i.value] for i in keywords]
        if vararg   : vararg   = vararg.id
        if kvararg  : kvararg  = kvararg.id
        return dict(args=args, kargs = keywords, vararg=vararg, kvararg=kvararg)

    else:
        if isinstance(n, ast.Module):
            print()
            print('--------------- getargs ----------------------')
            print(ast.dump(n))
            print(to_source(n))
            if len(n.body) > 1: raise Exception('getvars only support for FunctionDef, With and Module encapsulate with one FunctionDef/With')
            return getArgs(n.body[0])
        else:
            raise Exception('Uncaught usage exception, only support fetching args from Moudle/FunctionDef/Call ')

def decodeSlice(body_list):
    if IS(body_list[0], Subscript):
           return [ [bd.value, bd.slice] for bd in body_list]
    else:  return [ [bd,None] for bd in body_list]

def getValue(_x, exception=None):
    value = _x
    if   IS(value, Num)   : return value.n
    elif IS(value, Str)   : return value.s
    elif IS(value, List)  : return [getValue(e) for e in value.elts]
    elif IS(value, Tuple)  : return [getValue(e) for e in value.elts]
    elif IS(value, Dict)  : return { getValue(i[0]): getValue(i[1]) for i in zip(value.keys, value.values)}
    elif IS(value, Call)  : return value
    elif IS(value, Name)  : return value.id
    elif IS(value, BinOp) : return [getValue(value.left)] + [getValue(value.op,exception='binop')] +  [getValue(value.right) ]
    elif IS(value, Subscript): return [getValue(value.value), getValue(value.slice.value) ]
    elif IS(value, Assign):
        #---------------------------------------------------------------------
        targets =_x.targets
        if   IS(targets[0], ast.Name):  return targets[0].id, getValue(_x.value)
        elif IS(targets[0], ast.Tuple):
            if not IS(_x.value, Call):  return [[e[0].id, getValue(e[1])] for e in zip (targets[0].elts, _x.value.elts) ]
            else:                       return [e.id for e in targets[0].elts ], to_source(_x.value)
        else:                           raise  Exception('Uncaught exception')
        #---------------------------------------------------------------------
    else:
        if  exception == 'binop':  return binop[ type(_x) ]
        else:                      raise Exception('Uncaught exception', _x)


                       

def walk_field(child, conditions, do, stack,capture_field, parent_order):
    for rec in ast.iter_fields(child):
        fieldname, value = rec
        if capture_field(fieldname, value):
            print('       field', fieldname, value, stack[-1])
            do(stack[0], [fieldname, value], stack, mode = 'field' , parent_order = parent_order)
        else:
            if IS(value, list):
                for v in value:
                    walk_field(v, conditions, do, stack,capture_field,  parent_order)
            elif hasattr(value, '_fields'):
                walk_field(value, conditions, do , stack,capture_field, parent_order=parent_order)

def walk(node, conditions=None, do=None, stack=None):
    ''' stack[0]: captured_parent
        stack[1]: tree
        stack[2:]: stacked nodes        '''
    capture_current, capture_parent, capture_field = conditions
    if not IS(node, list): nodes = [node]
    else:                  nodes = node

    if not stack: stack = [None, node]
    else:         stack.append(node)

    for i, nd in enumerate(nodes):
        if capture_current(nd):
            print('node1')
            stack.append(nd)
            if not do(None, nd, stack, mode = 'node', parent_order = i):
                walk_field(nd, conditions, do, stack ,capture_field, parent_order= i)
        else:
            print('node2')
            if capture_parent(nd):
                        stack[0] = nd
            for child in ast.iter_child_nodes(nd):
                if capture_current(child):
                    stack.append(child)
                    if not do(stack[0], child, stack, mode = 'node', parent_order =i):
                        walk_field(child, conditions, do, stack,capture_field, i)
                else:
                    # if capture_parent(child):
                    #     stack[0] = child
                    walk(child, conditions, do, stack)
                          
   
def getBlockContentByFilter(body, conditions) :
    ret = []
    do = lambda x: ret.append(getValue(x))
    globAst(body, conditions , do)
    return ret

def getBlockArgs(block_body: ast.withitem) -> [ast.Subscript or ast.arguments]:
    args = block_body.items[0].context_expr.args
    return args


    
def safeReturn(obj):
    if IS(obj,str):
        return ast.parse(obj).body
    raise Exception ('Uncaught Exception')


def uniqueList(lst):
    seen = set()
    return [i for i in lst if not i in seen and not seen.add(i)]

def flattenList(lst, ret=None):
    if IS(lst, list):
        for l in lst:
            flattenList(l, ret)
    else:   ret.append(lst)


def replaceArgumentId(source_nodes, source_values, target_values):
    sources_targets = positionZip(source_values, target_values)
    #========BUG===========
    # need further test
    t = Transformer()
    for rec in sources_targets:
        from_source, to_target = rec
        funcdef_arg, call_arg = from_source, to_target
        t.FromCallArgsId = funcdef_arg.arg
        t.ToCallArgsId   = call_arg
        print('_________++++++++++++++', dump(funcdef_arg))
        print('_________++++++++++++++', call_arg)
    t.transform(source_nodes)

def flattenArguments(args, ret):
    flattenList( [i for i in args['args']] + [i[0] for i in args['kargs']] +
                          [args['vararg']]          + [args['kvararg']]   , ret)

def positionZip(a,b):
    return [ [ia,ib] for _ia,ia in enumerate(a) for _ib,ib in enumerate(b) if _ia == _ib]

def getArgsAndKargsFromNodes(nodes):
    args     = getArgs(nodes)
    callargs = []
    flattenArguments(args, callargs)
    if callargs:
        callargs = [ i for i in callargs if IS(i, str)]
        callargs = [ i for i in callargs if i[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstvuwxyz_' ]
        callargs = uniqueList(callargs)
    return callargs





#-----------------------------------------
#       copy from macropy enhancement
#-----------------------------------------
def genCall(funcname, args, keywords=[],starargs=None, kwargs=None):
    '''
    with function(a,b=1,c=2, *args, **kargs): pass
    ------------------------------------------------
    Call(
    func        =  Name(id='function', ctx=Load()),
    args        = [Name(id='a', ctx=Load())],
    keywords    = [keyword(arg ='b', value=Num(n=1)),
                   keyword(arg ='c', value=Num(n=2))],
    starargs    = Name(id='args',  ctx=Load()),
    kwargs      = Name(id='kargs', ctx=Load()))
    '''
    ret =  Call(func=Name(id=funcname, ctx=Load()) , args=args , keywords=keywords ,starargs=starargs, kwargs=kwargs)
    print(to_source(ret))
    return ret

def genAssignment(varname:str ,value) ->Assign:
    keywords    = []
    starargs    = None
    kwargs        = None
    if     IS(varname, str):     varname = [Name(id=varname, ctx=Store())]
    if not IS(varname, list):     varname = [varname]
    result = Assign(targets=varname, value=value, keywords=keywords, starargs=starargs,kwargs=kwargs)
    return result

def genFuncDef(name:str='',         args:[Name]=[], bodies:[stmt]=[Pass()], defaults=[], decorator=[], 
               kwonlyargs:[arg]=[], kw_defaults=[], kwarg:arg=None,         vararg:arg=None) -> FunctionDef :
    ''' ex node:
    def function(a,b,c=1,d=2,*args,e = 3, f= 4,**kargs): pass
    --------------------------------------------------------------
    FunctionDef(
    name           = 'function',
    args           = arguments(args=[arg(arg='a', annotation=None), 
                                     arg(arg='b', annotation=None), 
                                     arg(arg='c', annotation=None), 
                                     arg(arg='d', annotation=None)],
                                vararg      = arg(arg='args', annotation=None),
                                kwonlyargs  = [arg(arg='e', annotation=None),
                                arg(arg     = 'f', annotation=None)],
                                kw_defaults = [Num(n=3),
                                Num(n       = 4)],
                                kwarg       = arg(arg='kargs', annotation=None),
                                defaults    = [Num(n=1),
                                               Num(n=2)]),
    body           = [Pass()],
    decorator_list = [],
    returns        = None)])    '''
    nnkArgs    = arguments(args = [arg(arg=a.id, annotation=None) for a in args],
                           vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, 
                           kwarg=kwarg, defaults=defaults)
    nnkFuncDef = FunctionDef(name=name,args=nnkArgs, decorator_list=decorator, body=bodies    )
    return nnkFuncDef

def genResult(ret): pass


#---------------------------------------------
#           MACROS entry
#---------------------------------------------

@macro.astBlock
def saveCode(body, *args, **kargs):
    key = kargs['caller_node'].name
    saver(key, body)
    return body

def getRegisteredCode(key):
    if hasattr(code_registry,key):
        func     = code_registry[key]
        funcbd   = func['body']
        funcargs = func['args']
        return Object(body=func['body'], args=func['args'])
    else:
        return None





'''
=========================================================
            Macro definitions
--------------------------------------------------------- '''
@macro.expr
def assign(a,b):
    a = b

@macro.expr
def add(i, qq):
    i + qq



@macro.astBlock
def recursive(body, *args, **kargs):
    """ not implement yet
    ---------------------------
    try to translate from recursive call to while loop """
    print(123123)
    return body



@macro.astBlock
def inline(body, *args, **kargs):
    module     = kargs['module']
    callernode = kargs['caller_node']
    code_registry[module+callernode.name] = dict(body=body, args=args, kargs=kargs)
    return body


@macro.astBlock
def inlineable(body, *arg, **kargs)->AST:
    ret             = []
    capture_current = lambda node: IS(node, Call) and IS(node.func, Name)
    #capture_field   = lambda fieldname, value: fieldname == 'id'
    capture_parent  = lambda child: hasattr(child, 'body')
    capture_inline_field = lambda f,v: f == 'id'
    conditions      = [capture_current, capture_parent, lambda x,y: False]
    module          = __name__

    def do(parent, cur, stack, mode=None, parent_order=None):
        parent = stack[0]
        tree   = stack[1]
        stacks = stack[2:]
        if mode == 'node':
            inline_func = kargs['module'] + cur.func.id
            codes       = getRegisteredCode(inline_func)
            if codes:
                funcbd   = codes.body
                funcargs = codes.args
                callargs = getArgsAndKargsFromNodes(cur)
                sources_targets = positionZip(funcargs, callargs)
                #========BUG===========
                # need further test
                replaceArgumentId(funcbd, funcargs, callargs)
                #---------- insert function body------------
                if parent:  source = Seq(parent)
                else:       source = Seq(tree)
                source.insert_bulks(parent_order, funcbd)
                return True


    walk(body, conditions, do, stack=[None, body])
    print('final inlineable body:')
    print(body)
    return body




#=======================================================
#               Python TypeDef enhancement
#--------------------------------------------------------
class Struct(object):
    def __init__( self, **kargs): pass

class Obj(object):
    def __init__(self, **kargs): pass

class FuncDef(object):
    def __init__(self, **kargs):pass

class TypeGen(object):
    def parse_typ(self, data, rest=False):
        head = lambda x: x[0]
        tail = lambda x: [] if len(x) == 1 else x[1:]
        if not data: return data
        parse_typ = self.parse_typ
        x,xs = head(data), tail(data)
        if not IS(x, list):
            if len(xs) == 1:
                if not x in [TUNION, TLIST, TDICT, 'Tuple']:
                        # not mypy types
                        return str(x) + ':' + parse_typ(xs)
                else:
                        # Mypy types...
                        ret = x + '['
                        for i in xs[0]:
                            ret += parse_typ([i] ) + ','
                        ret = ret.strip(',')
                        ret += ']'
                        return ret
            elif not xs:return x
        else:           return parse_typ(x)

    def __init__(self, _data):
        repr = ''

        for data in _data:
            typ, defs = data
            if IS(defs, Call):
                name = defs.func.id
                args = getArgs(defs)['kargs']
                defs = [ [arg[0], getValue(arg[1])] for arg in args]
            else:
                name= defs[0]
                defs = defs[1]

            print(typ, name)
            if name in [TLIST, TDICT, TUNION,TOBJECT, TSTRUCT, TFUNC]:
                self.mode = 'defs'; self.name = name
                if name in [TOBJECT, TFUNC]:
                    ptn = ptns[name]
                    args = []
                    parse_typ = self.parse_typ
                    for _def in defs:
                       args.append(parse_typ(_def))

                    slots = [ i[0] for i in defs]; print(slots)
                    kargs = ','.join([i.split(':')[0] +'=' + i.split(':')[0] for i in args]); print(kargs)

                    if name == TOBJECT:
                        kargs_def = ','.join([i for i in args]); print(kargs_def)
                        defaults = 'dict({kargs})'.format(kargs=kargs); print(defaults)
                        repr = ptn.format( inher='object', slots=slots, kargs_def=kargs_def,defaults=defaults)
                    elif name == TFUNC:
                        kargs_def = ','.join([i for i in args[:-1]]); print(kargs_def)
                        rtype = args[-1]
                        repr = ptn.format( name=typ, kargs_def=kargs_def, rtype=rtype.split(':',1)[1] )

                else:
                    slice = str(defs).strip('[]')
                    repr = '{{name}} = {name}[{slice}]'.format(name=name, slice=slice)
                    if name == TLIST:
                        ptn = ptns[name]
                        repr = ptn.format( arg_def='data:'+repr.split('=')[1], arg='data')

                    elif name == TSTRUCT:
                        ptn = ptns[name]
                        repr = ptn.format(args=','.join( [ '='.join(a) for a in defs] ) )
            self.repr = repr

    def __repr__(self):
        return self.repr
    def __str__(self):
        return self.repr


@macro.astBlock
def Typedef(body, *args, **kargs):
    capture_current = lambda x: IS(x, Assign)
    capture_parent = lambda x: IS(x, With)
    capture_field = lambda f,v: False
    ret = []
    print('==================================================================')
    print('TypeDef')
    def do(parent, cur, stack, mode=None, parent_order=None):
        parent = stack[0]
        tree   = stack[1]
        stacks = stack[2:]
        last_node = stack[-1]
        print('typedef do')
        if mode == 'node':
            print('-------------------------------')
            print('found type def assignment')
            if IS(last_node.targets[0], Tuple):
                typs = [getValue(i) for i in last_node.targets[0].elts]
            else:
                typs = [getValue(i) for i in last_node.targets]

            if len(typs) >1:  defs = [ getValue(i) for i in last_node.value.elts]
            else:             defs = [ getValue(last_node.value) ]

            ret.append([i for i in zip(typs, defs)])
            return True

    conditions = [capture_current, capture_parent, capture_field]
    walk(body, conditions, do, stack=[None, body])
    output = []
    for rec in ret:
        typname = rec[0][0]
        typdef  = str(TypeGen(rec)).format(name = typname, L='{',R='}')
        output.append(typdef)

    output = '\n'.join(output)
    output = parse(output).body
    print()
    print('--------------- output ===============')
    print(to_source(output))
    print('00000000000000000000')
    return output



@macro.astBlock
def ObjectType(body, *args, **kargs):
    optional_vars      = kargs['optional_vars']
    optional_subscript = kargs['optional_subscript']
    call_args          = optional_subscript
    _name, _inher      = decodeSlice(args)[0]
    inher              = _inher.value.id if _inher else 'object'
    name               = _name.id
    assignments        = getBlockContentByFilter(body, lambda x:IS(x, ast.Assign) )
    tmp                = []
    for i,d  in enumerate(assignments):
        if IS(d, list) and IS(d[0], list):
            assignments[i] = ''
            tmp.extend(d)
        elif IS(d[1], Call):  assignments[i] = (d[0], to_source(d[1]))
        elif IS(d[1], str):   assignments[i] = (d[0], '"' + d[1] + '"')
    assignments += tmp
    defaults = {k[0]:k[1] for k in assignments if k}
    slots    = list(defaults.keys())
    #flatten = ','.join([ str(i[0]) for i in assignments if i] ) + '=' + \
    #          ','.join( str(i[1]) for i in assignments if i)
    obj      = objectPtn.format(name=name, inher=inher, slots=slots,  defaults=defaults)
    return safeReturn(obj)


#=============================================
#               EnumType
#============================================
@macro.astBlock 
def EnumType(body, *args, **kargs):
    enumgrp_ptn   = '''{gname} = EnumElem("{gname}")'''
    enum_ptn      = '''{ename} = EnumElem("{ename}", belong=EnumElts([{belong},{value}]), value={value}, mode="{mode}") '''
    optional_vars = kargs['optional_vars']
    capture_current = lambda node: IS(node, Assign) or IS(node, Tuple)
    capture_field   = lambda fieldname, value: fieldname == 'id'
    capture_parent  = lambda node: hasattr(node, 'body')
    conditions      = [capture_current, capture_parent, capture_field]
    ret             = []

    def do(parent, cur, stack, mode=None, parent_order=None):
        parent = stack[0]
        tree   = stack[1]
        stacks = stack[2:]
        last_node = stack[-1]
        if mode == 'field':  pass
        else:
            if IS(cur, Assign):
                targets = [getValue(i) for i in last_node.targets[0].elts]
                if len(targets) > 1:
                    if IS(last_node.value, Call):
                           values = [ to_source(Subscript(value=last_node.value,slice=Index(value=Num(n=i)))) for i,v in enumerate(targets)]
                    else:  values = [ getValue(i) for i in last_node.value.elts]
                else:      values = [ getValue(last_node.value) ]
                print(targets)
                print(values)
                ret.append([i for i in zip(targets, values)])
                return True
            elif IS(cur, Tuple):
                targets = [ getValue(i) for i in last_node.elts]
                values = list(range(len(targets)))
                print(targets)
                print(values)
                ret.append([i for i in zip(targets, values)])
                return True
            return False

    walk(body, conditions, do, stack=[None, body])
    belong = gname = args[0].id
    mode   = optional_vars
    nbody  = []
    nd     = safeReturn(enumgrp_ptn.format(gname = gname))[0]
    nbody.append(nd)
    for rec in ret:
        for elts in rec:
            ename, value = elts
            code = enum_ptn.format(ename =ename, belong=belong, value=value, mode=mode)
            nd   = safeReturn(code)[0]
            print('code', code)
            nbody.append(nd)
    return nbody


@macro.astBlock
def do(body, *args, **kargs):
    ''' do notation borrowed from Nim programming language
    ---------------------------------------------------------
    translate from:
    with arbtiraryFunction(*args_withoutar_kargs) as do[x,y]:
            do something herer...... body section    
    to:
    def do(x,y):
        do something herer...... body section 
    arbtiraryFunction(arg1, arg2.... , do)    '''

    print('args:', args)
    print('kargs:', kargs)
    caller_node        = kargs['caller_node']        # func node   => with func() as do[]:
    caller_func        = kargs['caller_func']        # func object => with func() as do[]:
    optional_vars      = kargs['optional_vars']      # do
    optional_vars_args = kargs['optional_vars_args'] # do[x,y]
    kwargs             = kargs['kargs']
    args               = [a for a in args]           # source is a tuple
    # @todo
    # 1) generate do function:
    # def do_domethine(x,y): put body here.....
    # 2) make call
    # function(args[0], args[1], do_something )
    func_name     = 'do_notation_line_' + str(caller_node.lineno+1)
    func_args     = [Name(id=i) for i in optional_vars_args]
    func_defaults = []
    func_body     = body
    func_decor    = []
    funcdef       = genFuncDef(name=func_name, args=func_args, bodies=func_body, decorator=func_decor )
    call_name     = caller_node.items[0].context_expr.func.id if IS(caller_node, With) else caller_node.name

    args.append(Name(id=func_name, ctx=Load()))
    call  = genCall(call_name, args, keywords=kwargs)
    obody = deepcopy(body)
    body  = [deepcopy(funcdef), Expr(value=call)]
    return body

@macro.astBlock
def nim_do(body, *args, **kargs):
    '''         not implement yet
    ------------------------------------------------------
    translate from:
    @ffi.callback( callback_from_nim('arbtiraryFunction') )
    with arbtiraryFunction(*args_withoutar_kargs) as do[x,y]:
            do something herer...... body section
    to:
    @ffi.callback( callback_from_nim('arbtiraryFunction') )
    def do(x,y):
        do something herer...... body section

    arbtiraryFunction(arg1, arg2.... , do)    '''
    pass



# @macro.astBlock
# def loop(body, L, R, optional_vars=None, optional_subscript=None, **kargs):
#     print('L,R,optional......',L,R,optional_vars)
#     return body

@macro.block
def loop(l, r, inc, optional_var):
    ''' with loop(2,10) as index:
            print(index)
    '''
    optional_var = l
    while optional_var < r:
        __body__
        optional_var += inc


'''
----------------------------------------------------------
borrowed from functional programming language: Haskell
----------------------------------------------------------'''
@macro.expr
def head(x):
    x[0]

@macro.expr
def tail(x):
    x[1:] if IS(x, list) and x else []

@macro.expr
def last(x):
    x[-1]

@macro.expr
def init(x):
    x[:-1] if IS(x, list) and x else []





@macro.expr
def obj(name,age,value):
    Object(**{'name':name,'age':age,'value':value})

# @macro.contextExpr
# def info(msg, **kargs):
#     funcname = kargs['funcnmae']
#     modulename = kargs['modulename']
#     return code





@macro.astBlock
def trace(body, *args, **kargs):
    # watch function "variables" and throw into provided "actions" with
    # optional informations like module name and ...
    # action could be another macro or a normal functions
    variables       = kargs['kargs'].get('variables') # ast List object
    if not variables: return body

    caller          = kargs['caller_node']
    caller_func     = kargs['caller_func']
    action          = kargs['kargs'].get('action') or Str(s='print')
    returns         = kargs['kargs'].get('returns')

    ret = {}
    variables = eval(to_source(variables))
    try:
        if isinstance(variables, list) or isinstance(variables, tuple):
            assert len(variables[0]) == 2
            variables = dict(variables)
    except:           raise Exception('trace variables only allows for [str, str, ...] object')

    print(dump(caller))
    module_name     = kargs['module']
    capture_current = lambda node: IS(node, Name) or IS(node, Return) if returns else IS(node,Name)
    capture_field   = lambda fieldname, value: False
    capture_parent  = lambda node: IS(node, Expr) or IS(node, Assign) or IS(node, IfExp) or IS(node, If)
    conditions      = [capture_current, capture_parent, capture_field]
    ret             = []

    print('module_name', module_name)
    print('*args', args)
    print('**kargs', kargs)
    print('variables == ', variables)
    print('action == ', action)
    def prints(argname, eval_name, cast_value, lineno, line_code):
        prefix = ''
        prefix += '"[{:15}({})]"'.format(module_name[:15], str(lineno)[-4:])
        output = action.s + '(' + prefix+', "{} =" ,{}'.format(argname,eval_name)  + ',' + str('"@'+line_code+'"') + ' )'
        return output


    class TraceVisitor(ContextualTransformer):
        def __init__(self):
            super(TraceVisitor, self).__init__(self)

        def visit_Name(self,node):
            if node.id in variables.keys():
                cast_value = node.id
                parent, id = self.get_line(-2)
                if id == 0 and isinstance(self.context, If):
                    line_code = self.context
                    line_code = to_source(line_code).split('\n')[0]
                elif hasattr(parent, 'body'):
                    line_code = parent
                    line_code = to_source(line_code).split('\n')[0]
                else:
                    line_code = to_source( self.get_line(-1)[0] ).split('\n')[0]
                if line_code:
                    eval_name = node.id + variables[node.id]
                    output = parse(prints(node.id,eval_name , cast_value, node.lineno, line_code)).body[0]
                    output.lineno = node.lineno + 1
                    self.result_stack.append( output)
                    return node
            return node

        def visit_Return(self, node):
            if returns:
                value =  to_source(node.value)
                lineno = node.lineno -1
                line_code = to_source(node)
                output = prints(value, value, value, lineno+1, line_code )
                output = parse(output).body[0]
                output.lineno = node.lineno-1
                self.result_stack.append(output)
                return node
            else: return node

    v = TraceVisitor()
    nbody = v.contextual_visit(body) # body[0] to switch back from module to body
    print('======================= RRRRRRRR+============')
    print(to_source(nbody))
    return nbody




# -----------------------------------------
# astExpr not implement yet
# usage:
# echo (result = [x for x in lst if x])
# translate to:
# result = [x for x in lst if x]
# print("result = [x for x in lst if x]", result)
@macro.astExpr
def echo(body, *args, **kargs):
    '''caller body is always empty as there is no caller body in expr macro'''
    print('echoMacro.... args = ',args)
    return Pass()






'''
# for performance enhancement
with loop(2,10) as i:
    pass

# always faster than
for i in range(2,10):
    pass

# experimental usage (not implement yet)
wtih loop(2,10

--------------------------------
with div() as DarkStyle:
    pos = 12
    size= 11
    def on_change():
        pass

with DarkStyle():
'''











