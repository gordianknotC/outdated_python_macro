#!/usr/bin/env 
import ast
from .codegen import to_source
import types


def genArguments(*args):
    ret = ast.arguments(args=[ast.Name(id=arg, ctx=ast.Param()) for arg in args])
    return ret

'''execute a ast node, has to be a single function and
no global variable
EX: not allowed for
----------------------
abc=123
def test(body,*args, **kargs): 
    def real(): return 12
    print(123)
    return 111111'''
def execAst(caller_nd, callee_node, *args, strip_decorator=None, filename=None, **kargs):
    print()
    print('=================================')
    print('       astmacro execution        ')
    print('---------------------------------')
    

    if not isinstance(callee_node, ast.Module):
        callee_node = ast.Module(body=[callee_node])
    ret = {}
    
    if len(callee_node.body) >1: 
        raise Exception('only support for executing ast that only has one function without global variables: bodies:{}'.format(callee_node.body))

    if callee_node.body[0].decorator_list:
        decorators = [i.attr.lower() for i in callee_node.body[0].decorator_list]
        if strip_decorator in decorators:
            callee_node.body[0].decorator_list.pop(decorators.index(strip_decorator))
    
    print('caller node:')
    print(to_source(caller_nd))


    print('callee_node:')
    print(to_source(callee_node))
    print('caller body:')
    for b in args[0]:
        print(to_source(b))
    print('caller_args')
    print( args[0], *args[1])
    print('caller_kargs')
    print(kargs)


    co  = compile(callee_node, filename, 'exec')
    names = co.co_names
    exec(co, ret)
    print(names)
   
    print(co.co_names)
    # call macro callee
    if not kargs: ret = ret[co.co_names[0]](args[0], *args[1])
    else:         ret = ret[co.co_names[0]](args[0], *args[1], **kargs)

    ret =  ast.Module(body=ret).body
    if ret == None:
        raise Exception('invalid astMacro usage, return a ast type is required')

    print('final caller body')
    for n in ret:
        print(  to_source(n) )
    # return code to macro caller
    return ret
    #return ast.If(test=ast.Num(n=1), body=ret, orelse=[])



def execMacro(macro_func, *args, filename=None, **kvarargs):
    print()
    print('=================================')
    print('       astmacro execution        ')
    print('---------------------------------')
    ret = {}
    if not kvarargs: ret = macro_func(args[0], *args[1])
    else:            ret = macro_func(args[0], *args[1], **kvarargs)

    _ret =  ret
    if _ret == None:
        raise Exception('invalid astMacro usage, return a ast type is required')

    print('final caller body', isinstance(_ret, list), type(_ret))
    if   isinstance(_ret, list):               pass
    elif isinstance(_ret, types.FunctionType): pass
    else:
        print(_ret)
        print(ast.dump(_ret))
        print(to_source(_ret))
        _ret = [_ret]
    # return code to macro caller
    return _ret
    #return ast.If(test=ast.Num(n=1), body=ret, orelse=[])



if __name__ == '__main__':
    import ast
    from ast import *
    s = """
with function(x,y) as do[a,b]:
    doSomething(a,b)
    """
    node = ast.parse(s)
    print(dump(node))
    print()

    print( [i.id for i in node.body[0].items[0].optional_vars.slice.value.elts] )


