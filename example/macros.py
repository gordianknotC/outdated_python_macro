from python_macro import macro
import sys

@macro.expr
def add(i, qq):
    i + qq

@macro.expr
def assign(n, v):
    n = v

@macro.expr
def first(x):
    x[0]

@macro.expr
def last(x):
    x[-1] if isinstance(x, list) and x else []



@macro.block
def custom_loop(i):
    for __x in range(i):
        print (__x)
        if __x < i-1:
            __body__


@macro.astBlock
def testmacro(body, *args, **kargs):
    print('=================== TETAMACRO ====================== ')
    print('-----------------------------------------------------')
    print('============================================================')
    return body



