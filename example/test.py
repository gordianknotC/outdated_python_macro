# always import macros from ``module.__macros__``, after the
# import hook is installed
from python_macro import redirect
from python_macro.example.macros.__macros__ import add, assign, custom_loop, testmacro, first, last
from python_macro.macro_enhancement.__macros__ import trace, EnumType, recursive, loop, ObjectType, do, inline, inlineable, Typedef, obj
from python_macro.macro_enhancement import singleton, EnumElem, EnumElts, Obj, Struct, Object
import sys
from typing import Union, List, Dict, typevar

# relative path from the main module where macro hook installed
redirect( r'example/test.py', r'example/translated_test.py', export=True)


def usage_expr():
    # usage of expression macros (nested calls possible)
    return add(1, 2) + add(3, 4) + add(add(3, 4), 5)

def usage_block():
    # usage of a block macro that does j = 1
    assign(j, 1)
    return j

def usage_3():
    # usage of a block macro with body
    with custom_loop(10):
        print ('loop continues...')


def test():

    with custom_loop(5):
        print('custom loop')

    with testmacro(11):
        print(123123123)
        print('second line')


head = lambda x: x[0]
tail = lambda x: x[1:]

with loop(3,11) as k:
    print('12')

def function(a,b, func):
    print(a,b)
    func(a,b)


with function('a','b') as do[x,y]:
    print('inside do')
    print('params x,y:', x,y)


@inline
def inline_function(a,b):
    print('executing inline function')
    print(sum(a,b))

@inlineable
def inline_host(x,y):
    x += 20 # x
    y += 11 # y
    kargs = dict(a=1,b=2)
    args = [1,2,3]
    print('inline test 1') # turn x,y to a,b
    inline_function(x,y,*args, name='inline_function',**kargs)
    print()
    print('inline test 2')
    inline_function (x+100, y+100)


with ObjectType(TPerson):
    name = 'HAHA'
    sex = "mediam"

with ObjectType(TMan[TPerson]):
    name = 'John'
    sex = 'man'
    age = 21

with EnumType(eDirection) as pure:
    eEast, eWest, eSouth, eNorth = 1,2,3,4

with EnumType(ePerson):
    eOccupation, eCountry, ePostCode


def echo( prefix, asign_str, arg, line_code):
    print(prefix, asign_str, arg, line_code)

@trace( variables=dict(a='',b='',cc=''), returns=True, action = "echo")
def ttt(a=1,b=2,c=3):
    arg = a + b
    argb = b
    argc = c
    if a == 1:
        k = argc
        cc = k
    elif b == 2:
        pass
    else:
        pass
    return True

xx = [5,6,7,8]
arg = first(xx)
arg2 = last(xx)
oo = obj("Steven", 13, 11)
print('+++++++++++++++++', arg, arg2)
print(oo)
print('===== test watch trace ======')
print(ttt())

