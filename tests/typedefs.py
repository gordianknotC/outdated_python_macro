# always import macros from ``module.__macros__``, after the
# import hook is installed
from python_macro import redirect
from python_macro.macro_enhancement.__macros__ import EnumType,   ObjectType, Typedef, trace, head, init, echo, assign
from python_macro.macro_enhancement import singleton, EnumElem, EnumElts, Obj, Struct, FuncDef
from typing import Union, List, Dict, typevar
import cython, os, sys
redirect(to_file = 'translated_typedef.py', export=True)
IS = isinstance
print(os.path.curdir)

some=10

with Typedef():
    Data     = Obj    (name=int, data=str, age=int, u=Union[int,str])  # puthon object
    Db       = Struct (name=int, data=str, age=int, lst=list)          # cython struct
    StrInt   = Union  [int, str] # Mypy union type
    StrOrInt = List   [StrInt]   # Mypy list type
    TAstGet  = FuncDef(name=str, data=str, elts=list, rtype=int)


with EnumType(eColor) as normal:
    eSouth, eNorth, eWest, eEast = range(0,4)

with EnumType(eCountry) as normal:
    eDirection, eRace, eEconomy = range(0,3)

with EnumType(eDirection) as normal:
    eSouth, eNorth, eWest, eEast = range(0,4)


# x = [1,2,23,4,5]
# xx = [x,x,x,x,x,x,x]
# init(x)
# print(init(x))
# print(init(init(xx)))
# print('------------')
# head(x)
# print(head(x))


if __name__ == 'typedefs':
    print('____________ mnain __________________', __name__)
    #@trace(variables=dict(eSouth='._belongs', eNorth='._belongs', eEast='._belongs'), returns=True)
    @trace(variables=dict(eSouth='._belongs', eNorth='._belongs', eEast='._belongs'), returns=True)
    def check():
        print(eSouth in eDirection)
        print(eNorth in eDirection)
        print(eEast in eDirection)
        if eSouth in eDirection:
            print('yes')
        elif eNorth in eDirection:
            print('no')
            print(eSouth)
        else: print(123)
        return True
    check()

    print('woijoewjfoj')

print('______________EWFEFEOFJEFOI')
echo('listAdd', 'listAdd')