from python_macro import redirect
from python_macro.macro_enhancement import singleton, EnumElem, EnumElts, Obj, Struct, FuncDef
from typing import Union, List, Dict, typevar
import cython, os, sys
pass
IS = isinstance
print(os.path.curdir)
some = 10


class Data(object):
    __slots__ = ['name', 'data', 'age', 'u']

    def __init__(self, name:int, data:str, age:int, u:Union[(int, str)]):
        defaults = dict(name=name, data=data, age=age, u=u)
        for (k, v) in defaults.items():
            setattr(self, k, v)

    def __repr__(self):
        ret = {i: getattr(self, i) for i in self.__slots__}
        return 'Data( {} )'.format(ret)
__struct = cython.struct(name=int, data=str, age=int, lst=list)
Db = cython.declare(__struct)
StrInt = Union[('int', 'str')]


class StrOrInt(list):

    def __init__(self, data:List[StrInt]):
        super().__init__(data)

    def insert_bulks(self, index, content):
        init = self[:index]
        tail = self[index + 1:]
        init.extend(content)
        init.extend(tail)
        self[:] = init

def TAstGet(name:str, data:str, elts:list):
    pass
eColor = EnumElem('eColor')
eSouth = EnumElem('eSouth', belong=EnumElts([eColor, range(0, 4)[0]]), value=range(0, 4)[0], mode='normal')
eNorth = EnumElem('eNorth', belong=EnumElts([eColor, range(0, 4)[1]]), value=range(0, 4)[1], mode='normal')
eWest = EnumElem('eWest', belong=EnumElts([eColor, range(0, 4)[2]]), value=range(0, 4)[2], mode='normal')
eEast = EnumElem('eEast', belong=EnumElts([eColor, range(0, 4)[3]]), value=range(0, 4)[3], mode='normal')
eCountry = EnumElem('eCountry')
eDirection = EnumElem('eDirection', belong=EnumElts([eCountry, range(0, 3)[0]]), value=range(0, 3)[0], mode='normal')
eRace = EnumElem('eRace', belong=EnumElts([eCountry, range(0, 3)[1]]), value=range(0, 3)[1], mode='normal')
eEconomy = EnumElem('eEconomy', belong=EnumElts([eCountry, range(0, 3)[2]]), value=range(0, 3)[2], mode='normal')
eDirection = EnumElem('eDirection')
eSouth = EnumElem('eSouth', belong=EnumElts([eDirection, range(0, 4)[0]]), value=range(0, 4)[0], mode='normal')
eNorth = EnumElem('eNorth', belong=EnumElts([eDirection, range(0, 4)[1]]), value=range(0, 4)[1], mode='normal')
eWest = EnumElem('eWest', belong=EnumElts([eDirection, range(0, 4)[2]]), value=range(0, 4)[2], mode='normal')
eEast = EnumElem('eEast', belong=EnumElts([eDirection, range(0, 4)[3]]), value=range(0, 4)[3], mode='normal')
if (__name__ == 'typedefs'):
    print('____________ mnain __________________', __name__)

    def check():
        print((eSouth in eDirection))
        print('[typedefs       (47)]', 'eSouth =', eSouth._belongs, '@print((eSouth in eDirection))')
        print((eNorth in eDirection))
        print('[typedefs       (48)]', 'eNorth =', eNorth._belongs, '@print((eNorth in eDirection))')
        print((eEast in eDirection))
        print('[typedefs       (49)]', 'eEast =', eEast._belongs, '@print((eEast in eDirection))')
        if (eSouth in eDirection):
            print('[typedefs       (50)]', 'eSouth =', eSouth._belongs, '@if (eSouth in eDirection):')
            print('yes')
        elif (eNorth in eDirection):
            print('[typedefs       (52)]', 'eNorth =', eNorth._belongs, '@if (eNorth in eDirection):')
            print('no')
            print(eSouth)
            print('[typedefs       (54)]', 'eSouth =', eSouth._belongs, '@print(eSouth)')
        else:
            print(123)
        print('[typedefs       (56)]', 'True =', True, '@return True')
        return True
    check()
    print('woijoewjfoj')
else:
    pass
print('______________EWFEFEOFJEFOI')
pass