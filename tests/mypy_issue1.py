from typing import Dict, List, Union, typevar, Generic, TypeAlias


AnyStr = typevar('AnyStr', values=(str, bytes))




T = typevar('T')

class ABC(object): pass

#StrInt = Union[str, int]

# with TypeDef:
#     StrInt = Union  [int, str]
#     AElts  = Object [name:str, arg:int]
#     AElts  = List   [StrInt]

class EnumElts(list):
    def __init__(self, d:List[Union[int,str, None,'EnumElem']]  ) -> None:
        super(EnumElts, self).__init__(d)

    def add(self, el:'EnumElts'):
          if not el in self: self.append(el)


class EnumElem(object):pass


def func(x:EnumElts  ):
    pass


func( EnumElts([1]) )
func( EnumElts(['str']) )
func( EnumElts([1,'str']) )
func( EnumElts([1,'str',1.23]) )
