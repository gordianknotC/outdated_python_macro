#!/usr/bin/python
# -*- coding: utf-8 -*-
from utils import *
import ast
from ast import *
from nose.tools import assert_equal, assert_raises
#!/usr/bin/python
# -*- coding: utf-8 -*-
from python_macro.utils import *
import ast
from ast import *
from nose.tools import assert_equal, assert_raises



def test_duplicate_enum():
    eDirection = EnumElem('eDirection')
    eEast = EnumElem( 'eEast',belong=eDirection, value= 1, mode='pure')
    eSouth = EnumElem('eSouth', belong=eDirection, value= 2, mode='pure')

    #------------------------------------------
    # test for duplicate elements
    def test_exception():
        # dupliacate value
        eSouth2 = EnumElem('eSouth', belong=eDirection, value= 2, mode='pure')

    def test_exception2():
        # dupliacate value
        eSouth = EnumElem('eSouth', belong=eDirection, value= 3, mode='pure')
    assert_raises(DuplicateEnumException, test_exception)
    assert_raises(DuplicateEnumException, test_exception2)
    EnumElem.clearRegistry(all=True)
#
# def test_pure_mode_enum():
#     eDirection = EnumElem('eDirection')
#     eEast = EnumElem( 'eEast',belong=eDirection, value= 1, mode='pure')
#     eSouth = EnumElem('eSouth', belong=eDirection, value= 2, mode='pure')
#
#     #------------------------------------------
#     # test for pure mode
#     eCountry = EnumElem('eCountry', belong=eEast, value=1, mode = 'pure')
#     assert_equal(eCountry in eEast, False)
#     assert_equal(eEast.eCountry in eEast, True)
#     EnumElem.clearRegistry(all=True)
#
#
# def test_normal_mode_enum():
#     eDirection = EnumElem('eDirection')
#     eEast = EnumElem( 'eEast',belong=eDirection, value= 1, mode='pure')
#     eSouth = EnumElem('eSouth', belong=eDirection, value= 2, mode='pure')
#     assert_equal(eEast in eDirection, True)
#     assert_equal(eSouth in eDirection, True)
#     EnumElem.clearRegistry(all=True)
#
# def test_nested_enum():
#     eWest       = EnumElem('eWest', mode='pure')
#     eCountry    = EnumElem('eCountry', belong=eWest, value=1, mode='None')
#     eReligion   = EnumElem('eReligion', belong=eWest, value=2, mode='None')
#
#     eDirection  = EnumElem('eDirection')
#     eEast       = EnumElem('eEast', belong=eDirection, value=1, mode='pure')
#     eWest       = EnumElem('eWest', belong=eDirection, value=2, mode='pure')
#     eSouth      = EnumElem('eSouth', belong=eDirection, value=3, mode='pure')
#     eNorth      = EnumElem('eNorth', belong=eDirection, value=4, mode='pure')
#
#     ePerson     = EnumElem('ePerson')
#     eOccupation = EnumElem('eOccupation', belong=ePerson, value=0, mode='None')
#     eCountry    = EnumElem('eCountry', belong=ePerson, value=2, mode='None')
#     ePostCode   = EnumElem('ePostCode', belong=ePerson, value=3, mode='None')
#
#     assert_equal(len(eWest._elements), 2)
#     assert_equal(len(eWest._belongs), 1)
#     assert_equal(len(eCountry._belongs), 2)
#     assert_equal(eWest.eCountry in eWest, True)
#     assert_equal(eCountry in ePerson, True)
#



s = 'self, name:int, data:str, q:Union(float, int, str), age:int, u:Union[(int, str)], p:List[int] '
def arg_spliter(s):
    # result = ['']
    # for i, d in enumerate(s.split(',')):
    #     if ':' in d: result.append(d)
    #     else:        result[-1] += d
    # return result
    enter_bracket = 0
    enter_square = 0
    result = []
    for d in s.split(','):
        if ':' in d:
            result.append(d)
            if '(' in d: enter_bracket +=1
            if '[' in d: enter_square += 1
        else:
            if not enter_bracket and not enter_square: result.append(d)
            else:                                      result[-1] += d
            if ')' in d: enter_bracket -= 1
            if ']' in d: enter_square -= 1

    return result

print(arg_spliter(s))


