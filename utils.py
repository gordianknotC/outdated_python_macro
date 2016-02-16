#!/usr/bin/python
# -*- coding: utf-8 -*-
from ast import *
from ast import AST
from collections import OrderedDict
import ast, sys
from typing import overload, Callable, Dict, Union, List as Lis


ast_map = Dict[str, AST](
    {'name' : Name,
    'assign' : Assign,
    'call' : Call,
    'if_' : If,
    'attribute' : Attribute,
    'expr' : Expr}
)

IS = isinstance
enum_registry = Dict['EnumElem', int]()

class DuplicateEnumException(Exception):
    def __init__(self, msg:str):
        super(DuplicateEnumException, self).__init__(msg)

class DynamicChangeEnumModeException(Exception):
    def __init__(self):
        super(DynamicChangeEnumModeException, self).__init__('Chaning Enum mode at runtime is not suported')


class EnumElts(list):
    def __init__(self, d: Lis[Union[int, None,'EnumElem']]  ) -> None:
        super(EnumElts, self).__init__(d)

    def add(self, el:'EnumElts'):
          if not el in self: self.append(el)

class EnumElem(object):
    #__slots__ = ['_belongs', '_value', '_elements', '_name', '__mode', '__name__','registery']
    def __init__(self, name:str, belong:EnumElts=None, value:int=None,
                 elements:EnumElts=None, mode:str=None):
        self._regist_name = module = __name__+ name
        if module in enum_registry:
            self.__dict__.update(enum_registry[module].__dict__)
            if belong:
                self._belongs.add(belong)
                belong[0].add_member(EnumElts([self, value]))
            if elements: self._elements.extend(elements)

            if mode:
                print('set mode to ', mode)
                if mode != self._mode: raise DynamicChangeEnumModeException()
        else:
            self._belongs  = EnumElts([]) if belong is None else belong
            self._elements = elements if elements is not None else EnumElts([])
            self._mode    = mode
            print('self._mode = mode', mode)
            self.__name__ = name
            if belong is not None:
                print('belong:', belong)
                print('self._belong:', self._belongs)
                self._belongs.add(belong)
                if value is None:
                    if belong[0]._elements: self._value = value = belong[0]._elements[-1][1] + 1
                    else:                self._value = value = 0
                belong[0].add_member(EnumElts([self, value]))

            self._value = value if value is not None else value
            enum_registry[module] = self

    @classmethod
    def clearRegistry(self, module=None, all=False):
        if not all:
            tmp = enum_registry
            for key in list(tmp.keys()):
                if module in key:
                    print('clear', key)
                    enum_registry.pop(key)
        else: enum_registry.clear()
    def checkDuplicate(self):
        seen1 = set([])
        retm = [i[0] for i in self._elements if not i[0] in seen1 and not seen1.add(i[0])]

        seen2 = set([])
        retv = [i[1] for i in self._elements if not i[1] in seen2 and not seen2.add(i[1])]

        retm = len(seen1) != len(self._elements)
        retv = len(seen2) != len(self._elements)
        if retm or retv:
            if retm:  msg = ' {} <=> {}'.format(self._elements, [i for i in seen1])
            if retv:  msg = ' {} <=> {}'.format(self._elements, [i for i in seen2])
            raise DuplicateEnumException('duplicate enum elements'+msg)

    def add_member(self, member:EnumElts):
        add_member = self.add_member
        self._elements.append(member)
        self.checkDuplicate()

    def __getattr__(self,name):
        print('getattr name = ', name)
        if '_mode' in self.__dict__:
            for d in self._elements:
                if d[0].__name__ == name and d[0]._mode == 'pure': return d
            raise AttributeError(name)
        else: return

    def __getitem__(self,key):
        print('get item key == ', key)
        return [i[0] for i in self._elements if i[1] == 'Enum_'+key][0]

    def __len__(self):
        return len(self._elements)

    def __show(self):
        return 'Enum {}:, {}'.format(self.__name__, 'value:{}, elements:{}, belongs:{}'.format(self._value, self._elements, self._belongs ))

    def __str__(self):
        return 'Enum_'+ self.__name__

    def __repr__(self):
        return 'Enum_'+self.__name__

    def __iter__ (self):
        for element in self._elements:
            yield element

    def __contains__(self, item):
        if IS(item, EnumElem):
            if item._mode == "normal":
                    if [i for i in self._elements if i[0] == item]: return True
                    return False
            else:   return False
        elif IS(item, EnumElts):
            if item[0]._mode == 'pure':
                if [i for i in self._elements if i == item]: return True
            else: return False
        return False

    def elements(self): return self._elements

    def __del__(self):
        #print(enum_registry.keys())
        if self._regist_name in enum_registry: enum_registry.pop(self._regist_name)



class Seq(list):
    def __init__(self, data):
        super().__init__(data)

    def insert_bulks(self, index, content):
        init = self[:index]
        tail = self[index+1:]
        init.extend(content)
        init.extend(tail)
        self[:] = init

class Object(object):
    def __init__(self, **kargs):
        for k,v in kargs.items():
            setattr(self,k,v)

class Data(dict):
    def __init__(self, **kargs):
        super().__init__(kargs)

    def __getattribute__(self, item):
        if item[:2] == '__':  return super().__getattribute__(item)
        else: return self[item]

class UncaughtFieldException(BaseException):
    def __init__(self, fieldname:str):
        msg = 'Invalid Usage, unsupported field name: {}'.format(fieldname)
        super(UncaughtFieldException, self).__init__(msg)

def flatten_list(lst, ret=None):
    if isinstance(lst, list):
        for l in lst:
            flatten_list(l, ret)
    else:   ret.append(lst)

class Transformer(object):
    from_source = []
    to_target = []
    interests_nodes =   ['call', 'assign', 'expr', 'if', 'attribute','name']
    interests_fields =  ['id', 'elts', 'n', 's', 'attr']

    class Tbase(object):
        __slots__ = ['value', '_fields', 'ast']
        def __init__(self, value):
            self.value = value
            if value in ast_map:
                self._fields = ast_map[value]._fields
                self.ast =ast_map[value]
            else:
                self._fields = None
                self.ast = None
    class Tnode(Tbase): pass
    class Tfield(Tbase): pass
    class Tvalue(Tbase):
        def __init__(self, value):
            self.value = value
            self.ast, self._fields = None, None

    # [V] tested okay
    def split_camel_case(self, attr):
        ret = []
        try:
            for i, ch in enumerate(attr):
                if ord(ch) == ord(ch.lower()): ret[-1] += ch
                else:                          ret.append(ch)
        except BaseException as e:
            print(e)
            tmp = [i for i in attr]
            tmp[i] = tmp[i].upper() if tmp[i].upper() != tmp[i] else tmp[i].lower()
            raise BaseException("invalid camelCase usage: {}, should be {}".format(attr, ''.join(tmp)  ), e)
        return [r.lower() for r in ret]

    def __setattr__(self, attr, value):
        if attr in ['from_source', 'to_target', 'interests_nodes', 'interests_fields']:
            super().__setattr__(attr, value)
            return
        attr = self.split_camel_case(attr)
        tail = attr[1:]
        from_source, to_target = self.from_source, self.to_target
        data = []
        if attr[0] == 'from':
            from_source.append([])
            data = from_source

        elif attr[0] == 'to':
            to_target.append([])
            data = to_target

        try:
            for i, nd in enumerate(tail):
                #print(i,nd)
                if nd in self.interests_nodes:
                    #print('interests nodes', nd)
                    data[-1].append(Transformer.Tnode(nd) )
                    #print(data[-1][-1]._fields)
                elif data[-1][-1]._fields:
                    if nd in data[-1][-1]._fields:
                        #print('nodes fields', nd)
                        data[-1].append(Transformer.Tfield(nd))
                elif nd in self.interests_fields:
                    #print('fields', nd)
                    data[-1].append(Transformer.Tfield(nd))
            if not type(data[-1][-1]) == Transformer.Tfield:
                raise UncaughtFieldException(nd)
            if type(data[-1][0]) != Transformer.Tnode:
                raise Exception('Invalid usage: only allowed Tnode for first node, you provide {}'.format(data[-1][0]))

        except BaseException as e:
            print(e)
            print('data:', data)
            print('nd:',nd)
            raise
        data[-1].append(Transformer.Tvalue(value))

    def find_fields(self, childs, rec:Tfield):
        find_fields = self.find_fields
        if type(childs) in [int, str, float]:      return []
        if not childs:                   return []
        if not isinstance(childs, list): childs = [childs]
        else:                            childs = [i for i in childs if i]
        _result = []
        tmp = []
        flatten_list(childs, tmp)
        for child in tmp:
            for fieldname, _value in ast.iter_fields(child):
                if fieldname == rec.value:
                    print('    match field:', rec.value, _value)
                    if isinstance(_value, list): _result.append([child, _value])
                    else:                        _result.append([child, _value])

                _result.extend(find_fields(_value, rec))
        if _result:
            return _result
        return []

    def find_nodes(self, node, rec):
        find_nodes = self.find_nodes
        for child in ast.iter_child_nodes(node):
            if isinstance(child, rec.ast): return [node, child]
            find_nodes(child, rec)
        return False

    def transform(self, body:AST):
        from_source = self.from_source
        to_target   = self.to_target
        find_nodes = self.find_nodes
        find_fields = self.find_fields

        for node in body:
            for csi, catch_set in enumerate(from_source):
                for icatch, _catch in enumerate(catch_set):
                    if icatch == 0:
                        parent, current = None, None
                        result = self.find_nodes(node, _catch)
                        if result:
                            parent, current = result
                            continue
                        else: break
                    else:
                        if _catch.ast:
                            result = self.find_nodes(current, _catch)
                            if result:
                                parent, current = result
                                continue
                            else: break
                        else:
                            if icatch != len(catch_set) -1:
                                result = self.find_fields(current, _catch)
                                if result:
                                    parent, current = [i[0] for i in result], [i[1] for i in result]
                                    continue
                                else: break
                            elif icatch == len(catch_set) -1:
                                seen         = set()
                                seenp        = set()
                                uniq_current = [ i for i in current if not i in seen and not seen.add(i)]
                                uniq_parent  = [ i for i in parent if not i in seenp and not seenp.add(i)]
                                print('      search final fields', _catch.value, uniq_current)
                                print('      parent:', uniq_parent)
                                print([i.value for i in catch_set])
                                for ci, cur in enumerate(uniq_current):
                                    if cur == _catch.value:
                                        target_value = to_target[csi][icatch].value
                                        print('        catch final fields:', _catch.value)
                                        setattr(uniq_parent[ci], catch_set[-2].value, target_value)
                                        print('setattr', uniq_parent[ci], catch_set[-2].value, target_value)
                                else: break
                            else:
                                raise Exception('uncaught exception')

Tbase = Transformer.Tbase
Tvalue = Transformer.Tvalue
Tnode = Transformer.Tnode
Tfield = Transformer.Tfield

#--------------------------
# depricate!!
# not a pratical solution yet
class Conditions(object):
    _conditons     = []
    _condition_map = OrderedDict()
    custom_commands = {}
    allowed = ['__call__', 'executeCondition', 'addConditionRules', '__getattribute__',
               '_conditions', '_condition_map', 'allowed', 'custom_commands']

    def __call__(self, *args, **kargs):
        return self.executeCondition(*args, **kargs)

    def executeCondition(self,*args, **kargs):
        def real(*_args, **_kargs):
            funcdata = kargs['__data']['func']
            funcdata(*_args)
        return real

    def addConditionRules(self, name, func,  test):
        data = {'func':func,   'test':test}
        name = name.lower()
        self._condition_map.update({name.lower():data})
        if  func.__code__.co_varnames != ('parent', 'cur', 'stack'):
            raise Exception('invalid function arguments numbers: {} \nShould be:parent, cur, stack, mode=None, parent_order=None'.format(func.__code__.co_varnames))

        for n in [name, 'set{}condition'.format(name), name+'condition']:
            self.custom_commands.update({n:data})

    def __iter__(self):
        for k, v in self._condition_map.items():
            yield v

    def __getattribute__(self, key):
        if key in super().__getattribute__('allowed'):
            return super().__getattribute__(key)
        key = key.lower()
        is_match = set(self.custom_commands).intersection( list(self._condition_map.keys()))
        if is_match:   return self.__call__(__data= self.custom_commands[key])
        else:          return super(Conditions, self).__getattribute__(key)




if __name__ == '__main__':
    print(__file__)
    # con = Conditions()
    # con.addConditionRules(name='field',func=lambda x,y: print('field', x,y),
    #                         test=lambda x: x=='field' )
    #
    # con.addConditionRules(name='current',func=lambda node: print('current', node),
    #                        test=lambda x: x=='current' )
    #
    # con.addConditionRules(name='parent',func=lambda x: print('parent', x),
    #                         test=lambda x: x=='parent' )
    #
    # print(con._condition_map)
    # con.fieldCondition('fieldname', 'value')
    # con.parentCondition('node')
    # con.currentCondition('node')
    #
    # for condition in con:
    #     print(condition)

    #     code = '''
    # call(a,b)
    # macro.func(1,2,3)
    # abc,df = 1,2
    # print('executing inline function')
    # print(sum(a,b))
    #     '''
    #     node = ast.parse(code).body
    #     for i in node:
    #         print(dump(i))
    #     print('-0------------------')
    #     t = Transformer()
    #     t.FromCallArgsId = 'a'
    #     t.ToCallArgsId = 'aa'
    #     t.FromCallArgsId = 'b'
    #     t.ToCallArgsId = 'bb'
    #     t.FromCallNameId = 'call'
    #     t.ToCallNameId = 'ccall'
    #     t.FromCallAttributeAttr = 'func'
    #     t.ToCallAttributeAttr = 'world'
    #     t.transform(node)
    #
    #
    #     print(node[0].value.func.id)
    #     print(node[0].value.args[0].id)
    #     print(node[0].value.args[1].id)
    #     print(node[1].value.func.attr)

    pass