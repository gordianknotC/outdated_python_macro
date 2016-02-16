#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'gordi_000'
from ast import *
import ast
from python_macro.astutil.codegen import to_source
from python_macro import fix_locations
from copy import deepcopy

class ContextualTransformer(NodeTransformer):
    def __init__(self, node):
        self.context = node
        self.contextual_stack = [None]

    def get_line(self, n:int):
        # fetch current line: get_line(-1)
        # fetch last line:    get_line(-2)
        return self.contextual_stack[n]

    def sequence_node(self, current_line_node):
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
        ret = []
        for bd in node.body:
            if not bd: continue
            self.contextual_stack.append( bd )
            if isinstance(bd, If):
                bd.test = self.generic_visit( bd.test)
                bd = self.contextual_visit(bd)
                if hasattr(bd, 'orelse'):
                    if bd.orelse:
                        if isinstance(bd.orelse[0], If): bd.orelse[0] = self.contextual_visit(bd.orelse[0])
                        else:                            bd.orelse[0] = self.generic_visit(bd.orelse[0])
                ret.append(bd)
            elif hasattr(bd, 'body'): ret.append( self.contextual_visit(bd) )
            else:                     ret.append( self.generic_visit( bd ) )
        node.body = ret
        return node

    def visit_Name(self, node):
        print("visit name: ", to_source(node))
        last = self.get_line(-1)
        if isinstance(last, Expr):
            print('--------')
            print(node.id)
        return node


    # def visit_Attribute(self, node):
    #     print('visit Attribute', to_source(node))
    #
    # def visit_List(self, node):
    #     print('visit List', to_source(node))
    #
    # def visit_Tuple(self, node):
    #     print('visit Tuple', to_source(node))
    #
    # def visit_Subscript(self, node):
    #     print('visit Subscript', to_source(node))
    #visit_Attribute = visit_Subscript = visit_List = visit_Tuple = visit_Name

class CallTransformer(NodeTransformer):
    def __init__(self, args, body=None):
        self.args = args
        self.body = body
        print('instantiate CallTransformer body:',body, 'args:', args)

code = """
import os,sys,cython
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
    print(eSouth in eDirection)
    node = self.generic_visit(node)
    body = self.body
    print('CallTransformer visitExpr')
    print('node:', ast.dump(node))
    print('body:', body)
    if body and isinstance(node.value, ast.Name) and node.value.id == '__body__':
        print('found __body__')
        return fix_locations(ast.If(ast.Num(1), body, []), node)
    return self.generic_visit(node)"""

node = parse(code)

visitor = ContextualTransformer(node)
result = visitor.contextual_visit(node)
print('result = ')
print(result)
print(dump(result))
print(to_source(result))
print(to_source(node))
print('\n'*10)
print('test get name')





ignored_modules = ['karnickel']
buildin_modules = ['sys']
module_import_list = []
IS  = isinstance


class BaseSearcher(NodeTransformer):
    def __init__(self, transformer=None):
        self.clear()
        self.ignore = None
        self.transformer = transformer
    def clear(self): self.value = set()
    def get_value(self):
        result = self.value
        self.clear()
        return result


class VarNameSearcher(BaseSearcher):
    def visit_list(self, lst):
        visit = self.visit
        return [visit(i) for i in lst]

    def visit_Name(self, node):
        self.value.add(node.id)
        if not self.transformer: return node
        else:                    return self.transformer(node)

class GlobalDefSearcher(BaseSearcher):
    def __init__(self):
        self.func_stack = []
        super(GlobalDefSearcher, self).__init__()

    def visit_FunctionDef(self,node):
        self.func_stack.append([node, []])
        return self.generic_visit(node)

    def visit_Global(self, node):
        print('global..', node.names)
        self.value.update(node.names)
        self.func_stack[-1][1].extend(node.names)
        return

class GlobalVarNameSearcher(BaseSearcher):
    # transform from globalName to sys.module[__name__].globalName
    def __init__(self, tree):
        super(GlobalVarNameSearcher, self).__init__()
        def transformer(node):
            result =  parse('module.{}'.format(node.id)).body[0].value
            result.ctx = node.ctx
            return result
        self.name_searcher = VarNameSearcher(transformer=transformer)
        self.tree = tree
        self.insert_data = []

    def combine_insert_data(self):
        body = self.tree.body
        [body.insert(i[0], i[1]) for i in self.insert_data]

    def visit_Assign(self, node):
        print('============================')
        print(type(node))
        print(dump(node))
        print(node._fields)
        if node.col_offset == 0:
            oid = deepcopy(node.targets[0])
            self.name_searcher.visit(node)
            if isinstance(node.targets[0], Tuple):
                print(to_source(node))
                node.targets.append(oid)
                print('altered tuple:')
                print(to_source(node))
            else:
                node.targets.append(oid)

            print(node.targets[0])
            #self.value.update(self.name_searcher.get_value())

        return node

    def visit_Expr(self, node):
        if node.col_offset == 0:
            print('found expr')
            if isinstance(node.value, Call):
                nargs = self.name_searcher.visit_list(node.value.args)
                node.value.args = nargs
            else:
                self.name_searcher.visit(node.value)
            self.value.update(self.name_searcher.get_value())
        return node

    def visit_FunctionDef(self, node):
        if node.col_offset == 0:
            self.value.add(node.name)
            id = self.tree.body.index(node)
            nnode = parse('module.{} = {}'.format(node.name,node.name)).body[0]
            self.insert_data.append([id+1+ len(self.insert_data), nnode])
        return node
    visit_ClassDef = visit_FunctionDef

class FuncDef_GlobalVar_Transformer(BaseSearcher):
    def __init__(self, funcs):
        super(FuncDef_GlobalVar_Transformer, self).__init__()
        self.func_stack = funcs
        self.current_func = None
        self.current_globals = None

    def visit_func_stack(self):
        for func in self.func_stack:
            func_node, func_globals = func
            if func_globals:
                self.current_func = func_node
                self.current_globals = func_globals
                self.visit(func_node)

    def visit_Name(self, node):
        name = node.id
        if name in self.current_globals:
            node.id = 'module.'+name
            print('global changer:', to_source(node))
            #node = parse('module.'+name).body[0].value
        return node

class ClassAttributeSearcher(BaseSearcher):
    def __init__(self,node):
        super(ClassAttributeSearcher, self).__init__()
        self.cls = []
        self.class_attributes = []
        self.initialize(node)

    def initialize(self, node):
        self.cls = node
        if node:
            self.col_offset = node.body[0].col_offset
        self.class_attributes = []

    def gen_function(self, body):
        return FunctionDef(name='__classproperties__',
                           args=arguments(args=[ast.arg(arg='self', annotation=None)],
                                          vararg=None, kwonlyargs=[],
                                          kw_defaults=[], kwarg=None, defaults=[]),
                           body=body, decorator_list=[], returns=None)

    def gen_classAttribute(self):
        body = self.class_attributes
        if body:
            class_attributes = self.gen_function(body)
            self.cls.body.insert(0,class_attributes)

    def visit_Assign(self, node):
        if node.col_offset == self.col_offset:
            nnode = self.generic_visit(node)
            self.class_attributes.append(nnode)
            return
        return node

    def visit_Name(self, node):
        return Attribute(value=Name(id='self', ctx=Load()), attr=node.id, ctx=node.ctx)


class ImportSearcher(BaseSearcher):
    def __init__(self):
        super(ImportSearcher, self).__init__()

    def visit_Import(self,node) ->None:
        modules = [[i.name, i.name or i.asname] for i in node.names if i.name not in ignored_modules and i not in module_import_list]
        module_import_list.extend(modules)
        if 'sys' in [i[0] for i in modules]:
            node.names = [alias(name='sys', asname = None)]
            return node

    def visit_ImportFrom(self, node):
        lineno, offset = node.lineno, node.col_offset
        modules = [[node.module + '.' + i.name,i.name or  i.asname] for i in node.names if node.module not in ignored_modules]
        module_import_list.extend([i for i in modules if not i in module_import_list])
        body =  [ast.Assign(targets=[Name(id=i[1], ctx=Store())], value=Name(id=i[0], ctx=Load())) for i in modules] or [ast.Pass()]
        #result = ast.If(test=ast.Num(n=0), body=body, orelse=[])
        for bd in body:
            bd.lineno = lineno
            bd.col_offset = offset
        result = body
        return result



s = """
from mytestsuite import testsuite
from multiple import ma,mb,mc
import rapydscript_testA
from karnickel import redirect

main_a = 1
main_b = 2
main_c, main_d = [1,3]
reduce(main_a)

def main_a_method(a:str = None, b:int = None):
    global main_b
    main_a = 'altered'
    main_b = 'altered2'
    print(main_a)
    print(main_b)

    def nested(e,f):
        global main_a
        main_a = 'nested'

def no_global(a,b):
    a = 1
    b = 2

class Sample(object):
    mode = None
    a,b,c = range(3)
    def __init__(self):
        global main_a
        if main_a == 1:
            print('Sample init')
        assert(main_a == 1)
        self.method(main_a)

    def method(self, m):
        print('Sample method m:', m)
        assert(m == 1)

    method2 = method

s = Sample()

"""

node = parse(s)
old = deepcopy(node)
searcher = VarNameSearcher()
searcher.visit(node)
print(searcher.value)

im_searcher = ImportSearcher()
node = im_searcher.visit(node)

gdef = GlobalDefSearcher()
gdef.visit(node)
func_stack = gdef.func_stack


func_global_changer = FuncDef_GlobalVar_Transformer(func_stack)
func_global_changer.visit_func_stack()

gsearcher = GlobalVarNameSearcher(node)
n_node = gsearcher.visit(node)
gsearcher.combine_insert_data()
print(gsearcher.value)
fix_locations(n_node.body, old.body)
print('n node:')
print(n_node)
print()
print()
print()



print(to_source(n_node))
code = compile(n_node, 'ast', 'exec')


