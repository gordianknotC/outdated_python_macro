#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'gordi_000'
import ast
from ast import *
IS=isinstance


code = '''
class ABC(object):pass

print(somthing)

with EnumType(eDirection) as pure:
    eEast, eWest, eSouth, eNorth = 1,2,3,4

with EnumType(ePerson):
    eOccupation, eCountry, ePostCode

'''
nodes = ast.parse(code).body



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
            for child in ast.iter_child_nodes(nd):
                if capture_current(child):
                    stack.append(child)
                    if not do(stack[0], child, stack, mode = 'node', parent_order =i):
                        walk_field(child, conditions, do, stack,capture_field, i)
                else:
                    if capture_parent(child):
                        stack[0] = child
                    walk(child, conditions, do, stack)

from astutil.codegen import to_source
def getValue(_x, exception=None):
    value = _x
    if   IS(value, Num)   : return value.n
    elif IS(value, Str)   : return value.s
    elif IS(value, List)  : return [getValue(e) for e in value.elts]
    elif IS(value, Dict)  : return { getValue(i[0]): getValue(i[1]) for i in zip(value.keys, value.values)}
    elif IS(value, Call)  : return value
    elif IS(value, Name)  : return value.id
    elif IS(value, BinOp) : return [getValue(value.left)] + [getValue(value.op,exception='binop')] +  [getValue(value.right) ]
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



def main():
    for node in nodes:
        if IS(node, With):
            print(dump(node))
            kargs = dict(optional_vars = node.items[0].optional_vars)
            args = node.items[0].context_expr.args
            body = node.body
            enumgrp_ptn   = '''{gname} = EnumElem("{gname}")'''
            enum_ptn      = '''{ename} = EnumElem("{ename}", belong={belong}, value={value], mode="{mode}") '''
            optional_vars = kargs['optional_vars']
            capture_current = lambda node: IS(node, Assign) or IS(node, Tuple)
            capture_field   = lambda fieldname, value: fieldname == 'id'
            capture_parent  = lambda node: hasattr(node, 'body')
            conditions      = [capture_current, capture_parent, capture_field]
            module          = __name__
            ret             = []

            def do(parent, cur, stack, mode=None, parent_order=None):
                parent = stack[0]
                tree   = stack[1]
                stacks = stack[2:]
                last_node = stack[-1]
                if mode == 'field':  pass
                else:
                    if IS(cur, Assign):
                        targets = last_node.targets[0].elts
                        values = [ getValue(i) for i in last_node.value.elts]
                        print(targets)
                        print(values)
                        ret.append([i for i in zip(targets, values)])
                        return True
                    elif IS(cur, Tuple):
                        targets = last_node.elts
                        values = list(range(len(targets)))
                        print(targets)
                        print(values)
                        ret.append([i for i in zip(targets, values)])
                        return True
                    return False

            belong = gname = args[0].id
            mode   = optional_vars
            nbody  = []
            nd     = parse(enumgrp_ptn.format(gname = gname))
            nbody.append(nd)
            for elts in ret:
                ename, value = elts
                code = enum_ptn.format(ename = ename, belong=belong, value=value, mode=mode)
                nd   = parse(code)
                nbody.append(nd)
            return nbody


            walk(body, conditions, do, stack=[None, body])

main()