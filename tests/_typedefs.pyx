#!/usr/bin/python
# -*- coding: utf-8 -*-
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
from cython cimport view as seq

#===============================================
#            Double, Int, Char, Hybrid Arrays Implementation
#===============================================
cdef:
    fused N:
        int
        double
        float

    struct TDb:
        char* name
        char* data
        int age

    struct TStruct:
        char* name
        int* values
        char** names

    struct TComplex:
        TStruct* structs
        TDb* dbs
    
    struct TData:
        char* svalue
        int ivalue
        float fvalue
        
    enum eDirection:
        eNorth,eSouth,eEast,eEst

ctypedef seq.array TArray

cdef:
    TDb Db(char* name, char* data, int age ):
        cdef TDb d
        d.name = name
        d.data = data
        d.age = age
        return d

    TStruct Struct(char* name, int* values, char** names):
        cdef TStruct d
        d.name = name
        d.values = values
        d.names = names
        return d

    TComplex Complex( TStruct* structs, TDb* dbs):
        cdef TComplex d
        d.structs = structs
        d.dbs = dbs
        return d
    
    TData Data(int ivalue=0, float fvalue=0.0, char* svalue=b''):
        cdef TData d
        d.ivalue = ivalue
        d.fvalue = fvalue
        d.svalue = svalue
        return d
    
    inline float* _farr(float* value):
        cdef float* d = value
        return d
    
    inline int* _iarr(int* value):
        cdef int* d = value
        return d
    
    inline char** _sarr(char** value):
        cdef char** d = value
        return d
    
    inline double* _darr(double* value):
        cdef double* d = value
        return d
        
    inline TArray Array(char format, tuple shape, list value):
        cdef TArray ret
        cdef int size
        if   format == 'c': size = sizeof(char)
        elif format == 'i': size = sizeof(int)
        elif format == 'f': size = sizeof(float)
        else:               size = sizeof(double)
        ret = TArray(shape=shape, itemsize=size, format=format)
        return ret
    

my_array = seq.array(shape=(10, 2), itemsize=sizeof(int), format="i")
cdef int[:, :] my_slice = my_array

cdef seq.array myarray = seq.array(shape=(10,), itemsize=sizeof(int), format="i")
cdef int[:] iarray = seq.array(shape=(10,), itemsize=sizeof(int), format="i")
for i in range(10):
    myarray[i] = i
    print(i, myarray[i])

