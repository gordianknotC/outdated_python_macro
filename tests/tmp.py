#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'gordi_000'


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b






