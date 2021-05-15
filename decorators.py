import numpy as np


# def f1():
#     print('f1 called')
#
#
# def f2(func):
#     func()


######################################


def f3(func):
    def wrapper():
        print('start')
        func()
        print('end')
    return wrapper


def f4():
    print('hello')


# f3(f4)()
# print()
# f = f3(f4)
# f()
# print()


######################################
# DECORATORS


def f1(func):
    def wrapper():
        print('start')
        func()
        print('end')
    return wrapper


@ f1  # whenever f2 is called, the decorator calls function f1 with argument f2
def f2():
    print('Hello')


# f2()


######################################
# DECORATORS WITH ARGS AND KWARGS (KEY WORD ARGUMENTS)


def one(func):
    def wrapper(*args, **kwargs):
        print('start')
        func(*args, **kwargs)
        print('end')
    return wrapper


@ one
def caller(word):
    print(word)


######################################
# DECORATORS WITH ARGS AND KWARGS (KEY WORD ARGUMENTS) AND RETURN


def function(func):
    def wrapper(*args, **kwargs):
        print('start')
        value = func(*args, **kwargs)
        print('end')
        return value
    return wrapper


@ function
def f(a, b):
    return a+b


print(
    f(5, 7)
)


























































































































































































































































































































































































































































