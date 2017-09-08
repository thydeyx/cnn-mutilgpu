# -*- encoding:utf-8 -*-

import sys

class Cat(object): # 类，Cat指向这个类对象
    def __init__(self, name='kitty'):
        self.name = name
    def sayHi(self): #  实例方法，sayHi指向这个方法对象，使用类或实例.sayHi访问
        print(self.name, 'says Hi!')# 访问名为name的字段，使用实例.name访问

if __name__ == '__main__':
    cat = Cat()
    for name in dir(cat):
        print(name, getattr(cat, name))