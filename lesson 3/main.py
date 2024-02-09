# Перенести все операции по работе с количеством объектов в класс Item  # + - * /
class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16

    def update_count(self, val):
        if 0 <= val <= self._max_count:
            return True
        else:
            return False

    def count(self):
        return self._count

    def __lt__(self, num):
        """ Сравнение меньше """
        return self._count < num

    def __gt__(self, num):
        """ Сравнение больше """
        return self._count > num

    def __le__(self, num):
        """ Сравнение меньше или равно """
        return self._count <= num

    def __ge__(self, num):
        """ Сравнение больше или равно """
        return self._count >= num

    def __eq__(self, num):
        """ Сравнение равно"""
        return self._count == num

    def __iadd__(self, num):
        self._count += num
        return self

    def __imul__(self, num):
        self._count *= num
        return self

    def __isub__(self, num):
        self._count -= num
        return self

    def __add__(self, num):
        """ Сложение с числом """
        return self._count + num

    def __sub__(self, num):
        """ Вычитыние числа """
        return self._count - num

    def __mul__(self, num):
        """ Умножение на число """
        return self._count * num

    def __floordiv__(self, num):
        """ Деление на число """
        return self._count // num


item = Item()
y = item.count()
x = 3
print(y, '+', x, '=', item + x)
print(y, '*', x, '=', item * x)
print(y, '-', x, '=', item - x)
print(y, ':', x, '=', item // x, '\n')

# Дополнить остальными опрерациями сравнения (>, <=, >=, ==), вычитания, а также выполнение операций +=, *=, -=. Все изменения количества должны быть в пределах [0, max_count]

# item = Item()
# a = item.count()
# b = 10
# b = b if item.update_count(b) else b == 16
# print(a, '>', b, ':', item > b)
# print(a, '<=', b, ':', item < b)
# print(a, '>=', b, ':', item >= b)
# print(a, '==', b, ':', item == b)
# item += b
# print(a, '+=', b, ':', item.count())
# item *= b
# print(a, '*=', b, ':', item.count())
# item -= b
# print(a, '-=', b, ':', item.count())

# Создать ещё 2 класса съедобных фруктов и 2 класса съедобных не фруктов
class Vegetable(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe


class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0


class Orange(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='orange', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Lemon(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='yellow', saturation=5):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Potato(Vegetable, Food):
    def __init__(self, ripe, count=1, max_count=32, color='brown', saturation=20):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Pickle(Vegetable, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=15):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe


apple = Apple(True, 2, color='red')
orange = Orange(True, 3, color='orange')
lemon = Lemon(True, 0, color='yellow')
potato = Potato(True, 5, color='brown')
pickle = Pickle(False, 6, color='green')
# Создать класс Inventory, который содержит в себе список фиксированной длины. Заполнить его None. Доступ в ячейку осуществляется по индексу.


class Inventory():
    def __init__(self, size):
        self._size = size
        self._items = [None] * size

    def __getitem__(self, index):
        return self._items[index] if 0 <= index < self._size else None

# 4.1 Добавить возможность добавлять в него съедобные объекты в определённые ячейки.
    def add_item(self, item, index):
        if item.eatable == True and 0 <= index < self._size:
            self._items[index] = item
        else:
            print('Food is not eatable!')

# 4.2 Добавить возможность уменьшать количество объектов в списке.
    def remove_item(self, index):
        if 0 <= index < self._size:
            self._items[index] = None

# 4.3 При достижении нуля, объект удаляется из инвенторя.
    def remove_empty(self, item):
        if item.count() <= 0:
            self.remove_item(self._items.index(item))


inventory = Inventory(5)

print('Число:', lemon.count())
print(potato < 3)
print('Цвет:', pickle.color)
print('Съедобно ли?', pickle.eatable)

inventory.add_item(apple, 0)  # добавление яблока и др в инвентарь
inventory.add_item(orange, 1)
inventory.add_item(lemon, 2)
inventory.add_item(potato, 3)
inventory.add_item(pickle, 4)

inventory.remove_empty(lemon)  # если лимонов нет, то объект удаляется из инвентаря
print('Есть ли лимоны в инвентаре?', inventory.__getitem__(2))
print(inventory.__getitem__(0))
print(inventory.__getitem__(1))
print('Есть ли огурцы в инвентаре?', inventory.__getitem__(4))  # pickle.eatable == False, поэтому в инвентаре его не будет

inventory.remove_item(0)  # удаление яблока
print(inventory.__getitem__(0))
