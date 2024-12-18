from src.cell import Cell

c = 0

c = Cell.setState(c, 1)
c = Cell.setPheroA(c, 5)
c = Cell.setPheroB(c, 4)

print(Cell.getState(c))
print(Cell.getPheroA(c))
print(Cell.getPheroB(c))
print()

c = Cell.setPheroA(c, 0x7FFFFFFF)

print(Cell.getState(c))
print(Cell.getPheroA(c))
print(Cell.getPheroB(c))
print()
print(0x7FFFFFFF)