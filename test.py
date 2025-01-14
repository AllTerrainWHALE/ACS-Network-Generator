# from src.cell import Cell

# c = 0

# c = Cell.setState(c, 1)
# c = Cell.setPheroA(c, 5)
# c = Cell.setPheroB(c, 4)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()

# c = Cell.setPheroA(c, 0x7FFFFFFF)

# print(Cell.getState(c))
# print(Cell.getPheroA(c))
# print(Cell.getPheroB(c))
# print()
# print(0x7FFFFFFF)

# xy_phero = 1
# for _ in range(2):
#     blur = xy_phero / 9

#     diffusionDelta = 0.5

#     diff = (diffusionDelta * xy_phero) + ((1-diffusionDelta) * blur)
#     diff_evap = max(0, diff - (0.01 * 1))

#     print(diff_evap)

#     xy_phero = diff_evap



arr = [0,1,2,3,4,5]
print(arr[-9:])