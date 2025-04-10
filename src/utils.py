from math import cos, sin, pi

# Gracefully stolen from https://stackoverflow.com/a/34325723
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd if iteration!=total else '')
    # Print New Line on Complete
    if iteration == total: 
        print()

# Gracefully stolen from https://www.geeksforgeeks.org/create-a-directory-in-python/
def make_dir (dir_name:str) -> bool:
    from os import mkdir

    success = False

    try:
        mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
        success = True
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
        success = True
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return success

# Gracefully stolen from ChatGPT
def regular_polygon(xy, d, n):
    """
    Returns `n` vertices of an equilateral shape of side length `d`
    centered at (x, y).
    
    Parameters:
        xy (float): (x, y)-coordinate of the center
        d (float): side length of the equilateral shape

    Returns:
        list of tuples: [(x1, y1), (x2, y2), ..., (xn, yn)] vertices of the shape
    """
    x,y = xy

    # For a regular polygon, the circumradius R is given by:
    R = d / (2 * sin(pi / n))
    
    vertices = []
    # We set the initial angle to pi/2 so that one vertex is straight up.
    for i in range(n):
        angle = pi / 2 + i * (2 * pi / n)
        vx = x + R * cos(angle)
        vy = y + R * sin(angle)
        vertices.append((int(vx), int(vy)))
    
    return vertices

# Gracefully stolen from https://stackoverflow.com/a/287944
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'