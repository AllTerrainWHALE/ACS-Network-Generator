import numpy as np
# from scipy.stats import norm 

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
# def normal_distribution_over_array(array, center_index, sigma=1.0):
#     # Create an array of indices corresponding to the positions in your array.
#     indices = np.arange(array.size)

#     # Compute the normal pdf at each index with the given mean and standard deviation.
#     pdf = norm.pdf(indices, loc=center_index, scale=sigma)

#     # Normalize the pdf so that it sums to 1.
#     pdf_normalized = pdf / pdf.sum()
#     return pdf_normalized

# def normal_distribution_by_values(values, center_value, sigma=1.0):
#     """
#     Computes a normalized Gaussian distribution based on the actual values.

#     Parameters:
#       values: array-like, the data values at each index.
#       center_value: float, the center (mean) of the Gaussian distribution.
#       sigma: float, the standard deviation controlling the spread.

#     Returns:
#       A numpy array of normalized weights corresponding to each value.
#     """
#     values = np.asarray(values)

#     # Compute the Gaussian probability density for each value
#     pdf = norm.pdf(values, loc=center_value, scale=sigma)
    
#     # Normalize so the sum is 1
#     pdf_normalized = pdf / pdf.sum()
#     return pdf_normalized

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