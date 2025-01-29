class Cell:

    # Masks for the 64-bit layout
    STATE_MASK = 3 << 62            # Mask for the first 2 bits
    PHEROA_MASK = 0x7FFFFFFF << 31  # Mask for the next 31 bits
    PHEROB_MASK = 0x7FFFFFFF        # Mask for the last 31 bits

    # Maximum values for each section
    MAX_STATE = 3                  # 2 bits: 0b11 -> 3
    MAX_PHEROA = 0x7FFFFFFF        # 31 bits: 0x7FFFFFFF -> 2147483647
    MAX_PHEROB = 0x7FFFFFFF        # 31 bits: 0x7FFFFFFF -> 2147483647
    
    # def __init__(self):
    #     self.value = 0

    
    @staticmethod
    def setState(whole_val, new_val: int):
        """
        2-bit long integer representing the state of the Cell in the environment.

        - `State.NONE = 0` -> Nothing
        - `State.FOOD = 1` -> Food
        - `State.NEST = 2` -> Nest
        - `State.OBJ = 3` -> Wall / Obstruction
        """
        # Clear the first 2 bits and set them
        whole_val = (int(whole_val) & ~Cell.STATE_MASK) | ((new_val & 0b11) << 62)
        return whole_val

    @staticmethod
    def setPheroA(whole_val, new_val: int):
        # Clear the next 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROA_MASK) | ((new_val & 0x7FFFFFFF) << 31)
        return whole_val

    @staticmethod
    def setPheroB(whole_val, new_val: int):
        # Clear the last 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROB_MASK) | (new_val & 0x7FFFFFFF)
        return whole_val
    


    @staticmethod
    def getState(whole_val):
        # Extract and return the first 2 bits
        return (int(whole_val) >> 62) & 0b11

    @staticmethod
    def getPheroA(whole_val):
        # Extract and return the next 31 bits
        return (int(whole_val) >> 31) & 0x7FFFFFFF

    @staticmethod
    def getPheroB(whole_val):
        # Extract and return the last 31 bits
        return int(whole_val) & 0x7FFFFFFF
    
    @staticmethod
    def getAll(whole_val):
        return Cell.getState(whole_val), Cell.getPheroA(whole_val), Cell.getPheroB(whole_val)
    


    @staticmethod
    def normalize(whole_val):
        return whole_val / 0xFFFFFFFFFFFFFFFF
    
class State:
    NONE = 0
    FOOD = 1
    NEST = 2
    OBJ = 3