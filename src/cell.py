class Cell:

    # Masks for the 64-bit layout
    STATE_MASK = 3 << 60            # Mask for the first 2 bits
    PHEROA_MASK = 0x3FFFFFFF << 30  # Mask for the next 31 bits
    PHEROB_MASK = 0x3FFFFFFF        # Mask for the last 31 bits

    # Maximum values for each section
    MAX_STATE = 3                  # 2 bits: 0b11 -> 3
    MAX_PHEROA = 0x3FFFFFFF        # 31 bits: 0x3FFFFFFF -> 2147483647
    MAX_PHEROB = 0x3FFFFFFF        # 31 bits: 0x3FFFFFFF -> 2147483647
    
    # def __init__(self):
    #     self.value = 0

    
    @staticmethod
    def setState(whole_val: int, new_val: int) -> int:
        """
        2-bit long integer representing the state of the Cell in the environment.

        - `State.NONE = 0` -> Nothing
        - `State.FOOD = 1` -> Food
        - `State.NEST = 2` -> Nest
        - `State.WALL = 3` -> Wall / Obstruction
        """
        # Clear the first 2 bits and set them
        whole_val = (int(whole_val) & ~Cell.STATE_MASK) | ((int(new_val) & 0b11) << 60)
        return whole_val

    @staticmethod
    def setPheroA(whole_val: int, new_val: int) -> int:
        # Clear the next 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROA_MASK) | ((int(new_val) & 0x3FFFFFFF) << 30)
        return whole_val

    @staticmethod
    def setPheroB(whole_val: int, new_val: int) -> int:
        # Clear the last 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROB_MASK) | (int(new_val) & 0x3FFFFFFF)
        return whole_val
    
    @staticmethod
    def setAll(whole_val: int, state: int, pheroA: int, pheroB: int) -> int:
        return Cell.setPheroB(
            Cell.setPheroA(
                Cell.setState(whole_val, state),
                pheroA
            ),
            pheroB
        )
    


    @staticmethod
    def getState(whole_val: int) -> int:
        # Extract and return the first 2 bits
        return (int(whole_val) >> 60) & 0b11

    @staticmethod
    def getPheroA(whole_val: int) -> int:
        # Extract and return the next 31 bits
        return (int(whole_val) >> 30) & 0x3FFFFFFF

    @staticmethod
    def getPheroB(whole_val: int) -> int:
        # Extract and return the last 31 bits
        return int(whole_val) & 0x3FFFFFFF
    
    @staticmethod
    def getAll(whole_val: int) -> int:
        return Cell.getState(whole_val), Cell.getPheroA(whole_val), Cell.getPheroB(whole_val)
    


    @staticmethod
    def excludeState(whole_val) -> int:
        """Removes the State bits and shifts PheroA and PheroB left to fill the gap."""
        # phero_a = Cell.getPheroA(whole_val)
        # phero_b = Cell.getPheroB(whole_val)
        return (Cell.getPheroA(whole_val) << 30) | Cell.getPheroB(whole_val)

    @staticmethod
    def excludePheroA(whole_val) -> int:
        """Removes the PheroA bits and shifts State and PheroB together."""
        # state = Cell.getState(whole_val)
        # phero_b = Cell.getPheroB(whole_val)
        return (Cell.getState(whole_val) << 30) | Cell.getPheroB(whole_val)

    @staticmethod
    def excludePheroB(whole_val) -> int:
        """Removes the PheroB bits and shifts State and PheroA together."""
        # state = Cell.getState(whole_val)
        # phero_a = Cell.getPheroA(whole_val)
        return (Cell.getState(whole_val) << 30) | Cell.getPheroA(whole_val)
    


    @staticmethod
    def normalize(whole_val):
        return whole_val / 0x3FFFFFFFFFFFFFFF
    
    class item:
        NONE = 0
        FOOD = 1
        NEST = 2
        WALL = 3