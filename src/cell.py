class Cell:

    #_ Info bit lengths
    ITEM_BITS = 2
    PHERO_BITS = 30

    #_ Maximum values for each section
    MAX_ITEM = (2**ITEM_BITS)-1 # 2 bits: 0b11 -> 3
    MAX_PHERO = (2**PHERO_BITS)-1 # 30 bits: 0x3FFFFFFF -> 1,073,741,823

    #_ Masks for the bit layout
    MASK_STEP = len(bin(MAX_PHERO))-2

    ITEM_MASK = 3 << (MASK_STEP*2)            # Mask for the items
    PHEROA_MASK = MAX_PHERO << MASK_STEP  # Mask for the next 30 bits
    PHEROB_MASK = MAX_PHERO        # Mask for the last 30 bits
    
    @staticmethod
    def setItem(whole_val: int, new_val: int) -> int:
        """
        2-bit long integer representing the state of the Cell in the environment.

        - `State.NONE = 0` -> Nothing
        - `State.FOOD = 1` -> Food
        - `State.NEST = 2` -> Nest
        - `State.WALL = 3` -> Wall / Obstruction
        """
        # Clear the first 2 bits and set them
        whole_val = (int(whole_val) & ~Cell.ITEM_MASK) | ((int(new_val) & Cell.MAX_ITEM) << (Cell.MASK_STEP*2))
        return whole_val

    @staticmethod
    def setPheroA(whole_val: int, new_val: int) -> int:
        # Clear the next 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROA_MASK) | ((int(new_val) & Cell.MAX_PHERO) << Cell.MASK_STEP)
        return whole_val

    @staticmethod
    def setPheroB(whole_val: int, new_val: int) -> int:
        # Clear the last 31 bits and set them
        whole_val = (int(whole_val) & ~Cell.PHEROB_MASK) | (int(new_val) & Cell.MAX_PHERO)
        return whole_val
    
    @staticmethod
    def setAll(whole_val: int, state: int, pheroA: int, pheroB: int) -> int:
        return Cell.setPheroB(
            Cell.setPheroA(
                Cell.setItem(whole_val, state),
                pheroA
            ),
            pheroB
        )
    


    @staticmethod
    def getItem(whole_val: int) -> int:
        # Extract and return the first 2 bits
        return (int(whole_val) >> (Cell.MASK_STEP*2)) & Cell.MAX_ITEM

    @staticmethod
    def getPheroA(whole_val: int) -> int:
        # Extract and return the next 31 bits
        return (int(whole_val) >> Cell.MASK_STEP) & Cell.MAX_PHERO

    @staticmethod
    def getPheroB(whole_val: int) -> int:
        # Extract and return the last 31 bits
        return int(whole_val) & Cell.MAX_PHERO
    
    @staticmethod
    def getAll(whole_val: int) -> int:
        return Cell.getItem(whole_val), Cell.getPheroA(whole_val), Cell.getPheroB(whole_val)
    


    @staticmethod
    def excludeItem(whole_val) -> int:
        """Removes the Item bits and shifts PheroA and PheroB bits left to fill the gap."""
        return (Cell.getPheroA(whole_val) << Cell.MASK_STEP) | Cell.getPheroB(whole_val)

    @staticmethod
    def excludePheroA(whole_val) -> int:
        """Removes the PheroA bits and shifts Item and PheroB bits together."""
        return (Cell.getItem(whole_val) << Cell.MASK_STEP) | Cell.getPheroB(whole_val)

    @staticmethod
    def excludePheroB(whole_val) -> int:
        """Removes the PheroB bits and shifts Item and PheroA bits together."""
        return (Cell.getItem(whole_val) << Cell.MASK_STEP) | Cell.getPheroA(whole_val)
    


    #// @staticmethod
    #// def normalize(whole_val):
    #//     return whole_val / int('1'*())
    
    class item:
        NONE = 0
        FOOD = 1
        NEST = 2
        WALL = 3