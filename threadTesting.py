import threading
import time

class SharedResource:
    def __init__(self):
        self.data = 0
        self.lock = threading.Lock()

    def update(self):
        # Simulate periodic updates to the shared data
        while True:
            with self.lock:
                self.data += 1
                print(f"Updated data to {self.data}")
            time.sleep(1)  # Simulate time-consuming work

    def get_data(self):
        # Safely access the data
        with self.lock:
            return self.data


# Function for the updating thread
def updater(resource):
    resource.update()


# Function for the reading thread
def reader(resource):
    while True:
        data = resource.get_data()
        print(f"Read data: {data}")
        time.sleep(0.5)  # Simulate periodic reads


# Main program
if __name__ == "__main__":
    shared_resource = SharedResource()

    # Create threads
    update_thread = threading.Thread(target=updater, args=(shared_resource,))
    read_thread = threading.Thread(target=reader, args=(shared_resource,))

    # Start threads
    update_thread.start()
    read_thread.start()

    # Join threads (optional: only if you want the main program to wait)
    update_thread.join()
    read_thread.join()
