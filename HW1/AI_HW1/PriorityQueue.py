
class PriorityQueue:
    """
        This is an implementation of Priority Queue. You can use this class but cannot change!
    """
    __items: list

    def __init__(self):
        """
            Constructor of the class.
        """
        self.__items = []

    def enqueue(self, item, value: float):
        """
            This method enqueues the given item. Also, it sorts the items in ascending order with the provided value.
        :param item: An item will be enqueue, Object
        :param value: Value of the item, float.
        :return: void.
        """
        self.__items.append([item, value])

        for i in reversed(range(1, len(self.__items))):
            if self.__items[i][1] < self.__items[i - 1][1]:
                self.__items[i], self.__items[i - 1] = self.__items[i - 1], self.__items[i]
            else:
                break

    def dequeue(self):
        """
            This method removes the next item and returns it.
        :return: Item in the queue.
        """
        return self.__items.pop(0)[0]

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, index: int):
        return self.__items[index][0]

    def __str__(self):
        return str([item[0] for item in self.__items])

    def __repr__(self):
        return self.__str__()

    def __contains__(self, item):
        return item in [i[0] for i in self.__items]

    def __eq__(self, other):
        return self.__items == other.__items
