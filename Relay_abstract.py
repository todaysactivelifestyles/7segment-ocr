import abc

class NOrelay(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def on(self):
        pass

    @abc.abstractmethod
    def off(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass