class Individual(object):
    def __init__(self, binary=None, fx=0, id=0, real=0):
        self.id = id
        self.binary = binary
        self.real = real
        self.fx = fx

    def __lt__(self, other):
        return self.fx > other.fx

    def __str__(self): 
        return "id: % s, fx: % s" % (self.id, self.fx)


class Test(object):
    def __init__(self, tau, generations_number, fx):
        self.tau = tau
        self.generations_number = generations_number
        self.fx = fx

    def __lt__(self, other):
        return self.fx > other.fx