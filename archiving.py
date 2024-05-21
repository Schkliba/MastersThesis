class Archive:
    def __init__(period):
        self.period = period
        self.gen_storage = {}
        self.pop_storage = []

    def store_individuals(self, generation, individual_list):
        if generation % self. period:
            self.gen_storage[generation] = individual_list
            self.pop_storage += individual_list

    def get_stored(self):
        return self.pop_storage