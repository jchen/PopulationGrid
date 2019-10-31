# Some things I like to import:
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from pylab import savefig
from tqdm import tnrange

class Household:
    """
    Households to use in our grid.
    """

    def __init__(self, home, work):
        """
        Creates a new household
        :param home: coordinates of the home
        :param work: coordinates of workplace
        """
        self.home = home
        self.work = work

    def travel_cost(self, work):
        """
        Calculates the travel cost given current home and arbitrary workplace
        :param work: new coordinates of work
        :return: the cost of travel, which is defined as taxicab distance
        """
        return (abs(self.home[0] - work[0]) +
                abs(self.home[1] - work[1]))

    def new_travel_cost(self, new_home):
        """
        Calculates the travel cost given current work and arbitrary home
        :param new_home: new coordinates of home
        :return: the cost of travel, taxicab distance
        """
        return (abs(new_home[0] - self.work[0]) +
                abs(new_home[1] - self.work[1]))

    def set_workplace(self, work):
        """
        sets own workplace to work
        :param work: new workplace
        :return: void
        """
        self.work = work

    def set_home(self, new_home):
        """
        sets own home to home
        :param new_home: new home
        :return: void
        """
        self.home = new_home

class Factory:
    """
    Factories to use in our grid
    """

    def __init__(self, location, capacity):
        """
        Initializes a factory located on our grid
        :param location: coordinates of the location
        :param capacity: capacity of this factory
        """
        self.loc = location
        self.cap = capacity
        self.available_workers = capacity

    def reduce_work(self):
        """
        Once a worker sets a factory, reduces the available work at this factory
        :return: void
        """
        self.available_workers += -1

class City:
    """
    Model class for a city.
    """

    def __init__(self, dim):
        """
        Creates a new city of dimension dim
        :param dim: dimension of square grid of city
        """
        self.dim = dim
        self.map = []
        for r in range(dim):
            row = []
            for c in range(dim):
                gridbox = []
                gridbox.append(Household((r, c), (0, 0)))
                row.append(gridbox)
            self.map.append(row)

    def population_density(self):
        """
        Gives the population density (household per grid) as an array
        :return: population density as an array
        """
        print("Population Density")
        all = []
        for r in self.map:
            row = []
            for box in r:
                row.append(len(box))
            all.append(row)
        return all
        # ax = sns.heatmap(np.array(all), annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, xticklabels=False,
        #                  yticklabels=False)
        # plt.savefig("output.png")

    def work_map(self):
        """
        Prints the map of works as an array
        :return: void
        """
        print("Work Map:")
        for r in self.map:
            rowText = ""
            for box in r:
                rowText += str(box[0].work) + " "
            print(rowText)

    def build_factories(self, numFactories):
        """
        Builds factories
        :param numFactories: Number of factories to build (can overlap)
        :return: void
        """
        self.factories = []
        seed = np.random.dirichlet(np.ones(numFactories), size=1) * (self.dim * self.dim * 20)
        for capacity in seed.tolist()[0]:
            self.factories.append(Factory((np.random.randint(0, self.dim),
                                           np.random.randint(0, self.dim)),
                                          int(capacity)))

    def map_factories(self):
        """
        Maps factories and returns array of factories
        :return: array of factory locations (can overlap)
        """
        print("mapFactories")
        factory_map = [[0 for i in range(self.dim)] for i in range(self.dim)]
        for factory in self.factories:
            #             factory_map[factory.loc[0]][factory.loc[1]] = factory.cap
            factory_map[factory.loc[0]][factory.loc[1]] = 1
        return factory_map
        # ax = sns.heatmap(np.array(factory_map), linewidths=.5, xticklabels=False, yticklabels=False, cbar=False)
        # plt.savefig("output.png")

    def map_available_work(self):
        """
        Maps available work and prints available work
        :return: void
        """
        print("mapAvailableWork")
        factory_map = [[0 for i in range(self.dim)] for i in range(self.dim)]
        for factory in self.factories:
            factory_map[factory.loc[0]][factory.loc[1]] = factory.available_workers
        print(np.matrix(factory_map))

    def assign_work(self):
        """
        Stochastically assigns all workers work around the grid nearest to them
        :return: void
        """
        all_workers = []
        for r in self.map:
            for box in r:
                for worker in box:
                    all_workers.append(worker)

        np.random.shuffle(all_workers)
        for worker in all_workers:
            min_travel_cost = worker.travel_cost((-self.dim, -self.dim))
            min_idx = -1
            for idx in range(len(self.factories)):
                factory = self.factories[idx]
                if factory.available_workers > 0:
                    if worker.travel_cost(factory.loc) < min_travel_cost:
                        min_travel_cost = worker.travel_cost(factory.loc)
                        min_idx = idx
            worker.set_workplace(self.factories[min_idx].loc)
            self.factories[min_idx].reduce_work()

    def move(self, rent_ratio):
        """
        Attempts to move a random worker to a better spot.
        :return: Whether the move was successful
        """
        all_workers = []
        for r in self.map:
            for box in r:
                for worker in box:
                    all_workers.append(worker)

        chosen_household = np.random.choice(all_workers)

        suggestion = (np.random.randint(0, self.dim), np.random.randint(0, self.dim))

        current_rent = (len(self.map[chosen_household.home[0]][chosen_household.home[1]])) * rent_ratio
        current_travel_cost = chosen_household.travel_cost(chosen_household.work)
        current_cost = current_rent + current_travel_cost

        density_at_suggestion = len(self.map[suggestion[0]][suggestion[1]])
        new_rent = (density_at_suggestion + 1) * rent_ratio
        new_travel_cost = chosen_household.new_travel_cost(suggestion)
        new_cost = new_rent + new_travel_cost

        if new_cost < current_cost:
            self.map[chosen_household.home[0]][chosen_household.home[1]].remove(chosen_household)
            chosen_household.set_home(suggestion)
            self.map[suggestion[0]][suggestion[1]].append(chosen_household)
            return True
        return False

    def average_density(self):
        """
        Calculates the average density of the map
        :return: the average density of occupied grids in the map
        """
        print("*")
        total = 0
        count = 0
        for row in self.map:
            for box in row:
                if len(box) != 0:
                    total += len(box)
                    count += 1
        return (1.0 * total) / (1.0 * count)
