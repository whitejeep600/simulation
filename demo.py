import copy
import random
import time

import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from numpy import mean, arange

from amai.model import EvacuationModel, InputPerson, Asset, RobotAgent
from amai.utils import get_boundary_fields, Spiral


def random_width(w):
    return random.randrange(0, w)


def random_height(h):
    return random.randrange(0, h)


def get_test_exit_coordinates(w, h, exits):
    return random.sample(get_boundary_fields(w, h), exits)


def get_test_people(w, h, conscious, unconscious):
    return [InputPerson(random_width(w), random_height(h), True) for i in range(0, conscious)] + \
           [InputPerson(random_width(w), random_height(h), False) for i in range(0, unconscious)]


def get_test_assets(w, h, assets):
    return [Asset(random.randrange(1, 10), random.randrange(1, 3), random_width(w), random_height(h))
            for i in range(0, assets)]


def visualise_agent(agent):
    x, y = agent.pos
    if type(agent) == RobotAgent:
        color = "blue"
    else:
        color = "green"
    plt.plot([x], [y], color=color, markersize=5, marker='o')

    if agent.prev_field:
        prev_x, prev_y = agent.prev_field
        plt.arrow(prev_x, prev_y, x - prev_x, y - prev_y, color=color)


def visualise_all_agents(model):
    for a in model.schedule.agents:
        visualise_agent(a)


def visualise_unconscious(model):
    for coors in model.object_grid.get_all_unconscious_coordinates():
        x, y = coors
        plt.plot([x], [y], color="red", markersize=5, marker='o')


def visualise_assets(model):
    for coors in model.object_grid.get_all_asset_coordinates():
        x, y = coors
        plt.plot([x], [y], color="pink", markersize=5, marker='d')


def visualise_exit_fields(model: EvacuationModel):
    for coors in model.exit_fields:
        x, y = coors
        plt.plot([x], [y], color="cyan", markersize=5, marker='s')


def get_evacuation_model(exits, robots, t_limit, conscious, unconscious, assets, w, h, no_chance, dist_penalty,
                         unconscious_value, conscious_value):
    return EvacuationModel(t_limit,
                           w,
                           h,
                           get_test_exit_coordinates(w, h, exits),
                           get_test_people(w, h, conscious, unconscious),
                           get_test_assets(w, h, assets), robots,
                           no_chance, dist_penalty,
                           unconscious_value,
                           conscious_value)


def run_test_with_visualisation(exits=5,
                                robots=8,
                                t_limit=100,
                                conscious=6,
                                unconscious=5,
                                assets=8,
                                w=40,
                                h=40):
    no_chance = 0.6
    dist_penalty = 0.8
    unconscious_value = 9
    conscious_value = 7
    model = get_evacuation_model(exits, robots, t_limit, conscious, unconscious, assets, w, h, no_chance,
                                 dist_penalty, unconscious_value, conscious_value)
    plt.ion()
    plt.ylim(-0.5, model.grid.height - 0.5)
    plt.xlim(-0.5, model.grid.width - 0.5)
    plt.gca().invert_yaxis()

    for i in range(t_limit):
        matrix = np.zeros((model.grid.height, model.grid.width))
        cmap = copy.copy(plt.cm.get_cmap('BrBG'))
        plt.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
        visualise_all_agents(model)
        visualise_unconscious(model)
        visualise_assets(model)
        visualise_exit_fields(model)
        plt.gca().invert_yaxis()
        plt.show()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.1)
        plt.clf()
        model.step()


def run_single_test_no_visualisation(exits, robots, t_limit, conscious, unconscious, assets, w, h, no_chance,
                                     dist_penalty, unconscious_value, conscious_value):
    model = get_evacuation_model(exits, robots, t_limit, conscious, unconscious, assets, w, h, no_chance,
                                 dist_penalty, unconscious_value, conscious_value)
    for i in range(t_limit):
        model.step()
    return model.get_evaluation()


# the adjustable parameters are: desirability penalty for objects that currently can't be carried
# in the robots' deliberation; desirability penalty for distance to helpers; and base desirability of
# conscious and unconscious people. They are explained at places of usage in model.py. This function
# will evaluate the given parameters based on random tests, giving them a score between 0 and 7.
def get_adjustable_parameters_evaluation(adjustable_params):
    # a list of model parameters which give us diverse versions of the battlefield
    reps = 6
    test_params = [(10, 7, 100, 30, 30, 20, 50, 50),
                   (10, 20, 20, 30, 30, 20, 50, 50),
                   (10, 5, 40, 3, 30, 20, 5, 50),
                   (1, 4, 60, 15, 3, 5, 20, 20),
                   (10, 5, 30, 30, 30, 20, 100, 1)]
    return mean([run_single_test_no_visualisation(*params, *adjustable_params) for params in reps * test_params])


# This was the first stage of trying to find the optimal parameters. It was done by trying out different
# sets of parameters by brute force, getting evaluation for each set and choosing the best one. However, I
# made the simplifying assumption that the perceived value of conscious and unconscious people could be the
# same, so as to save the computation time. This function returned (0.6, 0.8, 8, 8). It is important to note
# that all the values are within the intervals that I assumed were reasonable to test from - if some of the
# values had been the boundaries of their respective intervals, it would have been necessary to run the test
# again with adjusted intervals. Note: the tests are randomized so running it again might give a different
# set of parameters!
def get_best_parameters_with_same_conscious_and_unconscious():
    all_parameter_sets = [(no_chance, dist, conscious, conscious)
                          for no_chance in arange(0.2, 1, 0.1)
                          for dist in arange(0, 1, 0.1)
                          for conscious in range(6, 14)]
    parameters_with_scores = {get_adjustable_parameters_evaluation(params): params
                              for params in all_parameter_sets}
    return parameters_with_scores[max(parameters_with_scores.keys())]


# I rectified the aforementioned simplifying assumption by assuming the values of the first two
# adjustable parameters and trying out different pairs of values for the conscious_value and
# unconscious_value parameters. This returned (0.6, 0.8, 9, 7). The values of 9 and 7 are close
# to each other, which justifies that it would not be unreasonable to assume the same value for
# conscious and unconscious people. It also shows that collecting unconscious people should be
# preferred by the robots. This is not surprising, since conscious people can often evacuate
# without receiving help.
def get_best_conscious_and_unconscious():
    all_parameter_sets = [(0.6, 0.8, unconscious, conscious)
                          for unconscious in range(6, 14)
                          for conscious in range(6, 14)]
    parameters_with_scores = {get_adjustable_parameters_evaluation(params): params
                              for params in all_parameter_sets}
    return parameters_with_scores[max(parameters_with_scores.keys())]


# get_best_parameters_with_same_conscious_and_unconscious()  # returned (0.6, 0.8, 8, 8)
# print(get_best_conscious_and_unconscious()) # returned (0.6, 0.8, 9, 7)

#print(get_adjustable_parameters_evaluation((0.6, 0.8, 9, 7)))
# tends to give a score of about 2.5. The relative stability of the score shows that
# it doesn't depend much on the randomness of the model generation.
# The score of 2.5 out of a maximum of 7 is acceptable. Given the time limitations
# of the tests, I suppose the perfect algorithm would give a score of 3 or 4.

# run_test_with_visualisation()
# we can see that a lot of people are helped by the tactics of sticking to the boundaries.
# the robots' actions look mostly understandable.

# run_test_with_visualisation(exits=2, assets=6, conscious=0, unconscious=6, robots=10, w=15, h=15, t_limit=30)
# robots sometimes struggle to get organized, and jump between 'carriable' objects. To be fair, this
# showcases the shortcomings of the applied heuristics in carrying coordination.

# run_test_with_visualisation(exits=5, assets=6, conscious=0, unconscious=6, robots=10, w=15, h=15, t_limit=30)
# usually this case looks a bit better than the previous one, showing that better starting positions for robots
# can be very helpful if the simulation time is short. It also reduces the problem of robots sticking to each
# other in early stages.


