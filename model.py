import random

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid

from amai.utils import distance_sort_key, distance, next_field_towards, remove_carriable_from_map, get_id, \
    get_boundary_fields, get_field_to_right, get_field_down, get_field_to_left, get_field_up, robot_distance_sort_key, \
    Spiral, divide_or_one


# just for handling the input.
class InputPerson:
    def __init__(self, x: int, y: int, conscious: bool):
        self.x = x
        self.y = y
        self.conscious = conscious


# Conscious person or Carriable.
class Rescuable:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_value(self):
        raise "not implemented"


# Unconscious person or Asset.
class Carriable(Rescuable):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.carried = False  # only True if actually on the move
        self.carrying_robots = []  # can be non-empty if some agents are waiting for help picking something up.

    def get_weight(self) -> int:
        raise "not implemented"

    def still_required_to_carry(self) -> int:
        return self.get_weight() - len(self.carrying_robots) - 1


class Asset(Carriable):
    def __init__(self, value: float, weight: int, x: int, y: int):
        super().__init__(x, y)
        self.value = value
        self.weight = weight

    def get_value(self) -> float:
        return self.value

    def get_weight(self) -> int:
        return self.weight


class Unconscious(Carriable):
    def __init__(self, x: int, y: int, value: float):
        super().__init__(x, y)
        self.value = value

    def get_value(self) -> float:
        return self.value

    def get_weight(self) -> int:
        return 2


# an additional grid for Carriables.
class CarriableGrid:
    def __init__(self):
        self.coors_to_unconscious = dict()  # a dictionary mapping coordinates to lists of unconscious people
        self.coors_to_assets = dict()  # a dictionary mapping coordinates to lists of assets

    def add_asset(self, asset: Asset):
        if self.coors_to_assets.get((asset.x, asset.y)):
            self.coors_to_assets[(asset.x, asset.y)] += [asset]
        else:
            self.coors_to_assets[(asset.x, asset.y)] = [asset]

    def add_unconscious(self, x: int, y: int, new_unconscious: Unconscious):
        if self.coors_to_unconscious.get((x, y)):
            self.coors_to_unconscious[(x, y)] += [new_unconscious]
        else:
            self.coors_to_unconscious[(x, y)] = [new_unconscious]

    def get_all_unconscious_coordinates(self) -> [(int, int)]:
        return self.coors_to_unconscious.keys()

    def get_all_asset_coordinates(self) -> [(int, int)]:
        return self.coors_to_assets.keys()

    def move(self, carriable: Carriable, next_pos: (int, int)):
        (x, y) = next_pos
        if type(carriable) == Asset:
            self.remove_asset(carriable)
            carriable.x, carriable.y = x, y
            self.add_asset(carriable)
        else:
            self.remove_unconscious(carriable)
            carriable.x, carriable.y = x, y
            self.add_unconscious(carriable.x, carriable.y, carriable)

    def remove_asset(self, asset: Asset):
        remove_carriable_from_map(asset, self.coors_to_assets)

    def remove_unconscious(self, unconscious: Unconscious):
        remove_carriable_from_map(unconscious, self.coors_to_unconscious)


class EvacuationModel(Model):
    def __init__(self, total_time: int, width: int, height: int, exit_fields: (int, int), people: [InputPerson],
                 assets: [Asset], robots_number: [int], no_chance_desirability_parameter: float,
                 distance_penalty_parameter: float, unconscious_person_value: float, conscious_person_value: float):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.object_grid = CarriableGrid()
        self.schedule = SimultaneousActivation(self)
        self.exit_fields = exit_fields
        self.elapsed_time = 0
        self.total_time = total_time
        self.rescued_conscious = 0
        self.rescued_unconscious = 0
        self.secured_assets_value = 0
        self.person_agents = []
        self.robots = []
        self.unconscious = []
        self.assets = assets
        # parameters, explained at places of usage:
        self.no_chance_desirability_parameter = no_chance_desirability_parameter
        self.distance_penalty_parameter = distance_penalty_parameter
        self.conscious_person_value = conscious_person_value
        added_agents = 0
        for i in range(0, robots_number):
            robot = RobotAgent(added_agents, self)
            added_agents += 1
            self.schedule.add(robot)
            self.grid.place_agent(robot,
                                  random.choices(self.exit_fields, weights=self.get_exit_field_likelihoods(), k=1)[0])
            self.robots += [robot]

        for person in people:
            if person.conscious:
                person_agent = PersonAgent(added_agents, self, person.x, person.y)
                added_agents += 1
                self.schedule.add(person_agent)
                self.grid.place_agent(person_agent, (person.x, person.y))
                self.person_agents += [person_agent]
            else:
                new_unconscious = Unconscious(person.x, person.y, unconscious_person_value)
                self.unconscious += [new_unconscious]
                self.object_grid.add_unconscious(person.x, person.y, new_unconscious)

        for asset in assets:
            self.object_grid.add_asset(asset)

        self.running = True

    # robots will be placed on exit fields randomly, but with probabilities corresponding to how
    # close the assets amd unconscious people are to each exit field.
    def get_exit_field_likelihoods(self) -> [(int, int)]:
        return [1 / sum([self.get_expected_rescue_time_from_field(ef, res)
                         for res in self.unconscious + self.assets])
                for ef in self.exit_fields]

    def remove_conscious(self, conscious):
        self.schedule.remove(conscious)
        x, y = conscious.pos
        self.grid.remove_agent(conscious)
        self.rescued_conscious += 1
        self.person_agents.remove(conscious)
        print("{}, #{}, ({}, {})".format(self.elapsed_time, conscious.unique_id, x, y))

    def remove_carriable(self, carriable: Carriable):
        if type(carriable) == Asset:
            self.object_grid.remove_asset(carriable)
            self.assets.remove(carriable)
            self.secured_assets_value += carriable.value
        else:  # type = Unconscious
            self.object_grid.remove_unconscious(carriable)
            self.unconscious.remove(carriable)
            self.rescued_unconscious += 1

    def print_summary(self):
        original_conscious_number = len([a for a in self.person_agents if type(a) == PersonAgent]) \
                                    + self.rescued_conscious
        original_unconscious_number = len(self.unconscious) + self.rescued_unconscious
        total_assets_value = sum(a.value for a in self.assets) + self.secured_assets_value
        print("Total number of conscious people saved: " + str(self.rescued_conscious) +
              " out of " + str(original_conscious_number))
        print("Total number of unconscious people saved: " + str(self.rescued_unconscious) +
              " out of " + str(original_unconscious_number))
        print("Total value of secured assets: " + str(self.secured_assets_value) +
              " out of " + str(total_assets_value))

    def get_evaluation(self):
        original_conscious_number = len([a for a in self.person_agents if type(a) == PersonAgent]) \
                                    + self.rescued_conscious
        original_unconscious_number = len(self.unconscious) + self.rescued_unconscious
        total_assets_value = sum(a.value for a in self.assets) + self.secured_assets_value
        return 3 * divide_or_one(self.rescued_conscious, original_conscious_number) + \
               3 * divide_or_one(self.rescued_unconscious, original_unconscious_number) + \
               divide_or_one(self.secured_assets_value, total_assets_value)

    def step(self):
        if self.total_time - self.elapsed_time > 1:
            self.schedule.step()
            self.elapsed_time += 1
        elif self.running:
            self.print_summary()
            pass
        else:
            self.running = False

    def nearest_exit_for_field(self, position) -> (int, int):
        all_exits = self.exit_fields
        return sorted(all_exits, key=distance_sort_key(position))[0]

    def get_cell_list_contents(self, cell_list):
        # had to implement this here because the 'original' in Grid seems to be buggy.
        return [a for a in self.person_agents + self.robots if a.pos in cell_list]

    def get_expected_rescue_time_from_field(self, field, rescuable: Rescuable) -> int:
        rescuable_pos = (rescuable.x, rescuable.y)
        if type(rescuable) == Unconscious:
            slowdown = 2
        else:
            slowdown = 1
        return distance(field, rescuable_pos) + \
               slowdown * distance(rescuable_pos, self.nearest_exit_for_field(rescuable_pos))


# includes conscious people only. Unconscious people don't make sense to be modeled
# as agents since they take no independent actions.
class PersonAgent(Agent, Rescuable):
    def __init__(self, unique_id, model: EvacuationModel, x: int, y: int):
        super().__init__(unique_id, model)
        self.model = model
        self.evacuated = False
        self.next_field = None
        self.guiding_robot = None
        self.followed_person = None
        self.prev_field = None
        self.spiral = Spiral()

    @property
    def x(self):
        x, _ = self.pos
        return x

    @property
    def y(self):
        _, y = self.pos
        return y

    def get_value(self) -> float:
        return self.model.conscious_person_value

    def on_exit_field(self) -> bool:
        return self.pos in self.model.exit_fields

    def exit_battlefield(self):
        self.evacuated = True
        self.model.remove_conscious(self)

    def get_neighboring_exit_fields(self) -> [(int, int)]:
        neighboring = self.model.grid.get_neighborhood(self.pos, moore=True)
        return [field for field in neighboring if field in self.model.exit_fields]

    def get_human_cellmates(self) -> [Agent]:
        contents = [agent for agent in self.model.schedule.agents if agent.pos == self.pos and agent is not self]
        return [agent for agent in contents if type(agent) == PersonAgent]

    def join_person(self, person):
        if person.unique_id < self.unique_id:
            self.followed_person = person

    def people_to_left_or_up(self) -> [Agent]:
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        self_x, self_y = self.pos
        up_and_left_fields = [(x, y) for (x, y) in neighborhood if x <= self_x and y >= self_y]
        agents = self.model.get_cell_list_contents(up_and_left_fields)
        return [agent for agent in agents if type(agent) == PersonAgent]

    def people_to_right_or_down(self) -> [Agent]:
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        self_x, self_y = self.pos
        right_or_down_fields = [(x, y) for (x, y) in neighborhood if x >= self_x and y <= self_y]
        agents = self.model.get_cell_list_contents(right_or_down_fields)
        return [agent for agent in agents if type(agent) == PersonAgent]

    def visible_robots(self) -> [Agent]:
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        agents = self.model.get_cell_list_contents(neighborhood)
        return [agent for agent in agents if type(agent) == RobotAgent]

    def on_grid_boundary(self) -> bool:
        # I assume people can see if they are there.
        return self.pos in get_boundary_fields(self.model.grid.width, self.model.grid.height)

    def on_upper_grid_boundary(self) -> bool:
        return self.y == self.model.grid.height - 1

    def on_lower_grid_boundary(self) -> bool:
        return self.y == 0

    def on_left_grid_boundary(self) -> bool:
        return self.x == 0

    def on_right_grid_boundary(self) -> bool:
        return self.x == self.model.grid.width - 1

    def plan_boundary_move(self) -> (int, int):
        if self.on_upper_grid_boundary():
            if not self.on_right_grid_boundary():
                return get_field_to_right(self.pos)
            else:
                return get_field_down(self.pos)
        if self.on_right_grid_boundary():
            if not self.on_lower_grid_boundary():
                return get_field_down(self.pos)
            else:
                return get_field_to_left(self.pos)
        if self.on_lower_grid_boundary():
            if not self.on_left_grid_boundary():
                return get_field_to_left(self.pos)
            else:
                return get_field_up(self.pos)
        if self.on_left_grid_boundary():
            if not self.on_upper_grid_boundary():
                return get_field_up(self.pos)
            else:
                return get_field_to_right(self.pos)

    def plan_move(self) -> (int, int):
        if self.on_exit_field():
            return None
        neighboring_exit_fields = self.get_neighboring_exit_fields()
        if neighboring_exit_fields:
            return neighboring_exit_fields[0]
        if self.guiding_robot:
            return self.guiding_robot.next_field
        if self.followed_person:
            return self.followed_person.next_field
        human_cellmates = self.get_human_cellmates()
        if human_cellmates:
            self.join_person(sorted(human_cellmates, key=get_id)[0])
            if self.followed_person:
                return self.followed_person.next_field
        left_or_up = self.people_to_left_or_up()
        if left_or_up:  # move to join the person
            return left_or_up[0].pos
        right_or_down = self.people_to_right_or_down()
        if right_or_down:  # wait to be joined. broken symmetry ensures nothing funny will happen.
            return self.pos
        if self.visible_robots():  # wait to be picked up by the robot.
            return self.pos
        if self.on_grid_boundary():
            if self.model.grid.height >= 2 and self.model.grid.width >= 2:
                # sticking to grid boundaries gives higher chance of finding exit fields because these
                # are located on boundaries. However, it's less sensible if grid is very thin or low.
                return self.plan_boundary_move()
            else:
                return random.choice(self.model.grid.get_neighborhood(self.pos, moore=True))
        # if nothing more sensible can be done, go in a spiral to cover the biggest possible area.
        return self.spiral.next_for_field(self.pos)

    def step(self):
        self.next_field = self.plan_move()

    def advance(self):
        self.prev_field = self.pos
        if not self.next_field:
            self.exit_battlefield()
            return
        self.model.grid.move_agent(self, self.next_field)
        self.next_field = None


class RobotAgent(Agent):
    def __init__(self, unique_id, model: EvacuationModel):
        super().__init__(unique_id, model)
        self.model = model
        self.next_field = None
        self.objective_field = None
        self.carried_object = None
        self.waiting_for_help = False
        self.waited_first_step_while_carrying = False
        self.prev_field = None

    def get_unguided_human_cellmates(self) -> [PersonAgent]:
        contents = [agent for agent in self.model.schedule.agents if agent.pos == self.pos]
        people = [agent for agent in contents if type(agent) == PersonAgent]
        return [person for person in people if not person.guiding_robot]

    def nearest_exit(self) -> (int, int):
        return self.model.nearest_exit_for_field(self.pos)

    def get_maximum_grid_distance(self):
        return (self.model.grid.width ** 2 + self.model.grid.height ** 2) ** (1 / 2)

    def get_distance_to_helpers(self, carriable: Carriable, n):
        # returns the distance to the farthest one out of the nearest n robots that could help carry something.
        robots = [r for r in self.model.robots if not r.carried_object]
        if len(robots) < n:
            # means there's no chance of getting help. We should return something that means
            # this object is undesirable (but not for a desirability of zero). The higher
            # the no_chance_desirability_parameter, the lower the perceived desirability. The
            # parameter should be in range (0, 1].
            return self.get_maximum_grid_distance() * self.model.no_chance_desirability_parameter
        farthest = sorted(robots, key=robot_distance_sort_key((carriable.x, carriable.y)))[n - 1]
        return distance(farthest.pos, (carriable.x, carriable.y))

    def chance_to_get_help(self, carriable: Carriable) -> float:
        # a robot should be more willing to start carrying an object if it is likely to get help.
        # This likelihood is heuristically taken into account as a multiplier from range [0, 1]
        # which is used to adjust the object's perceived desirability (see below).
        # The distance penalty parameters measures how much we penalize distance to the object.
        # 1 means quite harshly, i. e. desirability=0 for an object a whole grid diagonal away.
        # 0 means no penalization at all.
        still_required = carriable.still_required_to_carry()
        if still_required == 0:
            return 1
        base_distance = self.get_distance_to_helpers(carriable, still_required) / self.get_maximum_grid_distance()
        return 1 - base_distance * self.model.distance_penalty_parameter

    def get_desirability(self, rescuable: Rescuable) -> float:
        expected_rescue_time = self.model.get_expected_rescue_time_from_field(self.pos, rescuable)
        if expected_rescue_time == 0:
            return 0
        base_desirability = rescuable.get_value() / expected_rescue_time
        if type(rescuable) in (Unconscious, Asset):
            return base_desirability / rescuable.get_weight() * self.chance_to_get_help(rescuable) * \
                   (rescuable.still_required_to_carry() + 1)
            # perceived desirability is lower if carrying an object will engage more robots.
            # it also depends on the estimated chance of getting help for carrying.
            # in addition, in order to make 'deadlocks' (situations in which many robots are waiting for help
            # instead of helping each other) less likely, we multiply the perceived desirability by
            # 1 + number of robots already waiting for help.
        else:
            return base_desirability

    def desirability_key(self):
        def calculate_desirability(entity):
            return self.get_desirability(entity)

        return calculate_desirability

    def get_uncarried_carriable(self) -> [Carriable]:
        return [carriable for carriable in self.model.assets + self.model.unconscious if not carriable.carried]

    def get_unguided_people(self) -> [PersonAgent]:
        return [person for person in self.model.person_agents if not person.guiding_robot]

    def get_rescuable_ranked_by_desirability(self) -> [Rescuable]:
        all_rescuable = self.get_uncarried_carriable() + self.get_unguided_people()
        sorted_rescuable = sorted(all_rescuable, key=self.desirability_key(), reverse=True)
        return sorted_rescuable

    def drop_object(self):
        self.model.remove_carriable(self.carried_object)
        for agent in self.carried_object.carrying_robots:
            agent.carried_object = None

    def start_carrying(self, carriable: Carriable):
        self.carried_object = carriable
        self.objective_field = self.nearest_exit()
        self.next_field = next_field_towards(self.pos, self.objective_field)
        self.waiting_for_help = False

    def guided_people(self) -> [PersonAgent]:
        return [person for person in self.model.person_agents if person.guiding_robot == self]

    def next_field_to_rescue(self, rescuable: Rescuable) -> (int, int):
        if self.pos == (rescuable.x, rescuable.y):
            # The case in which we should pick up conscious humans from current grid field has already been
            # handled. rescuable is, therefore, Carriable in this case.
            rescuable.carrying_robots += [self]
            if len(rescuable.carrying_robots) == rescuable.get_weight():
                rescuable.carried = True
                for agent in rescuable.carrying_robots:
                    agent.start_carrying(rescuable)
                return next_field_towards(self.pos, self.objective_field)
            else:
                self.waiting_for_help = True
                return self.pos
        else:
            if type(rescuable) != PersonAgent:  # we save some computation in next steps if target can't move
                self.objective_field = (rescuable.x, rescuable.y)
            return next_field_towards(self.pos, (rescuable.x, rescuable.y))

    def plan_action_while_waiting(self) -> (int, int):
        if type(self.carried_object) == Unconscious and self.get_unguided_human_cellmates():
            # this is the only situation in which robots use conscious humans' help in carrying unconscious ones.
            cellmate = self.get_unguided_human_cellmates()[0]
            cellmate.guiding_robot = self
            self.carried_object.carried = True
            self.start_carrying(self.carried_object)
            return next_field_towards(self.pos, self.objective_field)
        else:
            return self.pos

    def plan_move_for_guiding(self, unguided_cellmates):
        for cellmate in unguided_cellmates:
            cellmate.guiding_robot = self
        if not self.objective_field:
            self.objective_field = self.nearest_exit()

    def plan_move_for_new_target(self) -> (int, int):
        all_rescuable = self.get_rescuable_ranked_by_desirability()
        if not all_rescuable:
            return random.choice(self.model.grid.get_neighborhood(self.pos, moore=True))
        most_desirable = all_rescuable[0]
        return self.next_field_to_rescue(most_desirable)

    def plan_move(self) -> (int, int):
        if self.waiting_for_help:
            return self.plan_action_while_waiting()
        unguided_cellmates = self.get_unguided_human_cellmates()
        if unguided_cellmates:
            self.plan_move_for_guiding(unguided_cellmates)
        if self.objective_field:
            if self.objective_field == self.pos:
                self.objective_field = None
                if self.carried_object:
                    self.drop_object()
            else:
                return next_field_towards(self.pos, self.objective_field)
        return self.plan_move_for_new_target()

    def step(self):
        if self.carried_object and type(self.carried_object) == Unconscious:
            if not self.waited_first_step_while_carrying:
                self.waited_first_step_while_carrying = True
                self.next_field = self.pos
                return
            else:
                self.waited_first_step_while_carrying = False
        self.next_field = self.plan_move()

    def advance(self):
        self.prev_field = self.pos
        self.model.grid.move_agent(self, self.next_field)
        if self.carried_object:
            (x, y) = self.pos
            if x != self.carried_object.x or y != self.carried_object.y:
                self.model.object_grid.move(self.carried_object, self.next_field)
        self.next_field = None
