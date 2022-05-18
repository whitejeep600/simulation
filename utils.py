import random

from numpy import sign


def distance(field1, field2):
    x1, y1 = field1
    x2, y2 = field2
    return max(abs(x1-x2), abs(y1-y2))


def distance_sort_key(field):
    def key(other_field):
        return distance(field, other_field)
    return key


def robot_distance_sort_key(field):
    def key(robot):
        return distance(field, robot.pos)
    return key


def next_field_towards(field1, field2):
    x2, y2 = field2
    x1, y1 = field1
    next_x = x1 + sign(x2 - x1)
    next_y = y1 + sign(y2 - y1)
    return next_x, next_y


def remove_carriable_from_map(carriable, carriable_map):
    if carriable_map[(carriable.x, carriable.y)] == [carriable]:
        carriable_map.pop((carriable.x, carriable.y))
    else:
        carriable_map[(carriable.x, carriable.y)].remove(carriable)


def get_id(person):
    return person.unique_id


def get_boundary_fields(w, h):
    return [(0, i) for i in range(0, h)] + [(w - 1, i) for i in range(0, h)] + \
           [(i, 0) for i in range(1, w)] + [(i, h - 1) for i in range(1, w)]


def get_field_to_left(field: (int, int)) -> (int, int):
    (x, y) = field
    return x-1, y


def get_field_to_right(field: (int, int)) -> (int, int):
    (x, y) = field
    return x+1, y


def get_field_up(field: (int, int)) -> (int, int):
    (x, y) = field
    return x, y+1


def get_field_down(field: (int, int)) -> (int, int):
    (x, y) = field
    return x, y-1


def spiral_distances_generator():
    i = 5
    while True:
        yield i
        i += 2


class Spiral:
    def __init__(self):
        self.direction = random.choice([0, 1, 2, 3]) # 0 - up, 1 - right, 2 - down, 3 - left
        self.distances_gen = spiral_distances_generator()
        self.agenda = (3, self.direction)  # the number of fields to advance in a direction, and the direction

    def generate(self):
        self.direction = (self.direction + 1) % 4
        res = (next(self.distances_gen), self.direction)
        return res

    def next_direction(self):
        fields, direction = self.agenda
        if fields == 0:
            self.agenda = self.generate()
            fields, direction = self.agenda
        self.agenda = fields-1, direction
        return direction

    def next_for_field(self, field):
        x, y = field
        direction = self.next_direction()
        if direction == 0:
            return x, y+1
        elif direction == 0:
            return x+1, y
        elif direction == 0:
            return x, y-1
        else:
            return x-1, y


def divide_or_one(a, b):
    if b == 0:
        return 1
    else:
        return a / b
