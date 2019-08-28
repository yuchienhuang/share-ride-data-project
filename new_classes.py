import numpy as np
import pandas as pd
import dill
import datetime
from functools import reduce
from collections import defaultdict
from collections import deque
import heapq
import datetime
from dateutil import parser
import math
# from distance_estimator import *

xm, ym, xM, yM = (-74.04772962697038, 40.68291694544516, -73.90665099539478, 40.87903804730727)

harmonic_avg_speed = 2.0816481436507585e-05

class Ride(object):

    def __init__(self, Id, xi, yi, xf, yf, ti, tf, p_count, fare_amount, trip_distance, day, gridMap):
        self.Id = Id
        self.xi = np.array([xi, yi])
        self.xf = np.array([xf, yf])
        self.ti = parser.parse(ti)
        self.tf = parser.parse(tf)
        self.trip_distance = trip_distance
        self.day = day
        self.p_count = int(p_count)
        self.gridMap = gridMap


    def request_time(self, request_ahead=5 * 60):
        return self.ti - datetime.timedelta(seconds=request_ahead)

    def request_vehicle(self):

        self.vehicle = self.gridMap.find_vehicle(self)

class gridMap(object):
    def __init__(self, xy_minmax=(xm, xM, ym, yM), grid_length=0.02):
        self.xy_minmax = xy_minmax
        self.grid_length = grid_length


        self.online_vehicle_map = defaultdict(list)
        self.all_vehicles = dict()
        self.index = -1

#         self.condition_not_met = []

    def reset(self):
        self.online_vehicle_map = defaultdict(list)
        self.all_vehicles = dict()
        self.index = -1

    def folder_name(self, path):
        self.path = path

    def grid_index(self, position):

        xm, xM, ym, yM = self.xy_minmax
        x, y = position
        key = ((x - xm)//self.grid_length, (y- ym)//self.grid_length)
        return key


    def add_vehicle_to_map(self, vehicleId, position):
        grid = self.grid_index(position)
        self.online_vehicle_map[grid].append(vehicleId)


    def remove_vehicle_from_map(self, vehicleId, position):
        grid = self.grid_index(position)
        self.online_vehicle_map[grid].remove(vehicleId)


    def find_vehicle(self, ride):
        vehicle = self._from_existing_vehicle(ride)
        if not vehicle: #if no available vehicle, generate a new one
            self.index += 1
            Id = self.index
            vehicle = Vehicle(Id, ride)
            self.all_vehicles[Id] = vehicle
            self.online_vehicle_map[self.grid_index(ride.xi)].append(Id)

        return vehicle



    def _from_existing_vehicle(self, ride):

        location = ride.xi
        i, j = self.grid_index(location)
        p_count = ride.p_count

        neareast_grids = [(i - ii, j - jj) for ii in range(-1,2) for jj in range(-1,2)]
        nearby_vehicle_Ids = reduce(lambda ls1, ls2: ls1 + ls2, [self.online_vehicle_map[key] for key in neareast_grids])
        nearby_vehicles = [(self.all_vehicles[Id], sum((self.all_vehicles[Id].current_position - ride.xi)**2)) for Id in nearby_vehicle_Ids]
        nearby_vehicles.sort(key=lambda x: x[1])

        for v, _ in nearby_vehicles:
            conditions = self.pick_up_conditions(ride, v)
            if conditions:
                vehicle = v
                break
        else:
            return None



        turning_time, turning_point, new_moving_direction = conditions
        self.update_gridMap(vehicle, turning_point)
        vehicle.add_new_ride(turning_time, turning_point, new_moving_direction, ride)



        return vehicle

    def save_vehicle_data(self, vehicle):
        self.remove_vehicle_from_map(vehicle.Id, vehicle.current_position)
        completed = self.all_vehicles.pop(vehicle.Id)
        completed.end_of_time_statement()

        filename = self.path + datetime.datetime.now().strftime("%d-%H%M-%S%f") + '.pkd'
        dill.dump(completed.data, open(filename, 'wb'))            

    def clean_map(self):
        vehicles = list(self.all_vehicles.values())
        for vehicle in vehicles:
            self.save_vehicle_data(vehicle) 



    def pick_up_conditions(self, ride, vehicle, angle_para=np.pi / 2, detour_para=0.3, too_far_para=1.3):
        
        tr = ride.request_time()
        vehicle.update_vehicle(tr)

        # request before the very first request
        if tr <= vehicle.visited_list[0][1]:

            return False


        #request after the vehicle drop the last ride
        if not vehicle.active:
            self.save_vehicle_data(vehicle)
            
            return False


        # not enough space
        if vehicle.space < ride.p_count:

            return False


        # backtrack
        t_turning = tr
        x_turning = vehicle.location_at_t(t_turning)

        xr = ride.xi
        pick_up_direction = (xr - x_turning) / np.linalg.norm(xr - x_turning)

        def angle(v1, v2):
            cos = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
            try:
                assert(cos <= 1 and cos >= -1)
                return np.arccos(cos)
            except: # cos calculated above could be slightly out of range numerically
                if cos > -1:
                    return np.pi
                else:
                    return 0

        if angle(pick_up_direction, vehicle.moving_direction) > angle_para:

            return False

        # too much detour for the ride itself

        estimate_t = vehicle.arrival_time(ride.xf)

        original_duration = (ride.tf - ride.ti)
        share_ride_duration = (estimate_t - ride.ti)

        increased_ratio = (share_ride_duration - original_duration) / original_duration

        if increased_ratio > detour_para:

            return False

        # too much detour for the vehicle regarding the upcoming stop

        vehicle_current_location = vehicle.current_position
        _, _, original_next_stop, _ = vehicle.to_visit_list[0]


        original_distance = vehicle.distance(original_next_stop, vehicle_current_location)
        to_pick_this_ride_distance = vehicle.distance(original_next_stop, ride.xi) + vehicle.distance(ride.xi, vehicle_current_location)

        if to_pick_this_ride_distance / original_distance >= too_far_para:

            return False


        return (t_turning, x_turning, pick_up_direction)





    def update_gridMap(self, vehicle, new_position):

        self.remove_vehicle_from_map(vehicle.Id, vehicle.current_position)
        self.add_vehicle_to_map(vehicle.Id, new_position)

class Vehicle(object):

    def __init__(self, Id, first_ride, vehicle_speed=harmonic_avg_speed, space=6):
        self.active = True
        self.Id = Id
        self.vehicle_speed = vehicle_speed
        self.start_driving_time = first_ride.ti

        self.space = space - first_ride.p_count
        self.current_time = first_ride.ti
        self.current_position = first_ride.xi
        self.moving_direction = (first_ride.xf - first_ride.xi) / np.linalg.norm(first_ride.xf - first_ride.xi)


        self.velocity = self.vehicle_speed * self.moving_direction



        self.ride_list = deque([first_ride])
        self.completed_ride_list = []

        self.visited_list = [('i', first_ride.ti, first_ride.xi, first_ride.Id)]


        estimate_tf = first_ride.ti + self.duration(first_ride.xf, first_ride.xi)
        self.to_visit_list = deque([('f', estimate_tf, first_ride.xf, first_ride.Id)])







    def arrival_time(self, destination):


        _, currnt_final_t, current_final_x, _ = self.to_visit_list[-1]
        estimate_t = currnt_final_t + self.duration(destination, current_final_x)



        return estimate_t



    def add_new_ride(self, turning_time, turning_point, new_moving_direction, new_ride):
        self.space -= new_ride.p_count
        self.current_time = turning_time
        self.current_position = turning_point
        self.moving_direction = new_moving_direction

        self.velocity = self.vehicle_speed * self.moving_direction


        self.ride_list.append(new_ride)

        self.visited_list.append(('t', turning_time ,turning_point, new_ride.Id))

        estimate_tf = self.arrival_time(new_ride.xf)

        self.to_visit_list.appendleft(('i', new_ride.ti, new_ride.xi, new_ride.Id))
        self.to_visit_list.append(('f', estimate_tf, new_ride.xf, new_ride.Id))




    def location_at_t(self, time):
        return (time - self.current_time).total_seconds() * self.velocity + self.current_position


    def driving_distance(self, node_list):
        total_distance = 0


        for i in range(len(node_list) - 1):
            total_distance += self.distance(node_list[i+1], node_list[i])

        return total_distance


    def update_vehicle(self, time):

        next_event_time = self.to_visit_list[0][1]

        if next_event_time < time:

            if self.to_visit_list[0][0] == 'f':
                to_be_drop_ride = self.ride_list[0]
                self.space -= to_be_drop_ride.p_count
                self.completed_ride_list.append(self.ride_list.popleft())

            self.visited_list.append(self.to_visit_list.popleft())
            if not self.to_visit_list:
                self.active = False



    def distance(self, xf, xi):
        return np.sqrt(sum((xf - xi)**2))


    def duration(self, xf, xi):
        return datetime.timedelta(seconds = self.distance(xf, xi) / self.vehicle_speed)

    def end_of_time_statement(self):

        self.completed_ride_list.extend(list(self.ride_list))
        self.ride_list = deque()

        self.visited_list.extend(list(self.to_visit_list))
        self.to_visit_list = deque()

        no_share_distance = sum(ride.trip_distance for ride in self.completed_ride_list)

        share_distance = distance_predict(self.driving_distance([x for _,_,x,_ in self.visited_list]))[0]
        self.data = (share_distance, no_share_distance, len(self.completed_ride_list), self.visited_list)



