
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


# Tools
def vector_magnitude(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.yaw_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=min_speed,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)

    def control(self, current_velocity, target_velocity):
        cur_v = current_velocity.twist.linear.x
        lin_v = target_velocity.twist.linear.x
        ang_v = target_velocity.twist.angular.z

        steering = self.yaw_controller.get_steering(lin_v, ang_v, cur_v)

        # TODO write decent throttle/brake control (PID)
        throttle = (lin_v-cur_v)
        brake = 0.0
        if throttle > 1.0:
            throttle = 1.0
        if throttle < 0.0:
            throttle = 0.0

        # Return throttle, brake, steering
        return throttle, brake, steering
