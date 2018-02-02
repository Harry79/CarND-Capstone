
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
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle,th_kp,th_ki,th_kd,th_mn,th_mx,br_kp,br_ki,br_kd,br_mn,br_mx,rate):
        self.yaw_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=min_speed,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)
        self.throttle_controller = PID(
            kp=th_kp,
            ki=th_ki,
            kd=th_kd,
            mn=th_mn,
            mx=th_mx)
        self.brake_controller = PID(
            kp=br_kp,
            ki=br_ki,
            kd=br_kd,
            mn=br_mn,
            mx=br_mx )
        self.rate=rate

    def control(self, current_velocity, target_velocity):
        cur_v = current_velocity.twist.linear.x
        lin_v = target_velocity.twist.linear.x
        ang_v = target_velocity.twist.angular.z

        steering = self.yaw_controller.get_steering(lin_v, ang_v, cur_v)

        # TODO write decent throttle/brake control (PID)
        #throttle = (lin_v-cur_v)
       
        error = (lin_v-cur_v)
        dead_band = 0.05*cur_v
        sample_time = 1.0/self.rate


      
        if error > dead_band:
            throttle = self.throttle_controller.step(error, sample_time)
            brake = 0.0
            self.brake_controller.reset()
        elif error < -dead_band:
            brake = self.brake_controller.step(-error, sample_time)
            throttle = 0.0
            self.throttle_controller.reset()
        else:
            throttle = 0.0
            self.throttle_controller.reset()
            brake = 0.0
            self.brake_controller.reset()
     
        # Return throttle, brake, steering
        return throttle, brake, steering
