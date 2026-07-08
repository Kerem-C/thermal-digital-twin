class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float, min_out: float, max_out: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.min_out = min_out
        self.max_out = max_out

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_temp: float, dt: float) -> float:
        error = current_temp - self.setpoint

        p_out = self.kp * error

        self.integral += error * dt
        i_out = self.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_out = self.kd * derivative

        self.prev_error = error

        output = p_out + i_out + d_out

        return max(self.min_out, min(self.max_out, output))