### Plans

**00 - 10:**: Experimental
**10 - 19:**: Forward/backward, left/right straight flights
**20 - 29:**: Planar diagonal flights
**30 - 39:**: Turning left/right
**40 - 49:**: Eliptical and circular motion
**50 - 59:**: Takeoff/landing
**60 - 69:**: Limited step random walk


### Model

1. LSTM
2. NARX

Inputs:

- Potentiometer inputs (x,y,z,rot)
- Position inputs (x,y,z)
- Pitch, roll, yaw

Inferred:
- Velocity/Acceleration

Outputs:
- Position and pitch, roll, yaw

