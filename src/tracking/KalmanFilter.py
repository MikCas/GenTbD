import numpy as np

class KalmanFilter:
    """
    A Kalman Filter implementation for tracking objects in 2D space.

    The state vector includes position (x, y, w, h) and velocity (vx, vy, vw, vh).
    The filter predicts the next state based on a motion model and updates the state
    using new measurements (bounding boxes).

    Attributes:
        dt (float): Time step for the motion model.
        stdPosition (float): Standard deviation for position noise.
        stdVelocity (float): Standard deviation for velocity noise.
        numDimensions (int): Number of dimensions for position (x, y, w, h).
        state (np.ndarray): State vector [x, y, w, h, vx, vy, vw, vh].
        covariance (np.ndarray): Covariance matrix representing uncertainty in the state.
        stateTransition (np.ndarray): State transition matrix for motion model.
        stateUpdate (np.ndarray): Measurement matrix mapping state to measurement space.
    """

    def __init__(self, stdPosition=1.0 / 20, stdVelocity=1.0 / 160, dt=1.0):
        """
        Initialize the Kalman Filter.

        Args:
            stdPosition (float): Standard deviation for position noise.
            stdVelocity (float): Standard deviation for velocity noise.
            dt (float): Time step for the motion model.
        """
        self.dt = dt
        self.stdPosition = stdPosition
        self.stdVelocity = stdVelocity
        self.numDimensions = 4  # (x, y, w, h)

        # State vector [x, y, w, h, vx, vy, vw, vh]
        self.state = np.zeros((8, 1))

        # Covariance matrix (uncertainty in the state)
        self.covariance = np.eye(8)

        # State transition matrix (how the state evolves over time)
        self.stateTransition = np.eye(8)
        for i in range(self.numDimensions):
            self.stateTransition[i, self.numDimensions + i] = dt

        # Measurement matrix (maps state to measurement space)
        self.stateUpdate = np.eye(4, 8)
    
    def get_state(self) -> np.ndarray:
        """
        Get the current position (x, y, w, h) from the state vector.

        Returns:
            np.ndarray: Current position as [x, y, w, h].
        """
        return self.state[:4, 0]

    def initialise(self, boundingBox):
        """
        Initialize the state with a bounding box and zero velocities.

        Args:
            boundingBox (BoundingBox): Initial bounding box to set the state.
        """
        _, _, w, h = boundingBox.xywh()

        self.state = np.zeros((8, 1))
        self.state[:4, 0] = boundingBox.xywh()  # Initialize position
        self.state[4:, 0] = 0  # Initialize velocity to zero

        # Set standard deviations for position and velocity
        baseWStd = self.stdPosition * w
        baseHStd = self.stdPosition * h
        velWStd = self.stdVelocity * w
        velHStd = self.stdVelocity * h

        # Set covariance matrix
        stdDevs = np.array([
            2 * baseWStd, 2 * baseHStd, 2 * baseWStd, 2 * baseHStd,
            10 * velWStd, 10 * velHStd, 10 * velWStd, 10 * velHStd
        ])
        self.covariance = np.diag(stdDevs ** 2)  # Convert to variance and diagonalize

    def predict(self):
        """
        Predict the next state based on the motion model.
        """
        # Get current width and height to update noise covariance
        w, h = self.state[2, 0], self.state[3, 0]

        baseWStd = self.stdPosition * w
        baseHStd = self.stdPosition * h
        velWStd = self.stdVelocity * w
        velHStd = self.stdVelocity * h

        # Process noise standard deviations
        stdDevs = np.array([
            baseWStd, baseHStd, baseWStd, baseHStd,
            velWStd, velHStd, velWStd, velHStd
        ])
        processNoise = np.diag(stdDevs ** 2)  # Convert to variance

        # Predict the state and covariance
        self.state = np.dot(self.stateTransition, self.state)
        self.covariance = np.dot(
            np.dot(self.stateTransition, self.covariance),
            self.stateTransition.T
        ) + processNoise

    def update(self, boundingBox):
        """
        Update the state using a new detection bounding box.

        Args:
            boundingBox (BoundingBox): New detection bounding box.
        """
        measurement = np.array([boundingBox.xywh()]).T

        # Project the state and covariance to the measurement space
        projectedState, projectedCovariance = self.project()

        # Kalman gain
        B = np.dot(self.covariance, self.stateUpdate.T)
        kalmanGain = np.dot(B, np.linalg.inv(projectedCovariance))

        # Update state with new measurement
        innovation = measurement - projectedState
        self.state += np.dot(kalmanGain, innovation)

        # Update the covariance matrix
        self.covariance -= np.dot(
            kalmanGain,
            np.dot(projectedCovariance, kalmanGain.T)
        )

    def project(self) -> tuple:
        """
        Project the state and covariance into the measurement space.

        Returns:
            tuple: Projected state and covariance in the measurement space.
        """
        w, h = self.state[2, 0], self.state[3, 0]

        baseWStd = self.stdPosition * w
        baseHStd = self.stdPosition * h

        # Measurement noise standard deviations
        stdDevs = np.array([baseWStd, baseHStd, baseWStd, baseHStd])
        measurementNoise = np.diag(stdDevs ** 2)

        # Project the state and covariance into the measurement space
        projectedState = np.dot(self.stateUpdate, self.state)
        projectedCovariance = np.dot(
            np.dot(self.stateUpdate, self.covariance),
            self.stateUpdate.T
        ) + measurementNoise

        return projectedState, projectedCovariance

