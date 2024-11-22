import numpy as np

class KalmanFilter:
    
    def __init__(self, stdPosition=1./20, stdVelocity=1./160, dt=1.0):
        self.dt = dt
        self.stdPosition = stdPosition
        self.stdVelocity = stdVelocity
        
        self.numDimensions = 4  # (x, y, w, h)
        
        # State vector [x, y, w, h, vx, vy, vw, vh]
        self.state = np.zeros((8, 1))
        
        # Covariance Matrix (Uncertainty in the state)
        self.covariance = np.eye(8)
        
        # State transition matrix (how the state evolves)
        self.stateTransition = np.eye(8)
        for i in range(self.numDimensions):
            self.stateTransition[i, self.numDimensions + i] = dt
        
        # Measurement matrix (maps state to measurement space)
        self.stateUpdate = np.eye(4, 8)
    
    def getState(self):
        return self.state[:4, 0]
    
    def initialise(self, boundingBox):
        """Initialise state with bounding box and zero velocities."""
        _, _, w, h = boundingBox.xywh()
        
        self.state = np.zeros((8, 1))
        self.state[:4, 0] = boundingBox.xywh()  # Initialize state vector with bounding box
        self.state[4:, 0] = 0  # Initialize velocity to zero
        
        baseWStd = self.stdPosition * w
        baseHStd = self.stdPosition * h
        velWStd = self.stdVelocity * w
        velHStd = self.stdVelocity * h
        
        # Set standard deviations for position and velocity in covariance
        stdDevs = np.array([
            2 * baseWStd, 2 * baseHStd, 2 * baseWStd, 2 * baseHStd,
            10 * velWStd, 10 * velHStd, 10 * velWStd, 10 * velHStd
        ])
        
        self.covariance = np.diag(stdDevs ** 2)  # Convert to variance and diagonalize

    def predict(self):
        """Predict next state based on the motion model."""
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
        self.covariance = np.dot(np.dot(self.stateTransition, self.covariance), self.stateTransition.T) + processNoise

    def update(self, boundingBox):
        """Update state using new detection bounding box."""
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
        self.covariance -= np.dot(kalmanGain, np.dot(projectedCovariance, kalmanGain.T))

    def project(self):
        """Project the state to the measurement space."""
        w, h = self.state[2, 0], self.state[3, 0]
        
        baseWStd = self.stdPosition * w
        baseHStd = self.stdPosition * h
        
        # Measurement noise standard deviations
        stdDevs = np.array([baseWStd, baseHStd, baseWStd, baseHStd])
        measurementNoise = np.diag(stdDevs ** 2)
        
        # Project the state and covariance into the measurement space
        projectedState = np.dot(self.stateUpdate, self.state)
        projectedCovariance = np.dot(np.dot(self.stateUpdate, self.covariance), self.stateUpdate.T) + measurementNoise
        
        return projectedState, projectedCovariance
