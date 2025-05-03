import numpy as np
from scipy.stats import norm

class JointDistribution:
    def __init__(self, k):
        """
        Initialize the joint distribution with parameter k (0 <= k <= 1)
        
        Args:
            k (float): Distribution parameter between 0 and 1
        """
        if not 0 <= k <= 1:
            raise ValueError("k must be between 0 and 1")
        self.k = k
        
    def joint_yz(self):
        """
        Returns the joint distribution of (Y,Z) as a dictionary
        where keys are tuples (y,z) and values are probabilities
        """
        return {
            (1, 1): (1 + self.k)/4,
            (1, -1): (1 - self.k)/4,
            (-1, 1): (1 - self.k)/4,
            (-1, -1): (1 + self.k)/4
        }
       
    
    def conditional_x(self, y, z):
        """
        Returns the conditional distribution of X given Y=y and Z=z
        as a normal distribution object from scipy.stats
        
        Args:
            y (int): Binary label (+1 or -1)
            z (int): Binary label (+1 or -1)
        """
        if y not in [-1, 1] or z not in [-1, 1]:
            raise ValueError("y and z must be either +1 or -1")
            
        #Define four distinct cases
        if y == 1 and z == 1:
            return norm(loc=1.0 - self.k, scale=1)
        elif y == 1 and z == -1:
            return norm(loc=1.0, scale=1)
        elif y == -1 and z == 1:
            return norm(loc=0.0, scale=1)
        else:  # y == -1 and z == -1
            return norm(loc=0.0 + self.k, scale=1)

 
    
    def sample(self, size=1):
        """
        Generate samples from the joint distribution (X,Y,Z)
        
        Args:
            size (int): Number of samples to generate
            
        Returns:
            tuple: (X_samples, Y_samples, Z_samples) each of length 'size'
        """
        # First sample Y and Z based on their joint probability
        joint = self.joint_yz()
        yz_pairs = list(joint.keys())
        probs = list(joint.values())
        
        # Sample (Y, Z) pairs according to the joint distribution P(Y, Z)
        yz_indices = np.random.choice(len(yz_pairs), size=size, p=probs)
        y_samples = np.array([yz_pairs[i][0] for i in yz_indices], dtype=np.int8)
        z_samples = np.array([yz_pairs[i][1] for i in yz_indices], dtype=np.int8)
        
        # Sample X based on the conditional distribution P(X | Y, Z)
        x_samples = np.zeros(size, dtype=np.float32)
        for i in range(size):
            # Get the conditional distribution P(X | Y=y_i, Z=z_i)
            dist = self.conditional_x(y_samples[i], z_samples[i])
            # Sample X from this conditional distribution
            x_samples[i] = dist.rvs()
            
        return x_samples, y_samples, z_samples

# Example usage demonstrating class functionality
if __name__ == "__main__":
    # Initialize with k=0.5
    dist = JointDistribution(k=0.5)
    
    # Print joint distribution of Y,Z
    print("Joint distribution of Y,Z:")
    print(dist.joint_yz())
    
    # Print conditional distribution parameters for each Y,Z combination
    print("\nConditional distributions of X:")
    for y in [1, -1]:
        for z in [1, -1]:
            cond = dist.conditional_x(y, z)
            print(f"Y={y}, Z={z}: X ~ N({cond.mean():.2f}, {cond.std():.2f})")
    
    # Generate and print some samples
    print("\nGenerated samples:")
    x, y, z = dist.sample(size=5)
    for i in range(5):
        print(f"Sample {i+1}: X={x[i]:.2f}, Y={y[i]}, Z={z[i]}")