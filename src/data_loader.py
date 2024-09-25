import numpy as np
import random

from pyDOE import lhs


class DataLoader():
    """DataLoader class that provides the input data for the model.
       Configuration is based on Fisher's equation.
    """
    
    # From config (set as class attributes)
    args = ['seed', 'rho']


    def __init__(
            self, 
            config: dict,
        ) -> None:
        """Initialize DataLoader instance with arguments provided by config.
           Sets computational domain.

        Args:
            config (dict): Configuration dictionary.
        """
        for arg in self.args:
            setattr(self, arg, config[arg])  
        np.random.seed(self.seed)
        random.seed(self.seed)

        self._set_domain()


    def _set_domain(
            self
        ) -> None:
        """Sets the computational domain and system parameters.
           Domain information is used for sampling.
        """
        self.x_min, self.x_max = -5, 5
        self.t_min, self.t_max = 0, 0.004
        self.mu = 10

        # Continuous rho
        if isinstance(self.rho, list):
            (self.r_min, self.r_max) = self.rho
        # Discrete rho
        else:
            self.r_min = self.r_max = self.rho
                
        self.X_min = np.array([self.t_min, self.x_min, self.r_min])
        self.X_max = np.array([self.t_max, self.x_max, self.r_max])       
            

    def analytical_solution(
            self, 
            X: np.array,
        ) -> np.array:
        """Provides analytical solution to Fisher's equation.

        Args:
            X (np.array): Coordinates (t,x,rho) for which solution is provided.

        Returns:
            u_analytical (np.array): Analytical solution.
        """
        t, x, rho = np.hsplit(X, 3)
        u_analytical = (1 + np.exp(np.sqrt(rho/self.mu/6)*x - (5*rho/6)*t))**(-2)
        return u_analytical
    
    
    def sample_domain(
            self, 
            N: int = 1024,
            rho: float | list[float] | None = None,
        ) -> tuple[np.array, np.array]:
        """Samples data points using lating-hypercube sampling.
           Either from entire domain, or fixed rho value.
           Also provides analytical solution.

        Args:
            N (int): Number of randomly sampled data points.
            rho (float, None): Fixed rho value, default: entire rho domain

        Returns:
            X (np.array): Sampled data points.
            u (np.array): Respective analytical solution.
        """
        # Latin-hypercube sampling
        X = self.X_min + (self.X_max - self.X_min) * lhs(3, N) 
        # Overwrite values for fixed rho
        if rho is not None:
            if isinstance(rho, list):
                X[:, 2] = np.random.choice(rho, size=N)
            else:
                X[:, 2] = rho
        u = self.analytical_solution(X)
        return X, u
        
        
    def mesh(
            self, 
            rho: float | None = None, 
            N: int = 100
        ) -> tuple[np.array, np.array, np.array, np.array]:
        """Creates uniform mesh over computational domain for a fixed rho.
           Also provides analytical solution, as well as time and space ticks.

        Args:
            rho (float): Reaction rate coefficient for which mesh is generated.
            N (int): Extent of mesh, number of ticks for each dimension.

        Returns:
            X_mesh (np.array): Data points from generated mesh.
            u_mesh (np.array): Respective analytical solution.
            t_ticks (np.array): Ticks for time dimension.
            x_ticks (np.array): Ticks for space dimension.
        """
        # Uniform mesh
        t_ticks = np.linspace(self.t_min, self.t_max, N)
        x_ticks = np.linspace(self.x_min, self.x_max, N)
        tt, xx = np.meshgrid(t_ticks, x_ticks)
        # take default (min) rho if not provided
        if rho is None:
            rho = self.r_min
        X_mesh = np.stack([tt.flatten(), 
                           xx.flatten(), 
                           rho*np.ones(N**2)], axis=1) 
        u_mesh = self.analytical_solution(X_mesh)
        return X_mesh, u_mesh, t_ticks, x_ticks
    
    
    def sample_ICBC(
            self, 
            N: int = 1024
        ) -> tuple[np.array, np.array]:
        """Samples data points for initial and boundary condition.

        Args:
            N (int): Number of sampled data points.

        Returns:
            X_ICBC (np.array): Sampled data points.
            u_ICBC (np.array): Respective analytical solution.
        """
        # Initial conidition (t=0)
        X_IC, _ = self.sample_domain(N)
        X_IC[:,0:1] = np.zeros([N, 1])
        # Bottom boundary condition (x=x_min)
        X_BC_bot, _ = self.sample_domain(N)
        X_BC_bot[:,1:2] = self.x_min * np.ones([N, 1])
        # Top boundary condition (x=x_max)
        X_BC_top, _ = self.sample_domain(N)
        X_BC_top[:,1:2] = self.x_max * np.ones([N, 1])
        
        # Combine and ...
        X_ICBC = np.concatenate([X_IC, X_BC_bot, X_BC_top], axis=0)
        # sample the amount of data points specified
        indx_sample = np.random.choice(np.arange(X_ICBC.shape[0]), size=N)
        X_ICBC = X_ICBC[indx_sample]

        u_ICBC = self.analytical_solution(X_ICBC)
        return X_ICBC, u_ICBC


    def cut(
            self, 
            t: float = 0.04, 
            rho: float = 100, 
            N: int = 100
        ) -> tuple[np.array, np.array]:
        """Creates data points along x-dimension for a fixed time and rho.

        Args:
            t (float): Time at x-cut.
            rho (float): Reaction rate coefficient at x-cut.
            N (int): Number of data points along x-cut.

        Returns:
            X_cut (np.array): Created data points along x-cut.
            u_cut (np.array): Respective analytical solution.
        """
        x_ticks = np.linspace(self.x_min, self.x_max, N)
        
        X_cut = np.stack([t*np.ones(N), 
                          x_ticks, 
                          rho*np.ones(N)], axis=1)
        u_cut = self.analytical_solution(X_cut)
        return X_cut, u_cut