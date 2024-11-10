import numpy as np
from discretize import TensorMesh
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg.maps import ExpMap
from simpeg import (
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)

class LayeredInversion:
    def __init__(self, data_object, survey, referece_conductivity, depth_min=0.1, depth_max=3.0, 
                 geometric_factor=1.2, beta0_ratio=1.0, coolingFactor=1.5, coolingRate=2.0, chifact=1.0):
        """
        Initializes the LayeredInversion class with necessary parameters and configurations.
        
        Parameters:
        - data_object: Instance of SimPEG data.Data containing observed data and noise floor.
        - survey: The survey object for the simulation.
        - referece_conductivity: Soil conductivity value for setting up the starting model.
        - log_conductivity_map: Mapping for the conductivity model (typically an instance of maps.ExpMap).
        - depth_min: Minimum thickness of the top layer (default is 0.1).
        - depth_max: Maximum depth to the lowest layer (default is 3.0).
        - geometric_factor: Rate of thickness increase for layer depths (default is 1.2).
        - beta0_ratio, coolingFactor, coolingRate, chifact: Parameters for the inversion directives.
        """
        self.data_object = data_object
        self.survey = survey
        self.referece_conductivity = referece_conductivity
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.geometric_factor = geometric_factor

        # Parameters for inversion directives
        self.beta0_ratio = beta0_ratio
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate
        self.chifact = chifact
        
        # Placeholder for inversion, mesh, model, and simulation components
        self.layer_thicknesses = None
        self.regularization_mesh = None
        self.starting_model = None
        self.reference_conductivity_model = None
        self.simulation = None
        self.inv = None

        # Automatically set up layers, mesh, model, and simulation
        self.setup_layer_thicknesses()
        self.setup_regularization_mesh()
        self.setup_starting_model()
        self.setup_simulation()

    def setup_layer_thicknesses(self):
        """Define layer thicknesses based on geometric progression."""
        self.layer_thicknesses = [self.depth_min]
        while np.sum(self.layer_thicknesses) < self.depth_max:
            self.layer_thicknesses.append(self.geometric_factor * self.layer_thicknesses[-1])

    def setup_regularization_mesh(self):
        """Define 1D cell widths and create regularization mesh."""
        h = np.r_[self.layer_thicknesses, self.layer_thicknesses[-1]]
        h = np.flipud(h)
        self.regularization_mesh = TensorMesh([h], "N")

    def setup_starting_model(self):
        """Define the starting model, conductivity map, and reference conductivity model."""
        n_layers = len(self.layer_thicknesses) + 1
        self.starting_model = np.log(self.referece_conductivity * np.ones(n_layers))
        self.log_conductivity_map = ExpMap(nP=n_layers)
        self.reference_conductivity_model = self.starting_model.copy()
    
    def setup_simulation(self):
            """Initialize the 1D layered simulation."""
            self.simulation = fdem.Simulation1DLayered(
                survey=self.survey, 
                thicknesses=self.layer_thicknesses, 
                sigmaMap=self.log_conductivity_map
            )
    def create_misfit_term(self):
        """Create the data misfit term."""
        return data_misfit.L2DataMisfit(simulation=self.simulation, data=self.data_object)

    def create_regularization(self):
        """Create the regularization term."""
        return regularization.Sparse(
            mesh=self.regularization_mesh,
            reference_model=self.reference_conductivity_model,
            alpha_s=1e-3,
            alpha_x=1e-1,
            norms=[1, 0.5],
        )

    def create_optimization(self):
        """Create the optimization algorithm."""
        return optimization.InexactGaussNewton(
            maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
        )

    def create_directives(self):
        """Create a list of directives for the inversion using class attributes."""
        update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio)
        beta_schedule = directives.BetaSchedule(
            coolingFactor=self.coolingFactor,
            coolingRate=self.coolingRate
        )
        target_misfit = directives.TargetMisfit(chifact=self.chifact)
        sensitivity_weights = directives.UpdateSensitivityWeights()

        return [sensitivity_weights, update_jacobi, starting_beta, beta_schedule, target_misfit]

    def setup_inversion(self):
        """Set up the inversion problem using the misfit, regularization, and optimization."""
        dmis = self.create_misfit_term()
        reg = self.create_regularization()
        reg.depth_weighting = True
        opt = self.create_optimization()
        
        # Create the inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        
        # Attach directives
        directives_list = self.create_directives()
        
        # Define the inversion with the problem and directives
        self.inv = inversion.BaseInversion(inv_prob, directives_list)

    def run_inversion(self, starting_model=None):
        """
        Run the inversion process and return the recovered model.
        
        Parameters:
        - starting_model: Optional initial model for the inversion process. Uses default if None.
        
        Returns:
        - Recovered model from the inversion.
        """
        if self.inv is None:
            self.setup_inversion()
        
        # Run the inversion and return the model
        return self.inv.run(starting_model if starting_model is not None else self.starting_model)
