import numpy as np
import matplotlib.pyplot as plt
from simpeg.utils import plot_1d_layer_model

class Layered_Earth:
    """
    Represents a layered Earth model that includes a UXO object embedded in a 
    homogeneous background, enabling calculation of conductivity profiles at 
    specific points and visualizing the layered Earth structure.

    Attributes:
        background_conductivity (float): Conductivity of the surrounding Earth background.
        uxo_object: An instance of a UXO object with specified properties such as 
                    conductivity and geometry.
    """
    def __init__(self, background_conductivity: float, uxo_object) -> None:
        """
        Initializes the Layered_Earth model with a specified background conductivity
        and a UXO object instance.

        Args:
            background_conductivity (float): Conductivity of the background Earth layer.
            uxo_object: An instance of a UXO object representing a buried object.
        """
        self.background_conductivity = background_conductivity
        self.uxo_object = uxo_object
    
    def layers_at_pt(self, pt: tuple):
        """
        Determines the layer thicknesses and conductivities at a given point 
        by calculating intersections with the UXO object.

        Args:
            pt (tuple): A (x, y) coordinate tuple where the conductivity profile is calculated.

        Returns:
            tuple: A tuple containing:
                - layer_thicknesses (np.ndarray): Array of layer thicknesses (distance along the z-axis).
                - layer_conductivities (np.ndarray): Array of conductivities for each layer.
        """
        intersects = self.uxo_object.get_vertical_intersects(pt[0], pt[1])
        if intersects is None:
            layer_thicknesses = [3] # Depth of survey for plotting
            layer_conductivities = [self.background_conductivity]
        else:
            layer_thicknesses = np.abs(np.array(intersects))
            layer_conductivities = np.r_[self.background_conductivity,
                                            self.uxo_object.conductivity,
                                            self.background_conductivity]
        return layer_thicknesses, layer_conductivities

    def plot_layered_earth(self, pt: tuple, ax=None):
        """
        Plots the 1D layered Earth model conductivity profile at a specified point.

        Args:
            pt (tuple): A (x, y) coordinate tuple where the profile is evaluated and plotted.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new subplot is created.

        Returns:
            ax: aAxes object displaying the 1D conductivity profile of the Earth model.
        """
        layer_thicknesses, layer_conductivities = self.layers_at_pt(pt)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ax = plot_1d_layer_model(layer_thicknesses, layer_conductivities, scale="log", ax=ax)
        ax.grid(which="both")
        ax.set_xlabel(r"Conductivities ($S/m$)")
        ax.set_ylim([3, 0])
        ax.set_title(f"1D Layerd Earth at Point ({pt[0]}, {pt[1]})")

        return ax
        