import numpy as np
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps

class Forward_Freq_Survey:
    def __init__(self, frequencies: np.ndarray, z0: float, moment: float, coil_spacing: float):
        self.survey = self._create_survey(frequencies, z0, moment, coil_spacing)

    def _create_survey(self, frequencies: np.ndarray, z0: float, moment: float, coil_spacing: float):
        source_location = np.array([0.0, 0.0, z0])
        source_orientation = "z"

        # Receiver properties
        receiver_locations = np.array([coil_spacing, 0.0, z0])
        receiver_orientation = "z"
        data_type = "ppm"  # "secondary", "total" or "ppm"

        source_list = []  # create empty list for source objects

        # loop over all sources
        for freq in frequencies:
            # Define receivers that measure real and imaginary component
            # magnetic field data in ppm.
            receiver_list = []
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    receiver_locations,
                    orientation=receiver_orientation,
                    data_type=data_type,
                    component="real",
                )
            )
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    receiver_locations,
                    orientation=receiver_orientation,
                    data_type=data_type,
                    component="imag",
                )
            )

            # Define a magnetic dipole source at each frequency
            source_list.append(
                fdem.sources.MagDipole(
                    receiver_list=receiver_list,
                    frequency=freq,
                    location=source_location,
                    orientation=source_orientation,
                    moment=moment,
                )
            )

        # Define the FDEM survey
        return fdem.survey.Survey(source_list)
    
    def predict_data(self, layer_thicknesses, layer_conductivities):
        log_conductivities_model = np.log(layer_conductivities)
        log_conductivities_map = maps.ExpMap(nP=len(layer_thicknesses)+1)
        
        simulation= fdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=layer_thicknesses,
            sigmaMap=log_conductivities_map,
        )

        # Predict 1D FDEM Data
        return simulation.dpred(log_conductivities_model)
