# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import numpy as np
import logging; logger = logging.getLogger(__name__)

from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
from opendrift.models.physics_methods import hour_angle, solar_elevation

# Defining the properties for plastic capable of biofouling
class BioPlast(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for biofoulable plastic
    """

    variables = Lagrangian3DArray.add_variables([
        ('unfouled_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}),  # 
        ('unfouled_density', {'dtype': np.float32,
                                       'units':'kg/m^3',
                                       'default': 1028}),  # 
        ('biofilm_no_attached_algae', {'dtype': np.float32,
                                       'units': '',
                                       'default': 0}),
        ('total_density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),
        ('total_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014})])


class BioPlastDrift(OceanDrift):
    """Buoyant particle trajectory model based on the OpenDrift framework.

        Developed at MET Norway

        Generic module for particles that are subject to vertical turbulent
        mixing with the possibility for positive or negative buoyancy

        Particles could be e.g. oil droplets, plankton, or sediments

        Under construction.
    """

    ElementType = BioPlast

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_ice_area_fraction': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 1000},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'surface_downward_x_stress': {'fallback': 0},
        'surface_downward_y_stress': {'fallback': 0},
        'turbulent_kinetic_energy': {'fallback': 0},
        'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'mass_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water':{'fallback':0},
        'mass_concentration_of_phytoplankton_expressed_as_chl_in_sea_water':{'fallback':0},
        'surface_net_downward_radiative_flux':{'fallback':0}
      }

    # Default colors for plotting
    status_colors = {'initial': 'green', 'active': 'blue',
                     'hatched': 'red', 'eaten': 'yellow', 'died': 'magenta'}


    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(BioPlastDrift, self).__init__(*args, **kwargs)

        # By default, particles do not strand towards coastline
        self.set_config('general:coastline_action', 'previous')

        # Vertical mixing is enabled by default
        self.set_config('drift:vertical_mixing', True)

        # Parameters for biofilm module
        self._add_config({
            'biofilm:biofilm_density':{'type':'float', 'default':1388.,
                                'min': None, 'max': None, 'units':'kg/m^3',
                                'description': 'Density (rho) of material collected on plastic',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:grazing_rate':{'type':'float', 'default':0.39,
                                'min': None, 'max': None, 'units': 'day^-1',
                                'description': 'Grazing rate on biofilm',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:respiration_rate':{'type':'float', 'default':0.1,
                                'min': None, 'max': None, 'units': 'day^-1',
                                'description': 'Respiration rate of algae on biofilm',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:temperature_coeff_respiration':{'type':'float', 'default':2,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Q10',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:algal_cell_volume':{'type':'float', 'default':2*10**-16,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Va',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:algal_cell_radius':{'type':'float', 'default':7.78*10**-6,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Ra',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:algal_carbon_per_cell':{'type':'float', 'default':2726*10**-9,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Used to convert phytoplankton C to no of cells',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:shear':{'type':'float', 'default':1.7*10**-5,
                                'min': None, 'max': None, 'units': 'seconds',
                                'description': 'Shear rate',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:growth_temperature_max':{'type':'float', 'default':33.3,
                                'min': 32.3, 'max': 34.2, 'units': 'degrees C',
                                'description': 'Max temperature for algal growth',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:growth_temperature_min':{'type':'float', 'default':0.2,
                                'min': -3.7, 'max': 3.2, 'units': 'degrees C',
                                'description': 'Min temperature for algal growth',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:growth_temperature_optimal':{'type':'float', 'default':26.7,
                                'min': 26.0, 'max': 27.5, 'units': 'degrees C',
                                'description': 'Optimal temperature for algal growth',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:maximum_growth_rate':{'type':'float', 'default':1.85,
                                'min': 1.38, 'max': 2.17, 'units': 'days -1',
                                'description': 'Maximum algal growth rate',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:initial_slope':{'type':'float', 'default':0.12,
                                'min': 0.05, 'max': 0.21, 'units': 'days -1',
                                'description': 'Initial slope',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'biofilm:growth_light_optimal':{'type':'float', 'default':1.75392*19**13,
                                'min': None, 'max': None, 'units': 'micro Em-2d-1',
                                'description': 'Optimal light intensity algae growth',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'light:twilight':{'type':'float', 'default':15,
                                'min': 0., 'max': 90., 'units': 'degrees',
                                'description': 'angle below the horizon for twilight',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'light:nu':{'type':'float', 'default':430.,
                                'min': 100, 'max': 1000, 'units': 'nm',
                                'description': 'Wavelength used to calculate irradiance',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'light:k_water':{'type':'float', 'default':0.2,
                                'min': 0., 'max': 1, 'units': '',
                                'description': 'coefficient of exponential decay of light in water',
                                'level': self.CONFIG_LEVEL_ADVANCED},
            'light:k_chl':{'type':'float', 'default':0.02,
                                'min': 0, 'max': 1, 'units': '',
                                'description': 'coefficient of exponential decay of light in water due to algae',
                                'level': self.CONFIG_LEVEL_ADVANCED}})

    def get_seawater_viscosity(self):
        return 0.001*(1.7915 - 0.0538*self.environment.sea_water_temperature+ 0.007*(self.environment.sea_water_temperature**(2.0)) - 0.0023*self.environment.sea_water_salinity)

    def update_terminal_velocity(self, Tprofiles=None, Sprofiles=None, z_index=None):
        """Calculate terminal velocity for the plastic particles - taken from Pelagic Egg and adapted

        according to
        S. Sundby (1983): A one-dimensional model for the vertical
        distribution of pelagic fish eggs in the mixed layer
        Deep Sea Research (30) pp. 645-661

        Method copied from ibm.f90 module of LADIM:
        Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
        Fish. Oceanogr. (16) pp. 216-228
        """
        g = 9.81  # ms-2

        # Get the properies which determine buoyancy
        partsize = self.elements.total_diameter  

        DENSw = self.sea_water_density(T=self.environment.sea_water_temperature, S=self.environment.sea_water_salinity)
        DENSpart = self.elements.total_density
        dr = DENSw-DENSpart  # density difference

        # water viscosity
        my_w = self.get_seawater_viscosity()

        # terminal velocity for low Reynolds numbers
        W = (1.0/my_w)*(1.0/18.0)*g*partsize**2 * dr

        # check if we are in a Reynolds regime where Re > 0.5
        highRe = np.where(W*1000*partsize/my_w > 0.5)

        # Use empirical equations for terminal velocity in
        # high Reynolds numbers.
        # Empirical equations have length units in cm!
        my_w = 0.01854 * np.exp(-0.02783 * self.environment.sea_water_temperature)  # in cm2/s
        d0 = (partsize * 100) - 0.4 * \
            (9.0 * my_w**2 / (100 * g) * DENSw / dr)**(1.0 / 3.0)  # cm
        W2 = 19.0*d0*(0.001*dr)**(2.0/3.0)*(my_w*0.001*DENSw)**(-1.0/3.0)
        # cm/s
        W2 = W2/100.  # back to m/s

        W[highRe] = W2[highRe]
        self.elements.terminal_velocity = W

    def update_biofilm(self):
        # This applies the 4 terms in eqn 11 of Kooi 2017

        ### Collision of algae with particle
        # Encounter rate is sum of brownian motion, differential settling, and shear
        r_a = self.get_config('biofilm:algal_cell_radius')
        algae_conc_cells = self.environment.mass_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water / self.get_config('biofilm:algal_carbon_per_cell')
        particle_surface_area = np.pi * self.elements.total_diameter**2

        d_brown = 4*np.pi*(self.get_diffusivity(self.elements.total_diameter/2) + self.get_diffusivity(r_a))*((self.elements.total_diameter/2) + r_a)

        d_settle = 0.5*np.pi*((self.elements.total_diameter/2)**2)*-self.elements.terminal_velocity

        d_shear = 1.3 * self.get_config('biofilm:shear') * ((self.elements.total_diameter/2) + r_a)**3

        d_tot = d_brown + d_settle + d_shear

        collision_term = (d_tot*algae_conc_cells)/particle_surface_area

        ### Growth on particle 
        # light and temp limited, needs light model and growth curve
        #growth = optimal_growth(light, temperature) - possible light model in sealice model

        growth = self.get_light_growth() * self.get_temp_influence()

        ### Grazing on particle
        grazing = self.get_config('biofilm:grazing_rate')

        ### Respiration
        # This is temperature dependent
        respiration = self.get_config('biofilm:temperature_coeff_respiration')**((self.environment.sea_water_temperature - 20)/10) * self.get_config('biofilm:respiration_rate')

        # Add up and convert to thickness
        seconds_per_day = 60*60*24 # To convert from daily respiration/grazing to per second
        A = self.elements.biofilm_no_attached_algae
        newAttached = A + (collision_term + growth*A - (grazing/seconds_per_day)*A - (respiration/seconds_per_day)*A)* self.time_step.total_seconds()

        self.elements.biofilm_no_attached_algae = newAttached

    def get_temp_influence(self):
        # Equation 18 from Kooi et al 2017
        T_z = self.environment.sea_water_temperature
        T_max = self.get_config('biofilm:growth_temperature_max')
        T_min = self.get_config('biofilm:growth_temperature_min')
        T_opt = self.get_config('biofilm:growth_temperature_optimal')

        numerator = (T_z - T_max)*(T_z - T_min)**2
        denominator = ((T_opt - T_min)*((T_opt - T_min)*(T_z - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2*T_z)))

        return numerator/denominator

    def get_light_growth(self):
        u_max = self.get_config('biofilm:maximum_growth_rate')
        alpha = self.get_config('biofilm:initial_slope')
        I_opt = self.get_config('biofilm:growth_light_optimal')

        # Instead of eqn 20 from Kooi et al, use light model from sealice setup
        I_z = self.irradiance()
    
        # Eqn 17 from Kooi et al 2017
        denominator = I_z + (u_max/alpha)*((I_z/I_opt) - 1)**2
        u_opt = u_max * (I_z / denominator)
        
        return u_opt

    def irradiance(self):
        """
        From SeaLice model

        Distribute the daily energy from irradiance with a gaussian distribution.
        We use the twilight times for high sensitivity organisms
        Convert irradiance from W.m-2 to micromol photon.s-1.m-2
        https://www.berthold.com/en/bioanalytic/knowledge/faq/irradiance-to-photon-flux
        calculate the photon flux in the water according to exponential decay from the sea surface
        """
        tau= np.pi+2*np.deg2rad(self.get_config('light:twilight')) #width of solar irridation distribution
        nu = self.get_config('light:nu') # Wavelength used to calculalate irradiance
        k_water = self.get_config('light:k_water') # Extinction coefficient water 
        k_chl = self.get_config('light:k_chl') # Extinction coefficient algae
 
        k_tot = k_water + k_chl * self.environment.mass_concentration_of_phytoplankton_expressed_as_chl_in_sea_water
        solar_angle = np.deg2rad(hour_angle(self.time,np.mean(self.elements.lon)))

        solar_coeff= np.sqrt(tau/(2*np.pi)*np.exp(-tau/2*solar_angle**2))

        light = solar_coeff*self.environment.surface_net_downward_radiative_flux* \
                        nu*0.00836*np.exp(k_tot*self.elements.z)
        return light

    def get_diffusivity(self, radius):
        boltzmann = 1.0306*10**-13 
        seawater_visc = self.get_seawater_viscosity()
        
        return boltzmann*(self.environment.sea_water_temperature) / (6*np.pi*seawater_visc*radius)

    def update_density(self):
        biofilm_volume = self.elements.biofilm_no_attached_algae*self.get_config('biofilm:algal_cell_volume')
        original_volume = (1/6)*np.pi*(self.elements.unfouled_diameter**3)

        new_volume = biofilm_volume + original_volume

        original_mass = original_volume * self.elements.unfouled_density

        self.elements.total_diameter = (6*new_volume/np.pi)**(1/3)
        self.elements.total_density = (biofilm_volume*self.get_config('biofilm:biofilm_density')+ original_mass) / new_volume

    def update(self):
        """Update positions and properties of plastic particles."""
        # Biofilm
        self.update_biofilm()
        self.update_density()

        # Turbulent Mixing
        self.update_terminal_velocity()
        self.vertical_mixing()

        # Horizontal advection
        self.advect_ocean_current()

        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()
