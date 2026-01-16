"""
Visualization module for Bayesian Glacier Flow
"""

from .field_plots import (
    plot_velocity_field_2d,
    plot_fem_bnn_comparison,
    plot_vertical_profile
)

__all__ = [
    'plot_velocity_field_2d',
    'plot_fem_bnn_comparison',
    'plot_vertical_profile'
]
