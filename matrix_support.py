# -*- coding: utf-8 -*- 
from typing import Dict, Tuple 

import numpy as np 
import scipy.interpolate 

from config import COORDINATE_MAPPING 


def build_electrode_grid( 
    n_timesteps: int, 
    channel_values: Dict[str, np.ndarray], 
    coordinate_mapping: Dict[str, Tuple[int, int]] = COORDINATE_MAPPING, 
    grid_shape: Tuple[int, int] = (22, 43), 
    crop_rows: slice = slice(5, None), 
    crop_cols: slice = slice(1, 42), 
) -> np.ndarray: 
    """ 
    Build a (n_timesteps, rows, cols) interpolated grid from 
    sparse electrode readings 
    """ 
    matrix = np.full((n_timesteps, *grid_shape), np.nan) 

    for label, (row, col) in coordinate_mapping.items(): 
        if label not in channel_values: 
            continue 
        matrix[:, row, col] = channel_values[label] 

    x = np.arange(grid_shape[1]) 
    y = np.arange(grid_shape[0]) 
    xx, yy = np.meshgrid(x, y) 

    for t in range(n_timesteps): 
        frame = np.ma.masked_invalid(matrix[t]) 
        known_x = xx[~frame.mask] 
        known_y = yy[~frame.mask] 
        known_values = matrix[t][~frame.mask] 

        matrix[t] = scipy.interpolate.griddata( 
            (known_x, known_y), known_values.ravel(), (xx, yy), 
            fill_value=0, method="cubic", 
        ) 

    return matrix[:, crop_rows, crop_cols] 


if __name__ == "__main__": 
    # Test run 
    demo_timesteps = 3 
    demo_values = { 
        label: np.full(demo_timesteps, i) 
        for i, label in enumerate(COORDINATE_MAPPING) 
    } 
    grid = build_electrode_grid(demo_timesteps, demo_values) 
    print(f"Output shape: {grid.shape}  (expected: ({demo_timesteps}, 17, 41))") 
