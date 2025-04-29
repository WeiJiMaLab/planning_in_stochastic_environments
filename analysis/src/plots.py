import matplotlib.pyplot as plt
import matplotlib
import os, sys, inspect
font_path = os.path.join(os.getcwd(), 'fonts/Helvetica Neue.ttf')
matplotlib.font_manager.fontManager.addfont(font_path)
prop = matplotlib.font_manager.FontProperties(fname=font_path)
import matplotlib.colors as colors
from skimage.color import rgb2lab, lab2rgb
import numpy as np

plt.style.use('default')
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': prop.get_name(),
    'font.size': 20
})
plt.rcParams['svg.fonttype'] = 'none'

colors = ['#D76F6E', '#9FD5D9', '#5498AE','#1E5A80', '#002E59', '#0B003D']


def make_gradient(hex_codes, steps = 10):
    # Convert hex codes to RGB values
    rgb_colors = [colors.to_rgb(hex) for hex in hex_codes]

    # Convert RGB to Lab colorspace (perceptually uniform)
    lab_colors = [rgb2lab(color) for color in rgb_colors]
    
    intermediate_colors = []
    for c1, c2 in zip(lab_colors[:-1], lab_colors[1:]):
        interp = np.linspace(0, 1, 100, endpoint = False)
        intermediate_colors += [c1 * (1 - i) + c2 * i for i in interp]

    # Convert intermediate colors back to RGB
    rgb_gradient = [lab2rgb(color) for color in intermediate_colors]

    # Create a colormap from the generated colors
    cmap = colors.LinearSegmentedColormap.from_list("", rgb_gradient)

    # Generate positions evenly spaced between 0 and 1
    positions = np.linspace(0, 1, steps)

    # Generate the gradient using the colormap
    return cmap(positions)