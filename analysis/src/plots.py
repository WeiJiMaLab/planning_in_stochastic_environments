import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.colors as colors

# Load Helvetica font files
font_regular_path = os.path.join(os.getcwd(), 'fonts/HelveticaNeue-Roman.otf')
matplotlib.font_manager.fontManager.addfont(font_regular_path)
helvetica_regular = matplotlib.font_manager.FontProperties(fname=font_regular_path)

font_bold_path = os.path.join(os.getcwd(), 'fonts/HelveticaNeue-Bold.otf')
matplotlib.font_manager.fontManager.addfont(font_bold_path)
helvetica_bold = matplotlib.font_manager.FontProperties(fname=font_bold_path)

def set_helvetica_style():
    """Set Matplotlib default style with Helvetica font."""
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.family': helvetica_regular.get_name(),
        'font.size': 20
    })
    plt.rcParams['svg.fonttype'] = 'none'

    # Return both regular and bold FontProperties for convenience
    return helvetica_regular, helvetica_bold

colors = ['#D76F6E', '#9FD5D9', '#5498AE','#1E5A80', '#002E59', '#0B003D']
set_helvetica_style()

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

colormaps_ = {
    "blurple": ['#0c3679', '#807ad1', '#d067b9'],
    "political": ['#244b89', '#c6c9cd', '#b33756'],
    "popsicle": ['#ff5e57', '#ffbd69', '#2ec4b6'],
    "lavender": ['#240372', '#ecd5db', '#005ca3'],
    "sunset": ['#240372', '#d3033b', '#db7b51'],
    "arctic": ['#080745', '#246c99', '#92d9d4'],
    "easter": ['#eea79b', '#eca7dd', '#80b3ea', '#3cc3b3'],
    'countyfair': ['#f59e9e', '#a894d6', '#164374', '#2c9da5'],
    "playdough":['#d52320', '#2a78c0', '#06a288', '#fbae41'],
    "foliage": ['#d53f3f', '#ffc894', '#076e62'],
    "rouge": ['#6b0037', '#b40439', '#f2a673'],
    "berry": ['#64006b', '#b40462', '#f27373'],
    "sage": ['#043d48', '#196c2b', '#92a592'],
    "grass": ['#00663f', '#43896b', '#b9c997'],
}

colormaps = {name: plt.cm.colors.LinearSegmentedColormap.from_list(name, colors) for name, colors in colormaps_.items()}