import numpy as np
import json
import copy

MAXDEPTH = 8
MAXVAL = 10

#binary array is a full truth table -- i.e. all possible binary values in an array of size
#2 ** (MAXDEPTH - 1)
BINARR = np.array([list(np.binary_repr(x, MAXDEPTH)) for x in range(2 ** (MAXDEPTH - 1))]).astype(int)

def report_p_value(p): 
    if p < 1e-10: 
        return "p < 10^{-10}"
    else:
        # Convert to scientific notation with 2 sigfigs
        p_str = f"{p:.2e}"
        # Split into mantissa and exponent
        mantissa, exponent = p_str.split('e')
        # Remove leading zero from exponent
        exponent = str(int(exponent))
        return f"p = {mantissa} \\times 10^{{{exponent}}}"

def get_conditions(gametype):  
    if gametype == "T": 
        return [0, 0.125, 0.25, 0.375, 0.5]
    else: 
        return [0, 0.25, 0.5, 0.75, 1]

def get_data(gametype):
    datafile = "../data/raw/data_%s.json"%(gametype)
    with open(datafile, 'r') as f:
        data = json.load(f)
    return data

class mapdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def format_games(games): 
    '''
    removes the practice games from the list of games
    '''
    return [mapdict(g) for g in games if not "practice" in g["name"]]

def nan_add(a, b):
    return np.nansum(np.dstack((a,b)),2)

def copy_and_update(dict_orig, dict_update): 
    dict_copy = copy.deepcopy(dict_orig)
    dict_copy.update(dict_update)
    return dict_copy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
# Colormaps
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

def strsimplify(num):
    """
    Converts a number to a string with the minimum number of significant figures needed.
    """
    # Convert to string with many decimal places
    s = f"{num:.10f}"
    
    # Remove trailing zeros after decimal
    s = s.rstrip('0')
    
    # Remove decimal point if no decimals remain
    if s.endswith('.'):
        s = s[:-1]
        
    return s


import matplotlib.pyplot as plt
colormaps = {name: plt.cm.colors.LinearSegmentedColormap.from_list(name, colors) for name, colors in colormaps_.items()}
