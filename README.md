# Planning in Stochastic Environments

This repository contains code for an experiment that studies human planning in stochastic environments. The project includes both the experimental interface and analysis tools.

## Project Structure

```
.
├── analysis/                   
│   ├── data/                   # Retrieve data from OSF
│   │   ├── raw/                # Raw experimental data
│   │   │   ├── data_R.json     # Reliability condition data
│   │   │   ├── data_V.json     # Volatility condition data
│   │   │   └── data_T.json     # Controllability condition data
│   │   └── final_fit/          # Processed model fits
│   ├── demos/                  # Example notebooks and demonstrations
│   │   ├── analysis.ipynb      # Demo of how to create figures
│   │   └── figures.py          # Code for generating publication figures
│   ├── scripts/                
│   │   └── batch_fit.py        # Script for batch fitting models to data
│   └── src/                    
│       ├── analysis.py         # Core analysis functionality
│       └── utils.py            
├── experiment/                 # Experiment interface
│   └── src/                    # Source code for the experiment
│       ├── modules/            # JavaScript modules
│       │   ├── experiment.js   # Main experiment interface
│       │   ├── game.js         # Game logic and mechanics
│       │   └── utils.js        
│       └── index.html          # Main experiment page
├── resources/                  # Additional resources
├── requirements.txt            # Python dependencies
└── README.md                 
```

## Demo
A demo of the experiment can be found at: 
https://pirate-treasure-9e3ba.web.app/?type=demoR (Reliability)
https://pirate-treasure-9e3ba.web.app/?type=demoV (Volatility)
https://pirate-treasure-9e3ba.web.app/?type=demoT (Controllability)

## Data
Behavioral Data and Fitted Model Data can be found at: 
https://osf.io/vyh8u/?view_only=54e1761ac51a44c7a6b64212f5a176a4

data_R, data_V, data_T are in json format and final_fit.zip 
can be unzipped and placed in its spot above.

Place them in their respective places in the directory (shown above)

### Key Directories and Files

#### Analysis
- `analysis/data/`: Contains raw experimental data and processed results
- `analysis/demos/`: Example notebooks showing how to use the analysis tools
- `analysis/scripts/`: Scripts for batch processing and model fitting
- `analysis/src/`: Core analysis code and utility functions

#### Experiment
- `experiment/src/`: Web-based experiment interface
- `experiment/src/modules/`: JavaScript modules for the experiment
- `experiment/src/index.html`: Main experiment page

#### Root
- `requirements.txt`: Python package dependencies
- `resources/`: Additional resources and documentation

## Key Components

### 1. Experiment Interface (`experiment.js`)

The experiment is implemented as a web-based interface that presents participants with a treasure hunt game. Key features:
- Three types of stochastic environments: Reliability (R), Volatility (V), and Controllability (T)
- Practice rounds and comprehension checks
- Full-screen mode and browser compatibility checks
- Data collection and submission to Prolific

### 2. Analysis Tools

#### `analysis.ipynb`
A Jupyter notebook demonstrating how to:
- Load and analyze experimental data
- Generate camera-ready figures
- Compare different models of decision-making
- Analyze reaction times and choice patterns

#### `batch_fit.py`
A script for batch processing model fits:
- Supports multiple model types (filter functions and value functions)
- Handles different stochastic environments (R, V, T)
- Saves results in organized directories with timestamps

#### `figures.py`
Code for generating camera-ready figures:
- Model comparison plots
- Stochasticity vs. reaction time analysis
- Choice pattern analysis
- Reward analysis

## Usage

### Running the Experiment

1. Set up the experiment environment:
```bash
cd experiment
npm install  # Install dependencies
```

2. Configure the experiment type in `experiment.js`:
- R: Reliability
- V: Volatility
- T: Controllability

3. Run experiment
```bash
python -m http.server
```

The experiment should be accessible at `localhost:3000/?type_=demoR` (for demo)

### Fitting the Model
```bash
cd analysis/scripts
python batch_fit.py --type R  # (R for reliability, V for volatility, T for controllability)
```

### Generating the Figures
```bash
cd analysis/demos
python figures.py
```

## Model Types
The analysis supports several model types:

### Filter Functions
- `filter_depth`: Depth-based filtering
- `filter_rank`: Rank-based filtering
- `filter_value`: Value-based filtering

### Value Functions
- `value_EV`: Expected value
- `value_path`: Path value
- `value_max`: Maximum value
- `value_sum`: Sum value

## Dependencies
- Python packages: see `requirements.txt`
- JavaScript: jsPsych
- Web browser compatibility requirements

## License
MIT License

Copyright (c) 2024 Jordan Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

