# Gaussian Beam Propagation Calculator (ABCD Matrix Method)

## 📖 Overview

An interactive Gaussian beam propagation simulator based on the **ABCD matrix method**, featuring a Streamlit web interface. This tool calculates beam propagation characteristics through multi-lens systems, supports independent X and Y direction analysis, and is ideal for laser optics system design and education.

## ✨ Key Features

### 1. Beam Parameter Configuration
- **Wavelength Settings**: Supports 200-2000nm range
- **Waist Parameters**: Independent X and Y direction waist diameter settings (precision to 0.00001mm)
- **Beam Quality Factor M²**: Precision to 3 decimal places, supports non-ideal Gaussian beams
- **Waist Position**: Freely set initial waist position at any location

### 2. Lens System
- **Multi-Lens Support**: Up to 10 lenses
- **Lens Types**: Automatic recognition of converging (convex) and diverging (concave) lenses
- **Free Configuration**: Independent position and focal length settings for each lens

### 3. Visualization Features
- **2D Beam Envelope Plot**:
  - Upper half displays Y direction beam evolution (red)
  - Lower half displays X direction beam evolution (blue)
  - Marks waist positions and lens positions
  - Automatic region numbering
  
- **Wavefront Curvature Evolution Plot**:
  - Toggle between X or Y direction display
  - Real-time display of radius of curvature changes
  - Marks lens positions

### 4. Data Analysis
- **Specific Position Query**: Input any position to view beam radius and radius of curvature
- **Regional Gaussian Beam Parameters**:
  - N lenses divide space into N+1 regions
  - Each region displays independent waist position, waist radius, and Rayleigh length
  - Region numbering: Region containing initial waist is Region 0, backward is negative, forward is positive

## 🔬 Calculation Principles

### q-Parameter Method
Gaussian beams are described by complex q-parameter:
```
q(z) = (z - z₀) + i·z_R
```
Where:
- z₀: Waist position
- z_R = π·w₀²·M²/λ: Rayleigh length
- w₀: Waist radius
- M²: Beam quality factor

### ABCD Matrix Transformation
Optical element effects on q-parameter:
```
q_out = (A·q_in + B) / (C·q_in + D)
```

**Free Space Propagation**:
```
M = [1  d]
    [0  1]
```

**Thin Lens**:
```
M = [1    0  ]
    [-1/f  1  ]
```

### Beam Parameter Extraction
Extract physical quantities from q-parameter:
```
1/q = 1/R(z) - i·λ/(π·w²(z))
```
- w(z): Beam radius
- R(z): Wavefront radius of curvature

## 🚀 Installation and Running

### Requirements
- Python 3.8 or higher

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GaussionBeamPropagation.git
cd GaussionBeamPropagation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run simple_beam_app.py
```

4. Open the displayed URL in your browser (typically http://localhost:8501)

## 📊 User Guide

### Basic Workflow

1. **Set Beam Parameters**
   - Input wavelength in the left sidebar
   - Set X and Y direction waist diameters and M² factors separately
   - Set initial waist position

2. **Configure Lens System**
   - Select number of lenses
   - Set position and focal length for each lens
   - Positive focal length = convex lens, negative = concave lens

3. **Set Propagation Distance**
   - Input maximum propagation distance (cm)

4. **View Results**
   - 2D envelope plot shows beam propagation pattern
   - Curvature plot shows wavefront curvature evolution
   - Tables display parameters at specific positions and in each region

### Application Examples

<img width="2882" height="1654" alt="2e2c7c791e985fa589d3f718d916cc52" src="https://github.com/user-attachments/assets/a93acdcc-1eee-4b24-8b4b-79a2a69b927c" />


## 🛠️ Technology Stack

- **Python**: Core programming language
- **NumPy**: Numerical computation
- **Streamlit**: Web interface framework
- **Plotly**: Interactive charts
- **Pandas**: Data processing and table display

## 📝 Advanced Features

### Backward Propagation Support
- Supports waist position between lenses
- Automatic calculation of beam propagation before and after waist
- Uses inverse matrices for backward lens transformations

### Region Division
- Region containing the waist is marked as Region 0
- Forward beam propagation regions: Region 1, 2, 3...
- Backward beam propagation regions: Region -1, -2, -3...

### X/Y Direction Decoupling
- Completely independent X and Y direction analysis
- Supports elliptical beams and astigmatic beams
- Separate display of evolution in different directions


## 🔗 Related Resources

- [Gaussian Beam Optics Fundamentals](https://en.wikipedia.org/wiki/Gaussian_beam)
- [ABCD Matrix Method](https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis)

## 📈 Changelog

### v1.0.0 (2026-02-06)
- ✅ Support for independent X/Y direction analysis
- ✅ Arbitrary waist positioning
- ✅ Multi-lens system
- ✅ Regional Gaussian beam parameter calculation
- ✅ Interactive visualization
- ✅ Specific position parameter query
