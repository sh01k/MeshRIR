# About MeshRIR
MeshRIR is a dataset of acoustic room impulse responses (RIRs) at finely meshed grid points. Two subdatasets are currently available: one consists of IRs in a 3D cuboidal region from a single source, and the other consists of IRs in a 2D square region from an array of 32 sources. This dataset is suitable for evaluating sound field analysis and synthesis methods.

<div style="text-align:center">
<img src="./img/wave.gif" alt="wave" width="500">
</div>

# Download 
The dataset is available in the following link:
- [https://doi.org/10.5281/zenodo.5500451](https://doi.org/10.5281/zenodo.5500451)

Extract the data files in the folder "src". For processing the IR data, see "ir_view.html" (converted from .ipynb) for Python or "ir_view.m" for Matlab. Example codes for sound field analysis and synthesis are included in the folder "example". Latest codes are available here: 
- [https://github.com/sh01k/MeshRIR](https://github.com/sh01k/MeshRIR)

You can also try on Google Colab: 
- <a href="https://colab.research.google.com/github/sh01k/MeshRIR/blob/main/ir_view_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you use the MeshRIR dataset for your research, please cite the following paper ([preprint](https://arxiv.org/abs/2106.10801)):
~~~
@inproceedings{MeshRIR,
  author    = "Shoichi Koyama and Tomoya Nishida and Keisuke Kimura and 
               Takumi Abe and Natsuki Ueno and Jesper Brunnstr\"{o}m",
  title     = "{MeshRIR}: A Dataset of Room Impulse Responses on Meshed Grid Points For Evaluating Sound Field Analysis and Synthesis Methods",
  booktitle = "Proc. {IEEE} Int. Workshop Appl. Signal Process. Audio Acoust. (WASPAA)",
  year      = "2021"
}
~~~

# License
The MeshRIR dataset is provided under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://github.com/sh01k/MeshRIR/blob/main/LICENSE). 

# Detailed description
MeshRIR dataset consists of two subdatasets. 
#### S1-M3969
- IRs inside a 3D cuboidal region from a single source position

#### S32-M441
- IRs inside a 2D square region from 32 source positions

<figure id="position" style="text-align:center">
<img src="./img/pos_S1-M3969.png" alt="pos" width="280"> <img src="./img/pos_S32-M441.png" alt="pos" width="280">
<figcaption>Source and microphone positions of <strong>S1-M3969</strong> (left) and <strong>S32-M441</strong> (right)</figcaption>
</figure>

#### File format
The file formats are `.npy` for Python (Numpy) and `.mat` for MATLAB. All the additional data are provided as a JSON file. 

#### Measurement conditions
The IR at each position was measured using a Cartesian robot with an omnidirectional microphone (Primo, EM272J). The signal input and output were controlled by a PC with a Dante interface. The loudspeaker was DIATONE, DS-7 for <strong>S1-M3989</strong> and YAMAHA, VXS1MLB for <strong>S32-M441</strong>.

<table width="100%">
    <thead>
    <tr>
    <th width="30%"></th>
    <th width="35%" style="text-align:center">S1-M3969</th>
    <th width="35%" style="text-align:center">S32-M441</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Sampling rate</td>
    <td style="text-align:center" colspan="2">48000 Hz</td>
    </tr>
    <tr>
    <td>IR length</td>
    <td style="text-align:center" colspan="2">32768 samples</td>
    </tr>
    <tr>
    <td>Room dimensions</td>
    <td style="text-align:center" colspan="2">7.0 m × 6.4 m × 2.7 m</td>
    </tr>
    <tr>
    <td>Number of sources</td>
    <td style="text-align:center">1</td>
    <td style="text-align:center">32</td>
    </tr>
    <tr>
    <td>Measurement region</td>
    <td style="text-align:center">1.0 m × 1.0 m × 0.4 m</td>
    <td style="text-align:center">1.0 m × 1.0 m </td>
    </tr>
    <tr>
    <td>Intervals of mics</td>
    <td style="text-align:center" colspan="2">0.05 m</td>
    </tr>
    <tr>
    <td>Number of mics</td>
    <td style="text-align:center">21 × 21 × 9 points</td>
    <td style="text-align:center">21 × 21 points </td>
    </tr>
    <tr>
    <td>RT60</td>
    <td style="text-align:center">0.38 s</td>
    <td style="text-align:center">0.19 s</td>
    </tr>
    <tr>
    <td>Avg. temperature</td>
    <td style="text-align:center">26.3 °C</td>
    <td style="text-align:center">17.1 °C</td>
    </tr>
    </tbody>
</table>

<figure id="position" style="text-align:center">
<img src="./img/sf_measurement_spkarray.png" alt="pos" width="380"> 
<figcaption>IR measurement system for <strong>S32-M441</strong></figcaption>
</figure>

# Usage
### Basic usage examples (Python)
See [ir_view.ipynb](https://github.com/sh01k/MeshRIR/blob/main/ir_view.ipynb) for the details.


- Import [irutilities.py](https://github.com/sh01k/MeshRIR/blob/main/irutilities.py)
~~~python
import irutilities as irutil
~~~


- Load microphone and source positions, and IR data
~~~python
posMic, posSrc, ir = irutil.loadIR(path_to_data_folder)
~~~


- Plot IR of source `srcIdx` and mic `micIdx`
~~~python
irutil.irPlots(ir[srcIdx, micIdx, :], samplerate)
~~~

### Basic usage examples (MATLAB)
See [ir_view.m](https://github.com/sh01k/MeshRIR/blob/main/ir_view.m) for the details.

- Add folder "[matfiles](https://github.com/sh01k/MeshRIR/tree/main/matfiles)" to search path
~~~matlab
addpath('matfiles');
~~~

- Load microphone and source positions, and IR data
~~~matlab
[posMic, posSrc, ir] = loadIR(path_to_data_folder);
~~~

### Basic usage examples (Julia)

- Import [irutils.jl](https://github.com/sh01k/MeshRIR/blob/main/irutils.jl)
~~~julia
include("lib/utils.jl")
import .utils
~~~

- Load microphone and source positions, and IR data
~~~julia
posAll, posSrc, irAll = irutils.loadIR(path_to_data_folder)
~~~

### Application examples
- Sound field reconstruction: [examples/sf_reconst.ipynb](https://github.com/sh01k/MeshRIR/blob/main/example/sf_reconst.ipynb)
    - Estimation of pressure distribution using the method proposed in [2].
    - Microphone positions are selected by the MSE-based sensor placement method [3].

- Sound field control: [examples/sf_control.ipynb](https://github.com/sh01k/MeshRIR/blob/main/example/sf_control.ipynb)
    - Synthesis of planewave field by pressure matching.
    - Weighted pressure and mode matching [4,5] are also demonstrated. Details can be found in [6].

# References
1. S. Koyama, T. Nishida, K. Kimura, T. Abe, N. Ueno, and J. Brunnström, "MeshRIR: A dataset of room impulse responses on meshed grid points for evaluating sound field analysis and synthesis methods," in Proc. IEEE WASPAA, 2021. [[pdf]](https://arxiv.org/abs/2106.10801)
1. N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE SPL, 2018. [[pdf]](https://doi.org/10.1109/LSP.2017.2775242)
1. T. Nishida, N. Ueno, S. Koyama, and H. Saruwatari, “Region-restricted Sensor Placement Based on Gaussian Process for Sound Field Estimation,” IEEE Trans. SP, 2022. [[pdf]](https://doi.org/10.1109/TSP.2022.3156012)
1. S. Koyama and K. Arikawa, "Weighted pressure matching based on kernel interpolation for sound field reproduction," Proc. ICA, 2022. (to appear)
1. N. Ueno, S. Koyama, and H. Saruwatari, “Three-dimensional sound field reproduction based on weighted mode-matching method,” IEEE/ACM Trans. ASLP, 2019. [[pdf]](https://doi.org/10.1109/TASLP.2019.2934834)
1. S. Koyama, K. Kimura, and N. Ueno, "Weighted pressure and mode matching for sound field reproduction: Theoretical and experimental comparisons," J. AES, 2022. (in press)

# Author
- [Shoichi Koyama](https://www.sh01.org) (The University of Tokyo, Tokyo, Japan)
- Tomoya Nishida
- Keisuke Kimura
- Takumi Abe
- [Natsuki Ueno](https://natsuenono.github.io/)
- Jesper Brunnström
 
