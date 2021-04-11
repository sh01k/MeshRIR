# About MeshRIR
MeshRIR is a dataset of acoustic room impulse responses (RIRs) on finely meshed grid points. Two sub-datasets are currently available: one is IRs in 3D cuboid region from a single source, and the other is IRs in 2D square region from array of 32 sources. This dataset is suitable for evaluating sound field analysis and synthesis methods.

<img src="./img/wave.gif" alt="wave" width="400">

# Download 
TBA

# Detailed description
The MeshRIR dataset consists of two sub-datasets. 
- S1-M3969: IRs inside 3D cuvoid region from single source position
- S32-M441: IRs inside 2D square region from 32 source positions
Detailed measurement conditions are as follows.


|  | S1-M3969 | S32-M441 |
| :---- | :----: | :----: |
| Sampling rate | 48000 Hz ||
| IR length | 32768 samples ||
| Room dimensions | 7.0 m x 6.4 m x 2.7 m ||
| Number of source positions | 1 | 32 |
| Measurement region | 3D cuboid: 1.0 m x 1.0 m x 0.4 m | 2D square: 1.0 m x 1.0 m |
| Intervals of microphone positions | 0.05 m ||
| Number of microphone positions | 21 x 21 x 9 (=3969) points | 21 x 21 (=441) points |
| Reverberation time (T60) |  0.38 s | 0.19 s |
| Average temperature | 26.3 °C  | 17.1 °C |


# Usage
## Basic usage examples
See [ir_view.ipynb](https://github.com/sh01k/MeshRIR/blob/main/ir_view.ipynb) for the details.
- Import irutilities
    import irutilities as irutil
- Load microphone and source positions, and IR data
    posMic, posSrc, ir = irutil.loadIR(path_to_data_folder)

## Application examples
- Sound field reconstruction: [examples/sf_reconst.ipynb](https://github.com/sh01k/MeshRIR/blob/main/example/sf_reconst.ipynb)
- Sound field control: [examples/sf_control.ipynb](https://github.com/sh01k/MeshRIR/blob/main/example/sf_control.ipynb)

# References
TBA

# Author
- [Shoichi Koyama](https://www.sh01.org) (The University of Tokyo, Tokyo, Japan)
- Tomoya Nishida
- Keisuke Kimura
- Takumi Abe
- Natsuki Ueno
- Jesper Brunnström
 
