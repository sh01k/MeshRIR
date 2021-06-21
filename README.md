# MeshRIR

## Description
MeshRIR is a dataset of acoustic room impulse responses (RIRs) at finely meshed grid points. Two subdatasets are currently available: one consists of IRs in a 3D cuboidal region from a single source, and the other consists of IRs in a 2D square region from an array of 32 sources. This dataset is suitable for evaluating sound field analysis and synthesis methods.

See the link below for the details.
- https://sh01k.github.io/MeshRIR/

## Download
The dataset is available in the following link:
- https://doi.org/10.5281/zenodo.5002818

Extract the data files in the folder "src". For processing the IR data, see "ir_view.html" (converted from .ipynb) for Python or "ir_view.m" for Matlab. Example codes for sound field analysis and synthesis are included in the folder "example". Latest codes are available here: 
- https://github.com/sh01k/MeshRIR

If you use the MeshRIR dataset for your research, cite the following paper:
```
@inproceedings{MeshRIR,
  author    = "Shoichi Koyama and Tomoya Nishida and Keisuke Kimura and
               Takumi Abe and Natsuki Ueno and Jesper Brunnstr\"{o}m",
  title     = "{MeshRIR}: A Dataset of Room Impulse Responses on Meshed Grid Points For Evaluating Sound Field Analysis and Synthesis Methods",
  booktitle = "arXiv",
  year      = "2021"
}
```

## License
The MeshRIR dataset is provided under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://github.com/sh01k/MeshRIR/blob/main/LICENSE).

## Author
- [Shoichi Koyama](https://www.sh01.org) (The University of Tokyo, Tokyo, Japan)
- Tomoya Nishida
- Keisuke Kimura
- Takumi Abe
- [Natsuki Ueno](https://natsuenono.github.io/)
- Jesper Brunnström
