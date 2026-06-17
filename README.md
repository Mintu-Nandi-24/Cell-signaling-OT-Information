# Decoding cell signaling via optimal transport and information theory

This repository contains the source code and numerical source data associated with the manuscript:

**Decoding cell signaling via optimal transport and information theory**

The repository is organized to support reproducibility of the theoretical motif analysis, TNF signaling analysis, RAS-MAPK/SOS-RAF analysis, and the non-Gaussian illustrative example.

## Repository structure

```text
Source_Code/
Source_Data/
README.md
```

## Source code

The `Source_Code/` folder contains the Mathematica notebooks and Python scripts used to generate the theoretical and experimental analysis results.

Please see the README files inside `source_code/` for details on:

* theoretical motif analysis;
* TNF signaling analysis;
* RAS-MAPK/SOS-RAF analysis;
* non-Gaussian illustrative analysis;
* software requirements and expected inputs/outputs.

## Source data

The `Source_Data/` folder contains the numerical input data and processed output data underlying the main and supplementary figures.

Please see the README file inside `source_data/` and the accompanying `Source_Data_Index.xlsx` file for details on:

* figure-level source-data mapping;
* raw input-data descriptions;
* processed output-data descriptions;
* file manifest;
* column descriptions;
* external data sources and citations.

## External experimental data sources

* The TNF signaling input data were extracted from supplementary figures of:

  Cheong R, Rhee A, Wang CJ, Nemenman I, Levchenko A. Information transduction capacity of noisy biochemical signaling networks. Science. 2011;334:354–358. doi:10.1126/science.1204553.

* The SOS/RAF raw input data used for the RAS-MAPK reanalysis should be downloaded from the original repository associated with:

  Umeki N, Kabashima Y, Sako Y. Evaluation of information flows in the RAS-MAPK system using transfer entropy measurements. eLife. 2025;14:e104432. doi:10.7554/eLife.104432.

  Original repository:

  ```text
  https://github.com/YasushiSako/transfer_entropy_2/
  ```

* Details of the specific files used are provided in the README inside the `source_data/` folder.

## Citation

If you use the code or processed data in this repository, please cite the associated manuscript and the original experimental studies listed above.

## Author

Mintu Nandi,
Universal Biology Institute, The University of Tokyo
