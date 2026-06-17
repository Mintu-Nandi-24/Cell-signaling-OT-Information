### SOS-RAF input data

The SOS and RAF single-cell translocation data used for the RAS-MAPK reanalysis were obtained from the publicly available repository associated with the original study:

Umeki N, Kabashima Y, Sako Y. Evaluation of information flows in the RAS-MAPK system using transfer entropy measurements. eLife. 2025;14:e104432. doi:10.7554/eLife.104432.

Original repository: https://github.com/YasushiSako/transfer_entropy_2/

According to the original repository README, the CSV data files contain time courses of protein translocation in single cells. The first column indicates time after EGF stimulation in minutes, and subsequent columns indicate signal intensities in individual cells. The SOS and RAF responses with the same cell index were obtained simultaneously in the same cell.

Users can obtain the raw input data from the original repository and cite the original study. 

For the present analysis, only the following four files were used:

```text
SOS_wt_DMSO_EGF10ng.csv
RAF_wt_DMSO_EGF10ng.csv
SOS_wt_10nMMEKi_EGF10ng.csv
RAF_wt_10nMMEKi_EGF10ng.csv
