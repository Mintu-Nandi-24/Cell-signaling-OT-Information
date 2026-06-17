### TNF input data

The TNF signaling input data used in this repository were extracted from the supplementary figures of the following study:

Cheong R, Rhee A, Wang CJ, Nemenman I, Levchenko A. Information transduction capacity of noisy biochemical signaling networks. Science. 2011;334:354–358. doi:10.1126/science.1204553.

The present analysis uses TNF dose-dependent NF-κB and ATF-2 response statistics for WT and A20-deficient cells at 30 min and 4 hr after TNF stimulation. These values were extracted from the figures provided in Cheong et al.

Specifically, the mutual information values used as target values were extracted from Supplementary Fig. 3B of Cheong et al. For the present analysis, only the single-pathway information values were used, corresponding to the first two response cases: NF-κB only and ATF-2 only.

The dose-dependent mean and standard deviation values were extracted from Supplementary Fig. 10 of Cheong et al. These statistics were used to construct Gaussian conditional response distributions for each TNF dose, cell type, response pathway, and time point.

The TNF input files are organized as follows.

#### 30 min after TNF stimulation

```text
X-meanYcX-stdYcX-WT-30min-nfkb.txt
X-meanYcX-stdYcX-WT-30min-atf-2.txt
X-meanYcX-stdYcX-A20-30min-nfkb.txt
X-meanYcX-stdYcX-A20-30min-atf-2.txt
```

#### 4 hr after TNF stimulation

```text
X-meanYcX-stdYcX-WT-4hr-nfkb.txt
X-meanYcX-stdYcX-WT-4hr-atf-2.txt
X-meanYcX-stdYcX-A20-4hr-nfkb.txt
X-meanYcX-stdYcX-A20-4hr-atf-2.txt
```

#### Mutual information target values

```text
MI-nfkb.txt
MI-atf-2.txt
```

Each `X-meanYcX-stdYcX-*.txt` file contains three columns:

```text
TNF_dose_ng_per_mL    conditional_mean_response_a.u.    conditional_standard_deviation_a.u.
```

The `MI-nfkb.txt` file contains the extracted TNF-to-NF-κB mutual information values for WT and A20-deficient cells at 30 min and 4 hr. The `MI-atf-2.txt` file contains the corresponding TNF-to-ATF-2 mutual information values.

These extracted numerical values are used here only as input data for reproducing the present reanalysis. Users should cite the original Cheong et al. study when using these TNF response data. Small deviations may arise from digitization of the published supplementary figures.
