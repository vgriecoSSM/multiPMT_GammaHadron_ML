# multiPMT_GammaHadron_ML
A repository for Gamma-Hadron Discrimination with muliPMTs

If it's the first time using the project, build the environment:

     - ./build-env.sh

Else :

     - source activate-env.sh

Not enough storage for unzipped directories -> Do :

     - cd dfs_traces/
     - unzip HAWCSIM_array

To run the pipeline:

    - cd gamma_hadron_discrimination
    
    - set path to root files in config.yaml
    - set desired parameters in config.yaml

    - run -> 1_Traces_Extraction.ipynb
    - run -> 2_Features_Parquet_Builder.ipynb
    - run -> 3_Model_Train.ipynb
    - run -> 4_Apply_Model.ipynb
    - run -> (optional) 5_Model_Performance_Single_Station.ipynb
    - run -> 6_Gamma_Hadron_Discrimination_Performance.ipynb

Newly created features parquet files are NOT tracked as well as output files of any kind.
Newly created traces parquet are tracked.
Remove unzipped dfs_traces/HAWCSIM_array copy before pushing or zip an updated copy.
