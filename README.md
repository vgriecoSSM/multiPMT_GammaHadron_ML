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
    - run -> Features_Parquet_Builder
    - run -> Model_Train
    - run -> Apply_Model
    - run -> (optional) Model_Performance_Single_Station
    - run -> Gamma_Hadron_Discrimination_Performance

Newly created features parquet files are NOT tracked as well as output files of any kind.
Newly created traces parquet are tracked.
Remove unzipped dfs_traces/HAWCSIM_array copy before pushing or zip an updated copy.
