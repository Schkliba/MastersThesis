python adaptive_gridsearch.py -e lunarlander -a lambda -c fitness -o "./Data/grid_search/second_try" -r bash_run -H 3
python adaptive_gridsearch.py -e lunarlander -a diff -c fitness -o "./Data/grid_search/second_try" -r bash_run -H 3
python adaptive_gridsearch.py -e cartpole -a lambda -c novelty -o "./Data/grid_search/second_try" -r bash_run -H 3
python adaptive_gridsearch.py -e cartpole -a diff -c novelty -o "./Data/grid_search/second_try" -r bash_run -H 3
python adaptive_gridsearch.py -e lunarlander -a lambda -c add_novelty -o "./Data/grid_search/second_try" -r bash_run -H 3
python adaptive_gridsearch.py -e lunarlander -a diff -c add_novelty -o "./Data/grid_search/second_try" -r bash_run -H 3




