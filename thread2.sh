OUTPUT="./Data/grid_search/third_try" 

#python adaptive_gridsearch.py -e cartpole -a lambda -c fitness -o $OUTPUT -r bash_run -H 3
#python adaptive_gridsearch.py -e cartpole -a diff -c fitness -o $OUTPUT -r bash_run -H 3
#python adaptive_gridsearch.py -e lunarlander -a lambda -c novelty -o $OUTPUT -r bash_run -H 3
#python adaptive_gridsearch.py -e lunarlander -a diff -c novelty -o $OUTPUT -r bash_run -H 3
#python adaptive_gridsearch.py -e cartpole -a lambda -c add_novelty -o $OUTPUT -r bash_run -H 3
#python adaptive_gridsearch.py -e cartpole -a diff -c add_novelty -o $OUTPUT -r bash_run -H 3


python adaptive_gridsearch.py -e cartpole -a lambda -c fit_archiving -o $OUTPUT -r bash_run -H 3 &
python adaptive_gridsearch.py -e cartpole -a diff -c fit_archiving -o $OUTPUT -r bash_run -H 3 &
wait
python adaptive_gridsearch.py -e cartpole -a lambda -c novelty_archiving -o $OUTPUT -r bash_run -H 3 &
python adaptive_gridsearch.py -e cartpole -a diff -c novelty_archiving -o $OUTPUT -r bash_run -H 3 &
wait
