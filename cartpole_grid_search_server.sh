OUTPUT="./Data/grid_search/server_try0" 

declare -a arr=("fit_archiving" "elite_archiving" "novelty_archiving" "novelty_limit")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
   python adaptive_gridsearch.py -e cartpole -a lambda -c $i -o $OUTPUT -r bash_run -H 3 &
   python adaptive_gridsearch.py -e cartpole -a diff -c $i -o $OUTPUT -r bash_run -H 3 &
done
wait