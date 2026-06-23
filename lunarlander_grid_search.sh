OUTPUT="./Data/grid_search/server_try1" 

declare -a arr=("add_novelty" "sub_novelty")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
   python adaptive_gridsearch.py -e lunarlander -a lambda -c $i -o $OUTPUT -r bash_run -H 3 &
   python adaptive_gridsearch.py -e lunarlander -a diff -c $i -o $OUTPUT -r bash_run -H 3 &
   wait
done

