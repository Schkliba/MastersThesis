OUTPUT="server_try2" 

#declare -a arr=("add_novelty" "sub_novelty" "novelty_limit" "novelty_archiving", "novelty")
declare -a arr=("fitness" "fit_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e lunarlander -a lambda -c $i -o $OUTPUT -R grid_search &
    python generation_examination.py -e lunarlander -a diff -c $i -o $OUTPUT  -R grid_search &
    wait
done