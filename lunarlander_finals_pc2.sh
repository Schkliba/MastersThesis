OUTPUT="server_try2" 

#declare -a arr=("add_novelty" "sub_novelty" "novelty_limit" "novelty_archiving", "novelty")
declare -a arr=("fitness" "fit_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python final_examination.py -e lunarlander -a lambda -c $i -o $OUTPUT -R list &
    python final_examination.py -e lunarlander -a diff -c $i -o $OUTPUT  -R list &
    wait
done