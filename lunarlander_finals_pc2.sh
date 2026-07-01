OUTPUT="server_try3" 

declare -a arr=("add_novelty" "novelty_limit" "novelty")#
#declare -a arr=("fitness" "fit_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python final_examination.py -e lunarlander -a lambda -s 45 -c $i -o $OUTPUT -R list &
    python final_examination.py -e lunarlander -a diff -s 45 -c $i -o $OUTPUT  -R list &
    wait
done