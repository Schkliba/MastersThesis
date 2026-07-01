OUTPUT="server_try3" 

declare -a arr=("sub_novelty")
#declare -a arr=("fitness" "fit_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python final_examination.py -e lunarlander -s 45 -a lambda -c $i -o $OUTPUT -R list &
    python final_examination.py -e lunarlander -s 45 -a diff -c $i -o $OUTPUT  -R list &
    wait
done