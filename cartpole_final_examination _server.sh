OUTPUT="server_try0" 

declare -a arr=("fit_archving" "novelty_limit" "novelty_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python final_examination.py  -e cartpole -c $i -a lambda -s 45 -o $OUTPUT -R list &
    python final_examination.py  -e cartpole -c $i -a diff -s 45 -o $OUTPUT -R list &
    wait
done