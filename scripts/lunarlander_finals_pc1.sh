OUTPUT="server_try3" 

declare -a arr=( "fit_archiving" "novelty_archiving" "elite_archiving" )

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    echo $i 
    python final_examination.py -e lunarlander -a lambda -c $i -s 45 -o $OUTPUT -R list &
    python final_examination.py -e lunarlander -a diff -c $i -s 45 -o $OUTPUT  -R list &
    wait
done