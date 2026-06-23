OUTPUT="server_try1" 

declare -a arr=("fit_archiving" "elite_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e lunarlander -a lambda -c $i -o $OUTPUT -R grid_search 
    python generation_examination.py -e lunarlander -a diff -c $i -o $OUTPUT  -R grid_search 
    wait
done