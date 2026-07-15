OUTPUT="server_try3" 

declare -a arr=("fitness" "novelty")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e lunarlander -a lambda -c $i -o $OUTPUT -R grid_search 
    python generation_examination.py -e lunarlander -a diff -c $i -o $OUTPUT  -R grid_search 
done