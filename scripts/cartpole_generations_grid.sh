OUTPUT="server_try0" 

declare -a arr=("add_novelty" "sub_novelty" "novelty" "fitness")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e cartpole -a lambda -c $i -o $OUTPUT -R grid_search &
    python generation_examination.py -e cartpole -a diff -c $i -o $OUTPUT  -R grid_search &
    wait
done