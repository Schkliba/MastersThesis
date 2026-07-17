OUTPUT="server_try0" 

declare -a arr=("add_novelty" "sub_novelty" "novelty" "novelty_limit", "novelty_archiving")

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e cartpole -a lambda -c $i -o $OUTPUT -R grid_search -t add, -l 30 -u 60 -p 10 &
    python generation_examination.py -e cartpole -a diff -c $i -o $OUTPUT -R grid_search -t add, -l 30 -u 60 -p 10 &
    wait
done