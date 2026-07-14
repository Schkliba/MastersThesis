OUTPUT="server_diff" 

declare -a arr=( "add_novelty" "novelty" "fitness" "sub_novelty" "fit_archiving" "novelty_limit" "novelty_archiving" "elite_archiving" )

## loop through above array (quotes are important if your elements may contain spaces)
for i in "${arr[@]}"
do
    python generation_examination.py -e lunarlander -a diff -c $i -o $OUTPUT  -R grid_search &
    pids+=($!)

    count=$((count + 1))

    if (( count % 2 == 0 )); then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done