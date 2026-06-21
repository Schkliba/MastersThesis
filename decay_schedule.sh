python decay_examination.py -e cartpole -a diff -c add_novelty -o "repair_try9" -R grid_search & 
python decay_examination.py -e cartpole -a lambda -c add_novelty -o "repair_try9" -R grid_search & 
wait
python decay_examination.py -e cartpole -a lambda -c sub_novelty -o "repair_try9" -R grid_search &
python decay_examination.py -e cartpole -a diff -c sub_novelty -o "repair_try9" -R grid_search &
wait