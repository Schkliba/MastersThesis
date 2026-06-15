python decay_examination.py -e lunarlander -a diff -c add_novelty -o "first_try" -R grid_search & 
python decay_examination.py -e lunarlander -a lambda -c sub_novelty -o "first_try" -R grid_search &
wait 
python decay_examination.py -e lunarlander -a diff -c sub_novelty -o "first_try" -R grid_search 