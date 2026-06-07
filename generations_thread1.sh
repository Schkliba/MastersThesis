python generation_examination.py -e cartpole -a lambda -c novelty -o "third_try" -R grid_search
python generation_examination.py -e cartpole -a diff -c novelty -o "third_try" -R grid_search
python generation_examination.py -e cartpole -a lambda -c fitness -o "third_try" -R grid_search
python generation_examination.py -e cartpole -a diff -c fitness -o "third_try" -R grid_search
python generation_examination.py -e cartpole -a lambda -c add_novelty -o "third_try" -R grid_search
python generation_examination.py -e cartpole -a diff -c add_novelty -o "third_try" -R grid_search