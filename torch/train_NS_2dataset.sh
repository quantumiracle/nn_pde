echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
# export CUDA_VISIBLE_DEVICES=0

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train.py --id $DATE > log/$DATE$RAND.log &

# declare -a data=('500' '1000' '1500' '2000' '5000')
declare -a data=('500')
declare -a dim=('32' '64' '128' '256' '512')

for i in ${data[@]}; do
    for j in ${dim[@]}; do
        echo $i $j
        if [ "$j" -gt  256 ]; 
        then
        CUDA_VISIBLE_DEVICES=0 nohup python train_NS_2dataset.py --id $DATE --data $i --dim $j > log/$DATE$i$j.log &
        else 
        CUDA_VISIBLE_DEVICES=1 nohup python train_NS_2dataset.py --id $DATE --data $i --dim $j > log/$DATE$i$j.log &
        fi
    done
done 