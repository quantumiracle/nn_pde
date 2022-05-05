echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=1

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train.py --id $DATE > log/$DATE$RAND.log &

# declare -a data=('100')
# declare -a dim=('32' '64' '128' '256' '512')

declare -a dim=('20')
declare -a data=('1100' '1300' '1500' '1700' '1900')

for i in ${data[@]}; do
    for j in ${dim[@]}; do
        echo $i $j
        nohup python train_heat.py --id $DATE --data $i --dim $j > log/$DATE$i$j.log &
    done
done 
