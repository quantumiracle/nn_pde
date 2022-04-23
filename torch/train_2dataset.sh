echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=0

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train.py --id $DATE > log/$DATE$RAND.log &

# declare -a data=('500' '1000' '1500' '2000' '5000')
declare -a data=('1000')

for i in ${data[@]}; do
    echo $i
    nohup python train_2dataset.py --id $DATE --data $i > log/2data_$DATE$i.log &
done 
