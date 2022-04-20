echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=1

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train.py --id $DATE > log/$DATE$RAND.log &

declare -a data=('500' '1000' '2000' '3000' '4000')

for i in ${data[@]}; do
    echo $i
    nohup python train.py --id $DATE --data $i > log/$DATE$i.log &
done 
