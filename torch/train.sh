echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=0

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train.py --id $DATE > log/$DATE$RAND.log &

declare -a data=('5000' '10000' '20000' '30000' '40000' '50000')

for i in ${data[@]}; do
    echo $i
    nohup python train.py --id $DATE --data $i > log/$DATE$i.log &
done 
