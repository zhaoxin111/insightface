export OMP_NUM_THREADS=4

starttime=`date +'%Y-%m-%d %H:%M:%S'`


# /usr/bin/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 partial_fc.py --world_size=8 | tee hist.log


endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "运行时间:"$((end_seconds-start_seconds))"s"

ps -ef | grep "world_size" | grep -v grep | awk '{print "kill -9 "$2}' | sh
