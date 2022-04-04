root_dir=$1
size=$2
matcap=$3
cd $root_dir
count=0
for subd in */; do
	count=$((count+1))
done
index=0
for subd in */; do
	index=$((index+1))
	echo $index"/"$count
	cd $subd
	for f in *.obj; do
		mkdir render/
		pref=`echo $f | cut -d'.' -f1`
		/home/samk/Thea/Thea/Code/Build/Output/bin/RenderShape -u y -v 00- -z 1 -k $matcap $f "./render/"$pref".jpg" $size $size
	done
	cd ..
done
