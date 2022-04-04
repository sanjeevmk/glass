root_dir=$1
size=$2
matcap=$3
cd $root_dir
mkdir render/
for f in *.obj; 
do
	pref=`echo $f | cut -d'.' -f1`
	/home/samk/Thea/Thea/Code/Build/Output/bin/RenderShape -u y -v 00- -z 1 -k $matcap -a 4 $f "./render/"$pref".jpg" $size $size
done
