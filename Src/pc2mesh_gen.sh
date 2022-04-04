#python3 pc2mesh.py final_configs/Correspondence/FaustFullTest.json test_scan_021.ply ../Data/faust/tr_reg_000.obj
srcdir=$1
for f in $srcdir'/'*.ply; do
	fid=`echo $f | rev | cut -d'.' -f2 | cut -d'_' -f1 | rev`
        if [ "$fid" -lt "042" ];then
                continue
        fi
	echo $f
	python3 pc2mesh.py final_configs/Correspondence/Faust100_Gen.json $f ../Data/faust/tr_reg_000.obj
done
