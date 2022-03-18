# post-process images

cd workspace
for D in `find . -mindepth 1 -type d`
do 
    cd $D
    tail -1 output.txt > tail.txt
    filename=$(python3 ../../printname.py)
    cp optimized_design.png ../../figures/$filename 
    cd ..
done
cd ..
