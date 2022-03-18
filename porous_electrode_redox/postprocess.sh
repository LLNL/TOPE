# post-process images

cd workspace
for D in `find . -mindepth 1 -type d`
do 
    cd $D
    /Applications/ParaView-5.8.1.app/Contents/bin/pvpython ../../screenshot_design.py
    convert optimized_design.png -trim optimized_design.png
    cd ..
done
cd ..
