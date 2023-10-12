if [ "$1" == "--help" ]; then
    echo "Usage:"
    echo "adbTest.sh {test} {params...}"
    exit 1
fi

device_dir=/data/local/tmp/
adb shell "cd ${device_dir} && LD_LIBRARY_PATH=. ./$1 $2 $3 $4 $5 $6 $7 $8 $9"