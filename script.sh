cd "$(dirname "$0")";
CWD="$(pwd)"
echo $CWD
python main.py "$1" "$2"

