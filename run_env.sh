# project environment variables
PROJECT_DIR=$(pwd)

# setup path
export PATH="$PROJECT_DIR:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# run environment
pipenv install
pipenv shell
