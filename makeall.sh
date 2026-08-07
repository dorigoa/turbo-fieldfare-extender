git clone git@github.com:dorigoa/turbo-fieldfare-fork.git
python3 ./apply_gemma_qat.py --repo-path ./turbo-fieldfare/
cd turbo-fieldfare
mkdir -p Scratch
cp ../build-app.sh Scratch/
source Scratch/build-app.sh --install
