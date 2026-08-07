git clone https://github.com/drumih/turbo-fieldfare
python3 ./apply_gemma_qat.py --repo-path ./turbo-fieldfare/
cd turbo-fieldfare
mkdir -p Scratch
cp ../build-app.sh Scratch/
source Scratch/build-app.sh --install
