### Getting Started with [BERT]

docker build -t bert-ja-1.0 .

# Run in Env # Ubuntu
docker run -it --rm -v $PWD:/work -p 5000:5000 bert-ja-1.0

# Run in Window
docker run -it --rm -v %cd%:/work -p 5000:5000 bert-ja-1.0


pip install -r requirements.txt

# Run pretrain
python preprocess.py

# Run Train
python train.py

# Run app
python app.py
