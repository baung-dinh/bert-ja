import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from preprocess import prepare_data
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            print('URL:' + filename)
            file_path = os.path.join(app.config['UPLOAD_PATH'], filename)
            f.save(file_path)

            prepare_data(file_path)

            with open(file_path) as file:
                return render_template("index.html", results=file)
            # parseCSV(file_path)

            # with open(os.path.join(app.config['UPLOAD_PATH'], filename)) as f:
            #     file_content = f.read()
            # return render_template('index.html', results=file_content)
        else:
            print('Invalid Uplaod only txt, csv.')

        return render_template('index.html', results='')
        # file_content = file_content.replace('\n', '<br>')
        # return file_content
        # return redirect(url_for('index'))

def parseCSV(filePath):
    # CVS Column Names
    col_names = ['first_name','last_name','address', 'street', 'state' , 'zip']
    # Use Pandas to parse the CSV file
    csvData = pd.read_csv(filePath,names=col_names, header=None)
    # Loop through the Rows
    for i,row in csvData.iterrows():
        print(i,row['first_name'],row['last_name'],row['address'],row['street'],row['state'],row['zip'])

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')