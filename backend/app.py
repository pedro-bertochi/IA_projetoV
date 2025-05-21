from flask import Flask
from routes.api import api_bp
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'temp', 'uploads')
app.config['REPORT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'temp', 'relatorios')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)