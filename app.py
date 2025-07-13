from flask import Flask
from routes.predict import predict_bp
from routes.upload import upload_bp
from routes.home import home_bp
from config import load_environment

load_environment()  # Load .env configs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'user_data'

# Register blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(home_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
