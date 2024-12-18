from flask import Flask
from flask_cors import CORS
from routes import teapot_routes, ocs_routes, file_routes, db_routes

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.register_blueprint(teapot_routes)
    app.register_blueprint(ocs_routes)
    app.register_blueprint(file_routes)
    app.register_blueprint(db_routes)
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5050)
