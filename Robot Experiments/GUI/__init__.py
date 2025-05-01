# from flask import Flask
# import socket
#
# from .events import socketio
# from .routes import main
#
# def create_app():
#     app = Flask(__name__)
#     app.config["DEBUG"] = False
#     app.config["SECRET_KEY"] = 'ceff418fb561ebf2572221b1f28789a36e4e30f7da4df0a8'
#
#     app.register_blueprint(main)
#
#     socketio.init_app(app)
#
#     ip = socket.gethostbyname(socket.gethostname())
#     print(f'serving GUI on http://{ip}:8000/')
#
#     return app