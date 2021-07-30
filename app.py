
import camera
from flask import Flask,render_template,Response,redirect



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(camera.video_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start")
def start():
   
    return render_template('start.html')

@app.route("/stop")
def stop():
   
    return redirect("/")

if __name__=="__main__":
    app.run()