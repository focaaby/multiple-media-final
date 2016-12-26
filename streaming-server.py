import cv2
from PIL import Image
import threading
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
from datetime import datetime
from firebase import firebase

capture=None
id = 0
name = "Undefinded"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.createLBPHFaceRecognizer()
rec.load("recognizer/trainingData.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
firebase = firebase.FirebaseApplication('https://test-new-d1982.firebaseio.com/', None)
result = firebase.get('/users/', None)

class CamHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path.endswith('.mjpg'):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			while True:
				try:
					rc,img = capture.read()
					if not rc:
						continue
					# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					imgGARY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					faces = face_cascade.detectMultiScale(imgGARY, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
					for (x, y, w, h) in faces:
						cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
						id, conf = rec.predict(imgGARY[y:y+h, x:x+w])
						for val in result:
							if (id == result[val]["id"]):
								name = result[val]["name"]
								data = {"id":id, "name": name,"timestamp": str(datetime.now())}
								post_result = firebase.post('/check/', data)
								print post_result

						cv2.putText(img, str(name), (x, y + h), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
						cv2.putText(img, 'checked', (550, 450), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
					imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					jpg = Image.fromarray(imgRGB)
					tmpFile = StringIO.StringIO()
					jpg.save(tmpFile, 'JPEG')
					self.wfile.write("--jpgboundary")
					self.send_header('Content-type','image/jpeg')
					self.send_header('Content-length',str(tmpFile.len))
					self.end_headers()
					jpg.save(self.wfile,'JPEG')
					time.sleep(0.5)
				except KeyboardInterrupt:
					break
			return
		# if self.path.endswith('.html'):
		# 	self.send_response(200)
		# 	self.send_header('Content-type','text/html')
		# 	self.end_headers()
		# 	self.wfile.write('<html><head></head><body>')
		# 	self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
		# 	self.wfile.write('</body></html>')
		# 	return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

def main():
	global capture
	capture = cv2.VideoCapture(0)
	capture.set(3, 240);
	capture.set(4, 360);
	capture.set(12, 0.2);
	global img
	try:
		server = ThreadedHTTPServer(('localhost', 8080), CamHandler)
		print "server started"
		print "http://localhost:8080/cam.mjpg"
		server.serve_forever()
	except KeyboardInterrupt:
		capture.release()
		server.socket.close()

if __name__ == '__main__':
	main()
