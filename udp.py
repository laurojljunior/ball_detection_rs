import socket
import threading

class UdpServer(object):

	def __init__(self, _ip, _port ):
		self.ip = _ip
		self.port = _port
		self.message = None
		
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	
	def send_data(self):
		self.sock.sendto(self.message.encode(), (self.ip, self.port))

	def send_message(self, message):
		self.message = message
		upd_thread = threading.Thread(target=self.send_data)
		upd_thread.start()