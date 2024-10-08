import logging
from datetime import datetime
import requests
import cv2


class BankGuardBot:
    def __init__(self):
        # Telegram configuration
        self.TOKEN = "8106124538:AAFkPW8qQ7X9TtmsDV36glxCA60nG0zTinU"
        self.CHAT_ID = 5150162292  # Replace with your numerical chat ID
        # Initialize bot with request-based approach
        self.base_url = f"https://api.telegram.org/bot{self.TOKEN}"

        # Alert cooldown (to prevent spam)
        self.last_alert_time = datetime.now()
        self.alert_cooldown = 5  # seconds

    def built_connection(self):
        try:
            resp = requests.get(f"{self.base_url}.getMe")
            if resp.status_code == 200:
                logging.info("Successfully connected to Telegram bot")
            else:
                logging.error(f"Failed to connect to Telegram bot : {resp.text}")

        except Exception as e:
            logging.error(f"Failed to connect Telegram bot : {str(e)}")

    def send_alert(self, frame, message):
        """Send alert to Telegram with image and message"""
        try:
            self.built_connection()
            current_time = datetime.now()
            if (current_time - self.last_alert_time).seconds < self.alert_cooldown:
                return

            # Save the frame as an image
            alert_image_path = "alert.jpg"
            cv2.imwrite(alert_image_path, frame)

            # Prepare the message
            caption = f"⚠️ ALERT ⚠️\n{message}\nTime: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"

            # Send photo using requests
            with open(alert_image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.CHAT_ID, 'caption': caption}
                response = requests.post(f"{self.base_url}/sendPhoto", files=files, data=data)

            if response.status_code == 200:
                self.last_alert_time = current_time
                logging.info(f"Alert sent successfully: {message}")
            else:
                logging.error(f"Failed to send alert: {response.text}")

        except Exception as e:
            logging.error(f"Failed to send Telegram alert: {str(e)}")
