# Function to read the contacts from a given contact file and return a
# list of names and email addresses
from string import Template
import email, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

MY_ADDRESS = 'xaviertorres950829@gmail.com'
MY_PASSWD  = '3WGWEk4JAs3PqwV5'

def read_template(filename):
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)

def send_mail(subject = "No Subject", body = "No Body"):
    # set up the SMTP server
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, MY_PASSWD)
    email = 'xyttyxy@g.ucla.edu'

    # import necessary packages
    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template
    message = body
    # setup the parameters of the message
    msg['From']=MY_ADDRESS
    msg['To']=email
    msg['Subject']=subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    # send the message via the server set up earlier.
    try:
        s.send_message(msg)
    except smtplib.SMTPDataError:
        print('Sending failed, daily quota exceeded')
    del msg
          
