import requests
import json

Query = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "ASK_ASK_query"}}
Qr = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "Iteration 34"}}
# L = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "ITER"}}
# Query['message']['text']='ITER'
# r = requests.post('http://gr.tgram.ml', json= LogMessage + '"ITER"}}' )
r = requests.post('https://gr.tgram.ml', json= Query)
print(r.json)