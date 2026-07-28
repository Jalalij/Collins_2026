import requests 

response = requests.get("http://10.189.11.16/api/antenna/antennatuneinfo", timeout=10) 
print(response.json()) 
response = requests.get("http://10.189.11.16/api/gnss/gnssstats", timeout=10) 
print("\n") 
print(response.json()) 
response = requests.get("http://10.189.11.16/api/modem/lte_signal_info", timeout=10) 
print("\n") 
print(response.json()) 
response = requests.get("http://10.189.11.16/api/antenna/cached_messages", timeout=10) 
print("\n") 
print(response.json())