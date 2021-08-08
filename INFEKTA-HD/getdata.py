from ohsome import OhsomeClient
client = OhsomeClient()
print(client.base_api_url)
response = client.elements.count.post(bboxes=[8.625,49.3711,8.7334,49.4397],
				      time="2010-01-01/2020/01-01/PY1",
                                      filter="building=* and type:way")
print(response)