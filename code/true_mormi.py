import json
from geopandas.tools import geocode
import time
start_time = time.time()

'''add the functions of extracting coordinates using Google Map and other geocoders'''

def get_coordinates_nomi(place_name):
    try:
        geo = geocode([place_name], provider='nominatim', user_agent='tim_gruenemay', timeout=10)
        lat = geo['geometry'][0].centroid.y
        lon = geo['geometry'][0].centroid.x
    except:
        lat = 0
        lon = 0
    return lat, lon

for data in ['chennai','FGLOC','harvey','houston','mix_events']:
    io = open(data+'.json',"r")
    json_data = json.load(io)
    locations = []
    keys = []
    total_return_results = {}
    for key in json_data.keys():
        keys.append(key)
        print(len(keys))
        return_objects = []
        for item in json_data[key]:
            entity = {}
            # entity=item
            lat, lon = get_coordinates_nomi(item['LOC'])
            entity['LOC'] = item['LOC']
            entity['start'] = int(item['start'])
            entity['end'] = int(item['end'])
            entity['lat'] = lat
            entity['lon'] = lon
            return_objects.append(entity)
        total_return_results[key] = return_objects
        f = open('nominatim_'+data+'.json', "w")
        json.dump(total_return_results, f)
        f.close()
    '''change the prefix of the saved file, such as replace Google with nominatim'''
    f = open('nominatim_'+data+'.json', "w")
    json.dump(total_return_results, f)
    f.close()
end_time = time.time()

print('time',end_time-start_time)
