"""
Update json dicts to have records related to
data from RADAR_FRONT and CAM_FRONT only
Run this script inside the folder which contains the
json files
"""

import os.path as osp
import json
from nuscenes.nuscenes import NuScenes

def main():

	nusc = NuScenes(version='v1.0-mini', dataroot='/home/odysseas/thesis/data/sets/nuscenes/', verbose=True)
    #Fix sensor.json file
	sensor = nusc.sensor
	sensor[:] = [record for record in sensor if (record['channel'] == "CAM_FRONT") or (record['channel'] == "RADAR_FRONT")]

	with open('./sensor.json', 'w') as fout:
		json.dump(sensor , fout, indent=0)

	cam_front_token = sensor[0]["token"]
	radar_front_token = sensor[1]["token"]

	#Fix calibrated_sensor.json file
	calibrated_sensor = nusc.calibrated_sensor
	calibrated_sensor[:] = [record for record in calibrated_sensor if (record['sensor_token'] == cam_front_token) or (record['sensor_token'] == radar_front_token)]

	with open('./calibrated_sensor.json', 'w') as fout:
		json.dump(calibrated_sensor , fout, indent=0)

if __name__ == '__main__':
	main()
