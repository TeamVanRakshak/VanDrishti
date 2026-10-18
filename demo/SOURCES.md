# Demo clip sources

Run `python demo/fetch_clips.py` to download and prepare these clips in `demo/clips/` (not committed). Each clip is re-encoded to at most 1280×720 MP4; the content is otherwise unchanged.

| Clip | Scene | Source | Author | Licence |
|---|---|---|---|---|
| `day_elephant_herd.mp4` | Daytime elephant herd, Hoollongapar Gibbon Sanctuary, Jorhat, Assam | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A_herd_of_elephant_of_Hoollongapar_Gibbon_Sanctuary,_Jorhat_-_Assam.webm) | কুমুদ ঘোষ (Kumud Ghosh) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `drone_vehicle_forest.mp4` | Drone footage of a car driving through a snowy forest | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Aerial_views_of_a_car_driving_in_snow_through_a_forest.webm) | Vladislav Sanikovich | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) |
| `night_thermal_people.mp4` | Night thermal drone video with two people and six animals (BIRDSAI training sequence `0000000067_0000000059`) | [BIRDSAI / Conservation Drones, LILA BC](https://lila.science/datasets/conservationdrones) | Bondi et al., "BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos", WACV 2020 | [CDLA-Permissive-1.0](https://cdla.io/permissive-1-0/) |

`night_thermal_people.csv` holds the BIRDSAI ground-truth tracks for the thermal clip, in MOT format: frame, object id, x, y, w, h, class (0 animal, 1 human), species, occlusion, noise. BIRDSAI does not document the original frame rate; the clip is encoded at 25 fps.

The drone clip shows snow rather than an Indian landscape. It was the most suitable clearly licensed drone footage of a vehicle on a forest track.
