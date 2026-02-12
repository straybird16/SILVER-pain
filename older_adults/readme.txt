


------------------------------------------------------------------------

Under `original_files/` folder:

------------------------------------

`physiological_signal/` folder 

This foldercontains raw EmbracePlus recordings in .avro format downloaded from the Empatica data portal.
Each subject has various numbers of .avro files from disjoint period of time, with meta data and raw signal data. 
Raw data is stored in these separate segments (different .avro files) with missing values in between due to signal quality, device wearing conditions, and other artifacts or conditions.
Only BVP channel is on a perfect 64 Hz grid (15.625-millisecond-, or 15625-microsecond- steps).

See `extraction.py` under the `physiological_signal/` folder for the complete list of meta data and raw signal items, and the extraction of them to a python dictionary and .json file.
------------------------------------

`self_report/` folder 

This folder contains the log file in .csv format from the Apple iPad application with which subjects report pain levels (0-10 integers).
Each log file corresponds to one subject, and records all the actions from the subject.
"Trial" column represents the current stimulation trial number: for older adults, 5 TENS stimulation trials were conducted. `0` indicates resting period without stimulations.
`"Selected"` (as opposed to plain `Selected`) in "Action" column indicates the start of each TENS stimulation trial, and hence appears alongside the trail # flag.
Example: 2025-03-12 14:22:53.700,0,"Selected",1
"timestamp", "PainLevel", and "Action" columns are self-explanatory.

NOTE: self report logs are in Eastern Daylight Time (EDT) which is 4 HOURS BEHIND the Universal Coordiated Time (UTC) used in physiological signal files.
------------------------------------

`videa_data/` folder

This folder contains three modalities: 
1. deidentified facial action units extracted from the raw RGB video (permanently deleted per IRB regulations),
2. depth video
3. thermal video

1. and 2. were collected with the Intel RealSense D435i camera, 3. by the TOPDON TC004
------------------------------------

`screening_and_survey.xlsx`

This file contains each of the 7 older subject's answers in pre-screening, performance in minicog test, and answers in a few other survey questions.
------------------------------------
------------------------------------------------------------------------

`extraction_from_original_files/` folder:

Raw data directly extracted from each subject's .avro files. Multiple segments in each channel are merged into one .csv file.
No pre/post-processing.