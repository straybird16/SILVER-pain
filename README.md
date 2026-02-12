# README

The SILVER-Pain Dataset consists of two parts: an older cohort with 7 subjects, and a young cohort with 18 subjects.

Physiological signals in the young cohort were collected with the Empatica E4 wristband. Those from the older cohort was collected with the newer Empatica EmbracePlus wristband (which is, according to Empatica, a updated version of E4, after it was sunset).

Deidentified depth video, thermal video and facial expression data were both collected by (1) an Intel RealSense 435i Camera, (2) a thermal camera, and (3) converted from the RGB video colleceted by the RealSense camera with the open-source OpenFace library: [https://github.com/TadasBaltrusaitis/OpenFace]().

Original files and processed files are included.

Note that raw data is not synchronously and regularly sampled. There are also missing values and signal artifacts, and young adults' data format is slightly different from older adults (see readmes in subfolders). We provide basic functions with some config options to extract and preprocess the raw data into machine learning ready dataframe and data arrays. If you wish, you can build and use your own functions to extract and process the signals.



Please see subfolders for more specific information.
