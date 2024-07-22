
#  Automated Attendance Management System using Face Recognition

The Automated Attendance Management System emphasizes the utilization of Face Recognition Model for recognizing the individuals. By integrating facial recognition technology, the system offers a non-intrusive and contactless solution for attendance management. It can accurately identify and register students' attendance without requiring any manual intervention, reducing the potential for human errors and ensuring data integrity. Additionally, the system's ability to operate in real-time allows for instantaneous attendance updates, providing educators and administrators with up-to-date information for efficient decision-making. Various tools like Python, OpenCV, and the face_recognition library were used to build this project.
## Instructions

1. Run `app.py` to open an user-interface (Streamlit) to upload    the student images.

2. The uploaded images gets stored in "student_images" directory.
    
3. Ensure you have the necessary libraries installed (e.g.,OpenCV, face_recognition,Streamlit)
    
4. Run `precompute_encodings.py` to call the "load_images_from_folder" and "encode_faces" function from `face_recognition_utils.py` 
    
5. Run the `main.py` to open the webcam for live video capture and automatic attendance marking.

## Usage

* The system captures faces through the webcam and compares them with stored images.

* Detected faces are labeled with the corresponding student's name  and 'unknown' for faces not in the dataset.

* The attendance is then marked in a CSV file along with the absent tag for students not present.

## Authors

- Saurav S Suresh

