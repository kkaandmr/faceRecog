import cv2
import face_recognition
import dlib

known_face_encodings=[]
known_face_names=[]

known_person1_image= face_recognition.load_image_file("elon-musk.jpg")
known_person1_encoding= face_recognition.face_encodings(known_person1_image)[0]
known_face_encodings.append(known_person1_encoding)
known_face_names.append("Elon Musk")

known_person2_image= face_recognition.load_image_file("steve-jobs.jpg")
known_person2_encoding= face_recognition.face_encodings(known_person2_image)[0]
known_face_encodings.append(known_person2_encoding)
known_face_names.append("Steve Jobs")

known_person3_image= face_recognition.load_image_file("bill-gates.jpg")
known_person3_encoding= face_recognition.face_encodings(known_person3_image)[0]
known_face_encodings.append(known_person3_encoding)
known_face_names.append("Bill Gates")

known_person4_image= face_recognition.load_image_file("mark-zuckerberg.jpg")
known_person4_encoding= face_recognition.face_encodings(known_person4_image)[0]
known_face_encodings.append(known_person4_encoding)
known_face_names.append("Mark Zuckerberg")

known_person5_image= face_recognition.load_image_file("jeff-bezos.jpg")
known_person5_encoding= face_recognition.face_encodings(known_person5_image)[0]
known_face_encodings.append(known_person5_encoding)
known_face_names.append("Jeff Bezos")


video_capture=cv2.VideoCapture(0)
check_every_nth_frame = 5
frame_count = 0
while True:
    ret,frame=video_capture.read()
    frame_width = 480
    frame_height = 360
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_count % check_every_nth_frame == 0:
        face_locations=face_recognition.face_locations(frame)
        face_encodings=face_recognition.face_encodings(frame,face_locations)
        for(top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
            matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
            name="Unknown"

            if True in matches:
                first_match_index=matches.index(True)
                name=known_face_names[first_match_index]

            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)

    cv2.imshow("Face Recognition",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


