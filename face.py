import face_recognition
import sys

# Path to the default image (authorized person)
DEFAULT_IMAGE_PATH = "examplejpeg"

def recognize_face(known_image_path, unknown_image_path):
    """
    Recognize faces in the unknown image using the known image as reference.
    """
    try:
        known_image = face_recognition.load_image_file(known_image_path)
        unknown_image = face_recognition.load_image_file(unknown_image_path)

        
        known_encoding = face_recognition.face_encodings(known_image)[0]
        
        
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if not unknown_encodings:
            raise ValueError("No face detected in the unknown image.")
        
        
        results = face_recognition.compare_faces([known_encoding], unknown_encodings[0])
        return results[0]
    except FileNotFoundError:
        print("One or both image files not found.")
        sys.exit(1)
    except IndexError:
        print("No face detected in the known image.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    
    user_image_path = input("Enter the path to the user image: ")

    
    if recognize_face(DEFAULT_IMAGE_PATH, user_image_path):
        print("The faces match.")
    else:
        print("The faces do not match.")
