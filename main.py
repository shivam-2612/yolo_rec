from register import register_user
from recognize import recognize_faces

def main():
    print("========== Advanced Face Attendance System ==========")
    print("Select mode:")
    print("1 - Register New User")
    print("2 - Recognize & Mark Attendance")
    
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        register_user()
    elif choice == "2":
        recognize_faces()
    else:
        print("[ERROR] Invalid selection. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
