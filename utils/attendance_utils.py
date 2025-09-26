from datetime import datetime

def mark_attendance(db, user_id, name):
    today = datetime.now().date().isoformat()
    existing = db.attendance.find_one({"user_id": user_id, "date": today})
    if not existing:
        now_time = datetime.now().strftime("%H:%M:%S")
        db.attendance.insert_one({
            "user_id": user_id,
            "name": name,
            "date": today,
            "time": now_time
        })
        print(f"[INFO] Attendance marked for {name} at {now_time}")
    else:
        print(f"[INFO] {name} already marked today.")
