while True:
    sample = {
        "image_front": front_cam_path,
        "image_front_left": left_cam_path,
        "image_front_right": right_cam_path,
        "seg_front": seg_cam_path,
        "velocity_x": vx,
        "velocity_y": vy,
        "velocity_z": vz,
        "speed_kmh": speed,
        "nearest_object_dist": dist,
        "box_count": num_objects
    }

    control = predictor.predict(sample)

    car.apply_control(
        steer=control["steer"],
        throttle=control["throttle"],
        brake=control["brake"]
    )
