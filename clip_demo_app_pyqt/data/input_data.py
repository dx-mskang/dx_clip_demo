class InputData:
    default_sentence_threshold = [0.200, 0.250, 0.225]
    video_path_lists: list = [
        # [
        #     "/dev/video0"
        # ],
        [
            "demo_videos/fire_on_car",
        ],
        [
            "demo_videos/dam_explosion_short",
        ],
        [
            "demo_videos/violence_in_shopping_mall_short",
        ],
        [
            "demo_videos/gun_terrorism_in_airport",
        ],
        [
            "demo_videos/crowded_in_subway",
        ],
        [
            "demo_videos/heavy_structure_falling",
        ],
        [
            "demo_videos/electrical_outlet_is_emitting_smoke",
        ],
        [
            "demo_videos/pot_is_catching_fire",
        ],
        [
            "demo_videos/falldown_on_the_grass",
        ],
        [
            "demo_videos/fighting_on_field",
        ],
        [
            "demo_videos/fire_in_the_kitchen",
        ],
        [
            "demo_videos/group_fight_on_the_streat",
        ],
        [
            "demo_videos/iron_is_on_fire",
        ],
        [
            "demo_videos/someone_helps_old_man_who_is_fallting_down",
        ],
        [
            "demo_videos/the_pile_of_sockets_is_smoky_and_on_fire"
        ],
        [
            "demo_videos/two_childrens_are_fighting",
        ],
    ]
    sentence_alarm_threshold_list: list = [
        [0.27, 0.29, 0.28],  # "The subway is crowded with people",
        [0.27, 0.29, 0.28],  # "People is crowded in the subway",

        [0.21, 0.25, 0.225],  # "Heavy objects are fallen",

        [0.23, 0.25, 0.24],  # "Physical confrontation occurs between two people",
        [0.22, 0.25, 0.23],  # "Violence with kicking and punching",

        [0.27, 0.29, 0.28],  # "Terrorism is taking place at the airport",
        [0.23, 0.26, 0.247],  # "Terrorist is shooting at people",

        [0.24, 0.28, 0.255],  # "The water is exploding out",
        [0.24, 0.28, 0.255],  # "The water is gushing out",

        [0.23, 0.26, 0.24],  # "Fire is coming out of the car",
        [0.24, 0.28, 0.26],  # "The car is exploding",

        [0.23, 0.26, 0.24],  # "The electrical outlet on the wall is emitting smoke",
        [0.23, 0.26, 0.24],  # "Smoke is rising from the electrical outlet."

        [0.23, 0.26, 0.24],  # "A pot on the induction cooktop is catching fire.",
        [0.23, 0.26, 0.24],  # "A fire broke out in a pot in the kitchen."

        [0.23, 0.26, 0.24],  # "Two childrens are fighting.",
        [0.23, 0.26, 0.24],  # "Two children start crying after a fight."

        [0.23, 0.26, 0.24],  # "Several men are engaged in a fight.",
        [0.23, 0.26, 0.24],  # "Several people are fighting in the street.",

        [0.23, 0.26, 0.24],  # "An elderly man is complaining of pain on the street."
        [0.23, 0.26, 0.24],  # "An man is crouching on the street."

        [0.23, 0.26, 0.24],  # "Someone helps old man who is falling down."
        [0.23, 0.26, 0.24],  # "An elderly grandfather is lying on the floor."

        [0.23, 0.26, 0.24],  # "A fire has occurred in the electric iron."
        [0.23, 0.26, 0.24],  # "The electric iron on the table is on fire."

        [0.23, 0.26, 0.24],  # "Two men are engaging in mixed martial arts on the ring."
    ]

    sentence_list: list = [
        "The subway is crowded with people",
        "People is crowded in the subway",

        "Heavy objects are fallen",

        "Physical confrontation occurs between two people",
        "Violence with kicking and punching",

        "Terrorism is taking place at the airport",
        "Terrorist is shooting at people",

        "The water is exploding out",
        "The water is gushing out",

        "Fire is coming out of the car",
        "The car is exploding",

        "The electrical outlet on the wall is emitting smoke",
        "Smoke is rising from the electrical outlet.",

        "A pot on the induction cooktop is catching fire.",
        "A fire broke out in a pot in the kitchen.",

        "Two childrens are fighting.",
        "Two children start crying after a fight.",

        "Several men are engaged in a fight.",
        "Several people are fighting in the street.",

        "An elderly man is complaining of pain on the street.",
        "An man is crouching on the street.",

        "Someone helps old man who is falling down.",
        "An elderly grandfather is lying on the floor",

        "A fire has occurred in the electric iron.",
        "The electric iron on the table is on fire.",

        "Two men are engaging in mixed martial arts on the ring.",
    ]