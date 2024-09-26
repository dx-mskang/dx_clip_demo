class InputData:
    default_sentence_score_min = 0.200
    default_sentence_score_max = 0.250
    default_sentence_score_threshold = 0.225
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

    from clip_demo_app_pyqt.model.sentence_model import Sentence
    sentence_list: list[Sentence] = [
        # video : "demo_videos/crowded_in_subway",
        Sentence("The subway is crowded with people",
            0.27, 0.29, 0.28),
        Sentence("People is crowded in the subway",
            0.27, 0.29, 0.28),

        # video : "demo_videos/heavy_structure_falling",
        Sentence("Heavy objects are fallen",
            0.21, 0.25, 0.225),

        # video :
        Sentence("Physical confrontation occurs between two people",
            0.23, 0.25, 0.24),
        Sentence("Violence with kicking and punching",
            0.22, 0.25, 0.23),

        # video :
        Sentence("Terrorism is taking place at the airport",
            0.27, 0.29, 0.28),
        Sentence("Terrorist is shooting at people",
            0.23, 0.26, 0.247),

        # video :
        Sentence("The water is exploding out",
            0.24, 0.28, 0.255),
        Sentence("The water is gushing out",
            0.24, 0.28, 0.255),

        # video :
        Sentence("Fire is coming out of the car",
            0.23, 0.26, 0.24),
        Sentence("The car is exploding",
            0.24, 0.28, 0.26),

        # video :
        Sentence("The electrical outlet on the wall is emitting smoke",
            0.23, 0.26, 0.24),
        Sentence("Smoke is rising from the electrical outlet.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("A pot on the induction cooktop is catching fire.",
            0.23, 0.26, 0.24),
        Sentence("A fire broke out in a pot in the kitchen.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("Two childrens are fighting.",
            0.23, 0.26, 0.24),
        Sentence("Two children start crying after a fight.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("Several men are engaged in a fight.",
            0.23, 0.26, 0.24),
        Sentence("Several people are fighting in the street.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("An elderly man is complaining of pain on the street.",
            0.23, 0.26, 0.24),
        Sentence("An man is crouching on the street.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("Someone helps old man who is falling down.",
            0.23, 0.26, 0.24),
        Sentence("An elderly grandfather is lying on the floor",
            0.23, 0.26, 0.24),

        # video :
        Sentence("A fire has occurred in the electric iron.",
            0.23, 0.26, 0.24),
        Sentence("The electric iron on the table is on fire.",
            0.23, 0.26, 0.24),

        # video :
        Sentence("Two men are engaging in mixed martial arts on the ring.",
            0.23, 0.26, 0.24),
    ]
