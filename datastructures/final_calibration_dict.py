# intrinsics are updated after re-calibration on fiild on 20 June 2025
# after cleaning moisture from autovimation housings
# on 25.06.2025, dome to ouster calibration updted as per from liagdau
# on 27.06.2025, ouster to all 3 camera extrinsics updated after doing target-based calibration with undistorted images 

calibration_data = {
    "project_name": "inani",
    "reference_sensor": "ouster",
    "cameras": {
        "camera_01": {
            "image_width": 1920,
            "image_height": 1200,
            "intrinsic": [
                            [1189.298722, 0., 937.732109],
                            [0.     ,    1193.379998,    576.759555],
                            [0.     ,    0.     ,        1.]     
                        ],     

            "extrinsic": [  
                            [-0.91770636, -0.35578743, -0.17672107,  0.04730193],
                            [-0.00886123,  0.46307316, -0.88627576,  1.59187633],
                            [ 0.39716056, -0.81177494, -0.4281179,   0.43108148]
                        ],      
            "distortion": [-0.193992, 0.039889, -0.001975, 0.000858, 0.000000],
            "time_offset_msec": 30     
        },
        "camera_02": {
            "image_width": 1920,
            "image_height": 1200,
            "intrinsic": [
                            [1198.800084, 0., 945.108342],
                            [0., 1202.781813, 549.525267],
                            [0.     ,    0.     ,         1.]     
                        ],
            "extrinsic": [
                            [ 0.10708462, -0.99161217, -0.07237533,  0.20387646],
                            [-0.71540563, -0.02629601, -0.69821437,  0.91142134],
                            [ 0.69045468,  0.12654573, -0.71222083,  1.10544099]
                        ],
            "distortion": [-0.223967, 0.071287, -0.000750, 0.001043, 0.000000],
            "time_offset_msec": 50,
        },
        "camera_03": {
            "image_width": 1920,
            "image_height": 1200,
            "intrinsic": [
                            [1171.42979, 0., 921.05849],
                            [0., 1176.724, 591.97348],
                            [0.     ,    0.     ,        1.]     
                        ],
            "extrinsic": [
                            [ 0.94507554, -0.29234944,  0.14616437,  0.36075291],
                            [-0.00558908, -0.4615775,  -0.88708228,  1.422794  ],
                            [ 0.3268042,   0.83754284, -0.43785957,  0.13339311]
                        ],
            "distortion": [-0.234304, 0.073384, -0.000797, 0.002413, 0.000000],

            "time_offset_msec": 65
        }
    },
    "lidars": {
        "ouster": {
            "vertical_channels": 64,
            "horizontal_channels": 1024,
            "extrinsic": [
                            [1.0,    0.0,    0.0,    0.0],
                            [0.0,    1.0,    0.0,    0.0],
                            [0.0,    0.0,    1.0,    0.0],
                            [0.0,    0.0,    0.0,    1.0]
                        ]       
        },
        "dome": {
            "vertical_channels": 128,
            "horizontal_channels": 1024,
            "extrinsic": [  
                            [
                                0.9999902340243306,
                                0.004337761930234445,
                                0.0008459771874470788,
                                0.005537291470224765
                            ],
                            [
                                0.0043367757130853895,
                                -0.9999899185222066,
                                0.0011641436189942002,
                                0.007272841252342369
                            ],
                            [
                                0.0008510184366186476,
                                -0.0011604634366755965,
                                -0.9999989645455803,
                                -0.04632852758311232
                            ]
                        ]
        }
    },
    "ouster_to_ground":{
        "extrinsic": [
                            [1.0,    0.0,    0.0,    0.0],
                            [0.0,    1.0,    0.0,    0.0],
                            [0.0,    0.0,    1.0,    0.0],
                            [0.0,    0.0,    0.0,    1.0]
                        ]  
    }
}

    