
import numpy as np

#################################################################
roi_left_img = [0,0,0,0]

cfg_dict_init = {
    "arm1": {
        "gripper_port": '/dev/ttyUSB0',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB0
        "robot_ip_add": "192.168.9.134",  # robot IP address "192.168.31.134"
        "handeye_para": np.array([
            [-0.14495955, -0.82672254,  0.54361435, -0.46128096],
            [-0.98826554,  0.09424301, -0.120206  , -0.55922609],
            [ 0.04814516, -0.55466034, -0.83068282,  0.77215299],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])  # camera calib in 2024-11-20
    },
    "arm2": {
        "gripper_port": '/dev/ttyUSB1',  # for Ubuntu, sudo chmod 666 /dev/ttyUSB1
        "robot_ip_add": "192.168.9.135",  # robot IP address "192.168.31.135"
        "handeye_para": np.array([
            [ 0.15464175,  0.82413502, -0.54487375,  0.48699159],
            [ 0.98677   , -0.10165977,  0.12629433, -0.77214491],
            [ 0.04869184, -0.55719545, -0.82895256,  0.76490652],
            [-0.        ,  0.        , -0.        ,  1.        ]
        ])  # camera calib in 2024-11-20 
    },
}

#################################################################

video_id_str_list = [
    "drawer_01", "drawer_02", "drawer_03", "drawer_04",
    "pouring_01", "pouring_02", "pouring_03", "pouring_04", "pouring_05",
    "uncover_01", "uncover_02",
    "unscrew_01",
    "openbox_01",
    "invert_01",
    
    "redirect_01", "redicrect_02",
    "reorient_01",
    "dualpap_01",
    "insertpen_01",
    "stacking_01"
    ]

''' 1:open, 0:close, -1:twist, -2:lift, -3:pour, -4:uncover, -5:detach, -6:unclasp '''
video_info_dict = {
    #######################################################################
    "drawer_01": {
        "cam_l_name": "Video_20241107170835749_drawer",  # keyframe 271
        "cam_r_name": "Video_20241107170837409_drawer",  # keyframe 247
        "keyframe_id": {240:"R", 253:"L", 271:"R", 309:"R", 319:"L", 352:"L", 362:"L", 370:"L", 404:"L", 439:"R"},  # 10
        "gripper_state": {240:1, 253:1, 271:0, 309:0, 319:1, 352:0, 362:0, 370:1, 404:0, 439:1},  # 1:open, 0:close
        "showing_ids": [],
    },
    "drawer_02": {
        "cam_l_name": "Video_20241111164131621_drawer",  # keyframe 110
        "cam_r_name": "Video_20241111164133389_drawer",  # keyframe 56
        "keyframe_id": {87:"R", 100:"R", 108:"L", 124:"R", 131:"L", 143:"L", 158:"L", 167:"L", 192:"L", 216:"R"},  # 10
        "gripper_state": {87:1, 100:0, 108:1, 124:0, 131:1, 143:0, 158:0, 167:1, 192:1, 216:1},  # open + close + pull
        "showing_ids": [],
    }, 
    "drawer_03": {
        "cam_l_name": "Video_20241125173800357_drawer",  # keyframe 80
        "cam_r_name": "Video_20241125173801551_drawer",  # keyframe 64
        "keyframe_id": {44:"R", 60:"R", 61:"L", 80:"R", 85:"L", 100:"L", 112:"L", 120:"L", 131:"L", 144:"R"},  # 10
        "gripper_state": {44:1, 60:0, 61:1, 80:0, 85:1, 100:0, 112:0, 120:1, 131:1, 144:1},  # open + close + pull
        "showing_ids": [],
    },  
    "drawer_04": {
        "cam_l_name": "Video_20241128090342905_drawer",  # keyframe 71
        "cam_r_name": "Video_20241128090344246_drawer",  # keyframe 28
        # "keyframe_id": {54:"R", 70:"R", 72:"L", 100:"R", 109:"L", 125:"L", 140:"L", 155:"L", 167:"L", 180:"R"},  # 10 (for paper)
        # "gripper_state": {54:1, 70:0, 72:1, 100:0, 109:1, 125:0, 140:0, 155:1, 167:1, 180:1},  # open + close + pull
        "keyframe_id": {53:"R", 70:"R", 72:"L", 100:"R", 111:"L", 125:"L", 135:"L", 155:"L", 167:"L", 180:"R"},  # 10 (for real robot)
        "gripper_state": {53:1, 70:0, 72:1, 100:0, 111:1, 125:0, 135:0, 155:1, 167:1, 180:1},  # open + close + pull
        "showing_ids": [80, 140, 172],
    }, 
    # real atomic actions: 1+1+1+1+1+1+1+1+1+1 = 10
    #######################################################################
    "pouring_01": {
        "cam_l_name": "Video_20241112104908847_pouring",  # keyframe 116
        "cam_r_name": "Video_20241112104910609_pouring",  # keyframe 69
        "keyframe_id": {101:"R", 116:"L", 136:"L", 147:"R", 167:"R", 170:"R", 177:"R", 186:"R", 190:"L", 202:"R", 225:"R", 234:"L", 260:"R", 275:"L"},  # 14
        "gripper_state": {101:1, 116:0, 136:0, 147:-1, 167:-1, 170:0, 177:0, 186:1, 190:0, 202:0, 225:0, 234:-2, 260:1, 275:1},  # open + close + twist + pour
        "showing_ids": [],
    },
    "pouring_02": {
        "cam_l_name": "Video_20241112212035249_pouring",  # keyframe 115
        "cam_r_name": "Video_20241112212036925_pouring",  # keyframe 75
        "keyframe_id": {116:"L", 120:"R", 140:"L", 141:"R", 142:"R", 150:"R", 156:"R", 160:"L", 169:"R", 180:"R", 190:"L", 210:"R", 222:"L"},  # 13
        "gripper_state": {116:0, 120:1, 140:0, 141:-1, 142:0, 150:0, 156:1, 160:0, 169:0, 180:0, 190:-2, 210:1, 222:1},  # open + close + twist + pour
        "showing_ids": [],
    }, 
    "pouring_03": {
        "cam_l_name": "Video_20241115155332125_pouring",  # keyframe 62
        "cam_r_name": "Video_20241115155333066_pouring",  # keyframe 21
        "keyframe_id": {61:"L", 66:"R", 96:"L", 97:"R", 98:"R", 101:"R", 104:"R", 110:"R", 135:"L", 138:"R", 150:"R", 160:"L", 188:"R", 202:"L"},  # 14
        "gripper_state": {61:0, 66:1, 96:0, 97:-1, 98:0, 101:0, 104:0, 110:1, 135:0, 138:0, 150:0, 160:-2, 188:1, 202:1},  # open + close + twist + pour
        "showing_ids": [],
    },
    "pouring_04": {
        "cam_l_name": "Video_20241115195458756_pouring",  # keyframe 82
        "cam_r_name": "Video_20241115195500151_pouring",  # keyframe 40
        "keyframe_id": {85:"L", 86:"R", 91:"R", 96:"L", 102:"R", 120:"L", 174:"R", 178:"L", 187:"R"},  # 9
        "gripper_state": {85:0, 86:0, 91:0, 96:0, 102:0, 120:-2, 174:0, 178:1, 187:1},  # open + close + pour
        "showing_ids": [],
    }, 
    "pouring_05": {
        "cam_l_name": "Video_20241125112516999_pouring",  # keyframe 64
        "cam_r_name": "Video_20241125112518562_pouring",  # keyframe 37
        "keyframe_id": {62:"R", 63:"L", 82:"L", 90:"R", 110:"L", 156:"L", 169:"R"},  # 7
        "gripper_state": {62:-2, 63:-2, 82:0, 90:0, 110:-3, 156:1, 169:1},  # open + close + lift + pour
        "showing_ids": [70, 120, 168],
    },  # real atomic actions: 2+2+1+1+3+1+1 = 11
    #######################################################################
    "uncover_01": {
        "cam_l_name": "Video_20241115222347585_uncover",  # keyframe 127
        "cam_r_name": "Video_20241115222349751_uncover",  # keyframe 61
        "keyframe_id": {127:"LR", 136:"LR", 144:"LR", 152:"LR", 168:"LR", 170:"LR"},  # 6 x 2
        "gripper_state": {127:1, 136:1, 144:1, 152:1, 168:1, 170:-4},  # open + detach
        "showing_ids": [],
    },
    "uncover_02":{
        "cam_l_name": "Video_20241120181731061_uncover",  # keyframe 85
        "cam_r_name": "Video_20241120181732458_uncover",  # keyframe 40
        # "keyframe_id": {80:"LR", 96:"LR", 98:"LR", 100:"LR", 102:"LR", 105: "LR", 107: "LR", 110:"LR", 116:"LR"},  # 9 x 2
        # "gripper_state": {80:-4, 96:1, 98:1, 100:1, 102:1, 105:1, 107:1, 110:1, 116:-5},  # open + uncover + detach
        "keyframe_id": {80:"LR", 96:"LR", 98:"LR", 100:"LR", 102:"LR", 105: "LR", 107: "LR", 110:"LR", 113:"LR", 116:"LR"},  # 10 x 2
        "gripper_state": {80:-4, 96:1, 98:1, 100:1, 102:1, 105:1, 107:1, 110:1, 113:1, 116:-5},  # open + uncover + detach
        "showing_ids": [80, 106, 122],
    },  # real atomic actions: 2+1+1+1+1+1+1+1+1+2 = 12
    #######################################################################
    "unscrew_01":{
        "cam_l_name": "Video_20241122141747978_unscrew",  # keyframe 61
        "cam_r_name": "Video_20241122141748998_unscrew",  # keyframe 45
        "keyframe_id": {61:"L", 86:"L", 87:"R", 99:"R", 104:"L", 105:"R", 114:"L", 130:"L"},  # 8
        "gripper_state": {61:-2, 86:0, 87:-1, 99:0, 104:0, 105:1, 114:0, 130:1},  # open + close + twist
        "showing_ids": [66, 87, 108],
    },  # real atomic actions: 2+1+4+1+1+1+1+1 = 12
    #######################################################################
    "openbox_01":{
        "cam_l_name": "Video_20241122163050766_openbox",  # keyframe 50   
        "cam_r_name": "Video_20241122163051781_openbox",  # keyframe 30
        "keyframe_id": {50:"LR", 56:"LR", 58:"LR", 60:"LR", 65:"LR", 70:"LR", 75:"LR", 
                        95:"LR", 118:"LR", 121:"LR", 123:"LR", 125:"LR", 132:"LR", 139:"LR"},  # 14 x 2
        "gripper_state": {50:1, 56:1, 58:1, 60:1, 65:1, 70:1, 75:-6, 
                          95:1, 118:1, 121:1, 123:1, 125:1, 132:1, 139:-6},  # open + unclasp
        "showing_ids": [50, 80, 145],
    },  # real atomic actions: 1+1+1+1+1+1+2+1+1+1+1+1+1+2 = 16
    #######################################################################
    
    ####################################################################### 
    "invert_01":{  # place upside down woven basket
        "cam_l_name": "Video_20250122150737697_invert",  # keyframe 57   
        "cam_r_name": "Video_20250122150739025_invert",  # keyframe 37
        "keyframe_id": {57:"LR", 61:"LR", 65:"LR", 69:"LR", 73:"LR", 77:"LR", 85:"LR"},  # 7 x 2
        "gripper_state": {57:0, 61:0, 65:0, 69:0, 73:0, 77:0, 85:-5},  # approaching and moving + detach
        "showing_ids": [48, 66, 80],
    },  # real atomic actions: 1+1+1+1+1+1+2 = 8
    ####################################################################### 
    "redirect_01":{  # place two pens into the pen holder
        "cam_l_name": "Video_20250122193711380_redirect",  # keyframe 83   
        "cam_r_name": "Video_20250122193712851_redirect",  # keyframe 44
        "keyframe_id": {83:"R", 88:"L", 97:"R", 106:"L", 107:"R", 114:"R", 134:"R", 137:"R", 145:"R", 165:"L"},
        "gripper_state": {83:0, 88:0, 97:0, 106:0, 107:1, 114:1, 134:0, 137:0, 145:1, 165:1},  # cup + pen1 + pen2
        "showing_ids": [91, 108, 135, 147],
    },  # real atomic actions: 1+1+1+1 + 1+1+1+1+1 + 1 = 10
    "redirect_02":{  # place two pens into the pen holder
        "cam_l_name": "Video_20250219160553949_redirect",  # keyframe 94  
        "cam_r_name": "Video_20250219160556160_redirect",  # keyframe 42
        "keyframe_id": {93:"R", 94:"L", 100:"R", 105:"L", 106:"R", 110:"R", 120:"R", 125:"R", 135:"R", 158:"L"},
        "gripper_state": {93:0, 94:0, 100:0, 105:0, 106:1, 110:1, 120:0, 125:0, 135:1, 158:1},  # cup + pen1 + pen2
        "showing_ids": [96, 107, 121, 137],
    },  # real atomic actions: 1+1+1+1 + 1+1+1+1+1 + 1 = 10
    #######################################################################
    "reorient_01":{  # up side down the blackboard
        "cam_l_name": "Video_20250220144445293_reorient",  # keyframe 132  
        "cam_r_name": "Video_20250220144448278_reorient",  # keyframe 56
        "keyframe_id": {131:"R", 149:"R", 150:"L", 158:"R", 177:"L"},
        "gripper_state": {131:0, 149:0, 150:0, 158:1, 177:1},
        "showing_ids": [132, 151, 159, 178],
    },  # real atomic actions: 1+1+1 + 1 + 1 = 5
    #######################################################################
    "dualpap_01":{  # pick and place + reorientation
        "cam_l_name": "Video_20250220154026495_dualpap",  # keyframe 121  
        "cam_r_name": "Video_20250220154028914_dualpap",  # keyframe 58
        "keyframe_id": {121:"R", 123:"L", 136:"L", 137:"R", 167:"L"},
        "gripper_state": {121:0, 123:0, 136:0, 137:1, 167:1},
        "showing_ids": [121, 136, 166],
    },  # real atomic actions: 2+2 + 1 + 1 + 1 = 7
    #######################################################################
    "insertpen_01":{  # pick and place + insert pen
        "cam_l_name": "Video_20250225123729086_insertpen",  # keyframe 142  
        "cam_r_name": "Video_20250225123731630_insertpen",  # keyframe 74
        "keyframe_id": {141:"L", 142:"R", 183:"L", 184:"R", 220:"L", 221:"R", 295:"L"},
        "gripper_state": {141:0, 142:0, 183:0, 184:0, 220:0, 221:1, 295:1},
        "showing_ids": [143, 185, 222, 296],
    },  # real atomic actions: 1+1 + 1+1 + 1+1 + 1 = 7
    #######################################################################
    "stacking_01":{  # pick and place + stacking two cups
        "cam_l_name": "Video_20250304144113965_stacking",  # keyframe 89  
        "cam_r_name": "Video_20250304144115706_stacking",  # keyframe 42
        "keyframe_id": {88:"L", 89:"R", 103:"R", 104:"L", 125:"L"},
        "gripper_state": {88:-2, 89:-2, 103:1, 104:0, 125:1},
        "showing_ids": [90, 105, 126, 143],
    },  # real atomic actions: 2+2 + 1+1+1 = 7
   

    #######################################################################
    "pouring_06":{
        "cam_l_name": "Video_20250530194413211_pouring",  # keyframe 156/170  
        "cam_r_name": "Video_20250530194415130_pouring",  # keyframe 121/135
        "keyframe_id": {170:"R", 178:"L", 200:"L", 210:"R", 240:"L", 328:"L", 329:"R"},  # 7
        "gripper_state": {170:-2, 178:-2, 200:0, 210:0, 240:-3, 328:1, 329:1},  # open + close + lift + pour
        "showing_ids": [180, 215, 295, 340],
    },  # real atomic actions: 2+2+1+1+3+1+1 = 11
    "unscrew_02":{
        "cam_l_name": "Video_20250530194005505_unscrew",  # keyframe 158/171
        "cam_r_name": "Video_20250530194007289_unscrew",  # keyframe 113/126
        "keyframe_id": {171:"L", 190:"L", 210:"R", 228:"R", 230:"L", 241:"R", 265:"L", 280:"L"},  # 8
        "gripper_state": {171:-2, 190:0, 210:-1, 228:0, 230:0, 241:1, 265:0, 280:1},  # open + close + twist
        "showing_ids": [190, 215, 230, 280],
    },  # real atomic actions: 2+1+4+1+1+1+1+1 = 12
    
}