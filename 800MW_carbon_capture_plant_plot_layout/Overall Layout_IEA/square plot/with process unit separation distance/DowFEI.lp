\* Layout_Problem_Model *\
Minimize
OBJ: SumCD + SumOmega + SumVlc + TLC + sum_pumping_cost
Subject To
_C1: 0 O_direct_contact_cooler + l_direct_contact_cooler = 66
_C10: d_lean_amine_cooler + l_lean_amine_cooler = 60
_C100: - 1000 E1_absorber_lean_amine_cooler
 - 1000 E2_absorber_lean_amine_cooler - Y_absorber + Y_lean_amine_cooler
 - 0.5 d_absorber - 0.5 d_lean_amine_cooler >= -1985
_C1000: - 1000 E1_compressor_house_TEG_dehydration
 - 1000 E2_compressor_house_TEG_dehydration + Y_TEG_dehydration
 - Y_compressor_house - 0.5 d_TEG_dehydration - 0.5 d_compressor_house
 >= -1991
_C1001: L_compressor_house_TEG_dehydration
 - R_TEG_dehydration_compressor_house = 0
_C1002: - L_TEG_dehydration_compressor_house
 + R_compressor_house_TEG_dehydration = 0
_C1003: A_compressor_house_TEG_dehydration
 - B_TEG_dehydration_compressor_house = 0
_C1004: - A_TEG_dehydration_compressor_house
 + B_compressor_house_TEG_dehydration = 0
_C1005: Wx_TEG_dehydration_compressor_house
 + Wx_compressor_house_TEG_dehydration = 1
_C1006: Wy_TEG_dehydration_compressor_house
 + Wy_compressor_house_TEG_dehydration = 1
_C1007: - L_compressor_KOdrum_TEG_dehydration
 + R_compressor_KOdrum_TEG_dehydration + X_TEG_dehydration
 - X_compressor_KOdrum = 0
_C1008: A_compressor_KOdrum_TEG_dehydration
 - B_compressor_KOdrum_TEG_dehydration + Y_TEG_dehydration
 - Y_compressor_KOdrum = 0
_C1009: R_compressor_KOdrum_TEG_dehydration
 - 1000 Wx_compressor_KOdrum_TEG_dehydration <= 0
_C101: L_absorber_lean_amine_cooler - R_lean_amine_cooler_absorber = 0
_C1010: L_compressor_KOdrum_TEG_dehydration
 + 1000 Wx_compressor_KOdrum_TEG_dehydration <= 1000
_C1011: A_compressor_KOdrum_TEG_dehydration
 - 1000 Wy_compressor_KOdrum_TEG_dehydration <= 0
_C1012: B_compressor_KOdrum_TEG_dehydration
 + 1000 Wy_compressor_KOdrum_TEG_dehydration <= 1000
_C1013: - A_compressor_KOdrum_TEG_dehydration
 - B_compressor_KOdrum_TEG_dehydration + D_compressor_KOdrum_TEG_dehydration
 - L_compressor_KOdrum_TEG_dehydration - R_compressor_KOdrum_TEG_dehydration
 = 0
_C1014: - D_TEG_dehydration_compressor_KOdrum
 + D_compressor_KOdrum_TEG_dehydration = 0
_C1015: 1000 E1_compressor_KOdrum_TEG_dehydration
 + 1000 E2_compressor_KOdrum_TEG_dehydration - X_TEG_dehydration
 + X_compressor_KOdrum - 0.5 l_TEG_dehydration - 0.5 l_compressor_KOdrum >= 15
_C1016: - 1000 E1_compressor_KOdrum_TEG_dehydration
 + 1000 E2_compressor_KOdrum_TEG_dehydration + X_TEG_dehydration
 - X_compressor_KOdrum - 0.5 l_TEG_dehydration - 0.5 l_compressor_KOdrum
 >= -985
_C1017: 1000 E1_compressor_KOdrum_TEG_dehydration
 - 1000 E2_compressor_KOdrum_TEG_dehydration - Y_TEG_dehydration
 + Y_compressor_KOdrum - 0.5 d_TEG_dehydration - 0.5 d_compressor_KOdrum
 >= -985
_C1018: - 1000 E1_compressor_KOdrum_TEG_dehydration
 - 1000 E2_compressor_KOdrum_TEG_dehydration + Y_TEG_dehydration
 - Y_compressor_KOdrum - 0.5 d_TEG_dehydration - 0.5 d_compressor_KOdrum
 >= -1985
_C1019: L_compressor_KOdrum_TEG_dehydration
 - R_TEG_dehydration_compressor_KOdrum = 0
_C102: - L_lean_amine_cooler_absorber + R_absorber_lean_amine_cooler = 0
_C1020: - L_TEG_dehydration_compressor_KOdrum
 + R_compressor_KOdrum_TEG_dehydration = 0
_C1021: A_compressor_KOdrum_TEG_dehydration
 - B_TEG_dehydration_compressor_KOdrum = 0
_C1022: - A_TEG_dehydration_compressor_KOdrum
 + B_compressor_KOdrum_TEG_dehydration = 0
_C1023: Wx_TEG_dehydration_compressor_KOdrum
 + Wx_compressor_KOdrum_TEG_dehydration = 1
_C1024: Wy_TEG_dehydration_compressor_KOdrum
 + Wy_compressor_KOdrum_TEG_dehydration = 1
_C1025: - L_compressor_interstage_cooler_TEG_dehydration
 + R_compressor_interstage_cooler_TEG_dehydration + X_TEG_dehydration
 - X_compressor_interstage_cooler = 0
_C1026: A_compressor_interstage_cooler_TEG_dehydration
 - B_compressor_interstage_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_compressor_interstage_cooler = 0
_C1027: R_compressor_interstage_cooler_TEG_dehydration
 - 1000 Wx_compressor_interstage_cooler_TEG_dehydration <= 0
_C1028: L_compressor_interstage_cooler_TEG_dehydration
 + 1000 Wx_compressor_interstage_cooler_TEG_dehydration <= 1000
_C1029: A_compressor_interstage_cooler_TEG_dehydration
 - 1000 Wy_compressor_interstage_cooler_TEG_dehydration <= 0
_C103: A_absorber_lean_amine_cooler - B_lean_amine_cooler_absorber = 0
_C1030: B_compressor_interstage_cooler_TEG_dehydration
 + 1000 Wy_compressor_interstage_cooler_TEG_dehydration <= 1000
_C1031: - A_compressor_interstage_cooler_TEG_dehydration
 - B_compressor_interstage_cooler_TEG_dehydration
 + D_compressor_interstage_cooler_TEG_dehydration
 - L_compressor_interstage_cooler_TEG_dehydration
 - R_compressor_interstage_cooler_TEG_dehydration = 0
_C1032: - D_TEG_dehydration_compressor_interstage_cooler
 + D_compressor_interstage_cooler_TEG_dehydration = 0
_C1033: 1000 E1_compressor_interstage_cooler_TEG_dehydration
 + 1000 E2_compressor_interstage_cooler_TEG_dehydration - X_TEG_dehydration
 + X_compressor_interstage_cooler - 0.5 l_TEG_dehydration
 - 0.5 l_compressor_interstage_cooler >= 15
_C1034: - 1000 E1_compressor_interstage_cooler_TEG_dehydration
 + 1000 E2_compressor_interstage_cooler_TEG_dehydration + X_TEG_dehydration
 - X_compressor_interstage_cooler - 0.5 l_TEG_dehydration
 - 0.5 l_compressor_interstage_cooler >= -985
_C1035: 1000 E1_compressor_interstage_cooler_TEG_dehydration
 - 1000 E2_compressor_interstage_cooler_TEG_dehydration - Y_TEG_dehydration
 + Y_compressor_interstage_cooler - 0.5 d_TEG_dehydration
 - 0.5 d_compressor_interstage_cooler >= -985
_C1036: - 1000 E1_compressor_interstage_cooler_TEG_dehydration
 - 1000 E2_compressor_interstage_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_compressor_interstage_cooler - 0.5 d_TEG_dehydration
 - 0.5 d_compressor_interstage_cooler >= -1985
_C1037: L_compressor_interstage_cooler_TEG_dehydration
 - R_TEG_dehydration_compressor_interstage_cooler = 0
_C1038: - L_TEG_dehydration_compressor_interstage_cooler
 + R_compressor_interstage_cooler_TEG_dehydration = 0
_C1039: A_compressor_interstage_cooler_TEG_dehydration
 - B_TEG_dehydration_compressor_interstage_cooler = 0
_C104: - A_lean_amine_cooler_absorber + B_absorber_lean_amine_cooler = 0
_C1040: - A_TEG_dehydration_compressor_interstage_cooler
 + B_compressor_interstage_cooler_TEG_dehydration = 0
_C1041: Wx_TEG_dehydration_compressor_interstage_cooler
 + Wx_compressor_interstage_cooler_TEG_dehydration = 1
_C1042: Wy_TEG_dehydration_compressor_interstage_cooler
 + Wy_compressor_interstage_cooler_TEG_dehydration = 1
_C1043: - L_direct_contact_cooler_cooling_tower
 + R_direct_contact_cooler_cooling_tower + X_cooling_tower
 - X_direct_contact_cooler = 0
_C1044: A_direct_contact_cooler_cooling_tower
 - B_direct_contact_cooler_cooling_tower + Y_cooling_tower
 - Y_direct_contact_cooler = 0
_C1045: R_direct_contact_cooler_cooling_tower
 - 1000 Wx_direct_contact_cooler_cooling_tower <= 0
_C1046: L_direct_contact_cooler_cooling_tower
 + 1000 Wx_direct_contact_cooler_cooling_tower <= 1000
_C1047: A_direct_contact_cooler_cooling_tower
 - 1000 Wy_direct_contact_cooler_cooling_tower <= 0
_C1048: B_direct_contact_cooler_cooling_tower
 + 1000 Wy_direct_contact_cooler_cooling_tower <= 1000
_C1049: - A_direct_contact_cooler_cooling_tower
 - B_direct_contact_cooler_cooling_tower
 + D_direct_contact_cooler_cooling_tower
 - L_direct_contact_cooler_cooling_tower
 - R_direct_contact_cooler_cooling_tower = 0
_C105: Wx_absorber_lean_amine_cooler + Wx_lean_amine_cooler_absorber = 1
_C1050: - D_cooling_tower_direct_contact_cooler
 + D_direct_contact_cooler_cooling_tower = 0
_C1051: 1000 E1_direct_contact_cooler_cooling_tower
 + 1000 E2_direct_contact_cooler_cooling_tower - X_cooling_tower
 + X_direct_contact_cooler - 0.5 l_cooling_tower - 0.5 l_direct_contact_cooler
 >= 31
_C1052: - 1000 E1_direct_contact_cooler_cooling_tower
 + 1000 E2_direct_contact_cooler_cooling_tower + X_cooling_tower
 - X_direct_contact_cooler - 0.5 l_cooling_tower - 0.5 l_direct_contact_cooler
 >= -969
_C1053: 1000 E1_direct_contact_cooler_cooling_tower
 - 1000 E2_direct_contact_cooler_cooling_tower - Y_cooling_tower
 + Y_direct_contact_cooler - 0.5 d_cooling_tower - 0.5 d_direct_contact_cooler
 >= -969
_C1054: - 1000 E1_direct_contact_cooler_cooling_tower
 - 1000 E2_direct_contact_cooler_cooling_tower + Y_cooling_tower
 - Y_direct_contact_cooler - 0.5 d_cooling_tower - 0.5 d_direct_contact_cooler
 >= -1969
_C1055: L_direct_contact_cooler_cooling_tower
 - R_cooling_tower_direct_contact_cooler = 0
_C1056: - L_cooling_tower_direct_contact_cooler
 + R_direct_contact_cooler_cooling_tower = 0
_C1057: A_direct_contact_cooler_cooling_tower
 - B_cooling_tower_direct_contact_cooler = 0
_C1058: - A_cooling_tower_direct_contact_cooler
 + B_direct_contact_cooler_cooling_tower = 0
_C1059: Wx_cooling_tower_direct_contact_cooler
 + Wx_direct_contact_cooler_cooling_tower = 1
_C106: Wy_absorber_lean_amine_cooler + Wy_lean_amine_cooler_absorber = 1
_C1060: Wy_cooling_tower_direct_contact_cooler
 + Wy_direct_contact_cooler_cooling_tower = 1
_C1061: - L_absorber_cooling_tower + R_absorber_cooling_tower - X_absorber
 + X_cooling_tower = 0
_C1062: A_absorber_cooling_tower - B_absorber_cooling_tower - Y_absorber
 + Y_cooling_tower = 0
_C1063: R_absorber_cooling_tower - 1000 Wx_absorber_cooling_tower <= 0
_C1064: L_absorber_cooling_tower + 1000 Wx_absorber_cooling_tower <= 1000
_C1065: A_absorber_cooling_tower - 1000 Wy_absorber_cooling_tower <= 0
_C1066: B_absorber_cooling_tower + 1000 Wy_absorber_cooling_tower <= 1000
_C1067: - A_absorber_cooling_tower - B_absorber_cooling_tower
 + D_absorber_cooling_tower - L_absorber_cooling_tower
 - R_absorber_cooling_tower = 0
_C1068: D_absorber_cooling_tower - D_cooling_tower_absorber = 0
_C1069: 1000 E1_absorber_cooling_tower + 1000 E2_absorber_cooling_tower
 + X_absorber - X_cooling_tower - 0.5 l_absorber - 0.5 l_cooling_tower >= 31
_C107: - L_direct_contact_cooler_cross_heat_exchanger
 + R_direct_contact_cooler_cross_heat_exchanger + X_cross_heat_exchanger
 - X_direct_contact_cooler = 0
_C1070: - 1000 E1_absorber_cooling_tower + 1000 E2_absorber_cooling_tower
 - X_absorber + X_cooling_tower - 0.5 l_absorber - 0.5 l_cooling_tower >= -969
_C1071: 1000 E1_absorber_cooling_tower - 1000 E2_absorber_cooling_tower
 + Y_absorber - Y_cooling_tower - 0.5 d_absorber - 0.5 d_cooling_tower >= -969
_C1072: - 1000 E1_absorber_cooling_tower - 1000 E2_absorber_cooling_tower
 - Y_absorber + Y_cooling_tower - 0.5 d_absorber - 0.5 d_cooling_tower
 >= -1969
_C1073: L_absorber_cooling_tower - R_cooling_tower_absorber = 0
_C1074: - L_cooling_tower_absorber + R_absorber_cooling_tower = 0
_C1075: A_absorber_cooling_tower - B_cooling_tower_absorber = 0
_C1076: - A_cooling_tower_absorber + B_absorber_cooling_tower = 0
_C1077: Wx_absorber_cooling_tower + Wx_cooling_tower_absorber = 1
_C1078: Wy_absorber_cooling_tower + Wy_cooling_tower_absorber = 1
_C1079: - L_lean_amine_cooler_cooling_tower
 + R_lean_amine_cooler_cooling_tower + X_cooling_tower - X_lean_amine_cooler
 = 0
_C108: A_direct_contact_cooler_cross_heat_exchanger
 - B_direct_contact_cooler_cross_heat_exchanger + Y_cross_heat_exchanger
 - Y_direct_contact_cooler = 0
_C1080: A_lean_amine_cooler_cooling_tower - B_lean_amine_cooler_cooling_tower
 + Y_cooling_tower - Y_lean_amine_cooler = 0
_C1081: R_lean_amine_cooler_cooling_tower
 - 1000 Wx_lean_amine_cooler_cooling_tower <= 0
_C1082: L_lean_amine_cooler_cooling_tower
 + 1000 Wx_lean_amine_cooler_cooling_tower <= 1000
_C1083: A_lean_amine_cooler_cooling_tower
 - 1000 Wy_lean_amine_cooler_cooling_tower <= 0
_C1084: B_lean_amine_cooler_cooling_tower
 + 1000 Wy_lean_amine_cooler_cooling_tower <= 1000
_C1085: - A_lean_amine_cooler_cooling_tower
 - B_lean_amine_cooler_cooling_tower + D_lean_amine_cooler_cooling_tower
 - L_lean_amine_cooler_cooling_tower - R_lean_amine_cooler_cooling_tower = 0
_C1086: - D_cooling_tower_lean_amine_cooler
 + D_lean_amine_cooler_cooling_tower = 0
_C1087: 1000 E1_lean_amine_cooler_cooling_tower
 + 1000 E2_lean_amine_cooler_cooling_tower - X_cooling_tower
 + X_lean_amine_cooler - 0.5 l_cooling_tower - 0.5 l_lean_amine_cooler >= 31
_C1088: - 1000 E1_lean_amine_cooler_cooling_tower
 + 1000 E2_lean_amine_cooler_cooling_tower + X_cooling_tower
 - X_lean_amine_cooler - 0.5 l_cooling_tower - 0.5 l_lean_amine_cooler >= -969
_C1089: 1000 E1_lean_amine_cooler_cooling_tower
 - 1000 E2_lean_amine_cooler_cooling_tower - Y_cooling_tower
 + Y_lean_amine_cooler - 0.5 d_cooling_tower - 0.5 d_lean_amine_cooler >= -969
_C109: R_direct_contact_cooler_cross_heat_exchanger
 - 1000 Wx_direct_contact_cooler_cross_heat_exchanger <= 0
_C1090: - 1000 E1_lean_amine_cooler_cooling_tower
 - 1000 E2_lean_amine_cooler_cooling_tower + Y_cooling_tower
 - Y_lean_amine_cooler - 0.5 d_cooling_tower - 0.5 d_lean_amine_cooler
 >= -1969
_C1091: L_lean_amine_cooler_cooling_tower - R_cooling_tower_lean_amine_cooler
 = 0
_C1092: - L_cooling_tower_lean_amine_cooler
 + R_lean_amine_cooler_cooling_tower = 0
_C1093: A_lean_amine_cooler_cooling_tower - B_cooling_tower_lean_amine_cooler
 = 0
_C1094: - A_cooling_tower_lean_amine_cooler
 + B_lean_amine_cooler_cooling_tower = 0
_C1095: Wx_cooling_tower_lean_amine_cooler
 + Wx_lean_amine_cooler_cooling_tower = 1
_C1096: Wy_cooling_tower_lean_amine_cooler
 + Wy_lean_amine_cooler_cooling_tower = 1
_C1097: - L_cross_heat_exchanger_cooling_tower
 + R_cross_heat_exchanger_cooling_tower + X_cooling_tower
 - X_cross_heat_exchanger = 0
_C1098: A_cross_heat_exchanger_cooling_tower
 - B_cross_heat_exchanger_cooling_tower + Y_cooling_tower
 - Y_cross_heat_exchanger = 0
_C1099: R_cross_heat_exchanger_cooling_tower
 - 1000 Wx_cross_heat_exchanger_cooling_tower <= 0
_C11: X_lean_amine_cooler - 0.5 l_lean_amine_cooler >= 0
_C110: L_direct_contact_cooler_cross_heat_exchanger
 + 1000 Wx_direct_contact_cooler_cross_heat_exchanger <= 1000
_C1100: L_cross_heat_exchanger_cooling_tower
 + 1000 Wx_cross_heat_exchanger_cooling_tower <= 1000
_C1101: A_cross_heat_exchanger_cooling_tower
 - 1000 Wy_cross_heat_exchanger_cooling_tower <= 0
_C1102: B_cross_heat_exchanger_cooling_tower
 + 1000 Wy_cross_heat_exchanger_cooling_tower <= 1000
_C1103: - A_cross_heat_exchanger_cooling_tower
 - B_cross_heat_exchanger_cooling_tower + D_cross_heat_exchanger_cooling_tower
 - L_cross_heat_exchanger_cooling_tower - R_cross_heat_exchanger_cooling_tower
 = 0
_C1104: - D_cooling_tower_cross_heat_exchanger
 + D_cross_heat_exchanger_cooling_tower = 0
_C1105: 1000 E1_cross_heat_exchanger_cooling_tower
 + 1000 E2_cross_heat_exchanger_cooling_tower - X_cooling_tower
 + X_cross_heat_exchanger - 0.5 l_cooling_tower - 0.5 l_cross_heat_exchanger
 >= 31
_C1106: - 1000 E1_cross_heat_exchanger_cooling_tower
 + 1000 E2_cross_heat_exchanger_cooling_tower + X_cooling_tower
 - X_cross_heat_exchanger - 0.5 l_cooling_tower - 0.5 l_cross_heat_exchanger
 >= -969
_C1107: 1000 E1_cross_heat_exchanger_cooling_tower
 - 1000 E2_cross_heat_exchanger_cooling_tower - Y_cooling_tower
 + Y_cross_heat_exchanger - 0.5 d_cooling_tower - 0.5 d_cross_heat_exchanger
 >= -969
_C1108: - 1000 E1_cross_heat_exchanger_cooling_tower
 - 1000 E2_cross_heat_exchanger_cooling_tower + Y_cooling_tower
 - Y_cross_heat_exchanger - 0.5 d_cooling_tower - 0.5 d_cross_heat_exchanger
 >= -1969
_C1109: L_cross_heat_exchanger_cooling_tower
 - R_cooling_tower_cross_heat_exchanger = 0
_C111: A_direct_contact_cooler_cross_heat_exchanger
 - 1000 Wy_direct_contact_cooler_cross_heat_exchanger <= 0
_C1110: - L_cooling_tower_cross_heat_exchanger
 + R_cross_heat_exchanger_cooling_tower = 0
_C1111: A_cross_heat_exchanger_cooling_tower
 - B_cooling_tower_cross_heat_exchanger = 0
_C1112: - A_cooling_tower_cross_heat_exchanger
 + B_cross_heat_exchanger_cooling_tower = 0
_C1113: Wx_cooling_tower_cross_heat_exchanger
 + Wx_cross_heat_exchanger_cooling_tower = 1
_C1114: Wy_cooling_tower_cross_heat_exchanger
 + Wy_cross_heat_exchanger_cooling_tower = 1
_C1115: - L_stripper_cooling_tower + R_stripper_cooling_tower
 + X_cooling_tower - X_stripper = 0
_C1116: A_stripper_cooling_tower - B_stripper_cooling_tower + Y_cooling_tower
 - Y_stripper = 0
_C1117: R_stripper_cooling_tower - 1000 Wx_stripper_cooling_tower <= 0
_C1118: L_stripper_cooling_tower + 1000 Wx_stripper_cooling_tower <= 1000
_C1119: A_stripper_cooling_tower - 1000 Wy_stripper_cooling_tower <= 0
_C112: B_direct_contact_cooler_cross_heat_exchanger
 + 1000 Wy_direct_contact_cooler_cross_heat_exchanger <= 1000
_C1120: B_stripper_cooling_tower + 1000 Wy_stripper_cooling_tower <= 1000
_C1121: - A_stripper_cooling_tower - B_stripper_cooling_tower
 + D_stripper_cooling_tower - L_stripper_cooling_tower
 - R_stripper_cooling_tower = 0
_C1122: - D_cooling_tower_stripper + D_stripper_cooling_tower = 0
_C1123: 1000 E1_stripper_cooling_tower + 1000 E2_stripper_cooling_tower
 - X_cooling_tower + X_stripper - 0.5 l_cooling_tower - 0.5 l_stripper >= 31
_C1124: - 1000 E1_stripper_cooling_tower + 1000 E2_stripper_cooling_tower
 + X_cooling_tower - X_stripper - 0.5 l_cooling_tower - 0.5 l_stripper >= -969
_C1125: 1000 E1_stripper_cooling_tower - 1000 E2_stripper_cooling_tower
 - Y_cooling_tower + Y_stripper - 0.5 d_cooling_tower - 0.5 d_stripper >= -969
_C1126: - 1000 E1_stripper_cooling_tower - 1000 E2_stripper_cooling_tower
 + Y_cooling_tower - Y_stripper - 0.5 d_cooling_tower - 0.5 d_stripper
 >= -1969
_C1127: L_stripper_cooling_tower - R_cooling_tower_stripper = 0
_C1128: - L_cooling_tower_stripper + R_stripper_cooling_tower = 0
_C1129: A_stripper_cooling_tower - B_cooling_tower_stripper = 0
_C113: - A_direct_contact_cooler_cross_heat_exchanger
 - B_direct_contact_cooler_cross_heat_exchanger
 + D_direct_contact_cooler_cross_heat_exchanger
 - L_direct_contact_cooler_cross_heat_exchanger
 - R_direct_contact_cooler_cross_heat_exchanger = 0
_C1130: - A_cooling_tower_stripper + B_stripper_cooling_tower = 0
_C1131: Wx_cooling_tower_stripper + Wx_stripper_cooling_tower = 1
_C1132: Wy_cooling_tower_stripper + Wy_stripper_cooling_tower = 1
_C1133: - L_storage_tank_cooling_tower + R_storage_tank_cooling_tower
 + X_cooling_tower - X_storage_tank = 0
_C1134: A_storage_tank_cooling_tower - B_storage_tank_cooling_tower
 + Y_cooling_tower - Y_storage_tank = 0
_C1135: R_storage_tank_cooling_tower - 1000 Wx_storage_tank_cooling_tower <= 0
_C1136: L_storage_tank_cooling_tower + 1000 Wx_storage_tank_cooling_tower
 <= 1000
_C1137: A_storage_tank_cooling_tower - 1000 Wy_storage_tank_cooling_tower <= 0
_C1138: B_storage_tank_cooling_tower + 1000 Wy_storage_tank_cooling_tower
 <= 1000
_C1139: - A_storage_tank_cooling_tower - B_storage_tank_cooling_tower
 + D_storage_tank_cooling_tower - L_storage_tank_cooling_tower
 - R_storage_tank_cooling_tower = 0
_C114: - D_cross_heat_exchanger_direct_contact_cooler
 + D_direct_contact_cooler_cross_heat_exchanger = 0
_C1140: - D_cooling_tower_storage_tank + D_storage_tank_cooling_tower = 0
_C1141: 1000 E1_storage_tank_cooling_tower
 + 1000 E2_storage_tank_cooling_tower - X_cooling_tower + X_storage_tank
 - 0.5 l_cooling_tower - 0.5 l_storage_tank >= 76
_C1142: - 1000 E1_storage_tank_cooling_tower
 + 1000 E2_storage_tank_cooling_tower + X_cooling_tower - X_storage_tank
 - 0.5 l_cooling_tower - 0.5 l_storage_tank >= -924
_C1143: 1000 E1_storage_tank_cooling_tower
 - 1000 E2_storage_tank_cooling_tower - Y_cooling_tower + Y_storage_tank
 - 0.5 d_cooling_tower - 0.5 d_storage_tank >= -924
_C1144: - 1000 E1_storage_tank_cooling_tower
 - 1000 E2_storage_tank_cooling_tower + Y_cooling_tower - Y_storage_tank
 - 0.5 d_cooling_tower - 0.5 d_storage_tank >= -1924
_C1145: L_storage_tank_cooling_tower - R_cooling_tower_storage_tank = 0
_C1146: - L_cooling_tower_storage_tank + R_storage_tank_cooling_tower = 0
_C1147: A_storage_tank_cooling_tower - B_cooling_tower_storage_tank = 0
_C1148: - A_cooling_tower_storage_tank + B_storage_tank_cooling_tower = 0
_C1149: Wx_cooling_tower_storage_tank + Wx_storage_tank_cooling_tower = 1
_C115: 1000 E1_direct_contact_cooler_cross_heat_exchanger
 + 1000 E2_direct_contact_cooler_cross_heat_exchanger - X_cross_heat_exchanger
 + X_direct_contact_cooler - 0.5 l_cross_heat_exchanger
 - 0.5 l_direct_contact_cooler >= 15
_C1150: Wy_cooling_tower_storage_tank + Wy_storage_tank_cooling_tower = 1
_C1151: - L_reclaimer_cooling_tower + R_reclaimer_cooling_tower
 + X_cooling_tower - X_reclaimer = 0
_C1152: A_reclaimer_cooling_tower - B_reclaimer_cooling_tower
 + Y_cooling_tower - Y_reclaimer = 0
_C1153: R_reclaimer_cooling_tower - 1000 Wx_reclaimer_cooling_tower <= 0
_C1154: L_reclaimer_cooling_tower + 1000 Wx_reclaimer_cooling_tower <= 1000
_C1155: A_reclaimer_cooling_tower - 1000 Wy_reclaimer_cooling_tower <= 0
_C1156: B_reclaimer_cooling_tower + 1000 Wy_reclaimer_cooling_tower <= 1000
_C1157: - A_reclaimer_cooling_tower - B_reclaimer_cooling_tower
 + D_reclaimer_cooling_tower - L_reclaimer_cooling_tower
 - R_reclaimer_cooling_tower = 0
_C1158: - D_cooling_tower_reclaimer + D_reclaimer_cooling_tower = 0
_C1159: 1000 E1_reclaimer_cooling_tower + 1000 E2_reclaimer_cooling_tower
 - X_cooling_tower + X_reclaimer - 0.5 l_cooling_tower - 0.5 l_reclaimer >= 31
_C116: - 1000 E1_direct_contact_cooler_cross_heat_exchanger
 + 1000 E2_direct_contact_cooler_cross_heat_exchanger + X_cross_heat_exchanger
 - X_direct_contact_cooler - 0.5 l_cross_heat_exchanger
 - 0.5 l_direct_contact_cooler >= -985
_C1160: - 1000 E1_reclaimer_cooling_tower + 1000 E2_reclaimer_cooling_tower
 + X_cooling_tower - X_reclaimer - 0.5 l_cooling_tower - 0.5 l_reclaimer
 >= -969
_C1161: 1000 E1_reclaimer_cooling_tower - 1000 E2_reclaimer_cooling_tower
 - Y_cooling_tower + Y_reclaimer - 0.5 d_cooling_tower - 0.5 d_reclaimer
 >= -969
_C1162: - 1000 E1_reclaimer_cooling_tower - 1000 E2_reclaimer_cooling_tower
 + Y_cooling_tower - Y_reclaimer - 0.5 d_cooling_tower - 0.5 d_reclaimer
 >= -1969
_C1163: L_reclaimer_cooling_tower - R_cooling_tower_reclaimer = 0
_C1164: - L_cooling_tower_reclaimer + R_reclaimer_cooling_tower = 0
_C1165: A_reclaimer_cooling_tower - B_cooling_tower_reclaimer = 0
_C1166: - A_cooling_tower_reclaimer + B_reclaimer_cooling_tower = 0
_C1167: Wx_cooling_tower_reclaimer + Wx_reclaimer_cooling_tower = 1
_C1168: Wy_cooling_tower_reclaimer + Wy_reclaimer_cooling_tower = 1
_C1169: - L_compressor_house_cooling_tower + R_compressor_house_cooling_tower
 - X_compressor_house + X_cooling_tower = 0
_C117: 1000 E1_direct_contact_cooler_cross_heat_exchanger
 - 1000 E2_direct_contact_cooler_cross_heat_exchanger - Y_cross_heat_exchanger
 + Y_direct_contact_cooler - 0.5 d_cross_heat_exchanger
 - 0.5 d_direct_contact_cooler >= -985
_C1170: A_compressor_house_cooling_tower - B_compressor_house_cooling_tower
 - Y_compressor_house + Y_cooling_tower = 0
_C1171: R_compressor_house_cooling_tower
 - 1000 Wx_compressor_house_cooling_tower <= 0
_C1172: L_compressor_house_cooling_tower
 + 1000 Wx_compressor_house_cooling_tower <= 1000
_C1173: A_compressor_house_cooling_tower
 - 1000 Wy_compressor_house_cooling_tower <= 0
_C1174: B_compressor_house_cooling_tower
 + 1000 Wy_compressor_house_cooling_tower <= 1000
_C1175: - A_compressor_house_cooling_tower - B_compressor_house_cooling_tower
 + D_compressor_house_cooling_tower - L_compressor_house_cooling_tower
 - R_compressor_house_cooling_tower = 0
_C1176: D_compressor_house_cooling_tower - D_cooling_tower_compressor_house
 = 0
_C1177: 1000 E1_compressor_house_cooling_tower
 + 1000 E2_compressor_house_cooling_tower + X_compressor_house
 - X_cooling_tower - 0.5 l_compressor_house - 0.5 l_cooling_tower >= 31
_C1178: - 1000 E1_compressor_house_cooling_tower
 + 1000 E2_compressor_house_cooling_tower - X_compressor_house
 + X_cooling_tower - 0.5 l_compressor_house - 0.5 l_cooling_tower >= -969
_C1179: 1000 E1_compressor_house_cooling_tower
 - 1000 E2_compressor_house_cooling_tower + Y_compressor_house
 - Y_cooling_tower - 0.5 d_compressor_house - 0.5 d_cooling_tower >= -969
_C118: - 1000 E1_direct_contact_cooler_cross_heat_exchanger
 - 1000 E2_direct_contact_cooler_cross_heat_exchanger + Y_cross_heat_exchanger
 - Y_direct_contact_cooler - 0.5 d_cross_heat_exchanger
 - 0.5 d_direct_contact_cooler >= -1985
_C1180: - 1000 E1_compressor_house_cooling_tower
 - 1000 E2_compressor_house_cooling_tower - Y_compressor_house
 + Y_cooling_tower - 0.5 d_compressor_house - 0.5 d_cooling_tower >= -1969
_C1181: L_compressor_house_cooling_tower - R_cooling_tower_compressor_house
 = 0
_C1182: - L_cooling_tower_compressor_house + R_compressor_house_cooling_tower
 = 0
_C1183: A_compressor_house_cooling_tower - B_cooling_tower_compressor_house
 = 0
_C1184: - A_cooling_tower_compressor_house + B_compressor_house_cooling_tower
 = 0
_C1185: Wx_compressor_house_cooling_tower + Wx_cooling_tower_compressor_house
 = 1
_C1186: Wy_compressor_house_cooling_tower + Wy_cooling_tower_compressor_house
 = 1
_C1187: - L_compressor_KOdrum_cooling_tower
 + R_compressor_KOdrum_cooling_tower - X_compressor_KOdrum + X_cooling_tower
 = 0
_C1188: A_compressor_KOdrum_cooling_tower - B_compressor_KOdrum_cooling_tower
 - Y_compressor_KOdrum + Y_cooling_tower = 0
_C1189: R_compressor_KOdrum_cooling_tower
 - 1000 Wx_compressor_KOdrum_cooling_tower <= 0
_C119: L_direct_contact_cooler_cross_heat_exchanger
 - R_cross_heat_exchanger_direct_contact_cooler = 0
_C1190: L_compressor_KOdrum_cooling_tower
 + 1000 Wx_compressor_KOdrum_cooling_tower <= 1000
_C1191: A_compressor_KOdrum_cooling_tower
 - 1000 Wy_compressor_KOdrum_cooling_tower <= 0
_C1192: B_compressor_KOdrum_cooling_tower
 + 1000 Wy_compressor_KOdrum_cooling_tower <= 1000
_C1193: - A_compressor_KOdrum_cooling_tower
 - B_compressor_KOdrum_cooling_tower + D_compressor_KOdrum_cooling_tower
 - L_compressor_KOdrum_cooling_tower - R_compressor_KOdrum_cooling_tower = 0
_C1194: D_compressor_KOdrum_cooling_tower - D_cooling_tower_compressor_KOdrum
 = 0
_C1195: 1000 E1_compressor_KOdrum_cooling_tower
 + 1000 E2_compressor_KOdrum_cooling_tower + X_compressor_KOdrum
 - X_cooling_tower - 0.5 l_compressor_KOdrum - 0.5 l_cooling_tower >= 31
_C1196: - 1000 E1_compressor_KOdrum_cooling_tower
 + 1000 E2_compressor_KOdrum_cooling_tower - X_compressor_KOdrum
 + X_cooling_tower - 0.5 l_compressor_KOdrum - 0.5 l_cooling_tower >= -969
_C1197: 1000 E1_compressor_KOdrum_cooling_tower
 - 1000 E2_compressor_KOdrum_cooling_tower + Y_compressor_KOdrum
 - Y_cooling_tower - 0.5 d_compressor_KOdrum - 0.5 d_cooling_tower >= -969
_C1198: - 1000 E1_compressor_KOdrum_cooling_tower
 - 1000 E2_compressor_KOdrum_cooling_tower - Y_compressor_KOdrum
 + Y_cooling_tower - 0.5 d_compressor_KOdrum - 0.5 d_cooling_tower >= -1969
_C1199: L_compressor_KOdrum_cooling_tower - R_cooling_tower_compressor_KOdrum
 = 0
_C12: Y_lean_amine_cooler - 0.5 d_lean_amine_cooler >= 0
_C120: - L_cross_heat_exchanger_direct_contact_cooler
 + R_direct_contact_cooler_cross_heat_exchanger = 0
_C1200: - L_cooling_tower_compressor_KOdrum
 + R_compressor_KOdrum_cooling_tower = 0
_C1201: A_compressor_KOdrum_cooling_tower - B_cooling_tower_compressor_KOdrum
 = 0
_C1202: - A_cooling_tower_compressor_KOdrum
 + B_compressor_KOdrum_cooling_tower = 0
_C1203: Wx_compressor_KOdrum_cooling_tower
 + Wx_cooling_tower_compressor_KOdrum = 1
_C1204: Wy_compressor_KOdrum_cooling_tower
 + Wy_cooling_tower_compressor_KOdrum = 1
_C1205: - L_compressor_interstage_cooler_cooling_tower
 + R_compressor_interstage_cooler_cooling_tower
 - X_compressor_interstage_cooler + X_cooling_tower = 0
_C1206: A_compressor_interstage_cooler_cooling_tower
 - B_compressor_interstage_cooler_cooling_tower
 - Y_compressor_interstage_cooler + Y_cooling_tower = 0
_C1207: R_compressor_interstage_cooler_cooling_tower
 - 1000 Wx_compressor_interstage_cooler_cooling_tower <= 0
_C1208: L_compressor_interstage_cooler_cooling_tower
 + 1000 Wx_compressor_interstage_cooler_cooling_tower <= 1000
_C1209: A_compressor_interstage_cooler_cooling_tower
 - 1000 Wy_compressor_interstage_cooler_cooling_tower <= 0
_C121: A_direct_contact_cooler_cross_heat_exchanger
 - B_cross_heat_exchanger_direct_contact_cooler = 0
_C1210: B_compressor_interstage_cooler_cooling_tower
 + 1000 Wy_compressor_interstage_cooler_cooling_tower <= 1000
_C1211: - A_compressor_interstage_cooler_cooling_tower
 - B_compressor_interstage_cooler_cooling_tower
 + D_compressor_interstage_cooler_cooling_tower
 - L_compressor_interstage_cooler_cooling_tower
 - R_compressor_interstage_cooler_cooling_tower = 0
_C1212: D_compressor_interstage_cooler_cooling_tower
 - D_cooling_tower_compressor_interstage_cooler = 0
_C1213: 1000 E1_compressor_interstage_cooler_cooling_tower
 + 1000 E2_compressor_interstage_cooler_cooling_tower
 + X_compressor_interstage_cooler - X_cooling_tower
 - 0.5 l_compressor_interstage_cooler - 0.5 l_cooling_tower >= 31
_C1214: - 1000 E1_compressor_interstage_cooler_cooling_tower
 + 1000 E2_compressor_interstage_cooler_cooling_tower
 - X_compressor_interstage_cooler + X_cooling_tower
 - 0.5 l_compressor_interstage_cooler - 0.5 l_cooling_tower >= -969
_C1215: 1000 E1_compressor_interstage_cooler_cooling_tower
 - 1000 E2_compressor_interstage_cooler_cooling_tower
 + Y_compressor_interstage_cooler - Y_cooling_tower
 - 0.5 d_compressor_interstage_cooler - 0.5 d_cooling_tower >= -969
_C1216: - 1000 E1_compressor_interstage_cooler_cooling_tower
 - 1000 E2_compressor_interstage_cooler_cooling_tower
 - Y_compressor_interstage_cooler + Y_cooling_tower
 - 0.5 d_compressor_interstage_cooler - 0.5 d_cooling_tower >= -1969
_C1217: L_compressor_interstage_cooler_cooling_tower
 - R_cooling_tower_compressor_interstage_cooler = 0
_C1218: - L_cooling_tower_compressor_interstage_cooler
 + R_compressor_interstage_cooler_cooling_tower = 0
_C1219: A_compressor_interstage_cooler_cooling_tower
 - B_cooling_tower_compressor_interstage_cooler = 0
_C122: - A_cross_heat_exchanger_direct_contact_cooler
 + B_direct_contact_cooler_cross_heat_exchanger = 0
_C1220: - A_cooling_tower_compressor_interstage_cooler
 + B_compressor_interstage_cooler_cooling_tower = 0
_C1221: Wx_compressor_interstage_cooler_cooling_tower
 + Wx_cooling_tower_compressor_interstage_cooler = 1
_C1222: Wy_compressor_interstage_cooler_cooling_tower
 + Wy_cooling_tower_compressor_interstage_cooler = 1
_C1223: - L_TEG_dehydration_cooling_tower + R_TEG_dehydration_cooling_tower
 - X_TEG_dehydration + X_cooling_tower = 0
_C1224: A_TEG_dehydration_cooling_tower - B_TEG_dehydration_cooling_tower
 - Y_TEG_dehydration + Y_cooling_tower = 0
_C1225: R_TEG_dehydration_cooling_tower
 - 1000 Wx_TEG_dehydration_cooling_tower <= 0
_C1226: L_TEG_dehydration_cooling_tower
 + 1000 Wx_TEG_dehydration_cooling_tower <= 1000
_C1227: A_TEG_dehydration_cooling_tower
 - 1000 Wy_TEG_dehydration_cooling_tower <= 0
_C1228: B_TEG_dehydration_cooling_tower
 + 1000 Wy_TEG_dehydration_cooling_tower <= 1000
_C1229: - A_TEG_dehydration_cooling_tower - B_TEG_dehydration_cooling_tower
 + D_TEG_dehydration_cooling_tower - L_TEG_dehydration_cooling_tower
 - R_TEG_dehydration_cooling_tower = 0
_C123: Wx_cross_heat_exchanger_direct_contact_cooler
 + Wx_direct_contact_cooler_cross_heat_exchanger = 1
_C1230: D_TEG_dehydration_cooling_tower - D_cooling_tower_TEG_dehydration = 0
_C1231: 1000 E1_TEG_dehydration_cooling_tower
 + 1000 E2_TEG_dehydration_cooling_tower + X_TEG_dehydration - X_cooling_tower
 - 0.5 l_TEG_dehydration - 0.5 l_cooling_tower >= 31
_C1232: - 1000 E1_TEG_dehydration_cooling_tower
 + 1000 E2_TEG_dehydration_cooling_tower - X_TEG_dehydration + X_cooling_tower
 - 0.5 l_TEG_dehydration - 0.5 l_cooling_tower >= -969
_C1233: 1000 E1_TEG_dehydration_cooling_tower
 - 1000 E2_TEG_dehydration_cooling_tower + Y_TEG_dehydration - Y_cooling_tower
 - 0.5 d_TEG_dehydration - 0.5 d_cooling_tower >= -969
_C1234: - 1000 E1_TEG_dehydration_cooling_tower
 - 1000 E2_TEG_dehydration_cooling_tower - Y_TEG_dehydration + Y_cooling_tower
 - 0.5 d_TEG_dehydration - 0.5 d_cooling_tower >= -1969
_C1235: L_TEG_dehydration_cooling_tower - R_cooling_tower_TEG_dehydration = 0
_C1236: - L_cooling_tower_TEG_dehydration + R_TEG_dehydration_cooling_tower
 = 0
_C1237: A_TEG_dehydration_cooling_tower - B_cooling_tower_TEG_dehydration = 0
_C1238: - A_cooling_tower_TEG_dehydration + B_TEG_dehydration_cooling_tower
 = 0
_C1239: Wx_TEG_dehydration_cooling_tower + Wx_cooling_tower_TEG_dehydration
 = 1
_C124: Wy_cross_heat_exchanger_direct_contact_cooler
 + Wy_direct_contact_cooler_cross_heat_exchanger = 1
_C1240: Wy_TEG_dehydration_cooling_tower + Wy_cooling_tower_TEG_dehydration
 = 1
_C1241: - L_direct_contact_cooler_coolingwater_pumphouse
 + R_direct_contact_cooler_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_direct_contact_cooler = 0
_C1242: A_direct_contact_cooler_coolingwater_pumphouse
 - B_direct_contact_cooler_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_direct_contact_cooler = 0
_C1243: R_direct_contact_cooler_coolingwater_pumphouse
 - 1000 Wx_direct_contact_cooler_coolingwater_pumphouse <= 0
_C1244: L_direct_contact_cooler_coolingwater_pumphouse
 + 1000 Wx_direct_contact_cooler_coolingwater_pumphouse <= 1000
_C1245: A_direct_contact_cooler_coolingwater_pumphouse
 - 1000 Wy_direct_contact_cooler_coolingwater_pumphouse <= 0
_C1246: B_direct_contact_cooler_coolingwater_pumphouse
 + 1000 Wy_direct_contact_cooler_coolingwater_pumphouse <= 1000
_C1247: - A_direct_contact_cooler_coolingwater_pumphouse
 - B_direct_contact_cooler_coolingwater_pumphouse
 + D_direct_contact_cooler_coolingwater_pumphouse
 - L_direct_contact_cooler_coolingwater_pumphouse
 - R_direct_contact_cooler_coolingwater_pumphouse = 0
_C1248: - D_coolingwater_pumphouse_direct_contact_cooler
 + D_direct_contact_cooler_coolingwater_pumphouse = 0
_C1249: 1000 E1_direct_contact_cooler_coolingwater_pumphouse
 + 1000 E2_direct_contact_cooler_coolingwater_pumphouse
 - X_coolingwater_pumphouse + X_direct_contact_cooler
 - 0.5 l_coolingwater_pumphouse - 0.5 l_direct_contact_cooler >= 9
_C125: - L_absorber_cross_heat_exchanger + R_absorber_cross_heat_exchanger
 - X_absorber + X_cross_heat_exchanger = 0
_C1250: - 1000 E1_direct_contact_cooler_coolingwater_pumphouse
 + 1000 E2_direct_contact_cooler_coolingwater_pumphouse
 + X_coolingwater_pumphouse - X_direct_contact_cooler
 - 0.5 l_coolingwater_pumphouse - 0.5 l_direct_contact_cooler >= -991
_C1251: 1000 E1_direct_contact_cooler_coolingwater_pumphouse
 - 1000 E2_direct_contact_cooler_coolingwater_pumphouse
 - Y_coolingwater_pumphouse + Y_direct_contact_cooler
 - 0.5 d_coolingwater_pumphouse - 0.5 d_direct_contact_cooler >= -991
_C1252: - 1000 E1_direct_contact_cooler_coolingwater_pumphouse
 - 1000 E2_direct_contact_cooler_coolingwater_pumphouse
 + Y_coolingwater_pumphouse - Y_direct_contact_cooler
 - 0.5 d_coolingwater_pumphouse - 0.5 d_direct_contact_cooler >= -1991
_C1253: L_direct_contact_cooler_coolingwater_pumphouse
 - R_coolingwater_pumphouse_direct_contact_cooler = 0
_C1254: - L_coolingwater_pumphouse_direct_contact_cooler
 + R_direct_contact_cooler_coolingwater_pumphouse = 0
_C1255: A_direct_contact_cooler_coolingwater_pumphouse
 - B_coolingwater_pumphouse_direct_contact_cooler = 0
_C1256: - A_coolingwater_pumphouse_direct_contact_cooler
 + B_direct_contact_cooler_coolingwater_pumphouse = 0
_C1257: Wx_coolingwater_pumphouse_direct_contact_cooler
 + Wx_direct_contact_cooler_coolingwater_pumphouse = 1
_C1258: Wy_coolingwater_pumphouse_direct_contact_cooler
 + Wy_direct_contact_cooler_coolingwater_pumphouse = 1
_C1259: - L_absorber_coolingwater_pumphouse
 + R_absorber_coolingwater_pumphouse - X_absorber + X_coolingwater_pumphouse
 = 0
_C126: A_absorber_cross_heat_exchanger - B_absorber_cross_heat_exchanger
 - Y_absorber + Y_cross_heat_exchanger = 0
_C1260: A_absorber_coolingwater_pumphouse - B_absorber_coolingwater_pumphouse
 - Y_absorber + Y_coolingwater_pumphouse = 0
_C1261: R_absorber_coolingwater_pumphouse
 - 1000 Wx_absorber_coolingwater_pumphouse <= 0
_C1262: L_absorber_coolingwater_pumphouse
 + 1000 Wx_absorber_coolingwater_pumphouse <= 1000
_C1263: A_absorber_coolingwater_pumphouse
 - 1000 Wy_absorber_coolingwater_pumphouse <= 0
_C1264: B_absorber_coolingwater_pumphouse
 + 1000 Wy_absorber_coolingwater_pumphouse <= 1000
_C1265: - A_absorber_coolingwater_pumphouse
 - B_absorber_coolingwater_pumphouse + D_absorber_coolingwater_pumphouse
 - L_absorber_coolingwater_pumphouse - R_absorber_coolingwater_pumphouse = 0
_C1266: D_absorber_coolingwater_pumphouse - D_coolingwater_pumphouse_absorber
 = 0
_C1267: 1000 E1_absorber_coolingwater_pumphouse
 + 1000 E2_absorber_coolingwater_pumphouse + X_absorber
 - X_coolingwater_pumphouse - 0.5 l_absorber - 0.5 l_coolingwater_pumphouse
 >= 9
_C1268: - 1000 E1_absorber_coolingwater_pumphouse
 + 1000 E2_absorber_coolingwater_pumphouse - X_absorber
 + X_coolingwater_pumphouse - 0.5 l_absorber - 0.5 l_coolingwater_pumphouse
 >= -991
_C1269: 1000 E1_absorber_coolingwater_pumphouse
 - 1000 E2_absorber_coolingwater_pumphouse + Y_absorber
 - Y_coolingwater_pumphouse - 0.5 d_absorber - 0.5 d_coolingwater_pumphouse
 >= -991
_C127: R_absorber_cross_heat_exchanger - 1000 Wx_absorber_cross_heat_exchanger
 <= 0
_C1270: - 1000 E1_absorber_coolingwater_pumphouse
 - 1000 E2_absorber_coolingwater_pumphouse - Y_absorber
 + Y_coolingwater_pumphouse - 0.5 d_absorber - 0.5 d_coolingwater_pumphouse
 >= -1991
_C1271: L_absorber_coolingwater_pumphouse - R_coolingwater_pumphouse_absorber
 = 0
_C1272: - L_coolingwater_pumphouse_absorber
 + R_absorber_coolingwater_pumphouse = 0
_C1273: A_absorber_coolingwater_pumphouse - B_coolingwater_pumphouse_absorber
 = 0
_C1274: - A_coolingwater_pumphouse_absorber
 + B_absorber_coolingwater_pumphouse = 0
_C1275: Wx_absorber_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_absorber = 1
_C1276: Wy_absorber_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_absorber = 1
_C1277: - L_lean_amine_cooler_coolingwater_pumphouse
 + R_lean_amine_cooler_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_lean_amine_cooler = 0
_C1278: A_lean_amine_cooler_coolingwater_pumphouse
 - B_lean_amine_cooler_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_lean_amine_cooler = 0
_C1279: R_lean_amine_cooler_coolingwater_pumphouse
 - 1000 Wx_lean_amine_cooler_coolingwater_pumphouse <= 0
_C128: L_absorber_cross_heat_exchanger + 1000 Wx_absorber_cross_heat_exchanger
 <= 1000
_C1280: L_lean_amine_cooler_coolingwater_pumphouse
 + 1000 Wx_lean_amine_cooler_coolingwater_pumphouse <= 1000
_C1281: A_lean_amine_cooler_coolingwater_pumphouse
 - 1000 Wy_lean_amine_cooler_coolingwater_pumphouse <= 0
_C1282: B_lean_amine_cooler_coolingwater_pumphouse
 + 1000 Wy_lean_amine_cooler_coolingwater_pumphouse <= 1000
_C1283: - A_lean_amine_cooler_coolingwater_pumphouse
 - B_lean_amine_cooler_coolingwater_pumphouse
 + D_lean_amine_cooler_coolingwater_pumphouse
 - L_lean_amine_cooler_coolingwater_pumphouse
 - R_lean_amine_cooler_coolingwater_pumphouse = 0
_C1284: - D_coolingwater_pumphouse_lean_amine_cooler
 + D_lean_amine_cooler_coolingwater_pumphouse = 0
_C1285: 1000 E1_lean_amine_cooler_coolingwater_pumphouse
 + 1000 E2_lean_amine_cooler_coolingwater_pumphouse - X_coolingwater_pumphouse
 + X_lean_amine_cooler - 0.5 l_coolingwater_pumphouse
 - 0.5 l_lean_amine_cooler >= 9
_C1286: - 1000 E1_lean_amine_cooler_coolingwater_pumphouse
 + 1000 E2_lean_amine_cooler_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_lean_amine_cooler - 0.5 l_coolingwater_pumphouse
 - 0.5 l_lean_amine_cooler >= -991
_C1287: 1000 E1_lean_amine_cooler_coolingwater_pumphouse
 - 1000 E2_lean_amine_cooler_coolingwater_pumphouse - Y_coolingwater_pumphouse
 + Y_lean_amine_cooler - 0.5 d_coolingwater_pumphouse
 - 0.5 d_lean_amine_cooler >= -991
_C1288: - 1000 E1_lean_amine_cooler_coolingwater_pumphouse
 - 1000 E2_lean_amine_cooler_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_lean_amine_cooler - 0.5 d_coolingwater_pumphouse
 - 0.5 d_lean_amine_cooler >= -1991
_C1289: L_lean_amine_cooler_coolingwater_pumphouse
 - R_coolingwater_pumphouse_lean_amine_cooler = 0
_C129: A_absorber_cross_heat_exchanger - 1000 Wy_absorber_cross_heat_exchanger
 <= 0
_C1290: - L_coolingwater_pumphouse_lean_amine_cooler
 + R_lean_amine_cooler_coolingwater_pumphouse = 0
_C1291: A_lean_amine_cooler_coolingwater_pumphouse
 - B_coolingwater_pumphouse_lean_amine_cooler = 0
_C1292: - A_coolingwater_pumphouse_lean_amine_cooler
 + B_lean_amine_cooler_coolingwater_pumphouse = 0
_C1293: Wx_coolingwater_pumphouse_lean_amine_cooler
 + Wx_lean_amine_cooler_coolingwater_pumphouse = 1
_C1294: Wy_coolingwater_pumphouse_lean_amine_cooler
 + Wy_lean_amine_cooler_coolingwater_pumphouse = 1
_C1295: - L_cross_heat_exchanger_coolingwater_pumphouse
 + R_cross_heat_exchanger_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_cross_heat_exchanger = 0
_C1296: A_cross_heat_exchanger_coolingwater_pumphouse
 - B_cross_heat_exchanger_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_cross_heat_exchanger = 0
_C1297: R_cross_heat_exchanger_coolingwater_pumphouse
 - 1000 Wx_cross_heat_exchanger_coolingwater_pumphouse <= 0
_C1298: L_cross_heat_exchanger_coolingwater_pumphouse
 + 1000 Wx_cross_heat_exchanger_coolingwater_pumphouse <= 1000
_C1299: A_cross_heat_exchanger_coolingwater_pumphouse
 - 1000 Wy_cross_heat_exchanger_coolingwater_pumphouse <= 0
_C13: 0 O_cross_heat_exchanger + l_cross_heat_exchanger = 16
_C130: B_absorber_cross_heat_exchanger + 1000 Wy_absorber_cross_heat_exchanger
 <= 1000
_C1300: B_cross_heat_exchanger_coolingwater_pumphouse
 + 1000 Wy_cross_heat_exchanger_coolingwater_pumphouse <= 1000
_C1301: - A_cross_heat_exchanger_coolingwater_pumphouse
 - B_cross_heat_exchanger_coolingwater_pumphouse
 + D_cross_heat_exchanger_coolingwater_pumphouse
 - L_cross_heat_exchanger_coolingwater_pumphouse
 - R_cross_heat_exchanger_coolingwater_pumphouse = 0
_C1302: - D_coolingwater_pumphouse_cross_heat_exchanger
 + D_cross_heat_exchanger_coolingwater_pumphouse = 0
_C1303: 1000 E1_cross_heat_exchanger_coolingwater_pumphouse
 + 1000 E2_cross_heat_exchanger_coolingwater_pumphouse
 - X_coolingwater_pumphouse + X_cross_heat_exchanger
 - 0.5 l_coolingwater_pumphouse - 0.5 l_cross_heat_exchanger >= 9
_C1304: - 1000 E1_cross_heat_exchanger_coolingwater_pumphouse
 + 1000 E2_cross_heat_exchanger_coolingwater_pumphouse
 + X_coolingwater_pumphouse - X_cross_heat_exchanger
 - 0.5 l_coolingwater_pumphouse - 0.5 l_cross_heat_exchanger >= -991
_C1305: 1000 E1_cross_heat_exchanger_coolingwater_pumphouse
 - 1000 E2_cross_heat_exchanger_coolingwater_pumphouse
 - Y_coolingwater_pumphouse + Y_cross_heat_exchanger
 - 0.5 d_coolingwater_pumphouse - 0.5 d_cross_heat_exchanger >= -991
_C1306: - 1000 E1_cross_heat_exchanger_coolingwater_pumphouse
 - 1000 E2_cross_heat_exchanger_coolingwater_pumphouse
 + Y_coolingwater_pumphouse - Y_cross_heat_exchanger
 - 0.5 d_coolingwater_pumphouse - 0.5 d_cross_heat_exchanger >= -1991
_C1307: L_cross_heat_exchanger_coolingwater_pumphouse
 - R_coolingwater_pumphouse_cross_heat_exchanger = 0
_C1308: - L_coolingwater_pumphouse_cross_heat_exchanger
 + R_cross_heat_exchanger_coolingwater_pumphouse = 0
_C1309: A_cross_heat_exchanger_coolingwater_pumphouse
 - B_coolingwater_pumphouse_cross_heat_exchanger = 0
_C131: - A_absorber_cross_heat_exchanger - B_absorber_cross_heat_exchanger
 + D_absorber_cross_heat_exchanger - L_absorber_cross_heat_exchanger
 - R_absorber_cross_heat_exchanger = 0
_C1310: - A_coolingwater_pumphouse_cross_heat_exchanger
 + B_cross_heat_exchanger_coolingwater_pumphouse = 0
_C1311: Wx_coolingwater_pumphouse_cross_heat_exchanger
 + Wx_cross_heat_exchanger_coolingwater_pumphouse = 1
_C1312: Wy_coolingwater_pumphouse_cross_heat_exchanger
 + Wy_cross_heat_exchanger_coolingwater_pumphouse = 1
_C1313: - L_stripper_coolingwater_pumphouse
 + R_stripper_coolingwater_pumphouse + X_coolingwater_pumphouse - X_stripper
 = 0
_C1314: A_stripper_coolingwater_pumphouse - B_stripper_coolingwater_pumphouse
 + Y_coolingwater_pumphouse - Y_stripper = 0
_C1315: R_stripper_coolingwater_pumphouse
 - 1000 Wx_stripper_coolingwater_pumphouse <= 0
_C1316: L_stripper_coolingwater_pumphouse
 + 1000 Wx_stripper_coolingwater_pumphouse <= 1000
_C1317: A_stripper_coolingwater_pumphouse
 - 1000 Wy_stripper_coolingwater_pumphouse <= 0
_C1318: B_stripper_coolingwater_pumphouse
 + 1000 Wy_stripper_coolingwater_pumphouse <= 1000
_C1319: - A_stripper_coolingwater_pumphouse
 - B_stripper_coolingwater_pumphouse + D_stripper_coolingwater_pumphouse
 - L_stripper_coolingwater_pumphouse - R_stripper_coolingwater_pumphouse = 0
_C132: D_absorber_cross_heat_exchanger - D_cross_heat_exchanger_absorber = 0
_C1320: - D_coolingwater_pumphouse_stripper
 + D_stripper_coolingwater_pumphouse = 0
_C1321: 1000 E1_stripper_coolingwater_pumphouse
 + 1000 E2_stripper_coolingwater_pumphouse - X_coolingwater_pumphouse
 + X_stripper - 0.5 l_coolingwater_pumphouse - 0.5 l_stripper >= 9
_C1322: - 1000 E1_stripper_coolingwater_pumphouse
 + 1000 E2_stripper_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_stripper - 0.5 l_coolingwater_pumphouse - 0.5 l_stripper >= -991
_C1323: 1000 E1_stripper_coolingwater_pumphouse
 - 1000 E2_stripper_coolingwater_pumphouse - Y_coolingwater_pumphouse
 + Y_stripper - 0.5 d_coolingwater_pumphouse - 0.5 d_stripper >= -991
_C1324: - 1000 E1_stripper_coolingwater_pumphouse
 - 1000 E2_stripper_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_stripper - 0.5 d_coolingwater_pumphouse - 0.5 d_stripper >= -1991
_C1325: L_stripper_coolingwater_pumphouse - R_coolingwater_pumphouse_stripper
 = 0
_C1326: - L_coolingwater_pumphouse_stripper
 + R_stripper_coolingwater_pumphouse = 0
_C1327: A_stripper_coolingwater_pumphouse - B_coolingwater_pumphouse_stripper
 = 0
_C1328: - A_coolingwater_pumphouse_stripper
 + B_stripper_coolingwater_pumphouse = 0
_C1329: Wx_coolingwater_pumphouse_stripper
 + Wx_stripper_coolingwater_pumphouse = 1
_C133: 1000 E1_absorber_cross_heat_exchanger
 + 1000 E2_absorber_cross_heat_exchanger + X_absorber - X_cross_heat_exchanger
 - 0.5 l_absorber - 0.5 l_cross_heat_exchanger >= 15
_C1330: Wy_coolingwater_pumphouse_stripper
 + Wy_stripper_coolingwater_pumphouse = 1
_C1331: - L_storage_tank_coolingwater_pumphouse
 + R_storage_tank_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_storage_tank = 0
_C1332: A_storage_tank_coolingwater_pumphouse
 - B_storage_tank_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_storage_tank = 0
_C1333: R_storage_tank_coolingwater_pumphouse
 - 1000 Wx_storage_tank_coolingwater_pumphouse <= 0
_C1334: L_storage_tank_coolingwater_pumphouse
 + 1000 Wx_storage_tank_coolingwater_pumphouse <= 1000
_C1335: A_storage_tank_coolingwater_pumphouse
 - 1000 Wy_storage_tank_coolingwater_pumphouse <= 0
_C1336: B_storage_tank_coolingwater_pumphouse
 + 1000 Wy_storage_tank_coolingwater_pumphouse <= 1000
_C1337: - A_storage_tank_coolingwater_pumphouse
 - B_storage_tank_coolingwater_pumphouse
 + D_storage_tank_coolingwater_pumphouse
 - L_storage_tank_coolingwater_pumphouse
 - R_storage_tank_coolingwater_pumphouse = 0
_C1338: - D_coolingwater_pumphouse_storage_tank
 + D_storage_tank_coolingwater_pumphouse = 0
_C1339: 1000 E1_storage_tank_coolingwater_pumphouse
 + 1000 E2_storage_tank_coolingwater_pumphouse - X_coolingwater_pumphouse
 + X_storage_tank - 0.5 l_coolingwater_pumphouse - 0.5 l_storage_tank >= 76
_C134: - 1000 E1_absorber_cross_heat_exchanger
 + 1000 E2_absorber_cross_heat_exchanger - X_absorber + X_cross_heat_exchanger
 - 0.5 l_absorber - 0.5 l_cross_heat_exchanger >= -985
_C1340: - 1000 E1_storage_tank_coolingwater_pumphouse
 + 1000 E2_storage_tank_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_storage_tank - 0.5 l_coolingwater_pumphouse - 0.5 l_storage_tank >= -924
_C1341: 1000 E1_storage_tank_coolingwater_pumphouse
 - 1000 E2_storage_tank_coolingwater_pumphouse - Y_coolingwater_pumphouse
 + Y_storage_tank - 0.5 d_coolingwater_pumphouse - 0.5 d_storage_tank >= -924
_C1342: - 1000 E1_storage_tank_coolingwater_pumphouse
 - 1000 E2_storage_tank_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_storage_tank - 0.5 d_coolingwater_pumphouse - 0.5 d_storage_tank >= -1924
_C1343: L_storage_tank_coolingwater_pumphouse
 - R_coolingwater_pumphouse_storage_tank = 0
_C1344: - L_coolingwater_pumphouse_storage_tank
 + R_storage_tank_coolingwater_pumphouse = 0
_C1345: A_storage_tank_coolingwater_pumphouse
 - B_coolingwater_pumphouse_storage_tank = 0
_C1346: - A_coolingwater_pumphouse_storage_tank
 + B_storage_tank_coolingwater_pumphouse = 0
_C1347: Wx_coolingwater_pumphouse_storage_tank
 + Wx_storage_tank_coolingwater_pumphouse = 1
_C1348: Wy_coolingwater_pumphouse_storage_tank
 + Wy_storage_tank_coolingwater_pumphouse = 1
_C1349: - L_reclaimer_coolingwater_pumphouse
 + R_reclaimer_coolingwater_pumphouse + X_coolingwater_pumphouse - X_reclaimer
 = 0
_C135: 1000 E1_absorber_cross_heat_exchanger
 - 1000 E2_absorber_cross_heat_exchanger + Y_absorber - Y_cross_heat_exchanger
 - 0.5 d_absorber - 0.5 d_cross_heat_exchanger >= -985
_C1350: A_reclaimer_coolingwater_pumphouse
 - B_reclaimer_coolingwater_pumphouse + Y_coolingwater_pumphouse - Y_reclaimer
 = 0
_C1351: R_reclaimer_coolingwater_pumphouse
 - 1000 Wx_reclaimer_coolingwater_pumphouse <= 0
_C1352: L_reclaimer_coolingwater_pumphouse
 + 1000 Wx_reclaimer_coolingwater_pumphouse <= 1000
_C1353: A_reclaimer_coolingwater_pumphouse
 - 1000 Wy_reclaimer_coolingwater_pumphouse <= 0
_C1354: B_reclaimer_coolingwater_pumphouse
 + 1000 Wy_reclaimer_coolingwater_pumphouse <= 1000
_C1355: - A_reclaimer_coolingwater_pumphouse
 - B_reclaimer_coolingwater_pumphouse + D_reclaimer_coolingwater_pumphouse
 - L_reclaimer_coolingwater_pumphouse - R_reclaimer_coolingwater_pumphouse = 0
_C1356: - D_coolingwater_pumphouse_reclaimer
 + D_reclaimer_coolingwater_pumphouse = 0
_C1357: 1000 E1_reclaimer_coolingwater_pumphouse
 + 1000 E2_reclaimer_coolingwater_pumphouse - X_coolingwater_pumphouse
 + X_reclaimer - 0.5 l_coolingwater_pumphouse - 0.5 l_reclaimer >= 9
_C1358: - 1000 E1_reclaimer_coolingwater_pumphouse
 + 1000 E2_reclaimer_coolingwater_pumphouse + X_coolingwater_pumphouse
 - X_reclaimer - 0.5 l_coolingwater_pumphouse - 0.5 l_reclaimer >= -991
_C1359: 1000 E1_reclaimer_coolingwater_pumphouse
 - 1000 E2_reclaimer_coolingwater_pumphouse - Y_coolingwater_pumphouse
 + Y_reclaimer - 0.5 d_coolingwater_pumphouse - 0.5 d_reclaimer >= -991
_C136: - 1000 E1_absorber_cross_heat_exchanger
 - 1000 E2_absorber_cross_heat_exchanger - Y_absorber + Y_cross_heat_exchanger
 - 0.5 d_absorber - 0.5 d_cross_heat_exchanger >= -1985
_C1360: - 1000 E1_reclaimer_coolingwater_pumphouse
 - 1000 E2_reclaimer_coolingwater_pumphouse + Y_coolingwater_pumphouse
 - Y_reclaimer - 0.5 d_coolingwater_pumphouse - 0.5 d_reclaimer >= -1991
_C1361: L_reclaimer_coolingwater_pumphouse
 - R_coolingwater_pumphouse_reclaimer = 0
_C1362: - L_coolingwater_pumphouse_reclaimer
 + R_reclaimer_coolingwater_pumphouse = 0
_C1363: A_reclaimer_coolingwater_pumphouse
 - B_coolingwater_pumphouse_reclaimer = 0
_C1364: - A_coolingwater_pumphouse_reclaimer
 + B_reclaimer_coolingwater_pumphouse = 0
_C1365: Wx_coolingwater_pumphouse_reclaimer
 + Wx_reclaimer_coolingwater_pumphouse = 1
_C1366: Wy_coolingwater_pumphouse_reclaimer
 + Wy_reclaimer_coolingwater_pumphouse = 1
_C1367: - L_compressor_house_coolingwater_pumphouse
 + R_compressor_house_coolingwater_pumphouse - X_compressor_house
 + X_coolingwater_pumphouse = 0
_C1368: A_compressor_house_coolingwater_pumphouse
 - B_compressor_house_coolingwater_pumphouse - Y_compressor_house
 + Y_coolingwater_pumphouse = 0
_C1369: R_compressor_house_coolingwater_pumphouse
 - 1000 Wx_compressor_house_coolingwater_pumphouse <= 0
_C137: L_absorber_cross_heat_exchanger - R_cross_heat_exchanger_absorber = 0
_C1370: L_compressor_house_coolingwater_pumphouse
 + 1000 Wx_compressor_house_coolingwater_pumphouse <= 1000
_C1371: A_compressor_house_coolingwater_pumphouse
 - 1000 Wy_compressor_house_coolingwater_pumphouse <= 0
_C1372: B_compressor_house_coolingwater_pumphouse
 + 1000 Wy_compressor_house_coolingwater_pumphouse <= 1000
_C1373: - A_compressor_house_coolingwater_pumphouse
 - B_compressor_house_coolingwater_pumphouse
 + D_compressor_house_coolingwater_pumphouse
 - L_compressor_house_coolingwater_pumphouse
 - R_compressor_house_coolingwater_pumphouse = 0
_C1374: D_compressor_house_coolingwater_pumphouse
 - D_coolingwater_pumphouse_compressor_house = 0
_C1375: 1000 E1_compressor_house_coolingwater_pumphouse
 + 1000 E2_compressor_house_coolingwater_pumphouse + X_compressor_house
 - X_coolingwater_pumphouse - 0.5 l_compressor_house
 - 0.5 l_coolingwater_pumphouse >= 0
_C1376: - 1000 E1_compressor_house_coolingwater_pumphouse
 + 1000 E2_compressor_house_coolingwater_pumphouse - X_compressor_house
 + X_coolingwater_pumphouse - 0.5 l_compressor_house
 - 0.5 l_coolingwater_pumphouse >= -1000
_C1377: 1000 E1_compressor_house_coolingwater_pumphouse
 - 1000 E2_compressor_house_coolingwater_pumphouse + Y_compressor_house
 - Y_coolingwater_pumphouse - 0.5 d_compressor_house
 - 0.5 d_coolingwater_pumphouse >= -1000
_C1378: - 1000 E1_compressor_house_coolingwater_pumphouse
 - 1000 E2_compressor_house_coolingwater_pumphouse - Y_compressor_house
 + Y_coolingwater_pumphouse - 0.5 d_compressor_house
 - 0.5 d_coolingwater_pumphouse >= -2000
_C1379: L_compressor_house_coolingwater_pumphouse
 - R_coolingwater_pumphouse_compressor_house = 0
_C138: - L_cross_heat_exchanger_absorber + R_absorber_cross_heat_exchanger = 0
_C1380: - L_coolingwater_pumphouse_compressor_house
 + R_compressor_house_coolingwater_pumphouse = 0
_C1381: A_compressor_house_coolingwater_pumphouse
 - B_coolingwater_pumphouse_compressor_house = 0
_C1382: - A_coolingwater_pumphouse_compressor_house
 + B_compressor_house_coolingwater_pumphouse = 0
_C1383: Wx_compressor_house_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_compressor_house = 1
_C1384: Wy_compressor_house_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_compressor_house = 1
_C1385: - L_compressor_KOdrum_coolingwater_pumphouse
 + R_compressor_KOdrum_coolingwater_pumphouse - X_compressor_KOdrum
 + X_coolingwater_pumphouse = 0
_C1386: A_compressor_KOdrum_coolingwater_pumphouse
 - B_compressor_KOdrum_coolingwater_pumphouse - Y_compressor_KOdrum
 + Y_coolingwater_pumphouse = 0
_C1387: R_compressor_KOdrum_coolingwater_pumphouse
 - 1000 Wx_compressor_KOdrum_coolingwater_pumphouse <= 0
_C1388: L_compressor_KOdrum_coolingwater_pumphouse
 + 1000 Wx_compressor_KOdrum_coolingwater_pumphouse <= 1000
_C1389: A_compressor_KOdrum_coolingwater_pumphouse
 - 1000 Wy_compressor_KOdrum_coolingwater_pumphouse <= 0
_C139: A_absorber_cross_heat_exchanger - B_cross_heat_exchanger_absorber = 0
_C1390: B_compressor_KOdrum_coolingwater_pumphouse
 + 1000 Wy_compressor_KOdrum_coolingwater_pumphouse <= 1000
_C1391: - A_compressor_KOdrum_coolingwater_pumphouse
 - B_compressor_KOdrum_coolingwater_pumphouse
 + D_compressor_KOdrum_coolingwater_pumphouse
 - L_compressor_KOdrum_coolingwater_pumphouse
 - R_compressor_KOdrum_coolingwater_pumphouse = 0
_C1392: D_compressor_KOdrum_coolingwater_pumphouse
 - D_coolingwater_pumphouse_compressor_KOdrum = 0
_C1393: 1000 E1_compressor_KOdrum_coolingwater_pumphouse
 + 1000 E2_compressor_KOdrum_coolingwater_pumphouse + X_compressor_KOdrum
 - X_coolingwater_pumphouse - 0.5 l_compressor_KOdrum
 - 0.5 l_coolingwater_pumphouse >= 9
_C1394: - 1000 E1_compressor_KOdrum_coolingwater_pumphouse
 + 1000 E2_compressor_KOdrum_coolingwater_pumphouse - X_compressor_KOdrum
 + X_coolingwater_pumphouse - 0.5 l_compressor_KOdrum
 - 0.5 l_coolingwater_pumphouse >= -991
_C1395: 1000 E1_compressor_KOdrum_coolingwater_pumphouse
 - 1000 E2_compressor_KOdrum_coolingwater_pumphouse + Y_compressor_KOdrum
 - Y_coolingwater_pumphouse - 0.5 d_compressor_KOdrum
 - 0.5 d_coolingwater_pumphouse >= -991
_C1396: - 1000 E1_compressor_KOdrum_coolingwater_pumphouse
 - 1000 E2_compressor_KOdrum_coolingwater_pumphouse - Y_compressor_KOdrum
 + Y_coolingwater_pumphouse - 0.5 d_compressor_KOdrum
 - 0.5 d_coolingwater_pumphouse >= -1991
_C1397: L_compressor_KOdrum_coolingwater_pumphouse
 - R_coolingwater_pumphouse_compressor_KOdrum = 0
_C1398: - L_coolingwater_pumphouse_compressor_KOdrum
 + R_compressor_KOdrum_coolingwater_pumphouse = 0
_C1399: A_compressor_KOdrum_coolingwater_pumphouse
 - B_coolingwater_pumphouse_compressor_KOdrum = 0
_C14: d_cross_heat_exchanger + l_cross_heat_exchanger = 32
_C140: - A_cross_heat_exchanger_absorber + B_absorber_cross_heat_exchanger = 0
_C1400: - A_coolingwater_pumphouse_compressor_KOdrum
 + B_compressor_KOdrum_coolingwater_pumphouse = 0
_C1401: Wx_compressor_KOdrum_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_compressor_KOdrum = 1
_C1402: Wy_compressor_KOdrum_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_compressor_KOdrum = 1
_C1403: - L_compressor_interstage_cooler_coolingwater_pumphouse
 + R_compressor_interstage_cooler_coolingwater_pumphouse
 - X_compressor_interstage_cooler + X_coolingwater_pumphouse = 0
_C1404: A_compressor_interstage_cooler_coolingwater_pumphouse
 - B_compressor_interstage_cooler_coolingwater_pumphouse
 - Y_compressor_interstage_cooler + Y_coolingwater_pumphouse = 0
_C1405: R_compressor_interstage_cooler_coolingwater_pumphouse
 - 1000 Wx_compressor_interstage_cooler_coolingwater_pumphouse <= 0
_C1406: L_compressor_interstage_cooler_coolingwater_pumphouse
 + 1000 Wx_compressor_interstage_cooler_coolingwater_pumphouse <= 1000
_C1407: A_compressor_interstage_cooler_coolingwater_pumphouse
 - 1000 Wy_compressor_interstage_cooler_coolingwater_pumphouse <= 0
_C1408: B_compressor_interstage_cooler_coolingwater_pumphouse
 + 1000 Wy_compressor_interstage_cooler_coolingwater_pumphouse <= 1000
_C1409: - A_compressor_interstage_cooler_coolingwater_pumphouse
 - B_compressor_interstage_cooler_coolingwater_pumphouse
 + D_compressor_interstage_cooler_coolingwater_pumphouse
 - L_compressor_interstage_cooler_coolingwater_pumphouse
 - R_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C141: Wx_absorber_cross_heat_exchanger + Wx_cross_heat_exchanger_absorber = 1
_C1410: D_compressor_interstage_cooler_coolingwater_pumphouse
 - D_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C1411: 1000 E1_compressor_interstage_cooler_coolingwater_pumphouse
 + 1000 E2_compressor_interstage_cooler_coolingwater_pumphouse
 + X_compressor_interstage_cooler - X_coolingwater_pumphouse
 - 0.5 l_compressor_interstage_cooler - 0.5 l_coolingwater_pumphouse >= 9
_C1412: - 1000 E1_compressor_interstage_cooler_coolingwater_pumphouse
 + 1000 E2_compressor_interstage_cooler_coolingwater_pumphouse
 - X_compressor_interstage_cooler + X_coolingwater_pumphouse
 - 0.5 l_compressor_interstage_cooler - 0.5 l_coolingwater_pumphouse >= -991
_C1413: 1000 E1_compressor_interstage_cooler_coolingwater_pumphouse
 - 1000 E2_compressor_interstage_cooler_coolingwater_pumphouse
 + Y_compressor_interstage_cooler - Y_coolingwater_pumphouse
 - 0.5 d_compressor_interstage_cooler - 0.5 d_coolingwater_pumphouse >= -991
_C1414: - 1000 E1_compressor_interstage_cooler_coolingwater_pumphouse
 - 1000 E2_compressor_interstage_cooler_coolingwater_pumphouse
 - Y_compressor_interstage_cooler + Y_coolingwater_pumphouse
 - 0.5 d_compressor_interstage_cooler - 0.5 d_coolingwater_pumphouse >= -1991
_C1415: L_compressor_interstage_cooler_coolingwater_pumphouse
 - R_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C1416: - L_coolingwater_pumphouse_compressor_interstage_cooler
 + R_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C1417: A_compressor_interstage_cooler_coolingwater_pumphouse
 - B_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C1418: - A_coolingwater_pumphouse_compressor_interstage_cooler
 + B_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C1419: Wx_compressor_interstage_cooler_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_compressor_interstage_cooler = 1
_C142: Wy_absorber_cross_heat_exchanger + Wy_cross_heat_exchanger_absorber = 1
_C1420: Wy_compressor_interstage_cooler_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_compressor_interstage_cooler = 1
_C1421: - L_TEG_dehydration_coolingwater_pumphouse
 + R_TEG_dehydration_coolingwater_pumphouse - X_TEG_dehydration
 + X_coolingwater_pumphouse = 0
_C1422: A_TEG_dehydration_coolingwater_pumphouse
 - B_TEG_dehydration_coolingwater_pumphouse - Y_TEG_dehydration
 + Y_coolingwater_pumphouse = 0
_C1423: R_TEG_dehydration_coolingwater_pumphouse
 - 1000 Wx_TEG_dehydration_coolingwater_pumphouse <= 0
_C1424: L_TEG_dehydration_coolingwater_pumphouse
 + 1000 Wx_TEG_dehydration_coolingwater_pumphouse <= 1000
_C1425: A_TEG_dehydration_coolingwater_pumphouse
 - 1000 Wy_TEG_dehydration_coolingwater_pumphouse <= 0
_C1426: B_TEG_dehydration_coolingwater_pumphouse
 + 1000 Wy_TEG_dehydration_coolingwater_pumphouse <= 1000
_C1427: - A_TEG_dehydration_coolingwater_pumphouse
 - B_TEG_dehydration_coolingwater_pumphouse
 + D_TEG_dehydration_coolingwater_pumphouse
 - L_TEG_dehydration_coolingwater_pumphouse
 - R_TEG_dehydration_coolingwater_pumphouse = 0
_C1428: D_TEG_dehydration_coolingwater_pumphouse
 - D_coolingwater_pumphouse_TEG_dehydration = 0
_C1429: 1000 E1_TEG_dehydration_coolingwater_pumphouse
 + 1000 E2_TEG_dehydration_coolingwater_pumphouse + X_TEG_dehydration
 - X_coolingwater_pumphouse - 0.5 l_TEG_dehydration
 - 0.5 l_coolingwater_pumphouse >= 9
_C143: - L_lean_amine_cooler_cross_heat_exchanger
 + R_lean_amine_cooler_cross_heat_exchanger + X_cross_heat_exchanger
 - X_lean_amine_cooler = 0
_C1430: - 1000 E1_TEG_dehydration_coolingwater_pumphouse
 + 1000 E2_TEG_dehydration_coolingwater_pumphouse - X_TEG_dehydration
 + X_coolingwater_pumphouse - 0.5 l_TEG_dehydration
 - 0.5 l_coolingwater_pumphouse >= -991
_C1431: 1000 E1_TEG_dehydration_coolingwater_pumphouse
 - 1000 E2_TEG_dehydration_coolingwater_pumphouse + Y_TEG_dehydration
 - Y_coolingwater_pumphouse - 0.5 d_TEG_dehydration
 - 0.5 d_coolingwater_pumphouse >= -991
_C1432: - 1000 E1_TEG_dehydration_coolingwater_pumphouse
 - 1000 E2_TEG_dehydration_coolingwater_pumphouse - Y_TEG_dehydration
 + Y_coolingwater_pumphouse - 0.5 d_TEG_dehydration
 - 0.5 d_coolingwater_pumphouse >= -1991
_C1433: L_TEG_dehydration_coolingwater_pumphouse
 - R_coolingwater_pumphouse_TEG_dehydration = 0
_C1434: - L_coolingwater_pumphouse_TEG_dehydration
 + R_TEG_dehydration_coolingwater_pumphouse = 0
_C1435: A_TEG_dehydration_coolingwater_pumphouse
 - B_coolingwater_pumphouse_TEG_dehydration = 0
_C1436: - A_coolingwater_pumphouse_TEG_dehydration
 + B_TEG_dehydration_coolingwater_pumphouse = 0
_C1437: Wx_TEG_dehydration_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_TEG_dehydration = 1
_C1438: Wy_TEG_dehydration_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_TEG_dehydration = 1
_C1439: - L_cooling_tower_coolingwater_pumphouse
 + R_cooling_tower_coolingwater_pumphouse - X_cooling_tower
 + X_coolingwater_pumphouse = 0
_C144: A_lean_amine_cooler_cross_heat_exchanger
 - B_lean_amine_cooler_cross_heat_exchanger + Y_cross_heat_exchanger
 - Y_lean_amine_cooler = 0
_C1440: A_cooling_tower_coolingwater_pumphouse
 - B_cooling_tower_coolingwater_pumphouse - Y_cooling_tower
 + Y_coolingwater_pumphouse = 0
_C1441: R_cooling_tower_coolingwater_pumphouse
 - 1000 Wx_cooling_tower_coolingwater_pumphouse <= 0
_C1442: L_cooling_tower_coolingwater_pumphouse
 + 1000 Wx_cooling_tower_coolingwater_pumphouse <= 1000
_C1443: A_cooling_tower_coolingwater_pumphouse
 - 1000 Wy_cooling_tower_coolingwater_pumphouse <= 0
_C1444: B_cooling_tower_coolingwater_pumphouse
 + 1000 Wy_cooling_tower_coolingwater_pumphouse <= 1000
_C1445: - A_cooling_tower_coolingwater_pumphouse
 - B_cooling_tower_coolingwater_pumphouse
 + D_cooling_tower_coolingwater_pumphouse
 - L_cooling_tower_coolingwater_pumphouse
 - R_cooling_tower_coolingwater_pumphouse = 0
_C1446: D_cooling_tower_coolingwater_pumphouse
 - D_coolingwater_pumphouse_cooling_tower = 0
_C1447: 1000 E1_cooling_tower_coolingwater_pumphouse
 + 1000 E2_cooling_tower_coolingwater_pumphouse + X_cooling_tower
 - X_coolingwater_pumphouse - 0.5 l_cooling_tower
 - 0.5 l_coolingwater_pumphouse >= 31
_C1448: - 1000 E1_cooling_tower_coolingwater_pumphouse
 + 1000 E2_cooling_tower_coolingwater_pumphouse - X_cooling_tower
 + X_coolingwater_pumphouse - 0.5 l_cooling_tower
 - 0.5 l_coolingwater_pumphouse >= -969
_C1449: 1000 E1_cooling_tower_coolingwater_pumphouse
 - 1000 E2_cooling_tower_coolingwater_pumphouse + Y_cooling_tower
 - Y_coolingwater_pumphouse - 0.5 d_cooling_tower
 - 0.5 d_coolingwater_pumphouse >= -969
_C145: R_lean_amine_cooler_cross_heat_exchanger
 - 1000 Wx_lean_amine_cooler_cross_heat_exchanger <= 0
_C1450: - 1000 E1_cooling_tower_coolingwater_pumphouse
 - 1000 E2_cooling_tower_coolingwater_pumphouse - Y_cooling_tower
 + Y_coolingwater_pumphouse - 0.5 d_cooling_tower
 - 0.5 d_coolingwater_pumphouse >= -1969
_C1451: L_cooling_tower_coolingwater_pumphouse
 - R_coolingwater_pumphouse_cooling_tower = 0
_C1452: - L_coolingwater_pumphouse_cooling_tower
 + R_cooling_tower_coolingwater_pumphouse = 0
_C1453: A_cooling_tower_coolingwater_pumphouse
 - B_coolingwater_pumphouse_cooling_tower = 0
_C1454: - A_coolingwater_pumphouse_cooling_tower
 + B_cooling_tower_coolingwater_pumphouse = 0
_C1455: Wx_cooling_tower_coolingwater_pumphouse
 + Wx_coolingwater_pumphouse_cooling_tower = 1
_C1456: Wy_cooling_tower_coolingwater_pumphouse
 + Wy_coolingwater_pumphouse_cooling_tower = 1
_C1457: CD_direct_contact_cooler_direct_contact_cooler = 0
_C1458: CD_direct_contact_cooler_absorber
 - 428 D_direct_contact_cooler_absorber = 0
_C1459: CD_direct_contact_cooler_lean_amine_cooler = 0
_C146: L_lean_amine_cooler_cross_heat_exchanger
 + 1000 Wx_lean_amine_cooler_cross_heat_exchanger <= 1000
_C1460: CD_direct_contact_cooler_cross_heat_exchanger = 0
_C1461: CD_direct_contact_cooler_stripper = 0
_C1462: CD_direct_contact_cooler_storage_tank = 0
_C1463: CD_direct_contact_cooler_reclaimer = 0
_C1464: CD_direct_contact_cooler_compressor_house = 0
_C1465: CD_direct_contact_cooler_compressor_KOdrum = 0
_C1466: CD_direct_contact_cooler_compressor_interstage_cooler = 0
_C1467: CD_direct_contact_cooler_TEG_dehydration = 0
_C1468: CD_direct_contact_cooler_cooling_tower = 0
_C1469: CD_direct_contact_cooler_coolingwater_pumphouse = 0
_C147: A_lean_amine_cooler_cross_heat_exchanger
 - 1000 Wy_lean_amine_cooler_cross_heat_exchanger <= 0
_C1470: CD_absorber_direct_contact_cooler = 0
_C1471: CD_absorber_absorber = 0
_C1472: CD_absorber_lean_amine_cooler - 106 D_absorber_lean_amine_cooler = 0
_C1473: CD_absorber_cross_heat_exchanger - 112 D_absorber_cross_heat_exchanger
 = 0
_C1474: CD_absorber_stripper = 0
_C1475: CD_absorber_storage_tank = 0
_C1476: CD_absorber_reclaimer = 0
_C1477: CD_absorber_compressor_house = 0
_C1478: CD_absorber_compressor_KOdrum = 0
_C1479: CD_absorber_compressor_interstage_cooler = 0
_C148: B_lean_amine_cooler_cross_heat_exchanger
 + 1000 Wy_lean_amine_cooler_cross_heat_exchanger <= 1000
_C1480: CD_absorber_TEG_dehydration = 0
_C1481: CD_absorber_cooling_tower = 0
_C1482: CD_absorber_coolingwater_pumphouse = 0
_C1483: CD_lean_amine_cooler_direct_contact_cooler = 0
_C1484: CD_lean_amine_cooler_absorber = 0
_C1485: CD_lean_amine_cooler_lean_amine_cooler = 0
_C1486: CD_lean_amine_cooler_cross_heat_exchanger
 - 106 D_lean_amine_cooler_cross_heat_exchanger = 0
_C1487: CD_lean_amine_cooler_stripper = 0
_C1488: CD_lean_amine_cooler_storage_tank = 0
_C1489: CD_lean_amine_cooler_reclaimer = 0
_C149: - A_lean_amine_cooler_cross_heat_exchanger
 - B_lean_amine_cooler_cross_heat_exchanger
 + D_lean_amine_cooler_cross_heat_exchanger
 - L_lean_amine_cooler_cross_heat_exchanger
 - R_lean_amine_cooler_cross_heat_exchanger = 0
_C1490: CD_lean_amine_cooler_compressor_house = 0
_C1491: CD_lean_amine_cooler_compressor_KOdrum = 0
_C1492: CD_lean_amine_cooler_compressor_interstage_cooler = 0
_C1493: CD_lean_amine_cooler_TEG_dehydration = 0
_C1494: CD_lean_amine_cooler_cooling_tower = 0
_C1495: CD_lean_amine_cooler_coolingwater_pumphouse = 0
_C1496: CD_cross_heat_exchanger_direct_contact_cooler = 0
_C1497: CD_cross_heat_exchanger_absorber = 0
_C1498: CD_cross_heat_exchanger_lean_amine_cooler = 0
_C1499: CD_cross_heat_exchanger_cross_heat_exchanger = 0
_C15: X_cross_heat_exchanger - 0.5 l_cross_heat_exchanger >= 0
_C150: - D_cross_heat_exchanger_lean_amine_cooler
 + D_lean_amine_cooler_cross_heat_exchanger = 0
_C1500: CD_cross_heat_exchanger_stripper - 178 D_cross_heat_exchanger_stripper
 = 0
_C1501: CD_cross_heat_exchanger_storage_tank = 0
_C1502: CD_cross_heat_exchanger_reclaimer = 0
_C1503: CD_cross_heat_exchanger_compressor_house = 0
_C1504: CD_cross_heat_exchanger_compressor_KOdrum = 0
_C1505: CD_cross_heat_exchanger_compressor_interstage_cooler = 0
_C1506: CD_cross_heat_exchanger_TEG_dehydration = 0
_C1507: CD_cross_heat_exchanger_cooling_tower = 0
_C1508: CD_cross_heat_exchanger_coolingwater_pumphouse = 0
_C1509: CD_stripper_direct_contact_cooler = 0
_C151: 1000 E1_lean_amine_cooler_cross_heat_exchanger
 + 1000 E2_lean_amine_cooler_cross_heat_exchanger - X_cross_heat_exchanger
 + X_lean_amine_cooler - 0.5 l_cross_heat_exchanger - 0.5 l_lean_amine_cooler
 >= 15
_C1510: CD_stripper_absorber = 0
_C1511: CD_stripper_lean_amine_cooler = 0
_C1512: CD_stripper_cross_heat_exchanger = 0
_C1513: CD_stripper_stripper = 0
_C1514: CD_stripper_storage_tank = 0
_C1515: CD_stripper_reclaimer = 0
_C1516: CD_stripper_compressor_house = 0
_C1517: CD_stripper_compressor_KOdrum - 364 D_stripper_compressor_KOdrum = 0
_C1518: CD_stripper_compressor_interstage_cooler = 0
_C1519: CD_stripper_TEG_dehydration = 0
_C152: - 1000 E1_lean_amine_cooler_cross_heat_exchanger
 + 1000 E2_lean_amine_cooler_cross_heat_exchanger + X_cross_heat_exchanger
 - X_lean_amine_cooler - 0.5 l_cross_heat_exchanger - 0.5 l_lean_amine_cooler
 >= -985
_C1520: CD_stripper_cooling_tower = 0
_C1521: CD_stripper_coolingwater_pumphouse = 0
_C1522: CD_storage_tank_direct_contact_cooler = 0
_C1523: CD_storage_tank_absorber = 0
_C1524: CD_storage_tank_lean_amine_cooler = 0
_C1525: CD_storage_tank_cross_heat_exchanger
 - 112 D_storage_tank_cross_heat_exchanger = 0
_C1526: CD_storage_tank_stripper = 0
_C1527: CD_storage_tank_storage_tank = 0
_C1528: CD_storage_tank_reclaimer = 0
_C1529: CD_storage_tank_compressor_house = 0
_C153: 1000 E1_lean_amine_cooler_cross_heat_exchanger
 - 1000 E2_lean_amine_cooler_cross_heat_exchanger - Y_cross_heat_exchanger
 + Y_lean_amine_cooler - 0.5 d_cross_heat_exchanger - 0.5 d_lean_amine_cooler
 >= -985
_C1530: CD_storage_tank_compressor_KOdrum = 0
_C1531: CD_storage_tank_compressor_interstage_cooler = 0
_C1532: CD_storage_tank_TEG_dehydration = 0
_C1533: CD_storage_tank_cooling_tower = 0
_C1534: CD_storage_tank_coolingwater_pumphouse = 0
_C1535: CD_reclaimer_direct_contact_cooler = 0
_C1536: CD_reclaimer_absorber = 0
_C1537: CD_reclaimer_lean_amine_cooler = 0
_C1538: CD_reclaimer_cross_heat_exchanger
 - 106 D_reclaimer_cross_heat_exchanger = 0
_C1539: CD_reclaimer_stripper = 0
_C154: - 1000 E1_lean_amine_cooler_cross_heat_exchanger
 - 1000 E2_lean_amine_cooler_cross_heat_exchanger + Y_cross_heat_exchanger
 - Y_lean_amine_cooler - 0.5 d_cross_heat_exchanger - 0.5 d_lean_amine_cooler
 >= -1985
_C1540: CD_reclaimer_storage_tank = 0
_C1541: CD_reclaimer_reclaimer = 0
_C1542: CD_reclaimer_compressor_house = 0
_C1543: CD_reclaimer_compressor_KOdrum = 0
_C1544: CD_reclaimer_compressor_interstage_cooler = 0
_C1545: CD_reclaimer_TEG_dehydration = 0
_C1546: CD_reclaimer_cooling_tower = 0
_C1547: CD_reclaimer_coolingwater_pumphouse = 0
_C1548: CD_compressor_house_direct_contact_cooler = 0
_C1549: CD_compressor_house_absorber = 0
_C155: L_lean_amine_cooler_cross_heat_exchanger
 - R_cross_heat_exchanger_lean_amine_cooler = 0
_C1550: CD_compressor_house_lean_amine_cooler = 0
_C1551: CD_compressor_house_cross_heat_exchanger = 0
_C1552: CD_compressor_house_stripper = 0
_C1553: CD_compressor_house_storage_tank = 0
_C1554: CD_compressor_house_reclaimer = 0
_C1555: CD_compressor_house_compressor_house = 0
_C1556: CD_compressor_house_compressor_KOdrum = 0
_C1557: CD_compressor_house_compressor_interstage_cooler = 0
_C1558: CD_compressor_house_TEG_dehydration = 0
_C1559: CD_compressor_house_cooling_tower = 0
_C156: - L_cross_heat_exchanger_lean_amine_cooler
 + R_lean_amine_cooler_cross_heat_exchanger = 0
_C1560: CD_compressor_house_coolingwater_pumphouse = 0
_C1561: CD_compressor_KOdrum_direct_contact_cooler = 0
_C1562: CD_compressor_KOdrum_absorber = 0
_C1563: CD_compressor_KOdrum_lean_amine_cooler = 0
_C1564: CD_compressor_KOdrum_cross_heat_exchanger = 0
_C1565: CD_compressor_KOdrum_stripper = 0
_C1566: CD_compressor_KOdrum_storage_tank = 0
_C1567: CD_compressor_KOdrum_reclaimer = 0
_C1568: CD_compressor_KOdrum_compressor_house
 - 910 D_compressor_KOdrum_compressor_house = 0
_C1569: CD_compressor_KOdrum_compressor_KOdrum = 0
_C157: A_lean_amine_cooler_cross_heat_exchanger
 - B_cross_heat_exchanger_lean_amine_cooler = 0
_C1570: CD_compressor_KOdrum_compressor_interstage_cooler = 0
_C1571: CD_compressor_KOdrum_TEG_dehydration = 0
_C1572: CD_compressor_KOdrum_cooling_tower = 0
_C1573: CD_compressor_KOdrum_coolingwater_pumphouse = 0
_C1574: CD_compressor_interstage_cooler_direct_contact_cooler = 0
_C1575: CD_compressor_interstage_cooler_absorber = 0
_C1576: CD_compressor_interstage_cooler_lean_amine_cooler = 0
_C1577: CD_compressor_interstage_cooler_cross_heat_exchanger = 0
_C1578: CD_compressor_interstage_cooler_stripper = 0
_C1579: CD_compressor_interstage_cooler_storage_tank = 0
_C158: - A_cross_heat_exchanger_lean_amine_cooler
 + B_lean_amine_cooler_cross_heat_exchanger = 0
_C1580: CD_compressor_interstage_cooler_reclaimer = 0
_C1581: CD_compressor_interstage_cooler_compressor_house
 - 910 D_compressor_interstage_cooler_compressor_house = 0
_C1582: CD_compressor_interstage_cooler_compressor_KOdrum = 0
_C1583: CD_compressor_interstage_cooler_compressor_interstage_cooler = 0
_C1584: CD_compressor_interstage_cooler_TEG_dehydration = 0
_C1585: CD_compressor_interstage_cooler_cooling_tower = 0
_C1586: CD_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C1587: CD_TEG_dehydration_direct_contact_cooler = 0
_C1588: CD_TEG_dehydration_absorber = 0
_C1589: CD_TEG_dehydration_lean_amine_cooler = 0
_C159: Wx_cross_heat_exchanger_lean_amine_cooler
 + Wx_lean_amine_cooler_cross_heat_exchanger = 1
_C1590: CD_TEG_dehydration_cross_heat_exchanger = 0
_C1591: CD_TEG_dehydration_stripper = 0
_C1592: CD_TEG_dehydration_storage_tank = 0
_C1593: CD_TEG_dehydration_reclaimer = 0
_C1594: CD_TEG_dehydration_compressor_house
 - 364 D_TEG_dehydration_compressor_house = 0
_C1595: CD_TEG_dehydration_compressor_KOdrum = 0
_C1596: CD_TEG_dehydration_compressor_interstage_cooler = 0
_C1597: CD_TEG_dehydration_TEG_dehydration = 0
_C1598: CD_TEG_dehydration_cooling_tower = 0
_C1599: CD_TEG_dehydration_coolingwater_pumphouse = 0
_C16: Y_cross_heat_exchanger - 0.5 d_cross_heat_exchanger >= 0
_C160: Wy_cross_heat_exchanger_lean_amine_cooler
 + Wy_lean_amine_cooler_cross_heat_exchanger = 1
_C1600: CD_cooling_tower_direct_contact_cooler = 0
_C1601: CD_cooling_tower_absorber = 0
_C1602: CD_cooling_tower_lean_amine_cooler = 0
_C1603: CD_cooling_tower_cross_heat_exchanger = 0
_C1604: CD_cooling_tower_stripper = 0
_C1605: CD_cooling_tower_storage_tank = 0
_C1606: CD_cooling_tower_reclaimer = 0
_C1607: CD_cooling_tower_compressor_house = 0
_C1608: CD_cooling_tower_compressor_KOdrum = 0
_C1609: CD_cooling_tower_compressor_interstage_cooler = 0
_C161: - L_direct_contact_cooler_stripper + R_direct_contact_cooler_stripper
 - X_direct_contact_cooler + X_stripper = 0
_C1610: CD_cooling_tower_TEG_dehydration = 0
_C1611: CD_cooling_tower_cooling_tower = 0
_C1612: CD_cooling_tower_coolingwater_pumphouse
 - 768 D_cooling_tower_coolingwater_pumphouse = 0
_C1613: CD_coolingwater_pumphouse_direct_contact_cooler = 0
_C1614: CD_coolingwater_pumphouse_absorber = 0
_C1615: CD_coolingwater_pumphouse_lean_amine_cooler = 0
_C1616: CD_coolingwater_pumphouse_cross_heat_exchanger = 0
_C1617: CD_coolingwater_pumphouse_stripper = 0
_C1618: CD_coolingwater_pumphouse_storage_tank = 0
_C1619: CD_coolingwater_pumphouse_reclaimer = 0
_C162: A_direct_contact_cooler_stripper - B_direct_contact_cooler_stripper
 - Y_direct_contact_cooler + Y_stripper = 0
_C1620: CD_coolingwater_pumphouse_compressor_house = 0
_C1621: CD_coolingwater_pumphouse_compressor_KOdrum = 0
_C1622: CD_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C1623: CD_coolingwater_pumphouse_TEG_dehydration = 0
_C1624: CD_coolingwater_pumphouse_cooling_tower = 0
_C1625: CD_coolingwater_pumphouse_coolingwater_pumphouse = 0
_C1626: - CD_TEG_dehydration_TEG_dehydration - CD_TEG_dehydration_absorber
 - CD_TEG_dehydration_compressor_KOdrum - CD_TEG_dehydration_compressor_house
 - CD_TEG_dehydration_compressor_interstage_cooler
 - CD_TEG_dehydration_cooling_tower
 - CD_TEG_dehydration_coolingwater_pumphouse
 - CD_TEG_dehydration_cross_heat_exchanger
 - CD_TEG_dehydration_direct_contact_cooler
 - CD_TEG_dehydration_lean_amine_cooler - CD_TEG_dehydration_reclaimer
 - CD_TEG_dehydration_storage_tank - CD_TEG_dehydration_stripper
 - CD_absorber_TEG_dehydration - CD_absorber_absorber
 - CD_absorber_compressor_KOdrum - CD_absorber_compressor_house
 - CD_absorber_compressor_interstage_cooler - CD_absorber_cooling_tower
 - CD_absorber_coolingwater_pumphouse - CD_absorber_cross_heat_exchanger
 - CD_absorber_direct_contact_cooler - CD_absorber_lean_amine_cooler
 - CD_absorber_reclaimer - CD_absorber_storage_tank - CD_absorber_stripper
 - CD_compressor_KOdrum_TEG_dehydration - CD_compressor_KOdrum_absorber
 - CD_compressor_KOdrum_compressor_KOdrum
 - CD_compressor_KOdrum_compressor_house
 - CD_compressor_KOdrum_compressor_interstage_cooler
 - CD_compressor_KOdrum_cooling_tower
 - CD_compressor_KOdrum_coolingwater_pumphouse
 - CD_compressor_KOdrum_cross_heat_exchanger
 - CD_compressor_KOdrum_direct_contact_cooler
 - CD_compressor_KOdrum_lean_amine_cooler - CD_compressor_KOdrum_reclaimer
 - CD_compressor_KOdrum_storage_tank - CD_compressor_KOdrum_stripper
 - CD_compressor_house_TEG_dehydration - CD_compressor_house_absorber
 - CD_compressor_house_compressor_KOdrum
 - CD_compressor_house_compressor_house
 - CD_compressor_house_compressor_interstage_cooler
 - CD_compressor_house_cooling_tower
 - CD_compressor_house_coolingwater_pumphouse
 - CD_compressor_house_cross_heat_exchanger
 - CD_compressor_house_direct_contact_cooler
 - CD_compressor_house_lean_amine_cooler - CD_compressor_house_reclaimer
 - CD_compressor_house_storage_tank - CD_compressor_house_stripper
 - CD_compressor_interstage_cooler_TEG_dehydration
 - CD_compressor_interstage_cooler_absorber
 - CD_compressor_interstage_cooler_compressor_KOdrum
 - CD_compressor_interstage_cooler_compressor_house
 - CD_compressor_interstage_cooler_compressor_interstage_cooler
 - CD_compressor_interstage_cooler_cooling_tower
 - CD_compressor_interstage_cooler_coolingwater_pumphouse
 - CD_compressor_interstage_cooler_cross_heat_exchanger
 - CD_compressor_interstage_cooler_direct_contact_cooler
 - CD_compressor_interstage_cooler_lean_amine_cooler
 - CD_compressor_interstage_cooler_reclaimer
 - CD_compressor_interstage_cooler_storage_tank
 - CD_compressor_interstage_cooler_stripper - CD_cooling_tower_TEG_dehydration
 - CD_cooling_tower_absorber - CD_cooling_tower_compressor_KOdrum
 - CD_cooling_tower_compressor_house
 - CD_cooling_tower_compressor_interstage_cooler
 - CD_cooling_tower_cooling_tower - CD_cooling_tower_coolingwater_pumphouse
 - CD_cooling_tower_cross_heat_exchanger
 - CD_cooling_tower_direct_contact_cooler - CD_cooling_tower_lean_amine_cooler
 - CD_cooling_tower_reclaimer - CD_cooling_tower_storage_tank
 - CD_cooling_tower_stripper - CD_coolingwater_pumphouse_TEG_dehydration
 - CD_coolingwater_pumphouse_absorber
 - CD_coolingwater_pumphouse_compressor_KOdrum
 - CD_coolingwater_pumphouse_compressor_house
 - CD_coolingwater_pumphouse_compressor_interstage_cooler
 - CD_coolingwater_pumphouse_cooling_tower
 - CD_coolingwater_pumphouse_coolingwater_pumphouse
 - CD_coolingwater_pumphouse_cross_heat_exchanger
 - CD_coolingwater_pumphouse_direct_contact_cooler
 - CD_coolingwater_pumphouse_lean_amine_cooler
 - CD_coolingwater_pumphouse_reclaimer
 - CD_coolingwater_pumphouse_storage_tank - CD_coolingwater_pumphouse_stripper
 - CD_cross_heat_exchanger_TEG_dehydration - CD_cross_heat_exchanger_absorber
 - CD_cross_heat_exchanger_compressor_KOdrum
 - CD_cross_heat_exchanger_compressor_house
 - CD_cross_heat_exchanger_compressor_interstage_cooler
 - CD_cross_heat_exchanger_cooling_tower
 - CD_cross_heat_exchanger_coolingwater_pumphouse
 - CD_cross_heat_exchanger_cross_heat_exchanger
 - CD_cross_heat_exchanger_direct_contact_cooler
 - CD_cross_heat_exchanger_lean_amine_cooler
 - CD_cross_heat_exchanger_reclaimer - CD_cross_heat_exchanger_storage_tank
 - CD_cross_heat_exchanger_stripper - CD_direct_contact_cooler_TEG_dehydration
 - CD_direct_contact_cooler_absorber
 - CD_direct_contact_cooler_compressor_KOdrum
 - CD_direct_contact_cooler_compressor_house
 - CD_direct_contact_cooler_compressor_interstage_cooler
 - CD_direct_contact_cooler_cooling_tower
 - CD_direct_contact_cooler_coolingwater_pumphouse
 - CD_direct_contact_cooler_cross_heat_exchanger
 - CD_direct_contact_cooler_direct_contact_cooler
 - CD_direct_contact_cooler_lean_amine_cooler
 - CD_direct_contact_cooler_reclaimer - CD_direct_contact_cooler_storage_tank
 - CD_direct_contact_cooler_stripper - CD_lean_amine_cooler_TEG_dehydration
 - CD_lean_amine_cooler_absorber - CD_lean_amine_cooler_compressor_KOdrum
 - CD_lean_amine_cooler_compressor_house
 - CD_lean_amine_cooler_compressor_interstage_cooler
 - CD_lean_amine_cooler_cooling_tower
 - CD_lean_amine_cooler_coolingwater_pumphouse
 - CD_lean_amine_cooler_cross_heat_exchanger
 - CD_lean_amine_cooler_direct_contact_cooler
 - CD_lean_amine_cooler_lean_amine_cooler - CD_lean_amine_cooler_reclaimer
 - CD_lean_amine_cooler_storage_tank - CD_lean_amine_cooler_stripper
 - CD_reclaimer_TEG_dehydration - CD_reclaimer_absorber
 - CD_reclaimer_compressor_KOdrum - CD_reclaimer_compressor_house
 - CD_reclaimer_compressor_interstage_cooler - CD_reclaimer_cooling_tower
 - CD_reclaimer_coolingwater_pumphouse - CD_reclaimer_cross_heat_exchanger
 - CD_reclaimer_direct_contact_cooler - CD_reclaimer_lean_amine_cooler
 - CD_reclaimer_reclaimer - CD_reclaimer_storage_tank - CD_reclaimer_stripper
 - CD_storage_tank_TEG_dehydration - CD_storage_tank_absorber
 - CD_storage_tank_compressor_KOdrum - CD_storage_tank_compressor_house
 - CD_storage_tank_compressor_interstage_cooler
 - CD_storage_tank_cooling_tower - CD_storage_tank_coolingwater_pumphouse
 - CD_storage_tank_cross_heat_exchanger
 - CD_storage_tank_direct_contact_cooler - CD_storage_tank_lean_amine_cooler
 - CD_storage_tank_reclaimer - CD_storage_tank_storage_tank
 - CD_storage_tank_stripper - CD_stripper_TEG_dehydration
 - CD_stripper_absorber - CD_stripper_compressor_KOdrum
 - CD_stripper_compressor_house - CD_stripper_compressor_interstage_cooler
 - CD_stripper_cooling_tower - CD_stripper_coolingwater_pumphouse
 - CD_stripper_cross_heat_exchanger - CD_stripper_direct_contact_cooler
 - CD_stripper_lean_amine_cooler - CD_stripper_reclaimer
 - CD_stripper_storage_tank - CD_stripper_stripper + SumCD = 0
_C1627: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_direct_contact_cooler + 0.5 l_direct_contact_cooler <= 0
_C1628: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_direct_contact_cooler + 0.5 d_direct_contact_cooler <= 0
_C1629: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C163: R_direct_contact_cooler_stripper
 - 1000 Wx_direct_contact_cooler_stripper <= 0
_C1630: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_absorber + 0.5 l_absorber <= 0
_C1631: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_absorber + 0.5 d_absorber <= 0
_C1632: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1633: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_lean_amine_cooler + 0.5 l_lean_amine_cooler <= 0
_C1634: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_lean_amine_cooler + 0.5 d_lean_amine_cooler <= 0
_C1635: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1636: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_cross_heat_exchanger + 0.5 l_cross_heat_exchanger <= 0
_C1637: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_cross_heat_exchanger + 0.5 d_cross_heat_exchanger <= 0
_C1638: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1639: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_stripper + 0.5 l_stripper <= 0
_C164: L_direct_contact_cooler_stripper
 + 1000 Wx_direct_contact_cooler_stripper <= 1000
_C1640: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_stripper + 0.5 d_stripper <= 0
_C1641: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1642: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_storage_tank + 0.5 l_storage_tank <= 0
_C1643: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_storage_tank + 0.5 d_storage_tank <= 0
_C1644: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1645: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_reclaimer + 0.5 l_reclaimer <= 0
_C1646: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_reclaimer + 0.5 d_reclaimer <= 0
_C1647: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1648: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_compressor_house + 0.5 l_compressor_house <= 0
_C1649: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_compressor_house + 0.5 d_compressor_house <= 0
_C165: A_direct_contact_cooler_stripper
 - 1000 Wy_direct_contact_cooler_stripper <= 0
_C1650: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1651: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_compressor_KOdrum + 0.5 l_compressor_KOdrum <= 0
_C1652: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_compressor_KOdrum + 0.5 d_compressor_KOdrum <= 0
_C1653: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1654: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_compressor_interstage_cooler + 0.5 l_compressor_interstage_cooler <= 0
_C1655: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_compressor_interstage_cooler + 0.5 d_compressor_interstage_cooler <= 0
_C1656: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1657: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_TEG_dehydration + 0.5 l_TEG_dehydration <= 0
_C1658: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_TEG_dehydration + 0.5 d_TEG_dehydration <= 0
_C1659: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C166: B_direct_contact_cooler_stripper
 + 1000 Wy_direct_contact_cooler_stripper <= 1000
_C1660: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_cooling_tower + 0.5 l_cooling_tower <= 0
_C1661: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_cooling_tower + 0.5 d_cooling_tower <= 0
_C1662: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1663: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + X_coolingwater_pumphouse + 0.5 l_coolingwater_pumphouse <= 0
_C1664: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102
 - 206 Gn_103 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108
 - 218 Gn_109 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113
 - 228 Gn_114 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119
 - 24 Gn_12 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124
 - 250 Gn_125 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13
 - 260 Gn_130 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135
 - 272 Gn_136 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140
 - 282 Gn_141 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146
 - 294 Gn_147 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151
 - 304 Gn_152 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157
 - 316 Gn_158 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162
 - 326 Gn_163 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168
 - 338 Gn_169 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173
 - 348 Gn_174 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179
 - 36 Gn_18 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184
 - 370 Gn_185 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19
 - 380 Gn_190 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195
 - 392 Gn_196 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20
 - 400 Gn_200 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205
 - 412 Gn_206 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210
 - 422 Gn_211 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216
 - 434 Gn_217 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221
 - 444 Gn_222 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227
 - 456 Gn_228 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232
 - 466 Gn_233 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238
 - 478 Gn_239 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243
 - 488 Gn_244 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249
 - 50 Gn_25 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30
 - 62 Gn_31 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37
 - 76 Gn_38 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43
 - 88 Gn_44 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5
 - 100 Gn_50 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55
 - 112 Gn_56 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60
 - 122 Gn_61 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66
 - 134 Gn_67 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71
 - 144 Gn_72 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77
 - 156 Gn_78 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82
 - 166 Gn_83 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88
 - 178 Gn_89 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93
 - 188 Gn_94 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99
 + Y_coolingwater_pumphouse + 0.5 d_coolingwater_pumphouse <= 0
_C1665: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
 + Gn_106 + Gn_107 + Gn_108 + Gn_109 + Gn_11 + Gn_110 + Gn_111 + Gn_112
 + Gn_113 + Gn_114 + Gn_115 + Gn_116 + Gn_117 + Gn_118 + Gn_119 + Gn_12
 + Gn_120 + Gn_121 + Gn_122 + Gn_123 + Gn_124 + Gn_125 + Gn_126 + Gn_127
 + Gn_128 + Gn_129 + Gn_13 + Gn_130 + Gn_131 + Gn_132 + Gn_133 + Gn_134
 + Gn_135 + Gn_136 + Gn_137 + Gn_138 + Gn_139 + Gn_14 + Gn_140 + Gn_141
 + Gn_142 + Gn_143 + Gn_144 + Gn_145 + Gn_146 + Gn_147 + Gn_148 + Gn_149
 + Gn_15 + Gn_150 + Gn_151 + Gn_152 + Gn_153 + Gn_154 + Gn_155 + Gn_156
 + Gn_157 + Gn_158 + Gn_159 + Gn_16 + Gn_160 + Gn_161 + Gn_162 + Gn_163
 + Gn_164 + Gn_165 + Gn_166 + Gn_167 + Gn_168 + Gn_169 + Gn_17 + Gn_170
 + Gn_171 + Gn_172 + Gn_173 + Gn_174 + Gn_175 + Gn_176 + Gn_177 + Gn_178
 + Gn_179 + Gn_18 + Gn_180 + Gn_181 + Gn_182 + Gn_183 + Gn_184 + Gn_185
 + Gn_186 + Gn_187 + Gn_188 + Gn_189 + Gn_19 + Gn_190 + Gn_191 + Gn_192
 + Gn_193 + Gn_194 + Gn_195 + Gn_196 + Gn_197 + Gn_198 + Gn_199 + Gn_2 + Gn_20
 + Gn_200 + Gn_201 + Gn_202 + Gn_203 + Gn_204 + Gn_205 + Gn_206 + Gn_207
 + Gn_208 + Gn_209 + Gn_21 + Gn_210 + Gn_211 + Gn_212 + Gn_213 + Gn_214
 + Gn_215 + Gn_216 + Gn_217 + Gn_218 + Gn_219 + Gn_22 + Gn_220 + Gn_221
 + Gn_222 + Gn_223 + Gn_224 + Gn_225 + Gn_226 + Gn_227 + Gn_228 + Gn_229
 + Gn_23 + Gn_230 + Gn_231 + Gn_232 + Gn_233 + Gn_234 + Gn_235 + Gn_236
 + Gn_237 + Gn_238 + Gn_239 + Gn_24 + Gn_240 + Gn_241 + Gn_242 + Gn_243
 + Gn_244 + Gn_245 + Gn_246 + Gn_247 + Gn_248 + Gn_249 + Gn_25 + Gn_26 + Gn_27
 + Gn_28 + Gn_29 + Gn_3 + Gn_30 + Gn_31 + Gn_32 + Gn_33 + Gn_34 + Gn_35
 + Gn_36 + Gn_37 + Gn_38 + Gn_39 + Gn_4 + Gn_40 + Gn_41 + Gn_42 + Gn_43
 + Gn_44 + Gn_45 + Gn_46 + Gn_47 + Gn_48 + Gn_49 + Gn_5 + Gn_50 + Gn_51
 + Gn_52 + Gn_53 + Gn_54 + Gn_55 + Gn_56 + Gn_57 + Gn_58 + Gn_59 + Gn_6
 + Gn_60 + Gn_61 + Gn_62 + Gn_63 + Gn_64 + Gn_65 + Gn_66 + Gn_67 + Gn_68
 + Gn_69 + Gn_7 + Gn_70 + Gn_71 + Gn_72 + Gn_73 + Gn_74 + Gn_75 + Gn_76
 + Gn_77 + Gn_78 + Gn_79 + Gn_8 + Gn_80 + Gn_81 + Gn_82 + Gn_83 + Gn_84
 + Gn_85 + Gn_86 + Gn_87 + Gn_88 + Gn_89 + Gn_9 + Gn_90 + Gn_91 + Gn_92
 + Gn_93 + Gn_94 + Gn_95 + Gn_96 + Gn_97 + Gn_98 + Gn_99 = 1
_C1666: - 55.1 Gn_1 - 5510 Gn_10 - 551000 Gn_100 - 562075.1 Gn_101
 - 573260.4 Gn_102 - 584555.9 Gn_103 - 595961.6 Gn_104 - 607477.5 Gn_105
 - 619103.6 Gn_106 - 630839.9 Gn_107 - 642686.4 Gn_108 - 654643.1 Gn_109
 - 6667.1 Gn_11 - 666710 Gn_110 - 678887.1 Gn_111 - 691174.4 Gn_112
 - 703571.9 Gn_113 - 716079.6 Gn_114 - 728697.5 Gn_115 - 741425.6 Gn_116
 - 754263.9 Gn_117 - 767212.4 Gn_118 - 780271.1 Gn_119 - 7934.4 Gn_12
 - 793440 Gn_120 - 806719.1 Gn_121 - 820108.4 Gn_122 - 833607.9 Gn_123
 - 847217.6 Gn_124 - 860937.5 Gn_125 - 874767.6 Gn_126 - 888707.9 Gn_127
 - 902758.4 Gn_128 - 916919.1 Gn_129 - 9311.9 Gn_13 - 931190 Gn_130
 - 945571.1 Gn_131 - 960062.4 Gn_132 - 974663.9 Gn_133 - 989375.6 Gn_134
 - 1004197.5 Gn_135 - 1019129.6 Gn_136 - 1034171.9 Gn_137 - 1049324.4 Gn_138
 - 1064587.1 Gn_139 - 10799.6 Gn_14 - 1079960 Gn_140 - 1095443.1 Gn_141
 - 1111036.4 Gn_142 - 1126739.9 Gn_143 - 1142553.6 Gn_144 - 1158477.5 Gn_145
 - 1174511.6 Gn_146 - 1190655.9 Gn_147 - 1206910.4 Gn_148 - 1223275.1 Gn_149
 - 12397.5 Gn_15 - 1239750 Gn_150 - 1256335.1 Gn_151 - 1273030.4 Gn_152
 - 1289835.9 Gn_153 - 1306751.6 Gn_154 - 1323777.5 Gn_155 - 1340913.6 Gn_156
 - 1358159.9 Gn_157 - 1375516.4 Gn_158 - 1392983.1 Gn_159 - 14105.6 Gn_16
 - 1410560 Gn_160 - 1428247.1 Gn_161 - 1446044.4 Gn_162 - 1463951.9 Gn_163
 - 1481969.6 Gn_164 - 1500097.5 Gn_165 - 1518335.6 Gn_166 - 1536683.9 Gn_167
 - 1555142.4 Gn_168 - 1573711.1 Gn_169 - 15923.9 Gn_17 - 1592390 Gn_170
 - 1611179.1 Gn_171 - 1630078.4 Gn_172 - 1649087.9 Gn_173 - 1668207.6 Gn_174
 - 1687437.5 Gn_175 - 1706777.6 Gn_176 - 1726227.9 Gn_177 - 1745788.4 Gn_178
 - 1765459.1 Gn_179 - 17852.4 Gn_18 - 1785240 Gn_180 - 1805131.1 Gn_181
 - 1825132.4 Gn_182 - 1845243.9 Gn_183 - 1865465.6 Gn_184 - 1885797.5 Gn_185
 - 1906239.6 Gn_186 - 1926791.9 Gn_187 - 1947454.4 Gn_188 - 1968227.1 Gn_189
 - 19891.1 Gn_19 - 1989110 Gn_190 - 2010103.1 Gn_191 - 2031206.4 Gn_192
 - 2052419.9 Gn_193 - 2073743.6 Gn_194 - 2095177.5 Gn_195 - 2116721.6 Gn_196
 - 2138375.9 Gn_197 - 2160140.4 Gn_198 - 2182015.1 Gn_199 - 220.4 Gn_2
 - 22040 Gn_20 - 2204000 Gn_200 - 2226095.1 Gn_201 - 2248300.4 Gn_202
 - 2270615.9 Gn_203 - 2293041.6 Gn_204 - 2315577.5 Gn_205 - 2338223.6 Gn_206
 - 2360979.9 Gn_207 - 2383846.4 Gn_208 - 2406823.1 Gn_209 - 24299.1 Gn_21
 - 2429910 Gn_210 - 2453107.1 Gn_211 - 2476414.4 Gn_212 - 2499831.9 Gn_213
 - 2523359.6 Gn_214 - 2546997.5 Gn_215 - 2570745.6 Gn_216 - 2594603.9 Gn_217
 - 2618572.4 Gn_218 - 2642651.1 Gn_219 - 26668.4 Gn_22 - 2666840 Gn_220
 - 2691139.1 Gn_221 - 2715548.4 Gn_222 - 2740067.9 Gn_223 - 2764697.6 Gn_224
 - 2789437.5 Gn_225 - 2814287.6 Gn_226 - 2839247.9 Gn_227 - 2864318.4 Gn_228
 - 2889499.1 Gn_229 - 29147.9 Gn_23 - 2914790 Gn_230 - 2940191.1 Gn_231
 - 2965702.4 Gn_232 - 2991323.9 Gn_233 - 3017055.6 Gn_234 - 3042897.5 Gn_235
 - 3068849.6 Gn_236 - 3094911.9 Gn_237 - 3121084.4 Gn_238 - 3147367.1 Gn_239
 - 31737.6 Gn_24 - 3173760 Gn_240 - 3200263.1 Gn_241 - 3226876.4 Gn_242
 - 3253599.9 Gn_243 - 3280433.6 Gn_244 - 3307377.5 Gn_245 - 3334431.6 Gn_246
 - 3361595.9 Gn_247 - 3388870.4 Gn_248 - 3416255.1 Gn_249 - 34437.5 Gn_25
 - 37247.6 Gn_26 - 40167.9 Gn_27 - 43198.4 Gn_28 - 46339.1 Gn_29 - 495.9 Gn_3
 - 49590 Gn_30 - 52951.1 Gn_31 - 56422.4 Gn_32 - 60003.9 Gn_33 - 63695.6 Gn_34
 - 67497.5 Gn_35 - 71409.6 Gn_36 - 75431.9 Gn_37 - 79564.4 Gn_38
 - 83807.1 Gn_39 - 881.6 Gn_4 - 88160 Gn_40 - 92623.1 Gn_41 - 97196.4 Gn_42
 - 101879.9 Gn_43 - 106673.6 Gn_44 - 111577.5 Gn_45 - 116591.6 Gn_46
 - 121715.9 Gn_47 - 126950.4 Gn_48 - 132295.1 Gn_49 - 1377.5 Gn_5
 - 137750 Gn_50 - 143315.1 Gn_51 - 148990.4 Gn_52 - 154775.9 Gn_53
 - 160671.6 Gn_54 - 166677.5 Gn_55 - 172793.6 Gn_56 - 179019.9 Gn_57
 - 185356.4 Gn_58 - 191803.1 Gn_59 - 1983.6 Gn_6 - 198360 Gn_60
 - 205027.1 Gn_61 - 211804.4 Gn_62 - 218691.9 Gn_63 - 225689.6 Gn_64
 - 232797.5 Gn_65 - 240015.6 Gn_66 - 247343.9 Gn_67 - 254782.4 Gn_68
 - 262331.1 Gn_69 - 2699.9 Gn_7 - 269990 Gn_70 - 277759.1 Gn_71
 - 285638.4 Gn_72 - 293627.9 Gn_73 - 301727.6 Gn_74 - 309937.5 Gn_75
 - 318257.6 Gn_76 - 326687.9 Gn_77 - 335228.4 Gn_78 - 343879.1 Gn_79
 - 3526.4 Gn_8 - 352640 Gn_80 - 361511.1 Gn_81 - 370492.4 Gn_82
 - 379583.9 Gn_83 - 388785.6 Gn_84 - 398097.5 Gn_85 - 407519.6 Gn_86
 - 417051.9 Gn_87 - 426694.4 Gn_88 - 436447.1 Gn_89 - 4463.1 Gn_9
 - 446310 Gn_90 - 456283.1 Gn_91 - 466366.4 Gn_92 - 476559.9 Gn_93
 - 486863.6 Gn_94 - 497277.5 Gn_95 - 507801.6 Gn_96 - 518435.9 Gn_97
 - 529180.4 Gn_98 - 540035.1 Gn_99 + TLC = 0
_C1667: - D_absorber_direct_contact_cooler
 + Din_absorber_direct_contact_cooler + Dout_absorber_direct_contact_cooler
 = 0
_C1668: Din_absorber_direct_contact_cooler
 - 52.9431462755 Psie_absorber_direct_contact_cooler <= 0
_C1669: Dout_absorber_direct_contact_cooler
 + 52.9431462755 Psie_absorber_direct_contact_cooler >= 52.9431462755
_C167: - A_direct_contact_cooler_stripper - B_direct_contact_cooler_stripper
 + D_direct_contact_cooler_stripper - L_direct_contact_cooler_stripper
 - R_direct_contact_cooler_stripper = 0
_C1670: Dout_absorber_direct_contact_cooler
 + 1000 Psie_absorber_direct_contact_cooler <= 1000
_C1671: 65715.3804554 Din_absorber_direct_contact_cooler
 - 3479179 Psie_absorber_direct_contact_cooler
 + Ve2_absorber_direct_contact_cooler = 0
_C1672: Ve2_absorber_absorber = 0
_C1673: - D_absorber_lean_amine_cooler + Din_absorber_lean_amine_cooler
 + Dout_absorber_lean_amine_cooler = 0
_C1674: Din_absorber_lean_amine_cooler
 - 52.9431462755 Psie_absorber_lean_amine_cooler <= 0
_C1675: Dout_absorber_lean_amine_cooler
 + 52.9431462755 Psie_absorber_lean_amine_cooler >= 52.9431462755
_C1676: Dout_absorber_lean_amine_cooler + 1000 Psie_absorber_lean_amine_cooler
 <= 1000
_C1677: 225928.072687 Din_absorber_lean_amine_cooler
 - 11961343 Psie_absorber_lean_amine_cooler + Ve2_absorber_lean_amine_cooler
 = 0
_C1678: - D_absorber_cross_heat_exchanger + Din_absorber_cross_heat_exchanger
 + Dout_absorber_cross_heat_exchanger = 0
_C1679: Din_absorber_cross_heat_exchanger
 - 52.9431462755 Psie_absorber_cross_heat_exchanger <= 0
_C168: D_direct_contact_cooler_stripper - D_stripper_direct_contact_cooler = 0
_C1680: Dout_absorber_cross_heat_exchanger
 + 52.9431462755 Psie_absorber_cross_heat_exchanger >= 52.9431462755
_C1681: Dout_absorber_cross_heat_exchanger
 + 1000 Psie_absorber_cross_heat_exchanger <= 1000
_C1682: 38857.9286409 Din_absorber_cross_heat_exchanger
 - 2057261 Psie_absorber_cross_heat_exchanger
 + Ve2_absorber_cross_heat_exchanger = 0
_C1683: - D_absorber_stripper + Din_absorber_stripper + Dout_absorber_stripper
 = 0
_C1684: Din_absorber_stripper - 52.9431462755 Psie_absorber_stripper <= 0
_C1685: Dout_absorber_stripper + 52.9431462755 Psie_absorber_stripper
 >= 52.9431462755
_C1686: Dout_absorber_stripper + 1000 Psie_absorber_stripper <= 1000
_C1687: 489493.330546 Din_absorber_stripper - 25915317 Psie_absorber_stripper
 + Ve2_absorber_stripper = 0
_C1688: - D_absorber_storage_tank + Din_absorber_storage_tank
 + Dout_absorber_storage_tank = 0
_C1689: Din_absorber_storage_tank - 52.9431462755 Psie_absorber_storage_tank
 <= 0
_C169: 1000 E1_direct_contact_cooler_stripper
 + 1000 E2_direct_contact_cooler_stripper + X_direct_contact_cooler
 - X_stripper - 0.5 l_direct_contact_cooler - 0.5 l_stripper >= 15
_C1690: Dout_absorber_storage_tank + 52.9431462755 Psie_absorber_storage_tank
 >= 52.9431462755
_C1691: Dout_absorber_storage_tank + 1000 Psie_absorber_storage_tank <= 1000
_C1692: 168339.542074 Din_absorber_storage_tank
 - 8912425 Psie_absorber_storage_tank + Ve2_absorber_storage_tank = 0
_C1693: - D_absorber_reclaimer + Din_absorber_reclaimer
 + Dout_absorber_reclaimer = 0
_C1694: Din_absorber_reclaimer - 52.9431462755 Psie_absorber_reclaimer <= 0
_C1695: Dout_absorber_reclaimer + 52.9431462755 Psie_absorber_reclaimer
 >= 52.9431462755
_C1696: Dout_absorber_reclaimer + 1000 Psie_absorber_reclaimer <= 1000
_C1697: 49335.9421144 Din_absorber_reclaimer - 2612000 Psie_absorber_reclaimer
 + Ve2_absorber_reclaimer = 0
_C1698: - D_absorber_compressor_house + Din_absorber_compressor_house
 + Dout_absorber_compressor_house = 0
_C1699: Din_absorber_compressor_house
 - 52.9431462755 Psie_absorber_compressor_house <= 0
_C17: 0 O_stripper + l_stripper = 60
_C170: - 1000 E1_direct_contact_cooler_stripper
 + 1000 E2_direct_contact_cooler_stripper - X_direct_contact_cooler
 + X_stripper - 0.5 l_direct_contact_cooler - 0.5 l_stripper >= -985
_C1700: Dout_absorber_compressor_house
 + 52.9431462755 Psie_absorber_compressor_house >= 52.9431462755
_C1701: Dout_absorber_compressor_house + 1000 Psie_absorber_compressor_house
 <= 1000
_C1702: 515693.379043 Din_absorber_compressor_house
 - 27302430 Psie_absorber_compressor_house + Ve2_absorber_compressor_house = 0
_C1703: - D_absorber_compressor_KOdrum + Din_absorber_compressor_KOdrum
 + Dout_absorber_compressor_KOdrum = 0
_C1704: Din_absorber_compressor_KOdrum
 - 52.9431462755 Psie_absorber_compressor_KOdrum <= 0
_C1705: Dout_absorber_compressor_KOdrum
 + 52.9431462755 Psie_absorber_compressor_KOdrum >= 52.9431462755
_C1706: Dout_absorber_compressor_KOdrum + 1000 Psie_absorber_compressor_KOdrum
 <= 1000
_C1707: 98262.6150121 Din_absorber_compressor_KOdrum
 - 5202332 Psie_absorber_compressor_KOdrum + Ve2_absorber_compressor_KOdrum
 = 0
_C1708: - D_absorber_compressor_interstage_cooler
 + Din_absorber_compressor_interstage_cooler
 + Dout_absorber_compressor_interstage_cooler = 0
_C1709: Din_absorber_compressor_interstage_cooler
 - 52.9431462755 Psie_absorber_compressor_interstage_cooler <= 0
_C171: 1000 E1_direct_contact_cooler_stripper
 - 1000 E2_direct_contact_cooler_stripper + Y_direct_contact_cooler
 - Y_stripper - 0.5 d_direct_contact_cooler - 0.5 d_stripper >= -985
_C1710: Dout_absorber_compressor_interstage_cooler
 + 52.9431462755 Psie_absorber_compressor_interstage_cooler >= 52.9431462755
_C1711: Dout_absorber_compressor_interstage_cooler
 + 1000 Psie_absorber_compressor_interstage_cooler <= 1000
_C1712: 21937.2682152 Din_absorber_compressor_interstage_cooler
 - 1161428 Psie_absorber_compressor_interstage_cooler
 + Ve2_absorber_compressor_interstage_cooler = 0
_C1713: - D_absorber_TEG_dehydration + Din_absorber_TEG_dehydration
 + Dout_absorber_TEG_dehydration = 0
_C1714: Din_absorber_TEG_dehydration
 - 52.9431462755 Psie_absorber_TEG_dehydration <= 0
_C1715: Dout_absorber_TEG_dehydration
 + 52.9431462755 Psie_absorber_TEG_dehydration >= 52.9431462755
_C1716: Dout_absorber_TEG_dehydration + 1000 Psie_absorber_TEG_dehydration
 <= 1000
_C1717: 60064.4318238 Din_absorber_TEG_dehydration
 - 3180000 Psie_absorber_TEG_dehydration + Ve2_absorber_TEG_dehydration = 0
_C1718: - D_absorber_cooling_tower + Din_absorber_cooling_tower
 + Dout_absorber_cooling_tower = 0
_C1719: Din_absorber_cooling_tower - 52.9431462755 Psie_absorber_cooling_tower
 <= 0
_C172: - 1000 E1_direct_contact_cooler_stripper
 - 1000 E2_direct_contact_cooler_stripper - Y_direct_contact_cooler
 + Y_stripper - 0.5 d_direct_contact_cooler - 0.5 d_stripper >= -1985
_C1720: Dout_absorber_cooling_tower
 + 52.9431462755 Psie_absorber_cooling_tower >= 52.9431462755
_C1721: Dout_absorber_cooling_tower + 1000 Psie_absorber_cooling_tower <= 1000
_C1722: 542306.115519 Din_absorber_cooling_tower
 - 28711392 Psie_absorber_cooling_tower + Ve2_absorber_cooling_tower = 0
_C1723: - D_absorber_coolingwater_pumphouse
 + Din_absorber_coolingwater_pumphouse + Dout_absorber_coolingwater_pumphouse
 = 0
_C1724: Din_absorber_coolingwater_pumphouse
 - 52.9431462755 Psie_absorber_coolingwater_pumphouse <= 0
_C1725: Dout_absorber_coolingwater_pumphouse
 + 52.9431462755 Psie_absorber_coolingwater_pumphouse >= 52.9431462755
_C1726: Dout_absorber_coolingwater_pumphouse
 + 1000 Psie_absorber_coolingwater_pumphouse <= 1000
_C1727: 542306.115519 Din_absorber_coolingwater_pumphouse
 - 28711392 Psie_absorber_coolingwater_pumphouse
 + Ve2_absorber_coolingwater_pumphouse = 0
_C1728: - D_stripper_direct_contact_cooler
 + Din_stripper_direct_contact_cooler + Dout_stripper_direct_contact_cooler
 = 0
_C1729: Din_stripper_direct_contact_cooler
 - 76.0630289159 Psie_stripper_direct_contact_cooler <= 0
_C173: L_direct_contact_cooler_stripper - R_stripper_direct_contact_cooler = 0
_C1730: Dout_stripper_direct_contact_cooler
 + 76.0630289159 Psie_stripper_direct_contact_cooler >= 76.0630289159
_C1731: Dout_stripper_direct_contact_cooler
 + 1000 Psie_stripper_direct_contact_cooler <= 1000
_C1732: 45740.736986 Din_stripper_direct_contact_cooler
 - 3479179 Psie_stripper_direct_contact_cooler
 + Ve2_stripper_direct_contact_cooler = 0
_C1733: - D_stripper_absorber + Din_stripper_absorber + Dout_stripper_absorber
 = 0
_C1734: Din_stripper_absorber - 76.0630289159 Psie_stripper_absorber <= 0
_C1735: Dout_stripper_absorber + 76.0630289159 Psie_stripper_absorber
 >= 76.0630289159
_C1736: Dout_stripper_absorber + 1000 Psie_stripper_absorber <= 1000
_C1737: 847382.321197 Din_stripper_absorber - 64454466 Psie_stripper_absorber
 + Ve2_stripper_absorber = 0
_C1738: - D_stripper_lean_amine_cooler + Din_stripper_lean_amine_cooler
 + Dout_stripper_lean_amine_cooler = 0
_C1739: Din_stripper_lean_amine_cooler
 - 76.0630289159 Psie_stripper_lean_amine_cooler <= 0
_C174: - L_stripper_direct_contact_cooler + R_direct_contact_cooler_stripper
 = 0
_C1740: Dout_stripper_lean_amine_cooler
 + 76.0630289159 Psie_stripper_lean_amine_cooler >= 76.0630289159
_C1741: Dout_stripper_lean_amine_cooler + 1000 Psie_stripper_lean_amine_cooler
 <= 1000
_C1742: 157255.675595 Din_stripper_lean_amine_cooler
 - 11961343 Psie_stripper_lean_amine_cooler + Ve2_stripper_lean_amine_cooler
 = 0
_C1743: - D_stripper_cross_heat_exchanger + Din_stripper_cross_heat_exchanger
 + Dout_stripper_cross_heat_exchanger = 0
_C1744: Din_stripper_cross_heat_exchanger
 - 76.0630289159 Psie_stripper_cross_heat_exchanger <= 0
_C1745: Dout_stripper_cross_heat_exchanger
 + 76.0630289159 Psie_stripper_cross_heat_exchanger >= 76.0630289159
_C1746: Dout_stripper_cross_heat_exchanger
 + 1000 Psie_stripper_cross_heat_exchanger <= 1000
_C1747: 27046.7930258 Din_stripper_cross_heat_exchanger
 - 2057261 Psie_stripper_cross_heat_exchanger
 + Ve2_stripper_cross_heat_exchanger = 0
_C1748: Ve2_stripper_stripper = 0
_C1749: - D_stripper_storage_tank + Din_stripper_storage_tank
 + Dout_stripper_storage_tank = 0
_C175: A_direct_contact_cooler_stripper - B_stripper_direct_contact_cooler = 0
_C1750: Din_stripper_storage_tank - 76.0630289159 Psie_stripper_storage_tank
 <= 0
_C1751: Dout_stripper_storage_tank + 76.0630289159 Psie_stripper_storage_tank
 >= 76.0630289159
_C1752: Dout_stripper_storage_tank + 1000 Psie_stripper_storage_tank <= 1000
_C1753: 117171.57635 Din_stripper_storage_tank
 - 8912425 Psie_stripper_storage_tank + Ve2_stripper_storage_tank = 0
_C1754: - D_stripper_reclaimer + Din_stripper_reclaimer
 + Dout_stripper_reclaimer = 0
_C1755: Din_stripper_reclaimer - 76.0630289159 Psie_stripper_reclaimer <= 0
_C1756: Dout_stripper_reclaimer + 76.0630289159 Psie_stripper_reclaimer
 >= 76.0630289159
_C1757: Dout_stripper_reclaimer + 1000 Psie_stripper_reclaimer <= 1000
_C1758: 34339.9419827 Din_stripper_reclaimer - 2612000 Psie_stripper_reclaimer
 + Ve2_stripper_reclaimer = 0
_C1759: - D_stripper_compressor_house + Din_stripper_compressor_house
 + Dout_stripper_compressor_house = 0
_C176: - A_stripper_direct_contact_cooler + B_direct_contact_cooler_stripper
 = 0
_C1760: Din_stripper_compressor_house
 - 76.0630289159 Psie_stripper_compressor_house <= 0
_C1761: Dout_stripper_compressor_house
 + 76.0630289159 Psie_stripper_compressor_house >= 76.0630289159
_C1762: Dout_stripper_compressor_house + 1000 Psie_stripper_compressor_house
 <= 1000
_C1763: 358944.81707 Din_stripper_compressor_house
 - 27302430 Psie_stripper_compressor_house + Ve2_stripper_compressor_house = 0
_C1764: - D_stripper_compressor_KOdrum + Din_stripper_compressor_KOdrum
 + Dout_stripper_compressor_KOdrum = 0
_C1765: Din_stripper_compressor_KOdrum
 - 76.0630289159 Psie_stripper_compressor_KOdrum <= 0
_C1766: Dout_stripper_compressor_KOdrum
 + 76.0630289159 Psie_stripper_compressor_KOdrum >= 76.0630289159
_C1767: Dout_stripper_compressor_KOdrum + 1000 Psie_stripper_compressor_KOdrum
 <= 1000
_C1768: 68395.0149521 Din_stripper_compressor_KOdrum
 - 5202332 Psie_stripper_compressor_KOdrum + Ve2_stripper_compressor_KOdrum
 = 0
_C1769: - D_stripper_compressor_interstage_cooler
 + Din_stripper_compressor_interstage_cooler
 + Dout_stripper_compressor_interstage_cooler = 0
_C177: Wx_direct_contact_cooler_stripper + Wx_stripper_direct_contact_cooler
 = 1
_C1770: Din_stripper_compressor_interstage_cooler
 - 76.0630289159 Psie_stripper_compressor_interstage_cooler <= 0
_C1771: Dout_stripper_compressor_interstage_cooler
 + 76.0630289159 Psie_stripper_compressor_interstage_cooler >= 76.0630289159
_C1772: Dout_stripper_compressor_interstage_cooler
 + 1000 Psie_stripper_compressor_interstage_cooler <= 1000
_C1773: 15269.284126 Din_stripper_compressor_interstage_cooler
 - 1161428 Psie_stripper_compressor_interstage_cooler
 + Ve2_stripper_compressor_interstage_cooler = 0
_C1774: - D_stripper_TEG_dehydration + Din_stripper_TEG_dehydration
 + Dout_stripper_TEG_dehydration = 0
_C1775: Din_stripper_TEG_dehydration
 - 76.0630289159 Psie_stripper_TEG_dehydration <= 0
_C1776: Dout_stripper_TEG_dehydration
 + 76.0630289159 Psie_stripper_TEG_dehydration >= 76.0630289159
_C1777: Dout_stripper_TEG_dehydration + 1000 Psie_stripper_TEG_dehydration
 <= 1000
_C1778: 41807.4331949 Din_stripper_TEG_dehydration
 - 3180000 Psie_stripper_TEG_dehydration + Ve2_stripper_TEG_dehydration = 0
_C1779: - D_stripper_cooling_tower + Din_stripper_cooling_tower
 + Dout_stripper_cooling_tower = 0
_C178: Wy_direct_contact_cooler_stripper + Wy_stripper_direct_contact_cooler
 = 1
_C1780: Din_stripper_cooling_tower - 76.0630289159 Psie_stripper_cooling_tower
 <= 0
_C1781: Dout_stripper_cooling_tower
 + 76.0630289159 Psie_stripper_cooling_tower >= 76.0630289159
_C1782: Dout_stripper_cooling_tower + 1000 Psie_stripper_cooling_tower <= 1000
_C1783: 377468.428608 Din_stripper_cooling_tower
 - 28711392 Psie_stripper_cooling_tower + Ve2_stripper_cooling_tower = 0
_C1784: - D_stripper_coolingwater_pumphouse
 + Din_stripper_coolingwater_pumphouse + Dout_stripper_coolingwater_pumphouse
 = 0
_C1785: Din_stripper_coolingwater_pumphouse
 - 76.0630289159 Psie_stripper_coolingwater_pumphouse <= 0
_C1786: Dout_stripper_coolingwater_pumphouse
 + 76.0630289159 Psie_stripper_coolingwater_pumphouse >= 76.0630289159
_C1787: Dout_stripper_coolingwater_pumphouse
 + 1000 Psie_stripper_coolingwater_pumphouse <= 1000
_C1788: 377468.428608 Din_stripper_coolingwater_pumphouse
 - 28711392 Psie_stripper_coolingwater_pumphouse
 + Ve2_stripper_coolingwater_pumphouse = 0
_C1789: - D_storage_tank_direct_contact_cooler
 + Din_storage_tank_direct_contact_cooler
 + Dout_storage_tank_direct_contact_cooler = 0
_C179: - L_absorber_stripper + R_absorber_stripper - X_absorber + X_stripper
 = 0
_C1790: Din_storage_tank_direct_contact_cooler
 - 40.9449584123 Psie_storage_tank_direct_contact_cooler <= 0
_C1791: Dout_storage_tank_direct_contact_cooler
 + 40.9449584123 Psie_storage_tank_direct_contact_cooler >= 40.9449584123
_C1792: Dout_storage_tank_direct_contact_cooler
 + 1000 Psie_storage_tank_direct_contact_cooler <= 1000
_C1793: 84972.0975404 Din_storage_tank_direct_contact_cooler
 - 3479179 Psie_storage_tank_direct_contact_cooler
 + Ve2_storage_tank_direct_contact_cooler = 0
_C1794: - D_storage_tank_absorber + Din_storage_tank_absorber
 + Dout_storage_tank_absorber = 0
_C1795: Din_storage_tank_absorber - 40.9449584123 Psie_storage_tank_absorber
 <= 0
_C1796: Dout_storage_tank_absorber + 40.9449584123 Psie_storage_tank_absorber
 >= 40.9449584123
_C1797: Dout_storage_tank_absorber + 1000 Psie_storage_tank_absorber <= 1000
_C1798: 1574173.43916 Din_storage_tank_absorber
 - 64454466 Psie_storage_tank_absorber + Ve2_storage_tank_absorber = 0
_C1799: - D_storage_tank_lean_amine_cooler
 + Din_storage_tank_lean_amine_cooler + Dout_storage_tank_lean_amine_cooler
 = 0
_C18: d_stripper + l_stripper = 120
_C180: A_absorber_stripper - B_absorber_stripper - Y_absorber + Y_stripper = 0
_C1800: Din_storage_tank_lean_amine_cooler
 - 40.9449584123 Psie_storage_tank_lean_amine_cooler <= 0
_C1801: Dout_storage_tank_lean_amine_cooler
 + 40.9449584123 Psie_storage_tank_lean_amine_cooler >= 40.9449584123
_C1802: Dout_storage_tank_lean_amine_cooler
 + 1000 Psie_storage_tank_lean_amine_cooler <= 1000
_C1803: 292132.254222 Din_storage_tank_lean_amine_cooler
 - 11961343 Psie_storage_tank_lean_amine_cooler
 + Ve2_storage_tank_lean_amine_cooler = 0
_C1804: - D_storage_tank_cross_heat_exchanger
 + Din_storage_tank_cross_heat_exchanger
 + Dout_storage_tank_cross_heat_exchanger = 0
_C1805: Din_storage_tank_cross_heat_exchanger
 - 40.9449584123 Psie_storage_tank_cross_heat_exchanger <= 0
_C1806: Dout_storage_tank_cross_heat_exchanger
 + 40.9449584123 Psie_storage_tank_cross_heat_exchanger >= 40.9449584123
_C1807: Dout_storage_tank_cross_heat_exchanger
 + 1000 Psie_storage_tank_cross_heat_exchanger <= 1000
_C1808: 50244.549751 Din_storage_tank_cross_heat_exchanger
 - 2057261 Psie_storage_tank_cross_heat_exchanger
 + Ve2_storage_tank_cross_heat_exchanger = 0
_C1809: - D_storage_tank_stripper + Din_storage_tank_stripper
 + Dout_storage_tank_stripper = 0
_C181: R_absorber_stripper - 1000 Wx_absorber_stripper <= 0
_C1810: Din_storage_tank_stripper - 40.9449584123 Psie_storage_tank_stripper
 <= 0
_C1811: Dout_storage_tank_stripper + 40.9449584123 Psie_storage_tank_stripper
 >= 40.9449584123
_C1812: Dout_storage_tank_stripper + 1000 Psie_storage_tank_stripper <= 1000
_C1813: 632930.597683 Din_storage_tank_stripper
 - 25915317 Psie_storage_tank_stripper + Ve2_storage_tank_stripper = 0
_C1814: Ve2_storage_tank_storage_tank = 0
_C1815: - D_storage_tank_reclaimer + Din_storage_tank_reclaimer
 + Dout_storage_tank_reclaimer = 0
_C1816: Din_storage_tank_reclaimer - 40.9449584123 Psie_storage_tank_reclaimer
 <= 0
_C1817: Dout_storage_tank_reclaimer
 + 40.9449584123 Psie_storage_tank_reclaimer >= 40.9449584123
_C1818: Dout_storage_tank_reclaimer + 1000 Psie_storage_tank_reclaimer <= 1000
_C1819: 63792.9576994 Din_storage_tank_reclaimer
 - 2612000 Psie_storage_tank_reclaimer + Ve2_storage_tank_reclaimer = 0
_C182: L_absorber_stripper + 1000 Wx_absorber_stripper <= 1000
_C1820: - D_storage_tank_compressor_house + Din_storage_tank_compressor_house
 + Dout_storage_tank_compressor_house = 0
_C1821: Din_storage_tank_compressor_house
 - 40.9449584123 Psie_storage_tank_compressor_house <= 0
_C1822: Dout_storage_tank_compressor_house
 + 40.9449584123 Psie_storage_tank_compressor_house >= 40.9449584123
_C1823: Dout_storage_tank_compressor_house
 + 1000 Psie_storage_tank_compressor_house <= 1000
_C1824: 666808.101868 Din_storage_tank_compressor_house
 - 27302430 Psie_storage_tank_compressor_house
 + Ve2_storage_tank_compressor_house = 0
_C1825: - D_storage_tank_compressor_KOdrum
 + Din_storage_tank_compressor_KOdrum + Dout_storage_tank_compressor_KOdrum
 = 0
_C1826: Din_storage_tank_compressor_KOdrum
 - 40.9449584123 Psie_storage_tank_compressor_KOdrum <= 0
_C1827: Dout_storage_tank_compressor_KOdrum
 + 40.9449584123 Psie_storage_tank_compressor_KOdrum >= 40.9449584123
_C1828: Dout_storage_tank_compressor_KOdrum
 + 1000 Psie_storage_tank_compressor_KOdrum <= 1000
_C1829: 127056.717157 Din_storage_tank_compressor_KOdrum
 - 5202332 Psie_storage_tank_compressor_KOdrum
 + Ve2_storage_tank_compressor_KOdrum = 0
_C183: A_absorber_stripper - 1000 Wy_absorber_stripper <= 0
_C1830: - D_storage_tank_compressor_interstage_cooler
 + Din_storage_tank_compressor_interstage_cooler
 + Dout_storage_tank_compressor_interstage_cooler = 0
_C1831: Din_storage_tank_compressor_interstage_cooler
 - 40.9449584123 Psie_storage_tank_compressor_interstage_cooler <= 0
_C1832: Dout_storage_tank_compressor_interstage_cooler
 + 40.9449584123 Psie_storage_tank_compressor_interstage_cooler
 >= 40.9449584123
_C1833: Dout_storage_tank_compressor_interstage_cooler
 + 1000 Psie_storage_tank_compressor_interstage_cooler <= 1000
_C1834: 28365.5923717 Din_storage_tank_compressor_interstage_cooler
 - 1161428 Psie_storage_tank_compressor_interstage_cooler
 + Ve2_storage_tank_compressor_interstage_cooler = 0
_C1835: - D_storage_tank_TEG_dehydration + Din_storage_tank_TEG_dehydration
 + Dout_storage_tank_TEG_dehydration = 0
_C1836: Din_storage_tank_TEG_dehydration
 - 40.9449584123 Psie_storage_tank_TEG_dehydration <= 0
_C1837: Dout_storage_tank_TEG_dehydration
 + 40.9449584123 Psie_storage_tank_TEG_dehydration >= 40.9449584123
_C1838: Dout_storage_tank_TEG_dehydration
 + 1000 Psie_storage_tank_TEG_dehydration <= 1000
_C1839: 77665.2394655 Din_storage_tank_TEG_dehydration
 - 3180000 Psie_storage_tank_TEG_dehydration
 + Ve2_storage_tank_TEG_dehydration = 0
_C184: B_absorber_stripper + 1000 Wy_absorber_stripper <= 1000
_C1840: - D_storage_tank_cooling_tower + Din_storage_tank_cooling_tower
 + Dout_storage_tank_cooling_tower = 0
_C1841: Din_storage_tank_cooling_tower
 - 40.9449584123 Psie_storage_tank_cooling_tower <= 0
_C1842: Dout_storage_tank_cooling_tower
 + 40.9449584123 Psie_storage_tank_cooling_tower >= 40.9449584123
_C1843: Dout_storage_tank_cooling_tower + 1000 Psie_storage_tank_cooling_tower
 <= 1000
_C1844: 701219.224865 Din_storage_tank_cooling_tower
 - 28711392 Psie_storage_tank_cooling_tower + Ve2_storage_tank_cooling_tower
 = 0
_C1845: - D_storage_tank_coolingwater_pumphouse
 + Din_storage_tank_coolingwater_pumphouse
 + Dout_storage_tank_coolingwater_pumphouse = 0
_C1846: Din_storage_tank_coolingwater_pumphouse
 - 40.9449584123 Psie_storage_tank_coolingwater_pumphouse <= 0
_C1847: Dout_storage_tank_coolingwater_pumphouse
 + 40.9449584123 Psie_storage_tank_coolingwater_pumphouse >= 40.9449584123
_C1848: Dout_storage_tank_coolingwater_pumphouse
 + 1000 Psie_storage_tank_coolingwater_pumphouse <= 1000
_C1849: 701219.224865 Din_storage_tank_coolingwater_pumphouse
 - 28711392 Psie_storage_tank_coolingwater_pumphouse
 + Ve2_storage_tank_coolingwater_pumphouse = 0
_C185: - A_absorber_stripper - B_absorber_stripper + D_absorber_stripper
 - L_absorber_stripper - R_absorber_stripper = 0
_C1850: - Ve2_absorber_TEG_dehydration - Ve2_absorber_absorber
 - Ve2_absorber_compressor_KOdrum - Ve2_absorber_compressor_house
 - Ve2_absorber_compressor_interstage_cooler - Ve2_absorber_cooling_tower
 - Ve2_absorber_coolingwater_pumphouse - Ve2_absorber_cross_heat_exchanger
 - Ve2_absorber_direct_contact_cooler - Ve2_absorber_lean_amine_cooler
 - Ve2_absorber_reclaimer - Ve2_absorber_storage_tank - Ve2_absorber_stripper
 + Ve_absorber = 64454466
_C1851: Omega0_absorber - 0.00424705739876 Ve_absorber = 0
_C1852: - Ve2_stripper_TEG_dehydration - Ve2_stripper_absorber
 - Ve2_stripper_compressor_KOdrum - Ve2_stripper_compressor_house
 - Ve2_stripper_compressor_interstage_cooler - Ve2_stripper_cooling_tower
 - Ve2_stripper_coolingwater_pumphouse - Ve2_stripper_cross_heat_exchanger
 - Ve2_stripper_direct_contact_cooler - Ve2_stripper_lean_amine_cooler
 - Ve2_stripper_reclaimer - Ve2_stripper_storage_tank - Ve2_stripper_stripper
 + Ve_stripper = 25915317
_C1853: Omega0_stripper - 0.00601368634501 Ve_stripper = 0
_C1854: - Ve2_storage_tank_TEG_dehydration - Ve2_storage_tank_absorber
 - Ve2_storage_tank_compressor_KOdrum - Ve2_storage_tank_compressor_house
 - Ve2_storage_tank_compressor_interstage_cooler
 - Ve2_storage_tank_cooling_tower - Ve2_storage_tank_coolingwater_pumphouse
 - Ve2_storage_tank_cross_heat_exchanger
 - Ve2_storage_tank_direct_contact_cooler - Ve2_storage_tank_lean_amine_cooler
 - Ve2_storage_tank_reclaimer - Ve2_storage_tank_storage_tank
 - Ve2_storage_tank_stripper + Ve_storage_tank = 8912425
_C1855: Omega0_storage_tank - 0.00424705739876 Ve_storage_tank = 0
_C1856: - Omega0_absorber - Omega0_storage_tank - Omega0_stripper + SumOmega
 = 0
_C1857: - D_absorber_direct_contact_cooler
 + Dc_in_absorber_direct_contact_cooler
 + Dc_out_absorber_direct_contact_cooler = 0
_C1858: Dc_in_absorber_direct_contact_cooler
 - 1895.28430182 Psic_absorber_direct_contact_cooler_MEA <= 0
_C1859: Dc_out_absorber_direct_contact_cooler
 + 1895.28430182 Psic_absorber_direct_contact_cooler_MEA >= 1895.28430182
_C186: D_absorber_stripper - D_stripper_absorber = 0
_C1860: Dc_out_absorber_direct_contact_cooler
 + 1000 Psic_absorber_direct_contact_cooler_MEA <= 1000
_C1861: 94.9725589069 Dc_in_absorber_direct_contact_cooler
 - 180000 Psic_absorber_direct_contact_cooler_MEA
 + Vlc2_absorber_direct_contact_cooler_MEA = 0
_C1862: Vlc2_absorber_absorber_MEA = 0
_C1863: - D_absorber_lean_amine_cooler + Dc_in_absorber_lean_amine_cooler
 + Dc_out_absorber_lean_amine_cooler = 0
_C1864: Dc_in_absorber_lean_amine_cooler
 - 1895.28430182 Psic_absorber_lean_amine_cooler_MEA <= 0
_C1865: Dc_out_absorber_lean_amine_cooler
 + 1895.28430182 Psic_absorber_lean_amine_cooler_MEA >= 1895.28430182
_C1866: Dc_out_absorber_lean_amine_cooler
 + 1000 Psic_absorber_lean_amine_cooler_MEA <= 1000
_C1867: 94.9725589069 Dc_in_absorber_lean_amine_cooler
 - 180000 Psic_absorber_lean_amine_cooler_MEA
 + Vlc2_absorber_lean_amine_cooler_MEA = 0
_C1868: - D_absorber_cross_heat_exchanger
 + Dc_in_absorber_cross_heat_exchanger + Dc_out_absorber_cross_heat_exchanger
 = 0
_C1869: Dc_in_absorber_cross_heat_exchanger
 - 1895.28430182 Psic_absorber_cross_heat_exchanger_MEA <= 0
_C187: 1000 E1_absorber_stripper + 1000 E2_absorber_stripper + X_absorber
 - X_stripper - 0.5 l_absorber - 0.5 l_stripper >= 15
_C1870: Dc_out_absorber_cross_heat_exchanger
 + 1895.28430182 Psic_absorber_cross_heat_exchanger_MEA >= 1895.28430182
_C1871: Dc_out_absorber_cross_heat_exchanger
 + 1000 Psic_absorber_cross_heat_exchanger_MEA <= 1000
_C1872: 94.9725589069 Dc_in_absorber_cross_heat_exchanger
 - 180000 Psic_absorber_cross_heat_exchanger_MEA
 + Vlc2_absorber_cross_heat_exchanger_MEA = 0
_C1873: - D_absorber_stripper + Dc_in_absorber_stripper
 + Dc_out_absorber_stripper = 0
_C1874: Dc_in_absorber_stripper - 1895.28430182 Psic_absorber_stripper_MEA
 <= 0
_C1875: Dc_out_absorber_stripper + 1895.28430182 Psic_absorber_stripper_MEA
 >= 1895.28430182
_C1876: Dc_out_absorber_stripper + 1000 Psic_absorber_stripper_MEA <= 1000
_C1877: 94.9725589069 Dc_in_absorber_stripper
 - 180000 Psic_absorber_stripper_MEA + Vlc2_absorber_stripper_MEA = 0
_C1878: - D_absorber_storage_tank + Dc_in_absorber_storage_tank
 + Dc_out_absorber_storage_tank = 0
_C1879: Dc_in_absorber_storage_tank
 - 1895.28430182 Psic_absorber_storage_tank_MEA <= 0
_C188: - 1000 E1_absorber_stripper + 1000 E2_absorber_stripper - X_absorber
 + X_stripper - 0.5 l_absorber - 0.5 l_stripper >= -985
_C1880: Dc_out_absorber_storage_tank
 + 1895.28430182 Psic_absorber_storage_tank_MEA >= 1895.28430182
_C1881: Dc_out_absorber_storage_tank + 1000 Psic_absorber_storage_tank_MEA
 <= 1000
_C1882: 94.9725589069 Dc_in_absorber_storage_tank
 - 180000 Psic_absorber_storage_tank_MEA + Vlc2_absorber_storage_tank_MEA = 0
_C1883: - D_absorber_reclaimer + Dc_in_absorber_reclaimer
 + Dc_out_absorber_reclaimer = 0
_C1884: Dc_in_absorber_reclaimer - 1895.28430182 Psic_absorber_reclaimer_MEA
 <= 0
_C1885: Dc_out_absorber_reclaimer + 1895.28430182 Psic_absorber_reclaimer_MEA
 >= 1895.28430182
_C1886: Dc_out_absorber_reclaimer + 1000 Psic_absorber_reclaimer_MEA <= 1000
_C1887: 94.9725589069 Dc_in_absorber_reclaimer
 - 180000 Psic_absorber_reclaimer_MEA + Vlc2_absorber_reclaimer_MEA = 0
_C1888: - D_absorber_compressor_house + Dc_in_absorber_compressor_house
 + Dc_out_absorber_compressor_house = 0
_C1889: Dc_in_absorber_compressor_house
 - 1895.28430182 Psic_absorber_compressor_house_MEA <= 0
_C189: 1000 E1_absorber_stripper - 1000 E2_absorber_stripper + Y_absorber
 - Y_stripper - 0.5 d_absorber - 0.5 d_stripper >= -985
_C1890: Dc_out_absorber_compressor_house
 + 1895.28430182 Psic_absorber_compressor_house_MEA >= 1895.28430182
_C1891: Dc_out_absorber_compressor_house
 + 1000 Psic_absorber_compressor_house_MEA <= 1000
_C1892: 94.9725589069 Dc_in_absorber_compressor_house
 - 180000 Psic_absorber_compressor_house_MEA
 + Vlc2_absorber_compressor_house_MEA = 0
_C1893: - D_absorber_compressor_KOdrum + Dc_in_absorber_compressor_KOdrum
 + Dc_out_absorber_compressor_KOdrum = 0
_C1894: Dc_in_absorber_compressor_KOdrum
 - 1895.28430182 Psic_absorber_compressor_KOdrum_MEA <= 0
_C1895: Dc_out_absorber_compressor_KOdrum
 + 1895.28430182 Psic_absorber_compressor_KOdrum_MEA >= 1895.28430182
_C1896: Dc_out_absorber_compressor_KOdrum
 + 1000 Psic_absorber_compressor_KOdrum_MEA <= 1000
_C1897: 94.9725589069 Dc_in_absorber_compressor_KOdrum
 - 180000 Psic_absorber_compressor_KOdrum_MEA
 + Vlc2_absorber_compressor_KOdrum_MEA = 0
_C1898: - D_absorber_compressor_interstage_cooler
 + Dc_in_absorber_compressor_interstage_cooler
 + Dc_out_absorber_compressor_interstage_cooler = 0
_C1899: Dc_in_absorber_compressor_interstage_cooler
 - 1895.28430182 Psic_absorber_compressor_interstage_cooler_MEA <= 0
_C19: X_stripper - 0.5 l_stripper >= 0
_C190: - 1000 E1_absorber_stripper - 1000 E2_absorber_stripper - Y_absorber
 + Y_stripper - 0.5 d_absorber - 0.5 d_stripper >= -1985
_C1900: Dc_out_absorber_compressor_interstage_cooler
 + 1895.28430182 Psic_absorber_compressor_interstage_cooler_MEA
 >= 1895.28430182
_C1901: Dc_out_absorber_compressor_interstage_cooler
 + 1000 Psic_absorber_compressor_interstage_cooler_MEA <= 1000
_C1902: 94.9725589069 Dc_in_absorber_compressor_interstage_cooler
 - 180000 Psic_absorber_compressor_interstage_cooler_MEA
 + Vlc2_absorber_compressor_interstage_cooler_MEA = 0
_C1903: - D_absorber_TEG_dehydration + Dc_in_absorber_TEG_dehydration
 + Dc_out_absorber_TEG_dehydration = 0
_C1904: Dc_in_absorber_TEG_dehydration
 - 1895.28430182 Psic_absorber_TEG_dehydration_MEA <= 0
_C1905: Dc_out_absorber_TEG_dehydration
 + 1895.28430182 Psic_absorber_TEG_dehydration_MEA >= 1895.28430182
_C1906: Dc_out_absorber_TEG_dehydration
 + 1000 Psic_absorber_TEG_dehydration_MEA <= 1000
_C1907: 94.9725589069 Dc_in_absorber_TEG_dehydration
 - 180000 Psic_absorber_TEG_dehydration_MEA
 + Vlc2_absorber_TEG_dehydration_MEA = 0
_C1908: - D_absorber_cooling_tower + Dc_in_absorber_cooling_tower
 + Dc_out_absorber_cooling_tower = 0
_C1909: Dc_in_absorber_cooling_tower
 - 1895.28430182 Psic_absorber_cooling_tower_MEA <= 0
_C191: L_absorber_stripper - R_stripper_absorber = 0
_C1910: Dc_out_absorber_cooling_tower
 + 1895.28430182 Psic_absorber_cooling_tower_MEA >= 1895.28430182
_C1911: Dc_out_absorber_cooling_tower + 1000 Psic_absorber_cooling_tower_MEA
 <= 1000
_C1912: 94.9725589069 Dc_in_absorber_cooling_tower
 - 180000 Psic_absorber_cooling_tower_MEA + Vlc2_absorber_cooling_tower_MEA
 = 0
_C1913: - D_absorber_coolingwater_pumphouse
 + Dc_in_absorber_coolingwater_pumphouse
 + Dc_out_absorber_coolingwater_pumphouse = 0
_C1914: Dc_in_absorber_coolingwater_pumphouse
 - 1895.28430182 Psic_absorber_coolingwater_pumphouse_MEA <= 0
_C1915: Dc_out_absorber_coolingwater_pumphouse
 + 1895.28430182 Psic_absorber_coolingwater_pumphouse_MEA >= 1895.28430182
_C1916: Dc_out_absorber_coolingwater_pumphouse
 + 1000 Psic_absorber_coolingwater_pumphouse_MEA <= 1000
_C1917: 94.9725589069 Dc_in_absorber_coolingwater_pumphouse
 - 180000 Psic_absorber_coolingwater_pumphouse_MEA
 + Vlc2_absorber_coolingwater_pumphouse_MEA = 0
_C1918: - D_stripper_direct_contact_cooler
 + Dc_in_stripper_direct_contact_cooler
 + Dc_out_stripper_direct_contact_cooler = 0
_C1919: Dc_in_stripper_direct_contact_cooler
 - 2397.24554979 Psic_stripper_direct_contact_cooler_MEA <= 0
_C192: - L_stripper_absorber + R_absorber_stripper = 0
_C1920: Dc_out_stripper_direct_contact_cooler
 + 2397.24554979 Psic_stripper_direct_contact_cooler_MEA >= 2397.24554979
_C1921: Dc_out_stripper_direct_contact_cooler
 + 1000 Psic_stripper_direct_contact_cooler_MEA <= 1000
_C1922: 75.0861754715 Dc_in_stripper_direct_contact_cooler
 - 180000 Psic_stripper_direct_contact_cooler_MEA
 + Vlc2_stripper_direct_contact_cooler_MEA = 0
_C1923: - D_stripper_absorber + Dc_in_stripper_absorber
 + Dc_out_stripper_absorber = 0
_C1924: Dc_in_stripper_absorber - 2397.24554979 Psic_stripper_absorber_MEA
 <= 0
_C1925: Dc_out_stripper_absorber + 2397.24554979 Psic_stripper_absorber_MEA
 >= 2397.24554979
_C1926: Dc_out_stripper_absorber + 1000 Psic_stripper_absorber_MEA <= 1000
_C1927: 75.0861754715 Dc_in_stripper_absorber
 - 180000 Psic_stripper_absorber_MEA + Vlc2_stripper_absorber_MEA = 0
_C1928: - D_stripper_lean_amine_cooler + Dc_in_stripper_lean_amine_cooler
 + Dc_out_stripper_lean_amine_cooler = 0
_C1929: Dc_in_stripper_lean_amine_cooler
 - 2397.24554979 Psic_stripper_lean_amine_cooler_MEA <= 0
_C193: A_absorber_stripper - B_stripper_absorber = 0
_C1930: Dc_out_stripper_lean_amine_cooler
 + 2397.24554979 Psic_stripper_lean_amine_cooler_MEA >= 2397.24554979
_C1931: Dc_out_stripper_lean_amine_cooler
 + 1000 Psic_stripper_lean_amine_cooler_MEA <= 1000
_C1932: 75.0861754715 Dc_in_stripper_lean_amine_cooler
 - 180000 Psic_stripper_lean_amine_cooler_MEA
 + Vlc2_stripper_lean_amine_cooler_MEA = 0
_C1933: - D_stripper_cross_heat_exchanger
 + Dc_in_stripper_cross_heat_exchanger + Dc_out_stripper_cross_heat_exchanger
 = 0
_C1934: Dc_in_stripper_cross_heat_exchanger
 - 2397.24554979 Psic_stripper_cross_heat_exchanger_MEA <= 0
_C1935: Dc_out_stripper_cross_heat_exchanger
 + 2397.24554979 Psic_stripper_cross_heat_exchanger_MEA >= 2397.24554979
_C1936: Dc_out_stripper_cross_heat_exchanger
 + 1000 Psic_stripper_cross_heat_exchanger_MEA <= 1000
_C1937: 75.0861754715 Dc_in_stripper_cross_heat_exchanger
 - 180000 Psic_stripper_cross_heat_exchanger_MEA
 + Vlc2_stripper_cross_heat_exchanger_MEA = 0
_C1938: Vlc2_stripper_stripper_MEA = 0
_C1939: - D_stripper_storage_tank + Dc_in_stripper_storage_tank
 + Dc_out_stripper_storage_tank = 0
_C194: - A_stripper_absorber + B_absorber_stripper = 0
_C1940: Dc_in_stripper_storage_tank
 - 2397.24554979 Psic_stripper_storage_tank_MEA <= 0
_C1941: Dc_out_stripper_storage_tank
 + 2397.24554979 Psic_stripper_storage_tank_MEA >= 2397.24554979
_C1942: Dc_out_stripper_storage_tank + 1000 Psic_stripper_storage_tank_MEA
 <= 1000
_C1943: 75.0861754715 Dc_in_stripper_storage_tank
 - 180000 Psic_stripper_storage_tank_MEA + Vlc2_stripper_storage_tank_MEA = 0
_C1944: - D_stripper_reclaimer + Dc_in_stripper_reclaimer
 + Dc_out_stripper_reclaimer = 0
_C1945: Dc_in_stripper_reclaimer - 2397.24554979 Psic_stripper_reclaimer_MEA
 <= 0
_C1946: Dc_out_stripper_reclaimer + 2397.24554979 Psic_stripper_reclaimer_MEA
 >= 2397.24554979
_C1947: Dc_out_stripper_reclaimer + 1000 Psic_stripper_reclaimer_MEA <= 1000
_C1948: 75.0861754715 Dc_in_stripper_reclaimer
 - 180000 Psic_stripper_reclaimer_MEA + Vlc2_stripper_reclaimer_MEA = 0
_C1949: - D_stripper_compressor_house + Dc_in_stripper_compressor_house
 + Dc_out_stripper_compressor_house = 0
_C195: Wx_absorber_stripper + Wx_stripper_absorber = 1
_C1950: Dc_in_stripper_compressor_house
 - 2397.24554979 Psic_stripper_compressor_house_MEA <= 0
_C1951: Dc_out_stripper_compressor_house
 + 2397.24554979 Psic_stripper_compressor_house_MEA >= 2397.24554979
_C1952: Dc_out_stripper_compressor_house
 + 1000 Psic_stripper_compressor_house_MEA <= 1000
_C1953: 75.0861754715 Dc_in_stripper_compressor_house
 - 180000 Psic_stripper_compressor_house_MEA
 + Vlc2_stripper_compressor_house_MEA = 0
_C1954: - D_stripper_compressor_KOdrum + Dc_in_stripper_compressor_KOdrum
 + Dc_out_stripper_compressor_KOdrum = 0
_C1955: Dc_in_stripper_compressor_KOdrum
 - 2397.24554979 Psic_stripper_compressor_KOdrum_MEA <= 0
_C1956: Dc_out_stripper_compressor_KOdrum
 + 2397.24554979 Psic_stripper_compressor_KOdrum_MEA >= 2397.24554979
_C1957: Dc_out_stripper_compressor_KOdrum
 + 1000 Psic_stripper_compressor_KOdrum_MEA <= 1000
_C1958: 75.0861754715 Dc_in_stripper_compressor_KOdrum
 - 180000 Psic_stripper_compressor_KOdrum_MEA
 + Vlc2_stripper_compressor_KOdrum_MEA = 0
_C1959: - D_stripper_compressor_interstage_cooler
 + Dc_in_stripper_compressor_interstage_cooler
 + Dc_out_stripper_compressor_interstage_cooler = 0
_C196: Wy_absorber_stripper + Wy_stripper_absorber = 1
_C1960: Dc_in_stripper_compressor_interstage_cooler
 - 2397.24554979 Psic_stripper_compressor_interstage_cooler_MEA <= 0
_C1961: Dc_out_stripper_compressor_interstage_cooler
 + 2397.24554979 Psic_stripper_compressor_interstage_cooler_MEA
 >= 2397.24554979
_C1962: Dc_out_stripper_compressor_interstage_cooler
 + 1000 Psic_stripper_compressor_interstage_cooler_MEA <= 1000
_C1963: 75.0861754715 Dc_in_stripper_compressor_interstage_cooler
 - 180000 Psic_stripper_compressor_interstage_cooler_MEA
 + Vlc2_stripper_compressor_interstage_cooler_MEA = 0
_C1964: - D_stripper_TEG_dehydration + Dc_in_stripper_TEG_dehydration
 + Dc_out_stripper_TEG_dehydration = 0
_C1965: Dc_in_stripper_TEG_dehydration
 - 2397.24554979 Psic_stripper_TEG_dehydration_MEA <= 0
_C1966: Dc_out_stripper_TEG_dehydration
 + 2397.24554979 Psic_stripper_TEG_dehydration_MEA >= 2397.24554979
_C1967: Dc_out_stripper_TEG_dehydration
 + 1000 Psic_stripper_TEG_dehydration_MEA <= 1000
_C1968: 75.0861754715 Dc_in_stripper_TEG_dehydration
 - 180000 Psic_stripper_TEG_dehydration_MEA
 + Vlc2_stripper_TEG_dehydration_MEA = 0
_C1969: - D_stripper_cooling_tower + Dc_in_stripper_cooling_tower
 + Dc_out_stripper_cooling_tower = 0
_C197: - L_lean_amine_cooler_stripper + R_lean_amine_cooler_stripper
 - X_lean_amine_cooler + X_stripper = 0
_C1970: Dc_in_stripper_cooling_tower
 - 2397.24554979 Psic_stripper_cooling_tower_MEA <= 0
_C1971: Dc_out_stripper_cooling_tower
 + 2397.24554979 Psic_stripper_cooling_tower_MEA >= 2397.24554979
_C1972: Dc_out_stripper_cooling_tower + 1000 Psic_stripper_cooling_tower_MEA
 <= 1000
_C1973: 75.0861754715 Dc_in_stripper_cooling_tower
 - 180000 Psic_stripper_cooling_tower_MEA + Vlc2_stripper_cooling_tower_MEA
 = 0
_C1974: - D_stripper_coolingwater_pumphouse
 + Dc_in_stripper_coolingwater_pumphouse
 + Dc_out_stripper_coolingwater_pumphouse = 0
_C1975: Dc_in_stripper_coolingwater_pumphouse
 - 2397.24554979 Psic_stripper_coolingwater_pumphouse_MEA <= 0
_C1976: Dc_out_stripper_coolingwater_pumphouse
 + 2397.24554979 Psic_stripper_coolingwater_pumphouse_MEA >= 2397.24554979
_C1977: Dc_out_stripper_coolingwater_pumphouse
 + 1000 Psic_stripper_coolingwater_pumphouse_MEA <= 1000
_C1978: 75.0861754715 Dc_in_stripper_coolingwater_pumphouse
 - 180000 Psic_stripper_coolingwater_pumphouse_MEA
 + Vlc2_stripper_coolingwater_pumphouse_MEA = 0
_C1979: - D_storage_tank_direct_contact_cooler
 + Dc_in_storage_tank_direct_contact_cooler
 + Dc_out_storage_tank_direct_contact_cooler = 0
_C198: A_lean_amine_cooler_stripper - B_lean_amine_cooler_stripper
 - Y_lean_amine_cooler + Y_stripper = 0
_C1980: Dc_in_storage_tank_direct_contact_cooler
 - 4195.89276691 Psic_storage_tank_direct_contact_cooler_MEA <= 0
_C1981: Dc_out_storage_tank_direct_contact_cooler
 + 4195.89276691 Psic_storage_tank_direct_contact_cooler_MEA >= 4195.89276691
_C1982: Dc_out_storage_tank_direct_contact_cooler
 + 1000 Psic_storage_tank_direct_contact_cooler_MEA <= 1000
_C1983: 42.8990944238 Dc_in_storage_tank_direct_contact_cooler
 - 180000 Psic_storage_tank_direct_contact_cooler_MEA
 + Vlc2_storage_tank_direct_contact_cooler_MEA = 0
_C1984: - D_storage_tank_absorber + Dc_in_storage_tank_absorber
 + Dc_out_storage_tank_absorber = 0
_C1985: Dc_in_storage_tank_absorber
 - 4195.89276691 Psic_storage_tank_absorber_MEA <= 0
_C1986: Dc_out_storage_tank_absorber
 + 4195.89276691 Psic_storage_tank_absorber_MEA >= 4195.89276691
_C1987: Dc_out_storage_tank_absorber + 1000 Psic_storage_tank_absorber_MEA
 <= 1000
_C1988: 42.8990944238 Dc_in_storage_tank_absorber
 - 180000 Psic_storage_tank_absorber_MEA + Vlc2_storage_tank_absorber_MEA = 0
_C1989: - D_storage_tank_lean_amine_cooler
 + Dc_in_storage_tank_lean_amine_cooler
 + Dc_out_storage_tank_lean_amine_cooler = 0
_C199: R_lean_amine_cooler_stripper - 1000 Wx_lean_amine_cooler_stripper <= 0
_C1990: Dc_in_storage_tank_lean_amine_cooler
 - 4195.89276691 Psic_storage_tank_lean_amine_cooler_MEA <= 0
_C1991: Dc_out_storage_tank_lean_amine_cooler
 + 4195.89276691 Psic_storage_tank_lean_amine_cooler_MEA >= 4195.89276691
_C1992: Dc_out_storage_tank_lean_amine_cooler
 + 1000 Psic_storage_tank_lean_amine_cooler_MEA <= 1000
_C1993: 42.8990944238 Dc_in_storage_tank_lean_amine_cooler
 - 180000 Psic_storage_tank_lean_amine_cooler_MEA
 + Vlc2_storage_tank_lean_amine_cooler_MEA = 0
_C1994: - D_storage_tank_cross_heat_exchanger
 + Dc_in_storage_tank_cross_heat_exchanger
 + Dc_out_storage_tank_cross_heat_exchanger = 0
_C1995: Dc_in_storage_tank_cross_heat_exchanger
 - 4195.89276691 Psic_storage_tank_cross_heat_exchanger_MEA <= 0
_C1996: Dc_out_storage_tank_cross_heat_exchanger
 + 4195.89276691 Psic_storage_tank_cross_heat_exchanger_MEA >= 4195.89276691
_C1997: Dc_out_storage_tank_cross_heat_exchanger
 + 1000 Psic_storage_tank_cross_heat_exchanger_MEA <= 1000
_C1998: 42.8990944238 Dc_in_storage_tank_cross_heat_exchanger
 - 180000 Psic_storage_tank_cross_heat_exchanger_MEA
 + Vlc2_storage_tank_cross_heat_exchanger_MEA = 0
_C1999: - D_storage_tank_stripper + Dc_in_storage_tank_stripper
 + Dc_out_storage_tank_stripper = 0
_C2: d_direct_contact_cooler + l_direct_contact_cooler = 132
_C20: Y_stripper - 0.5 d_stripper >= 0
_C200: L_lean_amine_cooler_stripper + 1000 Wx_lean_amine_cooler_stripper
 <= 1000
_C2000: Dc_in_storage_tank_stripper
 - 4195.89276691 Psic_storage_tank_stripper_MEA <= 0
_C2001: Dc_out_storage_tank_stripper
 + 4195.89276691 Psic_storage_tank_stripper_MEA >= 4195.89276691
_C2002: Dc_out_storage_tank_stripper + 1000 Psic_storage_tank_stripper_MEA
 <= 1000
_C2003: 42.8990944238 Dc_in_storage_tank_stripper
 - 180000 Psic_storage_tank_stripper_MEA + Vlc2_storage_tank_stripper_MEA = 0
_C2004: Vlc2_storage_tank_storage_tank_MEA = 0
_C2005: - D_storage_tank_reclaimer + Dc_in_storage_tank_reclaimer
 + Dc_out_storage_tank_reclaimer = 0
_C2006: Dc_in_storage_tank_reclaimer
 - 4195.89276691 Psic_storage_tank_reclaimer_MEA <= 0
_C2007: Dc_out_storage_tank_reclaimer
 + 4195.89276691 Psic_storage_tank_reclaimer_MEA >= 4195.89276691
_C2008: Dc_out_storage_tank_reclaimer + 1000 Psic_storage_tank_reclaimer_MEA
 <= 1000
_C2009: 42.8990944238 Dc_in_storage_tank_reclaimer
 - 180000 Psic_storage_tank_reclaimer_MEA + Vlc2_storage_tank_reclaimer_MEA
 = 0
_C201: A_lean_amine_cooler_stripper - 1000 Wy_lean_amine_cooler_stripper <= 0
_C2010: - D_storage_tank_compressor_house
 + Dc_in_storage_tank_compressor_house + Dc_out_storage_tank_compressor_house
 = 0
_C2011: Dc_in_storage_tank_compressor_house
 - 4195.89276691 Psic_storage_tank_compressor_house_MEA <= 0
_C2012: Dc_out_storage_tank_compressor_house
 + 4195.89276691 Psic_storage_tank_compressor_house_MEA >= 4195.89276691
_C2013: Dc_out_storage_tank_compressor_house
 + 1000 Psic_storage_tank_compressor_house_MEA <= 1000
_C2014: 42.8990944238 Dc_in_storage_tank_compressor_house
 - 180000 Psic_storage_tank_compressor_house_MEA
 + Vlc2_storage_tank_compressor_house_MEA = 0
_C2015: - D_storage_tank_compressor_KOdrum
 + Dc_in_storage_tank_compressor_KOdrum
 + Dc_out_storage_tank_compressor_KOdrum = 0
_C2016: Dc_in_storage_tank_compressor_KOdrum
 - 4195.89276691 Psic_storage_tank_compressor_KOdrum_MEA <= 0
_C2017: Dc_out_storage_tank_compressor_KOdrum
 + 4195.89276691 Psic_storage_tank_compressor_KOdrum_MEA >= 4195.89276691
_C2018: Dc_out_storage_tank_compressor_KOdrum
 + 1000 Psic_storage_tank_compressor_KOdrum_MEA <= 1000
_C2019: 42.8990944238 Dc_in_storage_tank_compressor_KOdrum
 - 180000 Psic_storage_tank_compressor_KOdrum_MEA
 + Vlc2_storage_tank_compressor_KOdrum_MEA = 0
_C202: B_lean_amine_cooler_stripper + 1000 Wy_lean_amine_cooler_stripper
 <= 1000
_C2020: - D_storage_tank_compressor_interstage_cooler
 + Dc_in_storage_tank_compressor_interstage_cooler
 + Dc_out_storage_tank_compressor_interstage_cooler = 0
_C2021: Dc_in_storage_tank_compressor_interstage_cooler
 - 4195.89276691 Psic_storage_tank_compressor_interstage_cooler_MEA <= 0
_C2022: Dc_out_storage_tank_compressor_interstage_cooler
 + 4195.89276691 Psic_storage_tank_compressor_interstage_cooler_MEA
 >= 4195.89276691
_C2023: Dc_out_storage_tank_compressor_interstage_cooler
 + 1000 Psic_storage_tank_compressor_interstage_cooler_MEA <= 1000
_C2024: 42.8990944238 Dc_in_storage_tank_compressor_interstage_cooler
 - 180000 Psic_storage_tank_compressor_interstage_cooler_MEA
 + Vlc2_storage_tank_compressor_interstage_cooler_MEA = 0
_C2025: - D_storage_tank_TEG_dehydration + Dc_in_storage_tank_TEG_dehydration
 + Dc_out_storage_tank_TEG_dehydration = 0
_C2026: Dc_in_storage_tank_TEG_dehydration
 - 4195.89276691 Psic_storage_tank_TEG_dehydration_MEA <= 0
_C2027: Dc_out_storage_tank_TEG_dehydration
 + 4195.89276691 Psic_storage_tank_TEG_dehydration_MEA >= 4195.89276691
_C2028: Dc_out_storage_tank_TEG_dehydration
 + 1000 Psic_storage_tank_TEG_dehydration_MEA <= 1000
_C2029: 42.8990944238 Dc_in_storage_tank_TEG_dehydration
 - 180000 Psic_storage_tank_TEG_dehydration_MEA
 + Vlc2_storage_tank_TEG_dehydration_MEA = 0
_C203: - A_lean_amine_cooler_stripper - B_lean_amine_cooler_stripper
 + D_lean_amine_cooler_stripper - L_lean_amine_cooler_stripper
 - R_lean_amine_cooler_stripper = 0
_C2030: - D_storage_tank_cooling_tower + Dc_in_storage_tank_cooling_tower
 + Dc_out_storage_tank_cooling_tower = 0
_C2031: Dc_in_storage_tank_cooling_tower
 - 4195.89276691 Psic_storage_tank_cooling_tower_MEA <= 0
_C2032: Dc_out_storage_tank_cooling_tower
 + 4195.89276691 Psic_storage_tank_cooling_tower_MEA >= 4195.89276691
_C2033: Dc_out_storage_tank_cooling_tower
 + 1000 Psic_storage_tank_cooling_tower_MEA <= 1000
_C2034: 42.8990944238 Dc_in_storage_tank_cooling_tower
 - 180000 Psic_storage_tank_cooling_tower_MEA
 + Vlc2_storage_tank_cooling_tower_MEA = 0
_C2035: - D_storage_tank_coolingwater_pumphouse
 + Dc_in_storage_tank_coolingwater_pumphouse
 + Dc_out_storage_tank_coolingwater_pumphouse = 0
_C2036: Dc_in_storage_tank_coolingwater_pumphouse
 - 4195.89276691 Psic_storage_tank_coolingwater_pumphouse_MEA <= 0
_C2037: Dc_out_storage_tank_coolingwater_pumphouse
 + 4195.89276691 Psic_storage_tank_coolingwater_pumphouse_MEA >= 4195.89276691
_C2038: Dc_out_storage_tank_coolingwater_pumphouse
 + 1000 Psic_storage_tank_coolingwater_pumphouse_MEA <= 1000
_C2039: 42.8990944238 Dc_in_storage_tank_coolingwater_pumphouse
 - 180000 Psic_storage_tank_coolingwater_pumphouse_MEA
 + Vlc2_storage_tank_coolingwater_pumphouse_MEA = 0
_C204: D_lean_amine_cooler_stripper - D_stripper_lean_amine_cooler = 0
_C2040: 0.000255 Vlc2_absorber_TEG_dehydration_MEA
 + 0.000255 Vlc2_absorber_absorber_MEA
 + 0.000255 Vlc2_absorber_compressor_KOdrum_MEA
 + 0.000255 Vlc2_absorber_compressor_house_MEA
 + 0.000255 Vlc2_absorber_compressor_interstage_cooler_MEA
 + 0.000255 Vlc2_absorber_cooling_tower_MEA
 + 0.000255 Vlc2_absorber_coolingwater_pumphouse_MEA
 + 0.000255 Vlc2_absorber_cross_heat_exchanger_MEA
 + 0.000255 Vlc2_absorber_direct_contact_cooler_MEA
 + 0.000255 Vlc2_absorber_lean_amine_cooler_MEA
 + 0.000255 Vlc2_absorber_reclaimer_MEA
 + 0.000255 Vlc2_absorber_storage_tank_MEA
 + 0.000255 Vlc2_absorber_stripper_MEA - Vlc_absorber_TEG_dehydration
 - Vlc_absorber_absorber - Vlc_absorber_compressor_KOdrum
 - Vlc_absorber_compressor_house - Vlc_absorber_compressor_interstage_cooler
 - Vlc_absorber_cooling_tower - Vlc_absorber_coolingwater_pumphouse
 - Vlc_absorber_cross_heat_exchanger - Vlc_absorber_direct_contact_cooler
 - Vlc_absorber_lean_amine_cooler - Vlc_absorber_reclaimer
 - Vlc_absorber_storage_tank - Vlc_absorber_stripper = -45.9
_C2041: 0.000255 Vlc2_stripper_TEG_dehydration_MEA
 + 0.000255 Vlc2_stripper_absorber_MEA
 + 0.000255 Vlc2_stripper_compressor_KOdrum_MEA
 + 0.000255 Vlc2_stripper_compressor_house_MEA
 + 0.000255 Vlc2_stripper_compressor_interstage_cooler_MEA
 + 0.000255 Vlc2_stripper_cooling_tower_MEA
 + 0.000255 Vlc2_stripper_coolingwater_pumphouse_MEA
 + 0.000255 Vlc2_stripper_cross_heat_exchanger_MEA
 + 0.000255 Vlc2_stripper_direct_contact_cooler_MEA
 + 0.000255 Vlc2_stripper_lean_amine_cooler_MEA
 + 0.000255 Vlc2_stripper_reclaimer_MEA
 + 0.000255 Vlc2_stripper_storage_tank_MEA
 + 0.000255 Vlc2_stripper_stripper_MEA - Vlc_stripper_TEG_dehydration
 - Vlc_stripper_absorber - Vlc_stripper_compressor_KOdrum
 - Vlc_stripper_compressor_house - Vlc_stripper_compressor_interstage_cooler
 - Vlc_stripper_cooling_tower - Vlc_stripper_coolingwater_pumphouse
 - Vlc_stripper_cross_heat_exchanger - Vlc_stripper_direct_contact_cooler
 - Vlc_stripper_lean_amine_cooler - Vlc_stripper_reclaimer
 - Vlc_stripper_storage_tank - Vlc_stripper_stripper = -45.9
_C2042: 0.000255 Vlc2_storage_tank_TEG_dehydration_MEA
 + 0.000255 Vlc2_storage_tank_absorber_MEA
 + 0.000255 Vlc2_storage_tank_compressor_KOdrum_MEA
 + 0.000255 Vlc2_storage_tank_compressor_house_MEA
 + 0.000255 Vlc2_storage_tank_compressor_interstage_cooler_MEA
 + 0.000255 Vlc2_storage_tank_cooling_tower_MEA
 + 0.000255 Vlc2_storage_tank_coolingwater_pumphouse_MEA
 + 0.000255 Vlc2_storage_tank_cross_heat_exchanger_MEA
 + 0.000255 Vlc2_storage_tank_direct_contact_cooler_MEA
 + 0.000255 Vlc2_storage_tank_lean_amine_cooler_MEA
 + 0.000255 Vlc2_storage_tank_reclaimer_MEA
 + 0.000255 Vlc2_storage_tank_storage_tank_MEA
 + 0.000255 Vlc2_storage_tank_stripper_MEA - Vlc_storage_tank_TEG_dehydration
 - Vlc_storage_tank_absorber - Vlc_storage_tank_compressor_KOdrum
 - Vlc_storage_tank_compressor_house
 - Vlc_storage_tank_compressor_interstage_cooler
 - Vlc_storage_tank_cooling_tower - Vlc_storage_tank_coolingwater_pumphouse
 - Vlc_storage_tank_cross_heat_exchanger
 - Vlc_storage_tank_direct_contact_cooler - Vlc_storage_tank_lean_amine_cooler
 - Vlc_storage_tank_reclaimer - Vlc_storage_tank_storage_tank
 - Vlc_storage_tank_stripper = -45.9
_C2043: SumVlc - Vlc_absorber_TEG_dehydration - Vlc_absorber_absorber
 - Vlc_absorber_compressor_KOdrum - Vlc_absorber_compressor_house
 - Vlc_absorber_compressor_interstage_cooler - Vlc_absorber_cooling_tower
 - Vlc_absorber_coolingwater_pumphouse - Vlc_absorber_cross_heat_exchanger
 - Vlc_absorber_direct_contact_cooler - Vlc_absorber_lean_amine_cooler
 - Vlc_absorber_reclaimer - Vlc_absorber_storage_tank - Vlc_absorber_stripper
 - Vlc_storage_tank_TEG_dehydration - Vlc_storage_tank_absorber
 - Vlc_storage_tank_compressor_KOdrum - Vlc_storage_tank_compressor_house
 - Vlc_storage_tank_compressor_interstage_cooler
 - Vlc_storage_tank_cooling_tower - Vlc_storage_tank_coolingwater_pumphouse
 - Vlc_storage_tank_cross_heat_exchanger
 - Vlc_storage_tank_direct_contact_cooler - Vlc_storage_tank_lean_amine_cooler
 - Vlc_storage_tank_reclaimer - Vlc_storage_tank_storage_tank
 - Vlc_storage_tank_stripper - Vlc_stripper_TEG_dehydration
 - Vlc_stripper_absorber - Vlc_stripper_compressor_KOdrum
 - Vlc_stripper_compressor_house - Vlc_stripper_compressor_interstage_cooler
 - Vlc_stripper_cooling_tower - Vlc_stripper_coolingwater_pumphouse
 - Vlc_stripper_cross_heat_exchanger - Vlc_stripper_direct_contact_cooler
 - Vlc_stripper_lean_amine_cooler - Vlc_stripper_reclaimer
 - Vlc_stripper_storage_tank - Vlc_stripper_stripper = 0
_C2044: pressure_drop_direct_contact_cooler_direct_contact_cooler = 0
_C2045: pumping_power_direct_contact_cooler_direct_contact_cooler = 0
_C2046: pumping_cost_direct_contact_cooler_direct_contact_cooler = 0
_C2047: - 374.112 D_direct_contact_cooler_absorber
 + 3.04 pressure_drop_direct_contact_cooler_absorber = 0
_C2048: - 65 pressure_drop_direct_contact_cooler_absorber
 + 0.72 pumping_power_direct_contact_cooler_absorber = 0
_C2049: pumping_cost_direct_contact_cooler_absorber
 - 0.72 pumping_power_direct_contact_cooler_absorber = 0
_C205: 1000 E1_lean_amine_cooler_stripper + 1000 E2_lean_amine_cooler_stripper
 + X_lean_amine_cooler - X_stripper - 0.5 l_lean_amine_cooler - 0.5 l_stripper
 >= 15
_C2050: pressure_drop_direct_contact_cooler_lean_amine_cooler = 0
_C2051: pumping_power_direct_contact_cooler_lean_amine_cooler = 0
_C2052: pumping_cost_direct_contact_cooler_lean_amine_cooler = 0
_C2053: pressure_drop_direct_contact_cooler_cross_heat_exchanger = 0
_C2054: pumping_power_direct_contact_cooler_cross_heat_exchanger = 0
_C2055: pumping_cost_direct_contact_cooler_cross_heat_exchanger = 0
_C2056: pressure_drop_direct_contact_cooler_stripper = 0
_C2057: pumping_power_direct_contact_cooler_stripper = 0
_C2058: pumping_cost_direct_contact_cooler_stripper = 0
_C2059: pressure_drop_direct_contact_cooler_storage_tank = 0
_C206: - 1000 E1_lean_amine_cooler_stripper
 + 1000 E2_lean_amine_cooler_stripper - X_lean_amine_cooler + X_stripper
 - 0.5 l_lean_amine_cooler - 0.5 l_stripper >= -985
_C2060: pumping_power_direct_contact_cooler_storage_tank = 0
_C2061: pumping_cost_direct_contact_cooler_storage_tank = 0
_C2062: pressure_drop_direct_contact_cooler_reclaimer = 0
_C2063: pumping_power_direct_contact_cooler_reclaimer = 0
_C2064: pumping_cost_direct_contact_cooler_reclaimer = 0
_C2065: pressure_drop_direct_contact_cooler_compressor_house = 0
_C2066: pumping_power_direct_contact_cooler_compressor_house = 0
_C2067: pumping_cost_direct_contact_cooler_compressor_house = 0
_C2068: pressure_drop_direct_contact_cooler_compressor_KOdrum = 0
_C2069: pumping_power_direct_contact_cooler_compressor_KOdrum = 0
_C207: 1000 E1_lean_amine_cooler_stripper - 1000 E2_lean_amine_cooler_stripper
 + Y_lean_amine_cooler - Y_stripper - 0.5 d_lean_amine_cooler - 0.5 d_stripper
 >= -985
_C2070: pumping_cost_direct_contact_cooler_compressor_KOdrum = 0
_C2071: pressure_drop_direct_contact_cooler_compressor_interstage_cooler = 0
_C2072: pumping_power_direct_contact_cooler_compressor_interstage_cooler = 0
_C2073: pumping_cost_direct_contact_cooler_compressor_interstage_cooler = 0
_C2074: pressure_drop_direct_contact_cooler_TEG_dehydration = 0
_C2075: pumping_power_direct_contact_cooler_TEG_dehydration = 0
_C2076: pumping_cost_direct_contact_cooler_TEG_dehydration = 0
_C2077: pressure_drop_direct_contact_cooler_cooling_tower = 0
_C2078: pumping_power_direct_contact_cooler_cooling_tower = 0
_C2079: pumping_cost_direct_contact_cooler_cooling_tower = 0
_C208: - 1000 E1_lean_amine_cooler_stripper
 - 1000 E2_lean_amine_cooler_stripper - Y_lean_amine_cooler + Y_stripper
 - 0.5 d_lean_amine_cooler - 0.5 d_stripper >= -1985
_C2080: pressure_drop_direct_contact_cooler_coolingwater_pumphouse = 0
_C2081: pumping_power_direct_contact_cooler_coolingwater_pumphouse = 0
_C2082: pumping_cost_direct_contact_cooler_coolingwater_pumphouse = 0
_C2083: pressure_drop_absorber_direct_contact_cooler = 0
_C2084: pumping_power_absorber_direct_contact_cooler = 0
_C2085: pumping_cost_absorber_direct_contact_cooler = 0
_C2086: pressure_drop_absorber_absorber = 0
_C2087: pumping_power_absorber_absorber = 0
_C2088: pumping_cost_absorber_absorber = 0
_C2089: - 1164.0672 D_absorber_lean_amine_cooler
 + 0.288 pressure_drop_absorber_lean_amine_cooler = 0
_C209: L_lean_amine_cooler_stripper - R_stripper_lean_amine_cooler = 0
_C2090: - 49 pressure_drop_absorber_lean_amine_cooler
 + 598.8 pumping_power_absorber_lean_amine_cooler = 0
_C2091: pumping_cost_absorber_lean_amine_cooler
 - 0.72 pumping_power_absorber_lean_amine_cooler = 0
_C2092: - 1164.0672 D_absorber_cross_heat_exchanger
 + 0.3 pressure_drop_absorber_cross_heat_exchanger = 0
_C2093: - 53 pressure_drop_absorber_cross_heat_exchanger
 + 598.8 pumping_power_absorber_cross_heat_exchanger = 0
_C2094: pumping_cost_absorber_cross_heat_exchanger
 - 0.72 pumping_power_absorber_cross_heat_exchanger = 0
_C2095: pressure_drop_absorber_stripper = 0
_C2096: pumping_power_absorber_stripper = 0
_C2097: pumping_cost_absorber_stripper = 0
_C2098: pressure_drop_absorber_storage_tank = 0
_C2099: pumping_power_absorber_storage_tank = 0
_C21: 0 O_storage_tank + l_storage_tank = 34
_C210: - L_stripper_lean_amine_cooler + R_lean_amine_cooler_stripper = 0
_C2100: pumping_cost_absorber_storage_tank = 0
_C2101: pressure_drop_absorber_reclaimer = 0
_C2102: pumping_power_absorber_reclaimer = 0
_C2103: pumping_cost_absorber_reclaimer = 0
_C2104: pressure_drop_absorber_compressor_house = 0
_C2105: pumping_power_absorber_compressor_house = 0
_C2106: pumping_cost_absorber_compressor_house = 0
_C2107: pressure_drop_absorber_compressor_KOdrum = 0
_C2108: pumping_power_absorber_compressor_KOdrum = 0
_C2109: pumping_cost_absorber_compressor_KOdrum = 0
_C211: A_lean_amine_cooler_stripper - B_stripper_lean_amine_cooler = 0
_C2110: pressure_drop_absorber_compressor_interstage_cooler = 0
_C2111: pumping_power_absorber_compressor_interstage_cooler = 0
_C2112: pumping_cost_absorber_compressor_interstage_cooler = 0
_C2113: pressure_drop_absorber_TEG_dehydration = 0
_C2114: pumping_power_absorber_TEG_dehydration = 0
_C2115: pumping_cost_absorber_TEG_dehydration = 0
_C2116: pressure_drop_absorber_cooling_tower = 0
_C2117: pumping_power_absorber_cooling_tower = 0
_C2118: pumping_cost_absorber_cooling_tower = 0
_C2119: pressure_drop_absorber_coolingwater_pumphouse = 0
_C212: - A_stripper_lean_amine_cooler + B_lean_amine_cooler_stripper = 0
_C2120: pumping_power_absorber_coolingwater_pumphouse = 0
_C2121: pumping_cost_absorber_coolingwater_pumphouse = 0
_C2122: pressure_drop_lean_amine_cooler_direct_contact_cooler = 0
_C2123: pumping_power_lean_amine_cooler_direct_contact_cooler = 0
_C2124: pumping_cost_lean_amine_cooler_direct_contact_cooler = 0
_C2125: pressure_drop_lean_amine_cooler_absorber = 0
_C2126: pumping_power_lean_amine_cooler_absorber = 0
_C2127: pumping_cost_lean_amine_cooler_absorber = 0
_C2128: pressure_drop_lean_amine_cooler_lean_amine_cooler = 0
_C2129: pumping_power_lean_amine_cooler_lean_amine_cooler = 0
_C213: Wx_lean_amine_cooler_stripper + Wx_stripper_lean_amine_cooler = 1
_C2130: pumping_cost_lean_amine_cooler_lean_amine_cooler = 0
_C2131: - 1164.0672 D_lean_amine_cooler_cross_heat_exchanger
 + 0.288 pressure_drop_lean_amine_cooler_cross_heat_exchanger = 0
_C2132: - 49 pressure_drop_lean_amine_cooler_cross_heat_exchanger
 + 598.8 pumping_power_lean_amine_cooler_cross_heat_exchanger = 0
_C2133: pumping_cost_lean_amine_cooler_cross_heat_exchanger
 - 0.72 pumping_power_lean_amine_cooler_cross_heat_exchanger = 0
_C2134: pressure_drop_lean_amine_cooler_stripper = 0
_C2135: pumping_power_lean_amine_cooler_stripper = 0
_C2136: pumping_cost_lean_amine_cooler_stripper = 0
_C2137: pressure_drop_lean_amine_cooler_storage_tank = 0
_C2138: pumping_power_lean_amine_cooler_storage_tank = 0
_C2139: pumping_cost_lean_amine_cooler_storage_tank = 0
_C214: Wy_lean_amine_cooler_stripper + Wy_stripper_lean_amine_cooler = 1
_C2140: pressure_drop_lean_amine_cooler_reclaimer = 0
_C2141: pumping_power_lean_amine_cooler_reclaimer = 0
_C2142: pumping_cost_lean_amine_cooler_reclaimer = 0
_C2143: pressure_drop_lean_amine_cooler_compressor_house = 0
_C2144: pumping_power_lean_amine_cooler_compressor_house = 0
_C2145: pumping_cost_lean_amine_cooler_compressor_house = 0
_C2146: pressure_drop_lean_amine_cooler_compressor_KOdrum = 0
_C2147: pumping_power_lean_amine_cooler_compressor_KOdrum = 0
_C2148: pumping_cost_lean_amine_cooler_compressor_KOdrum = 0
_C2149: pressure_drop_lean_amine_cooler_compressor_interstage_cooler = 0
_C215: - L_cross_heat_exchanger_stripper + R_cross_heat_exchanger_stripper
 - X_cross_heat_exchanger + X_stripper = 0
_C2150: pumping_power_lean_amine_cooler_compressor_interstage_cooler = 0
_C2151: pumping_cost_lean_amine_cooler_compressor_interstage_cooler = 0
_C2152: pressure_drop_lean_amine_cooler_TEG_dehydration = 0
_C2153: pumping_power_lean_amine_cooler_TEG_dehydration = 0
_C2154: pumping_cost_lean_amine_cooler_TEG_dehydration = 0
_C2155: pressure_drop_lean_amine_cooler_cooling_tower = 0
_C2156: pumping_power_lean_amine_cooler_cooling_tower = 0
_C2157: pumping_cost_lean_amine_cooler_cooling_tower = 0
_C2158: pressure_drop_lean_amine_cooler_coolingwater_pumphouse = 0
_C2159: pumping_power_lean_amine_cooler_coolingwater_pumphouse = 0
_C216: A_cross_heat_exchanger_stripper - B_cross_heat_exchanger_stripper
 - Y_cross_heat_exchanger + Y_stripper = 0
_C2160: pumping_cost_lean_amine_cooler_coolingwater_pumphouse = 0
_C2161: pressure_drop_cross_heat_exchanger_direct_contact_cooler = 0
_C2162: pumping_power_cross_heat_exchanger_direct_contact_cooler = 0
_C2163: pumping_cost_cross_heat_exchanger_direct_contact_cooler = 0
_C2164: pressure_drop_cross_heat_exchanger_absorber = 0
_C2165: pumping_power_cross_heat_exchanger_absorber = 0
_C2166: pumping_cost_cross_heat_exchanger_absorber = 0
_C2167: pressure_drop_cross_heat_exchanger_lean_amine_cooler = 0
_C2168: pumping_power_cross_heat_exchanger_lean_amine_cooler = 0
_C2169: pumping_cost_cross_heat_exchanger_lean_amine_cooler = 0
_C217: R_cross_heat_exchanger_stripper - 1000 Wx_cross_heat_exchanger_stripper
 <= 0
_C2170: pressure_drop_cross_heat_exchanger_cross_heat_exchanger = 0
_C2171: pumping_power_cross_heat_exchanger_cross_heat_exchanger = 0
_C2172: pumping_cost_cross_heat_exchanger_cross_heat_exchanger = 0
_C2173: - 625.4424 D_cross_heat_exchanger_stripper
 + 0.392 pressure_drop_cross_heat_exchanger_stripper = 0
_C2174: - 53 pressure_drop_cross_heat_exchanger_stripper
 + 349.8 pumping_power_cross_heat_exchanger_stripper = 0
_C2175: pumping_cost_cross_heat_exchanger_stripper
 - 0.72 pumping_power_cross_heat_exchanger_stripper = 0
_C2176: pressure_drop_cross_heat_exchanger_storage_tank = 0
_C2177: pumping_power_cross_heat_exchanger_storage_tank = 0
_C2178: pumping_cost_cross_heat_exchanger_storage_tank = 0
_C2179: pressure_drop_cross_heat_exchanger_reclaimer = 0
_C218: L_cross_heat_exchanger_stripper + 1000 Wx_cross_heat_exchanger_stripper
 <= 1000
_C2180: pumping_power_cross_heat_exchanger_reclaimer = 0
_C2181: pumping_cost_cross_heat_exchanger_reclaimer = 0
_C2182: pressure_drop_cross_heat_exchanger_compressor_house = 0
_C2183: pumping_power_cross_heat_exchanger_compressor_house = 0
_C2184: pumping_cost_cross_heat_exchanger_compressor_house = 0
_C2185: pressure_drop_cross_heat_exchanger_compressor_KOdrum = 0
_C2186: pumping_power_cross_heat_exchanger_compressor_KOdrum = 0
_C2187: pumping_cost_cross_heat_exchanger_compressor_KOdrum = 0
_C2188: pressure_drop_cross_heat_exchanger_compressor_interstage_cooler = 0
_C2189: pumping_power_cross_heat_exchanger_compressor_interstage_cooler = 0
_C219: A_cross_heat_exchanger_stripper - 1000 Wy_cross_heat_exchanger_stripper
 <= 0
_C2190: pumping_cost_cross_heat_exchanger_compressor_interstage_cooler = 0
_C2191: pressure_drop_cross_heat_exchanger_TEG_dehydration = 0
_C2192: pumping_power_cross_heat_exchanger_TEG_dehydration = 0
_C2193: pumping_cost_cross_heat_exchanger_TEG_dehydration = 0
_C2194: pressure_drop_cross_heat_exchanger_cooling_tower = 0
_C2195: pumping_power_cross_heat_exchanger_cooling_tower = 0
_C2196: pumping_cost_cross_heat_exchanger_cooling_tower = 0
_C2197: pressure_drop_cross_heat_exchanger_coolingwater_pumphouse = 0
_C2198: pumping_power_cross_heat_exchanger_coolingwater_pumphouse = 0
_C2199: pumping_cost_cross_heat_exchanger_coolingwater_pumphouse = 0
_C22: d_storage_tank + l_storage_tank = 68
_C220: B_cross_heat_exchanger_stripper + 1000 Wy_cross_heat_exchanger_stripper
 <= 1000
_C2200: pressure_drop_stripper_direct_contact_cooler = 0
_C2201: pumping_power_stripper_direct_contact_cooler = 0
_C2202: pumping_cost_stripper_direct_contact_cooler = 0
_C2203: pressure_drop_stripper_absorber = 0
_C2204: pumping_power_stripper_absorber = 0
_C2205: pumping_cost_stripper_absorber = 0
_C2206: pressure_drop_stripper_lean_amine_cooler = 0
_C2207: pumping_power_stripper_lean_amine_cooler = 0
_C2208: pumping_cost_stripper_lean_amine_cooler = 0
_C2209: pressure_drop_stripper_cross_heat_exchanger = 0
_C221: - A_cross_heat_exchanger_stripper - B_cross_heat_exchanger_stripper
 + D_cross_heat_exchanger_stripper - L_cross_heat_exchanger_stripper
 - R_cross_heat_exchanger_stripper = 0
_C2210: pumping_power_stripper_cross_heat_exchanger = 0
_C2211: pumping_cost_stripper_cross_heat_exchanger = 0
_C2212: pressure_drop_stripper_stripper = 0
_C2213: pumping_power_stripper_stripper = 0
_C2214: pumping_cost_stripper_stripper = 0
_C2215: pressure_drop_stripper_storage_tank = 0
_C2216: pumping_power_stripper_storage_tank = 0
_C2217: pumping_cost_stripper_storage_tank = 0
_C2218: pressure_drop_stripper_reclaimer = 0
_C2219: pumping_power_stripper_reclaimer = 0
_C222: D_cross_heat_exchanger_stripper - D_stripper_cross_heat_exchanger = 0
_C2220: pumping_cost_stripper_reclaimer = 0
_C2221: pressure_drop_stripper_compressor_house = 0
_C2222: pumping_power_stripper_compressor_house = 0
_C2223: pumping_cost_stripper_compressor_house = 0
_C2224: - 199.584 D_stripper_compressor_KOdrum
 + 0.598 pressure_drop_stripper_compressor_KOdrum = 0
_C2225: - 4 pressure_drop_stripper_compressor_KOdrum
 + 1.188 pumping_power_stripper_compressor_KOdrum = 0
_C2226: pumping_cost_stripper_compressor_KOdrum
 - 0.72 pumping_power_stripper_compressor_KOdrum = 0
_C2227: pressure_drop_stripper_compressor_interstage_cooler = 0
_C2228: pumping_power_stripper_compressor_interstage_cooler = 0
_C2229: pumping_cost_stripper_compressor_interstage_cooler = 0
_C223: 1000 E1_cross_heat_exchanger_stripper
 + 1000 E2_cross_heat_exchanger_stripper + X_cross_heat_exchanger - X_stripper
 - 0.5 l_cross_heat_exchanger - 0.5 l_stripper >= 15
_C2230: pressure_drop_stripper_TEG_dehydration = 0
_C2231: pumping_power_stripper_TEG_dehydration = 0
_C2232: pumping_cost_stripper_TEG_dehydration = 0
_C2233: pressure_drop_stripper_cooling_tower = 0
_C2234: pumping_power_stripper_cooling_tower = 0
_C2235: pumping_cost_stripper_cooling_tower = 0
_C2236: pressure_drop_stripper_coolingwater_pumphouse = 0
_C2237: pumping_power_stripper_coolingwater_pumphouse = 0
_C2238: pumping_cost_stripper_coolingwater_pumphouse = 0
_C2239: pressure_drop_storage_tank_direct_contact_cooler = 0
_C224: - 1000 E1_cross_heat_exchanger_stripper
 + 1000 E2_cross_heat_exchanger_stripper - X_cross_heat_exchanger + X_stripper
 - 0.5 l_cross_heat_exchanger - 0.5 l_stripper >= -985
_C2240: pumping_power_storage_tank_direct_contact_cooler = 0
_C2241: pumping_cost_storage_tank_direct_contact_cooler = 0
_C2242: pressure_drop_storage_tank_absorber = 0
_C2243: pumping_power_storage_tank_absorber = 0
_C2244: pumping_cost_storage_tank_absorber = 0
_C2245: pressure_drop_storage_tank_lean_amine_cooler = 0
_C2246: pumping_power_storage_tank_lean_amine_cooler = 0
_C2247: pumping_cost_storage_tank_lean_amine_cooler = 0
_C2248: - 1143.936 D_storage_tank_cross_heat_exchanger
 + 0.3 pressure_drop_storage_tank_cross_heat_exchanger = 0
_C2249: - 53 pressure_drop_storage_tank_cross_heat_exchanger
 + 595.8 pumping_power_storage_tank_cross_heat_exchanger = 0
_C225: 1000 E1_cross_heat_exchanger_stripper
 - 1000 E2_cross_heat_exchanger_stripper + Y_cross_heat_exchanger - Y_stripper
 - 0.5 d_cross_heat_exchanger - 0.5 d_stripper >= -985
_C2250: pumping_cost_storage_tank_cross_heat_exchanger
 - 0.72 pumping_power_storage_tank_cross_heat_exchanger = 0
_C2251: pressure_drop_storage_tank_stripper = 0
_C2252: pumping_power_storage_tank_stripper = 0
_C2253: pumping_cost_storage_tank_stripper = 0
_C2254: pressure_drop_storage_tank_storage_tank = 0
_C2255: pumping_power_storage_tank_storage_tank = 0
_C2256: pumping_cost_storage_tank_storage_tank = 0
_C2257: pressure_drop_storage_tank_reclaimer = 0
_C2258: pumping_power_storage_tank_reclaimer = 0
_C2259: pumping_cost_storage_tank_reclaimer = 0
_C226: - 1000 E1_cross_heat_exchanger_stripper
 - 1000 E2_cross_heat_exchanger_stripper - Y_cross_heat_exchanger + Y_stripper
 - 0.5 d_cross_heat_exchanger - 0.5 d_stripper >= -1985
_C2260: pressure_drop_storage_tank_compressor_house = 0
_C2261: pumping_power_storage_tank_compressor_house = 0
_C2262: pumping_cost_storage_tank_compressor_house = 0
_C2263: pressure_drop_storage_tank_compressor_KOdrum = 0
_C2264: pumping_power_storage_tank_compressor_KOdrum = 0
_C2265: pumping_cost_storage_tank_compressor_KOdrum = 0
_C2266: pressure_drop_storage_tank_compressor_interstage_cooler = 0
_C2267: pumping_power_storage_tank_compressor_interstage_cooler = 0
_C2268: pumping_cost_storage_tank_compressor_interstage_cooler = 0
_C2269: pressure_drop_storage_tank_TEG_dehydration = 0
_C227: L_cross_heat_exchanger_stripper - R_stripper_cross_heat_exchanger = 0
_C2270: pumping_power_storage_tank_TEG_dehydration = 0
_C2271: pumping_cost_storage_tank_TEG_dehydration = 0
_C2272: pressure_drop_storage_tank_cooling_tower = 0
_C2273: pumping_power_storage_tank_cooling_tower = 0
_C2274: pumping_cost_storage_tank_cooling_tower = 0
_C2275: pressure_drop_storage_tank_coolingwater_pumphouse = 0
_C2276: pumping_power_storage_tank_coolingwater_pumphouse = 0
_C2277: pumping_cost_storage_tank_coolingwater_pumphouse = 0
_C2278: pressure_drop_reclaimer_direct_contact_cooler = 0
_C2279: pumping_power_reclaimer_direct_contact_cooler = 0
_C228: - L_stripper_cross_heat_exchanger + R_cross_heat_exchanger_stripper = 0
_C2280: pumping_cost_reclaimer_direct_contact_cooler = 0
_C2281: pressure_drop_reclaimer_absorber = 0
_C2282: pumping_power_reclaimer_absorber = 0
_C2283: pumping_cost_reclaimer_absorber = 0
_C2284: pressure_drop_reclaimer_lean_amine_cooler = 0
_C2285: pumping_power_reclaimer_lean_amine_cooler = 0
_C2286: pumping_cost_reclaimer_lean_amine_cooler = 0
_C2287: - 1143.936 D_reclaimer_cross_heat_exchanger
 + 0.288 pressure_drop_reclaimer_cross_heat_exchanger = 0
_C2288: - 49 pressure_drop_reclaimer_cross_heat_exchanger
 + 595.8 pumping_power_reclaimer_cross_heat_exchanger = 0
_C2289: pumping_cost_reclaimer_cross_heat_exchanger
 - 0.72 pumping_power_reclaimer_cross_heat_exchanger = 0
_C229: A_cross_heat_exchanger_stripper - B_stripper_cross_heat_exchanger = 0
_C2290: pressure_drop_reclaimer_stripper = 0
_C2291: pumping_power_reclaimer_stripper = 0
_C2292: pumping_cost_reclaimer_stripper = 0
_C2293: pressure_drop_reclaimer_storage_tank = 0
_C2294: pumping_power_reclaimer_storage_tank = 0
_C2295: pumping_cost_reclaimer_storage_tank = 0
_C2296: pressure_drop_reclaimer_reclaimer = 0
_C2297: pumping_power_reclaimer_reclaimer = 0
_C2298: pumping_cost_reclaimer_reclaimer = 0
_C2299: pressure_drop_reclaimer_compressor_house = 0
_C23: X_storage_tank - 0.5 l_storage_tank >= 0
_C230: - A_stripper_cross_heat_exchanger + B_cross_heat_exchanger_stripper = 0
_C2300: pumping_power_reclaimer_compressor_house = 0
_C2301: pumping_cost_reclaimer_compressor_house = 0
_C2302: pressure_drop_reclaimer_compressor_KOdrum = 0
_C2303: pumping_power_reclaimer_compressor_KOdrum = 0
_C2304: pumping_cost_reclaimer_compressor_KOdrum = 0
_C2305: pressure_drop_reclaimer_compressor_interstage_cooler = 0
_C2306: pumping_power_reclaimer_compressor_interstage_cooler = 0
_C2307: pumping_cost_reclaimer_compressor_interstage_cooler = 0
_C2308: pressure_drop_reclaimer_TEG_dehydration = 0
_C2309: pumping_power_reclaimer_TEG_dehydration = 0
_C231: Wx_cross_heat_exchanger_stripper + Wx_stripper_cross_heat_exchanger = 1
_C2310: pumping_cost_reclaimer_TEG_dehydration = 0
_C2311: pressure_drop_reclaimer_cooling_tower = 0
_C2312: pumping_power_reclaimer_cooling_tower = 0
_C2313: pumping_cost_reclaimer_cooling_tower = 0
_C2314: pressure_drop_reclaimer_coolingwater_pumphouse = 0
_C2315: pumping_power_reclaimer_coolingwater_pumphouse = 0
_C2316: pumping_cost_reclaimer_coolingwater_pumphouse = 0
_C2317: pressure_drop_compressor_house_direct_contact_cooler = 0
_C2318: pumping_power_compressor_house_direct_contact_cooler = 0
_C2319: pumping_cost_compressor_house_direct_contact_cooler = 0
_C232: Wy_cross_heat_exchanger_stripper + Wy_stripper_cross_heat_exchanger = 1
_C2320: pressure_drop_compressor_house_absorber = 0
_C2321: pumping_power_compressor_house_absorber = 0
_C2322: pumping_cost_compressor_house_absorber = 0
_C2323: pressure_drop_compressor_house_lean_amine_cooler = 0
_C2324: pumping_power_compressor_house_lean_amine_cooler = 0
_C2325: pumping_cost_compressor_house_lean_amine_cooler = 0
_C2326: pressure_drop_compressor_house_cross_heat_exchanger = 0
_C2327: pumping_power_compressor_house_cross_heat_exchanger = 0
_C2328: pumping_cost_compressor_house_cross_heat_exchanger = 0
_C2329: pressure_drop_compressor_house_stripper = 0
_C233: - L_direct_contact_cooler_storage_tank
 + R_direct_contact_cooler_storage_tank - X_direct_contact_cooler
 + X_storage_tank = 0
_C2330: pumping_power_compressor_house_stripper = 0
_C2331: pumping_cost_compressor_house_stripper = 0
_C2332: pressure_drop_compressor_house_storage_tank = 0
_C2333: pumping_power_compressor_house_storage_tank = 0
_C2334: pumping_cost_compressor_house_storage_tank = 0
_C2335: pressure_drop_compressor_house_reclaimer = 0
_C2336: pumping_power_compressor_house_reclaimer = 0
_C2337: pumping_cost_compressor_house_reclaimer = 0
_C2338: pressure_drop_compressor_house_compressor_house = 0
_C2339: pumping_power_compressor_house_compressor_house = 0
_C234: A_direct_contact_cooler_storage_tank
 - B_direct_contact_cooler_storage_tank - Y_direct_contact_cooler
 + Y_storage_tank = 0
_C2340: pumping_cost_compressor_house_compressor_house = 0
_C2341: pressure_drop_compressor_house_compressor_KOdrum = 0
_C2342: pumping_power_compressor_house_compressor_KOdrum = 0
_C2343: pumping_cost_compressor_house_compressor_KOdrum = 0
_C2344: pressure_drop_compressor_house_compressor_interstage_cooler = 0
_C2345: pumping_power_compressor_house_compressor_interstage_cooler = 0
_C2346: pumping_cost_compressor_house_compressor_interstage_cooler = 0
_C2347: pressure_drop_compressor_house_TEG_dehydration = 0
_C2348: pumping_power_compressor_house_TEG_dehydration = 0
_C2349: pumping_cost_compressor_house_TEG_dehydration = 0
_C235: R_direct_contact_cooler_storage_tank
 - 1000 Wx_direct_contact_cooler_storage_tank <= 0
_C2350: pressure_drop_compressor_house_cooling_tower = 0
_C2351: pumping_power_compressor_house_cooling_tower = 0
_C2352: pumping_cost_compressor_house_cooling_tower = 0
_C2353: pressure_drop_compressor_house_coolingwater_pumphouse = 0
_C2354: pumping_power_compressor_house_coolingwater_pumphouse = 0
_C2355: pumping_cost_compressor_house_coolingwater_pumphouse = 0
_C2356: pressure_drop_compressor_KOdrum_direct_contact_cooler = 0
_C2357: pumping_power_compressor_KOdrum_direct_contact_cooler = 0
_C2358: pumping_cost_compressor_KOdrum_direct_contact_cooler = 0
_C2359: pressure_drop_compressor_KOdrum_absorber = 0
_C236: L_direct_contact_cooler_storage_tank
 + 1000 Wx_direct_contact_cooler_storage_tank <= 1000
_C2360: pumping_power_compressor_KOdrum_absorber = 0
_C2361: pumping_cost_compressor_KOdrum_absorber = 0
_C2362: pressure_drop_compressor_KOdrum_lean_amine_cooler = 0
_C2363: pumping_power_compressor_KOdrum_lean_amine_cooler = 0
_C2364: pumping_cost_compressor_KOdrum_lean_amine_cooler = 0
_C2365: pressure_drop_compressor_KOdrum_cross_heat_exchanger = 0
_C2366: pumping_power_compressor_KOdrum_cross_heat_exchanger = 0
_C2367: pumping_cost_compressor_KOdrum_cross_heat_exchanger = 0
_C2368: pressure_drop_compressor_KOdrum_stripper = 0
_C2369: pumping_power_compressor_KOdrum_stripper = 0
_C237: A_direct_contact_cooler_storage_tank
 - 1000 Wy_direct_contact_cooler_storage_tank <= 0
_C2370: pumping_cost_compressor_KOdrum_stripper = 0
_C2371: pressure_drop_compressor_KOdrum_storage_tank = 0
_C2372: pumping_power_compressor_KOdrum_storage_tank = 0
_C2373: pumping_cost_compressor_KOdrum_storage_tank = 0
_C2374: pressure_drop_compressor_KOdrum_reclaimer = 0
_C2375: pumping_power_compressor_KOdrum_reclaimer = 0
_C2376: pumping_cost_compressor_KOdrum_reclaimer = 0
_C2377: - 199.584 D_compressor_KOdrum_compressor_house
 + 0.598 pressure_drop_compressor_KOdrum_compressor_house = 0
_C2378: - 4 pressure_drop_compressor_KOdrum_compressor_house
 + 1.188 pumping_power_compressor_KOdrum_compressor_house = 0
_C2379: pumping_cost_compressor_KOdrum_compressor_house
 - 1.8 pumping_power_compressor_KOdrum_compressor_house = 0
_C238: B_direct_contact_cooler_storage_tank
 + 1000 Wy_direct_contact_cooler_storage_tank <= 1000
_C2380: pressure_drop_compressor_KOdrum_compressor_KOdrum = 0
_C2381: pumping_power_compressor_KOdrum_compressor_KOdrum = 0
_C2382: pumping_cost_compressor_KOdrum_compressor_KOdrum = 0
_C2383: pressure_drop_compressor_KOdrum_compressor_interstage_cooler = 0
_C2384: pumping_power_compressor_KOdrum_compressor_interstage_cooler = 0
_C2385: pumping_cost_compressor_KOdrum_compressor_interstage_cooler = 0
_C2386: pressure_drop_compressor_KOdrum_TEG_dehydration = 0
_C2387: pumping_power_compressor_KOdrum_TEG_dehydration = 0
_C2388: pumping_cost_compressor_KOdrum_TEG_dehydration = 0
_C2389: pressure_drop_compressor_KOdrum_cooling_tower = 0
_C239: - A_direct_contact_cooler_storage_tank
 - B_direct_contact_cooler_storage_tank + D_direct_contact_cooler_storage_tank
 - L_direct_contact_cooler_storage_tank - R_direct_contact_cooler_storage_tank
 = 0
_C2390: pumping_power_compressor_KOdrum_cooling_tower = 0
_C2391: pumping_cost_compressor_KOdrum_cooling_tower = 0
_C2392: pressure_drop_compressor_KOdrum_coolingwater_pumphouse = 0
_C2393: pumping_power_compressor_KOdrum_coolingwater_pumphouse = 0
_C2394: pumping_cost_compressor_KOdrum_coolingwater_pumphouse = 0
_C2395: pressure_drop_compressor_interstage_cooler_direct_contact_cooler = 0
_C2396: pumping_power_compressor_interstage_cooler_direct_contact_cooler = 0
_C2397: pumping_cost_compressor_interstage_cooler_direct_contact_cooler = 0
_C2398: pressure_drop_compressor_interstage_cooler_absorber = 0
_C2399: pumping_power_compressor_interstage_cooler_absorber = 0
_C24: Y_storage_tank - 0.5 d_storage_tank >= 0
_C240: D_direct_contact_cooler_storage_tank
 - D_storage_tank_direct_contact_cooler = 0
_C2400: pumping_cost_compressor_interstage_cooler_absorber = 0
_C2401: pressure_drop_compressor_interstage_cooler_lean_amine_cooler = 0
_C2402: pumping_power_compressor_interstage_cooler_lean_amine_cooler = 0
_C2403: pumping_cost_compressor_interstage_cooler_lean_amine_cooler = 0
_C2404: pressure_drop_compressor_interstage_cooler_cross_heat_exchanger = 0
_C2405: pumping_power_compressor_interstage_cooler_cross_heat_exchanger = 0
_C2406: pumping_cost_compressor_interstage_cooler_cross_heat_exchanger = 0
_C2407: pressure_drop_compressor_interstage_cooler_stripper = 0
_C2408: pumping_power_compressor_interstage_cooler_stripper = 0
_C2409: pumping_cost_compressor_interstage_cooler_stripper = 0
_C241: 1000 E1_direct_contact_cooler_storage_tank
 + 1000 E2_direct_contact_cooler_storage_tank + X_direct_contact_cooler
 - X_storage_tank - 0.5 l_direct_contact_cooler - 0.5 l_storage_tank >= 76
_C2410: pressure_drop_compressor_interstage_cooler_storage_tank = 0
_C2411: pumping_power_compressor_interstage_cooler_storage_tank = 0
_C2412: pumping_cost_compressor_interstage_cooler_storage_tank = 0
_C2413: pressure_drop_compressor_interstage_cooler_reclaimer = 0
_C2414: pumping_power_compressor_interstage_cooler_reclaimer = 0
_C2415: pumping_cost_compressor_interstage_cooler_reclaimer = 0
_C2416: - 199.584 D_compressor_interstage_cooler_compressor_house
 + 0.598 pressure_drop_compressor_interstage_cooler_compressor_house = 0
_C2417: - 4 pressure_drop_compressor_interstage_cooler_compressor_house
 + 1.188 pumping_power_compressor_interstage_cooler_compressor_house = 0
_C2418: pumping_cost_compressor_interstage_cooler_compressor_house
 - 1.8 pumping_power_compressor_interstage_cooler_compressor_house = 0
_C2419: pressure_drop_compressor_interstage_cooler_compressor_KOdrum = 0
_C242: - 1000 E1_direct_contact_cooler_storage_tank
 + 1000 E2_direct_contact_cooler_storage_tank - X_direct_contact_cooler
 + X_storage_tank - 0.5 l_direct_contact_cooler - 0.5 l_storage_tank >= -924
_C2420: pumping_power_compressor_interstage_cooler_compressor_KOdrum = 0
_C2421: pumping_cost_compressor_interstage_cooler_compressor_KOdrum = 0
_C2422:
 pressure_drop_compressor_interstage_cooler_compressor_interstage_cooler = 0
_C2423:
 pumping_power_compressor_interstage_cooler_compressor_interstage_cooler = 0
_C2424: pumping_cost_compressor_interstage_cooler_compressor_interstage_cooler
 = 0
_C2425: pressure_drop_compressor_interstage_cooler_TEG_dehydration = 0
_C2426: pumping_power_compressor_interstage_cooler_TEG_dehydration = 0
_C2427: pumping_cost_compressor_interstage_cooler_TEG_dehydration = 0
_C2428: pressure_drop_compressor_interstage_cooler_cooling_tower = 0
_C2429: pumping_power_compressor_interstage_cooler_cooling_tower = 0
_C243: 1000 E1_direct_contact_cooler_storage_tank
 - 1000 E2_direct_contact_cooler_storage_tank + Y_direct_contact_cooler
 - Y_storage_tank - 0.5 d_direct_contact_cooler - 0.5 d_storage_tank >= -924
_C2430: pumping_cost_compressor_interstage_cooler_cooling_tower = 0
_C2431: pressure_drop_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C2432: pumping_power_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C2433: pumping_cost_compressor_interstage_cooler_coolingwater_pumphouse = 0
_C2434: pressure_drop_TEG_dehydration_direct_contact_cooler = 0
_C2435: pumping_power_TEG_dehydration_direct_contact_cooler = 0
_C2436: pumping_cost_TEG_dehydration_direct_contact_cooler = 0
_C2437: pressure_drop_TEG_dehydration_absorber = 0
_C2438: pumping_power_TEG_dehydration_absorber = 0
_C2439: pumping_cost_TEG_dehydration_absorber = 0
_C244: - 1000 E1_direct_contact_cooler_storage_tank
 - 1000 E2_direct_contact_cooler_storage_tank - Y_direct_contact_cooler
 + Y_storage_tank - 0.5 d_direct_contact_cooler - 0.5 d_storage_tank >= -1924
_C2440: pressure_drop_TEG_dehydration_lean_amine_cooler = 0
_C2441: pumping_power_TEG_dehydration_lean_amine_cooler = 0
_C2442: pumping_cost_TEG_dehydration_lean_amine_cooler = 0
_C2443: pressure_drop_TEG_dehydration_cross_heat_exchanger = 0
_C2444: pumping_power_TEG_dehydration_cross_heat_exchanger = 0
_C2445: pumping_cost_TEG_dehydration_cross_heat_exchanger = 0
_C2446: pressure_drop_TEG_dehydration_stripper = 0
_C2447: pumping_power_TEG_dehydration_stripper = 0
_C2448: pumping_cost_TEG_dehydration_stripper = 0
_C2449: pressure_drop_TEG_dehydration_storage_tank = 0
_C245: L_direct_contact_cooler_storage_tank
 - R_storage_tank_direct_contact_cooler = 0
_C2450: pumping_power_TEG_dehydration_storage_tank = 0
_C2451: pumping_cost_TEG_dehydration_storage_tank = 0
_C2452: pressure_drop_TEG_dehydration_reclaimer = 0
_C2453: pumping_power_TEG_dehydration_reclaimer = 0
_C2454: pumping_cost_TEG_dehydration_reclaimer = 0
_C2455: - 199.584 D_TEG_dehydration_compressor_house
 + 0.598 pressure_drop_TEG_dehydration_compressor_house = 0
_C2456: - 4 pressure_drop_TEG_dehydration_compressor_house
 + 1.188 pumping_power_TEG_dehydration_compressor_house = 0
_C2457: pumping_cost_TEG_dehydration_compressor_house
 - 0.72 pumping_power_TEG_dehydration_compressor_house = 0
_C2458: pressure_drop_TEG_dehydration_compressor_KOdrum = 0
_C2459: pumping_power_TEG_dehydration_compressor_KOdrum = 0
_C246: - L_storage_tank_direct_contact_cooler
 + R_direct_contact_cooler_storage_tank = 0
_C2460: pumping_cost_TEG_dehydration_compressor_KOdrum = 0
_C2461: pressure_drop_TEG_dehydration_compressor_interstage_cooler = 0
_C2462: pumping_power_TEG_dehydration_compressor_interstage_cooler = 0
_C2463: pumping_cost_TEG_dehydration_compressor_interstage_cooler = 0
_C2464: pressure_drop_TEG_dehydration_TEG_dehydration = 0
_C2465: pumping_power_TEG_dehydration_TEG_dehydration = 0
_C2466: pumping_cost_TEG_dehydration_TEG_dehydration = 0
_C2467: pressure_drop_TEG_dehydration_cooling_tower = 0
_C2468: pumping_power_TEG_dehydration_cooling_tower = 0
_C2469: pumping_cost_TEG_dehydration_cooling_tower = 0
_C247: A_direct_contact_cooler_storage_tank
 - B_storage_tank_direct_contact_cooler = 0
_C2470: pressure_drop_TEG_dehydration_coolingwater_pumphouse = 0
_C2471: pumping_power_TEG_dehydration_coolingwater_pumphouse = 0
_C2472: pumping_cost_TEG_dehydration_coolingwater_pumphouse = 0
_C2473: pressure_drop_cooling_tower_direct_contact_cooler = 0
_C2474: pumping_power_cooling_tower_direct_contact_cooler = 0
_C2475: pumping_cost_cooling_tower_direct_contact_cooler = 0
_C2476: pressure_drop_cooling_tower_absorber = 0
_C2477: pumping_power_cooling_tower_absorber = 0
_C2478: pumping_cost_cooling_tower_absorber = 0
_C2479: pressure_drop_cooling_tower_lean_amine_cooler = 0
_C248: - A_storage_tank_direct_contact_cooler
 + B_direct_contact_cooler_storage_tank = 0
_C2480: pumping_power_cooling_tower_lean_amine_cooler = 0
_C2481: pumping_cost_cooling_tower_lean_amine_cooler = 0
_C2482: pressure_drop_cooling_tower_cross_heat_exchanger = 0
_C2483: pumping_power_cooling_tower_cross_heat_exchanger = 0
_C2484: pumping_cost_cooling_tower_cross_heat_exchanger = 0
_C2485: pressure_drop_cooling_tower_stripper = 0
_C2486: pumping_power_cooling_tower_stripper = 0
_C2487: pumping_cost_cooling_tower_stripper = 0
_C2488: pressure_drop_cooling_tower_storage_tank = 0
_C2489: pumping_power_cooling_tower_storage_tank = 0
_C249: Wx_direct_contact_cooler_storage_tank
 + Wx_storage_tank_direct_contact_cooler = 1
_C2490: pumping_cost_cooling_tower_storage_tank = 0
_C2491: pressure_drop_cooling_tower_reclaimer = 0
_C2492: pumping_power_cooling_tower_reclaimer = 0
_C2493: pumping_cost_cooling_tower_reclaimer = 0
_C2494: pressure_drop_cooling_tower_compressor_house = 0
_C2495: pumping_power_cooling_tower_compressor_house = 0
_C2496: pumping_cost_cooling_tower_compressor_house = 0
_C2497: pressure_drop_cooling_tower_compressor_KOdrum = 0
_C2498: pumping_power_cooling_tower_compressor_KOdrum = 0
_C2499: pumping_cost_cooling_tower_compressor_KOdrum = 0
_C25: 0 O_reclaimer + l_reclaimer = 18
_C250: Wy_direct_contact_cooler_storage_tank
 + Wy_storage_tank_direct_contact_cooler = 1
_C2500: pressure_drop_cooling_tower_compressor_interstage_cooler = 0
_C2501: pumping_power_cooling_tower_compressor_interstage_cooler = 0
_C2502: pumping_cost_cooling_tower_compressor_interstage_cooler = 0
_C2503: pressure_drop_cooling_tower_TEG_dehydration = 0
_C2504: pumping_power_cooling_tower_TEG_dehydration = 0
_C2505: pumping_cost_cooling_tower_TEG_dehydration = 0
_C2506: pressure_drop_cooling_tower_cooling_tower = 0
_C2507: pumping_power_cooling_tower_cooling_tower = 0
_C2508: pumping_cost_cooling_tower_cooling_tower = 0
_C2509: - 1098.2952 D_cooling_tower_coolingwater_pumphouse
 + 0.408 pressure_drop_cooling_tower_coolingwater_pumphouse = 0
_C251: - L_absorber_storage_tank + R_absorber_storage_tank - X_absorber
 + X_storage_tank = 0
_C2510: - 98 pressure_drop_cooling_tower_coolingwater_pumphouse
 + 598.2 pumping_power_cooling_tower_coolingwater_pumphouse = 0
_C2511: pumping_cost_cooling_tower_coolingwater_pumphouse
 - 2.88 pumping_power_cooling_tower_coolingwater_pumphouse = 0
_C2512: pressure_drop_coolingwater_pumphouse_direct_contact_cooler = 0
_C2513: pumping_power_coolingwater_pumphouse_direct_contact_cooler = 0
_C2514: pumping_cost_coolingwater_pumphouse_direct_contact_cooler = 0
_C2515: pressure_drop_coolingwater_pumphouse_absorber = 0
_C2516: pumping_power_coolingwater_pumphouse_absorber = 0
_C2517: pumping_cost_coolingwater_pumphouse_absorber = 0
_C2518: pressure_drop_coolingwater_pumphouse_lean_amine_cooler = 0
_C2519: pumping_power_coolingwater_pumphouse_lean_amine_cooler = 0
_C252: A_absorber_storage_tank - B_absorber_storage_tank - Y_absorber
 + Y_storage_tank = 0
_C2520: pumping_cost_coolingwater_pumphouse_lean_amine_cooler = 0
_C2521: pressure_drop_coolingwater_pumphouse_cross_heat_exchanger = 0
_C2522: pumping_power_coolingwater_pumphouse_cross_heat_exchanger = 0
_C2523: pumping_cost_coolingwater_pumphouse_cross_heat_exchanger = 0
_C2524: pressure_drop_coolingwater_pumphouse_stripper = 0
_C2525: pumping_power_coolingwater_pumphouse_stripper = 0
_C2526: pumping_cost_coolingwater_pumphouse_stripper = 0
_C2527: pressure_drop_coolingwater_pumphouse_storage_tank = 0
_C2528: pumping_power_coolingwater_pumphouse_storage_tank = 0
_C2529: pumping_cost_coolingwater_pumphouse_storage_tank = 0
_C253: R_absorber_storage_tank - 1000 Wx_absorber_storage_tank <= 0
_C2530: pressure_drop_coolingwater_pumphouse_reclaimer = 0
_C2531: pumping_power_coolingwater_pumphouse_reclaimer = 0
_C2532: pumping_cost_coolingwater_pumphouse_reclaimer = 0
_C2533: pressure_drop_coolingwater_pumphouse_compressor_house = 0
_C2534: pumping_power_coolingwater_pumphouse_compressor_house = 0
_C2535: pumping_cost_coolingwater_pumphouse_compressor_house = 0
_C2536: pressure_drop_coolingwater_pumphouse_compressor_KOdrum = 0
_C2537: pumping_power_coolingwater_pumphouse_compressor_KOdrum = 0
_C2538: pumping_cost_coolingwater_pumphouse_compressor_KOdrum = 0
_C2539: pressure_drop_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C254: L_absorber_storage_tank + 1000 Wx_absorber_storage_tank <= 1000
_C2540: pumping_power_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C2541: pumping_cost_coolingwater_pumphouse_compressor_interstage_cooler = 0
_C2542: pressure_drop_coolingwater_pumphouse_TEG_dehydration = 0
_C2543: pumping_power_coolingwater_pumphouse_TEG_dehydration = 0
_C2544: pumping_cost_coolingwater_pumphouse_TEG_dehydration = 0
_C2545: pressure_drop_coolingwater_pumphouse_cooling_tower = 0
_C2546: pumping_power_coolingwater_pumphouse_cooling_tower = 0
_C2547: pumping_cost_coolingwater_pumphouse_cooling_tower = 0
_C2548: pressure_drop_coolingwater_pumphouse_coolingwater_pumphouse = 0
_C2549: pumping_power_coolingwater_pumphouse_coolingwater_pumphouse = 0
_C255: A_absorber_storage_tank - 1000 Wy_absorber_storage_tank <= 0
_C2550: pumping_cost_coolingwater_pumphouse_coolingwater_pumphouse = 0
_C2551: - pumping_cost_TEG_dehydration_TEG_dehydration
 - pumping_cost_TEG_dehydration_absorber
 - pumping_cost_TEG_dehydration_compressor_KOdrum
 - pumping_cost_TEG_dehydration_compressor_house
 - pumping_cost_TEG_dehydration_compressor_interstage_cooler
 - pumping_cost_TEG_dehydration_cooling_tower
 - pumping_cost_TEG_dehydration_coolingwater_pumphouse
 - pumping_cost_TEG_dehydration_cross_heat_exchanger
 - pumping_cost_TEG_dehydration_direct_contact_cooler
 - pumping_cost_TEG_dehydration_lean_amine_cooler
 - pumping_cost_TEG_dehydration_reclaimer
 - pumping_cost_TEG_dehydration_storage_tank
 - pumping_cost_TEG_dehydration_stripper
 - pumping_cost_absorber_TEG_dehydration - pumping_cost_absorber_absorber
 - pumping_cost_absorber_compressor_KOdrum
 - pumping_cost_absorber_compressor_house
 - pumping_cost_absorber_compressor_interstage_cooler
 - pumping_cost_absorber_cooling_tower
 - pumping_cost_absorber_coolingwater_pumphouse
 - pumping_cost_absorber_cross_heat_exchanger
 - pumping_cost_absorber_direct_contact_cooler
 - pumping_cost_absorber_lean_amine_cooler - pumping_cost_absorber_reclaimer
 - pumping_cost_absorber_storage_tank - pumping_cost_absorber_stripper
 - pumping_cost_compressor_KOdrum_TEG_dehydration
 - pumping_cost_compressor_KOdrum_absorber
 - pumping_cost_compressor_KOdrum_compressor_KOdrum
 - pumping_cost_compressor_KOdrum_compressor_house
 - pumping_cost_compressor_KOdrum_compressor_interstage_cooler
 - pumping_cost_compressor_KOdrum_cooling_tower
 - pumping_cost_compressor_KOdrum_coolingwater_pumphouse
 - pumping_cost_compressor_KOdrum_cross_heat_exchanger
 - pumping_cost_compressor_KOdrum_direct_contact_cooler
 - pumping_cost_compressor_KOdrum_lean_amine_cooler
 - pumping_cost_compressor_KOdrum_reclaimer
 - pumping_cost_compressor_KOdrum_storage_tank
 - pumping_cost_compressor_KOdrum_stripper
 - pumping_cost_compressor_house_TEG_dehydration
 - pumping_cost_compressor_house_absorber
 - pumping_cost_compressor_house_compressor_KOdrum
 - pumping_cost_compressor_house_compressor_house
 - pumping_cost_compressor_house_compressor_interstage_cooler
 - pumping_cost_compressor_house_cooling_tower
 - pumping_cost_compressor_house_coolingwater_pumphouse
 - pumping_cost_compressor_house_cross_heat_exchanger
 - pumping_cost_compressor_house_direct_contact_cooler
 - pumping_cost_compressor_house_lean_amine_cooler
 - pumping_cost_compressor_house_reclaimer
 - pumping_cost_compressor_house_storage_tank
 - pumping_cost_compressor_house_stripper
 - pumping_cost_compressor_interstage_cooler_TEG_dehydration
 - pumping_cost_compressor_interstage_cooler_absorber
 - pumping_cost_compressor_interstage_cooler_compressor_KOdrum
 - pumping_cost_compressor_interstage_cooler_compressor_house
 - pumping_cost_compressor_interstage_cooler_compressor_interstage_cooler
 - pumping_cost_compressor_interstage_cooler_cooling_tower
 - pumping_cost_compressor_interstage_cooler_coolingwater_pumphouse
 - pumping_cost_compressor_interstage_cooler_cross_heat_exchanger
 - pumping_cost_compressor_interstage_cooler_direct_contact_cooler
 - pumping_cost_compressor_interstage_cooler_lean_amine_cooler
 - pumping_cost_compressor_interstage_cooler_reclaimer
 - pumping_cost_compressor_interstage_cooler_storage_tank
 - pumping_cost_compressor_interstage_cooler_stripper
 - pumping_cost_cooling_tower_TEG_dehydration
 - pumping_cost_cooling_tower_absorber
 - pumping_cost_cooling_tower_compressor_KOdrum
 - pumping_cost_cooling_tower_compressor_house
 - pumping_cost_cooling_tower_compressor_interstage_cooler
 - pumping_cost_cooling_tower_cooling_tower
 - pumping_cost_cooling_tower_coolingwater_pumphouse
 - pumping_cost_cooling_tower_cross_heat_exchanger
 - pumping_cost_cooling_tower_direct_contact_cooler
 - pumping_cost_cooling_tower_lean_amine_cooler
 - pumping_cost_cooling_tower_reclaimer
 - pumping_cost_cooling_tower_storage_tank
 - pumping_cost_cooling_tower_stripper
 - pumping_cost_coolingwater_pumphouse_TEG_dehydration
 - pumping_cost_coolingwater_pumphouse_absorber
 - pumping_cost_coolingwater_pumphouse_compressor_KOdrum
 - pumping_cost_coolingwater_pumphouse_compressor_house
 - pumping_cost_coolingwater_pumphouse_compressor_interstage_cooler
 - pumping_cost_coolingwater_pumphouse_cooling_tower
 - pumping_cost_coolingwater_pumphouse_coolingwater_pumphouse
 - pumping_cost_coolingwater_pumphouse_cross_heat_exchanger
 - pumping_cost_coolingwater_pumphouse_direct_contact_cooler
 - pumping_cost_coolingwater_pumphouse_lean_amine_cooler
 - pumping_cost_coolingwater_pumphouse_reclaimer
 - pumping_cost_coolingwater_pumphouse_storage_tank
 - pumping_cost_coolingwater_pumphouse_stripper
 - pumping_cost_cross_heat_exchanger_TEG_dehydration
 - pumping_cost_cross_heat_exchanger_absorber
 - pumping_cost_cross_heat_exchanger_compressor_KOdrum
 - pumping_cost_cross_heat_exchanger_compressor_house
 - pumping_cost_cross_heat_exchanger_compressor_interstage_cooler
 - pumping_cost_cross_heat_exchanger_cooling_tower
 - pumping_cost_cross_heat_exchanger_coolingwater_pumphouse
 - pumping_cost_cross_heat_exchanger_cross_heat_exchanger
 - pumping_cost_cross_heat_exchanger_direct_contact_cooler
 - pumping_cost_cross_heat_exchanger_lean_amine_cooler
 - pumping_cost_cross_heat_exchanger_reclaimer
 - pumping_cost_cross_heat_exchanger_storage_tank
 - pumping_cost_cross_heat_exchanger_stripper
 - pumping_cost_direct_contact_cooler_TEG_dehydration
 - pumping_cost_direct_contact_cooler_absorber
 - pumping_cost_direct_contact_cooler_compressor_KOdrum
 - pumping_cost_direct_contact_cooler_compressor_house
 - pumping_cost_direct_contact_cooler_compressor_interstage_cooler
 - pumping_cost_direct_contact_cooler_cooling_tower
 - pumping_cost_direct_contact_cooler_coolingwater_pumphouse
 - pumping_cost_direct_contact_cooler_cross_heat_exchanger
 - pumping_cost_direct_contact_cooler_direct_contact_cooler
 - pumping_cost_direct_contact_cooler_lean_amine_cooler
 - pumping_cost_direct_contact_cooler_reclaimer
 - pumping_cost_direct_contact_cooler_storage_tank
 - pumping_cost_direct_contact_cooler_stripper
 - pumping_cost_lean_amine_cooler_TEG_dehydration
 - pumping_cost_lean_amine_cooler_absorber
 - pumping_cost_lean_amine_cooler_compressor_KOdrum
 - pumping_cost_lean_amine_cooler_compressor_house
 - pumping_cost_lean_amine_cooler_compressor_interstage_cooler
 - pumping_cost_lean_amine_cooler_cooling_tower
 - pumping_cost_lean_amine_cooler_coolingwater_pumphouse
 - pumping_cost_lean_amine_cooler_cross_heat_exchanger
 - pumping_cost_lean_amine_cooler_direct_contact_cooler
 - pumping_cost_lean_amine_cooler_lean_amine_cooler
 - pumping_cost_lean_amine_cooler_reclaimer
 - pumping_cost_lean_amine_cooler_storage_tank
 - pumping_cost_lean_amine_cooler_stripper
 - pumping_cost_reclaimer_TEG_dehydration - pumping_cost_reclaimer_absorber
 - pumping_cost_reclaimer_compressor_KOdrum
 - pumping_cost_reclaimer_compressor_house
 - pumping_cost_reclaimer_compressor_interstage_cooler
 - pumping_cost_reclaimer_cooling_tower
 - pumping_cost_reclaimer_coolingwater_pumphouse
 - pumping_cost_reclaimer_cross_heat_exchanger
 - pumping_cost_reclaimer_direct_contact_cooler
 - pumping_cost_reclaimer_lean_amine_cooler - pumping_cost_reclaimer_reclaimer
 - pumping_cost_reclaimer_storage_tank - pumping_cost_reclaimer_stripper
 - pumping_cost_storage_tank_TEG_dehydration
 - pumping_cost_storage_tank_absorber
 - pumping_cost_storage_tank_compressor_KOdrum
 - pumping_cost_storage_tank_compressor_house
 - pumping_cost_storage_tank_compressor_interstage_cooler
 - pumping_cost_storage_tank_cooling_tower
 - pumping_cost_storage_tank_coolingwater_pumphouse
 - pumping_cost_storage_tank_cross_heat_exchanger
 - pumping_cost_storage_tank_direct_contact_cooler
 - pumping_cost_storage_tank_lean_amine_cooler
 - pumping_cost_storage_tank_reclaimer
 - pumping_cost_storage_tank_storage_tank - pumping_cost_storage_tank_stripper
 - pumping_cost_stripper_TEG_dehydration - pumping_cost_stripper_absorber
 - pumping_cost_stripper_compressor_KOdrum
 - pumping_cost_stripper_compressor_house
 - pumping_cost_stripper_compressor_interstage_cooler
 - pumping_cost_stripper_cooling_tower
 - pumping_cost_stripper_coolingwater_pumphouse
 - pumping_cost_stripper_cross_heat_exchanger
 - pumping_cost_stripper_direct_contact_cooler
 - pumping_cost_stripper_lean_amine_cooler - pumping_cost_stripper_reclaimer
 - pumping_cost_stripper_storage_tank - pumping_cost_stripper_stripper
 + sum_pumping_cost = 0
_C256: B_absorber_storage_tank + 1000 Wy_absorber_storage_tank <= 1000
_C257: - A_absorber_storage_tank - B_absorber_storage_tank
 + D_absorber_storage_tank - L_absorber_storage_tank - R_absorber_storage_tank
 = 0
_C258: D_absorber_storage_tank - D_storage_tank_absorber = 0
_C259: 1000 E1_absorber_storage_tank + 1000 E2_absorber_storage_tank
 + X_absorber - X_storage_tank - 0.5 l_absorber - 0.5 l_storage_tank >= 76
_C26: d_reclaimer + l_reclaimer = 36
_C260: - 1000 E1_absorber_storage_tank + 1000 E2_absorber_storage_tank
 - X_absorber + X_storage_tank - 0.5 l_absorber - 0.5 l_storage_tank >= -924
_C261: 1000 E1_absorber_storage_tank - 1000 E2_absorber_storage_tank
 + Y_absorber - Y_storage_tank - 0.5 d_absorber - 0.5 d_storage_tank >= -924
_C262: - 1000 E1_absorber_storage_tank - 1000 E2_absorber_storage_tank
 - Y_absorber + Y_storage_tank - 0.5 d_absorber - 0.5 d_storage_tank >= -1924
_C263: L_absorber_storage_tank - R_storage_tank_absorber = 0
_C264: - L_storage_tank_absorber + R_absorber_storage_tank = 0
_C265: A_absorber_storage_tank - B_storage_tank_absorber = 0
_C266: - A_storage_tank_absorber + B_absorber_storage_tank = 0
_C267: Wx_absorber_storage_tank + Wx_storage_tank_absorber = 1
_C268: Wy_absorber_storage_tank + Wy_storage_tank_absorber = 1
_C269: - L_lean_amine_cooler_storage_tank + R_lean_amine_cooler_storage_tank
 - X_lean_amine_cooler + X_storage_tank = 0
_C27: X_reclaimer - 0.5 l_reclaimer >= 0
_C270: A_lean_amine_cooler_storage_tank - B_lean_amine_cooler_storage_tank
 - Y_lean_amine_cooler + Y_storage_tank = 0
_C271: R_lean_amine_cooler_storage_tank
 - 1000 Wx_lean_amine_cooler_storage_tank <= 0
_C272: L_lean_amine_cooler_storage_tank
 + 1000 Wx_lean_amine_cooler_storage_tank <= 1000
_C273: A_lean_amine_cooler_storage_tank
 - 1000 Wy_lean_amine_cooler_storage_tank <= 0
_C274: B_lean_amine_cooler_storage_tank
 + 1000 Wy_lean_amine_cooler_storage_tank <= 1000
_C275: - A_lean_amine_cooler_storage_tank - B_lean_amine_cooler_storage_tank
 + D_lean_amine_cooler_storage_tank - L_lean_amine_cooler_storage_tank
 - R_lean_amine_cooler_storage_tank = 0
_C276: D_lean_amine_cooler_storage_tank - D_storage_tank_lean_amine_cooler = 0
_C277: 1000 E1_lean_amine_cooler_storage_tank
 + 1000 E2_lean_amine_cooler_storage_tank + X_lean_amine_cooler
 - X_storage_tank - 0.5 l_lean_amine_cooler - 0.5 l_storage_tank >= 76
_C278: - 1000 E1_lean_amine_cooler_storage_tank
 + 1000 E2_lean_amine_cooler_storage_tank - X_lean_amine_cooler
 + X_storage_tank - 0.5 l_lean_amine_cooler - 0.5 l_storage_tank >= -924
_C279: 1000 E1_lean_amine_cooler_storage_tank
 - 1000 E2_lean_amine_cooler_storage_tank + Y_lean_amine_cooler
 - Y_storage_tank - 0.5 d_lean_amine_cooler - 0.5 d_storage_tank >= -924
_C28: Y_reclaimer - 0.5 d_reclaimer >= 0
_C280: - 1000 E1_lean_amine_cooler_storage_tank
 - 1000 E2_lean_amine_cooler_storage_tank - Y_lean_amine_cooler
 + Y_storage_tank - 0.5 d_lean_amine_cooler - 0.5 d_storage_tank >= -1924
_C281: L_lean_amine_cooler_storage_tank - R_storage_tank_lean_amine_cooler = 0
_C282: - L_storage_tank_lean_amine_cooler + R_lean_amine_cooler_storage_tank
 = 0
_C283: A_lean_amine_cooler_storage_tank - B_storage_tank_lean_amine_cooler = 0
_C284: - A_storage_tank_lean_amine_cooler + B_lean_amine_cooler_storage_tank
 = 0
_C285: Wx_lean_amine_cooler_storage_tank + Wx_storage_tank_lean_amine_cooler
 = 1
_C286: Wy_lean_amine_cooler_storage_tank + Wy_storage_tank_lean_amine_cooler
 = 1
_C287: - L_cross_heat_exchanger_storage_tank
 + R_cross_heat_exchanger_storage_tank - X_cross_heat_exchanger
 + X_storage_tank = 0
_C288: A_cross_heat_exchanger_storage_tank
 - B_cross_heat_exchanger_storage_tank - Y_cross_heat_exchanger
 + Y_storage_tank = 0
_C289: R_cross_heat_exchanger_storage_tank
 - 1000 Wx_cross_heat_exchanger_storage_tank <= 0
_C29: - 29.5 O_compressor_house + l_compressor_house = 8
_C290: L_cross_heat_exchanger_storage_tank
 + 1000 Wx_cross_heat_exchanger_storage_tank <= 1000
_C291: A_cross_heat_exchanger_storage_tank
 - 1000 Wy_cross_heat_exchanger_storage_tank <= 0
_C292: B_cross_heat_exchanger_storage_tank
 + 1000 Wy_cross_heat_exchanger_storage_tank <= 1000
_C293: - A_cross_heat_exchanger_storage_tank
 - B_cross_heat_exchanger_storage_tank + D_cross_heat_exchanger_storage_tank
 - L_cross_heat_exchanger_storage_tank - R_cross_heat_exchanger_storage_tank
 = 0
_C294: D_cross_heat_exchanger_storage_tank
 - D_storage_tank_cross_heat_exchanger = 0
_C295: 1000 E1_cross_heat_exchanger_storage_tank
 + 1000 E2_cross_heat_exchanger_storage_tank + X_cross_heat_exchanger
 - X_storage_tank - 0.5 l_cross_heat_exchanger - 0.5 l_storage_tank >= 76
_C296: - 1000 E1_cross_heat_exchanger_storage_tank
 + 1000 E2_cross_heat_exchanger_storage_tank - X_cross_heat_exchanger
 + X_storage_tank - 0.5 l_cross_heat_exchanger - 0.5 l_storage_tank >= -924
_C297: 1000 E1_cross_heat_exchanger_storage_tank
 - 1000 E2_cross_heat_exchanger_storage_tank + Y_cross_heat_exchanger
 - Y_storage_tank - 0.5 d_cross_heat_exchanger - 0.5 d_storage_tank >= -924
_C298: - 1000 E1_cross_heat_exchanger_storage_tank
 - 1000 E2_cross_heat_exchanger_storage_tank - Y_cross_heat_exchanger
 + Y_storage_tank - 0.5 d_cross_heat_exchanger - 0.5 d_storage_tank >= -1924
_C299: L_cross_heat_exchanger_storage_tank
 - R_storage_tank_cross_heat_exchanger = 0
_C3: X_direct_contact_cooler - 0.5 l_direct_contact_cooler >= 0
_C30: d_compressor_house + l_compressor_house = 45.5
_C300: - L_storage_tank_cross_heat_exchanger
 + R_cross_heat_exchanger_storage_tank = 0
_C301: A_cross_heat_exchanger_storage_tank
 - B_storage_tank_cross_heat_exchanger = 0
_C302: - A_storage_tank_cross_heat_exchanger
 + B_cross_heat_exchanger_storage_tank = 0
_C303: Wx_cross_heat_exchanger_storage_tank
 + Wx_storage_tank_cross_heat_exchanger = 1
_C304: Wy_cross_heat_exchanger_storage_tank
 + Wy_storage_tank_cross_heat_exchanger = 1
_C305: - L_stripper_storage_tank + R_stripper_storage_tank + X_storage_tank
 - X_stripper = 0
_C306: A_stripper_storage_tank - B_stripper_storage_tank + Y_storage_tank
 - Y_stripper = 0
_C307: R_stripper_storage_tank - 1000 Wx_stripper_storage_tank <= 0
_C308: L_stripper_storage_tank + 1000 Wx_stripper_storage_tank <= 1000
_C309: A_stripper_storage_tank - 1000 Wy_stripper_storage_tank <= 0
_C31: X_compressor_house - 0.5 l_compressor_house >= 0
_C310: B_stripper_storage_tank + 1000 Wy_stripper_storage_tank <= 1000
_C311: - A_stripper_storage_tank - B_stripper_storage_tank
 + D_stripper_storage_tank - L_stripper_storage_tank - R_stripper_storage_tank
 = 0
_C312: - D_storage_tank_stripper + D_stripper_storage_tank = 0
_C313: 1000 E1_stripper_storage_tank + 1000 E2_stripper_storage_tank
 - X_storage_tank + X_stripper - 0.5 l_storage_tank - 0.5 l_stripper >= 76
_C314: - 1000 E1_stripper_storage_tank + 1000 E2_stripper_storage_tank
 + X_storage_tank - X_stripper - 0.5 l_storage_tank - 0.5 l_stripper >= -924
_C315: 1000 E1_stripper_storage_tank - 1000 E2_stripper_storage_tank
 - Y_storage_tank + Y_stripper - 0.5 d_storage_tank - 0.5 d_stripper >= -924
_C316: - 1000 E1_stripper_storage_tank - 1000 E2_stripper_storage_tank
 + Y_storage_tank - Y_stripper - 0.5 d_storage_tank - 0.5 d_stripper >= -1924
_C317: L_stripper_storage_tank - R_storage_tank_stripper = 0
_C318: - L_storage_tank_stripper + R_stripper_storage_tank = 0
_C319: A_stripper_storage_tank - B_storage_tank_stripper = 0
_C32: Y_compressor_house - 0.5 d_compressor_house >= 0
_C320: - A_storage_tank_stripper + B_stripper_storage_tank = 0
_C321: Wx_storage_tank_stripper + Wx_stripper_storage_tank = 1
_C322: Wy_storage_tank_stripper + Wy_stripper_storage_tank = 1
_C323: - L_direct_contact_cooler_reclaimer + R_direct_contact_cooler_reclaimer
 - X_direct_contact_cooler + X_reclaimer = 0
_C324: A_direct_contact_cooler_reclaimer - B_direct_contact_cooler_reclaimer
 - Y_direct_contact_cooler + Y_reclaimer = 0
_C325: R_direct_contact_cooler_reclaimer
 - 1000 Wx_direct_contact_cooler_reclaimer <= 0
_C326: L_direct_contact_cooler_reclaimer
 + 1000 Wx_direct_contact_cooler_reclaimer <= 1000
_C327: A_direct_contact_cooler_reclaimer
 - 1000 Wy_direct_contact_cooler_reclaimer <= 0
_C328: B_direct_contact_cooler_reclaimer
 + 1000 Wy_direct_contact_cooler_reclaimer <= 1000
_C329: - A_direct_contact_cooler_reclaimer - B_direct_contact_cooler_reclaimer
 + D_direct_contact_cooler_reclaimer - L_direct_contact_cooler_reclaimer
 - R_direct_contact_cooler_reclaimer = 0
_C33: 0 O_compressor_KOdrum + l_compressor_KOdrum = 26
_C330: D_direct_contact_cooler_reclaimer - D_reclaimer_direct_contact_cooler
 = 0
_C331: 1000 E1_direct_contact_cooler_reclaimer
 + 1000 E2_direct_contact_cooler_reclaimer + X_direct_contact_cooler
 - X_reclaimer - 0.5 l_direct_contact_cooler - 0.5 l_reclaimer >= 15
_C332: - 1000 E1_direct_contact_cooler_reclaimer
 + 1000 E2_direct_contact_cooler_reclaimer - X_direct_contact_cooler
 + X_reclaimer - 0.5 l_direct_contact_cooler - 0.5 l_reclaimer >= -985
_C333: 1000 E1_direct_contact_cooler_reclaimer
 - 1000 E2_direct_contact_cooler_reclaimer + Y_direct_contact_cooler
 - Y_reclaimer - 0.5 d_direct_contact_cooler - 0.5 d_reclaimer >= -985
_C334: - 1000 E1_direct_contact_cooler_reclaimer
 - 1000 E2_direct_contact_cooler_reclaimer - Y_direct_contact_cooler
 + Y_reclaimer - 0.5 d_direct_contact_cooler - 0.5 d_reclaimer >= -1985
_C335: L_direct_contact_cooler_reclaimer - R_reclaimer_direct_contact_cooler
 = 0
_C336: - L_reclaimer_direct_contact_cooler + R_direct_contact_cooler_reclaimer
 = 0
_C337: A_direct_contact_cooler_reclaimer - B_reclaimer_direct_contact_cooler
 = 0
_C338: - A_reclaimer_direct_contact_cooler + B_direct_contact_cooler_reclaimer
 = 0
_C339: Wx_direct_contact_cooler_reclaimer + Wx_reclaimer_direct_contact_cooler
 = 1
_C34: d_compressor_KOdrum + l_compressor_KOdrum = 52
_C340: Wy_direct_contact_cooler_reclaimer + Wy_reclaimer_direct_contact_cooler
 = 1
_C341: - L_absorber_reclaimer + R_absorber_reclaimer - X_absorber
 + X_reclaimer = 0
_C342: A_absorber_reclaimer - B_absorber_reclaimer - Y_absorber + Y_reclaimer
 = 0
_C343: R_absorber_reclaimer - 1000 Wx_absorber_reclaimer <= 0
_C344: L_absorber_reclaimer + 1000 Wx_absorber_reclaimer <= 1000
_C345: A_absorber_reclaimer - 1000 Wy_absorber_reclaimer <= 0
_C346: B_absorber_reclaimer + 1000 Wy_absorber_reclaimer <= 1000
_C347: - A_absorber_reclaimer - B_absorber_reclaimer + D_absorber_reclaimer
 - L_absorber_reclaimer - R_absorber_reclaimer = 0
_C348: D_absorber_reclaimer - D_reclaimer_absorber = 0
_C349: 1000 E1_absorber_reclaimer + 1000 E2_absorber_reclaimer + X_absorber
 - X_reclaimer - 0.5 l_absorber - 0.5 l_reclaimer >= 15
_C35: X_compressor_KOdrum - 0.5 l_compressor_KOdrum >= 0
_C350: - 1000 E1_absorber_reclaimer + 1000 E2_absorber_reclaimer - X_absorber
 + X_reclaimer - 0.5 l_absorber - 0.5 l_reclaimer >= -985
_C351: 1000 E1_absorber_reclaimer - 1000 E2_absorber_reclaimer + Y_absorber
 - Y_reclaimer - 0.5 d_absorber - 0.5 d_reclaimer >= -985
_C352: - 1000 E1_absorber_reclaimer - 1000 E2_absorber_reclaimer - Y_absorber
 + Y_reclaimer - 0.5 d_absorber - 0.5 d_reclaimer >= -1985
_C353: L_absorber_reclaimer - R_reclaimer_absorber = 0
_C354: - L_reclaimer_absorber + R_absorber_reclaimer = 0
_C355: A_absorber_reclaimer - B_reclaimer_absorber = 0
_C356: - A_reclaimer_absorber + B_absorber_reclaimer = 0
_C357: Wx_absorber_reclaimer + Wx_reclaimer_absorber = 1
_C358: Wy_absorber_reclaimer + Wy_reclaimer_absorber = 1
_C359: - L_lean_amine_cooler_reclaimer + R_lean_amine_cooler_reclaimer
 - X_lean_amine_cooler + X_reclaimer = 0
_C36: Y_compressor_KOdrum - 0.5 d_compressor_KOdrum >= 0
_C360: A_lean_amine_cooler_reclaimer - B_lean_amine_cooler_reclaimer
 - Y_lean_amine_cooler + Y_reclaimer = 0
_C361: R_lean_amine_cooler_reclaimer - 1000 Wx_lean_amine_cooler_reclaimer
 <= 0
_C362: L_lean_amine_cooler_reclaimer + 1000 Wx_lean_amine_cooler_reclaimer
 <= 1000
_C363: A_lean_amine_cooler_reclaimer - 1000 Wy_lean_amine_cooler_reclaimer
 <= 0
_C364: B_lean_amine_cooler_reclaimer + 1000 Wy_lean_amine_cooler_reclaimer
 <= 1000
_C365: - A_lean_amine_cooler_reclaimer - B_lean_amine_cooler_reclaimer
 + D_lean_amine_cooler_reclaimer - L_lean_amine_cooler_reclaimer
 - R_lean_amine_cooler_reclaimer = 0
_C366: D_lean_amine_cooler_reclaimer - D_reclaimer_lean_amine_cooler = 0
_C367: 1000 E1_lean_amine_cooler_reclaimer
 + 1000 E2_lean_amine_cooler_reclaimer + X_lean_amine_cooler - X_reclaimer
 - 0.5 l_lean_amine_cooler - 0.5 l_reclaimer >= 15
_C368: - 1000 E1_lean_amine_cooler_reclaimer
 + 1000 E2_lean_amine_cooler_reclaimer - X_lean_amine_cooler + X_reclaimer
 - 0.5 l_lean_amine_cooler - 0.5 l_reclaimer >= -985
_C369: 1000 E1_lean_amine_cooler_reclaimer
 - 1000 E2_lean_amine_cooler_reclaimer + Y_lean_amine_cooler - Y_reclaimer
 - 0.5 d_lean_amine_cooler - 0.5 d_reclaimer >= -985
_C37: 0 O_compressor_interstage_cooler + l_compressor_interstage_cooler = 22
_C370: - 1000 E1_lean_amine_cooler_reclaimer
 - 1000 E2_lean_amine_cooler_reclaimer - Y_lean_amine_cooler + Y_reclaimer
 - 0.5 d_lean_amine_cooler - 0.5 d_reclaimer >= -1985
_C371: L_lean_amine_cooler_reclaimer - R_reclaimer_lean_amine_cooler = 0
_C372: - L_reclaimer_lean_amine_cooler + R_lean_amine_cooler_reclaimer = 0
_C373: A_lean_amine_cooler_reclaimer - B_reclaimer_lean_amine_cooler = 0
_C374: - A_reclaimer_lean_amine_cooler + B_lean_amine_cooler_reclaimer = 0
_C375: Wx_lean_amine_cooler_reclaimer + Wx_reclaimer_lean_amine_cooler = 1
_C376: Wy_lean_amine_cooler_reclaimer + Wy_reclaimer_lean_amine_cooler = 1
_C377: - L_cross_heat_exchanger_reclaimer + R_cross_heat_exchanger_reclaimer
 - X_cross_heat_exchanger + X_reclaimer = 0
_C378: A_cross_heat_exchanger_reclaimer - B_cross_heat_exchanger_reclaimer
 - Y_cross_heat_exchanger + Y_reclaimer = 0
_C379: R_cross_heat_exchanger_reclaimer
 - 1000 Wx_cross_heat_exchanger_reclaimer <= 0
_C38: d_compressor_interstage_cooler + l_compressor_interstage_cooler = 44
_C380: L_cross_heat_exchanger_reclaimer
 + 1000 Wx_cross_heat_exchanger_reclaimer <= 1000
_C381: A_cross_heat_exchanger_reclaimer
 - 1000 Wy_cross_heat_exchanger_reclaimer <= 0
_C382: B_cross_heat_exchanger_reclaimer
 + 1000 Wy_cross_heat_exchanger_reclaimer <= 1000
_C383: - A_cross_heat_exchanger_reclaimer - B_cross_heat_exchanger_reclaimer
 + D_cross_heat_exchanger_reclaimer - L_cross_heat_exchanger_reclaimer
 - R_cross_heat_exchanger_reclaimer = 0
_C384: D_cross_heat_exchanger_reclaimer - D_reclaimer_cross_heat_exchanger = 0
_C385: 1000 E1_cross_heat_exchanger_reclaimer
 + 1000 E2_cross_heat_exchanger_reclaimer + X_cross_heat_exchanger
 - X_reclaimer - 0.5 l_cross_heat_exchanger - 0.5 l_reclaimer >= 15
_C386: - 1000 E1_cross_heat_exchanger_reclaimer
 + 1000 E2_cross_heat_exchanger_reclaimer - X_cross_heat_exchanger
 + X_reclaimer - 0.5 l_cross_heat_exchanger - 0.5 l_reclaimer >= -985
_C387: 1000 E1_cross_heat_exchanger_reclaimer
 - 1000 E2_cross_heat_exchanger_reclaimer + Y_cross_heat_exchanger
 - Y_reclaimer - 0.5 d_cross_heat_exchanger - 0.5 d_reclaimer >= -985
_C388: - 1000 E1_cross_heat_exchanger_reclaimer
 - 1000 E2_cross_heat_exchanger_reclaimer - Y_cross_heat_exchanger
 + Y_reclaimer - 0.5 d_cross_heat_exchanger - 0.5 d_reclaimer >= -1985
_C389: L_cross_heat_exchanger_reclaimer - R_reclaimer_cross_heat_exchanger = 0
_C39: X_compressor_interstage_cooler - 0.5 l_compressor_interstage_cooler >= 0
_C390: - L_reclaimer_cross_heat_exchanger + R_cross_heat_exchanger_reclaimer
 = 0
_C391: A_cross_heat_exchanger_reclaimer - B_reclaimer_cross_heat_exchanger = 0
_C392: - A_reclaimer_cross_heat_exchanger + B_cross_heat_exchanger_reclaimer
 = 0
_C393: Wx_cross_heat_exchanger_reclaimer + Wx_reclaimer_cross_heat_exchanger
 = 1
_C394: Wy_cross_heat_exchanger_reclaimer + Wy_reclaimer_cross_heat_exchanger
 = 1
_C395: - L_stripper_reclaimer + R_stripper_reclaimer + X_reclaimer
 - X_stripper = 0
_C396: A_stripper_reclaimer - B_stripper_reclaimer + Y_reclaimer - Y_stripper
 = 0
_C397: R_stripper_reclaimer - 1000 Wx_stripper_reclaimer <= 0
_C398: L_stripper_reclaimer + 1000 Wx_stripper_reclaimer <= 1000
_C399: A_stripper_reclaimer - 1000 Wy_stripper_reclaimer <= 0
_C4: Y_direct_contact_cooler - 0.5 d_direct_contact_cooler >= 0
_C40: Y_compressor_interstage_cooler - 0.5 d_compressor_interstage_cooler >= 0
_C400: B_stripper_reclaimer + 1000 Wy_stripper_reclaimer <= 1000
_C401: - A_stripper_reclaimer - B_stripper_reclaimer + D_stripper_reclaimer
 - L_stripper_reclaimer - R_stripper_reclaimer = 0
_C402: - D_reclaimer_stripper + D_stripper_reclaimer = 0
_C403: 1000 E1_stripper_reclaimer + 1000 E2_stripper_reclaimer - X_reclaimer
 + X_stripper - 0.5 l_reclaimer - 0.5 l_stripper >= 15
_C404: - 1000 E1_stripper_reclaimer + 1000 E2_stripper_reclaimer + X_reclaimer
 - X_stripper - 0.5 l_reclaimer - 0.5 l_stripper >= -985
_C405: 1000 E1_stripper_reclaimer - 1000 E2_stripper_reclaimer - Y_reclaimer
 + Y_stripper - 0.5 d_reclaimer - 0.5 d_stripper >= -985
_C406: - 1000 E1_stripper_reclaimer - 1000 E2_stripper_reclaimer + Y_reclaimer
 - Y_stripper - 0.5 d_reclaimer - 0.5 d_stripper >= -1985
_C407: L_stripper_reclaimer - R_reclaimer_stripper = 0
_C408: - L_reclaimer_stripper + R_stripper_reclaimer = 0
_C409: A_stripper_reclaimer - B_reclaimer_stripper = 0
_C41: 0 O_TEG_dehydration + l_TEG_dehydration = 14
_C410: - A_reclaimer_stripper + B_stripper_reclaimer = 0
_C411: Wx_reclaimer_stripper + Wx_stripper_reclaimer = 1
_C412: Wy_reclaimer_stripper + Wy_stripper_reclaimer = 1
_C413: - L_storage_tank_reclaimer + R_storage_tank_reclaimer + X_reclaimer
 - X_storage_tank = 0
_C414: A_storage_tank_reclaimer - B_storage_tank_reclaimer + Y_reclaimer
 - Y_storage_tank = 0
_C415: R_storage_tank_reclaimer - 1000 Wx_storage_tank_reclaimer <= 0
_C416: L_storage_tank_reclaimer + 1000 Wx_storage_tank_reclaimer <= 1000
_C417: A_storage_tank_reclaimer - 1000 Wy_storage_tank_reclaimer <= 0
_C418: B_storage_tank_reclaimer + 1000 Wy_storage_tank_reclaimer <= 1000
_C419: - A_storage_tank_reclaimer - B_storage_tank_reclaimer
 + D_storage_tank_reclaimer - L_storage_tank_reclaimer
 - R_storage_tank_reclaimer = 0
_C42: d_TEG_dehydration + l_TEG_dehydration = 28
_C420: - D_reclaimer_storage_tank + D_storage_tank_reclaimer = 0
_C421: 1000 E1_storage_tank_reclaimer + 1000 E2_storage_tank_reclaimer
 - X_reclaimer + X_storage_tank - 0.5 l_reclaimer - 0.5 l_storage_tank >= 76
_C422: - 1000 E1_storage_tank_reclaimer + 1000 E2_storage_tank_reclaimer
 + X_reclaimer - X_storage_tank - 0.5 l_reclaimer - 0.5 l_storage_tank >= -924
_C423: 1000 E1_storage_tank_reclaimer - 1000 E2_storage_tank_reclaimer
 - Y_reclaimer + Y_storage_tank - 0.5 d_reclaimer - 0.5 d_storage_tank >= -924
_C424: - 1000 E1_storage_tank_reclaimer - 1000 E2_storage_tank_reclaimer
 + Y_reclaimer - Y_storage_tank - 0.5 d_reclaimer - 0.5 d_storage_tank
 >= -1924
_C425: L_storage_tank_reclaimer - R_reclaimer_storage_tank = 0
_C426: - L_reclaimer_storage_tank + R_storage_tank_reclaimer = 0
_C427: A_storage_tank_reclaimer - B_reclaimer_storage_tank = 0
_C428: - A_reclaimer_storage_tank + B_storage_tank_reclaimer = 0
_C429: Wx_reclaimer_storage_tank + Wx_storage_tank_reclaimer = 1
_C43: X_TEG_dehydration - 0.5 l_TEG_dehydration >= 0
_C430: Wy_reclaimer_storage_tank + Wy_storage_tank_reclaimer = 1
_C431: - L_direct_contact_cooler_compressor_house
 + R_direct_contact_cooler_compressor_house + X_compressor_house
 - X_direct_contact_cooler = 0
_C432: A_direct_contact_cooler_compressor_house
 - B_direct_contact_cooler_compressor_house + Y_compressor_house
 - Y_direct_contact_cooler = 0
_C433: R_direct_contact_cooler_compressor_house
 - 1000 Wx_direct_contact_cooler_compressor_house <= 0
_C434: L_direct_contact_cooler_compressor_house
 + 1000 Wx_direct_contact_cooler_compressor_house <= 1000
_C435: A_direct_contact_cooler_compressor_house
 - 1000 Wy_direct_contact_cooler_compressor_house <= 0
_C436: B_direct_contact_cooler_compressor_house
 + 1000 Wy_direct_contact_cooler_compressor_house <= 1000
_C437: - A_direct_contact_cooler_compressor_house
 - B_direct_contact_cooler_compressor_house
 + D_direct_contact_cooler_compressor_house
 - L_direct_contact_cooler_compressor_house
 - R_direct_contact_cooler_compressor_house = 0
_C438: - D_compressor_house_direct_contact_cooler
 + D_direct_contact_cooler_compressor_house = 0
_C439: 1000 E1_direct_contact_cooler_compressor_house
 + 1000 E2_direct_contact_cooler_compressor_house - X_compressor_house
 + X_direct_contact_cooler - 0.5 l_compressor_house
 - 0.5 l_direct_contact_cooler >= 9
_C44: Y_TEG_dehydration - 0.5 d_TEG_dehydration >= 0
_C440: - 1000 E1_direct_contact_cooler_compressor_house
 + 1000 E2_direct_contact_cooler_compressor_house + X_compressor_house
 - X_direct_contact_cooler - 0.5 l_compressor_house
 - 0.5 l_direct_contact_cooler >= -991
_C441: 1000 E1_direct_contact_cooler_compressor_house
 - 1000 E2_direct_contact_cooler_compressor_house - Y_compressor_house
 + Y_direct_contact_cooler - 0.5 d_compressor_house
 - 0.5 d_direct_contact_cooler >= -991
_C442: - 1000 E1_direct_contact_cooler_compressor_house
 - 1000 E2_direct_contact_cooler_compressor_house + Y_compressor_house
 - Y_direct_contact_cooler - 0.5 d_compressor_house
 - 0.5 d_direct_contact_cooler >= -1991
_C443: L_direct_contact_cooler_compressor_house
 - R_compressor_house_direct_contact_cooler = 0
_C444: - L_compressor_house_direct_contact_cooler
 + R_direct_contact_cooler_compressor_house = 0
_C445: A_direct_contact_cooler_compressor_house
 - B_compressor_house_direct_contact_cooler = 0
_C446: - A_compressor_house_direct_contact_cooler
 + B_direct_contact_cooler_compressor_house = 0
_C447: Wx_compressor_house_direct_contact_cooler
 + Wx_direct_contact_cooler_compressor_house = 1
_C448: Wy_compressor_house_direct_contact_cooler
 + Wy_direct_contact_cooler_compressor_house = 1
_C449: - L_absorber_compressor_house + R_absorber_compressor_house
 - X_absorber + X_compressor_house = 0
_C45: 0 O_cooling_tower + l_cooling_tower = 52
_C450: A_absorber_compressor_house - B_absorber_compressor_house - Y_absorber
 + Y_compressor_house = 0
_C451: R_absorber_compressor_house - 1000 Wx_absorber_compressor_house <= 0
_C452: L_absorber_compressor_house + 1000 Wx_absorber_compressor_house <= 1000
_C453: A_absorber_compressor_house - 1000 Wy_absorber_compressor_house <= 0
_C454: B_absorber_compressor_house + 1000 Wy_absorber_compressor_house <= 1000
_C455: - A_absorber_compressor_house - B_absorber_compressor_house
 + D_absorber_compressor_house - L_absorber_compressor_house
 - R_absorber_compressor_house = 0
_C456: D_absorber_compressor_house - D_compressor_house_absorber = 0
_C457: 1000 E1_absorber_compressor_house + 1000 E2_absorber_compressor_house
 + X_absorber - X_compressor_house - 0.5 l_absorber - 0.5 l_compressor_house
 >= 9
_C458: - 1000 E1_absorber_compressor_house + 1000 E2_absorber_compressor_house
 - X_absorber + X_compressor_house - 0.5 l_absorber - 0.5 l_compressor_house
 >= -991
_C459: 1000 E1_absorber_compressor_house - 1000 E2_absorber_compressor_house
 + Y_absorber - Y_compressor_house - 0.5 d_absorber - 0.5 d_compressor_house
 >= -991
_C46: d_cooling_tower + l_cooling_tower = 104
_C460: - 1000 E1_absorber_compressor_house - 1000 E2_absorber_compressor_house
 - Y_absorber + Y_compressor_house - 0.5 d_absorber - 0.5 d_compressor_house
 >= -1991
_C461: L_absorber_compressor_house - R_compressor_house_absorber = 0
_C462: - L_compressor_house_absorber + R_absorber_compressor_house = 0
_C463: A_absorber_compressor_house - B_compressor_house_absorber = 0
_C464: - A_compressor_house_absorber + B_absorber_compressor_house = 0
_C465: Wx_absorber_compressor_house + Wx_compressor_house_absorber = 1
_C466: Wy_absorber_compressor_house + Wy_compressor_house_absorber = 1
_C467: - L_lean_amine_cooler_compressor_house
 + R_lean_amine_cooler_compressor_house + X_compressor_house
 - X_lean_amine_cooler = 0
_C468: A_lean_amine_cooler_compressor_house
 - B_lean_amine_cooler_compressor_house + Y_compressor_house
 - Y_lean_amine_cooler = 0
_C469: R_lean_amine_cooler_compressor_house
 - 1000 Wx_lean_amine_cooler_compressor_house <= 0
_C47: X_cooling_tower - 0.5 l_cooling_tower >= 0
_C470: L_lean_amine_cooler_compressor_house
 + 1000 Wx_lean_amine_cooler_compressor_house <= 1000
_C471: A_lean_amine_cooler_compressor_house
 - 1000 Wy_lean_amine_cooler_compressor_house <= 0
_C472: B_lean_amine_cooler_compressor_house
 + 1000 Wy_lean_amine_cooler_compressor_house <= 1000
_C473: - A_lean_amine_cooler_compressor_house
 - B_lean_amine_cooler_compressor_house + D_lean_amine_cooler_compressor_house
 - L_lean_amine_cooler_compressor_house - R_lean_amine_cooler_compressor_house
 = 0
_C474: - D_compressor_house_lean_amine_cooler
 + D_lean_amine_cooler_compressor_house = 0
_C475: 1000 E1_lean_amine_cooler_compressor_house
 + 1000 E2_lean_amine_cooler_compressor_house - X_compressor_house
 + X_lean_amine_cooler - 0.5 l_compressor_house - 0.5 l_lean_amine_cooler >= 9
_C476: - 1000 E1_lean_amine_cooler_compressor_house
 + 1000 E2_lean_amine_cooler_compressor_house + X_compressor_house
 - X_lean_amine_cooler - 0.5 l_compressor_house - 0.5 l_lean_amine_cooler
 >= -991
_C477: 1000 E1_lean_amine_cooler_compressor_house
 - 1000 E2_lean_amine_cooler_compressor_house - Y_compressor_house
 + Y_lean_amine_cooler - 0.5 d_compressor_house - 0.5 d_lean_amine_cooler
 >= -991
_C478: - 1000 E1_lean_amine_cooler_compressor_house
 - 1000 E2_lean_amine_cooler_compressor_house + Y_compressor_house
 - Y_lean_amine_cooler - 0.5 d_compressor_house - 0.5 d_lean_amine_cooler
 >= -1991
_C479: L_lean_amine_cooler_compressor_house
 - R_compressor_house_lean_amine_cooler = 0
_C48: Y_cooling_tower - 0.5 d_cooling_tower >= 0
_C480: - L_compressor_house_lean_amine_cooler
 + R_lean_amine_cooler_compressor_house = 0
_C481: A_lean_amine_cooler_compressor_house
 - B_compressor_house_lean_amine_cooler = 0
_C482: - A_compressor_house_lean_amine_cooler
 + B_lean_amine_cooler_compressor_house = 0
_C483: Wx_compressor_house_lean_amine_cooler
 + Wx_lean_amine_cooler_compressor_house = 1
_C484: Wy_compressor_house_lean_amine_cooler
 + Wy_lean_amine_cooler_compressor_house = 1
_C485: - L_cross_heat_exchanger_compressor_house
 + R_cross_heat_exchanger_compressor_house + X_compressor_house
 - X_cross_heat_exchanger = 0
_C486: A_cross_heat_exchanger_compressor_house
 - B_cross_heat_exchanger_compressor_house + Y_compressor_house
 - Y_cross_heat_exchanger = 0
_C487: R_cross_heat_exchanger_compressor_house
 - 1000 Wx_cross_heat_exchanger_compressor_house <= 0
_C488: L_cross_heat_exchanger_compressor_house
 + 1000 Wx_cross_heat_exchanger_compressor_house <= 1000
_C489: A_cross_heat_exchanger_compressor_house
 - 1000 Wy_cross_heat_exchanger_compressor_house <= 0
_C49: - 9 O_coolingwater_pumphouse + l_coolingwater_pumphouse = 8
_C490: B_cross_heat_exchanger_compressor_house
 + 1000 Wy_cross_heat_exchanger_compressor_house <= 1000
_C491: - A_cross_heat_exchanger_compressor_house
 - B_cross_heat_exchanger_compressor_house
 + D_cross_heat_exchanger_compressor_house
 - L_cross_heat_exchanger_compressor_house
 - R_cross_heat_exchanger_compressor_house = 0
_C492: - D_compressor_house_cross_heat_exchanger
 + D_cross_heat_exchanger_compressor_house = 0
_C493: 1000 E1_cross_heat_exchanger_compressor_house
 + 1000 E2_cross_heat_exchanger_compressor_house - X_compressor_house
 + X_cross_heat_exchanger - 0.5 l_compressor_house
 - 0.5 l_cross_heat_exchanger >= 9
_C494: - 1000 E1_cross_heat_exchanger_compressor_house
 + 1000 E2_cross_heat_exchanger_compressor_house + X_compressor_house
 - X_cross_heat_exchanger - 0.5 l_compressor_house
 - 0.5 l_cross_heat_exchanger >= -991
_C495: 1000 E1_cross_heat_exchanger_compressor_house
 - 1000 E2_cross_heat_exchanger_compressor_house - Y_compressor_house
 + Y_cross_heat_exchanger - 0.5 d_compressor_house
 - 0.5 d_cross_heat_exchanger >= -991
_C496: - 1000 E1_cross_heat_exchanger_compressor_house
 - 1000 E2_cross_heat_exchanger_compressor_house + Y_compressor_house
 - Y_cross_heat_exchanger - 0.5 d_compressor_house
 - 0.5 d_cross_heat_exchanger >= -1991
_C497: L_cross_heat_exchanger_compressor_house
 - R_compressor_house_cross_heat_exchanger = 0
_C498: - L_compressor_house_cross_heat_exchanger
 + R_cross_heat_exchanger_compressor_house = 0
_C499: A_cross_heat_exchanger_compressor_house
 - B_compressor_house_cross_heat_exchanger = 0
_C5: 0 O_absorber + l_absorber = 46
_C50: d_coolingwater_pumphouse + l_coolingwater_pumphouse = 25
_C500: - A_compressor_house_cross_heat_exchanger
 + B_cross_heat_exchanger_compressor_house = 0
_C501: Wx_compressor_house_cross_heat_exchanger
 + Wx_cross_heat_exchanger_compressor_house = 1
_C502: Wy_compressor_house_cross_heat_exchanger
 + Wy_cross_heat_exchanger_compressor_house = 1
_C503: - L_stripper_compressor_house + R_stripper_compressor_house
 + X_compressor_house - X_stripper = 0
_C504: A_stripper_compressor_house - B_stripper_compressor_house
 + Y_compressor_house - Y_stripper = 0
_C505: R_stripper_compressor_house - 1000 Wx_stripper_compressor_house <= 0
_C506: L_stripper_compressor_house + 1000 Wx_stripper_compressor_house <= 1000
_C507: A_stripper_compressor_house - 1000 Wy_stripper_compressor_house <= 0
_C508: B_stripper_compressor_house + 1000 Wy_stripper_compressor_house <= 1000
_C509: - A_stripper_compressor_house - B_stripper_compressor_house
 + D_stripper_compressor_house - L_stripper_compressor_house
 - R_stripper_compressor_house = 0
_C51: X_coolingwater_pumphouse - 0.5 l_coolingwater_pumphouse >= 0
_C510: - D_compressor_house_stripper + D_stripper_compressor_house = 0
_C511: 1000 E1_stripper_compressor_house + 1000 E2_stripper_compressor_house
 - X_compressor_house + X_stripper - 0.5 l_compressor_house - 0.5 l_stripper
 >= 9
_C512: - 1000 E1_stripper_compressor_house + 1000 E2_stripper_compressor_house
 + X_compressor_house - X_stripper - 0.5 l_compressor_house - 0.5 l_stripper
 >= -991
_C513: 1000 E1_stripper_compressor_house - 1000 E2_stripper_compressor_house
 - Y_compressor_house + Y_stripper - 0.5 d_compressor_house - 0.5 d_stripper
 >= -991
_C514: - 1000 E1_stripper_compressor_house - 1000 E2_stripper_compressor_house
 + Y_compressor_house - Y_stripper - 0.5 d_compressor_house - 0.5 d_stripper
 >= -1991
_C515: L_stripper_compressor_house - R_compressor_house_stripper = 0
_C516: - L_compressor_house_stripper + R_stripper_compressor_house = 0
_C517: A_stripper_compressor_house - B_compressor_house_stripper = 0
_C518: - A_compressor_house_stripper + B_stripper_compressor_house = 0
_C519: Wx_compressor_house_stripper + Wx_stripper_compressor_house = 1
_C52: Y_coolingwater_pumphouse - 0.5 d_coolingwater_pumphouse >= 0
_C520: Wy_compressor_house_stripper + Wy_stripper_compressor_house = 1
_C521: - L_storage_tank_compressor_house + R_storage_tank_compressor_house
 + X_compressor_house - X_storage_tank = 0
_C522: A_storage_tank_compressor_house - B_storage_tank_compressor_house
 + Y_compressor_house - Y_storage_tank = 0
_C523: R_storage_tank_compressor_house - 1000 Wx_storage_tank_compressor_house
 <= 0
_C524: L_storage_tank_compressor_house + 1000 Wx_storage_tank_compressor_house
 <= 1000
_C525: A_storage_tank_compressor_house - 1000 Wy_storage_tank_compressor_house
 <= 0
_C526: B_storage_tank_compressor_house + 1000 Wy_storage_tank_compressor_house
 <= 1000
_C527: - A_storage_tank_compressor_house - B_storage_tank_compressor_house
 + D_storage_tank_compressor_house - L_storage_tank_compressor_house
 - R_storage_tank_compressor_house = 0
_C528: - D_compressor_house_storage_tank + D_storage_tank_compressor_house = 0
_C529: 1000 E1_storage_tank_compressor_house
 + 1000 E2_storage_tank_compressor_house - X_compressor_house + X_storage_tank
 - 0.5 l_compressor_house - 0.5 l_storage_tank >= 76
_C53: - L_direct_contact_cooler_absorber + R_direct_contact_cooler_absorber
 + X_absorber - X_direct_contact_cooler = 0
_C530: - 1000 E1_storage_tank_compressor_house
 + 1000 E2_storage_tank_compressor_house + X_compressor_house - X_storage_tank
 - 0.5 l_compressor_house - 0.5 l_storage_tank >= -924
_C531: 1000 E1_storage_tank_compressor_house
 - 1000 E2_storage_tank_compressor_house - Y_compressor_house + Y_storage_tank
 - 0.5 d_compressor_house - 0.5 d_storage_tank >= -924
_C532: - 1000 E1_storage_tank_compressor_house
 - 1000 E2_storage_tank_compressor_house + Y_compressor_house - Y_storage_tank
 - 0.5 d_compressor_house - 0.5 d_storage_tank >= -1924
_C533: L_storage_tank_compressor_house - R_compressor_house_storage_tank = 0
_C534: - L_compressor_house_storage_tank + R_storage_tank_compressor_house = 0
_C535: A_storage_tank_compressor_house - B_compressor_house_storage_tank = 0
_C536: - A_compressor_house_storage_tank + B_storage_tank_compressor_house = 0
_C537: Wx_compressor_house_storage_tank + Wx_storage_tank_compressor_house = 1
_C538: Wy_compressor_house_storage_tank + Wy_storage_tank_compressor_house = 1
_C539: - L_reclaimer_compressor_house + R_reclaimer_compressor_house
 + X_compressor_house - X_reclaimer = 0
_C54: A_direct_contact_cooler_absorber - B_direct_contact_cooler_absorber
 + Y_absorber - Y_direct_contact_cooler = 0
_C540: A_reclaimer_compressor_house - B_reclaimer_compressor_house
 + Y_compressor_house - Y_reclaimer = 0
_C541: R_reclaimer_compressor_house - 1000 Wx_reclaimer_compressor_house <= 0
_C542: L_reclaimer_compressor_house + 1000 Wx_reclaimer_compressor_house
 <= 1000
_C543: A_reclaimer_compressor_house - 1000 Wy_reclaimer_compressor_house <= 0
_C544: B_reclaimer_compressor_house + 1000 Wy_reclaimer_compressor_house
 <= 1000
_C545: - A_reclaimer_compressor_house - B_reclaimer_compressor_house
 + D_reclaimer_compressor_house - L_reclaimer_compressor_house
 - R_reclaimer_compressor_house = 0
_C546: - D_compressor_house_reclaimer + D_reclaimer_compressor_house = 0
_C547: 1000 E1_reclaimer_compressor_house + 1000 E2_reclaimer_compressor_house
 - X_compressor_house + X_reclaimer - 0.5 l_compressor_house - 0.5 l_reclaimer
 >= 9
_C548: - 1000 E1_reclaimer_compressor_house
 + 1000 E2_reclaimer_compressor_house + X_compressor_house - X_reclaimer
 - 0.5 l_compressor_house - 0.5 l_reclaimer >= -991
_C549: 1000 E1_reclaimer_compressor_house - 1000 E2_reclaimer_compressor_house
 - Y_compressor_house + Y_reclaimer - 0.5 d_compressor_house - 0.5 d_reclaimer
 >= -991
_C55: R_direct_contact_cooler_absorber
 - 1000 Wx_direct_contact_cooler_absorber <= 0
_C550: - 1000 E1_reclaimer_compressor_house
 - 1000 E2_reclaimer_compressor_house + Y_compressor_house - Y_reclaimer
 - 0.5 d_compressor_house - 0.5 d_reclaimer >= -1991
_C551: L_reclaimer_compressor_house - R_compressor_house_reclaimer = 0
_C552: - L_compressor_house_reclaimer + R_reclaimer_compressor_house = 0
_C553: A_reclaimer_compressor_house - B_compressor_house_reclaimer = 0
_C554: - A_compressor_house_reclaimer + B_reclaimer_compressor_house = 0
_C555: Wx_compressor_house_reclaimer + Wx_reclaimer_compressor_house = 1
_C556: Wy_compressor_house_reclaimer + Wy_reclaimer_compressor_house = 1
_C557: - L_direct_contact_cooler_compressor_KOdrum
 + R_direct_contact_cooler_compressor_KOdrum + X_compressor_KOdrum
 - X_direct_contact_cooler = 0
_C558: A_direct_contact_cooler_compressor_KOdrum
 - B_direct_contact_cooler_compressor_KOdrum + Y_compressor_KOdrum
 - Y_direct_contact_cooler = 0
_C559: R_direct_contact_cooler_compressor_KOdrum
 - 1000 Wx_direct_contact_cooler_compressor_KOdrum <= 0
_C56: L_direct_contact_cooler_absorber
 + 1000 Wx_direct_contact_cooler_absorber <= 1000
_C560: L_direct_contact_cooler_compressor_KOdrum
 + 1000 Wx_direct_contact_cooler_compressor_KOdrum <= 1000
_C561: A_direct_contact_cooler_compressor_KOdrum
 - 1000 Wy_direct_contact_cooler_compressor_KOdrum <= 0
_C562: B_direct_contact_cooler_compressor_KOdrum
 + 1000 Wy_direct_contact_cooler_compressor_KOdrum <= 1000
_C563: - A_direct_contact_cooler_compressor_KOdrum
 - B_direct_contact_cooler_compressor_KOdrum
 + D_direct_contact_cooler_compressor_KOdrum
 - L_direct_contact_cooler_compressor_KOdrum
 - R_direct_contact_cooler_compressor_KOdrum = 0
_C564: - D_compressor_KOdrum_direct_contact_cooler
 + D_direct_contact_cooler_compressor_KOdrum = 0
_C565: 1000 E1_direct_contact_cooler_compressor_KOdrum
 + 1000 E2_direct_contact_cooler_compressor_KOdrum - X_compressor_KOdrum
 + X_direct_contact_cooler - 0.5 l_compressor_KOdrum
 - 0.5 l_direct_contact_cooler >= 15
_C566: - 1000 E1_direct_contact_cooler_compressor_KOdrum
 + 1000 E2_direct_contact_cooler_compressor_KOdrum + X_compressor_KOdrum
 - X_direct_contact_cooler - 0.5 l_compressor_KOdrum
 - 0.5 l_direct_contact_cooler >= -985
_C567: 1000 E1_direct_contact_cooler_compressor_KOdrum
 - 1000 E2_direct_contact_cooler_compressor_KOdrum - Y_compressor_KOdrum
 + Y_direct_contact_cooler - 0.5 d_compressor_KOdrum
 - 0.5 d_direct_contact_cooler >= -985
_C568: - 1000 E1_direct_contact_cooler_compressor_KOdrum
 - 1000 E2_direct_contact_cooler_compressor_KOdrum + Y_compressor_KOdrum
 - Y_direct_contact_cooler - 0.5 d_compressor_KOdrum
 - 0.5 d_direct_contact_cooler >= -1985
_C569: L_direct_contact_cooler_compressor_KOdrum
 - R_compressor_KOdrum_direct_contact_cooler = 0
_C57: A_direct_contact_cooler_absorber
 - 1000 Wy_direct_contact_cooler_absorber <= 0
_C570: - L_compressor_KOdrum_direct_contact_cooler
 + R_direct_contact_cooler_compressor_KOdrum = 0
_C571: A_direct_contact_cooler_compressor_KOdrum
 - B_compressor_KOdrum_direct_contact_cooler = 0
_C572: - A_compressor_KOdrum_direct_contact_cooler
 + B_direct_contact_cooler_compressor_KOdrum = 0
_C573: Wx_compressor_KOdrum_direct_contact_cooler
 + Wx_direct_contact_cooler_compressor_KOdrum = 1
_C574: Wy_compressor_KOdrum_direct_contact_cooler
 + Wy_direct_contact_cooler_compressor_KOdrum = 1
_C575: - L_absorber_compressor_KOdrum + R_absorber_compressor_KOdrum
 - X_absorber + X_compressor_KOdrum = 0
_C576: A_absorber_compressor_KOdrum - B_absorber_compressor_KOdrum
 - Y_absorber + Y_compressor_KOdrum = 0
_C577: R_absorber_compressor_KOdrum - 1000 Wx_absorber_compressor_KOdrum <= 0
_C578: L_absorber_compressor_KOdrum + 1000 Wx_absorber_compressor_KOdrum
 <= 1000
_C579: A_absorber_compressor_KOdrum - 1000 Wy_absorber_compressor_KOdrum <= 0
_C58: B_direct_contact_cooler_absorber
 + 1000 Wy_direct_contact_cooler_absorber <= 1000
_C580: B_absorber_compressor_KOdrum + 1000 Wy_absorber_compressor_KOdrum
 <= 1000
_C581: - A_absorber_compressor_KOdrum - B_absorber_compressor_KOdrum
 + D_absorber_compressor_KOdrum - L_absorber_compressor_KOdrum
 - R_absorber_compressor_KOdrum = 0
_C582: D_absorber_compressor_KOdrum - D_compressor_KOdrum_absorber = 0
_C583: 1000 E1_absorber_compressor_KOdrum + 1000 E2_absorber_compressor_KOdrum
 + X_absorber - X_compressor_KOdrum - 0.5 l_absorber - 0.5 l_compressor_KOdrum
 >= 15
_C584: - 1000 E1_absorber_compressor_KOdrum
 + 1000 E2_absorber_compressor_KOdrum - X_absorber + X_compressor_KOdrum
 - 0.5 l_absorber - 0.5 l_compressor_KOdrum >= -985
_C585: 1000 E1_absorber_compressor_KOdrum - 1000 E2_absorber_compressor_KOdrum
 + Y_absorber - Y_compressor_KOdrum - 0.5 d_absorber - 0.5 d_compressor_KOdrum
 >= -985
_C586: - 1000 E1_absorber_compressor_KOdrum
 - 1000 E2_absorber_compressor_KOdrum - Y_absorber + Y_compressor_KOdrum
 - 0.5 d_absorber - 0.5 d_compressor_KOdrum >= -1985
_C587: L_absorber_compressor_KOdrum - R_compressor_KOdrum_absorber = 0
_C588: - L_compressor_KOdrum_absorber + R_absorber_compressor_KOdrum = 0
_C589: A_absorber_compressor_KOdrum - B_compressor_KOdrum_absorber = 0
_C59: - A_direct_contact_cooler_absorber - B_direct_contact_cooler_absorber
 + D_direct_contact_cooler_absorber - L_direct_contact_cooler_absorber
 - R_direct_contact_cooler_absorber = 0
_C590: - A_compressor_KOdrum_absorber + B_absorber_compressor_KOdrum = 0
_C591: Wx_absorber_compressor_KOdrum + Wx_compressor_KOdrum_absorber = 1
_C592: Wy_absorber_compressor_KOdrum + Wy_compressor_KOdrum_absorber = 1
_C593: - L_lean_amine_cooler_compressor_KOdrum
 + R_lean_amine_cooler_compressor_KOdrum + X_compressor_KOdrum
 - X_lean_amine_cooler = 0
_C594: A_lean_amine_cooler_compressor_KOdrum
 - B_lean_amine_cooler_compressor_KOdrum + Y_compressor_KOdrum
 - Y_lean_amine_cooler = 0
_C595: R_lean_amine_cooler_compressor_KOdrum
 - 1000 Wx_lean_amine_cooler_compressor_KOdrum <= 0
_C596: L_lean_amine_cooler_compressor_KOdrum
 + 1000 Wx_lean_amine_cooler_compressor_KOdrum <= 1000
_C597: A_lean_amine_cooler_compressor_KOdrum
 - 1000 Wy_lean_amine_cooler_compressor_KOdrum <= 0
_C598: B_lean_amine_cooler_compressor_KOdrum
 + 1000 Wy_lean_amine_cooler_compressor_KOdrum <= 1000
_C599: - A_lean_amine_cooler_compressor_KOdrum
 - B_lean_amine_cooler_compressor_KOdrum
 + D_lean_amine_cooler_compressor_KOdrum
 - L_lean_amine_cooler_compressor_KOdrum
 - R_lean_amine_cooler_compressor_KOdrum = 0
_C6: d_absorber + l_absorber = 92
_C60: - D_absorber_direct_contact_cooler + D_direct_contact_cooler_absorber
 = 0
_C600: - D_compressor_KOdrum_lean_amine_cooler
 + D_lean_amine_cooler_compressor_KOdrum = 0
_C601: 1000 E1_lean_amine_cooler_compressor_KOdrum
 + 1000 E2_lean_amine_cooler_compressor_KOdrum - X_compressor_KOdrum
 + X_lean_amine_cooler - 0.5 l_compressor_KOdrum - 0.5 l_lean_amine_cooler
 >= 15
_C602: - 1000 E1_lean_amine_cooler_compressor_KOdrum
 + 1000 E2_lean_amine_cooler_compressor_KOdrum + X_compressor_KOdrum
 - X_lean_amine_cooler - 0.5 l_compressor_KOdrum - 0.5 l_lean_amine_cooler
 >= -985
_C603: 1000 E1_lean_amine_cooler_compressor_KOdrum
 - 1000 E2_lean_amine_cooler_compressor_KOdrum - Y_compressor_KOdrum
 + Y_lean_amine_cooler - 0.5 d_compressor_KOdrum - 0.5 d_lean_amine_cooler
 >= -985
_C604: - 1000 E1_lean_amine_cooler_compressor_KOdrum
 - 1000 E2_lean_amine_cooler_compressor_KOdrum + Y_compressor_KOdrum
 - Y_lean_amine_cooler - 0.5 d_compressor_KOdrum - 0.5 d_lean_amine_cooler
 >= -1985
_C605: L_lean_amine_cooler_compressor_KOdrum
 - R_compressor_KOdrum_lean_amine_cooler = 0
_C606: - L_compressor_KOdrum_lean_amine_cooler
 + R_lean_amine_cooler_compressor_KOdrum = 0
_C607: A_lean_amine_cooler_compressor_KOdrum
 - B_compressor_KOdrum_lean_amine_cooler = 0
_C608: - A_compressor_KOdrum_lean_amine_cooler
 + B_lean_amine_cooler_compressor_KOdrum = 0
_C609: Wx_compressor_KOdrum_lean_amine_cooler
 + Wx_lean_amine_cooler_compressor_KOdrum = 1
_C61: 1000 E1_direct_contact_cooler_absorber
 + 1000 E2_direct_contact_cooler_absorber - X_absorber
 + X_direct_contact_cooler - 0.5 l_absorber - 0.5 l_direct_contact_cooler
 >= 15
_C610: Wy_compressor_KOdrum_lean_amine_cooler
 + Wy_lean_amine_cooler_compressor_KOdrum = 1
_C611: - L_cross_heat_exchanger_compressor_KOdrum
 + R_cross_heat_exchanger_compressor_KOdrum + X_compressor_KOdrum
 - X_cross_heat_exchanger = 0
_C612: A_cross_heat_exchanger_compressor_KOdrum
 - B_cross_heat_exchanger_compressor_KOdrum + Y_compressor_KOdrum
 - Y_cross_heat_exchanger = 0
_C613: R_cross_heat_exchanger_compressor_KOdrum
 - 1000 Wx_cross_heat_exchanger_compressor_KOdrum <= 0
_C614: L_cross_heat_exchanger_compressor_KOdrum
 + 1000 Wx_cross_heat_exchanger_compressor_KOdrum <= 1000
_C615: A_cross_heat_exchanger_compressor_KOdrum
 - 1000 Wy_cross_heat_exchanger_compressor_KOdrum <= 0
_C616: B_cross_heat_exchanger_compressor_KOdrum
 + 1000 Wy_cross_heat_exchanger_compressor_KOdrum <= 1000
_C617: - A_cross_heat_exchanger_compressor_KOdrum
 - B_cross_heat_exchanger_compressor_KOdrum
 + D_cross_heat_exchanger_compressor_KOdrum
 - L_cross_heat_exchanger_compressor_KOdrum
 - R_cross_heat_exchanger_compressor_KOdrum = 0
_C618: - D_compressor_KOdrum_cross_heat_exchanger
 + D_cross_heat_exchanger_compressor_KOdrum = 0
_C619: 1000 E1_cross_heat_exchanger_compressor_KOdrum
 + 1000 E2_cross_heat_exchanger_compressor_KOdrum - X_compressor_KOdrum
 + X_cross_heat_exchanger - 0.5 l_compressor_KOdrum
 - 0.5 l_cross_heat_exchanger >= 15
_C62: - 1000 E1_direct_contact_cooler_absorber
 + 1000 E2_direct_contact_cooler_absorber + X_absorber
 - X_direct_contact_cooler - 0.5 l_absorber - 0.5 l_direct_contact_cooler
 >= -985
_C620: - 1000 E1_cross_heat_exchanger_compressor_KOdrum
 + 1000 E2_cross_heat_exchanger_compressor_KOdrum + X_compressor_KOdrum
 - X_cross_heat_exchanger - 0.5 l_compressor_KOdrum
 - 0.5 l_cross_heat_exchanger >= -985
_C621: 1000 E1_cross_heat_exchanger_compressor_KOdrum
 - 1000 E2_cross_heat_exchanger_compressor_KOdrum - Y_compressor_KOdrum
 + Y_cross_heat_exchanger - 0.5 d_compressor_KOdrum
 - 0.5 d_cross_heat_exchanger >= -985
_C622: - 1000 E1_cross_heat_exchanger_compressor_KOdrum
 - 1000 E2_cross_heat_exchanger_compressor_KOdrum + Y_compressor_KOdrum
 - Y_cross_heat_exchanger - 0.5 d_compressor_KOdrum
 - 0.5 d_cross_heat_exchanger >= -1985
_C623: L_cross_heat_exchanger_compressor_KOdrum
 - R_compressor_KOdrum_cross_heat_exchanger = 0
_C624: - L_compressor_KOdrum_cross_heat_exchanger
 + R_cross_heat_exchanger_compressor_KOdrum = 0
_C625: A_cross_heat_exchanger_compressor_KOdrum
 - B_compressor_KOdrum_cross_heat_exchanger = 0
_C626: - A_compressor_KOdrum_cross_heat_exchanger
 + B_cross_heat_exchanger_compressor_KOdrum = 0
_C627: Wx_compressor_KOdrum_cross_heat_exchanger
 + Wx_cross_heat_exchanger_compressor_KOdrum = 1
_C628: Wy_compressor_KOdrum_cross_heat_exchanger
 + Wy_cross_heat_exchanger_compressor_KOdrum = 1
_C629: - L_stripper_compressor_KOdrum + R_stripper_compressor_KOdrum
 + X_compressor_KOdrum - X_stripper = 0
_C63: 1000 E1_direct_contact_cooler_absorber
 - 1000 E2_direct_contact_cooler_absorber - Y_absorber
 + Y_direct_contact_cooler - 0.5 d_absorber - 0.5 d_direct_contact_cooler
 >= -985
_C630: A_stripper_compressor_KOdrum - B_stripper_compressor_KOdrum
 + Y_compressor_KOdrum - Y_stripper = 0
_C631: R_stripper_compressor_KOdrum - 1000 Wx_stripper_compressor_KOdrum <= 0
_C632: L_stripper_compressor_KOdrum + 1000 Wx_stripper_compressor_KOdrum
 <= 1000
_C633: A_stripper_compressor_KOdrum - 1000 Wy_stripper_compressor_KOdrum <= 0
_C634: B_stripper_compressor_KOdrum + 1000 Wy_stripper_compressor_KOdrum
 <= 1000
_C635: - A_stripper_compressor_KOdrum - B_stripper_compressor_KOdrum
 + D_stripper_compressor_KOdrum - L_stripper_compressor_KOdrum
 - R_stripper_compressor_KOdrum = 0
_C636: - D_compressor_KOdrum_stripper + D_stripper_compressor_KOdrum = 0
_C637: 1000 E1_stripper_compressor_KOdrum + 1000 E2_stripper_compressor_KOdrum
 - X_compressor_KOdrum + X_stripper - 0.5 l_compressor_KOdrum - 0.5 l_stripper
 >= 15
_C638: - 1000 E1_stripper_compressor_KOdrum
 + 1000 E2_stripper_compressor_KOdrum + X_compressor_KOdrum - X_stripper
 - 0.5 l_compressor_KOdrum - 0.5 l_stripper >= -985
_C639: 1000 E1_stripper_compressor_KOdrum - 1000 E2_stripper_compressor_KOdrum
 - Y_compressor_KOdrum + Y_stripper - 0.5 d_compressor_KOdrum - 0.5 d_stripper
 >= -985
_C64: - 1000 E1_direct_contact_cooler_absorber
 - 1000 E2_direct_contact_cooler_absorber + Y_absorber
 - Y_direct_contact_cooler - 0.5 d_absorber - 0.5 d_direct_contact_cooler
 >= -1985
_C640: - 1000 E1_stripper_compressor_KOdrum
 - 1000 E2_stripper_compressor_KOdrum + Y_compressor_KOdrum - Y_stripper
 - 0.5 d_compressor_KOdrum - 0.5 d_stripper >= -1985
_C641: L_stripper_compressor_KOdrum - R_compressor_KOdrum_stripper = 0
_C642: - L_compressor_KOdrum_stripper + R_stripper_compressor_KOdrum = 0
_C643: A_stripper_compressor_KOdrum - B_compressor_KOdrum_stripper = 0
_C644: - A_compressor_KOdrum_stripper + B_stripper_compressor_KOdrum = 0
_C645: Wx_compressor_KOdrum_stripper + Wx_stripper_compressor_KOdrum = 1
_C646: Wy_compressor_KOdrum_stripper + Wy_stripper_compressor_KOdrum = 1
_C647: - L_storage_tank_compressor_KOdrum + R_storage_tank_compressor_KOdrum
 + X_compressor_KOdrum - X_storage_tank = 0
_C648: A_storage_tank_compressor_KOdrum - B_storage_tank_compressor_KOdrum
 + Y_compressor_KOdrum - Y_storage_tank = 0
_C649: R_storage_tank_compressor_KOdrum
 - 1000 Wx_storage_tank_compressor_KOdrum <= 0
_C65: L_direct_contact_cooler_absorber - R_absorber_direct_contact_cooler = 0
_C650: L_storage_tank_compressor_KOdrum
 + 1000 Wx_storage_tank_compressor_KOdrum <= 1000
_C651: A_storage_tank_compressor_KOdrum
 - 1000 Wy_storage_tank_compressor_KOdrum <= 0
_C652: B_storage_tank_compressor_KOdrum
 + 1000 Wy_storage_tank_compressor_KOdrum <= 1000
_C653: - A_storage_tank_compressor_KOdrum - B_storage_tank_compressor_KOdrum
 + D_storage_tank_compressor_KOdrum - L_storage_tank_compressor_KOdrum
 - R_storage_tank_compressor_KOdrum = 0
_C654: - D_compressor_KOdrum_storage_tank + D_storage_tank_compressor_KOdrum
 = 0
_C655: 1000 E1_storage_tank_compressor_KOdrum
 + 1000 E2_storage_tank_compressor_KOdrum - X_compressor_KOdrum
 + X_storage_tank - 0.5 l_compressor_KOdrum - 0.5 l_storage_tank >= 76
_C656: - 1000 E1_storage_tank_compressor_KOdrum
 + 1000 E2_storage_tank_compressor_KOdrum + X_compressor_KOdrum
 - X_storage_tank - 0.5 l_compressor_KOdrum - 0.5 l_storage_tank >= -924
_C657: 1000 E1_storage_tank_compressor_KOdrum
 - 1000 E2_storage_tank_compressor_KOdrum - Y_compressor_KOdrum
 + Y_storage_tank - 0.5 d_compressor_KOdrum - 0.5 d_storage_tank >= -924
_C658: - 1000 E1_storage_tank_compressor_KOdrum
 - 1000 E2_storage_tank_compressor_KOdrum + Y_compressor_KOdrum
 - Y_storage_tank - 0.5 d_compressor_KOdrum - 0.5 d_storage_tank >= -1924
_C659: L_storage_tank_compressor_KOdrum - R_compressor_KOdrum_storage_tank = 0
_C66: - L_absorber_direct_contact_cooler + R_direct_contact_cooler_absorber
 = 0
_C660: - L_compressor_KOdrum_storage_tank + R_storage_tank_compressor_KOdrum
 = 0
_C661: A_storage_tank_compressor_KOdrum - B_compressor_KOdrum_storage_tank = 0
_C662: - A_compressor_KOdrum_storage_tank + B_storage_tank_compressor_KOdrum
 = 0
_C663: Wx_compressor_KOdrum_storage_tank + Wx_storage_tank_compressor_KOdrum
 = 1
_C664: Wy_compressor_KOdrum_storage_tank + Wy_storage_tank_compressor_KOdrum
 = 1
_C665: - L_reclaimer_compressor_KOdrum + R_reclaimer_compressor_KOdrum
 + X_compressor_KOdrum - X_reclaimer = 0
_C666: A_reclaimer_compressor_KOdrum - B_reclaimer_compressor_KOdrum
 + Y_compressor_KOdrum - Y_reclaimer = 0
_C667: R_reclaimer_compressor_KOdrum - 1000 Wx_reclaimer_compressor_KOdrum
 <= 0
_C668: L_reclaimer_compressor_KOdrum + 1000 Wx_reclaimer_compressor_KOdrum
 <= 1000
_C669: A_reclaimer_compressor_KOdrum - 1000 Wy_reclaimer_compressor_KOdrum
 <= 0
_C67: A_direct_contact_cooler_absorber - B_absorber_direct_contact_cooler = 0
_C670: B_reclaimer_compressor_KOdrum + 1000 Wy_reclaimer_compressor_KOdrum
 <= 1000
_C671: - A_reclaimer_compressor_KOdrum - B_reclaimer_compressor_KOdrum
 + D_reclaimer_compressor_KOdrum - L_reclaimer_compressor_KOdrum
 - R_reclaimer_compressor_KOdrum = 0
_C672: - D_compressor_KOdrum_reclaimer + D_reclaimer_compressor_KOdrum = 0
_C673: 1000 E1_reclaimer_compressor_KOdrum
 + 1000 E2_reclaimer_compressor_KOdrum - X_compressor_KOdrum + X_reclaimer
 - 0.5 l_compressor_KOdrum - 0.5 l_reclaimer >= 15
_C674: - 1000 E1_reclaimer_compressor_KOdrum
 + 1000 E2_reclaimer_compressor_KOdrum + X_compressor_KOdrum - X_reclaimer
 - 0.5 l_compressor_KOdrum - 0.5 l_reclaimer >= -985
_C675: 1000 E1_reclaimer_compressor_KOdrum
 - 1000 E2_reclaimer_compressor_KOdrum - Y_compressor_KOdrum + Y_reclaimer
 - 0.5 d_compressor_KOdrum - 0.5 d_reclaimer >= -985
_C676: - 1000 E1_reclaimer_compressor_KOdrum
 - 1000 E2_reclaimer_compressor_KOdrum + Y_compressor_KOdrum - Y_reclaimer
 - 0.5 d_compressor_KOdrum - 0.5 d_reclaimer >= -1985
_C677: L_reclaimer_compressor_KOdrum - R_compressor_KOdrum_reclaimer = 0
_C678: - L_compressor_KOdrum_reclaimer + R_reclaimer_compressor_KOdrum = 0
_C679: A_reclaimer_compressor_KOdrum - B_compressor_KOdrum_reclaimer = 0
_C68: - A_absorber_direct_contact_cooler + B_direct_contact_cooler_absorber
 = 0
_C680: - A_compressor_KOdrum_reclaimer + B_reclaimer_compressor_KOdrum = 0
_C681: Wx_compressor_KOdrum_reclaimer + Wx_reclaimer_compressor_KOdrum = 1
_C682: Wy_compressor_KOdrum_reclaimer + Wy_reclaimer_compressor_KOdrum = 1
_C683: - L_compressor_house_compressor_KOdrum
 + R_compressor_house_compressor_KOdrum + X_compressor_KOdrum
 - X_compressor_house = 0
_C684: A_compressor_house_compressor_KOdrum
 - B_compressor_house_compressor_KOdrum + Y_compressor_KOdrum
 - Y_compressor_house = 0
_C685: R_compressor_house_compressor_KOdrum
 - 1000 Wx_compressor_house_compressor_KOdrum <= 0
_C686: L_compressor_house_compressor_KOdrum
 + 1000 Wx_compressor_house_compressor_KOdrum <= 1000
_C687: A_compressor_house_compressor_KOdrum
 - 1000 Wy_compressor_house_compressor_KOdrum <= 0
_C688: B_compressor_house_compressor_KOdrum
 + 1000 Wy_compressor_house_compressor_KOdrum <= 1000
_C689: - A_compressor_house_compressor_KOdrum
 - B_compressor_house_compressor_KOdrum + D_compressor_house_compressor_KOdrum
 - L_compressor_house_compressor_KOdrum - R_compressor_house_compressor_KOdrum
 = 0
_C69: Wx_absorber_direct_contact_cooler + Wx_direct_contact_cooler_absorber
 = 1
_C690: - D_compressor_KOdrum_compressor_house
 + D_compressor_house_compressor_KOdrum = 0
_C691: 1000 E1_compressor_house_compressor_KOdrum
 + 1000 E2_compressor_house_compressor_KOdrum - X_compressor_KOdrum
 + X_compressor_house - 0.5 l_compressor_KOdrum - 0.5 l_compressor_house >= 9
_C692: - 1000 E1_compressor_house_compressor_KOdrum
 + 1000 E2_compressor_house_compressor_KOdrum + X_compressor_KOdrum
 - X_compressor_house - 0.5 l_compressor_KOdrum - 0.5 l_compressor_house
 >= -991
_C693: 1000 E1_compressor_house_compressor_KOdrum
 - 1000 E2_compressor_house_compressor_KOdrum - Y_compressor_KOdrum
 + Y_compressor_house - 0.5 d_compressor_KOdrum - 0.5 d_compressor_house
 >= -991
_C694: - 1000 E1_compressor_house_compressor_KOdrum
 - 1000 E2_compressor_house_compressor_KOdrum + Y_compressor_KOdrum
 - Y_compressor_house - 0.5 d_compressor_KOdrum - 0.5 d_compressor_house
 >= -1991
_C695: L_compressor_house_compressor_KOdrum
 - R_compressor_KOdrum_compressor_house = 0
_C696: - L_compressor_KOdrum_compressor_house
 + R_compressor_house_compressor_KOdrum = 0
_C697: A_compressor_house_compressor_KOdrum
 - B_compressor_KOdrum_compressor_house = 0
_C698: - A_compressor_KOdrum_compressor_house
 + B_compressor_house_compressor_KOdrum = 0
_C699: Wx_compressor_KOdrum_compressor_house
 + Wx_compressor_house_compressor_KOdrum = 1
_C7: X_absorber - 0.5 l_absorber >= 0
_C70: Wy_absorber_direct_contact_cooler + Wy_direct_contact_cooler_absorber
 = 1
_C700: Wy_compressor_KOdrum_compressor_house
 + Wy_compressor_house_compressor_KOdrum = 1
_C701: - L_direct_contact_cooler_compressor_interstage_cooler
 + R_direct_contact_cooler_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_direct_contact_cooler = 0
_C702: A_direct_contact_cooler_compressor_interstage_cooler
 - B_direct_contact_cooler_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_direct_contact_cooler = 0
_C703: R_direct_contact_cooler_compressor_interstage_cooler
 - 1000 Wx_direct_contact_cooler_compressor_interstage_cooler <= 0
_C704: L_direct_contact_cooler_compressor_interstage_cooler
 + 1000 Wx_direct_contact_cooler_compressor_interstage_cooler <= 1000
_C705: A_direct_contact_cooler_compressor_interstage_cooler
 - 1000 Wy_direct_contact_cooler_compressor_interstage_cooler <= 0
_C706: B_direct_contact_cooler_compressor_interstage_cooler
 + 1000 Wy_direct_contact_cooler_compressor_interstage_cooler <= 1000
_C707: - A_direct_contact_cooler_compressor_interstage_cooler
 - B_direct_contact_cooler_compressor_interstage_cooler
 + D_direct_contact_cooler_compressor_interstage_cooler
 - L_direct_contact_cooler_compressor_interstage_cooler
 - R_direct_contact_cooler_compressor_interstage_cooler = 0
_C708: - D_compressor_interstage_cooler_direct_contact_cooler
 + D_direct_contact_cooler_compressor_interstage_cooler = 0
_C709: 1000 E1_direct_contact_cooler_compressor_interstage_cooler
 + 1000 E2_direct_contact_cooler_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_direct_contact_cooler
 - 0.5 l_compressor_interstage_cooler - 0.5 l_direct_contact_cooler >= 15
_C71: - L_direct_contact_cooler_lean_amine_cooler
 + R_direct_contact_cooler_lean_amine_cooler - X_direct_contact_cooler
 + X_lean_amine_cooler = 0
_C710: - 1000 E1_direct_contact_cooler_compressor_interstage_cooler
 + 1000 E2_direct_contact_cooler_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_direct_contact_cooler
 - 0.5 l_compressor_interstage_cooler - 0.5 l_direct_contact_cooler >= -985
_C711: 1000 E1_direct_contact_cooler_compressor_interstage_cooler
 - 1000 E2_direct_contact_cooler_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_direct_contact_cooler
 - 0.5 d_compressor_interstage_cooler - 0.5 d_direct_contact_cooler >= -985
_C712: - 1000 E1_direct_contact_cooler_compressor_interstage_cooler
 - 1000 E2_direct_contact_cooler_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_direct_contact_cooler
 - 0.5 d_compressor_interstage_cooler - 0.5 d_direct_contact_cooler >= -1985
_C713: L_direct_contact_cooler_compressor_interstage_cooler
 - R_compressor_interstage_cooler_direct_contact_cooler = 0
_C714: - L_compressor_interstage_cooler_direct_contact_cooler
 + R_direct_contact_cooler_compressor_interstage_cooler = 0
_C715: A_direct_contact_cooler_compressor_interstage_cooler
 - B_compressor_interstage_cooler_direct_contact_cooler = 0
_C716: - A_compressor_interstage_cooler_direct_contact_cooler
 + B_direct_contact_cooler_compressor_interstage_cooler = 0
_C717: Wx_compressor_interstage_cooler_direct_contact_cooler
 + Wx_direct_contact_cooler_compressor_interstage_cooler = 1
_C718: Wy_compressor_interstage_cooler_direct_contact_cooler
 + Wy_direct_contact_cooler_compressor_interstage_cooler = 1
_C719: - L_absorber_compressor_interstage_cooler
 + R_absorber_compressor_interstage_cooler - X_absorber
 + X_compressor_interstage_cooler = 0
_C72: A_direct_contact_cooler_lean_amine_cooler
 - B_direct_contact_cooler_lean_amine_cooler - Y_direct_contact_cooler
 + Y_lean_amine_cooler = 0
_C720: A_absorber_compressor_interstage_cooler
 - B_absorber_compressor_interstage_cooler - Y_absorber
 + Y_compressor_interstage_cooler = 0
_C721: R_absorber_compressor_interstage_cooler
 - 1000 Wx_absorber_compressor_interstage_cooler <= 0
_C722: L_absorber_compressor_interstage_cooler
 + 1000 Wx_absorber_compressor_interstage_cooler <= 1000
_C723: A_absorber_compressor_interstage_cooler
 - 1000 Wy_absorber_compressor_interstage_cooler <= 0
_C724: B_absorber_compressor_interstage_cooler
 + 1000 Wy_absorber_compressor_interstage_cooler <= 1000
_C725: - A_absorber_compressor_interstage_cooler
 - B_absorber_compressor_interstage_cooler
 + D_absorber_compressor_interstage_cooler
 - L_absorber_compressor_interstage_cooler
 - R_absorber_compressor_interstage_cooler = 0
_C726: D_absorber_compressor_interstage_cooler
 - D_compressor_interstage_cooler_absorber = 0
_C727: 1000 E1_absorber_compressor_interstage_cooler
 + 1000 E2_absorber_compressor_interstage_cooler + X_absorber
 - X_compressor_interstage_cooler - 0.5 l_absorber
 - 0.5 l_compressor_interstage_cooler >= 15
_C728: - 1000 E1_absorber_compressor_interstage_cooler
 + 1000 E2_absorber_compressor_interstage_cooler - X_absorber
 + X_compressor_interstage_cooler - 0.5 l_absorber
 - 0.5 l_compressor_interstage_cooler >= -985
_C729: 1000 E1_absorber_compressor_interstage_cooler
 - 1000 E2_absorber_compressor_interstage_cooler + Y_absorber
 - Y_compressor_interstage_cooler - 0.5 d_absorber
 - 0.5 d_compressor_interstage_cooler >= -985
_C73: R_direct_contact_cooler_lean_amine_cooler
 - 1000 Wx_direct_contact_cooler_lean_amine_cooler <= 0
_C730: - 1000 E1_absorber_compressor_interstage_cooler
 - 1000 E2_absorber_compressor_interstage_cooler - Y_absorber
 + Y_compressor_interstage_cooler - 0.5 d_absorber
 - 0.5 d_compressor_interstage_cooler >= -1985
_C731: L_absorber_compressor_interstage_cooler
 - R_compressor_interstage_cooler_absorber = 0
_C732: - L_compressor_interstage_cooler_absorber
 + R_absorber_compressor_interstage_cooler = 0
_C733: A_absorber_compressor_interstage_cooler
 - B_compressor_interstage_cooler_absorber = 0
_C734: - A_compressor_interstage_cooler_absorber
 + B_absorber_compressor_interstage_cooler = 0
_C735: Wx_absorber_compressor_interstage_cooler
 + Wx_compressor_interstage_cooler_absorber = 1
_C736: Wy_absorber_compressor_interstage_cooler
 + Wy_compressor_interstage_cooler_absorber = 1
_C737: - L_lean_amine_cooler_compressor_interstage_cooler
 + R_lean_amine_cooler_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_lean_amine_cooler = 0
_C738: A_lean_amine_cooler_compressor_interstage_cooler
 - B_lean_amine_cooler_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_lean_amine_cooler = 0
_C739: R_lean_amine_cooler_compressor_interstage_cooler
 - 1000 Wx_lean_amine_cooler_compressor_interstage_cooler <= 0
_C74: L_direct_contact_cooler_lean_amine_cooler
 + 1000 Wx_direct_contact_cooler_lean_amine_cooler <= 1000
_C740: L_lean_amine_cooler_compressor_interstage_cooler
 + 1000 Wx_lean_amine_cooler_compressor_interstage_cooler <= 1000
_C741: A_lean_amine_cooler_compressor_interstage_cooler
 - 1000 Wy_lean_amine_cooler_compressor_interstage_cooler <= 0
_C742: B_lean_amine_cooler_compressor_interstage_cooler
 + 1000 Wy_lean_amine_cooler_compressor_interstage_cooler <= 1000
_C743: - A_lean_amine_cooler_compressor_interstage_cooler
 - B_lean_amine_cooler_compressor_interstage_cooler
 + D_lean_amine_cooler_compressor_interstage_cooler
 - L_lean_amine_cooler_compressor_interstage_cooler
 - R_lean_amine_cooler_compressor_interstage_cooler = 0
_C744: - D_compressor_interstage_cooler_lean_amine_cooler
 + D_lean_amine_cooler_compressor_interstage_cooler = 0
_C745: 1000 E1_lean_amine_cooler_compressor_interstage_cooler
 + 1000 E2_lean_amine_cooler_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_lean_amine_cooler
 - 0.5 l_compressor_interstage_cooler - 0.5 l_lean_amine_cooler >= 15
_C746: - 1000 E1_lean_amine_cooler_compressor_interstage_cooler
 + 1000 E2_lean_amine_cooler_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_lean_amine_cooler
 - 0.5 l_compressor_interstage_cooler - 0.5 l_lean_amine_cooler >= -985
_C747: 1000 E1_lean_amine_cooler_compressor_interstage_cooler
 - 1000 E2_lean_amine_cooler_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_lean_amine_cooler
 - 0.5 d_compressor_interstage_cooler - 0.5 d_lean_amine_cooler >= -985
_C748: - 1000 E1_lean_amine_cooler_compressor_interstage_cooler
 - 1000 E2_lean_amine_cooler_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_lean_amine_cooler
 - 0.5 d_compressor_interstage_cooler - 0.5 d_lean_amine_cooler >= -1985
_C749: L_lean_amine_cooler_compressor_interstage_cooler
 - R_compressor_interstage_cooler_lean_amine_cooler = 0
_C75: A_direct_contact_cooler_lean_amine_cooler
 - 1000 Wy_direct_contact_cooler_lean_amine_cooler <= 0
_C750: - L_compressor_interstage_cooler_lean_amine_cooler
 + R_lean_amine_cooler_compressor_interstage_cooler = 0
_C751: A_lean_amine_cooler_compressor_interstage_cooler
 - B_compressor_interstage_cooler_lean_amine_cooler = 0
_C752: - A_compressor_interstage_cooler_lean_amine_cooler
 + B_lean_amine_cooler_compressor_interstage_cooler = 0
_C753: Wx_compressor_interstage_cooler_lean_amine_cooler
 + Wx_lean_amine_cooler_compressor_interstage_cooler = 1
_C754: Wy_compressor_interstage_cooler_lean_amine_cooler
 + Wy_lean_amine_cooler_compressor_interstage_cooler = 1
_C755: - L_cross_heat_exchanger_compressor_interstage_cooler
 + R_cross_heat_exchanger_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_cross_heat_exchanger = 0
_C756: A_cross_heat_exchanger_compressor_interstage_cooler
 - B_cross_heat_exchanger_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_cross_heat_exchanger = 0
_C757: R_cross_heat_exchanger_compressor_interstage_cooler
 - 1000 Wx_cross_heat_exchanger_compressor_interstage_cooler <= 0
_C758: L_cross_heat_exchanger_compressor_interstage_cooler
 + 1000 Wx_cross_heat_exchanger_compressor_interstage_cooler <= 1000
_C759: A_cross_heat_exchanger_compressor_interstage_cooler
 - 1000 Wy_cross_heat_exchanger_compressor_interstage_cooler <= 0
_C76: B_direct_contact_cooler_lean_amine_cooler
 + 1000 Wy_direct_contact_cooler_lean_amine_cooler <= 1000
_C760: B_cross_heat_exchanger_compressor_interstage_cooler
 + 1000 Wy_cross_heat_exchanger_compressor_interstage_cooler <= 1000
_C761: - A_cross_heat_exchanger_compressor_interstage_cooler
 - B_cross_heat_exchanger_compressor_interstage_cooler
 + D_cross_heat_exchanger_compressor_interstage_cooler
 - L_cross_heat_exchanger_compressor_interstage_cooler
 - R_cross_heat_exchanger_compressor_interstage_cooler = 0
_C762: - D_compressor_interstage_cooler_cross_heat_exchanger
 + D_cross_heat_exchanger_compressor_interstage_cooler = 0
_C763: 1000 E1_cross_heat_exchanger_compressor_interstage_cooler
 + 1000 E2_cross_heat_exchanger_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_cross_heat_exchanger
 - 0.5 l_compressor_interstage_cooler - 0.5 l_cross_heat_exchanger >= 15
_C764: - 1000 E1_cross_heat_exchanger_compressor_interstage_cooler
 + 1000 E2_cross_heat_exchanger_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_cross_heat_exchanger
 - 0.5 l_compressor_interstage_cooler - 0.5 l_cross_heat_exchanger >= -985
_C765: 1000 E1_cross_heat_exchanger_compressor_interstage_cooler
 - 1000 E2_cross_heat_exchanger_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_cross_heat_exchanger
 - 0.5 d_compressor_interstage_cooler - 0.5 d_cross_heat_exchanger >= -985
_C766: - 1000 E1_cross_heat_exchanger_compressor_interstage_cooler
 - 1000 E2_cross_heat_exchanger_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_cross_heat_exchanger
 - 0.5 d_compressor_interstage_cooler - 0.5 d_cross_heat_exchanger >= -1985
_C767: L_cross_heat_exchanger_compressor_interstage_cooler
 - R_compressor_interstage_cooler_cross_heat_exchanger = 0
_C768: - L_compressor_interstage_cooler_cross_heat_exchanger
 + R_cross_heat_exchanger_compressor_interstage_cooler = 0
_C769: A_cross_heat_exchanger_compressor_interstage_cooler
 - B_compressor_interstage_cooler_cross_heat_exchanger = 0
_C77: - A_direct_contact_cooler_lean_amine_cooler
 - B_direct_contact_cooler_lean_amine_cooler
 + D_direct_contact_cooler_lean_amine_cooler
 - L_direct_contact_cooler_lean_amine_cooler
 - R_direct_contact_cooler_lean_amine_cooler = 0
_C770: - A_compressor_interstage_cooler_cross_heat_exchanger
 + B_cross_heat_exchanger_compressor_interstage_cooler = 0
_C771: Wx_compressor_interstage_cooler_cross_heat_exchanger
 + Wx_cross_heat_exchanger_compressor_interstage_cooler = 1
_C772: Wy_compressor_interstage_cooler_cross_heat_exchanger
 + Wy_cross_heat_exchanger_compressor_interstage_cooler = 1
_C773: - L_stripper_compressor_interstage_cooler
 + R_stripper_compressor_interstage_cooler + X_compressor_interstage_cooler
 - X_stripper = 0
_C774: A_stripper_compressor_interstage_cooler
 - B_stripper_compressor_interstage_cooler + Y_compressor_interstage_cooler
 - Y_stripper = 0
_C775: R_stripper_compressor_interstage_cooler
 - 1000 Wx_stripper_compressor_interstage_cooler <= 0
_C776: L_stripper_compressor_interstage_cooler
 + 1000 Wx_stripper_compressor_interstage_cooler <= 1000
_C777: A_stripper_compressor_interstage_cooler
 - 1000 Wy_stripper_compressor_interstage_cooler <= 0
_C778: B_stripper_compressor_interstage_cooler
 + 1000 Wy_stripper_compressor_interstage_cooler <= 1000
_C779: - A_stripper_compressor_interstage_cooler
 - B_stripper_compressor_interstage_cooler
 + D_stripper_compressor_interstage_cooler
 - L_stripper_compressor_interstage_cooler
 - R_stripper_compressor_interstage_cooler = 0
_C78: D_direct_contact_cooler_lean_amine_cooler
 - D_lean_amine_cooler_direct_contact_cooler = 0
_C780: - D_compressor_interstage_cooler_stripper
 + D_stripper_compressor_interstage_cooler = 0
_C781: 1000 E1_stripper_compressor_interstage_cooler
 + 1000 E2_stripper_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_stripper
 - 0.5 l_compressor_interstage_cooler - 0.5 l_stripper >= 15
_C782: - 1000 E1_stripper_compressor_interstage_cooler
 + 1000 E2_stripper_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_stripper
 - 0.5 l_compressor_interstage_cooler - 0.5 l_stripper >= -985
_C783: 1000 E1_stripper_compressor_interstage_cooler
 - 1000 E2_stripper_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_stripper
 - 0.5 d_compressor_interstage_cooler - 0.5 d_stripper >= -985
_C784: - 1000 E1_stripper_compressor_interstage_cooler
 - 1000 E2_stripper_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_stripper
 - 0.5 d_compressor_interstage_cooler - 0.5 d_stripper >= -1985
_C785: L_stripper_compressor_interstage_cooler
 - R_compressor_interstage_cooler_stripper = 0
_C786: - L_compressor_interstage_cooler_stripper
 + R_stripper_compressor_interstage_cooler = 0
_C787: A_stripper_compressor_interstage_cooler
 - B_compressor_interstage_cooler_stripper = 0
_C788: - A_compressor_interstage_cooler_stripper
 + B_stripper_compressor_interstage_cooler = 0
_C789: Wx_compressor_interstage_cooler_stripper
 + Wx_stripper_compressor_interstage_cooler = 1
_C79: 1000 E1_direct_contact_cooler_lean_amine_cooler
 + 1000 E2_direct_contact_cooler_lean_amine_cooler + X_direct_contact_cooler
 - X_lean_amine_cooler - 0.5 l_direct_contact_cooler - 0.5 l_lean_amine_cooler
 >= 15
_C790: Wy_compressor_interstage_cooler_stripper
 + Wy_stripper_compressor_interstage_cooler = 1
_C791: - L_storage_tank_compressor_interstage_cooler
 + R_storage_tank_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_storage_tank = 0
_C792: A_storage_tank_compressor_interstage_cooler
 - B_storage_tank_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_storage_tank = 0
_C793: R_storage_tank_compressor_interstage_cooler
 - 1000 Wx_storage_tank_compressor_interstage_cooler <= 0
_C794: L_storage_tank_compressor_interstage_cooler
 + 1000 Wx_storage_tank_compressor_interstage_cooler <= 1000
_C795: A_storage_tank_compressor_interstage_cooler
 - 1000 Wy_storage_tank_compressor_interstage_cooler <= 0
_C796: B_storage_tank_compressor_interstage_cooler
 + 1000 Wy_storage_tank_compressor_interstage_cooler <= 1000
_C797: - A_storage_tank_compressor_interstage_cooler
 - B_storage_tank_compressor_interstage_cooler
 + D_storage_tank_compressor_interstage_cooler
 - L_storage_tank_compressor_interstage_cooler
 - R_storage_tank_compressor_interstage_cooler = 0
_C798: - D_compressor_interstage_cooler_storage_tank
 + D_storage_tank_compressor_interstage_cooler = 0
_C799: 1000 E1_storage_tank_compressor_interstage_cooler
 + 1000 E2_storage_tank_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_storage_tank
 - 0.5 l_compressor_interstage_cooler - 0.5 l_storage_tank >= 76
_C8: Y_absorber - 0.5 d_absorber >= 0
_C80: - 1000 E1_direct_contact_cooler_lean_amine_cooler
 + 1000 E2_direct_contact_cooler_lean_amine_cooler - X_direct_contact_cooler
 + X_lean_amine_cooler - 0.5 l_direct_contact_cooler - 0.5 l_lean_amine_cooler
 >= -985
_C800: - 1000 E1_storage_tank_compressor_interstage_cooler
 + 1000 E2_storage_tank_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_storage_tank
 - 0.5 l_compressor_interstage_cooler - 0.5 l_storage_tank >= -924
_C801: 1000 E1_storage_tank_compressor_interstage_cooler
 - 1000 E2_storage_tank_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_storage_tank
 - 0.5 d_compressor_interstage_cooler - 0.5 d_storage_tank >= -924
_C802: - 1000 E1_storage_tank_compressor_interstage_cooler
 - 1000 E2_storage_tank_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_storage_tank
 - 0.5 d_compressor_interstage_cooler - 0.5 d_storage_tank >= -1924
_C803: L_storage_tank_compressor_interstage_cooler
 - R_compressor_interstage_cooler_storage_tank = 0
_C804: - L_compressor_interstage_cooler_storage_tank
 + R_storage_tank_compressor_interstage_cooler = 0
_C805: A_storage_tank_compressor_interstage_cooler
 - B_compressor_interstage_cooler_storage_tank = 0
_C806: - A_compressor_interstage_cooler_storage_tank
 + B_storage_tank_compressor_interstage_cooler = 0
_C807: Wx_compressor_interstage_cooler_storage_tank
 + Wx_storage_tank_compressor_interstage_cooler = 1
_C808: Wy_compressor_interstage_cooler_storage_tank
 + Wy_storage_tank_compressor_interstage_cooler = 1
_C809: - L_reclaimer_compressor_interstage_cooler
 + R_reclaimer_compressor_interstage_cooler + X_compressor_interstage_cooler
 - X_reclaimer = 0
_C81: 1000 E1_direct_contact_cooler_lean_amine_cooler
 - 1000 E2_direct_contact_cooler_lean_amine_cooler + Y_direct_contact_cooler
 - Y_lean_amine_cooler - 0.5 d_direct_contact_cooler - 0.5 d_lean_amine_cooler
 >= -985
_C810: A_reclaimer_compressor_interstage_cooler
 - B_reclaimer_compressor_interstage_cooler + Y_compressor_interstage_cooler
 - Y_reclaimer = 0
_C811: R_reclaimer_compressor_interstage_cooler
 - 1000 Wx_reclaimer_compressor_interstage_cooler <= 0
_C812: L_reclaimer_compressor_interstage_cooler
 + 1000 Wx_reclaimer_compressor_interstage_cooler <= 1000
_C813: A_reclaimer_compressor_interstage_cooler
 - 1000 Wy_reclaimer_compressor_interstage_cooler <= 0
_C814: B_reclaimer_compressor_interstage_cooler
 + 1000 Wy_reclaimer_compressor_interstage_cooler <= 1000
_C815: - A_reclaimer_compressor_interstage_cooler
 - B_reclaimer_compressor_interstage_cooler
 + D_reclaimer_compressor_interstage_cooler
 - L_reclaimer_compressor_interstage_cooler
 - R_reclaimer_compressor_interstage_cooler = 0
_C816: - D_compressor_interstage_cooler_reclaimer
 + D_reclaimer_compressor_interstage_cooler = 0
_C817: 1000 E1_reclaimer_compressor_interstage_cooler
 + 1000 E2_reclaimer_compressor_interstage_cooler
 - X_compressor_interstage_cooler + X_reclaimer
 - 0.5 l_compressor_interstage_cooler - 0.5 l_reclaimer >= 15
_C818: - 1000 E1_reclaimer_compressor_interstage_cooler
 + 1000 E2_reclaimer_compressor_interstage_cooler
 + X_compressor_interstage_cooler - X_reclaimer
 - 0.5 l_compressor_interstage_cooler - 0.5 l_reclaimer >= -985
_C819: 1000 E1_reclaimer_compressor_interstage_cooler
 - 1000 E2_reclaimer_compressor_interstage_cooler
 - Y_compressor_interstage_cooler + Y_reclaimer
 - 0.5 d_compressor_interstage_cooler - 0.5 d_reclaimer >= -985
_C82: - 1000 E1_direct_contact_cooler_lean_amine_cooler
 - 1000 E2_direct_contact_cooler_lean_amine_cooler - Y_direct_contact_cooler
 + Y_lean_amine_cooler - 0.5 d_direct_contact_cooler - 0.5 d_lean_amine_cooler
 >= -1985
_C820: - 1000 E1_reclaimer_compressor_interstage_cooler
 - 1000 E2_reclaimer_compressor_interstage_cooler
 + Y_compressor_interstage_cooler - Y_reclaimer
 - 0.5 d_compressor_interstage_cooler - 0.5 d_reclaimer >= -1985
_C821: L_reclaimer_compressor_interstage_cooler
 - R_compressor_interstage_cooler_reclaimer = 0
_C822: - L_compressor_interstage_cooler_reclaimer
 + R_reclaimer_compressor_interstage_cooler = 0
_C823: A_reclaimer_compressor_interstage_cooler
 - B_compressor_interstage_cooler_reclaimer = 0
_C824: - A_compressor_interstage_cooler_reclaimer
 + B_reclaimer_compressor_interstage_cooler = 0
_C825: Wx_compressor_interstage_cooler_reclaimer
 + Wx_reclaimer_compressor_interstage_cooler = 1
_C826: Wy_compressor_interstage_cooler_reclaimer
 + Wy_reclaimer_compressor_interstage_cooler = 1
_C827: - L_compressor_house_compressor_interstage_cooler
 + R_compressor_house_compressor_interstage_cooler - X_compressor_house
 + X_compressor_interstage_cooler = 0
_C828: A_compressor_house_compressor_interstage_cooler
 - B_compressor_house_compressor_interstage_cooler - Y_compressor_house
 + Y_compressor_interstage_cooler = 0
_C829: R_compressor_house_compressor_interstage_cooler
 - 1000 Wx_compressor_house_compressor_interstage_cooler <= 0
_C83: L_direct_contact_cooler_lean_amine_cooler
 - R_lean_amine_cooler_direct_contact_cooler = 0
_C830: L_compressor_house_compressor_interstage_cooler
 + 1000 Wx_compressor_house_compressor_interstage_cooler <= 1000
_C831: A_compressor_house_compressor_interstage_cooler
 - 1000 Wy_compressor_house_compressor_interstage_cooler <= 0
_C832: B_compressor_house_compressor_interstage_cooler
 + 1000 Wy_compressor_house_compressor_interstage_cooler <= 1000
_C833: - A_compressor_house_compressor_interstage_cooler
 - B_compressor_house_compressor_interstage_cooler
 + D_compressor_house_compressor_interstage_cooler
 - L_compressor_house_compressor_interstage_cooler
 - R_compressor_house_compressor_interstage_cooler = 0
_C834: D_compressor_house_compressor_interstage_cooler
 - D_compressor_interstage_cooler_compressor_house = 0
_C835: 1000 E1_compressor_house_compressor_interstage_cooler
 + 1000 E2_compressor_house_compressor_interstage_cooler + X_compressor_house
 - X_compressor_interstage_cooler - 0.5 l_compressor_house
 - 0.5 l_compressor_interstage_cooler >= 9
_C836: - 1000 E1_compressor_house_compressor_interstage_cooler
 + 1000 E2_compressor_house_compressor_interstage_cooler - X_compressor_house
 + X_compressor_interstage_cooler - 0.5 l_compressor_house
 - 0.5 l_compressor_interstage_cooler >= -991
_C837: 1000 E1_compressor_house_compressor_interstage_cooler
 - 1000 E2_compressor_house_compressor_interstage_cooler + Y_compressor_house
 - Y_compressor_interstage_cooler - 0.5 d_compressor_house
 - 0.5 d_compressor_interstage_cooler >= -991
_C838: - 1000 E1_compressor_house_compressor_interstage_cooler
 - 1000 E2_compressor_house_compressor_interstage_cooler - Y_compressor_house
 + Y_compressor_interstage_cooler - 0.5 d_compressor_house
 - 0.5 d_compressor_interstage_cooler >= -1991
_C839: L_compressor_house_compressor_interstage_cooler
 - R_compressor_interstage_cooler_compressor_house = 0
_C84: - L_lean_amine_cooler_direct_contact_cooler
 + R_direct_contact_cooler_lean_amine_cooler = 0
_C840: - L_compressor_interstage_cooler_compressor_house
 + R_compressor_house_compressor_interstage_cooler = 0
_C841: A_compressor_house_compressor_interstage_cooler
 - B_compressor_interstage_cooler_compressor_house = 0
_C842: - A_compressor_interstage_cooler_compressor_house
 + B_compressor_house_compressor_interstage_cooler = 0
_C843: Wx_compressor_house_compressor_interstage_cooler
 + Wx_compressor_interstage_cooler_compressor_house = 1
_C844: Wy_compressor_house_compressor_interstage_cooler
 + Wy_compressor_interstage_cooler_compressor_house = 1
_C845: - L_compressor_KOdrum_compressor_interstage_cooler
 + R_compressor_KOdrum_compressor_interstage_cooler - X_compressor_KOdrum
 + X_compressor_interstage_cooler = 0
_C846: A_compressor_KOdrum_compressor_interstage_cooler
 - B_compressor_KOdrum_compressor_interstage_cooler - Y_compressor_KOdrum
 + Y_compressor_interstage_cooler = 0
_C847: R_compressor_KOdrum_compressor_interstage_cooler
 - 1000 Wx_compressor_KOdrum_compressor_interstage_cooler <= 0
_C848: L_compressor_KOdrum_compressor_interstage_cooler
 + 1000 Wx_compressor_KOdrum_compressor_interstage_cooler <= 1000
_C849: A_compressor_KOdrum_compressor_interstage_cooler
 - 1000 Wy_compressor_KOdrum_compressor_interstage_cooler <= 0
_C85: A_direct_contact_cooler_lean_amine_cooler
 - B_lean_amine_cooler_direct_contact_cooler = 0
_C850: B_compressor_KOdrum_compressor_interstage_cooler
 + 1000 Wy_compressor_KOdrum_compressor_interstage_cooler <= 1000
_C851: - A_compressor_KOdrum_compressor_interstage_cooler
 - B_compressor_KOdrum_compressor_interstage_cooler
 + D_compressor_KOdrum_compressor_interstage_cooler
 - L_compressor_KOdrum_compressor_interstage_cooler
 - R_compressor_KOdrum_compressor_interstage_cooler = 0
_C852: D_compressor_KOdrum_compressor_interstage_cooler
 - D_compressor_interstage_cooler_compressor_KOdrum = 0
_C853: 1000 E1_compressor_KOdrum_compressor_interstage_cooler
 + 1000 E2_compressor_KOdrum_compressor_interstage_cooler
 + X_compressor_KOdrum - X_compressor_interstage_cooler
 - 0.5 l_compressor_KOdrum - 0.5 l_compressor_interstage_cooler >= 15
_C854: - 1000 E1_compressor_KOdrum_compressor_interstage_cooler
 + 1000 E2_compressor_KOdrum_compressor_interstage_cooler
 - X_compressor_KOdrum + X_compressor_interstage_cooler
 - 0.5 l_compressor_KOdrum - 0.5 l_compressor_interstage_cooler >= -985
_C855: 1000 E1_compressor_KOdrum_compressor_interstage_cooler
 - 1000 E2_compressor_KOdrum_compressor_interstage_cooler
 + Y_compressor_KOdrum - Y_compressor_interstage_cooler
 - 0.5 d_compressor_KOdrum - 0.5 d_compressor_interstage_cooler >= -985
_C856: - 1000 E1_compressor_KOdrum_compressor_interstage_cooler
 - 1000 E2_compressor_KOdrum_compressor_interstage_cooler
 - Y_compressor_KOdrum + Y_compressor_interstage_cooler
 - 0.5 d_compressor_KOdrum - 0.5 d_compressor_interstage_cooler >= -1985
_C857: L_compressor_KOdrum_compressor_interstage_cooler
 - R_compressor_interstage_cooler_compressor_KOdrum = 0
_C858: - L_compressor_interstage_cooler_compressor_KOdrum
 + R_compressor_KOdrum_compressor_interstage_cooler = 0
_C859: A_compressor_KOdrum_compressor_interstage_cooler
 - B_compressor_interstage_cooler_compressor_KOdrum = 0
_C86: - A_lean_amine_cooler_direct_contact_cooler
 + B_direct_contact_cooler_lean_amine_cooler = 0
_C860: - A_compressor_interstage_cooler_compressor_KOdrum
 + B_compressor_KOdrum_compressor_interstage_cooler = 0
_C861: Wx_compressor_KOdrum_compressor_interstage_cooler
 + Wx_compressor_interstage_cooler_compressor_KOdrum = 1
_C862: Wy_compressor_KOdrum_compressor_interstage_cooler
 + Wy_compressor_interstage_cooler_compressor_KOdrum = 1
_C863: - L_direct_contact_cooler_TEG_dehydration
 + R_direct_contact_cooler_TEG_dehydration + X_TEG_dehydration
 - X_direct_contact_cooler = 0
_C864: A_direct_contact_cooler_TEG_dehydration
 - B_direct_contact_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_direct_contact_cooler = 0
_C865: R_direct_contact_cooler_TEG_dehydration
 - 1000 Wx_direct_contact_cooler_TEG_dehydration <= 0
_C866: L_direct_contact_cooler_TEG_dehydration
 + 1000 Wx_direct_contact_cooler_TEG_dehydration <= 1000
_C867: A_direct_contact_cooler_TEG_dehydration
 - 1000 Wy_direct_contact_cooler_TEG_dehydration <= 0
_C868: B_direct_contact_cooler_TEG_dehydration
 + 1000 Wy_direct_contact_cooler_TEG_dehydration <= 1000
_C869: - A_direct_contact_cooler_TEG_dehydration
 - B_direct_contact_cooler_TEG_dehydration
 + D_direct_contact_cooler_TEG_dehydration
 - L_direct_contact_cooler_TEG_dehydration
 - R_direct_contact_cooler_TEG_dehydration = 0
_C87: Wx_direct_contact_cooler_lean_amine_cooler
 + Wx_lean_amine_cooler_direct_contact_cooler = 1
_C870: - D_TEG_dehydration_direct_contact_cooler
 + D_direct_contact_cooler_TEG_dehydration = 0
_C871: 1000 E1_direct_contact_cooler_TEG_dehydration
 + 1000 E2_direct_contact_cooler_TEG_dehydration - X_TEG_dehydration
 + X_direct_contact_cooler - 0.5 l_TEG_dehydration
 - 0.5 l_direct_contact_cooler >= 15
_C872: - 1000 E1_direct_contact_cooler_TEG_dehydration
 + 1000 E2_direct_contact_cooler_TEG_dehydration + X_TEG_dehydration
 - X_direct_contact_cooler - 0.5 l_TEG_dehydration
 - 0.5 l_direct_contact_cooler >= -985
_C873: 1000 E1_direct_contact_cooler_TEG_dehydration
 - 1000 E2_direct_contact_cooler_TEG_dehydration - Y_TEG_dehydration
 + Y_direct_contact_cooler - 0.5 d_TEG_dehydration
 - 0.5 d_direct_contact_cooler >= -985
_C874: - 1000 E1_direct_contact_cooler_TEG_dehydration
 - 1000 E2_direct_contact_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_direct_contact_cooler - 0.5 d_TEG_dehydration
 - 0.5 d_direct_contact_cooler >= -1985
_C875: L_direct_contact_cooler_TEG_dehydration
 - R_TEG_dehydration_direct_contact_cooler = 0
_C876: - L_TEG_dehydration_direct_contact_cooler
 + R_direct_contact_cooler_TEG_dehydration = 0
_C877: A_direct_contact_cooler_TEG_dehydration
 - B_TEG_dehydration_direct_contact_cooler = 0
_C878: - A_TEG_dehydration_direct_contact_cooler
 + B_direct_contact_cooler_TEG_dehydration = 0
_C879: Wx_TEG_dehydration_direct_contact_cooler
 + Wx_direct_contact_cooler_TEG_dehydration = 1
_C88: Wy_direct_contact_cooler_lean_amine_cooler
 + Wy_lean_amine_cooler_direct_contact_cooler = 1
_C880: Wy_TEG_dehydration_direct_contact_cooler
 + Wy_direct_contact_cooler_TEG_dehydration = 1
_C881: - L_absorber_TEG_dehydration + R_absorber_TEG_dehydration
 + X_TEG_dehydration - X_absorber = 0
_C882: A_absorber_TEG_dehydration - B_absorber_TEG_dehydration
 + Y_TEG_dehydration - Y_absorber = 0
_C883: R_absorber_TEG_dehydration - 1000 Wx_absorber_TEG_dehydration <= 0
_C884: L_absorber_TEG_dehydration + 1000 Wx_absorber_TEG_dehydration <= 1000
_C885: A_absorber_TEG_dehydration - 1000 Wy_absorber_TEG_dehydration <= 0
_C886: B_absorber_TEG_dehydration + 1000 Wy_absorber_TEG_dehydration <= 1000
_C887: - A_absorber_TEG_dehydration - B_absorber_TEG_dehydration
 + D_absorber_TEG_dehydration - L_absorber_TEG_dehydration
 - R_absorber_TEG_dehydration = 0
_C888: - D_TEG_dehydration_absorber + D_absorber_TEG_dehydration = 0
_C889: 1000 E1_absorber_TEG_dehydration + 1000 E2_absorber_TEG_dehydration
 - X_TEG_dehydration + X_absorber - 0.5 l_TEG_dehydration - 0.5 l_absorber
 >= 15
_C89: - L_absorber_lean_amine_cooler + R_absorber_lean_amine_cooler
 - X_absorber + X_lean_amine_cooler = 0
_C890: - 1000 E1_absorber_TEG_dehydration + 1000 E2_absorber_TEG_dehydration
 + X_TEG_dehydration - X_absorber - 0.5 l_TEG_dehydration - 0.5 l_absorber
 >= -985
_C891: 1000 E1_absorber_TEG_dehydration - 1000 E2_absorber_TEG_dehydration
 - Y_TEG_dehydration + Y_absorber - 0.5 d_TEG_dehydration - 0.5 d_absorber
 >= -985
_C892: - 1000 E1_absorber_TEG_dehydration - 1000 E2_absorber_TEG_dehydration
 + Y_TEG_dehydration - Y_absorber - 0.5 d_TEG_dehydration - 0.5 d_absorber
 >= -1985
_C893: L_absorber_TEG_dehydration - R_TEG_dehydration_absorber = 0
_C894: - L_TEG_dehydration_absorber + R_absorber_TEG_dehydration = 0
_C895: A_absorber_TEG_dehydration - B_TEG_dehydration_absorber = 0
_C896: - A_TEG_dehydration_absorber + B_absorber_TEG_dehydration = 0
_C897: Wx_TEG_dehydration_absorber + Wx_absorber_TEG_dehydration = 1
_C898: Wy_TEG_dehydration_absorber + Wy_absorber_TEG_dehydration = 1
_C899: - L_lean_amine_cooler_TEG_dehydration
 + R_lean_amine_cooler_TEG_dehydration + X_TEG_dehydration
 - X_lean_amine_cooler = 0
_C9: 0 O_lean_amine_cooler + l_lean_amine_cooler = 30
_C90: A_absorber_lean_amine_cooler - B_absorber_lean_amine_cooler - Y_absorber
 + Y_lean_amine_cooler = 0
_C900: A_lean_amine_cooler_TEG_dehydration
 - B_lean_amine_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_lean_amine_cooler = 0
_C901: R_lean_amine_cooler_TEG_dehydration
 - 1000 Wx_lean_amine_cooler_TEG_dehydration <= 0
_C902: L_lean_amine_cooler_TEG_dehydration
 + 1000 Wx_lean_amine_cooler_TEG_dehydration <= 1000
_C903: A_lean_amine_cooler_TEG_dehydration
 - 1000 Wy_lean_amine_cooler_TEG_dehydration <= 0
_C904: B_lean_amine_cooler_TEG_dehydration
 + 1000 Wy_lean_amine_cooler_TEG_dehydration <= 1000
_C905: - A_lean_amine_cooler_TEG_dehydration
 - B_lean_amine_cooler_TEG_dehydration + D_lean_amine_cooler_TEG_dehydration
 - L_lean_amine_cooler_TEG_dehydration - R_lean_amine_cooler_TEG_dehydration
 = 0
_C906: - D_TEG_dehydration_lean_amine_cooler
 + D_lean_amine_cooler_TEG_dehydration = 0
_C907: 1000 E1_lean_amine_cooler_TEG_dehydration
 + 1000 E2_lean_amine_cooler_TEG_dehydration - X_TEG_dehydration
 + X_lean_amine_cooler - 0.5 l_TEG_dehydration - 0.5 l_lean_amine_cooler >= 15
_C908: - 1000 E1_lean_amine_cooler_TEG_dehydration
 + 1000 E2_lean_amine_cooler_TEG_dehydration + X_TEG_dehydration
 - X_lean_amine_cooler - 0.5 l_TEG_dehydration - 0.5 l_lean_amine_cooler
 >= -985
_C909: 1000 E1_lean_amine_cooler_TEG_dehydration
 - 1000 E2_lean_amine_cooler_TEG_dehydration - Y_TEG_dehydration
 + Y_lean_amine_cooler - 0.5 d_TEG_dehydration - 0.5 d_lean_amine_cooler
 >= -985
_C91: R_absorber_lean_amine_cooler - 1000 Wx_absorber_lean_amine_cooler <= 0
_C910: - 1000 E1_lean_amine_cooler_TEG_dehydration
 - 1000 E2_lean_amine_cooler_TEG_dehydration + Y_TEG_dehydration
 - Y_lean_amine_cooler - 0.5 d_TEG_dehydration - 0.5 d_lean_amine_cooler
 >= -1985
_C911: L_lean_amine_cooler_TEG_dehydration
 - R_TEG_dehydration_lean_amine_cooler = 0
_C912: - L_TEG_dehydration_lean_amine_cooler
 + R_lean_amine_cooler_TEG_dehydration = 0
_C913: A_lean_amine_cooler_TEG_dehydration
 - B_TEG_dehydration_lean_amine_cooler = 0
_C914: - A_TEG_dehydration_lean_amine_cooler
 + B_lean_amine_cooler_TEG_dehydration = 0
_C915: Wx_TEG_dehydration_lean_amine_cooler
 + Wx_lean_amine_cooler_TEG_dehydration = 1
_C916: Wy_TEG_dehydration_lean_amine_cooler
 + Wy_lean_amine_cooler_TEG_dehydration = 1
_C917: - L_cross_heat_exchanger_TEG_dehydration
 + R_cross_heat_exchanger_TEG_dehydration + X_TEG_dehydration
 - X_cross_heat_exchanger = 0
_C918: A_cross_heat_exchanger_TEG_dehydration
 - B_cross_heat_exchanger_TEG_dehydration + Y_TEG_dehydration
 - Y_cross_heat_exchanger = 0
_C919: R_cross_heat_exchanger_TEG_dehydration
 - 1000 Wx_cross_heat_exchanger_TEG_dehydration <= 0
_C92: L_absorber_lean_amine_cooler + 1000 Wx_absorber_lean_amine_cooler
 <= 1000
_C920: L_cross_heat_exchanger_TEG_dehydration
 + 1000 Wx_cross_heat_exchanger_TEG_dehydration <= 1000
_C921: A_cross_heat_exchanger_TEG_dehydration
 - 1000 Wy_cross_heat_exchanger_TEG_dehydration <= 0
_C922: B_cross_heat_exchanger_TEG_dehydration
 + 1000 Wy_cross_heat_exchanger_TEG_dehydration <= 1000
_C923: - A_cross_heat_exchanger_TEG_dehydration
 - B_cross_heat_exchanger_TEG_dehydration
 + D_cross_heat_exchanger_TEG_dehydration
 - L_cross_heat_exchanger_TEG_dehydration
 - R_cross_heat_exchanger_TEG_dehydration = 0
_C924: - D_TEG_dehydration_cross_heat_exchanger
 + D_cross_heat_exchanger_TEG_dehydration = 0
_C925: 1000 E1_cross_heat_exchanger_TEG_dehydration
 + 1000 E2_cross_heat_exchanger_TEG_dehydration - X_TEG_dehydration
 + X_cross_heat_exchanger - 0.5 l_TEG_dehydration - 0.5 l_cross_heat_exchanger
 >= 15
_C926: - 1000 E1_cross_heat_exchanger_TEG_dehydration
 + 1000 E2_cross_heat_exchanger_TEG_dehydration + X_TEG_dehydration
 - X_cross_heat_exchanger - 0.5 l_TEG_dehydration - 0.5 l_cross_heat_exchanger
 >= -985
_C927: 1000 E1_cross_heat_exchanger_TEG_dehydration
 - 1000 E2_cross_heat_exchanger_TEG_dehydration - Y_TEG_dehydration
 + Y_cross_heat_exchanger - 0.5 d_TEG_dehydration - 0.5 d_cross_heat_exchanger
 >= -985
_C928: - 1000 E1_cross_heat_exchanger_TEG_dehydration
 - 1000 E2_cross_heat_exchanger_TEG_dehydration + Y_TEG_dehydration
 - Y_cross_heat_exchanger - 0.5 d_TEG_dehydration - 0.5 d_cross_heat_exchanger
 >= -1985
_C929: L_cross_heat_exchanger_TEG_dehydration
 - R_TEG_dehydration_cross_heat_exchanger = 0
_C93: A_absorber_lean_amine_cooler - 1000 Wy_absorber_lean_amine_cooler <= 0
_C930: - L_TEG_dehydration_cross_heat_exchanger
 + R_cross_heat_exchanger_TEG_dehydration = 0
_C931: A_cross_heat_exchanger_TEG_dehydration
 - B_TEG_dehydration_cross_heat_exchanger = 0
_C932: - A_TEG_dehydration_cross_heat_exchanger
 + B_cross_heat_exchanger_TEG_dehydration = 0
_C933: Wx_TEG_dehydration_cross_heat_exchanger
 + Wx_cross_heat_exchanger_TEG_dehydration = 1
_C934: Wy_TEG_dehydration_cross_heat_exchanger
 + Wy_cross_heat_exchanger_TEG_dehydration = 1
_C935: - L_stripper_TEG_dehydration + R_stripper_TEG_dehydration
 + X_TEG_dehydration - X_stripper = 0
_C936: A_stripper_TEG_dehydration - B_stripper_TEG_dehydration
 + Y_TEG_dehydration - Y_stripper = 0
_C937: R_stripper_TEG_dehydration - 1000 Wx_stripper_TEG_dehydration <= 0
_C938: L_stripper_TEG_dehydration + 1000 Wx_stripper_TEG_dehydration <= 1000
_C939: A_stripper_TEG_dehydration - 1000 Wy_stripper_TEG_dehydration <= 0
_C94: B_absorber_lean_amine_cooler + 1000 Wy_absorber_lean_amine_cooler
 <= 1000
_C940: B_stripper_TEG_dehydration + 1000 Wy_stripper_TEG_dehydration <= 1000
_C941: - A_stripper_TEG_dehydration - B_stripper_TEG_dehydration
 + D_stripper_TEG_dehydration - L_stripper_TEG_dehydration
 - R_stripper_TEG_dehydration = 0
_C942: - D_TEG_dehydration_stripper + D_stripper_TEG_dehydration = 0
_C943: 1000 E1_stripper_TEG_dehydration + 1000 E2_stripper_TEG_dehydration
 - X_TEG_dehydration + X_stripper - 0.5 l_TEG_dehydration - 0.5 l_stripper
 >= 15
_C944: - 1000 E1_stripper_TEG_dehydration + 1000 E2_stripper_TEG_dehydration
 + X_TEG_dehydration - X_stripper - 0.5 l_TEG_dehydration - 0.5 l_stripper
 >= -985
_C945: 1000 E1_stripper_TEG_dehydration - 1000 E2_stripper_TEG_dehydration
 - Y_TEG_dehydration + Y_stripper - 0.5 d_TEG_dehydration - 0.5 d_stripper
 >= -985
_C946: - 1000 E1_stripper_TEG_dehydration - 1000 E2_stripper_TEG_dehydration
 + Y_TEG_dehydration - Y_stripper - 0.5 d_TEG_dehydration - 0.5 d_stripper
 >= -1985
_C947: L_stripper_TEG_dehydration - R_TEG_dehydration_stripper = 0
_C948: - L_TEG_dehydration_stripper + R_stripper_TEG_dehydration = 0
_C949: A_stripper_TEG_dehydration - B_TEG_dehydration_stripper = 0
_C95: - A_absorber_lean_amine_cooler - B_absorber_lean_amine_cooler
 + D_absorber_lean_amine_cooler - L_absorber_lean_amine_cooler
 - R_absorber_lean_amine_cooler = 0
_C950: - A_TEG_dehydration_stripper + B_stripper_TEG_dehydration = 0
_C951: Wx_TEG_dehydration_stripper + Wx_stripper_TEG_dehydration = 1
_C952: Wy_TEG_dehydration_stripper + Wy_stripper_TEG_dehydration = 1
_C953: - L_storage_tank_TEG_dehydration + R_storage_tank_TEG_dehydration
 + X_TEG_dehydration - X_storage_tank = 0
_C954: A_storage_tank_TEG_dehydration - B_storage_tank_TEG_dehydration
 + Y_TEG_dehydration - Y_storage_tank = 0
_C955: R_storage_tank_TEG_dehydration - 1000 Wx_storage_tank_TEG_dehydration
 <= 0
_C956: L_storage_tank_TEG_dehydration + 1000 Wx_storage_tank_TEG_dehydration
 <= 1000
_C957: A_storage_tank_TEG_dehydration - 1000 Wy_storage_tank_TEG_dehydration
 <= 0
_C958: B_storage_tank_TEG_dehydration + 1000 Wy_storage_tank_TEG_dehydration
 <= 1000
_C959: - A_storage_tank_TEG_dehydration - B_storage_tank_TEG_dehydration
 + D_storage_tank_TEG_dehydration - L_storage_tank_TEG_dehydration
 - R_storage_tank_TEG_dehydration = 0
_C96: D_absorber_lean_amine_cooler - D_lean_amine_cooler_absorber = 0
_C960: - D_TEG_dehydration_storage_tank + D_storage_tank_TEG_dehydration = 0
_C961: 1000 E1_storage_tank_TEG_dehydration
 + 1000 E2_storage_tank_TEG_dehydration - X_TEG_dehydration + X_storage_tank
 - 0.5 l_TEG_dehydration - 0.5 l_storage_tank >= 76
_C962: - 1000 E1_storage_tank_TEG_dehydration
 + 1000 E2_storage_tank_TEG_dehydration + X_TEG_dehydration - X_storage_tank
 - 0.5 l_TEG_dehydration - 0.5 l_storage_tank >= -924
_C963: 1000 E1_storage_tank_TEG_dehydration
 - 1000 E2_storage_tank_TEG_dehydration - Y_TEG_dehydration + Y_storage_tank
 - 0.5 d_TEG_dehydration - 0.5 d_storage_tank >= -924
_C964: - 1000 E1_storage_tank_TEG_dehydration
 - 1000 E2_storage_tank_TEG_dehydration + Y_TEG_dehydration - Y_storage_tank
 - 0.5 d_TEG_dehydration - 0.5 d_storage_tank >= -1924
_C965: L_storage_tank_TEG_dehydration - R_TEG_dehydration_storage_tank = 0
_C966: - L_TEG_dehydration_storage_tank + R_storage_tank_TEG_dehydration = 0
_C967: A_storage_tank_TEG_dehydration - B_TEG_dehydration_storage_tank = 0
_C968: - A_TEG_dehydration_storage_tank + B_storage_tank_TEG_dehydration = 0
_C969: Wx_TEG_dehydration_storage_tank + Wx_storage_tank_TEG_dehydration = 1
_C97: 1000 E1_absorber_lean_amine_cooler + 1000 E2_absorber_lean_amine_cooler
 + X_absorber - X_lean_amine_cooler - 0.5 l_absorber - 0.5 l_lean_amine_cooler
 >= 15
_C970: Wy_TEG_dehydration_storage_tank + Wy_storage_tank_TEG_dehydration = 1
_C971: - L_reclaimer_TEG_dehydration + R_reclaimer_TEG_dehydration
 + X_TEG_dehydration - X_reclaimer = 0
_C972: A_reclaimer_TEG_dehydration - B_reclaimer_TEG_dehydration
 + Y_TEG_dehydration - Y_reclaimer = 0
_C973: R_reclaimer_TEG_dehydration - 1000 Wx_reclaimer_TEG_dehydration <= 0
_C974: L_reclaimer_TEG_dehydration + 1000 Wx_reclaimer_TEG_dehydration <= 1000
_C975: A_reclaimer_TEG_dehydration - 1000 Wy_reclaimer_TEG_dehydration <= 0
_C976: B_reclaimer_TEG_dehydration + 1000 Wy_reclaimer_TEG_dehydration <= 1000
_C977: - A_reclaimer_TEG_dehydration - B_reclaimer_TEG_dehydration
 + D_reclaimer_TEG_dehydration - L_reclaimer_TEG_dehydration
 - R_reclaimer_TEG_dehydration = 0
_C978: - D_TEG_dehydration_reclaimer + D_reclaimer_TEG_dehydration = 0
_C979: 1000 E1_reclaimer_TEG_dehydration + 1000 E2_reclaimer_TEG_dehydration
 - X_TEG_dehydration + X_reclaimer - 0.5 l_TEG_dehydration - 0.5 l_reclaimer
 >= 15
_C98: - 1000 E1_absorber_lean_amine_cooler
 + 1000 E2_absorber_lean_amine_cooler - X_absorber + X_lean_amine_cooler
 - 0.5 l_absorber - 0.5 l_lean_amine_cooler >= -985
_C980: - 1000 E1_reclaimer_TEG_dehydration + 1000 E2_reclaimer_TEG_dehydration
 + X_TEG_dehydration - X_reclaimer - 0.5 l_TEG_dehydration - 0.5 l_reclaimer
 >= -985
_C981: 1000 E1_reclaimer_TEG_dehydration - 1000 E2_reclaimer_TEG_dehydration
 - Y_TEG_dehydration + Y_reclaimer - 0.5 d_TEG_dehydration - 0.5 d_reclaimer
 >= -985
_C982: - 1000 E1_reclaimer_TEG_dehydration - 1000 E2_reclaimer_TEG_dehydration
 + Y_TEG_dehydration - Y_reclaimer - 0.5 d_TEG_dehydration - 0.5 d_reclaimer
 >= -1985
_C983: L_reclaimer_TEG_dehydration - R_TEG_dehydration_reclaimer = 0
_C984: - L_TEG_dehydration_reclaimer + R_reclaimer_TEG_dehydration = 0
_C985: A_reclaimer_TEG_dehydration - B_TEG_dehydration_reclaimer = 0
_C986: - A_TEG_dehydration_reclaimer + B_reclaimer_TEG_dehydration = 0
_C987: Wx_TEG_dehydration_reclaimer + Wx_reclaimer_TEG_dehydration = 1
_C988: Wy_TEG_dehydration_reclaimer + Wy_reclaimer_TEG_dehydration = 1
_C989: - L_compressor_house_TEG_dehydration
 + R_compressor_house_TEG_dehydration + X_TEG_dehydration - X_compressor_house
 = 0
_C99: 1000 E1_absorber_lean_amine_cooler - 1000 E2_absorber_lean_amine_cooler
 + Y_absorber - Y_lean_amine_cooler - 0.5 d_absorber - 0.5 d_lean_amine_cooler
 >= -985
_C990: A_compressor_house_TEG_dehydration - B_compressor_house_TEG_dehydration
 + Y_TEG_dehydration - Y_compressor_house = 0
_C991: R_compressor_house_TEG_dehydration
 - 1000 Wx_compressor_house_TEG_dehydration <= 0
_C992: L_compressor_house_TEG_dehydration
 + 1000 Wx_compressor_house_TEG_dehydration <= 1000
_C993: A_compressor_house_TEG_dehydration
 - 1000 Wy_compressor_house_TEG_dehydration <= 0
_C994: B_compressor_house_TEG_dehydration
 + 1000 Wy_compressor_house_TEG_dehydration <= 1000
_C995: - A_compressor_house_TEG_dehydration
 - B_compressor_house_TEG_dehydration + D_compressor_house_TEG_dehydration
 - L_compressor_house_TEG_dehydration - R_compressor_house_TEG_dehydration = 0
_C996: - D_TEG_dehydration_compressor_house
 + D_compressor_house_TEG_dehydration = 0
_C997: 1000 E1_compressor_house_TEG_dehydration
 + 1000 E2_compressor_house_TEG_dehydration - X_TEG_dehydration
 + X_compressor_house - 0.5 l_TEG_dehydration - 0.5 l_compressor_house >= 9
_C998: - 1000 E1_compressor_house_TEG_dehydration
 + 1000 E2_compressor_house_TEG_dehydration + X_TEG_dehydration
 - X_compressor_house - 0.5 l_TEG_dehydration - 0.5 l_compressor_house >= -991
_C999: 1000 E1_compressor_house_TEG_dehydration
 - 1000 E2_compressor_house_TEG_dehydration - Y_TEG_dehydration
 + Y_compressor_house - 0.5 d_TEG_dehydration - 0.5 d_compressor_house >= -991
Bounds
X_TEG_dehydration = 192
X_absorber = 87
X_compressor_KOdrum = 139
X_compressor_house = 165
X_compressor_interstage_cooler = 189
X_cooling_tower free
X_coolingwater_pumphouse free
X_cross_heat_exchanger = 41
X_direct_contact_cooler = 158
X_lean_amine_cooler = 34
X_reclaimer = 9
X_storage_tank = 17
X_stripper = 140
Y_TEG_dehydration = 136
Y_absorber = 33
Y_compressor_KOdrum = 169
Y_compressor_house = 169
Y_compressor_interstage_cooler = 169
Y_cooling_tower free
Y_coolingwater_pumphouse free
Y_cross_heat_exchanger = 72
Y_direct_contact_cooler = 33
Y_lean_amine_cooler = 33
Y_reclaimer = 72
Y_storage_tank = 174
Y_stripper = 111
Binaries
E1_TEG_dehydration_cooling_tower
E1_TEG_dehydration_coolingwater_pumphouse
E1_absorber_TEG_dehydration
E1_absorber_compressor_KOdrum
E1_absorber_compressor_house
E1_absorber_compressor_interstage_cooler
E1_absorber_cooling_tower
E1_absorber_coolingwater_pumphouse
E1_absorber_cross_heat_exchanger
E1_absorber_lean_amine_cooler
E1_absorber_reclaimer
E1_absorber_storage_tank
E1_absorber_stripper
E1_compressor_KOdrum_TEG_dehydration
E1_compressor_KOdrum_compressor_interstage_cooler
E1_compressor_KOdrum_cooling_tower
E1_compressor_KOdrum_coolingwater_pumphouse
E1_compressor_house_TEG_dehydration
E1_compressor_house_compressor_KOdrum
E1_compressor_house_compressor_interstage_cooler
E1_compressor_house_cooling_tower
E1_compressor_house_coolingwater_pumphouse
E1_compressor_interstage_cooler_TEG_dehydration
E1_compressor_interstage_cooler_cooling_tower
E1_compressor_interstage_cooler_coolingwater_pumphouse
E1_cooling_tower_coolingwater_pumphouse
E1_cross_heat_exchanger_TEG_dehydration
E1_cross_heat_exchanger_compressor_KOdrum
E1_cross_heat_exchanger_compressor_house
E1_cross_heat_exchanger_compressor_interstage_cooler
E1_cross_heat_exchanger_cooling_tower
E1_cross_heat_exchanger_coolingwater_pumphouse
E1_cross_heat_exchanger_reclaimer
E1_cross_heat_exchanger_storage_tank
E1_cross_heat_exchanger_stripper
E1_direct_contact_cooler_TEG_dehydration
E1_direct_contact_cooler_absorber
E1_direct_contact_cooler_compressor_KOdrum
E1_direct_contact_cooler_compressor_house
E1_direct_contact_cooler_compressor_interstage_cooler
E1_direct_contact_cooler_cooling_tower
E1_direct_contact_cooler_coolingwater_pumphouse
E1_direct_contact_cooler_cross_heat_exchanger
E1_direct_contact_cooler_lean_amine_cooler
E1_direct_contact_cooler_reclaimer
E1_direct_contact_cooler_storage_tank
E1_direct_contact_cooler_stripper
E1_lean_amine_cooler_TEG_dehydration
E1_lean_amine_cooler_compressor_KOdrum
E1_lean_amine_cooler_compressor_house
E1_lean_amine_cooler_compressor_interstage_cooler
E1_lean_amine_cooler_cooling_tower
E1_lean_amine_cooler_coolingwater_pumphouse
E1_lean_amine_cooler_cross_heat_exchanger
E1_lean_amine_cooler_reclaimer
E1_lean_amine_cooler_storage_tank
E1_lean_amine_cooler_stripper
E1_reclaimer_TEG_dehydration
E1_reclaimer_compressor_KOdrum
E1_reclaimer_compressor_house
E1_reclaimer_compressor_interstage_cooler
E1_reclaimer_cooling_tower
E1_reclaimer_coolingwater_pumphouse
E1_storage_tank_TEG_dehydration
E1_storage_tank_compressor_KOdrum
E1_storage_tank_compressor_house
E1_storage_tank_compressor_interstage_cooler
E1_storage_tank_cooling_tower
E1_storage_tank_coolingwater_pumphouse
E1_storage_tank_reclaimer
E1_stripper_TEG_dehydration
E1_stripper_compressor_KOdrum
E1_stripper_compressor_house
E1_stripper_compressor_interstage_cooler
E1_stripper_cooling_tower
E1_stripper_coolingwater_pumphouse
E1_stripper_reclaimer
E1_stripper_storage_tank
E2_TEG_dehydration_cooling_tower
E2_TEG_dehydration_coolingwater_pumphouse
E2_absorber_TEG_dehydration
E2_absorber_compressor_KOdrum
E2_absorber_compressor_house
E2_absorber_compressor_interstage_cooler
E2_absorber_cooling_tower
E2_absorber_coolingwater_pumphouse
E2_absorber_cross_heat_exchanger
E2_absorber_lean_amine_cooler
E2_absorber_reclaimer
E2_absorber_storage_tank
E2_absorber_stripper
E2_compressor_KOdrum_TEG_dehydration
E2_compressor_KOdrum_compressor_interstage_cooler
E2_compressor_KOdrum_cooling_tower
E2_compressor_KOdrum_coolingwater_pumphouse
E2_compressor_house_TEG_dehydration
E2_compressor_house_compressor_KOdrum
E2_compressor_house_compressor_interstage_cooler
E2_compressor_house_cooling_tower
E2_compressor_house_coolingwater_pumphouse
E2_compressor_interstage_cooler_TEG_dehydration
E2_compressor_interstage_cooler_cooling_tower
E2_compressor_interstage_cooler_coolingwater_pumphouse
E2_cooling_tower_coolingwater_pumphouse
E2_cross_heat_exchanger_TEG_dehydration
E2_cross_heat_exchanger_compressor_KOdrum
E2_cross_heat_exchanger_compressor_house
E2_cross_heat_exchanger_compressor_interstage_cooler
E2_cross_heat_exchanger_cooling_tower
E2_cross_heat_exchanger_coolingwater_pumphouse
E2_cross_heat_exchanger_reclaimer
E2_cross_heat_exchanger_storage_tank
E2_cross_heat_exchanger_stripper
E2_direct_contact_cooler_TEG_dehydration
E2_direct_contact_cooler_absorber
E2_direct_contact_cooler_compressor_KOdrum
E2_direct_contact_cooler_compressor_house
E2_direct_contact_cooler_compressor_interstage_cooler
E2_direct_contact_cooler_cooling_tower
E2_direct_contact_cooler_coolingwater_pumphouse
E2_direct_contact_cooler_cross_heat_exchanger
E2_direct_contact_cooler_lean_amine_cooler
E2_direct_contact_cooler_reclaimer
E2_direct_contact_cooler_storage_tank
E2_direct_contact_cooler_stripper
E2_lean_amine_cooler_TEG_dehydration
E2_lean_amine_cooler_compressor_KOdrum
E2_lean_amine_cooler_compressor_house
E2_lean_amine_cooler_compressor_interstage_cooler
E2_lean_amine_cooler_cooling_tower
E2_lean_amine_cooler_coolingwater_pumphouse
E2_lean_amine_cooler_cross_heat_exchanger
E2_lean_amine_cooler_reclaimer
E2_lean_amine_cooler_storage_tank
E2_lean_amine_cooler_stripper
E2_reclaimer_TEG_dehydration
E2_reclaimer_compressor_KOdrum
E2_reclaimer_compressor_house
E2_reclaimer_compressor_interstage_cooler
E2_reclaimer_cooling_tower
E2_reclaimer_coolingwater_pumphouse
E2_storage_tank_TEG_dehydration
E2_storage_tank_compressor_KOdrum
E2_storage_tank_compressor_house
E2_storage_tank_compressor_interstage_cooler
E2_storage_tank_cooling_tower
E2_storage_tank_coolingwater_pumphouse
E2_storage_tank_reclaimer
E2_stripper_TEG_dehydration
E2_stripper_compressor_KOdrum
E2_stripper_compressor_house
E2_stripper_compressor_interstage_cooler
E2_stripper_cooling_tower
E2_stripper_coolingwater_pumphouse
E2_stripper_reclaimer
E2_stripper_storage_tank
Gn_1
Gn_10
Gn_100
Gn_101
Gn_102
Gn_103
Gn_104
Gn_105
Gn_106
Gn_107
Gn_108
Gn_109
Gn_11
Gn_110
Gn_111
Gn_112
Gn_113
Gn_114
Gn_115
Gn_116
Gn_117
Gn_118
Gn_119
Gn_12
Gn_120
Gn_121
Gn_122
Gn_123
Gn_124
Gn_125
Gn_126
Gn_127
Gn_128
Gn_129
Gn_13
Gn_130
Gn_131
Gn_132
Gn_133
Gn_134
Gn_135
Gn_136
Gn_137
Gn_138
Gn_139
Gn_14
Gn_140
Gn_141
Gn_142
Gn_143
Gn_144
Gn_145
Gn_146
Gn_147
Gn_148
Gn_149
Gn_15
Gn_150
Gn_151
Gn_152
Gn_153
Gn_154
Gn_155
Gn_156
Gn_157
Gn_158
Gn_159
Gn_16
Gn_160
Gn_161
Gn_162
Gn_163
Gn_164
Gn_165
Gn_166
Gn_167
Gn_168
Gn_169
Gn_17
Gn_170
Gn_171
Gn_172
Gn_173
Gn_174
Gn_175
Gn_176
Gn_177
Gn_178
Gn_179
Gn_18
Gn_180
Gn_181
Gn_182
Gn_183
Gn_184
Gn_185
Gn_186
Gn_187
Gn_188
Gn_189
Gn_19
Gn_190
Gn_191
Gn_192
Gn_193
Gn_194
Gn_195
Gn_196
Gn_197
Gn_198
Gn_199
Gn_2
Gn_20
Gn_200
Gn_201
Gn_202
Gn_203
Gn_204
Gn_205
Gn_206
Gn_207
Gn_208
Gn_209
Gn_21
Gn_210
Gn_211
Gn_212
Gn_213
Gn_214
Gn_215
Gn_216
Gn_217
Gn_218
Gn_219
Gn_22
Gn_220
Gn_221
Gn_222
Gn_223
Gn_224
Gn_225
Gn_226
Gn_227
Gn_228
Gn_229
Gn_23
Gn_230
Gn_231
Gn_232
Gn_233
Gn_234
Gn_235
Gn_236
Gn_237
Gn_238
Gn_239
Gn_24
Gn_240
Gn_241
Gn_242
Gn_243
Gn_244
Gn_245
Gn_246
Gn_247
Gn_248
Gn_249
Gn_25
Gn_26
Gn_27
Gn_28
Gn_29
Gn_3
Gn_30
Gn_31
Gn_32
Gn_33
Gn_34
Gn_35
Gn_36
Gn_37
Gn_38
Gn_39
Gn_4
Gn_40
Gn_41
Gn_42
Gn_43
Gn_44
Gn_45
Gn_46
Gn_47
Gn_48
Gn_49
Gn_5
Gn_50
Gn_51
Gn_52
Gn_53
Gn_54
Gn_55
Gn_56
Gn_57
Gn_58
Gn_59
Gn_6
Gn_60
Gn_61
Gn_62
Gn_63
Gn_64
Gn_65
Gn_66
Gn_67
Gn_68
Gn_69
Gn_7
Gn_70
Gn_71
Gn_72
Gn_73
Gn_74
Gn_75
Gn_76
Gn_77
Gn_78
Gn_79
Gn_8
Gn_80
Gn_81
Gn_82
Gn_83
Gn_84
Gn_85
Gn_86
Gn_87
Gn_88
Gn_89
Gn_9
Gn_90
Gn_91
Gn_92
Gn_93
Gn_94
Gn_95
Gn_96
Gn_97
Gn_98
Gn_99
O_TEG_dehydration
O_absorber
O_compressor_KOdrum
O_compressor_house
O_compressor_interstage_cooler
O_cooling_tower
O_coolingwater_pumphouse
O_cross_heat_exchanger
O_direct_contact_cooler
O_lean_amine_cooler
O_reclaimer
O_storage_tank
O_stripper
Psic_absorber_TEG_dehydration_MEA
Psic_absorber_compressor_KOdrum_MEA
Psic_absorber_compressor_house_MEA
Psic_absorber_compressor_interstage_cooler_MEA
Psic_absorber_cooling_tower_MEA
Psic_absorber_coolingwater_pumphouse_MEA
Psic_absorber_cross_heat_exchanger_MEA
Psic_absorber_direct_contact_cooler_MEA
Psic_absorber_lean_amine_cooler_MEA
Psic_absorber_reclaimer_MEA
Psic_absorber_storage_tank_MEA
Psic_absorber_stripper_MEA
Psic_storage_tank_TEG_dehydration_MEA
Psic_storage_tank_absorber_MEA
Psic_storage_tank_compressor_KOdrum_MEA
Psic_storage_tank_compressor_house_MEA
Psic_storage_tank_compressor_interstage_cooler_MEA
Psic_storage_tank_cooling_tower_MEA
Psic_storage_tank_coolingwater_pumphouse_MEA
Psic_storage_tank_cross_heat_exchanger_MEA
Psic_storage_tank_direct_contact_cooler_MEA
Psic_storage_tank_lean_amine_cooler_MEA
Psic_storage_tank_reclaimer_MEA
Psic_storage_tank_stripper_MEA
Psic_stripper_TEG_dehydration_MEA
Psic_stripper_absorber_MEA
Psic_stripper_compressor_KOdrum_MEA
Psic_stripper_compressor_house_MEA
Psic_stripper_compressor_interstage_cooler_MEA
Psic_stripper_cooling_tower_MEA
Psic_stripper_coolingwater_pumphouse_MEA
Psic_stripper_cross_heat_exchanger_MEA
Psic_stripper_direct_contact_cooler_MEA
Psic_stripper_lean_amine_cooler_MEA
Psic_stripper_reclaimer_MEA
Psic_stripper_storage_tank_MEA
Psie_absorber_TEG_dehydration
Psie_absorber_compressor_KOdrum
Psie_absorber_compressor_house
Psie_absorber_compressor_interstage_cooler
Psie_absorber_cooling_tower
Psie_absorber_coolingwater_pumphouse
Psie_absorber_cross_heat_exchanger
Psie_absorber_direct_contact_cooler
Psie_absorber_lean_amine_cooler
Psie_absorber_reclaimer
Psie_absorber_storage_tank
Psie_absorber_stripper
Psie_storage_tank_TEG_dehydration
Psie_storage_tank_absorber
Psie_storage_tank_compressor_KOdrum
Psie_storage_tank_compressor_house
Psie_storage_tank_compressor_interstage_cooler
Psie_storage_tank_cooling_tower
Psie_storage_tank_coolingwater_pumphouse
Psie_storage_tank_cross_heat_exchanger
Psie_storage_tank_direct_contact_cooler
Psie_storage_tank_lean_amine_cooler
Psie_storage_tank_reclaimer
Psie_storage_tank_stripper
Psie_stripper_TEG_dehydration
Psie_stripper_absorber
Psie_stripper_compressor_KOdrum
Psie_stripper_compressor_house
Psie_stripper_compressor_interstage_cooler
Psie_stripper_cooling_tower
Psie_stripper_coolingwater_pumphouse
Psie_stripper_cross_heat_exchanger
Psie_stripper_direct_contact_cooler
Psie_stripper_lean_amine_cooler
Psie_stripper_reclaimer
Psie_stripper_storage_tank
Wx_TEG_dehydration_absorber
Wx_TEG_dehydration_compressor_KOdrum
Wx_TEG_dehydration_compressor_house
Wx_TEG_dehydration_compressor_interstage_cooler
Wx_TEG_dehydration_cooling_tower
Wx_TEG_dehydration_coolingwater_pumphouse
Wx_TEG_dehydration_cross_heat_exchanger
Wx_TEG_dehydration_direct_contact_cooler
Wx_TEG_dehydration_lean_amine_cooler
Wx_TEG_dehydration_reclaimer
Wx_TEG_dehydration_storage_tank
Wx_TEG_dehydration_stripper
Wx_absorber_TEG_dehydration
Wx_absorber_compressor_KOdrum
Wx_absorber_compressor_house
Wx_absorber_compressor_interstage_cooler
Wx_absorber_cooling_tower
Wx_absorber_coolingwater_pumphouse
Wx_absorber_cross_heat_exchanger
Wx_absorber_direct_contact_cooler
Wx_absorber_lean_amine_cooler
Wx_absorber_reclaimer
Wx_absorber_storage_tank
Wx_absorber_stripper
Wx_compressor_KOdrum_TEG_dehydration
Wx_compressor_KOdrum_absorber
Wx_compressor_KOdrum_compressor_house
Wx_compressor_KOdrum_compressor_interstage_cooler
Wx_compressor_KOdrum_cooling_tower
Wx_compressor_KOdrum_coolingwater_pumphouse
Wx_compressor_KOdrum_cross_heat_exchanger
Wx_compressor_KOdrum_direct_contact_cooler
Wx_compressor_KOdrum_lean_amine_cooler
Wx_compressor_KOdrum_reclaimer
Wx_compressor_KOdrum_storage_tank
Wx_compressor_KOdrum_stripper
Wx_compressor_house_TEG_dehydration
Wx_compressor_house_absorber
Wx_compressor_house_compressor_KOdrum
Wx_compressor_house_compressor_interstage_cooler
Wx_compressor_house_cooling_tower
Wx_compressor_house_coolingwater_pumphouse
Wx_compressor_house_cross_heat_exchanger
Wx_compressor_house_direct_contact_cooler
Wx_compressor_house_lean_amine_cooler
Wx_compressor_house_reclaimer
Wx_compressor_house_storage_tank
Wx_compressor_house_stripper
Wx_compressor_interstage_cooler_TEG_dehydration
Wx_compressor_interstage_cooler_absorber
Wx_compressor_interstage_cooler_compressor_KOdrum
Wx_compressor_interstage_cooler_compressor_house
Wx_compressor_interstage_cooler_cooling_tower
Wx_compressor_interstage_cooler_coolingwater_pumphouse
Wx_compressor_interstage_cooler_cross_heat_exchanger
Wx_compressor_interstage_cooler_direct_contact_cooler
Wx_compressor_interstage_cooler_lean_amine_cooler
Wx_compressor_interstage_cooler_reclaimer
Wx_compressor_interstage_cooler_storage_tank
Wx_compressor_interstage_cooler_stripper
Wx_cooling_tower_TEG_dehydration
Wx_cooling_tower_absorber
Wx_cooling_tower_compressor_KOdrum
Wx_cooling_tower_compressor_house
Wx_cooling_tower_compressor_interstage_cooler
Wx_cooling_tower_coolingwater_pumphouse
Wx_cooling_tower_cross_heat_exchanger
Wx_cooling_tower_direct_contact_cooler
Wx_cooling_tower_lean_amine_cooler
Wx_cooling_tower_reclaimer
Wx_cooling_tower_storage_tank
Wx_cooling_tower_stripper
Wx_coolingwater_pumphouse_TEG_dehydration
Wx_coolingwater_pumphouse_absorber
Wx_coolingwater_pumphouse_compressor_KOdrum
Wx_coolingwater_pumphouse_compressor_house
Wx_coolingwater_pumphouse_compressor_interstage_cooler
Wx_coolingwater_pumphouse_cooling_tower
Wx_coolingwater_pumphouse_cross_heat_exchanger
Wx_coolingwater_pumphouse_direct_contact_cooler
Wx_coolingwater_pumphouse_lean_amine_cooler
Wx_coolingwater_pumphouse_reclaimer
Wx_coolingwater_pumphouse_storage_tank
Wx_coolingwater_pumphouse_stripper
Wx_cross_heat_exchanger_TEG_dehydration
Wx_cross_heat_exchanger_absorber
Wx_cross_heat_exchanger_compressor_KOdrum
Wx_cross_heat_exchanger_compressor_house
Wx_cross_heat_exchanger_compressor_interstage_cooler
Wx_cross_heat_exchanger_cooling_tower
Wx_cross_heat_exchanger_coolingwater_pumphouse
Wx_cross_heat_exchanger_direct_contact_cooler
Wx_cross_heat_exchanger_lean_amine_cooler
Wx_cross_heat_exchanger_reclaimer
Wx_cross_heat_exchanger_storage_tank
Wx_cross_heat_exchanger_stripper
Wx_direct_contact_cooler_TEG_dehydration
Wx_direct_contact_cooler_absorber
Wx_direct_contact_cooler_compressor_KOdrum
Wx_direct_contact_cooler_compressor_house
Wx_direct_contact_cooler_compressor_interstage_cooler
Wx_direct_contact_cooler_cooling_tower
Wx_direct_contact_cooler_coolingwater_pumphouse
Wx_direct_contact_cooler_cross_heat_exchanger
Wx_direct_contact_cooler_lean_amine_cooler
Wx_direct_contact_cooler_reclaimer
Wx_direct_contact_cooler_storage_tank
Wx_direct_contact_cooler_stripper
Wx_lean_amine_cooler_TEG_dehydration
Wx_lean_amine_cooler_absorber
Wx_lean_amine_cooler_compressor_KOdrum
Wx_lean_amine_cooler_compressor_house
Wx_lean_amine_cooler_compressor_interstage_cooler
Wx_lean_amine_cooler_cooling_tower
Wx_lean_amine_cooler_coolingwater_pumphouse
Wx_lean_amine_cooler_cross_heat_exchanger
Wx_lean_amine_cooler_direct_contact_cooler
Wx_lean_amine_cooler_reclaimer
Wx_lean_amine_cooler_storage_tank
Wx_lean_amine_cooler_stripper
Wx_reclaimer_TEG_dehydration
Wx_reclaimer_absorber
Wx_reclaimer_compressor_KOdrum
Wx_reclaimer_compressor_house
Wx_reclaimer_compressor_interstage_cooler
Wx_reclaimer_cooling_tower
Wx_reclaimer_coolingwater_pumphouse
Wx_reclaimer_cross_heat_exchanger
Wx_reclaimer_direct_contact_cooler
Wx_reclaimer_lean_amine_cooler
Wx_reclaimer_storage_tank
Wx_reclaimer_stripper
Wx_storage_tank_TEG_dehydration
Wx_storage_tank_absorber
Wx_storage_tank_compressor_KOdrum
Wx_storage_tank_compressor_house
Wx_storage_tank_compressor_interstage_cooler
Wx_storage_tank_cooling_tower
Wx_storage_tank_coolingwater_pumphouse
Wx_storage_tank_cross_heat_exchanger
Wx_storage_tank_direct_contact_cooler
Wx_storage_tank_lean_amine_cooler
Wx_storage_tank_reclaimer
Wx_storage_tank_stripper
Wx_stripper_TEG_dehydration
Wx_stripper_absorber
Wx_stripper_compressor_KOdrum
Wx_stripper_compressor_house
Wx_stripper_compressor_interstage_cooler
Wx_stripper_cooling_tower
Wx_stripper_coolingwater_pumphouse
Wx_stripper_cross_heat_exchanger
Wx_stripper_direct_contact_cooler
Wx_stripper_lean_amine_cooler
Wx_stripper_reclaimer
Wx_stripper_storage_tank
Wy_TEG_dehydration_absorber
Wy_TEG_dehydration_compressor_KOdrum
Wy_TEG_dehydration_compressor_house
Wy_TEG_dehydration_compressor_interstage_cooler
Wy_TEG_dehydration_cooling_tower
Wy_TEG_dehydration_coolingwater_pumphouse
Wy_TEG_dehydration_cross_heat_exchanger
Wy_TEG_dehydration_direct_contact_cooler
Wy_TEG_dehydration_lean_amine_cooler
Wy_TEG_dehydration_reclaimer
Wy_TEG_dehydration_storage_tank
Wy_TEG_dehydration_stripper
Wy_absorber_TEG_dehydration
Wy_absorber_compressor_KOdrum
Wy_absorber_compressor_house
Wy_absorber_compressor_interstage_cooler
Wy_absorber_cooling_tower
Wy_absorber_coolingwater_pumphouse
Wy_absorber_cross_heat_exchanger
Wy_absorber_direct_contact_cooler
Wy_absorber_lean_amine_cooler
Wy_absorber_reclaimer
Wy_absorber_storage_tank
Wy_absorber_stripper
Wy_compressor_KOdrum_TEG_dehydration
Wy_compressor_KOdrum_absorber
Wy_compressor_KOdrum_compressor_house
Wy_compressor_KOdrum_compressor_interstage_cooler
Wy_compressor_KOdrum_cooling_tower
Wy_compressor_KOdrum_coolingwater_pumphouse
Wy_compressor_KOdrum_cross_heat_exchanger
Wy_compressor_KOdrum_direct_contact_cooler
Wy_compressor_KOdrum_lean_amine_cooler
Wy_compressor_KOdrum_reclaimer
Wy_compressor_KOdrum_storage_tank
Wy_compressor_KOdrum_stripper
Wy_compressor_house_TEG_dehydration
Wy_compressor_house_absorber
Wy_compressor_house_compressor_KOdrum
Wy_compressor_house_compressor_interstage_cooler
Wy_compressor_house_cooling_tower
Wy_compressor_house_coolingwater_pumphouse
Wy_compressor_house_cross_heat_exchanger
Wy_compressor_house_direct_contact_cooler
Wy_compressor_house_lean_amine_cooler
Wy_compressor_house_reclaimer
Wy_compressor_house_storage_tank
Wy_compressor_house_stripper
Wy_compressor_interstage_cooler_TEG_dehydration
Wy_compressor_interstage_cooler_absorber
Wy_compressor_interstage_cooler_compressor_KOdrum
Wy_compressor_interstage_cooler_compressor_house
Wy_compressor_interstage_cooler_cooling_tower
Wy_compressor_interstage_cooler_coolingwater_pumphouse
Wy_compressor_interstage_cooler_cross_heat_exchanger
Wy_compressor_interstage_cooler_direct_contact_cooler
Wy_compressor_interstage_cooler_lean_amine_cooler
Wy_compressor_interstage_cooler_reclaimer
Wy_compressor_interstage_cooler_storage_tank
Wy_compressor_interstage_cooler_stripper
Wy_cooling_tower_TEG_dehydration
Wy_cooling_tower_absorber
Wy_cooling_tower_compressor_KOdrum
Wy_cooling_tower_compressor_house
Wy_cooling_tower_compressor_interstage_cooler
Wy_cooling_tower_coolingwater_pumphouse
Wy_cooling_tower_cross_heat_exchanger
Wy_cooling_tower_direct_contact_cooler
Wy_cooling_tower_lean_amine_cooler
Wy_cooling_tower_reclaimer
Wy_cooling_tower_storage_tank
Wy_cooling_tower_stripper
Wy_coolingwater_pumphouse_TEG_dehydration
Wy_coolingwater_pumphouse_absorber
Wy_coolingwater_pumphouse_compressor_KOdrum
Wy_coolingwater_pumphouse_compressor_house
Wy_coolingwater_pumphouse_compressor_interstage_cooler
Wy_coolingwater_pumphouse_cooling_tower
Wy_coolingwater_pumphouse_cross_heat_exchanger
Wy_coolingwater_pumphouse_direct_contact_cooler
Wy_coolingwater_pumphouse_lean_amine_cooler
Wy_coolingwater_pumphouse_reclaimer
Wy_coolingwater_pumphouse_storage_tank
Wy_coolingwater_pumphouse_stripper
Wy_cross_heat_exchanger_TEG_dehydration
Wy_cross_heat_exchanger_absorber
Wy_cross_heat_exchanger_compressor_KOdrum
Wy_cross_heat_exchanger_compressor_house
Wy_cross_heat_exchanger_compressor_interstage_cooler
Wy_cross_heat_exchanger_cooling_tower
Wy_cross_heat_exchanger_coolingwater_pumphouse
Wy_cross_heat_exchanger_direct_contact_cooler
Wy_cross_heat_exchanger_lean_amine_cooler
Wy_cross_heat_exchanger_reclaimer
Wy_cross_heat_exchanger_storage_tank
Wy_cross_heat_exchanger_stripper
Wy_direct_contact_cooler_TEG_dehydration
Wy_direct_contact_cooler_absorber
Wy_direct_contact_cooler_compressor_KOdrum
Wy_direct_contact_cooler_compressor_house
Wy_direct_contact_cooler_compressor_interstage_cooler
Wy_direct_contact_cooler_cooling_tower
Wy_direct_contact_cooler_coolingwater_pumphouse
Wy_direct_contact_cooler_cross_heat_exchanger
Wy_direct_contact_cooler_lean_amine_cooler
Wy_direct_contact_cooler_reclaimer
Wy_direct_contact_cooler_storage_tank
Wy_direct_contact_cooler_stripper
Wy_lean_amine_cooler_TEG_dehydration
Wy_lean_amine_cooler_absorber
Wy_lean_amine_cooler_compressor_KOdrum
Wy_lean_amine_cooler_compressor_house
Wy_lean_amine_cooler_compressor_interstage_cooler
Wy_lean_amine_cooler_cooling_tower
Wy_lean_amine_cooler_coolingwater_pumphouse
Wy_lean_amine_cooler_cross_heat_exchanger
Wy_lean_amine_cooler_direct_contact_cooler
Wy_lean_amine_cooler_reclaimer
Wy_lean_amine_cooler_storage_tank
Wy_lean_amine_cooler_stripper
Wy_reclaimer_TEG_dehydration
Wy_reclaimer_absorber
Wy_reclaimer_compressor_KOdrum
Wy_reclaimer_compressor_house
Wy_reclaimer_compressor_interstage_cooler
Wy_reclaimer_cooling_tower
Wy_reclaimer_coolingwater_pumphouse
Wy_reclaimer_cross_heat_exchanger
Wy_reclaimer_direct_contact_cooler
Wy_reclaimer_lean_amine_cooler
Wy_reclaimer_storage_tank
Wy_reclaimer_stripper
Wy_storage_tank_TEG_dehydration
Wy_storage_tank_absorber
Wy_storage_tank_compressor_KOdrum
Wy_storage_tank_compressor_house
Wy_storage_tank_compressor_interstage_cooler
Wy_storage_tank_cooling_tower
Wy_storage_tank_coolingwater_pumphouse
Wy_storage_tank_cross_heat_exchanger
Wy_storage_tank_direct_contact_cooler
Wy_storage_tank_lean_amine_cooler
Wy_storage_tank_reclaimer
Wy_storage_tank_stripper
Wy_stripper_TEG_dehydration
Wy_stripper_absorber
Wy_stripper_compressor_KOdrum
Wy_stripper_compressor_house
Wy_stripper_compressor_interstage_cooler
Wy_stripper_cooling_tower
Wy_stripper_coolingwater_pumphouse
Wy_stripper_cross_heat_exchanger
Wy_stripper_direct_contact_cooler
Wy_stripper_lean_amine_cooler
Wy_stripper_reclaimer
Wy_stripper_storage_tank
End
