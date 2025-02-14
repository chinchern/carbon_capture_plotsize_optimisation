\* Layout_Problem_Model *\
Minimize
OBJ: SumCD + TLC
Subject To
_C1: 0 O_ct_1 + l_ct_1 = 17
_C10: d_ct_3 + l_ct_3 = 34
_C100: - L_ct_4_ct_1 + R_ct_1_ct_4 = 0
_C101: A_ct_1_ct_4 - B_ct_4_ct_1 = 0
_C102: - A_ct_4_ct_1 + B_ct_1_ct_4 = 0
_C103: Wx_ct_1_ct_4 + Wx_ct_4_ct_1 = 1
_C104: Wy_ct_1_ct_4 + Wy_ct_4_ct_1 = 1
_C105: - L_ct_2_ct_4 + R_ct_2_ct_4 - X_ct_2 + X_ct_4 = 0
_C106: A_ct_2_ct_4 - B_ct_2_ct_4 - Y_ct_2 + Y_ct_4 = 0
_C107: R_ct_2_ct_4 - 1000 Wx_ct_2_ct_4 <= 0
_C108: L_ct_2_ct_4 + 1000 Wx_ct_2_ct_4 <= 1000
_C109: A_ct_2_ct_4 - 1000 Wy_ct_2_ct_4 <= 0
_C11: X_ct_3 - 0.5 l_ct_3 >= 0
_C110: B_ct_2_ct_4 + 1000 Wy_ct_2_ct_4 <= 1000
_C111: - A_ct_2_ct_4 - B_ct_2_ct_4 + D_ct_2_ct_4 - L_ct_2_ct_4 - R_ct_2_ct_4
 = 0
_C112: D_ct_2_ct_4 - D_ct_4_ct_2 = 0
_C113: 1000 E1_ct_2_ct_4 + 1000 E2_ct_2_ct_4 + X_ct_2 - X_ct_4 - 0.5 l_ct_2
 - 0.5 l_ct_4 >= 0
_C114: - 1000 E1_ct_2_ct_4 + 1000 E2_ct_2_ct_4 - X_ct_2 + X_ct_4 - 0.5 l_ct_2
 - 0.5 l_ct_4 >= -1000
_C115: 1000 E1_ct_2_ct_4 - 1000 E2_ct_2_ct_4 + Y_ct_2 - Y_ct_4 - 0.5 d_ct_2
 - 0.5 d_ct_4 >= -1000
_C116: - 1000 E1_ct_2_ct_4 - 1000 E2_ct_2_ct_4 - Y_ct_2 + Y_ct_4 - 0.5 d_ct_2
 - 0.5 d_ct_4 >= -2000
_C117: L_ct_2_ct_4 - R_ct_4_ct_2 = 0
_C118: - L_ct_4_ct_2 + R_ct_2_ct_4 = 0
_C119: A_ct_2_ct_4 - B_ct_4_ct_2 = 0
_C12: Y_ct_3 - 0.5 d_ct_3 >= 0
_C120: - A_ct_4_ct_2 + B_ct_2_ct_4 = 0
_C121: Wx_ct_2_ct_4 + Wx_ct_4_ct_2 = 1
_C122: Wy_ct_2_ct_4 + Wy_ct_4_ct_2 = 1
_C123: - L_ct_3_ct_4 + R_ct_3_ct_4 - X_ct_3 + X_ct_4 = 0
_C124: A_ct_3_ct_4 - B_ct_3_ct_4 - Y_ct_3 + Y_ct_4 = 0
_C125: R_ct_3_ct_4 - 1000 Wx_ct_3_ct_4 <= 0
_C126: L_ct_3_ct_4 + 1000 Wx_ct_3_ct_4 <= 1000
_C127: A_ct_3_ct_4 - 1000 Wy_ct_3_ct_4 <= 0
_C128: B_ct_3_ct_4 + 1000 Wy_ct_3_ct_4 <= 1000
_C129: - A_ct_3_ct_4 - B_ct_3_ct_4 + D_ct_3_ct_4 - L_ct_3_ct_4 - R_ct_3_ct_4
 = 0
_C13: 0 O_ct_4 + l_ct_4 = 17
_C130: D_ct_3_ct_4 - D_ct_4_ct_3 = 0
_C131: 1000 E1_ct_3_ct_4 + 1000 E2_ct_3_ct_4 + X_ct_3 - X_ct_4 - 0.5 l_ct_3
 - 0.5 l_ct_4 >= 0
_C132: - 1000 E1_ct_3_ct_4 + 1000 E2_ct_3_ct_4 - X_ct_3 + X_ct_4 - 0.5 l_ct_3
 - 0.5 l_ct_4 >= -1000
_C133: 1000 E1_ct_3_ct_4 - 1000 E2_ct_3_ct_4 + Y_ct_3 - Y_ct_4 - 0.5 d_ct_3
 - 0.5 d_ct_4 >= -1000
_C134: - 1000 E1_ct_3_ct_4 - 1000 E2_ct_3_ct_4 - Y_ct_3 + Y_ct_4 - 0.5 d_ct_3
 - 0.5 d_ct_4 >= -2000
_C135: L_ct_3_ct_4 - R_ct_4_ct_3 = 0
_C136: - L_ct_4_ct_3 + R_ct_3_ct_4 = 0
_C137: A_ct_3_ct_4 - B_ct_4_ct_3 = 0
_C138: - A_ct_4_ct_3 + B_ct_3_ct_4 = 0
_C139: Wx_ct_3_ct_4 + Wx_ct_4_ct_3 = 1
_C14: d_ct_4 + l_ct_4 = 34
_C140: Wy_ct_3_ct_4 + Wy_ct_4_ct_3 = 1
_C141: - L_ct_1_ct_5 + R_ct_1_ct_5 - X_ct_1 + X_ct_5 = 0
_C142: A_ct_1_ct_5 - B_ct_1_ct_5 - Y_ct_1 + Y_ct_5 = 0
_C143: R_ct_1_ct_5 - 1000 Wx_ct_1_ct_5 <= 0
_C144: L_ct_1_ct_5 + 1000 Wx_ct_1_ct_5 <= 1000
_C145: A_ct_1_ct_5 - 1000 Wy_ct_1_ct_5 <= 0
_C146: B_ct_1_ct_5 + 1000 Wy_ct_1_ct_5 <= 1000
_C147: - A_ct_1_ct_5 - B_ct_1_ct_5 + D_ct_1_ct_5 - L_ct_1_ct_5 - R_ct_1_ct_5
 = 0
_C148: D_ct_1_ct_5 - D_ct_5_ct_1 = 0
_C149: 1000 E1_ct_1_ct_5 + 1000 E2_ct_1_ct_5 + X_ct_1 - X_ct_5 - 0.5 l_ct_1
 - 0.5 l_ct_5 >= 0
_C15: X_ct_4 - 0.5 l_ct_4 >= 0
_C150: - 1000 E1_ct_1_ct_5 + 1000 E2_ct_1_ct_5 - X_ct_1 + X_ct_5 - 0.5 l_ct_1
 - 0.5 l_ct_5 >= -1000
_C151: 1000 E1_ct_1_ct_5 - 1000 E2_ct_1_ct_5 + Y_ct_1 - Y_ct_5 - 0.5 d_ct_1
 - 0.5 d_ct_5 >= -1000
_C152: - 1000 E1_ct_1_ct_5 - 1000 E2_ct_1_ct_5 - Y_ct_1 + Y_ct_5 - 0.5 d_ct_1
 - 0.5 d_ct_5 >= -2000
_C153: L_ct_1_ct_5 - R_ct_5_ct_1 = 0
_C154: - L_ct_5_ct_1 + R_ct_1_ct_5 = 0
_C155: A_ct_1_ct_5 - B_ct_5_ct_1 = 0
_C156: - A_ct_5_ct_1 + B_ct_1_ct_5 = 0
_C157: Wx_ct_1_ct_5 + Wx_ct_5_ct_1 = 1
_C158: Wy_ct_1_ct_5 + Wy_ct_5_ct_1 = 1
_C159: - L_ct_2_ct_5 + R_ct_2_ct_5 - X_ct_2 + X_ct_5 = 0
_C16: Y_ct_4 - 0.5 d_ct_4 >= 0
_C160: A_ct_2_ct_5 - B_ct_2_ct_5 - Y_ct_2 + Y_ct_5 = 0
_C161: R_ct_2_ct_5 - 1000 Wx_ct_2_ct_5 <= 0
_C162: L_ct_2_ct_5 + 1000 Wx_ct_2_ct_5 <= 1000
_C163: A_ct_2_ct_5 - 1000 Wy_ct_2_ct_5 <= 0
_C164: B_ct_2_ct_5 + 1000 Wy_ct_2_ct_5 <= 1000
_C165: - A_ct_2_ct_5 - B_ct_2_ct_5 + D_ct_2_ct_5 - L_ct_2_ct_5 - R_ct_2_ct_5
 = 0
_C166: D_ct_2_ct_5 - D_ct_5_ct_2 = 0
_C167: 1000 E1_ct_2_ct_5 + 1000 E2_ct_2_ct_5 + X_ct_2 - X_ct_5 - 0.5 l_ct_2
 - 0.5 l_ct_5 >= 0
_C168: - 1000 E1_ct_2_ct_5 + 1000 E2_ct_2_ct_5 - X_ct_2 + X_ct_5 - 0.5 l_ct_2
 - 0.5 l_ct_5 >= -1000
_C169: 1000 E1_ct_2_ct_5 - 1000 E2_ct_2_ct_5 + Y_ct_2 - Y_ct_5 - 0.5 d_ct_2
 - 0.5 d_ct_5 >= -1000
_C17: 0 O_ct_5 + l_ct_5 = 17
_C170: - 1000 E1_ct_2_ct_5 - 1000 E2_ct_2_ct_5 - Y_ct_2 + Y_ct_5 - 0.5 d_ct_2
 - 0.5 d_ct_5 >= -2000
_C171: L_ct_2_ct_5 - R_ct_5_ct_2 = 0
_C172: - L_ct_5_ct_2 + R_ct_2_ct_5 = 0
_C173: A_ct_2_ct_5 - B_ct_5_ct_2 = 0
_C174: - A_ct_5_ct_2 + B_ct_2_ct_5 = 0
_C175: Wx_ct_2_ct_5 + Wx_ct_5_ct_2 = 1
_C176: Wy_ct_2_ct_5 + Wy_ct_5_ct_2 = 1
_C177: - L_ct_3_ct_5 + R_ct_3_ct_5 - X_ct_3 + X_ct_5 = 0
_C178: A_ct_3_ct_5 - B_ct_3_ct_5 - Y_ct_3 + Y_ct_5 = 0
_C179: R_ct_3_ct_5 - 1000 Wx_ct_3_ct_5 <= 0
_C18: d_ct_5 + l_ct_5 = 34
_C180: L_ct_3_ct_5 + 1000 Wx_ct_3_ct_5 <= 1000
_C181: A_ct_3_ct_5 - 1000 Wy_ct_3_ct_5 <= 0
_C182: B_ct_3_ct_5 + 1000 Wy_ct_3_ct_5 <= 1000
_C183: - A_ct_3_ct_5 - B_ct_3_ct_5 + D_ct_3_ct_5 - L_ct_3_ct_5 - R_ct_3_ct_5
 = 0
_C184: D_ct_3_ct_5 - D_ct_5_ct_3 = 0
_C185: 1000 E1_ct_3_ct_5 + 1000 E2_ct_3_ct_5 + X_ct_3 - X_ct_5 - 0.5 l_ct_3
 - 0.5 l_ct_5 >= 0
_C186: - 1000 E1_ct_3_ct_5 + 1000 E2_ct_3_ct_5 - X_ct_3 + X_ct_5 - 0.5 l_ct_3
 - 0.5 l_ct_5 >= -1000
_C187: 1000 E1_ct_3_ct_5 - 1000 E2_ct_3_ct_5 + Y_ct_3 - Y_ct_5 - 0.5 d_ct_3
 - 0.5 d_ct_5 >= -1000
_C188: - 1000 E1_ct_3_ct_5 - 1000 E2_ct_3_ct_5 - Y_ct_3 + Y_ct_5 - 0.5 d_ct_3
 - 0.5 d_ct_5 >= -2000
_C189: L_ct_3_ct_5 - R_ct_5_ct_3 = 0
_C19: X_ct_5 - 0.5 l_ct_5 >= 0
_C190: - L_ct_5_ct_3 + R_ct_3_ct_5 = 0
_C191: A_ct_3_ct_5 - B_ct_5_ct_3 = 0
_C192: - A_ct_5_ct_3 + B_ct_3_ct_5 = 0
_C193: Wx_ct_3_ct_5 + Wx_ct_5_ct_3 = 1
_C194: Wy_ct_3_ct_5 + Wy_ct_5_ct_3 = 1
_C195: - L_ct_4_ct_5 + R_ct_4_ct_5 - X_ct_4 + X_ct_5 = 0
_C196: A_ct_4_ct_5 - B_ct_4_ct_5 - Y_ct_4 + Y_ct_5 = 0
_C197: R_ct_4_ct_5 - 1000 Wx_ct_4_ct_5 <= 0
_C198: L_ct_4_ct_5 + 1000 Wx_ct_4_ct_5 <= 1000
_C199: A_ct_4_ct_5 - 1000 Wy_ct_4_ct_5 <= 0
_C2: d_ct_1 + l_ct_1 = 34
_C20: Y_ct_5 - 0.5 d_ct_5 >= 0
_C200: B_ct_4_ct_5 + 1000 Wy_ct_4_ct_5 <= 1000
_C201: - A_ct_4_ct_5 - B_ct_4_ct_5 + D_ct_4_ct_5 - L_ct_4_ct_5 - R_ct_4_ct_5
 = 0
_C202: D_ct_4_ct_5 - D_ct_5_ct_4 = 0
_C203: 1000 E1_ct_4_ct_5 + 1000 E2_ct_4_ct_5 + X_ct_4 - X_ct_5 - 0.5 l_ct_4
 - 0.5 l_ct_5 >= 0
_C204: - 1000 E1_ct_4_ct_5 + 1000 E2_ct_4_ct_5 - X_ct_4 + X_ct_5 - 0.5 l_ct_4
 - 0.5 l_ct_5 >= -1000
_C205: 1000 E1_ct_4_ct_5 - 1000 E2_ct_4_ct_5 + Y_ct_4 - Y_ct_5 - 0.5 d_ct_4
 - 0.5 d_ct_5 >= -1000
_C206: - 1000 E1_ct_4_ct_5 - 1000 E2_ct_4_ct_5 - Y_ct_4 + Y_ct_5 - 0.5 d_ct_4
 - 0.5 d_ct_5 >= -2000
_C207: L_ct_4_ct_5 - R_ct_5_ct_4 = 0
_C208: - L_ct_5_ct_4 + R_ct_4_ct_5 = 0
_C209: A_ct_4_ct_5 - B_ct_5_ct_4 = 0
_C21: 0 O_ct_6 + l_ct_6 = 17
_C210: - A_ct_5_ct_4 + B_ct_4_ct_5 = 0
_C211: Wx_ct_4_ct_5 + Wx_ct_5_ct_4 = 1
_C212: Wy_ct_4_ct_5 + Wy_ct_5_ct_4 = 1
_C213: - L_ct_1_ct_6 + R_ct_1_ct_6 - X_ct_1 + X_ct_6 = 0
_C214: A_ct_1_ct_6 - B_ct_1_ct_6 - Y_ct_1 + Y_ct_6 = 0
_C215: R_ct_1_ct_6 - 1000 Wx_ct_1_ct_6 <= 0
_C216: L_ct_1_ct_6 + 1000 Wx_ct_1_ct_6 <= 1000
_C217: A_ct_1_ct_6 - 1000 Wy_ct_1_ct_6 <= 0
_C218: B_ct_1_ct_6 + 1000 Wy_ct_1_ct_6 <= 1000
_C219: - A_ct_1_ct_6 - B_ct_1_ct_6 + D_ct_1_ct_6 - L_ct_1_ct_6 - R_ct_1_ct_6
 = 0
_C22: d_ct_6 + l_ct_6 = 34
_C220: D_ct_1_ct_6 - D_ct_6_ct_1 = 0
_C221: 1000 E1_ct_1_ct_6 + 1000 E2_ct_1_ct_6 + X_ct_1 - X_ct_6 - 0.5 l_ct_1
 - 0.5 l_ct_6 >= 0
_C222: - 1000 E1_ct_1_ct_6 + 1000 E2_ct_1_ct_6 - X_ct_1 + X_ct_6 - 0.5 l_ct_1
 - 0.5 l_ct_6 >= -1000
_C223: 1000 E1_ct_1_ct_6 - 1000 E2_ct_1_ct_6 + Y_ct_1 - Y_ct_6 - 0.5 d_ct_1
 - 0.5 d_ct_6 >= -1000
_C224: - 1000 E1_ct_1_ct_6 - 1000 E2_ct_1_ct_6 - Y_ct_1 + Y_ct_6 - 0.5 d_ct_1
 - 0.5 d_ct_6 >= -2000
_C225: L_ct_1_ct_6 - R_ct_6_ct_1 = 0
_C226: - L_ct_6_ct_1 + R_ct_1_ct_6 = 0
_C227: A_ct_1_ct_6 - B_ct_6_ct_1 = 0
_C228: - A_ct_6_ct_1 + B_ct_1_ct_6 = 0
_C229: Wx_ct_1_ct_6 + Wx_ct_6_ct_1 = 1
_C23: X_ct_6 - 0.5 l_ct_6 >= 0
_C230: Wy_ct_1_ct_6 + Wy_ct_6_ct_1 = 1
_C231: - L_ct_2_ct_6 + R_ct_2_ct_6 - X_ct_2 + X_ct_6 = 0
_C232: A_ct_2_ct_6 - B_ct_2_ct_6 - Y_ct_2 + Y_ct_6 = 0
_C233: R_ct_2_ct_6 - 1000 Wx_ct_2_ct_6 <= 0
_C234: L_ct_2_ct_6 + 1000 Wx_ct_2_ct_6 <= 1000
_C235: A_ct_2_ct_6 - 1000 Wy_ct_2_ct_6 <= 0
_C236: B_ct_2_ct_6 + 1000 Wy_ct_2_ct_6 <= 1000
_C237: - A_ct_2_ct_6 - B_ct_2_ct_6 + D_ct_2_ct_6 - L_ct_2_ct_6 - R_ct_2_ct_6
 = 0
_C238: D_ct_2_ct_6 - D_ct_6_ct_2 = 0
_C239: 1000 E1_ct_2_ct_6 + 1000 E2_ct_2_ct_6 + X_ct_2 - X_ct_6 - 0.5 l_ct_2
 - 0.5 l_ct_6 >= 0
_C24: Y_ct_6 - 0.5 d_ct_6 >= 0
_C240: - 1000 E1_ct_2_ct_6 + 1000 E2_ct_2_ct_6 - X_ct_2 + X_ct_6 - 0.5 l_ct_2
 - 0.5 l_ct_6 >= -1000
_C241: 1000 E1_ct_2_ct_6 - 1000 E2_ct_2_ct_6 + Y_ct_2 - Y_ct_6 - 0.5 d_ct_2
 - 0.5 d_ct_6 >= -1000
_C242: - 1000 E1_ct_2_ct_6 - 1000 E2_ct_2_ct_6 - Y_ct_2 + Y_ct_6 - 0.5 d_ct_2
 - 0.5 d_ct_6 >= -2000
_C243: L_ct_2_ct_6 - R_ct_6_ct_2 = 0
_C244: - L_ct_6_ct_2 + R_ct_2_ct_6 = 0
_C245: A_ct_2_ct_6 - B_ct_6_ct_2 = 0
_C246: - A_ct_6_ct_2 + B_ct_2_ct_6 = 0
_C247: Wx_ct_2_ct_6 + Wx_ct_6_ct_2 = 1
_C248: Wy_ct_2_ct_6 + Wy_ct_6_ct_2 = 1
_C249: - L_ct_3_ct_6 + R_ct_3_ct_6 - X_ct_3 + X_ct_6 = 0
_C25: 0 O_ct_7 + l_ct_7 = 17
_C250: A_ct_3_ct_6 - B_ct_3_ct_6 - Y_ct_3 + Y_ct_6 = 0
_C251: R_ct_3_ct_6 - 1000 Wx_ct_3_ct_6 <= 0
_C252: L_ct_3_ct_6 + 1000 Wx_ct_3_ct_6 <= 1000
_C253: A_ct_3_ct_6 - 1000 Wy_ct_3_ct_6 <= 0
_C254: B_ct_3_ct_6 + 1000 Wy_ct_3_ct_6 <= 1000
_C255: - A_ct_3_ct_6 - B_ct_3_ct_6 + D_ct_3_ct_6 - L_ct_3_ct_6 - R_ct_3_ct_6
 = 0
_C256: D_ct_3_ct_6 - D_ct_6_ct_3 = 0
_C257: 1000 E1_ct_3_ct_6 + 1000 E2_ct_3_ct_6 + X_ct_3 - X_ct_6 - 0.5 l_ct_3
 - 0.5 l_ct_6 >= 0
_C258: - 1000 E1_ct_3_ct_6 + 1000 E2_ct_3_ct_6 - X_ct_3 + X_ct_6 - 0.5 l_ct_3
 - 0.5 l_ct_6 >= -1000
_C259: 1000 E1_ct_3_ct_6 - 1000 E2_ct_3_ct_6 + Y_ct_3 - Y_ct_6 - 0.5 d_ct_3
 - 0.5 d_ct_6 >= -1000
_C26: d_ct_7 + l_ct_7 = 34
_C260: - 1000 E1_ct_3_ct_6 - 1000 E2_ct_3_ct_6 - Y_ct_3 + Y_ct_6 - 0.5 d_ct_3
 - 0.5 d_ct_6 >= -2000
_C261: L_ct_3_ct_6 - R_ct_6_ct_3 = 0
_C262: - L_ct_6_ct_3 + R_ct_3_ct_6 = 0
_C263: A_ct_3_ct_6 - B_ct_6_ct_3 = 0
_C264: - A_ct_6_ct_3 + B_ct_3_ct_6 = 0
_C265: Wx_ct_3_ct_6 + Wx_ct_6_ct_3 = 1
_C266: Wy_ct_3_ct_6 + Wy_ct_6_ct_3 = 1
_C267: - L_ct_4_ct_6 + R_ct_4_ct_6 - X_ct_4 + X_ct_6 = 0
_C268: A_ct_4_ct_6 - B_ct_4_ct_6 - Y_ct_4 + Y_ct_6 = 0
_C269: R_ct_4_ct_6 - 1000 Wx_ct_4_ct_6 <= 0
_C27: X_ct_7 - 0.5 l_ct_7 >= 0
_C270: L_ct_4_ct_6 + 1000 Wx_ct_4_ct_6 <= 1000
_C271: A_ct_4_ct_6 - 1000 Wy_ct_4_ct_6 <= 0
_C272: B_ct_4_ct_6 + 1000 Wy_ct_4_ct_6 <= 1000
_C273: - A_ct_4_ct_6 - B_ct_4_ct_6 + D_ct_4_ct_6 - L_ct_4_ct_6 - R_ct_4_ct_6
 = 0
_C274: D_ct_4_ct_6 - D_ct_6_ct_4 = 0
_C275: 1000 E1_ct_4_ct_6 + 1000 E2_ct_4_ct_6 + X_ct_4 - X_ct_6 - 0.5 l_ct_4
 - 0.5 l_ct_6 >= 0
_C276: - 1000 E1_ct_4_ct_6 + 1000 E2_ct_4_ct_6 - X_ct_4 + X_ct_6 - 0.5 l_ct_4
 - 0.5 l_ct_6 >= -1000
_C277: 1000 E1_ct_4_ct_6 - 1000 E2_ct_4_ct_6 + Y_ct_4 - Y_ct_6 - 0.5 d_ct_4
 - 0.5 d_ct_6 >= -1000
_C278: - 1000 E1_ct_4_ct_6 - 1000 E2_ct_4_ct_6 - Y_ct_4 + Y_ct_6 - 0.5 d_ct_4
 - 0.5 d_ct_6 >= -2000
_C279: L_ct_4_ct_6 - R_ct_6_ct_4 = 0
_C28: Y_ct_7 - 0.5 d_ct_7 >= 0
_C280: - L_ct_6_ct_4 + R_ct_4_ct_6 = 0
_C281: A_ct_4_ct_6 - B_ct_6_ct_4 = 0
_C282: - A_ct_6_ct_4 + B_ct_4_ct_6 = 0
_C283: Wx_ct_4_ct_6 + Wx_ct_6_ct_4 = 1
_C284: Wy_ct_4_ct_6 + Wy_ct_6_ct_4 = 1
_C285: - L_ct_5_ct_6 + R_ct_5_ct_6 - X_ct_5 + X_ct_6 = 0
_C286: A_ct_5_ct_6 - B_ct_5_ct_6 - Y_ct_5 + Y_ct_6 = 0
_C287: R_ct_5_ct_6 - 1000 Wx_ct_5_ct_6 <= 0
_C288: L_ct_5_ct_6 + 1000 Wx_ct_5_ct_6 <= 1000
_C289: A_ct_5_ct_6 - 1000 Wy_ct_5_ct_6 <= 0
_C29: 0 O_ct_8 + l_ct_8 = 17
_C290: B_ct_5_ct_6 + 1000 Wy_ct_5_ct_6 <= 1000
_C291: - A_ct_5_ct_6 - B_ct_5_ct_6 + D_ct_5_ct_6 - L_ct_5_ct_6 - R_ct_5_ct_6
 = 0
_C292: D_ct_5_ct_6 - D_ct_6_ct_5 = 0
_C293: 1000 E1_ct_5_ct_6 + 1000 E2_ct_5_ct_6 + X_ct_5 - X_ct_6 - 0.5 l_ct_5
 - 0.5 l_ct_6 >= 0
_C294: - 1000 E1_ct_5_ct_6 + 1000 E2_ct_5_ct_6 - X_ct_5 + X_ct_6 - 0.5 l_ct_5
 - 0.5 l_ct_6 >= -1000
_C295: 1000 E1_ct_5_ct_6 - 1000 E2_ct_5_ct_6 + Y_ct_5 - Y_ct_6 - 0.5 d_ct_5
 - 0.5 d_ct_6 >= -1000
_C296: - 1000 E1_ct_5_ct_6 - 1000 E2_ct_5_ct_6 - Y_ct_5 + Y_ct_6 - 0.5 d_ct_5
 - 0.5 d_ct_6 >= -2000
_C297: L_ct_5_ct_6 - R_ct_6_ct_5 = 0
_C298: - L_ct_6_ct_5 + R_ct_5_ct_6 = 0
_C299: A_ct_5_ct_6 - B_ct_6_ct_5 = 0
_C3: X_ct_1 - 0.5 l_ct_1 >= 0
_C30: d_ct_8 + l_ct_8 = 34
_C300: - A_ct_6_ct_5 + B_ct_5_ct_6 = 0
_C301: Wx_ct_5_ct_6 + Wx_ct_6_ct_5 = 1
_C302: Wy_ct_5_ct_6 + Wy_ct_6_ct_5 = 1
_C303: - L_ct_1_ct_7 + R_ct_1_ct_7 - X_ct_1 + X_ct_7 = 0
_C304: A_ct_1_ct_7 - B_ct_1_ct_7 - Y_ct_1 + Y_ct_7 = 0
_C305: R_ct_1_ct_7 - 1000 Wx_ct_1_ct_7 <= 0
_C306: L_ct_1_ct_7 + 1000 Wx_ct_1_ct_7 <= 1000
_C307: A_ct_1_ct_7 - 1000 Wy_ct_1_ct_7 <= 0
_C308: B_ct_1_ct_7 + 1000 Wy_ct_1_ct_7 <= 1000
_C309: - A_ct_1_ct_7 - B_ct_1_ct_7 + D_ct_1_ct_7 - L_ct_1_ct_7 - R_ct_1_ct_7
 = 0
_C31: X_ct_8 - 0.5 l_ct_8 >= 0
_C310: D_ct_1_ct_7 - D_ct_7_ct_1 = 0
_C311: 1000 E1_ct_1_ct_7 + 1000 E2_ct_1_ct_7 + X_ct_1 - X_ct_7 - 0.5 l_ct_1
 - 0.5 l_ct_7 >= 0
_C312: - 1000 E1_ct_1_ct_7 + 1000 E2_ct_1_ct_7 - X_ct_1 + X_ct_7 - 0.5 l_ct_1
 - 0.5 l_ct_7 >= -1000
_C313: 1000 E1_ct_1_ct_7 - 1000 E2_ct_1_ct_7 + Y_ct_1 - Y_ct_7 - 0.5 d_ct_1
 - 0.5 d_ct_7 >= -1000
_C314: - 1000 E1_ct_1_ct_7 - 1000 E2_ct_1_ct_7 - Y_ct_1 + Y_ct_7 - 0.5 d_ct_1
 - 0.5 d_ct_7 >= -2000
_C315: L_ct_1_ct_7 - R_ct_7_ct_1 = 0
_C316: - L_ct_7_ct_1 + R_ct_1_ct_7 = 0
_C317: A_ct_1_ct_7 - B_ct_7_ct_1 = 0
_C318: - A_ct_7_ct_1 + B_ct_1_ct_7 = 0
_C319: Wx_ct_1_ct_7 + Wx_ct_7_ct_1 = 1
_C32: Y_ct_8 - 0.5 d_ct_8 >= 0
_C320: Wy_ct_1_ct_7 + Wy_ct_7_ct_1 = 1
_C321: - L_ct_2_ct_7 + R_ct_2_ct_7 - X_ct_2 + X_ct_7 = 0
_C322: A_ct_2_ct_7 - B_ct_2_ct_7 - Y_ct_2 + Y_ct_7 = 0
_C323: R_ct_2_ct_7 - 1000 Wx_ct_2_ct_7 <= 0
_C324: L_ct_2_ct_7 + 1000 Wx_ct_2_ct_7 <= 1000
_C325: A_ct_2_ct_7 - 1000 Wy_ct_2_ct_7 <= 0
_C326: B_ct_2_ct_7 + 1000 Wy_ct_2_ct_7 <= 1000
_C327: - A_ct_2_ct_7 - B_ct_2_ct_7 + D_ct_2_ct_7 - L_ct_2_ct_7 - R_ct_2_ct_7
 = 0
_C328: D_ct_2_ct_7 - D_ct_7_ct_2 = 0
_C329: 1000 E1_ct_2_ct_7 + 1000 E2_ct_2_ct_7 + X_ct_2 - X_ct_7 - 0.5 l_ct_2
 - 0.5 l_ct_7 >= 0
_C33: - L_ct_1_ct_2 + R_ct_1_ct_2 - X_ct_1 + X_ct_2 = 0
_C330: - 1000 E1_ct_2_ct_7 + 1000 E2_ct_2_ct_7 - X_ct_2 + X_ct_7 - 0.5 l_ct_2
 - 0.5 l_ct_7 >= -1000
_C331: 1000 E1_ct_2_ct_7 - 1000 E2_ct_2_ct_7 + Y_ct_2 - Y_ct_7 - 0.5 d_ct_2
 - 0.5 d_ct_7 >= -1000
_C332: - 1000 E1_ct_2_ct_7 - 1000 E2_ct_2_ct_7 - Y_ct_2 + Y_ct_7 - 0.5 d_ct_2
 - 0.5 d_ct_7 >= -2000
_C333: L_ct_2_ct_7 - R_ct_7_ct_2 = 0
_C334: - L_ct_7_ct_2 + R_ct_2_ct_7 = 0
_C335: A_ct_2_ct_7 - B_ct_7_ct_2 = 0
_C336: - A_ct_7_ct_2 + B_ct_2_ct_7 = 0
_C337: Wx_ct_2_ct_7 + Wx_ct_7_ct_2 = 1
_C338: Wy_ct_2_ct_7 + Wy_ct_7_ct_2 = 1
_C339: - L_ct_3_ct_7 + R_ct_3_ct_7 - X_ct_3 + X_ct_7 = 0
_C34: A_ct_1_ct_2 - B_ct_1_ct_2 - Y_ct_1 + Y_ct_2 = 0
_C340: A_ct_3_ct_7 - B_ct_3_ct_7 - Y_ct_3 + Y_ct_7 = 0
_C341: R_ct_3_ct_7 - 1000 Wx_ct_3_ct_7 <= 0
_C342: L_ct_3_ct_7 + 1000 Wx_ct_3_ct_7 <= 1000
_C343: A_ct_3_ct_7 - 1000 Wy_ct_3_ct_7 <= 0
_C344: B_ct_3_ct_7 + 1000 Wy_ct_3_ct_7 <= 1000
_C345: - A_ct_3_ct_7 - B_ct_3_ct_7 + D_ct_3_ct_7 - L_ct_3_ct_7 - R_ct_3_ct_7
 = 0
_C346: D_ct_3_ct_7 - D_ct_7_ct_3 = 0
_C347: 1000 E1_ct_3_ct_7 + 1000 E2_ct_3_ct_7 + X_ct_3 - X_ct_7 - 0.5 l_ct_3
 - 0.5 l_ct_7 >= 0
_C348: - 1000 E1_ct_3_ct_7 + 1000 E2_ct_3_ct_7 - X_ct_3 + X_ct_7 - 0.5 l_ct_3
 - 0.5 l_ct_7 >= -1000
_C349: 1000 E1_ct_3_ct_7 - 1000 E2_ct_3_ct_7 + Y_ct_3 - Y_ct_7 - 0.5 d_ct_3
 - 0.5 d_ct_7 >= -1000
_C35: R_ct_1_ct_2 - 1000 Wx_ct_1_ct_2 <= 0
_C350: - 1000 E1_ct_3_ct_7 - 1000 E2_ct_3_ct_7 - Y_ct_3 + Y_ct_7 - 0.5 d_ct_3
 - 0.5 d_ct_7 >= -2000
_C351: L_ct_3_ct_7 - R_ct_7_ct_3 = 0
_C352: - L_ct_7_ct_3 + R_ct_3_ct_7 = 0
_C353: A_ct_3_ct_7 - B_ct_7_ct_3 = 0
_C354: - A_ct_7_ct_3 + B_ct_3_ct_7 = 0
_C355: Wx_ct_3_ct_7 + Wx_ct_7_ct_3 = 1
_C356: Wy_ct_3_ct_7 + Wy_ct_7_ct_3 = 1
_C357: - L_ct_4_ct_7 + R_ct_4_ct_7 - X_ct_4 + X_ct_7 = 0
_C358: A_ct_4_ct_7 - B_ct_4_ct_7 - Y_ct_4 + Y_ct_7 = 0
_C359: R_ct_4_ct_7 - 1000 Wx_ct_4_ct_7 <= 0
_C36: L_ct_1_ct_2 + 1000 Wx_ct_1_ct_2 <= 1000
_C360: L_ct_4_ct_7 + 1000 Wx_ct_4_ct_7 <= 1000
_C361: A_ct_4_ct_7 - 1000 Wy_ct_4_ct_7 <= 0
_C362: B_ct_4_ct_7 + 1000 Wy_ct_4_ct_7 <= 1000
_C363: - A_ct_4_ct_7 - B_ct_4_ct_7 + D_ct_4_ct_7 - L_ct_4_ct_7 - R_ct_4_ct_7
 = 0
_C364: D_ct_4_ct_7 - D_ct_7_ct_4 = 0
_C365: 1000 E1_ct_4_ct_7 + 1000 E2_ct_4_ct_7 + X_ct_4 - X_ct_7 - 0.5 l_ct_4
 - 0.5 l_ct_7 >= 0
_C366: - 1000 E1_ct_4_ct_7 + 1000 E2_ct_4_ct_7 - X_ct_4 + X_ct_7 - 0.5 l_ct_4
 - 0.5 l_ct_7 >= -1000
_C367: 1000 E1_ct_4_ct_7 - 1000 E2_ct_4_ct_7 + Y_ct_4 - Y_ct_7 - 0.5 d_ct_4
 - 0.5 d_ct_7 >= -1000
_C368: - 1000 E1_ct_4_ct_7 - 1000 E2_ct_4_ct_7 - Y_ct_4 + Y_ct_7 - 0.5 d_ct_4
 - 0.5 d_ct_7 >= -2000
_C369: L_ct_4_ct_7 - R_ct_7_ct_4 = 0
_C37: A_ct_1_ct_2 - 1000 Wy_ct_1_ct_2 <= 0
_C370: - L_ct_7_ct_4 + R_ct_4_ct_7 = 0
_C371: A_ct_4_ct_7 - B_ct_7_ct_4 = 0
_C372: - A_ct_7_ct_4 + B_ct_4_ct_7 = 0
_C373: Wx_ct_4_ct_7 + Wx_ct_7_ct_4 = 1
_C374: Wy_ct_4_ct_7 + Wy_ct_7_ct_4 = 1
_C375: - L_ct_5_ct_7 + R_ct_5_ct_7 - X_ct_5 + X_ct_7 = 0
_C376: A_ct_5_ct_7 - B_ct_5_ct_7 - Y_ct_5 + Y_ct_7 = 0
_C377: R_ct_5_ct_7 - 1000 Wx_ct_5_ct_7 <= 0
_C378: L_ct_5_ct_7 + 1000 Wx_ct_5_ct_7 <= 1000
_C379: A_ct_5_ct_7 - 1000 Wy_ct_5_ct_7 <= 0
_C38: B_ct_1_ct_2 + 1000 Wy_ct_1_ct_2 <= 1000
_C380: B_ct_5_ct_7 + 1000 Wy_ct_5_ct_7 <= 1000
_C381: - A_ct_5_ct_7 - B_ct_5_ct_7 + D_ct_5_ct_7 - L_ct_5_ct_7 - R_ct_5_ct_7
 = 0
_C382: D_ct_5_ct_7 - D_ct_7_ct_5 = 0
_C383: 1000 E1_ct_5_ct_7 + 1000 E2_ct_5_ct_7 + X_ct_5 - X_ct_7 - 0.5 l_ct_5
 - 0.5 l_ct_7 >= 0
_C384: - 1000 E1_ct_5_ct_7 + 1000 E2_ct_5_ct_7 - X_ct_5 + X_ct_7 - 0.5 l_ct_5
 - 0.5 l_ct_7 >= -1000
_C385: 1000 E1_ct_5_ct_7 - 1000 E2_ct_5_ct_7 + Y_ct_5 - Y_ct_7 - 0.5 d_ct_5
 - 0.5 d_ct_7 >= -1000
_C386: - 1000 E1_ct_5_ct_7 - 1000 E2_ct_5_ct_7 - Y_ct_5 + Y_ct_7 - 0.5 d_ct_5
 - 0.5 d_ct_7 >= -2000
_C387: L_ct_5_ct_7 - R_ct_7_ct_5 = 0
_C388: - L_ct_7_ct_5 + R_ct_5_ct_7 = 0
_C389: A_ct_5_ct_7 - B_ct_7_ct_5 = 0
_C39: - A_ct_1_ct_2 - B_ct_1_ct_2 + D_ct_1_ct_2 - L_ct_1_ct_2 - R_ct_1_ct_2
 = 0
_C390: - A_ct_7_ct_5 + B_ct_5_ct_7 = 0
_C391: Wx_ct_5_ct_7 + Wx_ct_7_ct_5 = 1
_C392: Wy_ct_5_ct_7 + Wy_ct_7_ct_5 = 1
_C393: - L_ct_6_ct_7 + R_ct_6_ct_7 - X_ct_6 + X_ct_7 = 0
_C394: A_ct_6_ct_7 - B_ct_6_ct_7 - Y_ct_6 + Y_ct_7 = 0
_C395: R_ct_6_ct_7 - 1000 Wx_ct_6_ct_7 <= 0
_C396: L_ct_6_ct_7 + 1000 Wx_ct_6_ct_7 <= 1000
_C397: A_ct_6_ct_7 - 1000 Wy_ct_6_ct_7 <= 0
_C398: B_ct_6_ct_7 + 1000 Wy_ct_6_ct_7 <= 1000
_C399: - A_ct_6_ct_7 - B_ct_6_ct_7 + D_ct_6_ct_7 - L_ct_6_ct_7 - R_ct_6_ct_7
 = 0
_C4: Y_ct_1 - 0.5 d_ct_1 >= 0
_C40: D_ct_1_ct_2 - D_ct_2_ct_1 = 0
_C400: D_ct_6_ct_7 - D_ct_7_ct_6 = 0
_C401: 1000 E1_ct_6_ct_7 + 1000 E2_ct_6_ct_7 + X_ct_6 - X_ct_7 - 0.5 l_ct_6
 - 0.5 l_ct_7 >= 0
_C402: - 1000 E1_ct_6_ct_7 + 1000 E2_ct_6_ct_7 - X_ct_6 + X_ct_7 - 0.5 l_ct_6
 - 0.5 l_ct_7 >= -1000
_C403: 1000 E1_ct_6_ct_7 - 1000 E2_ct_6_ct_7 + Y_ct_6 - Y_ct_7 - 0.5 d_ct_6
 - 0.5 d_ct_7 >= -1000
_C404: - 1000 E1_ct_6_ct_7 - 1000 E2_ct_6_ct_7 - Y_ct_6 + Y_ct_7 - 0.5 d_ct_6
 - 0.5 d_ct_7 >= -2000
_C405: L_ct_6_ct_7 - R_ct_7_ct_6 = 0
_C406: - L_ct_7_ct_6 + R_ct_6_ct_7 = 0
_C407: A_ct_6_ct_7 - B_ct_7_ct_6 = 0
_C408: - A_ct_7_ct_6 + B_ct_6_ct_7 = 0
_C409: Wx_ct_6_ct_7 + Wx_ct_7_ct_6 = 1
_C41: 1000 E1_ct_1_ct_2 + 1000 E2_ct_1_ct_2 + X_ct_1 - X_ct_2 - 0.5 l_ct_1
 - 0.5 l_ct_2 >= 0
_C410: Wy_ct_6_ct_7 + Wy_ct_7_ct_6 = 1
_C411: - L_ct_1_ct_8 + R_ct_1_ct_8 - X_ct_1 + X_ct_8 = 0
_C412: A_ct_1_ct_8 - B_ct_1_ct_8 - Y_ct_1 + Y_ct_8 = 0
_C413: R_ct_1_ct_8 - 1000 Wx_ct_1_ct_8 <= 0
_C414: L_ct_1_ct_8 + 1000 Wx_ct_1_ct_8 <= 1000
_C415: A_ct_1_ct_8 - 1000 Wy_ct_1_ct_8 <= 0
_C416: B_ct_1_ct_8 + 1000 Wy_ct_1_ct_8 <= 1000
_C417: - A_ct_1_ct_8 - B_ct_1_ct_8 + D_ct_1_ct_8 - L_ct_1_ct_8 - R_ct_1_ct_8
 = 0
_C418: D_ct_1_ct_8 - D_ct_8_ct_1 = 0
_C419: 1000 E1_ct_1_ct_8 + 1000 E2_ct_1_ct_8 + X_ct_1 - X_ct_8 - 0.5 l_ct_1
 - 0.5 l_ct_8 >= 0
_C42: - 1000 E1_ct_1_ct_2 + 1000 E2_ct_1_ct_2 - X_ct_1 + X_ct_2 - 0.5 l_ct_1
 - 0.5 l_ct_2 >= -1000
_C420: - 1000 E1_ct_1_ct_8 + 1000 E2_ct_1_ct_8 - X_ct_1 + X_ct_8 - 0.5 l_ct_1
 - 0.5 l_ct_8 >= -1000
_C421: 1000 E1_ct_1_ct_8 - 1000 E2_ct_1_ct_8 + Y_ct_1 - Y_ct_8 - 0.5 d_ct_1
 - 0.5 d_ct_8 >= -1000
_C422: - 1000 E1_ct_1_ct_8 - 1000 E2_ct_1_ct_8 - Y_ct_1 + Y_ct_8 - 0.5 d_ct_1
 - 0.5 d_ct_8 >= -2000
_C423: L_ct_1_ct_8 - R_ct_8_ct_1 = 0
_C424: - L_ct_8_ct_1 + R_ct_1_ct_8 = 0
_C425: A_ct_1_ct_8 - B_ct_8_ct_1 = 0
_C426: - A_ct_8_ct_1 + B_ct_1_ct_8 = 0
_C427: Wx_ct_1_ct_8 + Wx_ct_8_ct_1 = 1
_C428: Wy_ct_1_ct_8 + Wy_ct_8_ct_1 = 1
_C429: - L_ct_2_ct_8 + R_ct_2_ct_8 - X_ct_2 + X_ct_8 = 0
_C43: 1000 E1_ct_1_ct_2 - 1000 E2_ct_1_ct_2 + Y_ct_1 - Y_ct_2 - 0.5 d_ct_1
 - 0.5 d_ct_2 >= -1000
_C430: A_ct_2_ct_8 - B_ct_2_ct_8 - Y_ct_2 + Y_ct_8 = 0
_C431: R_ct_2_ct_8 - 1000 Wx_ct_2_ct_8 <= 0
_C432: L_ct_2_ct_8 + 1000 Wx_ct_2_ct_8 <= 1000
_C433: A_ct_2_ct_8 - 1000 Wy_ct_2_ct_8 <= 0
_C434: B_ct_2_ct_8 + 1000 Wy_ct_2_ct_8 <= 1000
_C435: - A_ct_2_ct_8 - B_ct_2_ct_8 + D_ct_2_ct_8 - L_ct_2_ct_8 - R_ct_2_ct_8
 = 0
_C436: D_ct_2_ct_8 - D_ct_8_ct_2 = 0
_C437: 1000 E1_ct_2_ct_8 + 1000 E2_ct_2_ct_8 + X_ct_2 - X_ct_8 - 0.5 l_ct_2
 - 0.5 l_ct_8 >= 0
_C438: - 1000 E1_ct_2_ct_8 + 1000 E2_ct_2_ct_8 - X_ct_2 + X_ct_8 - 0.5 l_ct_2
 - 0.5 l_ct_8 >= -1000
_C439: 1000 E1_ct_2_ct_8 - 1000 E2_ct_2_ct_8 + Y_ct_2 - Y_ct_8 - 0.5 d_ct_2
 - 0.5 d_ct_8 >= -1000
_C44: - 1000 E1_ct_1_ct_2 - 1000 E2_ct_1_ct_2 - Y_ct_1 + Y_ct_2 - 0.5 d_ct_1
 - 0.5 d_ct_2 >= -2000
_C440: - 1000 E1_ct_2_ct_8 - 1000 E2_ct_2_ct_8 - Y_ct_2 + Y_ct_8 - 0.5 d_ct_2
 - 0.5 d_ct_8 >= -2000
_C441: L_ct_2_ct_8 - R_ct_8_ct_2 = 0
_C442: - L_ct_8_ct_2 + R_ct_2_ct_8 = 0
_C443: A_ct_2_ct_8 - B_ct_8_ct_2 = 0
_C444: - A_ct_8_ct_2 + B_ct_2_ct_8 = 0
_C445: Wx_ct_2_ct_8 + Wx_ct_8_ct_2 = 1
_C446: Wy_ct_2_ct_8 + Wy_ct_8_ct_2 = 1
_C447: - L_ct_3_ct_8 + R_ct_3_ct_8 - X_ct_3 + X_ct_8 = 0
_C448: A_ct_3_ct_8 - B_ct_3_ct_8 - Y_ct_3 + Y_ct_8 = 0
_C449: R_ct_3_ct_8 - 1000 Wx_ct_3_ct_8 <= 0
_C45: L_ct_1_ct_2 - R_ct_2_ct_1 = 0
_C450: L_ct_3_ct_8 + 1000 Wx_ct_3_ct_8 <= 1000
_C451: A_ct_3_ct_8 - 1000 Wy_ct_3_ct_8 <= 0
_C452: B_ct_3_ct_8 + 1000 Wy_ct_3_ct_8 <= 1000
_C453: - A_ct_3_ct_8 - B_ct_3_ct_8 + D_ct_3_ct_8 - L_ct_3_ct_8 - R_ct_3_ct_8
 = 0
_C454: D_ct_3_ct_8 - D_ct_8_ct_3 = 0
_C455: 1000 E1_ct_3_ct_8 + 1000 E2_ct_3_ct_8 + X_ct_3 - X_ct_8 - 0.5 l_ct_3
 - 0.5 l_ct_8 >= 0
_C456: - 1000 E1_ct_3_ct_8 + 1000 E2_ct_3_ct_8 - X_ct_3 + X_ct_8 - 0.5 l_ct_3
 - 0.5 l_ct_8 >= -1000
_C457: 1000 E1_ct_3_ct_8 - 1000 E2_ct_3_ct_8 + Y_ct_3 - Y_ct_8 - 0.5 d_ct_3
 - 0.5 d_ct_8 >= -1000
_C458: - 1000 E1_ct_3_ct_8 - 1000 E2_ct_3_ct_8 - Y_ct_3 + Y_ct_8 - 0.5 d_ct_3
 - 0.5 d_ct_8 >= -2000
_C459: L_ct_3_ct_8 - R_ct_8_ct_3 = 0
_C46: - L_ct_2_ct_1 + R_ct_1_ct_2 = 0
_C460: - L_ct_8_ct_3 + R_ct_3_ct_8 = 0
_C461: A_ct_3_ct_8 - B_ct_8_ct_3 = 0
_C462: - A_ct_8_ct_3 + B_ct_3_ct_8 = 0
_C463: Wx_ct_3_ct_8 + Wx_ct_8_ct_3 = 1
_C464: Wy_ct_3_ct_8 + Wy_ct_8_ct_3 = 1
_C465: - L_ct_4_ct_8 + R_ct_4_ct_8 - X_ct_4 + X_ct_8 = 0
_C466: A_ct_4_ct_8 - B_ct_4_ct_8 - Y_ct_4 + Y_ct_8 = 0
_C467: R_ct_4_ct_8 - 1000 Wx_ct_4_ct_8 <= 0
_C468: L_ct_4_ct_8 + 1000 Wx_ct_4_ct_8 <= 1000
_C469: A_ct_4_ct_8 - 1000 Wy_ct_4_ct_8 <= 0
_C47: A_ct_1_ct_2 - B_ct_2_ct_1 = 0
_C470: B_ct_4_ct_8 + 1000 Wy_ct_4_ct_8 <= 1000
_C471: - A_ct_4_ct_8 - B_ct_4_ct_8 + D_ct_4_ct_8 - L_ct_4_ct_8 - R_ct_4_ct_8
 = 0
_C472: D_ct_4_ct_8 - D_ct_8_ct_4 = 0
_C473: 1000 E1_ct_4_ct_8 + 1000 E2_ct_4_ct_8 + X_ct_4 - X_ct_8 - 0.5 l_ct_4
 - 0.5 l_ct_8 >= 0
_C474: - 1000 E1_ct_4_ct_8 + 1000 E2_ct_4_ct_8 - X_ct_4 + X_ct_8 - 0.5 l_ct_4
 - 0.5 l_ct_8 >= -1000
_C475: 1000 E1_ct_4_ct_8 - 1000 E2_ct_4_ct_8 + Y_ct_4 - Y_ct_8 - 0.5 d_ct_4
 - 0.5 d_ct_8 >= -1000
_C476: - 1000 E1_ct_4_ct_8 - 1000 E2_ct_4_ct_8 - Y_ct_4 + Y_ct_8 - 0.5 d_ct_4
 - 0.5 d_ct_8 >= -2000
_C477: L_ct_4_ct_8 - R_ct_8_ct_4 = 0
_C478: - L_ct_8_ct_4 + R_ct_4_ct_8 = 0
_C479: A_ct_4_ct_8 - B_ct_8_ct_4 = 0
_C48: - A_ct_2_ct_1 + B_ct_1_ct_2 = 0
_C480: - A_ct_8_ct_4 + B_ct_4_ct_8 = 0
_C481: Wx_ct_4_ct_8 + Wx_ct_8_ct_4 = 1
_C482: Wy_ct_4_ct_8 + Wy_ct_8_ct_4 = 1
_C483: - L_ct_5_ct_8 + R_ct_5_ct_8 - X_ct_5 + X_ct_8 = 0
_C484: A_ct_5_ct_8 - B_ct_5_ct_8 - Y_ct_5 + Y_ct_8 = 0
_C485: R_ct_5_ct_8 - 1000 Wx_ct_5_ct_8 <= 0
_C486: L_ct_5_ct_8 + 1000 Wx_ct_5_ct_8 <= 1000
_C487: A_ct_5_ct_8 - 1000 Wy_ct_5_ct_8 <= 0
_C488: B_ct_5_ct_8 + 1000 Wy_ct_5_ct_8 <= 1000
_C489: - A_ct_5_ct_8 - B_ct_5_ct_8 + D_ct_5_ct_8 - L_ct_5_ct_8 - R_ct_5_ct_8
 = 0
_C49: Wx_ct_1_ct_2 + Wx_ct_2_ct_1 = 1
_C490: D_ct_5_ct_8 - D_ct_8_ct_5 = 0
_C491: 1000 E1_ct_5_ct_8 + 1000 E2_ct_5_ct_8 + X_ct_5 - X_ct_8 - 0.5 l_ct_5
 - 0.5 l_ct_8 >= 0
_C492: - 1000 E1_ct_5_ct_8 + 1000 E2_ct_5_ct_8 - X_ct_5 + X_ct_8 - 0.5 l_ct_5
 - 0.5 l_ct_8 >= -1000
_C493: 1000 E1_ct_5_ct_8 - 1000 E2_ct_5_ct_8 + Y_ct_5 - Y_ct_8 - 0.5 d_ct_5
 - 0.5 d_ct_8 >= -1000
_C494: - 1000 E1_ct_5_ct_8 - 1000 E2_ct_5_ct_8 - Y_ct_5 + Y_ct_8 - 0.5 d_ct_5
 - 0.5 d_ct_8 >= -2000
_C495: L_ct_5_ct_8 - R_ct_8_ct_5 = 0
_C496: - L_ct_8_ct_5 + R_ct_5_ct_8 = 0
_C497: A_ct_5_ct_8 - B_ct_8_ct_5 = 0
_C498: - A_ct_8_ct_5 + B_ct_5_ct_8 = 0
_C499: Wx_ct_5_ct_8 + Wx_ct_8_ct_5 = 1
_C5: 0 O_ct_2 + l_ct_2 = 17
_C50: Wy_ct_1_ct_2 + Wy_ct_2_ct_1 = 1
_C500: Wy_ct_5_ct_8 + Wy_ct_8_ct_5 = 1
_C501: - L_ct_6_ct_8 + R_ct_6_ct_8 - X_ct_6 + X_ct_8 = 0
_C502: A_ct_6_ct_8 - B_ct_6_ct_8 - Y_ct_6 + Y_ct_8 = 0
_C503: R_ct_6_ct_8 - 1000 Wx_ct_6_ct_8 <= 0
_C504: L_ct_6_ct_8 + 1000 Wx_ct_6_ct_8 <= 1000
_C505: A_ct_6_ct_8 - 1000 Wy_ct_6_ct_8 <= 0
_C506: B_ct_6_ct_8 + 1000 Wy_ct_6_ct_8 <= 1000
_C507: - A_ct_6_ct_8 - B_ct_6_ct_8 + D_ct_6_ct_8 - L_ct_6_ct_8 - R_ct_6_ct_8
 = 0
_C508: D_ct_6_ct_8 - D_ct_8_ct_6 = 0
_C509: 1000 E1_ct_6_ct_8 + 1000 E2_ct_6_ct_8 + X_ct_6 - X_ct_8 - 0.5 l_ct_6
 - 0.5 l_ct_8 >= 0
_C51: - L_ct_1_ct_3 + R_ct_1_ct_3 - X_ct_1 + X_ct_3 = 0
_C510: - 1000 E1_ct_6_ct_8 + 1000 E2_ct_6_ct_8 - X_ct_6 + X_ct_8 - 0.5 l_ct_6
 - 0.5 l_ct_8 >= -1000
_C511: 1000 E1_ct_6_ct_8 - 1000 E2_ct_6_ct_8 + Y_ct_6 - Y_ct_8 - 0.5 d_ct_6
 - 0.5 d_ct_8 >= -1000
_C512: - 1000 E1_ct_6_ct_8 - 1000 E2_ct_6_ct_8 - Y_ct_6 + Y_ct_8 - 0.5 d_ct_6
 - 0.5 d_ct_8 >= -2000
_C513: L_ct_6_ct_8 - R_ct_8_ct_6 = 0
_C514: - L_ct_8_ct_6 + R_ct_6_ct_8 = 0
_C515: A_ct_6_ct_8 - B_ct_8_ct_6 = 0
_C516: - A_ct_8_ct_6 + B_ct_6_ct_8 = 0
_C517: Wx_ct_6_ct_8 + Wx_ct_8_ct_6 = 1
_C518: Wy_ct_6_ct_8 + Wy_ct_8_ct_6 = 1
_C519: - L_ct_7_ct_8 + R_ct_7_ct_8 - X_ct_7 + X_ct_8 = 0
_C52: A_ct_1_ct_3 - B_ct_1_ct_3 - Y_ct_1 + Y_ct_3 = 0
_C520: A_ct_7_ct_8 - B_ct_7_ct_8 - Y_ct_7 + Y_ct_8 = 0
_C521: R_ct_7_ct_8 - 1000 Wx_ct_7_ct_8 <= 0
_C522: L_ct_7_ct_8 + 1000 Wx_ct_7_ct_8 <= 1000
_C523: A_ct_7_ct_8 - 1000 Wy_ct_7_ct_8 <= 0
_C524: B_ct_7_ct_8 + 1000 Wy_ct_7_ct_8 <= 1000
_C525: - A_ct_7_ct_8 - B_ct_7_ct_8 + D_ct_7_ct_8 - L_ct_7_ct_8 - R_ct_7_ct_8
 = 0
_C526: D_ct_7_ct_8 - D_ct_8_ct_7 = 0
_C527: 1000 E1_ct_7_ct_8 + 1000 E2_ct_7_ct_8 + X_ct_7 - X_ct_8 - 0.5 l_ct_7
 - 0.5 l_ct_8 >= 0
_C528: - 1000 E1_ct_7_ct_8 + 1000 E2_ct_7_ct_8 - X_ct_7 + X_ct_8 - 0.5 l_ct_7
 - 0.5 l_ct_8 >= -1000
_C529: 1000 E1_ct_7_ct_8 - 1000 E2_ct_7_ct_8 + Y_ct_7 - Y_ct_8 - 0.5 d_ct_7
 - 0.5 d_ct_8 >= -1000
_C53: R_ct_1_ct_3 - 1000 Wx_ct_1_ct_3 <= 0
_C530: - 1000 E1_ct_7_ct_8 - 1000 E2_ct_7_ct_8 - Y_ct_7 + Y_ct_8 - 0.5 d_ct_7
 - 0.5 d_ct_8 >= -2000
_C531: L_ct_7_ct_8 - R_ct_8_ct_7 = 0
_C532: - L_ct_8_ct_7 + R_ct_7_ct_8 = 0
_C533: A_ct_7_ct_8 - B_ct_8_ct_7 = 0
_C534: - A_ct_8_ct_7 + B_ct_7_ct_8 = 0
_C535: Wx_ct_7_ct_8 + Wx_ct_8_ct_7 = 1
_C536: Wy_ct_7_ct_8 + Wy_ct_8_ct_7 = 1
_C537: CD_ct_1_ct_1 = 0
_C538: CD_ct_1_ct_2 - D_ct_1_ct_2 = 0
_C539: CD_ct_1_ct_3 = 0
_C54: L_ct_1_ct_3 + 1000 Wx_ct_1_ct_3 <= 1000
_C540: CD_ct_1_ct_4 = 0
_C541: CD_ct_1_ct_5 = 0
_C542: CD_ct_1_ct_6 = 0
_C543: CD_ct_1_ct_7 = 0
_C544: CD_ct_1_ct_8 = 0
_C545: CD_ct_2_ct_1 = 0
_C546: CD_ct_2_ct_2 = 0
_C547: CD_ct_2_ct_3 - D_ct_2_ct_3 = 0
_C548: CD_ct_2_ct_4 = 0
_C549: CD_ct_2_ct_5 = 0
_C55: A_ct_1_ct_3 - 1000 Wy_ct_1_ct_3 <= 0
_C550: CD_ct_2_ct_6 = 0
_C551: CD_ct_2_ct_7 = 0
_C552: CD_ct_2_ct_8 = 0
_C553: CD_ct_3_ct_1 = 0
_C554: CD_ct_3_ct_2 = 0
_C555: CD_ct_3_ct_3 = 0
_C556: CD_ct_3_ct_4 - D_ct_3_ct_4 = 0
_C557: CD_ct_3_ct_5 = 0
_C558: CD_ct_3_ct_6 = 0
_C559: CD_ct_3_ct_7 = 0
_C56: B_ct_1_ct_3 + 1000 Wy_ct_1_ct_3 <= 1000
_C560: CD_ct_3_ct_8 = 0
_C561: CD_ct_4_ct_1 = 0
_C562: CD_ct_4_ct_2 = 0
_C563: CD_ct_4_ct_3 = 0
_C564: CD_ct_4_ct_4 = 0
_C565: CD_ct_4_ct_5 - D_ct_4_ct_5 = 0
_C566: CD_ct_4_ct_6 = 0
_C567: CD_ct_4_ct_7 = 0
_C568: CD_ct_4_ct_8 = 0
_C569: CD_ct_5_ct_1 = 0
_C57: - A_ct_1_ct_3 - B_ct_1_ct_3 + D_ct_1_ct_3 - L_ct_1_ct_3 - R_ct_1_ct_3
 = 0
_C570: CD_ct_5_ct_2 = 0
_C571: CD_ct_5_ct_3 = 0
_C572: CD_ct_5_ct_4 = 0
_C573: CD_ct_5_ct_5 = 0
_C574: CD_ct_5_ct_6 - D_ct_5_ct_6 = 0
_C575: CD_ct_5_ct_7 = 0
_C576: CD_ct_5_ct_8 = 0
_C577: CD_ct_6_ct_1 = 0
_C578: CD_ct_6_ct_2 = 0
_C579: CD_ct_6_ct_3 = 0
_C58: D_ct_1_ct_3 - D_ct_3_ct_1 = 0
_C580: CD_ct_6_ct_4 = 0
_C581: CD_ct_6_ct_5 = 0
_C582: CD_ct_6_ct_6 = 0
_C583: CD_ct_6_ct_7 - D_ct_6_ct_7 = 0
_C584: CD_ct_6_ct_8 = 0
_C585: CD_ct_7_ct_1 = 0
_C586: CD_ct_7_ct_2 = 0
_C587: CD_ct_7_ct_3 = 0
_C588: CD_ct_7_ct_4 = 0
_C589: CD_ct_7_ct_5 = 0
_C59: 1000 E1_ct_1_ct_3 + 1000 E2_ct_1_ct_3 + X_ct_1 - X_ct_3 - 0.5 l_ct_1
 - 0.5 l_ct_3 >= 0
_C590: CD_ct_7_ct_6 = 0
_C591: CD_ct_7_ct_7 = 0
_C592: CD_ct_7_ct_8 - D_ct_7_ct_8 = 0
_C593: CD_ct_8_ct_1 = 0
_C594: CD_ct_8_ct_2 = 0
_C595: CD_ct_8_ct_3 = 0
_C596: CD_ct_8_ct_4 = 0
_C597: CD_ct_8_ct_5 = 0
_C598: CD_ct_8_ct_6 = 0
_C599: CD_ct_8_ct_7 = 0
_C6: d_ct_2 + l_ct_2 = 34
_C60: - 1000 E1_ct_1_ct_3 + 1000 E2_ct_1_ct_3 - X_ct_1 + X_ct_3 - 0.5 l_ct_1
 - 0.5 l_ct_3 >= -1000
_C600: CD_ct_8_ct_8 = 0
_C601: - CD_ct_1_ct_1 - CD_ct_1_ct_2 - CD_ct_1_ct_3 - CD_ct_1_ct_4
 - CD_ct_1_ct_5 - CD_ct_1_ct_6 - CD_ct_1_ct_7 - CD_ct_1_ct_8 - CD_ct_2_ct_1
 - CD_ct_2_ct_2 - CD_ct_2_ct_3 - CD_ct_2_ct_4 - CD_ct_2_ct_5 - CD_ct_2_ct_6
 - CD_ct_2_ct_7 - CD_ct_2_ct_8 - CD_ct_3_ct_1 - CD_ct_3_ct_2 - CD_ct_3_ct_3
 - CD_ct_3_ct_4 - CD_ct_3_ct_5 - CD_ct_3_ct_6 - CD_ct_3_ct_7 - CD_ct_3_ct_8
 - CD_ct_4_ct_1 - CD_ct_4_ct_2 - CD_ct_4_ct_3 - CD_ct_4_ct_4 - CD_ct_4_ct_5
 - CD_ct_4_ct_6 - CD_ct_4_ct_7 - CD_ct_4_ct_8 - CD_ct_5_ct_1 - CD_ct_5_ct_2
 - CD_ct_5_ct_3 - CD_ct_5_ct_4 - CD_ct_5_ct_5 - CD_ct_5_ct_6 - CD_ct_5_ct_7
 - CD_ct_5_ct_8 - CD_ct_6_ct_1 - CD_ct_6_ct_2 - CD_ct_6_ct_3 - CD_ct_6_ct_4
 - CD_ct_6_ct_5 - CD_ct_6_ct_6 - CD_ct_6_ct_7 - CD_ct_6_ct_8 - CD_ct_7_ct_1
 - CD_ct_7_ct_2 - CD_ct_7_ct_3 - CD_ct_7_ct_4 - CD_ct_7_ct_5 - CD_ct_7_ct_6
 - CD_ct_7_ct_7 - CD_ct_7_ct_8 - CD_ct_8_ct_1 - CD_ct_8_ct_2 - CD_ct_8_ct_3
 - CD_ct_8_ct_4 - CD_ct_8_ct_5 - CD_ct_8_ct_6 - CD_ct_8_ct_7 - CD_ct_8_ct_8
 + SumCD = 0
_C602: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_1
 + 0.5 l_ct_1 <= 0
_C603: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_1
 + 0.5 d_ct_1 <= 0
_C604: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C605: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_2
 + 0.5 l_ct_2 <= 0
_C606: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_2
 + 0.5 d_ct_2 <= 0
_C607: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C608: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_3
 + 0.5 l_ct_3 <= 0
_C609: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_3
 + 0.5 d_ct_3 <= 0
_C61: 1000 E1_ct_1_ct_3 - 1000 E2_ct_1_ct_3 + Y_ct_1 - Y_ct_3 - 0.5 d_ct_1
 - 0.5 d_ct_3 >= -1000
_C610: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C611: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_4
 + 0.5 l_ct_4 <= 0
_C612: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_4
 + 0.5 d_ct_4 <= 0
_C613: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C614: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_5
 + 0.5 l_ct_5 <= 0
_C615: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_5
 + 0.5 d_ct_5 <= 0
_C616: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C617: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_6
 + 0.5 l_ct_6 <= 0
_C618: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_6
 + 0.5 d_ct_6 <= 0
_C619: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C62: - 1000 E1_ct_1_ct_3 - 1000 E2_ct_1_ct_3 - Y_ct_1 + Y_ct_3 - 0.5 d_ct_1
 - 0.5 d_ct_3 >= -2000
_C620: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_7
 + 0.5 l_ct_7 <= 0
_C621: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_7
 + 0.5 d_ct_7 <= 0
_C622: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C623: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + X_ct_8
 + 0.5 l_ct_8 <= 0
_C624: - 2 Gn_1 - 20 Gn_10 - 200 Gn_100 - 202 Gn_101 - 204 Gn_102 - 206 Gn_103
 - 208 Gn_104 - 210 Gn_105 - 212 Gn_106 - 214 Gn_107 - 216 Gn_108 - 218 Gn_109
 - 22 Gn_11 - 220 Gn_110 - 222 Gn_111 - 224 Gn_112 - 226 Gn_113 - 228 Gn_114
 - 230 Gn_115 - 232 Gn_116 - 234 Gn_117 - 236 Gn_118 - 238 Gn_119 - 24 Gn_12
 - 240 Gn_120 - 242 Gn_121 - 244 Gn_122 - 246 Gn_123 - 248 Gn_124 - 250 Gn_125
 - 252 Gn_126 - 254 Gn_127 - 256 Gn_128 - 258 Gn_129 - 26 Gn_13 - 260 Gn_130
 - 262 Gn_131 - 264 Gn_132 - 266 Gn_133 - 268 Gn_134 - 270 Gn_135 - 272 Gn_136
 - 274 Gn_137 - 276 Gn_138 - 278 Gn_139 - 28 Gn_14 - 280 Gn_140 - 282 Gn_141
 - 284 Gn_142 - 286 Gn_143 - 288 Gn_144 - 290 Gn_145 - 292 Gn_146 - 294 Gn_147
 - 296 Gn_148 - 298 Gn_149 - 30 Gn_15 - 300 Gn_150 - 302 Gn_151 - 304 Gn_152
 - 306 Gn_153 - 308 Gn_154 - 310 Gn_155 - 312 Gn_156 - 314 Gn_157 - 316 Gn_158
 - 318 Gn_159 - 32 Gn_16 - 320 Gn_160 - 322 Gn_161 - 324 Gn_162 - 326 Gn_163
 - 328 Gn_164 - 330 Gn_165 - 332 Gn_166 - 334 Gn_167 - 336 Gn_168 - 338 Gn_169
 - 34 Gn_17 - 340 Gn_170 - 342 Gn_171 - 344 Gn_172 - 346 Gn_173 - 348 Gn_174
 - 350 Gn_175 - 352 Gn_176 - 354 Gn_177 - 356 Gn_178 - 358 Gn_179 - 36 Gn_18
 - 360 Gn_180 - 362 Gn_181 - 364 Gn_182 - 366 Gn_183 - 368 Gn_184 - 370 Gn_185
 - 372 Gn_186 - 374 Gn_187 - 376 Gn_188 - 378 Gn_189 - 38 Gn_19 - 380 Gn_190
 - 382 Gn_191 - 384 Gn_192 - 386 Gn_193 - 388 Gn_194 - 390 Gn_195 - 392 Gn_196
 - 394 Gn_197 - 396 Gn_198 - 398 Gn_199 - 4 Gn_2 - 40 Gn_20 - 400 Gn_200
 - 402 Gn_201 - 404 Gn_202 - 406 Gn_203 - 408 Gn_204 - 410 Gn_205 - 412 Gn_206
 - 414 Gn_207 - 416 Gn_208 - 418 Gn_209 - 42 Gn_21 - 420 Gn_210 - 422 Gn_211
 - 424 Gn_212 - 426 Gn_213 - 428 Gn_214 - 430 Gn_215 - 432 Gn_216 - 434 Gn_217
 - 436 Gn_218 - 438 Gn_219 - 44 Gn_22 - 440 Gn_220 - 442 Gn_221 - 444 Gn_222
 - 446 Gn_223 - 448 Gn_224 - 450 Gn_225 - 452 Gn_226 - 454 Gn_227 - 456 Gn_228
 - 458 Gn_229 - 46 Gn_23 - 460 Gn_230 - 462 Gn_231 - 464 Gn_232 - 466 Gn_233
 - 468 Gn_234 - 470 Gn_235 - 472 Gn_236 - 474 Gn_237 - 476 Gn_238 - 478 Gn_239
 - 48 Gn_24 - 480 Gn_240 - 482 Gn_241 - 484 Gn_242 - 486 Gn_243 - 488 Gn_244
 - 490 Gn_245 - 492 Gn_246 - 494 Gn_247 - 496 Gn_248 - 498 Gn_249 - 50 Gn_25
 - 52 Gn_26 - 54 Gn_27 - 56 Gn_28 - 58 Gn_29 - 6 Gn_3 - 60 Gn_30 - 62 Gn_31
 - 64 Gn_32 - 66 Gn_33 - 68 Gn_34 - 70 Gn_35 - 72 Gn_36 - 74 Gn_37 - 76 Gn_38
 - 78 Gn_39 - 8 Gn_4 - 80 Gn_40 - 82 Gn_41 - 84 Gn_42 - 86 Gn_43 - 88 Gn_44
 - 90 Gn_45 - 92 Gn_46 - 94 Gn_47 - 96 Gn_48 - 98 Gn_49 - 10 Gn_5 - 100 Gn_50
 - 102 Gn_51 - 104 Gn_52 - 106 Gn_53 - 108 Gn_54 - 110 Gn_55 - 112 Gn_56
 - 114 Gn_57 - 116 Gn_58 - 118 Gn_59 - 12 Gn_6 - 120 Gn_60 - 122 Gn_61
 - 124 Gn_62 - 126 Gn_63 - 128 Gn_64 - 130 Gn_65 - 132 Gn_66 - 134 Gn_67
 - 136 Gn_68 - 138 Gn_69 - 14 Gn_7 - 140 Gn_70 - 142 Gn_71 - 144 Gn_72
 - 146 Gn_73 - 148 Gn_74 - 150 Gn_75 - 152 Gn_76 - 154 Gn_77 - 156 Gn_78
 - 158 Gn_79 - 16 Gn_8 - 160 Gn_80 - 162 Gn_81 - 164 Gn_82 - 166 Gn_83
 - 168 Gn_84 - 170 Gn_85 - 172 Gn_86 - 174 Gn_87 - 176 Gn_88 - 178 Gn_89
 - 18 Gn_9 - 180 Gn_90 - 182 Gn_91 - 184 Gn_92 - 186 Gn_93 - 188 Gn_94
 - 190 Gn_95 - 192 Gn_96 - 194 Gn_97 - 196 Gn_98 - 198 Gn_99 + Y_ct_8
 + 0.5 d_ct_8 <= 0
_C625: Gn_1 + Gn_10 + Gn_100 + Gn_101 + Gn_102 + Gn_103 + Gn_104 + Gn_105
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
_C626: - 55.1 Gn_1 - 5510 Gn_10 - 551000 Gn_100 - 562075.1 Gn_101
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
_C63: L_ct_1_ct_3 - R_ct_3_ct_1 = 0
_C64: - L_ct_3_ct_1 + R_ct_1_ct_3 = 0
_C65: A_ct_1_ct_3 - B_ct_3_ct_1 = 0
_C66: - A_ct_3_ct_1 + B_ct_1_ct_3 = 0
_C67: Wx_ct_1_ct_3 + Wx_ct_3_ct_1 = 1
_C68: Wy_ct_1_ct_3 + Wy_ct_3_ct_1 = 1
_C69: - L_ct_2_ct_3 + R_ct_2_ct_3 - X_ct_2 + X_ct_3 = 0
_C7: X_ct_2 - 0.5 l_ct_2 >= 0
_C70: A_ct_2_ct_3 - B_ct_2_ct_3 - Y_ct_2 + Y_ct_3 = 0
_C71: R_ct_2_ct_3 - 1000 Wx_ct_2_ct_3 <= 0
_C72: L_ct_2_ct_3 + 1000 Wx_ct_2_ct_3 <= 1000
_C73: A_ct_2_ct_3 - 1000 Wy_ct_2_ct_3 <= 0
_C74: B_ct_2_ct_3 + 1000 Wy_ct_2_ct_3 <= 1000
_C75: - A_ct_2_ct_3 - B_ct_2_ct_3 + D_ct_2_ct_3 - L_ct_2_ct_3 - R_ct_2_ct_3
 = 0
_C76: D_ct_2_ct_3 - D_ct_3_ct_2 = 0
_C77: 1000 E1_ct_2_ct_3 + 1000 E2_ct_2_ct_3 + X_ct_2 - X_ct_3 - 0.5 l_ct_2
 - 0.5 l_ct_3 >= 0
_C78: - 1000 E1_ct_2_ct_3 + 1000 E2_ct_2_ct_3 - X_ct_2 + X_ct_3 - 0.5 l_ct_2
 - 0.5 l_ct_3 >= -1000
_C79: 1000 E1_ct_2_ct_3 - 1000 E2_ct_2_ct_3 + Y_ct_2 - Y_ct_3 - 0.5 d_ct_2
 - 0.5 d_ct_3 >= -1000
_C8: Y_ct_2 - 0.5 d_ct_2 >= 0
_C80: - 1000 E1_ct_2_ct_3 - 1000 E2_ct_2_ct_3 - Y_ct_2 + Y_ct_3 - 0.5 d_ct_2
 - 0.5 d_ct_3 >= -2000
_C81: L_ct_2_ct_3 - R_ct_3_ct_2 = 0
_C82: - L_ct_3_ct_2 + R_ct_2_ct_3 = 0
_C83: A_ct_2_ct_3 - B_ct_3_ct_2 = 0
_C84: - A_ct_3_ct_2 + B_ct_2_ct_3 = 0
_C85: Wx_ct_2_ct_3 + Wx_ct_3_ct_2 = 1
_C86: Wy_ct_2_ct_3 + Wy_ct_3_ct_2 = 1
_C87: - L_ct_1_ct_4 + R_ct_1_ct_4 - X_ct_1 + X_ct_4 = 0
_C88: A_ct_1_ct_4 - B_ct_1_ct_4 - Y_ct_1 + Y_ct_4 = 0
_C89: R_ct_1_ct_4 - 1000 Wx_ct_1_ct_4 <= 0
_C9: 0 O_ct_3 + l_ct_3 = 17
_C90: L_ct_1_ct_4 + 1000 Wx_ct_1_ct_4 <= 1000
_C91: A_ct_1_ct_4 - 1000 Wy_ct_1_ct_4 <= 0
_C92: B_ct_1_ct_4 + 1000 Wy_ct_1_ct_4 <= 1000
_C93: - A_ct_1_ct_4 - B_ct_1_ct_4 + D_ct_1_ct_4 - L_ct_1_ct_4 - R_ct_1_ct_4
 = 0
_C94: D_ct_1_ct_4 - D_ct_4_ct_1 = 0
_C95: 1000 E1_ct_1_ct_4 + 1000 E2_ct_1_ct_4 + X_ct_1 - X_ct_4 - 0.5 l_ct_1
 - 0.5 l_ct_4 >= 0
_C96: - 1000 E1_ct_1_ct_4 + 1000 E2_ct_1_ct_4 - X_ct_1 + X_ct_4 - 0.5 l_ct_1
 - 0.5 l_ct_4 >= -1000
_C97: 1000 E1_ct_1_ct_4 - 1000 E2_ct_1_ct_4 + Y_ct_1 - Y_ct_4 - 0.5 d_ct_1
 - 0.5 d_ct_4 >= -1000
_C98: - 1000 E1_ct_1_ct_4 - 1000 E2_ct_1_ct_4 - Y_ct_1 + Y_ct_4 - 0.5 d_ct_1
 - 0.5 d_ct_4 >= -2000
_C99: L_ct_1_ct_4 - R_ct_4_ct_1 = 0
Bounds
X_ct_1 free
X_ct_2 free
X_ct_3 free
X_ct_4 free
X_ct_5 free
X_ct_6 free
X_ct_7 free
X_ct_8 free
Y_ct_1 free
Y_ct_2 free
Y_ct_3 free
Y_ct_4 free
Y_ct_5 free
Y_ct_6 free
Y_ct_7 free
Y_ct_8 free
Binaries
E1_ct_1_ct_2
E1_ct_1_ct_3
E1_ct_1_ct_4
E1_ct_1_ct_5
E1_ct_1_ct_6
E1_ct_1_ct_7
E1_ct_1_ct_8
E1_ct_2_ct_3
E1_ct_2_ct_4
E1_ct_2_ct_5
E1_ct_2_ct_6
E1_ct_2_ct_7
E1_ct_2_ct_8
E1_ct_3_ct_4
E1_ct_3_ct_5
E1_ct_3_ct_6
E1_ct_3_ct_7
E1_ct_3_ct_8
E1_ct_4_ct_5
E1_ct_4_ct_6
E1_ct_4_ct_7
E1_ct_4_ct_8
E1_ct_5_ct_6
E1_ct_5_ct_7
E1_ct_5_ct_8
E1_ct_6_ct_7
E1_ct_6_ct_8
E1_ct_7_ct_8
E2_ct_1_ct_2
E2_ct_1_ct_3
E2_ct_1_ct_4
E2_ct_1_ct_5
E2_ct_1_ct_6
E2_ct_1_ct_7
E2_ct_1_ct_8
E2_ct_2_ct_3
E2_ct_2_ct_4
E2_ct_2_ct_5
E2_ct_2_ct_6
E2_ct_2_ct_7
E2_ct_2_ct_8
E2_ct_3_ct_4
E2_ct_3_ct_5
E2_ct_3_ct_6
E2_ct_3_ct_7
E2_ct_3_ct_8
E2_ct_4_ct_5
E2_ct_4_ct_6
E2_ct_4_ct_7
E2_ct_4_ct_8
E2_ct_5_ct_6
E2_ct_5_ct_7
E2_ct_5_ct_8
E2_ct_6_ct_7
E2_ct_6_ct_8
E2_ct_7_ct_8
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
O_ct_1
O_ct_2
O_ct_3
O_ct_4
O_ct_5
O_ct_6
O_ct_7
O_ct_8
Wx_ct_1_ct_2
Wx_ct_1_ct_3
Wx_ct_1_ct_4
Wx_ct_1_ct_5
Wx_ct_1_ct_6
Wx_ct_1_ct_7
Wx_ct_1_ct_8
Wx_ct_2_ct_1
Wx_ct_2_ct_3
Wx_ct_2_ct_4
Wx_ct_2_ct_5
Wx_ct_2_ct_6
Wx_ct_2_ct_7
Wx_ct_2_ct_8
Wx_ct_3_ct_1
Wx_ct_3_ct_2
Wx_ct_3_ct_4
Wx_ct_3_ct_5
Wx_ct_3_ct_6
Wx_ct_3_ct_7
Wx_ct_3_ct_8
Wx_ct_4_ct_1
Wx_ct_4_ct_2
Wx_ct_4_ct_3
Wx_ct_4_ct_5
Wx_ct_4_ct_6
Wx_ct_4_ct_7
Wx_ct_4_ct_8
Wx_ct_5_ct_1
Wx_ct_5_ct_2
Wx_ct_5_ct_3
Wx_ct_5_ct_4
Wx_ct_5_ct_6
Wx_ct_5_ct_7
Wx_ct_5_ct_8
Wx_ct_6_ct_1
Wx_ct_6_ct_2
Wx_ct_6_ct_3
Wx_ct_6_ct_4
Wx_ct_6_ct_5
Wx_ct_6_ct_7
Wx_ct_6_ct_8
Wx_ct_7_ct_1
Wx_ct_7_ct_2
Wx_ct_7_ct_3
Wx_ct_7_ct_4
Wx_ct_7_ct_5
Wx_ct_7_ct_6
Wx_ct_7_ct_8
Wx_ct_8_ct_1
Wx_ct_8_ct_2
Wx_ct_8_ct_3
Wx_ct_8_ct_4
Wx_ct_8_ct_5
Wx_ct_8_ct_6
Wx_ct_8_ct_7
Wy_ct_1_ct_2
Wy_ct_1_ct_3
Wy_ct_1_ct_4
Wy_ct_1_ct_5
Wy_ct_1_ct_6
Wy_ct_1_ct_7
Wy_ct_1_ct_8
Wy_ct_2_ct_1
Wy_ct_2_ct_3
Wy_ct_2_ct_4
Wy_ct_2_ct_5
Wy_ct_2_ct_6
Wy_ct_2_ct_7
Wy_ct_2_ct_8
Wy_ct_3_ct_1
Wy_ct_3_ct_2
Wy_ct_3_ct_4
Wy_ct_3_ct_5
Wy_ct_3_ct_6
Wy_ct_3_ct_7
Wy_ct_3_ct_8
Wy_ct_4_ct_1
Wy_ct_4_ct_2
Wy_ct_4_ct_3
Wy_ct_4_ct_5
Wy_ct_4_ct_6
Wy_ct_4_ct_7
Wy_ct_4_ct_8
Wy_ct_5_ct_1
Wy_ct_5_ct_2
Wy_ct_5_ct_3
Wy_ct_5_ct_4
Wy_ct_5_ct_6
Wy_ct_5_ct_7
Wy_ct_5_ct_8
Wy_ct_6_ct_1
Wy_ct_6_ct_2
Wy_ct_6_ct_3
Wy_ct_6_ct_4
Wy_ct_6_ct_5
Wy_ct_6_ct_7
Wy_ct_6_ct_8
Wy_ct_7_ct_1
Wy_ct_7_ct_2
Wy_ct_7_ct_3
Wy_ct_7_ct_4
Wy_ct_7_ct_5
Wy_ct_7_ct_6
Wy_ct_7_ct_8
Wy_ct_8_ct_1
Wy_ct_8_ct_2
Wy_ct_8_ct_3
Wy_ct_8_ct_4
Wy_ct_8_ct_5
Wy_ct_8_ct_6
Wy_ct_8_ct_7
End
