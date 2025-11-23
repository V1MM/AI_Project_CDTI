# #Pattern Recognition - Probability Theory

# # p(B=r) - ความน่าจะเป็นที่จะหยิบกล่องแดง (Red Box)
# P_R = 4 / 10
# # p(B=b) - ความน่าจะเป็นที่จะหยิบกล่องน้ำเงิน (Blue Box)
# P_B = 6 / 10

# #ความน่าจะเป็นแบบมีเงื่อนไข
# # กล่องแดง (r):
# # p(F=o|B=r) - ส้มจากแดง (6/8 = 0.75)
# P_FO_BR = 6/8
# # p(F=a|B=r) - แอปเปิ้ลจากแดง (2/8 = 0.25)
# P_FA_BR = 2/8

# # กล่องน้ำเงิน (b):
# # p(F=o|B=b) - ส้มจากน้ำเงิน (1/4 = 0.25)
# P_FO_BB = 1/4
# # p(F=a|B=b) - แอปเปิ้ลจากน้ำเงิน (3/4 = 0.75)
# P_FA_BB = 3/4

# #คำนวณความน่าจะเป็นรวม (Law of Total Probability)
# #ความน่าจะเป็นที่จะหยิบได้ Orange (P(F=o))
# # P(O) = P(O|R)P(R) + P(O|B)P(B)
# P_FO = (P_FO_BR*P_R) + (P_FO_BB*P_B)

# #ความน่าจะเป็นที่จะหยิบได้ Apple (P(F=a))
# # P(A) = P(A|R)P(R) + P(A|B)P(B)
# P_FA = (P_FA_BR*P_R) + (P_FA_BB*P_B)

# #คำนวณความน่าจะเป็นย้อนหลัง (Bayes' Theorem)
# #ความน่าจะเป็นที่จะมาจากกล่องแดง เมื่อรู้ว่าได้ Orange (P(B=r|F=o))
# # P(R|O) = [P(O|R) * P(R)] / P(O)
# P_BR_FO = (P_FO_BR * P_R) / P_FO

# #OUTPUT
# print(f"ความน่าจะเป็นที่จะหยิบได้ Orange (P(F=o)): {P_FO:.4f} ({P_FO*100:.2f}%)")
# print(f"ความน่าจะเป็นที่จะหยิบได้ Apple (P(F=a)): {P_FA:.4f} ({P_FA*100:.2f}%)")
# print(f"ความน่าจะเป็นที่จะมาจากกล่องแดง เมื่อรู้ว่าได้ Orange (P(B=r|F=o)): {P_BR_FO:.4f} ({P_BR_FO*100:.2f}%)")

def calculate_probability(P_R, P_B, P_FO_BR, P_FA_BR, P_FO_BB, P_FA_BB):
    """
    คำนวณความน่าจะเป็นรวมและความน่าจะเป็นย้อนหลัง (Bayes' Theorem)
    P_R, P_B: ความน่าจะเป็นในการเลือกกล่องแดง, น้ำเงิน
    P_FO_BR, P_FA_BR: ความน่าจะเป็น ส้ม/แอปเปิ้ล เมื่อมาจากกล่องแดง
    P_FO_BB, P_FA_BB: ความน่าจะเป็น ส้ม/แอปเปิ้ล เมื่อมาจากกล่องน้ำเงิน
    """
    
    # ตรวจสอบความสมบูรณ์ของ Prior Probability
    if P_R + P_B != 1.0: #100%
        print("คำเตือน: P_R + P_B ไม่เท่ากับ 1.0 อาจทำให้ผลลัพธ์ผิดพลาด")

    # --- 1. คำนวณความน่าจะเป็นรวม (Law of Total Probability) ---
    P_FO = (P_FO_BR * P_R) + (P_FO_BB * P_B)
    P_FA = (P_FA_BR * P_R) + (P_FA_BB * P_B)

    # --- 2. คำนวณความน่าจะเป็นย้อนหลัง (Bayes' Theorem) ---
    
    # P(B=r|F=o) - โอกาสมาจากกล่องแดง เมื่อรู้ว่าได้ Orange
    if P_FO == 0:
        P_BR_FO = 0.0
    else:
        P_BR_FO = (P_FO_BR * P_R) / P_FO
        
    # P(B=b|F=a) - โอกาสมาจากกล่องน้ำเงิน เมื่อรู้ว่าได้ Apple (คำนวณเพิ่มให้)
    if P_FA == 0:
        P_BB_FA = 0.0
    else:
        P_BB_FA = (P_FA_BB * P_B) / P_FA

    result = [
        f"ความน่าจะเป็นที่จะหยิบได้ Orange (P(F=o)): {P_FO:.4f} ({P_FO*100:.2f}%)",
        f"ความน่าจะเป็นที่จะหยิบได้ Apple (P(F=a)): {P_FA:.4f} ({P_FA*100:.2f}%)",
        f"ความน่าจะเป็นที่จะมาจากกล่องแดง เมื่อรู้ว่าได้ Orange (P(B=r|F=o)): {P_BR_FO:.4f} ({P_BR_FO*100:.2f}%)",
        f"ความน่าจะเป็นที่จะมาจากกล่องน้ำเงิน เมื่อรู้ว่าได้ Apple (P(B=b|F=a)): {P_BB_FA:.4f} ({P_BB_FA*100:.2f}%) (คำนวณเพิ่มเติม)"
    ]
    return result

if __name__ == "__main__":
    print("="*50)
    print("Pattern Recognition - Probability Theory (User Input)")
    print("="*50)
    
    # 1. รับค่าความน่าจะเป็นในการเลือกกล่อง
    try:
        P_R_input = int(input("Probability of picking red box (P_R) %: "))
        P_B_input = int(input("Probability of picking blue box (P_B) %: "))
        P_R = P_R_input / 100
        P_B = P_B_input / 100
    except ValueError:
        print("ข้อผิดพลาด: โปรดป้อนตัวเลขสำหรับเปอร์เซ็นต์")
        exit()
    
    # 2. รับจำนวนผลไม้
    try:
        MaxRB = int(input("\nMaximum total fruits in Red Box: "))
        MaxBB = int(input("Maximum total fruits in Blue Box: "))
        
        A_R = int(input("How many Apple (เขียว) in Red Box: ")) 
        O_R = int(input("How many Orange (ส้ม) in Red Box: "))
        
        A_B = int(input("How many Apple (เขียว) in Blue Box: "))
        O_B = int(input("How many Orange (ส้ม) in Blue Box: "))
        
    except ValueError:
        print("ข้อผิดพลาด: โปรดป้อนตัวเลขสำหรับจำนวนผลไม้")
        exit()

    if A_R + O_R != MaxRB or A_B + O_B != MaxBB:
        print("\nคำเตือน: จำนวนผลไม้ที่นับได้ไม่ตรงกับ Maximum total fruits!")
        MaxRB = A_R + O_R
        MaxBB = A_B + O_B
    
    if MaxRB == 0:
        P_FA_BR = P_FO_BR = 0.0
    else:
        P_FA_BR = A_R / MaxRB  # P(A|R)
        P_FO_BR = O_R / MaxRB  # P(O|R)

    if MaxBB == 0:
        P_FA_BB = P_FO_BB = 0.0
    else:
        P_FA_BB = A_B / MaxBB  # P(A|B)
        P_FO_BB = O_B / MaxBB  # P(O|B)
        
    final_results = calculate_probability(P_R, P_B, P_FO_BR, P_FA_BR, P_FO_BB, P_FA_BB)
    
    print("\n" + "="*50)
    print("ผลการคำนวณ Bayes' Theorem")
    print("="*50)
    print("\n".join(final_results))
