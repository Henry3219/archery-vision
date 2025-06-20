import cv2
import numpy as np

# 在图像上绘制圆环和靶心
def draw_results(img, cen, radii_list):
    out_img = img.copy()
    
    # BGR颜色定义
    cols = {
        "yellow": (0, 255, 255), "red": (0, 0, 255),
        "blue": (255, 0, 0), "black": (0, 0, 0)
    }

    # 各环颜色 (X, 10, 9为黄; 8, 7为红; 6, 5为蓝)
    ring_cols = [
        cols["yellow"], cols["yellow"], cols["yellow"], 
        cols["red"], cols["red"], 
        cols["blue"], cols["blue"],   
    ]
    num_draw = 7 # 绘制X, 10, 9, 8, 7, 6, 5 共7个环线

    if radii_list:
        # 从内向外绘制
        for i in range(min(len(radii_list), num_draw)):
            r_val = radii_list[i]
            if r_val <= 0: continue # 跳过无效半径
            
            # cv2.circle参数需为整数
            cv2.circle(out_img, (int(cen[0]), int(cen[1])), int(r_val), ring_cols[i], 4) # 线宽为4

    # 绘制靶心十字
    if cen is not None:
        cx, cy = int(cen[0]), int(cen[1])
        cv2.line(out_img, (cx-15,cy), (cx+15,cy), (0,255,0), 3) # 绿色十字
        cv2.line(out_img, (cx,cy-15), (cx,cy+15), (0,255,0), 3)
        cv2.putText(out_img, f"C:({cx},{cy})", (cx+20,cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2) # 中心坐标
    return out_img

# 保存图像
def save_image(img, path):
    try:
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        print(f"保存图像 {path} 失败: {e}")
        return False