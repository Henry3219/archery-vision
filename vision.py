import cv2
import numpy as np
import settings
import os

# 在图像中定位靶纸并校正其透视形变
def find_target(img, fn="debug"):
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 步骤1: 通过颜色找到蓝色最外环作为初步定位信标
    b_mask = cv2.inRange(hsv, settings.BLUE_HSV_LOWER, settings.BLUE_HSV_UPPER)
    cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: return None, "未找到蓝色信标"
        
    # 步骤2: 筛选"圆"轮廓
    circ_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0 or area < 100: continue # 忽略小轮廓
        circ_cnts.append(c)
            
    if not circ_cnts:
        if settings.DEBUG_MODE: cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_debug_b_mask_nc.jpg"), b_mask)
        return None, "未找到圆形蓝色信标"
        
    # 步骤3: 确定最大信标轮廓及中心
    b_cnt = max(circ_cnts, key=cv2.contourArea)
    b_area = cv2.contourArea(b_cnt)
    M = cv2.moments(b_cnt)
    if M["m00"] == 0: return None, "信标无效"
    b_cen = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    # 步骤4: 在信标周围定义更大的搜索区域(ROI)
    x, y, wc, hc = cv2.boundingRect(b_cnt)
    pad_f = settings.ROI_PADDING_FACTOR # 从settings读取padding因子
    pad_w = int(wc * pad_f)
    pad_h = int(hc * pad_f)
    xs = max(0, x - pad_w)
    ys = max(0, y - pad_h)
    ws = min(w - xs, wc + 2 * pad_w)
    hs = min(h - ys, hc + 2 * pad_h)
    
    roi = img[ys:ys+hs, xs:xs+ws]
    if roi.size == 0: return None, "ROI无效"

    # 步骤5: 在ROI内进行边缘检测以找到靶纸方形边界
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # 对比度增强
    enh = clahe.apply(gray)
    edges = cv2.Canny(enh, 50, 150) 
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) # 形态学核
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, krn) # 闭运算连接边缘

    if settings.DEBUG_MODE: cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_debug_1_edges.jpg"), closed)
            
    roi_cnts, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    # 步骤6: 筛选四边形轮廓作为靶纸候选
    q_cand = []
    for c in roi_cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # 多边形逼近
        if len(approx) != 4: continue # 必须是四边形
        area = cv2.contourArea(approx)
        if area < b_area : continue # 面积必须比蓝色环大
        c_glob = approx + np.array([xs, ys]) # 转为全局坐标检查
        if cv2.pointPolygonTest(c_glob.astype(np.float32), b_cen, False) < 0: continue # 必须包含蓝色环中心
        q_cand.append(approx) # 存储ROI内的局部坐标

    if settings.DEBUG_MODE and q_cand:
        dbg_roi = roi.copy()
        cv2.drawContours(dbg_roi, q_cand, -1, (0,255,0), 2)
        cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_debug_2_cand.jpg"), dbg_roi)
        
    if not q_cand: return None, "未找到靶纸候选框"
        
    best_c = max(q_cand, key=cv2.contourArea) # 选择面积最大的候选
            
    # 步骤7: 进行透视变换，校正图像
    pts_loc = best_c
    pts_glob = pts_loc + np.array([xs, ys]) # 转为全局坐标
    pts = pts_glob.reshape(4, 2).astype("float32")
    rect = np.zeros((4,2), dtype="float32") # 存储有序角点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上角
    rect[2] = pts[np.argmax(s)] # 右下角
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上角
    rect[3] = pts[np.argmax(diff)] # 左下角
    
    out_s = settings.OUTPUT_SIZE
    dst_pts = np.array([[0,0], [out_s-1,0], [out_s-1,out_s-1], [0,out_s-1]], dtype="float32")
    M_persp = cv2.getPerspectiveTransform(rect, dst_pts) # 计算变换矩阵
    warped = cv2.warpPerspective(img, M_persp, (out_s, out_s)) # 应用变换

    if settings.DEBUG_MODE: cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_debug_3_warped.jpg"), warped)
    
    return warped, "靶纸已校正"

# 通过颜色检测靶心（黄色区域）
def detect_center_by_color(img_corr, fn="debug"):
    hsv = cv2.cvtColor(img_corr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, settings.YELLOW_HSV_LOWER, settings.YELLOW_HSV_UPPER)
    krn = np.ones((5,5), np.uint8) # 形态学核
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, krn) # 开运算去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, krn, iterations=2) # 闭运算连接
    
    if settings.DEBUG_MODE: cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn)[0]}_debug_4_y_mask.jpg"), mask)
        
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None
    tgt_cnt = max(cnts, key=cv2.contourArea) # 最大黄色区域
    if cv2.contourArea(tgt_cnt) < 50: return None, None # 最小面积阈值
    (x,y), rad = cv2.minEnclosingCircle(tgt_cnt) # 最小外接圆定中心
    cen = (int(x), int(y))
    return cen, tgt_cnt

# 使用最小外接圆测量主要色带（黄红蓝）的外部半径
def measure_ring_radii(img, cen, fn_pref="target"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img.shape
    out_r = {"yellow": None, "red": None, "blue": None} # 存储半径结果

    # 各色带的HSV定义
    defs = {
        "yellow": (settings.YELLOW_HSV_LOWER, settings.YELLOW_HSV_UPPER, None, None),
        "red": (settings.RED_HSV_LOWER, settings.RED_HSV_UPPER, settings.RED2_HSV_LOWER, settings.RED2_HSV_UPPER),
        "blue": (settings.BLUE_HSV_LOWER, settings.BLUE_HSV_UPPER, None, None)
    }
    krn = np.ones((5,5), np.uint8) # 形态学核

    for name, (l1, u1, l2, u2) in defs.items():
        m1 = cv2.inRange(hsv, l1, u1)
        mask = m1
        if l2 is not None and u2 is not None: # 处理跨HUE边界的颜色（如红色）
            m2 = cv2.inRange(hsv, l2, u2)
            mask = cv2.bitwise_or(m1, m2)

        # 形态学处理颜色掩码
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, krn, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, krn, iterations=2)

        if settings.DEBUG_MODE: cv2.imwrite(os.path.join(settings.OUTPUT_DIR, f"{os.path.splitext(fn_pref)[0]}_debug_mask_{name}.jpg"), mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue

        # 筛选最佳轮廓
        best_c = None
        max_a = 0
        min_a = w * h * 0.005 # 最小面积，占图像0.5%

        for c_ in cnts:
            area = cv2.contourArea(c_)
            if area > max_a and area > min_a:
                # 检查轮廓是否大致包围已知靶心
                if cv2.pointPolygonTest(c_, cen, False) >= -int(w*0.05): # 允许中心点在轮廓边缘或内部
                    max_a = area
                    best_c = c_
        
        if best_c is None: continue
            
        (xf, yf), rf = cv2.minEnclosingCircle(best_c) # 拟合最小外接圆
        
        # 验证拟合圆心与靶心是否接近
        dist_sq = (xf - cen[0])**2 + (yf - cen[1])**2
        if dist_sq < (w * 0.05)**2 : # 允许偏差为图像宽度的5%
            out_r[name] = int(rf)
    return out_r

# 按比例计算所有得分环的半径
def calculate_all_ring_radii_proportional(maj_r, incl_x=True):
    # 黄色带外圈(9环)半径Ry = 2W, 红色(7环)Rr = 4W, 蓝色(5环)Rb = 6W (W为单位环宽)
    ry, rr, rb = maj_r.get("yellow"), maj_r.get("red"), maj_r.get("blue")
    w_ring = None # 单位环宽度

    # 优先用内侧色带计算W
    if ry is not None and ry > 1: w_ring = ry / 2.0
    elif rr is not None and rr > 1: w_ring = rr / 4.0
    elif rb is not None and rb > 1: w_ring = rb / 6.0
    
    if w_ring is None or w_ring <= 1.0: return [] # W必须有效

    all_r = []
    if incl_x: all_r.append(int(round(0.5 * w_ring))) # X环半径 (0.5W)
    for i in range(1, 8): # 10环线到4环线 (1W 到 7W)
        all_r.append(int(round(i * w_ring)))
    
    # 去重并排序，确保半径大于0
    return sorted(list(set(r for r in all_r if r > 0)))

# 处理流程
def analyze_image(img, fn):
    img_corr, msg = find_target(img, fn) # 校正靶纸
    if img_corr is None:
        return {"success": False, "message": msg, "corrected_img": None, "center": None, "radii": []}

    cen, _ = detect_center_by_color(img_corr, fn) # 定位靶心
    if cen is None:
        return {"success": False, "corrected_img": img_corr, "center": None, "radii": [], "message": "未找到靶心"}
    
    maj_r = measure_ring_radii(img_corr, cen, fn_pref=fn) # 测量主色带半径
    if all(v is None for v in maj_r.values()): # 至少一个主色带被测到
        return {"success": True, "corrected_img": img_corr, "center": cen, "radii": [], "message": "未测量到主环"}

    all_r = calculate_all_ring_radii_proportional(maj_r, incl_x=True) # 计算所有环半径
    if not all_r:
        return {"success": True, "corrected_img": img_corr, "center": cen, "radii": [], "message": "比例半径计算失败"}

    return {"success": True, "corrected_img": img_corr, "center": cen, "radii": all_r, "message": "分析完成！"}