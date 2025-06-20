import os
import cv2
import vision
import utils
import settings

def run():
    # 检查输入输出文件夹
    if not os.path.exists(settings.INPUT_DIR): 
        os.makedirs(settings.INPUT_DIR)
        print(f"已创建输入文件夹 '{settings.INPUT_DIR}'。请添加图片后重新运行。")
        return
    if not os.path.exists(settings.OUTPUT_DIR): 
        os.makedirs(settings.OUTPUT_DIR)

    # 获取图片列表
    img_files = [f for f in os.listdir(settings.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files: 
        print(f"文件夹 '{settings.INPUT_DIR}' 中没有图片。")
        return

    print(f"--- 开始处理 {len(img_files)} 张图片 ---")
    for fn in img_files: # 遍历图片文件
        print(f"\n>> 正在处理: {fn}")
        img_path = os.path.join(settings.INPUT_DIR, fn)
        img = cv2.imread(img_path) # 读取图片
        if img is None: 
            print(f"   无法读取图片 {fn}，已跳过。")
            continue
        
        res = vision.analyze_image(img, fn) # 调用核心分析函数
        print(f"   处理消息: {res['message']}")
        
        # 根据分析结果保存图像
        if res["success"]:
            final_img = utils.draw_results(res["corrected_img"], res["center"], res["radii"]) # 绘制结果
            base_fn = os.path.splitext(fn)[0]
            out_path = os.path.join(settings.OUTPUT_DIR, f"{base_fn}_result.jpg")
            utils.save_image(final_img, out_path) # 保存最终结果图
            print(f"   成功！最终结果图已保存到: {out_path}")
        elif "corrected_img" in res and res["corrected_img"] is not None: # 如果仅校正成功
             base_fn = os.path.splitext(fn)[0]
             corr_path = os.path.join(settings.OUTPUT_DIR, f"{base_fn}_corrected.jpg")
             utils.save_image(res["corrected_img"], corr_path) # 保存校正后的图像
             print(f"   部分成功，校正图已保存。")

    print("\n--- 所有图片处理完毕 ---")

if __name__ == "__main__":
    run()