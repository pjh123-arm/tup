#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的演示程序，包含褐色矩形框以展示完整功能
"""

import cv2
import numpy as np
from rectangle_detection import RectangleDetector

def create_enhanced_demo_images():
    """创建包含褐色矩形框的增强演示图像"""
    
    # 创建基准图片 (包含红色矩形框)
    base_image = np.zeros((500, 700, 3), dtype=np.uint8)
    base_image.fill(240)  # 浅灰色背景
    
    # 红色矩形框1
    cv2.rectangle(base_image, (50, 50), (250, 180), (0, 0, 255), 4)
    # 红色矩形框2
    cv2.rectangle(base_image, (350, 200), (550, 330), (0, 0, 255), 4)
    # 红色矩形框3 (这个只会包含青色点)
    cv2.rectangle(base_image, (100, 350), (300, 450), (0, 0, 255), 4)
    
    cv2.imwrite('enhanced_base_image.png', base_image)
    print("✅ 创建增强基准图片: enhanced_base_image.png")
    
    # 创建对比图片 (包含褐色矩形框和青色点)
    compare_image = np.zeros((500, 700, 3), dtype=np.uint8)
    compare_image.fill(220)  # 浅灰色背景
    
    # 褐色矩形框1 (在红色矩形框1内) - 使用明确的褐色
    cv2.rectangle(compare_image, (80, 80), (220, 150), (19, 69, 139), -1)  # 棕色
    
    # 褐色矩形框2 (在红色矩形框2内) - 使用明确的褐色
    cv2.rectangle(compare_image, (380, 230), (520, 300), (19, 69, 139), -1)  # 棕色
    
    # 青色点 (在红色矩形框1内)
    cv2.circle(compare_image, (150, 160), 8, (255, 255, 0), -1)
    
    # 青色点 (在红色矩形框2内)  
    cv2.circle(compare_image, (450, 250), 6, (255, 255, 0), -1)
    cv2.circle(compare_image, (400, 280), 7, (255, 255, 0), -1)
    
    # 青色点 (在红色矩形框3内，只有青色点，没有褐色矩形框)
    cv2.circle(compare_image, (150, 400), 9, (255, 255, 0), -1)
    cv2.circle(compare_image, (250, 380), 5, (255, 255, 0), -1)
    
    cv2.imwrite('enhanced_compare_image.png', compare_image)
    print("✅ 创建增强对比图片: enhanced_compare_image.png")

def main():
    """运行增强演示"""
    print("🚀 启动增强矩形检测演示程序")
    print("=" * 60)
    
    # 创建增强示例图像
    create_enhanced_demo_images()
    
    # 创建调试模式的检测器
    detector = RectangleDetector(debug_mode=True)
    
    # 处理图像
    detector.process_images('enhanced_base_image.png', 'enhanced_compare_image.png')

if __name__ == "__main__":
    main()