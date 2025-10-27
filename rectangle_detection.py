#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于公式的双重对比逻辑矩形检测程序
支持调试模式，用于检测红色矩形框和褐色矩形框以及青色点
"""

import cv2
import numpy as np
import os
import json
import datetime
from typing import List, Tuple, Dict, Any
import logging

class RectangleDetector:
    """矩形检测器类"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化矩形检测器
        
        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        self.debug_output_dir = "debug_output"
        self.detection_results = {
            "red_rectangles": [],
            "brown_rectangles": [],
            "cyan_points": [],
            "z1_values": [],
            "z2_values": [],
            "first_layer_accuracy": [],
            "second_layer_accuracy": 0.0,
            "excluded_red_rectangles": [],
            "matching_statistics": {}
        }
        
        # 设置日志
        if self.debug_mode:
            self._create_debug_directory()
            self._setup_logging()
    
    def _setup_logging(self):
        """设置调试日志"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.debug_output_dir}/debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_debug_directory(self):
        """创建调试输出目录"""
        if not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)
        
        # 创建子目录
        subdirs = ['images', 'data', 'analysis']
        for subdir in subdirs:
            subdir_path = os.path.join(self.debug_output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
    
    def debug_mode_output(self):
        """调试模式入口函数"""
        if not self.debug_mode:
            return
        
        print("=" * 60)
        print("🔧 基于公式的双重对比逻辑矩形检测程序 - 调试模式")
        print("=" * 60)
        print(f"⏰ 程序启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 调试输出目录: {self.debug_output_dir}")
        print("🎯 检测目标:")
        print("   - 红色矩形框（基准图片）")
        print("   - 褐色矩形框（对比图片）")
        print("   - 青色点（对比图片）")
        print("📊 计算指标:")
        print("   - 第一层准确率: P = 1 - |Z1 - Z2| / MAX(Z1, Z2)")
        print("   - 第二层准确率: P2 = (Σi=1^n Si/Mi) / n")
        print("🔍 特殊逻辑: 排除只包含青色点的红色框")
        print("-" * 60)
        
        self.logger.info("矩形检测程序调试模式已启动")
    
    def detect_red_rectangles(self, image: np.ndarray, image_name: str = "base_image") -> List[Dict]:
        """
        检测红色矩形框（基准图片）
        
        Args:
            image: 输入图像
            image_name: 图像名称
            
        Returns:
            检测到的红色矩形框列表
        """
        if self.debug_mode:
            print(f"\n🔴 开始检测红色矩形框 - {image_name}")
            self.logger.info(f"开始检测红色矩形框: {image_name}")
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义红色范围（HSV）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_rectangles = []
        for i, contour in enumerate(contours):
            # 计算矩形边界框
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 过滤小的区域
            if area > 500:
                rect_info = {
                    "id": f"red_{i}",
                    "x": int(x),
                    "y": int(y), 
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "center": (int(x + w/2), int(y + h/2))
                }
                red_rectangles.append(rect_info)
                
                if self.debug_mode:
                    print(f"   ✅ 检测到红色矩形 #{i}: 位置({x},{y}) 尺寸({w}x{h}) 面积:{area}")
        
        # 保存调试图像
        if self.debug_mode:
            debug_image = image.copy()
            for rect in red_rectangles:
                cv2.rectangle(debug_image, 
                            (rect["x"], rect["y"]), 
                            (rect["x"] + rect["width"], rect["y"] + rect["height"]), 
                            (0, 0, 255), 2)
                cv2.putText(debug_image, rect["id"], 
                          (rect["x"], rect["y"]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(f'{self.debug_output_dir}/images/red_rectangles_{image_name}.png', debug_image)
            cv2.imwrite(f'{self.debug_output_dir}/images/red_mask_{image_name}.png', red_mask)
            
            print(f"   📸 调试图像已保存: red_rectangles_{image_name}.png")
            print(f"   📸 红色掩码已保存: red_mask_{image_name}.png")
            print(f"   📊 检测统计: 共检测到 {len(red_rectangles)} 个红色矩形框")
        
        self.detection_results["red_rectangles"].extend(red_rectangles)
        return red_rectangles
    
    def detect_brown_rectangles_and_cyan_points(self, image: np.ndarray, image_name: str = "compare_image") -> Tuple[List[Dict], List[Dict]]:
        """
        检测褐色矩形框和青色点（对比图片）
        
        Args:
            image: 输入图像
            image_name: 图像名称
            
        Returns:
            (褐色矩形框列表, 青色点列表)
        """
        if self.debug_mode:
            print(f"\n🟤 开始检测褐色矩形框和青色点 - {image_name}")
            self.logger.info(f"开始检测褐色矩形框和青色点: {image_name}")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测褐色矩形框 - 扩大颜色范围
        lower_brown = np.array([5, 40, 40])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 检测青色点
        lower_cyan = np.array([85, 50, 50])
        upper_cyan = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # 检测褐色矩形框
        brown_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brown_rectangles = []
        
        for i, contour in enumerate(brown_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > 300:
                rect_info = {
                    "id": f"brown_{i}",
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "center": (int(x + w/2), int(y + h/2))
                }
                brown_rectangles.append(rect_info)
                
                if self.debug_mode:
                    print(f"   ✅ 检测到褐色矩形 #{i}: 位置({x},{y}) 尺寸({w}x{h}) 面积:{area}")
        
        # 检测青色点
        cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cyan_points = []
        
        for i, contour in enumerate(cyan_contours):
            # 计算轮廓的中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                
                if area > 10:  # 过滤噪声点
                    point_info = {
                        "id": f"cyan_{i}",
                        "x": cx,
                        "y": cy,
                        "area": float(area)
                    }
                    cyan_points.append(point_info)
                    
                    if self.debug_mode:
                        print(f"   ✅ 检测到青色点 #{i}: 位置({cx},{cy}) 面积:{area:.2f}")
        
        # 保存调试图像
        if self.debug_mode:
            debug_image = image.copy()
            
            # 绘制褐色矩形框
            for rect in brown_rectangles:
                cv2.rectangle(debug_image,
                            (rect["x"], rect["y"]),
                            (rect["x"] + rect["width"], rect["y"] + rect["height"]),
                            (42, 42, 165), 2)  # 褐色
                cv2.putText(debug_image, rect["id"],
                          (rect["x"], rect["y"]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (42, 42, 165), 1)
            
            # 绘制青色点
            for point in cyan_points:
                cv2.circle(debug_image, (point["x"], point["y"]), 5, (255, 255, 0), -1)  # 青色
                cv2.putText(debug_image, point["id"],
                          (point["x"]+10, point["y"]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imwrite(f'{self.debug_output_dir}/images/brown_cyan_{image_name}.png', debug_image)
            cv2.imwrite(f'{self.debug_output_dir}/images/brown_mask_{image_name}.png', brown_mask)
            cv2.imwrite(f'{self.debug_output_dir}/images/cyan_mask_{image_name}.png', cyan_mask)
            
            print(f"   📸 调试图像已保存: brown_cyan_{image_name}.png")
            print(f"   📸 褐色掩码已保存: brown_mask_{image_name}.png")
            print(f"   📸 青色掩码已保存: cyan_mask_{image_name}.png")
            print(f"   📊 检测统计: 褐色矩形框 {len(brown_rectangles)} 个, 青色点 {len(cyan_points)} 个")
        
        self.detection_results["brown_rectangles"].extend(brown_rectangles)
        self.detection_results["cyan_points"].extend(cyan_points)
        
        return brown_rectangles, cyan_points
    
    def calculate_z_values(self, red_rectangles: List[Dict], brown_rectangles: List[Dict]) -> Tuple[List[float], List[float]]:
        """
        计算Z1和Z2值
        
        Args:
            red_rectangles: 红色矩形框列表
            brown_rectangles: 褐色矩形框列表
            
        Returns:
            (Z1值列表, Z2值列表)
        """
        if self.debug_mode:
            print(f"\n📐 开始计算Z1和Z2值")
            self.logger.info("开始计算Z1和Z2值")
        
        z1_values = []
        z2_values = []
        
        # Z1值计算（基于红色矩形框的面积比）
        total_red_area = sum(rect["area"] for rect in red_rectangles)
        for rect in red_rectangles:
            if total_red_area > 0:
                z1 = rect["area"] / total_red_area
                z1_values.append(z1)
                
                if self.debug_mode:
                    print(f"   🔴 红色矩形 {rect['id']}: Z1 = {z1:.6f} (面积 {rect['area']} / 总面积 {total_red_area})")
        
        # Z2值计算（基于褐色矩形框的面积比）
        total_brown_area = sum(rect["area"] for rect in brown_rectangles)
        for rect in brown_rectangles:
            if total_brown_area > 0:
                z2 = rect["area"] / total_brown_area
                z2_values.append(z2)
                
                if self.debug_mode:
                    print(f"   🟤 褐色矩形 {rect['id']}: Z2 = {z2:.6f} (面积 {rect['area']} / 总面积 {total_brown_area})")
        
        if self.debug_mode:
            print(f"   📊 Z值统计: Z1平均值 = {np.mean(z1_values):.6f}, Z2平均值 = {np.mean(z2_values):.6f}")
        
        self.detection_results["z1_values"] = z1_values
        self.detection_results["z2_values"] = z2_values
        
        return z1_values, z2_values
    
    def calculate_first_layer_accuracy(self, z1_values: List[float], z2_values: List[float]) -> List[float]:
        """
        计算第一层准确率: P = 1 - |Z1 - Z2| / MAX(Z1, Z2)
        
        Args:
            z1_values: Z1值列表
            z2_values: Z2值列表
            
        Returns:
            第一层准确率列表
        """
        if self.debug_mode:
            print(f"\n🎯 计算第一层准确率: P = 1 - |Z1 - Z2| / MAX(Z1, Z2)")
            self.logger.info("开始计算第一层准确率")
        
        first_layer_accuracies = []
        
        # 确保两个列表长度一致
        min_len = min(len(z1_values), len(z2_values))
        
        for i in range(min_len):
            z1 = z1_values[i]
            z2 = z2_values[i]
            
            if max(z1, z2) > 0:
                accuracy = 1 - abs(z1 - z2) / max(z1, z2)
                first_layer_accuracies.append(accuracy)
                
                if self.debug_mode:
                    print(f"   📊 配对 #{i+1}: Z1={z1:.6f}, Z2={z2:.6f} => P={accuracy:.6f}")
            else:
                first_layer_accuracies.append(0.0)
                if self.debug_mode:
                    print(f"   ⚠️  配对 #{i+1}: Z1={z1:.6f}, Z2={z2:.6f} => P=0.000000 (分母为0)")
        
        avg_accuracy = np.mean(first_layer_accuracies) if first_layer_accuracies else 0.0
        
        if self.debug_mode:
            print(f"   🎯 第一层准确率统计:")
            print(f"      - 配对数量: {len(first_layer_accuracies)}")
            print(f"      - 平均准确率: {avg_accuracy:.6f}")
            print(f"      - 最高准确率: {max(first_layer_accuracies):.6f}" if first_layer_accuracies else "      - 最高准确率: N/A")
            print(f"      - 最低准确率: {min(first_layer_accuracies):.6f}" if first_layer_accuracies else "      - 最低准确率: N/A")
        
        self.detection_results["first_layer_accuracy"] = first_layer_accuracies
        return first_layer_accuracies
    
    def calculate_second_layer_accuracy(self, red_rectangles: List[Dict], brown_rectangles: List[Dict], cyan_points: List[Dict]) -> float:
        """
        计算第二层准确率: P2 = (Σi=1^n Si/Mi) / n
        
        Args:
            red_rectangles: 红色矩形框列表
            brown_rectangles: 褐色矩形框列表
            cyan_points: 青色点列表
            
        Returns:
            第二层准确率
        """
        if self.debug_mode:
            print(f"\n🎯 计算第二层准确率: P2 = (Σi=1^n Si/Mi) / n")
            self.logger.info("开始计算第二层准确率")
        
        if not red_rectangles:
            if self.debug_mode:
                print("   ⚠️  没有红色矩形框，第二层准确率为0")
            return 0.0
        
        sum_ratios = 0.0
        valid_matches = 0
        
        for i, red_rect in enumerate(red_rectangles):
            # 计算该红色矩形框内的匹配情况
            matches_in_red = 0
            total_targets_in_red = 0
            
            # 检查褐色矩形框是否在红色矩形框内
            for brown_rect in brown_rectangles:
                if self._is_inside_rectangle(brown_rect["center"], red_rect):
                    matches_in_red += 1
                    total_targets_in_red += 1
            
            # 检查青色点是否在红色矩形框内
            for cyan_point in cyan_points:
                if self._is_inside_rectangle((cyan_point["x"], cyan_point["y"]), red_rect):
                    total_targets_in_red += 1
            
            # 计算比例 Si/Mi
            if total_targets_in_red > 0:
                ratio = matches_in_red / total_targets_in_red
                sum_ratios += ratio
                valid_matches += 1
                
                if self.debug_mode:
                    print(f"   📊 红色矩形 {red_rect['id']}: 匹配{matches_in_red}/总计{total_targets_in_red} = {ratio:.6f}")
            else:
                if self.debug_mode:
                    print(f"   📊 红色矩形 {red_rect['id']}: 无目标对象")
        
        # 计算第二层准确率
        second_layer_accuracy = sum_ratios / valid_matches if valid_matches > 0 else 0.0
        
        if self.debug_mode:
            print(f"   🎯 第二层准确率统计:")
            print(f"      - 有效匹配数: {valid_matches}")
            print(f"      - 比例总和: {sum_ratios:.6f}")
            print(f"      - 第二层准确率: {second_layer_accuracy:.6f}")
        
        self.detection_results["second_layer_accuracy"] = second_layer_accuracy
        return second_layer_accuracy
    
    def _is_inside_rectangle(self, point: Tuple[int, int], rectangle: Dict) -> bool:
        """
        检查点是否在矩形内
        
        Args:
            point: 点坐标 (x, y)
            rectangle: 矩形信息字典
            
        Returns:
            是否在矩形内
        """
        x, y = point
        return (rectangle["x"] <= x <= rectangle["x"] + rectangle["width"] and
                rectangle["y"] <= y <= rectangle["y"] + rectangle["height"])
    
    def exclude_cyan_only_red_rectangles(self, red_rectangles: List[Dict], brown_rectangles: List[Dict], cyan_points: List[Dict]) -> List[Dict]:
        """
        排除只包含青色点的红色框
        
        Args:
            red_rectangles: 红色矩形框列表
            brown_rectangles: 褐色矩形框列表
            cyan_points: 青色点列表
            
        Returns:
            过滤后的红色矩形框列表
        """
        if self.debug_mode:
            print(f"\n🔍 执行排除逻辑: 排除只包含青色点的红色框")
            self.logger.info("开始执行排除只包含青色点的红色框逻辑")
        
        filtered_red_rectangles = []
        excluded_rectangles = []
        
        for red_rect in red_rectangles:
            has_brown_rectangle = False
            has_cyan_point = False
            
            # 检查是否包含褐色矩形框
            for brown_rect in brown_rectangles:
                if self._is_inside_rectangle(brown_rect["center"], red_rect):
                    has_brown_rectangle = True
                    break
            
            # 检查是否包含青色点
            for cyan_point in cyan_points:
                if self._is_inside_rectangle((cyan_point["x"], cyan_point["y"]), red_rect):
                    has_cyan_point = True
                    break
            
            # 排除逻辑：如果只包含青色点而没有褐色矩形框，则排除
            if has_cyan_point and not has_brown_rectangle:
                excluded_rectangles.append(red_rect)
                if self.debug_mode:
                    print(f"   ❌ 排除红色矩形 {red_rect['id']}: 只包含青色点，无褐色矩形框")
            else:
                filtered_red_rectangles.append(red_rect)
                if self.debug_mode:
                    status = "包含褐色矩形框" if has_brown_rectangle else "不包含任何目标"
                    print(f"   ✅ 保留红色矩形 {red_rect['id']}: {status}")
        
        if self.debug_mode:
            print(f"   📊 排除统计:")
            print(f"      - 原始红色矩形框: {len(red_rectangles)}")
            print(f"      - 排除的矩形框: {len(excluded_rectangles)}")
            print(f"      - 保留的矩形框: {len(filtered_red_rectangles)}")
        
        self.detection_results["excluded_red_rectangles"] = excluded_rectangles
        return filtered_red_rectangles
    
    def generate_matching_statistics(self, red_rectangles: List[Dict], brown_rectangles: List[Dict], cyan_points: List[Dict]):
        """
        生成匹配统计信息
        
        Args:
            red_rectangles: 红色矩形框列表
            brown_rectangles: 褐色矩形框列表
            cyan_points: 青色点列表
        """
        if self.debug_mode:
            print(f"\n📈 生成匹配统计信息")
            self.logger.info("开始生成匹配统计信息")
        
        statistics = {
            "detection_summary": {
                "red_rectangles_count": len(red_rectangles),
                "brown_rectangles_count": len(brown_rectangles),
                "cyan_points_count": len(cyan_points),
                "excluded_red_rectangles_count": len(self.detection_results["excluded_red_rectangles"])
            },
            "accuracy_summary": {
                "first_layer_accuracies": self.detection_results["first_layer_accuracy"],
                "first_layer_avg_accuracy": np.mean(self.detection_results["first_layer_accuracy"]) if self.detection_results["first_layer_accuracy"] else 0.0,
                "second_layer_accuracy": self.detection_results["second_layer_accuracy"]
            },
            "z_values_summary": {
                "z1_values": self.detection_results["z1_values"],
                "z2_values": self.detection_results["z2_values"],
                "z1_avg": np.mean(self.detection_results["z1_values"]) if self.detection_results["z1_values"] else 0.0,
                "z2_avg": np.mean(self.detection_results["z2_values"]) if self.detection_results["z2_values"] else 0.0
            },
            "processing_timestamp": datetime.datetime.now().isoformat()
        }
        
        self.detection_results["matching_statistics"] = statistics
        
        if self.debug_mode:
            print(f"   📊 检测汇总:")
            print(f"      - 红色矩形框: {statistics['detection_summary']['red_rectangles_count']} 个")
            print(f"      - 褐色矩形框: {statistics['detection_summary']['brown_rectangles_count']} 个")
            print(f"      - 青色点: {statistics['detection_summary']['cyan_points_count']} 个")
            print(f"      - 排除的红色框: {statistics['detection_summary']['excluded_red_rectangles_count']} 个")
            
            print(f"   🎯 准确率汇总:")
            print(f"      - 第一层平均准确率: {statistics['accuracy_summary']['first_layer_avg_accuracy']:.6f}")
            print(f"      - 第二层准确率: {statistics['accuracy_summary']['second_layer_accuracy']:.6f}")
            
            print(f"   📐 Z值汇总:")
            print(f"      - Z1平均值: {statistics['z_values_summary']['z1_avg']:.6f}")
            print(f"      - Z2平均值: {statistics['z_values_summary']['z2_avg']:.6f}")
    
    def save_debug_data(self):
        """保存调试数据到文件"""
        if not self.debug_mode:
            return
        
        # 保存详细检测结果
        results_file = f'{self.debug_output_dir}/data/detection_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.detection_results, f, ensure_ascii=False, indent=2)
        
        # 保存匹配统计
        stats_file = f'{self.debug_output_dir}/data/matching_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.detection_results["matching_statistics"], f, ensure_ascii=False, indent=2)
        
        # 生成分析报告
        self._generate_analysis_report()
        
        print(f"\n💾 调试数据保存完成:")
        print(f"   📄 检测结果: {results_file}")
        print(f"   📄 匹配统计: {stats_file}")
        print(f"   📄 分析报告: {self.debug_output_dir}/analysis/analysis_report.txt")
        
        self.logger.info("调试数据保存完成")
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        report_file = f'{self.debug_output_dir}/analysis/analysis_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("基于公式的双重对比逻辑矩形检测程序 - 调试模式分析报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            stats = self.detection_results["matching_statistics"]
            
            f.write("1. 检测结果汇总\n")
            f.write("-" * 40 + "\n")
            f.write(f"红色矩形框数量: {stats['detection_summary']['red_rectangles_count']}\n")
            f.write(f"褐色矩形框数量: {stats['detection_summary']['brown_rectangles_count']}\n")
            f.write(f"青色点数量: {stats['detection_summary']['cyan_points_count']}\n")
            f.write(f"排除的红色框数量: {stats['detection_summary']['excluded_red_rectangles_count']}\n\n")
            
            f.write("2. 准确率分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"第一层准确率 (P = 1 - |Z1 - Z2| / MAX(Z1, Z2)):\n")
            f.write(f"  平均值: {stats['accuracy_summary']['first_layer_avg_accuracy']:.6f}\n")
            if stats['accuracy_summary']['first_layer_accuracies']:
                f.write(f"  最大值: {max(stats['accuracy_summary']['first_layer_accuracies']):.6f}\n")
                f.write(f"  最小值: {min(stats['accuracy_summary']['first_layer_accuracies']):.6f}\n")
            f.write(f"\n第二层准确率 (P2 = (Σi=1^n Si/Mi) / n):\n")
            f.write(f"  值: {stats['accuracy_summary']['second_layer_accuracy']:.6f}\n\n")
            
            f.write("3. Z值分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"Z1值 (基于红色矩形框面积比):\n")
            f.write(f"  平均值: {stats['z_values_summary']['z1_avg']:.6f}\n")
            f.write(f"  数量: {len(stats['z_values_summary']['z1_values'])}\n")
            f.write(f"\nZ2值 (基于褐色矩形框面积比):\n")
            f.write(f"  平均值: {stats['z_values_summary']['z2_avg']:.6f}\n")
            f.write(f"  数量: {len(stats['z_values_summary']['z2_values'])}\n\n")
            
            f.write("4. 生成的调试文件\n")
            f.write("-" * 40 + "\n")
            f.write("调试图片文件:\n")
            f.write("  - red_rectangles_*.png: 红色矩形框检测结果\n")
            f.write("  - red_mask_*.png: 红色颜色掩码\n")
            f.write("  - brown_cyan_*.png: 褐色矩形框和青色点检测结果\n")
            f.write("  - brown_mask_*.png: 褐色颜色掩码\n")
            f.write("  - cyan_mask_*.png: 青色颜色掩码\n\n")
            f.write("数据文件:\n")
            f.write("  - detection_results.json: 详细检测结果\n")
            f.write("  - matching_statistics.json: 匹配统计信息\n")
            f.write("  - debug.log: 调试日志\n\n")
            f.write("分析文件:\n")
            f.write("  - analysis_report.txt: 本分析报告\n")
    
    def process_images(self, base_image_path: str, compare_image_path: str):
        """
        处理图像的完整流程
        
        Args:
            base_image_path: 基准图片路径
            compare_image_path: 对比图片路径
        """
        if self.debug_mode:
            self.debug_mode_output()
        
        try:
            # 读取图像
            base_image = cv2.imread(base_image_path)
            compare_image = cv2.imread(compare_image_path)
            
            if base_image is None:
                raise FileNotFoundError(f"无法加载基准图片: {base_image_path}")
            if compare_image is None:
                raise FileNotFoundError(f"无法加载对比图片: {compare_image_path}")
            
            if self.debug_mode:
                print(f"\n📂 图像加载成功:")
                print(f"   基准图片: {base_image_path} ({base_image.shape[1]}x{base_image.shape[0]})")
                print(f"   对比图片: {compare_image_path} ({compare_image.shape[1]}x{compare_image.shape[0]})")
            
            # 检测红色矩形框
            red_rectangles = self.detect_red_rectangles(base_image, "base")
            
            # 检测褐色矩形框和青色点
            brown_rectangles, cyan_points = self.detect_brown_rectangles_and_cyan_points(compare_image, "compare")
            
            # 排除只包含青色点的红色框
            filtered_red_rectangles = self.exclude_cyan_only_red_rectangles(red_rectangles, brown_rectangles, cyan_points)
            
            # 计算Z值
            z1_values, z2_values = self.calculate_z_values(filtered_red_rectangles, brown_rectangles)
            
            # 计算第一层准确率
            first_layer_accuracies = self.calculate_first_layer_accuracy(z1_values, z2_values)
            
            # 计算第二层准确率
            second_layer_accuracy = self.calculate_second_layer_accuracy(filtered_red_rectangles, brown_rectangles, cyan_points)
            
            # 生成匹配统计
            self.generate_matching_statistics(filtered_red_rectangles, brown_rectangles, cyan_points)
            
            # 保存调试数据
            if self.debug_mode:
                self.save_debug_data()
                
                print(f"\n🎉 程序执行完成!")
                print(f"⏰ 完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
        
        except Exception as e:
            if self.debug_mode:
                print(f"\n❌ 程序执行出错: {str(e)}")
                self.logger.error(f"程序执行出错: {str(e)}")
            raise

def main():
    """主函数 - 演示程序"""
    # 创建调试模式的检测器
    detector = RectangleDetector(debug_mode=True)
    
    # 创建示例图像用于演示
    print("📝 正在创建示例图像用于演示...")
    
    # 创建基准图片 (包含红色矩形框)
    base_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(base_image, (50, 50), (200, 150), (0, 0, 255), 3)  # 红色矩形框
    cv2.rectangle(base_image, (300, 200), (450, 300), (0, 0, 255), 3)  # 红色矩形框
    cv2.imwrite('base_image_example.png', base_image)
    
    # 创建对比图片 (包含褐色矩形框和青色点)
    compare_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(compare_image, (60, 60), (190, 140), (42, 42, 165), -1)  # 褐色矩形框
    cv2.circle(compare_image, (320, 220), 8, (255, 255, 0), -1)  # 青色点
    cv2.circle(compare_image, (400, 250), 6, (255, 255, 0), -1)  # 青色点
    cv2.imwrite('compare_image_example.png', compare_image)
    
    # 处理图像
    detector.process_images('base_image_example.png', 'compare_image_example.png')

if __name__ == "__main__":
    main()