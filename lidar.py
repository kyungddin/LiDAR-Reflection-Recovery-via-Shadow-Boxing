#!/usr/bin/env python3
# LiDAR 거울 탐지 알고리즘 (CUDA Accelerated with PyTorch)

########################################################################################################################
# 1. Import Library
########################################################################################################################

# 1. Python 표준 라이브러리
from threading import Lock
# 2. 서드파티 라이브러리 (계산 및 데이터 처리)
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.t.geometry as o3dt
import torch
from scipy.spatial.transform import Rotation, Slerp
# 3. ROS 핵심 및 유틸리티 라이브러리
import rospy
import sensor_msgs.point_cloud2 as pc2
# 4. ROS 메시지 타입
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


########################################################################################################################
# 2. 클래스 및 초기화, 콜백, 유틸리티 메서드
########################################################################################################################

class MirrorDetector:
    def __init__(self):
        """클래스 초기화, ROS 노드, 파라미터, Publisher/Subscriber 설정"""
        rospy.loginfo("▶ 거울 탐지 및 복원 노드 시작 (CUDA with PyTorch)")

        # --- 상태 변수 초기화 ---
        self.points_lock = Lock()
        self.first_return_pcd = None
        self.second_return_pcd_gpu = None
        self.last_mirror_state = None
        self.last_restored_pcd = None
        self.frames_since_detection = 0
        self.frames_since_restoration = 0

        # --- FPS 계산용 변수들 ---
        self.fps_log_interval = 2.0
        self.fps_frame_count = 0
        self.fps_start_time = None

        # --- CUDA 장치 설정 ---
        self.o3d_device = o3c.Device("CUDA:0")
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using PyTorch device: {self.torch_device}")

        # --- ROS 파라미터 불러오기 ---
        self._load_params()

        # --- ROS Publisher 및 Subscriber 설정 ---
        self._setup_ros_communications()

    def _load_params(self):
        """ROS 파라미터 서버에서 설정값들을 불러옵니다."""
        self.MAX_DISTANCE_THRESHOLD = rospy.get_param("~filtering/max_distance_threshold", 2.0)
        self.PLANE_REFINEMENT_ENABLED = rospy.get_param("~plane/refinement_enabled", True)
        self.MANUAL_YAW_CORRECTION_DEGREES = rospy.get_param("~plane/manual_yaw_correction_degrees", 0.0)
        self.DETECTION_TTL = rospy.get_param("~smoothing/detection_ttl", 40)
        self.SMOOTHING_FACTOR = rospy.get_param("~smoothing/smoothing_factor", 0.15)
        # <<< 파라미터 수정 >>>
        self.MAX_TRANSLATIONAL_VELOCITY = rospy.get_param("~gate/max_translational_velocity", 0.8)
        self.MAX_ROTATIONAL_VELOCITY_DEG = rospy.get_param("~gate/max_rotational_velocity_deg", 25.0)
        # <<< 파라미터 수정 끝 >>>
        self.CULLING_DISTANCE_FROM_MIRROR = rospy.get_param("~restoration/culling_distance", 0.5)
        self.Z_CORRECTION_FACTOR = rospy.get_param("~restoration/z_correction_factor", 1.5)
        self.RESTORATION_TTL = rospy.get_param("~smoothing/restoration_ttl", 10)

    def _setup_ros_communications(self):
        """Publisher와 Subscriber를 초기화합니다."""
        rospy.Subscriber('/ouster/points', PointCloud2, self._points1_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber('/ouster/points2', PointCloud2, self._points2_callback, queue_size=1, buff_size=2 ** 24)
        self.marker_pub = rospy.Publisher('/mirror_bounding_box', Marker, queue_size=10)
        self.filtered_points1_pub = rospy.Publisher('/filtered_points1', PointCloud2, queue_size=2)
        self.filtered_points2_pub = rospy.Publisher('/filtered_points2', PointCloud2, queue_size=2)
        self.restored_points_pub = rospy.Publisher('/restored_points', PointCloud2, queue_size=2)
        self.final_points_pub = rospy.Publisher('/final_points', PointCloud2, queue_size=2)

    def _points1_callback(self, msg):
        """/ouster/points 토픽 콜백 함수"""
        if self.fps_start_time is None:
            self.fps_start_time = rospy.Time.now()
        self.fps_frame_count += 1
        elapsed_time = (rospy.Time.now() - self.fps_start_time).to_sec()
        if elapsed_time >= self.fps_log_interval:
            avg_fps = self.fps_frame_count / elapsed_time
            rospy.loginfo(f"Average FPS: {avg_fps:.2f} (over {elapsed_time:.2f} seconds)")
            self.fps_frame_count = 0
            self.fps_start_time = rospy.Time.now()

        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points.shape[0] == 0: return
        with self.points_lock:
            self.first_return_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd1_down = self.first_return_pcd.voxel_down_sample(voxel_size=0.05)
        self.filtered_points1_pub.publish(self._o3d_to_pointcloud2(pcd1_down, frame_id=msg.header.frame_id))
        self._process_mirror_detection(msg.header.frame_id)

    def _points2_callback(self, msg):
        """/ouster/points2 토픽 콜백 함수"""
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points.shape[0] > 0:
            mask = np.sum(points ** 2, axis=1) < self.MAX_DISTANCE_THRESHOLD ** 2
            points = points[mask]
        pcd_cpu = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        with self.points_lock:
            self.second_return_pcd_gpu = o3dt.PointCloud.from_legacy(pcd_cpu, device=self.o3d_device)

    def _o3d_to_pointcloud2(self, o3d_cloud, frame_id="ouster"):
        """Open3D Cloud -> Ros Cloud"""
        points = np.asarray(o3d_cloud.points)
        header = rospy.Header(stamp=rospy.Time.now(), frame_id=frame_id)
        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        if o3d_cloud.has_colors():
            fields.append(PointField('rgb', 12, PointField.UINT32, 1))
            dtype_list.append(('rgb', np.uint32))
        packed_points = np.zeros(len(points), dtype=dtype_list)
        packed_points['x'], packed_points['y'], packed_points['z'] = points[:, 0], points[:, 1], points[:, 2]
        if o3d_cloud.has_colors():
            colors = (np.asarray(o3d_cloud.colors) * 255).astype(np.uint8)
            packed_points['rgb'] = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | \
                                   colors[:, 2].astype(np.uint32)
        return pc2.create_cloud(header, fields, packed_points)

    ########################################################################################################################
    # 3. 유틸리티 메서드 (시각화 등)
    ########################################################################################################################

    def _publish_bounding_box(self, center, extent, orientation_q, frame_id="ouster", marker_id=0, normal_vector=None):
        header = rospy.Header(frame_id=frame_id, stamp=rospy.Time.now())
        marker = Marker(header=header, ns="bounding_box", id=marker_id, type=Marker.CUBE, action=Marker.ADD,
                        pose=Pose(Point(*center), Quaternion(*orientation_q)), scale=Point(extent[0], extent[1], 0.01),
                        color=ColorRGBA(0.0, 0.0, 1.0, 0.5), lifetime=rospy.Duration(0.5))
        self.marker_pub.publish(marker)
        if normal_vector is not None:
            for i, (text, color, offset) in enumerate(
                    [("Front", ColorRGBA(0, 1, 0, 0.8), -0.15), ("Back", ColorRGBA(1, 0, 0, 0.8), 0.15)]):
                text_marker = Marker(header=header, ns="front_back_text", id=marker_id * 2 + i,
                                     type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                                     pose=Pose(Point(*(center + normal_vector * offset)), Quaternion(w=1.0)),
                                     scale=Point(z=0.15), color=color, text=text, lifetime=rospy.Duration(0.5))
                self.marker_pub.publish(text_marker)

    def _publish_shadow_box(self, search_obbs, frame_id="ouster"):
        for i, obb in enumerate(search_obbs):
            q = Rotation.from_matrix(obb.R.copy()).as_quat()
            marker = Marker(header=rospy.Header(frame_id=frame_id, stamp=rospy.Time.now()), ns="shadow_box", id=i,
                            type=Marker.CUBE, action=Marker.ADD, pose=Pose(Point(*obb.center), Quaternion(*q)),
                            scale=Point(*obb.extent), color=ColorRGBA(0.0, 0.7, 0.3, 0.4), lifetime=rospy.Duration(0.5))
            self.marker_pub.publish(marker)

    def _clear_all_markers(self):
        self.marker_pub.publish(Marker(action=Marker.DELETEALL))

    def run(self):
        rospy.spin()

    ########################################################################################################################
    # 4. 메인 로직 및 하위 메서드
    ########################################################################################################################

    def _process_mirror_detection(self, frame_id):
        """거울 탐지 및 포인트 복원 메인 로직을 관리하고 각 기능별 메서드를 호출합니다."""
        with self.points_lock:
            pcd2_gpu = self.second_return_pcd_gpu
            pcd1_cpu = self.first_return_pcd
            if pcd2_gpu is None or pcd1_cpu is None or pcd2_gpu.is_empty() or not pcd1_cpu.has_points():
                return
            pcd1_tree = o3d.geometry.KDTreeFlann(pcd1_cpu)

        candidate = self._find_mirror_candidate(pcd1_cpu, pcd2_gpu, pcd1_tree, frame_id)
        self._update_mirror_state(candidate)

        if self.frames_since_detection < self.DETECTION_TTL and self.last_mirror_state is not None:
            self._restore_points_and_publish(self.last_mirror_state, pcd1_cpu, frame_id)
        else:
            self._handle_detection_loss(frame_id)

    def _find_mirror_candidate(self, pcd1, pcd2_gpu, pcd1_tree, frame_id):
        """GPU를 사용하여 pcd2에서 평면 클러스터를 찾아 거울 후보 정보를 반환합니다."""
        # <<< 파라미터 수정 >>>
        labels_tensor = pcd2_gpu.cluster_dbscan(eps=0.15, min_points=40)
        # <<< 파라미터 수정 끝 >>>
        labels_np = labels_tensor.cpu().numpy()
        unique_labels = np.unique(labels_np[labels_np != -1])
        pcd2_cpu = pcd2_gpu.to_legacy()

        if len(unique_labels) > 0:
            denoised_indices = np.where(labels_np != -1)[0]
            pcd2_denoised = pcd2_cpu.select_by_index(denoised_indices)
            pcd2_denoised.paint_uniform_color([0, 1, 0])
            self.filtered_points2_pub.publish(self._o3d_to_pointcloud2(pcd2_denoised, frame_id=frame_id))

        for label in unique_labels:
            cluster_indices = np.where(labels_np == label)[0]
            cluster_pcd = pcd2_cpu.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            if len(points) < 50: continue

            mean, cov_matrix = np.mean(points, axis=0), np.cov(points - mean, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sum_eigenvalues = np.sum(eigenvalues)
            if sum_eigenvalues < 1e-6: continue
            planarity = np.min(eigenvalues) / sum_eigenvalues
            if planarity > 0.01: continue

            cluster_pcd_gpu = o3dt.PointCloud.from_legacy(cluster_pcd, device=self.o3d_device)
            plane_model_tensor, inliers_tensor = cluster_pcd_gpu.segment_plane(distance_threshold=0.02, ransac_n=3,
                                                                               num_iterations=1000)
            inliers = inliers_tensor.cpu().numpy()
            plane_model = plane_model_tensor.cpu().numpy()

            if len(inliers) < 15: continue
            pcd2_inliers = cluster_pcd.select_by_index(inliers)

            mirror_candidate_pcd = pcd2_inliers
            if self.PLANE_REFINEMENT_ENABLED:
                pcd1_neighbor_indices = [idx[0] for point in pcd2_inliers.points for k, idx, _ in
                                         [pcd1_tree.search_knn_vector_3d(point, 1)] if k > 0]
                if pcd1_neighbor_indices:
                    pcd1_neighbors = pcd1.select_by_index(list(set(pcd1_neighbor_indices)))
                    combined_pcd = pcd2_inliers + pcd1_neighbors
                    combined_pcd_gpu = o3dt.PointCloud.from_legacy(combined_pcd, device=self.o3d_device)
                    plane_model_tensor, inliers_tensor = combined_pcd_gpu.segment_plane(distance_threshold=0.02,
                                                                                        ransac_n=3, num_iterations=1000)
                    inliers = inliers_tensor.cpu().numpy()
                    plane_model = plane_model_tensor.cpu().numpy()
                    if len(inliers) >= 15:
                        mirror_candidate_pcd = combined_pcd.select_by_index(inliers)

            center = mirror_candidate_pcd.get_center()
            normal_vec = plane_model[:3] / np.linalg.norm(plane_model[:3])

            temp_z_axis = normal_vec
            ref_vec = np.array([1, 0, 0]) if np.abs(np.dot(temp_z_axis, [0, 0, 1])) > 0.95 else np.array([0, 0, 1])
            temp_y_axis = np.cross(temp_z_axis, ref_vec);
            temp_y_axis /= np.linalg.norm(temp_y_axis)
            temp_x_axis = np.cross(temp_y_axis, temp_z_axis);
            temp_x_axis /= np.linalg.norm(temp_x_axis)
            temp_rotation_matrix = np.stack([temp_x_axis, temp_y_axis, temp_z_axis], axis=1)
            transformed_points = np.dot(np.asarray(mirror_candidate_pcd.points) - center, temp_rotation_matrix)
            extent = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
            z_axis = normal_vec
            if np.linalg.norm(center + z_axis * (extent[2] / 2.0)) > np.linalg.norm(
                center - z_axis * (extent[2] / 2.0)): z_axis = -z_axis
            ref_vec = np.array([1, 0, 0]) if np.abs(np.dot(z_axis, [0, 0, 1])) > 0.95 else np.array([0, 0, 1])
            y_axis = np.cross(z_axis, ref_vec);
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis);
            x_axis /= np.linalg.norm(x_axis)
            rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

            if np.sort(extent)[1] < 0.3: continue

            current_rot = Rotation.from_matrix(rotation_matrix)
            return {"center": center, "extent": extent, "rotation": current_rot, "plane_model": plane_model}

        return None

    def _update_mirror_state(self, candidate):
        if candidate is None:
            self.frames_since_detection += 1
            return
        if self.last_mirror_state is None:
            self.last_mirror_state = candidate
        else:
            dist_diff = np.linalg.norm(candidate["center"] - self.last_mirror_state["center"])
            angle_diff_deg = np.linalg.norm(
                (self.last_mirror_state["rotation"].inv() * candidate["rotation"]).as_rotvec()) * 180 / np.pi
            if dist_diff > self.MAX_TRANSLATIONAL_VELOCITY or angle_diff_deg > self.MAX_ROTATIONAL_VELOCITY_DEG:
                rospy.logwarn(f"Outlier detected. Ignoring. Dist: {dist_diff:.2f}m, Angle: {angle_diff_deg:.2f}deg")
                self.frames_since_detection += 1
                return
            alpha = self.SMOOTHING_FACTOR
            self.last_mirror_state["center"] = alpha * candidate["center"] + (1 - alpha) * self.last_mirror_state[
                "center"]
            self.last_mirror_state["extent"] = alpha * candidate["extent"] + (1 - alpha) * self.last_mirror_state[
                "extent"]
            self.last_mirror_state["plane_model"] = alpha * candidate["plane_model"] + (1 - alpha) * \
                                                    self.last_mirror_state["plane_model"]
            slerp = Slerp([0, 1], Rotation.from_matrix(
                [self.last_mirror_state["rotation"].as_matrix(), candidate["rotation"].as_matrix()]))
            self.last_mirror_state["rotation"] = slerp([alpha])[0]
        self.frames_since_detection = 0

    def _handle_detection_loss(self, frame_id):
        if self.frames_since_detection >= self.DETECTION_TTL:
            self.last_mirror_state, self.last_restored_pcd = None, None
            self._clear_all_markers()
            self.filtered_points2_pub.publish(self._o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id))
            self.restored_points_pub.publish(self._o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id))
        if self.first_return_pcd:
            pcd1_down = self.first_return_pcd.voxel_down_sample(voxel_size=0.05)
            self.final_points_pub.publish(self._o3d_to_pointcloud2(pcd1_down, frame_id))

    def _restore_points_and_publish(self, state, pcd1, frame_id):
        """포인트 복원 및 모든 결과 발행을 처리합니다."""
        center, extent, rotation = state["center"], state["extent"], state["rotation"]
        corrected_rotation = rotation * Rotation.from_euler('z', self.MANUAL_YAW_CORRECTION_DEGREES, degrees=True)
        mirror_rotation_matrix = corrected_rotation.as_matrix()
        mirror_z_axis, mirror_y_axis = mirror_rotation_matrix[:, 2], mirror_rotation_matrix[:, 1]
        self._publish_bounding_box(center, extent, corrected_rotation.as_quat(), frame_id, 0,
                                   normal_vector=mirror_z_axis)
        vec_for_yaw = center / (np.linalg.norm(center) + np.finfo(float).eps)
        vec_for_pitch = mirror_z_axis
        yaw_horiz_direction = np.array([vec_for_yaw[0], vec_for_yaw[1], 0.0])
        yaw_horiz_direction /= (np.linalg.norm(yaw_horiz_direction) + np.finfo(float).eps)
        pitch_horiz_magnitude = np.sqrt(vec_for_pitch[0] ** 2 + vec_for_pitch[1] ** 2)
        pitch_vert_magnitude = vec_for_pitch[2]
        final_z_axis = yaw_horiz_direction * pitch_horiz_magnitude + np.array([0, 0, pitch_vert_magnitude])
        final_z_axis /= (np.linalg.norm(final_z_axis) + np.finfo(float).eps)
        final_x_axis = np.cross(mirror_y_axis, final_z_axis);
        final_x_axis /= (np.linalg.norm(final_x_axis) + np.finfo(float).eps)
        final_y_axis = np.cross(final_z_axis, final_x_axis);
        final_y_axis /= (np.linalg.norm(final_y_axis) + np.finfo(float).eps)
        search_box_rotation_matrix = np.stack([final_x_axis, final_y_axis, final_z_axis], axis=1)
        search_obbs = []
        num_boxes, step_distance, height_reduction, base_width, width_increment = 1, 1.5, 1.5, 1.5, 0.6
        for i in range(num_boxes):
            box_center = center + final_z_axis * (step_distance * i + (step_distance / 2.0))
            size_multiplier = base_width + (width_increment * i)
            box_extent = np.array([extent[0] * height_reduction, extent[1] * size_multiplier, step_distance])
            search_obbs.append(o3d.geometry.OrientedBoundingBox(box_center, search_box_rotation_matrix, box_extent))
        self._publish_shadow_box(search_obbs, frame_id)
        final_pcd_to_publish = pcd1.voxel_down_sample(voxel_size=0.05)
        reflected_indices = []
        for obb in search_obbs:
            reflected_indices.extend(obb.get_point_indices_within_bounding_box(pcd1.points))

        if reflected_indices:
            initial_indices = np.unique(reflected_indices)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd1)
            neighbor_indices = []
            for index in initial_indices:
                [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd1.points[index], 0.15)
                neighbor_indices.extend(idx)
            candidate_indices = np.unique(np.concatenate((initial_indices, neighbor_indices)))
            candidate_points = np.asarray(pcd1.select_by_index(candidate_indices).points)
            keep_mask = np.ones(len(candidate_points), dtype=bool)
            if self.CULLING_DISTANCE_FROM_MIRROR > 0.0:
                distances = np.dot(candidate_points - state["center"], mirror_z_axis)
                keep_mask = distances > self.CULLING_DISTANCE_FROM_MIRROR
            final_indices_to_process = candidate_indices[keep_mask]
            points_to_restore = candidate_points[keep_mask]

            if len(points_to_restore) > 0:
                self.frames_since_restoration = 0
                n, Q = mirror_z_axis, center + mirror_z_axis * 0.005

                # <<< PyTorch 변경점 시작 >>>
                points_tensor = torch.tensor(points_to_restore, dtype=torch.float32, device=self.torch_device)
                n_tensor = torch.tensor(n, dtype=torch.float32, device=self.torch_device)
                Q_tensor = torch.tensor(Q, dtype=torch.float32, device=self.torch_device)
                reflection_matrix_tensor = torch.eye(3, device=self.torch_device) - 2 * torch.outer(n_tensor, n_tensor)
                restored_coords_tensor = (points_tensor - Q_tensor) @ reflection_matrix_tensor.T + Q_tensor
                if self.Z_CORRECTION_FACTOR != 0.0:
                    restored_coords_tensor[:, 2] += -mirror_z_axis[2] * self.Z_CORRECTION_FACTOR
                restored_coords = restored_coords_tensor.cpu().numpy()
                # <<< PyTorch 변경점 끝 >>>

                restored_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(restored_coords))
                restored_pcd.paint_uniform_color([0, 0, 1])
                self.last_restored_pcd = restored_pcd
                pcd1_cleaned = pcd1.select_by_index(final_indices_to_process, invert=True)
                final_pcd_to_publish = pcd1_cleaned.voxel_down_sample(voxel_size=0.05)
            else:
                self.frames_since_restoration += 1
        else:
            self.frames_since_restoration += 1

        if self.frames_since_restoration < self.RESTORATION_TTL and self.last_restored_pcd is not None:
            self.restored_points_pub.publish(self._o3d_to_pointcloud2(self.last_restored_pcd, frame_id))
        else:
            self.restored_points_pub.publish(self._o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id))
        self.final_points_pub.publish(self._o3d_to_pointcloud2(final_pcd_to_publish, frame_id))


########################################################################################################################
# 5. 메인 함수
########################################################################################################################

if __name__ == '__main__':
    try:
        rospy.init_node('mirror_detector_node', anonymous=True)
        detector = MirrorDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass