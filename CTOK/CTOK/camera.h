#ifndef CAMERA_H
#define CAMERA_H

#include "opencv2/opencv.hpp"

using namespace cv;

//摄像头移动速度
#define MOVESPEEDLR 0.1
#define MOVESPEEDFB	0.1
//摄像头旋转速度
#define ROTSPEED 0.01

// 窗口的高度和宽度
extern int glWinWidth;
extern int glWinHeight;
// 窗口的初始位置坐标
extern int curWinPosX;
extern int curWinPosY;

class Camera {
public:
	Camera();
	// 用来返回Camera信息
	Vec3d position(){ return m_vPosition; }
	Vec3d view(){ return m_vView; }
	Vec3d upVector(){ return m_vUpVector; }
	Vec3d strafe(){ return m_vStrafe; }

	void setMouseState(bool state) { mouseInWindow = state; }

	// 初始化Camera属性
	void positionCamera(double posX, double posY, double posZ,
		double viewX, double viewY, double viewZ,
		double upX, double upY, double upZ);
	void positionCamera(const Vec3d &pos, const Vec3d &view, const Vec3d &up);

	// 使用gluLookAt()在场景中摆放Camera
	void look();

	// 通过移动鼠标移动Camera的方向(第一人称)
	void setViewByMouse(void); 

	// 绕给出的轴旋转Camera的方向
	void rotateView(double angle, const Vec3d &vAxis);

	// 左右移动Camera(速度:speed)
	void strafeCamera(double speed);

	// 前后移动Camera(速度:speed)
	void moveCamera(double speed);

	// 重置相机位置
	void reset();

private:
	// 更新Camere的方向和其他信息
	void update();

	bool mouseInWindow;		// 鼠标是否在窗体内

	Vec3d m_vPosition;		// Camera 的位置
	Vec3d m_vView;			// Camera 的视点
	Vec3d m_vUpVector;		// Camera 向上的向量
	Vec3d m_vStrafe;		// Camera 水平的向量
};

#endif
