#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"

//摄像头移动速度
#define MOVESPEEDLR 10.0f
#define MOVESPEEDFB	50.0f
//摄像头旋转速度
#define ROTSPEED 0.01f

// 窗口的高度和宽度
#define Window_Width	640
#define Window_Height	480

class Camera {
public:
	Camera();
	// 用来返回Camera信息
	Vec3f position(){ return m_vPosition; }
	Vec3f view(){ return m_vView; }
	Vec3f upVector(){ return m_vUpVector; }
	Vec3f strafe(){ return m_vStrafe; }

	void setMouseState(bool state) { mouseInWindow = state; }

	// 初始化Camera属性
	void positionCamera(float posX, float posY, float posZ,
		float viewX, float viewY, float viewZ,
		float upX, float upY, float upZ);
	void positionCamera(const Vec3f &pos, const Vec3f &view, const Vec3f &up);

	// 使用gluLookAt()在场景中摆放Camera
	void look();

	// 通过移动鼠标移动Camera的方向(第一人称)
	void setViewByMouse(void); 

	// 绕给出的轴旋转Camera的方向
	void rotateView(float angle, const Vec3f &vAxis);

	// 左右移动Camera(速度:speed)
	void strafeCamera(float speed);

	// 前后移动Camera(速度:speed)
	void moveCamera(float speed);

	// 重置相机位置
	void reset();

private:
	// 更新Camere的方向和其他信息
	void update();

	bool mouseInWindow;		// 鼠标是否在窗体内

	Vec3f m_vPosition;		// Camera 的位置
	Vec3f m_vView;			// Camera 的视点
	Vec3f m_vUpVector;		// Camera 向上的向量
	Vec3f m_vStrafe;		// Camera 水平的向量
};

#endif


