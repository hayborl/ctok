#ifndef CAMERA_H
#define CAMERA_H

#include "opencv2/opencv.hpp"

using namespace cv;

//����ͷ�ƶ��ٶ�
#define MOVESPEEDLR 0.1
#define MOVESPEEDFB	0.1
//����ͷ��ת�ٶ�
#define ROTSPEED 0.01

// ���ڵĸ߶ȺͿ��
extern int glWinWidth;
extern int glWinHeight;
// ���ڵĳ�ʼλ������
extern int curWinPosX;
extern int curWinPosY;

class Camera {
public:
	Camera();
	// ��������Camera��Ϣ
	Vec3d position(){ return m_vPosition; }
	Vec3d view(){ return m_vView; }
	Vec3d upVector(){ return m_vUpVector; }
	Vec3d strafe(){ return m_vStrafe; }

	void setMouseState(bool state) { mouseInWindow = state; }

	// ��ʼ��Camera����
	void positionCamera(double posX, double posY, double posZ,
		double viewX, double viewY, double viewZ,
		double upX, double upY, double upZ);
	void positionCamera(const Vec3d &pos, const Vec3d &view, const Vec3d &up);

	// ʹ��gluLookAt()�ڳ����аڷ�Camera
	void look();

	// ͨ���ƶ�����ƶ�Camera�ķ���(��һ�˳�)
	void setViewByMouse(void); 

	// �Ƹ���������תCamera�ķ���
	void rotateView(double angle, const Vec3d &vAxis);

	// �����ƶ�Camera(�ٶ�:speed)
	void strafeCamera(double speed);

	// ǰ���ƶ�Camera(�ٶ�:speed)
	void moveCamera(double speed);

	// �������λ��
	void reset();

private:
	// ����Camere�ķ����������Ϣ
	void update();

	bool mouseInWindow;		// ����Ƿ��ڴ�����

	Vec3d m_vPosition;		// Camera ��λ��
	Vec3d m_vView;			// Camera ���ӵ�
	Vec3d m_vUpVector;		// Camera ���ϵ�����
	Vec3d m_vStrafe;		// Camera ˮƽ������
};

#endif
