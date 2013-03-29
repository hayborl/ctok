#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"

//����ͷ�ƶ��ٶ�
#define MOVESPEEDLR 10.0f
#define MOVESPEEDFB	50.0f
//����ͷ��ת�ٶ�
#define ROTSPEED 0.01f

// ���ڵĸ߶ȺͿ��
#define Window_Width	640
#define Window_Height	480

class Camera {
public:
	Camera();
	// ��������Camera��Ϣ
	Vec3f position(){ return m_vPosition; }
	Vec3f view(){ return m_vView; }
	Vec3f upVector(){ return m_vUpVector; }
	Vec3f strafe(){ return m_vStrafe; }

	void setMouseState(bool state) { mouseInWindow = state; }

	// ��ʼ��Camera����
	void positionCamera(float posX, float posY, float posZ,
		float viewX, float viewY, float viewZ,
		float upX, float upY, float upZ);
	void positionCamera(const Vec3f &pos, const Vec3f &view, const Vec3f &up);

	// ʹ��gluLookAt()�ڳ����аڷ�Camera
	void look();

	// ͨ���ƶ�����ƶ�Camera�ķ���(��һ�˳�)
	void setViewByMouse(void); 

	// �Ƹ���������תCamera�ķ���
	void rotateView(float angle, const Vec3f &vAxis);

	// �����ƶ�Camera(�ٶ�:speed)
	void strafeCamera(float speed);

	// ǰ���ƶ�Camera(�ٶ�:speed)
	void moveCamera(float speed);

	// �������λ��
	void reset();

private:
	// ����Camere�ķ����������Ϣ
	void update();

	bool mouseInWindow;		// ����Ƿ��ڴ�����

	Vec3f m_vPosition;		// Camera ��λ��
	Vec3f m_vView;			// Camera ���ӵ�
	Vec3f m_vUpVector;		// Camera ���ϵ�����
	Vec3f m_vStrafe;		// Camera ˮƽ������
};

#endif


