#include "camera.h"
#include <GL/freeglut.h>

// Ĭ�ϳ�ʼ��Camera����
Camera::Camera()
{
	// ��ʼ��CameraΪOpenGL��Ĭ�Ϸ���
	Vec3d vPos(0.0, 0.0, 0.0);
	Vec3d vView(0.0, 0.0,-1.0);
	Vec3d vUp(0.0, 1.0, 0.0);

	m_vPosition	= vPos;
	m_vView		= vView;
	m_vUpVector	= vUp;

	mouseInWindow = true;

	SetCursorPos(glWinPosX + (glWinWidth >> 1), 
		glWinPosY + (glWinHeight >> 1));
	//ShowCursor(FALSE);
}

// ����Cameraλ��
void Camera::positionCamera(double posX, double posY, double posZ,
	double viewX, double viewY, double viewZ,
	double upX, double upY, double upZ)
{
	// ��ʼ��Camera
	Vec3d vPos	= Vec3d(posX, posY, posZ);
	Vec3d vView	= Vec3d(viewX, viewY, viewZ);
	Vec3d vUp	= Vec3d(upX, upY, upZ);

	m_vPosition = vPos;
	m_vView     = vView;
	m_vUpVector = vUp;

	// ��m_vView��m_vPosition��������λ��
	Vec3d tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
}

// ����Cameraλ��
void Camera::positionCamera(const Vec3d &pos, const Vec3d &view, const Vec3d &up)
{
	// ��ʼ��Camera
	m_vPosition = pos;
	m_vView = view;
	m_vUpVector = up;

	// ��m_vView��m_vPosition��ʸ����λ��
	Vec3d tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
}

// ͨ���ƶ�����ƶ�Camera�ķ���(��һ�˳�)
void Camera::setViewByMouse(void)
{
	POINT mousePos;									// �洢���λ�õĽṹ��
	int middleX = glWinPosX + (glWinWidth  >> 1);	// ���ڿ�ȵ�һ��
	int middleY = glWinPosY + (glWinHeight >> 1);	// ���ڸ߶ȵ�һ��
	double angleY = 0.0;							// �洢���Ͽ������¿��ĽǶ�
	double angleZ = 0.0;							// �洢���󿴡����ҿ��ĽǶ�
	static double currentRotX = 0.0;				// �洢�ܵ����ϡ����µ���ת�Ƕ�

	// ��õ�ǰ���λ��
	GetCursorPos(&mousePos);						

	// ��������Ȼ�����м䣬�����³���
	if ((mousePos.x == middleX) && (mousePos.y == middleY))
	{
		return;
	}

	// ������û���Ļ������
	SetCursorPos(middleX, middleY);

	// �������ƶ��ķ���
	angleY = (double)((middleX - mousePos.x)) / 1000.0;
	angleZ = (double)((middleY - mousePos.y)) / 1000.0;

	// ����һ����ǰ���ϻ�������ת���ĽǶȣ���������Camera������360����ת
	currentRotX -= angleZ;  

	// �����ǰ����ת���ȴ���1.3������Camera����������ת
	if (currentRotX > 1.3)
	{
		currentRotX = 1.3;
	}
	// �����ǰ����ת����С��1.3������Camera����������ת
	else if (currentRotX < -1.3)
	{
		currentRotX = -1.3;
	}
	// ����������λ����ת����
	else
	{
		// ��Camera��ˮƽ������תCamera�����£�
		rotateView(angleZ, m_vStrafe);
	}
	// ��Camera������������תCamera�����ң�
	rotateView(angleY, Vec3d(0,1,0));
}

// �Ƹ���������תCamera�ķ���
void Camera::rotateView(double angle, const Vec3d &vAxis)
{
	Vec3d vNewView;
	Vec3d vView = m_vView - m_vPosition;

	if (angle == 0.0)
	{
		vNewView = vView;
	}
	else
	{
		Vec3d u;
		normalize(vAxis, u);

		double cosTheta = cos(angle);
		double sinTheta = sin(angle);

		vNewView[0]  = (cosTheta + (1 - cosTheta) * u[0] * u[0]) * vView[0];
		vNewView[0] += ((1 - cosTheta) * u[0] * u[1] - u[2] * sinTheta) * vView[1];
		vNewView[0] += ((1 - cosTheta) * u[0] * u[2] + u[1] * sinTheta) * vView[2];

		vNewView[1]  = ((1 - cosTheta) * u[0] * u[1] + u[2] * sinTheta) * vView[0];
		vNewView[1] += (cosTheta + (1 - cosTheta) * u[1] * u[1]) * vView[1];
		vNewView[1] += ((1 - cosTheta) * u[1] * u[2] - u[0] * sinTheta) * vView[2];

		vNewView[2]  = ((1 - cosTheta) * u[0] * u[2] - u[1] * sinTheta) * vView[0];
		vNewView[2] += ((1 - cosTheta) * u[1] * u[2] + u[0] * sinTheta) * vView[1];
		vNewView[2] += (cosTheta + (1 - cosTheta) * u[2] * u[2]) * vView[2];
	}

	m_vView = m_vPosition + vNewView;
}

// �����ƶ�Camera
void Camera::strafeCamera(double speed)
{
	m_vPosition[0] += m_vStrafe[0] * speed;
	m_vPosition[2] += m_vStrafe[2] * speed;
	m_vView[0] += m_vStrafe[0] * speed;
	m_vView[2] += m_vStrafe[2] * speed;
}

// ǰ���ƶ�Camera
void Camera::moveCamera(double speed)
{
	Vec3d vView = m_vView - m_vPosition;
	m_vPosition[0] += vView[0] * speed;
	m_vPosition[2] += vView[2] * speed;
	m_vView[0] += vView[0] * speed;
	m_vView[2] += vView[2] * speed;
}

void Camera::reset()
{
	positionCamera(0.0, 1.8, 100.0, 0.0, 1.8, 0.0, 0.0, 1.0, 0.0);
}

// ����Cameraλ�úͷ���
void Camera::update(void)
{
	// ����Cameraˮƽ����
	m_vStrafe = ((m_vView - m_vPosition).cross(m_vUpVector));
	normalize(m_vStrafe, m_vStrafe);

	if (mouseInWindow)
	{
		// ����ƶ�Camera
		setViewByMouse();
	}
}

// �ڳ����з���Camera
void Camera::look(void)
{
	// ����Camera
	gluLookAt(m_vPosition[0], m_vPosition[1], m_vPosition[2],
		m_vView[0],	 m_vView[1],     m_vView[2],
		m_vUpVector[0], m_vUpVector[1], m_vUpVector[2]);
	// ����Camera��Ϣ
	update();
}