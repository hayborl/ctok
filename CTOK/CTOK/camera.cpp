#include "camera.h"
#include <GL/freeglut.h>

// Ĭ�ϳ�ʼ��Camera����
Camera::Camera()
{
	// ��ʼ��CameraΪOpenGL��Ĭ�Ϸ���
	Vec3f vPos(0.0, 0.0, 0.0);
	Vec3f vView(0.0, 0.0,-1.0);
	Vec3f vUp(0.0, 1.0, 0.0);

	m_vPosition	= vPos;
	m_vView		= vView;
	m_vUpVector	= vUp;

	mouseInWindow = true;

	SetCursorPos(Window_Width >> 1, Window_Height >> 1);
	//ShowCursor(FALSE);
}

// ����Cameraλ��
void Camera::positionCamera(float posX, float posY, float posZ,
	float viewX, float viewY, float viewZ,
	float upX, float upY, float upZ)
{
	// ��ʼ��Camera
	Vec3f vPos	= Vec3f(posX, posY, posZ);
	Vec3f vView	= Vec3f(viewX, viewY, viewZ);
	Vec3f vUp	= Vec3f(upX, upY, upZ);

	m_vPosition = vPos;
	m_vView     = vView;
	m_vUpVector = vUp;

	// ��m_vView��m_vPosition��������λ��
	Vec3f tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
	//SetCursorPos(Window_Width >> 1, Window_Height >> 1);
}

// ����Cameraλ��
void Camera::positionCamera(const Vec3f &pos, const Vec3f &view, const Vec3f &up)
{
	// ��ʼ��Camera
	m_vPosition = pos;
	m_vView = view;
	m_vUpVector = up;

	// ��m_vView��m_vPosition��ʸ����λ��
	Vec3f tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
	//SetCursorPos(Window_Width >> 1, Window_Height >> 1);
}

// ͨ���ƶ�����ƶ�Camera�ķ���(��һ�˳�)
void Camera::setViewByMouse(void)
{
	POINT mousePos;						// �洢���λ�õĽṹ��
	int middleX = Window_Width  >> 1;	// ���ڿ�ȵ�һ��
	int middleY = Window_Height >> 1;	// ���ڸ߶ȵ�һ��
	float angleY = 0.0f;				// �洢���Ͽ������¿��ĽǶ�
	float angleZ = 0.0f;				// �洢���󿴡����ҿ��ĽǶ�
	static float currentRotX = 0.0f;	// �洢�ܵ����ϡ����µ���ת�Ƕ�

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
	angleY = (float)((middleX - mousePos.x)) / 1000.0f;
	angleZ = (float)((middleY - mousePos.y)) / 1000.0f;

	// ����һ����ǰ���ϻ�������ת���ĽǶȣ���������Camera������360����ת
	currentRotX -= angleZ;  

	// �����ǰ����ת���ȴ���1.3������Camera����������ת
	if (currentRotX > 1.3f)
	{
		currentRotX = 1.3f;
	}
	// �����ǰ����ת����С��1.3������Camera����������ת
	else if (currentRotX < -1.3f)
	{
		currentRotX = -1.3f;
	}
	// ����������λ����ת����
	else
	{
		// ��Camera��ˮƽ������תCamera�����£�
		rotateView(angleZ, m_vStrafe);
	}
	// ��Camera������������תCamera�����ң�
	rotateView(angleY, Vec3f(0,1,0));
}

// �Ƹ���������תCamera�ķ���
void Camera::rotateView(float angle, const Vec3f &vAxis)
{
	Vec3f vNewView;
	Vec3f vView = m_vView - m_vPosition;

	if (angle == 0.0)
	{
		vNewView = vView;
	}
	else
	{
		Vec3f u;
		normalize(vAxis, u);

		float cosTheta = (float)cos(angle);
		float sinTheta = (float)sin(angle);

		vNewView[0]  = (float)(cosTheta + (1 - cosTheta) * u[0] * u[0]) * vView[0];
		vNewView[0] += (float)((1 - cosTheta) * u[0] * u[1] - u[2] * sinTheta) * vView[1];
		vNewView[0] += (float)((1 - cosTheta) * u[0] * u[2] + u[1] * sinTheta) * vView[2];

		vNewView[1]  = (float)((1 - cosTheta) * u[0] * u[1] + u[2] * sinTheta) * vView[0];
		vNewView[1] += (float)(cosTheta + (1 - cosTheta) * u[1] * u[1]) * vView[1];
		vNewView[1] += (float)((1 - cosTheta) * u[1] * u[2] - u[0] * sinTheta) * vView[2];

		vNewView[2]  = (float)((1 - cosTheta) * u[0] * u[2] - u[1] * sinTheta) * vView[0];
		vNewView[2] += (float)((1 - cosTheta) * u[1] * u[2] + u[0] * sinTheta) * vView[1];
		vNewView[2] += (float)(cosTheta + (1 - cosTheta) * u[2] * u[2]) * vView[2];
	}

	m_vView = m_vPosition + vNewView;
}

// �����ƶ�Camera
void Camera::strafeCamera(float speed)
{
	m_vPosition[0] += m_vStrafe[0] * speed;
	m_vPosition[2] += m_vStrafe[2] * speed;
	m_vView[0] += m_vStrafe[0] * speed;
	m_vView[2] += m_vStrafe[2] * speed;
}

// ǰ���ƶ�Camera
void Camera::moveCamera(float speed)
{
	Vec3f vView = m_vView - m_vPosition;
	m_vPosition[0] += vView[0] * speed;
	m_vPosition[2] += vView[2] * speed;
	m_vView[0] += vView[0] * speed;
	m_vView[2] += vView[2] * speed;
}

void Camera::reset()
{
	positionCamera(0.0f, 1.8f, 100.0f, 0.0f, 1.8f, 0.0f, 0.0f, 1.0f, 0.0f);
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