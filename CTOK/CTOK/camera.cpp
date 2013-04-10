#include "camera.h"
#include <GL/freeglut.h>

// 默认初始化Camera属性
Camera::Camera()
{
	// 初始化Camera为OpenGL的默认方向
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

// 设置Camera位置
void Camera::positionCamera(double posX, double posY, double posZ,
	double viewX, double viewY, double viewZ,
	double upX, double upY, double upZ)
{
	// 初始化Camera
	Vec3d vPos	= Vec3d(posX, posY, posZ);
	Vec3d vView	= Vec3d(viewX, viewY, viewZ);
	Vec3d vUp	= Vec3d(upX, upY, upZ);

	m_vPosition = vPos;
	m_vView     = vView;
	m_vUpVector = vUp;

	// 将m_vView到m_vPosition的向量单位化
	Vec3d tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
}

// 设置Camera位置
void Camera::positionCamera(const Vec3d &pos, const Vec3d &view, const Vec3d &up)
{
	// 初始化Camera
	m_vPosition = pos;
	m_vView = view;
	m_vUpVector = up;

	// 将m_vView到m_vPosition的矢量单位化
	Vec3d tmp;
	normalize(m_vView-m_vPosition, tmp);
	m_vView = m_vPosition + tmp;
}

// 通过移动鼠标移动Camera的方向(第一人称)
void Camera::setViewByMouse(void)
{
	POINT mousePos;									// 存储鼠标位置的结构体
	int middleX = glWinPosX + (glWinWidth  >> 1);	// 窗口宽度的一半
	int middleY = glWinPosY + (glWinHeight >> 1);	// 窗口高度的一半
	double angleY = 0.0;							// 存储向上看、向下看的角度
	double angleZ = 0.0;							// 存储向左看、向右看的角度
	static double currentRotX = 0.0;				// 存储总的向上、向下的旋转角度

	// 获得当前鼠标位置
	GetCursorPos(&mousePos);						

	// 如果鼠标仍然在正中间，不更新场景
	if ((mousePos.x == middleX) && (mousePos.y == middleY))
	{
		return;
	}

	// 将鼠标置回屏幕的中央
	SetCursorPos(middleX, middleY);

	// 获得鼠标移动的方向
	angleY = (double)((middleX - mousePos.x)) / 1000.0;
	angleZ = (double)((middleY - mousePos.y)) / 1000.0;

	// 保存一个当前向上或向下旋转过的角度，可以限制Camera上下做360度旋转
	currentRotX -= angleZ;  

	// 如果当前的旋转弧度大于1.3，不让Camera继续向上旋转
	if (currentRotX > 1.3)
	{
		currentRotX = 1.3;
	}
	// 如果当前的旋转弧度小于1.3，不让Camera继续向下旋转
	else if (currentRotX < -1.3)
	{
		currentRotX = -1.3;
	}
	// 否则绕所处位置旋转视线
	else
	{
		// 绕Camera的水平向量旋转Camera（上下）
		rotateView(angleZ, m_vStrafe);
	}
	// 绕Camera的向上向量旋转Camera（左右）
	rotateView(angleY, Vec3d(0,1,0));
}

// 绕给出的轴旋转Camera的方向
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

// 左右移动Camera
void Camera::strafeCamera(double speed)
{
	m_vPosition[0] += m_vStrafe[0] * speed;
	m_vPosition[2] += m_vStrafe[2] * speed;
	m_vView[0] += m_vStrafe[0] * speed;
	m_vView[2] += m_vStrafe[2] * speed;
}

// 前后移动Camera
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

// 更新Camera位置和方向
void Camera::update(void)
{
	// 更新Camera水平向量
	m_vStrafe = ((m_vView - m_vPosition).cross(m_vUpVector));
	normalize(m_vStrafe, m_vStrafe);

	if (mouseInWindow)
	{
		// 鼠标移动Camera
		setViewByMouse();
	}
}

// 在场景中放置Camera
void Camera::look(void)
{
	// 放置Camera
	gluLookAt(m_vPosition[0], m_vPosition[1], m_vPosition[2],
		m_vView[0],	 m_vView[1],     m_vView[2],
		m_vUpVector[0], m_vUpVector[1], m_vUpVector[2]);
	// 更新Camera信息
	update();
}