#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 
#include <stdarg.h>
#include "GL/glut.h" 
#include "glm.h"
#include "MatrixAlgebraLib.h"

//////////////////////////////////////////////////////////////////////
// Grphics Pipeline section
//////////////////////////////////////////////////////////////////////

#define WIN_SIZE 500
#define TEXTURE_SIZE 512
#define CAMERA_DISTANCE_FROM_AXIS_CENTER 10

typedef struct {
	GLfloat point3D[4];
	GLfloat normal[4];
	GLfloat point3DeyeCoordinates[4];
	GLfloat NormalEyeCoordinates[4];
	GLfloat pointScreen[4];
	GLfloat PixelValue;
	GLfloat TextureCoordinates[2];

} Vertex;

enum ProjectionTypeEnum { ORTHOGRAPHIC = 1, PERSPECTIVE };
enum DisplayTypeEnum {
	FACE_VERTEXES = 11, FACE_COLOR,
	LIGHTING_FLAT, LIGHTING_GOURARD, LIGHTING_PHONG,TEXTURE,
	TEXTURE_NEAREST, TEXTURE_LINEAR, TEXTURE_LIGHTING_PHONG
};
enum DisplayNormalEnum { DISPLAY_NORMAL_YES = 21, DISPLAY_NORMAL_NO };

typedef struct {
	GLfloat ModelMinVec[3]; //(left, bottom, near) of a model.
	GLfloat ModelMaxVec[3]; //(right, top, far) of a model.
	GLfloat CameraPos[3];
	GLfloat ModelScale;
	GLfloat ModelTranslateVector[3];
	enum ProjectionTypeEnum ProjectionType;
	enum DisplayTypeEnum DisplayType;
	enum DisplayNormalEnum DisplayNormals;
	GLfloat Lighting_Diffuse;
	GLfloat Lighting_Specular;
	GLfloat Lighting_Ambient;
	GLfloat Lighting_sHininess;
	GLfloat LightPosition[3];
} GuiParamsForYou;

GuiParamsForYou GlobalGuiParamsForYou;

//written for you
void setPixel(GLint x, GLint y, GLfloat r, GLfloat g, GLfloat b);

//you should write
void ModelProcessing();
void VertexProcessing(Vertex* v);
void FaceProcessing(Vertex* v1, Vertex* v2, Vertex* v3, GLfloat FaceColor[3]);
GLfloat LightingEquation(GLfloat point[3], GLfloat PointNormal[3], GLfloat LightPos[3], GLfloat Kd, GLfloat Ks, GLfloat Ka, GLfloat n);
void DrawLineDDA(GLint x1, GLint y1, GLint x2, GLint y2, GLfloat r, GLfloat g, GLfloat b);
int ifVertexNeedToBeDrawn(int x, int y, Vertex* v1, Vertex* v2, Vertex* v3, GLfloat* alpha, GLfloat* beta, GLfloat* gamma);
void lineCoefficients(Vertex* v1, Vertex* v2, GLfloat* A, GLfloat* B, GLfloat* C);
GLfloat PixelDepth(GLfloat alpha, GLfloat beta, GLfloat gamma, GLfloat p1, GLfloat p2, GLfloat p3);
GLint nearest(GLfloat num);

GLMmodel* model_ptr;
void ClearColorBuffer();
void DisplayColorBuffer();

void GraphicsPipeline()
{
	static GLuint i;
	static GLMgroup* group;
	static GLMtriangle* triangle;
	Vertex v1, v2, v3;
	GLfloat FaceColor[3];

	//calling ModelProcessing every time refreshing screen
	ModelProcessing();

	//call VertexProcessing for every vertrx
	//and then call FaceProcessing for every face
	group = model_ptr->groups;
	srand(0);
	while (group) {
		for (i = 0; i < group->numtriangles; i++) {
			triangle = &(model_ptr->triangles[group->triangles[i]]);

			MatrixCopy(v1.point3D, &model_ptr->vertices[3 * triangle->vindices[0]], 3);
			v1.point3D[3] = 1;
			MatrixCopy(v1.normal, &model_ptr->normals[3 * triangle->nindices[0]], 3);
			v1.normal[3] = 1;
			if (model_ptr->numtexcoords != 0)
				MatrixCopy(v1.TextureCoordinates, &model_ptr->texcoords[2 * triangle->tindices[0]], 2);
			else {
				v1.TextureCoordinates[0] = -1; v1.TextureCoordinates[1] = -1; //v1.TextureCoordinates[2] = -1;
			}
			VertexProcessing(&v1);

			MatrixCopy(v2.point3D, &model_ptr->vertices[3 * triangle->vindices[1]], 3);
			v2.point3D[3] = 1;
			MatrixCopy(v2.normal, &model_ptr->normals[3 * triangle->nindices[1]], 3);
			v2.normal[3] = 1;
			if (model_ptr->numtexcoords != 0)
				MatrixCopy(v2.TextureCoordinates, &model_ptr->texcoords[2 * triangle->tindices[1]], 2);
			else {
				v2.TextureCoordinates[0] = -1; v2.TextureCoordinates[1] = -1; //v2.TextureCoordinates[2] = -1;
			}
			VertexProcessing(&v2);

			MatrixCopy(v3.point3D, &model_ptr->vertices[3 * triangle->vindices[2]], 3);
			v3.point3D[3] = 1;
			MatrixCopy(v3.normal, &model_ptr->normals[3 * triangle->nindices[2]], 3);
			v3.normal[3] = 1;
			if (model_ptr->numtexcoords != 0)
				MatrixCopy(v3.TextureCoordinates, &model_ptr->texcoords[2 * triangle->tindices[2]], 2);
			else {
				v3.TextureCoordinates[0] = -1; v3.TextureCoordinates[1] = -1; //v3.TextureCoordinates[2] = -1;
			}
			VertexProcessing(&v3);

			FaceColor[0] = (GLfloat)rand() / ((GLfloat)RAND_MAX + 1);
			FaceColor[1] = (GLfloat)rand() / ((GLfloat)RAND_MAX + 1);
			FaceColor[2] = (GLfloat)rand() / ((GLfloat)RAND_MAX + 1);
			FaceProcessing(&v1, &v2, &v3, FaceColor);
		}
		group = group->next;
	}

	DisplayColorBuffer();
}

GLfloat Zbuffer[WIN_SIZE][WIN_SIZE];
GLfloat Mmodeling[16];
GLfloat Mlookat[16];
GLfloat Mprojection[16];
GLfloat Mviewport[16];
GLubyte TextureImage[TEXTURE_SIZE][TEXTURE_SIZE][3];

void ModelProcessing()
{
	int i, j;
	GLfloat right = 1, left = -1, top = 1, bottom = -1, near = CAMERA_DISTANCE_FROM_AXIS_CENTER - 1, far = CAMERA_DISTANCE_FROM_AXIS_CENTER + 1;
	GLfloat w[3], v[3], u[3], up[3] = { 0.0,1.0,0.0 };
	GLfloat vectorsM[16]; // transposed u,v,w vectors
	GLfloat eyeM[16]; //  -eye vector


					  // ex2-3-extra: calculating model scaling and translating transformation matrix
					  //////////////////////////////////////////////////////////////////////////////////

					  // ex2-3: calculating translate transformation matrix
					  //////////////////////////////////////////////////////////////////////////////////

	M4x4identity(Mmodeling);
	for (int i = 12; i < 15; i++)
	{
		Mmodeling[i] = GlobalGuiParamsForYou.ModelTranslateVector[i - 12];
	}

	// ex2-3: calculating scale transformation matrix
	//////////////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < 11; i += 5)
	{
		Mmodeling[i] = GlobalGuiParamsForYou.ModelScale;
	}


	// ex2-4: calculating lookat transformation matrix
	//////////////////////////////////////////////////////////////////////////////////

	//calculate w vector (center is  [0,0,0])
	w[0] = GlobalGuiParamsForYou.CameraPos[0];
	w[1] = GlobalGuiParamsForYou.CameraPos[1];
	w[2] = GlobalGuiParamsForYou.CameraPos[2];
	V3Normalize(w);

	//calculate u vector 
	V3cross(u, up, w);
	V3Normalize(u);

	//calculate v vector
	V3cross(v, w, u);

	//initializing the eye matrix
	M4x4identity(eyeM);
	for (int i = 12; i < 15; i++)
	{
		eyeM[i] = -GlobalGuiParamsForYou.CameraPos[i - 12];
	}

	//initialize the u,v,w matrix
	M4x4identity(vectorsM);
	vectorsM[0] = u[0];
	vectorsM[1] = v[0];
	vectorsM[2] = w[0];
	vectorsM[4] = u[1];
	vectorsM[5] = v[1];
	vectorsM[6] = w[1];
	vectorsM[8] = u[2];
	vectorsM[9] = v[2];
	vectorsM[10] = w[2];

	//create the look at matrix
	M4multiplyM4(Mlookat, vectorsM, eyeM);


	// ex2-2: calculating Orthographic or Perspective projection transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	M4x4identity(Mprojection);
	if (GlobalGuiParamsForYou.ProjectionType == PERSPECTIVE)
	{
		Mprojection[0] = (2.0 * near) / (right - left);
		Mprojection[5] = (2.0 * near) / (top - bottom);
		Mprojection[8] = (right + left) / (right - left);
		Mprojection[9] = (top + bottom) / (top - bottom);
		Mprojection[10] = -(far + near) / (far - near);
		Mprojection[11] = -1.0;
		Mprojection[14] = -(2.0 * far * near) / (far - near);
		Mprojection[15] = 0.0;

	}
	else if (GlobalGuiParamsForYou.ProjectionType == ORTHOGRAPHIC)
	{
		Mprojection[0] = 2.0 / (right - left);
		Mprojection[5] = 2.0 / (top - bottom);
		Mprojection[10] = -2.0 / (far - near);
		Mprojection[12] = -(right + left) / (right - left);
		Mprojection[13] = -(top + bottom) / (top - bottom);
		Mprojection[14] = -(far + near) / (far - near);
	}


	// ex2-2: calculating viewport transformation matrix
	//////////////////////////////////////////////////////////////////////////////////
	M4x4identity(Mviewport);

	Mviewport[0] = Mviewport[12] = WIN_SIZE / 2.0;
	Mviewport[5] = Mviewport[13] = WIN_SIZE / 2.0;
	Mviewport[10] = Mviewport[14] = 0.5;


	// ex3: clearing color and Z-buffer
	//////////////////////////////////////////////////////////////////////////////////
	for (i = 0; i < WIN_SIZE; i++)
	{
		for (j = 0; j < WIN_SIZE; j++)
		{
			Zbuffer[i][j] = 1.0;
		}
	}
	ClearColorBuffer(); // setting color buffer to background color

}


void VertexProcessing(Vertex* v)
{
	GLfloat point3DafterModelingTrans[4];
	GLfloat temp1[4], temp2[4];
	GLfloat point3D_plusNormal_screen[4];
	GLfloat Mmodeling3x3[9], Mlookat3x3[9];

	// ex2-3: modeling transformation v->point3D --> point3DafterModelingTrans
	//////////////////////////////////////////////////////////////////////////////////
	M4multiplyV4(point3DafterModelingTrans, Mmodeling, v->point3D);

	// ex2-4: lookat transformation point3DafterModelingTrans --> v->point3DeyeCoordinates
	//////////////////////////////////////////////////////////////////////////////////
	M4multiplyV4(v->point3DeyeCoordinates, Mlookat, point3DafterModelingTrans);

	// ex2-2: transformation from eye coordinates to screen coordinates v->point3DeyeCoordinates --> v->pointScreen
	//////////////////////////////////////////////////////////////////////////////////
	M4multiplyV4(temp1, Mprojection, v->point3DeyeCoordinates);
	M4multiplyV4(v->pointScreen, Mviewport, temp1);


	// ex2-5: transformation normal from object coordinates to eye coordinates v->normal --> v->NormalEyeCoordinates
	//////////////////////////////////////////////////////////////////////////////////
	M3fromM4(Mmodeling3x3, Mmodeling);
	M3fromM4(Mlookat3x3, Mlookat);
	M3multiplyV3(temp1, Mmodeling3x3, v->normal);
	M3multiplyV3(v->NormalEyeCoordinates, Mlookat3x3, temp1);
	V3Normalize(v->NormalEyeCoordinates);
	v->NormalEyeCoordinates[3] = 1;

	// ex2-5: drawing normals 
	//////////////////////////////////////////////////////////////////////////////////
	if (GlobalGuiParamsForYou.DisplayNormals == DISPLAY_NORMAL_YES) {
		V4HomogeneousDivide(v->point3DeyeCoordinates);
		VscalarMultiply(temp1, v->NormalEyeCoordinates, 0.05, 3);
		Vplus(temp2, v->point3DeyeCoordinates, temp1, 4);
		temp2[3] = 1;
		M4multiplyV4(temp1, Mprojection, temp2);
		M4multiplyV4(point3D_plusNormal_screen, Mviewport, temp1);
		V4HomogeneousDivide(point3D_plusNormal_screen);
		V4HomogeneousDivide(v->pointScreen);
		DrawLineDDA(round(v->pointScreen[0]), round(v->pointScreen[1]), round(point3D_plusNormal_screen[0]), round(point3D_plusNormal_screen[1]), 0, 0, 1);
	}

	// ex3: calculating lighting for vertex
	//////////////////////////////////////////////////////////////////////////////////
	v->PixelValue = LightingEquation(v->point3DeyeCoordinates, v->NormalEyeCoordinates, GlobalGuiParamsForYou.LightPosition,
		GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);
}


void FaceProcessing(Vertex* v1, Vertex* v2, Vertex* v3, GLfloat FaceColor[3])
{

	GLfloat minimumX, minimumY, maximumX, maximumY;
	GLfloat pixColor1 = FaceColor[0], pixColor2 = FaceColor[1], pixColor3 = FaceColor[2];
	GLfloat Alpha, Beta, Gamma,light;
	GLfloat pos[3], normal[3];
	int i = 0; // x index
	int j = 0; // y index
	int s, t;
	unsigned int tempRed, tempGreen, tempBlue, temp1, temp2, temp3;
	unsigned char r, g, b;

	V4HomogeneousDivide(v1->pointScreen);
	V4HomogeneousDivide(v2->pointScreen);
	V4HomogeneousDivide(v3->pointScreen);


	if (GlobalGuiParamsForYou.DisplayType == FACE_VERTEXES)
	{
		DrawLineDDA(round(v1->pointScreen[0]), round(v1->pointScreen[1]), round(v2->pointScreen[0]), round(v2->pointScreen[1]), 1, 1, 1);
		DrawLineDDA(round(v2->pointScreen[0]), round(v2->pointScreen[1]), round(v3->pointScreen[0]), round(v3->pointScreen[1]), 1, 1, 1);
		DrawLineDDA(round(v3->pointScreen[0]), round(v3->pointScreen[1]), round(v1->pointScreen[0]), round(v1->pointScreen[1]), 1, 1, 1);


	}
	else {
		//ex3: Barycentric Coordinates and lighting
		//////////////////////////////////////////////////////////////////////////////////

		//Calculating rectangle's borders:
		minimumX = min(v1->pointScreen[0], min(v2->pointScreen[0], v3->pointScreen[0]));
		maximumX = max(v1->pointScreen[0], max(v2->pointScreen[0], v3->pointScreen[0]));
		minimumY = min(v1->pointScreen[1], min(v2->pointScreen[1], v3->pointScreen[1]));
		maximumY = max(v1->pointScreen[1], max(v2->pointScreen[1], v3->pointScreen[1]));

		//Checking if we don't exceed the buffer 

		if (minimumX < 0) minimumX = 0;
		if (maximumX > WIN_SIZE) maximumX = WIN_SIZE;
		if (minimumY < 0) minimumY = 0;
		if (maximumY > WIN_SIZE) maximumY = WIN_SIZE;

		// Checking which pixels in the rectangle we have to draw:
		for (i = minimumX; i < maximumX; i++)
		{
			for (j = minimumY; j < maximumY; j++)
			{
				if (ifVertexNeedToBeDrawn(i, j, v1, v2, v3, &Alpha, &Beta, &Gamma))			// If we have to draw the pixel, check how to draw it and choose colors.
				{
					switch (GlobalGuiParamsForYou.DisplayType)
					{
					case FACE_COLOR:
						pixColor1 = FaceColor[0];
						pixColor2 = FaceColor[1];
						pixColor3 = FaceColor[2];
						break;
					case LIGHTING_FLAT:
						pixColor1 = (v1->PixelValue + v2->PixelValue + v3->PixelValue) / 3.0;
						pixColor2 = (v1->PixelValue + v2->PixelValue + v3->PixelValue) / 3.0;
						pixColor3 = (v1->PixelValue + v2->PixelValue + v3->PixelValue) / 3.0;
						break;
					case LIGHTING_GOURARD:
						pixColor1 = Alpha * v3->PixelValue + Beta * v1->PixelValue + Gamma * v2->PixelValue;
						pixColor2 = Alpha * v3->PixelValue + Beta * v1->PixelValue + Gamma * v2->PixelValue;
						pixColor3 = Alpha * v3->PixelValue + Beta * v1->PixelValue + Gamma * v2->PixelValue;
						break;
						case LIGHTING_PHONG:
						{
							// calculation of position and normal vector:
							for (int k = 0; k < 3; k++)
							{
								pos[k] = (Alpha * v3->point3DeyeCoordinates[k]) + (Beta * v1->point3DeyeCoordinates[k]) + (Gamma * v2->point3DeyeCoordinates[k]);
								normal[k] = (Alpha * v3->NormalEyeCoordinates[k]) + (Beta * v1->NormalEyeCoordinates[k]) + (Gamma * v2->NormalEyeCoordinates[k]);
							}
							// Lighting calculation:
							pixColor1 = LightingEquation(pos, normal, GlobalGuiParamsForYou.LightPosition, GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);
							pixColor2 = pixColor1;
							pixColor3 = pixColor1;
				
							break;
						}
						case TEXTURE:
						{
							// Calculation of the coordinates of the pixel in the texture image that contains the appropriate color of that pixel. 
							t = nearest(Alpha * v3->TextureCoordinates[0] * (TEXTURE_SIZE - 1) + Beta * v1->TextureCoordinates[0] * (TEXTURE_SIZE - 1) + Gamma * v2->TextureCoordinates[0] * (TEXTURE_SIZE - 1));	
							s = nearest(Alpha * v3->TextureCoordinates[1] * (TEXTURE_SIZE - 1) + Beta * v1->TextureCoordinates[1] * (TEXTURE_SIZE - 1) + Gamma * v2->TextureCoordinates[1] * (TEXTURE_SIZE - 1));

							/*
							pixColor1 = (GLfloat)(GLint)TextureImage[t][s][0];
							pixColor2 = (GLfloat)(GLint)TextureImage[t][s][1];
							pixColor3 = (GLfloat)(GLint)TextureImage[t][s][2];
							pixColor1 /= 255.0;
							pixColor2 /= 255.0;
							pixColor3 /= 255.0;*/

							/*
							pixColor1 = TextureImage[t][s][0];
							pixColor2 = TextureImage[t][s][1];
							pixColor3 = TextureImage[t][s][2];
							pixColor1 /= 255.0;
							pixColor2 /= 255.0;
							pixColor3 /= 255.0;*/

							// Conversion from TextureImage[s][t][] to pixColor: 
							r = TextureImage[s][t][0];
							g = TextureImage[s][t][1];
							b = TextureImage[s][t][2];
							temp1 = r + '0';
							temp2 = g + '0';
							temp3 = b + '0';
							pixColor1 = temp1/255.0;
							pixColor2 = temp2/255.0;
							pixColor3 = temp3/255.0; 
							break;

						}

						case TEXTURE_LIGHTING_PHONG:
						{
							// Calculation of the coordinates of the pixel in the texture image that contains the appropriate color of that pixel. 
							t = nearest(Alpha * v3->TextureCoordinates[0] * (TEXTURE_SIZE - 1) + Beta * v1->TextureCoordinates[0] * (TEXTURE_SIZE - 1) + Gamma * v2->TextureCoordinates[0] * (TEXTURE_SIZE - 1));
							s = nearest(Alpha * v3->TextureCoordinates[1] * (TEXTURE_SIZE - 1) + Beta * v1->TextureCoordinates[1] * (TEXTURE_SIZE - 1) + Gamma * v2->TextureCoordinates[1] * (TEXTURE_SIZE - 1));
							for (int k = 0; k < 3; k++)
							{
								pos[k] = (Alpha * v3->point3DeyeCoordinates[k]) + (Beta * v1->point3DeyeCoordinates[k]) + (Gamma * v2->point3DeyeCoordinates[k]);
								normal[k] = (Alpha * v3->NormalEyeCoordinates[k]) + (Beta * v1->NormalEyeCoordinates[k]) + (Gamma * v2->NormalEyeCoordinates[k]);
							}
							// Lighting calculation:
							light= LightingEquation(pos, normal, GlobalGuiParamsForYou.LightPosition, GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);

							// Conversion from TextureImage[s][t][] to pixColor:
							r = TextureImage[s][t][0];
							g = TextureImage[s][t][1];
							b = TextureImage[s][t][2];
							temp1 = r + '0';
							temp2 = g + '0';
							temp3 = b + '0';
							pixColor1 = temp1 / 255.0;
							pixColor2 = temp2 / 255.0;
							pixColor3 = temp3 / 255.0;

							// (pixColor1,pixColor2,pixColor3) = (light*TextureRed, light*TextureGreen, light*TextureBlue)
							pixColor1 *= light;
							pixColor2 *= light;
							pixColor3 *= light;
							break;
						}

					}
					// After we find the colors, we will color the pixel in those colors:
					setPixel(i, j, pixColor1, pixColor2, pixColor3);	
				}

			}
		}
	}
}


int ifVertexNeedToBeDrawn(int x, int y, Vertex* v1, Vertex* v2, Vertex* v3, GLfloat* alpha, GLfloat* beta, GLfloat* gamma)
{
	GLfloat multiplaier = 8;
	GLfloat Alpha, Beta, Gamma;
	GLfloat pixelDepth;
	GLfloat A, B, C;

	// Calculating Alpha:
	lineCoefficients(v1, v2, &A, &B, &C);
	Alpha = (A * x + B * y + C) / (A * v3->pointScreen[0] + B * v3->pointScreen[1] + C);
	// Calculating Beta:
	lineCoefficients(v2, v3, &A, &B, &C);
	Beta = (A * x + B * y + C) / (A * v1->pointScreen[0] + B * v1->pointScreen[1] + C);
	// Calculating Gamma:
	lineCoefficients(v1, v3, &A, &B, &C);
	Gamma = (A * x + B * y + C) / (A * v2->pointScreen[0] + B * v2->pointScreen[1] + C);

	// Save alpha beta and gamma
	*alpha = Alpha;
	*beta = Beta;
	*gamma = Gamma;

	if (Alpha > 1.0 || Beta > 1.0 || Gamma > 1.0 || Alpha < 0.0 || Beta < 0.0 || Gamma < 0.0)// Out side of the triangle
	{
		return 0;
	}

	// ex3: z-buffer
	//////////////////////////////////////////////////////////////////////////////////

	pixelDepth = PixelDepth(Alpha, Beta, Gamma, v3->pointScreen[2], v1->pointScreen[2], v2->pointScreen[2]);
	if (Zbuffer[x][y] >= pixelDepth)
	{
		Zbuffer[x][y] = pixelDepth;
		return 1;
	}
	return 0;
}

void lineCoefficients(Vertex* v1, Vertex* v2, GLfloat* A, GLfloat* B, GLfloat* C)
{
	*A = v1->pointScreen[1] - v2->pointScreen[1]; // y1-y2
	*B = v2->pointScreen[0] - v1->pointScreen[0]; // x2-x1
	*C = (v1->pointScreen[0] * v2->pointScreen[1]) - (v2->pointScreen[0] * v1->pointScreen[1]); //x1*y2 - x2*y1
}

GLfloat PixelDepth(GLfloat alpha, GLfloat beta, GLfloat gamma, GLfloat p1, GLfloat p2, GLfloat p3)
{
	return alpha * p1 + beta * p2 + gamma * p3;
}
void DrawLineDDA(GLint x1, GLint y1, GLint x2, GLint y2, GLfloat r, GLfloat g, GLfloat b)
{
	//ex2.1: 
	//////////////////////////////////////////////////////////////////////////////////
	float dx, dy, x, y, a, x1_, y1_, x2_, y2_;

	if ((y2 - y1) > -(x2 - x1)) {
		x1_ = x1;
		y1_ = y1;
		x2_ = x2;
		y2_ = y2;
	}
	else
	{
		x1_ = x2;
		y1_ = y2;
		x2_ = x1;
		y2_ = y1;
	}

	dx = x2_ - x1_;
	dy = y2_ - y1_;
	if (fabs(dx) > fabs(dy)) {
		a = dy / dx;
		y = y1_;
		for (x = x1_; x < x2_; x++) {
			setPixel(x, round(y), 1, 1, 1);
			y = y + a;
		}
	}
	else {
		a = dx / dy;
		x = x1_;
		for (y = y1_; y < y2_; y++) {
			setPixel(round(x), y, 1, 1, 1);
			x = x + a;
		}
	}
}





GLfloat LightingEquation(GLfloat point[3], GLfloat PointNormal[3], GLfloat LightPos[3], GLfloat Kd, GLfloat Ks, GLfloat Ka, GLfloat n)
{
	//ex3: calculate lighting equation
	//////////////////////////////////////////////////////////////////////////////////
	//ex3: calculate lighting equation

	float specular = 0, diffuse;
	float normal[3], r[3], l[3], v[3], tmp[3], result;

	// Diffuse calculation:
	MatrixCopy(normal, PointNormal, 3);
	V3Normalize(normal);
	Vminus(l, LightPos, point, 3);
	V3Normalize(l);
	diffuse = Kd * V3dot(normal, l);

	// Specular calculation:
	Vminus(v, PointNormal, point, 3);
	V3Normalize(v);
	VscalarMultiply(tmp, normal, 2 * V3dot(l, normal), 3);
	Vminus(r, tmp, l, 3);
	V3Normalize(r);
	if (n >= 0)
		specular = Ks * (min(powf(V3dot(v, r), n), pow(V3dot(v, r), n + 1)));
	result = max(0, diffuse) + max(0, specular) + max(0, Ka);
	return result;
}








//////////////////////////////////////////////////////////////////////
// GUI section
//////////////////////////////////////////////////////////////////////

//function declerations

void InitGuiGlobalParams();
void drawingCB(void);
void reshapeCB(int width, int height);
void keyboardCB(unsigned char key, int x, int y);
void keyboardSpecialCB(int key, int x, int y);
void MouseClickCB(int button, int state, int x, int y);
void MouseMotionCB(int x, int y);
void menuCB(int value);
void TerminationErrorFunc(char* ErrorString);
void LoadModelFile();
void DisplayColorBuffer();
void drawstr(char* FontName, int FontSize, GLuint x, GLuint y, char* format, ...);
void TerminationErrorFunc(char* ErrorString);
GLubyte *readBMP(char *imagepath, int *width, int *height);

enum FileNumberEnum {
	TEAPOT = 100, TEDDY, PUMPKIN, COW,
	SIMPLE_PYRAMID, FIRST_EXAMPLE, SIMPLE_3D_EXAMPLE, SPHERE,
	TRIANGLE, Z_BUFFER_EXAMPLE, TEXTURE_BOX, TEXTURE_TRIANGLE,
	TEXTURE_BARREL, TEXTURE_SHEEP
};


typedef struct {
	enum FileNumberEnum FileNum;
	GLfloat CameraRaduis;
	GLint   CameraAnleHorizontal;
	GLint   CameraAnleVertical;
	GLint   MouseLastPos[2];
} GuiCalculations;

GuiCalculations GlobalGuiCalculations;

GLuint ColorBuffer[WIN_SIZE][WIN_SIZE][3];

int main(int argc, char** argv)
{
	GLint submenu1_id, submenu2_id, submenu3_id, submenu4_id;

	//initizlizing GLUT
	glutInit(&argc, argv);

	//initizlizing GUI globals
	InitGuiGlobalParams();

	//initializing window
	glutInitWindowSize(WIN_SIZE, WIN_SIZE);
	glutInitWindowPosition(900, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("Computer Graphics HW");

	//registering callbacks
	glutDisplayFunc(drawingCB);
	glutReshapeFunc(reshapeCB);
	glutKeyboardFunc(keyboardCB);
	glutSpecialFunc(keyboardSpecialCB);
	glutMouseFunc(MouseClickCB);
	glutMotionFunc(MouseMotionCB);

	//registering and creating menu
	submenu1_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("open teapot.obj", TEAPOT);
	glutAddMenuEntry("open teddy.obj", TEDDY);
	glutAddMenuEntry("open pumpkin.obj", PUMPKIN);
	glutAddMenuEntry("open cow.obj", COW);
	glutAddMenuEntry("open Simple3Dexample.obj", SIMPLE_3D_EXAMPLE);
	glutAddMenuEntry("open SimplePyramid.obj", SIMPLE_PYRAMID);
	glutAddMenuEntry("open sphere.obj", SPHERE);
	glutAddMenuEntry("open triangle.obj", TRIANGLE);
	glutAddMenuEntry("open FirstExample.obj", FIRST_EXAMPLE);
	glutAddMenuEntry("open ZbufferExample.obj", Z_BUFFER_EXAMPLE);
	glutAddMenuEntry("open TriangleTexture.obj", TEXTURE_TRIANGLE);
	glutAddMenuEntry("open box.obj", TEXTURE_BOX);
	glutAddMenuEntry("open barrel.obj", TEXTURE_BARREL);
	glutAddMenuEntry("open sheep.obj", TEXTURE_SHEEP);
	submenu2_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Orthographic", ORTHOGRAPHIC);
	glutAddMenuEntry("Perspective", PERSPECTIVE);
	submenu3_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Face Vertexes", FACE_VERTEXES);
	glutAddMenuEntry("Face Color", FACE_COLOR);
	glutAddMenuEntry("Lighting Flat", LIGHTING_FLAT);
	glutAddMenuEntry("Lighting Gourard", LIGHTING_GOURARD);
	glutAddMenuEntry("Lighting Phong", LIGHTING_PHONG);
	glutAddMenuEntry("Texture", TEXTURE);
	glutAddMenuEntry("Texture lighting Phong", TEXTURE_LIGHTING_PHONG);
	submenu4_id = glutCreateMenu(menuCB);
	glutAddMenuEntry("Yes", DISPLAY_NORMAL_YES);
	glutAddMenuEntry("No", DISPLAY_NORMAL_NO);
	glutCreateMenu(menuCB);
	glutAddSubMenu("Open Model File", submenu1_id);
	glutAddSubMenu("Projection Type", submenu2_id);
	glutAddSubMenu("Display type", submenu3_id);
	glutAddSubMenu("Display Normals", submenu4_id);
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	LoadModelFile();
	/*for (int r = 0; r < TEXTURE_SIZE; r++) {
		for (int o = 0; o < TEXTURE_SIZE; o++) {
			printf("\nred = %f, green = %f, blue = %f\n", TextureImage[r][o][0], TextureImage[r][o][1], TextureImage[r][o][2]);
		}
	}*/
	//starting main loop
	glutMainLoop();
}

void drawingCB(void)
{
	GLenum er;

	char DisplayString1[200], DisplayString2[200];

	//clearing the background
	glClearColor(0, 0, 0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//initializing modelview transformation matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	GraphicsPipeline();

	glColor3f(0, 1, 0);
	sprintf(DisplayString1, "Scale:%.1f , Translate: (%.1f,%.1f,%.1f), Camera angles:(%d,%d) position:(%.1f,%.1f,%.1f) ", GlobalGuiParamsForYou.ModelScale, GlobalGuiParamsForYou.ModelTranslateVector[0], GlobalGuiParamsForYou.ModelTranslateVector[1], GlobalGuiParamsForYou.ModelTranslateVector[2], GlobalGuiCalculations.CameraAnleHorizontal, GlobalGuiCalculations.CameraAnleVertical, GlobalGuiParamsForYou.CameraPos[0], GlobalGuiParamsForYou.CameraPos[1], GlobalGuiParamsForYou.CameraPos[2]);
	drawstr("helvetica", 12, 15, 25, DisplayString1);
	sprintf(DisplayString2, "Lighting reflection - Diffuse:%1.2f, Specular:%1.2f, Ambient:%1.2f, sHininess:%1.2f", GlobalGuiParamsForYou.Lighting_Diffuse, GlobalGuiParamsForYou.Lighting_Specular, GlobalGuiParamsForYou.Lighting_Ambient, GlobalGuiParamsForYou.Lighting_sHininess);
	drawstr("helvetica", 12, 15, 10, DisplayString2);

	//swapping buffers and displaying
	glutSwapBuffers();

	//check for errors
	er = glGetError();  //get errors. 0 for no error, find the error codes in: https://www.opengl.org/wiki/OpenGL_Error
	if (er) printf("error: %d\n", er);
}


void LoadModelFile()
{
	int width, height;
	GLubyte *ImageData;

	if (model_ptr) {
		glmDelete(model_ptr);
		model_ptr = 0;
	}

	switch (GlobalGuiCalculations.FileNum) {
	case TEAPOT:
		model_ptr = glmReadOBJ("teapot.obj");
		break;
	case TEDDY:
		model_ptr = glmReadOBJ("teddy.obj");
		break;
	case PUMPKIN:
		model_ptr = glmReadOBJ("pumpkin.obj");
		break;
	case COW:
		model_ptr = glmReadOBJ("cow.obj");
		break;
	case SIMPLE_PYRAMID:
		model_ptr = glmReadOBJ("SimplePyramid.obj");
		break;
	case FIRST_EXAMPLE:
		model_ptr = glmReadOBJ("FirstExample.obj");
		break;
	case SIMPLE_3D_EXAMPLE:
		model_ptr = glmReadOBJ("Simple3Dexample.obj");
		break;
	case SPHERE:
		model_ptr = glmReadOBJ("sphere.obj");
		break;
	case TRIANGLE:
		model_ptr = glmReadOBJ("triangle.obj");
		break;
	case Z_BUFFER_EXAMPLE:
		model_ptr = glmReadOBJ("ZbufferExample.obj");
		break;
	case TEXTURE_TRIANGLE:
		model_ptr = glmReadOBJ("TriangleTexture.obj");
		ImageData = readBMP("TriangleTexture.bmp", &width, &height);
		if (width != TEXTURE_SIZE || height != TEXTURE_SIZE)
			TerminationErrorFunc("Invalid texture size");
		memcpy(TextureImage, ImageData, TEXTURE_SIZE*TEXTURE_SIZE * 3);
		free(ImageData);
		break;
	case TEXTURE_BOX:
		model_ptr = glmReadOBJ("box.obj");
		ImageData = readBMP("box.bmp", &width, &height);
		if (width != TEXTURE_SIZE || height != TEXTURE_SIZE)
			TerminationErrorFunc("Invalid texture size");
		memcpy(TextureImage, ImageData, TEXTURE_SIZE*TEXTURE_SIZE * 3);
		free(ImageData);
		break;
	case TEXTURE_BARREL:
		model_ptr = glmReadOBJ("barrel.obj");
		ImageData = readBMP("barrel.bmp", &width, &height);
		if (width != TEXTURE_SIZE || height != TEXTURE_SIZE)
			TerminationErrorFunc("Invalid texture size");
		memcpy(TextureImage, ImageData, TEXTURE_SIZE*TEXTURE_SIZE * 3);
		free(ImageData);
		break;
	case TEXTURE_SHEEP:
		model_ptr = glmReadOBJ("sheep.obj");
		ImageData = readBMP("sheep.bmp", &width, &height);
		if (width != TEXTURE_SIZE || height != TEXTURE_SIZE)
			TerminationErrorFunc("Invalid texture size");
		memcpy(TextureImage, ImageData, TEXTURE_SIZE*TEXTURE_SIZE * 3);
		free(ImageData);
		break;
	default:
		TerminationErrorFunc("File number not valid");
		break;
	}

	if (!model_ptr)
		TerminationErrorFunc("can not load 3D model");
	//glmUnitize(model_ptr);  //"unitize" a model by translating it

	//to the origin and scaling it to fit in a unit cube around
	//the origin
	glmFacetNormals(model_ptr);  //adding facet normals
	glmVertexNormals(model_ptr, 90.0);  //adding vertex normals

	glmBoundingBox(model_ptr, GlobalGuiParamsForYou.ModelMinVec, GlobalGuiParamsForYou.ModelMaxVec);
}

void ClearColorBuffer()
{
	GLuint x, y;
	for (y = 0; y < WIN_SIZE; y++) {
		for (x = 0; x < WIN_SIZE; x++) {
			ColorBuffer[y][x][0] = 0;
			ColorBuffer[y][x][1] = 0;
			ColorBuffer[y][x][2] = 0;
		}
	}
}

void setPixel(GLint x, GLint y, GLfloat r, GLfloat g, GLfloat b)
{
	if (x >= 0 && x < WIN_SIZE && y >= 0 && y < WIN_SIZE) {
		ColorBuffer[y][x][0] = round(r * 255);
		ColorBuffer[y][x][1] = round(g * 255);
		ColorBuffer[y][x][2] = round(b * 255);
	}
}

GLint nearest(GLfloat num) {

	if (num <= ((int)num + 0.5))
		return (int)num;
	return ((int)num + 1);
}


void DisplayColorBuffer()
{
	GLuint x, y;
	glBegin(GL_POINTS);
	for (y = 0; y < WIN_SIZE; y++) {
		for (x = 0; x < WIN_SIZE; x++) {
			glColor3ub(min(255, ColorBuffer[y][x][0]), min(255, ColorBuffer[y][x][1]), min(255, ColorBuffer[y][x][2]));
			glVertex2f(x + 0.5, y + 0.5);   // The 0.5 is to target pixel
		}
	}
	glEnd();
}


void InitGuiGlobalParams()
{
	GlobalGuiCalculations.FileNum = TEAPOT;
	GlobalGuiCalculations.CameraRaduis = CAMERA_DISTANCE_FROM_AXIS_CENTER;
	GlobalGuiCalculations.CameraAnleHorizontal = 0;
	GlobalGuiCalculations.CameraAnleVertical = 0;

	GlobalGuiParamsForYou.CameraPos[0] = 0;
	GlobalGuiParamsForYou.CameraPos[1] = 0;
	GlobalGuiParamsForYou.CameraPos[2] = GlobalGuiCalculations.CameraRaduis;

	GlobalGuiParamsForYou.ModelScale = 1;

	GlobalGuiParamsForYou.ModelTranslateVector[0] = 0;
	GlobalGuiParamsForYou.ModelTranslateVector[1] = 0;
	GlobalGuiParamsForYou.ModelTranslateVector[2] = 0;
	GlobalGuiParamsForYou.DisplayType = FACE_VERTEXES;
	GlobalGuiParamsForYou.ProjectionType = ORTHOGRAPHIC;
	GlobalGuiParamsForYou.DisplayNormals = DISPLAY_NORMAL_NO;
	GlobalGuiParamsForYou.Lighting_Diffuse = 0.75;
	GlobalGuiParamsForYou.Lighting_Specular = 0.2;
	GlobalGuiParamsForYou.Lighting_Ambient = 0.2;
	GlobalGuiParamsForYou.Lighting_sHininess = 40;
	GlobalGuiParamsForYou.LightPosition[0] = 10;
	GlobalGuiParamsForYou.LightPosition[1] = 5;
	GlobalGuiParamsForYou.LightPosition[2] = 0;

}


void reshapeCB(int width, int height)
{
	if (width != WIN_SIZE || height != WIN_SIZE)
	{
		glutReshapeWindow(WIN_SIZE, WIN_SIZE);
	}

	//update viewport
	glViewport(0, 0, width, height);

	//clear the transformation matrices (load identity)
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//projection
	gluOrtho2D(0, WIN_SIZE, 0, WIN_SIZE);
}


void keyboardCB(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		exit(0);
		break;
	case '+':
		GlobalGuiParamsForYou.ModelScale += 0.1;
		glutPostRedisplay();
		break;
	case '-':
		GlobalGuiParamsForYou.ModelScale -= 0.1;
		glutPostRedisplay();
		break;
	case 'd':
	case 'D':
		GlobalGuiParamsForYou.Lighting_Diffuse += 0.05;
		glutPostRedisplay();
		break;
	case 'c':
	case 'C':
		GlobalGuiParamsForYou.Lighting_Diffuse -= 0.05;
		glutPostRedisplay();
		break;
	case 's':
	case 'S':
		GlobalGuiParamsForYou.Lighting_Specular += 0.05;
		glutPostRedisplay();
		break;
	case 'x':
	case 'X':
		GlobalGuiParamsForYou.Lighting_Specular -= 0.05;
		glutPostRedisplay();
		break;
	case 'a':
	case 'A':
		GlobalGuiParamsForYou.Lighting_Ambient += 0.05;
		glutPostRedisplay();
		break;
	case 'z':
	case 'Z':
		GlobalGuiParamsForYou.Lighting_Ambient -= 0.05;
		glutPostRedisplay();
		break;
	case 'h':
	case 'H':
		GlobalGuiParamsForYou.Lighting_sHininess += 1;
		glutPostRedisplay();
		break;
	case 'n':
	case 'N':
		GlobalGuiParamsForYou.Lighting_sHininess -= 1;
		glutPostRedisplay();
		break;
	default:
		printf("Key not valid (language shold be english)\n");
	}
}


void keyboardSpecialCB(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_LEFT:
		GlobalGuiParamsForYou.ModelTranslateVector[0] -= 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_RIGHT:
		GlobalGuiParamsForYou.ModelTranslateVector[0] += 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_DOWN:
		GlobalGuiParamsForYou.ModelTranslateVector[2] -= 0.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_UP:
		GlobalGuiParamsForYou.ModelTranslateVector[2] += 0.1;
		glutPostRedisplay();
		break;
	}
}


void MouseClickCB(int button, int state, int x, int y)
{
	GlobalGuiCalculations.MouseLastPos[0] = x;
	GlobalGuiCalculations.MouseLastPos[1] = y;
}

void MouseMotionCB(int x, int y)
{
	GlobalGuiCalculations.CameraAnleHorizontal += (x - GlobalGuiCalculations.MouseLastPos[0]) / 40;
	GlobalGuiCalculations.CameraAnleVertical -= (y - GlobalGuiCalculations.MouseLastPos[1]) / 40;

	if (GlobalGuiCalculations.CameraAnleVertical > 30)
		GlobalGuiCalculations.CameraAnleVertical = 30;
	if (GlobalGuiCalculations.CameraAnleVertical < -30)
		GlobalGuiCalculations.CameraAnleVertical = -30;

	GlobalGuiCalculations.CameraAnleHorizontal = (GlobalGuiCalculations.CameraAnleHorizontal + 360) % 360;
	//	GlobalGuiCalculations.CameraAnleVertical   = (GlobalGuiCalculations.CameraAnleVertical   + 360) % 360;

	GlobalGuiParamsForYou.CameraPos[0] = GlobalGuiCalculations.CameraRaduis * sin((float)(GlobalGuiCalculations.CameraAnleVertical + 90) * M_PI / 180) * cos((float)(GlobalGuiCalculations.CameraAnleHorizontal + 90) * M_PI / 180);
	GlobalGuiParamsForYou.CameraPos[2] = GlobalGuiCalculations.CameraRaduis * sin((float)(GlobalGuiCalculations.CameraAnleVertical + 90) * M_PI / 180) * sin((float)(GlobalGuiCalculations.CameraAnleHorizontal + 90) * M_PI / 180);
	GlobalGuiParamsForYou.CameraPos[1] = GlobalGuiCalculations.CameraRaduis * cos((float)(GlobalGuiCalculations.CameraAnleVertical + 90) * M_PI / 180);
	glutPostRedisplay();
}

void menuCB(int value)
{
	switch (value) {
	case ORTHOGRAPHIC:
	case PERSPECTIVE:
		GlobalGuiParamsForYou.ProjectionType = value;
		glutPostRedisplay();
		break;
	case FACE_VERTEXES:
	case FACE_COLOR:
	case LIGHTING_FLAT:
	case LIGHTING_GOURARD:
	case LIGHTING_PHONG:
		GlobalGuiParamsForYou.DisplayType = value;
		glutPostRedisplay();
		break;
	case TEXTURE:
		GlobalGuiParamsForYou.DisplayType = value;
		glutPostRedisplay();
		break;
	case TEXTURE_LIGHTING_PHONG:
		GlobalGuiParamsForYou.DisplayType = value;
		glutPostRedisplay();
		break;
	case DISPLAY_NORMAL_YES:
	case DISPLAY_NORMAL_NO:
		GlobalGuiParamsForYou.DisplayNormals = value;
		glutPostRedisplay();
		break;
	default:
		GlobalGuiCalculations.FileNum = value;
		LoadModelFile();
		glutPostRedisplay();
	}
}



void drawstr(char* FontName, int FontSize, GLuint x, GLuint y, char* format, ...)
{
	va_list args;
	char buffer[255], *s;

	GLvoid* font_style = GLUT_BITMAP_TIMES_ROMAN_10;

	font_style = GLUT_BITMAP_HELVETICA_10;
	if (strcmp(FontName, "helvetica") == 0) {
		if (FontSize == 12)
			font_style = GLUT_BITMAP_HELVETICA_12;
		else if (FontSize == 18)
			font_style = GLUT_BITMAP_HELVETICA_18;
	}
	else if (strcmp(FontName, "times roman") == 0) {
		font_style = GLUT_BITMAP_TIMES_ROMAN_10;
		if (FontSize == 24)
			font_style = GLUT_BITMAP_TIMES_ROMAN_24;
	}
	else if (strcmp(FontName, "8x13") == 0) {
		font_style = GLUT_BITMAP_8_BY_13;
	}
	else if (strcmp(FontName, "9x15") == 0) {
		font_style = GLUT_BITMAP_9_BY_15;
	}

	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	glRasterPos2i(x, y);
	for (s = buffer; *s; s++)
		glutBitmapCharacter(font_style, *s);
}


void TerminationErrorFunc(char* ErrorString)
{
	char string[256];
	printf(ErrorString);
	fgets(string, 256, stdin);

	exit(0);
}

// Function to load bmp file
// buffer for the image is allocated in this function, you should free this buffer
GLubyte *readBMP(char *imagepath, int *width, int *height)
{
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned int imageSize;   // = width*height*3
	unsigned char * data;
	unsigned char tmp;
	int i;

	// Open the file
	FILE * file = fopen(imagepath, "rb");
	if (!file) {
		TerminationErrorFunc("Image could not be opened\n");
	}

	if (fread(header, 1, 54, file) != 54) { // If not 54 bytes read : problem
		TerminationErrorFunc("Not a correct BMP file\n");
	}

	if (header[0] != 'B' || header[1] != 'M') {
		TerminationErrorFunc("Not a correct BMP file\n");
	}

	// Read ints from the byte array
	dataPos = *(int*)&(header[0x0A]);
	imageSize = *(int*)&(header[0x22]);
	*width = *(int*)&(header[0x12]);
	*height = *(int*)&(header[0x16]);

	// Some BMP files are misformatted, guess missing information
	if (imageSize == 0)
		imageSize = *width**height * 3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)
		dataPos = 54; // The BMP header is done that way

					  // Create a buffer
	data = malloc(imageSize * sizeof(GLubyte));

	// Read the actual data from the file into the buffer
	fread(data, 1, imageSize, file);


	//swap the r and b values to get RGB (bitmap is BGR)
	for (i = 0; i<*width**height; i++)
	{
		tmp = data[i * 3];
		data[i * 3] = data[i * 3 + 2];
		data[i * 3 + 2] = tmp;
	}


	//Everything is in memory now, the file can be closed
	fclose(file);

	return data;
}
