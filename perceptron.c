#include <stdio.h>
#include <math.h>

// Función de activación sigmoide
float sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

float x;
float y, y2, y3;

float w1 = 0.73467124;
float w2 = 0.74910873;
float w3 = -6.1940413;
float w4 = -6.8395452;

float b1 = 8.292889;
float b2 = -8.280191;
float b3 = 6.2032886;

int main()
{

    while (1)
    {
        printf("\nInsert an input value:");
        scanf("%f", &x);

        y = sigmoid(x * w1 + b1);
        y2 = sigmoid(x * w2 + b2);

        y3 = sigmoid((y * w3) + (y2 * w4) + b3);

        printf("\nInput value: %f Output value: %f", x, y3);
    }
    return 0;
}