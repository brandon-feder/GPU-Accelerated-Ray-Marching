__device__
int color(int  r, int g, int b)
{
  int rgba = r;
  rgba =  (rgba<<8) + g;
  rgba =  (rgba<<8) + b;
  rgba =  (rgba<<8) + 255;
  return rgba;
}
