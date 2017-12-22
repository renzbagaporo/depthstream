/*
 * This file is part of the Computer Vision Toolkit (cvkit).
 *
 * Author: Heiko Hirschmueller
 *
 * Copyright (c) 2014, Institute of Robotics and Mechatronics, German Aerospace Center
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "camera.h"
#include "minpack.h"

#include <limits>
#include <cstring>

using std::string;
using std::ostringstream;
using std::isfinite;
using std::numeric_limits;

using gutil::Properties;
using gutil::IOException;
using gutil::InvalidArgumentException;

namespace gmath
{

string Camera::getCameraKey(const char *key, int id) const
{
    ostringstream os;
    
    os << "camera.";
    if (id >= 0)
      os << id << '.';
    
    os << key;
    
    return os.str();
}

Camera::Camera(const Properties &prop, int id)
{
    prop.getValue(getCameraKey("R", id).c_str(), R, "[1 0 0; 0 1 0; 0 0 1]");
    prop.getValue(getCameraKey("T", id).c_str(), T, "[0 0 0]");
    prop.getValue(getCameraKey("width", id).c_str(), width, "0");
    prop.getValue(getCameraKey("height", id).c_str(), height, "0");
    
    prop.getValue(getCameraKey("zmin", id).c_str(), zmin, "0");
    prop.getValue(getCameraKey("zmax", id).c_str(), zmax, "0");
    
    prop.getStringVector(getCameraKey("match", id).c_str(), match, "", ',');
}

bool Camera::isInside(const Vector2d &p) const
{
    return p[0] >= 0 && p[0] < width && p[1] >= 0 && p[1] < height;
}

bool Camera::isInside(const Vector3d &p) const
{
    return p[0] >= 0 && p[0] < width && p[1] >= 0 && p[1] < height &&
      !isfinite(p[2]);
}

void Camera::setDownscaled(int ds)
{
    if (ds > 1)
    {
      width=(width+ds-1)/ds;
      height=(height+ds-1)/ds;
    }
}

void Camera::setPart(long x, long y, long w, long h)
{
    width=w;
    height=h;
}

void Camera::getProperties(Properties &prop, int id) const
{
    prop.putValue(getCameraKey("R", id).c_str(), R);
    prop.putValue(getCameraKey("T", id).c_str(), T);
    
    if (width != 0)
      prop.putValue(getCameraKey("width", id).c_str(), width);
    
    if (height != 0)
      prop.putValue(getCameraKey("height", id).c_str(), height);
    
    if (zmax > 0)
    {
      prop.putValue(getCameraKey("zmin", id).c_str(), zmin);
      prop.putValue(getCameraKey("zmax", id).c_str(), zmax);
    }
    
    if (match.size() > 0)
      prop.putStringVector(getCameraKey("match", id).c_str(), match, ',');
}

PinholeCamera::PinholeCamera()
{
    rho=0;
    dist=false;
    kd[0]=0;
    kd[1]=0;
    kd[2]=0;
    pd[0]=0;
    pd[1]=0;
}

PinholeCamera::PinholeCamera(const Properties &prop, int id)
  : Camera(prop, id)
{
    Matrix33d P;
    prop.getValue(getCameraKey("A", id).c_str(), P);
    
    string origin;
    prop.getValue("origin", origin, "corner");
    if (origin == "center")
    {
      A(0, 2)+=0.5;
      A(1, 2)+=0.5;
    }
    
    setA(P);
    
    prop.getValue(getCameraKey("rho", id).c_str(), rho, "0");
    
    if (rho == 0)
      prop.getValue("rho", rho, "0");
    
    if (rho == 0)
    {
      double f, t;
      prop.getValue("f", f, "0");
      prop.getValue("t", t, "0");
      rho=f*t;
    }
    
    prop.getValue(getCameraKey("k1", id).c_str(), kd[0], "0");
    prop.getValue(getCameraKey("k2", id).c_str(), kd[1], "0");
    prop.getValue(getCameraKey("k3", id).c_str(), kd[2], "0");
    prop.getValue(getCameraKey("p1", id).c_str(), pd[0], "0");
    prop.getValue(getCameraKey("p2", id).c_str(), pd[1], "0");
    
    dist=(kd[0] != 0) || (kd[1] != 0) || (kd[2] != 0) || (pd[0] != 0) ||
      (pd[1] != 0);
}

Camera *PinholeCamera::clone() const
{
    PinholeCamera *ret=new PinholeCamera();
    
    *ret=*this;
    
    return ret;
}

void PinholeCamera::setA(const Matrix33d &AA)
{
    A=AA;
    
    if (A(1, 0) != 0 || A(2, 0) != 0 || A(2, 1) != 0 || A(2, 2) != 1)
    {
      ostringstream out;
      out << "Invalid camera matrix: " << A;
      throw IOException(out.str().c_str());
    }
}

void PinholeCamera::setDownscaled(int ds)
{
    Camera::setDownscaled(ds);
    
    if (ds > 1)
    {
      A/=ds;
      A(2, 2)=1.0;
      rho/=ds;
    }
}

void PinholeCamera::setPart(long x, long y, long w, long h)
{
    Camera::setPart(x, y, w, h);
    
    A(0, 2)-=x;
    A(1, 2)-=y;
}

void PinholeCamera::getProperties(Properties &prop, int id) const
{
    Camera::getProperties(prop);
    
    prop.putValue(getCameraKey("A", id).c_str(), A);
    prop.putValue(getCameraKey("rho", id).c_str(), rho);
    
    if (dist)
    {
      prop.putValue(getCameraKey("k1", id).c_str(), kd[0]);
      prop.putValue(getCameraKey("k2", id).c_str(), kd[1]);
      prop.putValue(getCameraKey("k3", id).c_str(), kd[2]);
      prop.putValue(getCameraKey("p1", id).c_str(), pd[0]);
      prop.putValue(getCameraKey("p2", id).c_str(), pd[1]);
    }
}

double PinholeCamera::projectPoint(Vector2d &p, const Vector3d &Pw) const
{
      // transform into camera coordiante system
    
    Vector3d Pc=transpose(getR())*(Pw-getT());
    
      // if rho is given, then compute disparity and determine if the point
      // is behind the camera
    
    double d=numeric_limits<double>::infinity();
    
    if (rho != 0)
      d=rho/Pc[2];
    
    if (Pc[2] <= 0)
      d=-1;
    
      // apply lens distortion
    
    Pc/=Pc[2];
    
    if (dist)
    {
      double x=Pc[0];
      double y=Pc[1];
      
      double r2=x*x+y*y;
      double s=1.0+kd[0]*r2+kd[1]*r2*r2+kd[2]*r2*r2*r2;
      
      Pc[0]=x*s+2*pd[0]*x*y+pd[1]*(r2+2*x*x);
      Pc[1]=y*s+pd[0]*(r2+2*y*y)+2*pd[1]*x*y;
    }
    
      // apply camera matrix
    
    p[0]=A(0, 0)*Pc[0]+A(0, 1)*Pc[1]+A(0, 2);
    p[1]=A(1, 1)*Pc[1]+A(1, 2);
    
    return d;
}

void PinholeCamera::reconstructPoint(Vector3d &Pw, const Vector2d &p, double d)
  const
{
    if (rho == 0)
      throw IOException("Cannot reconstruct point with unknown rho");
    
    assert(isfinite(d));
    
    Vector3d Pc;
    reconstructLocal(Pc, p);
    
    Pc*=rho/d;
    
    Pw=getR()*Pc+getT();
}

void PinholeCamera::reconstructRay(Vector3d &V, Vector3d &C, const Vector2d &p) const
{
    Vector3d Pc;
    reconstructLocal(Pc, p);
    
    V=getR()*Pc;
    C=getT();
}

void PinholeCamera::projectPointLocal(Vector2d &p, const Vector3d &Pc) const
{
      // apply lens distortion
    
    Vector3d P=Pc/Pc[2];
    
    if (dist)
    {
      double x=P[0];
      double y=P[1];
      
      double r2=x*x+y*y;
      double s=1.0+kd[0]*r2+kd[1]*r2*r2+kd[2]*r2*r2*r2;
      
      P[0]=x*s+2*pd[0]*x*y+pd[1]*(r2+2*x*x);
      P[1]=y*s+pd[0]*(r2+2*y*y)+2*pd[1]*x*y;
    }
    
      // apply camera matrix
    
    p[0]=A(0, 0)*P[0]+A(0, 1)*P[1]+A(0, 2);
    p[1]=A(1, 1)*P[1]+A(1, 2);
}

namespace
{

struct DistortionParameter
{
    double kd[3];
    double pd[2];
    double xd, yd;
};

int computeDistortion(int n, double x[], int m, double fvec[], double fjac[],
  void *up)
{
    DistortionParameter *p=static_cast<DistortionParameter *>(up);
    
    if (fvec != 0)
    {
      double r2=x[0]*x[0]+x[1]*x[1];
      double s=1.0+p->kd[0]*r2+p->kd[1]*r2*r2+p->kd[2]*r2*r2*r2;
      
      fvec[0]=p->xd-(x[0]*s+2*p->pd[0]*x[0]*x[1]+p->pd[1]*(r2+2*x[0]*x[0]));
      fvec[1]=p->yd-(x[1]*s+p->pd[0]*(r2+2*x[1]*x[1])+2*p->pd[1]*x[0]*x[1]);
    }
    
    if (fjac != 0)
    {
      double r2=x[0]*x[0]+x[1]*x[1];
      
      double f=1.0+p->kd[0]*r2+p->kd[1]*r2*r2+p->kd[2]*r2*r2*r2;
      double fs=p->kd[0]+2*p->kd[1]*r2+3*p->kd[2]*r2*r2;
      
      fjac[0]=-(f+2*x[0]*x[0]*fs+2*p->pd[0]*x[1]+6*p->pd[1]*x[0]);
      fjac[1]=-(2*x[0]*x[1]*fs+2*p->pd[0]*x[0]+2*p->pd[1]*x[1]);
      fjac[2]=-fjac[1];
      fjac[3]=-(f+2*x[1]*x[1]*fs+6*p->pd[0]*x[1]+2*p->pd[1]*x[0]);
    }
    
    return 0;
}

}

void PinholeCamera::reconstructLocal(Vector3d &q, const Vector2d &p) const
{
      // apply inverse camera matrix
    
    q[0]=p[0]/A(0, 0)-p[1]*A(0, 1)/(A(0, 0)*A(1, 1))+
      (A(0, 1)*A(1, 2)-A(1, 1)*A(0, 2))/
      (A(0, 0)*A(1, 1)*A(2, 2));
    q[1]=p[1]/A(1, 1)-A(1, 2)/(A(2, 2)*A(1, 1));
    q[2]=1;
    
      // apply inverse lense distortion
    
    if (dist)
    {
      DistortionParameter param;
      
      param.kd[0]=kd[0];
      param.kd[1]=kd[1];
      param.kd[2]=kd[2];
      param.pd[0]=pd[0];
      param.pd[1]=pd[1];
      param.xd=q[0];
      param.yd=q[1];
      
      double x[2];
      double fvec[2];
      
      x[0]=q[0];
      x[1]=q[1];
      
      long ltmp[2]={0, 0};
      double dtmp[16];
      
      memset(dtmp, 0, 16*sizeof(double));
      slmder(computeDistortion, 2, 2, x, fvec, &param, 1e-6, ltmp, dtmp);
      
      q[0]=x[0];
      q[1]=x[1];
    }
}

OrthoCamera::OrthoCamera(const Properties &prop)
  : Camera(prop, -1)
{
    prop.getValue("resolution", res);
    prop.getValue("depth.resolution", dres, "1");
    
    if (prop.contains("origin.T"))
    {
      Vector3d T;
      prop.getValue("origin.T", T);
      
      string origin;
      prop.getValue("origin", origin, "corner");
      if (origin == "center")
      {
        T[0]-=res/2;
        T[1]+=res/2;
      }
      
      setT(T);
    }
}

Camera *OrthoCamera::clone() const
{
    OrthoCamera *ret=new OrthoCamera();
    
    *ret=*this;
    
    return ret;
}

void OrthoCamera::setDownscaled(int ds)
{
    Camera::setDownscaled(ds);
    
    if (ds > 1)
    {
      res*=ds;
      dres*=ds;
    }
}

void OrthoCamera::setPart(long x, long y, long w, long h)
{
    Camera::setPart(x, y, w, h);
    
    Vector3d T=getT();
    
    T[0]+=x*res;
    T[1]-=y*res;
    
    setT(T);
}

void OrthoCamera::getProperties(Properties &prop, int id) const
{
    Camera::getProperties(prop, -1);
    
    prop.putValue("resolution", res);
    prop.putValue("depth.resolution", dres);
}

double OrthoCamera::projectPoint(Vector2d &p, const Vector3d &Pw) const
{
    Vector3d Pc=transpose(getR())*(Pw-getT());
    
    p[0]=Pc[0]/res;
    p[1]=-Pc[1]/res;
    
    return Pc[2]/dres;
}

void OrthoCamera::reconstructPoint(Vector3d &Pw, const Vector2d &p, double d)
  const
{
    if (!isfinite(d))
      throw InvalidArgumentException("Cannot reconstruct invalid point");
    
    Vector3d Pc;
    
    Pc[0]=p[0]*res;
    Pc[1]=-p[1]*res;
    Pc[2]=d*dres;
    
    Pw=getR()*Pc+getT();
}

void OrthoCamera::reconstructRay(Vector3d &V, Vector3d &C, const Vector2d &p)
  const
{
    V=getR()*Vector3d(0, 0, -1);
    
    C[0]=p[0]*res;
    C[1]=-p[1]*res;
    C[2]=0;
    
    C+=getT();
}

}
