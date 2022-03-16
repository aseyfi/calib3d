#!/usr/bin/env python

import cv2 as cv
import numpy as np
import glob
import sys

def Rodrigues(v):
    # rotation vecctor input
    if len(v.shape) == 1:
        v = v.reshape(3,1)
        theta = np.linalg.norm(v)
        u = v/theta
        u_x, u_y, u_z = u
        u_cross = np.array([[0, -u_z, u_y],[u_z, 0, -u_x],[-u_y, u_x, 0]])
        uuT = np.dot(u, u.T)
        # Calculate rotation matrix
        R = np.cos(theta) * np.eye(3) + np.sin(theta) * u_cross + (1 - np.cos(theta)) * uuT

        # Calculate jacobian
        A = np.zeros((4,3), dtype = float)
        A[0,0] = A[1,1] = A[2,2] = 1
        A[3,:] = u.T

        B = np.zeros((4,4), dtype = float)
        B[0,0] = B[1,1] = B[2,2] = 1/theta
        B[0:3,3] = -u.reshape(-1)/theta
        B[3,3] = 1
        #print(A)

        dRdu_x = np.sin(theta) * np.array([0, 0, 0, 0, 0, -1, 0, 1, 0], dtype = float).reshape(-1,1)
        dRdu_x = dRdu_x + (1 - np.cos(theta)) * np.array([2 * u_x, u_y, u_z, u_y, 0, 0, u_z, 0, 0], dtype = float).reshape(-1,1)

        dRdu_y = np.sin(theta) * np.array([0, 0, 1, 0, 0, 0, -1, 0, 0], dtype = float).reshape(-1,1)
        dRdu_y = dRdu_y + (1 - np.cos(theta)) * np.array([0, u_x, 0, u_x, 2 * u_y, u_z, 0, u_z, 0], dtype = float).reshape(-1,1)

        dRdu_z = np.sin(theta) * np.array([0, -1, 0, 1, 0, 0, 0, 0, 0], dtype = float).reshape(-1,1)
        dRdu_z = dRdu_z + (1 - np.cos(theta)) * np.array([0, 0, u_x, 0, 0, u_y, u_x, u_y, 2 * u_z], dtype = float).reshape(-1,1)

        dRdtheta = -np.sin(theta) * np.eye(3).reshape(-1,1) + np.cos(theta) *  u_cross.reshape(-1,1) + np.sin(theta) * uuT.reshape(-1,1)

        J_int = np.dot(B, A)
        J = np.dot(np.hstack((dRdu_x, dRdu_y, dRdu_z, dRdtheta)), J_int)

        return R, J

def project_points(objpt, fx, fy, cx, cy, alpha, k, rvec, t):
    #xx = a * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      2*kc(3)*a*b + kc(4)*(r^2 + 2*a^2);
    #yy = b * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      kc(3)*(r^2 + 2*b^2) + 2*kc(4)*a*b;
    R, _ = Rodrigues(rvec)
    X = np.matrix(R) * np.matrix(objpt).T + t.reshape(3,1)
    X = np.array(X)
    x  = X[0,:]
    y  = X[1,:]
    z  = X[2,:]
    a = x / z
    b = y / z
    r = np.sqrt(a**2 + b**2)
    xx = a * (1 + k[0] * r**2 + k[1] * r**4 + k[4] * r**6) + 2 * k[2] * a * b + k[3] * (r**2 + 2 * a**2)
    yy = b * (1 + k[0] * r**2 + k[1] * r**4 + k[4] * r**6) + k[2] * (r**2 + 2 * b**2) + 2 * k[3] * a * b
    xxp = fx * (xx + alpha * yy) + cx
    yyp = fy * (yy) + cy

    return np.hstack((xxp.reshape(-1,1), yyp.reshape(-1,1)))


def project_points_jacobian(objpt, fx, fy, cx, cy, alpha, k, rvec, t):
    def params_to_list(fx, fy, cx, cy, alpha, k, rvec, t):
        params = [fx] + [fy] + [cx] + [cy] + [alpha]
        params.extend(k.reshape(-1).tolist())
        params.extend(rvec.reshape(-1).tolist())
        params.extend(t.reshape(-1).tolist())
        return params

    def list_to_params(params):
        fx, fy, cx, cy, alpha = params[0:5]
        k = params[5:10]
        rvec = params[10:13]
        t = params[13:]
        return fx, fy, cx, cy, alpha, np.array(k), np.array(rvec), np.array(t)

    delta = 1e-6
    params = params_to_list(fx, fy, cx, cy, alpha, k, rvec, t)

    J = []
    for i in range(len(params)):
        params1 = list(params)
        params1[i] -= delta
        params2 = list(params)
        params2[i] += delta
        uv1 = project_points(objpt, *list_to_params(params1))
        uv2 = project_points(objpt, *list_to_params(params2))
        J_i = (uv2 - uv1)/(2 * delta)
        J.append(J_i.reshape(-1,1)) # row 1 = du/d., row2 = dv/d.

    J = np.hstack(tuple(J))
    return J


def compute_extrinsic_refine(imgpt, objpt, fx, fy, cx, cy, alpha, k, rvec_init, t_init):

    rvec = rvec_init
    t = t_init
    params = []
    params.extend(rvec.reshape(-1).tolist())
    params.extend(t.reshape(-1).tolist())
    params = np.array(params)


    for i in range(10):
        projected_imgpt = project_points(objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        J = project_points_jacobian(objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        # We are only optimizing for rvec and t, so we only need those columns
        JJ = np.matrix(J[:, 10:])

        # Optimization:
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
        ex = (imgpt - projected_imgpt).reshape(-1,1) # row 1 = u error, row2 = v error

        # calculate innovation
        JJ2 = JJ.T * JJ
        param_innov = np.array(np.linalg.inv(JJ2) * JJ.T * ex).reshape(-1)
        param_up = params + param_innov
        change = np.linalg.norm(param_innov)/np.linalg.norm(param_up)
        params = param_up
        rvec = np.array(params[:3]).reshape(-1)
        t = np.array(params[3:]).reshape(-1)

        if change < 1e-6:
            break

    return(rvec, t)

def normalize_pixel(imgpt,fx, fy, cx, cy, alpha, k):
    pt_distort_x = (imgpt[:,0] - cx)/fx; pt_distort_x = pt_distort_x.reshape(-1,1)
    pt_distort_y = (imgpt[:,1] - cy)/fy; pt_distort_y = pt_distort_y.reshape(-1,1)
    pt_distort = np.hstack((pt_distort_x, pt_distort_y))
    # undo skew
    pt_distort[:,0] = pt_distort[:,0] - alpha * pt_distort[:,1]
    return pt_distort


def homography_2d(src, dst):
    src = np.hstack((src, np.ones((src.shape[0], 1))))
    dst = np.hstack((dst, np.ones((dst.shape[0], 1))))
    # Section 4.1 of the book
    # dst = H*src1111
    # dst = [x_p, y_p, w_p = 1]
    # Algorithm 4.2, The normalized DLT for 2D homographies

    # i,ii) Normalization of x (In page 108 it says that this is not an optional step)
    def normalization_matrix(pts):
        avg = np.mean(pts, axis = 0)
        pts = pts - avg
        dist = np.linalg.norm(pts, axis = 1)
        dist_mean = np.mean(dist)
        scale = np.sqrt(2)/dist_mean
        Hnorm = np.matrix([[scale, 0, -avg[0]],[0, scale, -avg[1]],[0, 0, 1]])
        Hnorm_inv = np.matrix([[1/scale, 0, +avg[0]],[0, 1/scale, +avg[1]],[0, 0, 1]])
        return Hnorm, Hnorm_inv

    Tsrc, Tsrc_inv = normalization_matrix(src[:,:2])
    Tdst, Tdst_inv = normalization_matrix(dst[:,:2])
    src_n = np.array((Tsrc * np.matrix(src).T).T)
    dst_n = np.array((Tdst * np.matrix(dst).T).T)

    # iii) Apply algorithm 4.1
    A = np.zeros((src.shape[0] * 2, 9))
    for i in range(src.shape[0]):
        # writing equation 4.3 for for each pair of points:
        x = src_n[i,:]
        xp = dst_n[i,:]
        x[2] = 1
        xp[2] = 1
        A[i * 2, 3:6] = -xp[2]*x
        A[i * 2, 6:9] = xp[1]*x

        A[i * 2 + 1, :3] = xp[2]*x
        A[i * 2 + 1, 6:9] = -xp[0]*x

    # calculate SDV
    H = solution(A)
    H = np.matrix(H.reshape(3,3))
    H = Tdst_inv * H * Tsrc
    H = H / H[2,2]

    # Gauss-Newton Optimization
    if src.shape[0] > 4:
        # Optimization itteations
        for iter in range(10):
            # prepare the parameter vector
            hhv = np.array(H).reshape(-1)[:8]
            J = np.zeros((src.shape[0]*2, 8))
            imgpt_proj = np.array(H * np.matrix(src).T).T # Nx3
            imgpt_div_w = imgpt_proj / imgpt_proj[:,2].reshape(-1,1)
            objpt_div_w = src / imgpt_proj[:,2].reshape(-1,1) # MMM
            err = (dst[:,:2] - imgpt_div_w[:,:2]).reshape(-1,1)
            MM2 = objpt_div_w * imgpt_div_w[:,0].reshape(-1,1)
            MM3 = objpt_div_w * imgpt_div_w[:,1].reshape(-1,1)
            J[::2, :3] = -objpt_div_w
            J[::2, 6:] = MM2[:,:2]
            J[1::2, 3:6] = -objpt_div_w
            J[1::2, 6:] = MM3[:,:2]
            J = np.matrix(J)
            JJ = J.T * J
            param_innov = np.array(np.linalg.inv(JJ) * J.T * np.matrix(err)).reshape(-1)
            hhv_up = hhv - param_innov
            H = np.hstack((hhv_up, [1])).reshape(3,3)
            change = np.linalg.norm(param_innov)/np.linalg.norm(hhv_up)
            if change < 1e-9:
                break

    return np.array(H)

def compute_extrinsic_init(imgpt, objpt, fx, fy, cx, cy, alpha, k):
    # normaliize points to points on a plane at Z = 1
    # or fx=fy= 0, cx=cy= 0, alpha= 0
    imgpt_n = normalize_pixel(imgpt,fx, fy, cx, cy, alpha, k)
    # Assuming Planar Structure:
    H = homography_2d(objpt[:,:2], imgpt_n[:,:2])

    # from end of section 3.1 of paper
    # camera matrix is an identity matrix, because image points are normalized,
    # r1 = lambda * inv(A) * h1 = lambda * h1, I think setting norm(r1) = 1 will normalize for lambda
    # r2 = lambda * inv(A) * h2 = lambda * h2 = h2
    # r3 = r1 * r2

    r1 = H[:,0] / np.linalg.norm(H[:,0])
    r2 = H[:,1] / np.linalg.norm(H[:,1])
    r3 = np.cross(r1, r2)
    R = np.hstack((r1.reshape(3,1), r2.reshape(3,1), r3.reshape(3,1)))
    # Paper: R does not in general satisfy the properties of a rotation matrix.
    # Appendix C describes a method to estimate the best rotation matrix from a general 3 x 3 matrix
    # Bouguet is converting to rot vector and then back to rot matrix
    # ToDo: See appendix C of the paper
    rotvec, _ = cv.Rodrigues(R)
    #R, _ = cv.Rodrigues(rotvec)

    # It seems norm of h1 and h2 are different.
    # To calculate T we need lambda which is calculated as 1/norm(h1) = 1/norm(h2)
    # This is what Bouguet is doing. I am doing the same:
    lam =  1 / np.mean(np.linalg.norm(H[:,:2], axis = 0))
    T = lam * H[:,2].reshape(3,1)
    return (rotvec.reshape(-1), T.reshape(-1))

# Estimate Extrinsic values
def comp_ext_calib(imgpoints, objpoints, fx, fy, cx, cy, alpha, k):
    tc = []
    omc = []
    for imgpt, objpt in zip(imgpoints, objpoints):
        # format image and board 2D points
        imgpt = imgpt.reshape(-1,2)
        objpt = objpt.reshape(-1,3)
        (rvec, t) = compute_extrinsic_init(imgpt, objpt, fx, fy, cx, cy, alpha, k)
        # The paper says: The above solution is obtained through minimizing an algebraic distance which is not
        # physically meaningful. We can refine it through maximum likelihood inference.
        # Bouguet is doing the same.
        (rvec, t) = compute_extrinsic_refine(imgpt, objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        tc.append(t)
        omc.append(rvec)

    return omc, tc

# TZhang's paper: he solution to Vx = 0 is well known as the eigenvector of V^T*V associated with the smallest eigenvalue
# https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)]

# From equation 7 of Zhang's paper.
# Returns a row vector of 6 elements
def v_ij(hi, hj):
    v1 = hi[0] * hj[0]
    v2 = hi[0] * hj[1] + hi[1] * hj[0]
    v3 = hi[1] * hj[1]
    v4 = hi[2] * hj[0] + hi[0] * hj[2]
    v5 = hi[2] * hj[1] + hi[1] * hj[2]
    v6 = hi[2] * hj[2]
    return np.array([v1, v2, v3, v4, v5, v6])

def init_intrinsic_param(imgpoints, objpoints, (ny, nx)):
    V = []
    for imgpt, objpt in zip(imgpoints, objpoints):
        # format image and board 2D points
        imgpt = imgpt.reshape(-1,2)
        objpt = objpt.reshape(-1,3); objpt = objpt[:,0:2]
        # Calculate homography. See Algorithm 4.2 of MVG
        H = homography_2d(objpt[:,:2], imgpt[:,:2])
        # Calculate camera parameters from homography matrix. See Algorithm 8.2 of MVG
        V.append(v_ij(H[:,0], H[:,1])) # V12
        V.append(v_ij(H[:,0], H[:,0]) - v_ij(H[:,1], H[:,1])) # V11 - V22

    V = np.array(V)
    # We can use SDV to solve V*b = 0 where b = [w1, w2, w3, w4, w5, w6]^T
    [w1, w2, w3, w4, w5, w6] = solution(V)

    # Appendix B of the paper
    cy = (w2*w4 - w1*w5) / (w1*w3 - w2 ** 2)
    lam = w6 - (w4**2 + cy * (w2*w4 - w1*w5))/w1
    fx = np.sqrt(lam / w1)
    fy = np.sqrt(lam * w1 / (w1*w3 - w2**2))
    alpha = -w2 * fx**2 * fy / lam # image skew
    cx = alpha * cy / fy - w4 * fx**2/lam
    return (fx, fy, cx, cy, alpha)

def visualize_jacobian(J):
    img = np.zeros(J.shape)
    img[J > 0] = 255
    cv.imshow('Jacobian', cv.resize(img, (500, 500)))
    cv.waitKey(0)

def ml_optmize(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec):
    # main optimization
    print('Gradient descent iterations')
    # Initialize parameters
    params = [fx] + [fy] + [cx] + [cy] + [alpha]
    params.extend(k.reshape(-1).tolist())
    len_int = len(params)
    len_ext = 6
    for n in range(len(objpoints)):
        params.extend(rvec[n].reshape(-1).tolist())
        params.extend(tvec[n].reshape(-1).tolist())

    params = np.array(params)

    n_planes = len(objpoints)
    n_rows_per_plane = len(objpoints[0]) * 2

    # main optimization loop
    for ii in range(10):
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        alpha = params[4]
        k = params[5:10]

        ex = np.zeros((n_planes * n_rows_per_plane, 1))
        JJ = np.zeros((n_planes * n_rows_per_plane, len_int + len_ext * n_planes)) # 10 intrinsic and 6 * N extrinsics

        for n in range(n_planes):
            # update extrinsic parameters
            ext_params = params[len_int + len_ext * n:len_int + len_ext * (n+1)]
            rvec[n] = rvec_n = ext_params[:3]
            tvec[n] = tvec_n = ext_params[3:]

            # calculate error and javobian
            objpt_n = objpoints[n]
            imgpt_n = imgpoints[n]
            imgpt_n_prj = project_points(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            J_n = project_points_jacobian(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            ex_n = (imgpt_n - imgpt_n_prj).reshape(-1,1) # row 1 = u error, row2 = v error
            # Some Sparse optimization by separating parameter vector two bloxks, intrinsic and extrinsic
            # This method is called The Schur Complement Trick
            # https://grail.cs.washington.edu/projects/bal/bal.pdf
            # Starting with a non-efficient version
            ex[n * n_rows_per_plane :(n + 1) * n_rows_per_plane] = ex_n
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, :len_int] = J_n[:,:len_int]
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, len_int + len_ext * n:len_int + len_ext * (n+1)] = J_n[:,len_int:]

        #visualize_jacobian(JJ
        # calculate innovation
        JJ = np.matrix(JJ)
        JJ2 = JJ.T * JJ
        param_innov = np.array(np.linalg.inv(JJ2) * JJ.T * ex).reshape(-1)
        param_up = params + param_innov
        change = np.linalg.norm(param_innov)/np.linalg.norm(param_up)
        params = param_up
        if change < 1e-10:
            break

    print(np.mean(np.linalg.norm(ex, axis = 1)))
    return fx, fy, cx, cy, alpha, k, rvec, tvec

def sparse_ml_optmize(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec):
    # main optimization
    print('Gradient descent iterations')
    # Initialize parameters
    params = [fx] + [fy] + [cx] + [cy] + [alpha]
    params.extend(k.reshape(-1).tolist())
    len_int = len(params)
    len_ext = 6
    for n in range(len(objpoints)):
        params.extend(rvec[n].reshape(-1).tolist())
        params.extend(tvec[n].reshape(-1).tolist())

    params = np.array(params)

    n_planes = len(objpoints)
    n_rows_per_plane = len(objpoints[0]) * 2

    # main optimization loop
    lambda_smooth = 1.0
    for ii in range(100):
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        alpha = params[4]
        k = params[5:10]

        ex = np.zeros((n_planes * n_rows_per_plane, 1))
        JJ = np.zeros((n_planes * n_rows_per_plane, len_int + len_ext * n_planes)) # 10 intrinsic and 6 * N extrinsics

        for n in range(n_planes):
            # update extrinsic parameters
            ext_params = params[len_int + len_ext * n:len_int + len_ext * (n+1)]
            rvec[n] = rvec_n = ext_params[:3]
            tvec[n] = tvec_n = ext_params[3:]

            # calculate error and javobian
            objpt_n = objpoints[n]
            imgpt_n = imgpoints[n]
            imgpt_n_prj = project_points(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            J_n = project_points_jacobian(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            ex_n = (imgpt_n - imgpt_n_prj).reshape(-1,1) # row 1 = u error, row2 = v error
            # Some Sparse optimization by separating parameter vector two bloxks, intrinsic and extrinsic
            # This method is called The Schur Complement Trick
            # https://grail.cs.washington.edu/projects/bal/bal.pdf
            # Starting with a non-efficient version
            ex[n * n_rows_per_plane :(n + 1) * n_rows_per_plane] = ex_n
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, :len_int] = J_n[:,:len_int]
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, len_int + len_ext * n:len_int + len_ext * (n+1)] = J_n[:,len_int:]


        #visualize_jacobian(JJ
        # calculate innovation
        JJ = np.matrix(JJ)
        JJ2 = JJ.T * JJ
        # we can write JJ2 = [A, B;C, D]
        B = JJ2[:len_int, :len_int]
        E = JJ2[:len_int, len_int:]
        ET = JJ2[len_int:, :len_int]
        C = JJ2[len_int:, len_int:]

        # Apply damping factor
        lambda_smooth2  = 1-(1-lambda_smooth)**(ii + 1)
        vw = JJ.T*ex
        v = vw[:len_int]
        w = vw[len_int:]

        S = B - E * np.linalg.inv(C) * ET
        C_inv = np.linalg.inv(C)
        S_inv = np.linalg.inv(S)
        delta_y = S_inv * (v - E * C_inv*w)
        delta_z = C_inv * (w - ET * delta_y)

        param_innov = np.vstack((np.array(delta_y), np.array(delta_z))).reshape(-1)
        param_innov = lambda_smooth2 * param_innov

        #param_innov = np.array(np.linalg.inv(JJ2) * JJ.T * ex).reshape(-1)
        param_up = params + param_innov
        change = np.linalg.norm(param_innov)/np.linalg.norm(param_up)
        params = param_up
        if change < 1e-10:
            break

    print(np.sqrt(np.mean(ex**2)))
    return fx, fy, cx, cy, alpha, k, rvec, tvec


def go_calib_optim_iter(imgpoints, objpoints, (ny, nx)):
    #
    quick_init = False
    #
    nx = float(nx)
    ny = float(ny)
    print('Image Size = %sx%s' %(nx, ny))
    print('Initialization of the principal point at the center of the image')
    (cx, cy) = [(nx-1)/2, (ny-1)/2]
    print('Initialization of the image skew to zero')
    alpha = 0.0
    print('Initialization of distortion to zero')
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    print('Initialization of the focal length')
    if quick_init:
        fov_deg = 70
        print('Initialization of the focal length to an fov %s' %fov_deg)
        fov = fov_deg * (np.pi/180) # Assuming 70 degrees fov
        fx = fy = nx/(2 * np.tan(fov/2))
    else:
        print('nitialization of the intrinsic parameters using the vanishing points of planar patterns')
        (fx, fy, cx, cy, alpha) = init_intrinsic_param(imgpoints, objpoints, (ny, nx))

    rvec, tvec = comp_ext_calib(imgpoints, objpoints, fx, fy, cx, cy, alpha, k)

    # We have all of the initial values that we need
    fx, fy, cx, cy, alpha, k, rvec, tvec = ml_optmize(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec)
    #fx, fy, cx, cy, alpha, k, rvec, tvec = sparse_ml_optmize(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec)
    return fx, fy, cx, cy, alpha, k, rvec, tvec

def draw_corners(img, imgpt):
    radius = 3
    color = (255, 0, 0)
    thickness = 2

    for u , v in zip(imgpt[:,0], imgpt[:,1]):
        center_coordinates = (int(u), int(v))
        image = cv.circle(img, center_coordinates, radius, color, thickness)

    return image

def comp_fisheye_distortion(pt, k):
    # input: image points normalized to fx=fy=1, cx=cy=0
    # ouput: points with no fisheye distortion effect
    theta_d = np.linalg.norm(pt, axis = 1).reshape(-1,1)
    # solve equation theta_d = theta * (1 + k[0]*theta^2 + k[1]*theta^2 + k[2]*theta^6 + k[3]*theta^8)
    # Theta can be formulated as a function of thera as follows:
    # theta = theta_d /(1 + k[0]*theta^2 + k[1]*theta^4 + k[2]*theta^6 + k[3]*theta^8)
    # theta(K+1) = theta_d /(1 + k[0]*theta(k)^2 + k[1]*theta(k)^4 + k[2]*theta(k)^6 + k[3]*theta(k)^8)
    # where theta(k+1) and theta(k) are theta at itteration K+1 and k respectively.
    theta = theta_d # initial value

    for i in range(20):
        theta = theta_d /(1 + k[0]*theta**2 + k[1]*theta**4 + k[2]*theta**6 + k[3]*theta**8)

    scaling = np.tan(theta)/theta_d
    return scaling * pt

def normalize_pixel_fisheye(imgpt,fx, fy, cx, cy, alpha, k):
    pt_distort_x = (imgpt[:,0] - cx)/fx; pt_distort_x = pt_distort_x.reshape(-1,1)
    pt_distort_y = (imgpt[:,1] - cy)/fy; pt_distort_y = pt_distort_y.reshape(-1,1)
    pt_distort = np.hstack((pt_distort_x, pt_distort_y))
    # undo skew
    pt_distort[:,0] = pt_distort[:,0] - alpha * pt_distort[:,1]
    # remove fisheye distortiion effect
    pt_normalized = comp_fisheye_distortion(pt_distort, k)
    return pt_normalized

def compute_extrinsic_init_fisheye(imgpt, objpt, fx, fy, cx, cy, alpha, k):
    # normaliize points to points on a plane at Z = 1
    # or fx=fy= 0, cx=cy= 0, alpha= 0
    imgpt_n = normalize_pixel_fisheye(imgpt,fx, fy, cx, cy, alpha, k)
    # Assuming Planar Structure:
    H = homography_2d(objpt[:,:2], imgpt_n[:,:2])

    # from end of section 3.1 of paper
    # camera matrix is an identity matrix, because image points are normalized,
    # r1 = lambda * inv(A) * h1 = lambda * h1, I think setting norm(r1) = 1 will normalize for lambda
    # r2 = lambda * inv(A) * h2 = lambda * h2 = h2
    # r3 = r1 * r2

    r1 = H[:,0] / np.linalg.norm(H[:,0])
    r2 = H[:,1] / np.linalg.norm(H[:,1])
    r3 = np.cross(r1, r2)
    R = np.hstack((r1.reshape(3,1), r2.reshape(3,1), r3.reshape(3,1)))
    # Paper: R does not in general satisfy the properties of a rotation matrix.
    # Appendix C describes a method to estimate the best rotation matrix from a general 3 x 3 matrix
    # Bouguet is converting to rot vector and then back to rot matrix
    # ToDo: See appendix C of the paper
    rotvec, _ = cv.Rodrigues(R)
    #R, _ = cv.Rodrigues(rotvec)

    # It seems norm of h1 and h2 are different.
    # To calculate T we need lambda which is calculated as 1/norm(h1) = 1/norm(h2)
    # This is what Bouguet is doing for plumb bob model, but not for fisheye.
    # I am doing the same for fishey too:
    lam =  1 / np.mean(np.linalg.norm(H[:,:2], axis = 0))
    T = lam * H[:,2].reshape(3,1)
    return (rotvec.reshape(-1), T.reshape(-1))

def project_points_fisheye(objpt, fx, fy, cx, cy, alpha, k, rvec, t):
    #xx = a * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      2*kc(3)*a*b + kc(4)*(r^2 + 2*a^2);
    #yy = b * (1 + kc(1)*r^2 + kc(2)*r^4 + kc(5)*r^6)      +      kc(3)*(r^2 + 2*b^2) + 2*kc(4)*a*b;
    R, _ = Rodrigues(rvec)
    X = np.matrix(R) * np.matrix(objpt).T + t.reshape(3,1)
    X = np.array(X)
    x  = X[0,:]
    y  = X[1,:]
    z  = X[2,:]
    a = x / z
    b = y / z
    r = np.sqrt(a**2 + b**2)
    theta = np.arctan(r)
    theta_d =  theta * (1 + k[0]*theta**2 + k[1]*theta**4 + k[2]*theta**6 + k[3]*theta**8)

    xx = (theta_d / r) * a
    yy = (theta_d / r) * b

    xxp = fx * (xx + alpha * yy) + cx
    yyp = fy * (yy) + cy

    return np.hstack((xxp.reshape(-1,1), yyp.reshape(-1,1)))


def project_points_jacobian_fisheye(objpt, fx, fy, cx, cy, alpha, k, rvec, t):
    def params_to_list(fx, fy, cx, cy, alpha, k, rvec, t):
        params = [fx] + [fy] + [cx] + [cy] + [alpha]
        params.extend(k.reshape(-1).tolist())
        params.extend(rvec.reshape(-1).tolist())
        params.extend(t.reshape(-1).tolist())
        return params

    def list_to_params(params):
        fx, fy, cx, cy, alpha = params[0:5]
        k = params[5:9]
        rvec = params[9:12]
        t = params[12:]
        return fx, fy, cx, cy, alpha, np.array(k), np.array(rvec), np.array(t)

    delta = 1e-8
    params = params_to_list(fx, fy, cx, cy, alpha, k, rvec, t)

    J = []
    for i in range(len(params)):
        params1 = list(params)
        params1[i] -= delta
        params2 = list(params)
        params2[i] += delta
        uv1 = project_points_fisheye(objpt, *list_to_params(params1))
        uv2 = project_points_fisheye(objpt, *list_to_params(params2))
        J_i = (uv2 - uv1)/(2 * delta)
        J.append(J_i.reshape(-1,1)) # row 1 = du/d., row2 = dv/d.

    J = np.hstack(tuple(J))
    return J


def compute_extrinsic_refine_fisheye(imgpt, objpt, fx, fy, cx, cy, alpha, k, rvec_init, t_init):

    rvec = rvec_init
    t = t_init
    params = []
    params.extend(rvec.reshape(-1).tolist())
    params.extend(t.reshape(-1).tolist())
    params = np.array(params)


    for i in range(10):
        projected_imgpt = project_points_fisheye(objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        J = project_points_jacobian_fisheye(objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        # We are only optimizing for rvec and t, so we only need those columns
        JJ = np.matrix(J[:, 9:])

        # Optimization:
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
        ex = (imgpt - projected_imgpt).reshape(-1,1) # row 1 = u error, row2 = v error

        # calculate innovation
        JJ2 = JJ.T * JJ
        param_innov = np.array(np.linalg.inv(JJ2) * JJ.T * ex).reshape(-1)
        param_up = params + param_innov
        change = np.linalg.norm(param_innov)/np.linalg.norm(param_up)
        params = param_up
        rvec = np.array(params[:3]).reshape(-1)
        t = np.array(params[3:]).reshape(-1)
        if change < 1e-6:
            break

    return(rvec, t)

def comp_ext_calib_fisheye(imgpoints, objpoints, fx, fy, cx, cy, alpha, k):
    tc = []
    omc = []
    for imgpt, objpt in zip(imgpoints, objpoints):
        # format image and board 2D points
        imgpt = imgpt.reshape(-1,2)
        objpt = objpt.reshape(-1,3)
        (rvec, t) = compute_extrinsic_init_fisheye(imgpt, objpt, fx, fy, cx, cy, alpha, k)
        # The paper says: The above solution is obtained through minimizing an algebraic distance which is not
        # physically meaningful. We can refine it through maximum likelihood inference.
        # Bouguet is doing the same.
        (rvec, t) = compute_extrinsic_refine_fisheye(imgpt, objpt, fx, fy, cx, cy, alpha, k, rvec, t)
        tc.append(t)
        omc.append(rvec)

    return omc, tc


def ml_optmize_fisheye(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec):
    # main optimization
    print('Gradient descent iterations')
    # Initialize parameters
    params = [fx] + [fy] + [cx] + [cy] + [alpha]
    params.extend(k.reshape(-1).tolist())
    len_int = len(params)
    len_ext = 6
    for n in range(len(objpoints)):
        params.extend(rvec[n].reshape(-1).tolist())
        params.extend(tvec[n].reshape(-1).tolist())

    params = np.array(params)

    n_planes = len(objpoints)
    n_rows_per_plane = len(objpoints[0]) * 2

    # main optimization loop
    for ii in range(20):
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        alpha = params[4]
        k = params[5:9]

        ex = np.zeros((n_planes * n_rows_per_plane, 1))
        JJ = np.zeros((n_planes * n_rows_per_plane, len_int + len_ext * n_planes)) # 10 intrinsic and 6 * N extrinsics

        for n in range(n_planes):
            # update extrinsic parameters
            ext_params = params[len_int + len_ext * n:len_int + len_ext * (n+1)]
            rvec[n] = rvec_n = ext_params[:3]
            tvec[n] = tvec_n = ext_params[3:]

            # calculate error and javobian
            objpt_n = objpoints[n]
            imgpt_n = imgpoints[n]
            imgpt_n_prj = project_points_fisheye(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            J_n = project_points_jacobian_fisheye(objpt_n, fx, fy, cx, cy, alpha, k, rvec_n, tvec_n)
            ex_n = (imgpt_n - imgpt_n_prj).reshape(-1,1) # row 1 = u error, row2 = v error
            # Some Sparse optimization by separating parameter vector two bloxks, intrinsic and extrinsic
            # This method is called The Schur Complement Trick
            # https://grail.cs.washington.edu/projects/bal/bal.pdf
            # Starting with a non-efficient version
            ex[n * n_rows_per_plane :(n + 1) * n_rows_per_plane] = ex_n
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, :len_int] = J_n[:,:len_int]
            JJ[n * n_rows_per_plane :(n + 1) * n_rows_per_plane, len_int + len_ext * n:len_int + len_ext * (n+1)] = J_n[:,len_int:]

        #visualize_jacobian(JJ
        # calculate innovation
        JJ = np.matrix(JJ)
        JJ2 = JJ.T * JJ
        param_innov = np.array(np.linalg.inv(JJ2) * JJ.T * ex).reshape(-1)
        param_up = params + param_innov
        change = np.linalg.norm(param_innov)/np.linalg.norm(param_up)
        params = param_up
        if change < 1e-10:
            break

    print(np.mean(np.linalg.norm(ex, axis = 1)))
    return fx, fy, cx, cy, alpha, k, rvec, tvec


def go_calib_optim_iter_fisheye(imgpoints, objpoints, (ny, nx)):
    # Initialize intrinsics
    #fx = fy = np.max((ny, nx)) / np.pi
    fx = 300.0
    fy = 300.0

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    alpha = 0.0
    k = np.array([0.0,0,0,0])

    # initialize extrinsics
    rvec, tvec = comp_ext_calib_fisheye(imgpoints, objpoints, fx, fy, cx, cy, alpha, k)
    fx, fy, cx, cy, alpha, k, rvec, tvec = ml_optmize_fisheye(imgpoints, objpoints, fx, fy, cx, cy, alpha, k, rvec, tvec)
    return fx, fy, cx, cy, alpha, k, rvec, tvec

def main():
    params = {'board_size' : (8,11), 'vis_detection': False, 'model':'fisheye'}
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Generate Object Points
    board_size = params['board_size']
    objp = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = np.hstack((objp.astype(float),np.zeros((objp.shape[0],1))))
    objp = objp.reshape(-1,1,3)

    # Load the images
    #img_dir = 'sample_images/AR0144_narrow_fov'
    img_dir = 'sample_images/AR0144_fisheye'
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = []
    for image_name in glob.glob(img_dir + '/*.png'):
        img = cv.imread(image_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board_size, flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

        # If found, add object points, image points (after refining them)
        if ret == True:
            images.append(gray)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2.reshape(-1,2))
            # Draw and display the corners
            cv.drawChessboardCorners(img, board_size, corners2, ret)

        if params['vis_detection']:
            cv.imshow("image", img)
            if cv.waitKey(0) & 0xFF == ord('q'):
                sys.exit(0)

    if params['model'] == 'plumb_bob':
        fx, fy, cx, cy, alpha, k, rvec, tvec = go_calib_optim_iter(imgpoints, objpoints, gray.shape)

        print('fx = %s' %fx)
        print('fy = %s' %fy)
        print('cx = %s' %cx)
        print('cy = %s' %cy)
        print('skew = %s' %alpha)
        print('k = %s' %k.tolist())

        for pt3d, r, t, img in zip(objpoints, rvec, tvec, images):
            pt2d_projected = project_points(pt3d, fx, fy, cx, cy, alpha, k, r, t)
            image = draw_corners(img, pt2d_projected)
            cv.imshow("IMG", image)
            key = cv.waitKey(0)
            if key == ord('q'):
                cv.destroyAllWindows()
                sys.exit(0)

    elif params['model'] == 'fisheye':
        fx, fy, cx, cy, alpha, k, rvec, tvec = go_calib_optim_iter_fisheye(imgpoints, objpoints, gray.shape)
        print('fx = %s' %fx)
        print('fy = %s' %fy)
        print('cx = %s' %cx)
        print('cy = %s' %cy)
        print('skew = %s' %alpha)
        print('k = %s' %k.tolist())

        for pt3d, r, t, img in zip(objpoints, rvec, tvec, images):
            pt2d_projected = project_points_fisheye(pt3d, fx, fy, cx, cy, alpha, k, r, t)
            image = draw_corners(img, pt2d_projected)
            cv.imshow("IMG", image)
            key = cv.waitKey(0)
            if key == ord('q'):
                cv.destroyAllWindows()
                sys.exit(0)

if __name__ == "__main__":
    main()
