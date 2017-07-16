#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;

  n_x_ = 5; //  (px, py, v, yaw, yaw_rate)

  n_aug_ = 7;  // (px, py, v, yaw, yaw_rate, noise_a, noise_y)

  n_sig_ = 15; // augmentation dimension

  n_z_radar_ = 3; // radar dimensions
  n_z_lidar_ = 2; // lidar dimension

  lambda_= -4; // sigma scaling

  x_ = VectorXd::Zero(n_x_);
  P_ = MatrixXd::Zero(n_x_, n_x_);
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);
  weights_ = VectorXd::Zero(n_sig_);
  R_lidar_ = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);
  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);


  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Zero(n_x_,   n_x_);

  Q_ = MatrixXd::Zero(2, 2); // Process Noise Covariance Matrix


  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.4; // default 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // default 30

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;



}

float NormalizeAngle(float angle) {
  return atan2(sin(angle), cos(angle));
}

float Atan2M(float y, float x){
  return atan2(y, x);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  // Initialisation
  if(!is_initialized_)
  {
    float px = 0, py = 0, v = 0;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      // extract the RADAR measurements and convert from
      // Polar to Cartesian coordinates
      float range = meas_package.raw_measurements_[0];
      float bearing = meas_package.raw_measurements_[1];
      float range_rate = meas_package.raw_measurements_[2];

      // calculate position and velocity
      px = range * cos(bearing);
      py = range * sin(bearing);
      v = range_rate;

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      // if it is laser, just grab the raw x, y coordinates
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
    }

    x_ << px , py , v, 0, 0;
    P_ << 1, 0, 0, 0, 0,
                  0, 1, 0, 0, 0,
                  0, 0, 100, 0, 0,
                  0, 0, 0, 100, 0,
                  0, 0, 0, 0, 1;

    // weights initialisation
    weights_.fill(0.5 / (n_aug_ + lambda_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    // init radar matrix
    R_radar_ << std_radr_ * std_radr_ , 0, 0,
                    0, std_radphi_ * std_radphi_, 0,
                    0, 0, std_radrd_ * std_radrd_;

    // lidar matrix
    R_lidar_ << std_laspx_ * std_laspx_ , 0,
                    0, std_laspy_ * std_laspy_;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  }

  // prediction
  // elapsed seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);


  // update
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
          UpdateRadar(meas_package);
  }
  else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
          UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  // generate sigma points
  VectorXd x_aug = VectorXd::Zero(n_aug_); // augmentation vector
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_); // augmented state covariance
  MatrixXd Q = MatrixXd::Zero(n_aug_ - n_x_, n_aug_ - n_x_); // process noise covariance
  Q << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

  // sigma points matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);
  x_aug << x_, 0, 0;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;
  MatrixXd A_aug = P_aug.llt().matrixL();

  // create sigma points
  MatrixXd term_aug = sqrt(lambda_ + n_aug_) * A_aug;
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = term_aug.colwise() + x_aug;
  Xsig_aug.block(0, 1 + n_aug_, n_aug_, n_aug_) = (-1 * term_aug).colwise() + x_aug;

  // sigma point prediction ------

  VectorXd vs = Xsig_aug.row(2);
  VectorXd yaws = Xsig_aug.row(3);
  VectorXd yawds = Xsig_aug.row(4);
  VectorXd nu_as = Xsig_aug.row(5);
  VectorXd nu_yawdds = Xsig_aug.row(6);

  ArrayXd new_yaws = (yaws + yawds * delta_t).array();
  ArrayXd ratios = vs.array() / yawds.array();
  MatrixXd dvalues = MatrixXd::Zero(n_x_, n_sig_);
  dvalues.row(0) = (yawds.array() > 0).select(ratios * (new_yaws.sin() - yaws.array().sin()),
          delta_t * vs.array() * yaws.array().cos());
  dvalues.row(1) = (yawds.array() > 0).select(ratios * (yaws.array().cos() - new_yaws.cos()), delta_t * vs.array() * yaws.array().sin());
  dvalues.row(3) = yawds * delta_t;

  MatrixXd noises = MatrixXd::Zero(n_x_, n_sig_);
  noises.row(0) = delta_t * delta_t * 0.5 * nu_as.array() * yaws.array().cos();
  noises.row(1) = delta_t * delta_t * 0.5 * nu_as.array() * yaws.array().sin();
  noises.row(2) = delta_t * nu_as;
  noises.row(3) = delta_t * delta_t * 0.5 * nu_yawdds;
  noises.row(4) = delta_t * nu_yawdds;

  Xsig_pred_ = Xsig_aug.topRows(n_x_) + dvalues + noises;

  // predict mean and convariance matrix

  x_ = (Xsig_pred_ * weights_.asDiagonal()).rowwise().sum();
  MatrixXd diff = Xsig_pred_.colwise() - x_;
  diff.row(3) = diff.row(3).array().unaryExpr(&NormalizeAngle);
  P_ = (diff * weights_.asDiagonal()) * diff.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  MatrixXd Zsig = MatrixXd::Zero(n_z_lidar_, n_sig_);
  VectorXd z_pred = VectorXd::Zero(n_z_lidar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  Zsig.row(0) = Xsig_pred_.row(0);
  Zsig.row(1) = Xsig_pred_.row(1);

  z_pred = (Zsig * weights_.asDiagonal()).rowwise().sum();

  MatrixXd diff = Zsig.colwise() - z_pred;
  diff.row(1) = diff.row(1).array().unaryExpr(&NormalizeAngle);
  S = (diff * weights_.asDiagonal()) * diff.transpose();
  S = S + R_lidar_;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_lidar_);
  MatrixXd x_diffs = Xsig_pred_.colwise() - x_;
  MatrixXd z_diffs = Zsig.colwise() - z_pred;
  x_diffs.row(3) = x_diffs.row(3).array().unaryExpr(&NormalizeAngle);
  z_diffs.row(1) = z_diffs.row(1).array().unaryExpr(&NormalizeAngle);
  Tc = (x_diffs * weights_.asDiagonal()) * z_diffs.transpose();

  MatrixXd K = Tc * S.inverse();
  VectorXd z = VectorXd(n_z_lidar_);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  VectorXd z_diff = z - z_pred;

  z_diff(1) = NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  ArrayXd pxs = Xsig_pred_.row(0);
  ArrayXd pxs_sq = pxs * pxs;

  ArrayXd pys = Xsig_pred_.row(1);
  ArrayXd pys_sq = pys * pys;

  ArrayXd vs = Xsig_pred_.row(2);
  ArrayXd yaws = Xsig_pred_.row(3);
  ArrayXd vs1 = yaws.cos() * vs;
  ArrayXd vs2 = yaws.sin() * vs;

  ArrayXd rs = (pxs_sq + pys_sq).sqrt();
  rs = (rs > 0).select(rs, 0);
  Zsig.row(0) = rs;
  Zsig.row(1) = pys.binaryExpr(pxs, &Atan2M);
  Zsig.row(2) = (pxs * vs1 + pys * vs2) / rs;

  z_pred = (Zsig * weights_.asDiagonal()).rowwise().sum();

  MatrixXd diff = Zsig.colwise() - z_pred;
  diff.row(1) = diff.row(1).array().unaryExpr(&NormalizeAngle);
  S = (diff * weights_.asDiagonal()) * diff.transpose();
  S = S + R_radar_;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
  MatrixXd x_diffs = Xsig_pred_.colwise() - x_;
  MatrixXd z_diffs = Zsig.colwise() - z_pred;
  x_diffs.row(3) = x_diffs.row(3).array().unaryExpr(&NormalizeAngle);
  z_diffs.row(1) = z_diffs.row(1).array().unaryExpr(&NormalizeAngle);
  Tc = (x_diffs * weights_.asDiagonal()) * z_diffs.transpose();


  MatrixXd K = Tc * S.inverse();

  VectorXd z = VectorXd(n_z_radar_);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      meas_package.raw_measurements_[2];
  VectorXd z_diff = z - z_pred;

  z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

