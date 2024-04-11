/*
 *
 *    Copyright (c) 2013-2015
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */
#include<iostream>
#include<cmath>
#include<exception>
#include<cassert>

//#include "include/logging.h"
#include "include/ars.h"
#include "include/constants.h"
//#include "include/fpenvironment.h"

namespace Smash {

namespace Rejection {
std::ostream &operator<<(std::ostream &out, const Point &p) {
  out << "Point= (" << p.x << ',' << p.y << ',' << p.expy << ")\n";
  return out;
}

std::ostream &operator<<(std::ostream &out, const Line &l) {
  out << "Line= " << l.m << " * x + " << l.b << "\n";
  return out;
}

auto ran = Random::uniform_dist<double>(0.0, 1.0);
/** contrustructor for AdaptiveRejectionSampler
 * param: func distribution function f_(x)
 */
AdaptiveRejectionSampler::AdaptiveRejectionSampler(
    std::function<double(double)> func, double xmin, double xmax):
    f_(func), xmin_(xmin), xmax_(xmax) {
  // const auto &log = logger<LogArea::Sampling>();
  /** judge if f_(xmin_)<FLT_MIN or f_(xmax_)<FLT_MIN,
   * change the range automatically to make it work
   * in ARS since we need log(f_(x))*/
  int nloop = 1;

  {
    /** disable double traps here since probability can goes to
     * really small as we expected; we need to judge it and
     * shrink the range (xmin, xmax) to get ride of it */
    // DisabledoubleTraps guard(FE_DIVBYZERO | FE_INVALID);


    //std::cout << "f_(xmin_) = " << f_(xmin_) << std::endl;
    //std::cout << "f_(xmax_) = " << f_(xmax_) << std::endl;
    while ( f_(xmin_) < std::numeric_limits<float>::min() ) {
      xmin_ += nloop*really_small;
      nloop *= 2;
      // log.debug() << "xmin_ is changed to " << xmin_ << std::endl;
      // std::cout << "xmin_ is changed to " << xmin_ << std::endl;

      if ( xmin_ > xmax_ ) {
        // log.fatal() << "xmin_ > xmax_ " << std::endl;
        throw std::runtime_error(
            "Error: xmin_ > xmax_ in ARS during shrinking range");
      }
    }

    nloop = 1;
    while ( f_(xmax_) < std::numeric_limits<float>::min() ) {
      xmax_ -= nloop*really_small;
      nloop *= 2;
      // log.debug() << "xmax_ is changed to " << xmax_ << std::endl;
      std::cout << "xmax_ is changed to " << xmax_ << std::endl;
      if ( xmin_ > xmax_ ) {
        // log.fatal() << "xmax_ < xmin_ " << std::endl;
        throw std::runtime_error(
            "Error: xmin_ > xmax_ in ARS during shrinking range");
      }
    }
  }  // only disable underflow double traps inside the brackets

  double dx = (xmax_ - xmin_)/static_cast<double>(init_npoint_-1);

  Point p;
  for ( int i=0; i < init_npoint_; i++ ) {
    p.x = xmin_+i*dx;
    p.expy = f_(p.x);
    p.y = std::log(p.expy);
    points_.push_back(p);
  }

  init_scant();
  init_inter();
  update_area();
}

/** Set max_refine_loops by hand */
void AdaptiveRejectionSampler::reset_max_refine_loops(const int
                                                      new_max_refine_loops) {
  max_refine_loops_ = new_max_refine_loops;
}


inline Line AdaptiveRejectionSampler::create_line(Point p0, Point p1) {
  Line l1;
  // const auto &log = logger<LogArea::Sampling>();
  if ( std::abs((p1).x-(p0).x) < really_small ) {
    // log.fatal() << "the slope is too big" << std::endl;
    // log.fatal() << "p1.x=" << p1.x << " p0.x=" << p0.x << std::endl;
    throw std::runtime_error(
        "Error: unsafe to create scant from 2 points "
        "whose positions are too close since we need the slope.");
  }

  l1.m = ((p1).y - (p0).y)/((p1).x-(p0).x);
  l1.b = (p0).y - l1.m*(p0).x;
  return l1;
}

/** Get scants_ according to points_ */
void AdaptiveRejectionSampler::init_scant() {
  for ( auto p0=points_.begin(); p0 != std::prev(points_.end(), 1);
       p0++ ) {
    auto p1 = std::next(p0, 1);
    scants_.emplace_back(create_line(*p0, *p1));
  }
}

/** Calc intersection of two scants_
 * \todo: looks like in linux: the most left intersection is wrongly 
 * calculateted by calling this function*/
inline Point AdaptiveRejectionSampler::create_inter(Line l0, Line l2) {
  if ( std::abs((l0).m-(l2).m) < really_small ) {
       //throw std::runtime_error(
       //"Error: unable to get intersection of 2 lines "
       //"that are parallel to each other.");
      std::cerr << "two parallel scants dont interact" << std::endl;
  }

  double x, y, expy;
  x = (l2.b-l0.b)/(l0.m-l2.m);
  y = l0.b+l0.m*x;
  expy = std::exp(y);

  Point p;
  p.x = x;
  p.y = y;
  p.expy = expy;

  if ( std::isnan(x) ) {
    std::cerr << "calc intersection:\n";
    std::cerr << "l0=" << l0;
    std::cerr << "l2=" << l2;
    throw std::runtime_error(
        "Error: the x coordinates of intersection is nan.");
  }
  return p;
}

/** leftmost point in upper bounds */
inline void AdaptiveRejectionSampler::create_leftend() {
  auto l1 = std::next(scants_.begin(), 1);
  Point p0;
  p0.x = (*points_.begin()).x;
  p0.y = (*l1).b + (*l1).m*p0.x;
  p0.expy = std::exp(p0.y);
  inters_.insert(inters_.begin(), p0);
}

/** rightmost point in upper bounds */
inline void AdaptiveRejectionSampler::create_rightend() {
  Point p0;
  auto l1 = std::prev(scants_.end(), 2);
  p0.x = (*std::prev(points_.end(), 1)).x;
  p0.y = (*l1).b + (*l1).m*p0.x;
  p0.expy = std::exp(p0.y);
  inters_.emplace_back(std::move(p0));
}

/** get all intersection points in upper bounds */
void AdaptiveRejectionSampler::init_inter() {
  for (auto l0=scants_.begin(); l0 != std::prev(scants_.end(), 2); l0++) {
    auto l2 = std::next(l0, 2);
    inters_.emplace_back(create_inter(*l0, *l2));
  }
  create_leftend();
  create_rightend();
}

/// construct the jth upper bounds from scants_
inline Line AdaptiveRejectionSampler::upper(int j) {
  // j&1 == j%2, while j&1 should be faster
  return (j&1) ? scants_.at((j+1)/2-1) : scants_.at((j+1)/2+1);
}

/// construct the jth lower bounds from scants_
Line AdaptiveRejectionSampler::lower(int j) {
  return scants_.at((j+1)/2);
}


/** get areas_ below piecewise exponential upper bounds
*/
void AdaptiveRejectionSampler::update_area() {
  areas_.resize(0);
  upper_bounds_.resize(0);
  auto it0 = inters_.begin();
  auto it1 = std::next(points_.begin(), 1);
  int j = 0;

  // Aj: The area under the jth piece exponential function
  double Aj;

  int loops = 0;
  for (; it0 != std::prev(inters_.end(), 1); it0++, it1++) {
    // (left) intersection--->(right) point
    if ( std::abs(upper(j).m) > really_small ) {
      Aj = ((*it1).expy-(*it0).expy)/upper(j).m;
    } else {
      // if it happens in really tiny opportunity that the slope == 0
      Aj = (*it1).expy * ((*it1).x - (*it0).x);
    }
    areas_.push_back(Aj);
    upper_bounds_.push_back({*it0, *it1, upper(j)});

    j++;
    // (left) point--->(right) intersection
    if ( std::abs(upper(j).m) > really_small ) {
      Aj = ((*std::next(it0, 1)).expy-(*it1).expy)/upper(j).m;
    } else {
      // if it happens in really tiny opportunity that the slope == 0
      Aj = (*it1).expy * ((*std::next(it0, 1)).x - (*it1).x);
    }
    areas_.push_back(Aj);
    upper_bounds_.push_back({*it1, *std::next(it0, 1), upper(j)});
    j++;

    loops++;
    if (loops > 100000) {
        std::cerr << "update too many times; it0 may go wild" << std::endl;
    }
  }
  discrete_distribution_.reset_weights(areas_);
}


/** refine scants_, intersections, upper and lowers bounds with
 * the new rejected point, recalculate area sequence
 * param: j the id of the scant, not the id of area
 */
void AdaptiveRejectionSampler::adaptive_update(
    const int j, const Point & new_rejection) {
  const int nscants = scants_.size();
  // const auto & log = logger<LogArea::Sampling>();
  Line l[2];
  Point ints[4];

  // don't update if the new point is too close to existing points
  // and intersections

  if ( std::fabs(new_rejection.x - points_.at(j).x) < really_small ||
      std::fabs(new_rejection.x - points_.at(j+1).x) < really_small ||
      std::fabs(new_rejection.x - inters_.at(j).x) < really_small ) {
    return;
  }

  points_.insert(std::next(points_.begin(), j+1), new_rejection);

  /// scants_ -1 +2 for all cases with the new rejection
  auto it = points_.begin();
  auto p0 = std::next(it, j);
  auto p1 = std::next(it, j+1);
  auto p2 = std::next(it, j+2);
  l[0] = create_line(*p0, *p1);

  if ( std::isinf(l[0].m) ) {
    std::cerr << "l0, j=" << j;
    std::cerr << "\n p0=" << *p0;
    std::cerr << "\n p1=" << *p1;
  }

  if ( std::isinf(l[1].m) ) {
    std::cerr << "l1, j=" << j;
  }

  l[1] = create_line(*p1, *p2);
  scants_.insert(std::next(scants_.begin(), j), {l[0], l[1]});
  scants_.erase(std::next(scants_.begin(), j+2));

  auto it0 = inters_.begin();
  auto it1 = inters_.begin();

  if ( j > 1 && j < nscants-2 ) {
    /// most of the case, intersections -3 +4
    ints[0] = create_inter(l[0], *std::next(scants_.begin(), j-2));
    ints[1] = create_inter(l[1], *std::next(scants_.begin(), j-1));
    ints[2] = create_inter(l[0], *std::next(scants_.begin(), j+2));
    ints[3] = create_inter(l[1], *std::next(scants_.begin(), j+3));
    inters_.insert(std::next(inters_.begin(), j-1),
                   {ints[0], ints[1], ints[2], ints[3]});
    it0 = std::next(inters_.begin(), j+3);
    it1 = std::next(inters_.begin(), j+6);
    inters_.erase(it0, it1);
  } else if ( j == 0 ) {
    /// for left most piece, boundary intersections -2 +3
    ints[0] = create_inter(l[0], *std::next(scants_.begin(), j+2));
    ints[1] = create_inter(l[1], *std::next(scants_.begin(), j+3));
    inters_.insert(std::next(inters_.begin(), j), {ints[0], ints[1]});
    it0 = std::next(inters_.begin(), j+2);
    it1 = std::next(inters_.begin(), j+4);
    inters_.erase(it0, it1);
    create_leftend();
  } else if ( j == 1 ) {
    /// for next to left most piece, intersections -3 +4
    ints[1] = create_inter(l[1], *std::next(scants_.begin(), j-1));
    ints[2] = create_inter(l[0], *std::next(scants_.begin(), j+2));
    ints[3] = create_inter(l[1], *std::next(scants_.begin(), j+3));
    inters_.insert(std::next(inters_.begin(), j-1),
                   {ints[1], ints[2], ints[3]});
    it0 = std::next(inters_.begin(), j+2);
    it1 = std::next(inters_.begin(), j+5);
    inters_.erase(it0, it1);
    create_leftend();
  } else if ( j == nscants-2 ) {
    /// for prev to right most piece, intersections -3 +4
    ints[0] = create_inter(l[0], *std::next(scants_.begin(), j-2));
    ints[1] = create_inter(l[1], *std::next(scants_.begin(), j-1));
    ints[2] = create_inter(l[0], *std::next(scants_.begin(), j+2));
    inters_.insert(std::next(inters_.begin(), j-1),
                   {ints[0], ints[1], ints[2]});
    it0 = std::next(inters_.begin(), j+2);
    it1 = std::next(inters_.begin(), j+5);
    inters_.erase(it0, it1);
    create_rightend();
  } else if ( j == nscants-1 ) {
    /// for right most piece, intersections -2 +3
    ints[0] = create_inter(l[0], *std::next(scants_.begin(), j-2));
    ints[1] = create_inter(l[1], *std::next(scants_.begin(), j-1));
    inters_.insert(std::next(inters_.begin(), j-1), {ints[0], ints[1]});
    it0 = std::next(inters_.begin(), j+1);
    it1 = std::next(inters_.begin(), j+3);
    inters_.erase(it0, it1);
    create_rightend();
  }  else {
    // log.fatal() << "The rejection point is not in the range\n";
      std::cerr << "The rejection point is not in the range\n";
  }
  update_area();
}


/** draw region from discrete distribution with weight areas_ */
inline int AdaptiveRejectionSampler::sample_j() {
  /*used to sample j from area list*/
  return discrete_distribution_();
}

/** sampler x in range [xj, xj+1) */
inline double AdaptiveRejectionSampler::sample_x(int j) {
  double r = Random::canonical<double>();
  double m = upper_bounds_.at(j).piecewise_linear_line.m;
  double x;
  // m != 0, sample from piecewise exponential distribution
  if ( std::abs(m) > really_small ) {
    x = std::log(r*std::exp(m*upper_bounds_.at(j).right_point.x)+
                    (1.0f-r)*std::exp(m*upper_bounds_.at(j).left_point.x))/m;
  } else {
    // if it happens in really tiny opportunity that the slope m == 0
    // sample from unifrom distribution
    x = r*upper_bounds_.at(j).right_point.x +
        (1.0f-r)*upper_bounds_.at(j).left_point.x;
  }

  if ( std::isnan(x) ) {
    std::cerr << "m=" << m << "r=" << r << std::endl;
    std::cerr << "upper_bounds_.at(j).right_point.x = " << upper_bounds_.at(j).right_point.x << std::endl;
    std::cerr << "upper_bounds_.at(j).left_point.x  = " << upper_bounds_.at(j).left_point.x << std::endl;
  }

  return x;
}


/** if squeezing_test==true, do not need rejection_test (time save) */
inline bool AdaptiveRejectionSampler::squeezing_test(const double x,
                                            const int j, const double rand) {
  return rand <= std::exp(lower(j).eval(x) -
                          upper_bounds_.at(j).piecewise_linear_line.eval(x));
}

/** if squeezing_test==false, do rejection_test
 * if rejection_test=true, keep the sampling*/
inline bool AdaptiveRejectionSampler::rejection_test(const double x,
                                            const int j, const double rand) {
  assert(j < upper_bounds_.size());
  return rand*std::exp(upper_bounds_.at(j).piecewise_linear_line.eval(x))
      <= f_(x);
}


/** get one x from distribution function f_(x)*/
double AdaptiveRejectionSampler::get_one_sample() {
  // const auto & log = logger<LogArea::Sampling>();

  int rejections = 0;
  double x;
  while ( true ) {
    int j = sample_j();
    x = sample_x(j);
    double rand = Random::canonical<double>();
    if ( squeezing_test(x, j, rand) ) {
      return x;
    } else if ( rejection_test(x, j, rand) ) {
      return x;
    }else {
      if ( total_refine_loops_ < 1000 ) {
        Point rej;
        rej.x = x;
        rej.expy = f_(x);
        rej.y = std::log(rej.expy);
        try {
            adaptive_update((j+1)/2, rej);
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of range:" << oor.what() << std::endl;
            throw std::runtime_error("Error: Out of range error in ARS");
        }
        total_refine_loops_++;
      }
      rejections++;
      //std::cout << "don't do adaptive_update to test memory leak \n";
    }

    if ( rejections == max_refine_loops_ ) {
      // log.fatal() << "In AdaptiveRejectionSampler:";
      // log.fatal() << "reject too many time!\n";
      std::cout << "f(0.1)=" << f_(0.1) << std::endl;
      std::cout << "x, f(x), log(f)=" << x << " " << f_(x) << " " << std::log(f_(x)) << std::endl;
      throw std::runtime_error(
          "Error: Reject more than 40 times to get one sample "
          "is not resonable in ARS method!");
    }
  }
}


}  // end namespace Rejection
}  // end namespace Smash
