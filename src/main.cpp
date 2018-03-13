#include <ros/ros.h>
#include <Particle_Filter.h>

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "particle_filter");

  Particle_Filter pf;

  ros::spin();

  return 0;
}
