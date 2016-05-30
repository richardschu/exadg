/*
 * FE_Parameters.h
 *
 *  Created on: May 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FE_PARAMETERS_H_
#define INCLUDE_FE_PARAMETERS_H_

#include "InputParameters.h"

template<typename value_type>
class FEParameters
{
public:
  FEParameters(InputParameters const & param)
    :
    viscosity(param.viscosity),
    cs(param.cs),
    ml(param.ml),
    variabletauw(param.variabletauw),
    dtauw(param.dtauw)
  {
    xwallstatevec.resize(2);
  }

  double const viscosity;
  double const cs;
  double const ml;
  bool const variabletauw;
  double const dtauw;
  std::vector<parallel::distributed::Vector<value_type> > xwallstatevec;
};

#endif /* INCLUDE_FE_PARAMETERS_H_ */