#include "controllers/control_blocks.h"

void summation2(const double *value1, const double *value2, double *result)
{
    *result = *value1 + *value2;
}

void summation3(const double *value1, const double *value2,
                const double *value3, double *result)
{
    *result = *value1 + *value2 + *value3;
}

void subtraction(const double *start_value, const double *reduction_value, double *difference)
{
    *difference = *start_value - *reduction_value;
}

void multiply2(const double *value1, const double *value2, double *result)
{
    *result = *value1 * *value2;
}

void multiply3(const double *value1, const double *value2, const double *value3,
               double *result)
{
    *result = *value1 * *value2 * *value3;
}

void division(const double *dividend, const double *divisor, double *quotient)
{
    *quotient = *dividend / *divisor;
}

void integrator(const double *input, const double *dt, double *output)
{
    *output += *input * *dt;
}

void differentiator(const double *current_value, const double *previous_value, const double *dt, double *output)
{
    *output = (*current_value - *previous_value) / *dt;
}

void saturation(double *input, const double *lower_limit, const double *upper_limit)
{
    double output;
    if (*input < *lower_limit)
    {
        *input = *lower_limit;
    }
    else if (*input > *upper_limit)
    {
        *input = *upper_limit;
    }
}

void integrate_when_in_desired_interval(const double *input,
                                        const double *lower_limit,
                                        const double *upper_limit,
                                        const double *dt, double *output)
{
    if (*input > *lower_limit && *input < *upper_limit)
    {
        integrator(input, dt, output);
    }
    else
    {
        *output = 0;
    }
}

void map_control_command(const double *control_command,
                         const double *control_command_lower_limit,
                         const double *control_command_upper_limit,
                         const double *control_command_mapped_lower_limit,
                         const double *control_command_mapped_upper_limit,
                         double *mapped_control_command)
{
    // normalize
    *mapped_control_command =
        (*control_command - *control_command_lower_limit) /
        (*control_command_upper_limit - *control_command_lower_limit);

    // scale
    *mapped_control_command =
        *mapped_control_command * (*control_command_mapped_upper_limit -
                                   *control_command_mapped_lower_limit);
    // translate
    *mapped_control_command += *control_command_mapped_lower_limit;
}

void get_sign(const double *value, double *sign)
{
    if (*value > 0)
    {
        *sign = 1;
    }
    else if (*value < 0)
    {
        *sign = -1;
    }
    else
    {
        *sign = 0;
    }
}

void hside(const double *value, double *hside_output)
{
    if (*value >= 0)
    {
        *hside_output = 1.0;
    }
    else
    {
        *hside_output = 0.0;
    }
}

void get_abs(const double *value, double *abs_value)
{
    if (*value < 0)
    {
        *abs_value = -(*value);
    }
    else
    {
        *abs_value = *value;
    }
}

void set_value_of_first_to_second_variable(const double *first, double *second)
{
    *second = *first;
}