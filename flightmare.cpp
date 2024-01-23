#include <iostream>
#include <iomanip>
#include <Eigen/Eigen> //3.3.4
#include <cmath>

// --- Setup some Eigen quick shorthands ---

// Define the scalar type used.
using Scalar = double; // numpy float64
static constexpr int Dynamic = Eigen::Dynamic;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template <int rows = Dynamic, int cols = Dynamic>
using Matrix = Eigen::Matrix<Scalar, rows, cols>; // lets you do Matrix<3,3> or Matrix<3,1> etc;

template <int rows = Dynamic>
using Vector = Matrix<rows, 1>; // lets you do Vector<3> or Vector<4> etc;

using Quaternion = Eigen::Quaternion<Scalar>; // type Quaternion instead of Eigen::Quaternion<Scalar>

// --- Constants ---
static constexpr Scalar Gz = -9.81;
static constexpr Scalar motor_tau = 0.02;
constexpr Scalar integrator_dt = 0.005;
constexpr Scalar motor_tau_inv = 1.0 / motor_tau;
constexpr Scalar sim_dt = 0.008;
constexpr Scalar max_t = 8; // unused atm, ep length
constexpr Scalar motor_constant = 0.000000135;
constexpr Scalar rotor_drag_coef = 0.000175;
constexpr Scalar mass = 1.6;
constexpr Scalar arm_l = 0.225;
const Vector<3> _gz{0.0, 0.0, Gz};
const Matrix<3, 3> J = mass / 12.0 * arm_l * arm_l * Vector<3>(4.5, 4.5, 7).asDiagonal(); // inertia matrix
const Matrix<3, 3> J_inv = J.inverse();
const Matrix<3, 4> t_BM_ = arm_l * sqrt(0.5) *
                           (Matrix<3, 4>() << 1, -1, -1, 1, -1, -1, 1, 1, 0, 0, 0, 0).finished(); // thrust to moment matrix
const Matrix<4, 4> B_allocation_ = (Matrix<4, 4>() << Vector<4>::Ones().transpose(), t_BM_.topRows<2>(),
                                    0.0115 * Vector<4>(1, -1, 1, -1).transpose())
                                       .finished(); // Used to generate thrust equations (see torque matrix eq. )
const Matrix<4, 4> B_allocation_inv_ = B_allocation_.inverse();
const Vector<7> rk7_sum_vec = {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
const Vector<4> rk4_sum_vec = {1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0};

/**
 * @brief State of the environment and quadcopter.
 */
class State
{
public:
    /**
     * @brief Indexes for the state vector
     */
    enum IDX : int
    {
        // position
        POS = 0,
        POSX = 0,
        POSY = 1,
        POSZ = 2,
        NPOS = 3,
        // quaternion
        ATT = 3,
        ATTW = 3,
        ATTX = 4,
        ATTY = 5,
        ATTZ = 6,
        NATT = 4,
        // linear velocity
        VEL = 7,
        VELX = 7,
        VELY = 8,
        VELZ = 9,
        NVEL = 3,
        // body rate
        OME = 10,
        OMEX = 10,
        OMEY = 11,
        OMEZ = 12,
        NOME = 3,
        // linear acceleration
        ACC = 13,
        ACCX = 13,
        ACCY = 14,
        ACCZ = 15,
        NACC = 3,
        // body-torque
        TAU = 16,
        TAUX = 16,
        TAUY = 17,
        TAUZ = 18,
        NTAU = 3,
        //
        BOME = 19,
        BOMEX = 19,
        BOMEY = 20,
        BOMEZ = 21,
        NBOME = 3,
        //
        BACC = 22,
        BACCX = 22,
        BACCY = 23,
        BACCZ = 24,
        NBACC = 3,
        //
        SIZE = 25,
        NDYM = 19
    };

    Vector<4> prev_motor_speed;
    Vector<4> motor_speed;
    Vector<3> goal_position;
    Vector<4> prev_action;

    Vector<IDX::SIZE> x;

    Vector<3> pos() const { return x.segment<IDX::NPOS>(IDX::POS); }
    Vector<4> att() const { return x.segment<IDX::NATT>(IDX::ATT); }
    Vector<3> vel() const { return x.segment<IDX::NVEL>(IDX::VEL); }
    Vector<3> ome() const { return x.segment<IDX::NOME>(IDX::OME); }
    Vector<3> acc() const { return x.segment<IDX::NACC>(IDX::ACC); }
    Vector<3> tau() const { return x.segment<IDX::NTAU>(IDX::TAU); }
    Vector<3> bome() const { return x.segment<IDX::NBOME>(IDX::BOME); }
    Vector<3> bacc() const { return x.segment<IDX::NBACC>(IDX::BACC); }
    void set_pos(const Vector<3> &pos) { x.segment<IDX::NPOS>(IDX::POS) = pos; }
    void set_att(const Vector<4> &att) { x.segment<IDX::NATT>(IDX::ATT) = att; }
    void set_vel(const Vector<3> &vel) { x.segment<IDX::NVEL>(IDX::VEL) = vel; }
    void set_ome(const Vector<3> &ome) { x.segment<IDX::NOME>(IDX::OME) = ome; }
    void set_acc(const Vector<3> &acc) { x.segment<IDX::NACC>(IDX::ACC) = acc; }
    void set_tau(const Vector<3> &tau) { x.segment<IDX::NTAU>(IDX::TAU) = tau; }
    void set_bome(const Vector<3> &bome) { x.segment<IDX::NBOME>(IDX::BOME) = bome; }
    void set_bacc(const Vector<3> &bacc) { x.segment<IDX::NBACC>(IDX::BACC) = bacc; }

    /**
     * @brief Returns a quaternion from the state attitude vector.
     *
     * @return Quaternion
     */
    Quaternion q() const { return Quaternion(att()[0], att()[1], att()[2], att()[3]); }

    /**
     * @brief Set all values in the current state to zero and resets quadcopter position.
     *
     */
    void setZero()
    {
        x.setZero();
        x(IDX::ATTW) = 1.0;
        x(IDX::POSZ) = 0;
        x(IDX::ACCZ) = -9.81;
        prev_motor_speed.setZero();
        motor_speed.setZero();
        goal_position = Vector<3>{Scalar(1), Scalar(1), Scalar(2)};
    }

    void setZero(bool doNotSetMotors)
    {
        x.setZero();
        x(IDX::ATTW) = 1.0;

        x(IDX::ATTW) = 1.0;
        x(IDX::POSZ) = 0;
    }

    /**
     * @brief Returns the current observation from the quadcopters state, ready for the policy.
     *
     * @return Vector<16> Observation vector
     */
    Vector<16> getObservation()
    {
        Vector<16> observation;
        Vector<3> distance_to_target = goal_position - pos();
        observation << pos(), q().toRotationMatrix().eulerAngles(2, 1, 0), vel(), ome(), prev_action;
        return observation;
    }
};

//--- Helper functions ---
/**
 * @brief Returns the right-hand side of the quaternion multiplication.
 *
 * @param q Quaternion to multiply
 * @return Matrix<4, 4> Right-hand side of quaternion multiplication
 */
Matrix<4, 4>
Q_right(const Quaternion &q)
{
    return (Matrix<4, 4>() << q.w(), -q.x(), -q.y(), -q.z(), q.x(), q.w(), q.z(),
            -q.y(), q.y(), -q.z(), q.w(), q.x(), q.z(), q.y(), -q.x(), q.w())
        .finished();
}

/**
 * @brief Clamps motor rpm speed between (min) and (max)
 *
 * @param omega Current motor_rpm variable
 * @param min  Minimum motor_rpm
 * @param max  Maximum motor_rpm
 * @return Vector<4> Updated motor_rpm variable
 */
Vector<4> clamp(const Vector<4> &omega, Scalar min, Scalar max)
{
    return omega.cwiseMax(min).cwiseMin(max);
}

/**
 * @brief Dynamics function to update quadcopter state.
 *
 * @param state Current state, with acceleration and torque applied.
 * @param next_state Blank object, to be filled with updated state.
 * @return State Updated state.
 */
State dynamics(State state, State next_state) // integrated dynamics
{
    next_state.set_pos(state.vel());
    next_state.set_att(0.5 * Q_right(Quaternion(0, state.ome()[0], state.ome()[1], state.ome()[2])) * state.att());
    next_state.set_vel(state.acc());
    next_state.set_ome((J_inv * (state.tau() - state.ome().cross(J * state.ome()))));
    return next_state;
}

/**
 * @brief Runge-Kutta 7/8 integrator
 *
 * @param initial initial starting state
 * @param dt Time step to integrate over
 * @param final State to be filled with updated state
 * @return State
 */
State rk7(State initial, Scalar dt, State final)
{
    Matrix<> k = Matrix<>::Zero(initial.x.rows(), 7);

    final = initial;

    // K_1
    State k1;
    k1.setZero();
    k1 = dynamics(final, k1);

    // K_2
    State k2;
    k2.setZero();
    final.x = initial.x + 0.125 * dt * k1.x;
    k2 = dynamics(final, k2);

    // K_3
    State k3;
    k3.setZero();
    final.x = initial.x + 0.5 * dt * k1.x;
    k3 = dynamics(final, k3);

    // K_4
    State k4;
    k4.setZero();
    final.x = initial.x + 0.5 * dt * k2.x;
    k4 = dynamics(final, k4);

    // K_5
    State k5;
    k5.setZero();
    final.x = initial.x + dt * k3.x;
    k5 = dynamics(final, k5);

    // K_6
    State k6;
    k6.setZero();
    final.x = initial.x + 0.875 * dt * k4.x;
    k6 = dynamics(final, k6);

    // K_7
    State k7;
    k7.setZero();
    final.x = initial.x + dt * k5.x;
    k7 = dynamics(final, k7);

    k.col(0) = k1.x;
    k.col(1) = k2.x;
    k.col(2) = k3.x;
    k.col(3) = k4.x;
    k.col(4) = k5.x;
    k.col(5) = k6.x;
    k.col(6) = k7.x;

    final.x = initial.x + dt * k * rk7_sum_vec;

    return final;
}

/**
 * @brief Runge-Kutta 4 integrator
 *
 * @param initial initial starting state
 * @param dt Time step to integrate over
 * @param final State to be filled with updated state
 * @return State
 */
State rk4(State inital, Scalar dt, State final)
{
    Matrix<> k = Matrix<>::Zero(inital.x.rows(), 4);

    final = inital;

    // K_1
    State k1;
    k1.setZero();
    k1 = dynamics(final, k1);

    // K_2
    State k2;
    k2.setZero();
    final.x = inital.x + 0.5 * dt * k1.x;
    k2 = dynamics(final, k2);

    // k_3
    State k3;
    k3.setZero();
    final.x = inital.x + 0.5 * dt * k2.x;
    k3 = dynamics(final, k3);

    // k_4
    State k4;
    k4.setZero();
    final.x = inital.x + dt * k3.x;
    k4 = dynamics(final, k4);

    k.col(0) = k1.x;
    k.col(1) = k2.x;
    k.col(2) = k3.x;
    k.col(3) = k4.x;

    final.x = inital.x + dt * k * rk4_sum_vec;

    return final;
}

//-- DORMAND PRINCE 7(8) INTEGRATOR. REMOVED FOR NOW. --
// constexpr Scalar safety_factor = 0.9;    // Safety factor for step size control
// constexpr Scalar min_step_size = 1e-6;   // Minimum allowable step size
// constexpr Scalar max_step_size = 1.0;    // Maximum allowable step size
// constexpr Scalar error_threshold = 1e-2; // Maximum allowable step size
// Scalar calculate_error(const State &y1, const State &y2)
// {
//     // Implement the error calculation based on your problem.
//     // For example, you can use the Euclidean norm of the difference.
//     return (y1.x - y2.x).norm();
// }

// State integrate_dormand_prince(State initial, Scalar dt)
// {
//     State y = initial;

//     while (true)
//     {
//         // Compute RK8 step
//         State y_temp = rk4(y, dt, y);

//         // Compute two smaller RK4 steps
//         State y_half_dt = rk4(y, 0.5 * dt, y);
//         y_half_dt = rk4(y_half_dt, 0.5 * dt, y_half_dt);

//         // Calculate the error
//         Scalar error = calculate_error(y_temp, y_half_dt);

//         // Adjust the step size
//         Scalar scale_factor = std::pow((1.0 / error), 1.0 / 8.0);
//         Scalar new_dt = safety_factor * dt * scale_factor;

//         // Check if the error is within the tolerance
//         if (error < 1.0)
//         {
//             // Accept the step and update the state
//             y = y_temp;

//             // Update the time step for the next iteration
//             dt = std::min(std::max(new_dt, min_step_size), max_step_size);

//             // Output the state or perform other actions as needed

//             // Check if the integration is complete based on your criteria
//             if (error < error_threshold)
//             {
//                 break;
//             }
//         }
//         else
//         {
//             // Retry the step with a smaller step size
//             dt = std::min(new_dt, dt * 0.9f);
//         }
//     }

//     return y;
// }
// State euler(State initial, Scalar dt)
// {
//     State y = initial;

//     // Perform one Euler step
//     State k1;
//     k1.setZero();
//     k1 = dynamics(y, k1);

//     y.x += dt * k1.x;

//     // You can add state validation or additional processing here if needed

//     return y;
// }

Scalar step_motors(Scalar rpm, Scalar desired_rpm, Scalar dt)
{
        float c = std::exp(-dt * motor_tau_inv);
        rpm = c * rpm + (1 - c) * desired_rpm;
        return rpm;
}

Scalar step_motors(Scalar rpm, Scalar desired_rpm, Scalar dt)
{
    float a = 2.0 / std::pow(dt, 2);
    float b = (desired_rpm - rpm) / dt - a * dt;
    rpm = a * std::pow(dt, 2) + b * dt + rpm;
    return rpm;
}


/**
 * @brief Step function to update the environment and quadcopter state.
 *
 * @param state Current state object
 * @param init_omega  Initial motor rpm
 * @param desired_thurst_pc Desired thrust percentage
 * @param dt How long this step will last
 * @return State Updated state object
 */
State step(State state, Vector<4> &init_omega, Vector<4> desired_thurst_pc, Scalar sim_dt)
{

    State old_state = state;
    State next_state = state;
    // Scalar true_hover_thrust = 0.6074674579665363;
    // Scalar remaining_thrust = 1.0 - true_hover_thrust;
    // desired_thurst_pc[0] = desired_thurst_pc[0] <= 0.0f ? true_hover_thrust * desired_thurst_pc[0] : remaining_thrust * desired_thurst_pc[0];
    // desired_thurst_pc[1] = desired_thurst_pc[1] <= 0.0f ? true_hover_thrust * desired_thurst_pc[1] : remaining_thrust * desired_thurst_pc[1];
    // desired_thurst_pc[2] = desired_thurst_pc[2] <= 0.0f ? true_hover_thrust * desired_thurst_pc[2] : remaining_thrust * desired_thurst_pc[2];
    // desired_thurst_pc[3] = desired_thurst_pc[3] <= 0.0f ? true_hover_thrust * desired_thurst_pc[3] : remaining_thrust * desired_thurst_pc[3];
    // desired_thurst_pc = desired_thurst_pc + Vector<4>(true_hover_thrust, true_hover_thrust, true_hover_thrust, true_hover_thrust);

    Vector<4> desired_rpm = (desired_thurst_pc * 10000) + Vector<4>(100, 100, 100, 100);
    // std::cout << "desired_rpm " << desired_rpm.transpose() << std::endl;

    Scalar max_dt = 2.5e-3;
    Scalar remain_ctrl_dt = sim_dt;
    Vector<4> motor_omega = init_omega;
    int i = 0;

    while (remain_ctrl_dt > 0)
    {
        Vector<4> recieved_rpm = state.prev_motor_speed;
        // delayed response to input command
        if (i >= 1)
        {
            recieved_rpm = desired_rpm;
        }
        i++;

        // time step
        Scalar dt = std::min(remain_ctrl_dt, max_dt);
        remain_ctrl_dt -= dt;

        // clamp motor rpm
        Vector<4> clamped = clamp(recieved_rpm, 100.0f, 10100.0f);

        // rotor ramp up time
        float c = std::exp(-dt * motor_tau_inv);
        motor_omega = c * motor_omega + (1 - c) * clamped;

        // armed quadcopter rpm is 100. will always be above 100rpm. max is
        motor_omega = clamp(motor_omega, 100.0f, 10100.0f);

        // clamp motor thrust
        Vector<4> motor_thrusts = motor_omega.cwiseProduct(motor_omega) * motor_constant;
        motor_thrusts = motor_thrusts.cwiseMax(0.0584f);
        motor_thrusts = motor_thrusts.cwiseMin(9.0f);

        // forces & moments
        Vector<4> force_torques = B_allocation_ * motor_thrusts;

        // thrust
        Vector<3> force(0, 0, force_torques[0]);

        // acceleration
        state.set_acc(state.q() * force * 1.0 / mass + _gz);

        // body torque
        Vector<3> torque = force_torques.tail(3);
        state.set_tau(torque);

        // integrate dynamics function
        next_state = rk4(state, dt, next_state);

        state = next_state;

        // TODO, separate quadcopter and state into their own objects. The environment doesn't contain motor speed, quadcopter does.
        state.motor_speed = motor_omega;
    }
    state.prev_motor_speed = desired_rpm;
    // std::cout << "state x: " << state.x << std::endl;
    state.prev_action = desired_thurst_pc;

    // Do not let quadcopter fall through the floor
    if (state.pos()[2] < 0)
    {

        Scalar lvz = state.vel()[2];
        Scalar laz = state.acc()[2];
        state.setZero(true);

        if (lvz > 0)
        {
            state.set_vel({0, 0, lvz});
        }
        if (laz > 0)
        {
            state.set_acc({0, 0, laz});
        }
    }

    return state;
}

//-- TODO, reward function. --
int main()
{
    State quad;
    quad.setZero();

    // Vector<4> desired_thrust(0.407654, 0.912659, 0.242549, 0.245255);
    Vector<4> init_omega = {0.0, 0.0, 0.0, 0.0};
    NeuralNetwork nn;
    // quad.prev_motor_speed = {0.413210, 0.909381, 0.239145, 0.247557};
    // Vector<3> pos{-0.334669, -0.184764, 2.445248};
    // Vector<3> lv{0.000413, -0.022922, 0.059665};
    // Vector<4> att{0.999775, 0.006836, -0.002675, -0.019910};

    // quad.set_att(att);
    // quad.set_vel(lv);
    // quad.set_pos(pos - quad.goal_position);

    for (int i = 0; i < 1000 - 1; i++)
    {
        // Vector<4> desired_thrust = {control1[i], control3[i], control0[i], control2[i]};
        Vector<4> desired_thrust = {0.7, 0.7, 0.7, 0.7};

        // Scalar timestep_length = timestamp[i + 1] - timestamp[i];
        // std::cout << "timestep_length " << timestep_length << std::endl;

        // std::cout << "Desired thrust: " << desired_thrust.transpose() << std::endl;
        quad = step(quad, init_omega, desired_thrust, 0.008);
        std::cout << "Quad state: " << quad.getObservation().transpose() << std::endl;
        init_omega = quad.motor_speed;
    }
    return 0;
}
