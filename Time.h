#ifndef FINITE_DEFORMATION_SOLVER_TIME_H
#define FINITE_DEFORMATION_SOLVER_TIME_H

#include <deal.II/base/function_time.h>
#include <exception>

class IncrementExcededException : public exception {
    const char *what() const throw() override {
        return "The maximum number of time increments for this stage has been exceded.";
    }
} increment_exceded;

class PredictIncrementExcededException : public exception {
public:
    PredictIncrementExcededException(unsigned int max_incs, unsigned int predicted_incs) :
            max_incs(max_incs), predicted_incs(predicted_incs) {
        message = "The number of increments to complete the stage with the current step size is " +
                  to_string(predicted_incs) + ". However, maximum number of increments allowed is set to" +
                  to_string(max_incs) + ".";
    };

    const char *what() const throw() override {
        return message.c_str();
    }

private:
    unsigned int max_incs;
    unsigned int predicted_incs;
    string message;
};


class Time : public FunctionTime<double> {
public:
    Time() {}

//    Time(const json &time_config, const double & start_time, const unsigned int & stage)
//            : time_end(time_config["end time"]), stage(stage), time(start_time) {
//        if(time_config.contains("time increment")) delta_t = (double) time_config["time increment"];
//        else if (time_config.contains("n steps")) delta_t = (time_end - time) / ((double) time_config["n steps"]);
//        else throw invalid_argument("Must either specify 'time increment' or 'n steps' but neither was.");
//
//        delta_t_original = delta_t;
//
//        if(time_config.contains("output increment")) output_delta_t = (double) time_config["output increment"];
//        else if (time_config.contains("n outputs")) output_delta_t = (time_end - time) / ((double) time_config["n outputs"]);
//        else output_delta_t = delta_t;
//
//        if(time_config.contains("max increments")){
//            set_max_incs = true;
//            max_incs = (unsigned int) time_config["max increments"];
//            if(time_config.contains("use predictive max increments")) predictive_max_incs =
//                    (bool) time_config["use predictive max increments"];
//        }
//
//        if(time_config.contains("update with cut")) update_with_cut = (bool) time_config["update with cut"];
//
//        next_output = time + output_delta_t;
//
//    };

    Time(const double time_end,
         unsigned int n_steps,
         unsigned int n_steps_out = 0,
         const double time_start = 0)
            : timestep(0),
              time_end(time_end),
              delta_t((time_end - time_start) / n_steps),
              output_delta_t((n_steps_out == 0) ?
              (time_end - time_start) / n_steps
              : (time_end - time_start) / n_steps_out) {}

    Time(const double time_end, const double delta_t)
            : timestep(0), time_end(time_end), delta_t(delta_t), time(0.0) {}

    virtual ~Time() = default;

    double current() const {
        return time;
    }

    double end() const {
        return time_end;
    }

    double get_delta_t() const {
        return delta_t;
    }

    void set_delta_t(const double &new_delta_t) {
        delta_t = new_delta_t;
    }

    unsigned int get_timestep() const {
        return timestep;
    }

    unsigned int get_stage() const {
        return stage;
    }

    bool increment() {
        if (cut) {
            if (update_with_cut) delta_t = delta_t / pow(2, n_cuts);
            n_cuts = 0;
            cut = false;
        }
        bool output = false;
        double next_time = this->time + delta_t;
        current_delta_t = delta_t;
        if (next_time > time_end) {
            current_delta_t = time_end - this->time;
            this->time = time_end;
            output = true;
        } else this->time = next_time;

        if (this->time >= next_output) {
            output = true;
            current_delta_t = next_output - this->time;
            this->time = next_output;
            next_output += output_delta_t;
            output_step++;
        }

        timestep++;

        if (set_max_incs && timestep > max_incs) throw increment_exceded;
        if (predictive_max_incs) {
            unsigned int tot_incs = timestep + (int) (time_end - time) / delta_t;
            if (tot_incs > max_incs) throw PredictIncrementExcededException(max_incs, tot_incs);
        }

        return output;
    }

    void next_stage(const double end_time, const double dt) {
        stage++;
        this->time_end = end_time;
        this->delta_t = dt;
    }

    void set_time_end(const double &new_time_end) {
        this->time_end = new_time_end;
    }

    void cut_step() {
        cut = true;
        n_cuts++;
        time -= delta_t / pow(2, n_cuts);
    }

    void increase_dt() {
        delta_t = delta_t * 1.5;
    }

    double stage_pct() {
        return (time - time_start) / (time_end - time_start);
    }

    double delta_stage_pct() {
        return delta_t / (time_end - time_start);
    }

private:
    unsigned int timestep = -1;
    unsigned int stage = 0;
    unsigned int n_cuts = 0;
    bool cut = false;
    bool update_with_cut = true;
    double time_start = 0;
    double time_end = 0;
    double delta_t_original = 0;
    double delta_t = 0;
    double current_delta_t = 0;
    unsigned int output_step = 0;
    double output_delta_t = 0;
    double next_output = 0;
    double time = 0;
    unsigned int max_incs = 0;
    bool set_max_incs = false;
    bool predictive_max_incs = false;
};


#endif //FINITE_DEFORMATION_SOLVER_TIME_H
