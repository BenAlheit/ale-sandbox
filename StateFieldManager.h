//
// Created by alhei on 2022/08/11.
//

#ifndef ALE_STATEFIELDMANAGER_H
#define ALE_STATEFIELDMANAGER_H

#include "MaterialStates.h"
#include <unordered_map>
#include <array>
#include <boost/functional/hash.hpp>

using namespace std;

struct array_hash {
    template <class T1, long unsigned int n_elements>
    size_t operator () (const array<T1, n_elements> &p) const {
        return boost::hash_range(p.begin(), p.end());
    }
};

template <unsigned int dim, class State>
class StateFieldManager{
public:
    StateFieldManager() {};
    StateFieldManager(const DoFHandler<dim> & dof_handler,
                      FEValues<dim> & fe_values,
                      const State* new_state);
    State* get_state(unsigned int level, unsigned int el_id, unsigned int qp) const;
    void add_state(unsigned int level, unsigned int el_id, unsigned int qp, State* state);
    unordered_map<array<unsigned int, 3>, State*, array_hash> material_states;

//private:
};

template<unsigned int dim, class State>
State* StateFieldManager<dim, State>::get_state(unsigned int level, unsigned int el_id, unsigned int qp) const {
    return material_states.at({level, el_id, qp});
}

template<unsigned int dim, class State>
void StateFieldManager<dim, State>::add_state(unsigned int level, unsigned int el_id, unsigned int qp, State * new_state) {
    material_states[{level, el_id, qp}] = new_state;
}

template<unsigned int dim, class State>
StateFieldManager<dim, State>::StateFieldManager(const DoFHandler<dim> & dof_handler,
                                          FEValues<dim> & fe_values,
                                          const State *state) {
    material_states.clear();
    array<unsigned int, 3> input{};
    pair<array<unsigned int, 3>, State*> p;
//    unsigned int level;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        input[0] = cell->level();
        input[1] = cell->index();
        for (const unsigned int q_point:
                fe_values.quadrature_point_indices()){
            input[2] = q_point;
            p.first=input;
            p.second=state->copy();
            material_states.insert(p);
        }
    }
}

#endif //ALE_STATEFIELDMANAGER_H
