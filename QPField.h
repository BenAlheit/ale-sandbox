//
// Created by alhei on 2022/08/11.
//

#ifndef ALE_QPFIELD_H
#define ALE_QPFIELD_H

#include <array>
#include "StateFieldManager.h"

template <unsigned int dim, class type>
class QPField{
public:
    QPField() {};
    type get_field_value(unsigned int level, unsigned int el_id, unsigned int qp) const {return field.at({level, el_id, qp});};
    void add_field_value(unsigned int level, unsigned int el_id, unsigned int qp, type value);

private:
    unordered_map<array<unsigned int, 3>, type, array_hash> field;
};

template<unsigned int dim, class type>
void QPField<dim, type>::add_field_value(unsigned int level, unsigned int el_id, unsigned int qp, type value) {
    array<unsigned int, 3> input{level, el_id, qp};
    pair<array<unsigned int, 3>, type> p;

    p.first=input;
    p.second=value;
    field.insert(p);
}


#endif //ALE_QPFIELD_H
