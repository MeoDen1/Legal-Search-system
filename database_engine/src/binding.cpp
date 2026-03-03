#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for std::vector <-> Python list
#include "database.h"
#include "searcher.h"
#include "ingestion_manager.h"

namespace py = pybind11;

py::object json_to_py(const nlohmann::json& j) {
    if (j.is_null())    return py::none();
    if (j.is_boolean())    return py::bool_(j.get<bool>());
    if (j.is_number_integer()) return py::int_(j.get<int64_t>());
    if (j.is_number_float())   return py::float_(j.get<double>());
    if (j.is_string())  return py::str(j.get<std::string>());

    if (j.is_array()) {
        py::list list;
        for (const auto& item : j) {
            list.append(json_to_py(item));
        }
        return list;
    }

    if (j.is_object()) {
        py::dict dict;
        for (auto& [key, value] : j.items()) {
            dict[py::str(key)] = json_to_py(value);
        }
        return dict;
    }

    return py::none();
}

PYBIND11_MODULE(database_engine, m) {
    m.doc() = "High-performance Legal AI Search Database";

    py::class_<Database>(m, "Database")
        .def(py::init<const std::string&>(), py::arg("db_path"));

    py::class_<Searcher>(m, "Searcher")
        .def(py::init([](Database& db) {
            return new Searcher(db);
        }), py::arg("db"), py::keep_alive<1, 2>())
        .def("search", [](Searcher &s, const std::vector<float>& embedding, const std::string& query) {
            // We wrap the search call to ensure it returns a Python-friendly dict
            // nlohmann::json can be converted to a string, then parsed in Python,
            nlohmann::json result = s.search(embedding, query);
            return json_to_py(result);
        }, py::arg("embedding"), py::arg("query"));

    py::class_<IngestionManager>(m, "IngestionManager")
        .def(py::init<const std::string&, const std::string&>(), py::arg("data_path"), py::arg("db_path"))
        .def("ingest", &IngestionManager::ingest);
}
