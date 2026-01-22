
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

#include <concurrentqueue.h>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>

namespace duckdb {


unique_ptr<IndexBuildBindData> HNSWIndexBuildBind(IndexBuildBindInput &input) {
	return nullptr;
}

//-------------------------------------------------------------
// Global State
//-------------------------------------------------------------

class HNSWIndexBuildState : public IndexBuildState {
public:
	HNSWIndexBuildState() {}

public:
	//! Global index to be added to the table
	unique_ptr<HNSWIndex> global_index;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	// Parallel scan state
	ColumnDataParallelScanState scan_state;

	// Track which phase we're in
	atomic<bool> is_building = {false};
	atomic<idx_t> loaded_count = {0};
	atomic<idx_t> built_count = {0};

	// estimated cardinality (for progress)
	idx_t estimated_cardinality;
};

unique_ptr<IndexBuildState> HNSWInitBuildState(IndexBuildInitStateInput &input) {
	auto global_state = make_uniq<HNSWIndexBuildState>();

	vector<LogicalType> data_types = {input.expressions[0]->return_type, LogicalType::ROW_TYPE};
	global_state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(input.context), data_types);
	global_state->context = input.context.shared_from_this();
	global_state->estimated_cardinality = input.estimated_cardinality;

	// Create the index
	auto &storage = input.table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = input.info.constraint_type;
	auto &db = storage.db;

	// TODO; we should also be able to do this somehow with the create_instance callback
	global_state->global_index =
		make_uniq<HNSWIndex>(input.info.index_name, constraint_type, input.storage_ids, table_manager, input.expressions, db,
							 input.info.options, IndexStorageInfo(), input.estimated_cardinality);

	return std::move(global_state);
}

class HNSWIndexBuildSinkState final : public IndexBuildSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<IndexBuildSinkState> HNSWInitSinkState(IndexBuildInitSinkInput &input) {
	auto state = make_uniq<HNSWIndexBuildSinkState>();
	// TODO: Get the type of the sink chunk somehow
	vector<LogicalType> data_types = {input.data_types[0], LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(input.context), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}


void HNSWIndexBuildSink(IndexBuildSinkInput &state, DataChunk &key_chunk, DataChunk &row_chunk) {
	// Create a Datachunk
	DataChunk chunk;
	chunk.InitializeEmpty({key_chunk.data[0].GetType(), row_chunk.data[0].GetType()});
	chunk.data[0].Reference(key_chunk.data[0]);
	chunk.data[1].Reference(row_chunk.data[0]);
	chunk.SetCardinality(key_chunk.size());

	auto &lstate = state.local_state->Cast<HNSWIndexBuildSinkState>();
	auto &gstate = state.global_state->Cast<HNSWIndexBuildState>();

	lstate.collection->Append(lstate.append_state, chunk);
	gstate.loaded_count += chunk.size();
}

//-------------------------------------------------------------
// Combine
//-------------------------------------------------------------
// void (*index_build_sink_combine_t)(IndexBuildSinkCombineInput &input);
void HNSWIndexBuildSinkCombine(IndexBuildSinkCombineInput &input) {
	auto &lstate = input.local_state->Cast<HNSWIndexBuildSinkState>();
	auto &gstate = input.global_state->Cast<HNSWIndexBuildState>();

	if (lstate.collection->Count() == 0) {
		return;
	}

	// lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}
}

// Midpoint! We have sunk everything into the columndata collection
// now prepare to build the index in parallel
void HNSWIndexBuildPrepare(IndexBuildPrepareInput &input) {
	auto &gstate = input.global_state->Cast<HNSWIndexBuildState>();

	// Update our status
	gstate.is_building = true;

	// Reserve index size
	auto &ts = TaskScheduler::GetScheduler(input.context);
	auto &index = gstate.global_index->index;
	index.reserve({static_cast<size_t>(gstate.collection->Count()), static_cast<size_t>(ts.NumberOfThreads())});

	// Initialize a parallel scan for the index construction
	gstate.collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);
}

class HNSWIndexBuildWorkState final : public IndexBuildWorkState {
public:
	DataChunk scan_chunk;
	ColumnDataLocalScanState local_scan_state;
};

// former Constructor of IndexConstructTask
unique_ptr<IndexBuildWorkState> HNSWInitWorkState(IndexBuildInitWorkInput &input) {
	auto lstate = make_uniq<HNSWIndexBuildWorkState>();
	auto &gstate = input.global_state->Cast<HNSWIndexBuildState>();
	gstate.collection->InitializeScanChunk(lstate->scan_chunk);
	return std::move(lstate);
}

// Former ExecuteTask()
bool HNSWIndexBuildWork(IndexBuildWorkInput &input) {
	auto &gstate = input.global_state->Cast<HNSWIndexBuildState>();
	auto &lstate = input.local_state->Cast<HNSWIndexBuildWorkState>();
	// TODO: Figure out what makes sense to return here
	auto &index = gstate.global_index->index;
	auto &scan_state = gstate.scan_state;
	auto &collection = gstate.collection;

	const auto array_size = ArrayType::GetSize(lstate.scan_chunk.data[0].GetType());
	const auto keep_going = collection->Scan(scan_state, lstate.local_scan_state, lstate.scan_chunk);

	if (!keep_going) {
		return false;
	}

	const auto count = lstate.scan_chunk.size();
	auto &vec_vec = lstate.scan_chunk.data[0];
	auto &data_vec = ArrayVector::GetEntry(vec_vec);
	auto &rowid_vec = lstate.scan_chunk.data[1];

	UnifiedVectorFormat vec_format;
	UnifiedVectorFormat data_format;
	UnifiedVectorFormat rowid_format;

	vec_vec.ToUnifiedFormat(count, vec_format);
	data_vec.ToUnifiedFormat(count * array_size, data_format);
	rowid_vec.ToUnifiedFormat(count, rowid_format);

	const auto row_ptr = UnifiedVectorFormat::GetData<row_t>(rowid_format);
	const auto data_ptr = UnifiedVectorFormat::GetData<float>(data_format);

	for (idx_t i = 0; i < count; i++) {
		const auto vec_idx = vec_format.sel->get_index(i);
		const auto row_idx = rowid_format.sel->get_index(i);

		// Check for NULL values
		const auto vec_valid = vec_format.validity.RowIsValid(vec_idx);
		const auto rowid_valid = rowid_format.validity.RowIsValid(row_idx);

		if (!vec_valid || !rowid_valid) {
			throw InvalidInputException("Invalid data in HNSW index construction: Cannot construct index with NULL values.");
		}

		// Add the vector to the index
		const auto result = index.add(row_ptr[row_idx], data_ptr + (vec_idx * array_size), input.thread_id);

		// Check for errors
		if (!result) {
			throw InvalidInputException(result.error.what());
		}
	}

	// Update the built count
	gstate.built_count += count;

	// Keep going!
	return true;
}

unique_ptr<BoundIndex> HNSWFinalizeBuild(IndexBuildFinalizeInput &input) {
	auto &gstate = input.global_state.Cast<HNSWIndexBuildState>();

	gstate.global_index->SetDirty();
	gstate.global_index->SyncSize();

	return std::move(gstate.global_index);
}

//------------------------------------------------------------------------------
// Progress (for the progress bar)
//------------------------------------------------------------------------------

ProgressData HNSWIndexSinkProgress(IndexBuildProgressInput &input) {
	// The "source_progress" is not relevant for CREATE INDEX statements
	ProgressData res;

	const auto &state = input.global_state.Cast<HNSWIndexBuildState>();
	// First half of the progress is appending to the collection
	if (!state.is_building) {
		res.done = state.loaded_count + 0.0;
		res.total = state.estimated_cardinality + state.estimated_cardinality;
	} else {
		res.done = state.loaded_count + state.built_count;
		res.total = state.loaded_count + state.loaded_count;
	}
	return res;
}

//------------------------------------------------------------------------------
// Register Index Type
//------------------------------------------------------------------------------
void HNSWModule::RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;

	index_type.name = HNSWIndex::TYPE_NAME;

	// we can possible remove this as well
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<HNSWIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
										input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};

	// not necessary anymore
	// index_type.create_plan = HNSWIndex::CreatePlan;

	// Setup index creation callbacks!
	index_type.build_bind = HNSWIndexBuildBind;
	index_type.build_init = HNSWInitBuildState;
	index_type.build_sink_init = HNSWInitSinkState;
	index_type.build_sink = HNSWIndexBuildSink;
	index_type.build_sink_combine = HNSWIndexBuildSinkCombine;
	index_type.build_prepare = HNSWIndexBuildPrepare;
	index_type.build_work_init = HNSWInitWorkState;
	index_type.build_work = HNSWIndexBuildWork;
	// No build work combine?
	index_type.build_finalize = HNSWFinalizeBuild;
	index_type.build_sink_progress = HNSWIndexSinkProgress;

	// Register persistence option
	db.config.AddExtensionOption("hnsw_enable_experimental_persistence",
								 "experimental: enable creating HNSW indexes in persistent databases",
								 LogicalType::BOOLEAN, Value::BOOLEAN(false));

	// Register scan option
	db.config.AddExtensionOption("hnsw_ef_search",
								 "experimental: override the ef_search parameter when scanning HNSW indexes",
								 LogicalType::BIGINT);

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}

} // namespace duckdb