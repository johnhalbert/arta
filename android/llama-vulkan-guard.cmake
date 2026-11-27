# The project hook runs before llama.cpp creates its library target.
function(arta_configure_vulkan guard_source)
    target_sources(llama PRIVATE "${guard_source}")

    # Derive a build-local source from the pinned dependency without editing Cargo's source copy.
    get_target_property(vulkan_dir ggml-vulkan SOURCE_DIR)
    set(vulkan_source "${vulkan_dir}/ggml-vulkan.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${vulkan_source}")
    file(READ "${vulkan_source}" source)

    set(matrix_dispatch [=[                ggml_type src0_type = op->src[0]->type;
                if (op->op == GGML_OP_MUL_MAT_ID) {]=])
    set(matrix_dispatch_compat [=[                ggml_type src0_type = op->src[0]->type;
                // The proprietary Adreno 830 driver rejects Q4_K matvec pipelines.
                // Reject the matrix ops during placement so ggml can schedule them on CPU.
                // https://github.com/ggml-org/llama.cpp/issues/28635
                if (src0_type == GGML_TYPE_Q4_K &&
                    device->vendor_id == VK_VENDOR_ID_QUALCOMM &&
                    device->driver_id == vk::DriverId::eQualcommProprietary &&
                    device->properties.deviceID == 0x44050001) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID) {]=])
    string(REPLACE "${matrix_dispatch}" "${matrix_dispatch_compat}" patched "${source}")
    if(patched STREQUAL source)
        message(FATAL_ERROR "The pinned Vulkan matrix dispatch moved. Review the Adreno 830 workaround.")
    endif()

    set(pipeline_error [=[        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::endl;
        throw;]=])
    set(pipeline_error_named [=[        throw std::runtime_error("Vulkan compute pipeline " + pipeline->name + " failed (" + e.what() + ")");]=])
    string(REPLACE "${pipeline_error}" "${pipeline_error_named}" generated "${patched}")
    if(generated STREQUAL patched)
        message(FATAL_ERROR "The pinned Vulkan pipeline error handler moved. Review the Android diagnostic.")
    endif()

    set(generated_source "${CMAKE_CURRENT_BINARY_DIR}/arta-ggml-vulkan.cpp")
    file(CONFIGURE OUTPUT "${generated_source}" CONTENT "${generated}" @ONLY)
    get_target_property(vulkan_sources ggml-vulkan SOURCES)
    list(REMOVE_ITEM vulkan_sources ggml-vulkan.cpp "${vulkan_source}")
    set_property(TARGET ggml-vulkan PROPERTY SOURCES "${vulkan_sources};${generated_source}")
    target_include_directories(ggml-vulkan PRIVATE "${vulkan_dir}")
endfunction()

cmake_language(EVAL CODE
    "cmake_language(DEFER CALL arta_configure_vulkan [[${CMAKE_CURRENT_LIST_DIR}/native/llama-vulkan-guard.cpp]])"
)
