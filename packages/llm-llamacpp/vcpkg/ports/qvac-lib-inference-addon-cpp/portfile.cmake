# Dev overlay: build qvac-lib-inference-addon-cpp from the addon-cpp snapshot
# branch (jesusmb1995/qvac @ continuousBatchingD40AddonCpp1) instead of the
# published registry version, so llm-llamacpp picks up unpublished changes —
# the multi-job scheduler in 1.3.0 plus the IModelJobLifecycle::jobStarting
# dequeue announcement. Bump REF/SHA512 (and the overlay port-version) when
# that branch moves. To fall back to the registry version, remove this port
# directory and the "overlay-ports" entry in vcpkg-configuration.json.
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jesusmb1995/qvac
  REF d28056d687b7fb5229ccbc67224172b72ea8141d
  SHA512 a2efd29b1632d5ecc9b4a927c0eae148385ac3f3a4a8e2260787ac7c12cec2acb7864f1559e71e434c83c423d8c309972b757a908102dff48a9f6709a122d169
  HEAD_REF continuousBatchingD40AddonCpp1
)

vcpkg_check_features(
  OUT_FEATURE_OPTIONS FEATURE_OPTIONS
  FEATURES
    tests BUILD_TESTING
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}/packages/inference-addon-cpp"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

file(
  INSTALL "${SOURCE_PATH}/packages/inference-addon-cpp/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright
)
