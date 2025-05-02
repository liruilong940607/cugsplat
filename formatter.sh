# install via: sudo apt-get install clang-format
find cugsplat/ tests/ \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i {} \;

# install via: pip install black==22.3.0
# black . cugsplat/ tests/ examples/ profiling/