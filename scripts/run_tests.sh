# Features are mutually exclusive with no features,
# so we should run the test twice

# No features activated
cargo test
status_test_with_no_features=$?

# Features activated
if [[ "$status_test_with_no_features" -eq 0 ]];
    then cargo test --all-features
fi