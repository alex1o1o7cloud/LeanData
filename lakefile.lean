import Lake
open Lake DSL

package leanData where
  -- add any package configuration options here
  require mathlib from git
    "https://github.com/leanprover-community/mathlib4"

@[default_target]
lean_lib Main 

lean_lib NuminaMath