import Lake
open Lake DSL

package «leanData» {
  -- add any package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.7.0"


@[default_target]
lean_lib «Main» {
  -- add any library configuration options here
}

lean_lib «NuminaMath» {
  -- add any library configuration options here
}
