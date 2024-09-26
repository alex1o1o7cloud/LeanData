import Lake
open Lake DSL

package «LeanData» where
  -- add any package-wide dependencies here

@[default_target]
lean_lib «NuminaMath» where
  -- Point to the NuminaMath directory
  srcDir := "NuminaMath"
  -- Explicitly state that we don't have a root file
  roots := #[]

lean_exe «LeanData» where
  root := `Main
  -- Link the NuminaMath library
  dependencies := #[`NuminaMath]