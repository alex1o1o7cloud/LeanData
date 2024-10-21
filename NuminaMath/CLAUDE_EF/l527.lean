import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_prime_factors_of_1170_l527_52725

theorem sum_of_extreme_prime_factors_of_1170 : 
  let factors := Nat.factors 1170
  ∃ (min max : Nat), min ∈ factors ∧ max ∈ factors ∧ 
    (∀ p ∈ factors, min ≤ p) ∧ 
    (∀ p ∈ factors, p ≤ max) ∧ 
    min + max = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_prime_factors_of_1170_l527_52725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l527_52734

theorem square_difference (x y : ℤ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  ∃ a b : ℤ, x = a^2 ∧ y = b^2 →
  100 ≤ (x + y) / 2 →
  (x + y) / 2 < 1000 →
  ∃ p q r : ℤ, (x + y) / 2 = 100*p + 10*q + r →
  ∃ s t u : ℤ, Int.sqrt (x * y) = 100*s + 10*t + u →
  ({p, q, r} : Finset ℤ) = {s, t, u} →
  (x + y) % 5 = 0 →
  |x - y| = 65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l527_52734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l527_52704

-- Define the binary operation on nonzero real numbers
def diamond (a b : ℝ) : ℝ := a * b

-- State the properties of the operation
axiom diamond_assoc (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) : 
  diamond a (diamond b c) = (diamond a b) / c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- State the theorem to be proved
theorem solution_equation (x : ℝ) (hx : x ≠ 0) :
  diamond 504 (diamond 7 x) = 50 → x = 25 / 1766 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l527_52704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_25_and_200_l527_52764

theorem multiples_of_15_between_25_and_200 : 
  (((200 / 15) * 15 - ((25 / 15 + 1) * 15)) / 15 + 1 = 12) := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_25_and_200_l527_52764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l527_52707

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the foci
def focus (x y : ℝ) : Prop := hyperbola x y ∧ y = 0 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2)

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := x = y ∨ x = -y

-- Theorem statement
theorem focus_to_asymptote_distance :
  ∀ (fx fy ax ay : ℝ),
    focus fx fy →
    asymptote ax ay →
    ∃ (d : ℝ), d = 1 ∧ d = abs (ax * fx + ay * fy) / Real.sqrt (ax^2 + ay^2) :=
by
  sorry

#check focus_to_asymptote_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l527_52707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_x_axis_with_distance_l527_52769

theorem point_on_x_axis_with_distance (M : ℝ × ℝ × ℝ) (d : ℝ) :
  M = (4, 1, 2) →
  d = Real.sqrt 30 →
  ∃ (P : ℝ × ℝ × ℝ),
    (P.2.1 = 0 ∧ P.2.2 = 0) ∧
    Real.sqrt ((P.1 - M.1)^2 + (P.2.1 - M.2.1)^2 + (P.2.2 - M.2.2)^2) = d ∧
    (P = (9, 0, 0) ∨ P = (-1, 0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_x_axis_with_distance_l527_52769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_arrangement_theorem_l527_52776

def OctagonArrangement := Fin 8 → Fin 8

def SumOfThree (arr : OctagonArrangement) (i : Fin 8) : ℕ :=
  (arr i).val + (arr ((i + 1) % 8)).val + (arr ((i + 2) % 8)).val

def ValidArrangement (arr : OctagonArrangement) : Prop :=
  (∀ i : Fin 8, arr i ≠ arr ((i + 1) % 8)) ∧
  (∀ i : Fin 8, (arr i).val ∈ Finset.range 8)

theorem octagon_arrangement_theorem :
  (∃ arr : OctagonArrangement, ValidArrangement arr ∧ 
    ∀ i : Fin 8, SumOfThree arr i > 11) ∧
  ¬(∃ arr : OctagonArrangement, ValidArrangement arr ∧ 
    ∀ i : Fin 8, SumOfThree arr i > 13) := by
  sorry

#check octagon_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_arrangement_theorem_l527_52776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l527_52782

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1 / 2

-- Define the theorem
theorem function_properties :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 0) ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -3 / 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -3 / 2) ∧
  (∀ (A B C : ℝ), 
    0 < B ∧ B < Real.pi ∧
    f B = 0 ∧
    2 * Real.sin A = 3 * Real.sin B ∧
    2 = 3 * Real.sin C ∧
    Real.cos C = 1 / 2 →
    Real.sin A = 3 * Real.sqrt 21 / 14) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l527_52782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l527_52730

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the specific triangle
def special_triangle : Triangle where
  A := sorry
  B := sorry
  C := sorry
  a := 2
  b := sorry
  c := sorry

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : Real.sin (t.B / 2) = Real.sqrt 5 / 5)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 4) :
  Real.cos t.B = 3/5 ∧ 
  t.b = Real.sqrt 17 ∧ 
  t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l527_52730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_is_3_l527_52700

-- Define the points
def A : ℚ × ℚ := (3, 4)

-- Define the reflection over y-axis
def reflect_y (p : ℚ × ℚ) : ℚ × ℚ := (-p.1, p.2)

-- Define the reflection over y = -x
def reflect_neg_x (p : ℚ × ℚ) : ℚ × ℚ := (p.2, p.1)

-- Define point B
def B : ℚ × ℚ := reflect_y A

-- Define point C
def C : ℚ × ℚ := reflect_neg_x B

-- Calculate the area of triangle ABC
noncomputable def area_triangle_ABC : ℚ :=
  let base := |A.1 - B.1|
  let height := |C.2 - A.2|
  (1/2) * base * height

-- Theorem statement
theorem area_triangle_ABC_is_3 : area_triangle_ABC = 3 := by
  -- Unfold definitions
  unfold area_triangle_ABC
  unfold C
  unfold B
  unfold reflect_neg_x
  unfold reflect_y
  unfold A
  -- Simplify
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_is_3_l527_52700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_completion_time_l527_52709

/-- Represents a runner on the circular track -/
inductive Runner
| One
| Two
| Three

/-- Represents an encounter between two runners -/
structure Encounter where
  runner1 : Runner
  runner2 : Runner
  time : ℕ

/-- The circular track with three runners -/
structure CircularTrack where
  runners : List Runner
  speed : ℝ
  encounters : List Encounter

/-- The problem setup -/
def runnersProblem : CircularTrack :=
  { runners := [Runner.One, Runner.Two, Runner.Three]
  , speed := 1  -- Arbitrary constant speed
  , encounters :=
    [ { runner1 := Runner.One, runner2 := Runner.Two, time := 0 }
    , { runner1 := Runner.Two, runner2 := Runner.Three, time := 20 }
    , { runner1 := Runner.Three, runner2 := Runner.One, time := 50 }
    ]
  }

/-- Calculate the track completion time for a runner -/
def track_completion_time_for_runner (track : CircularTrack) (r : Runner) : ℕ :=
  100 -- Placeholder value, replace with actual calculation if needed

/-- The theorem to be proved -/
theorem track_completion_time (track : CircularTrack) :
  track = runnersProblem →
  ∃ (t : ℕ), t = 100 ∧ (∀ (r : Runner), r ∈ track.runners → t = track_completion_time_for_runner track r) :=
by
  intro h
  use 100
  constructor
  · rfl
  · intro r hr
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_completion_time_l527_52709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l527_52736

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n * n) % d = 4) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l527_52736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_divisible_by_d_l527_52790

theorem ones_divisible_by_d (d : ℕ) (h_coprime : Nat.Coprime d 10) :
  ∃ n : ℕ, (10^n - 1) / 9 ∣ d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_divisible_by_d_l527_52790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l527_52717

theorem cos_shift_equivalence (x : ℝ) : 
  Real.cos (2*x + π/3) = Real.cos (2*(x + π/6)) := by
  have h : 2*x + π/3 = 2*(x + π/6) := by
    ring
  rw [h]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l527_52717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_diagonals_l527_52796

/-- A regular decagon -/
structure RegularDecagon where
  vertices : Fin 10 → ℝ × ℝ

/-- A diagonal in a regular decagon -/
structure Diagonal (d : RegularDecagon) where
  start : Fin 10
  finish : Fin 10
  ne : start ≠ finish

/-- The number of vertices on one side of a diagonal -/
def k (d : RegularDecagon) (diag : Diagonal d) : Nat :=
  sorry

/-- Predicate for perpendicular diagonals -/
def perpendicular (d : RegularDecagon) (diag1 diag2 : Diagonal d) : Prop :=
  sorry

/-- Predicate for parallel diagonals -/
def parallel_diag (d : RegularDecagon) (diag1 diag2 : Diagonal d) : Prop :=
  sorry

/-- Number of perpendicular diagonals to a given diagonal -/
def num_perpendicular (d : RegularDecagon) (diag : Diagonal d) : Nat :=
  sorry

/-- Number of parallel diagonals to a given diagonal -/
def num_parallel (d : RegularDecagon) (diag : Diagonal d) : Nat :=
  sorry

/-- Number of parts the longest diagonal is divided into by perpendicular diagonals -/
def num_parts (d : RegularDecagon) (diag : Diagonal d) : Nat :=
  sorry

/-- Lengths of parts the longest diagonal is divided into by perpendicular diagonals -/
def part_lengths (d : RegularDecagon) (diag : Diagonal d) : List ℝ :=
  sorry

theorem regular_decagon_diagonals (d : RegularDecagon) :
  (∀ diag : Diagonal d, ∃ perp parallel : Diagonal d,
    perp.start ≠ diag.start ∧ perp.finish ≠ diag.finish ∧
    parallel.start ≠ diag.start ∧ parallel.finish ≠ diag.finish ∧
    perpendicular d perp diag ∧ parallel_diag d parallel diag) ∧
  (∀ diag : Diagonal d,
    (k d diag % 2 = 1 → num_perpendicular d diag = 3 ∧ num_parallel d diag = 3) ∧
    (k d diag % 2 = 0 → num_perpendicular d diag = 4 ∧ num_parallel d diag = 2)) ∧
  (∃ longest : Diagonal d,
    num_parts d longest = 5 ∧
    part_lengths d longest = [0.382, 0.5, 0.618, 0.5, 0.382]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_diagonals_l527_52796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_simplification_l527_52792

/-- Prove that the given vector expression simplifies to the expected result. -/
theorem vector_expression_simplification (a b : ℝ × ℝ × ℝ) :
  (1/3 : ℝ) • ((1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b)) = 2 • b - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_simplification_l527_52792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l527_52713

theorem terminal_side_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ Real.cos α = x ∧ Real.sin α = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l527_52713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l527_52772

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = 6 * Real.pi ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l527_52772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_closest_to_1700_l527_52732

theorem product_closest_to_1700 : 
  let product := (0.000258 : ℝ) * 6539721
  ∀ x ∈ ({1600, 1800, 1900, 2000} : Set ℝ), |product - 1700| < |product - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_closest_to_1700_l527_52732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_diff_max_l527_52705

-- Define the two functions
def f (x : ℝ) : ℝ := 5 - x^2 + x^3
def g (x : ℝ) : ℝ := 1 + x^2 + x^3

-- Theorem statement
theorem intersection_y_diff_max :
  let intersection_points := {x : ℝ | f x = g x}
  let y_diff := fun x₁ x₂ => |f x₁ - f x₂|
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    y_diff x₁ x₂ = 4 * Real.sqrt 2 ∧
    ∀ (y₁ y₂ : ℝ), y₁ ∈ intersection_points → y₂ ∈ intersection_points →
      y_diff y₁ y₂ ≤ 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_diff_max_l527_52705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l527_52718

-- Define the plane
variable (x y : ℝ)

-- Define points F and Q
def F : ℝ × ℝ := (0, 1)
def Q (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define vectors
def QP (x y : ℝ) : ℝ × ℝ := (0, y + 1)
def QF (x : ℝ) : ℝ × ℝ := (-x, 2)
def FP (x y : ℝ) : ℝ × ℝ := (x, y - 1)
def FQ (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition for point P
def condition (x y : ℝ) : Prop :=
  dot_product (QP x y) (QF x) = dot_product (FP x y) (FQ x)

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define the distance function to the line y = x - 3
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y - 3| / Real.sqrt 2

-- Main theorem
theorem main_theorem :
  ∀ x y, condition x y → trajectory x y ∧
  ∃ x₀ y₀, trajectory x₀ y₀ ∧
  x₀ = 2 ∧ y₀ = 1 ∧
  ∀ x' y', trajectory x' y' →
  distance_to_line x₀ y₀ ≤ distance_to_line x' y' ∧
  distance_to_line x₀ y₀ = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l527_52718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_with_difference_divisible_by_2022_l527_52791

theorem two_integers_with_difference_divisible_by_2022 
  (integers : Finset ℤ) 
  (h : integers.card = 2023) : 
  ∃ a b, a ∈ integers ∧ b ∈ integers ∧ a ≠ b ∧ (a - b) % 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_with_difference_divisible_by_2022_l527_52791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_cut_pyramid_specific_l527_52770

/-- The volume of a cut-off pyramid -/
noncomputable def volume_cut_pyramid (base_edge slant_edge cut_height : ℝ) : ℝ :=
  let original_height := Real.sqrt (slant_edge ^ 2 - (base_edge * Real.sqrt 2 / 2) ^ 2)
  let new_height := original_height - cut_height
  let ratio := new_height / original_height
  48 * (ratio ^ 3) * new_height

/-- Theorem: Volume of the cut-off pyramid -/
theorem volume_cut_pyramid_specific :
  volume_cut_pyramid 12 15 5 = 48 * ((Real.sqrt 153 - 5) / Real.sqrt 153) ^ 3 * (Real.sqrt 153 - 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_cut_pyramid_specific_l527_52770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duration_is_five_years_l527_52752

/-- The duration (in years) for which B gains a certain amount by borrowing and lending money -/
noncomputable def duration_years (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (gain : ℝ) : ℝ :=
  gain / (principal * (lend_rate - borrow_rate))

/-- Proof that the duration is 5 years under the given conditions -/
theorem duration_is_five_years :
  let principal := (3200 : ℝ)
  let borrow_rate := (0.12 : ℝ)
  let lend_rate := (0.145 : ℝ)
  let gain := (400 : ℝ)
  duration_years principal borrow_rate lend_rate gain = 5 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duration_is_five_years_l527_52752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l527_52748

/-- A power function that passes through the point (2, 8) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

theorem power_function_through_point_value (a : ℝ) (h : f a 2 = 8) : f a (1/2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l527_52748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_average_height_l527_52742

/-- Calculates the volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Calculates the height of the center of each sphere in the snowman -/
def sphereHeight (r : ℝ) (h : ℝ) : ℝ := r + h

theorem snowman_average_height :
  let r₁ : ℝ := 10
  let r₂ : ℝ := 8
  let r₃ : ℝ := 6
  let v₁ := sphereVolume r₁
  let v₂ := sphereVolume r₂
  let v₃ := sphereVolume r₃
  let h₁ := r₁
  let h₂ := sphereHeight r₂ (2 * r₁)
  let h₃ := sphereHeight r₃ (2 * r₁ + 2 * r₂)
  let totalVolume := v₁ + v₂ + v₃
  let weightedHeight := v₁ * h₁ + v₂ * h₂ + v₃ * h₃
  weightedHeight / totalVolume = 58 / 3 := by
  sorry

#eval (58 : Nat) + (3 : Nat)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_average_height_l527_52742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_light_difference_l527_52761

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light
deriving BEq, Repr

/-- Represents a row in the grid -/
def Row := List Square

/-- Represents the entire grid -/
def Grid := List Row

/-- Creates a row starting with the given square -/
def createRow (startSquare : Square) : Row :=
  match startSquare with
  | Square.Dark => [Square.Dark, Square.Light, Square.Dark, Square.Light, Square.Dark, Square.Light, Square.Dark, Square.Light, Square.Dark]
  | Square.Light => [Square.Light, Square.Dark, Square.Light, Square.Dark, Square.Light, Square.Dark, Square.Light, Square.Dark, Square.Light]

/-- Creates the 5x9 grid -/
def createGrid : Grid :=
  [createRow Square.Dark, createRow Square.Light, createRow Square.Dark, createRow Square.Light, createRow Square.Dark]

/-- Counts the number of dark squares in a row -/
def countDarkInRow (row : Row) : Nat :=
  row.filter (· == Square.Dark) |>.length

/-- Counts the number of light squares in a row -/
def countLightInRow (row : Row) : Nat :=
  row.filter (· == Square.Light) |>.length

/-- Counts the total number of dark squares in the grid -/
def countDarkInGrid (grid : Grid) : Nat :=
  grid.map countDarkInRow |>.sum

/-- Counts the total number of light squares in the grid -/
def countLightInGrid (grid : Grid) : Nat :=
  grid.map countLightInRow |>.sum

theorem dark_light_difference :
  let grid := createGrid
  countDarkInGrid grid - countLightInGrid grid = 5 := by sorry

#eval let grid := createGrid; countDarkInGrid grid - countLightInGrid grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_light_difference_l527_52761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l527_52785

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def point_on_segment (C D Q : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D

def segment_ratio (C D Q : V) (r : ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ Q = (1 - t) • C + t • D ∧ t / (1 - t) = r

theorem segment_division (C D Q : V) :
  point_on_segment C D Q →
  segment_ratio C D Q (4 / 1) →
  Q = (1 / 5) • C + (4 / 5) • D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_l527_52785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l527_52758

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y - 5 = 0

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = Real.sqrt 5 - 1 ∧
  ∀ (P : ℝ × ℝ), circle_C P.1 P.2 →
  ∀ (Q : ℝ × ℝ), line_l Q.1 Q.2 →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l527_52758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l527_52760

/-- The parabola defined by the equation y^2 = 4x has its focus at the point (1, 0). -/
theorem parabola_focus (x y : ℝ) : y^2 = 4*x → (1, 0) = (x + 1, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l527_52760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l527_52756

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 1 - 1/x - 1/(1-x))

theorem functional_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  f x + f (1/(1-x)) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_l527_52756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l527_52789

theorem trig_problem (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = -4*Real.sqrt 3/5) 
  (h2 : -π/2 < α ∧ α < 0) : 
  Real.cos (α + 2*π/3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l527_52789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_root_polynomial_l527_52777

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  b : ℤ
  c : ℤ

/-- Checks if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (r : ℤ), r * r = p.b * p.b - 4 * p.c

/-- Represents a valid step in the transformation process -/
inductive Step
  | change_b : Step
  | change_c : Step

/-- Applies a step to a quadratic polynomial -/
def apply_step (p : QuadraticPolynomial) (s : Step) : QuadraticPolynomial :=
  match s with
  | Step.change_b => ⟨p.b + 1, p.c⟩
  | Step.change_c => ⟨p.b, p.c - 1⟩

/-- Theorem: There exists a quadratic polynomial with integer roots in the transformation sequence -/
theorem exists_integer_root_polynomial :
  ∃ (steps : List Step),
    let init := QuadraticPolynomial.mk 10 20
    let final := QuadraticPolynomial.mk 20 10
    let sequence := init :: (steps.scanl apply_step init)
    (final ∈ sequence) ∧ (∃ (p : QuadraticPolynomial), p ∈ sequence ∧ has_integer_roots p) :=
  sorry

#check exists_integer_root_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_root_polynomial_l527_52777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_below_diagonal_equals_total_area_l527_52780

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  s : ℝ  -- Length of the short side
  long_side : ℝ  -- Length of the long side
  long_twice_short : long_side = 2 * s

/-- Point P on the rectangle -/
noncomputable def point_P (rect : SpecialRectangle) : ℝ × ℝ := (2 * rect.s / 3, 0)

/-- Point Q on the rectangle -/
noncomputable def point_Q (rect : SpecialRectangle) : ℝ × ℝ := (rect.s, rect.s / 2)

/-- Area below the diagonal passing through P and Q -/
noncomputable def area_below_diagonal (rect : SpecialRectangle) : ℝ := sorry

/-- Total area of the rectangle -/
noncomputable def total_area (rect : SpecialRectangle) : ℝ := rect.s * rect.long_side

/-- Theorem stating that the area below the diagonal is equal to the total area -/
theorem area_below_diagonal_equals_total_area (rect : SpecialRectangle) :
  area_below_diagonal rect = total_area rect := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_below_diagonal_equals_total_area_l527_52780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l527_52731

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

/-- Theorem stating that the maximum value of f(x) is 3 -/
theorem f_max_value :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l527_52731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_vectors_l527_52784

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ t.c = 3 ∧ (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

-- Theorem statement
theorem dot_product_of_vectors (t : Triangle) 
  (h : triangle_conditions t) : t.a * t.c * Real.cos t.B = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_vectors_l527_52784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_subsets_l527_52787

-- Define the property of having arbitrarily large gaps
def has_arbitrarily_large_gaps (S : Set ℕ) : Prop :=
  ∀ M : ℕ, ∃ k : ℕ, ∃ x y : ℕ, x ∈ S ∧ y ∈ S ∧ x < y ∧ y - x ≥ M

-- Define the property of having infinitely many composite numbers
def has_infinitely_many_composites (S : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S ∧ ¬ Nat.Prime m

theorem existence_of_special_subsets :
  ∃ A B : Set ℕ,
    (A.Nonempty ∧ B.Nonempty) ∧
    (A ∩ B = {1}) ∧
    (∀ n : ℕ, ∃ a ∈ A, ∃ b ∈ B, n = a * b) ∧
    (∀ p : ℕ, Nat.Prime p → (∃ a ∈ A, p ∣ a) ∧ (∃ b ∈ B, p ∣ b)) ∧
    (has_arbitrarily_large_gaps A ∨ has_arbitrarily_large_gaps B) ∧
    (has_infinitely_many_composites A ∧ has_infinitely_many_composites B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_subsets_l527_52787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_one_solution_equation_two_solutions_l527_52767

-- Define the ﹫ operation
def at_op (x y : ℝ) : ℝ := 3 * x - y

-- Theorem for equation 1
theorem equation_one_solution :
  ∃ x : ℝ, at_op x (at_op 2 3) = 1 ∧ x = 4/3 := by sorry

-- Theorem for equation 2
theorem equation_two_solutions :
  ∃ x : ℝ, at_op (x^2) 2 = 10 ∧ (x = 2 ∨ x = -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_one_solution_equation_two_solutions_l527_52767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l527_52722

/-- Given an ellipse C and a line l with specific properties, 
    prove the equation of C and that a line AB passes through a fixed point N. -/
theorem ellipse_and_line_properties 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (C : Set (ℝ × ℝ)) 
  (h_C : C = {(x, y) | x^2/a^2 + y^2/b^2 = 1}) 
  (h_ecc : (a^2 - b^2)/a^2 = 1/2) 
  (l : Set (ℝ × ℝ)) 
  (h_l : l = {(x, y) | x - y + Real.sqrt 2 = 0}) 
  (h_tangent : ∃ (p : ℝ × ℝ), p ∈ l ∧ p.1^2 + p.2^2 = b^2) 
  (M : ℝ × ℝ) 
  (h_M : M ∈ C ∧ M.2 = b) 
  (A B : ℝ × ℝ) 
  (h_A : A ∈ C) (h_B : B ∈ C) 
  (k₁ k₂ : ℝ) 
  (h_MA : A.2 - M.2 = k₁ * (A.1 - M.1)) 
  (h_MB : B.2 - M.2 = k₂ * (B.1 - M.1)) 
  (h_k : k₁ + k₂ = 4) : 
  (C = {(x, y) | x^2/2 + y^2 = 1}) ∧ 
  (∃ (t : ℝ), (1-t) • A + t • B = (-1/2, -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l527_52722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_greater_l527_52762

/-- Represents a student's quiz scores and weights -/
structure QuizData where
  first_three_average : ℚ
  third_quiz_score : ℚ
  first_three_weight : ℚ
  last_two_weight : ℚ
  last_two_min_score : ℚ

/-- Calculates the new weighted average given quiz data -/
def newWeightedAverage (data : QuizData) : ℚ :=
  (data.first_three_average * 3 * data.first_three_weight + 2 * data.last_two_min_score * data.last_two_weight) /
  (3 * data.first_three_weight + 2 * data.last_two_weight)

/-- Theorem stating that the new weighted average is greater than the original average -/
theorem new_average_greater (data : QuizData) 
    (h1 : data.first_three_average = 94)
    (h2 : data.third_quiz_score = 92)
    (h3 : data.first_three_weight = 15/100)
    (h4 : data.last_two_weight = 1/5)
    (h5 : data.last_two_min_score > data.first_three_average) :
    newWeightedAverage data > data.first_three_average := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_greater_l527_52762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_p_is_false_q_is_true_not_p_and_q_not_p_and_not_q_not_p_or_q_not_p_or_not_q_l527_52763

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Theorem statements
theorem problem_solution :
  (¬p) ∧ q ∧ ¬(p ∧ q) ∧ ¬(p ∧ ¬q) ∧ ((¬p) ∨ q) ∧ ((¬p) ∨ (¬q)) := by
  sorry

-- Individual theorems for each part
theorem p_is_false : ¬p := by
  sorry

theorem q_is_true : q := by
  sorry

theorem not_p_and_q : ¬(p ∧ q) := by
  sorry

theorem not_p_and_not_q : ¬(p ∧ ¬q) := by
  sorry

theorem not_p_or_q : (¬p) ∨ q := by
  sorry

theorem not_p_or_not_q : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_p_is_false_q_is_true_not_p_and_q_not_p_and_not_q_not_p_or_q_not_p_or_not_q_l527_52763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l527_52701

/-- Represents a parabola with equation x² = 16y -/
structure Parabola where
  equation : ∀ (x y : ℝ), x^2 = 16*y

/-- The directrix of a parabola -/
def directrix (p : Parabola) : Set (ℝ × ℝ) :=
  {pair | pair.2 = -4}

theorem parabola_directrix (p : Parabola) : 
  directrix p = {pair : ℝ × ℝ | pair.2 = -4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l527_52701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_of_symmetric_difference_l527_52754

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem intersection_size_of_symmetric_difference
  (x y : Finset ℤ)
  (hx : x.card = 8)
  (hy : y.card = 18)
  (hxy : (symmetric_difference x y).card = 14) :
  (x ∩ y).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_of_symmetric_difference_l527_52754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supply_exceeds_demand_and_inventory_increases_l527_52788

/-- Represents the annual production or sales of a product at time t -/
def LinearGrowth (rate : ℝ) (initial : ℝ) (t : ℝ) : ℝ := initial + rate * t

/-- The production function -/
def Production (t : ℝ) : ℝ := LinearGrowth 2 100 t

/-- The sales function -/
def Sales (t : ℝ) : ℝ := LinearGrowth 1 100 t

/-- The inventory at time t -/
def Inventory (t : ℝ) : ℝ := Production t - Sales t

theorem supply_exceeds_demand_and_inventory_increases :
  (∀ t > 0, Production t > Sales t) ∧ 
  (∀ t₁ t₂, t₁ < t₂ → Inventory t₁ < Inventory t₂) := by
  sorry

#check supply_exceeds_demand_and_inventory_increases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supply_exceeds_demand_and_inventory_increases_l527_52788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rounded_to_nearest_dollar_l527_52728

def purchase1 : ℚ := 2.47
def purchase2 : ℚ := 7.51
def purchase3 : ℚ := 11.56
def purchase4 : ℚ := 4.98

def roundToNearestDollar (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

def totalPurchases : ℚ := purchase1 + purchase2 + purchase3 + purchase4

theorem total_rounded_to_nearest_dollar :
  roundToNearestDollar totalPurchases = 27 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rounded_to_nearest_dollar_l527_52728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l527_52724

/-- The function f(x) = a^x + b^x - c^x --/
noncomputable def f (a b c x : ℝ) : ℝ := a^x + b^x - c^x

/-- Theorem stating the three properties of the function f --/
theorem f_properties (a b c : ℝ) 
  (h1 : c > a) (h2 : a > 0) (h3 : c > b) (h4 : b > 0) 
  (h5 : a + b > c) : 
  (∀ x < 1, f a b c x > 0) ∧ 
  (∃ x > 0, ¬(a^x + b^x > c^x ∧ b^x + c^x > a^x ∧ c^x + a^x > b^x)) ∧
  ((a^2 + b^2 < c^2) → ∃ x ∈ Set.Ioo 1 2, f a b c x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l527_52724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_theorem_l527_52735

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line tangent to the circle
def tangent_line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y - 3 = 0

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

-- Define the intersecting line
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1/3

theorem ellipse_and_circle_theorem
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 2 / 2)
  (h4 : ∃ (x y : ℝ), tangent_line x y ∧ (x + c)^2 + y^2 = (2*c)^2)
  (h5 : ellipse a b 0 (-1/3)) :
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2/2 + y^2 = 1) ∧
  (∀ (k : ℝ), ∃ (A B : ℝ × ℝ),
    ellipse a b A.1 A.2 ∧
    ellipse a b B.1 B.2 ∧
    intersecting_line k A.1 A.2 ∧
    intersecting_line k B.1 B.2 ∧
    (A.1 - 0)^2 + (A.2 - 1)^2 = (B.1 - 0)^2 + (B.2 - 1)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_theorem_l527_52735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l527_52757

-- Define the curve C'
def curve_C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function to minimize
noncomputable def f (x y : ℝ) : ℝ := x + 2 * Real.sqrt 3 * y

-- Theorem statement
theorem min_value_on_curve :
  ∃ (min : ℝ), min = -4 ∧ ∀ (x y : ℝ), curve_C' x y → f x y ≥ min :=
by
  -- We'll use -4 as our minimum value
  use -4
  constructor
  · -- Prove that min = -4
    rfl
  · -- Prove that for all points on the curve, f(x,y) ≥ -4
    intros x y h_curve
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l527_52757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_is_24_l527_52733

/-- The polynomial z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The property of k being the smallest positive integer such that f(z) divides z^k - 1 -/
def is_smallest_divisor (k : ℕ) : Prop :=
  (∀ z : ℂ, f z = 0 → z^k = 1) ∧
  (∀ m : ℕ, 0 < m → m < k → ∃ z : ℂ, f z = 0 ∧ z^m ≠ 1)

/-- The theorem stating that 24 is the smallest positive integer k such that f(z) divides z^k - 1 -/
theorem smallest_divisor_is_24 : is_smallest_divisor 24 := by sorry

#check smallest_divisor_is_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_is_24_l527_52733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_8cos_x_range_l527_52747

theorem cos_2x_minus_8cos_x_range :
  ∀ x : ℝ, -7 ≤ Real.cos (2 * x) - 8 * Real.cos x ∧ Real.cos (2 * x) - 8 * Real.cos x ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_8cos_x_range_l527_52747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digit_occurrences_l527_52771

/-- Helper function to count the occurrences of a digit in a natural number -/
def count_occurrences (d : ℕ) (n : ℕ) : ℕ := sorry

/-- For any two natural numbers m and n, there exists a natural number c such that
    c·m and c·n have the same number of occurrences for each non-zero digit. -/
theorem same_digit_occurrences (m n : ℕ) : ∃ c : ℕ,
  ∀ d : ℕ, d ≠ 0 → (count_occurrences d (c * m) = count_occurrences d (c * n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digit_occurrences_l527_52771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_a1efd1_is_4_sqrt_13_l527_52743

/-- Rectangular prism with given dimensions and volume ratios -/
structure RectangularPrism where
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  v1 : ℝ
  v2 : ℝ
  v3 : ℝ

/-- The area of face A₁EFD₁ in the given rectangular prism -/
noncomputable def areaA1EFD1 (p : RectangularPrism) : ℝ :=
  p.ad * Real.sqrt 13

/-- Theorem stating that the area of face A₁EFD₁ is 4√13 -/
theorem area_a1efd1_is_4_sqrt_13 (p : RectangularPrism)
  (h1 : p.ab = 6)
  (h2 : p.ad = 4)
  (h3 : p.aa1 = 3)
  (h4 : p.v1 + p.v2 + p.v3 = p.ab * p.ad * p.aa1)
  (h5 : p.v1 = 1)
  (h6 : p.v2 = 4)
  (h7 : p.v3 = 1) :
  areaA1EFD1 p = 4 * Real.sqrt 13 := by
  sorry

#check area_a1efd1_is_4_sqrt_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_a1efd1_is_4_sqrt_13_l527_52743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_five_tons_profit_three_at_two_tons_l527_52720

/-- Daily cost function -/
noncomputable def C (x : ℝ) : ℝ := x + 5

/-- Daily sales revenue function -/
noncomputable def S (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then 3 * x + k / (x - 8) + 7 else 16

/-- Daily profit function -/
noncomputable def L (x : ℝ) (k : ℝ) : ℝ := S x k - C x

/-- The value of k satisfying the given condition -/
def k : ℝ := 18

theorem max_profit_at_five_tons :
  ∀ x : ℝ, L x k ≤ 6 ∧ L 5 k = 6 := by
  sorry

theorem profit_three_at_two_tons : L 2 k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_five_tons_profit_three_at_two_tons_l527_52720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abs_diff_l527_52739

/-- The sum of absolute differences for a list of numbers -/
def sumAbsDiff (x : List ℝ) : ℝ :=
  List.sum (List.map (fun (pair : ℝ × ℝ) => |pair.1 - pair.2|)
    (List.join (List.mapIdx (fun i xi =>
      List.map (fun xj => (xi, xj)) (List.drop (i + 1) x)) x)))

/-- The maximum sum of absolute differences for n numbers between 0 and 1 -/
noncomputable def S (n : ℕ) : ℝ :=
  ⨆ (x : List ℝ) (h1 : x.length = n) (h2 : ∀ i, i ∈ List.range n → 0 ≤ x[i]! ∧ x[i]! ≤ 1), sumAbsDiff x

/-- The theorem stating the maximum sum of absolute differences -/
theorem max_sum_abs_diff (n : ℕ) : S n = ⌊(n : ℝ)^2 / 4⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abs_diff_l527_52739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l527_52759

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a + 3)*x - 4*a + 3 else a^x

theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  1 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l527_52759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l527_52716

/-- The original function before transformation -/
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x * Real.cos x

/-- The line along which the graph is translated -/
noncomputable def translation_line (x : ℝ) : ℝ := Real.sqrt 3 * x

/-- The resulting function after transformation -/
noncomputable def f (x : ℝ) : ℝ := 1/2 * Real.sin (2*x + 2) - Real.sqrt 3

/-- Theorem stating that f is the result of translating the original function -/
theorem translation_theorem :
  ∀ x : ℝ, f x = original_function (x + 2) - translation_line 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l527_52716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l527_52775

/-- An arithmetic sequence with a_5 = 2 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a_5_eq_2 : a 5 = 2

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: For an arithmetic sequence with a_5 = 2, 2S_6 + S_12 = 48 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 2 * S seq 6 + S seq 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l527_52775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l527_52723

-- Define the function f on [-1, 1]
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 0 then 1 / (4^x) - b / (2^x) else 2^x - 4^x

-- State the theorem
theorem odd_function_properties :
  ∃ (b : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f b (-x) = -(f b x)) ∧  -- f is odd on [-1, 1]
    (∀ x ∈ Set.Icc (-1 : ℝ) 0, f b x = 1 / (4^x) - b / (2^x)) ∧  -- f definition on [-1, 0]
    (b = 1) ∧  -- b equals 1
    (∀ x ∈ Set.Icc 0 1, f b x = 2^x - 4^x) ∧  -- f definition on [0, 1]
    (∀ x ∈ Set.Icc 0 1, f b x ≤ 0) ∧  -- maximum value is 0
    (∃ x ∈ Set.Icc 0 1, f b x = 0) ∧  -- maximum value is achieved
    (∀ x ∈ Set.Icc 0 1, f b x ≥ -2) ∧  -- minimum value is -2
    (∃ x ∈ Set.Icc 0 1, f b x = -2)  -- minimum value is achieved
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l527_52723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l527_52766

-- Define a type for lines in a plane
structure Line where
  id : ℕ

-- Define a function to represent the distance between two lines
noncomputable def distance (l1 l2 : Line) : ℝ := sorry

-- Define a predicate for parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_lines_distance 
  (a b c : Line) 
  (parallel_ab : parallel a b) 
  (parallel_bc : parallel b c) 
  (parallel_ac : parallel a c) 
  (dist_ab : distance a b = 5) 
  (dist_ac : distance a c = 3) : 
  distance b c = 2 ∨ distance b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l527_52766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l527_52751

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (x + Real.pi/4))^2 + 2 * Real.sin (Real.pi/4 - x) * Real.cos (Real.pi/4 - x)

theorem min_value_of_f :
  ∃ (x : ℝ), Real.pi/2 ≤ x ∧ x ≤ 3*Real.pi/4 ∧
  (∀ (y : ℝ), Real.pi/2 ≤ y ∧ y ≤ 3*Real.pi/4 → f x ≤ f y) ∧
  f x = 1 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l527_52751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_exists_max_sin_cos_l527_52753

theorem max_sin_cos (x : ℝ) : Real.sin x * Real.cos x ≤ 1/2 := by
  sorry

theorem exists_max_sin_cos : ∃ x : ℝ, Real.sin x * Real.cos x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_exists_max_sin_cos_l527_52753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_with_2010_factors_l527_52703

theorem least_factorial_with_2010_factors : 
  (∃ n : ℕ, (Nat.factorial n).factors.length ≥ 2010) ∧ 
  (∀ m : ℕ, m < 14 → (Nat.factorial m).factors.length < 2010) ∧ 
  (Nat.factorial 14).factors.length ≥ 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_with_2010_factors_l527_52703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_common_points_inequality_proof_l527_52744

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.exp x
def g (x : ℝ) := Real.log x

-- Part I
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ + 2 = g x₀ ∧ k = 1 / x₀) → k = Real.exp (-3) :=
sorry

-- Part II
def h (x : ℝ) := Real.exp x / (x ^ 2)

theorem common_points (m : ℝ) (hm : m > 0) :
  (m < Real.exp 2 / 4 → ∀ x, x > 0 → f x ≠ m * x^2) ∧
  (m = Real.exp 2 / 4 → ∃! x, x > 0 ∧ f x = m * x^2) ∧
  (m > Real.exp 2 / 4 → ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f x₁ = m * x₁^2 ∧ f x₂ = m * x₂^2) :=
sorry

-- Part III
theorem inequality_proof (a b : ℝ) (hab : a < b) :
  (f a + f b) / 2 > (f b - f a) / (b - a) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_common_points_inequality_proof_l527_52744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l527_52765

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  5 * x^2 + 20 * x + 9 * y^2 - 36 * y + 36 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 4 * Real.sqrt 5 * Real.pi / 3

/-- Theorem stating that the area of the ellipse defined by the given equation is 4√5π/3 -/
theorem area_of_ellipse :
  ∀ x y : ℝ, ellipse_equation x y → ∃ A : ℝ, A = ellipse_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l527_52765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_theorem_l527_52778

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the angle between two vectors -/
noncomputable def angle (v1 v2 : Point → Point) : ℝ := sorry

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point lies on a line segment -/
def onSegment (p1 p2 p : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (p1 p2 p3 p4 : Point) : Prop := sorry

theorem parallelogram_intersection_theorem 
  (ABCD : Parallelogram) 
  (E : Point) 
  (F : Point) :
  angle (fun p => ABCD.B) (fun p => ABCD.C) = 100 * π / 180 →
  distance ABCD.A ABCD.B = 12 →
  distance ABCD.B ABCD.C = 8 →
  onSegment ABCD.B ABCD.C E →
  distance ABCD.C E = 6 →
  intersect ABCD.A E ABCD.B ABCD.D →
  onSegment ABCD.B ABCD.D F →
  ∃ ε > 0, |distance F ABCD.D - 4.3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_theorem_l527_52778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2023_lt_tan_53_tan_period_l527_52714

open Real

theorem tan_2023_lt_tan_53 :
  tan (2023 * π / 180) < tan (53 * π / 180) :=
by
  sorry

-- Definitions and assumptions
theorem tan_period (x : ℝ) : tan x = tan (x + π) :=
by
  sorry

-- Note: We use radians instead of degrees in Lean,
-- so we convert degrees to radians using the formula: radians = degrees * π / 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2023_lt_tan_53_tan_period_l527_52714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_ratio_l527_52710

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent lines and their properties
structure TangentLine where
  line : ℝ × ℝ → ℝ × ℝ → Prop
  touchPoint1 : ℝ × ℝ
  touchPoint2 : ℝ × ℝ

-- Define a membership relation for points on a circle
def OnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Define the theorem
theorem tangent_circles_area_ratio 
  (ω₁ ω₂ : Circle) 
  (a b c : TangentLine) 
  (h_nonoverlap : ω₁ ≠ ω₂) 
  (h_external1 : OnCircle a.touchPoint1 ω₁ ∧ OnCircle a.touchPoint2 ω₂)
  (h_external2 : OnCircle b.touchPoint1 ω₁ ∧ OnCircle b.touchPoint2 ω₂)
  (h_internal : OnCircle c.touchPoint1 ω₁ ∧ OnCircle c.touchPoint2 ω₂)
  : (area_triangle a.touchPoint1 b.touchPoint1 c.touchPoint1) / 
    (area_triangle a.touchPoint2 b.touchPoint2 c.touchPoint2) = 
    ω₁.radius / ω₂.radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_area_ratio_l527_52710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_function_minimum_l527_52741

theorem inequality_and_function_minimum (a : ℕ) 
  (h1 : |3/2 - 2| < (a : ℝ))
  (h2 : |1/2 - 2| ≥ (a : ℝ)) : 
  (a = 1) ∧ 
  (∀ x : ℝ, |x + (a : ℝ)| + |x - 2| ≥ 3) ∧
  (∃ x : ℝ, |x + (a : ℝ)| + |x - 2| = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_function_minimum_l527_52741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_calls_max_calls_achievable_l527_52715

/-- Represents a class with boys and girls -/
structure ClassRoom where
  boys : Finset Nat
  girls : Finset Nat
  calls : Finset (Nat × Nat)
  unique_pairing : Finset (Nat × Nat)

/-- The conditions of the problem -/
def problem_conditions (c : ClassRoom) : Prop :=
  c.boys.card = 15 ∧
  c.girls.card = 15 ∧
  c.calls ⊆ c.boys.product c.girls ∧
  (∀ b g₁ g₂, (b, g₁) ∈ c.calls → (b, g₂) ∈ c.calls → g₁ = g₂) ∧
  c.unique_pairing.card = 15 ∧
  (∀ p, p ∈ c.unique_pairing → p.1 ∈ c.boys ∧ p.2 ∈ c.girls ∧ p ∈ c.calls) ∧
  (∀ b, b ∈ c.boys → ∃! g, g ∈ c.girls ∧ (b, g) ∈ c.unique_pairing) ∧
  (∀ g, g ∈ c.girls → ∃! b, b ∈ c.boys ∧ (b, g) ∈ c.unique_pairing)

/-- The main theorem -/
theorem max_calls (c : ClassRoom) (h : problem_conditions c) : c.calls.card ≤ 120 := by
  sorry

/-- The maximum number of calls is achievable -/
theorem max_calls_achievable : ∃ c : ClassRoom, problem_conditions c ∧ c.calls.card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_calls_max_calls_achievable_l527_52715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_675_degrees_l527_52779

theorem cos_675_degrees : Real.cos (675 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_675_degrees_l527_52779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_forms_of_z_l527_52786

noncomputable def z : ℂ := 3 - 3 * Complex.I * Real.sqrt 3

theorem complex_forms_of_z :
  (∃ (r : ℝ) (θ : ℝ), z = r * (Complex.cos θ + Complex.I * Complex.sin θ) ∧
                       r = 6 ∧
                       θ = 5 * Real.pi / 3) ∧
  (∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧
                       r = 6 ∧
                       θ = 5 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_forms_of_z_l527_52786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l527_52726

-- Problem 1
theorem problem_1 : (Real.sqrt 2)^2 - abs (1 - Real.sqrt 3) + Real.sqrt ((-3)^2) + Real.sqrt 81 = 15 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (x - 2*y)^2 - (x + 2*y + 3)*(x + 2*y - 3) = -8*x*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l527_52726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_bound_l527_52737

theorem triangle_geometric_sequence_bound (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.sin A = b * Real.sin B →
  b * Real.sin B = c * Real.sin C →
  ∃ q : ℝ, b = a * q ∧ c = a * q^2 →
  (Real.sqrt 5 - 1) / 2 < Real.sin A * (1 / Real.tan A + 1 / Real.tan B) ∧
  Real.sin A * (1 / Real.tan A + 1 / Real.tan B) < (Real.sqrt 5 + 1) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_bound_l527_52737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l527_52712

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) 
  (h1 : total = 850)
  (h2 : muslim_percent = 46 / 100)
  (h3 : hindu_percent = 28 / 100)
  (h4 : sikh_percent = 10 / 100) :
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 136 := by
  sorry

#check other_communities_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l527_52712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_squares_theorem_l527_52773

/-- Represents the number of squares in a rectangle --/
def RectangleSquares : Type := ℕ

/-- Represents the number of segments in a rectangle --/
def Segments : Type := ℕ

/-- The total number of segments in the given rectangle --/
def totalSegments : ℕ := 1997

/-- The width of the sheet in cm --/
def sheetWidth : ℝ := 21

/-- The height of the sheet in cm --/
def sheetHeight : ℝ := 29.7

/-- The side length of each small square in cm --/
def squareSide : ℝ := 0.5

/-- Function to calculate the number of segments given the number of squares on each side --/
def segmentsFromSquares (m n : ℕ) : ℕ :=
  m * (n + 1) + n * (m + 1)

/-- Theorem stating that the rectangle can only have 399, 117, or 42 squares --/
theorem rectangle_squares_theorem : 
  ∀ (squares : ℕ), 
    (∃ (m n : ℕ), segmentsFromSquares m n = totalSegments ∧ squares = m * n) → 
    (squares = 399 ∨ squares = 117 ∨ squares = 42) :=
by
  sorry

#eval totalSegments
#eval segmentsFromSquares 2 399
#eval segmentsFromSquares 8 117
#eval segmentsFromSquares 23 42

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_squares_theorem_l527_52773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_l527_52797

/-- The time it takes for a ball to hit the ground when thrown downward -/
def time_to_ground : ℝ := 2.5

/-- The height equation for the ball's motion -/
def height_equation (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 180

/-- Theorem stating that the ball hits the ground at the calculated time -/
theorem ball_hits_ground : 
  height_equation time_to_ground = 0 ∧ 
  ∀ t, 0 < t ∧ t < time_to_ground → height_equation t > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_l527_52797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_kids_l527_52721

/-- The number of kids Julia played with on a given day -/
def kids_played_with (day : String) : ℕ := sorry

/-- The total number of kids Julia played with on Monday and Tuesday -/
def total_monday_tuesday : ℕ := 33

theorem tuesday_kids :
  kids_played_with "Monday" = 15 →
  total_monday_tuesday = kids_played_with "Monday" + kids_played_with "Tuesday" →
  kids_played_with "Tuesday" = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_kids_l527_52721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l527_52749

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1007 + a 1008 + a 1009 = 18) :
  sum_arithmetic a 2015 = 12090 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l527_52749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_selected_l527_52708

-- Define the set of people
inductive Person : Type
| A | B | C | D

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of A being selected
def prob_A_selected : ℚ :=
  (choose 3 1 : ℚ) / (choose 4 2 : ℚ)

-- Theorem statement
theorem probability_A_selected :
  prob_A_selected = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_selected_l527_52708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l527_52746

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (25 - 5^x)

theorem f_range : Set.range f = Set.Icc 0 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l527_52746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_cow_price_is_460_l527_52729

def total_animals : ℕ := 2 + 8
def total_cost : ℕ := 1400
def num_cows : ℕ := 2
def num_goats : ℕ := 8
def avg_goat_price : ℕ := 60

def avg_cow_price : ℕ := (total_cost - num_goats * avg_goat_price) / num_cows

theorem avg_cow_price_is_460 (h1 : total_animals = num_cows + num_goats)
                             (h2 : total_cost = num_cows * avg_cow_price + num_goats * avg_goat_price) :
  avg_cow_price = 460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_cow_price_is_460_l527_52729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l527_52795

/-- Bill's miles on Saturday -/
def bill_saturday : ℕ := sorry

/-- Bill's miles on Sunday -/
def bill_sunday : ℕ := bill_saturday + 4

/-- Julia's miles on Sunday -/
def julia_sunday : ℕ := 2 * bill_sunday

/-- Total miles run by Bill and Julia on both days -/
def total_miles : ℕ := 32

theorem bill_sunday_miles :
  bill_saturday + bill_sunday + julia_sunday = total_miles →
  bill_sunday = 9 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_sunday_miles_l527_52795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomial_l527_52781

/-- A polynomial that takes integer values for n+1 consecutive integers takes integer values for all integers -/
theorem integer_valued_polynomial (n : ℕ) (f : ℤ → ℤ) (a : Fin (n + 1) → ℤ) (k : ℤ) :
  (∀ (i : Fin (n + 1)), f (k + ↑i) ∈ Set.range f) →
  (∀ (x : ℤ), ∃ (y : ℤ), f x = y) →
  ∀ (x : ℤ), f x ∈ Set.range f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomial_l527_52781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relationships_l527_52755

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Hypothesis that m and n are different lines
variable (h_diff_lines : m ≠ n)

-- Hypothesis that α and β are different planes
variable (h_diff_planes : α ≠ β)

-- Theorem statement
theorem geometric_relationships :
  ¬(∀ (m n : Line) (α β : Plane),
    (perpendicular_line_plane m α ∧ perpendicular_plane_plane α β → parallel_line_plane m β) ∧
    (parallel_line_plane n α ∧ parallel_plane_plane α β → parallel_line_plane n β) ∧
    (parallel_line_plane n α ∧ perpendicular_plane_plane α β → perpendicular_line_plane n β)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relationships_l527_52755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_fail_prob_worker_b_disqualification_prob_l527_52798

-- Define the probabilities of passing for Worker A and Worker B
variable (p_a p_b : ℝ)

-- Assumptions about the probabilities
variable (h_a : 0 ≤ p_a ∧ p_a ≤ 1)
variable (h_b : 0 ≤ p_b ∧ p_b ≤ 1)

-- Define the probability functions
def probability_of_failing_at_least_once_in_three_months (p : ℝ) : ℝ := 1 - p^3

def probability_of_disqualification_after_four_tests (p : ℝ) : ℝ := p^2 * (1-p)^2 + p * (1-p)^3

-- Theorem for Worker A
theorem worker_a_fail_prob (p_a : ℝ) (h_a : 0 ≤ p_a ∧ p_a ≤ 1) :
  1 - p_a^3 = probability_of_failing_at_least_once_in_three_months p_a := by
  sorry

-- Theorem for Worker B
theorem worker_b_disqualification_prob (p_b : ℝ) (h_b : 0 ≤ p_b ∧ p_b ≤ 1) :
  p_b^2 * (1-p_b)^2 + p_b * (1-p_b)^3 = probability_of_disqualification_after_four_tests p_b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_fail_prob_worker_b_disqualification_prob_l527_52798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_theorem_l527_52745

noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := (x^2 + D*x + E) / (3*x^2 - F*x - 18)

theorem asymptote_theorem (D E F : ℤ) :
  (∀ x, x ≠ -3 ∧ x ≠ 4 → g D E F x ≠ 0) →
  (∀ x > 4, g D E F x > 0.3) →
  (∀ ε > 0, ∃ M, ∀ x > M, |g D E F x - 1/3| < ε) →
  D + E + F = -42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_theorem_l527_52745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_initial_gift_l527_52794

def initial_amount (final_amount : ℚ) (value_multiplier : ℚ) (share_fraction : ℚ) : ℚ :=
  final_amount / (share_fraction * value_multiplier)

theorem blake_initial_gift :
  let final_amount : ℚ := 30000
  let value_multiplier : ℚ := 3
  let share_fraction : ℚ := 1/2
  initial_amount final_amount value_multiplier share_fraction = 20000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_initial_gift_l527_52794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_stake_ratio_l527_52768

/-- Represents a player in the card game -/
structure Player where
  id : ℕ
  stake : ℝ

/-- Represents the card game -/
structure CardGame where
  players : Fin 36 → Player
  totalPot : ℝ

/-- The probability of drawing the ace of diamonds -/
noncomputable def aceProbability : ℝ := 1 / 36

/-- The probability of not drawing the ace of diamonds -/
noncomputable def notAceProbability : ℝ := 35 / 36

/-- The expected value for a player in the game -/
noncomputable def expectedValue (game : CardGame) (player : Player) : ℝ :=
  (game.totalPot - player.stake) * aceProbability * (notAceProbability ^ (player.id - 1)) +
  (-player.stake) * (1 - aceProbability * (notAceProbability ^ (player.id - 1)))

/-- A game is fair if the expected value for all players is zero -/
def isFairGame (game : CardGame) : Prop :=
  ∀ k : Fin 36, expectedValue game (game.players k) = 0

theorem fair_game_stake_ratio (game : CardGame) (h : isFairGame game) :
  ∀ k : Fin 35, (game.players k.succ).stake = (game.players k).stake * (35 / 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_game_stake_ratio_l527_52768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_ratio_l527_52738

/-- A regular square pyramid with an inscribed cube -/
structure SquarePyramidWithCube where
  /-- Length of the base edge of the pyramid -/
  base_edge : ℝ
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the inscribed cube -/
  cube_side : ℝ
  /-- The height is twice the base edge length -/
  height_twice_base : height = 2 * base_edge
  /-- The cube side length is 2/3 of the base edge -/
  cube_side_relation : cube_side = (2/3) * base_edge

/-- Volume of a regular square pyramid -/
noncomputable def pyramid_volume (p : SquarePyramidWithCube) : ℝ :=
  (1/3) * p.base_edge^2 * p.height

/-- Volume of the inscribed cube -/
noncomputable def cube_volume (p : SquarePyramidWithCube) : ℝ :=
  p.cube_side^3

/-- The theorem stating the volume ratio -/
theorem inscribed_cube_volume_ratio (p : SquarePyramidWithCube) :
  cube_volume p / pyramid_volume p = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_ratio_l527_52738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4050_degrees_undefined_l527_52740

theorem tan_4050_degrees_undefined :
  ¬∃ (x : ℝ), Real.tan (4050 * π / 180) = x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4050_degrees_undefined_l527_52740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l527_52750

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_problem (principal interest time : ℝ) 
  (h_principal : principal = 800)
  (h_interest : interest = 200)
  (h_time : time = 4) :
  ∃ rate : ℝ, simple_interest principal rate time = interest ∧ rate = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l527_52750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l527_52799

theorem class_average_problem (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) 
  (high_score : ℕ) (class_average : ℚ) :
  total_students = 25 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 49.6 →
  (total_students - high_scorers - zero_scorers : ℚ) * 
    ((class_average * total_students - high_scorers * high_score) / 
     (total_students - high_scorers - zero_scorers)) = 45 * 
     (total_students - high_scorers - zero_scorers) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l527_52799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_three_tosses_l527_52719

-- Define a fair coin
noncomputable def fair_coin_prob : ℝ := 1 / 2

-- Define the number of tosses
def num_tosses : ℕ := 3

-- Define the number of desired heads
def num_heads : ℕ := 2

-- Binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability of getting exactly k successes in n trials
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem probability_two_heads_three_tosses :
  binomial_probability num_tosses num_heads fair_coin_prob = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_three_tosses_l527_52719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l527_52793

theorem negation_of_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l527_52793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_difference_l527_52774

def spring_mows : ℕ := 8
def summer_mows : ℕ := 5
def fall_mows : ℕ := 12

def seasons_mows : List ℕ := [spring_mows, summer_mows, fall_mows]

theorem lawn_mowing_difference :
  (List.maximum seasons_mows).getD 0 - (List.minimum seasons_mows).getD 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_difference_l527_52774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bee_problem_l527_52711

/-- Proof of flower and bee problem -/
theorem flower_bee_problem :
  let num_flowers : ℕ := 5
  let petals_per_flower : ℕ := 2
  let num_bees : ℕ := 3
  let wings_per_bee : ℕ := 4
  let leaves_per_flower : ℕ := 3
  let flowers_visited_per_bee : ℕ := 2

  let total_petals : ℕ := num_flowers * petals_per_flower
  let total_wings : ℕ := num_bees * wings_per_bee
  let total_leaves : ℕ := num_flowers * leaves_per_flower
  let total_visits : ℕ := num_bees * flowers_visited_per_bee

  let wings_minus_petals : ℤ := total_wings - total_petals
  let unvisited_leaves : ℕ := 0

  wings_minus_petals = 2 ∧ unvisited_leaves = 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bee_problem_l527_52711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_fraction_l527_52706

theorem remaining_money_fraction (total_money : ℝ) (total_items : ℝ) 
  (h1 : total_money > 0) (h2 : total_items > 0) : 
  (total_money - ((2 * ((1/3) * total_money)) + ((1/6) * total_money))) = (1/6) * total_money := by
  -- Define variables
  let cost_half_items := (1/3) * total_money
  let cost_all_items := 2 * cost_half_items
  let additional_expense := (1/6) * total_money
  let remaining_money := total_money - (cost_all_items + additional_expense)
  
  -- Proof steps
  sorry  -- We'll skip the actual proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_fraction_l527_52706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_no_axis_intersection_l527_52702

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := -5 / x

-- Theorem stating that the function cannot intersect with the coordinate axes
theorem inverse_proportion_no_axis_intersection :
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧ (∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_no_axis_intersection_l527_52702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MN_perp_OP_l527_52727

-- Define the circle Γ
variable (Γ : Set (ℝ × ℝ))

-- Define points A, B, C, D, O, P, M, N
variable (A B C D O P M N : ℝ × ℝ)

-- A and C are on Γ
axiom A_on_Γ : A ∈ Γ
axiom C_on_Γ : C ∈ Γ

-- O is on the perpendicular bisector of [AC]
axiom O_on_perp_bisector : ∃ (mid : ℝ × ℝ), mid = (A + C) / 2 ∧ ((O.1 - mid.1) * (C.1 - A.1) + (O.2 - mid.2) * (C.2 - A.2) = 0)

-- B and D are intersections of angle bisectors of [OA] and [OC] with Γ
axiom B_on_bisector : ∃ (t : ℝ), 0 < t ∧ B = (O.1 + t * (A.1 - O.1 + A.2 - O.2), O.2 + t * (A.2 - O.2 - A.1 + O.1))
axiom D_on_bisector : ∃ (s : ℝ), 0 < s ∧ D = (O.1 + s * (C.1 - O.1 + C.2 - O.2), O.2 + s * (C.2 - O.2 - C.1 + O.1))
axiom B_on_Γ : B ∈ Γ
axiom D_on_Γ : D ∈ Γ

-- A, B, C, D are in this order on Γ
axiom order_on_Γ : ∃ (θ₁ θ₂ θ₃ θ₄ : ℝ), 
  0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ < θ₃ ∧ θ₃ < θ₄ ∧ θ₄ < 2*Real.pi ∧
  A = (O.1 + Real.cos θ₁, O.2 + Real.sin θ₁) ∧
  B = (O.1 + Real.cos θ₂, O.2 + Real.sin θ₂) ∧
  C = (O.1 + Real.cos θ₃, O.2 + Real.sin θ₃) ∧
  D = (O.1 + Real.cos θ₄, O.2 + Real.sin θ₄)

-- P is the intersection of lines (AB) and (CD)
axiom P_intersection : ∃ (t s : ℝ), P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) ∧
                                    P = (C.1 + s * (D.1 - C.1), C.2 + s * (D.2 - C.2))

-- M is the midpoint of [AB]
axiom M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- N is the midpoint of [CD]
axiom N_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Theorem to prove
theorem MN_perp_OP : (N.1 - M.1) * (P.1 - O.1) + (N.2 - M.2) * (P.2 - O.2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_MN_perp_OP_l527_52727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_20_pentagons_l527_52783

/-- Represents a polygon --/
structure Polygon :=
  (sides : ℕ)

/-- Represents a cut operation on polygons --/
def cut (polygons : List Polygon) : List Polygon :=
  sorry

/-- The number of pentagons in a list of polygons --/
def count_pentagons (polygons : List Polygon) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required to obtain 20 pentagons --/
theorem min_cuts_for_20_pentagons :
  ∃ (n : ℕ),
    n = 38 ∧
    (∀ (m : ℕ),
      (∃ (cuts : List (List Polygon)),
        cuts.length = m ∧
        cuts.head! = [Polygon.mk 5] ∧
        (∀ (i : ℕ), i < m - 1 → cuts[i + 1]! = cut cuts[i]!) ∧
        count_pentagons (cuts.getLast (by sorry)) = 20) →
      m ≥ n) ∧
    (∃ (cuts : List (List Polygon)),
      cuts.length = n ∧
      cuts.head! = [Polygon.mk 5] ∧
      (∀ (i : ℕ), i < n - 1 → cuts[i + 1]! = cut cuts[i]!) ∧
      count_pentagons (cuts.getLast (by sorry)) = 20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_20_pentagons_l527_52783
