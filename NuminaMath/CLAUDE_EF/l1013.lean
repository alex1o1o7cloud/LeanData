import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_truncated_pyramid_l1013_101301

/-- A unit cube is a cube with side length 1 -/
def UnitCube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- A point on the opposite face of the unit cube from (0,0,0) -/
def OppositeFacePoint : (Fin 3 → ℝ) → Prop :=
  λ p ↦ p 0 = 1 ∧ p 1 = 1 ∧ 0 ≤ p 2 ∧ p 2 ≤ 1

/-- The midpoint of the opposite face -/
noncomputable def MidpointOppositeFace : Fin 3 → ℝ :=
  λ i => if i = 2 then 1/2 else 1

/-- The volume of the truncated pyramid formed by slicing the unit cube -/
noncomputable def VolumeTruncatedPyramid : ℝ :=
  5/6

theorem volume_truncated_pyramid :
  VolumeTruncatedPyramid = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_truncated_pyramid_l1013_101301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_thousand_gram_in_three_weighings_l1013_101382

/-- Represents a weight with a specific mass in grams -/
structure Weight where
  mass : ℕ

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftLighter : WeighingResult
  | RightLighter : WeighingResult

/-- Represents a weighing operation that compares two lists of weights -/
def weighing (left : List Weight) (right : List Weight) : WeighingResult :=
  sorry

/-- Represents the process of identifying the 1000g weight -/
def identifyThousandGram (weights : List Weight) : Option Weight :=
  sorry

/-- Theorem stating that it's possible to identify the 1000g weight in exactly 3 weighings -/
theorem identify_thousand_gram_in_three_weighings 
  (weights : List Weight)
  (h1 : weights.length = 5)
  (h2 : ∃ w, w ∈ weights ∧ w.mass = 1000)
  (h3 : ∀ w, w ∈ weights → w.mass = 1000 ∨ w.mass = 1001 ∨ w.mass = 1002 ∨ w.mass = 1004 ∨ w.mass = 1007)
  (h4 : ∀ w1 w2, w1 ∈ weights → w2 ∈ weights → w1 ≠ w2 → w1.mass ≠ w2.mass) :
  ∃ (process : List (List Weight × List Weight)), 
    process.length = 3 ∧ 
    (identifyThousandGram weights).isSome ∧
    (∀ (left right : List Weight), (left, right) ∈ process → weighing left right ≠ WeighingResult.Equal) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_thousand_gram_in_three_weighings_l1013_101382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_theorem_l1013_101379

/-- Represents a chemical mixture -/
structure ChemicalMixture where
  volume : ℝ
  percentA : ℝ
  percentB : ℝ
  percentC : ℝ
  sum_to_hundred : percentA + percentB + percentC = 100

/-- Calculates the combined percentages of two chemical mixtures -/
noncomputable def combineMixtures (m1 m2 : ChemicalMixture) : (ℝ × ℝ × ℝ) :=
  let totalVolume := m1.volume + m2.volume
  let totalA := (m1.percentA * m1.volume + m2.percentA * m2.volume) / totalVolume
  let totalB := (m1.percentB * m1.volume + m2.percentB * m2.volume) / totalVolume
  let totalC := (m1.percentC * m1.volume + m2.percentC * m2.volume) / totalVolume
  (totalA, totalB, totalC)

theorem chemical_mixture_theorem (m1 m2 : ChemicalMixture)
  (h1 : m1.volume = 40 ∧ m1.percentA = 50 ∧ m1.percentB = 30 ∧ m1.percentC = 20)
  (h2 : m2.volume = 60 ∧ m2.percentA = 40 ∧ m2.percentB = 10 ∧ m2.percentC = 50) :
  combineMixtures m1 m2 = (44, 18, 38) := by
  sorry

#check chemical_mixture_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_theorem_l1013_101379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1013_101302

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1013_101302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_width_calculation_l1013_101352

/-- The width of a track formed by two concentric circles -/
noncomputable def track_width (inner_radius outer_radius : ℝ) : ℝ := outer_radius - inner_radius

/-- The difference in circumferences of two circles -/
noncomputable def circumference_difference (r1 r2 : ℝ) : ℝ := 2 * Real.pi * (r1 - r2)

theorem track_width_calculation (inner_radius outer_radius : ℝ) :
  inner_radius = 20 →
  circumference_difference outer_radius inner_radius = 20 * Real.pi →
  track_width inner_radius outer_radius = 10 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_width_calculation_l1013_101352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1013_101336

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * x^2 - x - 1/4
  else Real.log x / Real.log a - 1

/-- The function f is decreasing for all x₁ ≠ x₂ -/
def is_decreasing (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0

/-- Main theorem: If f is decreasing, then a is in [1/4, 1/2] -/
theorem a_range (a : ℝ) (h : is_decreasing a) : a ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1013_101336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1013_101365

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Checks if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop := p.x = 0

/-- Checks if a point is on the line y = -1 -/
def isOnLineYNegOne (p : Point) : Prop := p.y = -1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The length of the major axis of the given ellipse is 4 -/
theorem ellipse_major_axis_length :
  ∀ (e : Ellipse),
    e.focus1 = Point.mk 3 (-3 + 2 * Real.sqrt 2) →
    e.focus2 = Point.mk 3 (-3 - 2 * Real.sqrt 2) →
    (∃ (p : Point), isOnYAxis p ∧ distance p e.focus1 = distance p e.focus2) →
    (∃ (q : Point), isOnLineYNegOne q ∧ distance q e.focus1 = distance q e.focus2) →
    ∃ (v1 v2 : Point), distance v1 v2 = 4 ∧
      distance v1 e.focus1 + distance v1 e.focus2 =
      distance v2 e.focus1 + distance v2 e.focus2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1013_101365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1013_101367

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (3 - (Real.sqrt 2 / 2) * t, Real.sqrt 5 + (Real.sqrt 2 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - Real.sqrt 5)^2 = 5

-- Define point P
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 5)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    (∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂) ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    (point_P.1 - A.1)^2 + (point_P.2 - A.2)^2 +
    (point_P.1 - B.1)^2 + (point_P.2 - B.2)^2 = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1013_101367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficients_bound_l1013_101356

theorem quadratic_coefficients_bound (A B : ℝ) (a b c d e f : ℝ)
  (h1 : 0 ≤ A ∧ A ≤ 1) (h2 : 0 ≤ B ∧ B ≤ 1)
  (h3 : ∀ x y : ℝ, a * x^2 + b * x * y + c * y^2 = (A * x + (1 - A) * y)^2)
  (h4 : ∀ x y : ℝ, (A * x + (1 - A) * y) * (B * x + (1 - B) * y) = d * x^2 + e * x * y + f * y^2) :
  (∃ t ∈ ({a, b, c} : Set ℝ), t ≥ 4/9) ∧ (∃ t ∈ ({d, e, f} : Set ℝ), t ≥ 4/9) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficients_bound_l1013_101356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_l1013_101344

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x - 1 else Real.sqrt x

-- State the theorem
theorem f_greater_than_one (x₀ : ℝ) :
  f x₀ > 1 ↔ x₀ < -1 ∨ x₀ > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_l1013_101344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_l1013_101309

/-- Given two vectors a and b in ℝ², prove that |a + 2b| = 2√3 -/
theorem magnitude_of_sum (a b : ℝ × ℝ) : 
  (a.1 = Real.sqrt 3 ∧ a.2 = 1) →  -- a = (√3, 1)
  ‖b‖ = 1 →  -- |b| = 1
  Real.cos (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖))) = 1/2 →  -- angle between a and b is 60°
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_l1013_101309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_naval_battle_minimum_shots_l1013_101304

/-- A type representing the possible ship shapes -/
inductive ShipShape
  | A  -- ⬜⬜⬜⬜
  | B  -- ⬜⬜
       -- ⬜⬜
  | C  -- ⬜⬜⬜
       -- ⬜
  | D  -- ⬜
       -- ⬜⬜⬜

/-- A position on the grid -/
structure Position where
  x : Fin 7
  y : Fin 7

/-- A ship placement on the grid -/
structure Ship where
  shape : ShipShape
  position : Position

/-- A shot on the grid -/
structure Shot where
  position : Position

/-- Check if a shot hits a ship -/
def hits (shot : Shot) (ship : Ship) : Prop := sorry

/-- The main theorem -/
theorem naval_battle_minimum_shots :
  ∀ (shots : Finset Shot),
    (∀ (ship : Ship), ∃ (shot : Shot), shot ∈ shots ∧ hits shot ship) →
    Finset.card shots ≥ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_naval_battle_minimum_shots_l1013_101304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_problem_l1013_101380

/-- The average weight problem -/
theorem average_weight_problem 
  (weight_A weight_B weight_C weight_D : ℝ)
  (h1 : (weight_A + weight_B + weight_C) / 3 = 60)
  (h2 : (weight_A + weight_B + weight_C + weight_D) / 4 = 65)
  (h3 : weight_A = 87) :
  (weight_B + weight_C + weight_D + (weight_D + 3)) / 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_problem_l1013_101380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nucleic_acid_testing_equation_l1013_101316

/-- Represents the rate at which Team A tests people per hour -/
def x : ℝ := sorry

/-- The difference in testing rate between Team A and Team B -/
def rate_difference : ℝ := 15

/-- The number of people Team A tests -/
def team_a_people : ℝ := 600

/-- The number of people Team B tests -/
def team_b_people : ℝ := 500

/-- The percentage reduction in time for Team A compared to Team B -/
def time_reduction : ℝ := 0.1

theorem nucleic_acid_testing_equation :
  team_a_people / x = (team_b_people / (x - rate_difference)) * (1 - time_reduction) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nucleic_acid_testing_equation_l1013_101316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_12_unoccupied_cells_l1013_101358

/-- Represents a 6x6 checkerboard -/
def Board := Fin 6 → Fin 6 → Bool

/-- Represents the position of a grasshopper on the board -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the state of the board with grasshoppers -/
def BoardState := Position → ℕ

/-- Initial state of the board with one grasshopper in each cell -/
def initialState : BoardState := fun _ => 1

/-- Checks if a position is within the board -/
def isValidPosition (p : Position) : Bool :=
  p.row < 6 ∧ p.col < 6

/-- Represents a diagonal jump over one cell -/
def diagonalJump (p : Position) : Position :=
  ⟨(p.row + 2) % 6, (p.col + 2) % 6⟩

/-- The state of the board after all grasshoppers have jumped -/
def finalState (initial : BoardState) : BoardState :=
  fun p => initial (diagonalJump p)

/-- Counts the number of unoccupied cells in a given board state -/
def countUnoccupiedCells (state : BoardState) : ℕ :=
  (List.range 6).foldl (fun acc row => 
    acc + ((List.range 6).foldl (fun inner_acc col => 
      inner_acc + (if state ⟨row, col⟩ = 0 then 1 else 0)
    ) 0)
  ) 0

/-- Theorem stating that there will be at least 12 unoccupied cells after the jump -/
theorem at_least_12_unoccupied_cells :
  countUnoccupiedCells (finalState initialState) ≥ 12 := by
  sorry

#eval countUnoccupiedCells (finalState initialState)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_12_unoccupied_cells_l1013_101358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_conditions_l1013_101340

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def satisfies_conditions (s : Set ℕ) : Prop :=
  (∀ n, n ∈ s → n > 0) ∧
  (∀ n, n ∈ s → is_two_digit (n + 50)) ∧
  (∀ n, n ∈ s → is_two_digit (n - 32) ∧ n - 32 > 0) ∧
  (∀ a b, a ∈ s → b ∈ s → a ≠ b → Nat.gcd a b = 1)

theorem unique_set_satisfying_conditions :
  ∃! s : Set ℕ, satisfies_conditions s ∧ 
    ∃ k, s = {k, k+1, k+2, k+3, k+4, k+5, k+6, k+7} ∧
    k = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_conditions_l1013_101340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1013_101308

-- Define the equation of the region
def region_equation (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 + 6*x - 8*y + 10*z = -19

-- Define the volume of the region
noncomputable def region_volume : ℝ := (4/3) * Real.pi * 19^(3/2)

-- Theorem statement
theorem volume_of_region :
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ),
    (∀ (x y z : ℝ), region_equation x y z ↔ 
      (x - center.1)^2 + (y - center.2.1)^2 + (y - center.2.2)^2 = radius^2) ∧
    region_volume = (4/3) * Real.pi * radius^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1013_101308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_row_first_five_l1013_101317

/-- Represents the digits that can be used to fill the grid -/
inductive Digit
  | Two
  | Zero
  | One
  | Five
  | Empty

/-- A 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Digit

/-- Check if a grid satisfies the conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 5, ∀ d : Digit, ∃! j : Fin 5, g i j = d) ∧ 
  (∀ j : Fin 5, ∀ d : Digit, ∃! i : Fin 5, g i j = d) ∧
  (∀ i j : Fin 5, g i (j+1) ≠ g (i+1) j ∧ g i j ≠ g (i+1) (j+1))

/-- The theorem to be proved -/
theorem fifth_row_first_five (g : Grid) (h : is_valid_grid g) : 
  (g 4 0 = Digit.Two) ∧ 
  (g 4 1 = Digit.Zero) ∧ 
  (g 4 2 = Digit.One) ∧ 
  (g 4 3 = Digit.Five) ∧ 
  (g 4 4 = Digit.Empty) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_row_first_five_l1013_101317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_cistern_fill_time_l1013_101387

/-- Represents the time it takes to fill a portion of the cistern -/
def fill_time (portion : ℝ) : ℝ := sorry

/-- The given time to fill half of the cistern -/
def given_half_fill_time : ℝ := sorry

/-- Theorem stating that the time to fill half the cistern is equal to the given time -/
theorem half_cistern_fill_time : fill_time (1/2) = given_half_fill_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_cistern_fill_time_l1013_101387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_opposite_signs_l1013_101398

theorem roots_opposite_signs (m n k : ℝ) (hm : m ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ m * x^2 + n * x + k
  (g k * g (1/m) < 0) →
  ∃ r₁ r₂ : ℝ, g r₁ = 0 ∧ g r₂ = 0 ∧ r₁ * r₂ < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_opposite_signs_l1013_101398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sin_l1013_101399

open Set
open Function
open Real

-- Define the function f with domain (0,1]
def f : {x : ℝ | 0 < x ∧ x ≤ 1} → ℝ := sorry

-- Define the set of real numbers x such that 0 < sin x ≤ 1
def sin_domain : Set ℝ := {x : ℝ | 0 < sin x ∧ sin x ≤ 1}

-- Define the set of real numbers x such that x is in (2kπ, 2kπ + π) for some integer k
def periodic_domain : Set ℝ := {x : ℝ | ∃ k : ℤ, 2 * k * π < x ∧ x < 2 * k * π + π}

-- Theorem statement
theorem domain_of_f_sin (x : ℝ) : 
  x ∈ periodic_domain ↔ x ∈ sin_domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sin_l1013_101399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spelling_bee_contestants_l1013_101327

theorem spelling_bee_contestants (initial_students : ℕ) : initial_students = 600 :=
  let after_first_round := (2 : ℚ) / 5 * initial_students
  let after_second_round := (1 : ℚ) / 4 * after_first_round
  let after_third_round := (1 : ℚ) / 4 * after_second_round
  have h_final : after_third_round = 15 := by sorry
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spelling_bee_contestants_l1013_101327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_when_a_4_parallel_for_other_a_a_4_sufficient_not_necessary_l1013_101315

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (2+a)x+3ay+1=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -(2+a)/(3*a)

/-- The slope of the line (a-2)x+ay-3=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -(a-2)/a

/-- The lines are parallel when a=4 -/
theorem parallel_when_a_4 : are_parallel (slope1 4) (slope2 4) := by sorry

/-- There exists a value of a ≠ 4 for which the lines are parallel -/
theorem parallel_for_other_a : ∃ a : ℝ, a ≠ 4 ∧ are_parallel (slope1 a) (slope2 a) := by sorry

/-- a=4 is a sufficient but not necessary condition for the lines to be parallel -/
theorem a_4_sufficient_not_necessary : 
  (∃ a : ℝ, a ≠ 4 ∧ are_parallel (slope1 a) (slope2 a)) ∧ 
  are_parallel (slope1 4) (slope2 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_when_a_4_parallel_for_other_a_a_4_sufficient_not_necessary_l1013_101315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_reversible_number_l1013_101329

def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

def reverse_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).reverse.foldl (fun acc d => acc * 10 + d) 0

theorem unique_reversible_number : ∃! n : Nat, is_five_digit n ∧ 9 * n = reverse_digits n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_reversible_number_l1013_101329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l1013_101389

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 - x + 2

-- Theorem statement
theorem f_has_one_zero : ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l1013_101389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1013_101357

-- Define auxiliary propositions used in the statement
def IsHyperbola (x y : ℝ) : Prop := sorry
def FociOnYAxis (x y : ℝ) : Prop := sorry

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (2 - m) + y^2 / (m - 1) = 1 → IsHyperbola x y ∧ FociOnYAxis x y

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0

-- Define the theorem
theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(q m)) : m ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1013_101357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l1013_101347

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  (π * (2 * r)^2 - π * r^2) / (π * r^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_increase_l1013_101347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_correct_l1013_101355

/-- The area of a quadrilateral with vertices at (0, 0), (0, 2), (3, 2), and (5, 5) -/
noncomputable def quadrilateral_area : ℝ := (6 + 3 * Real.sqrt 13) / 2

/-- Function to calculate the area of a quadrilateral given its vertices -/
noncomputable def area_of_quadrilateral (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the area of the quadrilateral is correct -/
theorem quadrilateral_area_correct :
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 2)
  let v3 : ℝ × ℝ := (3, 2)
  let v4 : ℝ × ℝ := (5, 5)
  area_of_quadrilateral v1 v2 v3 v4 = quadrilateral_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_correct_l1013_101355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l1013_101313

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

-- State the theorem
theorem f_max_min_values :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y ≤ 2 * π →
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b ≤ 2 * π → f a b ≤ f x y) →
  f x y = 3 * Real.sqrt 3 / 2 ∧
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b ≤ 2 * π → f a b ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l1013_101313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1013_101384

noncomputable def h (x : ℝ) : ℝ := (x^3 + 11*x - 2) / ((x^2 - 9) + |x + 1|)

def domain_h : Set ℝ := {x | x < -3 ∨ (-3 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) ∨ 5 < x}

theorem h_domain : {x : ℝ | h x ≠ 0} = domain_h := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l1013_101384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_l1013_101320

/-- Represents a segment of the journey -/
structure Segment where
  distance : ℚ
  speed : ℚ

/-- Calculates the time taken for a segment -/
def time (s : Segment) : ℚ := s.distance / s.speed

/-- The journey consisting of 6 segments -/
def journey : List Segment := [
  { distance := 40, speed := 20 },
  { distance := 10, speed := 15 },
  { distance := 5, speed := 10 },
  { distance := 180, speed := 60 },
  { distance := 25, speed := 30 },
  { distance := 20, speed := 45 }
]

/-- The total distance of the journey -/
def totalDistance : ℚ := 280

/-- Theorem: The average speed of the entire trip is approximately 37.62 mph -/
theorem average_speed_approx : 
  let totalTime := (journey.map time).sum
  let avgSpeed := totalDistance / totalTime
  ∀ ε > 0, |avgSpeed - 37.62| < ε := by
  sorry

#eval (journey.map time).sum
#eval totalDistance / (journey.map time).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_l1013_101320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_percent_daisies_l1013_101325

/-- Represents a flower bouquet with specific proportions of flowers. -/
structure FlowerBouquet where
  total : ℕ
  red_ratio : ℚ
  pink_ratio : ℚ
  yellow_ratio : ℚ
  pink_roses_ratio : ℚ
  red_carnations_ratio : ℚ

/-- The conditions of the flower bouquet problem. -/
def problem_bouquet : FlowerBouquet where
  total := 100  -- We use 100 for easier percentage calculations
  red_ratio := 1/2
  pink_ratio := 1/5
  yellow_ratio := 3/10
  pink_roses_ratio := 1/4
  red_carnations_ratio := 1/2

/-- Theorem stating that 30% of the flowers in the problem bouquet are daisies. -/
theorem thirty_percent_daisies (b : FlowerBouquet) 
  (h1 : b.red_ratio = 1/2)
  (h2 : b.pink_ratio = 1/5)
  (h3 : b.yellow_ratio = 1 - (b.red_ratio + b.pink_ratio))
  (h4 : b.pink_roses_ratio = 1/4)
  (h5 : b.red_carnations_ratio = 1/2) :
  b.yellow_ratio = 3/10 := by
  sorry

#eval (problem_bouquet.yellow_ratio : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_percent_daisies_l1013_101325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_5_2_hours_l1013_101385

/-- Calculates the total time for a journey with two segments at different speeds -/
noncomputable def total_journey_time (total_distance : ℝ) (distance_at_speed1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  let time_at_speed1 := distance_at_speed1 / speed1
  let distance_at_speed2 := total_distance - distance_at_speed1
  let time_at_speed2 := distance_at_speed2 / speed2
  time_at_speed1 + time_at_speed2

/-- Proves that the total journey time for the given conditions is 5.2 hours -/
theorem journey_time_is_5_2_hours :
  total_journey_time 250 124 40 60 = 5.2 := by
  -- Unfold the definition of total_journey_time
  unfold total_journey_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_5_2_hours_l1013_101385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_m_l1013_101396

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns the slope of a line in the form ax + by + c = 0 -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Given two points (x₁, y₁) and (x₂, y₂), calculates the slope of the line passing through them -/
noncomputable def pointSlope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

theorem parallel_lines_equal_m (m : ℝ) : 
  (pointSlope (-2) m m 10 = Line.slope ⟨2, -1, -1⟩) → m = 2 := by
  sorry

#check parallel_lines_equal_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_m_l1013_101396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1013_101342

-- Define the function f(x) = 2e^x
noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x

-- State the theorem
theorem tangent_line_at_zero (x y : ℝ) :
  (deriv f 0) * x - y + f 0 = 0 ↔ 2 * x - y + 2 = 0 := by
  sorry

-- Note: deriv f represents the derivative of f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1013_101342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_distances_l1013_101321

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 9) + (P.2^2 / 4) = 1

-- Define points A and B
noncomputable def A : ℝ × ℝ := (Real.sqrt 5, 0)
def B : ℝ × ℝ := (7, 5)

-- Define distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem ellipse_sum_distances (P : ℝ × ℝ) (h : is_on_ellipse P) :
  distance A P + distance P B = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_distances_l1013_101321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_left_of_kolya_l1013_101311

/-- Represents a person in the line -/
inductive Person : Type where
  | kolya : Person
  | sasha : Person
  | other : ℕ → Person

/-- The number of people to the left of a given person -/
def left_of : Person → ℕ := sorry

/-- The number of people to the right of a given person -/
def right_of : Person → ℕ := sorry

/-- The total number of people in the line -/
def total_people : ℕ := sorry

theorem people_left_of_kolya
  (h1 : right_of Person.kolya = 12)
  (h2 : left_of Person.sasha = 20)
  (h3 : right_of Person.sasha = 8)
  (h4 : total_people = left_of Person.sasha + right_of Person.sasha + 1)
  (h5 : total_people = left_of Person.kolya + right_of Person.kolya + 1) :
  left_of Person.kolya = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_left_of_kolya_l1013_101311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_walking_speed_l1013_101374

theorem initial_walking_speed 
  (distance : ℝ) 
  (v : ℝ) 
  (miss_time : ℝ) 
  (early_time : ℝ) 
  (faster_speed : ℝ) 
  (h1 : distance = 13.5) 
  (h2 : miss_time = 12/60) 
  (h3 : early_time = 15/60) 
  (h4 : faster_speed = 6) 
  (h5 : distance/v - distance/faster_speed = miss_time + early_time) :
  v = 5 := by
  sorry

#check initial_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_walking_speed_l1013_101374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_40_not_in_A_l1013_101334

noncomputable def A : Set ℝ := {x | x ≤ 1/2}

noncomputable def m : ℝ := Real.sin (40 * Real.pi / 180)

theorem sin_40_not_in_A : m ∉ A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_40_not_in_A_l1013_101334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_food_cost_is_three_l1013_101376

/-- The cost per ounce of special fish food -/
def special_food_cost_per_ounce (total_goldfish : ℕ) (food_per_fish : ℚ) 
  (special_food_percentage : ℚ) (total_special_food_cost : ℚ) : ℚ :=
  let special_fish_count := (special_food_percentage * total_goldfish : ℚ).floor
  let special_food_ounces := special_fish_count * food_per_fish
  total_special_food_cost / special_food_ounces

/-- Theorem stating the cost per ounce of special fish food is $3 -/
theorem special_food_cost_is_three :
  special_food_cost_per_ounce 50 (3/2) (1/5) 45 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_food_cost_is_three_l1013_101376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_l1013_101318

theorem sin_2alpha (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : Real.cos (α - β) = 12 / 13) 
  (h5 : Real.sin (α + β) = -3 / 5) : 
  Real.sin (2 * α) = -56 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_l1013_101318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1013_101381

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)

theorem triangle_theorem (t : Triangle) 
  (h : triangle_condition t) : 
  (t.A = 2 * t.B → t.C = 5 * Real.pi / 8) ∧ 
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1013_101381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_nonneg_f_sum_equals_six_f_neg_l1013_101346

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 1 else 1 - 2^(-x)

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem f_nonneg (x : ℝ) (h : x ≥ 0) : f x = 2^x - 1 := by sorry

theorem f_sum_equals_six : f 3 + f (-1) = 6 := by sorry

theorem f_neg (x : ℝ) (h : x < 0) : f x = 1 - 2^(-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_nonneg_f_sum_equals_six_f_neg_l1013_101346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_49_65_l1013_101314

/-- The price of a candy bar in dollars -/
noncomputable def candy_bar_price : ℝ := 2 * 3

/-- The price of cotton candy in dollars -/
noncomputable def cotton_candy_price : ℝ := (4 * candy_bar_price) / 2

/-- The discounted price of a candy bar in dollars -/
noncomputable def discounted_candy_bar_price : ℝ := candy_bar_price * (1 - 0.1)

/-- The discounted price of a caramel in dollars -/
noncomputable def discounted_caramel_price : ℝ := 3 * (1 - 0.15)

/-- The discounted price of cotton candy in dollars -/
noncomputable def discounted_cotton_candy_price : ℝ := cotton_candy_price * (1 - 0.2)

/-- The total cost of 6 candy bars, 3 caramels, and 1 cotton candy after discounts -/
noncomputable def total_cost : ℝ := 6 * discounted_candy_bar_price + 3 * discounted_caramel_price + discounted_cotton_candy_price

theorem total_cost_is_49_65 : total_cost = 49.65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_49_65_l1013_101314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1013_101394

/-- Represents a train with its length and time to cross a telegraph post -/
structure Train where
  length : ℝ
  crossTime : ℝ

/-- Calculates the speed of a train -/
noncomputable def trainSpeed (t : Train) : ℝ := t.length / t.crossTime

/-- Calculates the time for two trains to cross each other when traveling in opposite directions -/
noncomputable def timeToCross (t1 t2 : Train) : ℝ :=
  (t1.length + t2.length) / (trainSpeed t1 + trainSpeed t2)

/-- Theorem stating the time for two specific trains to cross each other -/
theorem trains_crossing_time :
  let train1 : Train := { length := 240, crossTime := 3 }
  let train2 : Train := { length := 300, crossTime := 10 }
  |timeToCross train1 train2 - 4.91| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1013_101394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forces_right_angle_arithmetic_progression_l1013_101364

-- Define the magnitudes of the two forces and their resultant
noncomputable def force1 : ℝ → ℝ := λ a ↦ a
noncomputable def force2 : ℝ → ℝ := λ a ↦ a + (a / 3)
noncomputable def resultant : ℝ → ℝ := λ a ↦ a + 2 * (a / 3)

-- Theorem statement
theorem forces_right_angle_arithmetic_progression (a : ℝ) (h : a > 0) :
  (force1 a)^2 + (force2 a)^2 = (resultant a)^2 ∧
  force2 a - force1 a = resultant a - force2 a →
  (force1 a : ℝ) / (force1 a : ℝ) = 3 / 3 ∧
  (force2 a : ℝ) / (force1 a : ℝ) = 4 / 3 ∧
  (resultant a : ℝ) / (force1 a : ℝ) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forces_right_angle_arithmetic_progression_l1013_101364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_max_convergence_parameter_l1013_101371

noncomputable def sequence_x (a : ℝ) : ℕ → ℝ
| 0 => 1
| 1 => 0
| (n + 2) => (sequence_x a n)^2 / 4 + (sequence_x a (n + 1))^2 / 4 + a

theorem sequence_convergence :
  ∃ (L : ℝ), Filter.Tendsto (sequence_x 0) Filter.atTop (nhds L) := by sorry

theorem max_convergence_parameter :
  ∀ (a : ℝ), (∃ (L : ℝ), Filter.Tendsto (sequence_x a) Filter.atTop (nhds L)) ↔ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_max_convergence_parameter_l1013_101371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l1013_101305

/-- Represents the time difference between two runners in a race --/
noncomputable def timeDifference (raceDistance : ℝ) (distanceAhead : ℝ) (timeA : ℝ) : ℝ :=
  let distanceB := raceDistance - distanceAhead
  let timeB := raceDistance * timeA / distanceB
  timeB - timeA

/-- Proves that the time difference is approximately 10 seconds --/
theorem race_time_difference :
  let raceDistance : ℝ := 1000
  let distanceAhead : ℝ := 16
  let timeA : ℝ := 615
  abs (timeDifference raceDistance distanceAhead timeA - 10) < 0.5 := by
  sorry

-- Use #eval only for computable functions
def approxTimeDifference : Nat := 10

#eval approxTimeDifference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l1013_101305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_in_specific_cylinder_l1013_101383

/-- Represents a cylinder with two tangent balls and an intersecting plane. -/
structure CylinderWithBalls where
  height : ℝ
  baseRadius : ℝ
  ballRadius : ℝ
  planeTangentToBalls : Bool

/-- The area of the ellipse formed by the intersection of the plane and cylinder edge. -/
noncomputable def ellipseArea (c : CylinderWithBalls) : ℝ :=
  2 * Real.pi * c.baseRadius * (c.height - 2 * c.ballRadius)

/-- Theorem stating the area of the ellipse in the given problem. -/
theorem ellipse_area_in_specific_cylinder :
  let c := CylinderWithBalls.mk 10 1 1 true
  ellipseArea c = 8 * Real.pi :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_in_specific_cylinder_l1013_101383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1013_101353

-- Define the expression
noncomputable def f (a : ℝ) : ℝ :=
  (a^3 - 3*a^2 + 4 + (a^2 - 4)*Real.sqrt (a^2 - 1)) /
  (a^3 + 3*a^2 - 4 + (a^2 - 4)*Real.sqrt (a^2 - 1))

-- State the theorem
theorem expression_simplification (a : ℝ) 
  (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) :
  f a = ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1013_101353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_increase_l1013_101322

theorem average_weight_increase (n : ℕ) (old_weight new_weight : ℝ) :
  n = 10 ∧ old_weight = 65 ∧ new_weight = 128 →
  (λ A : ℝ ↦ (n * A - old_weight + new_weight) / n - A) = λ _ : ℝ ↦ 6.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_increase_l1013_101322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_collection_l1013_101335

noncomputable def yoongi_collection : ℝ := 4
noncomputable def jungkook_collection : ℝ := 6 / 3
noncomputable def yuna_collection : ℝ := 5

theorem biggest_collection :
  max yoongi_collection (max jungkook_collection yuna_collection) = yuna_collection :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biggest_collection_l1013_101335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1013_101310

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 - Real.sqrt 3 / 2 * t, 1 + 1 / 2 * t)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the point P
def point_P : ℝ × ℝ := (-1, 1)

-- Define the slope angle of line l
noncomputable def slope_angle : ℝ := 5 * Real.pi / 6

-- Define the vector e
noncomputable def vector_e : ℝ × ℝ := (-Real.sqrt 3 / 2, 1 / 2)

-- State the theorem
theorem intersection_dot_product :
  ∃ (t₁ t₂ : ℝ),
    curve_C (line_l t₁).1 (line_l t₁).2 ∧
    curve_C (line_l t₂).1 (line_l t₂).2 ∧
    t₁ ≠ t₂ ∧
    (t₁ * vector_e.1 * t₂ * vector_e.1 + t₁ * vector_e.2 * t₂ * vector_e.2) = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1013_101310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tails_after_flipping_l1013_101319

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a circle of coins -/
def CoinCircle (n : ℕ) := Fin (2*n + 1) → CoinState

/-- Flips a coin state -/
def flipCoin (s : CoinState) : CoinState :=
  match s with
  | CoinState.Heads => CoinState.Tails
  | CoinState.Tails => CoinState.Heads

/-- Performs the flipping sequence on the coin circle -/
def flipSequence (n : ℕ) (c : CoinCircle n) : CoinCircle n :=
  sorry

/-- Counts the number of tails in the coin circle -/
def countTails (n : ℕ) (c : CoinCircle n) : ℕ :=
  sorry

/-- The main theorem -/
theorem one_tails_after_flipping (n : ℕ) :
  ∃ (c : CoinCircle n), (∀ i, c i = CoinState.Heads) →
    countTails n (flipSequence n c) = 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tails_after_flipping_l1013_101319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_2_l1013_101391

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x^2 - 1)

-- State the theorem
theorem f_x_plus_2 (x : ℝ) (h : x^2 ≠ 1) : 
  f (x + 2) = (x^2 + 4*x + 5) / (x^2 + 4*x + 3) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_2_l1013_101391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_rational_inequalities_l1013_101330

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set of the quadratic inequality
noncomputable def solution_set (a b : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ b}

-- Define the rational function
noncomputable def g (a b x : ℝ) : ℝ := (x + 3) / (a * x - b)

theorem quadratic_and_rational_inequalities 
  (a b : ℝ) 
  (h1 : ∀ x, x ∈ solution_set a b ↔ f a x ≤ 0) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ x, g a b x > 0 ↔ (x > 2 ∨ x < -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_rational_inequalities_l1013_101330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l1013_101375

/-- The initial investment that grows to a given amount after a certain period at a given interest rate --/
noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / (1 + interest_rate) ^ years

/-- The problem statement --/
theorem investment_problem : 
  let final_amount : ℝ := 439.23
  let years : ℕ := 3
  let interest_rate : ℝ := 0.08
  let result := initial_investment final_amount years interest_rate
  ∃ ε > 0, |result - 348.68| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l1013_101375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_and_5_l1013_101360

def digits : List Nat := [5, 6, 7]

def is_multiple_of_3_and_5 (n : Nat) : Bool :=
  n % 3 = 0 && n % 5 = 0

def three_digit_numbers (ds : List Nat) : List Nat :=
  ds.foldr (fun d1 acc =>
    ds.foldr (fun d2 acc =>
      ds.foldr (fun d3 acc =>
        if d1 ≠ d2 && d2 ≠ d3 && d1 ≠ d3 && d1 ≠ 0
        then (100 * d1 + 10 * d2 + d3) :: acc
        else acc
      ) acc
    ) acc
  ) []

theorem probability_multiple_3_and_5 :
  let numbers := three_digit_numbers digits
  let valid_numbers := numbers.filter is_multiple_of_3_and_5
  (valid_numbers.length : Rat) / (numbers.length : Rat) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_and_5_l1013_101360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l1013_101307

noncomputable def ellipse_focus (major_axis_start major_axis_end minor_axis_start minor_axis_end : ℝ × ℝ) : ℝ × ℝ :=
  let center_x := (major_axis_start.1 + major_axis_end.1) / 2
  let center_y := major_axis_start.2  -- Assuming y-coordinates are the same for major axis endpoints
  let semi_major_axis := (major_axis_end.1 - major_axis_start.1) / 2
  let semi_minor_axis := (minor_axis_start.2 - minor_axis_end.2) / 2
  let c := Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)
  (center_x + c, center_y)

theorem ellipse_focus_coordinates :
  ellipse_focus (0, -1) (8, -1) (3, 2) (3, -5) = (4 + Real.sqrt 3.75, -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l1013_101307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1013_101323

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℕ => ⌈(n : ℚ) / 101⌉ + 1 > (n : ℚ) / 100) (Finset.range 15050)).card = 15049 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1013_101323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1013_101392

/-- Calculates the length of a bridge given train parameters --/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Theorem stating the length of the bridge given specific parameters --/
theorem bridge_length_calculation :
  let train_length := (360 : ℝ)
  let train_speed_kmh := (30 : ℝ)
  let time_to_pass := (60 : ℝ)
  ∃ ε > 0, |bridge_length train_length train_speed_kmh time_to_pass - 139.8| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval bridge_length 360 30 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1013_101392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_destroy_tank_l1013_101337

/-- Represents a grid cell --/
structure Cell where
  x : Fin 41
  y : Fin 41
deriving DecidableEq

/-- Represents the state of the tank --/
inductive TankState
  | Unhit
  | HitOnce
  | Destroyed

/-- Represents the game state --/
structure GameState where
  tankPosition : Cell
  tankState : TankState

/-- Represents a strategy for shooting --/
def Strategy := List Cell

/-- Checks if two cells are adjacent --/
def areAdjacent (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ (c1.y = c2.y - 1 ∨ c1.y = c2.y + 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x - 1 ∨ c1.x = c2.x + 1))

/-- Applies a shot to the game state --/
def applyShot (state : GameState) (target : Cell) : GameState :=
  if state.tankPosition = target then
    match state.tankState with
    | TankState.Unhit => { tankPosition := sorry, tankState := TankState.HitOnce }
    | TankState.HitOnce => { tankPosition := state.tankPosition, tankState := TankState.Destroyed }
    | TankState.Destroyed => state
  else
    state

/-- Checks if a strategy guarantees tank destruction --/
def guaranteesDestruction (s : Strategy) : Prop :=
  ∀ initialState : GameState,
    ∃ finalState : GameState,
      finalState.tankState = TankState.Destroyed ∧
      finalState = s.foldl applyShot initialState

/-- The main theorem --/
theorem min_shots_to_destroy_tank :
  ∃ (s : Strategy),
    s.length = 2521 ∧
    guaranteesDestruction s ∧
    ∀ (s' : Strategy),
      guaranteesDestruction s' → s'.length ≥ 2521 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_destroy_tank_l1013_101337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1013_101339

theorem largest_expression (x : ℝ) (h : x = (1 : ℝ) / 10^2024) :
  (5/x > 5+x) ∧ (5/x > 5-x) ∧ (5/x > 5*x) ∧ (5/x > x/5) :=
by
  -- Substitute the value of x
  rw [h]
  -- Split the conjunction into separate goals
  apply And.intro
  · -- Prove 5/x > 5+x
    sorry
  apply And.intro
  · -- Prove 5/x > 5-x
    sorry
  apply And.intro
  · -- Prove 5/x > 5*x
    sorry
  · -- Prove 5/x > x/5
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1013_101339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inverse_sqrt_l1013_101354

theorem sqrt_sum_inverse_sqrt (a : ℝ) (h1 : a > 0) (h2 : a + a⁻¹ = 3) :
  a^(1/2 : ℝ) + a^(-(1/2 : ℝ)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inverse_sqrt_l1013_101354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_is_15cm_l1013_101370

/-- Represents the dimensions of a rectangular vessel base -/
structure VesselBase where
  length : ℝ
  breadth : ℝ

/-- Represents a cubical box -/
structure CubicalBox where
  edge : ℝ

/-- Calculates the volume of a cubical box -/
noncomputable def boxVolume (box : CubicalBox) : ℝ := box.edge ^ 3

/-- Calculates the base area of a vessel -/
noncomputable def vesselBaseArea (base : VesselBase) : ℝ := base.length * base.breadth

/-- Calculates the rise in water level when a box is immersed in a vessel -/
noncomputable def waterRise (base : VesselBase) (box : CubicalBox) : ℝ :=
  boxVolume box / vesselBaseArea base

/-- Theorem stating that for the given dimensions, the water rise is 15 cm -/
theorem water_rise_is_15cm (base : VesselBase) (box : CubicalBox)
    (h1 : base.length = 60)
    (h2 : base.breadth = 30)
    (h3 : box.edge = 30) :
    waterRise base box = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_is_15cm_l1013_101370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1013_101350

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 4)

-- State the theorem
theorem function_properties :
  ∃ (min_value : ℝ) (min_point : ℝ),
    (∀ x > 4, f x ≥ min_value) ∧
    (f min_point = min_value) ∧
    (min_value = 10) ∧
    (min_point = 7) ∧
    (∀ M : ℝ, ∃ x > 4, f x > M) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1013_101350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l1013_101303

/-- Represents different types of measurements --/
inductive MeasurementType
  | Speed
  | Length
  | Weight

/-- Represents different units of measurement --/
inductive MeasurementUnit
  | Kilometer
  | Millimeter
  | Ton
  | Kilogram

/-- Determines if a unit is appropriate for a given measurement type and magnitude --/
def is_appropriate_unit (t : MeasurementType) (magnitude : ℕ) (u : MeasurementUnit) : Prop :=
  match t, u with
  | MeasurementType.Speed, MeasurementUnit.Kilometer => magnitude = 80
  | MeasurementType.Length, MeasurementUnit.Millimeter => magnitude = 7
  | MeasurementType.Weight, MeasurementUnit.Ton => magnitude = 4
  | MeasurementType.Weight, MeasurementUnit.Kilogram => magnitude = 35
  | _, _ => False

/-- Theorem stating that the given units are appropriate for the given measurements --/
theorem appropriate_units :
  is_appropriate_unit MeasurementType.Speed 80 MeasurementUnit.Kilometer ∧
  is_appropriate_unit MeasurementType.Length 7 MeasurementUnit.Millimeter ∧
  is_appropriate_unit MeasurementType.Weight 4 MeasurementUnit.Ton ∧
  is_appropriate_unit MeasurementType.Weight 35 MeasurementUnit.Kilogram :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l1013_101303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_moon_distance_in_hundred_million_l1013_101372

/-- The average distance between the Earth and the Moon in meters -/
noncomputable def earth_moon_distance : ℝ := 384400000

/-- One hundred million (亿) in numeric form -/
noncomputable def hundred_million : ℝ := 100000000

/-- Conversion of earth_moon_distance to hundred million meters -/
noncomputable def distance_in_hundred_million : ℝ := earth_moon_distance / hundred_million

theorem earth_moon_distance_in_hundred_million :
  distance_in_hundred_million = 3.844 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_moon_distance_in_hundred_million_l1013_101372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1013_101361

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x - 1 + x^2 / 2

-- State the theorem
theorem f_monotonicity :
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x > f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1013_101361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_is_500_l1013_101369

/-- Represents a company with factory and office workers -/
structure Company where
  factory_workers : ℕ
  office_workers : ℕ
  factory_payroll : ℚ
  office_payroll : ℚ

/-- Calculates the average salary for a given number of workers and total payroll -/
def average_salary (workers : ℕ) (payroll : ℚ) : ℚ :=
  payroll / workers

/-- Theorem: The difference in average salaries between office and factory workers is $500 -/
theorem salary_difference_is_500 (j : Company)
  (h1 : j.factory_workers = 15)
  (h2 : j.office_workers = 30)
  (h3 : j.factory_payroll = 30000)
  (h4 : j.office_payroll = 75000) :
  average_salary j.office_workers j.office_payroll - average_salary j.factory_workers j.factory_payroll = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_is_500_l1013_101369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_ratio_l1013_101363

def lemonade_problem (total_cups : ℕ) (construction_crew_fraction : ℚ) 
  (kids_bikes_cups : ℕ) (hazel_cups : ℕ) : Prop :=
  let construction_crew_cups := (total_cups : ℚ) * construction_crew_fraction
  let friends_cups := total_cups - (construction_crew_cups.floor + kids_bikes_cups + hazel_cups)
  (friends_cups : ℚ) / kids_bikes_cups = 1 / 2

theorem lemonade_ratio : 
  lemonade_problem 56 (1/2) 18 1 := by
  sorry

#eval (56 : ℚ) * (1/2)
#eval ((56 : ℚ) * (1/2)).floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_ratio_l1013_101363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1013_101386

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Coordinates of point P -/
def P : ℝ × ℝ := (-3, 2)

/-- Coordinates of point Q -/
def Q : ℝ × ℝ := (1, 7)

/-- Coordinates of point R -/
def R : ℝ × ℝ := (4, -1)

/-- The area of triangle PQR is 23.5 square units -/
theorem triangle_PQR_area :
  triangleArea P.1 P.2 Q.1 Q.2 R.1 R.2 = 23.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1013_101386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_zero_l1013_101341

/-- Represents the score distribution of a quiz --/
structure ScoreDistribution where
  score_60 : ℝ
  score_75 : ℝ
  score_80 : ℝ
  score_88 : ℝ
  score_92 : ℝ
  sum_to_100 : score_60 + score_75 + score_80 + score_88 + score_92 = 100

/-- Calculates the mean score given a score distribution --/
noncomputable def mean_score (dist : ScoreDistribution) : ℝ :=
  (60 * dist.score_60 + 75 * dist.score_75 + 80 * dist.score_80 + 
   88 * dist.score_88 + 92 * dist.score_92) / 100

/-- Calculates the median score given a score distribution --/
noncomputable def median_score (dist : ScoreDistribution) : ℝ :=
  if dist.score_60 + dist.score_75 > 50 then 75
  else if dist.score_60 + dist.score_75 + dist.score_80 > 50 then 80
  else if dist.score_60 + dist.score_75 + dist.score_80 + dist.score_88 > 50 then 88
  else 92

/-- Theorem: The difference between the mean and median score is 0 --/
theorem mean_median_difference_zero (dist : ScoreDistribution) 
  (h1 : dist.score_60 = 15)
  (h2 : dist.score_75 = 20)
  (h3 : dist.score_80 = 25)
  (h4 : dist.score_88 = 20) :
  |mean_score dist - median_score dist| = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_zero_l1013_101341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1013_101351

/-- The function f(x) = a/x - x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a/x - x

/-- The theorem statement -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (1/4 : ℝ) 1, (f a x) * |x - 1/2| ≤ 1) →
  a ≤ 17/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1013_101351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_l1013_101331

/-- A function that checks if an arrangement of numbers satisfies the condition -/
def is_valid_arrangement (arr : List ℕ) : Prop :=
  ∀ k ∈ arr, k ≤ arr.length / 2 →
    ∃ i j, i < j ∧ arr.get? i = some k ∧ arr.get? j = some k ∧ j - i = k + 1

/-- The main theorem stating that only n = 1 and n = 2 satisfy the condition -/
theorem valid_arrangements :
  ∀ n : ℕ, n > 0 →
    (∃ arr : List ℕ, arr.length = 2 * n ∧
      (∀ k ≤ n, (arr.count k = 2)) ∧
      is_valid_arrangement arr) ↔ (n = 1 ∨ n = 2) := by
  sorry

#check valid_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_l1013_101331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l1013_101395

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => a n + a 0 + n

theorem sequence_sum_theorem :
  let s : ℕ → ℚ := fun n => (1 : ℚ) / a n
  (∀ m n : ℕ, a (m + n) = a m + a n + (m * n : ℚ)) →
  (Finset.sum (Finset.range 2017) (fun i => s (i + 1)) = 4034 / 2018) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l1013_101395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_l1013_101324

/-- The actual diameter of a circular tissue, given its magnified image diameter and magnification factor. -/
noncomputable def actual_diameter (magnified_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  magnified_diameter / magnification_factor

/-- Theorem: The actual diameter of a circular tissue is 0.0002 centimeters when its image is magnified 1,000 times and has a diameter of 0.2 centimeters. -/
theorem tissue_diameter :
  let magnified_diameter : ℝ := 0.2
  let magnification_factor : ℝ := 1000
  actual_diameter magnified_diameter magnification_factor = 0.0002 := by
  -- Unfold the definition of actual_diameter
  unfold actual_diameter
  -- Perform the division
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_l1013_101324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_subtraction_example_l1013_101306

/-- Converts a list of binary digits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n |>.reverse

theorem binary_subtraction_example :
  let a := [true, true, false, true, true]  -- 11011 in binary
  let b := [true, false, true]              -- 101 in binary
  let result := [false, true, true, false, true]  -- 10110 in binary
  binary_to_nat a - binary_to_nat b = binary_to_nat result :=
by
  sorry

#eval binary_to_nat [true, true, false, true, true]  -- Should output 27
#eval binary_to_nat [true, false, true]              -- Should output 5
#eval nat_to_binary 22                               -- Should output [false, true, true, false, true]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_subtraction_example_l1013_101306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l1013_101300

-- Define the points as pairs of real numbers
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
def is_semicircle_center (A : ℝ × ℝ) : Prop := sorry
def on_base (P : ℝ × ℝ) : Prop := sorry
def on_circular_portion (E : ℝ × ℝ) : Prop := sorry
def is_right_angle (E B A : ℝ × ℝ) : Prop := sorry
def extend_line (E A C : ℝ × ℝ) : Prop := sorry
def on_line (F C D : ℝ × ℝ) : Prop := sorry
def is_line (E B F : ℝ × ℝ) : Prop := sorry

-- Define the given lengths
noncomputable def EA_length : ℝ := 1
noncomputable def AC_length : ℝ := Real.sqrt 2
noncomputable def BF_length : ℝ := (2 - Real.sqrt 2) / 4
noncomputable def CF_length : ℝ := (2 * Real.sqrt 5 + Real.sqrt 10) / 4
noncomputable def DF_length : ℝ := (2 * Real.sqrt 5 - Real.sqrt 10) / 4

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem semicircle_problem (A B C D E F : ℝ × ℝ) 
  (h1 : is_semicircle_center A)
  (h2 : on_base B)
  (h3 : on_base D)
  (h4 : on_circular_portion E)
  (h5 : is_right_angle E B A)
  (h6 : extend_line E A C)
  (h7 : on_line F C D)
  (h8 : is_line E B F)
  (h9 : distance E A = EA_length)
  (h10 : distance A C = AC_length)
  (h11 : distance B F = BF_length)
  (h12 : distance C F = CF_length)
  (h13 : distance D F = DF_length) :
  distance D E = Real.sqrt (2 - Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l1013_101300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1013_101366

noncomputable def square_diagonal : ℝ := 10
noncomputable def circle_diameter : ℝ := 10

noncomputable def square_area : ℝ := (square_diagonal ^ 2) / 2
noncomputable def circle_area : ℝ := Real.pi * ((circle_diameter / 2) ^ 2)

noncomputable def area_difference : ℝ := circle_area - square_area

theorem area_difference_approx : 
  ∃ ε > 0, |area_difference - 28.5| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1013_101366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABD_range_l1013_101332

/-- The ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The left focus F1 -/
def F1 : ℝ × ℝ := (-1, 0)

/-- The point D -/
def D : ℝ × ℝ := (-1, -1)

/-- The condition on PF1 + QF1 -/
def PF1_QF1_sum (P Q : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
  Real.sqrt ((Q.1 - F1.1)^2 + (Q.2 - F1.2)^2) = 2 * Real.sqrt 2

/-- The slope of line AB -/
def k_positive (k : ℝ) : Prop := k > 0

/-- The condition on vector AF1 and F1B -/
def vector_condition (A B : ℝ × ℝ) (lambda : ℝ) : Prop :=
  (F1.1 - A.1, F1.2 - A.2) = lambda • (B.1 - F1.1, B.2 - F1.2)

/-- The range of lambda -/
def lambda_range (lambda : ℝ) : Prop := 3 ≤ lambda ∧ lambda ≤ 2 + Real.sqrt 3

/-- The area of triangle ABD -/
noncomputable def area_ABD (A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (B.2 - A.2))

theorem area_ABD_range (a b : ℝ) (P Q A B : ℝ × ℝ) (k lambda : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b P.1 P.2 ∧ ellipse a b Q.1 Q.2 ∧
  P.1 = -Q.1 ∧
  PF1_QF1_sum P Q ∧
  ellipse a b A.1 A.2 ∧ ellipse a b B.1 B.2 ∧
  k_positive k ∧
  A.2 - F1.2 = k * (A.1 - F1.1) ∧
  B.2 - F1.2 = k * (B.1 - F1.1) ∧
  vector_condition A B lambda ∧
  lambda_range lambda →
  2/3 ≤ area_ABD A B ∧ area_ABD A B ≤ Real.sqrt 3 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABD_range_l1013_101332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangle_properties_l1013_101378

/-- Given an oblique triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure ObliqueTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem oblique_triangle_properties (t : ObliqueTriangle) 
  (h : Real.sin t.A = Real.cos t.B) : 
  /- Part 1 -/
  t.A - t.B = Real.pi / 2 ∧ 
  /- Part 2 -/
  (t.a = 1 → 
    ∃ (min : Real), min = 2 * Real.sqrt 2 - 3 ∧ 
    ∀ (v : Real), v = t.b * t.c * Real.cos t.A → min ≤ v) ∧
  /- Part 3 -/
  (Real.sin t.A = Real.cos t.B ∧ 
   Real.sin t.A = (3 / 2) * Real.tan t.C → 
   t.A = 2 * Real.pi / 3 ∧ t.B = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangle_properties_l1013_101378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_values_l1013_101312

-- Define the lines l₁ and l₂
def l₁ (x : ℝ) : ℝ := x
def l₂ (k : ℝ) (x : ℝ) : ℝ := k * x - k + 1

-- Define the intersection point A
def A : ℝ × ℝ := (1, 1)

-- Define the point B where l₂ intersects the x-axis
noncomputable def B (k : ℝ) : ℝ × ℝ := 
  let x := (k - 1) / k
  (x, 0)

-- Define the area of triangle OAB
noncomputable def area_OAB (k : ℝ) : ℝ := 
  let (x_b, _) := B k
  (1/2) * |x_b|

-- Theorem statement
theorem k_values : 
  ∀ k : ℝ, k ≠ 0 → 
  (area_OAB k = 2 ↔ k = -1/3 ∨ k = 1/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_values_l1013_101312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1013_101393

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

/-- The equation of the ellipse -/
def ellipse_equation (x y m : ℝ) : Prop := x^2 / m + y^2 = 1

/-- Theorem stating the possible values of m for the given ellipse -/
theorem ellipse_m_values :
  ∃ (m : ℝ), (m = 4 ∨ m = 1/4) ∧
  ∀ (x y : ℝ), ellipse_equation x y m →
  (Real.sqrt (1 - 1/m) = eccentricity ∨ Real.sqrt (1 - m) = eccentricity) := by
  sorry

#check ellipse_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1013_101393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_snail_ratio_l1013_101397

/-- Proves that the ratio of snails found by the remaining ducklings to the mother duck's snails is 1:1 -/
theorem duck_snail_ratio :
  -- Total number of ducklings
  ∀ (total_ducklings : ℕ)
  -- Number of snails found by each duckling in the first group
  (snails_per_duckling_group1 : ℕ)
  -- Number of snails found by each duckling in the second group
  (snails_per_duckling_group2 : ℕ)
  -- Total number of snails found by the family
  (total_snails : ℕ),
  -- Conditions
  total_ducklings = 8 →
  snails_per_duckling_group1 = 5 →
  snails_per_duckling_group2 = 9 →
  total_snails = 294 →
  -- Calculations
  let first_group_snails := 3 * snails_per_duckling_group1;
  let second_group_snails := 3 * snails_per_duckling_group2;
  let mother_duck_snails := 3 * (first_group_snails + second_group_snails);
  let remaining_ducklings_snails := total_snails - (first_group_snails + second_group_snails + mother_duck_snails);
  -- Theorem statement
  remaining_ducklings_snails = mother_duck_snails := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_snail_ratio_l1013_101397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_positions_l1013_101345

/-- The circle with equation (x - 1)^2 + (y + 3)^2 = 25 -/
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 25

/-- Point A -/
def A : ℝ × ℝ := (0, 0)

/-- Point B -/
def B : ℝ × ℝ := (-2, 1)

/-- Point C -/
def C : ℝ × ℝ := (3, 3)

/-- Point D -/
def D : ℝ × ℝ := (2, -1)

theorem point_positions :
  (A.1 - 1)^2 + (A.2 + 3)^2 < 25 ∧
  (B.1 - 1)^2 + (B.2 + 3)^2 = 25 ∧
  (C.1 - 1)^2 + (C.2 + 3)^2 > 25 ∧
  (D.1 - 1)^2 + (D.2 + 3)^2 < 25 := by
  sorry

#check point_positions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_positions_l1013_101345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1013_101359

def vector_a (l : ℝ) : Fin 2 → ℝ := ![l, 1]
def vector_b (l : ℝ) : Fin 2 → ℝ := ![l + 2, 1]

theorem vector_equality (l : ℝ) :
  (‖vector_a l + vector_b l‖ = ‖vector_a l - vector_b l‖) → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1013_101359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1013_101338

/-- The function f(x) defined as sin(ωx) + cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

/-- Theorem stating the conditions and the result to be proved -/
theorem omega_value (ω : ℝ) :
  ω > 0 ∧
  (∀ x y : ℝ, -ω < x ∧ x < y ∧ y < ω → f ω x < f ω y) ∧
  (∀ x : ℝ, f ω (ω + x) = f ω (ω - x)) →
  ω = Real.sqrt Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1013_101338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_e_l1013_101328

-- Define the force function
noncomputable def F (x : ℝ) : ℝ := 1 + Real.exp x

-- Define the work done by the force
noncomputable def work (F : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, F x

-- Theorem statement
theorem work_equals_e :
  work F 0 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_equals_e_l1013_101328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l1013_101333

theorem cubic_root_inequality (x : ℝ) : 
  (2 * x) ^ (1/3) + 3 / ((2 * x) ^ (1/3) + 4) ≤ 0 ↔ 
  -32 < x ∧ x < -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l1013_101333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_f_minimum_value_f_min_is_minimum_l1013_101343

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Part 1: Condition for f(x) to be increasing on [1, +∞)
theorem f_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, ∀ y ≥ x, f a y ≥ f a x) ↔ a ≥ -2 :=
sorry

-- Part 2: Minimum value of f(x) on [-2, 1]
noncomputable def f_min (a : ℝ) : ℝ :=
  if a > 4 then 8 - 2*a
  else if a ≥ -2 then 4 - a^2/4
  else 5 + a

theorem f_minimum_value (a : ℝ) :
  ∀ x ∈ Set.Icc (-2) 1, f a x ≥ f_min a :=
sorry

-- Prove that f_min is indeed the minimum value
theorem f_min_is_minimum (a : ℝ) :
  ∃ x ∈ Set.Icc (-2) 1, f a x = f_min a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_f_minimum_value_f_min_is_minimum_l1013_101343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_players_l1013_101349

/-- Represents the number of girls in the tournament -/
def num_girls : ℕ := sorry

/-- Represents the number of boys in the tournament -/
def num_boys : ℕ := 5 * num_girls

/-- Represents the total number of players in the tournament -/
def total_players : ℕ := num_girls + num_boys

/-- Represents the total number of games played in the tournament -/
def total_games : ℕ := total_players * (total_players - 1)

/-- Represents the total points scored by girls -/
def girls_points : ℕ := sorry

/-- Represents the total points scored by boys -/
def boys_points : ℕ := 2 * girls_points

theorem chess_tournament_players :
  (num_boys = 5 * num_girls) →
  (total_games = total_players * (total_players - 1)) →
  (boys_points = 2 * girls_points) →
  (total_players = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_players_l1013_101349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_y_intercept_of_specific_line_l1013_101326

/-- The y-intercept of a line is the point where the line intersects the y-axis (x = 0) -/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ := (0, c / b)

/-- A point (x, y) lies on a line ax + by = c if and only if the equation holds -/
def on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem y_intercept_of_line (a b c : ℝ) (h : b ≠ 0) :
  on_line a b c (y_intercept a b c) :=
by sorry

theorem y_intercept_of_specific_line :
  y_intercept 4 6 24 = (0, 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_y_intercept_of_specific_line_l1013_101326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inclination_angle_l1013_101368

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x + 2 + 2*Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x - 3 + 2/x

-- Define IsTangentLine (since it's not defined in the standard library)
def IsTangentLine (f : ℝ → ℝ) (l : Set (ℝ × ℝ)) (x : ℝ) : Prop :=
  ∃ y, (x, y) ∈ l ∧ f x = y ∧ ∀ x' y', (x', y') ∈ l → y' - y = (f' x) * (x' - x)

-- Define IsInclinationAngle (since it's not defined in the standard library)
def IsInclinationAngle (l : Set (ℝ × ℝ)) (α : ℝ) : Prop :=
  ∃ m, (∀ x y x' y', (x, y) ∈ l → (x', y') ∈ l → y' - y = m * (x' - x)) ∧ α = Real.arctan m

-- Theorem statement
theorem min_inclination_angle (l : Set (ℝ × ℝ)) :
  (∃ x > 0, IsTangentLine f l x) →
  (∃ α : ℝ, α = Real.pi/4 ∧ ∀ β, IsInclinationAngle l β → α ≤ β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inclination_angle_l1013_101368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_a_value_positivity_implies_a_range_l1013_101388

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 6

-- Part 1: Critical point condition
theorem critical_point_implies_a_value (a : ℝ) (h : a > 0) :
  (∀ x, deriv (f a) x = 0 → x = 1) → a = 2 :=
sorry

-- Part 2: Positivity condition
theorem positivity_implies_a_range (a : ℝ) (h : a > 2) :
  (∀ x, x ∈ Set.Icc (-1) 1 → f a x > 0) → 2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_a_value_positivity_implies_a_range_l1013_101388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1013_101390

noncomputable def sequence1 : List ℝ := [2, 4, 6, 8]
noncomputable def sequence2 : List ℝ := [1, 2, 3, 4]

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := s.sum / s.length
  (s.map (fun x => (x - mean)^2)).sum / s.length

theorem variance_relation :
  variance sequence1 = 4 * variance sequence2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1013_101390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1013_101373

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Distance from apex to any vertex of the base -/
  apexToVertex : ℝ

/-- The height of a right pyramid -/
noncomputable def pyramidHeight (p : RightPyramid) : ℝ :=
  Real.sqrt (p.apexToVertex ^ 2 - 2 * (p.baseSide / 2) ^ 2)

/-- Theorem stating the height of a specific right pyramid -/
theorem specific_pyramid_height :
  let p := RightPyramid.mk 10 12
  pyramidHeight p = 3 * Real.sqrt (47 / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1013_101373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_solution_sequence_l1013_101362

def is_solution (x : Int) : Bool := (x^2 - 2*x - 3 < 0)

def arithmetic_sequence_of_solutions : List Int :=
  (List.range 10).filter is_solution

theorem fourth_term_of_solution_sequence :
  arithmetic_sequence_of_solutions.get? 3 = some 3 ∨
  arithmetic_sequence_of_solutions.get? 3 = some (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_solution_sequence_l1013_101362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l1013_101377

/-- The length of the diagonal of a rectangular prism with length 3, width 4, and height 5 is 5√2 -/
theorem rectangular_prism_diagonal : 
  let length : ℝ := 3
  let width : ℝ := 4
  let height : ℝ := 5
  let diagonal := Real.sqrt (length^2 + width^2 + height^2)
  diagonal = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonal_l1013_101377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_property_l1013_101348

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the fixed point M
noncomputable def M : ℝ × ℝ := (-11/8, 0)

-- Define the point through which all lines pass
noncomputable def P : ℝ × ℝ := (-1, 0)

-- Define the dot product of vectors MA and MB
noncomputable def dot_product (A B : ℝ × ℝ) : ℝ :=
  let (x_a, y_a) := A
  let (x_b, y_b) := B
  let (m_x, _) := M
  (x_a - m_x) * (x_b - m_x) + y_a * y_b

-- Theorem statement
theorem ellipse_fixed_point_property :
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    ∃ (k : ℝ), (A.2 - P.2) = k * (A.1 - P.1) ∧ (B.2 - P.2) = k * (B.1 - P.1) →
    dot_product A B = -135/64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_property_l1013_101348
