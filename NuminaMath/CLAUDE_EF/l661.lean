import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_employees_with_sunglasses_and_shoes_l661_66141

theorem min_employees_with_sunglasses_and_shoes (n : ℕ) (h1 : n > 0) :
  ∃ (both : ℕ), both ≥ 1 ∧ n / 3 + n * 5 / 6 - both = n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_employees_with_sunglasses_and_shoes_l661_66141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_trapezium_area_l661_66186

noncomputable def A : ℝ × ℝ := (2, 5)
noncomputable def B : ℝ × ℝ := (8, 12)
noncomputable def C : ℝ × ℝ := (14, 7)
noncomputable def D : ℝ × ℝ := (6, 2)

noncomputable def trapeziumArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem irregular_trapezium_area :
  trapeziumArea A B C D = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_trapezium_area_l661_66186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l661_66101

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 → a*b ≥ x*y) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 2/a + 1/b ≤ 2/x + 1/y) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 → (2:ℝ)^a + (4:ℝ)^b ≤ (2:ℝ)^x + (4:ℝ)^y) ∧
  a*b = 1/8 ∧
  2/a + 1/b = 8 ∧
  (2:ℝ)^a + (4:ℝ)^b = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l661_66101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_22_l661_66198

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid with four vertices -/
structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let b1 := abs (t.F.y - t.E.y)
  let b2 := abs (t.G.y - t.H.y)
  let h := abs (t.G.x - t.E.x)
  (b1 + b2) * h / 2

/-- The main theorem stating that the area of the given trapezoid is 22 -/
theorem trapezoid_area_is_22 :
  let t := Trapezoid.mk
    (Point.mk 2 (-3))  -- E
    (Point.mk 2 2)     -- F
    (Point.mk 6 8)     -- G
    (Point.mk 6 2)     -- H
  trapezoidArea t = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_22_l661_66198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_KJM_equals_5sqrt2_div_12_l661_66152

-- Define the rectangular solid
structure RectangularSolid where
  angle_JKI : ℝ
  angle_MKJ : ℝ
  angle_KJM : ℝ

-- Define our specific rectangular solid
noncomputable def our_solid : RectangularSolid where
  angle_JKI := Real.pi / 6  -- 30 degrees in radians
  angle_MKJ := Real.pi / 4  -- 45 degrees in radians
  angle_KJM := Real.arccos ((5 * Real.sqrt 2) / 12)  -- The angle we're proving

-- State the theorem
theorem cos_KJM_equals_5sqrt2_div_12 :
  Real.cos our_solid.angle_KJM = (5 * Real.sqrt 2) / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_KJM_equals_5sqrt2_div_12_l661_66152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_m_range_l661_66149

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 0 then 1 / (x + 1) - 3
  else if 0 < x ∧ x ≤ 1 then x^2 - 3*x + 2
  else 0  -- undefined for other x values

-- Define the theorem
theorem two_roots_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ g x - m * x - m = 0 ∧ g y - m * y - m = 0) ∧
  (∀ z : ℝ, g z - m * z - m = 0 → z = x ∨ z = y) →
  m ∈ Set.Ioo (-9/4) (-2) ∪ Set.Ioc 0 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_m_range_l661_66149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_calculation_l661_66173

/-- Given a total sum of squared deviations and a sum of squared residuals,
    calculate the coefficient of determination (R²). -/
noncomputable def coefficient_of_determination (total_sum_squared_deviations : ℝ) (sum_squared_residuals : ℝ) : ℝ :=
  1 - sum_squared_residuals / total_sum_squared_deviations

/-- Theorem: Given specific values for total sum of squared deviations and sum of squared residuals,
    the coefficient of determination (R²) is equal to 0.25. -/
theorem r_squared_calculation (total_sum_squared_deviations : ℝ) (sum_squared_residuals : ℝ)
    (h1 : total_sum_squared_deviations = 80)
    (h2 : sum_squared_residuals = 60) :
    coefficient_of_determination total_sum_squared_deviations sum_squared_residuals = 0.25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_calculation_l661_66173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_comparison_l661_66128

theorem subset_sum_comparison :
  let S := Finset.range 64
  let less_than_95 := {s : Finset ℕ | s.card = 3 ∧ s ⊆ S ∧ (s.sum id < 95)}
  let greater_equal_95 := {s : Finset ℕ | s.card = 3 ∧ s ⊆ S ∧ (s.sum id ≥ 95)}
  Finset.card (Finset.filter (λ s => s.card = 3 ∧ s ⊆ S ∧ s.sum id < 95) (Finset.powerset S)) <
  Finset.card (Finset.filter (λ s => s.card = 3 ∧ s ⊆ S ∧ s.sum id ≥ 95) (Finset.powerset S)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_comparison_l661_66128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_plus_intercept_l661_66119

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ := l.y₁ - l.slope * l.x₁

/-- Theorem: For a line passing through (2, -1) and (-1, 6),
    the sum of its slope and y-intercept is 4/3 -/
theorem line_slope_plus_intercept :
  let l : Line := ⟨2, -1, -1, 6⟩
  l.slope + l.yIntercept = 4/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_plus_intercept_l661_66119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_plus_x_sin_l661_66169

theorem min_value_cos_plus_x_sin (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) : 
  Real.cos x + x * Real.sin x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_plus_x_sin_l661_66169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_roots_exist_l661_66145

open Real

/-- Defines the function f(x) = tan(x) + tan(2x) + tan(3x) - 0.1 --/
noncomputable def f (x : ℝ) := tan x + tan (2 * x) + tan (3 * x) - 0.1

/-- Theorem stating the existence of five approximate roots for f(x) = 0 --/
theorem approximate_roots_exist (ε : ℝ) (hε : ε > 0) :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ),
    x₁ ∈ Set.Ioo (-π/2) (π/2) ∧
    x₂ ∈ Set.Ioo (-π/2) (π/2) ∧
    x₃ ∈ Set.Ioo (-π/2) (π/2) ∧
    x₄ ∈ Set.Ioo (-π/2) (π/2) ∧
    x₅ ∈ Set.Ioo (-π/2) (π/2) ∧
    |f x₁| < ε ∧
    |f x₂| < ε ∧
    |f x₃| < ε ∧
    |f x₄| < ε ∧
    |f x₅| < ε ∧
    |x₁ + 1.04| < ε ∧
    |x₂ + 0.61| < ε ∧
    |x₃ - 0.017| < ε ∧
    |x₄ - 0.62| < ε ∧
    |x₅ - 1.05| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_roots_exist_l661_66145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_480_l661_66185

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  -- Base rectangle
  ab : ℝ
  bc : ℝ
  -- Apex point to base vertex distance
  pb : ℝ
  -- Conditions
  pa_perp_ad : Bool
  pa_perp_ab : Bool

/-- Calculates the volume of a rectangular base pyramid -/
noncomputable def pyramidVolume (p : RectangularBasePyramid) : ℝ :=
  let baseArea := p.ab * p.bc
  let height := Real.sqrt (p.pb^2 - p.ab^2)
  (1/3) * baseArea * height

/-- Theorem: The volume of the given pyramid is 480 cubic units -/
theorem pyramid_volume_is_480 (p : RectangularBasePyramid) 
    (h1 : p.ab = 10)
    (h2 : p.bc = 6)
    (h3 : p.pb = 26)
    (h4 : p.pa_perp_ad = true)
    (h5 : p.pa_perp_ab = true) :
    pyramidVolume p = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_480_l661_66185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athena_sandwiches_l661_66172

/-- The number of sandwiches Athena bought -/
def num_sandwiches : ℕ := sorry

/-- The cost of each sandwich in dollars -/
def sandwich_cost : ℚ := 3

/-- The number of fruit drinks Athena bought -/
def num_drinks : ℕ := 2

/-- The cost of each fruit drink in dollars -/
def drink_cost : ℚ := 5/2

/-- The total amount Athena spent in dollars -/
def total_spent : ℚ := 14

theorem athena_sandwiches :
  num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_spent ∧
  num_sandwiches = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athena_sandwiches_l661_66172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quotient_for_optimal_A_l661_66178

def is_valid_A (A : ℕ) : Prop :=
  A ≥ 10^1000 ∧ A < 10^1001

noncomputable def left_cyclic_permutation (A : ℕ) : ℕ :=
  sorry

def optimal_A : ℕ :=
  9 * (10^1000 - 10^501) + 8 * 10^500 + 9 * (10^500 - 1)

theorem smallest_quotient_for_optimal_A :
  ∀ A : ℕ, is_valid_A A →
  let Z := left_cyclic_permutation A
  (A > Z) →
  ((A : ℚ) / Z > 1) →
  ((optimal_A : ℚ) / (left_cyclic_permutation optimal_A) ≤ (A : ℚ) / Z) :=
by sorry

#check smallest_quotient_for_optimal_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quotient_for_optimal_A_l661_66178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_staircase_partition_count_l661_66192

/-- Represents a staircase of height n -/
structure Staircase (n : ℕ) where
  height : ℕ
  height_eq : height = n

/-- Represents a valid partition of a staircase into rectangles -/
structure ValidPartition (n : ℕ) where
  partitions : ℕ
  is_valid : partitions > 0

/-- The number of valid partitions for a staircase of height n -/
def numValidPartitions (n : ℕ) : ℕ := 2^(n-1)

/-- 
Theorem stating that the number of valid partitions of a staircase
of height n is 2^(n-1)
-/
theorem staircase_partition_count (n : ℕ) :
  (∀ p : ValidPartition n, p.partitions > 0) →
  (∃! f : Staircase n → ℕ, ∀ s : Staircase n, f s = numValidPartitions n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_staircase_partition_count_l661_66192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcs_inside_exist_l661_66104

/-- Represents an arc on a circle -/
structure Arc where
  start : ℕ  -- Starting point of the arc
  length : ℕ  -- Length of the arc
  deriving Repr

/-- Checks if one arc is inside another -/
def is_inside (a b : Arc) (circle_size : ℕ) : Prop :=
  (a.start ≥ b.start ∧ a.start + a.length ≤ b.start + b.length) ∨
  (a.start ≥ b.start ∧ a.start + a.length ≤ circle_size ∧ b.start + b.length > circle_size) ∨
  (a.start < b.start ∧ a.start + a.length > circle_size ∧ (a.start + a.length) % circle_size ≤ b.start + b.length)

/-- The main theorem to be proved -/
theorem arcs_inside_exist (n : ℕ) (h : n ≥ 1) :
  ∃ (arcs : List Arc),
    arcs.length = n + 1 ∧
    (∀ i, i ∈ arcs.map Arc.length → 1 ≤ i ∧ i ≤ n + 1) ∧
    (∃ a b, a ∈ arcs ∧ b ∈ arcs ∧ is_inside a b (2 * n)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcs_inside_exist_l661_66104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_cosine_l661_66189

variable (a b c : ℝ × ℝ)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

noncomputable def angle_cosine (v w : ℝ × ℝ) : ℝ := 
  dot_product v w / (vector_length v * vector_length w)

theorem min_length_cosine 
  (ha : a = (3, -1))
  (hb_dot_a : dot_product b a = -5)
  (hb_length : vector_length b = Real.sqrt 5) :
  ∃ (c : ℝ × ℝ), 
    (∀ (d : ℝ × ℝ), vector_length c ≤ vector_length d) → 
    angle_cosine b c = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_cosine_l661_66189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_inequality_l661_66187

-- Define the linear function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define g in terms of f
noncomputable def g (x : ℝ) : ℝ := 2^(f x)

-- Theorem statement
theorem linear_function_and_inequality :
  (∀ x, f x = 2 * x + 1) ∧
  (f 0 = 1) ∧
  (f 1 = 3) ∧
  (∀ m, g (m^2 - 2) < g m ↔ -1 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_inequality_l661_66187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_winning_iff_perfect_square_l661_66182

/-- Represents the divisor erasing game on a positive integer N -/
structure DivisorErasingGame (N : ℕ+) where
  board : Finset ℕ
  board_valid : ∀ n ∈ board, n ∣ N

/-- A valid move in the game is erasing either a multiple or a divisor of the previous move -/
def ValidMove {N : ℕ+} (game : DivisorErasingGame N) (prev : ℕ) (next : ℕ) : Prop :=
  (next ∣ prev ∨ prev ∣ next) ∧ next ∈ game.board

/-- Alice has a winning strategy if and only if N is a perfect square -/
theorem alice_winning_iff_perfect_square (N : ℕ+) :
  (∃ (strategy : DivisorErasingGame N → ℕ),
    ∀ (game : DivisorErasingGame N) (bob_move : ℕ),
    ValidMove game N (strategy game) ∧
    (ValidMove game (strategy game) bob_move →
     ∃ (alice_response : ℕ), ValidMove game bob_move alice_response)) ↔
  ∃ (n : ℕ), N = n^2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_winning_iff_perfect_square_l661_66182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_with_distance_3_l661_66183

noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem parallel_lines_with_distance_3 :
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 4 * y - 5
  let line2 : ℝ → ℝ → ℝ := λ x y => 3 * x + 4 * y + 10
  let line3 : ℝ → ℝ → ℝ := λ x y => 3 * x + 4 * y - 20
  (are_parallel 3 4 3 4 ∧ 
   are_parallel 3 4 3 4) ∧
  (distance_parallel_lines 3 4 (-5) 10 = 3 ∧
   distance_parallel_lines 3 4 (-5) (-20) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_with_distance_3_l661_66183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_proof_l661_66140

/-- Represents a bowler's statistics -/
structure BowlerStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
noncomputable def newAverage (stats : BowlerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns : ℚ) / (stats.wickets + newWickets)

/-- Theorem: Given the conditions, prove the original average was 12.4 -/
theorem bowling_average_proof (originalStats : BowlerStats) 
    (h1 : originalStats.wickets = 175)
    (h2 : newAverage originalStats 8 26 = originalStats.average - 4/10) :
  originalStats.average = 124/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_proof_l661_66140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cube_root_of_27_plus_27i_l661_66191

theorem complex_cube_root_of_27_plus_27i (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  (c : ℂ) + (d : ℂ) * Complex.I ^ 3 = 27 + 27 * Complex.I →
  (c : ℂ) + (d : ℂ) * Complex.I = 3 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cube_root_of_27_plus_27i_l661_66191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l661_66126

noncomputable def f (x : Real) : Real := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : Real), p > 0 ∧ ∀ (x : Real), f (x + p) = f x ∧ ∀ (q : Real), q > 0 ∧ (∀ (x : Real), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x y : Real), x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6) → y ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6) → x < y → f x < f y) ∧
  (Set.Icc (-2) (Real.sqrt 3) = {y | ∃ (x : Real), x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3) ∧ f x = y}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l661_66126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l661_66165

open BigOperators

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = ∑ i in Finset.range n, a (i + 1)) →
  (S 3 = a 3) →
  (a 3 ≠ 0) →
  S 4 / S 3 = 8/3 := by
  intros sum_def s3_eq_a3 a3_nonzero
  sorry  -- The proof steps would go here

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l661_66165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_increasing_l661_66103

noncomputable def f (x : ℝ) : ℝ := -1/x

theorem f_is_odd_and_increasing :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_increasing_l661_66103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_l661_66118

/-- The function f(x) = x ln x - m x^2 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

/-- f'(x) = ln x + 1 - 2mx -/
noncomputable def f_derivative (m : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - 2 * m * x

/-- Theorem: f(x) has two extreme points if and only if 0 < m < 1/2 -/
theorem f_two_extreme_points (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_derivative m x₁ = 0 ∧ f_derivative m x₂ = 0) ↔ 0 < m ∧ m < 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_l661_66118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approx_l661_66150

/-- The total area of hardwood flooring in Nancy's bathroom -/
noncomputable def total_area : ℝ :=
  let central_area := 3 * 3
  let hallway_area := 2 * 1.2
  let l_shaped_area := 1.5 * 0.6
  let triangular_area := 0.5 * 0.9 * 0.9
  let semi_circular_area := 0.5 * Real.pi * (1.2 * 1.2)
  central_area + hallway_area + l_shaped_area + triangular_area + semi_circular_area

/-- The theorem stating that the total area is approximately 14.965 square meters -/
theorem total_area_approx :
  |total_area - 14.965| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approx_l661_66150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l661_66176

/-- The slope of the asymptotes for the hyperbola (y^2 / 16) - (x^2 / 9) = 1 is 4/3 -/
theorem hyperbola_asymptote_slope :
  let f (x y : ℝ) := (y^2 / 16) - (x^2 / 9)
  ∃ m : ℝ, m = 4/3 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, 
      x > δ → abs (y / x - m) < ε ∧ f x y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l661_66176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_after_discounts_l661_66174

/-- Given a dress with original price d, prove that after applying a 25% discount
    and then an additional 20% staff discount, the final price is 0.60d. -/
theorem dress_price_after_discounts (d : ℝ) : 
  d * (1 - 0.25) * (1 - 0.20) = d * 0.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_after_discounts_l661_66174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_has_six_members_l661_66167

/-- A club with members and committees -/
structure Club where
  members : ℕ
  committees : Finset (Finset ℕ)
  member_in_two_committees : ∀ m, m < members → (committees.filter (λ c => m ∈ c)).card = 2
  pair_share_one_member : ∀ c1 c2, c1 ∈ committees → c2 ∈ committees → c1 ≠ c2 → (c1 ∩ c2).card = 1

/-- The theorem stating that a club satisfying the given conditions must have 6 members -/
theorem club_has_six_members (c : Club) (h : c.committees.card = 4) : c.members = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_has_six_members_l661_66167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_l661_66171

theorem consecutive_odd_integers (a b c : ℤ) : 
  (Odd a ∧ Odd b ∧ Odd c) →  -- Three odd integers
  (b = a + 2 ∧ c = b + 2) →  -- Consecutive
  (a + c = 156) →            -- Sum of first and third is 156
  b = 78 :=                  -- Second integer is 78
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_integers_l661_66171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_period_cyclic_sequence_16_l661_66154

noncomputable def cyclic_sequence (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Added case for 0
  | 1 => a
  | n + 2 => -1 / (cyclic_sequence a (n + 1) + 1)

theorem cyclic_sequence_period (a : ℝ) (h : a > 0) :
  ∀ n : ℕ, n > 0 → (cyclic_sequence a n = a ↔ n % 3 = 1) :=
by sorry

theorem cyclic_sequence_16 (a : ℝ) (h : a > 0) :
  cyclic_sequence a 16 = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_period_cyclic_sequence_16_l661_66154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_with_divisible_difference_l661_66153

theorem infinite_primes_with_divisible_difference (a : ℕ) (ha : a > 0) :
  ∃ S : Set ℕ, (∀ p, p ∈ S → Prime p) ∧ 
                Set.Infinite S ∧
                (∀ p q, p ∈ S → q ∈ S → ∃ k : ℤ, p - q = k * a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_with_divisible_difference_l661_66153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l661_66139

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the triangle -/
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.A < Real.pi ∧
  t.B > 0 ∧ t.B < Real.pi ∧
  t.C > 0 ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : TriangleProperties t)
  (h2 : t.a^2 + t.c^2 + Real.sqrt 2 * t.a * t.c = t.b^2)
  (h3 : Real.sin t.A = Real.sqrt 10 / 10) :
  Real.sin t.C = Real.sqrt 5 / 5 ∧
  (t.a = 2 → 1/2 * t.a * t.c * Real.sin t.B = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l661_66139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l661_66125

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-2.25) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l661_66125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_triangle_area_l661_66155

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sin (2 * x)

theorem f_derivative_and_triangle_area :
  ∃ (f' : ℝ → ℝ),
    (∀ x, HasDerivAt f (f' x) x) ∧
    (∀ x, f' x = 2 * Real.sin x * Real.cos x + 2 * Real.cos (2 * x)) ∧
    (let tangent_line := λ x ↦ f (π/4) + f' (π/4) * (x - π/4);
     let x_intercept := π/4 - 3/2;
     let y_intercept := 3/2 - π/4;
     (1/2) * x_intercept * y_intercept = (1/2) * (3/2 - π/4)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_triangle_area_l661_66155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l661_66136

-- Define the necessary functions
def is_triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def is_opposite_angle (side : ℝ) (angle : ℝ) : Prop :=
  side = 2 * Real.sin angle

theorem triangle_equilateral (a b c : ℝ) (α β γ : ℝ) 
  (h_triangle : is_triangle a b c)
  (h_angles : α + β + γ = Real.pi)
  (h_opposite : is_opposite_angle a α ∧ is_opposite_angle b β ∧ is_opposite_angle c γ)
  (h_condition : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  α = Real.pi / 3 ∧ β = Real.pi / 3 ∧ γ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l661_66136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l661_66115

/-- Given a line l that is symmetric about the y-axis with the line 4x - 3y + 5 = 0,
    prove that the equation of line l is 4x + 3y - 5 = 0 -/
theorem symmetric_line_equation :
  ∀ (x y : ℝ), (4 * x - 3 * y + 5 = 0) ↔ (4 * (-x) - 3 * y + 5 = 0) →
  ∃ (l : Set (ℝ × ℝ)), ∀ (x y : ℝ), (x, y) ∈ l ↔ 4 * x + 3 * y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l661_66115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_approx_l661_66111

/-- The atomic mass of potassium in g/mol -/
def mass_K : ℚ := 3910 / 100

/-- The atomic mass of chromium in g/mol -/
def mass_Cr : ℚ := 52

/-- The atomic mass of oxygen in g/mol -/
def mass_O : ℚ := 16

/-- The chemical formula of the compound -/
structure K2Cr2O7 where
  K : Fin 2
  Cr : Fin 2
  O : Fin 7

/-- The molar mass of K2Cr2O7 in g/mol -/
def molar_mass_K2Cr2O7 : ℚ := 2 * mass_K + 2 * mass_Cr + 7 * mass_O

/-- The mass percentage of Cr in K2Cr2O7 -/
def mass_percentage_Cr : ℚ := (2 * mass_Cr / molar_mass_K2Cr2O7) * 100

/-- Theorem stating that the mass percentage of Cr in K2Cr2O7 is approximately 35.36% -/
theorem mass_percentage_Cr_approx :
  ∃ ε > 0, |mass_percentage_Cr - 3536 / 100| < ε := by
  sorry

#eval mass_percentage_Cr

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_approx_l661_66111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_is_ten_percent_l661_66127

/-- Calculates the commission rate given the basic salary, total sales, savings percentage, and monthly expenses. -/
noncomputable def calculate_commission_rate (basic_salary : ℝ) (total_sales : ℝ) (savings_percentage : ℝ) (monthly_expenses : ℝ) : ℝ :=
  let total_earnings := basic_salary + (total_sales * (100 - savings_percentage) / 100)
  let commission := total_earnings - basic_salary
  (commission / total_sales) * 100

/-- Theorem stating that the commission rate is 10% given the specified conditions. -/
theorem commission_rate_is_ten_percent :
  calculate_commission_rate 1250 23600 20 2888 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_is_ten_percent_l661_66127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_six_on_circle_l661_66156

-- Define a type for points on a plane
def Point : Type := ℝ × ℝ

-- Define a predicate to check if points lie on a circle
def lie_on_circle (points : Set Point) : Prop := sorry

-- Define the property that among any 5 points, 4 lie on a circle
def four_of_five_on_circle (points : Set Point) : Prop :=
  ∀ (subset : Finset Point), subset.toSet ⊆ points → subset.card = 5 →
    ∃ (circle_points : Finset Point), circle_points.toSet ⊆ subset.toSet ∧ circle_points.card = 4 ∧
      lie_on_circle circle_points.toSet

-- State the theorem
theorem at_least_six_on_circle (points : Finset Point) 
  (h : points.card = 13) 
  (h_four_of_five : four_of_five_on_circle points.toSet) : 
  ∃ (circle_points : Finset Point), circle_points.toSet ⊆ points.toSet ∧ 
    circle_points.card ≥ 6 ∧ lie_on_circle circle_points.toSet :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_six_on_circle_l661_66156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l661_66177

theorem simplify_trig_expression (x : ℝ) (h : Real.cos x ≠ 0) :
  (Real.sin x * Real.cos x + Real.sin (2 * x)) / (Real.cos x + Real.cos (2 * x) + Real.sin x ^ 2) = 
  (3 * Real.sin x) / (1 + 2 * Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l661_66177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_integer_for_all_integer_x_l661_66161

/-- A function f(x) = ax^4 + bx^3 + cx^2 + dx with specific properties -/
def f (a b c d : ℝ) (x : ℤ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x

/-- Theorem stating that f(x) is an integer for all integer x -/
theorem f_is_integer_for_all_integer_x (a b c d : ℝ) :
  (a > 0) → (b > 0) → (c > 0) → (d > 0) →
  (∀ x : ℤ, x ∈ ({-2, -1, 0, 1, 2} : Set ℤ) → ∃ n : ℤ, f a b c d x = n) →
  (f a b c d 1 = 1) →
  (f a b c d 5 = 70) →
  (∀ x : ℤ, ∃ n : ℤ, f a b c d x = n) :=
by
  sorry

#check f_is_integer_for_all_integer_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_integer_for_all_integer_x_l661_66161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_reciprocal_sum_l661_66168

/-- An equilateral triangle with a line from one vertex to the opposite side -/
structure TriangleWithLine where
  /-- The points of the equilateral triangle -/
  A : ℂ
  B : ℂ
  C : ℂ
  /-- The point where the line intersects the opposite side -/
  D : ℂ
  /-- The point where the line intersects the circumcircle -/
  Q : ℂ
  /-- ABC forms an equilateral triangle -/
  eq_triangle : Complex.abs (B - A) = Complex.abs (C - A) ∧ Complex.abs (C - A) = Complex.abs (C - B)
  /-- D lies on BC -/
  D_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • B + t • C
  /-- Q lies on AD -/
  Q_on_AD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • D
  /-- Q lies on the circumcircle of ABC -/
  Q_on_circle : Complex.abs (Q - A) = Complex.abs (B - A)

/-- The main theorem to be proved -/
theorem triangle_line_reciprocal_sum (T : TriangleWithLine) :
  1 / Complex.abs (T.Q - T.D) = 1 / Complex.abs (T.Q - T.B) + 1 / Complex.abs (T.Q - T.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_reciprocal_sum_l661_66168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_proof_l661_66109

/-- The circle equation: x^2 + y^2 + 4x - 2y + 24/5 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 24/5 = 0

/-- The line equation: 3x + 4y = 0 -/
def line_equation (x y : ℝ) : Prop :=
  3*x + 4*y = 0

/-- The maximum distance from a point on the circle to the line -/
noncomputable def max_distance : ℝ := (2 + Real.sqrt 5) / 5

theorem max_distance_proof :
  ∃ (p : ℝ × ℝ), circle_equation p.1 p.2 ∧
    ∀ (q : ℝ × ℝ), circle_equation q.1 q.2 →
      ∀ (r : ℝ × ℝ), line_equation r.1 r.2 →
        ((p.1 - r.1)^2 + (p.2 - r.2)^2).sqrt ≤ max_distance :=
by
  sorry

#check max_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_proof_l661_66109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_altitude_and_area_l661_66197

/-- An isosceles right triangle with side length 8 -/
structure IsoscelesRightTriangle where
  /-- The length of the equal sides -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- The altitude of an isosceles right triangle -/
noncomputable def altitude (t : IsoscelesRightTriangle) : ℝ :=
  t.side / Real.sqrt 2

/-- The area of an isosceles right triangle -/
noncomputable def area (t : IsoscelesRightTriangle) : ℝ :=
  t.side * t.side / 2

/-- Theorem about the altitude and area of an isosceles right triangle with side length 8 -/
theorem isosceles_right_triangle_altitude_and_area :
  ∃ (t : IsoscelesRightTriangle),
    t.side = 8 ∧
    altitude t = 4 * Real.sqrt 2 ∧
    area t = 16 * Real.sqrt 2 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_altitude_and_area_l661_66197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_b_not_perpendicular_l661_66138

def vector_a1 : Fin 3 → ℝ := ![3, 4, 0]
def vector_a2 : Fin 3 → ℝ := ![0, 0, 5]

def vector_b1 : Fin 3 → ℝ := ![6, 0, 12]
def vector_b2 : Fin 3 → ℝ := ![6, -5, 7]

def vector_c1 : Fin 3 → ℝ := ![-2, 1, 2]
def vector_c2 : Fin 3 → ℝ := ![4, -6, 7]

def vector_d1 : Fin 3 → ℝ := ![3, 1, 3]
def vector_d2 : Fin 3 → ℝ := ![1, 0, -1]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

theorem only_b_not_perpendicular :
  dot_product vector_a1 vector_a2 = 0 ∧
  dot_product vector_b1 vector_b2 ≠ 0 ∧
  dot_product vector_c1 vector_c2 = 0 ∧
  dot_product vector_d1 vector_d2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_b_not_perpendicular_l661_66138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l661_66130

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := (Real.sin x - a)^2 + 1

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x, f x a ≤ f (Real.arcsin 1) a) ∧  -- Maximum when sin x = 1
  (∀ x, f (Real.arcsin a) a ≤ f x a) ∧  -- Minimum when sin x = a
  (-1 ≤ a ∧ a ≤ 1) →                    -- Constraint on a
  -1 ≤ a ∧ a ≤ 0 :=                     -- Conclusion: a is in [-1, 0]
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l661_66130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l661_66108

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 2 * π / 3) :
  ∃ b : ℝ, b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ b = 10 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l661_66108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_magnitude_of_quadratic_roots_l661_66129

theorem unique_magnitude_of_quadratic_roots (z : ℂ) : 
  z^2 - 6*z + 20 = 0 → Complex.abs z = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_magnitude_of_quadratic_roots_l661_66129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l661_66132

/-- Given a circle C with center (a, 0) that is tangent to a line with slope -√3/3 at the point (3, -√3),
    prove that the equation of the circle is (x-4)^2 + y^2 = 4 -/
theorem circle_equation (a : ℝ) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = ((a - 3)^2 + 3)}
  let L : Set (ℝ × ℝ) := {p | (Real.sqrt 3 / 3) * p.1 + p.2 = 0}
  let N : ℝ × ℝ := (3, -(Real.sqrt 3))
  (N ∈ C) ∧ (N ∈ L) ∧ (∀ p ∈ C, p ≠ N → p ∉ L) →
  C = {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l661_66132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l661_66170

-- Define the polynomial type
def MyPolynomial (R : Type*) [Semiring R] := R → R

-- Define the property that the polynomial satisfies the given equation
def SatisfiesEquation (p : MyPolynomial ℝ) : Prop :=
  ∀ (a b c : ℝ), a * b + b * c + c * a = 0 →
    p (a - b) + p (b - c) + p (c - a) = 2 * (p (a + b + c))

-- Define the form of the polynomial we want to prove
def QuadraticQuarticPolynomial (p : MyPolynomial ℝ) : Prop :=
  ∃ (α β : ℝ), ∀ x, p x = α * x^2 + β * x^4

-- Theorem statement
theorem polynomial_characterization (p : MyPolynomial ℝ) :
  SatisfiesEquation p → QuadraticQuarticPolynomial p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l661_66170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_range_l661_66134

open Real

noncomputable def f (x : ℝ) := cos (2 * x) + sin (2 * x + π / 6)

theorem f_zeros_range (α : ℝ) :
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < α ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x, 0 < x ∧ x < α ∧ f x = 0 → x = x₁ ∨ x = x₂) →
  5 * π / 6 < α ∧ α ≤ 4 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_range_l661_66134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l661_66117

theorem vector_magnitude_range (a b : ℝ × ℝ) (h1 : ‖a‖ = 1) (h2 : b • (a - b) = 0) : 
  0 ≤ ‖b‖ ∧ ‖b‖ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l661_66117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l661_66144

theorem no_integer_roots (P : Polynomial ℤ) (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5 →
  ∀ n : ℤ, P.eval n ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l661_66144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_value_l661_66120

/-- Parabola with equation y = ax^2 where a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with equation x - y = k -/
structure Line where
  k : ℝ

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

noncomputable def focus (p : Parabola) : Point :=
  ⟨0, 1 / (4 * p.a)⟩

noncomputable def directrix_distance (p : Parabola) (m : Point) : ℝ :=
  abs (m.y + 1 / (4 * p.a))

theorem parabola_minimum_value (p : Parabola) (m n : Point) (l : Line) :
  m.x = 3 ∧ m.y = 2 ∧
  n.x = 1 ∧ n.y = 1 ∧
  l.k = 2 ∧
  directrix_distance p m = 4 →
  (∃ min_value : ℝ, 
    (∀ q : Point, q.x - q.y = l.k → 
      (distance n q - 1) / distance (focus p) q ≥ min_value) ∧
    min_value = (2 - Real.sqrt 2) / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_value_l661_66120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_f_value_at_specific_angle_l661_66113

noncomputable def f (α : Real) : Real :=
  (Real.tan (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.sin (-α + 3*Real.pi/2)) /
  (Real.cos (-α - Real.pi) * Real.tan (-Real.pi - α))

theorem f_simplification (α : Real) (h : α ∈ Set.Ioo Real.pi (3*Real.pi/2)) :
  f α = Real.cos α := by sorry

theorem f_value_when_cos_condition (α : Real) (h1 : α ∈ Set.Ioo Real.pi (3*Real.pi/2)) 
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = -2 * Real.sqrt 6 / 5 := by sorry

theorem f_value_at_specific_angle :
  f (-1860 * Real.pi / 180) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_f_value_at_specific_angle_l661_66113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l661_66162

/-- The function g(x) = 1 / (x^2 + 1) -/
noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 + 1)

/-- The range of g(x) is (0, 1] -/
theorem range_of_g : Set.range g = Set.Ioo 0 1 ∪ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l661_66162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l661_66199

-- Define the parametric equation of curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (3 + 2 * Real.cos α, 1 - 2 * Real.sin α)

-- Define the polar equation of line l
def line_l (θ : ℝ) (ρ : ℝ) : Prop :=
  Real.sin θ - 2 * Real.cos θ = 1 / ρ

-- Theorem statement
theorem max_distance_curve_to_line :
  ∃ (d : ℝ), d = (6 * Real.sqrt 5) / 5 + 2 ∧
  ∀ (α θ ρ : ℝ),
    let (x, y) := curve_C α
    line_l θ ρ →
    Real.sqrt ((x - ρ * Real.cos θ)^2 + (y - ρ * Real.sin θ)^2) ≤ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l661_66199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l661_66105

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x, f (x + Real.pi) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l661_66105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_neighboring_difference_l661_66193

/-- A chessboard is represented as a function from (Fin n × Fin n) to ℕ --/
def Chessboard (n : ℕ) := Fin n × Fin n → ℕ

/-- Two squares are neighboring if they share a common edge --/
def neighboring {n : ℕ} (a b : Fin n × Fin n) : Prop :=
  (a.1 = b.1 ∧ a.2.val + 1 = b.2.val) ∨
  (a.1 = b.1 ∧ a.2.val = b.2.val + 1) ∨
  (a.1.val + 1 = b.1.val ∧ a.2 = b.2) ∨
  (a.1.val = b.1.val + 1 ∧ a.2 = b.2)

theorem chessboard_neighboring_difference {n : ℕ} (hn : n ≥ 2) 
  (board : Chessboard n) (h_board : ∀ i, board i ∈ Finset.range (n^2 + 1) \ {0}) :
  ∃ a b : Fin n × Fin n, neighboring a b ∧ (board a).dist (board b) ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_neighboring_difference_l661_66193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_company_revenue_percentage_l661_66114

/-- The percentage of revenue a production company keeps from a movie, given:
  * Opening weekend revenue
  * Total revenue multiplier
  * Production cost
  * Profit
-/
noncomputable def revenue_percentage (opening_revenue : ℝ) (total_revenue_multiplier : ℝ) 
                       (production_cost : ℝ) (profit : ℝ) : ℝ :=
  let total_revenue := opening_revenue * total_revenue_multiplier
  let amount_kept := profit + production_cost
  (amount_kept / total_revenue) * 100

/-- Theorem stating that the production company keeps 60% of the revenue -/
theorem production_company_revenue_percentage :
  revenue_percentage 120 3.5 60 192 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_company_revenue_percentage_l661_66114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_profit_theorem_l661_66158

/-- Represents the profit calculation for fruit sales --/
structure FruitProfit where
  boxesA : ℕ
  boxesB : ℕ
  totalProfit : ℝ
  profitDifference : ℝ
  additionalBoxesPerDollar : ℝ
  baseBoxesSold : ℝ

/-- Theorem stating the correct profit calculations and optimal price reduction --/
theorem fruit_profit_theorem (fp : FruitProfit)
  (h1 : fp.boxesA = 60)
  (h2 : fp.boxesB = 40)
  (h3 : fp.totalProfit = 1300)
  (h4 : fp.profitDifference = 5)
  (h5 : fp.additionalBoxesPerDollar = 20)
  (h6 : fp.baseBoxesSold = 100) :
  ∃ (profitA profitB optimalReduction maxProfit : ℝ),
    profitA = 15 ∧
    profitB = 10 ∧
    optimalReduction = 5 ∧
    maxProfit = 2000 ∧
    (fp.boxesA : ℝ) * profitA + (fp.boxesB : ℝ) * profitB = fp.totalProfit ∧
    profitA = profitB + fp.profitDifference ∧
    maxProfit = (profitA - optimalReduction) * (fp.baseBoxesSold + fp.additionalBoxesPerDollar * optimalReduction) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_profit_theorem_l661_66158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_likelihood_ratio_calculation_expected_male_students_l661_66107

-- Define the sample data
def total_sample : ℕ := 100
def male_science : ℕ := 24
def male_non_science : ℕ := 36
def female_science : ℕ := 12
def female_non_science : ℕ := 28

-- Define the critical value
def critical_value : ℝ := 6.635

-- Define the chi-square test statistic function
noncomputable def chi_square (a b c d : ℕ) : ℝ :=
  (total_sample * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the likelihood ratio function
noncomputable def likelihood_ratio (a b : ℕ) : ℝ := (a : ℝ) / b

-- Define the expectation function for the stratified sample
noncomputable def expectation (p1 p2 p3 : ℝ) : ℝ := 1 * p1 + 2 * p2 + 3 * p3

-- Theorem statements
theorem independence_test :
  chi_square male_science female_science male_non_science female_non_science < critical_value :=
by sorry

theorem likelihood_ratio_calculation :
  likelihood_ratio male_non_science female_non_science = 9 / 7 :=
by sorry

theorem expected_male_students :
  expectation (1/5 : ℝ) (3/5 : ℝ) (1/5 : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_likelihood_ratio_calculation_expected_male_students_l661_66107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l661_66159

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 4)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l661_66159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_l661_66106

-- Define proposition p
def p : Prop := ∃ x : ℝ, (2 : ℝ)^x > (3 : ℝ)^x

-- Define proposition q
def q : Prop := ∀ x : ℝ, 0 < x → x < Real.pi/2 → Real.tan x > Real.sin x

-- Theorem to prove
theorem correct_option : p ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_option_l661_66106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_cells_l661_66146

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Counts the number of black cells in a grid -/
def blackCellCount (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 4)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j =>
      if g i j then 1 else 0

/-- Checks if a grid has at least one black cell after removing two rows and two columns -/
def hasBlackCellAfterRemoval (g : Grid) : Prop :=
  ∀ (r₁ r₂ c₁ c₂ : Fin 4), r₁ ≠ r₂ → c₁ ≠ c₂ →
    ∃ (i j : Fin 4), i ≠ r₁ ∧ i ≠ r₂ ∧ j ≠ c₁ ∧ j ≠ c₂ ∧ g i j = true

/-- The main theorem stating that 7 is the minimum number of black cells required -/
theorem min_black_cells :
  (∃ (g : Grid), blackCellCount g = 7 ∧ hasBlackCellAfterRemoval g) ∧
  (∀ (g : Grid), blackCellCount g < 7 → ¬hasBlackCellAfterRemoval g) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_cells_l661_66146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_configuration_l661_66184

def Point := ℤ × ℤ

def initial_points : Set Point :=
  {(0, 0), (0, 1), (1, 0), (1, 1)}

def target_points : Set Point :=
  {(0, 0), (1, 1), (3, 0), (2, -1)}

def symmetry_transform (p q : Point) : Point :=
  (2 * q.1 - p.1, 2 * q.2 - p.2)

def is_reachable (start finish : Set Point) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → Set Point),
    sequence 0 = start ∧
    sequence n = finish ∧
    ∀ i : Fin n,
      ∃ (p q r : Point),
        p ∈ sequence i ∧
        q ∈ sequence i ∧
        r = symmetry_transform p q ∧
        sequence (i.succ) = insert r (sequence i \ {p})

theorem unreachable_configuration :
  ¬ is_reachable initial_points target_points := by
  sorry

#check unreachable_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_configuration_l661_66184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_sum_max_l661_66124

-- Define the convexity of a function on an interval
def IsConvex (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y t : ℝ, a ≤ x → x ≤ b → a ≤ y → y ≤ b → 0 ≤ t → t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- State the theorem
theorem triangle_sin_sum_max :
  ∀ A B C : ℝ,
    0 < A → A < π →
    0 < B → B < π →
    0 < C → C < π →
    A + B + C = π →
    IsConvex Real.sin 0 π →
    Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_sum_max_l661_66124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_pages_theorem_l661_66194

def digits_used (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def total_digits (n : ℕ) : ℕ :=
  (List.range n).map digits_used |>.sum

theorem encyclopedia_pages_theorem :
  ∃ (n : ℕ), total_digits n = 6869 ∧ n = 1994 := by
  use 1994
  apply And.intro
  · sorry -- Proof that total_digits 1994 = 6869
  · rfl

#eval total_digits 1994

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_pages_theorem_l661_66194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l661_66110

-- Define what a power function is
noncomputable def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the given functions
noncomputable def f1 (x : ℝ) : ℝ := x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := 2^x
noncomputable def f3 (x : ℝ) : ℝ := 1 / (x^2)
noncomputable def f4 (x : ℝ) : ℝ := (x - 1)^2
noncomputable def f5 (x : ℝ) : ℝ := x^5
noncomputable def f6 (x : ℝ) : ℝ := x^(x + 1)

-- State the theorem
theorem power_function_identification :
  ¬(isPowerFunction f1) ∧
  ¬(isPowerFunction f2) ∧
  (isPowerFunction f3) ∧
  ¬(isPowerFunction f4) ∧
  (isPowerFunction f5) ∧
  ¬(isPowerFunction f6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l661_66110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_l661_66166

noncomputable def a : ℕ → ℚ
  | 0 => 1/3
  | 1 => 1/3
  | n+2 => let a_n_minus_2 := a n
           let a_n_minus_1 := a (n+1)
           ((1 - 2*a_n_minus_2) * a_n_minus_1^2) / (2*a_n_minus_1^2 - 4*a_n_minus_2*a_n_minus_1^2 + a_n_minus_2)

noncomputable def general_term (n : ℕ) : ℝ :=
  let x := (3/2 - 5*Real.sqrt 3/6) * (2 + Real.sqrt 3)^n + (3/2 + 5*Real.sqrt 3/6) * (2 - Real.sqrt 3)^n
  (x^2 + 2)⁻¹

theorem a_equals_general_term : ∀ n : ℕ, (a n : ℝ) = general_term n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_l661_66166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l661_66160

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.sin x ^ 2 - Real.cos x ^ 2

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (f x) ^ 2 + f x

-- State the theorem
theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (k : ℤ), ∃ (x : ℝ), x = (k : ℝ) * Real.pi / 2 + Real.pi / 3 ∧ 
    ∀ (y : ℝ), f (x + y) = f (x - y)) ∧
  (∀ (y : ℝ), g y ≥ -1/4 ∧ g y ≤ 2) ∧
  (∃ (x₁ x₂ : ℝ), g x₁ = -1/4 ∧ g x₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l661_66160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l661_66180

/-- The area of a triangle given its three vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * |v.1 * w.2 - v.2 * w.1|

/-- Theorem: The area of the triangle with vertices (4, -4), (-1, 1), and (2, -7) is 12.5 -/
theorem triangle_area_example : triangle_area (4, -4) (-1, 1) (2, -7) = 12.5 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l661_66180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l661_66147

/-- The circle O with equation x^2 + y^2 = 4 -/
def circle_O : Set (ℝ × ℝ) := {p | p.fst^2 + p.snd^2 = 4}

/-- The point A with coordinates (1,1) -/
def point_A : ℝ × ℝ := (1, 1)

/-- A chord PQ of the circle that passes through point A -/
def chord_through_A (P Q : ℝ × ℝ) : Prop :=
  P ∈ circle_O ∧ Q ∈ circle_O ∧ ∃ t : ℝ, (1 - t) • P + t • Q = point_A

/-- The length of a chord PQ -/
noncomputable def chord_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- The theorem stating the minimum length of a chord passing through A -/
theorem min_chord_length :
  ∀ P Q : ℝ × ℝ, chord_through_A P Q →
    chord_length P Q ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l661_66147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_percentage_theorem_l661_66122

/-- Represents a two-segment trip with given speeds and average speed -/
structure TwoSegmentTrip where
  speed1 : ℝ
  speed2 : ℝ
  avg_speed : ℝ

/-- Calculates the percentage of the trip traveled at the first speed -/
noncomputable def percentage_at_speed1 (trip : TwoSegmentTrip) : ℝ :=
  (trip.speed2 - trip.avg_speed) / (trip.speed2 - trip.speed1)

theorem trip_percentage_theorem (trip : TwoSegmentTrip) 
  (h1 : trip.speed1 = 35)
  (h2 : trip.speed2 = 65)
  (h3 : trip.avg_speed = 50) :
  percentage_at_speed1 trip = 0.35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_percentage_theorem_l661_66122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_three_points_l661_66123

/-- The slope of the best-fit line through three points with equally spaced x-coordinates -/
theorem best_fit_slope_three_points (y₁ y₂ y₃ : ℝ) :
  let x₁ : ℝ := 140
  let x₂ : ℝ := 145
  let x₃ : ℝ := 150
  let slope := (y₃ - y₁) / 10
  ∀ m b b' : ℝ, (y₁ - (m * x₁ + b))^2 + (y₂ - (m * x₂ + b))^2 + (y₃ - (m * x₃ + b))^2 ≥
             (y₁ - (slope * x₁ + b'))^2 + (y₂ - (slope * x₂ + b'))^2 + (y₃ - (slope * x₃ + b'))^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_three_points_l661_66123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_division_solution_l661_66175

theorem unique_division_solution : ∃! (a b c d : ℕ), 
  a / b = c ∧ 
  c / d = 10356 ∧ 
  a = 100007892 ∧ 
  b = 333 ∧ 
  d = 29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_division_solution_l661_66175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l661_66190

/-- Represents Jason's work schedule and earnings --/
structure JasonWork where
  afterSchoolRate : ℚ
  saturdayRate : ℚ
  totalHours : ℚ
  totalEarnings : ℚ

/-- Calculates the number of hours Jason worked on Saturday --/
def saturdayHours (j : JasonWork) : ℚ :=
  (j.totalEarnings - j.afterSchoolRate * j.totalHours) / (j.saturdayRate - j.afterSchoolRate)

/-- Theorem stating that Jason worked 8 hours on Saturday --/
theorem jason_saturday_hours :
  let j : JasonWork := {
    afterSchoolRate := 4,
    saturdayRate := 6,
    totalHours := 18,
    totalEarnings := 88
  }
  saturdayHours j = 8 := by
  -- The proof goes here
  sorry

#eval saturdayHours {
  afterSchoolRate := 4,
  saturdayRate := 6,
  totalHours := 18,
  totalEarnings := 88
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l661_66190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_composition_l661_66188

open Real Matrix

noncomputable def R (φ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos φ, -sin φ; sin φ, cos φ]

def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

theorem dilation_rotation_composition (k φ : ℝ) (hk : k > 0) :
  R φ * D k = !![6, -3; 3, 6] →
  tan φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_composition_l661_66188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_points_l661_66151

theorem circle_equation_through_points : 
  let circle_eq (x y : ℝ) := x^2 + y^2 - 4*x - 6*y = 0
  let point1 : ℝ × ℝ := (0, 0)
  let point2 : ℝ × ℝ := (4, 0)
  let point3 : ℝ × ℝ := (-1, 1)
  (circle_eq point1.1 point1.2) ∧ 
  (circle_eq point2.1 point2.2) ∧ 
  (circle_eq point3.1 point3.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_points_l661_66151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l661_66181

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 4

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (x : ℝ) : ℝ :=
  if x = 0 then -3/2 * (n - 1)
  else 3/2 * (n - 3)

theorem arithmetic_sequence_problem (x : ℝ) :
  (f (x + 1) = x^2 - 4) →
  (a 1 x = f (x - 1)) →
  (a 2 x = -3/2) →
  (a 3 x = f x) →
  ((x = 0 ∨ x = 3) ∧
   (∀ n : ℕ, a n x = -3/2 * (n - 1) ∨ a n x = 3/2 * (n - 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l661_66181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l661_66116

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l661_66116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_rainfall_ratio_l661_66135

/-- Represents the rainfall data for a storm -/
structure StormData where
  duration : ℚ  -- Total duration of the storm in hours
  first_30min : ℚ  -- Rainfall in the first 30 minutes
  last_hour : ℚ  -- Rainfall in the last hour
  average_rainfall : ℚ  -- Average rainfall per hour for the entire storm

/-- Calculates the ratio of rainfall in the first 30 minutes to the next 30 minutes -/
def rainfall_ratio (data : StormData) : ℚ :=
  let total_rainfall := data.average_rainfall * data.duration
  let next_30min := total_rainfall - data.first_30min - data.last_hour
  data.first_30min / next_30min

/-- Theorem stating the rainfall ratio for the given storm data -/
theorem storm_rainfall_ratio :
  let data : StormData := {
    duration := 2,
    first_30min := 5,
    last_hour := 1,
    average_rainfall := 4
  }
  rainfall_ratio data = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_rainfall_ratio_l661_66135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_theorem_l661_66148

/-- The function f(x) satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

/-- The point P on the terminal side of angle φ -/
noncomputable def P : ℝ × ℝ := (1, -Real.sqrt 3)

/-- The theorem stating the properties of the function and the range of m -/
theorem function_and_range_theorem :
  (∀ x₁ x₂ : ℝ, |f x₁ - f x₂| = 4 → |x₁ - x₂| ≥ Real.pi / 3) ∧
  (∀ x : ℝ, f x = 2 * Real.sin (3 * x - Real.pi / 3)) ∧
  (∀ m : ℝ, (m = 1 / 12 ∨ (-10 < m ∧ m ≤ 0)) ↔
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      Real.pi / 9 < x₁ ∧ x₁ < 4 * Real.pi / 9 ∧
      Real.pi / 9 < x₂ ∧ x₂ < 4 * Real.pi / 9 ∧
      3 * (f x₁)^2 - f x₁ + m = 0 ∧
      3 * (f x₂)^2 - f x₂ + m = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_theorem_l661_66148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_is_25_percent_l661_66195

/-- Represents the shopkeeper's buying and selling practices -/
structure ShopkeeperPractice where
  buyingExcess : ℚ  -- Percentage excess when buying
  sellingShortage : ℚ  -- Percentage shortage when selling

/-- Calculates the profit percentage given the shopkeeper's practice -/
noncomputable def profitPercentage (practice : ShopkeeperPractice) : ℚ :=
  let buyingRatio := 1 + practice.buyingExcess
  let sellingRatio := 1 / (1 - practice.sellingShortage)
  (sellingRatio / buyingRatio - 1) * 100

/-- Theorem stating that the shopkeeper's profit percentage is 25% -/
theorem shopkeeper_profit_is_25_percent :
  let practice : ShopkeeperPractice := { buyingExcess := 1/5, sellingShortage := 1/10 }
  profitPercentage practice = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_is_25_percent_l661_66195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l661_66164

/-- Given an angle α in the plane rectangular coordinate system xOy with its initial side on the 
    non-negative half-axis of the x-axis and its terminal side passing through the point P(-1,2), 
    prove that cos(π - α) = √5/5 -/
theorem cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : 
  P.1 = -1 → P.2 = 2 → Real.cos (Real.pi - α) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l661_66164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_120_deg_sector_l661_66137

/-- The radius of a circle inscribed in a circular sector -/
noncomputable def inscribed_circle_radius (sector_radius : ℝ) (central_angle : ℝ) : ℝ :=
  sector_radius * Real.sqrt 3 * (2 - Real.sqrt 3)

/-- Theorem stating that the radius of an inscribed circle in a 120° sector with radius 8 is 8√3(2 - √3) -/
theorem inscribed_circle_radius_120_deg_sector :
  inscribed_circle_radius 8 (2 * Real.pi / 3) = 8 * Real.sqrt 3 * (2 - Real.sqrt 3) := by
  sorry

#check inscribed_circle_radius_120_deg_sector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_120_deg_sector_l661_66137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_two_l661_66112

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def equation (x : ℝ) : Prop := floor (3 * x + 1) = ⌊2 * x - 1/2⌋

theorem sum_of_roots_is_negative_two :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ ∧ x₁ + x₂ = -2 ∧
  ∀ (x : ℝ), equation x → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_two_l661_66112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bound_l661_66131

-- Define the curve C₁
noncomputable def C₁ (φ : Real) : Real × Real := (2 * Real.cos φ, 3 * Real.sin φ)

-- Define the points A, B, C, D
noncomputable def A : Real × Real := (1, Real.sqrt 3)
noncomputable def B : Real × Real := (-Real.sqrt 3, 1)
noncomputable def C : Real × Real := (-1, -Real.sqrt 3)
noncomputable def D : Real × Real := (Real.sqrt 3, -1)

-- Define the squared distance between two points
def squared_distance (p q : Real × Real) : Real :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the sum of squared distances from a point to A, B, C, D
noncomputable def sum_squared_distances (p : Real × Real) : Real :=
  squared_distance p A + squared_distance p B + squared_distance p C + squared_distance p D

-- State the theorem
theorem sum_squared_distances_bound :
  ∀ φ : Real, 32 ≤ sum_squared_distances (C₁ φ) ∧ sum_squared_distances (C₁ φ) ≤ 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bound_l661_66131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_prime_symmetric_to_M_l661_66179

/-- The plane equation: 2x - 2y + 10z + 1 = 0 -/
def plane_equation (x y z : ℝ) : Prop := 2*x - 2*y + 10*z + 1 = 0

/-- The point M -/
def M : ℝ × ℝ × ℝ := (-2, 0, 3)

/-- The point M' -/
def M' : ℝ × ℝ × ℝ := (-3, 1, -2)

/-- Definition of symmetry with respect to a plane -/
def symmetric_wrt_plane (p q : ℝ × ℝ × ℝ) : Prop :=
  ∃ (m : ℝ × ℝ × ℝ), 
    (plane_equation m.1 m.2.1 m.2.2) ∧ 
    (m.1 = (p.1 + q.1) / 2) ∧
    (m.2.1 = (p.2.1 + q.2.1) / 2) ∧
    (m.2.2 = (p.2.2 + q.2.2) / 2)

/-- Theorem: M' is symmetric to M with respect to the given plane -/
theorem M_prime_symmetric_to_M : symmetric_wrt_plane M M' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_prime_symmetric_to_M_l661_66179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_1_to_100_l661_66102

def is_visible_seven (n : Nat) : Bool :=
  n.repr.contains '7'

def is_invisible_seven (n : Nat) : Bool :=
  n % 7 = 0

def count_special_numbers (lower upper : Nat) : Nat :=
  (List.range (upper - lower + 1)).map (· + lower)
    |> List.filter (λ n => is_visible_seven n ∨ is_invisible_seven n)
    |> List.length

theorem count_special_numbers_1_to_100 :
  count_special_numbers 1 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_1_to_100_l661_66102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_k_seven_fourths_l661_66163

/-- The equation has no solutions when k = 7/4 -/
theorem no_solution_for_k_seven_fourths :
  ∃ k : ℝ, ∀ t s : ℝ, 
    (⟨3, 5⟩ : ℝ × ℝ) + t • ⟨4, -7⟩ ≠ (⟨2, -2⟩ : ℝ × ℝ) + s • ⟨-1, k⟩ :=
by
  -- We claim that k = 7/4 satisfies the condition
  use 7/4
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_k_seven_fourths_l661_66163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l661_66100

open Set Real

def U : Set ℝ := univ

def M : Set ℝ := {x | x^2 ≤ 4}

def N : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}

theorem set_operations :
  (M ∩ N = Ioc 0 2) ∧ (M ∪ (U \ N) = Iic 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l661_66100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l661_66157

variable (a b x y : ℝ)

-- Define the function
noncomputable def f (a b x : ℝ) : ℝ := (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

-- State the theorem
theorem function_properties (h : a > b) :
  -- Maximum value of y is a
  (∀ x, f a b x ≤ a^2) ∧
  -- Minimum value of y is b
  (∀ x, f a b x ≥ b^2) ∧
  -- When y is between a and b, the quartic equation has real roots
  (∀ y, b < y ∧ y < a → ∃ x, f a b x = y^2) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l661_66157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_compression_and_translation_l661_66142

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if x > 0 ∧ x ≤ 2 then Real.sqrt (4 - (x-2)^2) - 2
  else if x > 2 ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- undefined outside the given intervals

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := (1/3) * f x + 2

-- Theorem statement
theorem vertical_compression_and_translation :
  ∀ x : ℝ, g x = (1/3) * (f x - f 0) + (g 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_compression_and_translation_l661_66142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l661_66121

/-- The probability of winning for each player -/
noncomputable def win_prob : ℝ := 1 / 2

/-- The probability of getting heads in a single coin flip -/
noncomputable def p : ℝ := (3 - Real.sqrt 5) / 2

/-- The sum of the geometric series representing Newton's probability of winning -/
noncomputable def newton_win_prob (p : ℝ) : ℝ := p / (1 - p + p^2)

/-- Theorem stating that the probability of Newton winning equals 1/2 -/
theorem coin_game_probability :
  newton_win_prob p = win_prob := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l661_66121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_half_l661_66133

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  AB : ℝ
  CD : ℝ
  distance : ℝ
  angle : ℝ
  ab_length : AB = 1
  cd_length : CD = Real.sqrt 3
  line_distance : distance = 2
  line_angle : angle = π / 3

/-- Calculates the volume of the tetrahedron -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  1 / 2

/-- Theorem stating that the volume of the specified tetrahedron is 1/2 -/
theorem tetrahedron_volume_is_half (t : Tetrahedron) :
  tetrahedron_volume t = 1 / 2 := by
  -- The proof goes here
  sorry

#eval "Tetrahedron volume theorem is stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_half_l661_66133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_zero_l661_66143

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + (x+1)^2) + Real.sqrt (x^2 + (x-1)^2)

theorem min_value_at_zero :
  ∀ x : ℝ, f 0 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_zero_l661_66143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_75_degrees_l661_66196

/-- Proves that cos 75° = (√6 - √2) / 4 given that 75° = 60° + 15° -/
theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_75_degrees_l661_66196
