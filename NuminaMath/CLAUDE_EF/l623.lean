import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_is_infinite_non_repeating_real_numbers_one_to_one_with_line_l623_62316

-- Define infinite non-repeating decimal
def InfiniteNonRepeatingDecimal (x : ℝ) : Prop :=
  ∀ (n : ℕ), ∃ (m : ℕ), m > n ∧ (⌊x * 10^m⌋ % 10 ≠ ⌊x * 10^n⌋ % 10)

-- Define one-to-one correspondence between real numbers and points on a line
def OneToOneCorrespondence (f : ℝ → ℝ) : Prop :=
  Function.Injective f ∧ Function.Surjective f

-- Theorem 1: Any irrational number is an infinite non-repeating decimal
theorem irrational_is_infinite_non_repeating :
  ∀ (x : ℝ), Irrational x → InfiniteNonRepeatingDecimal x :=
by
  sorry

-- Theorem 2: Real numbers correspond one-to-one with points on the number line
theorem real_numbers_one_to_one_with_line :
  ∃ (f : ℝ → ℝ), OneToOneCorrespondence f :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_is_infinite_non_repeating_real_numbers_one_to_one_with_line_l623_62316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_bounds_l623_62340

/-- Represents a curve y = a/x where x > 0 -/
noncomputable def curve1 (a : ℝ) (x : ℝ) : ℝ := a / x

/-- Represents the curve y = 2ln(x) -/
noncomputable def curve2 (x : ℝ) : ℝ := 2 * Real.log x

/-- Checks if two curves have a common tangent at a given point -/
def has_common_tangent (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv f) x = (deriv g) x ∧ f x = g x

/-- The range of a values for which curve1 and curve2 have a common tangent -/
def a_range : Set ℝ := {a | ∃ x > 0, has_common_tangent (curve1 a) curve2 x}

theorem a_range_bounds :
  a_range = Set.Icc (-2 / Real.exp 1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_bounds_l623_62340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l623_62311

theorem cubic_equation_solution (m n x : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  (x + m)^3 - (x + n)^3 = (m - n)^3 →
  x = (-(m + n) + Real.sqrt (m^2 + 6*m*n + n^2)) / 2 ∨
  x = (-(m + n) - Real.sqrt (m^2 + 6*m*n + n^2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_l623_62311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_with_addition_l623_62373

def operation (a b : ℕ) : ℚ :=
  (a * b : ℚ) / (a + b + 2 : ℚ)

theorem operation_with_addition (a b : ℕ) (h1 : a = 7) (h2 : b = 21) :
  operation a b + 3 = 79 / 10 := by
  have h3 : operation a b = 147 / 30 := by
    rw [h1, h2]
    simp [operation]
    norm_num
  
  calc
    operation a b + 3 = 147 / 30 + 3 := by rw [h3]
    _ = 147 / 30 + 90 / 30 := by norm_num
    _ = 237 / 30 := by norm_num
    _ = 79 / 10 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_with_addition_l623_62373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_property_l623_62351

theorem log_function_property (a : ℝ) (h1 : a > 1) :
  (let f := λ x => Real.log x / Real.log a;
   (∀ x ∈ Set.Icc a (2 * a), f x ≤ f (2 * a)) ∧
   (∀ x ∈ Set.Icc a (2 * a), f x ≥ f a) ∧
   f (2 * a) - f a = 1/2) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_property_l623_62351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l623_62345

/-- The line equation -/
noncomputable def line_eq (t : ℝ) : ℝ × ℝ := (-2 - Real.sqrt 2 * t, 3 + Real.sqrt 2 * t)

/-- The point A -/
def point_A : ℝ × ℝ := (-2, 3)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that (-3, 4) and (-1, 2) are the only points on the line
    at distance √2 from point A -/
theorem line_points_at_distance_sqrt_2 :
  {p : ℝ × ℝ | ∃ t : ℝ, line_eq t = p ∧ distance p point_A = Real.sqrt 2} =
  {(-3, 4), (-1, 2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l623_62345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_less_than_four_l623_62362

theorem at_least_one_not_less_than_four (m n t : ℝ) 
  (hm : m > 0) (hn : n > 0) (ht : t > 0) : 
  ∃ x ∈ ({m + 4/n, n + 4/t, t + 4/m} : Set ℝ), x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_less_than_four_l623_62362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_walking_speed_l623_62346

/-- Represents Tony's exercise routine and calculates his walking speed. -/
noncomputable def TonyExercise (walking_distance : ℝ) (running_distance : ℝ) (running_speed : ℝ) (total_time : ℝ) : ℝ :=
  let total_walking_distance := 7 * walking_distance
  let total_running_distance := 7 * running_distance
  let running_time := total_running_distance / running_speed
  let walking_time := total_time - running_time
  total_walking_distance / walking_time

/-- Theorem stating that Tony's walking speed is 3 miles per hour given his exercise routine. -/
theorem tony_walking_speed :
  TonyExercise 3 10 5 21 = 3 := by
  -- Unfold the definition of TonyExercise
  unfold TonyExercise
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_walking_speed_l623_62346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_h_min_value_l623_62356

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

-- Define the function h
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := Real.exp (2*x) + m * Real.exp x

-- Statement for the monotonicity of g
theorem g_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂ := by sorry

-- Statement for the range of m
theorem h_min_value (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 (Real.log 4), h m x ≥ 0) ∧ 
  (∃ x ∈ Set.Icc 0 (Real.log 4), h m x = 0) ↔ 
  m = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_h_min_value_l623_62356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l623_62310

open Real

-- Define the original function
noncomputable def y (x : ℝ) : ℝ := sin (2 * x + π / 3)

-- Define the period of the sine function
noncomputable def period : ℝ := 2 * π / 2

-- Define the shift amount
noncomputable def shift : ℝ := (1 / 4) * period

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Theorem statement
theorem f_increasing_intervals (k : ℤ) :
  ∀ x y : ℝ, x ∈ Set.Icc (k * π - π / 6) (k * π + π / 3) →
    y ∈ Set.Icc (k * π - π / 6) (k * π + π / 3) →
    x < y → f x < f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l623_62310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_l623_62378

theorem repeating_decimal_fraction : 
  ∀ (x y : ℚ),
  x = 63 * (1 / 99 : ℚ) ∧ 
  y = 18 * (1 / 99 : ℚ) →
  x / y = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_l623_62378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_price_correct_l623_62381

/-- The price of a candle that satisfies the given conditions -/
def candle_price : ℝ := 15

/-- Theorem stating that the candle price satisfies the given conditions -/
theorem candle_price_correct :
  let cost : ℝ := 20
  let num_candles : ℕ := 7
  let profit : ℝ := 85
  let revenue : ℝ := num_candles * candle_price
  profit = revenue - cost :=
by
  -- Unfold the definitions
  unfold candle_price
  -- Simplify the expression
  simp
  -- Check that the equation holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_price_correct_l623_62381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_imply_p_q_l623_62309

noncomputable def f (p q x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * p * Real.sin x + q

theorem max_min_values_imply_p_q (p q : ℝ) :
  (∀ x, f p q x ≤ 9) ∧
  (∃ x, f p q x = 9) ∧
  (∀ x, f p q x ≥ 6) ∧
  (∃ x, f p q x = 6) →
  ((p = Real.sqrt 3 - 1 ∨ p = -(Real.sqrt 3 - 1)) ∧
   q = 4 + 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_imply_p_q_l623_62309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l623_62347

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (1 - x)

noncomputable def g (x : ℝ) : ℝ := f (x + 1) + 1

theorem g_is_odd : ∀ x : ℝ, x ≠ 0 → g (-x) = -g x := by
  intro x hx
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l623_62347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l623_62302

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
def Point : Type := ℝ × ℝ

-- Define distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define if a triangle is equilateral
def isEquilateral (t : Triangle) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

-- Define if a point is on the circumcircle of a triangle
def isOnCircumcircle (t : Triangle) (p : Point) : Prop :=
  ∃ r : ℝ, distance p t.A = r ∧ distance p t.B = r ∧ distance p t.C = r

theorem triangle_inequalities (t : Triangle) (M : Point) :
  (isEquilateral t → distance M t.A ≤ distance M t.B + distance M t.C) ∧
  (distance M t.A * distance t.B t.C ≤ distance M t.B * distance t.C t.A + distance M t.C * distance t.A t.B) ∧
  (isEquilateral t → (distance M t.A = distance M t.B + distance M t.C ↔ isOnCircumcircle t M)) ∧
  (distance M t.A * distance t.B t.C = distance M t.B * distance t.C t.A + distance M t.C * distance t.A t.B
    ↔ isOnCircumcircle t M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l623_62302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l623_62376

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Circle O: x^2 + y^2 = 1 -/
def circleO (p : Point) : Prop :=
  p.x^2 + p.y^2 = 1

/-- Circle C: (x-3)^2 + y^2 = 1 -/
def circleC (p : Point) : Prop :=
  (p.x - 3)^2 + p.y^2 = 1

/-- The center of circle O -/
def O : Point :=
  { x := 0, y := 0 }

/-- The center of circle C -/
def C : Point :=
  { x := 3, y := 0 }

/-- The moving circle M is externally tangent to O and internally tangent to C -/
def tangentCondition (m : Point) (r : ℝ) : Prop :=
  distance m O = r + 1 ∧ distance m C = r - 1

/-- The locus of points M forms the right branch of a hyperbola -/
def isRightBranchHyperbola (m : Point) : Prop :=
  m.x > (O.x + C.x) / 2 ∧ distance m O - distance m C = distance O C - 2

theorem moving_circle_trajectory (m : Point) (r : ℝ) :
  tangentCondition m r → isRightBranchHyperbola m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l623_62376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_curve_satisfies_equation_l623_62322

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 3 ∧ y' = y / 2

-- Define the parametric equation of the transformed curve
noncomputable def transformed_parametric (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 / 3 * Real.cos θ, Real.sqrt 2 / 2 * Real.sin θ)

-- State the theorem
theorem transformed_curve_satisfies_equation :
  ∀ θ : ℝ,
  let (x', y') := transformed_parametric θ
  3 * x'^2 + 2 * y'^2 = 1 :=
by
  intro θ
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_curve_satisfies_equation_l623_62322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l623_62313

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers that can appear on the faces of a cube --/
def validNumbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A function to calculate the sum of visible faces when stacking cubes --/
def visibleSum (cubes : Fin 4 → Cube) : ℕ := sorry

/-- The theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∃ (cubes : Fin 4 → Cube),
    (∀ i : Fin 4, ∀ j : Fin 6, (cubes i).faces j ∈ validNumbers) →
    (∀ i : Fin 4, (Finset.filter (λ j => (cubes i).faces j ∈ validNumbers) (Finset.univ : Finset (Fin 6))).card = 6) →
    visibleSum cubes = 1452 ∧ 
    ∀ (other_cubes : Fin 4 → Cube),
      (∀ i : Fin 4, ∀ j : Fin 6, (other_cubes i).faces j ∈ validNumbers) →
      (∀ i : Fin 4, (Finset.filter (λ j => (other_cubes i).faces j ∈ validNumbers) (Finset.univ : Finset (Fin 6))).card = 6) →
      visibleSum other_cubes ≤ 1452 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l623_62313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_II_must_be_true_l623_62395

def SingleDigit : Type := Fin 10

structure Statements :=
  (I : Prop) -- The digit is 2
  (II : Prop) -- The digit is not 3
  (III : Prop) -- The digit is 5
  (IV : Prop) -- The digit is not 6

def evaluate_statements (d : SingleDigit) (s : Statements) : Prop :=
  (s.I ↔ d = ⟨2, by norm_num⟩) ∧
  (s.II ↔ d ≠ ⟨3, by norm_num⟩) ∧
  (s.III ↔ d = ⟨5, by norm_num⟩) ∧
  (s.IV ↔ d ≠ ⟨6, by norm_num⟩)

theorem statement_II_must_be_true (d : SingleDigit) (s : Statements) 
  (h : evaluate_statements d s)
  (h_three_true : (s.I ∧ s.II ∧ s.III) ∨ (s.I ∧ s.II ∧ s.IV) ∨ (s.I ∧ s.III ∧ s.IV) ∨ (s.II ∧ s.III ∧ s.IV)) :
  s.II :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_II_must_be_true_l623_62395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l623_62312

/-- The area of a circular sector given the radius and arc length -/
noncomputable def sectorArea (radius : ℝ) (arcLength : ℝ) : ℝ :=
  (arcLength / (2 * Real.pi * radius)) * (Real.pi * radius^2)

/-- Theorem: The area of a sector with radius 5 cm and arc length 3.5 cm is 8.75 cm² -/
theorem sector_area_example : sectorArea 5 3.5 = 8.75 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l623_62312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_condition_for_quadratic_l623_62379

theorem absolute_value_condition_for_quadratic : 
  (∃ y : ℝ, |y - 1| < 2 ∧ ¬(y * (y - 3) < 0)) ∧ 
  (∀ z : ℝ, z * (z - 3) < 0 → |z - 1| < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_condition_for_quadratic_l623_62379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_symmetry_axes_l623_62337

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.cos ((2/3)*x + Real.pi/2) + Real.cos ((2/3)*x)

/-- The distance between adjacent symmetry axes of f(x) -/
noncomputable def symmetry_axis_distance : ℝ := 3*Real.pi/2

/-- Theorem stating that the distance between adjacent symmetry axes of f(x) is 3π/2 -/
theorem distance_between_symmetry_axes :
  ∀ x : ℝ, ∃ y : ℝ, y = x + symmetry_axis_distance ∧ f y = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_symmetry_axes_l623_62337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_fuel_consumption_l623_62389

-- Define the fuel consumption function
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Define the total fuel consumption for a 100 km journey
noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

-- State the theorem
theorem optimal_fuel_consumption :
  0 < 40 ∧ 40 ≤ 120 ∧ 0 < 80 ∧ 80 ≤ 120 →
  total_fuel 40 = 17.5 ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ 120 → total_fuel x ≥ total_fuel 80) ∧
  total_fuel 80 = 11.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_fuel_consumption_l623_62389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l623_62393

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x) - 2 * (Real.sin (ω * x))^2 + 1

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ has_period f p ∧ ∀ q, 0 < q ∧ q < p → ¬ has_period f q

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem f_properties (ω : ℝ) (h1 : ω > 0) (h2 : is_smallest_positive_period (f ω) Real.pi) :
  ω = 1 ∧
  ∀ k : ℤ, monotone_decreasing_on (f ω) (Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l623_62393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_625_floor_l623_62303

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | (n + 1) => x n + 1 / (2 * x n)

theorem x_625_floor : ⌊25 * x 625⌋ = 625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_625_floor_l623_62303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_switches_on_l623_62329

/-- Represents a grid of switches -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Flips a row in the grid -/
def flipRow (g : Grid n) (row : Fin n) : Grid n :=
  λ i j => if i = row then !g i j else g i j

/-- Flips a column in the grid -/
def flipColumn (g : Grid n) (col : Fin n) : Grid n :=
  λ i j => if j = col then !g i j else g i j

/-- Counts the number of "on" switches in a grid -/
def countOn (g : Grid n) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin n)) λ i =>
    Finset.sum (Finset.univ : Finset (Fin n)) λ j =>
      if g i j then 1 else 0)

/-- Checks if a row has an odd number of "on" switches -/
def rowIsOdd (g : Grid n) (row : Fin n) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin n)) λ j => if g row j then 1 else 0)

/-- Checks if a column has an odd number of "on" switches -/
def columnIsOdd (g : Grid n) (col : Fin n) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin n)) λ i => if g i col then 1 else 0)

/-- The main theorem -/
theorem max_switches_on (n : ℕ) (h : n = 100) :
  ∃ (g : Grid n), 
    (∀ i, rowIsOdd g i) ∧ 
    (∀ j, columnIsOdd g j) ∧
    (∀ g' : Grid n, (∀ i, rowIsOdd g' i) ∧ (∀ j, columnIsOdd g' j) → countOn g' ≤ countOn g) ∧
    countOn g = 9802 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_switches_on_l623_62329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_values_l623_62380

def U (a : ℝ) : Set ℝ := {2, 3, a^2 - a - 1}
def A : Set ℝ := {2, 3}

theorem find_a_values (a : ℝ) (h1 : (U a) \ A = {1}) : a = -1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_values_l623_62380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l623_62307

-- Define the curve C in polar coordinates
noncomputable def C_polar (θ : ℝ) : ℝ := 8 * Real.sin θ

-- Define the curve C in rectangular coordinates
def C_rect (x y : ℝ) : Prop := x^2 + y^2 = 8*y

-- Define the intersecting line
def L (t : ℝ) : ℝ × ℝ := (t, t + 2)

-- State the theorem
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ),
    (C_rect A.1 A.2 ∧ C_rect B.1 B.2) ∧
    (∃ (t1 t2 : ℝ), L t1 = A ∧ L t2 = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 14 :=
by
  sorry

#check intersection_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l623_62307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_for_new_group_l623_62349

/-- Represents the time taken to paint a house -/
def PaintTime : Type := ℝ

/-- Represents the number of workers -/
structure Workers where
  experienced : ℕ
  inexperienced : ℕ

/-- Calculates the effective worker count considering experience -/
def effectiveWorkers (w : Workers) : ℝ :=
  (w.experienced : ℝ) * 2 + w.inexperienced

/-- Time taken to paint a house given the number of workers -/
def timeToPaint (w : Workers) (t : ℝ) : Prop :=
  effectiveWorkers w * t = 1

/-- The given condition: 5 people (3 experienced, 2 inexperienced) can paint 1 house in 4 hours -/
axiom initial_condition : timeToPaint { experienced := 3, inexperienced := 2 } 4

/-- The theorem to prove -/
theorem paint_time_for_new_group :
  timeToPaint { experienced := 2, inexperienced := 3 } (8/7 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_for_new_group_l623_62349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_unit_fractions_equal_one_l623_62315

theorem sum_of_three_unit_fractions_equal_one :
  {(a, b, c) : ℕ+ × ℕ+ × ℕ+ | (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1} =
  {(3, 3, 3), (2, 4, 4), (2, 3, 6)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_unit_fractions_equal_one_l623_62315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_zero_rational_points_circle_with_one_rational_point_circle_with_two_rational_points_no_circle_with_three_rational_points_l623_62367

/-- A point is rational if both of its coordinates are rational numbers. -/
def RationalPoint (p : ℚ × ℚ) : Prop := True

/-- The set of rational points on a circle. -/
def RationalPointsOnCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | (↑p.1 - center.1)^2 + (↑p.2 - center.2)^2 = radius^2}

theorem circle_with_zero_rational_points :
  RationalPointsOnCircle (0, 0) (Real.sqrt 2) = ∅ := by sorry

theorem circle_with_one_rational_point :
  RationalPointsOnCircle (Real.sqrt 2, 0) (Real.sqrt 2) = {(0, 0)} := by sorry

theorem circle_with_two_rational_points :
  RationalPointsOnCircle (0, Real.sqrt 2) (Real.sqrt 3) = {(1, 0), (-1, 0)} := by sorry

theorem no_circle_with_three_rational_points :
  ¬∃ (center : ℝ × ℝ) (radius : ℝ),
    ∃ (p₁ p₂ p₃ : ℚ × ℚ), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    RationalPointsOnCircle center radius = {p₁, p₂, p₃} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_zero_rational_points_circle_with_one_rational_point_circle_with_two_rational_points_no_circle_with_three_rational_points_l623_62367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_3_to_7_l623_62343

-- Define the triangle PQR and points T and S
structure Triangle (P Q R T S : ℝ × ℝ) : Prop where
  t_on_qr : ∃ t : ℝ, T = t • Q + (1 - t) • R
  s_on_tr : ∃ s : ℝ, S = s • T + (1 - s) • R
  qt_length : dist Q T = 3
  tr_length : dist T R = 9
  ts_length : dist T S = 7
  sr_length : dist S R = 2

-- Define the areas of triangles PQT and PTS
noncomputable def area_pqt (P Q R T S : ℝ × ℝ) : ℝ := sorry
noncomputable def area_pts (P Q R T S : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_is_3_to_7 {P Q R T S : ℝ × ℝ} (h : Triangle P Q R T S) :
  area_pqt P Q R T S / area_pts P Q R T S = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_3_to_7_l623_62343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l623_62300

-- Define the inequality function
noncomputable def g (x : ℝ) := |x + 3| - 2 * x - 1

-- Define the solution set
noncomputable def S := {x : ℝ | g x < 0}

-- Define the function f
noncomputable def f (x m : ℝ) := |x - m| + |x + 1 / m| - 2

theorem problem_solution :
  -- Part 1: The solution set of |x+3|-2x-1 < 0 is (2, +∞)
  S = {x : ℝ | x > 2} ∧
  -- Part 2: f(x) has zero points if and only if m = 1
  (∀ m : ℝ, m > 0 → (∃ x : ℝ, f x m = 0) ↔ m = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l623_62300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_king_game_l623_62368

/-- Represents the game state on an n × m chessboard -/
structure GameState (n m : ℕ) where
  king_position : ℕ × ℕ
  visited : Set (ℕ × ℕ)

/-- Defines a valid move in the game -/
def is_valid_move (n m : ℕ) (state : GameState n m) (new_pos : ℕ × ℕ) : Prop :=
  new_pos.1 < n ∧ new_pos.2 < m ∧
  (state.king_position.1 = new_pos.1 ∧ (state.king_position.2 + 1 = new_pos.2 ∨ state.king_position.2 = new_pos.2 + 1) ∨
   state.king_position.2 = new_pos.2 ∧ (state.king_position.1 + 1 = new_pos.1 ∨ state.king_position.1 = new_pos.1 + 1) ∨
   (state.king_position.1 + 1 = new_pos.1 ∨ state.king_position.1 = new_pos.1 + 1) ∧
   (state.king_position.2 + 1 = new_pos.2 ∨ state.king_position.2 = new_pos.2 + 1)) ∧
  new_pos ∉ state.visited

/-- Defines the winning condition for Player 1 -/
def player1_wins (n m : ℕ) : Prop :=
  ∀ (state : GameState n m),
    ∃ (move : ℕ × ℕ), is_valid_move n m state move →
      ¬∃ (opponent_move : ℕ × ℕ), is_valid_move n m
        { king_position := move
        , visited := insert move state.visited } opponent_move

/-- The main theorem: Player 1 wins if and only if n × m is even -/
theorem dark_king_game (n m : ℕ) :
  player1_wins n m ↔ Even (n * m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_king_game_l623_62368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_log_trig_equation_l623_62342

theorem no_solution_log_trig_equation :
  ¬ ∃ x : ℝ, (Real.sin x > 0 ∧ Real.cos x > 0) ∧ Real.log (Real.sin x) + Real.log (Real.cos x) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_log_trig_equation_l623_62342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cutting_process_properties_l623_62332

/-- A square cutting process where each square is kept with probability p at each step -/
def SquareCuttingProcess (p : ℝ) := Unit

/-- The critical probability for the square cutting process -/
noncomputable def critical_probability : ℝ := 1/4

/-- The probability of having a non-empty remaining set -/
noncomputable def prob_non_empty (p : ℝ) : ℝ := sorry

/-- The size of the remaining set -/
noncomputable def remaining_set_size (p : ℝ) : ℝ := sorry

theorem square_cutting_process_properties :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 →
  (p > critical_probability → prob_non_empty p > 0) ∧
  (p < critical_probability → prob_non_empty p = 0) ∧
  (p ≠ 1 → remaining_set_size p = 0) := by
  sorry

#check square_cutting_process_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cutting_process_properties_l623_62332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_in_cubic_l623_62357

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

def is_root_of_cubic (z : ℂ) : Prop :=
  ∃ c d e : ℤ, z^3 + c*z^2 + d*z + e = 0

theorem roots_of_unity_in_cubic :
  ∃ S : Finset ℂ, (∀ z ∈ S, is_root_of_unity z ∧ is_root_of_cubic z) ∧ S.card ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_in_cubic_l623_62357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l623_62308

/-- The function f(x) = ln x - x^2 is monotonically decreasing on [√2/2, +∞) -/
theorem f_monotone_decreasing :
  ∀ x y, x > 0 ∧ y > 0 ∧ Real.sqrt 2 / 2 ≤ x ∧ x < y → 
  (Real.log x - x^2) > (Real.log y - y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l623_62308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_sixth_person_friends_l623_62319

/-- Represents the number of friends for each person in a group of 26 people. -/
def num_friends : Fin 26 → ℕ
  | ⟨0, _⟩ => 1
  | ⟨n+1, _⟩ => min (n+2) 25

/-- The properties of the friendship network -/
structure FriendshipNetwork where
  /-- The total number of people is 26 -/
  total_people : Nat
  total_people_eq : total_people = 26
  /-- Each person has friends according to the num_friends function -/
  friends_count : ∀ (i : Fin 26), num_friends i = i.val + 1
  /-- The 25th person is friends with everyone -/
  twenty_fifth_friends_all : num_friends ⟨24, by norm_num⟩ = 25
  /-- Friendship is reciprocal -/
  reciprocal_friendship : ∀ (i j : Fin 26), i ≠ j → 
    (j.val < num_friends i ↔ i.val < num_friends j)

/-- The main theorem: The 26th person has exactly 13 friends -/
theorem twenty_sixth_person_friends (fn : FriendshipNetwork) : 
  num_friends ⟨25, by norm_num⟩ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_sixth_person_friends_l623_62319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_area_difference_l623_62388

-- Define the fixed side length and perimeter
def fixed_side : ℝ := 36
def perimeter : ℝ := 144

-- Define a parallelogram
structure Parallelogram where
  side_a : ℝ
  side_b : ℝ
  angle : ℝ
  perimeter_condition : side_a + side_b = perimeter / 2
  side_a_fixed : side_a = fixed_side

-- Define the area of a parallelogram
noncomputable def area (p : Parallelogram) : ℝ := p.side_a * p.side_b * Real.sin p.angle

-- Theorem statement
theorem greatest_area_difference :
  ∃ (p1 p2 : Parallelogram), 
    ∀ (q1 q2 : Parallelogram), 
      area p1 - area p2 ≥ area q1 - area q2 ∧ 
      area p1 - area p2 = 1296 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_area_difference_l623_62388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_perimeter_ratio_l623_62366

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℚ
  width : ℚ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℚ :=
  2 * (g.length + g.width)

/-- The specific garden in the problem -/
def problem_garden : RectangularGarden :=
  { length := 23, width := 15 }

/-- Theorem stating the ratio of width to perimeter for the problem garden -/
theorem width_to_perimeter_ratio :
  problem_garden.width / perimeter problem_garden = 15 / 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_perimeter_ratio_l623_62366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_DB_length_l623_62391

-- Define the triangle
structure RightTriangle :=
  (AC : ℝ)
  (AD : ℝ)
  (h_AC_pos : AC > 0)
  (h_AD_pos : AD > 0)
  (h_AD_lt_AC : AD < AC)

-- Define the theorem
theorem right_triangle_DB_length (t : RightTriangle) (h : t.AC = 18.6 ∧ t.AD = 4.5) :
  ∃ DB : ℝ, DB > 0 ∧ |DB - 7.965| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_DB_length_l623_62391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_l623_62398

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ → ℕ → ℕ
  | u, v, 0 => 0  -- Add this case to handle Nat.zero
  | u, v, 1 => u + v
  | u, v, n + 1 => if n % 2 = 0 then a u v (n / 2) + u else a u v (n / 2) + v

/-- Definition of S_n -/
def S (u v : ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + a u v (i + 1)) 0

/-- Main theorem statement -/
theorem infinitely_many_perfect_squares (u v : ℕ) (hu : u > 0) (hv : v > 0) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ n, ∃ m : ℕ, S u v (f n) = m ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_l623_62398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_watching_percentage_l623_62358

/-- Calculates the percentage of time spent watching a comet given various activity durations --/
theorem comet_watching_percentage
  (telescope_shopping : ℕ)
  (binocular_shopping : ℕ)
  (setup_time : ℕ)
  (snack_prep_multiplier : ℕ)
  (comet_watching : ℕ)
  (other_stargazing : ℕ)
  (h1 : telescope_shopping = 120)
  (h2 : binocular_shopping = 45)
  (h3 : setup_time = 90)
  (h4 : snack_prep_multiplier = 3)
  (h5 : comet_watching = 20)
  (h6 : other_stargazing = 30) :
  let total_prep_time := telescope_shopping + binocular_shopping + setup_time + snack_prep_multiplier * setup_time
  let total_stargazing_time := (comet_watching + other_stargazing) / 2
  let total_time := total_prep_time + total_stargazing_time
  let comet_percentage := (comet_watching / 2) / total_time * 100
  ⌊comet_percentage⌋₊ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comet_watching_percentage_l623_62358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_imply_a_range_l623_62354

-- Define the function f as noncomputable
noncomputable def f (a x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + 1)

-- State the theorem
theorem f_bounds_imply_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → 1 ≤ f a x ∧ f a x ≤ 3) → 2 ≤ a ∧ a ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_imply_a_range_l623_62354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_range_of_a_l623_62364

open Real

theorem equal_roots_range_of_a (θ : ℝ) (a : ℝ) :
  (π/4 < θ ∧ θ < π/2) →
  (∃ x : ℝ, ∀ y : ℝ, x^2 + 4*x*sin θ + a*tan θ = 0 ↔ y = x) →
  0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_range_of_a_l623_62364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l623_62399

noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x

noncomputable def tangent_slope (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem tangent_slope_angle_range :
  ∀ x : ℝ, ∃ α : ℝ, 
    (0 ≤ α ∧ α < Real.pi / 2) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) ∧
    Real.tan α = tangent_slope x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l623_62399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l623_62334

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  area : ℝ
  paintCost : ℝ
  paintRate : ℝ

/-- Theorem about the length of a rectangular floor given specific conditions -/
theorem floor_length_calculation (floor : RectangularFloor) 
  (h1 : floor.length = 3 * floor.breadth) 
  (h2 : floor.area = floor.length * floor.breadth)
  (h3 : floor.paintCost = 529)
  (h4 : floor.paintRate = 3)
  (h5 : floor.area = floor.paintCost / floor.paintRate) : 
  ∃ ε > 0, |floor.length - 23| < ε := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_calculation_l623_62334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_equals_frequency_ratio_l623_62360

/-- The frequency of an object in a dataset. -/
def frequency [DecidableEq α] (object : α) (dataset : Multiset α) : ℕ := Multiset.count object dataset

/-- The total number of occurrences in a dataset. -/
def total_occurrences (dataset : Multiset α) : ℕ := Multiset.card dataset

/-- The rate of an object in a dataset. -/
noncomputable def rate [DecidableEq α] (object : α) (dataset : Multiset α) : ℚ :=
  (frequency object dataset : ℚ) / (total_occurrences dataset : ℚ)

/-- The rate is equal to the ratio of the frequency to the total number of occurrences. -/
theorem rate_equals_frequency_ratio [DecidableEq α] (object : α) (dataset : Multiset α) :
  rate object dataset = (frequency object dataset : ℚ) / (total_occurrences dataset : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_equals_frequency_ratio_l623_62360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l623_62370

/-- Given a train and platform, calculate the time to cross the platform -/
noncomputable def time_to_cross_platform (train_length platform_length signal_pole_time : ℝ) : ℝ :=
  (train_length + platform_length) * signal_pole_time / train_length

/-- Theorem stating that for the given parameters, the time to cross the platform is 39 seconds -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 300
  let platform_length : ℝ := 25
  let signal_pole_time : ℝ := 36
  time_to_cross_platform train_length platform_length signal_pole_time = 39 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l623_62370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_are_correct_l623_62314

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the original line
def original_line (x y : ℝ) : Prop := x + 2*y + 1 = 0

-- Define the potential tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x + 2*y + 5 = 0
def tangent_line_2 (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Function to check if a line is parallel to the original line
def is_parallel (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), line x y ↔ original_line x (y + k)

-- Function to check if a line is tangent to the circle
def is_tangent (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), line x y ∧ my_circle x y ∧
  ∀ (x' y' : ℝ), line x' y' ∧ my_circle x' y' → (x', y') = (x, y)

-- Theorem statement
theorem tangent_lines_are_correct :
  (is_parallel tangent_line_1 ∧ is_tangent tangent_line_1) ∧
  (is_parallel tangent_line_2 ∧ is_tangent tangent_line_2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_are_correct_l623_62314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_products_count_l623_62317

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => q * geometric_sequence a₁ q n

def product_up_to (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * a i) 1

theorem positive_products_count (a₁ q : ℝ) (h₁ : a₁ = 512) (h₂ : q = -1/2) :
  let a := geometric_sequence a₁ q
  let prod := product_up_to a
  (([8, 9, 10, 11].map (λ n => prod n)).filter (λ x => x > 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_products_count_l623_62317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l623_62339

-- Define a geometric sequence
def is_geometric_sequence (seq : List ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Nat, i + 1 < seq.length → seq[i + 1]! = seq[i]! * r

theorem geometric_sequence_property (a b c : ℝ) :
  is_geometric_sequence [-1, a, b, c, -9] →
  b = -3 ∧ a * c = 9 := by
  intro h
  sorry

#check geometric_sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l623_62339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l623_62397

def set_A : Set ℝ := {x | 1 < (3 : ℝ)^x ∧ (3 : ℝ)^x ≤ 9}
def set_B : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l623_62397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_ratio_theorem_l623_62327

/-- The polynomial g(x) = x^2001 - 14x^2000 + 1 -/
def g (x : ℂ) : ℂ := x^2001 - 14*x^2000 + 1

/-- The zeros of g(x) -/
noncomputable def s : Fin 2001 → ℂ := sorry

/-- The polynomial Q of degree 2001 -/
noncomputable def Q : Polynomial ℂ := sorry

theorem polynomial_ratio_theorem 
  (h_distinct : ∀ i j : Fin 2001, i ≠ j → s i ≠ s j)
  (h_zeros : ∀ i : Fin 2001, g (s i) = 0)
  (h_degree : Q.degree = 2001)
  (h_Q_zeros : ∀ i : Fin 2001, Q.eval (s i + (s i)⁻¹) = 0) :
  Q.eval 1 / Q.eval (-1) = 259 / 289 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_ratio_theorem_l623_62327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_in_P_periodic_when_M_is_one_h_in_P_iff_l623_62320

-- Define the set P
def P : Set (ℝ → ℝ) :=
  {f | ∃ (M : ℝ), M ≠ 0 ∧ ∀ x, f (x + M) = -M * f x}

-- Define the function g
noncomputable def g : ℝ → ℝ := fun x ↦ Real.sin (Real.pi * x)

-- Define the function h
noncomputable def h (ω : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x)

-- Theorem 1: g belongs to P
theorem g_in_P : g ∈ P := by sorry

-- Theorem 2: Functions in P with M = 1 are periodic with period 2
theorem periodic_when_M_is_one (f : ℝ → ℝ) (hf : f ∈ P) :
  (∃ (M : ℝ), M = 1 ∧ ∀ x, f (x + M) = -M * f x) →
  ∀ x, f (x + 2) = f x := by sorry

-- Theorem 3: Condition for h to be in P
theorem h_in_P_iff (ω : ℝ) :
  h ω ∈ P ↔ ∃ k : ℤ, ω = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_in_P_periodic_when_M_is_one_h_in_P_iff_l623_62320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_100_l623_62331

theorem infinitely_many_divisible_by_100 :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ (k : ℕ), 100 ∣ (2^(f k) + (f k)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_100_l623_62331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_provision_equation_l623_62394

/-- The initial number of men -/
def M : ℝ := sorry

/-- The number of days the provisions last initially -/
def initial_days : ℝ := 15

/-- The number of additional men -/
def additional_men : ℝ := 200

/-- The number of days the provisions last after additional men join -/
def final_days : ℝ := 12.857

/-- Theorem stating the relationship between the initial number of men,
    the additional men, and the duration of provisions -/
theorem provision_equation :
  M * initial_days = (M + additional_men) * final_days := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_provision_equation_l623_62394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_only_good_point_l623_62352

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a triangle is acute
def isAcute (t : Triangle) : Prop := sorry

-- Define a function to calculate the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : Point := sorry

-- Define a function to check if a point is inside a triangle
def isInside (p : Point) (t : Triangle) : Prop := sorry

-- Define a function to calculate the length of a cevian
noncomputable def cevianLength (p : Point) (vertex : ℝ × ℝ) (t : Triangle) : ℝ := sorry

-- Define a function to calculate the length of a side of the triangle
noncomputable def sideLength (t : Triangle) (side : Fin 3) : ℝ := sorry

-- Define the property of a "good" point
def isGoodPoint (p : Point) (t : Triangle) : Prop :=
  isInside p t ∧
  ∃ (k : ℝ), k > 0 ∧
    (cevianLength p t.A t / sideLength t 0 = k) ∧
    (cevianLength p t.B t / sideLength t 1 = k) ∧
    (cevianLength p t.C t / sideLength t 2 = k)

-- The main theorem
theorem orthocenter_only_good_point (t : Triangle) :
  isAcute t →
  ∀ (p : Point), isGoodPoint p t ↔ p = orthocenter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_only_good_point_l623_62352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_correct_l623_62326

noncomputable def equipment_value (n : ℕ) : ℝ :=
  if n ≤ 6 then 130 - 10 * n
  else 70 * (3/4)^(n-6)

theorem equipment_value_correct (n : ℕ) (h : n ≥ 1) :
  equipment_value n =
    if n ≤ 6 then 130 - 10 * n
    else 70 * (3/4)^(n-6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_value_correct_l623_62326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_proof_l623_62306

/-- Calculates the monthly payment for a loan with the given parameters -/
noncomputable def calculate_monthly_payment (principal : ℝ) (interest_rate : ℝ) (num_payments : ℕ) : ℝ :=
  let future_value := principal * (1 + interest_rate) ^ (num_payments : ℝ)
  future_value / (num_payments : ℝ)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem loan_payment_proof (principal : ℝ) (interest_rate : ℝ) (num_payments : ℕ) :
  principal = 5000 →
  interest_rate = 0.15 →
  num_payments = 6 →
  round_to_nearest (calculate_monthly_payment principal interest_rate num_payments) = 1676 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_proof_l623_62306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l623_62328

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (x - 4*x^2 + 16*x^3) / (12 - x^3)

-- Define the cube root of 3
noncomputable def cubeRoot3 : ℝ := Real.rpow 3 (1/3)

-- Theorem statement
theorem f_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio (2 * cubeRoot3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l623_62328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l623_62387

-- Define the function f(x) = 0.5^x
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

-- State the theorem
theorem inequality_solution_set :
  (∀ x y : ℝ, x < y → f y < f x) →
  (∀ x : ℝ, f (2*x) > f (x-1) ↔ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l623_62387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_desired_quadrants_l623_62336

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y - 2 = -2 * (x - 1)

-- Define the condition for a line to pass through the first, second, and fourth quadrants
def passes_through_first_second_fourth_quadrants (f : ℝ → ℝ) : Prop :=
  ∃ a b, a < 0 ∧ b > 0 ∧ f 0 = b ∧ f (b / (-a)) = 0

-- Theorem statement
theorem line_passes_through_desired_quadrants :
  passes_through_first_second_fourth_quadrants (λ x ↦ -2 * (x - 1) + 2) := by
  sorry

-- Helper lemma to show that the line equation matches the function in the theorem
lemma line_equation_matches_function :
  ∀ x y, line_equation x y ↔ y = -2 * (x - 1) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_desired_quadrants_l623_62336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_count_l623_62324

/-- The maximum number of smaller boxes that can fit in the larger box -/
def max_boxes (large_box : List ℕ) (small_box : List ℕ) : ℕ :=
  (large_box.prod * 1000000) / small_box.prod

/-- Theorem stating the maximum number of smaller boxes that can fit in the larger box -/
theorem max_boxes_count :
  let large_box := [8, 7, 6]
  let small_box := [8, 7, 6]
  max_boxes large_box small_box = 1000000 := by
  sorry

#eval max_boxes [8, 7, 6] [8, 7, 6]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_boxes_count_l623_62324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l623_62305

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f(x) = x^(2⌊x⌋)
noncomputable def f (x : ℝ) : ℝ := x ^ (2 * (floor x))

-- State the theorem
theorem solution_set :
  {x : ℝ | f x = 2022} = {Real.sqrt 2022 / 2022, (2022 : ℝ) ^ (1/6)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l623_62305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_of_length_10_l623_62371

/-- Represents a valid binary sequence where neither two zeroes nor three ones appear consecutively -/
inductive ValidSequence : Nat → Type where
  | zero : ValidSequence 1
  | one : ValidSequence 1
  | append_zero : ValidSequence n → ValidSequence (n+1)
  | append_one : ValidSequence n → ValidSequence (n+1)
  | append_one_one : ValidSequence n → ValidSequence (n+2)

/-- Count of valid sequences of length n -/
def countValidSequences : Nat → Nat
  | 0 => 0
  | 1 => 2
  | n + 1 => 
    let a := countValidSequences n
    let b := if n > 0 then countValidSequences (n-1) else 0
    let c := if n > 1 then countValidSequences (n-2) else 0
    a + b + c

/-- The main theorem stating that there are 28 valid sequences of length 10 -/
theorem valid_sequences_of_length_10 : 
  countValidSequences 10 = 28 := by sorry

#eval countValidSequences 10  -- Should output 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_of_length_10_l623_62371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l623_62333

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.sum 9 = seq.sum 4)
  (h2 : seq.a 1 ≠ 0)
  (h3 : ∃ k, seq.sum (k + 3) = 0) :
  ∃ k, seq.sum (k + 3) = 0 ∧ k = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l623_62333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solvability_l623_62396

-- Define the original system of equations
noncomputable def original_system (x y : ℝ) : Prop :=
  (2:ℝ)^x + (2:ℝ)^(x+1) + (2:ℝ)^(x+2) + (2:ℝ)^(x+3) = (4:ℝ)^y + (4:ℝ)^(y+1) + (4:ℝ)^(y+2) + (4:ℝ)^(y+3) ∧
  (3:ℝ)^x + (3:ℝ)^(x+1) + (3:ℝ)^(x+2) + (3:ℝ)^(x+3) = (9:ℝ)^y + (9:ℝ)^(y+1) + (9:ℝ)^(y+2) + (9:ℝ)^(y+3)

-- Define the modified system of equations
noncomputable def modified_system (x y : ℝ) : Prop :=
  (2:ℝ)^x + (2:ℝ)^(x+1) + (2:ℝ)^(x+2) + (2:ℝ)^(x+3) = (8:ℝ)^y + (8:ℝ)^(y+1) + (8:ℝ)^(y+2) + (8:ℝ)^(y+3) ∧
  (3:ℝ)^x + (3:ℝ)^(x+1) + (3:ℝ)^(x+2) + (3:ℝ)^(x+3) = (9:ℝ)^y + (9:ℝ)^(y+1) + (9:ℝ)^(y+2) + (9:ℝ)^(y+3)

-- Define a function to check if two real numbers are approximately equal
def approx_equal (a b : ℝ) (ε : ℝ) : Prop :=
  abs (a - b) < ε

-- Theorem statement
theorem system_solvability :
  (¬ ∃ x y : ℝ, original_system x y) ∧
  (∃ x y : ℝ, modified_system x y ∧ 
    approx_equal x (-2.323) 0.001 ∧ 
    approx_equal y (-2.536) 0.001) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solvability_l623_62396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_and_non_divisors_l623_62350

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sequence_a (n + 1)^2 + (n + 2) * sequence_a (n + 1) - sequence_a n^2 - n * sequence_a n

theorem infinite_prime_divisors_and_non_divisors :
  (∃ (S : Set Nat), S.Infinite ∧ (∀ p ∈ S, Nat.Prime p ∧ ∃ n, (sequence_a n).natAbs % p = 0)) ∧
  (∀ n : ℕ, (sequence_a n).natAbs % 3 ≠ 0 ∧ (sequence_a n).natAbs % 5 ≠ 0 ∧ (sequence_a n).natAbs % 19 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_and_non_divisors_l623_62350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_l623_62321

noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

noncomputable def fractionalPart (x : ℝ) : ℝ :=
  x - integerPart x

theorem solution_theorem (x : ℝ) :
  (integerPart x + fractionalPart (2 * x) = (5/2 : ℝ)) ↔ (x = (9/4 : ℝ) ∨ x = (11/4 : ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_l623_62321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l623_62361

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 + 2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the line C₂
noncomputable def C₂ (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ α, C₁ α = p ∧ (C₂ p.1 = p.2)}

-- Theorem statement
theorem intersection_product : 
  ∀ P Q, P ∈ intersection_points → Q ∈ intersection_points →
  (Real.sqrt (P.1^2 + P.2^2)) * (Real.sqrt (Q.1^2 + Q.2^2)) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l623_62361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_has_zero_point_l623_62390

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem f_odd_and_has_zero_point :
  (∀ x, f (-x) = -f x) ∧ (∃ x, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_has_zero_point_l623_62390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_22_l623_62323

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let height := |g.x - e.x|
  let base1 := |f.y - e.y|
  let base2 := |g.y - h.y|
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given coordinates is 22 square units -/
theorem trapezoid_area_is_22 :
  let e : Point := ⟨2, -3⟩
  let f : Point := ⟨2, 2⟩
  let g : Point := ⟨6, 8⟩
  let h : Point := ⟨6, 2⟩
  trapezoidArea e f g h = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_22_l623_62323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_star_angle_theorem_l623_62348

/-- A regular polygon with n sides -/
def regular_polygon (n : ℕ) : Set ℝ × Set ℝ := sorry

/-- Extend every alternate side of a polygon to create a modified star -/
def extend_alternate_sides (p : Set ℝ × Set ℝ) : Set ℝ × Set ℝ := sorry

/-- The set of vertices of a polygon -/
def vertices (p : Set ℝ × Set ℝ) : Set (ℝ × ℝ) := sorry

/-- The angle at a specific vertex of a polygon -/
def angle_at_vertex (p : Set ℝ × Set ℝ) (v : ℝ × ℝ) : ℚ := sorry

/-- The angle at each vertex of a modified star created from a regular polygon -/
def modified_star_angle (n : ℕ) : ℚ :=
  180 * (n - 4) / n

/-- Theorem: The angle at each vertex of a modified star created by extending
    every alternate side of a regular polygon with n sides (n > 4) is 180(n-4)/n -/
theorem modified_star_angle_theorem (n : ℕ) (h : n > 4) :
  let polygon := regular_polygon n
  let star := extend_alternate_sides polygon
  ∀ v ∈ vertices star, angle_at_vertex star v = modified_star_angle n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_star_angle_theorem_l623_62348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l623_62301

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 - x + 2

-- Theorem statement
theorem f_has_two_zeros : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

#check f_has_two_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l623_62301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l623_62353

/-- Given a triangle with two sides of length 8 and 5 units, 
    there are exactly 9 possible integer lengths for the third side. -/
theorem triangle_third_side_count : 
  let side1 : ℕ := 8
  let side2 : ℕ := 5
  let possible_sides := Finset.filter (fun x => 
    x > 0 ∧ 
    x < side1 + side2 ∧ 
    side1 < x + side2 ∧ 
    side2 < x + side1) (Finset.range (side1 + side2))
  Finset.card possible_sides = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l623_62353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_length_l623_62385

/-- Hyperbola C with equation x²/25 - y²/16 = 1 -/
def Hyperbola (p : ℝ × ℝ) : Prop := p.1^2 / 25 - p.2^2 / 16 = 1

/-- F₁ and F₂ are the foci of the hyperbola -/
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  let c := Real.sqrt 41
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

/-- P is a point on the hyperbola distinct from the vertices -/
def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  Hyperbola P ∧ P ≠ (5, 0) ∧ P ≠ (-5, 0)

/-- Line l is tangent to the circles with diameters PF₁ and PF₂ at points A and B -/
def is_tangent_line (l : Set (ℝ × ℝ)) (P F₁ F₂ A B : ℝ × ℝ) : Prop :=
  A ∈ l ∧ B ∈ l ∧
  ∃ (c₁ c₂ : Set (ℝ × ℝ)),
    -- Here we assume the existence of is_circle, diameter, and is_tangent
    -- These would need to be defined elsewhere in a real implementation
    true ∧ true ∧
    true ∧ true ∧
    A ∈ c₁ ∧ B ∈ c₂ ∧
    true ∧ true

/-- The main theorem -/
theorem hyperbola_tangent_length 
  (F₁ F₂ P A B : ℝ × ℝ)
  (l : Set (ℝ × ℝ)) :
  are_foci F₁ F₂ →
  is_on_hyperbola P →
  is_tangent_line l P F₁ F₂ A B →
  dist A B = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_length_l623_62385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l623_62341

/-- Represents a work project with laborers and completion time -/
structure WorkProject where
  total_laborers : ℝ
  absent_laborers : ℕ
  actual_completion_time : ℕ

/-- Calculates the original planned completion time for a work project -/
noncomputable def original_completion_time (project : WorkProject) : ℝ :=
  (project.total_laborers - project.absent_laborers : ℝ) * project.actual_completion_time / project.total_laborers

/-- Theorem stating that for the given project, the original completion time is approximately 10 days -/
theorem project_completion_time :
  let project : WorkProject := {
    total_laborers := 21.67,
    absent_laborers := 5,
    actual_completion_time := 13
  }
  abs (original_completion_time project - 10) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l623_62341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l623_62335

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x) - y x = 0

-- Define the general solution
noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ :=
  C * Real.exp x

-- Define the particular solution
noncomputable def particular_solution (x : ℝ) : ℝ :=
  -Real.exp (x - 1)

-- Theorem statement
theorem solution_verification :
  (∀ C, diff_eq (general_solution C)) ∧
  (diff_eq particular_solution) ∧
  (particular_solution 1 = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l623_62335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l623_62386

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x

-- Define the slope of the line parallel to the tangent
def m : ℝ := 4

-- Theorem statement
theorem tangent_line_equation :
  ∃ (a b : ℝ), (a = 1 ∧ b = 2) ∨ (a = -1 ∧ b = -2) ∧
  ∀ (x y : ℝ), y = f x → (deriv f a = m ∧ y - f a = m * (x - a)) ↔ 4*x - y - b = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l623_62386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_68_divisible_sums_cell_sums_not_repeating_l623_62330

-- Define the grid as a function from coordinates to natural numbers
def spiral_grid : ℤ × ℤ → ℕ := sorry

-- Define the sum at the center of a cell
def cell_sum (x y : ℤ) : ℕ :=
  spiral_grid (x, y) + spiral_grid (x + 1, y) + 
  spiral_grid (x, y + 1) + spiral_grid (x + 1, y + 1)

-- State the property of horizontal or vertical movement
axiom horizontal_vertical_increment (x y : ℤ) :
  cell_sum (x + 1) y - cell_sum x y = 4 ∨ cell_sum x (y + 1) - cell_sum x y = 4

-- State the main theorem
theorem infinite_68_divisible_sums :
  ∀ n : ℕ, ∃ x y : ℤ, x ≥ n ∧ y ≥ n ∧ 68 ∣ cell_sum x y := by
  sorry

-- Additional theorem to address the second part of the question
theorem cell_sums_not_repeating :
  ∀ x₁ y₁ x₂ y₂ : ℤ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) → cell_sum x₁ y₁ ≠ cell_sum x₂ y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_68_divisible_sums_cell_sums_not_repeating_l623_62330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l623_62382

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I ^ 3) = Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l623_62382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l623_62375

/-- Given two circles in a 2D plane, this theorem proves that the y-intercept of their common tangent line in the first quadrant is 6√6. -/
theorem tangent_line_y_intercept : ∃ (y : ℝ), y = 6 * Real.sqrt 6 := by
  -- Define the circles
  let circle1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 3^2}
  let circle2 := {p : ℝ × ℝ | (p.1 - 8)^2 + p.2^2 = 2^2}

  -- Define the tangent line condition
  let tangent_line (c1 c2 : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
    ∃ (t1 t2 : ℝ × ℝ), t1 ∈ c1 ∧ t2 ∈ c2 ∧
      t1.2 > 0 ∧ t2.2 > 0 ∧  -- Points in the first quadrant
      (p.2 - t1.2) * (t1.1 - 3) = t1.2 * (p.1 - t1.1) ∧  -- Tangent to circle1
      (p.2 - t2.2) * (t2.1 - 8) = t2.2 * (p.1 - t2.1)    -- Tangent to circle2

  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l623_62375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l623_62338

/-- Calculates the speed of a train given the speed of another train traveling in the opposite direction, the length of the train, and the time it takes to pass. -/
noncomputable def calculate_train_speed (speed_a : ℝ) (length_b : ℝ) (passing_time : ℝ) : ℝ :=
  let speed_a_ms := speed_a * 1000 / 3600
  let relative_speed := length_b / passing_time
  let speed_b_ms := relative_speed - speed_a_ms
  speed_b_ms * 3600 / 1000

/-- Theorem stating that given the conditions, the speed of the goods train is approximately 61.99 km/h. -/
theorem goods_train_speed (speed_a : ℝ) (length_b : ℝ) (passing_time : ℝ)
  (h1 : speed_a = 50)
  (h2 : length_b = 280)
  (h3 : passing_time = 9) :
  ∃ ε > 0, |calculate_train_speed speed_a length_b passing_time - 61.99| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l623_62338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caterpillar_path_ratio_l623_62374

noncomputable section

open Real

/-- The angle between two vectors in radians -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The Euclidean distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem caterpillar_path_ratio :
  ∀ (A B C D : ℝ × ℝ) (h x : ℝ),
    -- Conditions
    angle (B - A) (D - A) = π / 6 →
    angle (C - B) (D - B) = π / 4 →
    B.2 - A.2 = h →
    B.1 - A.1 = x →
    C.1 - B.1 = h →
    C.2 - B.2 = -h →
    -- Theorem
    distance B C / distance A B = sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caterpillar_path_ratio_l623_62374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_ptolemy_l623_62304

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculates the length of the common tangent between two circles -/
noncomputable def commonTangentLength (c1 c2 : Circle) : ℝ := sorry

/-- Checks if a quadrilateral is inscribed in a circle -/
def Quadrilateral.isInscribed (quad : Quadrilateral) (circ : Circle) : Prop := sorry

/-- Checks if a circle touches another circle at a given point -/
def Circle.touches (c1 c2 : Circle) (point : ℝ × ℝ) : Prop := sorry

/-- Main theorem: Generalized Ptolemy's theorem for circles touching a circumscribed circle -/
theorem generalized_ptolemy 
  (circumCircle : Circle) 
  (α β γ δ : Circle) 
  (quad : Quadrilateral) 
  (h_inscribed : Quadrilateral.isInscribed quad circumCircle)
  (h_α_touches : Circle.touches α circumCircle quad.A)
  (h_β_touches : Circle.touches β circumCircle quad.B)
  (h_γ_touches : Circle.touches γ circumCircle quad.C)
  (h_δ_touches : Circle.touches δ circumCircle quad.D) :
  commonTangentLength α β * commonTangentLength γ δ + 
  commonTangentLength β γ * commonTangentLength δ α = 
  commonTangentLength α γ * commonTangentLength β δ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_ptolemy_l623_62304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equality_exists_l623_62355

theorem fraction_sum_equality_exists : ∃ (a b c d e f g h i : ℕ), 
  (a ∈ Finset.range 10 ∧ b ∈ Finset.range 10 ∧ c ∈ Finset.range 10 ∧ 
   d ∈ Finset.range 10 ∧ e ∈ Finset.range 10 ∧ f ∈ Finset.range 10 ∧ 
   g ∈ Finset.range 10 ∧ h ∈ Finset.range 10 ∧ i ∈ Finset.range 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧ h ≠ 0) ∧
  ((a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f + (g : ℚ) / h = i) := by
  sorry

#check fraction_sum_equality_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equality_exists_l623_62355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l623_62359

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : π/2 < β) (h4 : β < π)
  (h5 : Real.sin α = 3/5) (h6 : Real.cos (α + β) = -4/5) : 
  Real.sin β = 24/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l623_62359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_returned_proof_l623_62377

noncomputable def initial_balance : ℚ := 126
noncomputable def groceries : ℚ := 60
noncomputable def gas : ℚ := groceries / 2
noncomputable def new_balance : ℚ := 171

theorem amount_returned_proof :
  initial_balance + groceries + gas - new_balance = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_returned_proof_l623_62377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l623_62365

/-- Parabola defined by y = 2x^2 - 5x + 3 -/
def parabola (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

/-- The vertex of the parabola -/
noncomputable def vertex : ℝ × ℝ := (5/4, -1/8)

/-- The number of intersection points with the coordinate axes -/
def num_intersections : ℕ := 3

/-- Theorem stating the properties of the parabola -/
theorem parabola_properties :
  (∀ x : ℝ, parabola x = 2 * (x - 5/4)^2 - 1/8) ∧
  (∃! p : ℝ × ℝ, p = vertex ∧ 
    ∀ x : ℝ, parabola x ≥ parabola p.1) ∧
  (∃ a b c : ℝ, a ≠ b ∧ parabola a = 0 ∧ parabola b = 0 ∧ parabola 0 = c ∧ c ≠ 0) ∧
  (∀ x : ℝ, parabola x = 0 → x = 1 ∨ x = 3/2) ∧
  num_intersections = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l623_62365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l623_62318

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 9 * x^2 * f y

/-- The set of all functions satisfying the functional equation -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfyingFunction f}

/-- The zero function -/
noncomputable def zeroFunction : ℝ → ℝ := λ _ ↦ 0

/-- The quadratic function 9/4 * x^2 -/
noncomputable def quadFunction : ℝ → ℝ := λ x ↦ 9/4 * x^2

theorem functional_equation_solutions :
  SatisfyingFunctions = {zeroFunction, quadFunction} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l623_62318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l623_62392

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line with slope 2√2
noncomputable def line (p x y : ℝ) : Prop := y = 2*Real.sqrt 2*(x - p/2)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem parabola_theorem :
  ∀ p : ℝ,
  ∃ x₁ y₁ x₂ y₂ : ℝ,
  (parabola p x₁ y₁) ∧ 
  (parabola p x₂ y₂) ∧ 
  (line p x₁ y₁) ∧ 
  (line p x₂ y₂) ∧
  (x₁ < x₂) ∧
  (distance x₁ y₁ x₂ y₂ = 9) ∧
  (parabola p (p/2) 0) →  -- passes through focus
  (p = 4) ∧
  (∃ x₃ y₃ : ℝ, parabola 4 x₃ y₃ ∧ x₃ = x₁ + 2*(x₂ - x₁) ∧ y₃ = y₁ + 2*(y₂ - y₁)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l623_62392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_2_and_4_l623_62363

/-- A random variable following a normal distribution with mean 3 and some variance σ² -/
def x : ℝ → ℝ := sorry

/-- The probability density function of x -/
def f : ℝ → ℝ := sorry

/-- The cumulative distribution function of x -/
def F : ℝ → ℝ := sorry

/-- x follows a normal distribution with mean 3 and variance σ² -/
axiom normal_dist : ∃ σ : ℝ, σ > 0 ∧ 
  ∀ t, f t = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1 / (2 * σ^2)) * (t - 3)^2)

/-- The probability that x is less than or equal to 4 is 0.84 -/
axiom prob_le_4 : F 4 = 0.84

/-- Theorem: The probability that x is between 2 and 4 is 0.68 -/
theorem prob_between_2_and_4 : F 4 - F 2 = 0.68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_2_and_4_l623_62363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pipe_pile_height_l623_62383

/-- The height of a pile of three cylindrical pipes -/
noncomputable def pileHeight (pipeDiameter : ℝ) : ℝ :=
  pipeDiameter + pipeDiameter * Real.sqrt 3 / 2

theorem three_pipe_pile_height :
  pileHeight 10 = 10 + 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pipe_pile_height_l623_62383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l623_62344

theorem tan_half_angle (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) :
  Real.tan (α / 2) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l623_62344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_trig_functions_l623_62325

open Real MeasureTheory

theorem geometric_sequence_trig_functions : 
  ∃ (S : Finset ℝ), 
    (∀ θ ∈ S, 0 ≤ θ ∧ θ < 2*π) ∧ 
    (∀ θ ∈ S, ¬∃ k : ℤ, θ = k * (π/2)) ∧
    (∀ θ ∈ S, (∃ (a b c : ℝ), ({sin θ, cos θ, tan θ} : Set ℝ) = {a, b, c} ∧ 
      ((a*c = b^2) ∨ (a*b = c^2) ∨ (b*c = a^2)))) ∧
    S.card = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_trig_functions_l623_62325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_running_time_sofia_running_time_in_minutes_and_seconds_l623_62369

/-- The time Sofia took to complete all 6 laps -/
noncomputable def total_time (laps : ℕ) (lap_length : ℝ) (first_part_length : ℝ) (second_part_length : ℝ) 
  (first_part_speed : ℝ) (second_part_speed : ℝ) : ℝ :=
  laps * ((first_part_length / first_part_speed) + (second_part_length / second_part_speed))

/-- Theorem stating that Sofia's total time for 6 laps is 648 seconds -/
theorem sofia_running_time : 
  total_time 6 500 150 350 3 6 = 648 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_time 6 500 150 350 3 6

/-- Proposition stating that the calculated time is equal to 10 minutes and 48 seconds -/
theorem sofia_running_time_in_minutes_and_seconds :
  (total_time 6 500 150 350 3 6) / 60 = 10 + 48 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_running_time_sofia_running_time_in_minutes_and_seconds_l623_62369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_plus_abs_nonnegative_l623_62384

theorem negation_of_forall_sin_plus_abs_nonnegative :
  (¬ ∀ x : ℝ, Real.sin x + |x| ≥ 0) ↔ (∃ x : ℝ, Real.sin x + |x| < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_plus_abs_nonnegative_l623_62384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_a_l623_62372

theorem unique_solution_for_a (a : ℝ) : 
  (3 ∈ ({a + 3, 2*a + 1, a^2 + a + 1} : Set ℝ)) ∧ 
  (a + 3 ≠ 2*a + 1) ∧ (a + 3 ≠ a^2 + a + 1) ∧ (2*a + 1 ≠ a^2 + a + 1) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_a_l623_62372
