import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_right_triangular_prism_l52_5285

noncomputable section

-- Define the radius of the sphere
variable (R : ℝ)

-- Define the side length of the equilateral triangle base
variable (a : ℝ)

-- Define the volume of the triangular prism
noncomputable def prism_volume (R a : ℝ) : ℝ := (1/2) * Real.sqrt ((3 * R^2 - a^2) * a^4)

-- Statement of the theorem
theorem max_volume_right_triangular_prism (R : ℝ) (h : R > 0) :
  ∃ (a : ℝ), a > 0 ∧ a < R * Real.sqrt 3 ∧ 
  ∀ (b : ℝ), b > 0 → b < R * Real.sqrt 3 → prism_volume R b ≤ prism_volume R a ∧ prism_volume R a ≤ R^3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_right_triangular_prism_l52_5285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_sides_l52_5279

/-- Predicate for an isosceles triangle -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

/-- An isosceles triangle with perimeter 10 and one side of length 3 has the other two sides of lengths either 3 and 4, or 3.5 and 3.5. -/
theorem isosceles_triangle_sides (a b c : ℝ) : 
  a + b + c = 10 → -- perimeter is 10
  (a = 3 ∨ b = 3 ∨ c = 3) → -- one side is 3
  IsoscelesTriangle a b c → -- the triangle is isosceles
  ((a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 3.5 ∧ b = 3.5 ∧ c = 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 3) ∨ (a = 3.5 ∧ b = 3 ∧ c = 3.5) ∨
   (a = 4 ∧ b = 3 ∧ c = 3) ∨ (a = 3 ∧ b = 3.5 ∧ c = 3.5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_sides_l52_5279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l52_5273

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of line l1: ax - y + 3 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := a

/-- The slope of line l2: 2x - (a+1)y + 4 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := 2 / (a + 1)

/-- a = -2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = -2 → parallel_lines (slope_l1 a) (slope_l2 a)) ∧
  ¬(parallel_lines (slope_l1 a) (slope_l2 a) → a = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l52_5273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l52_5244

-- Define the basic structures
structure Line where
  -- Placeholder for line properties
  dummy : Unit

structure Point where
  -- Placeholder for point properties
  dummy : Unit

-- Define the given conditions
variable (l₁ l₂ l₃ : Line)
variable (O A : Point)

-- Define intersection and "on" relations
def Line.intersect : Line → Line → Option Point := sorry
def Point.on : Point → Line → Prop := sorry

-- Define the given conditions as axioms
axiom intersection_point : 
  (l₁.intersect l₂ = some O) ∧ (l₂.intersect l₃ = some O) ∧ (l₃.intersect l₁ = some O)
axiom A_on_l₁ : A.on l₁

-- Define the properties of the triangle
def angle_bisector (p q r : Point) : Line :=
  sorry

def is_valid_triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- State the theorem
theorem triangle_construction_theorem :
  ∃ (B C : Point), is_valid_triangle A B C ∧
    angle_bisector A B C = l₁ ∧
    angle_bisector B C A = l₂ ∧
    angle_bisector C A B = l₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l52_5244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l52_5245

-- Define the function f(x) = 2x^3 - 3x^2 - 12x
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

-- Define the interval [-2, 3]
def interval : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 7 ∧ min = -20 ∧
  (∀ x ∈ interval, f x ≤ max) ∧
  (∃ x ∈ interval, f x = max) ∧
  (∀ x ∈ interval, min ≤ f x) ∧
  (∃ x ∈ interval, f x = min) := by
  sorry

#check f_max_min_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l52_5245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l52_5214

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | Real.exp (x * Real.log 2) < 2}

theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l52_5214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l52_5258

-- Define the radius of the original circle
def original_radius : ℝ := 3

-- Define the number of arcs
def num_arcs : ℕ := 6

-- Define the number of smaller circles formed
def num_smaller_circles : ℕ := 3

-- Theorem statement
theorem circle_area_ratio :
  let original_area := π * original_radius^2
  let arc_length := 2 * π * original_radius / (num_arcs : ℝ)
  let smaller_radius := arc_length / (2 * π)
  let smaller_area := π * smaller_radius^2
  let total_smaller_area := (num_smaller_circles : ℝ) * smaller_area
  total_smaller_area / original_area = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l52_5258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l52_5208

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3)) + ((1 / b + 6 * c) ^ (1/3)) + ((1 / c + 6 * a) ^ (1/3))) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l52_5208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_and_solution_set_l52_5210

noncomputable def f (x : ℝ) := 1 / (9 * Real.sin x ^ 2) + 4 / (9 * Real.cos x ^ 2)

theorem min_value_of_f_and_solution_set :
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x ≥ 1) ∧
  {x : ℝ | |x + 1| + |x - 2| ≥ 5} = {x : ℝ | x ≤ -2 ∨ x ≥ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_and_solution_set_l52_5210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l52_5278

/-- A function g satisfying certain properties -/
noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- Theorem stating the unique number not in the range of g -/
theorem unique_number_not_in_range
  (p q r s : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h23 : g p q r s 23 = 23)
  (h101 : g p q r s 101 = 101)
  (hg : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 62 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l52_5278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l52_5225

/-- The equation of circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 4*y = -y^2 + 2*y + 13

/-- The center of circle D -/
def center : ℝ × ℝ := (4, 3)

/-- The radius of circle D -/
noncomputable def radius : ℝ := Real.sqrt 38

/-- The sum of center coordinates and radius -/
noncomputable def sum : ℝ := center.1 + center.2 + radius

theorem circle_properties :
  (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  sum = 7 + Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l52_5225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l52_5290

/-- The volume of the cistern in litres -/
noncomputable def cistern_volume : ℝ := 39.99999999999999

/-- The time taken for pipe B to fill the cistern in minutes -/
noncomputable def pipe_b_time : ℝ := 5

/-- The rate at which pipe C empties the cistern in litres per minute -/
noncomputable def pipe_c_rate : ℝ := 14

/-- The time taken to empty the full cistern when all pipes are open in minutes -/
noncomputable def empty_time : ℝ := 60

/-- The time taken for pipe A to fill the cistern in minutes -/
noncomputable def pipe_a_time : ℝ := cistern_volume / ((cistern_volume / pipe_b_time) + pipe_c_rate - (cistern_volume / empty_time))

theorem pipe_a_fill_time :
  ∃ (ε : ℝ), ε > 0 ∧ |pipe_a_time - 7.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l52_5290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coins_for_success_l52_5269

/-- Represents a circular arrangement of boxes with coins --/
structure CircularArrangement where
  numBoxes : Nat
  numCoins : Nat
  coinDistribution : Fin numBoxes → Nat

/-- Represents a move by Player B --/
def PlayerBMove (arr : CircularArrangement) : CircularArrangement :=
  sorry

/-- Represents a move by Player A --/
def PlayerAMove (arr : CircularArrangement) : CircularArrangement :=
  sorry

/-- Checks if all boxes have at least one coin --/
def allBoxesNonEmpty (arr : CircularArrangement) : Prop :=
  ∀ i, arr.coinDistribution i ≥ 1

/-- Helper function to simulate multiple moves --/
def applyMoves (arr : CircularArrangement) (numMoves : Nat) : CircularArrangement :=
  match numMoves with
  | 0 => arr
  | n + 1 => PlayerAMove (PlayerBMove (applyMoves arr n))

/-- The main theorem stating the minimum number of coins required --/
theorem min_coins_for_success (n : Nat) (h : n = 2012) :
  ∃ N : Nat,
    N = 2 * n - 2 ∧
    (∀ arr : CircularArrangement,
      arr.numBoxes = n →
      arr.numCoins = N →
      allBoxesNonEmpty arr →
      ∀ k : Nat,
        let arrAfterMoves := applyMoves arr k
        allBoxesNonEmpty arrAfterMoves) ∧
    (∀ M : Nat,
      M < N →
      ∃ arr : CircularArrangement,
        arr.numBoxes = n ∧
        arr.numCoins = M ∧
        allBoxesNonEmpty arr ∧
        ∃ k : Nat,
          let arrAfterMoves := applyMoves arr k
          ¬allBoxesNonEmpty arrAfterMoves) :=
  sorry

#check min_coins_for_success

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coins_for_success_l52_5269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l52_5209

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 + 2 * tan x + 6 / tan x + 9 / (tan x ^ 2) + 4

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧
  (∀ (y : ℝ), 0 < y ∧ y < π / 2 → f y ≥ f x) ∧
  f x = 10 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l52_5209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_printing_volume_l52_5200

/-- Represents the monthly growth rate of book printing volume -/
def x : ℝ := sorry

/-- Represents the printing volume in March in thousands of books -/
def y : ℝ := sorry

/-- The initial printing volume in January in thousands of books -/
def january_volume : ℝ := 500

/-- Theorem stating the relationship between March printing volume, initial volume, and growth rate -/
theorem march_printing_volume : 
  y = 50 * (1 + x)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_printing_volume_l52_5200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l52_5253

theorem tan_ratio_from_sin_ratio (α β m n : ℝ) 
  (h : Real.sin (α + β) / Real.sin (α - β) = m / n) :
  Real.tan β / Real.tan α = (m - n) / (m + n) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l52_5253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l52_5234

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through (0,1)
def line_through_point (m : ℝ) (x y : ℝ) : Prop := y = m*x + 1

-- Define the condition for a line to intersect the parabola at exactly one point
def single_intersection (m : ℝ) : Prop :=
  ∃! (x y : ℝ), parabola x y ∧ line_through_point m x y

-- Theorem statement
theorem exactly_three_lines :
  ∃! (l : Finset ℝ), (∀ m ∈ l, single_intersection m) ∧ l.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l52_5234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l52_5249

theorem coefficient_x_cubed_in_expansion : 
  let f : ℕ → ℤ := fun n => (Nat.choose 5 2 * (2^(5-2) * 1^2)) - (Nat.choose 5 3 * (2^(5-3) * 1^3))
  f 3 = -30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l52_5249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l52_5213

/-- The circle with center (1, -3) and radius 1 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 1

/-- The point M -/
def M : ℝ × ℝ := (2, 4)

/-- A line passing through point M -/
def LineM (k : ℝ) (x y : ℝ) : Prop := y = k * (x - M.1) + M.2

/-- Distance from a point to a line -/
noncomputable def DistancePointLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem tangent_line_equation :
  ∀ x y : ℝ,
  (∃ k : ℝ, LineM k x y ∧ DistancePointLine (1, -3) 1 (-k) (4 - 2*k) = 1) ∨
  (x = 2 ∧ DistancePointLine (1, -3) 1 0 (-2) = 1) ↔
  x = 2 ∨ 24 * x - 7 * y - 20 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l52_5213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_leftover_money_l52_5212

/-- Represents the cost and discount information for a grocery item -/
structure GroceryItem where
  name : String
  quantity : Nat
  price : ℝ
  discount : ℝ

/-- Calculates the total cost of grocery shopping -/
def calculateTotalCost (items : List GroceryItem) (taxRate : ℝ) : ℝ :=
  let itemsCost := items.foldl (fun acc item =>
    acc + (item.quantity : ℝ) * item.price * (1 - item.discount)
  ) 0
  itemsCost * (1 + taxRate)

/-- Theorem: Jerry will have approximately $49.71 left after grocery shopping -/
theorem jerry_leftover_money (budget : ℝ) (items : List GroceryItem) (taxRate : ℝ) :
  budget = 100 ∧
  items = [
    { name := "Mustard Oil", quantity := 2, price := 13, discount := 0.1 },
    { name := "Pasta", quantity := 2, price := 4, discount := 0 },
    { name := "Pasta Sauce", quantity := 1, price := 5, discount := 0 },
    { name := "Eggs", quantity := 1, price := 2.54, discount := 0 },
    { name := "Sugar", quantity := 4, price := 2.5, discount := 0.15 }
  ] ∧
  taxRate = 0.06 →
  abs (budget - calculateTotalCost items taxRate - 49.71) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_leftover_money_l52_5212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_approx_l52_5262

-- Define the number of responses needed
noncomputable def responses_needed : ℝ := 300

-- Define the number of questionnaires mailed
noncomputable def questionnaires_mailed : ℝ := 483.87

-- Define the response rate percentage
noncomputable def response_rate_percentage : ℝ := (responses_needed / questionnaires_mailed) * 100

-- Theorem to prove
theorem response_rate_approx :
  |response_rate_percentage - 62.02| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_response_rate_approx_l52_5262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_max_value_l52_5288

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the maximum value function
noncomputable def g (t : ℝ) : ℝ :=
  if t < -2 then -t^2 - 4*t - 2
  else if t ≤ 0 then 2
  else -t^2 + 2

-- Theorem statement
theorem quadratic_function_and_max_value 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (f a b c 0 = 2) →
  (∀ x, f a b c (x + 1) - f a b c x = -2*x - 1) →
  (∀ x, f a b c x = -x^2 + 2) ∧
  (∀ t x, t ≤ x ∧ x ≤ t + 2 → f a b c x ≤ g t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_max_value_l52_5288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l52_5206

/-- Circle passing through two points with center on a line -/
structure CircleWithCenter where
  center : ℝ × ℝ
  radius : ℝ
  -- Circle passes through points A(1, 3) and B(5, 1)
  passes_through_A : (1 - center.1) ^ 2 + (3 - center.2) ^ 2 = radius ^ 2
  passes_through_B : (5 - center.1) ^ 2 + (1 - center.2) ^ 2 = radius ^ 2
  -- Center lies on the line x - y + 1 = 0
  center_on_line : center.1 - center.2 + 1 = 0

/-- Line tangent to the circle and passing through a point -/
structure TangentLine where
  slope : ℝ
  intercept : ℝ
  -- Line passes through (0, 3)
  passes_through : 0 * slope + intercept = 3
  -- Line is tangent to the circle
  is_tangent : (5 * slope - 6 + intercept) ^ 2 = 25 * (slope ^ 2 + 1)

/-- Main theorem statement -/
theorem circle_and_tangent_line 
  (c : CircleWithCenter) (l : TangentLine) : 
  (c.center = (5, 6) ∧ c.radius = 5) ∧ 
  (l.slope = 0 ∨ (l.slope = -8/15 ∧ l.intercept = 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l52_5206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_avg_speed_l52_5298

/-- The position function of the particle -/
noncomputable def s (t : ℝ) : ℝ := 3 + t^2

/-- The average speed function -/
noncomputable def avg_speed (t₁ t₂ : ℝ) : ℝ := (s t₂ - s t₁) / (t₂ - t₁)

/-- Theorem stating that the average speed in the interval [2, 2.1] is 4.1 -/
theorem particle_avg_speed : 
  ∀ ε > 0, |avg_speed 2 2.1 - 4.1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_avg_speed_l52_5298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_primes_l52_5207

theorem right_triangle_from_primes (p q : ℕ) : 
  Prime p → Prime q → 5 * p^2 + 3 * q = 59 →
  (p + 3)^2 + (1 - p + q)^2 = (2 * p + q - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_from_primes_l52_5207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_trig_values_l52_5284

theorem compare_trig_values : 
  Real.cos (3/2) < Real.sin (1/10) ∧ Real.sin (1/10) < -Real.cos (7/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_trig_values_l52_5284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l52_5296

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.log x + x + 1

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 1 / x + 1

-- Theorem statement
theorem tangent_line_equation (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : f' x₀ = 2) :
  ∃ y₀ : ℝ, ∀ x y : ℝ, y - f x₀ = 2 * (x - x₀) ↔ y = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l52_5296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l52_5218

theorem max_prime_factors (a b : ℕ) 
  (h1 : (Nat.gcd a b).factors.length = 10)
  (h2 : (Nat.lcm a b).factors.length = 35)
  (h3 : a.factors.length < b.factors.length) :
  a.factors.length ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l52_5218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_angles_equal_sine_l52_5292

-- Define a type for angles
def Angle : Type := ℝ

-- Define a predicate for symmetry with respect to y-axis
def symmetric_wrt_y_axis (α β : Angle) : Prop := sorry

-- Theorem statement
theorem symmetric_angles_equal_sine (α β : Angle) :
  symmetric_wrt_y_axis α β → Real.sin α = Real.sin β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_angles_equal_sine_l52_5292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_graph_proportions_l52_5201

-- Define the parts of the circle graph
inductive CirclePart
| white
| black
| gray
| light_gray

open CirclePart

-- Define the size of each part
noncomputable def size : CirclePart → ℝ
| white => 1
| black => 1/2
| gray => 1/4
| light_gray => 1/8

-- Define the total size of the circle
noncomputable def total_size : ℝ := size white + size black + size gray + size light_gray

-- Define the proportion of each part
noncomputable def proportion (part : CirclePart) : ℝ := size part / total_size

-- Theorem statement
theorem circle_graph_proportions :
  proportion white = 1/2 ∧
  proportion black = 1/4 ∧
  proportion gray = 1/8 ∧
  proportion light_gray = 1/16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_graph_proportions_l52_5201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_for_all_c_l52_5224

theorem odd_sum_for_all_c (a b c : ℕ) (ha : Even a) (hb : Even b) :
  Odd (3^a + (b - 2)^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_for_all_c_l52_5224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l52_5268

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (c^2 = a^2 + b^2) →
  (2 * c = 2 * b^2 / a) →
  c / a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l52_5268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l52_5248

/-- The time (in seconds) it takes for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  train_length / relative_speed_ms

/-- Theorem stating that the time for a 110 m long train moving at 30 km/h to pass a man 
    moving at 3 km/h in the opposite direction is approximately 12 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, abs (train_passing_time 110 30 3 - 12) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l52_5248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wins_by_three_minutes_l52_5219

-- Define the given conditions
noncomputable def john_speed : ℚ := 15  -- mph
noncomputable def race_distance : ℚ := 5  -- miles
noncomputable def next_fastest_time : ℚ := 23  -- minutes

-- Define John's race time in minutes
noncomputable def john_time : ℚ := (race_distance / john_speed) * 60

-- Theorem to prove
theorem john_wins_by_three_minutes :
  next_fastest_time - john_time = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wins_by_three_minutes_l52_5219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_mats_touch_mat_corners_on_edge_l52_5255

/-- The radius of the round table -/
def table_radius : ℝ := 5

/-- The number of place mats on the table -/
def num_mats : ℕ := 8

/-- The width of each place mat -/
def mat_width : ℝ := 1

/-- The length of each place mat -/
noncomputable def mat_length : ℝ := (Real.sqrt 99) / 2 + 1 / 2

/-- Theorem stating that the length of each place mat is approximately 5.475 units -/
theorem place_mat_length :
  ‖mat_length - 5.475‖ < 0.001 := by sorry

/-- Theorem stating that the inner corners of adjacent mats touch -/
theorem mats_touch :
  ∀ i : Fin num_mats,
    ∃ p : ℝ × ℝ,
      p.1^2 + p.2^2 = table_radius^2 ∧
      (p.1 + mat_width / 2)^2 + (p.2 + mat_length / 2)^2 = table_radius^2 := by sorry

/-- Theorem stating that two corners of each mat lie on the edge of the table -/
theorem mat_corners_on_edge :
  ∀ i : Fin num_mats,
    ∃ p q : ℝ × ℝ,
      p.1^2 + p.2^2 = table_radius^2 ∧
      q.1^2 + q.2^2 = table_radius^2 ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = mat_length^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_mats_touch_mat_corners_on_edge_l52_5255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l52_5226

open Real

-- Define the equation
def equation (x a : ℝ) : Prop :=
  log x + 2 * Real.exp 1 * x^2 = x^3 + (a / Real.exp 1) * x

-- Define the condition of having two different real roots
def has_two_different_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ equation x a ∧ equation y a

-- State the theorem
theorem range_of_a (a : ℝ) :
  has_two_different_real_roots a → a < (Real.exp 1)^3 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l52_5226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_equals_3645_l52_5240

noncomputable section

-- Define the radius of the semicircle
def radius : ℝ := 3

-- Define ω as the fifth root of unity
noncomputable def ω : ℂ := Complex.exp (Real.pi * Complex.I / 5)

-- Define the positions of P, Q, and D_k on the complex plane
def P : ℂ := radius
def Q : ℂ := -radius
noncomputable def D (k : ℕ) : ℂ := radius * ω ^ k

-- Define the length of a chord
noncomputable def chord_length (A B : ℂ) : ℝ := Complex.abs (A - B)

-- Define the product of all chord lengths
noncomputable def chord_product : ℝ :=
  (chord_length P (D 1)) * (chord_length P (D 2)) * (chord_length P (D 3)) *
  (chord_length Q (D 1)) * (chord_length Q (D 2)) * (chord_length Q (D 3)) *
  (chord_length P (D 4)) * (chord_length Q (D 4))

-- Theorem statement
theorem chord_product_equals_3645 : chord_product = 3645 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_equals_3645_l52_5240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l52_5267

/-- The distance between the foci of a hyperbola -/
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 16*y^2 - 16*y = 48

theorem hyperbola_focal_distance :
  ∃ (a b : ℝ), a^2 = 60 ∧ b^2 = 3.75 ∧
  (∀ (x y : ℝ), hyperbola_equation x y ↔ 
    (x - 4)^2 / a^2 - (y + 1/2)^2 / b^2 = 1) ∧
  focal_distance a b = 2 * Real.sqrt 63.75 := by
  sorry

#check hyperbola_focal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l52_5267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l52_5277

/-- Represents a trapezoid with specific properties -/
structure SpecialTrapezoid where
  R : ℝ
  perpendicular_leg : ℝ
  perpendicular_leg_eq : perpendicular_leg = 2 * R
  other_leg_segments : Fin 3 → ℝ
  segment_ratio : other_leg_segments 0 = 12 * (other_leg_segments 2) ∧
                  other_leg_segments 1 = 15 * (other_leg_segments 2) ∧
                  other_leg_segments 2 = 5 * (other_leg_segments 2)

/-- The area of the special trapezoid -/
noncomputable def area (t : SpecialTrapezoid) : ℝ := 7 * t.R^2 / Real.sqrt 15

/-- Theorem stating that the area of the special trapezoid is 7R^2/√15 -/
theorem special_trapezoid_area (t : SpecialTrapezoid) : 
  area t = 7 * t.R^2 / Real.sqrt 15 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l52_5277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_unique_l52_5259

/-- A function f: ℝ → ℝ satisfying f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1 for all x, y ∈ ℝ -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The unique solution to the functional equation -/
noncomputable def Solution : ℝ → ℝ := fun x ↦ 1 - x^2 / 2

theorem solution_unique :
  ∃! f : ℝ → ℝ, FunctionalEquation f ∧ f = Solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_unique_l52_5259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l52_5221

/-- Represents a circle with a center point and a radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if they touch at exactly one point. -/
def AreTangent (c1 c2 : Circle) : Prop := sorry

/-- The distance between the centers of two circles. -/
def CenterDistance (c1 c2 : Circle) : ℝ := sorry

theorem tangent_circles_distance (c1 c2 : Circle) :
  c1.radius = 3 ∧ c2.radius = 4 ∧ AreTangent c1 c2 →
  CenterDistance c1 c2 = 1 ∨ CenterDistance c1 c2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l52_5221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tails_one_head_is_one_eighth_l52_5250

-- Define a fair coin
noncomputable def fair_coin (outcome : Bool) : ℝ := 1 / 2

-- Define the probability of a specific sequence of three flips
noncomputable def prob_two_tails_one_head : ℝ :=
  fair_coin false * fair_coin false * fair_coin true

-- Theorem statement
theorem prob_two_tails_one_head_is_one_eighth :
  prob_two_tails_one_head = 1 / 8 := by
  unfold prob_two_tails_one_head fair_coin
  -- Simplify the expression
  simp [mul_assoc]
  -- Evaluate the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tails_one_head_is_one_eighth_l52_5250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l52_5231

/-- Proves that sin(-870°)cos930° + cos(-990°)sin(23π/6) + tan(13π/4) = √3/4 + 1 -/
theorem trigonometric_identity :
  Real.sin (-870 * Real.pi / 180) * Real.cos (930 * Real.pi / 180) +
  Real.cos (-990 * Real.pi / 180) * Real.sin (23 * Real.pi / 6) +
  Real.tan (13 * Real.pi / 4) = Real.sqrt 3 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l52_5231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_median_mode_difference_l52_5242

def data : List Int := [17, 17, 17, 19, 21, 21, 25, 30, 30, 30, 34, 42, 42, 46, 46, 53, 53, 53, 58]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem data_median_mode_difference :
  Int.natAbs (median data - mode data) = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_median_mode_difference_l52_5242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l52_5280

/-- The polynomial x^3 - 6x^2 + 11x - 6 -/
def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

/-- Sum of the kth powers of the roots of the polynomial -/
noncomputable def s : ℕ → ℝ := sorry

/-- The recurrence relation for s_k -/
def recurrence (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)

theorem sum_of_coefficients :
  s 0 = 3 →
  s 1 = 6 →
  s 2 = 11 →
  ∃ a b c : ℝ, recurrence a b c ∧ a + b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l52_5280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_purchase_price_l52_5202

/-- Given a marked price, discount rate, and profit margin, calculates the purchase price -/
noncomputable def calculate_purchase_price (marked_price : ℝ) (discount_rate : ℝ) (profit_margin : ℝ) : ℝ :=
  (marked_price * (1 - discount_rate)) / (1 + profit_margin)

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_margin : ℝ := 0.1
  calculate_purchase_price marked_price discount_rate profit_margin = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_purchase_price_l52_5202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_bought_l52_5265

def total_money : ℚ := 50
def cupcake_cost : ℚ := 3
def pastry_cost : ℚ := 5/2

def max_cupcakes : ℕ := Int.toNat ⌊total_money / cupcake_cost⌋

def remaining_money : ℚ := total_money - (↑max_cupcakes * cupcake_cost)

def max_pastries : ℕ := Int.toNat ⌊remaining_money / pastry_cost⌋

theorem total_items_bought : max_cupcakes + max_pastries = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_bought_l52_5265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l52_5256

theorem triangle_inequality (A B C k : Real) 
  (triangle_cond : A + B + C = Real.pi)
  (k_range : 1 ≤ k ∧ k ≤ 2) : 
  (1 / (k - Real.cos A)) + (1 / (k - Real.cos B)) + (1 / (k - Real.cos C)) ≥ 6 / (2 * k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l52_5256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_conditions_l52_5233

-- Define the line l
noncomputable def Line (m : ℝ) (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b}

-- Define the area of the triangle formed by the line and coordinate axes
noncomputable def TriangleArea (m : ℝ) (b : ℝ) : ℝ :=
  |b * (b / m)| / 2

-- Define the condition that the line passes through a point
def PassesThrough (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b

-- Theorem statement
theorem line_equation_with_conditions :
  ∀ (m b : ℝ),
    m = 1 →  -- Slope is 1
    PassesThrough m b (-3, 4) →  -- Passes through P(-3,4)
    TriangleArea m b = 3 →  -- Forms a triangle with area 3
    (∃ (k : ℝ), (2 * k + 3) * b = 6 ∧ k * b = 2) ∨  -- 2x + 3y - 6 = 0
    (∃ (k : ℝ), (8 * k + 3) * b = -12 ∧ k * b = 8) :=  -- 8x + 3y + 12 = 0
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_conditions_l52_5233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_iff_excircle_product_l52_5299

/-- A triangle with sides a, b, c, semi-perimeter s, area T, and excircle radii ρa, ρb, ρc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ
  T : ℝ
  ρa : ℝ
  ρb : ℝ
  ρc : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_semiperimeter : s = (a + b + c) / 2
  h_area : T = Real.sqrt (s * (s - a) * (s - b) * (s - c))
  h_excircle_a : T = ρa * (s - a)
  h_excircle_b : T = ρb * (s - b)
  h_excircle_c : T = ρc * (s - c)

/-- A triangle is right-angled if and only if the product of the radii of any two of its excircles
    is equal to the area of the triangle -/
theorem right_triangle_iff_excircle_product (t : Triangle) :
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2 ↔ t.ρa * t.ρb = t.T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_iff_excircle_product_l52_5299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamshid_fifty_percent_faster_l52_5275

/-- The time (in hours) it takes Jamshid to paint the fence alone -/
noncomputable def jamshid_time : ℝ := 7.5

/-- The time (in hours) it takes Taimour to paint the fence alone -/
noncomputable def taimour_time : ℝ := 15

/-- The time (in hours) it takes Jamshid and Taimour to paint the fence together -/
noncomputable def combined_time : ℝ := 5

/-- Jamshid can paint the fence faster than Taimour -/
axiom jamshid_faster : jamshid_time < taimour_time

/-- The combined work rate equals the sum of individual work rates -/
axiom work_rate_sum : 1 / jamshid_time + 1 / taimour_time = 1 / combined_time

/-- The percentage of time Jamshid takes less than Taimour -/
noncomputable def time_difference_percentage : ℝ := (taimour_time - jamshid_time) / taimour_time * 100

theorem jamshid_fifty_percent_faster : time_difference_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamshid_fifty_percent_faster_l52_5275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_comparison_theorem_l52_5257

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Define the solution set
def solution_set : Set ℝ := Set.Iic (-3) ∪ Set.Ici 1

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | f x ≥ 4 - x} = solution_set :=
sorry

-- Theorem for the comparison of 2(a+b) and ab+4
theorem comparison_theorem :
  ∀ a b : ℝ, (∃ x : ℝ, f x = a) → (∃ x : ℝ, f x = b) →
  2 * (a + b) < a * b + 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_comparison_theorem_l52_5257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l52_5251

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (x - 1)

-- State the theorem
theorem f_min_value : 
  ∀ x : ℝ, x ≥ 1 → f x ≥ f 1 ∧ f 1 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l52_5251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_fractions_l52_5260

theorem reciprocal_sum_fractions : 
  (1 / 2 + 2 / 5 + 3 / 4 : ℚ)⁻¹ = 20 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_fractions_l52_5260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l52_5237

theorem point_on_line_cos_value (a : ℝ) :
  (Real.sin a = -2 * Real.cos a) →
  Real.cos (2 * a + π / 2) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l52_5237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l52_5232

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2*x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ (c : ℝ), (∀ (x : ℝ), f (c + x) = f (c - x)) ∧ c = -π/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l52_5232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_downhill_speed_l52_5238

/-- Proves that Jill's downhill speed is 12 feet/second given the conditions -/
theorem jills_downhill_speed
  (hill_height : ℝ)
  (uphill_speed : ℝ)
  (total_time : ℝ)
  (h1 : hill_height = 900)
  (h2 : uphill_speed = 9)
  (h3 : total_time = 175) :
  (let time_up := hill_height / uphill_speed
   let time_down := total_time - time_up
   let downhill_speed := hill_height / time_down
   downhill_speed) = 12 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_downhill_speed_l52_5238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_9801_l52_5289

theorem largest_prime_factor_of_9801 :
  (Nat.factors 9801).maximum? = some 199 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_9801_l52_5289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_c_d_l52_5217

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | Real.log (3*x + 2*y) = z ∧ Real.log (x^3 + y^3) = 2*z}

-- State the theorem
theorem exists_c_d :
  ∃ (c d : ℝ), ∀ (x y z : ℝ), (x, y, z) ∈ T → x^2 + y^2 = c * (10 : ℝ)^(2*z) + d * (10 : ℝ)^z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_c_d_l52_5217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_plates_l52_5246

/-- Represents the sets of letters for each position in the license plate --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card

/-- The initial license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'B', 'F', 'G', 'T', 'Y'},
    second := {'E', 'U'},
    third := {'K', 'S', 'W'} }

/-- The number of new letters to be added --/
def newLettersCount : Nat := 3

/-- Theorem: The maximum number of additional license plates is 50 --/
theorem max_additional_plates :
  ∃ (optimalSets : LicensePlateSets),
    (totalPlates optimalSets - totalPlates initialSets = 50) ∧
    (∀ (anySets : LicensePlateSets),
      (optimalSets.first.card + optimalSets.second.card + optimalSets.third.card =
       initialSets.first.card + initialSets.second.card + initialSets.third.card + newLettersCount) →
      (totalPlates anySets - totalPlates initialSets ≤ 50)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_plates_l52_5246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_lift_sphere_formula_l52_5272

/-- Work required to lift a sphere out of water -/
noncomputable def work_to_lift_sphere (R H δ : ℝ) : ℝ :=
  (4/3) * Real.pi * R^3 * ((δ - 1) * H + R * (2 * δ - 1))

/-- Theorem stating the work required to lift a sphere out of water -/
theorem work_to_lift_sphere_formula (R H δ : ℝ) (hR : R > 0) (hH : H > 0) (hδ : δ > 1) :
  ∃ Q : ℝ, Q = work_to_lift_sphere R H δ ∧ 
  Q = (4/3) * Real.pi * R^3 * ((δ - 1) * H + R * (2 * δ - 1)) := by
  use work_to_lift_sphere R H δ
  constructor
  · rfl
  · rfl

#check work_to_lift_sphere_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_to_lift_sphere_formula_l52_5272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_contrapositive_relation_l52_5297

theorem inverse_contrapositive_relation (p q r : Prop) :
  (q ↔ (¬p → ¬q)) →
  (r ↔ (¬q → ¬p)) →
  (q ↔ ¬r) ∧ (r ↔ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_contrapositive_relation_l52_5297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_GCD_l52_5216

-- Define the square ABCD
def square_ABCD : Set (ℝ × ℝ) := sorry

-- Define the area of the square
noncomputable def area_ABCD : ℝ := 256

-- Define point E on side BC
def point_E : ℝ × ℝ := sorry

-- Define the ratio of BE to EC
noncomputable def BE_EC_ratio : ℝ := 3/1

-- Define point F as midpoint of AE
def point_F : ℝ × ℝ := sorry

-- Define point G as midpoint of DE
def point_G : ℝ × ℝ := sorry

-- Define the area of quadrilateral BEGF
noncomputable def area_BEGF : ℝ := 56

-- Helper function to get points of the square
def point_A (s : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def point_B (s : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def point_C (s : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def point_D (s : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Helper function for midpoint
def is_midpoint (m : ℝ × ℝ) (a b : ℝ × ℝ) : Prop := sorry

-- Helper function for triangle area
noncomputable def area_of_triangle (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_GCD : 
  area_ABCD = 256 ∧ 
  BE_EC_ratio = 3/1 ∧ 
  is_midpoint point_F (point_A square_ABCD) point_E ∧
  is_midpoint point_G (point_D square_ABCD) point_E ∧
  area_BEGF = 56 →
  area_of_triangle point_G (point_C square_ABCD) (point_D square_ABCD) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_GCD_l52_5216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_at_distance_5_l52_5252

/-- A lattice point in 3D space -/
structure LatticePoint3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The squared distance of a lattice point from the origin -/
def squaredDistance (p : LatticePoint3D) : ℤ :=
  p.x^2 + p.y^2 + p.z^2

/-- The set of all lattice points with a squared distance of 25 from the origin -/
def latticePointsAtDistance25 : Set LatticePoint3D :=
  {p : LatticePoint3D | squaredDistance p = 25}

/-- The set of lattice points at distance 25 is finite -/
instance : Fintype latticePointsAtDistance25 := by
  sorry

theorem count_lattice_points_at_distance_5 : 
  Fintype.card latticePointsAtDistance25 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_at_distance_5_l52_5252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_3_l52_5293

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | (n + 2) => (1 + a (n + 1)) / (1 - a (n + 1))

/-- The 2018th term of the sequence equals -3 -/
theorem a_2018_eq_neg_3 : a 2018 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_eq_neg_3_l52_5293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_ratio_l52_5222

/-- A regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point on a line segment -/
def PointOnSegment (A B : ℝ × ℝ) := { P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B }

/-- The ratio of distances AM:AC -/
def ratio (A M C : ℝ × ℝ) : ℝ := sorry

/-- Three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

theorem hexagon_diagonal_ratio 
  (ABCDEF : RegularHexagon) 
  (M : PointOnSegment (ABCDEF.vertices 0) (ABCDEF.vertices 2)) 
  (K : PointOnSegment (ABCDEF.vertices 2) (ABCDEF.vertices 4)) 
  (n : ℝ) 
  (h1 : ratio (ABCDEF.vertices 0) M.val (ABCDEF.vertices 2) = n)
  (h2 : ratio (ABCDEF.vertices 2) K.val (ABCDEF.vertices 4) = n)
  (h3 : collinear (ABCDEF.vertices 1) M.val K.val) :
  n = Real.sqrt 3 / 3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_ratio_l52_5222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_252_equiv_rotate_72_then_180_rotate_252_is_B_l52_5236

/-- Represents a regular pentagon -/
structure RegularPentagon where
  -- Add necessary fields (we'll leave this empty for now)

/-- Represents the result of rotating a regular pentagon -/
inductive RotationResult
  | A | B | C | D | E

/-- Rotates a regular pentagon by the given angle (in degrees) clockwise -/
def rotate (p : RegularPentagon) (angle : ℝ) : RotationResult :=
  sorry

/-- Rotates a regular pentagon and returns another pentagon -/
def rotatePentagon (p : RegularPentagon) (angle : ℝ) : RegularPentagon :=
  sorry

/-- Theorem stating that rotating a regular pentagon by 252° is equivalent to 
    rotating it by 72° and then by 180° -/
theorem rotate_252_equiv_rotate_72_then_180 (p : RegularPentagon) :
  rotate p 252 = rotate p (72 + 180) := by
  sorry

/-- Theorem stating that the result of rotating by 252° is option B -/
theorem rotate_252_is_B (p : RegularPentagon) :
  rotate p 252 = RotationResult.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_252_equiv_rotate_72_then_180_rotate_252_is_B_l52_5236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l52_5271

open Real

-- Define the function
noncomputable def f (x : ℝ) := abs (sin x)

-- State the theorem
theorem f_properties :
  (∀ x y, 0 < x → x < y → y < π / 2 → f x < f y) ∧ 
  (∀ x, f (-x) = f x) ∧
  (∀ x, f (x + π) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l52_5271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_correct_l52_5223

/-- The area enclosed by a curve consisting of 8 congruent circular arcs, where each arc has length 5π/6 and the centers of the corresponding circles are vertices of a regular octagon with side length 3 -/
noncomputable def enclosedArea : ℝ :=
  let numArcs : ℕ := 8
  let arcLength : ℝ := 5 * Real.pi / 6
  let octagonSideLength : ℝ := 3
  18 * (1 + Real.sqrt 2) + 250 * Real.pi / 108

/-- Theorem stating that the enclosed area is correct -/
theorem enclosed_area_correct : enclosedArea = 18 * (1 + Real.sqrt 2) + 250 * Real.pi / 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_correct_l52_5223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_set_for_all_degree_polynomials_l52_5204

theorem no_finite_set_for_all_degree_polynomials :
  ¬ ∃ (M : Set ℝ), (M.Finite ∧ (0 : ℝ) ∉ M) ∧
    (∀ n : ℕ, ∃ (p : Polynomial ℝ),
      (p.degree ≥ n) ∧
      (∀ i : ℕ, p.coeff i ∈ M) ∧
      (∀ r : ℝ, p.eval r = 0 → r ∈ M)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_set_for_all_degree_polynomials_l52_5204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l52_5266

theorem divisibility_problem (n : ℕ) : 
  (n = 1010) →
  (∀ k ∈ ({12, 18, 21, 28} : Set ℕ), (n - 2) % k = 0) →
  (∀ m < n, ∃ k ∈ ({12, 18, 21, 28} : Set ℕ), (m - 2) % k ≠ 0) →
  ((n - 2) % 4 = 0) ∧ (4 ∉ ({12, 18, 21, 28} : Set ℕ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l52_5266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l52_5235

/-- Given two parallel vectors a and b, prove that m = -1/2 --/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![2, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l52_5235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_range_l52_5261

open Real Set

theorem function_value_range (x : ℝ) :
  let y := (sin x * cos x) / (1 + sin x - cos x)
  let t := sin x - cos x
  (t ≠ -1) →
  y ∈ Icc (-(sqrt 2 + 1) / 2) ((sqrt 2 - 1) / 2) ∧
  y ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_range_l52_5261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_s_squared_minus_t_squared_l52_5203

theorem min_value_s_squared_minus_t_squared 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let s := Real.sqrt (x + 2) + Real.sqrt (y + 5) + Real.sqrt (z + 10)
  let t := Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1)
  s^2 - t^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_s_squared_minus_t_squared_l52_5203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_of_similar_polynomial_l52_5276

/-- Two polynomials are similar if they have the same degree and their coefficients are permutations of each other -/
def similar (P Q : Polynomial ℤ) : Prop :=
  (P.natDegree = Q.natDegree) ∧ 
  ∃ π : Fin (P.natDegree + 1) ≃ Fin (Q.natDegree + 1), 
    ∀ i : Fin (P.natDegree + 1), Q.coeff i = P.coeff (π i)

theorem smallest_absolute_value_of_similar_polynomial 
  (P Q : Polynomial ℤ) (h_similar : similar P Q) (h_eval : P.eval 16 = 3^2012) :
  ∃ k : ℤ, k = |Q.eval (3^2012)| ∧ k ≥ 1 ∧ ∀ m : ℤ, m = |Q.eval (3^2012)| → m ≥ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_of_similar_polynomial_l52_5276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l52_5281

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) (θ : ℝ) := cos (2 * x + θ)

-- State the theorem
theorem odd_function_properties (θ : ℝ) (α : ℝ) 
  (h1 : 0 < θ) (h2 : θ < π)
  (h3 : π / 2 < α) (h4 : α < π)
  (h5 : ∀ x, f x θ = -f (-x) θ)  -- f is odd
  (h6 : f (α / 2) θ = -4 / 5) :
  -- 1. Minimum positive period is π
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) θ = f x θ) ∧ 
    (∀ S, S > 0 → (∀ x, f (x + S) θ = f x θ) → T ≤ S)) ∧
  -- 2. f(x) = -sin(2x)
  (∀ x, f x θ = -sin (2 * x)) ∧
  -- 3. sin(α + π/3) = (4 - 3√3)/10
  sin (α + π / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l52_5281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_l52_5263

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/3 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 3/5 → 
  n + k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_l52_5263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_u_at_A_towards_B_l52_5247

/-- The function u(x, y, z) = x^2 - arctan(y + z) -/
noncomputable def u (x y z : ℝ) : ℝ := x^2 - Real.arctan (y + z)

/-- Point A -/
def A : Fin 3 → ℝ := ![2, 1, 1]

/-- Point B -/
def B : Fin 3 → ℝ := ![2, 4, -3]

/-- Direction vector from A to B -/
def direction : Fin 3 → ℝ := λ i => B i - A i

/-- Theorem: The directional derivative of u at point A in the direction of B is 1/25 -/
theorem directional_derivative_u_at_A_towards_B :
  let grad_u := ![2 * A 0, -1 / (1 + (A 1 + A 2)^2), -1 / (1 + (A 1 + A 2)^2)]
  let norm := Real.sqrt (direction 0^2 + direction 1^2 + direction 2^2)
  let dir_normalized := λ i => direction i / norm
  (grad_u 0 * dir_normalized 0 + grad_u 1 * dir_normalized 1 + grad_u 2 * dir_normalized 2) = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_u_at_A_towards_B_l52_5247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l52_5205

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) (D : ℝ) :
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b →
  t.A + t.B + t.C = π →
  D ∈ Set.Icc 0 t.c →
  6^2 + 4^2 = 8^2 + D * (t.c - D) →
  Real.tan t.C = Real.sqrt 3 ∧ 
  t.c = 3 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l52_5205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_number_gcd_bound_triangular_number_gcd_max_l52_5227

def T (n : ℕ) : ℚ := (n + 1) * n / 2

theorem triangular_number_gcd_bound (n : ℕ) : 
  Nat.gcd (Int.toNat ⌊3 * T n + n⌋) (n + 3) ≤ 12 :=
sorry

theorem triangular_number_gcd_max : 
  ∃ n : ℕ, Nat.gcd (Int.toNat ⌊3 * T n + n⌋) (n + 3) = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_number_gcd_bound_triangular_number_gcd_max_l52_5227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_l52_5291

/-- Given a triangle with medians of lengths 3, 4, and 5, its area is 8. -/
theorem triangle_area_from_medians (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let area := (4 / 3) * Real.sqrt (3 * (a^2 + b^2 + c^2) - (a + b + c)^2)
  area = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_l52_5291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_equality_l52_5283

-- Define the circle and points
variable (O : Type) (Point : Type)
variable (A B C D M : Point)
variable (circle : O → Type)
variable (on_circle : Point → O → Prop)
variable (intersect : Point → Point → Point → Point → Point → Prop)

-- Define distance between points
variable (dist : Point → Point → ℝ)

-- State the theorem
theorem intersection_ratio_equality 
  {o : O}
  (h_circle : on_circle A o ∧ on_circle B o ∧ on_circle C o ∧ on_circle D o)
  (h_intersect : intersect A B C D M) :
  (dist A C * dist A D) / dist A M = (dist B C * dist B D) / dist B M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_equality_l52_5283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_is_correct_l52_5295

/-- The inequality mx^2 + 2mx - 4 < 2x^2 + 4x holds for all real x -/
def inequality_holds_for_all_x (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x

/-- The range of m for which the inequality holds for all x -/
def m_range : Set ℝ := {m : ℝ | inequality_holds_for_all_x m}

theorem m_range_is_correct : m_range = Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_is_correct_l52_5295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l52_5243

/-- The area of the triangle formed by two lines and the y-axis -/
theorem triangle_area (a b c d : ℝ) (h : a ≠ b) :
  (1/2) * (d - c) * ((d - c) / (a - b)) = 32.4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l52_5243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_zero_l52_5294

theorem complex_expression_equals_zero : 
  (3 + I) / (1 - 3*I) + (1 / I) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_zero_l52_5294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_M_N_l52_5229

def U : Set ℕ := {x : ℕ | 0 < x ∧ x < 9}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union_M_N : (M ∪ N)ᶜ = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_M_N_l52_5229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_smaller_cubes_l52_5239

def original_cube_edge : ℕ := 4

def is_valid_partition (partition : List ℕ) : Prop :=
  (partition.sum = original_cube_edge ^ 3) ∧
  (∀ x, x ∈ partition → x > 0 ∧ x ≤ original_cube_edge ^ 3) ∧
  (∃ a b, a ∈ partition ∧ b ∈ partition ∧ a ≠ b)

def max_partition_size : ℕ := 57

theorem max_smaller_cubes :
  ∀ partition : List ℕ,
    is_valid_partition partition →
    partition.length ≤ max_partition_size :=
by
  intro partition h
  sorry

#check max_smaller_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_smaller_cubes_l52_5239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_pythagorean_triple_l52_5228

/-- A function to check if a triple of natural numbers is a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of numbers -/
def set_A : List ℕ := [2, 3, 4]
def set_B : List ℚ := [3/10, 4/10, 5/10]
def set_C : List ℕ := [8, 11, 12]
def set_D : List ℕ := [6, 8, 10]

/-- Theorem stating that only set D is a Pythagorean triple -/
theorem only_D_is_pythagorean_triple :
  (¬ is_pythagorean_triple (set_A.get! 0) (set_A.get! 1) (set_A.get! 2)) ∧
  (¬ is_pythagorean_triple (set_C.get! 0) (set_C.get! 1) (set_C.get! 2)) ∧
  (is_pythagorean_triple (set_D.get! 0) (set_D.get! 1) (set_D.get! 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_pythagorean_triple_l52_5228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l52_5211

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | |x^2 - 2| < 2}

-- State the theorem
theorem inequality_solution_set : 
  solution_set = Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l52_5211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_solution_l52_5230

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (0, 3)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)

-- Define the function to calculate the area of a triangle given its base and height
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

-- Define the function to calculate the area of the right part of the triangle
noncomputable def rightAreaFunction (a : ℝ) : ℝ := triangleArea (10 - a) 3

-- Theorem statement
theorem equal_area_division (a : ℝ) :
  a = 5 ↔ rightAreaFunction a = triangleArea 10 3 / 2 := by
  sorry

-- Additional theorem to explicitly state the solution
theorem solution : ∃ a : ℝ, a = 5 ∧ rightAreaFunction a = triangleArea 10 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_solution_l52_5230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_p_plus_s_zero_l52_5264

-- Define the curve equation
noncomputable def curve_equation (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- Theorem statement
theorem symmetry_implies_p_plus_s_zero
  (p q r s : ℝ)
  (h_p : p ≠ 0)
  (h_q : q ≠ 0)
  (h_r : r ≠ 0)
  (h_s : s ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = curve_equation p q r s x → x = curve_equation p q r s y) :
  p + s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_p_plus_s_zero_l52_5264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l52_5220

/-- Approximate perimeter formula for an ellipse -/
noncomputable def approximate_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

/-- Area formula for an ellipse -/
noncomputable def ellipse_area (a b : ℝ) : ℝ := Real.pi * a * b

theorem ellipse_area_theorem :
  ∀ a b : ℝ,
  a > 0 ∧ b > 0 ∧
  a = b + 4 ∧
  approximate_perimeter a b = 18 →
  ellipse_area a b = 5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l52_5220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_term_is_98_l52_5286

def mySequence : List ℕ := [3, 15, 17, 51, 53]

def pattern (n : ℕ) : ℕ :=
  if n % 2 = 1 then 12 + 11 * ((n - 1) / 2) else 2

theorem last_term_is_98 :
  (mySequence ++ [mySequence.getLast! + pattern mySequence.length]).getLast! = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_term_is_98_l52_5286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l52_5254

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

-- State the theorem
theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a y ≥ f a x) ∧
  (∃ x ∈ Set.Icc 1 4, f a x = -16/3) →
  (∃ z ∈ Set.Icc 1 4, ∀ w ∈ Set.Icc 1 4, f a w ≤ f a z) ∧
  (∃ z ∈ Set.Icc 1 4, f a z = 10/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l52_5254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_set_of_integer_sums_l52_5215

theorem finite_set_of_integer_sums (a b : ℕ+) :
  Set.Finite {n : ℕ | ∃ k : ℤ, ((a : ℝ) + 1/2)^n + ((b : ℝ) + 1/2)^n = k} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_set_of_integer_sums_l52_5215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l52_5274

-- Define the function f(x) = ln(x) / e^x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

-- Theorem statement
theorem f_properties :
  -- f(x) < 0 for all x in (0, 1)
  (∀ x : ℝ, 0 < x → x < 1 → f x < 0) ∧
  -- f(x) has exactly one critical point
  (∃! x : ℝ, x > 0 ∧ (deriv f) x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l52_5274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l52_5241

/-- The time taken for a person to cover the entire length of an escalator -/
noncomputable def escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) : ℝ :=
  escalator_length / (escalator_speed + person_speed)

/-- Theorem: The time taken for a person to cover the entire length of an escalator is 8 seconds -/
theorem escalator_problem :
  let escalator_speed : ℝ := 12
  let person_speed : ℝ := 8
  let escalator_length : ℝ := 160
  escalator_time escalator_speed person_speed escalator_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l52_5241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firm_ratio_theorem_l52_5282

/-- Represents the ratio of partners to associates in a firm -/
structure FirmRatio where
  partners : ℕ
  associates : ℕ
deriving Repr

/-- Calculates the new ratio after hiring more associates -/
def newRatio (initial : FirmRatio) (current_partners : ℕ) (new_associates : ℕ) : FirmRatio :=
  let original_associates := (initial.associates * current_partners) / initial.partners
  let total_associates := original_associates + new_associates
  ⟨current_partners, total_associates⟩

/-- Simplifies a ratio by dividing both parts by their GCD -/
def simplifyRatio (r : FirmRatio) : FirmRatio :=
  let gcd := Nat.gcd r.partners r.associates
  ⟨r.partners / gcd, r.associates / gcd⟩

theorem firm_ratio_theorem (initial : FirmRatio) (current_partners : ℕ) (new_associates : ℕ) :
  initial.partners = 2 →
  initial.associates = 63 →
  current_partners = 14 →
  new_associates = 35 →
  simplifyRatio (newRatio initial current_partners new_associates) = ⟨1, 34⟩ := by
  sorry

#eval simplifyRatio (newRatio ⟨2, 63⟩ 14 35)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firm_ratio_theorem_l52_5282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equation_solution_l52_5287

-- Define the functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 1
def g (x : ℝ) : ℝ := x^4 - x^3 + 4*x^2 + 8*x + 8
def h (x : ℝ) : ℝ := x + 1

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Define the solutions
noncomputable def solution1 : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def solution2 : ℝ := (3 - Real.sqrt 5) / 2

-- Theorem statement
theorem inverse_function_equation_solution :
  (∀ x, f (f_inv x) = x) →
  (f_inv (g solution1) = h solution1) ∧
  (f_inv (g solution2) = h solution2) ∧
  (∀ x, f_inv (g x) = h x → x = solution1 ∨ x = solution2) := by
  sorry

#check inverse_function_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equation_solution_l52_5287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l52_5270

-- Define the line l
noncomputable def line_l (t φ : ℝ) : ℝ × ℝ := (t * Real.cos φ, -2 + t * Real.sin φ)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the intersection points
def intersection_points (φ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, line_l t φ = p ∧ p ∈ Set.range curve_C}

-- Statement of the theorem
theorem midpoint_trajectory (φ : ℝ) (P₁ P₂ : ℝ × ℝ) :
  φ ∈ Set.Ioo (Real.pi / 3) (2 * Real.pi / 3) →
  P₁ ∈ intersection_points φ →
  P₂ ∈ intersection_points φ →
  P₁ ≠ P₂ →
  ∃ t, line_l t φ = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) →
  (Real.sin (2 * φ), -1 - Real.cos (2 * φ)) = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l52_5270
