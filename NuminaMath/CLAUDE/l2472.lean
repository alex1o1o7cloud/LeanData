import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2472_247244

theorem expression_evaluation :
  let x : ℚ := -1
  let y : ℚ := -2
  ((x + y)^2 - (3*x - y)*(3*x + y) - 2*y^2) / (-2*x) = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2472_247244


namespace NUMINAMATH_CALUDE_circle_center_l2472_247288

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = 16

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = 16 is (4, -2) -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 4 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2472_247288


namespace NUMINAMATH_CALUDE_nested_expression_equals_4094_l2472_247258

def nested_expression : ℕ := 2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2*(1+2))))))))))

theorem nested_expression_equals_4094 : nested_expression = 4094 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_4094_l2472_247258


namespace NUMINAMATH_CALUDE_equation_solution_l2472_247272

theorem equation_solution : ∃ (x y : ℝ), x + y + x*y = 4 ∧ 3*x*y = 4 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2472_247272


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2472_247275

/-- The equation of the tangent line to the parabola y = x² that is parallel to y = 2x -/
theorem tangent_line_to_parabola (x y : ℝ) : 
  (∀ t, y = t^2 → (2 * t = 2 → x = t ∧ y = t^2)) →
  (2 * x - y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2472_247275


namespace NUMINAMATH_CALUDE_watch_cost_price_l2472_247293

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp * 0.79 = cp * 1.04 - 140) ∧ 
  cp = 560 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2472_247293


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l2472_247200

theorem number_exceeding_percentage : ∃ x : ℝ, x = 75 ∧ x = 0.16 * x + 63 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l2472_247200


namespace NUMINAMATH_CALUDE_parallelogram_area_48_36_l2472_247267

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters -/
theorem parallelogram_area_48_36 : parallelogram_area 48 36 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_48_36_l2472_247267


namespace NUMINAMATH_CALUDE_cos_alpha_proof_l2472_247289

def angle_alpha : ℝ := sorry

def point_P : ℝ × ℝ := (-4, 3)

theorem cos_alpha_proof :
  point_P.1 = -4 ∧ point_P.2 = 3 →
  point_P ∈ {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p = (r * Real.cos angle_alpha, r * Real.sin angle_alpha)} →
  Real.cos angle_alpha = -4/5 := by
  sorry

#check cos_alpha_proof

end NUMINAMATH_CALUDE_cos_alpha_proof_l2472_247289


namespace NUMINAMATH_CALUDE_salary_increase_after_three_years_l2472_247202

-- Define the annual raise rate
def annual_raise : ℝ := 1.15

-- Define the number of years
def years : ℕ := 3

-- Theorem statement
theorem salary_increase_after_three_years :
  (annual_raise ^ years - 1) * 100 = 52.0875 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_three_years_l2472_247202


namespace NUMINAMATH_CALUDE_original_number_proof_l2472_247276

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 5/2) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2472_247276


namespace NUMINAMATH_CALUDE_tan_half_angle_second_quadrant_l2472_247273

/-- If α is an angle in the second quadrant and 3sinα + 4cosα = 0, then tan(α/2) = 2 -/
theorem tan_half_angle_second_quadrant (α : Real) : 
  π/2 < α ∧ α < π → -- α is in the second quadrant
  3 * Real.sin α + 4 * Real.cos α = 0 → -- given equation
  Real.tan (α/2) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_half_angle_second_quadrant_l2472_247273


namespace NUMINAMATH_CALUDE_karens_cookies_l2472_247206

/-- Theorem: Karen's Cookies --/
theorem karens_cookies (
  kept_for_self : ℕ)
  (given_to_grandparents : ℕ)
  (class_size : ℕ)
  (cookies_per_person : ℕ)
  (h1 : kept_for_self = 10)
  (h2 : given_to_grandparents = 8)
  (h3 : class_size = 16)
  (h4 : cookies_per_person = 2)
  : kept_for_self + given_to_grandparents + class_size * cookies_per_person = 50 := by
  sorry

end NUMINAMATH_CALUDE_karens_cookies_l2472_247206


namespace NUMINAMATH_CALUDE_original_function_equation_l2472_247265

/-- Given a vector OA and a quadratic function transformed by OA,
    prove that the original function has the form y = x^2 + 2x - 2 -/
theorem original_function_equation
  (OA : ℝ × ℝ)
  (h_OA : OA = (4, 3))
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^2 + b*x + c)
  (h_tangent : ∀ x y, y = f (x - 4) + 3 → (4*x + y - 8 = 0 ↔ x = 1 ∧ y = 4)) :
  b = 2 ∧ c = -2 :=
sorry

end NUMINAMATH_CALUDE_original_function_equation_l2472_247265


namespace NUMINAMATH_CALUDE_union_of_sets_l2472_247232

theorem union_of_sets : 
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-1, 0}
  A ∪ B = {-1, 0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2472_247232


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l2472_247233

/-- Proves that Austin bought robots for 7 friends given the problem conditions --/
theorem austin_robot_purchase (robot_cost : ℚ) (tax : ℚ) (change : ℚ) (initial_amount : ℚ) 
  (h1 : robot_cost = 8.75)
  (h2 : tax = 7.22)
  (h3 : change = 11.53)
  (h4 : initial_amount = 80) :
  (initial_amount - (change + tax)) / robot_cost = 7 := by
  sorry

#eval (80 : ℚ) - (11.53 + 7.22)
#eval ((80 : ℚ) - (11.53 + 7.22)) / 8.75

end NUMINAMATH_CALUDE_austin_robot_purchase_l2472_247233


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2472_247229

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs -/
def pairSequence : ℕ → IntPair :=
  sorry

/-- The 60th pair in the sequence -/
def sixtiethPair : IntPair :=
  pairSequence 60

theorem sixtieth_pair_is_five_seven :
  sixtiethPair = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2472_247229


namespace NUMINAMATH_CALUDE_expression_evaluation_l2472_247283

theorem expression_evaluation (a b : ℚ) (ha : a = 7) (hb : b = 5) :
  3 * (a^3 + b^3) / (a^2 - a*b + b^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2472_247283


namespace NUMINAMATH_CALUDE_group_formation_count_l2472_247290

def total_people : ℕ := 7
def group_size_1 : ℕ := 3
def group_size_2 : ℕ := 4

theorem group_formation_count :
  Nat.choose total_people group_size_1 = 35 :=
by sorry

end NUMINAMATH_CALUDE_group_formation_count_l2472_247290


namespace NUMINAMATH_CALUDE_palic_function_is_quadratic_l2472_247255

-- Define the Palić function
def PalicFunction (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

-- Define the theorem
theorem palic_function_is_quadratic 
  (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : PalicFunction f a b c) : 
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x^2 + B * x + C := by
sorry

end NUMINAMATH_CALUDE_palic_function_is_quadratic_l2472_247255


namespace NUMINAMATH_CALUDE_solve_equation_l2472_247254

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem solve_equation (a : ℝ) (ha : a > 0) (h : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_solve_equation_l2472_247254


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_in_cones_l2472_247294

theorem liquid_rise_ratio_in_cones (r₁ r₂ r_marble : ℝ) 
  (h₁ h₂ : ℝ) (V : ℝ) :
  r₁ = 4 →
  r₂ = 8 →
  r_marble = 2 →
  V = (1/3) * π * r₁^2 * h₁ →
  V = (1/3) * π * r₂^2 * h₂ →
  let V_marble := (4/3) * π * r_marble^3
  let h₁' := h₁ + V_marble / ((1/3) * π * r₁^2)
  let h₂' := h₂ + V_marble / ((1/3) * π * r₂^2)
  (h₁' - h₁) / (h₂' - h₂) = 4 :=
by sorry

#check liquid_rise_ratio_in_cones

end NUMINAMATH_CALUDE_liquid_rise_ratio_in_cones_l2472_247294


namespace NUMINAMATH_CALUDE_age_difference_mandy_sarah_l2472_247282

/-- Given the ages and relationships of Mandy's siblings, prove the age difference between Mandy and Sarah. -/
theorem age_difference_mandy_sarah :
  let mandy_age : ℕ := 3
  let tom_age : ℕ := 5 * mandy_age
  let julia_age : ℕ := tom_age - 3
  let max_age : ℕ := 2 * julia_age + 2
  let sarah_age : ℕ := max_age + 4
  sarah_age - mandy_age = 27 := by sorry

end NUMINAMATH_CALUDE_age_difference_mandy_sarah_l2472_247282


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2472_247299

theorem hyperbola_sum (F₁ F₂ : ℝ × ℝ) (h k a b : ℝ) :
  F₁ = (-2, 0) →
  F₂ = (2, 0) →
  a > 0 →
  b > 0 →
  (∀ P : ℝ × ℝ, |dist P F₁ - dist P F₂| = 2 ↔ 
    (P.1 - h)^2 / a^2 - (P.2 - k)^2 / b^2 = 1) →
  h + k + a + b = 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2472_247299


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l2472_247250

/-- Represents the cost in dollars to fill a bathtub with jello --/
def jelloBathtubCost (jelloMixPerPound : Real) (bathtubCapacity : Real) 
  (cubicFeetToGallons : Real) (poundsPerGallon : Real) (jelloMixCost : Real) : Real :=
  jelloMixPerPound * bathtubCapacity * cubicFeetToGallons * poundsPerGallon * jelloMixCost

/-- Theorem stating the cost to fill a bathtub with jello is $270 --/
theorem jello_bathtub_cost_is_270 :
  jelloBathtubCost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#eval jelloBathtubCost 1.5 6 7.5 8 0.5

end NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l2472_247250


namespace NUMINAMATH_CALUDE_total_cost_equals_12_46_l2472_247278

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 249/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 187/100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem total_cost_equals_12_46 : total_cost = 1246/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_equals_12_46_l2472_247278


namespace NUMINAMATH_CALUDE_smallest_consecutive_sum_theorem_l2472_247240

/-- The smallest natural number that can be expressed as the sum of 9, 10, and 11 consecutive non-zero natural numbers. -/
def smallest_consecutive_sum : ℕ := 495

/-- Checks if a natural number can be expressed as the sum of n consecutive non-zero natural numbers. -/
def is_sum_of_consecutive (x n : ℕ) : Prop :=
  ∃ k : ℕ, x = (n * (2*k + n + 1)) / 2

/-- The main theorem stating that smallest_consecutive_sum is the smallest natural number
    that can be expressed as the sum of 9, 10, and 11 consecutive non-zero natural numbers. -/
theorem smallest_consecutive_sum_theorem :
  (is_sum_of_consecutive smallest_consecutive_sum 9) ∧
  (is_sum_of_consecutive smallest_consecutive_sum 10) ∧
  (is_sum_of_consecutive smallest_consecutive_sum 11) ∧
  (∀ m : ℕ, m < smallest_consecutive_sum →
    ¬(is_sum_of_consecutive m 9 ∧ is_sum_of_consecutive m 10 ∧ is_sum_of_consecutive m 11)) :=
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sum_theorem_l2472_247240


namespace NUMINAMATH_CALUDE_factorial_product_not_perfect_square_l2472_247226

theorem factorial_product_not_perfect_square (n : ℕ) (hn : n ≥ 100) :
  ¬ ∃ m : ℕ, n.factorial * (n + 1).factorial = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_not_perfect_square_l2472_247226


namespace NUMINAMATH_CALUDE_triangle_inequality_l2472_247266

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2472_247266


namespace NUMINAMATH_CALUDE_number_of_teams_l2472_247251

theorem number_of_teams (n : ℕ) (k : ℕ) : n = 10 → k = 5 → Nat.choose n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teams_l2472_247251


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2472_247249

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 208 / 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2472_247249


namespace NUMINAMATH_CALUDE_grid_sum_invariant_l2472_247216

/-- Represents a 5x5 grid where each cell contains a natural number -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Represents a sequence of 25 moves to fill the grid -/
def MoveSequence := Fin 25 → Fin 5 × Fin 5

/-- Checks if two cells are adjacent in the grid -/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Generates a grid based on a move sequence -/
def generateGrid (moves : MoveSequence) : Grid :=
  sorry

/-- Calculates the sum of all numbers in a grid -/
def gridSum (g : Grid) : ℕ :=
  sorry

/-- The main theorem: the sum of all numbers in the grid is always 40 -/
theorem grid_sum_invariant (moves : MoveSequence) :
  gridSum (generateGrid moves) = 40 :=
  sorry

end NUMINAMATH_CALUDE_grid_sum_invariant_l2472_247216


namespace NUMINAMATH_CALUDE_paper_I_passing_percentage_l2472_247219

/-- Calculates the passing percentage for an exam given the maximum marks,
    the marks secured by a candidate, and the marks by which they failed. -/
def calculate_passing_percentage (max_marks : ℕ) (secured_marks : ℕ) (failed_by : ℕ) : ℚ :=
  let passing_marks : ℕ := secured_marks + failed_by
  (passing_marks : ℚ) / max_marks * 100

/-- Theorem stating that the passing percentage for Paper I is 40% -/
theorem paper_I_passing_percentage :
  calculate_passing_percentage 150 40 20 = 40 := by
sorry

end NUMINAMATH_CALUDE_paper_I_passing_percentage_l2472_247219


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2472_247221

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 250 →
  crossing_time = 20 →
  train_speed = 66.6 →
  (train_speed * crossing_time) - train_length = 1082 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2472_247221


namespace NUMINAMATH_CALUDE_one_intersection_iff_a_in_set_l2472_247201

-- Define the function
def f (a x : ℝ) : ℝ := a * x^2 - a * x + 3 * x + 1

-- Define the condition for exactly one intersection point
def has_one_intersection (a : ℝ) : Prop :=
  ∃! x, f a x = 0

-- Theorem statement
theorem one_intersection_iff_a_in_set :
  ∀ a : ℝ, has_one_intersection a ↔ a ∈ ({0, 1, 9} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_one_intersection_iff_a_in_set_l2472_247201


namespace NUMINAMATH_CALUDE_debby_jogged_nine_km_on_wednesday_l2472_247284

/-- The distance Debby jogged on Monday in kilometers -/
def monday_distance : ℕ := 2

/-- The distance Debby jogged on Tuesday in kilometers -/
def tuesday_distance : ℕ := 5

/-- The total distance Debby jogged over three days in kilometers -/
def total_distance : ℕ := 16

/-- The distance Debby jogged on Wednesday in kilometers -/
def wednesday_distance : ℕ := total_distance - (monday_distance + tuesday_distance)

theorem debby_jogged_nine_km_on_wednesday :
  wednesday_distance = 9 := by sorry

end NUMINAMATH_CALUDE_debby_jogged_nine_km_on_wednesday_l2472_247284


namespace NUMINAMATH_CALUDE_circle_equation_l2472_247287

open Real

/-- A circle C in polar coordinates -/
structure PolarCircle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a line in polar form -/
def polarLine (θ₀ : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun ρ θ ↦ ρ * sin (θ - θ₀) = k

theorem circle_equation (C : PolarCircle) 
  (h1 : C.passesThrough = (2 * sqrt 2, π/4))
  (h2 : C.center.1 = 2 ∧ C.center.2 = 0)
  (h3 : polarLine (π/3) (-sqrt 3) C.center.1 C.center.2) :
  ∀ θ, ∃ ρ, ρ = 4 * cos θ ∧ (ρ * cos θ - C.center.1)^2 + (ρ * sin θ - C.center.2)^2 = (2 * sqrt 2 - C.center.1)^2 + (2 * sqrt 2 - C.center.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2472_247287


namespace NUMINAMATH_CALUDE_trig_sum_zero_l2472_247238

theorem trig_sum_zero : Real.sin (0 * π / 180) + Real.cos (90 * π / 180) + Real.tan (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l2472_247238


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2472_247263

theorem complex_reciprocal_sum_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2472_247263


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2472_247256

/-- Given two points (m, n) and (m + 3, n + 9) in the coordinate plane,
    prove that the equation y = 3x + (n - 3m) represents the line passing through these points. -/
theorem line_equation_through_points (m n : ℝ) :
  ∀ x y : ℝ, y = 3 * x + (n - 3 * m) ↔ (∃ t : ℝ, x = m + 3 * t ∧ y = n + 9 * t) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2472_247256


namespace NUMINAMATH_CALUDE_starting_number_proof_l2472_247291

theorem starting_number_proof : ∃ (n : ℕ), 
  n = 220 ∧ 
  n < 580 ∧ 
  (∃ (m : ℕ), m = 6 ∧ 
    (∀ k : ℕ, n ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n)) ∧
  (∀ n' : ℕ, n < n' → n' < 580 → 
    ¬(∃ (m : ℕ), m = 6 ∧ 
      (∀ k : ℕ, n' ≤ k ∧ k ≤ 580 → (k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0) ↔ k ∈ Finset.range (m + 1) ∧ k ≠ n'))) :=
by sorry

end NUMINAMATH_CALUDE_starting_number_proof_l2472_247291


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2472_247260

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2472_247260


namespace NUMINAMATH_CALUDE_factorial_34_representation_l2472_247253

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def decimal_rep (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_34_representation (a b : ℕ) :
  decimal_rep (factorial 34) = [2, 9, 5, 2, 3, 2, 7, 9, 9, 0, 3, 9, a, 0, 4, 1, 4, 0, 8, 4, 7, 6, 1, 8, 6, 0, 9, 6, 4, 3, 5, b, 0, 0, 0, 0, 0, 0, 0] →
  a = 6 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_34_representation_l2472_247253


namespace NUMINAMATH_CALUDE_range_of_a_l2472_247224

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then (a + 1) * x - 2 * a else Real.log x / Real.log 3

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ∈ Set.Ioi (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2472_247224


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l2472_247264

-- Define the height of the sculpture in inches
def sculpture_height : ℕ := 2 * 12 + 10

-- Define the height of the base in inches
def base_height : ℕ := 2

-- Define the total height in inches
def total_height : ℕ := sculpture_height + base_height

-- Theorem to prove
theorem sculpture_and_base_height :
  total_height / 12 = 3 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l2472_247264


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2472_247215

/-- Given a parabola with equation x² = 8y, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), x^2 = 8*y → (∃ (focus_distance : ℝ), focus_distance = 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2472_247215


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2472_247205

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4, 6, 8, 10}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2472_247205


namespace NUMINAMATH_CALUDE_function_property_l2472_247269

/-- A function satisfying f(x) + 3f(1 - x) = 4x^3 for all real x has f(4) = -72.5 -/
theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : 
  f 4 = -72.5 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2472_247269


namespace NUMINAMATH_CALUDE_anne_heavier_than_douglas_l2472_247279

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

/-- Theorem stating that Anne is 15 pounds heavier than Douglas -/
theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_heavier_than_douglas_l2472_247279


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2472_247252

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m*y^2 = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1 - 1/m

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  1 = 2 * (1/m).sqrt

/-- Theorem: For an ellipse with equation x^2 + my^2 = 1, where the foci are on the x-axis
    and the length of the major axis is twice the length of the minor axis, m = 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
    (h1 : foci_on_x_axis e) (h2 : major_axis_twice_minor e) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2472_247252


namespace NUMINAMATH_CALUDE_y_relationship_l2472_247248

/-- The function f(x) = -x² + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 5

/-- y₁ is the y-coordinate of the point (-4, y₁) on the graph of f -/
def y₁ : ℝ := f (-4)

/-- y₂ is the y-coordinate of the point (-1, y₂) on the graph of f -/
def y₂ : ℝ := f (-1)

/-- y₃ is the y-coordinate of the point (2, y₃) on the graph of f -/
def y₃ : ℝ := f 2

theorem y_relationship : y₂ > y₃ ∧ y₃ > y₁ := by sorry

end NUMINAMATH_CALUDE_y_relationship_l2472_247248


namespace NUMINAMATH_CALUDE_parabola_point_value_l2472_247210

/-- Given points A(a,m), B(b,m), P(a+b,n) on the parabola y=x^2-2x-2, prove that n = -2 -/
theorem parabola_point_value (a b m n : ℝ) : 
  (m = a^2 - 2*a - 2) →  -- A is on the parabola
  (m = b^2 - 2*b - 2) →  -- B is on the parabola
  (n = (a+b)^2 - 2*(a+b) - 2) →  -- P is on the parabola
  (n = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_value_l2472_247210


namespace NUMINAMATH_CALUDE_function_extrema_l2472_247231

/-- The function f(x) = (1/3)x³ - ax² + x has exactly one maximum value and one minimum value
if and only if a < -1 or a > 1. -/
theorem function_extrema (a : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (∀ x : ℝ, (1/3 : ℝ) * x^3 - a * x^2 + x ≤ (1/3 : ℝ) * x₁^3 - a * x₁^2 + x₁) ∧
    (∀ x : ℝ, (1/3 : ℝ) * x^3 - a * x^2 + x ≥ (1/3 : ℝ) * x₂^3 - a * x₂^2 + x₂)) ↔
  (a < -1 ∨ a > 1) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_l2472_247231


namespace NUMINAMATH_CALUDE_velocity_of_point_C_l2472_247268

/-- Given the equation relating distances and time, prove the velocity of point C. -/
theorem velocity_of_point_C 
  (a R L T : ℝ) 
  (x : ℝ) 
  (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R :=
sorry

end NUMINAMATH_CALUDE_velocity_of_point_C_l2472_247268


namespace NUMINAMATH_CALUDE_sum_of_negatives_l2472_247228

theorem sum_of_negatives : (-3) + (-9) = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_negatives_l2472_247228


namespace NUMINAMATH_CALUDE_permutations_of_four_l2472_247261

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_l2472_247261


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_system_l2472_247203

theorem quadratic_and_inequality_system :
  (∃ x1 x2 : ℝ, x1 = 1 ∧ x2 = 5 ∧ 
    (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x = x1 ∨ x = x2))) ∧
  (∀ x : ℝ, x + 3 > 0 ∧ 2*(x - 1) < 4 ↔ -3 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_system_l2472_247203


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2472_247227

theorem sum_of_fractions_equals_one 
  (a b c x y z : ℝ) 
  (h1 : 13 * x + b * y + c * z = 0)
  (h2 : a * x + 23 * y + c * z = 0)
  (h3 : a * x + b * y + 42 * z = 0)
  (h4 : a ≠ 13)
  (h5 : x ≠ 0) :
  a / (a - 13) + b / (b - 23) + c / (c - 42) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2472_247227


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l2472_247217

theorem initial_children_on_bus (children_off : ℕ) (children_on : ℕ) (final_children : ℕ) :
  children_off = 10 →
  children_on = 5 →
  final_children = 16 →
  final_children + (children_off - children_on) = 21 :=
by sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l2472_247217


namespace NUMINAMATH_CALUDE_shaded_circle_fraction_l2472_247298

/-- Given a circle divided into equal regions, this theorem proves that
    if there are 4 regions and 1 is shaded, then the shaded fraction is 1/4. -/
theorem shaded_circle_fraction (total_regions shaded_regions : ℕ) :
  total_regions = 4 →
  shaded_regions = 1 →
  (shaded_regions : ℚ) / total_regions = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_circle_fraction_l2472_247298


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2472_247209

def solution_set : Set ℝ := {x | x ≤ -5/2}

def inequality (x : ℝ) : Prop := |x - 2| + |x + 3| ≥ 4

theorem solution_set_equivalence :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2472_247209


namespace NUMINAMATH_CALUDE_simplify_expression_l2472_247223

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2472_247223


namespace NUMINAMATH_CALUDE_problem_statements_l2472_247286

theorem problem_statements :
  (∃ x : ℝ, x^3 < 1) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∀ x : ℕ, x^3 > x^2) ∧
  (∀ x : ℝ, x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2472_247286


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2472_247297

def b (n : ℕ) : ℕ := 2^n * n.factorial + n

theorem max_gcd_consecutive_terms :
  ∀ n : ℕ, ∃ m : ℕ, m ≤ n → Nat.gcd (b m) (b (m + 1)) = 1 ∧
  ∀ k : ℕ, k ≤ n → Nat.gcd (b k) (b (k + 1)) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2472_247297


namespace NUMINAMATH_CALUDE_equation_solutions_l2472_247207

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  (13*x - x^2) / (x + 1) * (x + (13 - x) / (x + 1)) = 42

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2472_247207


namespace NUMINAMATH_CALUDE_color_fractions_l2472_247259

-- Define the color type
inductive Color
  | Red
  | Blue

-- Define the coloring function
def color : ℚ → Color := sorry

-- Define the coloring rules
axiom color_one : color 1 = Color.Red
axiom color_diff_one (x : ℚ) : color (x + 1) ≠ color x
axiom color_reciprocal (x : ℚ) (h : x ≠ 1) : color (1 / x) ≠ color x

-- State the theorem
theorem color_fractions :
  color (2013 / 2014) = Color.Red ∧ color (2 / 7) = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_color_fractions_l2472_247259


namespace NUMINAMATH_CALUDE_bundle_limit_points_l2472_247218

-- Define the types of bundles
inductive BundleType
  | Hyperbolic
  | Parabolic
  | Elliptic

-- Define a function that returns the number of limit points for a given bundle type
def limitPoints (b : BundleType) : Nat :=
  match b with
  | BundleType.Hyperbolic => 2
  | BundleType.Parabolic => 1
  | BundleType.Elliptic => 0

-- Theorem statement
theorem bundle_limit_points (b : BundleType) :
  (b = BundleType.Hyperbolic → limitPoints b = 2) ∧
  (b = BundleType.Parabolic → limitPoints b = 1) ∧
  (b = BundleType.Elliptic → limitPoints b = 0) :=
by sorry

end NUMINAMATH_CALUDE_bundle_limit_points_l2472_247218


namespace NUMINAMATH_CALUDE_hoseok_result_l2472_247270

theorem hoseok_result (X : ℤ) (h : X - 46 = 15) : X - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_result_l2472_247270


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l2472_247236

theorem imaginary_part_of_one_over_one_minus_i :
  Complex.im (1 / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l2472_247236


namespace NUMINAMATH_CALUDE_first_tank_height_l2472_247239

/-- Represents a cylindrical water tank being pumped at a constant rate -/
structure WaterTank where
  height : ℝ
  emptyTime : ℝ

/-- Proves that given two water tanks with specified properties, the height of the first tank is 12.5 meters -/
theorem first_tank_height (tank1 tank2 : WaterTank) 
  (h1 : tank2.height = 10)
  (h2 : tank2.emptyTime = 8)
  (h3 : tank1.emptyTime = 5)
  (h4 : (3 / 5) * tank1.height = (3 / 4) * tank2.height) : 
  tank1.height = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_first_tank_height_l2472_247239


namespace NUMINAMATH_CALUDE_equation_solution_l2472_247237

theorem equation_solution : 
  let S : Set ℝ := {x | (x^4 + 4*x^3*Real.sqrt 3 + 12*x^2 + 8*x*Real.sqrt 3 + 4) + (x^2 + 2*x*Real.sqrt 3 + 3) = 0}
  S = {-Real.sqrt 3, -Real.sqrt 3 + 1, -Real.sqrt 3 - 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2472_247237


namespace NUMINAMATH_CALUDE_marble_bag_problem_l2472_247247

theorem marble_bag_problem (red blue : ℕ) (p : ℚ) (total : ℕ) : 
  red = 12 →
  blue = 8 →
  p = 81 / 256 →
  (((total - red : ℚ) / total) ^ 4 = p) →
  total = 48 :=
by sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l2472_247247


namespace NUMINAMATH_CALUDE_fraction_problem_l2472_247274

theorem fraction_problem : ∃ x : ℚ, (65 / 100 * 40 : ℚ) = x * 25 + 6 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2472_247274


namespace NUMINAMATH_CALUDE_circle_properties_1_circle_properties_2_l2472_247234

/-- Definition of a circle in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  eq : ∀ x y : ℝ, x^2 + y^2 + a*x + b*y + c = 0

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem for the first circle -/
theorem circle_properties_1 :
  let C : Circle := {
    a := 2
    b := -4
    c := -3
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-1, 2) ∧ props.radius = 2 * Real.sqrt 2 := by sorry

/-- Theorem for the second circle -/
theorem circle_properties_2 (m : ℝ) :
  let C : Circle := {
    a := 2*m
    b := 0
    c := 0
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-m, 0) ∧ props.radius = |m| := by sorry

end NUMINAMATH_CALUDE_circle_properties_1_circle_properties_2_l2472_247234


namespace NUMINAMATH_CALUDE_original_shirt_price_l2472_247245

/-- Proves that if a shirt's current price is $6 and this price is 25% of the original price, 
    then the original price was $24. -/
theorem original_shirt_price (current_price : ℝ) (original_price : ℝ) : 
  current_price = 6 → 
  current_price = 0.25 * original_price →
  original_price = 24 := by
sorry

end NUMINAMATH_CALUDE_original_shirt_price_l2472_247245


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2472_247208

theorem rational_equation_solution :
  ∃ y : ℝ, y ≠ 3 ∧ y ≠ (1/3) ∧
  (y^2 - 7*y + 12)/(y - 3) + (3*y^2 + 5*y - 8)/(3*y - 1) = -8 ∧
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2472_247208


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l2472_247295

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1
def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def total_tickets : ℕ := 21

theorem ferris_wheel_cost :
  total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost) = ferris_wheel_rides := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l2472_247295


namespace NUMINAMATH_CALUDE_cycling_route_length_l2472_247222

theorem cycling_route_length (upper_segments : List ℝ) (left_segments : List ℝ) :
  upper_segments = [4, 7, 2] →
  left_segments = [6, 7] →
  2 * (upper_segments.sum + left_segments.sum) = 52 := by
  sorry

end NUMINAMATH_CALUDE_cycling_route_length_l2472_247222


namespace NUMINAMATH_CALUDE_equation_proof_l2472_247285

theorem equation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2472_247285


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2472_247243

/-- The radius of the inscribed circle in a right triangle with side lengths 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_right_triangle :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 5
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := (s * (s - a) * (s - b) * (s - c))^(1/2)
  a^2 + b^2 = c^2 → -- Pythagorean theorem to ensure it's a right triangle
  area / s = 1 := by
sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2472_247243


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2472_247257

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2472_247257


namespace NUMINAMATH_CALUDE_cheetah_catches_deer_l2472_247230

/-- The time it takes for a cheetah to catch up with a deer given their speeds and initial time difference -/
theorem cheetah_catches_deer (deer_speed cheetah_speed : ℝ) (initial_time_diff : ℝ) : 
  deer_speed = 50 →
  cheetah_speed = 60 →
  initial_time_diff = 2 / 60 →
  (initial_time_diff + (cheetah_speed - deer_speed)⁻¹ * deer_speed * initial_time_diff) * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cheetah_catches_deer_l2472_247230


namespace NUMINAMATH_CALUDE_ryan_coin_value_l2472_247235

/-- Represents the types of coins Ryan has --/
inductive Coin
| Penny
| Nickel

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- Ryan's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  total_coins : pennies + nickels = 17
  equal_count : pennies = nickels

theorem ryan_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + c.nickels * coinValue Coin.Nickel = 49 := by
  sorry

#check ryan_coin_value

end NUMINAMATH_CALUDE_ryan_coin_value_l2472_247235


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_3148_l2472_247262

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 3) - 9 * (x^4 - 4*x^2 + 4)

/-- The sum of squares of coefficients of the fully simplified expression -/
def sum_of_squared_coefficients : ℝ := 3148

/-- Theorem stating that the sum of squares of coefficients of the fully simplified expression is 3148 -/
theorem sum_of_squared_coefficients_is_3148 :
  ∃ (a b c d : ℝ), ∀ (x : ℝ), 
    expression x = a*x^4 + b*x^3 + c*x^2 + d ∧
    a^2 + b^2 + c^2 + d^2 = sum_of_squared_coefficients :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_3148_l2472_247262


namespace NUMINAMATH_CALUDE_tangent_difference_l2472_247212

noncomputable section

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.log (abs x)

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the property of being a tangent line to the curve
def is_tangent (a k : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ curve a x = tangent_line k x ∧
    (∀ y : ℝ, y ≠ x → curve a y ≠ tangent_line k y)

-- Theorem statement
theorem tangent_difference (a k₁ k₂ : ℝ) :
  is_tangent a k₁ → is_tangent a k₂ → k₁ > k₂ → k₁ - k₂ = 4 / Real.exp 1 := by
  sorry

end

end NUMINAMATH_CALUDE_tangent_difference_l2472_247212


namespace NUMINAMATH_CALUDE_trick_or_treat_total_l2472_247280

/-- Calculates the total number of treats received by children while trick-or-treating. -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_child_per_house

/-- Proves that given the specific conditions, the total number of treats is 180. -/
theorem trick_or_treat_total (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) 
    (h1 : num_children = 3)
    (h2 : hours_out = 4)
    (h3 : houses_per_hour = 5)
    (h4 : treats_per_child_per_house = 3) :
  total_treats num_children hours_out houses_per_hour treats_per_child_per_house = 180 := by
  sorry

end NUMINAMATH_CALUDE_trick_or_treat_total_l2472_247280


namespace NUMINAMATH_CALUDE_condition_C_necessary_for_A_l2472_247281

-- Define the conditions as propositions
variable (A B C D : Prop)

-- Define the relationship between the conditions
variable (h : (C → D) → (A → B))

-- Theorem to prove
theorem condition_C_necessary_for_A (h : (C → D) → (A → B)) : A → C :=
  sorry

end NUMINAMATH_CALUDE_condition_C_necessary_for_A_l2472_247281


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l2472_247225

theorem polynomial_non_negative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l2472_247225


namespace NUMINAMATH_CALUDE_election_result_l2472_247271

/-- Represents an election with three candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.votes_A = (35 * e.total_votes) / 100 ∧
  e.votes_C = (25 * e.total_votes) / 100 ∧
  e.votes_B = e.votes_A + 2460 ∧
  e.total_votes = e.votes_A + e.votes_B + e.votes_C

theorem election_result (e : Election) (h : valid_election e) :
  e.votes_B = (40 * e.total_votes) / 100 ∧ e.total_votes = 49200 := by
  sorry


end NUMINAMATH_CALUDE_election_result_l2472_247271


namespace NUMINAMATH_CALUDE_melanie_dimes_given_l2472_247241

/-- The number of dimes Melanie gave to her dad -/
def dimes_given_to_dad : ℕ := 7

/-- The initial number of dimes Melanie had -/
def initial_dimes : ℕ := 8

/-- The number of dimes Melanie received from her mother -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 5

theorem melanie_dimes_given :
  initial_dimes - dimes_given_to_dad + dimes_from_mother = current_dimes :=
by sorry

end NUMINAMATH_CALUDE_melanie_dimes_given_l2472_247241


namespace NUMINAMATH_CALUDE_select_five_from_eight_with_book_a_l2472_247220

/-- The number of ways to select 5 books from 8 books, always including "Book A" -/
def select_books (total_books : ℕ) (books_to_select : ℕ) : ℕ :=
  Nat.choose (total_books - 1) (books_to_select - 1)

/-- Theorem: Selecting 5 books from 8 books, always including "Book A", can be done in 35 ways -/
theorem select_five_from_eight_with_book_a : select_books 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_with_book_a_l2472_247220


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l2472_247211

theorem reciprocal_of_sum : (1 / (1/2 + 1/3) : ℚ) = 6/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l2472_247211


namespace NUMINAMATH_CALUDE_count_divisible_integers_l2472_247204

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (7 * n) % (n * (n + 1) / 2) = 0) ∧ 
    Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l2472_247204


namespace NUMINAMATH_CALUDE_integer_bounds_l2472_247246

theorem integer_bounds (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : x < 18)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∀ y : ℤ, x > y → y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_integer_bounds_l2472_247246


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2472_247242

/-- Proves that a rectangle with specific properties has dimensions 7 and 4 -/
theorem rectangle_dimensions :
  ∀ l w : ℝ,
  l = w + 3 →
  l * w = 2 * (2 * l + 2 * w) →
  l = 7 ∧ w = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2472_247242


namespace NUMINAMATH_CALUDE_sum_first_four_eq_40_l2472_247213

/-- A geometric sequence with a_2 = 6 and a_3 = -18 -/
def geometric_sequence (n : ℕ) : ℝ :=
  let q := -3  -- common ratio
  let a1 := -2 -- first term
  a1 * q^(n-1)

/-- The sum of the first four terms of the geometric sequence -/
def sum_first_four : ℝ :=
  (geometric_sequence 1) + (geometric_sequence 2) + (geometric_sequence 3) + (geometric_sequence 4)

/-- Theorem stating that the sum of the first four terms equals 40 -/
theorem sum_first_four_eq_40 : sum_first_four = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_four_eq_40_l2472_247213


namespace NUMINAMATH_CALUDE_parlor_game_solution_l2472_247292

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : a > 0

/-- Calculates the sum of permutations for a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.c + n.b +
  100 * n.a + 10 * n.b + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.b + 10 * n.a + n.c +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

/-- The main theorem -/
theorem parlor_game_solution :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = 4326 ∧ n.a = 3 ∧ n.b = 9 ∧ n.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parlor_game_solution_l2472_247292


namespace NUMINAMATH_CALUDE_circle_diameter_l2472_247214

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_l2472_247214


namespace NUMINAMATH_CALUDE_carnation_fraction_l2472_247296

def flower_bouquet (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) : Prop :=
  pink_roses + red_roses + pink_carnations + red_carnations = total ∧
  pink_roses + pink_carnations = (7/10) * total ∧
  pink_roses = (1/2) * (pink_roses + pink_carnations) ∧
  red_carnations = (5/6) * (red_roses + red_carnations)

theorem carnation_fraction (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) 
  (h : flower_bouquet total pink_roses red_roses pink_carnations red_carnations) :
  (pink_carnations + red_carnations) / total = 3/5 :=
sorry

end NUMINAMATH_CALUDE_carnation_fraction_l2472_247296


namespace NUMINAMATH_CALUDE_inequality_proof_l2472_247277

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2472_247277
