import Mathlib

namespace NUMINAMATH_CALUDE_smallest_integer_power_l3561_356130

theorem smallest_integer_power (x : ℕ) (h : x = 9 * 3) :
  (∀ c : ℕ, x^c > 3^24 → c ≥ 9) ∧ x^9 > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l3561_356130


namespace NUMINAMATH_CALUDE_triangle_formation_l3561_356137

/-- Two lines in the Cartesian coordinate system -/
structure CartesianLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  k : ℝ

/-- Condition for two lines to form a triangle with the x-axis -/
def formsTriangle (lines : CartesianLines) : Prop :=
  lines.k ≠ 0 ∧ lines.k ≠ -1/2

/-- Theorem: The given lines form a triangle with the x-axis if and only if k ≠ -1/2 -/
theorem triangle_formation (lines : CartesianLines) 
  (h1 : lines.line1 = fun x ↦ -0.5 * x - 2)
  (h2 : lines.line2 = fun x ↦ lines.k * x + 3) :
  formsTriangle lines ↔ lines.k ≠ -1/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3561_356137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3561_356153

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3561_356153


namespace NUMINAMATH_CALUDE_circle_equation_l3561_356132

/-- The equation of a circle with center (±2, 1) and radius 2, given specific conditions -/
theorem circle_equation (x y : ℝ) : 
  (∃ t : ℝ, t = 2 ∨ t = -2) →
  (1 : ℝ) = (1/4) * t^2 →
  (abs t = abs ((1/4) * t^2 + 1)) →
  (x^2 + y^2 + 4*x - 2*y - 1 = 0) ∨ (x^2 + y^2 - 4*x - 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3561_356132


namespace NUMINAMATH_CALUDE_john_writing_speed_l3561_356165

/-- The number of books John writes -/
def num_books : ℕ := 3

/-- The number of pages in each book -/
def pages_per_book : ℕ := 400

/-- The number of days it takes John to write the books -/
def total_days : ℕ := 60

/-- The number of pages John writes per day -/
def pages_per_day : ℕ := (num_books * pages_per_book) / total_days

theorem john_writing_speed : pages_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_writing_speed_l3561_356165


namespace NUMINAMATH_CALUDE_value_of_a_l3561_356133

theorem value_of_a (x a : ℝ) (hx : x ≠ 1) : 
  (8 * a) / (1 - x^32) = 2 / (1 - x) + 2 / (1 + x) + 4 / (1 + x^2) + 
                         8 / (1 + x^4) + 16 / (1 + x^8) + 32 / (1 + x^16) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3561_356133


namespace NUMINAMATH_CALUDE_triangle_theorem_l3561_356183

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A * Real.sin t.B + t.b * (Real.cos t.A)^2 = 4/3 * t.a

/-- The additional condition for part 2 -/
def additional_condition (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + 1/4 * t.b^2

theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : additional_condition t) : 
  t.b / t.a = 4/3 ∧ t.C = π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l3561_356183


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3561_356150

theorem sqrt_inequality (a : ℝ) (h : a > 5) :
  Real.sqrt (a - 5) - Real.sqrt (a - 3) < Real.sqrt (a - 2) - Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3561_356150


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3561_356112

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3561_356112


namespace NUMINAMATH_CALUDE_max_length_theorem_l3561_356136

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a line passing through (0,1)
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection points
def IntersectionPoints (k : ℝ) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the lengths |A₁B₁| and |A₂B₂|
def Length_A1B1 (k : ℝ) : ℝ := sorry
def Length_A2B2 (k : ℝ) : ℝ := sorry

-- The main theorem
theorem max_length_theorem :
  ∃ k : ℝ, Length_A1B1 k = max_length_A1B1 ∧ Length_A2B2 k = 2 * Real.sqrt 30 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_length_theorem_l3561_356136


namespace NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l3561_356126

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to roll a sum of 5 with two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l3561_356126


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3561_356162

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3561_356162


namespace NUMINAMATH_CALUDE_root_product_simplification_l3561_356146

theorem root_product_simplification (a : ℝ) (ha : 0 < a) :
  (a ^ (1 / Real.sqrt a)) * (a ^ (1 / 3)) = a ^ (5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_root_product_simplification_l3561_356146


namespace NUMINAMATH_CALUDE_tangent_length_is_three_l3561_356110

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- The point P -/
def P : ℝ × ℝ := (-1, 4)

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangentLength (p : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The length of the tangent line from P to the circle is 3 -/
theorem tangent_length_is_three :
  tangentLength P = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_length_is_three_l3561_356110


namespace NUMINAMATH_CALUDE_gecko_cost_is_fifteen_l3561_356179

/-- Represents the cost of feeding Harry's pets -/
structure PetFeedingCost where
  geckos : ℕ
  iguanas : ℕ
  snakes : ℕ
  snake_cost : ℕ
  iguana_cost : ℕ
  total_annual_cost : ℕ

/-- Calculates the monthly cost per gecko -/
def gecko_monthly_cost (p : PetFeedingCost) : ℚ :=
  (p.total_annual_cost / 12 - (p.snakes * p.snake_cost + p.iguanas * p.iguana_cost)) / p.geckos

/-- Theorem stating that the monthly cost per gecko is $15 -/
theorem gecko_cost_is_fifteen (p : PetFeedingCost) 
    (h1 : p.geckos = 3)
    (h2 : p.iguanas = 2)
    (h3 : p.snakes = 4)
    (h4 : p.snake_cost = 10)
    (h5 : p.iguana_cost = 5)
    (h6 : p.total_annual_cost = 1140) :
    gecko_monthly_cost p = 15 := by
  sorry

end NUMINAMATH_CALUDE_gecko_cost_is_fifteen_l3561_356179


namespace NUMINAMATH_CALUDE_paint_needed_for_columns_l3561_356103

-- Define constants
def num_columns : ℕ := 20
def column_height : ℝ := 20
def column_diameter : ℝ := 10
def paint_coverage : ℝ := 350

-- Theorem statement
theorem paint_needed_for_columns :
  ∃ (gallons : ℕ),
    gallons * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) ∧
    ∀ (g : ℕ), g * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) → g ≥ gallons :=
by sorry

end NUMINAMATH_CALUDE_paint_needed_for_columns_l3561_356103


namespace NUMINAMATH_CALUDE_inverse_of_complex_l3561_356161

theorem inverse_of_complex (z : ℂ) (h : z = (1 : ℝ) / 2 + (Real.sqrt 3 / 2) * I) : 
  z⁻¹ = (1 : ℝ) / 2 - (Real.sqrt 3 / 2) * I := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_l3561_356161


namespace NUMINAMATH_CALUDE_age_difference_proof_l3561_356119

theorem age_difference_proof (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b + 3 = 3 * (10 * b + a + 3)) → 
  (10 * a + b) - (10 * b + a) = 36 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3561_356119


namespace NUMINAMATH_CALUDE_nine_b_equals_eighteen_l3561_356185

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end NUMINAMATH_CALUDE_nine_b_equals_eighteen_l3561_356185


namespace NUMINAMATH_CALUDE_grapes_purchased_l3561_356172

/-- Represents the price of grapes per kilogram -/
def grape_price : ℕ := 70

/-- Represents the price of mangoes per kilogram -/
def mango_price : ℕ := 55

/-- Represents the amount of mangoes purchased in kilograms -/
def mango_amount : ℕ := 11

/-- Represents the total amount paid -/
def total_paid : ℕ := 1165

/-- Theorem stating that the amount of grapes purchased is 8 kg -/
theorem grapes_purchased : ∃ (g : ℕ), g * grape_price + mango_amount * mango_price = total_paid ∧ g = 8 := by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l3561_356172


namespace NUMINAMATH_CALUDE_inequality_implies_sum_l3561_356105

/-- Given that (x-a)(x-b)/(x-c) ≤ 0 if and only if x < -6 or |x-30| ≤ 2, and a < b,
    prove that a + 2b + 3c = 74 -/
theorem inequality_implies_sum (a b c : ℝ) :
  (∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  a < b →
  a + 2*b + 3*c = 74 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_l3561_356105


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3561_356109

/-- Given a quadratic equation x^2 + kx - 2 = 0 where x = 1 is one root,
    prove that x = -2 is the other root. -/
theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 2 = 0 → -2^2 + k * (-2) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3561_356109


namespace NUMINAMATH_CALUDE_non_intersecting_to_concentric_l3561_356155

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- An inversion transformation --/
structure Inversion where
  center : ℝ × ℝ
  power : ℝ
  power_pos : power > 0

/-- Two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Two circles are concentric --/
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

/-- The image of a circle under inversion --/
def inversion_image (i : Inversion) (c : Circle) : Circle :=
  sorry

/-- The main theorem --/
theorem non_intersecting_to_concentric :
  ∀ (S1 S2 : Circle), non_intersecting S1 S2 →
  ∃ (i : Inversion), concentric (inversion_image i S1) (inversion_image i S2) :=
sorry

end NUMINAMATH_CALUDE_non_intersecting_to_concentric_l3561_356155


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_f_upper_bound_implies_a_range_l3561_356168

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 1 / x

theorem tangent_slope_implies_a (a : ℝ) :
  f_deriv a 1 = -1 → a = -1 := by sorry

theorem f_upper_bound_implies_a_range (a : ℝ) :
  a < 0 →
  (∀ x > 0, f a x ≤ -1/2) →
  a ≤ -1/2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_slope_implies_a_f_upper_bound_implies_a_range_l3561_356168


namespace NUMINAMATH_CALUDE_two_rectangle_formations_l3561_356106

def square_sides : List ℕ := [3, 5, 9, 11, 14, 19, 20, 24, 31, 33, 36, 39, 42]

def rectangle_width : ℕ := 75
def rectangle_height : ℕ := 112

def forms_rectangle (subset : List ℕ) : Prop :=
  (subset.map (λ x => x^2)).sum = rectangle_width * rectangle_height

theorem two_rectangle_formations :
  ∃ (subset1 subset2 : List ℕ),
    subset1 ⊆ square_sides ∧
    subset2 ⊆ square_sides ∧
    subset1 ∩ subset2 = ∅ ∧
    forms_rectangle subset1 ∧
    forms_rectangle subset2 :=
sorry

end NUMINAMATH_CALUDE_two_rectangle_formations_l3561_356106


namespace NUMINAMATH_CALUDE_crow_eating_time_l3561_356190

/-- Given a constant eating rate where 1/5 of the nuts are eaten in 8 hours,
    prove that it takes 10 hours to eat 1/4 of the nuts. -/
theorem crow_eating_time (eating_rate : ℝ → ℝ) (h1 : eating_rate (8 : ℝ) = 1/5) 
    (h2 : ∀ t1 t2 : ℝ, eating_rate (t1 + t2) = eating_rate t1 + eating_rate t2) : 
    ∃ t : ℝ, eating_rate t = 1/4 ∧ t = 10 := by
  sorry


end NUMINAMATH_CALUDE_crow_eating_time_l3561_356190


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3561_356174

theorem equation_one_solutions (x : ℝ) :
  x - 2 = 4 * (x - 2)^2 ↔ x = 2 ∨ x = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3561_356174


namespace NUMINAMATH_CALUDE_isosceles_triangle_obtuse_iff_quadratic_roots_l3561_356102

theorem isosceles_triangle_obtuse_iff_quadratic_roots 
  (A B C : Real) 
  (triangle_sum : A + B + C = π) 
  (isosceles : A = C) : 
  (B > π / 2) ↔ 
  ∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ A * x₁^2 + B * x₁ + C = 0 ∧ A * x₂^2 + B * x₂ + C = 0 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_obtuse_iff_quadratic_roots_l3561_356102


namespace NUMINAMATH_CALUDE_problem_solution_l3561_356194

theorem problem_solution (a b : ℝ) (m n : ℕ) 
  (h : (2 * a^m * b^(m+n))^3 = 8 * a^9 * b^15) : 
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3561_356194


namespace NUMINAMATH_CALUDE_sum_base4_numbers_l3561_356182

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem sum_base4_numbers : 
  let a := [2, 0, 2]  -- 202₄
  let b := [0, 3, 3]  -- 330₄
  let c := [0, 0, 0, 1]  -- 1000₄
  let sum_base10 := base4ToBase10 a + base4ToBase10 b + base4ToBase10 c
  base10ToBase4 sum_base10 = [2, 3, 1, 2] ∧ sum_base10 = 158 := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_numbers_l3561_356182


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3561_356138

/-- Given two lines l₁ and l₂, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2*x + (a-1)*y + a = 0 ↔ a*x + y + 2 = 0) → -- l₁ and l₂ are parallel
  (2*2 ≠ a^2) →                                         -- Additional condition
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3561_356138


namespace NUMINAMATH_CALUDE_water_bottle_capacity_l3561_356152

/-- The capacity of a water bottle in milliliters -/
def bottle_capacity : ℕ := 12800

/-- The volume of the smaller cup in milliliters -/
def small_cup : ℕ := 250

/-- The volume of the larger cup in milliliters -/
def large_cup : ℕ := 600

/-- The number of times water is scooped with the smaller cup -/
def small_cup_scoops : ℕ := 20

/-- The number of times water is scooped with the larger cup -/
def large_cup_scoops : ℕ := 13

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1 / 1000

theorem water_bottle_capacity :
  (bottle_capacity : ℚ) * ml_to_l = 12.8 ∧
  bottle_capacity = small_cup * small_cup_scoops + large_cup * large_cup_scoops :=
sorry

end NUMINAMATH_CALUDE_water_bottle_capacity_l3561_356152


namespace NUMINAMATH_CALUDE_function_and_inequality_solution_l3561_356178

noncomputable section

variables (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem function_and_inequality_solution 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : f 0 = 2020)
  (h3 : ∀ x, f' x = f x - 2) :
  (∀ x, f x = 2 + 2018 * Real.exp x) ∧ 
  {x : ℝ | f x + 4034 > 2 * f' x} = {x : ℝ | x < Real.log 2} := by
  sorry

end

end NUMINAMATH_CALUDE_function_and_inequality_solution_l3561_356178


namespace NUMINAMATH_CALUDE_savings_account_calculation_final_amount_is_690_l3561_356164

/-- Calculates the final amount in a savings account after two years with given conditions --/
theorem savings_account_calculation (initial_deposit : ℝ) (first_year_rate : ℝ) 
  (withdrawal_percentage : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_balance := initial_deposit * (1 + first_year_rate)
  let remaining_after_withdrawal := first_year_balance * (1 - withdrawal_percentage)
  let final_balance := remaining_after_withdrawal * (1 + second_year_rate)
  final_balance

/-- Proves that the final amount in the account is $690 given the specified conditions --/
theorem final_amount_is_690 : 
  savings_account_calculation 1000 0.20 0.50 0.15 = 690 := by
sorry

end NUMINAMATH_CALUDE_savings_account_calculation_final_amount_is_690_l3561_356164


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3561_356171

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3561_356171


namespace NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l3561_356145

def months_in_year : ℕ := 12
def replacement_interval : ℕ := 7
def target_replacement : ℕ := 18

def month_of_replacement (n : ℕ) : ℕ :=
  ((n - 1) * replacement_interval) % months_in_year + 1

theorem eighteenth_replacement_in_december :
  month_of_replacement target_replacement = 12 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l3561_356145


namespace NUMINAMATH_CALUDE_triangle_side_length_l3561_356135

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  b = 1 →
  c = Real.sqrt 3 →
  C = 2 * Real.pi / 3 →
  a^2 + b^2 - 2*a*b*(Real.cos C) = c^2 →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3561_356135


namespace NUMINAMATH_CALUDE_range_of_x_l3561_356148

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  (x ∈ Set.Ioo 1 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3561_356148


namespace NUMINAMATH_CALUDE_circuit_reliability_l3561_356144

-- Define the probabilities of element failures
def p1 : ℝ := 0.2
def p2 : ℝ := 0.3
def p3 : ℝ := 0.4

-- Define the probability of the circuit not breaking
def circuit_not_break : ℝ := (1 - p1) * (1 - p2) * (1 - p3)

-- Theorem statement
theorem circuit_reliability : circuit_not_break = 0.336 := by
  sorry

end NUMINAMATH_CALUDE_circuit_reliability_l3561_356144


namespace NUMINAMATH_CALUDE_reciprocal_of_product_l3561_356170

theorem reciprocal_of_product : (((1 : ℚ) / 3) * (3 / 4))⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_product_l3561_356170


namespace NUMINAMATH_CALUDE_max_factors_bound_l3561_356180

/-- The number of positive factors of b^n, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
def max_factors (b n : ℕ+) : ℕ :=
  if b ≤ 20 ∧ n ≤ 20 then
    -- Placeholder for the actual calculation of factors
    0
  else
    0

/-- The maximum number of positive factors of b^n is 861, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
theorem max_factors_bound :
  ∃ (b n : ℕ+), b ≤ 20 ∧ n ≤ 20 ∧ max_factors b n = 861 ∧
  ∀ (b' n' : ℕ+), b' ≤ 20 → n' ≤ 20 → max_factors b' n' ≤ 861 :=
sorry

end NUMINAMATH_CALUDE_max_factors_bound_l3561_356180


namespace NUMINAMATH_CALUDE_topsoil_cost_l3561_356195

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℚ := 7

/-- The volume of topsoil in cubic yards -/
def volume_cubic_yards : ℚ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℚ := 27

/-- The total cost of topsoil in dollars -/
def total_cost : ℚ := 1512

theorem topsoil_cost :
  cost_per_cubic_foot * volume_cubic_yards * cubic_yards_to_cubic_feet = total_cost := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3561_356195


namespace NUMINAMATH_CALUDE_square_root_of_10_factorial_div_210_l3561_356154

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem square_root_of_10_factorial_div_210 :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (factorial 10 : ℝ) / 210 ∧ x = 24 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_10_factorial_div_210_l3561_356154


namespace NUMINAMATH_CALUDE_streak_plate_method_claim_incorrect_l3561_356147

/-- Represents the capability of the streak plate method -/
structure StreakPlateMethod where
  can_separate : Bool
  can_count : Bool

/-- The actual capabilities of the streak plate method -/
def actual_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := false }

/-- The claimed capabilities of the streak plate method in the statement -/
def claimed_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := true }

/-- Theorem stating that the claim about the streak plate method is incorrect -/
theorem streak_plate_method_claim_incorrect :
  actual_streak_plate_method ≠ claimed_streak_plate_method :=
by sorry

end NUMINAMATH_CALUDE_streak_plate_method_claim_incorrect_l3561_356147


namespace NUMINAMATH_CALUDE_problem_solution_l3561_356159

def row1 (n : ℕ) : ℤ := (-2) ^ n

def row2 (n : ℕ) : ℤ := row1 n + 2

def row3 (n : ℕ) : ℤ := (-2) ^ (n - 1)

theorem problem_solution :
  (row1 4 = 16) ∧
  (∀ n : ℕ, row2 n = row1 n + 2) ∧
  (∃ k : ℕ, row3 k + row3 (k + 1) + row3 (k + 2) = -192 ∧
            row3 k = -64 ∧ row3 (k + 1) = 128 ∧ row3 (k + 2) = -256) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3561_356159


namespace NUMINAMATH_CALUDE_first_pay_cut_percentage_l3561_356111

theorem first_pay_cut_percentage 
  (overall_decrease : Real) 
  (second_cut : Real) 
  (third_cut : Real) 
  (h1 : overall_decrease = 27.325)
  (h2 : second_cut = 10)
  (h3 : third_cut = 15) : 
  ∃ (first_cut : Real), 
    first_cut = 5 ∧ 
    (1 - overall_decrease / 100) = 
    (1 - first_cut / 100) * (1 - second_cut / 100) * (1 - third_cut / 100) := by
  sorry


end NUMINAMATH_CALUDE_first_pay_cut_percentage_l3561_356111


namespace NUMINAMATH_CALUDE_two_digit_multiplication_error_l3561_356163

theorem two_digit_multiplication_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  a * b = 936 →
  ((a + 40) * b = 2496 ∨ a * (b + 40) = 2496) →
  a + b = 63 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_error_l3561_356163


namespace NUMINAMATH_CALUDE_equation_solutions_l3561_356114

theorem equation_solutions : 
  {x : ℝ | x^6 + (2-x)^6 = 272} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3561_356114


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l3561_356156

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1 ∨ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l3561_356156


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l3561_356177

/-- Represents a quadrilateral with extended sides --/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  WZ : ℝ
  WX : ℝ
  XY : ℝ
  YZ : ℝ
  -- Extended sides
  ZW' : ℝ
  XX' : ℝ
  YY' : ℝ
  Z'W : ℝ
  -- Area of original quadrilateral
  area : ℝ

/-- Theorem stating the area of the extended quadrilateral --/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) 
  (h1 : q.WZ = 10 ∧ q.ZW' = 10)
  (h2 : q.WX = 6 ∧ q.XX' = 6)
  (h3 : q.XY = 7 ∧ q.YY' = 7)
  (h4 : q.YZ = 12 ∧ q.Z'W = 12)
  (h5 : q.area = 15) :
  ∃ (area_extended : ℝ), area_extended = 45 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l3561_356177


namespace NUMINAMATH_CALUDE_eggs_needed_is_84_l3561_356134

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
| ThreeEgg
| FourEgg

/-- Represents an hour of operation with customer orders -/
structure HourlyOrder where
  customerCount : Nat
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourlyOrder) : Nat :=
  orders.foldl (fun acc order => 
    acc + order.customerCount * match order.omeletteType with
      | OmeletteType.ThreeEgg => 3
      | OmeletteType.FourEgg => 4
  ) 0

/-- The main theorem stating that given the specific orders, 84 eggs are needed -/
theorem eggs_needed_is_84 : 
  let orders : List HourlyOrder := [
    { customerCount := 5, omeletteType := OmeletteType.ThreeEgg },
    { customerCount := 7, omeletteType := OmeletteType.FourEgg },
    { customerCount := 3, omeletteType := OmeletteType.ThreeEgg },
    { customerCount := 8, omeletteType := OmeletteType.FourEgg }
  ]
  totalEggsNeeded orders = 84 := by
  sorry

end NUMINAMATH_CALUDE_eggs_needed_is_84_l3561_356134


namespace NUMINAMATH_CALUDE_blueberry_pie_count_l3561_356125

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 30)
  (h_ratio : apple_ratio + blueberry_ratio + cherry_ratio = 10)
  (h_apple : apple_ratio = 2)
  (h_blueberry : blueberry_ratio = 3)
  (h_cherry : cherry_ratio = 5) :
  (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) * blueberry_ratio = 9 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pie_count_l3561_356125


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l3561_356198

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l3561_356198


namespace NUMINAMATH_CALUDE_area_bounded_by_cos_sin_squared_l3561_356189

theorem area_bounded_by_cos_sin_squared (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos x * Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), f x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_cos_sin_squared_l3561_356189


namespace NUMINAMATH_CALUDE_largest_alpha_l3561_356157

theorem largest_alpha : ∃ (α : ℝ), (α = 3) ∧ 
  (∀ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 → α / n^2 ≤ 7 - (m : ℝ)^2 / n^2) ∧
  (∀ (β : ℝ), β > α → 
    ∃ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 ∧ β / n^2 > 7 - (m : ℝ)^2 / n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_alpha_l3561_356157


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3561_356121

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 3}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the intersection of P and Q
def PQ_intersection : Set ℝ := P ∩ Q

-- Define the half-open interval [3,4)
def interval_3_4 : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : PQ_intersection = interval_3_4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3561_356121


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3561_356120

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- Expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution X with E(X) = 3 and D(X) = 2, n = 9 and p = 1/3 -/
theorem binomial_distribution_unique_parameters :
  ∀ X : BinomialDistribution,
    expectation X = 3 →
    variance X = 2 →
    X.n = 9 ∧ X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3561_356120


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_81_l3561_356158

theorem alpha_plus_beta_equals_81 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 945) / (x^2 + 45*x - 3240)) : 
  α + β = 81 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_81_l3561_356158


namespace NUMINAMATH_CALUDE_sin_equation_holds_ten_degrees_is_acute_l3561_356127

theorem sin_equation_holds : 
  (Real.sin (10 * Real.pi / 180)) * (1 + Real.sqrt 3 * Real.tan (70 * Real.pi / 180)) = 1 := by
  sorry

-- Additional definition to ensure 10° is acute
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem ten_degrees_is_acute : is_acute_angle (10 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_equation_holds_ten_degrees_is_acute_l3561_356127


namespace NUMINAMATH_CALUDE_barge_length_is_125_steps_l3561_356104

/-- Represents the scenario of Jake walking along a barge on a river -/
structure BargeProblem where
  -- Length of Jake's step upstream
  step_length : ℝ
  -- Length the barge moves while Jake takes one step
  barge_speed : ℝ
  -- Length of the barge
  barge_length : ℝ
  -- Jake walks faster than the barge
  jake_faster : barge_speed < step_length
  -- 300 steps downstream from back to front
  downstream_eq : 300 * (1.5 * step_length) = barge_length + 300 * barge_speed
  -- 60 steps upstream from front to back
  upstream_eq : 60 * step_length = barge_length - 60 * barge_speed

/-- The length of the barge is 125 times Jake's upstream step length -/
theorem barge_length_is_125_steps (p : BargeProblem) : p.barge_length = 125 * p.step_length := by
  sorry


end NUMINAMATH_CALUDE_barge_length_is_125_steps_l3561_356104


namespace NUMINAMATH_CALUDE_smallest_number_l3561_356123

theorem smallest_number (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 2) (hd : d = -1/2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3561_356123


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l3561_356118

theorem max_area_rectangle_with_fixed_perimeter :
  ∀ (width height : ℝ),
  width > 0 → height > 0 →
  width + height = 50 →
  width * height ≤ 625 :=
by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l3561_356118


namespace NUMINAMATH_CALUDE_pseudoprime_construction_infinite_pseudoprimes_l3561_356188

/-- A number n is a pseudoprime to base a if it's composite and a^(n-1) ≡ 1 (mod n) -/
def IsPseudoprime (n : ℕ) (a : ℕ) : Prop :=
  ¬ Nat.Prime n ∧ a^(n-1) % n = 1

/-- Given a pseudoprime m, 2^m - 1 is also a pseudoprime -/
theorem pseudoprime_construction (m : ℕ) (a : ℕ) (h : IsPseudoprime m a) :
  ∃ b : ℕ, IsPseudoprime (2^m - 1) b :=
sorry

/-- There are infinitely many pseudoprimes -/
theorem infinite_pseudoprimes : ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ a : ℕ, IsPseudoprime m a :=
sorry

end NUMINAMATH_CALUDE_pseudoprime_construction_infinite_pseudoprimes_l3561_356188


namespace NUMINAMATH_CALUDE_max_truthful_students_2015_l3561_356124

/-- The maximum number of truthful students in the described arrangement --/
def max_truthful_students (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that for n = 2015, the maximum number of truthful students is 2031120 --/
theorem max_truthful_students_2015 :
  max_truthful_students 2015 = 2031120 := by
  sorry

#eval max_truthful_students 2015

end NUMINAMATH_CALUDE_max_truthful_students_2015_l3561_356124


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3561_356117

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3561_356117


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l3561_356181

theorem book_sale_gain_percentage (initial_sale_price : ℝ) (loss_percentage : ℝ) (desired_sale_price : ℝ) : 
  initial_sale_price = 810 →
  loss_percentage = 10 →
  desired_sale_price = 990 →
  let cost_price := initial_sale_price / (1 - loss_percentage / 100)
  let gain := desired_sale_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l3561_356181


namespace NUMINAMATH_CALUDE_problem_solution_l3561_356142

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  c - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3561_356142


namespace NUMINAMATH_CALUDE_socks_theorem_l3561_356184

def socks_problem (initial_pairs : ℕ) : Prop :=
  let week1 := 12
  let week2 := week1 + 4
  let week3 := (week1 + week2) / 2
  let week4 := week3 - 3
  let total := 57
  initial_pairs = total - (week1 + week2 + week3 + week4)

theorem socks_theorem : ∃ (x : ℕ), socks_problem x :=
sorry

end NUMINAMATH_CALUDE_socks_theorem_l3561_356184


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3561_356166

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 120

/-- The difference in water volume between 70% full and 40% full, in liters. -/
def volume_difference : ℝ := 36

/-- Theorem stating the total capacity of the tank given the volume difference between two fill levels. -/
theorem tank_capacity_proof :
  tank_capacity * (0.7 - 0.4) = volume_difference :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3561_356166


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3561_356186

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3561_356186


namespace NUMINAMATH_CALUDE_simplify_expression_l3561_356107

theorem simplify_expression : (2^8 + 4^5) * (1^3 - (-1)^3)^8 = 327680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3561_356107


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_two_l3561_356175

theorem sum_of_squares_equals_two (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) : 
  x^2 + y^2 + z^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_two_l3561_356175


namespace NUMINAMATH_CALUDE_volume_per_part_l3561_356169

/-- Given two rectangular prisms and a number of equal parts filling these prisms,
    calculate the volume of each part. -/
theorem volume_per_part
  (length width height : ℝ)
  (num_prisms num_parts : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_num_prisms : num_prisms = 2)
  (h_num_parts : num_parts = 16) :
  (num_prisms * length * width * height) / num_parts = 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_per_part_l3561_356169


namespace NUMINAMATH_CALUDE_band_member_earnings_l3561_356115

theorem band_member_earnings 
  (attendees : ℕ) 
  (revenue_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : revenue_share = 7/10) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) :
  (attendees * ticket_price * revenue_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l3561_356115


namespace NUMINAMATH_CALUDE_dorothy_and_sister_ages_l3561_356143

/-- Proves the ages of Dorothy and her sister given the conditions -/
theorem dorothy_and_sister_ages :
  ∀ (d s : ℕ),
  d = 3 * s →
  d + 5 = 2 * (s + 5) →
  d = 15 ∧ s = 5 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_and_sister_ages_l3561_356143


namespace NUMINAMATH_CALUDE_wall_ratio_l3561_356191

/-- Given a wall with specific dimensions, prove the ratio of its length to height --/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = w * h * l →
  w = 6.999999999999999 →
  volume = 86436 →
  l / h = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l3561_356191


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3561_356131

theorem arithmetic_computation : -9 * 3 - (-7 * -4) + (-11 * -6) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3561_356131


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3561_356149

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 7) ↔ m^2 ≥ (9/50) := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3561_356149


namespace NUMINAMATH_CALUDE_sum_of_roots_l3561_356176

theorem sum_of_roots (M : ℝ) : (∃ M₁ M₂ : ℝ, M₁ * (M₁ - 8) = 7 ∧ M₂ * (M₂ - 8) = 7 ∧ M₁ + M₂ = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3561_356176


namespace NUMINAMATH_CALUDE_prob_green_ball_l3561_356173

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def probGreen (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The four containers described in the problem -/
def containerA : Container := ⟨5, 7⟩
def containerB : Container := ⟨7, 3⟩
def containerC : Container := ⟨8, 2⟩
def containerD : Container := ⟨4, 6⟩

/-- The probability of selecting each container -/
def probContainer : ℚ := 1 / 4

/-- Theorem stating the probability of selecting a green ball -/
theorem prob_green_ball : 
  probContainer * probGreen containerA +
  probContainer * probGreen containerB +
  probContainer * probGreen containerC +
  probContainer * probGreen containerD = 101 / 240 := by
  sorry


end NUMINAMATH_CALUDE_prob_green_ball_l3561_356173


namespace NUMINAMATH_CALUDE_great_eighteen_games_l3561_356193

/-- The Great Eighteen Hockey League -/
structure HockeyLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of scheduled games in the league -/
def total_scheduled_games (league : HockeyLeague) : Nat :=
  let total_intra_division_games := league.total_teams * (league.teams_per_division - 1) * league.intra_division_games
  let total_inter_division_games := league.total_teams * league.teams_per_division * league.inter_division_games
  (total_intra_division_games + total_inter_division_games) / 2

/-- The Great Eighteen Hockey League satisfies the given conditions -/
def great_eighteen : HockeyLeague :=
  { total_teams := 18
  , teams_per_division := 9
  , intra_division_games := 3
  , inter_division_games := 2
  }

/-- Theorem: The total number of scheduled games in the Great Eighteen Hockey League is 378 -/
theorem great_eighteen_games :
  total_scheduled_games great_eighteen = 378 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_games_l3561_356193


namespace NUMINAMATH_CALUDE_possible_c_value_l3561_356160

/-- An obtuse-angled triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  obtuse : c^2 > a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that 2√5 is a possible value for c in the given obtuse triangle -/
theorem possible_c_value (t : ObtuseTriangle) 
  (ha : t.a = Real.sqrt 2)
  (hb : t.b = 2 * Real.sqrt 2)
  (hc : t.c > t.b) :
  ∃ (t' : ObtuseTriangle), t'.a = t.a ∧ t'.b = t.b ∧ t'.c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_possible_c_value_l3561_356160


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l3561_356108

/-- The number of different positive six-digit integers formed using the digits 1, 1, 1, 5, 9, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    formed using the digits 1, 1, 1, 5, 9, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l3561_356108


namespace NUMINAMATH_CALUDE_max_product_sum_l3561_356192

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 24) :
  (A * M * C + A * M + M * C + C * A) ≤ 704 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l3561_356192


namespace NUMINAMATH_CALUDE_total_sum_is_71_rupees_l3561_356140

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := 20 * coins_20_paise
  let value_25_paise := 25 * coins_25_paise
  (value_20_paise + value_25_paise : ℚ) / 100

/-- Theorem stating that given 342 total coins with 290 being 20 paise coins, 
    the total sum of money is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_in_rupees 342 290 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_is_71_rupees_l3561_356140


namespace NUMINAMATH_CALUDE_work_completion_proof_l3561_356128

/-- A's work rate in days -/
def a_rate : ℚ := 1 / 15

/-- B's work rate in days -/
def b_rate : ℚ := 1 / 20

/-- The fraction of work left after A and B work together -/
def work_left : ℚ := 65 / 100

/-- The number of days A and B worked together -/
def days_worked : ℚ := 3

theorem work_completion_proof :
  (a_rate + b_rate) * days_worked = 1 - work_left :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3561_356128


namespace NUMINAMATH_CALUDE_certain_number_proof_l3561_356122

theorem certain_number_proof : ∃! N : ℕ, 
  N % 101 = 8 ∧ 
  5161 % 101 = 10 ∧ 
  N = 5159 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3561_356122


namespace NUMINAMATH_CALUDE_nahco3_equals_nano3_l3561_356101

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction :=
  (naHCO3 : Moles)
  (hNO3 : Moles)
  (naNO3 : Moles)
  (h2O : Moles)
  (cO2 : Moles)

/-- The chemical equation is balanced -/
axiom balanced_equation (r : Reaction) : r.naHCO3 = r.hNO3 ∧ r.naHCO3 = r.naNO3

/-- The number of moles of HNO3 combined equals the number of moles of NaNO3 formed -/
axiom hno3_equals_nano3 (r : Reaction) : r.hNO3 = r.naNO3

/-- The stoichiometric ratio of NaHCO3 to NaNO3 is 1:1 -/
axiom stoichiometric_ratio (r : Reaction) : r.naHCO3 = r.naNO3

/-- Theorem: The number of moles of NaHCO3 combined equals the number of moles of NaNO3 formed -/
theorem nahco3_equals_nano3 (r : Reaction) : r.naHCO3 = r.naNO3 := by
  sorry

end NUMINAMATH_CALUDE_nahco3_equals_nano3_l3561_356101


namespace NUMINAMATH_CALUDE_average_income_calculation_l3561_356141

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of a specific pair. -/
theorem average_income_calculation (P Q R : ℕ) : 
  (P + Q) / 2 = 2050 →
  (Q + R) / 2 = 5250 →
  P = 3000 →
  (P + R) / 2 = 6200 := by
sorry

end NUMINAMATH_CALUDE_average_income_calculation_l3561_356141


namespace NUMINAMATH_CALUDE_prob_four_draws_ge_ten_expected_value_two_draws_l3561_356116

-- Define the bags and their contents
def bagA : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}
def bagB : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}

-- Define the probabilities of drawing each color
def probRedA : ℝ := 0.8
def probWhiteA : ℝ := 0.2
def probYellowB : ℝ := 0.9
def probBlackB : ℝ := 0.1

-- Define the scoring system
def scoreRed : ℤ := 4
def scoreWhite : ℤ := -1
def scoreYellow : ℤ := 6
def scoreBlack : ℤ := -2

-- Define the game rules
def fourDraws : ℕ := 4
def minScore : ℤ := 10

-- Theorem for Question 1
theorem prob_four_draws_ge_ten (p : ℝ) : 
  p = probRedA^4 + 4 * probRedA^3 * probWhiteA → p = 0.8192 := by sorry

-- Theorem for Question 2
theorem expected_value_two_draws (ev : ℝ) :
  ev = scoreRed * probRedA * probYellowB + 
        scoreRed * probRedA * probBlackB + 
        scoreWhite * probWhiteA * probYellowB + 
        scoreWhite * probWhiteA * probBlackB → ev = 8.2 := by sorry

end NUMINAMATH_CALUDE_prob_four_draws_ge_ten_expected_value_two_draws_l3561_356116


namespace NUMINAMATH_CALUDE_guitar_ratio_proof_l3561_356167

/-- Proves that the ratio of Barbeck's guitars to Steve's guitars is 2:1 given the problem conditions -/
theorem guitar_ratio_proof (total_guitars : ℕ) (davey_guitars : ℕ) (barbeck_guitars : ℕ) (steve_guitars : ℕ) : 
  total_guitars = 27 →
  davey_guitars = 18 →
  barbeck_guitars = steve_guitars →
  davey_guitars = 3 * barbeck_guitars →
  total_guitars = davey_guitars + barbeck_guitars + steve_guitars →
  (barbeck_guitars : ℚ) / steve_guitars = 2 / 1 :=
by sorry


end NUMINAMATH_CALUDE_guitar_ratio_proof_l3561_356167


namespace NUMINAMATH_CALUDE_digital_earth_has_info_at_fingertips_l3561_356196

-- Define the set of technologies
inductive Technology
| Internet
| VirtualWorld
| DigitalEarth
| InformationSuperhighway

-- Define the property of "information at your fingertips"
def hasInfoAtFingertips (t : Technology) : Prop :=
  match t with
  | Technology.DigitalEarth => true
  | _ => false

-- Theorem statement
theorem digital_earth_has_info_at_fingertips :
  hasInfoAtFingertips Technology.DigitalEarth :=
by
  sorry

#check digital_earth_has_info_at_fingertips

end NUMINAMATH_CALUDE_digital_earth_has_info_at_fingertips_l3561_356196


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3561_356100

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle is 18π. -/
theorem cone_lateral_surface_area (r : ℝ) (l : ℝ) : 
  r = 3 →
  π * l = 2 * π * r →
  π * r * l = 18 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3561_356100


namespace NUMINAMATH_CALUDE_min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l3561_356199

-- Part Ⅰ
theorem min_value_f (x : ℝ) (hx : x > 0) : 12/x + 3*x ≥ 12 := by
  sorry

theorem min_value_f_attained : ∃ x : ℝ, x > 0 ∧ 12/x + 3*x = 12 := by
  sorry

-- Part Ⅱ
theorem max_value_g (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x*(1 - 3*x) ≤ 1/12 := by
  sorry

theorem max_value_g_attained : ∃ x : ℝ, x > 0 ∧ x < 1/3 ∧ x*(1 - 3*x) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l3561_356199


namespace NUMINAMATH_CALUDE_stuffed_toy_dogs_boxes_l3561_356139

theorem stuffed_toy_dogs_boxes (dogs_per_box : ℕ) (total_dogs : ℕ) (h1 : dogs_per_box = 4) (h2 : total_dogs = 28) :
  total_dogs / dogs_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_toy_dogs_boxes_l3561_356139


namespace NUMINAMATH_CALUDE_john_hat_days_l3561_356113

def total_cost : ℕ := 700
def hat_cost : ℕ := 50

theorem john_hat_days : (total_cost / hat_cost : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_hat_days_l3561_356113


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3561_356187

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - 3 * x) ↔ x ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3561_356187


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l3561_356197

theorem parallelogram_perimeter (a b : ℝ) (ha : a = Real.sqrt 20) (hb : b = Real.sqrt 125) :
  2 * (a + b) = 14 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l3561_356197


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l3561_356129

theorem power_mod_seventeen : 5^2021 ≡ 11 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l3561_356129


namespace NUMINAMATH_CALUDE_decimal_123_in_base7_has_three_consecutive_digits_l3561_356151

/-- Represents a number in base 7 --/
def Base7 := Nat

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : Base7 :=
  sorry

/-- Checks if a Base7 number has three consecutive digits --/
def hasThreeConsecutiveDigits (n : Base7) : Prop :=
  sorry

/-- The decimal number we're working with --/
def decimalNumber : Nat := 123

theorem decimal_123_in_base7_has_three_consecutive_digits :
  hasThreeConsecutiveDigits (toBase7 decimalNumber) :=
sorry

end NUMINAMATH_CALUDE_decimal_123_in_base7_has_three_consecutive_digits_l3561_356151
