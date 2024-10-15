import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_l2042_204248

/-- A quadratic function of the form (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_inequality (m x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₁ + x₂ < 3) :
  f m x₁ > f m x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2042_204248


namespace NUMINAMATH_CALUDE_expression_evaluation_l2042_204259

theorem expression_evaluation : 15 - 6 / (-2) + |3| * (-5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2042_204259


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l2042_204236

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (remaining_jelly_beans : ℕ) (boy_girl_difference : ℕ) : 
  total_jelly_beans = 500 →
  remaining_jelly_beans = 10 →
  boy_girl_difference = 4 →
  ∃ (girls boys : ℕ),
    girls + boys = 32 ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - remaining_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l2042_204236


namespace NUMINAMATH_CALUDE_percentage_decrease_l2042_204229

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) : 
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decreased_value := original * (1 - (25 : ℝ) / 100)
  increased_value - decreased_value = difference :=
by
  sorry

#check percentage_decrease

end NUMINAMATH_CALUDE_percentage_decrease_l2042_204229


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2042_204289

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 3/2 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2042_204289


namespace NUMINAMATH_CALUDE_coefficient_x3_equals_neg16_l2042_204218

/-- The coefficient of x^3 in the expansion of (1-ax)^2(1+x)^6 -/
def coefficient_x3 (a : ℝ) : ℝ :=
  20 - 30*a + 6*a^2

theorem coefficient_x3_equals_neg16 (a : ℝ) :
  coefficient_x3 a = -16 → a = 2 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3_equals_neg16_l2042_204218


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_prime_reciprocals_l2042_204235

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List Nat) : ℚ :=
  let reciprocals := numbers.map (λ x => (1 : ℚ) / x)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_prime_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_prime_reciprocals_l2042_204235


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2042_204212

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 4 ∧ x + 3*y = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2042_204212


namespace NUMINAMATH_CALUDE_factorial_difference_l2042_204200

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2042_204200


namespace NUMINAMATH_CALUDE_abs_equation_quadratic_coefficients_l2042_204266

theorem abs_equation_quadratic_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_abs_equation_quadratic_coefficients_l2042_204266


namespace NUMINAMATH_CALUDE_graph_shift_l2042_204257

theorem graph_shift (x : ℝ) : (10 : ℝ) ^ (x + 3) = (10 : ℝ) ^ ((x + 4) - 1) := by
  sorry

end NUMINAMATH_CALUDE_graph_shift_l2042_204257


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2042_204245

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2042_204245


namespace NUMINAMATH_CALUDE_y_divisibility_l2042_204288

def y : ℕ := 72 + 108 + 144 + 180 + 324 + 396 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 12 * k) ∧
  (∃ k : ℕ, y = 18 * k) ∧
  (∃ k : ℕ, y = 36 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l2042_204288


namespace NUMINAMATH_CALUDE_parabola_tangent_intersection_l2042_204231

/-- Parabola defined by x^2 = 2y -/
def parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- Tangent line to the parabola at a given point (a, a^2/2) -/
def tangent_line (a x y : ℝ) : Prop := y - (a^2/2) = a*(x - a)

/-- Point of intersection of two lines -/
def intersection (m₁ b₁ m₂ b₂ x y : ℝ) : Prop :=
  y = m₁*x + b₁ ∧ y = m₂*x + b₂

theorem parabola_tangent_intersection :
  ∃ (x y : ℝ),
    intersection 4 (-8) (-2) (-2) x y ∧
    y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_intersection_l2042_204231


namespace NUMINAMATH_CALUDE_kants_clock_problem_l2042_204285

/-- Kant's Clock Problem -/
theorem kants_clock_problem (T_F T_2 T_S : ℝ) :
  ∃ T : ℝ, T = T_F + (T_2 - T_S) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_kants_clock_problem_l2042_204285


namespace NUMINAMATH_CALUDE_baby_shower_parking_lot_wheels_l2042_204278

/-- Calculates the total number of car wheels in a parking lot --/
def total_wheels (guest_cars : ℕ) (parent_cars : ℕ) (wheels_per_car : ℕ) : ℕ :=
  (guest_cars + parent_cars) * wheels_per_car

/-- Theorem statement for the baby shower parking lot problem --/
theorem baby_shower_parking_lot_wheels : 
  total_wheels 10 2 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_baby_shower_parking_lot_wheels_l2042_204278


namespace NUMINAMATH_CALUDE_company_attendees_l2042_204255

theorem company_attendees (total : ℕ) (other : ℕ) (h_total : total = 185) (h_other : other = 20) : 
  ∃ (a : ℕ), 
    a + (2 * a) + (a + 10) + (a + 5) + other = total ∧ 
    a = 30 := by
  sorry

end NUMINAMATH_CALUDE_company_attendees_l2042_204255


namespace NUMINAMATH_CALUDE_binary_1011011_equals_91_l2042_204299

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1011011_equals_91 :
  binary_to_decimal [true, true, false, true, true, false, true] = 91 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011011_equals_91_l2042_204299


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2042_204294

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y > 0 ∧ Nat.lcm x (Nat.lcm 15 21) = 210) → x ≤ 70 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2042_204294


namespace NUMINAMATH_CALUDE_seedling_probability_l2042_204279

def total_seedlings : ℕ := 14
def selechenskaya_seedlings : ℕ := 6
def vologda_seedlings : ℕ := 8
def selected_seedlings : ℕ := 3

theorem seedling_probability :
  (Nat.choose selechenskaya_seedlings selected_seedlings : ℚ) / 
  (Nat.choose total_seedlings selected_seedlings : ℚ) = 5 / 91 := by
  sorry

end NUMINAMATH_CALUDE_seedling_probability_l2042_204279


namespace NUMINAMATH_CALUDE_correct_sum_calculation_l2042_204298

theorem correct_sum_calculation (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : n % 10 = 9) (h3 : (n - 3 + 57) = 1823) : n + 57 = 1826 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sum_calculation_l2042_204298


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_existence_n_6_largest_n_is_6_l2042_204237

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem existence_n_6 : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_6 : 
  ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) ∧
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_existence_n_6_largest_n_is_6_l2042_204237


namespace NUMINAMATH_CALUDE_kite_profit_theorem_l2042_204291

/-- Cost price of type B kite -/
def cost_B : ℝ := 80

/-- Cost price of type A kite -/
def cost_A : ℝ := cost_B + 20

/-- Selling price of type B kite -/
def sell_B : ℝ := 120

/-- Selling price of type A kite -/
def sell_A (m : ℝ) : ℝ := 2 * (130 - m)

/-- Total number of kites -/
def total_kites : ℕ := 300

/-- Profit function -/
def profit (m : ℝ) : ℝ := (sell_A m - cost_A) * m + (sell_B - cost_B) * (total_kites - m)

/-- Theorem stating the cost prices and maximum profit -/
theorem kite_profit_theorem :
  (∀ m : ℝ, 50 ≤ m → m ≤ 150 → profit m ≤ 13000) ∧
  (20000 / cost_A = 2 * (8000 / cost_B)) ∧
  (profit 50 = 13000) := by sorry

end NUMINAMATH_CALUDE_kite_profit_theorem_l2042_204291


namespace NUMINAMATH_CALUDE_carol_initial_peanuts_l2042_204268

/-- The number of peanuts Carol initially collected -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Carol's father gave her -/
def fathers_peanuts : ℕ := 5

/-- The total number of peanuts Carol has after receiving peanuts from her father -/
def total_peanuts : ℕ := 7

/-- Theorem: Carol initially collected 2 peanuts -/
theorem carol_initial_peanuts : 
  initial_peanuts + fathers_peanuts = total_peanuts ∧ initial_peanuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_carol_initial_peanuts_l2042_204268


namespace NUMINAMATH_CALUDE_distance_sf_to_atlantis_l2042_204267

theorem distance_sf_to_atlantis : 
  let sf : ℂ := 0
  let atlantis : ℂ := 1300 + 3120 * I
  Complex.abs (atlantis - sf) = 3380 := by
sorry

end NUMINAMATH_CALUDE_distance_sf_to_atlantis_l2042_204267


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2042_204276

/-- Given a triangle with side lengths x-1, x+1, and 7, where x = 10, the perimeter of the triangle is 27. -/
theorem triangle_perimeter (x : ℝ) : x = 10 → (x - 1) + (x + 1) + 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2042_204276


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2042_204209

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 8 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2042_204209


namespace NUMINAMATH_CALUDE_product_equals_zero_l2042_204256

def product_sequence (a : ℤ) : ℤ := (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_equals_zero : product_sequence 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2042_204256


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2042_204234

theorem cube_root_equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (5 - 1/x)^(1/3 : ℝ) = -6 ↔ x = 1/221 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2042_204234


namespace NUMINAMATH_CALUDE_custom_op_equality_l2042_204282

/-- Custom operation ⊗ -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality of the expression and its simplified form -/
theorem custom_op_equality (a b : ℝ) : 
  custom_op a b + custom_op (b - a) b = b^2 - b := by sorry

end NUMINAMATH_CALUDE_custom_op_equality_l2042_204282


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_over_pi_halves_l2042_204263

open Real MeasureTheory

theorem integral_one_plus_sin_over_pi_halves : 
  ∫ x in (-π/2)..(π/2), (1 + Real.sin x) = π := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_over_pi_halves_l2042_204263


namespace NUMINAMATH_CALUDE_range_of_a_l2042_204274

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

-- Define the property of having one positive and one negative root
def has_pos_neg_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
  quadratic_eq a x₁ = 0 ∧ quadratic_eq a x₂ = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_pos_neg_roots a ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2042_204274


namespace NUMINAMATH_CALUDE_function_inequality_l2042_204242

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (-x) = f (2 + x))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) 
  (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) > f (2 - x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2042_204242


namespace NUMINAMATH_CALUDE_lcm_18_35_l2042_204292

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l2042_204292


namespace NUMINAMATH_CALUDE_fraction_woodwind_brass_this_year_l2042_204225

-- Define the fractions of students for each instrument last year
def woodwind_last_year : ℚ := 1/2
def brass_last_year : ℚ := 2/5
def percussion_last_year : ℚ := 1 - (woodwind_last_year + brass_last_year)

-- Define the fractions of students who left for each instrument
def woodwind_left : ℚ := 1/2
def brass_left : ℚ := 1/4
def percussion_left : ℚ := 0

-- Calculate the fractions of students for each instrument this year
def woodwind_this_year : ℚ := woodwind_last_year * (1 - woodwind_left)
def brass_this_year : ℚ := brass_last_year * (1 - brass_left)
def percussion_this_year : ℚ := percussion_last_year * (1 - percussion_left)

-- Theorem to prove
theorem fraction_woodwind_brass_this_year :
  woodwind_this_year + brass_this_year = 11/20 := by sorry

end NUMINAMATH_CALUDE_fraction_woodwind_brass_this_year_l2042_204225


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l2042_204250

-- Define the total number of contestants
def total_contestants : ℕ := 20

-- Define the number of tribes
def num_tribes : ℕ := 4

-- Define the number of contestants per tribe
def contestants_per_tribe : ℕ := 5

-- Define the number of quitters
def num_quitters : ℕ := 3

-- Theorem statement
theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_contestants num_quitters
  let same_tribe_ways := num_tribes * Nat.choose contestants_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 2 / 57 := by
  sorry


end NUMINAMATH_CALUDE_survivor_quitters_probability_l2042_204250


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2042_204246

theorem arithmetic_operations :
  (3 - (-2) = 5) ∧
  ((-4) * (-3) = 12) ∧
  (0 / (-3) = 0) ∧
  (|(-12)| + (-4) = 8) ∧
  ((3) - 14 - (-5) + (-16) = -22) ∧
  ((-5) / (-1/5) * (-5) = -125) ∧
  (-24 * ((-5/6) + (3/8) - (1/12)) = 13) ∧
  (3 * (-4) + 18 / (-6) - (-2) = -12) ∧
  ((-99 - 15/16) * 4 = -399 - 3/4) := by
sorry

#eval 3 - (-2)
#eval (-4) * (-3)
#eval 0 / (-3)
#eval |(-12)| + (-4)
#eval 3 - 14 - (-5) + (-16)
#eval (-5) / (-1/5) * (-5)
#eval -24 * ((-5/6) + (3/8) - (1/12))
#eval 3 * (-4) + 18 / (-6) - (-2)
#eval (-99 - 15/16) * 4

end NUMINAMATH_CALUDE_arithmetic_operations_l2042_204246


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l2042_204270

def baseball_cards_problem (initial_cards : ℝ) (promised_cards : ℝ) (bought_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - promised_cards

theorem mary_baseball_cards :
  baseball_cards_problem 18.0 26.0 40.0 = 32.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l2042_204270


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2042_204284

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 7) = -4 * s + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2042_204284


namespace NUMINAMATH_CALUDE_jake_brought_four_balloons_l2042_204254

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := 2

/-- The total number of balloons Allan and Jake had -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := total_balloons - allan_balloons

theorem jake_brought_four_balloons : jake_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_four_balloons_l2042_204254


namespace NUMINAMATH_CALUDE_circle_chord_tangent_relation_l2042_204260

/-- Given a circle with diameter AB and radius r, chord BF extended to meet
    the tangent at A at point C, and point E on BC extended such that BE = DC,
    prove that h = √(r² - d²), where d is the distance from E to the tangent at B
    and h is the distance from E to the diameter AB. -/
theorem circle_chord_tangent_relation (r d h : ℝ) : h = Real.sqrt (r^2 - d^2) :=
sorry

end NUMINAMATH_CALUDE_circle_chord_tangent_relation_l2042_204260


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l2042_204290

theorem tan_x_minus_pi_sixth (x : ℝ) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l2042_204290


namespace NUMINAMATH_CALUDE_area_of_specific_figure_l2042_204241

/-- A figure composed of squares and triangles -/
structure Figure where
  num_squares : ℕ
  num_triangles : ℕ

/-- The area of a figure in square centimeters -/
def area (f : Figure) : ℝ :=
  f.num_squares + (f.num_triangles * 0.5)

/-- Theorem: The area of a specific figure is 10.5 cm² -/
theorem area_of_specific_figure :
  ∃ (f : Figure), f.num_squares = 8 ∧ f.num_triangles = 5 ∧ area f = 10.5 :=
sorry

end NUMINAMATH_CALUDE_area_of_specific_figure_l2042_204241


namespace NUMINAMATH_CALUDE_addition_proof_l2042_204273

theorem addition_proof : 9873 + 3927 = 13800 := by
  sorry

end NUMINAMATH_CALUDE_addition_proof_l2042_204273


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l2042_204261

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l2042_204261


namespace NUMINAMATH_CALUDE_remaining_money_l2042_204265

def salary : ℝ := 8123.08

theorem remaining_money (food_fraction : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ)
  (h_food : food_fraction = 1/3)
  (h_rent : rent_fraction = 1/4)
  (h_clothes : clothes_fraction = 1/5) :
  let total_expenses := salary * (food_fraction + rent_fraction + clothes_fraction)
  ∃ ε > 0, |salary - total_expenses - 1759.00| < ε :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_l2042_204265


namespace NUMINAMATH_CALUDE_F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l2042_204223

-- Define the function F(x, t)
def F (x t : ℝ) : ℝ := |2*x + t| + x^2 + x + 1

-- Theorem for the minimum value when t = -1
theorem F_minimum_at_negative_one :
  ∃ (x_min : ℝ), F x_min (-1) = 7/4 ∧ ∀ (x : ℝ), F x (-1) ≥ 7/4 :=
sorry

-- Theorems for monotonic intervals
theorem F_monotonic_intervals_t_ge_3 (t : ℝ) (h : t ≥ 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -3/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -3/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_between (t : ℝ) (h1 : -1 < t) (h2 : t < 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -t/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -t/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_le_neg_one (t : ℝ) (h : t ≤ -1) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, 1/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

end NUMINAMATH_CALUDE_F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l2042_204223


namespace NUMINAMATH_CALUDE_expression_value_l2042_204244

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2042_204244


namespace NUMINAMATH_CALUDE_power_of_three_implies_large_prime_factor_l2042_204283

theorem power_of_three_implies_large_prime_factor (n : ℕ+) :
  (∃ k : ℕ, 125 * n + 22 = 3^k) →
  ∃ p : ℕ, p > 100 ∧ Prime p ∧ p ∣ (125 * n + 29) :=
by sorry

end NUMINAMATH_CALUDE_power_of_three_implies_large_prime_factor_l2042_204283


namespace NUMINAMATH_CALUDE_g_at_zero_l2042_204217

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_constant_term : f.coeff 0 = 5

-- Define the constant term of h
axiom h_constant_term : h.coeff 0 = -10

-- Theorem to prove
theorem g_at_zero : g.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_g_at_zero_l2042_204217


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2042_204208

def henry_age : ℕ := 27
def jill_age : ℕ := 16

theorem age_ratio_proof :
  (henry_age + jill_age = 43) →
  (henry_age - 5) / (jill_age - 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2042_204208


namespace NUMINAMATH_CALUDE_student_groups_l2042_204295

theorem student_groups (group_size : ℕ) (left_early : ℕ) (remaining : ℕ) : 
  group_size = 8 → left_early = 2 → remaining = 22 → 
  (remaining + left_early) / group_size = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_student_groups_l2042_204295


namespace NUMINAMATH_CALUDE_exists_claw_count_for_total_time_specific_grooming_problem_l2042_204213

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  clipTime : ℕ -- Time to clip one nail in seconds
  earCleanTime : ℕ -- Time to clean one ear in seconds
  shampooTime : ℕ -- Time to shampoo in seconds

/-- Theorem stating that there exists a number of claws that results in the given total grooming time -/
theorem exists_claw_count_for_total_time 
  (g : GroomingTime) 
  (totalTime : ℕ) : 
  ∃ (clawCount : ℕ), 
    g.clipTime * clawCount + g.earCleanTime * 2 + g.shampooTime * 60 = totalTime :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem specific_grooming_problem : 
  ∃ (clawCount : ℕ), 
    10 * clawCount + 90 * 2 + 5 * 60 = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_claw_count_for_total_time_specific_grooming_problem_l2042_204213


namespace NUMINAMATH_CALUDE_range_of_a_l2042_204275

def proposition_p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

def proposition_q (x : ℝ) : Prop :=
  x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

def negation_p_necessary_not_sufficient_for_negation_q (a : ℝ) : Prop :=
  ∀ x, ¬(proposition_q x) → ¬(proposition_p a x) ∧
  ∃ x, ¬(proposition_p a x) ∧ proposition_q x

theorem range_of_a :
  ∀ a : ℝ, negation_p_necessary_not_sufficient_for_negation_q a →
  (a < 0 ∧ a > -4) ∨ a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2042_204275


namespace NUMINAMATH_CALUDE_game_night_sandwiches_l2042_204214

theorem game_night_sandwiches (num_friends : ℕ) (sandwiches_per_friend : ℕ) 
  (h1 : num_friends = 7) (h2 : sandwiches_per_friend = 5) : 
  num_friends * sandwiches_per_friend = 35 := by
  sorry

end NUMINAMATH_CALUDE_game_night_sandwiches_l2042_204214


namespace NUMINAMATH_CALUDE_cyclist_distance_l2042_204271

/-- The distance between two points A and B for two cyclists with given conditions -/
theorem cyclist_distance (a k : ℝ) (ha : a > 0) (hk : k > 0) : ∃ (z x y : ℝ),
  z > 0 ∧ x > y ∧ y > 0 ∧
  (z + a) / (z - a) = x / y ∧
  (2 * k + 1) * z / ((2 * k - 1) * z) = x / y ∧
  z = 2 * a * k :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l2042_204271


namespace NUMINAMATH_CALUDE_sum_s_r_x_is_negative_fifteen_l2042_204206

def r (x : ℝ) : ℝ := |x| - 3
def s (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_x_is_negative_fifteen :
  (x_values.map (λ x => s (r x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_s_r_x_is_negative_fifteen_l2042_204206


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l2042_204277

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l2042_204277


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2042_204272

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2042_204272


namespace NUMINAMATH_CALUDE_fraction_equality_l2042_204280

theorem fraction_equality (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x - y) / (x + y) = -1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2042_204280


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2042_204222

/-- Given a point M(a, b) outside the unit circle, prove that the line ax + by = 1 intersects the circle -/
theorem line_intersects_circle (a b : ℝ) (h : a^2 + b^2 > 1) :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a * x + b * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2042_204222


namespace NUMINAMATH_CALUDE_range_of_expression_l2042_204252

theorem range_of_expression (x y a b : ℝ) : 
  x ≥ 1 →
  y ≥ 2 →
  x + y ≤ 4 →
  2*a + b ≥ 1 →
  3*a - b ≥ 2 →
  5*a ≤ 4 →
  (b + 2) / (a - 1) ≥ -12 ∧ (b + 2) / (a - 1) ≤ -9/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l2042_204252


namespace NUMINAMATH_CALUDE_hospital_patient_distribution_l2042_204233

/-- Represents the number of patients each doctor takes care of in a hospital -/
def patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) : ℕ :=
  total_patients / total_doctors

/-- Theorem stating that given 400 patients and 16 doctors, each doctor takes care of 25 patients -/
theorem hospital_patient_distribution :
  patients_per_doctor 400 16 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hospital_patient_distribution_l2042_204233


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l2042_204281

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry

-- Theorem statements
theorem perpendicular_parallel_implies_perpendicular 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel_lines n α → perpendicular m n := by sorry

theorem parallel_planes_line_in_plane_implies_parallel 
  (m : Line) (α β : Plane) :
  parallel_planes α β → line_in_plane m α → parallel_lines m β := by sorry

theorem parallel_lines_planes_implies_equal_angles 
  (m n : Line) (α β : Plane) :
  parallel_lines m n → parallel_planes α β → 
  angle_with_plane m α = angle_with_plane n β := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l2042_204281


namespace NUMINAMATH_CALUDE_sum_of_defined_values_l2042_204227

theorem sum_of_defined_values : 
  let x : ℝ := -2 + 3
  let y : ℝ := |(-5)|
  let z : ℝ := 4 * (-1/4)
  x + y + z = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_defined_values_l2042_204227


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l2042_204251

theorem compare_sqrt_expressions : 3 * Real.sqrt 5 > 2 * Real.sqrt 11 := by sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l2042_204251


namespace NUMINAMATH_CALUDE_triangle_area_side_a_value_l2042_204204

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosA : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.cosA = 1/2 ∧ t.b * t.c = 3

-- Theorem 1: Area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2 : ℝ) * t.b * t.c * Real.sqrt (1 - t.cosA^2) = Real.sqrt 3 / 2 := by
  sorry

-- Theorem 2: Value of side a when c = 1
theorem side_a_value (t : Triangle) (h : triangle_conditions t) (h_c : t.c = 1) :
  t.a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_side_a_value_l2042_204204


namespace NUMINAMATH_CALUDE_sequence_expression_evaluation_l2042_204202

theorem sequence_expression_evaluation :
  ∀ (x : ℝ),
  (∀ (n : ℕ), n > 0 → n = 2^(n-1) * x) →
  x = 1 →
  2*x * 6*x + 5*x / (4*x) - 56*x = 69/8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_expression_evaluation_l2042_204202


namespace NUMINAMATH_CALUDE_sum_is_34_l2042_204211

/-- Represents a 4x4 grid filled with integers from 1 to 16 -/
def Grid : Type := Fin 4 → Fin 4 → Fin 16

/-- Fills the grid sequentially from 1 to 16 -/
def fillGrid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Represents a selection of 4 positions in the grid, each from a different row and column -/
structure Selection :=
  (pos : Fin 4 → Fin 4 × Fin 4)
  (different_rows : ∀ i j, i ≠ j → (pos i).1 ≠ (pos j).1)
  (different_cols : ∀ i j, i ≠ j → (pos i).2 ≠ (pos j).2)

/-- The main theorem to be proved -/
theorem sum_is_34 (s : Selection) : 
  (Finset.univ.sum fun i => (fillGrid (s.pos i).1 (s.pos i).2).val) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_34_l2042_204211


namespace NUMINAMATH_CALUDE_angle_problem_l2042_204226

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define complementary angles
def complementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 90 * 60

-- Define supplementary angles
def supplementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 180 * 60

-- State the theorem
theorem angle_problem (angle1 angle2 angle3 : Angle) :
  complementary angle1 angle2 →
  supplementary angle2 angle3 →
  angle1 = Angle.mk 67 12 →
  angle3 = Angle.mk 157 12 :=
by sorry

end NUMINAMATH_CALUDE_angle_problem_l2042_204226


namespace NUMINAMATH_CALUDE_max_product_of_digits_divisible_by_25_l2042_204201

theorem max_product_of_digits_divisible_by_25 (a b : Nat) : 
  a ≤ 9 →
  b ≤ 9 →
  (10 * a + b) % 25 = 0 →
  b * a ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_digits_divisible_by_25_l2042_204201


namespace NUMINAMATH_CALUDE_weight_of_a_l2042_204216

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l2042_204216


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l2042_204215

theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = a + 2 →        -- legs differ by 2
  c = 53 →           -- hypotenuse is 53
  a + b = 104 :=     -- sum of legs is 104
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l2042_204215


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2042_204228

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a ≠ 0) :
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a = 1 ∧ b = -6 ∧ c = -7) →
  (r1 + r2 = 6 ∧ r1 - r2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2042_204228


namespace NUMINAMATH_CALUDE_election_win_margin_l2042_204249

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = 837 →
    winner_votes = (62 * total_votes) / 100 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 324 := by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l2042_204249


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_four_l2042_204219

/-- Given a function f where f(x) = f'(π/4) * cos(x) + sin(x), prove that f(π/4) = 1 -/
theorem function_value_at_pi_over_four (f : ℝ → ℝ) 
  (h : ∀ x, f x = (deriv f (π/4)) * Real.cos x + Real.sin x) : 
  f (π/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_four_l2042_204219


namespace NUMINAMATH_CALUDE_rearrangeable_natural_segments_l2042_204293

theorem rearrangeable_natural_segments (A B : Fin 1961 → ℕ) : 
  ∃ (σ τ : Equiv.Perm (Fin 1961)) (m : ℕ),
    ∀ (i : Fin 1961), A (σ i) + B (τ i) = m + i.val :=
sorry

end NUMINAMATH_CALUDE_rearrangeable_natural_segments_l2042_204293


namespace NUMINAMATH_CALUDE_product_of_cubes_l2042_204230

theorem product_of_cubes (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) : r * s = (1/48)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubes_l2042_204230


namespace NUMINAMATH_CALUDE_joanie_wants_three_cups_l2042_204253

-- Define the relationship between tablespoons of kernels and cups of popcorn
def kernels_to_popcorn (tablespoons : ℕ) : ℕ := 2 * tablespoons

-- Define the amount of popcorn each person wants
def mitchell_popcorn : ℕ := 4
def miles_davis_popcorn : ℕ := 6
def cliff_popcorn : ℕ := 3

-- Define the total amount of kernels needed
def total_kernels : ℕ := 8

-- Define Joanie's popcorn amount
def joanie_popcorn : ℕ := kernels_to_popcorn total_kernels - (mitchell_popcorn + miles_davis_popcorn + cliff_popcorn)

-- Theorem statement
theorem joanie_wants_three_cups :
  joanie_popcorn = 3 := by sorry

end NUMINAMATH_CALUDE_joanie_wants_three_cups_l2042_204253


namespace NUMINAMATH_CALUDE_reciprocal_roots_implies_p_zero_l2042_204264

-- Define the quadratic equation
def quadratic (p : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition for reciprocal roots
def has_reciprocal_roots (p : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r * s = 1 ∧
  quadratic p r = 0 ∧ quadratic p s = 0

-- Theorem statement
theorem reciprocal_roots_implies_p_zero :
  has_reciprocal_roots p → p = 0 := by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_implies_p_zero_l2042_204264


namespace NUMINAMATH_CALUDE_abc_value_l2042_204221

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 3 (1/3))
  (hac : a * c = 42 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 90 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2042_204221


namespace NUMINAMATH_CALUDE_investment_ratio_l2042_204203

/-- Represents the business investment scenario -/
structure Investment where
  nandan_amount : ℝ
  nandan_time : ℝ
  krishan_amount : ℝ
  krishan_time : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The theorem representing the investment problem -/
theorem investment_ratio (i : Investment) 
  (h1 : i.krishan_amount = 4 * i.nandan_amount)
  (h2 : i.total_gain = 26000)
  (h3 : i.nandan_gain = 2000)
  (h4 : ∃ (k : ℝ), i.nandan_gain / i.total_gain = 
       (i.nandan_amount * i.nandan_time) / 
       (i.nandan_amount * i.nandan_time + i.krishan_amount * i.krishan_time)) :
  i.krishan_time / i.nandan_time = 3 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_l2042_204203


namespace NUMINAMATH_CALUDE_cube_prime_factorization_l2042_204287

theorem cube_prime_factorization (x y : ℕ+) (p : ℕ) :
  (x + y) * (x^2 + 9*y) = p^3 ∧ Nat.Prime p ↔ (x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_prime_factorization_l2042_204287


namespace NUMINAMATH_CALUDE_base4_odd_digits_317_l2042_204240

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_odd_digits_317 :
  countOddDigits (toBase4 317) = 4 := by
  sorry

end NUMINAMATH_CALUDE_base4_odd_digits_317_l2042_204240


namespace NUMINAMATH_CALUDE_square_sum_value_l2042_204205

theorem square_sum_value (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l2042_204205


namespace NUMINAMATH_CALUDE_tree_cutting_theorem_l2042_204224

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def james_solo_days : ℕ := 2

/-- The number of days the brothers help -/
def brother_help_days : ℕ := 3

/-- The number of brothers helping -/
def num_brothers : ℕ := 2

/-- The percentage reduction in trees cut by brothers compared to James -/
def brother_reduction_percent : ℚ := 20 / 100

/-- The total number of trees cut down -/
def total_trees_cut : ℕ := 136

theorem tree_cutting_theorem :
  james_trees_per_day * james_solo_days + 
  (james_trees_per_day * (1 - brother_reduction_percent) * num_brothers * brother_help_days).floor = 
  total_trees_cut :=
sorry

end NUMINAMATH_CALUDE_tree_cutting_theorem_l2042_204224


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2042_204210

/-- 
Given an initial angle of 40 degrees that is rotated 480 degrees clockwise,
the resulting acute angle measures 80 degrees.
-/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 40 →
  rotation = 480 →
  (rotation % 360 - initial_angle) % 180 = 80 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2042_204210


namespace NUMINAMATH_CALUDE_arcade_play_time_l2042_204297

def weekly_pay : ℕ := 100
def arcade_budget : ℕ := weekly_pay / 2
def food_cost : ℕ := 10
def token_budget : ℕ := arcade_budget - food_cost
def play_cost : ℕ := 8
def total_play_time : ℕ := 300

theorem arcade_play_time : 
  (token_budget / play_cost) * (total_play_time / (token_budget / play_cost)) = total_play_time :=
by sorry

end NUMINAMATH_CALUDE_arcade_play_time_l2042_204297


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_60_429_l2042_204238

theorem gcd_lcm_sum_60_429 : Nat.gcd 60 429 + Nat.lcm 60 429 = 8583 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_60_429_l2042_204238


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l2042_204269

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2 ∨ d = 3

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, 
    valid_digits n → 
    digit_sum n = 13 → 
    n ≤ 222211111 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l2042_204269


namespace NUMINAMATH_CALUDE_milk_water_ratio_l2042_204258

/-- 
Given a mixture of milk and water with total volume 145 liters,
if adding 58 liters of water changes the ratio of milk to water to 3:4,
then the initial ratio of milk to water was 3:2.
-/
theorem milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (milk : ℝ) 
  (water : ℝ) : 
  total_volume = 145 →
  added_water = 58 →
  milk + water = total_volume →
  milk / (water + added_water) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l2042_204258


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l2042_204286

-- Define q as the largest prime with 2011 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2011 digits
axiom q_digits : 10^2010 ≤ q ∧ q < 10^2011

-- Define the property we want to prove
def is_divisible_by_15 (m : ℕ) : Prop :=
  ∃ k : ℤ, (q^2 - m : ℤ) = 15 * k

-- Theorem statement
theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ is_divisible_by_15 m ∧
  ∀ n : ℕ, 0 < n ∧ n < m → ¬is_divisible_by_15 n :=
sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l2042_204286


namespace NUMINAMATH_CALUDE_dartboard_region_angle_l2042_204232

def circular_dartboard (total_area : ℝ) : Prop := total_area > 0

def region_probability (prob : ℝ) : Prop := prob = 1 / 8

def central_angle (angle : ℝ) : Prop := 
  0 ≤ angle ∧ angle ≤ 360

theorem dartboard_region_angle 
  (total_area : ℝ) 
  (prob : ℝ) 
  (angle : ℝ) :
  circular_dartboard total_area →
  region_probability prob →
  central_angle angle →
  prob = angle / 360 →
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_dartboard_region_angle_l2042_204232


namespace NUMINAMATH_CALUDE_inspection_probability_l2042_204220

theorem inspection_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90)
  (h2 : pass_rate2 = 0.95) :
  let fail_rate1 := 1 - pass_rate1
  let fail_rate2 := 1 - pass_rate2
  pass_rate1 * fail_rate2 + fail_rate1 * pass_rate2 = 0.14 :=
by sorry

end NUMINAMATH_CALUDE_inspection_probability_l2042_204220


namespace NUMINAMATH_CALUDE_increasing_function_integral_inequality_l2042_204207

theorem increasing_function_integral_inequality
  (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ (a b c : ℝ), a < b → b < c →
    (c - b) * ∫ x in a..b, f x ≤ (b - a) * ∫ x in b..c, f x) ↔
  Monotone f :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_integral_inequality_l2042_204207


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2042_204243

/-- The value of c for which the line x - y + c = 0 is tangent to the circle (x - 1)^2 + y^2 = 2 -/
theorem line_tangent_to_circle (x y : ℝ) :
  (∃! p : ℝ × ℝ, (p.1 - p.2 + c = 0) ∧ ((p.1 - 1)^2 + p.2^2 = 2)) →
  c = -1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2042_204243


namespace NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2042_204296

/-- Represents the outcome of throwing a coin -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of throwing 3 coins simultaneously -/
def ThreeCoinsOutcome := (CoinOutcome × CoinOutcome × CoinOutcome)

/-- Counts the number of heads in a ThreeCoinsOutcome -/
def countHeads : ThreeCoinsOutcome → Nat
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Heads) => 3
  | (CoinOutcome.Heads, CoinOutcome.Heads, CoinOutcome.Tails) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Heads) => 2
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Heads) => 2
  | (CoinOutcome.Heads, CoinOutcome.Tails, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Heads, CoinOutcome.Tails) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Heads) => 1
  | (CoinOutcome.Tails, CoinOutcome.Tails, CoinOutcome.Tails) => 0

/-- Event: At most one head facing up -/
def atMostOneHead (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≤ 1

/-- Event: At least two heads facing up -/
def atLeastTwoHeads (outcome : ThreeCoinsOutcome) : Prop :=
  countHeads outcome ≥ 2

/-- Theorem: The events "at most one head facing up" and "at least two heads facing up" 
    are mutually exclusive when throwing 3 coins simultaneously -/
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : ThreeCoinsOutcome), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by sorry

end NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_l2042_204296


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2042_204262

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (a^2 - 4*a + 3 + Complex.I * (a - 1) : ℂ).im ≠ 0 → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2042_204262


namespace NUMINAMATH_CALUDE_zero_exponent_is_one_l2042_204239

theorem zero_exponent_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_is_one_l2042_204239


namespace NUMINAMATH_CALUDE_quadratic_and_slope_l2042_204247

-- Define the quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def passes_through (a b c : ℝ) : Prop :=
  quadratic a b c 1 = -2 ∧
  quadratic a b c 2 = 4 ∧
  quadratic a b c 3 = 10

-- Define the slope of the tangent line
def tangent_slope (a b c : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Theorem statement
theorem quadratic_and_slope :
  ∃ a b c : ℝ,
    passes_through a b c ∧
    (∀ x : ℝ, quadratic a b c x = 6 * x - 8) ∧
    tangent_slope a b c 2 = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_and_slope_l2042_204247
