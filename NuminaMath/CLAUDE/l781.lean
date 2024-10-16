import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_of_116_million_l781_78151

theorem scientific_notation_of_116_million :
  (116000000 : ℝ) = 1.16 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_116_million_l781_78151


namespace NUMINAMATH_CALUDE_percentage_problem_l781_78115

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 200) : 
  (1200 / x) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l781_78115


namespace NUMINAMATH_CALUDE_max_value_is_58_l781_78150

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def cave_problem :=
  let stone7 : Stone := { weight := 7, value := 16 }
  let stone3 : Stone := { weight := 3, value := 9 }
  let stone2 : Stone := { weight := 2, value := 4 }
  let max_weight : ℕ := 20
  let max_stone7 : ℕ := 2
  (stone7, stone3, stone2, max_weight, max_stone7)

/-- The function to maximize the value of stones -/
def maximize_value (p : Stone × Stone × Stone × ℕ × ℕ) : ℕ :=
  let (stone7, stone3, stone2, max_weight, max_stone7) := p
  sorry -- The actual maximization logic would go here

/-- The theorem stating that the maximum value is 58 -/
theorem max_value_is_58 : maximize_value cave_problem = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_58_l781_78150


namespace NUMINAMATH_CALUDE_mary_paid_three_more_than_john_l781_78131

/-- Represents the pizza sharing scenario between John and Mary -/
structure PizzaSharing where
  total_slices : Nat
  base_price : ℚ
  cheese_price : ℚ
  mary_cheese_slices : Nat
  mary_plain_slices : Nat

/-- Calculates the price difference between Mary's and John's payments -/
def price_difference (p : PizzaSharing) : ℚ :=
  let total_price := p.base_price + p.cheese_price
  let price_per_slice := total_price / p.total_slices
  let plain_price_per_slice := p.base_price / p.total_slices
  let mary_payment := price_per_slice * p.mary_cheese_slices + plain_price_per_slice * p.mary_plain_slices
  let john_slices := p.total_slices - p.mary_cheese_slices - p.mary_plain_slices
  let john_payment := plain_price_per_slice * john_slices
  mary_payment - john_payment

/-- Theorem stating that Mary paid $3 more than John -/
theorem mary_paid_three_more_than_john :
  ∃ (p : PizzaSharing),
    p.total_slices = 12 ∧
    p.base_price = 12 ∧
    p.cheese_price = 3 ∧
    p.mary_cheese_slices = 4 ∧
    p.mary_plain_slices = 3 ∧
    price_difference p = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_paid_three_more_than_john_l781_78131


namespace NUMINAMATH_CALUDE_expected_sophomores_in_sample_l781_78130

/-- Given a school with 1000 students, of which 320 are sophomores,
    the expected number of sophomores in a random sample of 200 students is 64. -/
theorem expected_sophomores_in_sample
  (total_students : ℕ)
  (sophomores : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 1000)
  (h2 : sophomores = 320)
  (h3 : sample_size = 200) :
  (sophomores : ℝ) / total_students * sample_size = 64 := by
  sorry

end NUMINAMATH_CALUDE_expected_sophomores_in_sample_l781_78130


namespace NUMINAMATH_CALUDE_isosceles_triangle_solution_l781_78173

/-- Represents a system of linear equations in two variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The main theorem -/
theorem isosceles_triangle_solution (a : ℝ) : 
  let system : LinearSystem := {
    eq1 := fun x y a => 3 * x - y - (2 * a - 5)
    eq2 := fun x y a => x + 2 * y - (3 * a + 3)
  }
  let x := a - 1
  let y := a + 2
  (x > 0 ∧ y > 0) →
  (∃ t : IsoscelesTriangle, t.leg = x ∧ t.base = y ∧ 2 * t.leg + t.base = 12) →
  system.eq1 x y a = 0 ∧ system.eq2 x y a = 0 →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_solution_l781_78173


namespace NUMINAMATH_CALUDE_smallest_winning_number_l781_78177

def B (x : ℕ) : ℕ := 3 * x

def S (x : ℕ) : ℕ := x + 100

def game_sequence (N : ℕ) : ℕ := B (S (B (S (B N))))

theorem smallest_winning_number :
  ∀ N : ℕ, 0 ≤ N ∧ N ≤ 1999 →
    (∀ M : ℕ, 0 ≤ M ∧ M < N → S (B (S (B M))) ≤ 2000) ∧
    2000 < game_sequence N ∧
    S (B (S (B N))) ≤ 2000 →
    N = 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l781_78177


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l781_78117

/-- Given a function f(x) = kx³ + 3(k-1)x² - k² + 1 where k > 0,
    and f(x) is decreasing in the interval (0,4),
    prove that k = 4. -/
theorem function_decreasing_interval (k : ℝ) (h1 : k > 0) : 
  (∀ x ∈ Set.Ioo 0 4, 
    (deriv (fun x => k*x^3 + 3*(k-1)*x^2 - k^2 + 1) x) < 0) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l781_78117


namespace NUMINAMATH_CALUDE_solution_set_inequality_l781_78142

theorem solution_set_inequality (x : ℝ) : (2 * x + 5) / (x - 2) < 1 ↔ -7 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l781_78142


namespace NUMINAMATH_CALUDE_fraction_problem_l781_78113

theorem fraction_problem : ∃ x : ℚ, x < 20 / 100 * 180 ∧ x * 180 = 24 := by
  use 2 / 15
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l781_78113


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_l781_78134

/-- The slope of the given line x - 2y + 3 = 0 -/
def m₁ : ℚ := 1 / 2

/-- The point P through which the perpendicular line passes -/
def P : ℚ × ℚ := (-1, 3)

/-- The equation of the perpendicular line in the form ax + by + c = 0 -/
def perpendicular_line (x y : ℚ) : Prop := 2 * x + y - 1 = 0

theorem perpendicular_line_correct :
  /- The line passes through point P -/
  perpendicular_line P.1 P.2 ∧
  /- The line is perpendicular to x - 2y + 3 = 0 -/
  (∃ m₂ : ℚ, m₂ * m₁ = -1 ∧
    ∀ x y : ℚ, perpendicular_line x y ↔ y - P.2 = m₂ * (x - P.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_l781_78134


namespace NUMINAMATH_CALUDE_cubic_root_from_quadratic_l781_78101

theorem cubic_root_from_quadratic : ∀ r : ℝ, 
  (r^2 = r + 2) → (r^3 = 3*r + 2) ∧ (3 * 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_from_quadratic_l781_78101


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l781_78188

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l781_78188


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l781_78146

open Complex

/-- A complex-valued function g defined as g(z) = (5 + 3i)z^3 + βz + δ -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (5 + 3*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| given the conditions -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ,
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = 0 →
  (∃ β₀ δ₀ : ℂ, ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-I)).im = 0 →
    Complex.abs β₀ + Complex.abs δ₀ ≤ Complex.abs β + Complex.abs δ) →
  ∃ β₀ δ₀ : ℂ, Complex.abs β₀ + Complex.abs δ₀ = Real.sqrt 73 :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l781_78146


namespace NUMINAMATH_CALUDE_g_increasing_intervals_l781_78124

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_increasing_intervals :
  ∃ (a b c : ℝ), a = -1 ∧ b = 0 ∧ c = 1 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g x ≤ g y) ∧
  (∀ x y, c ≤ x ∧ x < y → g x < g y) :=
sorry

end NUMINAMATH_CALUDE_g_increasing_intervals_l781_78124


namespace NUMINAMATH_CALUDE_f_minimum_at_two_l781_78155

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_minimum_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_two_l781_78155


namespace NUMINAMATH_CALUDE_tangent_inequality_parameter_range_l781_78175

open Real

theorem tangent_inequality_parameter_range 
  (a b : ℝ) 
  (h_a : a ∈ Set.Icc (-1 : ℝ) 2) :
  (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4), 
    tan x ^ 2 + 4 * (a + b) * tan x + a ^ 2 + b ^ 2 - 18 < 0) ↔ 
  b ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_inequality_parameter_range_l781_78175


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bound_one_half_is_greatest_bound_l781_78110

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_ratio_bound (t : Triangle) :
  (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 > 1 / 2 :=
sorry

theorem one_half_is_greatest_bound :
  ∀ ε > 0, ∃ t : Triangle, (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 < 1 / 2 + ε :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bound_one_half_is_greatest_bound_l781_78110


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l781_78167

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 * a 4 * a 6 * a 8 = 120 →
  1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7 / 60 →
  S a 9 = 63 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l781_78167


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l781_78144

theorem max_value_x_plus_y :
  ∃ (x y : ℝ),
    (2 * Real.sin x - 1) * (2 * Real.cos y - Real.sqrt 3) = 0 ∧
    x ∈ Set.Icc 0 (3 * Real.pi / 2) ∧
    y ∈ Set.Icc Real.pi (2 * Real.pi) ∧
    ∀ (x' y' : ℝ),
      (2 * Real.sin x' - 1) * (2 * Real.cos y' - Real.sqrt 3) = 0 →
      x' ∈ Set.Icc 0 (3 * Real.pi / 2) →
      y' ∈ Set.Icc Real.pi (2 * Real.pi) →
      x + y ≥ x' + y' ∧
    x + y = 8 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l781_78144


namespace NUMINAMATH_CALUDE_probability_theorem_l781_78119

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The probability of drawing two balls of different colors without replacement -/
def prob_different_without_replacement : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2 : ℚ)

/-- The probability of drawing two balls of different colors with replacement -/
def prob_different_with_replacement : ℚ :=
  2 * (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls

theorem probability_theorem :
  prob_different_without_replacement = 3/5 ∧
  prob_different_with_replacement = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l781_78119


namespace NUMINAMATH_CALUDE_rabbit_log_cutting_l781_78199

theorem rabbit_log_cutting (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  ∃ logs : ℕ, logs + cuts = pieces ∧ logs = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_log_cutting_l781_78199


namespace NUMINAMATH_CALUDE_soccer_enjoyment_misreporting_l781_78108

theorem soccer_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let dont_enjoy := 0.3 * total
  let say_dont_but_do := 0.25 * enjoy
  let say_dont_and_dont := 0.85 * dont_enjoy
  say_dont_but_do / (say_dont_but_do + say_dont_and_dont) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_soccer_enjoyment_misreporting_l781_78108


namespace NUMINAMATH_CALUDE_fliers_remaining_l781_78137

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 2500)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).floor - (afternoon_fraction * (total - (morning_fraction * total).floor)).floor = 1500 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l781_78137


namespace NUMINAMATH_CALUDE_dish_price_proof_l781_78198

/-- The original price of a dish satisfying the given conditions -/
def original_price : ℝ := 34

/-- The discount rate applied to the original price -/
def discount_rate : ℝ := 0.1

/-- The tip rate applied to either the original or discounted price -/
def tip_rate : ℝ := 0.15

/-- The difference in total payments between the two people -/
def payment_difference : ℝ := 0.51

theorem dish_price_proof :
  let discounted_price := original_price * (1 - discount_rate)
  let payment1 := discounted_price + original_price * tip_rate
  let payment2 := discounted_price + discounted_price * tip_rate
  payment1 - payment2 = payment_difference := by sorry

end NUMINAMATH_CALUDE_dish_price_proof_l781_78198


namespace NUMINAMATH_CALUDE_prob_all_even_four_dice_l781_78114

/-- The probability of a single standard six-sided die showing an even number -/
def prob_even_single : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- Theorem: The probability of all four standard six-sided dice showing even numbers
    when tossed simultaneously is 1/16 -/
theorem prob_all_even_four_dice :
  (prob_even_single ^ num_dice : ℚ) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_even_four_dice_l781_78114


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sum_l781_78176

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sum
  (a b : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_arith : arithmetic_sequence b)
  (h_eq : a 7 = b 7)
  (h_prod : a 3 * a 11 = 4 * a 7) :
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sum_l781_78176


namespace NUMINAMATH_CALUDE_factor_expression_l781_78190

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l781_78190


namespace NUMINAMATH_CALUDE_train_speed_l781_78145

/-- A train journey with two segments and a given average speed -/
structure TrainJourney where
  x : ℝ  -- distance of the first segment
  V : ℝ  -- speed of the train in the first segment
  avg_speed : ℝ  -- average speed for the entire journey

/-- The train journey satisfies the given conditions -/
def valid_journey (j : TrainJourney) : Prop :=
  j.x > 0 ∧ j.V > 0 ∧ j.avg_speed = 16 ∧
  (j.x / j.V + (2 * j.x) / 20) = (3 * j.x) / j.avg_speed

theorem train_speed (j : TrainJourney) (h : valid_journey j) : j.V = 40 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l781_78145


namespace NUMINAMATH_CALUDE_fraction_square_equals_twentyfive_l781_78157

theorem fraction_square_equals_twentyfive : (123456^2 : ℚ) / (24691^2 : ℚ) = 25 := by sorry

end NUMINAMATH_CALUDE_fraction_square_equals_twentyfive_l781_78157


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l781_78189

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1056 + y))) ∧ (23 ∣ (1056 + x)) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l781_78189


namespace NUMINAMATH_CALUDE_joggers_speed_ratio_l781_78141

theorem joggers_speed_ratio : 
  ∀ (v₁ v₂ : ℝ), v₁ > v₂ → v₁ > 0 → v₂ > 0 →
  (v₁ + v₂) * 2 = 12 →
  (v₁ - v₂) * 6 = 12 →
  v₁ / v₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_joggers_speed_ratio_l781_78141


namespace NUMINAMATH_CALUDE_inverse_inequality_l781_78112

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l781_78112


namespace NUMINAMATH_CALUDE_f_properties_l781_78160

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 4| - 3

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≠ -2) :
  (f a a > f a (-2)) ∧
  (∃ x y : ℝ, x < y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z ∈ Set.Ioo x y, f a z > 0) ↔
  a ∈ Set.Ioc (-5) (-7/2) ∪ Set.Ico (-1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l781_78160


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l781_78174

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 164) : 
  a * b = -50 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l781_78174


namespace NUMINAMATH_CALUDE_simplify_expression_l781_78182

theorem simplify_expression (x : ℝ) : (2 + x) * (1 - x) + (x + 2)^2 = 5 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l781_78182


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l781_78195

theorem quadratic_equation_roots (x : ℝ) : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x^2 - 4*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l781_78195


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l781_78105

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x + 3 < 3*x - 4 → x ≥ 4 ∧ 4 + 3 < 3*4 - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l781_78105


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l781_78196

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical balls into 3 distinct boxes,
    where box i must contain at least i balls (for i = 1, 2, 3) --/
def distributeWithMinimum (n : ℕ) : ℕ :=
  distribute (n - (1 + 2 + 3)) 3

theorem ball_distribution_problem :
  distributeWithMinimum 10 = 15 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l781_78196


namespace NUMINAMATH_CALUDE_inequality_implications_l781_78165

theorem inequality_implications (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) ∧
  ¬(∀ a b : ℝ, a > 0 ∧ 0 > b → 1/a < 1/b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l781_78165


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l781_78135

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) * (x + 1) ≥ 0}
def B : Set ℝ := {x | x < -4/5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l781_78135


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l781_78116

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the Triangle Angle Sum Theorem
axiom triangle_angle_sum (t : Triangle) : t.angle1 + t.angle2 + t.angle3 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : complementary t.angle1 t.angle2 ∨ complementary t.angle1 t.angle3 ∨ complementary t.angle2 t.angle3) : 
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l781_78116


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l781_78129

theorem sum_of_reciprocals (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 0) :
  1 / (b^3 + c^3 - a^3) + 1 / (a^3 + c^3 - b^3) + 1 / (a^3 + b^3 - c^3) = 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l781_78129


namespace NUMINAMATH_CALUDE_building_population_l781_78123

/-- Calculates the total number of people housed in a building -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem: A 25-story building with 4 apartments per floor and 2 people per apartment houses 200 people -/
theorem building_population : total_people 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_population_l781_78123


namespace NUMINAMATH_CALUDE_triangle_existence_l781_78133

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l781_78133


namespace NUMINAMATH_CALUDE_quadratic_root_range_l781_78138

theorem quadratic_root_range (k : ℝ) (α β : ℝ) : 
  (∃ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0) →
  (∀ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0 → x = α ∨ x = β) →
  0 < α → α < 1 → 1 < β → β < 2 →
  (3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l781_78138


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l781_78156

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z + 2)^5 = 64 * z^5 → Complex.abs (z + 2/15) = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l781_78156


namespace NUMINAMATH_CALUDE_inverse_function_sum_l781_78187

-- Define the function g and its inverse
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) : 
  (∀ x : ℝ, g c d (g_inv c d x) = x) ∧ 
  (∀ x : ℝ, g_inv c d (g c d x) = x) → 
  c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l781_78187


namespace NUMINAMATH_CALUDE_max_cylinder_surface_area_in_sphere_l781_78170

/-- The maximum surface area of a cylinder inscribed in a sphere -/
theorem max_cylinder_surface_area_in_sphere (R : ℝ) (h_pos : R > 0) :
  ∃ (r h : ℝ),
    r > 0 ∧ h > 0 ∧
    R^2 = r^2 + (h/2)^2 ∧
    ∀ (r' h' : ℝ),
      r' > 0 → h' > 0 → R^2 = r'^2 + (h'/2)^2 →
      2 * π * r * (h + r) ≤ 2 * π * r' * (h' + r') →
      2 * π * r * (h + r) = R^2 * π * (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_max_cylinder_surface_area_in_sphere_l781_78170


namespace NUMINAMATH_CALUDE_twentyfifth_triangular_number_l781_78180

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentyfifth_triangular_number :
  triangular_number 25 = 325 := by sorry

end NUMINAMATH_CALUDE_twentyfifth_triangular_number_l781_78180


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l781_78139

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y + 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (1, -2)

/-- First tangent line equation -/
def tangent1 (x y : ℝ) : Prop :=
  5*x - 12*y - 29 = 0

/-- Second tangent line equation -/
def tangent2 (x : ℝ) : Prop :=
  x = 1

/-- Theorem stating that the tangent lines from P to the circle have the given equations -/
theorem tangent_lines_to_circle :
  ∃ (x y : ℝ), circle_equation x y ∧
  ((tangent1 x y ∧ (x, y) ≠ P) ∨ (tangent2 x ∧ y ≠ -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l781_78139


namespace NUMINAMATH_CALUDE_chocolate_milk_students_l781_78140

theorem chocolate_milk_students (strawberry_milk : ℕ) (regular_milk : ℕ) (total_milk : ℕ) :
  strawberry_milk = 15 →
  regular_milk = 3 →
  total_milk = 20 →
  total_milk - (strawberry_milk + regular_milk) = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_students_l781_78140


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l781_78111

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt (3 * x + 7) - Real.sqrt (2 * x - 1) + 2 = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l781_78111


namespace NUMINAMATH_CALUDE_vector_dot_product_equality_l781_78153

/-- Given vectors a and b in ℝ², and a scalar t, prove that if the dot product of a and c
    is equal to the dot product of b and c, where c = a + t*b, then t = 13/2. -/
theorem vector_dot_product_equality (a b : ℝ × ℝ) (t : ℝ) :
  a = (5, 12) →
  b = (2, 0) →
  let c := a + t • b
  (a.1 * c.1 + a.2 * c.2) = (b.1 * c.1 + b.2 * c.2) →
  t = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equality_l781_78153


namespace NUMINAMATH_CALUDE_john_tax_difference_l781_78148

/-- Represents the tax rates and incomes before and after the change -/
structure TaxData where
  old_rate : ℝ
  new_rate : ℝ
  old_income : ℝ
  new_income : ℝ

/-- Calculates the difference in tax payments given the tax data -/
def tax_difference (data : TaxData) : ℝ :=
  data.new_rate * data.new_income - data.old_rate * data.old_income

/-- The specific tax data for John's situation -/
def john_tax_data : TaxData :=
  { old_rate := 0.20
    new_rate := 0.30
    old_income := 1000000
    new_income := 1500000 }

/-- Theorem stating that the difference in John's tax payments is $250,000 -/
theorem john_tax_difference :
  tax_difference john_tax_data = 250000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_difference_l781_78148


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l781_78149

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem f_satisfies_conditions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l781_78149


namespace NUMINAMATH_CALUDE_solution_set_and_roots_negative_at_two_implies_bound_l781_78127

def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

theorem solution_set_and_roots (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := by sorry

theorem negative_at_two_implies_bound (b : ℝ) :
  (∀ a, f a b 2 < 0) →
  b < -1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_and_roots_negative_at_two_implies_bound_l781_78127


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l781_78197

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of the given hyperbola is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (C : Hyperbola) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_axes = true)
  (h3 : C.asymptote_angle = π / 3) :
  eccentricity C = 2 ∨ eccentricity C = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l781_78197


namespace NUMINAMATH_CALUDE_valid_selections_count_l781_78169

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def selected_teachers : ℕ := 3

def all_selections : ℕ := (total_teachers.choose selected_teachers)
def all_male_selections : ℕ := (male_teachers.choose selected_teachers)
def all_female_selections : ℕ := (female_teachers.choose selected_teachers)

theorem valid_selections_count : 
  all_selections - (all_male_selections + all_female_selections) = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l781_78169


namespace NUMINAMATH_CALUDE_eventual_period_two_l781_78104

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, a k ≠ 0 ∧ a (k + 1) = ⌊a k⌋ * (a k - ⌊a k⌋)

theorem eventual_period_two (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ N : ℕ, ∀ k ≥ N, a k = a (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_eventual_period_two_l781_78104


namespace NUMINAMATH_CALUDE_bankers_discount_l781_78166

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℝ) (rate : ℝ) (time : ℝ) : 
  bankers_gain = 270 → rate = 12 → time = 3 → 
  ∃ (bankers_discount : ℝ), abs (bankers_discount - 421.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l781_78166


namespace NUMINAMATH_CALUDE_mean_homeruns_is_12_08_l781_78185

def total_hitters : ℕ := 12

def april_homeruns : List (ℕ × ℕ) := [(5, 4), (6, 4), (8, 2), (10, 1)]
def may_homeruns : List (ℕ × ℕ) := [(5, 2), (6, 2), (8, 3), (10, 2), (11, 1)]

def total_homeruns : ℕ := 
  (april_homeruns.map (λ p => p.1 * p.2)).sum + 
  (may_homeruns.map (λ p => p.1 * p.2)).sum

theorem mean_homeruns_is_12_08 : 
  (total_homeruns : ℚ) / total_hitters = 12.08 := by sorry

end NUMINAMATH_CALUDE_mean_homeruns_is_12_08_l781_78185


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l781_78152

theorem quadratic_function_uniqueness (a b c : ℝ) :
  (∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c) →
  (∀ x : ℝ, a * x^2 + b * x + c ≥ x) →
  (∀ x : ℝ, x > 0 ∧ x < 2 → a * x^2 + b * x + c ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c) →
  (∃ x : ℝ, a * x^2 + b * x + c = 0) →
  (∀ x : ℝ, a * x^2 + b * x + c = (1/4) * (x + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l781_78152


namespace NUMINAMATH_CALUDE_one_is_monomial_l781_78183

/-- A monomial is an algebraic expression with only one term. -/
def is_monomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ m : ℕ, expr m = if m = n then c else 0

/-- The constant function that always returns 1 -/
def const_one : ℕ → ℚ := λ _ => 1

theorem one_is_monomial : is_monomial const_one :=
sorry

end NUMINAMATH_CALUDE_one_is_monomial_l781_78183


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l781_78181

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 100 ↔ x = -10 ∨ x = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l781_78181


namespace NUMINAMATH_CALUDE_sum_of_cubes_negative_l781_78168

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_negative_l781_78168


namespace NUMINAMATH_CALUDE_a_ratio_l781_78102

def a (n : ℕ) : ℚ := 3 - 2^n

theorem a_ratio : a 2 / a 3 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_a_ratio_l781_78102


namespace NUMINAMATH_CALUDE_cookies_distribution_l781_78103

def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

theorem cookies_distribution :
  total_cookies / cookies_per_person = 14 :=
by sorry

end NUMINAMATH_CALUDE_cookies_distribution_l781_78103


namespace NUMINAMATH_CALUDE_max_distance_theorem_l781_78109

def vector_a : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def vector_b : ℝ × ℝ := (1, 2)

theorem max_distance_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = Real.sqrt 5 + 1 ∧
    ∀ (a : ℝ × ℝ), a = vector_a (x, y) →
      ‖a - vector_b‖ ≤ max_val ∧
      ∃ (a' : ℝ × ℝ), a' = vector_a (x', y') ∧ x'^2 + y'^2 = 1 ∧ ‖a' - vector_b‖ = max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l781_78109


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l781_78143

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 9 * π → (4 / 3) * π * r^3 = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l781_78143


namespace NUMINAMATH_CALUDE_task_completion_time_l781_78136

/-- Given workers A, B, and C with work rates a, b, and c respectively,
    if A and B together complete a task in 8 hours,
    A and C together complete it in 6 hours,
    and B and C together complete it in 4.8 hours,
    then A, B, and C working together will complete the task in 4 hours. -/
theorem task_completion_time (a b c : ℝ) 
  (hab : 8 * (a + b) = 1)
  (hac : 6 * (a + c) = 1)
  (hbc : 4.8 * (b + c) = 1) :
  (a + b + c)⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l781_78136


namespace NUMINAMATH_CALUDE_can_empty_table_l781_78186

/-- Represents a 2x2 table of natural numbers -/
def Table := Fin 2 → Fin 2 → ℕ

/-- Represents a move on the table -/
inductive Move
| RemoveRow (row : Fin 2) : Move
| DoubleColumn (col : Fin 2) : Move

/-- Applies a move to a table -/
def applyMove (t : Table) (m : Move) : Table :=
  match m with
  | Move.RemoveRow row => fun i j => if i = row ∧ t i j > 0 then t i j - 1 else t i j
  | Move.DoubleColumn col => fun i j => if j = col then 2 * t i j else t i j

/-- Checks if a table is empty (all cells are zero) -/
def isEmptyTable (t : Table) : Prop :=
  ∀ i j, t i j = 0

/-- The main theorem: any non-empty table can be emptied -/
theorem can_empty_table (t : Table) (h : ∀ i j, t i j > 0) :
  ∃ (moves : List Move), isEmptyTable (moves.foldl applyMove t) :=
sorry

end NUMINAMATH_CALUDE_can_empty_table_l781_78186


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l781_78147

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((6 * n + 15) / 6)) → n ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l781_78147


namespace NUMINAMATH_CALUDE_expression_value_at_three_l781_78161

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x ^ (x - 1)) = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l781_78161


namespace NUMINAMATH_CALUDE_count_five_digit_integers_l781_78122

/-- The number of different positive five-digit integers that can be formed using the digits 1, 1, 1, 2, and 2 -/
def num_five_digit_integers : ℕ := 10

/-- The multiset of digits used to form the integers -/
def digit_multiset : Multiset ℕ := {1, 1, 1, 2, 2}

/-- The theorem stating that the number of different positive five-digit integers
    that can be formed using the digits 1, 1, 1, 2, and 2 is equal to 10 -/
theorem count_five_digit_integers :
  (Multiset.card digit_multiset = 5) →
  (Multiset.card (Multiset.erase digit_multiset 1) = 2) →
  (Multiset.card (Multiset.erase digit_multiset 2) = 3) →
  num_five_digit_integers = 10 := by
  sorry


end NUMINAMATH_CALUDE_count_five_digit_integers_l781_78122


namespace NUMINAMATH_CALUDE_initial_retail_price_l781_78121

/-- Calculates the initial retail price of a machine given wholesale price, shipping, tax, discount, and profit margin. -/
theorem initial_retail_price
  (wholesale_with_shipping : ℝ)
  (shipping : ℝ)
  (tax_rate : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_with_shipping = 90)
  (h2 : shipping = 10)
  (h3 : tax_rate = 0.05)
  (h4 : discount_rate = 0.10)
  (h5 : profit_rate = 0.20) :
  let wholesale := (wholesale_with_shipping - shipping) / (1 + tax_rate)
  let cost := wholesale_with_shipping
  let initial_price := cost / (1 - profit_rate - discount_rate + discount_rate * profit_rate)
  initial_price = 125 := by sorry

end NUMINAMATH_CALUDE_initial_retail_price_l781_78121


namespace NUMINAMATH_CALUDE_remaining_squares_after_removal_l781_78171

/-- Represents the initial arrangement of matchsticks -/
structure Arrangement where
  matchsticks : ℕ
  squares : ℕ

/-- Represents a claim about the arrangement after removing matchsticks -/
inductive Claim
  | A : Claim  -- 5 squares of size 1x1 remain
  | B : Claim  -- 3 squares of size 2x2 remain
  | C : Claim  -- All 3x3 squares remain
  | D : Claim  -- Removed matchsticks are all on different lines
  | E : Claim  -- Four of the removed matchsticks are on the same line

/-- The main theorem to be proved -/
theorem remaining_squares_after_removal 
  (initial : Arrangement)
  (removed : ℕ)
  (incorrect_claims : Finset Claim)
  (h1 : initial.matchsticks = 40)
  (h2 : initial.squares = 30)
  (h3 : removed = 5)
  (h4 : incorrect_claims.card = 2)
  (h5 : Claim.A ∈ incorrect_claims)
  (h6 : Claim.D ∈ incorrect_claims)
  (h7 : Claim.E ∉ incorrect_claims)
  (h8 : Claim.B ∉ incorrect_claims)
  (h9 : Claim.C ∉ incorrect_claims) :
  ∃ (final : Arrangement), final.squares = 28 :=
sorry

end NUMINAMATH_CALUDE_remaining_squares_after_removal_l781_78171


namespace NUMINAMATH_CALUDE_problem_statement_l781_78154

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a^2 / (b - c) + b^2 / (c - a) + c^2 / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l781_78154


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l781_78164

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (num_boys : ℕ) 
  (num_playing_soccer : ℕ) 
  (num_girls_not_playing : ℕ) 
  (h1 : total_students = 500) 
  (h2 : num_boys = 350) 
  (h3 : num_playing_soccer = 250) 
  (h4 : num_girls_not_playing = 115) :
  (num_boys - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer * 100 = 86 := by
sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l781_78164


namespace NUMINAMATH_CALUDE_dress_design_count_l781_78172

/-- The number of fabric colors available -/
def num_colors : Nat := 5

/-- The number of patterns available -/
def num_patterns : Nat := 5

/-- The number of fabric materials available -/
def num_materials : Nat := 2

/-- A dress design consists of exactly one color, one pattern, and one material -/
structure DressDesign where
  color : Fin num_colors
  pattern : Fin num_patterns
  material : Fin num_materials

/-- The total number of possible dress designs -/
def total_designs : Nat := num_colors * num_patterns * num_materials

theorem dress_design_count : total_designs = 50 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l781_78172


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l781_78128

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 2 ∧ 
  ∀ x y : ℝ, x^2 - y^2 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧ 
      c^2 = a^2 + b^2 ∧ 
      e = c / a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l781_78128


namespace NUMINAMATH_CALUDE_jovanas_shells_l781_78126

theorem jovanas_shells (x : ℝ) : x + 15 + 17 = 37 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l781_78126


namespace NUMINAMATH_CALUDE_peanut_butter_recipe_l781_78107

/-- Peanut butter recipe proof -/
theorem peanut_butter_recipe (total_weight oil_to_peanut_ratio honey_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : oil_to_peanut_ratio = 3 / 12)
  (h3 : honey_weight = 2) :
  let peanut_weight := total_weight * (1 / (1 + oil_to_peanut_ratio + honey_weight / total_weight))
  let oil_weight := peanut_weight * oil_to_peanut_ratio
  oil_weight + honey_weight = 8 := by
sorry


end NUMINAMATH_CALUDE_peanut_butter_recipe_l781_78107


namespace NUMINAMATH_CALUDE_max_stores_visited_l781_78120

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (two_store_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : two_store_visitors = 8) 
  (h5 : two_store_visitors ≤ total_shoppers) 
  (h6 : 2 * two_store_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits ∧
  (total_visits = 2 * two_store_visitors + 
    (total_shoppers - two_store_visitors) + 
    (individual_visits - 1)) :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l781_78120


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l781_78179

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (second_day_speed : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) :
  total_time = 14 →
  second_day_speed = 4 →
  speed_difference = 0.5 →
  time_difference = 2 →
  ∃ (first_day_time second_day_time : ℝ),
    first_day_time + second_day_time = total_time ∧
    second_day_time = first_day_time - time_difference ∧
    ∃ (first_day_speed : ℝ),
      first_day_speed = second_day_speed - speed_difference ∧
      first_day_speed * first_day_time + second_day_speed * second_day_time = 52 :=
by sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l781_78179


namespace NUMINAMATH_CALUDE_rectangle_original_length_l781_78191

theorem rectangle_original_length :
  ∀ (original_length : ℝ),
    (original_length * 10 = 25 * 7.2) →
    original_length = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_original_length_l781_78191


namespace NUMINAMATH_CALUDE_certain_number_proof_l781_78178

def smallest_number : ℕ := 3153
def increase : ℕ := 3
def divisor1 : ℕ := 70
def divisor2 : ℕ := 25
def divisor3 : ℕ := 21

theorem certain_number_proof :
  ∃ (n : ℕ), n > 0 ∧
  (smallest_number + increase) % n = 0 ∧
  n % divisor1 = 0 ∧
  n % divisor2 = 0 ∧
  n % divisor3 = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (smallest_number + increase) % m = 0 →
    m % divisor1 = 0 →
    m % divisor2 = 0 →
    m % divisor3 = 0 →
    n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l781_78178


namespace NUMINAMATH_CALUDE_carters_additional_cakes_l781_78118

/-- The number of additional cakes Carter bakes in a week when tripling his usual production. -/
theorem carters_additional_cakes 
  (cheesecakes muffins red_velvet : ℕ) 
  (h1 : cheesecakes = 6)
  (h2 : muffins = 5)
  (h3 : red_velvet = 8) :
  3 * (cheesecakes + muffins + red_velvet) - (cheesecakes + muffins + red_velvet) = 38 :=
by sorry


end NUMINAMATH_CALUDE_carters_additional_cakes_l781_78118


namespace NUMINAMATH_CALUDE_davids_biology_marks_l781_78132

/-- Given David's marks in four subjects and his average marks across five subjects,
    proves that his marks in Biology are 90. -/
theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℚ)
  (h1 : english = 74)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 75.6)
  (h6 : average = (english + mathematics + physics + chemistry + biology) / 5) :
  biology = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_davids_biology_marks_l781_78132


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l781_78106

theorem sufficient_but_not_necessary : 
  (∀ x₁ x₂ : ℝ, x₁ > 3 ∧ x₂ > 3 → x₁ * x₂ > 9 ∧ x₁ + x₂ > 6) ∧
  (∃ x₁ x₂ : ℝ, x₁ * x₂ > 9 ∧ x₁ + x₂ > 6 ∧ ¬(x₁ > 3 ∧ x₂ > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l781_78106


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l781_78158

theorem subtraction_and_simplification :
  (9 : ℚ) / 23 - 5 / 69 = 22 / 69 ∧ 
  ∀ (a b : ℤ), (a : ℚ) / b = 22 / 69 → (a.gcd b = 1 → a = 22 ∧ b = 69) :=
by sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l781_78158


namespace NUMINAMATH_CALUDE_unique_base_nine_l781_78163

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_nine :
  ∃! b : Nat, b > 1 ∧ 
    to_decimal [1, 5, 2] b + to_decimal [1, 4, 3] b = to_decimal [3, 0, 5] b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_nine_l781_78163


namespace NUMINAMATH_CALUDE_alice_outfits_l781_78159

/-- The number of different outfits Alice can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Alice can create with her wardrobe -/
theorem alice_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_alice_outfits_l781_78159


namespace NUMINAMATH_CALUDE_bicycle_journey_l781_78193

theorem bicycle_journey (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  (5 * t₁ + 15 * t₂) / (t₁ + t₂) = 10 → t₂ / (t₁ + t₂) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_journey_l781_78193


namespace NUMINAMATH_CALUDE_kristin_reads_half_l781_78125

/-- The number of books Peter and Kristin need to read -/
def total_books : ℕ := 20

/-- Peter's reading speed in hours per book -/
def peter_speed : ℚ := 18

/-- The ratio of Kristin's reading speed to Peter's -/
def speed_ratio : ℚ := 3

/-- The time Kristin has to read in hours -/
def kristin_time : ℚ := 540

/-- The portion of books Kristin reads in the given time -/
def kristin_portion : ℚ := kristin_time / (peter_speed * speed_ratio * total_books)

theorem kristin_reads_half :
  kristin_portion = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_kristin_reads_half_l781_78125


namespace NUMINAMATH_CALUDE_probability_consecutive_points_l781_78162

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of points in the quadrilateral -/
def q : ℕ := 4

/-- The number of points in the triangle -/
def t : ℕ := 3

/-- The number of ways to select 3 consecutive points from n points on a circle -/
def consecutive_selections (n : ℕ) : ℕ := n

/-- The total number of ways to select 3 points from n points -/
def total_selections (n : ℕ) : ℕ := n.choose 3

/-- The probability of selecting 3 consecutive points out of 7 points on a circle,
    given that 4 points have already been selected to form a quadrilateral -/
theorem probability_consecutive_points : 
  (consecutive_selections n : ℚ) / (total_selections n : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_consecutive_points_l781_78162


namespace NUMINAMATH_CALUDE_range_of_f_l781_78100

def f (x : ℝ) : ℝ := x^2 - 3*x

def domain : Set ℝ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l781_78100


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l781_78192

/-- Given that -1, a, b, c, -4 form a geometric sequence, prove that a * b * c = -8 -/
theorem geometric_sequence_product (a b c : ℝ) 
  (h : ∃ (r : ℝ), a = -1 * r ∧ b = a * r ∧ c = b * r ∧ -4 = c * r) : 
  a * b * c = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l781_78192


namespace NUMINAMATH_CALUDE_smallest_value_l781_78184

theorem smallest_value : 
  54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l781_78184


namespace NUMINAMATH_CALUDE_correct_conclusions_l781_78194

theorem correct_conclusions :
  (∀ a b : ℝ, a + b < 0 ∧ b / a > 0 → |a + 2*b| = -a - 2*b) ∧
  (∀ m : ℚ, |m| + m ≥ 0) ∧
  (∀ a b c : ℝ, c < 0 ∧ 0 < a ∧ a < b → (a - b)*(b - c)*(c - a) > 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l781_78194
