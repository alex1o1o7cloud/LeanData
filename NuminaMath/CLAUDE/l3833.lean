import Mathlib

namespace textbook_savings_l3833_383384

/-- Calculates the savings when buying textbooks from alternative bookshops instead of the school bookshop -/
theorem textbook_savings 
  (math_school_price : ℝ) 
  (science_school_price : ℝ) 
  (literature_school_price : ℝ)
  (math_discount : ℝ) 
  (science_discount : ℝ) 
  (literature_discount : ℝ)
  (school_tax_rate : ℝ)
  (alt_tax_rate : ℝ)
  (shipping_cost : ℝ)
  (h1 : math_school_price = 45)
  (h2 : science_school_price = 60)
  (h3 : literature_school_price = 35)
  (h4 : math_discount = 0.2)
  (h5 : science_discount = 0.25)
  (h6 : literature_discount = 0.15)
  (h7 : school_tax_rate = 0.07)
  (h8 : alt_tax_rate = 0.06)
  (h9 : shipping_cost = 10) :
  let school_total := (math_school_price + science_school_price + literature_school_price) * (1 + school_tax_rate)
  let alt_total := (math_school_price * (1 - math_discount) + 
                    science_school_price * (1 - science_discount) + 
                    literature_school_price * (1 - literature_discount)) * (1 + alt_tax_rate) + shipping_cost
  school_total - alt_total = 22.4 := by
  sorry


end textbook_savings_l3833_383384


namespace weeks_per_season_l3833_383306

def weekly_earnings : ℕ := 1357
def num_seasons : ℕ := 73
def total_earnings : ℕ := 22090603

theorem weeks_per_season : 
  (total_earnings / weekly_earnings) / num_seasons = 223 :=
sorry

end weeks_per_season_l3833_383306


namespace sum_product_squares_ratio_l3833_383310

theorem sum_product_squares_ratio (x y z a : ℝ) (h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h2 : x + y + z = a) (h3 : a ≠ 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = 1/3 := by
sorry

end sum_product_squares_ratio_l3833_383310


namespace cos_thirty_degrees_l3833_383379

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l3833_383379


namespace min_value_quadratic_l3833_383353

theorem min_value_quadratic (x : ℝ) : x^2 + 4*x + 5 ≥ 1 ∧ (x^2 + 4*x + 5 = 1 ↔ x = -2) := by
  sorry

end min_value_quadratic_l3833_383353


namespace expression_evaluation_l3833_383349

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end expression_evaluation_l3833_383349


namespace onions_removed_l3833_383311

/-- Proves that 5 onions were removed from the scale given the problem conditions -/
theorem onions_removed (total_onions : ℕ) (remaining_onions : ℕ) (total_weight : ℚ) 
  (avg_weight_remaining : ℚ) (avg_weight_removed : ℚ) :
  total_onions = 40 →
  remaining_onions = 35 →
  total_weight = 768/100 →
  avg_weight_remaining = 190/1000 →
  avg_weight_removed = 206/1000 →
  total_onions - remaining_onions = 5 := by
  sorry

#check onions_removed

end onions_removed_l3833_383311


namespace dividend_calculation_l3833_383390

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end dividend_calculation_l3833_383390


namespace not_prime_if_two_square_sums_l3833_383325

theorem not_prime_if_two_square_sums (p a b x y : ℤ) 
  (sum1 : p = a^2 + b^2) 
  (sum2 : p = x^2 + y^2) 
  (diff_repr : (a, b) ≠ (x, y) ∧ (a, b) ≠ (y, x)) : 
  ¬ Nat.Prime p.natAbs := by
  sorry

end not_prime_if_two_square_sums_l3833_383325


namespace b_value_l3833_383397

theorem b_value (a b c m : ℝ) (h : m = (c * a * b) / (a + b)) : 
  b = (m * a) / (c * a - m) :=
sorry

end b_value_l3833_383397


namespace water_heater_problem_l3833_383316

/-- Represents the capacity and current water level of a water heater -/
structure WaterHeater where
  capacity : ℚ
  fillRatio : ℚ

/-- Calculates the total amount of water in all water heaters -/
def totalWater (wallace catherine albert belinda : WaterHeater) : ℚ :=
  wallace.capacity * wallace.fillRatio +
  catherine.capacity * catherine.fillRatio +
  albert.capacity * albert.fillRatio - 5 +
  belinda.capacity * belinda.fillRatio

theorem water_heater_problem 
  (wallace catherine albert belinda : WaterHeater)
  (h1 : wallace.capacity = 2 * catherine.capacity)
  (h2 : albert.capacity = 3/2 * wallace.capacity)
  (h3 : wallace.capacity = 40)
  (h4 : wallace.fillRatio = 3/4)
  (h5 : albert.fillRatio = 2/3)
  (h6 : belinda.capacity = 1/2 * catherine.capacity)
  (h7 : belinda.fillRatio = 5/8)
  (h8 : catherine.fillRatio = 7/8) :
  totalWater wallace catherine albert belinda = 89 := by
  sorry


end water_heater_problem_l3833_383316


namespace max_min_kangaroo_weight_l3833_383382

theorem max_min_kangaroo_weight :
  ∀ (a b c : ℕ),
    a > 0 → b > 0 → c > 0 →
    a + b + c = 97 →
    min a (min b c) ≤ 32 ∧
    ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 97 ∧ min x (min y z) = 32 :=
by sorry

end max_min_kangaroo_weight_l3833_383382


namespace tan_fifteen_identity_l3833_383352

theorem tan_fifteen_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_identity_l3833_383352


namespace hockey_pads_cost_l3833_383308

def initial_amount : ℕ := 150
def remaining_amount : ℕ := 25

def cost_of_skates : ℕ := initial_amount / 2

def cost_of_pads : ℕ := initial_amount - cost_of_skates - remaining_amount

theorem hockey_pads_cost : cost_of_pads = 50 := by
  sorry

end hockey_pads_cost_l3833_383308


namespace optimal_price_and_range_l3833_383321

-- Define the linear relationship between quantity and price
def quantity (x : ℝ) : ℝ := -2 * x + 100

-- Define the cost per item
def cost : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * quantity x

-- Statement to prove
theorem optimal_price_and_range :
  -- The price that maximizes profit is 35
  (∃ (x_max : ℝ), x_max = 35 ∧ ∀ (x : ℝ), profit x ≤ profit x_max) ∧
  -- The range of prices that ensures at least 30 items sold and a profit of at least 400
  (∀ (x : ℝ), 30 ≤ x ∧ x ≤ 35 ↔ quantity x ≥ 30 ∧ profit x ≥ 400) :=
by sorry

end optimal_price_and_range_l3833_383321


namespace teacher_assignment_theorem_l3833_383367

/-- The number of ways to assign 4 teachers to 3 schools, with each school having at least 1 teacher -/
def teacher_assignment_count : ℕ := 36

/-- The number of teachers -/
def num_teachers : ℕ := 4

/-- The number of schools -/
def num_schools : ℕ := 3

theorem teacher_assignment_theorem :
  (∀ assignment : Fin num_teachers → Fin num_schools,
    (∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s) →
    (∃ s : Fin num_schools, ∃ t₁ t₂ : Fin num_teachers, t₁ ≠ t₂ ∧ assignment t₁ = s ∧ assignment t₂ = s)) →
  (Fintype.card {assignment : Fin num_teachers → Fin num_schools |
    ∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s}) = teacher_assignment_count :=
by sorry

#check teacher_assignment_theorem

end teacher_assignment_theorem_l3833_383367


namespace sum_x_y_is_three_sevenths_l3833_383302

theorem sum_x_y_is_three_sevenths (x y : ℚ) 
  (eq1 : 2 * x + y = 3)
  (eq2 : 3 * x - 2 * y = 12) : 
  x + y = 3 / 7 := by
  sorry

end sum_x_y_is_three_sevenths_l3833_383302


namespace tangent_line_sin_at_pi_l3833_383342

theorem tangent_line_sin_at_pi :
  let f (x : ℝ) := Real.sin x
  let x₀ : ℝ := Real.pi
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.cos x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - Real.pi = 0) := by sorry

end tangent_line_sin_at_pi_l3833_383342


namespace roots_sum_greater_than_a_l3833_383391

noncomputable section

-- Define the function f(x) = x ln x
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := Real.log x + 1

-- Define the function F(x) = x^2 - a[x + f'(x)] + 2x
def F (a : ℝ) (x : ℝ) : ℝ := x^2 - a * (x + f' x) + 2 * x

-- Theorem statement
theorem roots_sum_greater_than_a (a m x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : F a x₁ = m)
  (h₃ : F a x₂ = m)
  : x₁ + x₂ > a :=
sorry

end

end roots_sum_greater_than_a_l3833_383391


namespace class_average_mark_l3833_383313

theorem class_average_mark (n1 n2 : ℕ) (avg2 avg_total : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
    (h3 : avg2 = 90) (h4 : avg_total = 71.25) : 
  (n1 + n2 : ℝ) * avg_total = n1 * ((n1 + n2 : ℝ) * avg_total - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

end class_average_mark_l3833_383313


namespace barry_head_standing_duration_l3833_383360

def head_standing_duration (total_time minutes_between_turns num_turns : ℕ) : ℚ :=
  (total_time - minutes_between_turns * (num_turns - 1)) / num_turns

theorem barry_head_standing_duration :
  ∃ (x : ℕ), x ≥ 11 ∧ x < 12 ∧ head_standing_duration 120 5 8 < x :=
sorry

end barry_head_standing_duration_l3833_383360


namespace tangent_lines_to_circle_l3833_383335

/-- Circle equation: x^2 + y^2 - 4x - 6y - 3 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y - 3 = 0

/-- Point M(-2, 0) -/
def point_M : ℝ × ℝ := (-2, 0)

/-- First tangent line equation: x + 2 = 0 -/
def tangent_line1 (x y : ℝ) : Prop :=
  x + 2 = 0

/-- Second tangent line equation: 7x + 24y + 14 = 0 -/
def tangent_line2 (x y : ℝ) : Prop :=
  7*x + 24*y + 14 = 0

/-- Theorem stating that the given lines are tangent to the circle through point M -/
theorem tangent_lines_to_circle :
  (∀ x y, tangent_line1 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) ∧
  (∀ x y, tangent_line2 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) :=
sorry

end tangent_lines_to_circle_l3833_383335


namespace polynomial_has_three_real_roots_l3833_383300

def P (x : ℝ) : ℝ := x^5 + x^4 - x^3 - x^2 - 2*x - 2

theorem polynomial_has_three_real_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, P x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end polynomial_has_three_real_roots_l3833_383300


namespace infinite_series_sum_l3833_383320

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 1 / 3 := by sorry

end infinite_series_sum_l3833_383320


namespace derivative_limit_theorem_l3833_383329

open Real

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_limit_theorem (h : HasDerivAt f (-3) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ →
    |((f (x₀ + h) - f (x₀ - 3 * h)) / h) - (-12)| < ε :=
sorry

end derivative_limit_theorem_l3833_383329


namespace certain_number_proof_l3833_383354

theorem certain_number_proof (x : ℚ) : 
  (5 / 6 : ℚ) * x = (5 / 16 : ℚ) * x + 300 → x = 576 := by
  sorry

end certain_number_proof_l3833_383354


namespace climb_eight_steps_climb_ways_eq_fib_l3833_383304

/-- Fibonacci sequence starting with 1, 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Number of ways to climb n steps -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays n + climbWays (n + 1)

theorem climb_eight_steps : climbWays 8 = 34 := by
  sorry

theorem climb_ways_eq_fib (n : ℕ) : climbWays n = fib n := by
  sorry

end climb_eight_steps_climb_ways_eq_fib_l3833_383304


namespace sum_abc_l3833_383331

theorem sum_abc (a b c : ℝ) 
  (h1 : a - (b + c) = 16)
  (h2 : a^2 - (b + c)^2 = 1664) : 
  a + b + c = 104 := by
sorry

end sum_abc_l3833_383331


namespace triangle_inequality_l3833_383369

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (ABC : Triangle) (P : ℝ × ℝ) :
  let S := area ABC
  distance P ABC.A + distance P ABC.B + distance P ABC.C ≥ 2 * (3 ^ (1/4)) * Real.sqrt S := by
  sorry

end triangle_inequality_l3833_383369


namespace complement_union_M_N_l3833_383380

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {5} := by sorry

end complement_union_M_N_l3833_383380


namespace three_roots_range_l3833_383377

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem three_roots_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) →
  -2 < a ∧ a < 2 :=
sorry

end three_roots_range_l3833_383377


namespace smartphone_transactions_l3833_383336

def initial_price : ℝ := 300
def selling_price : ℝ := 255
def repurchase_price : ℝ := 275

theorem smartphone_transactions :
  (((initial_price - selling_price) / initial_price) * 100 = 15) ∧
  (((initial_price - repurchase_price) / repurchase_price) * 100 = 9.09) := by
sorry

end smartphone_transactions_l3833_383336


namespace incorrect_multiplication_result_l3833_383324

theorem incorrect_multiplication_result (x : ℕ) : 
  x * 153 = 109395 → x * 152 = 108680 := by
sorry

end incorrect_multiplication_result_l3833_383324


namespace original_average_age_of_class_l3833_383372

theorem original_average_age_of_class (A : ℝ) : 
  (12 : ℝ) * A + (12 : ℝ) * 32 = (24 : ℝ) * (A - 4) → A = 40 := by
  sorry

end original_average_age_of_class_l3833_383372


namespace k_range_l3833_383386

theorem k_range (x y k : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → Real.sqrt x + 3 * Real.sqrt y < k * Real.sqrt (x + y)) : 
  k > Real.sqrt 10 := by
  sorry

end k_range_l3833_383386


namespace area_between_curves_l3833_383338

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = 4*x
def curve2 (x y : ℝ) : Prop := x^2 = 4*y

-- Define the bounded area
def bounded_area (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ A ↔ (curve1 x y ∧ x ≥ 0 ∧ y ≥ 0) ∨ (curve2 x y ∧ x ≥ 0 ∧ y ≥ 0)

-- State the theorem
theorem area_between_curves :
  ∃ (A : Set (ℝ × ℝ)), bounded_area A ∧ MeasureTheory.volume A = 16/3 := by
  sorry

end area_between_curves_l3833_383338


namespace find_multiple_of_q_l3833_383323

theorem find_multiple_of_q (q : ℤ) (m : ℤ) : 
  let x := 55 + 2*q
  let y := m*q + 41
  (q = 7 → x = y) → m = 4 := by
sorry

end find_multiple_of_q_l3833_383323


namespace algebraic_simplification_l3833_383339

theorem algebraic_simplification (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x + 2) = -44 := by
  sorry

end algebraic_simplification_l3833_383339


namespace sport_formulation_water_amount_l3833_383315

/-- Prove that the sport formulation of a flavored drink contains 30 ounces of water -/
theorem sport_formulation_water_amount :
  -- Standard formulation ratio
  let standard_ratio : Fin 3 → ℚ := ![1, 12, 30]
  -- Sport formulation ratios relative to standard
  let sport_flavoring_corn_ratio := 3
  let sport_flavoring_water_ratio := 1 / 2
  -- Amount of corn syrup in sport formulation
  let sport_corn_syrup := 2

  -- The amount of water in the sport formulation
  ∃ (water : ℚ),
    -- Sport formulation flavoring to corn syrup ratio
    sport_flavoring_corn_ratio * standard_ratio 0 / standard_ratio 1 = sport_corn_syrup / 2 ∧
    -- Sport formulation flavoring to water ratio
    sport_flavoring_water_ratio * standard_ratio 0 / standard_ratio 2 = 2 / water ∧
    -- The amount of water is 30 ounces
    water = 30 :=
by
  sorry

end sport_formulation_water_amount_l3833_383315


namespace test_probabilities_l3833_383366

def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.6
def prob_C : ℝ := 0.5

theorem test_probabilities :
  let prob_all := prob_A * prob_B * prob_C
  let prob_none := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)
  let prob_at_least_one := 1 - prob_none
  prob_all = 0.24 ∧ prob_at_least_one = 0.96 := by
  sorry

end test_probabilities_l3833_383366


namespace solution_range_l3833_383395

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end solution_range_l3833_383395


namespace quadratic_inequality_bound_l3833_383385

theorem quadratic_inequality_bound (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x - 3) < d ↔ -5/2 < x ∧ x < 3) ↔ d = 39 := by
  sorry

end quadratic_inequality_bound_l3833_383385


namespace total_packs_is_35_l3833_383364

/-- The number of packs sold by Lucy -/
def lucy_packs : ℕ := 19

/-- The number of packs sold by Robyn -/
def robyn_packs : ℕ := 16

/-- The total number of packs sold by Robyn and Lucy -/
def total_packs : ℕ := lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold is 35 -/
theorem total_packs_is_35 : total_packs = 35 := by
  sorry

end total_packs_is_35_l3833_383364


namespace min_m_value_l3833_383344

theorem min_m_value (x y m : ℝ) 
  (hx : 2 ≤ x ∧ x ≤ 3) 
  (hy : 3 ≤ y ∧ y ≤ 6) 
  (h : ∀ x y, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x*y + y^2 ≥ 0) : 
  m ≥ 0 :=
sorry

end min_m_value_l3833_383344


namespace sequence_expression_l3833_383330

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^(n - 1)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
by sorry

end sequence_expression_l3833_383330


namespace zoe_calories_l3833_383389

/-- The number of calories Zoe ate from her snack -/
def total_calories (strawberry_count : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (calories_per_yogurt_ounce : ℕ) : ℕ :=
  strawberry_count * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

/-- Theorem stating that Zoe ate 150 calories -/
theorem zoe_calories : total_calories 12 6 4 17 = 150 := by
  sorry

end zoe_calories_l3833_383389


namespace givenPointInFirstQuadrant_l3833_383314

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end givenPointInFirstQuadrant_l3833_383314


namespace tournament_committee_count_l3833_383322

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the winning team -/
def winning_team_selection : ℕ := 3

/-- The number of members selected from each non-winning team -/
def other_team_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of possible tournament committees -/
def num_committees : ℕ := 4917248

theorem tournament_committee_count :
  num_committees = 
    num_teams * (Nat.choose team_size winning_team_selection) * 
    (Nat.choose team_size other_team_selection) ^ (num_teams - 1) := by
  sorry

end tournament_committee_count_l3833_383322


namespace sum_of_squares_175_l3833_383326

theorem sum_of_squares_175 (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a^2 + b^2 + c^2 + d^2 = 175 →
  a + b + c + d = 23 := by
sorry

end sum_of_squares_175_l3833_383326


namespace power_equality_l3833_383399

theorem power_equality (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end power_equality_l3833_383399


namespace sufficient_not_necessary_condition_l3833_383355

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a = -2 → |a| = 2) ∧ 
  (∃ a, |a| = 2 ∧ a ≠ -2) := by
  sorry

end sufficient_not_necessary_condition_l3833_383355


namespace inequality_solution_set_l3833_383333

theorem inequality_solution_set (a x : ℝ) :
  (a^2 - 4) * x^2 + 4 * x - 1 > 0 ↔
  (a = 2 ∨ a = -2 → x > 1/4) ∧
  (a > 2 → x > 1/(a + 2) ∨ x < 1/(2 - a)) ∧
  (a < -2 → x < 1/(a + 2) ∨ x > 1/(2 - a)) ∧
  (-2 < a ∧ a < 2 → 1/(a + 2) < x ∧ x < 1/(2 - a)) :=
by sorry

end inequality_solution_set_l3833_383333


namespace a_values_l3833_383375

def A : Set ℝ := {x | 2 * x^2 - 7 * x - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (h : B a ⊆ A) : a = 0 ∨ a = -2 ∨ a = 1/4 := by
  sorry

end a_values_l3833_383375


namespace puzzle_pieces_problem_l3833_383371

theorem puzzle_pieces_problem (pieces_first : ℕ) (pieces_second : ℕ) (pieces_third : ℕ) :
  pieces_second = pieces_third ∧
  pieces_second = (3 : ℕ) / 2 * pieces_first ∧
  pieces_first + pieces_second + pieces_third = 4000 →
  pieces_first = 1000 := by
  sorry

end puzzle_pieces_problem_l3833_383371


namespace jason_gave_four_cards_l3833_383318

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

theorem jason_gave_four_cards :
  let initial_cards : ℕ := 9
  let current_cards : ℕ := 5
  cards_given initial_cards current_cards = 4 := by
  sorry

end jason_gave_four_cards_l3833_383318


namespace unique_solution_l3833_383376

/-- Sum of digits function for positive integers in base 10 -/
def S (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 17 is the only positive integer solution to the equation -/
theorem unique_solution : ∀ n : ℕ+, (n : ℕ)^3 = 8 * (S n)^3 + 6 * (S n) * (n : ℕ) + 1 ↔ n = 17 := by
  sorry

end unique_solution_l3833_383376


namespace quadratic_equation_solution_l3833_383368

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = 2*d) →
  c = 1/2 ∧ d = -1/2 :=
by sorry

end quadratic_equation_solution_l3833_383368


namespace A_intersect_B_l3833_383387

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end A_intersect_B_l3833_383387


namespace fraction_sum_integer_l3833_383332

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end fraction_sum_integer_l3833_383332


namespace trapezoid_top_width_l3833_383392

/-- Proves that a trapezoid with given dimensions has a top width of 14 meters -/
theorem trapezoid_top_width :
  ∀ (area bottom_width height top_width : ℝ),
    area = 880 →
    bottom_width = 8 →
    height = 80 →
    area = (1 / 2) * (top_width + bottom_width) * height →
    top_width = 14 :=
by
  sorry

#check trapezoid_top_width

end trapezoid_top_width_l3833_383392


namespace perimeter_after_cut_l3833_383305

/-- The perimeter of the figure remaining after cutting a square corner from a larger square -/
def remaining_perimeter (original_side_length cut_side_length : ℝ) : ℝ :=
  2 * original_side_length + 3 * (original_side_length - cut_side_length)

/-- Theorem stating that the perimeter of the remaining figure is 17 -/
theorem perimeter_after_cut :
  remaining_perimeter 4 1 = 17 := by
  sorry

#eval remaining_perimeter 4 1

end perimeter_after_cut_l3833_383305


namespace sufficient_not_necessary_l3833_383328

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 3 → a^2 = 9) ∧ (∃ b : ℝ, b ≠ 3 ∧ b^2 = 9) := by sorry

end sufficient_not_necessary_l3833_383328


namespace raven_current_age_l3833_383373

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- The current ages of Raven and Phoebe -/
def raven_phoebe_ages : Person × Person → Prop
  | (raven, phoebe) => 
    -- In 5 years, Raven will be 4 times as old as Phoebe
    raven.age + 5 = 4 * (phoebe.age + 5) ∧
    -- Phoebe is currently 10 years old
    phoebe.age = 10

/-- Theorem stating Raven's current age -/
theorem raven_current_age : 
  ∀ (raven phoebe : Person), raven_phoebe_ages (raven, phoebe) → raven.age = 55 := by
  sorry

end raven_current_age_l3833_383373


namespace temp_at_six_km_l3833_383343

/-- The temperature drop per kilometer of altitude increase -/
def temp_drop_per_km : ℝ := 5

/-- The temperature at ground level in Celsius -/
def ground_temp : ℝ := 25

/-- The height in kilometers at which we want to calculate the temperature -/
def target_height : ℝ := 6

/-- Calculates the temperature at a given height -/
def temp_at_height (h : ℝ) : ℝ := ground_temp - temp_drop_per_km * h

/-- Theorem stating that the temperature at 6 km height is -5°C -/
theorem temp_at_six_km : temp_at_height target_height = -5 := by sorry

end temp_at_six_km_l3833_383343


namespace negative_143_coterminal_with_37_l3833_383327

/-- An angle is coterminal with 37° if it can be represented as 37° + 180°k, where k is an integer -/
def is_coterminal_with_37 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = 37 + 180 * k

/-- Theorem: -143° is coterminal with 37° -/
theorem negative_143_coterminal_with_37 : is_coterminal_with_37 (-143) := by
  sorry

end negative_143_coterminal_with_37_l3833_383327


namespace complement_A_in_U_l3833_383317

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end complement_A_in_U_l3833_383317


namespace son_age_problem_l3833_383357

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by
sorry

end son_age_problem_l3833_383357


namespace part_one_part_two_l3833_383388

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x + a|

-- Part 1
theorem part_one :
  {x : ℝ | f 3 x ≤ 1/2} = {x : ℝ | x ≥ -11/4} := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  ({x : ℝ | f a x ≤ a} = Set.univ) → a ≥ 1 := by sorry

end part_one_part_two_l3833_383388


namespace square_area_error_percentage_l3833_383358

theorem square_area_error_percentage (x : ℝ) (h : x > 0) :
  let measured_side := 1.12 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 25.44 := by sorry

end square_area_error_percentage_l3833_383358


namespace parabola_with_focus_at_origin_five_l3833_383347

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The vertex of the parabola -/
  vertex : ℝ × ℝ

/-- The equation of a parabola given its focus and vertex -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

theorem parabola_with_focus_at_origin_five : 
  let p : Parabola := { focus := (0, 5), vertex := (0, 0) }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 20*y :=
sorry

end parabola_with_focus_at_origin_five_l3833_383347


namespace sum_six_consecutive_even_integers_l3833_383301

/-- The sum of six consecutive even integers, starting from m, is equal to 6m + 30 -/
theorem sum_six_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 := by
  sorry

end sum_six_consecutive_even_integers_l3833_383301


namespace expression_simplification_l3833_383370

theorem expression_simplification (a : ℚ) (h : a = 3) :
  (((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4*a + 4) / (a^2 - a))) = 3/5 := by
  sorry

end expression_simplification_l3833_383370


namespace least_n_with_j_geq_10_remainder_M_mod_100_l3833_383393

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation -/
def j (n : ℕ) : ℕ := sorry

/-- The least value of n such that j(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_j_geq_10 : M = 14 := by sorry

theorem remainder_M_mod_100 : M % 100 = 14 := by sorry

end least_n_with_j_geq_10_remainder_M_mod_100_l3833_383393


namespace path_count_l3833_383374

/-- The number of paths from A to each blue arrow -/
def paths_to_blue : ℕ := 2

/-- The number of blue arrows -/
def num_blue_arrows : ℕ := 2

/-- The number of distinct ways from each blue arrow to each green arrow -/
def paths_blue_to_green : ℕ := 3

/-- The number of green arrows -/
def num_green_arrows : ℕ := 2

/-- The number of distinct final approaches from each green arrow to C -/
def paths_green_to_C : ℕ := 2

/-- The total number of paths from A to C -/
def total_paths : ℕ := 
  paths_to_blue * num_blue_arrows * 
  (paths_blue_to_green * num_blue_arrows) * num_green_arrows * 
  paths_green_to_C

theorem path_count : total_paths = 288 := by
  sorry

end path_count_l3833_383374


namespace billion_to_scientific_notation_l3833_383383

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^8
  let original_number : ℝ := 4947.66 * billion
  original_number = 4.94766 * 10^11 := by sorry

end billion_to_scientific_notation_l3833_383383


namespace solve_equation_l3833_383381

theorem solve_equation : ∃ x : ℝ, 60 + 5 * x / (180 / 3) = 61 ∧ x = 12 := by
  sorry

end solve_equation_l3833_383381


namespace largest_fraction_sum_l3833_383394

theorem largest_fraction_sum (a b c d : ℤ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 7) :
  (max ((a : ℚ) / b) ((a : ℚ) / c) + max ((b : ℚ) / a) ((b : ℚ) / c) + 
   max ((c : ℚ) / a) ((c : ℚ) / b) + max ((d : ℚ) / a) ((d : ℚ) / b)) ≤ 23 / 6 :=
by sorry

end largest_fraction_sum_l3833_383394


namespace line_slope_condition_l3833_383348

/-- Given a line passing through points (5, m) and (m, 8), prove that its slope is greater than 1
    if and only if m is in the open interval (5, 13/2). -/
theorem line_slope_condition (m : ℝ) :
  (8 - m) / (m - 5) > 1 ↔ 5 < m ∧ m < 13 / 2 := by
sorry

end line_slope_condition_l3833_383348


namespace circular_arrangement_l3833_383309

/-- 
Given a circular arrangement of n people numbered 1 to n,
if the distance from person 31 to person 7 is equal to 
the distance from person 31 to person 14, then n = 41.
-/
theorem circular_arrangement (n : ℕ) : 
  n ≥ 31 → 
  (min ((7 - 31 + n) % n) ((31 - 7) % n) = min ((14 - 31 + n) % n) ((31 - 14) % n)) → 
  n = 41 := by
  sorry

end circular_arrangement_l3833_383309


namespace interval_intersection_l3833_383351

theorem interval_intersection (x : ℝ) : 
  (3/4 < x ∧ x < 5/4) ↔ (2 < 3*x ∧ 3*x < 4) ∧ (3 < 4*x ∧ 4*x < 5) := by
  sorry

end interval_intersection_l3833_383351


namespace hyperbola_eccentricity_l3833_383361

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → e = Real.sqrt 5 / 2) :=
by sorry

end hyperbola_eccentricity_l3833_383361


namespace circle_radius_five_l3833_383378

/-- The value of c for which the circle x^2 + 8x + y^2 - 2y + c = 0 has radius 5 -/
theorem circle_radius_five (x y : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x^2 + 8*x + y^2 - 2*y + c = 0 ↔ (x + 4)^2 + (y - 1)^2 = 5^2) →
  (∃ c : ℝ, c = -8) :=
by sorry

end circle_radius_five_l3833_383378


namespace elias_bananas_l3833_383312

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elias ate -/
def eaten : ℕ := 1

/-- The number of bananas left after Elias ate some -/
def bananas_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem: If Elias bought a dozen bananas and ate 1, he has 11 left -/
theorem elias_bananas : bananas_left dozen eaten = 11 := by
  sorry

end elias_bananas_l3833_383312


namespace arthurs_hamburgers_l3833_383334

/-- Given the prices and quantities of hamburgers and hot dogs purchased over two days,
    prove that Arthur bought 2 hamburgers on the second day. -/
theorem arthurs_hamburgers (H D x : ℚ) : 
  3 * H + 4 * D = 10 →  -- Day 1 purchase
  x * H + 3 * D = 7 →   -- Day 2 purchase
  D = 1 →               -- Price of a hot dog
  x = 2 := by            
  sorry

end arthurs_hamburgers_l3833_383334


namespace function_graph_relationship_l3833_383350

theorem function_graph_relationship (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, x > 0 → Real.log x < a * x^2 - 1/2) → a > 1/2 := by
  sorry

end function_graph_relationship_l3833_383350


namespace triangle_exists_l3833_383337

/-- Represents a point in 2D space with integer coordinates -/
structure Point :=
  (x : Int) (y : Int)

/-- Calculates the square of the distance between two points -/
def distanceSquared (p q : Point) : Int :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : Rat :=
  let det := a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
  Rat.ofInt (abs det) / 2

/-- Theorem stating the existence of a triangle with the specified properties -/
theorem triangle_exists : ∃ (a b c : Point),
  (triangleArea a b c < 1) ∧
  (distanceSquared a b > 4) ∧
  (distanceSquared b c > 4) ∧
  (distanceSquared c a > 4) :=
sorry

end triangle_exists_l3833_383337


namespace fractional_equation_solution_range_l3833_383303

theorem fractional_equation_solution_range (m : ℝ) (x : ℝ) : 
  (m / (2 * x - 1) + 2 = 0) → (x > 0) → (m < 2 ∧ m ≠ 0) := by
  sorry

end fractional_equation_solution_range_l3833_383303


namespace unused_card_is_one_l3833_383356

def cards : Finset Nat := {1, 3, 4}

def largest_two_digit (a b : Nat) : Nat := 10 * max a b + min a b

def is_largest_two_digit (n : Nat) : Prop :=
  ∃ (a b : Nat), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧
  n = largest_two_digit a b ∧
  ∀ (x y : Nat), x ∈ cards → y ∈ cards → x ≠ y →
  largest_two_digit x y ≤ n

theorem unused_card_is_one :
  ∃ (n : Nat), is_largest_two_digit n ∧ (cards \ {n.div 10, n.mod 10}).toList = [1] := by
  sorry

end unused_card_is_one_l3833_383356


namespace inequality_system_solution_fractional_equation_no_solution_l3833_383362

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  (1 - x ≤ 2) ∧ ((x + 1) / 2 + (x - 1) / 3 < 1)

theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 1} :=
sorry

-- Part 2: Fractional equation
def fractional_equation (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4)

theorem fractional_equation_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end inequality_system_solution_fractional_equation_no_solution_l3833_383362


namespace power_sum_equality_l3833_383363

theorem power_sum_equality (a : ℕ) (h : 2^50 = a) :
  2^50 + 2^51 + 2^52 + 2^53 + 2^54 + 2^55 + 2^56 + 2^57 + 2^58 + 2^59 +
  2^60 + 2^61 + 2^62 + 2^63 + 2^64 + 2^65 + 2^66 + 2^67 + 2^68 + 2^69 +
  2^70 + 2^71 + 2^72 + 2^73 + 2^74 + 2^75 + 2^76 + 2^77 + 2^78 + 2^79 +
  2^80 + 2^81 + 2^82 + 2^83 + 2^84 + 2^85 + 2^86 + 2^87 + 2^88 + 2^89 +
  2^90 + 2^91 + 2^92 + 2^93 + 2^94 + 2^95 + 2^96 + 2^97 + 2^98 + 2^99 + 2^100 = 2*a^2 - a := by
  sorry

end power_sum_equality_l3833_383363


namespace pencil_distribution_theorem_l3833_383307

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) (max_pencils : ℕ) : ℕ := 
  sorry

/-- Theorem stating the number of ways to distribute 10 pencils among 4 friends -/
theorem pencil_distribution_theorem :
  distribute_pencils 10 4 1 5 = 64 := by sorry

end pencil_distribution_theorem_l3833_383307


namespace work_completion_time_proportional_aarti_work_completion_time_l3833_383365

/-- If a person can complete a piece of work in a given number of days,
    then the time to complete a multiple of that work is proportional to the multiple. -/
theorem work_completion_time_proportional
  (original_days : ℕ) (work_multiple : ℕ) :
  original_days * work_multiple = original_days * work_multiple :=
by sorry

/-- Aarti's work completion time -/
theorem aarti_work_completion_time :
  let original_days : ℕ := 6
  let work_multiple : ℕ := 3
  original_days * work_multiple = 18 :=
by sorry

end work_completion_time_proportional_aarti_work_completion_time_l3833_383365


namespace angle_y_value_l3833_383319

-- Define the angles in the problem
def angle_ACB : ℝ := 45
def angle_ABC : ℝ := 90
def angle_CDE : ℝ := 72

-- Define the theorem
theorem angle_y_value : 
  ∀ (angle_BAC angle_ADE angle_AED angle_DEB : ℝ),
  -- Triangle ABC
  angle_BAC + angle_ACB + angle_ABC = 180 →
  -- Angle ADC is a straight angle
  angle_ADE + angle_CDE = 180 →
  -- Triangle AED
  angle_AED + angle_ADE + angle_BAC = 180 →
  -- Angle AEB is a straight angle
  angle_AED + angle_DEB = 180 →
  -- Conclusion
  angle_DEB = 153 :=
by sorry

end angle_y_value_l3833_383319


namespace cube_volume_from_side_area_l3833_383340

theorem cube_volume_from_side_area (side_area : ℝ) (h : side_area = 64) :
  let side_length := Real.sqrt side_area
  side_length ^ 3 = 512 := by sorry

end cube_volume_from_side_area_l3833_383340


namespace cube_sum_and_reciprocal_l3833_383341

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end cube_sum_and_reciprocal_l3833_383341


namespace student_sister_weight_ratio_l3833_383345

/-- Proves that the ratio of a student's weight after losing 5 kg to his sister's weight is 2:1 -/
theorem student_sister_weight_ratio 
  (student_weight : ℕ) 
  (total_weight : ℕ) 
  (weight_loss : ℕ) :
  student_weight = 75 →
  total_weight = 110 →
  weight_loss = 5 →
  (student_weight - weight_loss) / (total_weight - student_weight) = 2 := by
  sorry

end student_sister_weight_ratio_l3833_383345


namespace chocolate_division_l3833_383346

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 24 / 7 := by
  sorry

end chocolate_division_l3833_383346


namespace signup_ways_4_3_l3833_383396

/-- The number of ways 4 students can sign up for one of 3 interest groups -/
def signup_ways (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem stating that the number of ways 4 students can sign up for one of 3 interest groups is 81 -/
theorem signup_ways_4_3 : signup_ways 4 3 = 81 := by
  sorry

end signup_ways_4_3_l3833_383396


namespace geometric_sequence_sixth_term_l3833_383359

theorem geometric_sequence_sixth_term
  (a : ℕ+) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 3)
  (h2 : a * r^3 = 243) -- fourth term condition
  : a * r^5 = 729 := by -- sixth term
sorry

end geometric_sequence_sixth_term_l3833_383359


namespace car_count_total_l3833_383398

/-- Given the car counting scenario, prove the total count of cars. -/
theorem car_count_total (jared_count : ℕ) (ann_count : ℕ) (alfred_count : ℕ) :
  jared_count = 300 →
  ann_count = (115 * jared_count) / 100 →
  alfred_count = ann_count - 7 →
  jared_count + ann_count + alfred_count = 983 :=
by sorry

end car_count_total_l3833_383398
