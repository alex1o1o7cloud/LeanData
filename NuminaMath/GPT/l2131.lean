import Mathlib

namespace twelfth_term_of_arithmetic_sequence_l2131_213194

/-- Condition: a_1 = 1/2 -/
def a1 : ℚ := 1 / 2

/-- Condition: common difference d = 1/3 -/
def d : ℚ := 1 / 3

/-- Prove that the 12th term in the arithmetic sequence is 25/6 given the conditions. -/
theorem twelfth_term_of_arithmetic_sequence : a1 + 11 * d = 25 / 6 := by
  sorry

end twelfth_term_of_arithmetic_sequence_l2131_213194


namespace exponential_function_range_l2131_213163

noncomputable def exponential_function (a x : ℝ) : ℝ := a^x

theorem exponential_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : exponential_function a (-2) < exponential_function a (-3)) : 
  0 < a ∧ a < 1 :=
by
  sorry

end exponential_function_range_l2131_213163


namespace part1_solution_set_part2_range_of_a_l2131_213126

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l2131_213126


namespace gcd_lcm_mul_l2131_213115

theorem gcd_lcm_mul (a b : ℤ) : (Int.gcd a b) * (Int.lcm a b) = a * b := by
  sorry

end gcd_lcm_mul_l2131_213115


namespace simplify_expression_l2131_213141

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l2131_213141


namespace perimeter_of_park_is_66_l2131_213113

-- Given width and length of the flower bed
variables (w l : ℝ)
-- Given that the length is four times the width
variable (h1 : l = 4 * w)
-- Given the area of the flower bed
variable (h2 : l * w = 100)
-- Given the width of the walkway
variable (walkway_width : ℝ := 2)

-- The total width and length of the park, including the walkway
def w_park := w + 2 * walkway_width
def l_park := l + 2 * walkway_width

-- The proof statement: perimeter of the park equals 66 meters
theorem perimeter_of_park_is_66 :
  2 * (l_park + w_park) = 66 :=
by
  -- The full proof can be filled in here
  sorry

end perimeter_of_park_is_66_l2131_213113


namespace find_b_days_l2131_213197

theorem find_b_days 
  (a_days b_days c_days : ℕ)
  (a_wage b_wage c_wage : ℕ)
  (total_earnings : ℕ)
  (ratio_3_4_5 : a_wage * 5 = b_wage * 4 ∧ b_wage * 5 = c_wage * 4 ∧ a_wage * 5 = c_wage * 3)
  (c_wage_val : c_wage = 110)
  (a_days_val : a_days = 6)
  (c_days_val : c_days = 4) 
  (total_earnings_val : total_earnings = 1628)
  (earnings_eq : a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings) :
  b_days = 9 := by
  sorry

end find_b_days_l2131_213197


namespace brenda_cakes_l2131_213168

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l2131_213168


namespace wrench_force_inv_proportional_l2131_213121

theorem wrench_force_inv_proportional (F₁ : ℝ) (L₁ : ℝ) (F₂ : ℝ) (L₂ : ℝ) (k : ℝ)
  (h₁ : F₁ * L₁ = k) (h₂ : L₁ = 12) (h₃ : F₁ = 300) (h₄ : L₂ = 18) :
  F₂ = 200 :=
by
  sorry

end wrench_force_inv_proportional_l2131_213121


namespace sum_of_cubes_equality_l2131_213153

theorem sum_of_cubes_equality (a b p n : ℕ) (hp : Nat.Prime p) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^3 + b^3 = p^n) ↔ 
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end sum_of_cubes_equality_l2131_213153


namespace rains_at_least_once_l2131_213169

noncomputable def prob_rains_on_weekend : ℝ :=
  let prob_rain_saturday := 0.60
  let prob_rain_sunday := 0.70
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let independent_events := prob_no_rain_saturday * prob_no_rain_sunday
  1 - independent_events

theorem rains_at_least_once :
  prob_rains_on_weekend = 0.88 :=
by sorry

end rains_at_least_once_l2131_213169


namespace midpoint_of_hyperbola_l2131_213107

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l2131_213107


namespace claire_balloons_l2131_213179

def initial_balloons : ℕ := 50
def balloons_lost : ℕ := 12
def balloons_given_away : ℕ := 9
def balloons_received : ℕ := 11

theorem claire_balloons : initial_balloons - balloons_lost - balloons_given_away + balloons_received = 40 :=
by
  sorry

end claire_balloons_l2131_213179


namespace correct_sum_is_132_l2131_213135

-- Let's define the conditions:
-- The ones digit B is mistakenly taken as 1 (when it should be 7)
-- The tens digit C is mistakenly taken as 6 (when it should be 4)
-- The incorrect sum is 146

def correct_ones_digit (mistaken_ones_digit : Nat) : Nat :=
  -- B was mistaken for 1, so B should be 7
  if mistaken_ones_digit = 1 then 7 else mistaken_ones_digit

def correct_tens_digit (mistaken_tens_digit : Nat) : Nat :=
  -- C was mistaken for 6, so C should be 4
  if mistaken_tens_digit = 6 then 4 else mistaken_tens_digit

def correct_sum (incorrect_sum : Nat) : Nat :=
  -- Correcting the sum based on the mistakes
  incorrect_sum + 6 - 20 -- 6 to correct ones mistake, minus 20 to correct tens mistake

theorem correct_sum_is_132 : correct_sum 146 = 132 :=
  by
    -- The theorem is here to check that the corrected sum equals 132
    sorry

end correct_sum_is_132_l2131_213135


namespace students_not_invited_count_l2131_213191

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students not invited to the event
def not_invited_students : ℕ := 14

-- Define the sets representing different levels of friends of Anna
-- This demonstrates that the total invited students can be derived from given conditions

def anna_immediate_friends : ℕ := 4
def anna_second_level_friends : ℕ := (12 - anna_immediate_friends)
def anna_third_level_friends : ℕ := (16 - 12)

-- Define total invited students
def invited_students : ℕ := 
  anna_immediate_friends + 
  anna_second_level_friends +
  anna_third_level_friends

-- Prove that the number of not invited students is 14
theorem students_not_invited_count : (total_students - invited_students) = not_invited_students :=
by
  sorry

end students_not_invited_count_l2131_213191


namespace find_divisor_l2131_213192

theorem find_divisor (D : ℕ) : 
  let dividend := 109
  let quotient := 9
  let remainder := 1
  (dividend = D * quotient + remainder) → D = 12 :=
by
  sorry

end find_divisor_l2131_213192


namespace quadratic_root_3_m_value_l2131_213154

theorem quadratic_root_3_m_value (m : ℝ) : (∃ x : ℝ, 2*x*x - m*x + 3 = 0 ∧ x = 3) → m = 7 :=
by
  sorry

end quadratic_root_3_m_value_l2131_213154


namespace evaluate_expression_l2131_213125

theorem evaluate_expression : (4 - 3) * 2 = 2 := by
  sorry

end evaluate_expression_l2131_213125


namespace solve_system_of_equations_l2131_213146

def system_solution : Prop := ∃ x y : ℚ, 4 * x - 6 * y = -14 ∧ 8 * x + 3 * y = -15 ∧ x = -11 / 5 ∧ y = 2.6 / 3

theorem solve_system_of_equations : system_solution := sorry

end solve_system_of_equations_l2131_213146


namespace sin_is_odd_and_has_zero_point_l2131_213110

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem sin_is_odd_and_has_zero_point :
  is_odd_function sin ∧ has_zero_point sin := 
  by sorry

end sin_is_odd_and_has_zero_point_l2131_213110


namespace percentage_decrease_l2131_213184

variables (S : ℝ) (D : ℝ)
def initial_increase (S : ℝ) : ℝ := 1.5 * S
def final_gain (S : ℝ) : ℝ := 1.15 * S
def salary_after_decrease (S D : ℝ) : ℝ := (initial_increase S) * (1 - D)

theorem percentage_decrease :
  salary_after_decrease S D = final_gain S → D = 0.233333 :=
by
  sorry

end percentage_decrease_l2131_213184


namespace find_value_of_k_l2131_213127

theorem find_value_of_k (k : ℤ) : 
  (2 + 3 * k * -1/3 = -7 * 4) → k = 30 := 
by
  sorry

end find_value_of_k_l2131_213127


namespace calculate_expression_l2131_213116

theorem calculate_expression : ((-3: ℤ) ^ 3 + (5: ℤ) ^ 2 - ((-2: ℤ) ^ 2)) = -6 := by
  sorry

end calculate_expression_l2131_213116


namespace arithmetic_sequence_property_l2131_213195

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property
  (h1 : a 6 + a 8 = 10)
  (h2 : a 3 = 1)
  (property : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q)
  : a 11 = 9 :=
by
  sorry

end arithmetic_sequence_property_l2131_213195


namespace Marcus_ate_more_than_John_l2131_213147

theorem Marcus_ate_more_than_John:
  let John_eaten := 28
  let Marcus_eaten := 40
  Marcus_eaten - John_eaten = 12 :=
by
  sorry

end Marcus_ate_more_than_John_l2131_213147


namespace mary_needs_to_add_6_25_more_cups_l2131_213189

def total_flour_needed : ℚ := 8.5
def flour_already_added : ℚ := 2.25
def flour_to_add : ℚ := total_flour_needed - flour_already_added

theorem mary_needs_to_add_6_25_more_cups :
  flour_to_add = 6.25 :=
sorry

end mary_needs_to_add_6_25_more_cups_l2131_213189


namespace point_translation_proof_l2131_213180

def Point := (ℝ × ℝ)

def translate_right (p : Point) (d : ℝ) : Point := (p.1 + d, p.2)

theorem point_translation_proof :
  let A : Point := (1, 2)
  let A' := translate_right A 2
  A' = (3, 2) :=
by
  let A : Point := (1, 2)
  let A' := translate_right A 2
  show A' = (3, 2)
  sorry

end point_translation_proof_l2131_213180


namespace prime_sum_l2131_213132

theorem prime_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : 2 * p + 3 * q = 6 * r) : 
  p + q + r = 7 := 
sorry

end prime_sum_l2131_213132


namespace no_three_consecutive_geo_prog_l2131_213148

theorem no_three_consecutive_geo_prog (n k m: ℕ) (h: n ≠ k ∧ n ≠ m ∧ k ≠ m) :
  ¬(∃ a b c: ℕ, 
    (a = 2^n + 1 ∧ b = 2^k + 1 ∧ c = 2^m + 1) ∧ 
    (b^2 = a * c)) :=
by sorry

end no_three_consecutive_geo_prog_l2131_213148


namespace area_of_regular_octagon_l2131_213152

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end area_of_regular_octagon_l2131_213152


namespace similar_triangles_height_l2131_213151

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l2131_213151


namespace drink_costs_l2131_213103

theorem drink_costs (cost_of_steak_per_person : ℝ) (total_tip_paid : ℝ) (tip_percentage : ℝ) (billy_tip_coverage_percentage : ℝ) (total_tip_percentage : ℝ) :
  cost_of_steak_per_person = 20 → 
  total_tip_paid = 8 → 
  tip_percentage = 0.20 → 
  billy_tip_coverage_percentage = 0.80 → 
  total_tip_percentage = 0.20 → 
  ∃ (cost_of_drink : ℝ), cost_of_drink = 1.60 :=
by
  intros
  sorry

end drink_costs_l2131_213103


namespace apples_total_l2131_213105

theorem apples_total :
  ∀ (Marin David Amanda : ℕ),
  Marin = 6 →
  David = 2 * Marin →
  Amanda = David + 5 →
  Marin + David + Amanda = 35 :=
by
  intros Marin David Amanda hMarin hDavid hAmanda
  sorry

end apples_total_l2131_213105


namespace area_of_set_R_is_1006point5_l2131_213172

-- Define the set of points R as described in the problem
def isPointInSetR (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y ≤ 2013 ∧ ⌈x⌉ * ⌊y⌋ = ⌊x⌋ * ⌈y⌉

noncomputable def computeAreaOfSetR : ℝ :=
  1006.5

theorem area_of_set_R_is_1006point5 :
  (∃ x y : ℝ, isPointInSetR x y) → computeAreaOfSetR = 1006.5 := by
  sorry

end area_of_set_R_is_1006point5_l2131_213172


namespace rectangles_fit_l2131_213185

theorem rectangles_fit :
  let width := 50
  let height := 90
  let r_width := 1
  let r_height := (10 * Real.sqrt 2)
  ∃ n : ℕ, 
  n = 315 ∧
  (∃ w_cuts h_cuts : ℕ, 
    w_cuts = Int.floor (width / r_height) ∧
    h_cuts = Int.floor (height / r_height) ∧
    n = ((Int.floor (width / r_height) * Int.floor (height / r_height)) + 
         (Int.floor (height / r_width) * Int.floor (width / r_height)))) := 
sorry

end rectangles_fit_l2131_213185


namespace min_value_fraction_sum_l2131_213130

theorem min_value_fraction_sum : 
  ∀ (n : ℕ), n > 0 → (n / 3 + 27 / n) ≥ 6 :=
by
  sorry

end min_value_fraction_sum_l2131_213130


namespace perpendicular_line_through_circle_center_l2131_213149

theorem perpendicular_line_through_circle_center :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (y = m * x + b) → (x = -1 ∧ y = 0) ) ∧ m = 1 ∧ b = 1 ∧ (∀ (x y : ℝ), (y = x + 1) → (x - y + 1 = 0)) :=
sorry

end perpendicular_line_through_circle_center_l2131_213149


namespace range_of_a_l2131_213100

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x - a ^ 2 + 3

theorem range_of_a (h : ∀ x, x ≥ 0 → f x a - x ^ 2 ≥ 0) :
  -Real.sqrt 5 ≤ a ∧ a ≤ 3 - Real.log 3 := sorry

end range_of_a_l2131_213100


namespace overall_sale_price_per_kg_l2131_213137

-- Defining the quantities and prices
def tea_A_quantity : ℝ := 80
def tea_A_cost_per_kg : ℝ := 15
def tea_B_quantity : ℝ := 20
def tea_B_cost_per_kg : ℝ := 20
def tea_C_quantity : ℝ := 50
def tea_C_cost_per_kg : ℝ := 25
def tea_D_quantity : ℝ := 40
def tea_D_cost_per_kg : ℝ := 30

-- Defining the profit percentages
def tea_A_profit_percentage : ℝ := 0.30
def tea_B_profit_percentage : ℝ := 0.25
def tea_C_profit_percentage : ℝ := 0.20
def tea_D_profit_percentage : ℝ := 0.15

-- Desired sale price per kg
theorem overall_sale_price_per_kg : 
  (tea_A_quantity * tea_A_cost_per_kg * (1 + tea_A_profit_percentage) +
   tea_B_quantity * tea_B_cost_per_kg * (1 + tea_B_profit_percentage) +
   tea_C_quantity * tea_C_cost_per_kg * (1 + tea_C_profit_percentage) +
   tea_D_quantity * tea_D_cost_per_kg * (1 + tea_D_profit_percentage)) / 
  (tea_A_quantity + tea_B_quantity + tea_C_quantity + tea_D_quantity) = 26 := 
by
  sorry

end overall_sale_price_per_kg_l2131_213137


namespace solve_system_l2131_213129

variable {R : Type*} [CommRing R]

-- Given conditions
variables (a b c x y z : R)

-- Assuming the given system of equations
axiom eq1 : x + a*y + a^2*z + a^3 = 0
axiom eq2 : x + b*y + b^2*z + b^3 = 0
axiom eq3 : x + c*y + c^2*z + c^3 = 0

-- The goal is to prove the mathematical equivalence
theorem solve_system : x = -a*b*c ∧ y = a*b + b*c + c*a ∧ z = -(a + b + c) :=
by
  sorry

end solve_system_l2131_213129


namespace passes_through_origin_l2131_213119

def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2 * x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

theorem passes_through_origin : 
  (parabola_A 0 ≠ 0) ∧
  (parabola_B 0 ≠ 0) ∧
  (parabola_C 0 = 0) ∧
  (parabola_D 0 ≠ 0) := 
by 
  sorry

end passes_through_origin_l2131_213119


namespace find_x_plus_y_l2131_213159

theorem find_x_plus_y (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x - y > 0) : x + y = 8 ∨ x + y = 2 :=
by
  sorry

end find_x_plus_y_l2131_213159


namespace initial_people_count_l2131_213164

theorem initial_people_count (x : ℕ) (h : (x - 2) + 2 = 10) : x = 10 :=
by
  sorry

end initial_people_count_l2131_213164


namespace solution1_solution2_solution3_l2131_213112

noncomputable def problem1 : Real :=
3.5 * 101

noncomputable def problem2 : Real :=
11 * 5.9 - 5.9

noncomputable def problem3 : Real :=
88 - 17.5 - 12.5

theorem solution1 : problem1 = 353.5 :=
by
  sorry

theorem solution2 : problem2 = 59 :=
by
  sorry

theorem solution3 : problem3 = 58 :=
by
  sorry

end solution1_solution2_solution3_l2131_213112


namespace water_added_l2131_213157

-- Definitions and constants based on conditions
def initial_volume : ℝ := 80
def initial_jasmine_percentage : ℝ := 0.10
def jasmine_added : ℝ := 5
def final_jasmine_percentage : ℝ := 0.13

-- Problem statement
theorem water_added (W : ℝ) :
  (initial_volume * initial_jasmine_percentage + jasmine_added) / (initial_volume + jasmine_added + W) = final_jasmine_percentage → 
  W = 15 :=
by
  sorry

end water_added_l2131_213157


namespace find_p_q_d_l2131_213156

def f (p q d : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

theorem find_p_q_d :
  ∃ p q d : ℕ, f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 ∧ (p + q + d = 13) :=
by
  sorry

end find_p_q_d_l2131_213156


namespace surface_area_of_sphere_l2131_213196

-- Define the conditions from the problem.

variables (r R : ℝ) -- r is the radius of the cross-section, R is the radius of the sphere.
variables (π : ℝ := Real.pi) -- Define π using the real pi constant.
variables (h_dist : 1 = 1) -- Distance from the plane to the center is 1 unit.
variables (h_area_cross_section : π = π * r^2) -- Area of the cross-section is π.

-- State to prove the surface area of the sphere is 8π.
theorem surface_area_of_sphere :
    ∃ (R : ℝ), (R^2 = 2) → (4 * π * R^ 2 = 8 * π) := sorry

end surface_area_of_sphere_l2131_213196


namespace ab_minus_a_inv_b_l2131_213136

theorem ab_minus_a_inv_b (a : ℝ) (b : ℚ) (h1 : a > 1) (h2 : 0 < (b : ℝ)) (h3 : (a ^ (b : ℝ)) + (a ^ (-(b : ℝ))) = 2 * Real.sqrt 2) :
  (a ^ (b : ℝ)) - (a ^ (-(b : ℝ))) = 2 := 
sorry

end ab_minus_a_inv_b_l2131_213136


namespace cuboid_height_l2131_213171

-- Given conditions
def volume_cuboid : ℝ := 1380 -- cubic meters
def base_area_cuboid : ℝ := 115 -- square meters

-- Prove that the height of the cuboid is 12 meters
theorem cuboid_height : volume_cuboid / base_area_cuboid = 12 := by
  sorry

end cuboid_height_l2131_213171


namespace acute_angle_parallel_vectors_l2131_213109

theorem acute_angle_parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
    (h₁ : a = (Real.sin x, 1))
    (h₂ : b = (1 / 2, Real.cos x))
    (h₃ : ∃ k : ℝ, a = k • b ∧ k ≠ 0) :
    x = Real.pi / 4 :=
by
  sorry

end acute_angle_parallel_vectors_l2131_213109


namespace fraction_position_1991_1949_l2131_213188

theorem fraction_position_1991_1949 :
  ∃ (row position : ℕ), 
    ∀ (i j : ℕ), 
      (∃ k : ℕ, k = i + j - 1 ∧ k = 3939) ∧
      (∃ p : ℕ, p = j ∧ p = 1949) → 
      row = 3939 ∧ position = 1949 := 
sorry

end fraction_position_1991_1949_l2131_213188


namespace garrett_granola_bars_l2131_213117

theorem garrett_granola_bars :
  ∀ (oatmeal_raisin peanut total : ℕ),
  peanut = 8 →
  total = 14 →
  oatmeal_raisin + peanut = total →
  oatmeal_raisin = 6 :=
by
  intros oatmeal_raisin peanut total h_peanut h_total h_sum
  sorry

end garrett_granola_bars_l2131_213117


namespace prime_implies_n_eq_3k_l2131_213183

theorem prime_implies_n_eq_3k (n : ℕ) (p : ℕ) (k : ℕ) (h_pos : k > 0)
  (h_prime : Prime p) (h_eq : p = 1 + 2^n + 4^n) :
  ∃ k : ℕ, k > 0 ∧ n = 3^k :=
by
  sorry

end prime_implies_n_eq_3k_l2131_213183


namespace no_solution_exists_l2131_213158

def product_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else (x / 10) * (x % 10)

theorem no_solution_exists :
  ¬ ∃ x : ℕ, product_of_digits x = x^2 - 10 * x - 22 :=
by
  sorry

end no_solution_exists_l2131_213158


namespace average_of_r_s_t_l2131_213143

theorem average_of_r_s_t
  (r s t : ℝ)
  (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l2131_213143


namespace exist_elements_inequality_l2131_213170

open Set

theorem exist_elements_inequality (A : Set ℝ) (a_1 a_2 a_3 a_4 : ℝ)
(hA : A = {a_1, a_2, a_3, a_4})
(h_ineq1 : 0 < a_1 )
(h_ineq2 : a_1 < a_2 )
(h_ineq3 : a_2 < a_3 )
(h_ineq4 : a_3 < a_4 ) :
∃ (x y : ℝ), x ∈ A ∧ y ∈ A ∧ (2 + Real.sqrt 3) * |x - y| < (x + 1) * (y + 1) + x * y := 
sorry

end exist_elements_inequality_l2131_213170


namespace simplify_product_l2131_213190

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l2131_213190


namespace prove_partial_fractions_identity_l2131_213139

def partial_fraction_identity (x : ℚ) (A B C a b c : ℚ) : Prop :=
  a = 0 ∧ b = 1 ∧ c = -1 ∧
  (A / (x - a) + B / (x - b) + C / (x - c) = 4*x - 2 ∧ x^3 - x ≠ 0)

theorem prove_partial_fractions_identity :
  (partial_fraction_identity x 2 1 (-3) 0 1 (-1)) :=
by {
  sorry
}

end prove_partial_fractions_identity_l2131_213139


namespace find_side_PR_of_PQR_l2131_213199

open Real

noncomputable def triangle_PQR (PQ PM PH PR : ℝ) : Prop :=
  let HQ := sqrt (PQ^2 - PH^2)
  let MH := sqrt (PM^2 - PH^2)
  let MQ := MH - HQ
  let RH := HQ + 2 * MQ
  PR = sqrt (PH^2 + RH^2)

theorem find_side_PR_of_PQR (PQ PM PH : ℝ) (h_PQ : PQ = 3) (h_PM : PM = sqrt 14) (h_PH : PH = sqrt 5) (h_angle : ∀ QPR PRQ : ℝ, QPR + PRQ < 90) : 
  triangle_PQR PQ PM PH (sqrt 21) :=
by
  rw [h_PQ, h_PM, h_PH]
  exact sorry

end find_side_PR_of_PQR_l2131_213199


namespace men_in_first_group_l2131_213176

theorem men_in_first_group (M : ℕ) : (M * 18 = 27 * 24) → M = 36 :=
by
  sorry

end men_in_first_group_l2131_213176


namespace older_brother_age_is_25_l2131_213138

noncomputable def age_of_older_brother (father_age current_n : ℕ) (younger_brother_age : ℕ) : ℕ := 
  (father_age - current_n) / 2

theorem older_brother_age_is_25 
  (father_age : ℕ) 
  (h1 : father_age = 50) 
  (younger_brother_age : ℕ)
  (current_n : ℕ) 
  (h2 : (2 * (younger_brother_age + current_n)) = father_age + current_n) : 
  age_of_older_brother father_age current_n younger_brother_age = 25 := 
by
  sorry

end older_brother_age_is_25_l2131_213138


namespace domain_f_log_l2131_213134

noncomputable def domain_f (u : Real) : u ∈ Set.Icc (1 : Real) 2 := sorry

theorem domain_f_log (x : Real) : (x ∈ Set.Icc (4 : Real) 16) :=
by
  have h : ∀ x, (1 : Real) ≤ 2^x ∧ 2^x ≤ 2
  { intro x
    sorry }
  have h_log : ∀ x, 2 ≤ x ∧ x ≤ 4 
  { intro x
    sorry }
  have h_domain : ∀ x, 4 ≤ x ∧ x ≤ 16
  { intro x
    sorry }
  exact sorry

end domain_f_log_l2131_213134


namespace evlyn_can_buy_grapes_l2131_213181

theorem evlyn_can_buy_grapes 
  (price_pears price_oranges price_lemons price_grapes : ℕ)
  (h1 : 10 * price_pears = 5 * price_oranges)
  (h2 : 4 * price_oranges = 6 * price_lemons)
  (h3 : 3 * price_lemons = 2 * price_grapes) :
  (20 * price_pears = 10 * price_grapes) :=
by
  -- The proof is omitted using sorry
  sorry

end evlyn_can_buy_grapes_l2131_213181


namespace radio_price_position_l2131_213155

theorem radio_price_position (n : ℕ) (h₁ : n = 42)
  (h₂ : ∃ m : ℕ, m = 18 ∧ 
    (∀ k : ℕ, k < m → (∃ x : ℕ, x > k))) : 
    ∃ m : ℕ, m = 24 :=
by
  sorry

end radio_price_position_l2131_213155


namespace greatest_value_x_plus_y_l2131_213175

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 ∨ x + y = -6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_x_plus_y_l2131_213175


namespace tennis_tournament_boxes_needed_l2131_213187

theorem tennis_tournament_boxes_needed (n : ℕ) (h : n = 199) : 
  ∃ m, m = 198 ∧
    (∀ k, k < n → (n - k - 1 = m)) :=
by
  sorry

end tennis_tournament_boxes_needed_l2131_213187


namespace no_solution_inequalities_l2131_213108

theorem no_solution_inequalities (a : ℝ) : (¬ ∃ x : ℝ, 2 * x - 4 > 0 ∧ x - a < 0) → a ≤ 2 := 
by 
  sorry

end no_solution_inequalities_l2131_213108


namespace triangle_BC_length_l2131_213186

theorem triangle_BC_length (A B C X : Type) (AB AC BC BX CX : ℕ)
  (h1 : AB = 75)
  (h2 : AC = 85)
  (h3 : BC = BX + CX)
  (h4 : BX * (BX + CX) = 1600)
  (h5 : BX + CX = 80) :
  BC = 80 :=
by
  sorry

end triangle_BC_length_l2131_213186


namespace no_int_solutions_for_quadratics_l2131_213102

theorem no_int_solutions_for_quadratics :
  ¬ ∃ a b c : ℤ, (∃ x1 x2 : ℤ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
                (∃ y1 y2 : ℤ, (a + 1) * y1^2 + (b + 1) * y1 + (c + 1) = 0 ∧ 
                              (a + 1) * y2^2 + (b + 1) * y2 + (c + 1) = 0) :=
by
  sorry

end no_int_solutions_for_quadratics_l2131_213102


namespace clothing_price_reduction_l2131_213174

def price_reduction (original_profit_per_piece : ℕ) (original_sales_volume : ℕ) (target_profit : ℕ) (increase_in_sales_per_unit_price_reduction : ℕ) : ℕ :=
  sorry

theorem clothing_price_reduction :
  ∃ x : ℕ, (40 - x) * (20 + 2 * x) = 1200 :=
sorry

end clothing_price_reduction_l2131_213174


namespace letter_at_position_in_pattern_l2131_213131

/-- Determine the 150th letter in the repeating pattern XYZ is "Z"  -/
theorem letter_at_position_in_pattern :
  ∀ (pattern : List Char) (position : ℕ), pattern = ['X', 'Y', 'Z'] → position = 150 → pattern.get! ((position - 1) % pattern.length) = 'Z' :=
by
  intros pattern position
  intro hPattern hPosition
  rw [hPattern, hPosition]
  -- pattern = ['X', 'Y', 'Z'] and position = 150
  sorry

end letter_at_position_in_pattern_l2131_213131


namespace sam_memorized_digits_l2131_213162

theorem sam_memorized_digits (c s m : ℕ) 
  (h1 : s = c + 6) 
  (h2 : m = 6 * c)
  (h3 : m = 24) : 
  s = 10 :=
by
  sorry

end sam_memorized_digits_l2131_213162


namespace total_miles_walked_l2131_213198

-- Definition of the conditions
def num_islands : ℕ := 4
def miles_per_day_island1 : ℕ := 20
def miles_per_day_island2 : ℕ := 25
def days_per_island : ℚ := 1.5

-- Mathematically Equivalent Proof Problem
theorem total_miles_walked :
  let total_miles_island1 := 2 * (miles_per_day_island1 * days_per_island)
  let total_miles_island2 := 2 * (miles_per_day_island2 * days_per_island)
  total_miles_island1 + total_miles_island2 = 135 := by
  sorry

end total_miles_walked_l2131_213198


namespace sum_h_k_a_b_l2131_213145

def h : ℤ := 3
def k : ℤ := -5
def a : ℤ := 7
def b : ℤ := 4

theorem sum_h_k_a_b : h + k + a + b = 9 := by
  sorry

end sum_h_k_a_b_l2131_213145


namespace simplify_expr_l2131_213161

noncomputable def expr : ℝ := Real.sqrt 12 - 3 * Real.sqrt (1 / 3) + Real.sqrt 27 + (Real.pi + 1)^0

theorem simplify_expr : expr = 4 * Real.sqrt 3 + 1 := by
  sorry

end simplify_expr_l2131_213161


namespace remainder_abc_div9_l2131_213167

theorem remainder_abc_div9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) 
    (h1 : a + 2 * b + 3 * c ≡ 0 [MOD 9]) 
    (h2 : 2 * a + 3 * b + c ≡ 5 [MOD 9]) 
    (h3 : 3 * a + b + 2 * c ≡ 5 [MOD 9]) : 
    (a * b * c) % 9 = 0 := 
sorry

end remainder_abc_div9_l2131_213167


namespace compute_f_f_f_19_l2131_213178

def f (x : Int) : Int :=
  if x < 10 then x^2 - 9 else x - 15

theorem compute_f_f_f_19 : f (f (f 19)) = 40 := by
  sorry

end compute_f_f_f_19_l2131_213178


namespace smallest_b_in_AP_l2131_213177

theorem smallest_b_in_AP (a b c : ℝ) (d : ℝ) (ha : a = b - d) (hc : c = b + d) (habc : a * b * c = 125) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    b = 5 :=
by
  -- Proof needed here
  sorry

end smallest_b_in_AP_l2131_213177


namespace Grisha_owes_correct_l2131_213118

noncomputable def Grisha_owes (dish_cost : ℝ) : ℝ × ℝ :=
  let misha_paid := 3 * dish_cost
  let sasha_paid := 2 * dish_cost
  let friends_contribution := 50
  let equal_payment := 50 / 2
  (misha_paid - equal_payment, sasha_paid - equal_payment)

theorem Grisha_owes_correct :
  ∀ (dish_cost : ℝ), (dish_cost = 30) → Grisha_owes dish_cost = (40, 10) :=
by
  intro dish_cost h
  rw [h]
  unfold Grisha_owes
  simp
  sorry

end Grisha_owes_correct_l2131_213118


namespace log_prime_factor_inequality_l2131_213101

open Real

noncomputable def num_prime_factors (n : ℕ) : ℕ := sorry 

theorem log_prime_factor_inequality (n : ℕ) (h : 0 < n) : 
  log n ≥ num_prime_factors n * log 2 := 
sorry

end log_prime_factor_inequality_l2131_213101


namespace tan_theta_eq_neg_2sqrt2_to_expression_l2131_213166

theorem tan_theta_eq_neg_2sqrt2_to_expression (θ : ℝ) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2)) ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 1 :=
by
  sorry

end tan_theta_eq_neg_2sqrt2_to_expression_l2131_213166


namespace three_digit_2C4_not_multiple_of_5_l2131_213104

theorem three_digit_2C4_not_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬(∃ n : ℕ, 2 * 100 + C * 10 + 4 = 5 * n) :=
by
  sorry

end three_digit_2C4_not_multiple_of_5_l2131_213104


namespace BD_range_l2131_213122

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) (BD : ℤ) :=
  AB = 7 ∧ BC = 15 ∧ CD = 7 ∧ DA = 11 ∧ (9 ≤ BD ∧ BD ≤ 17)

theorem BD_range : 
  ∀ (AB BC CD DA : ℕ) (BD : ℤ),
  quadrilateral_BD AB BC CD DA BD → 
  9 ≤ BD ∧ BD ≤ 17 :=
by
  intros AB BC CD DA BD h
  cases h
  -- We would then prove the conditions
  sorry

end BD_range_l2131_213122


namespace unique_right_triangle_construction_l2131_213120

noncomputable def right_triangle_condition (c f : ℝ) : Prop :=
  f < c / 2

theorem unique_right_triangle_construction (c f : ℝ) (h_c : 0 < c) (h_f : 0 < f) :
  right_triangle_condition c f :=
  sorry

end unique_right_triangle_construction_l2131_213120


namespace max_three_kopecks_l2131_213193

def is_coin_placement_correct (n1 n2 n3 : ℕ) : Prop :=
  -- Conditions for the placement to be valid
  ∀ (i j : ℕ), i < j → 
  ((j - i > 1 → n1 = 0) ∧ (j - i > 2 → n2 = 0) ∧ (j - i > 3 → n3 = 0))

theorem max_three_kopecks (n1 n2 n3 : ℕ) (h : n1 + n2 + n3 = 101) (placement_correct : is_coin_placement_correct n1 n2 n3) :
  n3 = 25 ∨ n3 = 26 :=
sorry

end max_three_kopecks_l2131_213193


namespace scientific_notation_of_100000_l2131_213182

theorem scientific_notation_of_100000 :
  100000 = 1 * 10^5 :=
by sorry

end scientific_notation_of_100000_l2131_213182


namespace hot_sauce_container_size_l2131_213165

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end hot_sauce_container_size_l2131_213165


namespace value_of_first_equation_l2131_213106

theorem value_of_first_equation (x y : ℚ) 
  (h1 : 5 * x + 6 * y = 7) 
  (h2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 :=
sorry

end value_of_first_equation_l2131_213106


namespace problem1_problem2_problem3_l2131_213123

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 - 3 * x = 2) : 1 + 2 * x^2 - 6 * x = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x^2 - 3 * x - 4 = 0) : 1 + 3 * x - x^2 = -3 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) (p q : ℝ) (h1 : x = 1 → p * x^3 + q * x + 1 = 5) (h2 : p + q = 4) (hx : x = -1) : p * x^3 + q * x + 1 = -3 :=
by
  sorry

end problem1_problem2_problem3_l2131_213123


namespace quadratic_inequality_solution_l2131_213160

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l2131_213160


namespace parabola_opens_upward_l2131_213124

structure QuadraticFunction :=
  (a b c : ℝ)

def quadratic_y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

def points : List (ℝ × ℝ) :=
  [(-1, 10), (0, 5), (1, 2), (2, 1), (3, 2)]

theorem parabola_opens_upward (f : QuadraticFunction)
  (h_values : ∀ (x : ℝ), (x, quadratic_y f x) ∈ points) :
  f.a > 0 :=
sorry

end parabola_opens_upward_l2131_213124


namespace quarters_count_l2131_213150

noncomputable def num_coins := 12
noncomputable def total_value := 166 -- in cents
noncomputable def min_value := 1 + 5 + 10 + 25 + 50 -- minimum value from one of each type
noncomputable def remaining_value := total_value - min_value
noncomputable def remaining_coins := num_coins - 5

theorem quarters_count :
  ∀ (p n d q h : ℕ), 
  p + n + d + q + h = num_coins ∧
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ h ≥ 1 ∧
  (p + 5*n + 10*d + 25*q + 50*h = total_value) → 
  q = 3 := 
by 
  sorry

end quarters_count_l2131_213150


namespace local_minimum_f_eval_integral_part_f_l2131_213144

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sin x * Real.sqrt (1 - Real.cos x))

theorem local_minimum_f :
  (0 < x) -> (x < π) -> f x >= 1 :=
  by sorry

theorem eval_integral_part_f :
  ∫ x in (↑(π / 2))..(↑(2 * π / 3)), f x = sorry :=
  by sorry

end local_minimum_f_eval_integral_part_f_l2131_213144


namespace Irja_wins_probability_l2131_213173

noncomputable def probability_irja_wins : ℚ :=
  let X0 : ℚ := 4 / 7
  X0

theorem Irja_wins_probability :
  probability_irja_wins = 4 / 7 :=
sorry

end Irja_wins_probability_l2131_213173


namespace number_of_boys_l2131_213128

theorem number_of_boys (x g : ℕ) (h1 : x + g = 100) (h2 : g = x) : x = 50 := by
  sorry

end number_of_boys_l2131_213128


namespace diophantine_solution_unique_l2131_213133

theorem diophantine_solution_unique (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 = k * x * y - 1 ↔ k = 3 :=
by sorry

end diophantine_solution_unique_l2131_213133


namespace bill_bought_60_rats_l2131_213140

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l2131_213140


namespace non_drinkers_count_l2131_213142

-- Define the total number of businessmen and the sets of businessmen drinking each type of beverage.
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def soda_drinkers : ℕ := 8
def coffee_tea_drinkers : ℕ := 7
def tea_soda_drinkers : ℕ := 3
def coffee_soda_drinkers : ℕ := 2
def all_three_drinkers : ℕ := 1

-- Statement to prove:
theorem non_drinkers_count :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers - coffee_tea_drinkers - tea_soda_drinkers - coffee_soda_drinkers + all_three_drinkers) = 6 :=
by
  -- Skip the proof for now.
  sorry

end non_drinkers_count_l2131_213142


namespace candy_factory_days_l2131_213114

noncomputable def candies_per_hour := 50
noncomputable def total_candies := 4000
noncomputable def working_hours_per_day := 10
noncomputable def total_hours_needed := total_candies / candies_per_hour
noncomputable def total_days_needed := total_hours_needed / working_hours_per_day

theorem candy_factory_days :
  total_days_needed = 8 := 
by
  -- (Proof steps will be filled here)
  sorry

end candy_factory_days_l2131_213114


namespace joshua_additional_cents_needed_l2131_213111

def cost_of_pen_cents : ℕ := 600
def money_joshua_has_cents : ℕ := 500
def money_borrowed_cents : ℕ := 68

def additional_cents_needed (cost money has borrowed : ℕ) : ℕ :=
  cost - (has + borrowed)

theorem joshua_additional_cents_needed :
  additional_cents_needed cost_of_pen_cents money_joshua_has_cents money_borrowed_cents = 32 :=
by
  sorry

end joshua_additional_cents_needed_l2131_213111
