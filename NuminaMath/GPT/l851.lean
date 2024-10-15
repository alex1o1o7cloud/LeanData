import Mathlib

namespace NUMINAMATH_GPT_area_of_overlap_l851_85104

def area_of_square_1 : ℝ := 1
def area_of_square_2 : ℝ := 4
def area_of_square_3 : ℝ := 9
def area_of_square_4 : ℝ := 16
def total_area_of_rectangle : ℝ := 27.5
def unshaded_area : ℝ := 1.5

def total_area_of_squares : ℝ := area_of_square_1 + area_of_square_2 + area_of_square_3 + area_of_square_4
def total_area_covered_by_squares : ℝ := total_area_of_rectangle - unshaded_area

theorem area_of_overlap :
  total_area_of_squares - total_area_covered_by_squares = 4 := 
sorry

end NUMINAMATH_GPT_area_of_overlap_l851_85104


namespace NUMINAMATH_GPT_max_positive_n_l851_85117

def a (n : ℕ) : ℤ := 19 - 2 * n

theorem max_positive_n (n : ℕ) (h : a n > 0) : n ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_max_positive_n_l851_85117


namespace NUMINAMATH_GPT_men_in_first_group_l851_85164

theorem men_in_first_group (M : ℕ) 
  (h1 : (M * 25 : ℝ) = (15 * 26.666666666666668 : ℝ)) : 
  M = 16 := 
by 
  sorry

end NUMINAMATH_GPT_men_in_first_group_l851_85164


namespace NUMINAMATH_GPT_div2_implies_div2_of_either_l851_85199

theorem div2_implies_div2_of_either (a b : ℕ) (h : 2 ∣ a * b) : (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end NUMINAMATH_GPT_div2_implies_div2_of_either_l851_85199


namespace NUMINAMATH_GPT_inequality_solution_l851_85146

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l851_85146


namespace NUMINAMATH_GPT_maximum_visibility_sum_l851_85116

theorem maximum_visibility_sum (X Y : ℕ) (h : X + 2 * Y = 30) :
  X * Y ≤ 112 :=
by
  sorry

end NUMINAMATH_GPT_maximum_visibility_sum_l851_85116


namespace NUMINAMATH_GPT_polynomial_not_product_of_single_var_l851_85182

theorem polynomial_not_product_of_single_var :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ), 
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = (x^200) * (y^200) + 1) := sorry

end NUMINAMATH_GPT_polynomial_not_product_of_single_var_l851_85182


namespace NUMINAMATH_GPT_base_conversion_l851_85162

theorem base_conversion {b : ℕ} (h : 5 * 6 + 2 = b * b + b + 1) : b = 5 :=
by
  -- Begin omitted steps to solve the proof
  sorry

end NUMINAMATH_GPT_base_conversion_l851_85162


namespace NUMINAMATH_GPT_find_largest_m_l851_85131

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end NUMINAMATH_GPT_find_largest_m_l851_85131


namespace NUMINAMATH_GPT_simplify_trig_expression_l851_85175

open Real

theorem simplify_trig_expression (α : ℝ) : 
  sin (2 * π - α)^2 + (cos (π + α) * cos (π - α)) + 1 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l851_85175


namespace NUMINAMATH_GPT_num_packages_l851_85166

-- Defining the given conditions
def packages_count_per_package := 6
def total_tshirts := 426

-- The statement to be proved
theorem num_packages : (total_tshirts / packages_count_per_package) = 71 :=
by sorry

end NUMINAMATH_GPT_num_packages_l851_85166


namespace NUMINAMATH_GPT_movement_left_3m_l851_85161

-- Define the condition
def movement_right_1m : ℝ := 1

-- Define the theorem stating that movement to the left by 3m should be denoted as -3
theorem movement_left_3m : movement_right_1m * (-3) = -3 :=
by
  sorry

end NUMINAMATH_GPT_movement_left_3m_l851_85161


namespace NUMINAMATH_GPT_modulus_of_complex_l851_85180

-- Define the conditions
variables {x y : ℝ}
def i := Complex.I

-- State the conditions of the problem
def condition1 : 1 + x * i = (2 - y) - 3 * i :=
by sorry

-- State the hypothesis and the goal
theorem modulus_of_complex (h : 1 + x * i = (2 - y) - 3 * i) : Complex.abs (x + y * i) = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_modulus_of_complex_l851_85180


namespace NUMINAMATH_GPT_unique_identity_function_l851_85125

theorem unique_identity_function (f : ℕ+ → ℕ+) :
  (∀ (x y : ℕ+), 
    let a := x 
    let b := f y 
    let c := f (y + f x - 1)
    a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x, f x = x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_unique_identity_function_l851_85125


namespace NUMINAMATH_GPT_hair_growth_l851_85115

theorem hair_growth (initial final : ℝ) (h_init : initial = 18) (h_final : final = 24) : final - initial = 6 :=
by
  sorry

end NUMINAMATH_GPT_hair_growth_l851_85115


namespace NUMINAMATH_GPT_one_plus_x_pow_gt_one_plus_nx_l851_85142

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end NUMINAMATH_GPT_one_plus_x_pow_gt_one_plus_nx_l851_85142


namespace NUMINAMATH_GPT_amount_spent_per_trip_l851_85159

def trips_per_month := 4
def months_per_year := 12
def initial_amount := 200
def final_amount := 104

def total_amount_spent := initial_amount - final_amount
def total_trips := trips_per_month * months_per_year

theorem amount_spent_per_trip :
  (total_amount_spent / total_trips) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_amount_spent_per_trip_l851_85159


namespace NUMINAMATH_GPT_ratio_G_to_C_is_1_1_l851_85111

variable (R C G : ℕ)

-- Given conditions
def Rover_has_46_spots : Prop := R = 46
def Cisco_has_half_R_minus_5 : Prop := C = R / 2 - 5
def Granger_Cisco_combined_108 : Prop := G + C = 108
def Granger_Cisco_equal : Prop := G = C

-- Theorem stating the final answer to the problem
theorem ratio_G_to_C_is_1_1 (h1 : Rover_has_46_spots R) 
                            (h2 : Cisco_has_half_R_minus_5 C R) 
                            (h3 : Granger_Cisco_combined_108 G C) 
                            (h4 : Granger_Cisco_equal G C) : 
                            G / C = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_G_to_C_is_1_1_l851_85111


namespace NUMINAMATH_GPT_pos_sum_inequality_l851_85173

theorem pos_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := 
sorry

end NUMINAMATH_GPT_pos_sum_inequality_l851_85173


namespace NUMINAMATH_GPT_glass_bottles_in_second_scenario_l851_85101

theorem glass_bottles_in_second_scenario
  (G P x : ℕ)
  (h1 : 3 * G = 600)
  (h2 : G = P + 150)
  (h3 : x * G + 5 * P = 1050) :
  x = 4 :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_glass_bottles_in_second_scenario_l851_85101


namespace NUMINAMATH_GPT_find_sum_invested_l851_85132

theorem find_sum_invested (P : ℝ) 
  (SI_1: ℝ) (SI_2: ℝ)
  (h1 : SI_1 = P * (15 / 100) * 2)
  (h2 : SI_2 = P * (12 / 100) * 2)
  (h3 : SI_1 - SI_2 = 900) :
  P = 15000 := by
sorry

end NUMINAMATH_GPT_find_sum_invested_l851_85132


namespace NUMINAMATH_GPT_total_length_proof_l851_85174

noncomputable def total_length_climbed (keaton_ladder_height : ℕ) (keaton_times : ℕ) (shortening : ℕ) (reece_times : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - shortening
  let keaton_total := keaton_ladder_height * keaton_times
  let reece_total := reece_ladder_height * reece_times
  (keaton_total + reece_total) * 100

theorem total_length_proof :
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end NUMINAMATH_GPT_total_length_proof_l851_85174


namespace NUMINAMATH_GPT_clock_angle_150_at_5pm_l851_85185

theorem clock_angle_150_at_5pm :
  (∀ t : ℕ, (t = 5) ↔ (∀ θ : ℝ, θ = 150 → θ = (30 * t))) := sorry

end NUMINAMATH_GPT_clock_angle_150_at_5pm_l851_85185


namespace NUMINAMATH_GPT_solution_inequality_l851_85126

theorem solution_inequality (m : ℝ) :
  (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ m ∈ Set.Icc 3 (-1) ∪ Set.Icc 6 7) →
  m = -1/2 ∨ m = 13/2 :=
sorry

end NUMINAMATH_GPT_solution_inequality_l851_85126


namespace NUMINAMATH_GPT_proof_problem_l851_85149

noncomputable def calc_a_star_b (a b : ℤ) : ℚ :=
1 / (a:ℚ) + 1 / (b:ℚ)

theorem proof_problem (a b : ℤ) (h1 : a + b = 10) (h2 : a * b = 24) :
  calc_a_star_b a b = 5 / 12 ∧ (a * b > a + b) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l851_85149


namespace NUMINAMATH_GPT_valid_m_values_l851_85198

theorem valid_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_valid_m_values_l851_85198


namespace NUMINAMATH_GPT_sine_sum_square_greater_l851_85195

variable {α β : Real} (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1)

theorem sine_sum_square_greater (α β : Real) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
  (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : 
  Real.sin (α + β) ^ 2 > Real.sin α ^ 2 + Real.sin β ^ 2 :=
sorry

end NUMINAMATH_GPT_sine_sum_square_greater_l851_85195


namespace NUMINAMATH_GPT_vacation_cost_division_l851_85147

theorem vacation_cost_division (total_cost : ℕ) (cost_per_person3 different_cost : ℤ) (n : ℕ)
  (h1 : total_cost = 375)
  (h2 : cost_per_person3 = total_cost / 3)
  (h3 : different_cost = cost_per_person3 - 50)
  (h4 : different_cost = total_cost / n) :
  n = 5 :=
  sorry

end NUMINAMATH_GPT_vacation_cost_division_l851_85147


namespace NUMINAMATH_GPT_side_length_of_square_l851_85191

theorem side_length_of_square 
  (x : ℝ) 
  (h₁ : 4 * x = 2 * (x * x)) :
  x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_side_length_of_square_l851_85191


namespace NUMINAMATH_GPT_smallest_non_six_digit_palindrome_l851_85156

-- Definition of a four-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

-- Definition of a six-digit number
def is_six_digit (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000

-- Definition of a non-palindrome
def not_palindrome (n : ℕ) : Prop :=
  ¬ is_palindrome n

-- Find the smallest four-digit palindrome whose product with 103 is not a six-digit palindrome
theorem smallest_non_six_digit_palindrome :
  ∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ is_palindrome n ∧ not_palindrome (103 * n)
  ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ is_palindrome m ∧ not_palindrome (103 * m) → n ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_non_six_digit_palindrome_l851_85156


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l851_85133

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (a > b - 1) ∧ ¬(a > b - 1 → a > b) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l851_85133


namespace NUMINAMATH_GPT_solve_for_k_l851_85179

theorem solve_for_k (t s k : ℝ) :
  (∀ t s : ℝ, (∃ t s : ℝ, (⟨1, 4⟩ : ℝ × ℝ) + t • ⟨5, -3⟩ = ⟨0, 1⟩ + s • ⟨-2, k⟩) → false) ↔ k = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l851_85179


namespace NUMINAMATH_GPT_extreme_points_range_l851_85190

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end NUMINAMATH_GPT_extreme_points_range_l851_85190


namespace NUMINAMATH_GPT_min_value_of_quadratic_l851_85124

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 12 * x + 35

theorem min_value_of_quadratic :
  ∀ x : ℝ, quadratic_function x ≥ quadratic_function 6 :=
by sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l851_85124


namespace NUMINAMATH_GPT_single_colony_habitat_limit_reach_time_l851_85103

noncomputable def doubling_time (n : ℕ) : ℕ := 2^n

theorem single_colony_habitat_limit_reach_time :
  ∀ (S : ℕ), ∀ (n : ℕ), doubling_time (n + 1) = S → doubling_time (2 * (n - 1)) = S → n + 1 = 16 :=
by
  intros S n H1 H2
  sorry

end NUMINAMATH_GPT_single_colony_habitat_limit_reach_time_l851_85103


namespace NUMINAMATH_GPT_partial_fraction_product_is_correct_l851_85130

-- Given conditions
def fraction_decomposition (x A B C : ℝ) :=
  ( (x^2 + 5 * x - 14) / (x^3 - 3 * x^2 - x + 3) = A / (x - 1) + B / (x - 3) + C / (x + 1) )

-- Statement we want to prove
theorem partial_fraction_product_is_correct (A B C : ℝ) (h : ∀ x : ℝ, fraction_decomposition x A B C) :
  A * B * C = -25 / 2 :=
sorry

end NUMINAMATH_GPT_partial_fraction_product_is_correct_l851_85130


namespace NUMINAMATH_GPT_calculate_expression_l851_85186

theorem calculate_expression : 1 + (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 2 - Real.sqrt 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l851_85186


namespace NUMINAMATH_GPT_total_games_High_School_Nine_l851_85178

-- Define the constants and assumptions.
def num_teams := 9
def games_against_non_league := 6

-- Calculation of the number of games within the league.
def games_within_league := (num_teams * (num_teams - 1) / 2) * 2

-- Calculation of the number of games against non-league teams.
def games_non_league := num_teams * games_against_non_league

-- The total number of games.
def total_games := games_within_league + games_non_league

-- The statement to prove.
theorem total_games_High_School_Nine : total_games = 126 := 
by
  -- You do not need to provide the proof.
  sorry

end NUMINAMATH_GPT_total_games_High_School_Nine_l851_85178


namespace NUMINAMATH_GPT_multiply_identity_l851_85154

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end NUMINAMATH_GPT_multiply_identity_l851_85154


namespace NUMINAMATH_GPT_system_of_equations_has_two_solutions_l851_85183

theorem system_of_equations_has_two_solutions :
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  xy + yz = 63 ∧ 
  xz + yz = 23 :=
sorry

end NUMINAMATH_GPT_system_of_equations_has_two_solutions_l851_85183


namespace NUMINAMATH_GPT_infinite_n_exists_l851_85106

theorem infinite_n_exists (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : 7 < p) :
  ∃ᶠ n in at_top, (n ≡ 1 [MOD 2016]) ∧ (p ∣ 2^n + n) :=
sorry

end NUMINAMATH_GPT_infinite_n_exists_l851_85106


namespace NUMINAMATH_GPT_MrSami_sold_20_shares_of_stock_x_l851_85108

theorem MrSami_sold_20_shares_of_stock_x
    (shares_v : ℕ := 68)
    (shares_w : ℕ := 112)
    (shares_x : ℕ := 56)
    (shares_y : ℕ := 94)
    (shares_z : ℕ := 45)
    (additional_shares_y : ℕ := 23)
    (increase_in_range : ℕ := 14)
    : (shares_x - (shares_y + additional_shares_y - ((shares_w - shares_z + increase_in_range) - shares_y - additional_shares_y)) = 20) :=
by
  sorry

end NUMINAMATH_GPT_MrSami_sold_20_shares_of_stock_x_l851_85108


namespace NUMINAMATH_GPT_consecutive_odd_product_l851_85167

theorem consecutive_odd_product (n : ℤ) :
  (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by sorry

end NUMINAMATH_GPT_consecutive_odd_product_l851_85167


namespace NUMINAMATH_GPT_angle_B_plus_angle_D_105_l851_85153

theorem angle_B_plus_angle_D_105
(angle_A : ℝ) (angle_AFG angle_AGF : ℝ)
(h1 : angle_A = 30)
(h2 : angle_AFG = angle_AGF)
: angle_B + angle_D = 105 := sorry

end NUMINAMATH_GPT_angle_B_plus_angle_D_105_l851_85153


namespace NUMINAMATH_GPT_sum_of_numbers_is_twenty_l851_85160

-- Given conditions
variables {a b c : ℝ}

-- Prove that the sum of a, b, and c is 20 given the conditions
theorem sum_of_numbers_is_twenty (h1 : a^2 + b^2 + c^2 = 138) (h2 : ab + bc + ca = 131) :
  a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_is_twenty_l851_85160


namespace NUMINAMATH_GPT_olivia_worked_hours_on_wednesday_l851_85100

-- Define the conditions
def hourly_rate := 9
def hours_monday := 4
def hours_friday := 6
def total_earnings := 117
def earnings_monday := hours_monday * hourly_rate
def earnings_friday := hours_friday * hourly_rate
def earnings_wednesday := total_earnings - (earnings_monday + earnings_friday)

-- Define the number of hours worked on Wednesday
def hours_wednesday := earnings_wednesday / hourly_rate

-- The theorem to prove
theorem olivia_worked_hours_on_wednesday : hours_wednesday = 3 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_olivia_worked_hours_on_wednesday_l851_85100


namespace NUMINAMATH_GPT_volume_in_barrel_l851_85127

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end NUMINAMATH_GPT_volume_in_barrel_l851_85127


namespace NUMINAMATH_GPT_men_wages_l851_85170

-- Conditions
variable (M W B : ℝ)
variable (h1 : 15 * M = W)
variable (h2 : W = 12 * B)
variable (h3 : 15 * M + W + B = 432)

-- Statement to prove
theorem men_wages : 15 * M = 144 :=
by
  sorry

end NUMINAMATH_GPT_men_wages_l851_85170


namespace NUMINAMATH_GPT_one_third_times_seven_times_nine_l851_85176

theorem one_third_times_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_GPT_one_third_times_seven_times_nine_l851_85176


namespace NUMINAMATH_GPT_graph_quadrant_l851_85158

theorem graph_quadrant (x y : ℝ) : 
  y = 3 * x - 4 → ¬ ((x < 0) ∧ (y > 0)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_graph_quadrant_l851_85158


namespace NUMINAMATH_GPT_total_stones_l851_85120

theorem total_stones (x : ℕ) 
  (h1 : x + 6 * x = x * 7 ∧ 7 * x + 6 * x = 2 * x) 
  (h2 : 2 * x = 7 * x - 10) 
  (h3 : 14 * x / 2 = 7 * x) :
  2 * 2 + 14 * 2 + 2 + 7 * 2 + 6 * 2 = 60 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_stones_l851_85120


namespace NUMINAMATH_GPT_radishes_per_row_l851_85188

theorem radishes_per_row 
  (bean_seedlings : ℕ) (beans_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (rows_per_bed : ℕ) (plant_beds : ℕ)
  (h1 : bean_seedlings = 64) (h2 : beans_per_row = 8)
  (h3 : pumpkin_seeds = 84) (h4 : pumpkins_per_row = 7)
  (h5 : radishes = 48) (h6 : rows_per_bed = 2) (h7 : plant_beds = 14) : 
  (radishes / ((plant_beds * rows_per_bed) - (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row))) = 6 := 
by sorry

end NUMINAMATH_GPT_radishes_per_row_l851_85188


namespace NUMINAMATH_GPT_find_a_l851_85122

theorem find_a (a : ℝ) (h : (2 - -3) / (1 - a) = Real.tan (135 * Real.pi / 180)) : a = 6 :=
sorry

end NUMINAMATH_GPT_find_a_l851_85122


namespace NUMINAMATH_GPT_stephanie_fewer_forks_l851_85128

noncomputable def fewer_forks := 
  (60 - 44) / 4

theorem stephanie_fewer_forks : fewer_forks = 4 := by
  sorry

end NUMINAMATH_GPT_stephanie_fewer_forks_l851_85128


namespace NUMINAMATH_GPT_probability_neither_orange_nor_white_l851_85184

/-- Define the problem conditions. -/
def num_orange_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 6

/-- Define the total number of balls. -/
def total_balls : ℕ := num_orange_balls + num_black_balls + num_white_balls

/-- Define the probability of picking a black ball (neither orange nor white). -/
noncomputable def probability_black_ball : ℚ := num_black_balls / total_balls

/-- The main statement to be proved: The probability is 1/3. -/
theorem probability_neither_orange_nor_white : probability_black_ball = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_neither_orange_nor_white_l851_85184


namespace NUMINAMATH_GPT_system_solution_xz_y2_l851_85157

theorem system_solution_xz_y2 (x y z : ℝ) (k : ℝ)
  (h : (x + 2 * k * y + 4 * z = 0) ∧
       (4 * x + k * y - 3 * z = 0) ∧
       (3 * x + 5 * y - 2 * z = 0) ∧
       x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k = 95 / 12) :
  (x * z) / (y ^ 2) = 10 :=
by sorry

end NUMINAMATH_GPT_system_solution_xz_y2_l851_85157


namespace NUMINAMATH_GPT_smallest_positive_period_and_monotonic_increase_max_min_in_interval_l851_85192

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem smallest_positive_period_and_monotonic_increase :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∃ a b : ℝ, (k * π - π / 3 ≤ a ∧ a ≤ x) ∧ (x ≤ b ∧ b ≤ k * π + π / 6) → f x = 1) := sorry

theorem max_min_in_interval :
  (∀ x : ℝ, (-π / 4 ≤ x ∧ x ≤ π / 6) → (1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3)) := sorry

end NUMINAMATH_GPT_smallest_positive_period_and_monotonic_increase_max_min_in_interval_l851_85192


namespace NUMINAMATH_GPT_solve_congruence_l851_85150

theorem solve_congruence :
  ∃ a m : ℕ, m ≥ 2 ∧ a < m ∧ a + m = 27 ∧ (10 * x + 3 ≡ 7 [MOD 15]) → x ≡ 12 [MOD 15] := 
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l851_85150


namespace NUMINAMATH_GPT_solve_for_x_l851_85112

theorem solve_for_x (x : ℝ) (h1 : x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 := sorry

end NUMINAMATH_GPT_solve_for_x_l851_85112


namespace NUMINAMATH_GPT_find_selling_price_functional_relationship_and_max_find_value_of_a_l851_85139

section StoreProduct

variable (x : ℕ) (y : ℕ) (a k b : ℝ)

-- Definitions for the given conditions
def cost_price : ℝ := 50
def selling_price := x 
def sales_quantity := y 
def future_cost_increase := a

-- Given points
def point1 : ℝ × ℕ := (55, 90) 
def point2 : ℝ × ℕ := (65, 70)

-- Linear relationship between selling price and sales quantity
def linearfunc := y = k * x + b

-- Proof of the first statement
theorem find_selling_price (k := -2) (b := 200) : 
    (profit = 800 → (x = 60 ∨ x = 90)) :=
by
  -- People prove the theorem here
  sorry

-- Proof for the functional relationship between W and x
theorem functional_relationship_and_max (x := 75) : 
    W = -2*x^2 + 300*x - 10000 ∧ W_max = 1250 :=
by
  -- People prove the theorem here
  sorry

-- Proof for the value of a when the cost price increases
theorem find_value_of_a (cost_increase := 4) : 
    (W'_max = 960 → a = 4) :=
by
  -- People prove the theorem here
  sorry

end StoreProduct

end NUMINAMATH_GPT_find_selling_price_functional_relationship_and_max_find_value_of_a_l851_85139


namespace NUMINAMATH_GPT_find_number_l851_85189

theorem find_number (x : ℝ) (h : 5 * 1.6 - (2 * 1.4) / x = 4) : x = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l851_85189


namespace NUMINAMATH_GPT_value_of_3a_minus_b_l851_85121
noncomputable def solveEquation : Type := sorry

theorem value_of_3a_minus_b (a b : ℝ) (h1 : a = 3 + Real.sqrt 15) (h2 : b = 3 - Real.sqrt 15) (h3 : a ≥ b) :
  3 * a - b = 6 + 4 * Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_value_of_3a_minus_b_l851_85121


namespace NUMINAMATH_GPT_commission_percentage_l851_85171

-- Define the conditions
def cost_of_item := 18.0
def observed_price := 27.0
def profit_percentage := 0.20
def desired_selling_price := cost_of_item + profit_percentage * cost_of_item
def commission_amount := observed_price - desired_selling_price

-- Prove the commission percentage taken by the online store
theorem commission_percentage : (commission_amount / desired_selling_price) * 100 = 25 :=
by
  -- Here the proof would normally be implemented
  sorry

end NUMINAMATH_GPT_commission_percentage_l851_85171


namespace NUMINAMATH_GPT_no_power_of_q_l851_85137

theorem no_power_of_q (n : ℕ) (hn : n > 0) (q : ℕ) (hq : Prime q) : ¬ (∃ k : ℕ, n^q + ((n-1)/2)^2 = q^k) := 
by
  sorry  -- proof steps are not required as per instructions

end NUMINAMATH_GPT_no_power_of_q_l851_85137


namespace NUMINAMATH_GPT_compound_interest_correct_l851_85113

-- define the problem conditions
def P : ℝ := 3000
def r : ℝ := 0.07
def n : ℕ := 25

-- the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- state the theorem we want to prove
theorem compound_interest_correct :
  compound_interest P r n = 16281 := 
by
  sorry

end NUMINAMATH_GPT_compound_interest_correct_l851_85113


namespace NUMINAMATH_GPT_longer_diagonal_rhombus_l851_85152

theorem longer_diagonal_rhombus (a b d1 d2 : ℝ) 
  (h1 : a = 35) 
  (h2 : d1 = 42) : 
  d2 = 56 := 
by 
  sorry

end NUMINAMATH_GPT_longer_diagonal_rhombus_l851_85152


namespace NUMINAMATH_GPT_arun_remaining_work_days_l851_85110

noncomputable def arun_and_tarun_work_in_days (W : ℝ) := 10
noncomputable def arun_alone_work_in_days (W : ℝ) := 60
noncomputable def arun_tarun_together_days := 4

theorem arun_remaining_work_days (W : ℝ) :
  (arun_and_tarun_work_in_days W = 10) ∧
  (arun_alone_work_in_days W = 60) ∧
  (let complete_work_days := arun_tarun_together_days;
  let remaining_work := W - (complete_work_days / arun_and_tarun_work_in_days W * W);
  let arun_remaining_days := (remaining_work / W) * arun_alone_work_in_days W;
  arun_remaining_days = 36) :=
sorry

end NUMINAMATH_GPT_arun_remaining_work_days_l851_85110


namespace NUMINAMATH_GPT_find_alpha_l851_85143

theorem find_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) 
  (l1 : ∀ x y : ℝ, x * Real.cos α - y - 1 = 0) 
  (l2 : ∀ x y : ℝ, x + y * Real.sin α + 1 = 0) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_alpha_l851_85143


namespace NUMINAMATH_GPT_cumulative_percentage_decrease_l851_85136

theorem cumulative_percentage_decrease :
  let original_price := 100
  let first_reduction := original_price * 0.85
  let second_reduction := first_reduction * 0.90
  let third_reduction := second_reduction * 0.95
  let fourth_reduction := third_reduction * 0.80
  let final_price := fourth_reduction
  (original_price - final_price) / original_price * 100 = 41.86 := by
  sorry

end NUMINAMATH_GPT_cumulative_percentage_decrease_l851_85136


namespace NUMINAMATH_GPT_solve_system_of_eqns_l851_85119

theorem solve_system_of_eqns :
  ∃ x y : ℝ, (x^2 + x * y + y = 1 ∧ y^2 + x * y + x = 5) ∧ ((x = -1 ∧ y = 3) ∨ (x = -1 ∧ y = -2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_eqns_l851_85119


namespace NUMINAMATH_GPT_malesWithCollegeDegreesOnly_l851_85172

-- Define the parameters given in the problem
def totalEmployees : ℕ := 180
def totalFemales : ℕ := 110
def employeesWithAdvancedDegrees : ℕ := 90
def employeesWithCollegeDegreesOnly : ℕ := totalEmployees - employeesWithAdvancedDegrees
def femalesWithAdvancedDegrees : ℕ := 55

-- Define the question as a theorem
theorem malesWithCollegeDegreesOnly : 
  totalEmployees = 180 →
  totalFemales = 110 →
  employeesWithAdvancedDegrees = 90 →
  employeesWithCollegeDegreesOnly = 90 →
  femalesWithAdvancedDegrees = 55 →
  ∃ (malesWithCollegeDegreesOnly : ℕ), 
    malesWithCollegeDegreesOnly = 35 := 
by
  intros
  sorry

end NUMINAMATH_GPT_malesWithCollegeDegreesOnly_l851_85172


namespace NUMINAMATH_GPT_no_solution_to_inequalities_l851_85102

theorem no_solution_to_inequalities :
  ∀ (x y z t : ℝ), 
    ¬ (|x| > |y - z + t| ∧
       |y| > |x - z + t| ∧
       |z| > |x - y + t| ∧
       |t| > |x - y + z|) :=
by
  intro x y z t
  sorry

end NUMINAMATH_GPT_no_solution_to_inequalities_l851_85102


namespace NUMINAMATH_GPT_total_pieces_in_10_row_triangle_l851_85168

open Nat

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_in_10_row_triangle : 
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  unit_rods + connectors = 231 :=
by
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  show unit_rods + connectors = 231
  sorry

end NUMINAMATH_GPT_total_pieces_in_10_row_triangle_l851_85168


namespace NUMINAMATH_GPT_gcd_m_n_is_one_l851_85177

-- Definitions of m and n
def m : ℕ := 101^2 + 203^2 + 307^2
def n : ℕ := 100^2 + 202^2 + 308^2

-- The main theorem stating the gcd of m and n
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_is_one_l851_85177


namespace NUMINAMATH_GPT_parabola_b_value_l851_85169

variable {q : ℝ}

theorem parabola_b_value (a b c : ℝ) (h_a : a = -3 / q)
  (h_eq : ∀ x : ℝ, (a * x^2 + b * x + c) = a * (x - q)^2 + q)
  (h_intercept : (a * 0^2 + b * 0 + c) = -2 * q)
  (h_q_nonzero : q ≠ 0) :
  b = 6 / q := 
sorry

end NUMINAMATH_GPT_parabola_b_value_l851_85169


namespace NUMINAMATH_GPT_x_finishes_in_nine_days_l851_85151

-- Definitions based on the conditions
def x_work_rate : ℚ := 1 / 24
def y_work_rate : ℚ := 1 / 16
def y_days_worked : ℚ := 10
def y_work_done : ℚ := y_work_rate * y_days_worked
def remaining_work : ℚ := 1 - y_work_done
def x_days_to_finish : ℚ := remaining_work / x_work_rate

-- Statement to be proven
theorem x_finishes_in_nine_days : x_days_to_finish = 9 := 
by
  -- Skipping actual proof steps as instructed
  sorry

end NUMINAMATH_GPT_x_finishes_in_nine_days_l851_85151


namespace NUMINAMATH_GPT_bricks_needed_l851_85118

noncomputable def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := length * width * height

theorem bricks_needed
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (hl : brick_length = 40)
  (hw : brick_width = 11.25)
  (hh : brick_height = 6)
  (wl : wall_length = 800)
  (wh : wall_height = 600)
  (wt : wall_thickness = 22.5) :
  (volume wall_length wall_height wall_thickness / volume brick_length brick_width brick_height) = 4000 := by
  sorry

end NUMINAMATH_GPT_bricks_needed_l851_85118


namespace NUMINAMATH_GPT_sum_of_areas_lt_side_length_square_l851_85145

variable (n : ℕ) (a : ℝ)
variable (S : Fin n → ℝ) (d : Fin n → ℝ)

-- Conditions
axiom areas_le_one : ∀ i, S i ≤ 1
axiom sum_d_le_a : (Finset.univ).sum d ≤ a
axiom areas_less_than_diameters : ∀ i, S i < d i

-- Theorem Statement
theorem sum_of_areas_lt_side_length_square :
  ((Finset.univ : Finset (Fin n)).sum S) < a :=
sorry

end NUMINAMATH_GPT_sum_of_areas_lt_side_length_square_l851_85145


namespace NUMINAMATH_GPT_MatthewSharedWithTwoFriends_l851_85165

theorem MatthewSharedWithTwoFriends
  (crackers : ℕ)
  (cakes : ℕ)
  (cakes_per_person : ℕ)
  (persons : ℕ)
  (H1 : crackers = 29)
  (H2 : cakes = 30)
  (H3 : cakes_per_person = 15)
  (H4 : persons * cakes_per_person = cakes) :
  persons = 2 := by
  sorry

end NUMINAMATH_GPT_MatthewSharedWithTwoFriends_l851_85165


namespace NUMINAMATH_GPT_circular_arc_sum_l851_85109

theorem circular_arc_sum (n : ℕ) (h₁ : n > 0) :
  ∀ s : ℕ, (1 ≤ s ∧ s ≤ (n * (n + 1)) / 2) →
  ∃ arc_sum : ℕ, arc_sum = s := 
by
  sorry

end NUMINAMATH_GPT_circular_arc_sum_l851_85109


namespace NUMINAMATH_GPT_num_ordered_pairs_c_d_l851_85123

def is_solution (c d x y : ℤ) : Prop :=
  c * x + d * y = 2 ∧ x^2 + y^2 = 65

theorem num_ordered_pairs_c_d : 
  ∃ (S : Finset (ℤ × ℤ)), S.card = 136 ∧ 
  ∀ (c d : ℤ), (c, d) ∈ S ↔ ∃ (x y : ℤ), is_solution c d x y :=
sorry

end NUMINAMATH_GPT_num_ordered_pairs_c_d_l851_85123


namespace NUMINAMATH_GPT_proof_completion_l851_85197

namespace MathProof

def p : ℕ := 10 * 7

def r : ℕ := p - 3

def q : ℚ := (3 / 5) * r

theorem proof_completion : q = 40.2 := by
  sorry

end MathProof

end NUMINAMATH_GPT_proof_completion_l851_85197


namespace NUMINAMATH_GPT_part1_part2_l851_85193

-- Define the function f
def f (x m : ℝ) : ℝ := abs (x + m) + abs (2 * x - 1)

-- First part of the problem
theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 := 
by sorry

-- Second part of the problem
theorem part2 (m : ℝ) : 
  (∀ x, 3 / 4 ≤ x → x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 := 
by sorry

end NUMINAMATH_GPT_part1_part2_l851_85193


namespace NUMINAMATH_GPT_marbles_problem_l851_85107

theorem marbles_problem (n : ℕ) :
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) → 
  n - 10 = 830 :=
sorry

end NUMINAMATH_GPT_marbles_problem_l851_85107


namespace NUMINAMATH_GPT_ratio_of_means_l851_85196

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_means_l851_85196


namespace NUMINAMATH_GPT_aimee_poll_l851_85105

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end NUMINAMATH_GPT_aimee_poll_l851_85105


namespace NUMINAMATH_GPT_fruit_fly_cell_division_l851_85155

/-- Genetic properties of fruit flies:
  1. Fruit flies have 2N = 8 chromosomes.
  2. Alleles A/a and B/b are inherited independently.
  3. Genotype AaBb is given.
  4. This genotype undergoes cell division without chromosomal variation.

Prove that:
Cells with a genetic composition of AAaaBBbb contain 8 or 16 chromosomes.
-/
theorem fruit_fly_cell_division (genotype : ℕ → ℕ) (A a B b : ℕ) :
  genotype 2 = 8 ∧
  (A + a + B + b = 8) ∧
  (genotype 0 = 2 * 4) →
  (genotype 1 = 8 ∨ genotype 1 = 16) :=
by
  sorry

end NUMINAMATH_GPT_fruit_fly_cell_division_l851_85155


namespace NUMINAMATH_GPT_petrol_price_l851_85144

theorem petrol_price (P : ℝ) (h : 0.9 * P = 0.9 * P) : (250 / (0.9 * P) - 250 / P = 5) → P = 5.56 :=
by
  sorry

end NUMINAMATH_GPT_petrol_price_l851_85144


namespace NUMINAMATH_GPT_line_through_point_parallel_to_given_line_l851_85140

theorem line_through_point_parallel_to_given_line 
  (x y : ℝ) 
  (h₁ : (x, y) = (1, -4)) 
  (h₂ : ∀ m : ℝ, 2 * 1 + 3 * (-4) + m = 0 → m = 10)
  : 2 * x + 3 * y + 10 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_point_parallel_to_given_line_l851_85140


namespace NUMINAMATH_GPT_product_of_two_numbers_ratio_l851_85194

theorem product_of_two_numbers_ratio (x y : ℝ)
  (h1 : x - y ≠ 0)
  (h2 : x + y = 4 * (x - y))
  (h3 : x * y = 18 * (x - y)) :
  x * y = 86.4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_ratio_l851_85194


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_square_l851_85163

theorem sufficient_but_not_necessary_condition_for_square (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ (¬(x^2 > 4 → x > 3)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_square_l851_85163


namespace NUMINAMATH_GPT_determine_C_plus_D_l851_85148

theorem determine_C_plus_D (A B C D : ℕ) 
  (hA : A ≠ 0) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D → 
  C + D = 5 :=
by
    sorry

end NUMINAMATH_GPT_determine_C_plus_D_l851_85148


namespace NUMINAMATH_GPT_probability_x_lt_2y_in_rectangle_l851_85114

-- Define the rectangle and the conditions
def in_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3

-- Define the condition x < 2y
def condition_x_lt_2y (x y : ℝ) : Prop :=
  x < 2 * y

-- Define the probability calculation
theorem probability_x_lt_2y_in_rectangle :
  let rectangle_area := (4:ℝ) * 3
  let triangle_area := (1:ℝ) / 2 * 4 * 2
  let probability := triangle_area / rectangle_area
  probability = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_x_lt_2y_in_rectangle_l851_85114


namespace NUMINAMATH_GPT_sequence_term_expression_l851_85129

theorem sequence_term_expression (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = 3^n + 1) :
  (a 1 = 4) ∧ (∀ n, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_expression_l851_85129


namespace NUMINAMATH_GPT_cannot_achieve_80_cents_l851_85135

def is_possible_value (n : ℕ) : Prop :=
  ∃ (n_nickels n_dimes n_quarters n_half_dollars : ℕ), 
    n_nickels + n_dimes + n_quarters + n_half_dollars = 5 ∧
    5 * n_nickels + 10 * n_dimes + 25 * n_quarters + 50 * n_half_dollars = n

theorem cannot_achieve_80_cents : ¬ is_possible_value 80 :=
by sorry

end NUMINAMATH_GPT_cannot_achieve_80_cents_l851_85135


namespace NUMINAMATH_GPT_sin_150_eq_half_l851_85141

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_150_eq_half_l851_85141


namespace NUMINAMATH_GPT_solution_l851_85181

-- Define the equations and their solution sets
def eq1 (x p : ℝ) : Prop := x^2 - p * x + 6 = 0
def eq2 (x q : ℝ) : Prop := x^2 + 6 * x - q = 0

-- Define the condition that the solution sets intersect at {2}
def intersect_at_2 (p q : ℝ) : Prop :=
  eq1 2 p ∧ eq2 2 q

-- The main theorem stating the value of p + q given the conditions
theorem solution (p q : ℝ) (h : intersect_at_2 p q) : p + q = 21 :=
by
  sorry

end NUMINAMATH_GPT_solution_l851_85181


namespace NUMINAMATH_GPT_Trishul_investment_percentage_l851_85187

-- Definitions from the conditions
def Vishal_invested (T : ℝ) : ℝ := 1.10 * T
def total_investment (T : ℝ) (V : ℝ) : ℝ := T + V + 2000

-- Problem statement
theorem Trishul_investment_percentage (T : ℝ) (V : ℝ) (H1 : V = Vishal_invested T) (H2 : total_investment T V = 5780) :
  ((2000 - T) / 2000) * 100 = 10 :=
sorry

end NUMINAMATH_GPT_Trishul_investment_percentage_l851_85187


namespace NUMINAMATH_GPT_determine_c_div_d_l851_85134

theorem determine_c_div_d (x y c d : ℝ) (h1 : 4 * x + 8 * y = c) (h2 : 5 * x - 10 * y = d) (h3 : d ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) : c / d = -4 / 5 :=
by
sorry

end NUMINAMATH_GPT_determine_c_div_d_l851_85134


namespace NUMINAMATH_GPT_find_values_of_ABC_l851_85138

-- Define the given conditions
def condition1 (A B C : ℕ) : Prop := A + B + C = 36
def condition2 (A B C : ℕ) : Prop := 
  (A + B) * 3 * 4 = (B + C) * 2 * 4 ∧ 
  (B + C) * 2 * 4 = (A + C) * 2 * 3

-- State the problem
theorem find_values_of_ABC (A B C : ℕ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  A = 12 ∧ B = 4 ∧ C = 20 :=
sorry

end NUMINAMATH_GPT_find_values_of_ABC_l851_85138
