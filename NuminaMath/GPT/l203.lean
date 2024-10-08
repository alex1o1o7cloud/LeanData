import Mathlib

namespace gcd_2210_145_l203_203749

-- defining the constants a and b
def a : ℕ := 2210
def b : ℕ := 145

-- theorem stating that gcd(a, b) = 5
theorem gcd_2210_145 : Nat.gcd a b = 5 :=
sorry

end gcd_2210_145_l203_203749


namespace sphere_volume_of_hexagonal_prism_l203_203692

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_volume_of_hexagonal_prism
  (a h : ℝ)
  (volume : ℝ)
  (base_perimeter : ℝ)
  (vertices_on_sphere : ∀ (x y : ℝ) (hx : x^2 + y^2 = a^2) (hy : y = h / 2), x^2 + y^2 = 1) :
  volume = 9 / 8 ∧ base_perimeter = 3 →
  volume_of_sphere 1 = 4 * Real.pi / 3 :=
by
  sorry

end sphere_volume_of_hexagonal_prism_l203_203692


namespace factor_is_given_sum_l203_203421

theorem factor_is_given_sum (P Q : ℤ)
  (h1 : ∀ x : ℝ, (x^2 + 3 * x + 7) * (x^2 + (-3) * x + 7) = x^4 + P * x^2 + Q) :
  P + Q = 54 := 
sorry

end factor_is_given_sum_l203_203421


namespace vector_calculation_l203_203524

variables (a b : ℝ × ℝ)

def a_def : Prop := a = (3, 5)
def b_def : Prop := b = (-2, 1)

theorem vector_calculation (h1 : a_def a) (h2 : b_def b) : a - 2 • b = (7, 3) :=
sorry

end vector_calculation_l203_203524


namespace find_k_l203_203119

theorem find_k (k : ℕ) : (1/2)^18 * (1/81)^k = (1/18)^18 → k = 9 :=
by
  intro h
  sorry

end find_k_l203_203119


namespace checkerboard_disc_coverage_l203_203108

/-- A circular disc with a diameter of 5 units is placed on a 10 x 10 checkerboard with each square having a side length of 1 unit such that the centers of both the disc and the checkerboard coincide.
    Prove that the number of checkerboard squares that are completely covered by the disc is 36. -/
theorem checkerboard_disc_coverage :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let side_length : ℝ := 1
  let board_size : ℕ := 10
  let disc_center : ℝ × ℝ := (board_size / 2, board_size / 2)
  ∃ (count : ℕ), count = 36 := 
  sorry

end checkerboard_disc_coverage_l203_203108


namespace bret_spends_77_dollars_l203_203174

def num_people : ℕ := 4
def main_meal_cost : ℝ := 12.0
def num_appetizers : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0

def total_cost (num_people : ℕ) (main_meal_cost : ℝ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_order_fee : ℝ) : ℝ :=
  let main_meal_total := num_people * main_meal_cost
  let appetizer_total := num_appetizers * appetizer_cost
  let subtotal := main_meal_total + appetizer_total
  let tip := tip_rate * subtotal
  subtotal + tip + rush_order_fee

theorem bret_spends_77_dollars :
  total_cost num_people main_meal_cost num_appetizers appetizer_cost tip_rate rush_order_fee = 77.0 :=
by
  sorry

end bret_spends_77_dollars_l203_203174


namespace seven_b_value_l203_203977

theorem seven_b_value (a b : ℚ) (h₁ : 8 * a + 3 * b = 0) (h₂ : a = b - 3) :
  7 * b = 168 / 11 :=
sorry

end seven_b_value_l203_203977


namespace alfred_gain_percent_l203_203008

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end alfred_gain_percent_l203_203008


namespace katy_books_ratio_l203_203337

theorem katy_books_ratio (J : ℕ) (H1 : 8 + J + (J - 3) = 37) : J / 8 = 2 := 
by
  sorry

end katy_books_ratio_l203_203337


namespace matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l203_203988

def num_teams_group1 : ℕ := 3
def num_teams_group2 : ℕ := 4

def num_matches_round1_group1 (n : ℕ) : ℕ := n * (n - 1) / 2
def num_matches_round1_group2 (n : ℕ) : ℕ := n * (n - 1) / 2

def num_matches_round2 (n1 n2 : ℕ) : ℕ := n1 * n2

theorem matches_in_round1_group1 : num_matches_round1_group1 num_teams_group1 = 3 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round1_group2 : num_matches_round1_group2 num_teams_group2 = 6 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round2 : num_matches_round2 num_teams_group1 num_teams_group2 = 12 := 
by
  -- Exact proof steps should be filled in here.
  sorry

end matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l203_203988


namespace arithmetic_sequence_problem_l203_203502

noncomputable def arithmetic_sequence_sum : ℕ → ℕ := sorry  -- Define S_n here

theorem arithmetic_sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 8 - S 3 = 10)
    (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) (h3 : a 6 = 2) : S 11 = 22 :=
  sorry

end arithmetic_sequence_problem_l203_203502


namespace sin_cos_15_degrees_proof_l203_203531

noncomputable
def sin_cos_15_degrees : Prop := (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4)

theorem sin_cos_15_degrees_proof : sin_cos_15_degrees :=
by
  sorry

end sin_cos_15_degrees_proof_l203_203531


namespace numbers_divisible_by_three_l203_203656

theorem numbers_divisible_by_three (a b : ℕ) (h1 : a = 150) (h2 : b = 450) :
  ∃ n : ℕ, ∀ x : ℕ, (a < x) → (x < b) → (x % 3 = 0) → (x = 153 + 3 * (n - 1)) :=
by
  sorry

end numbers_divisible_by_three_l203_203656


namespace period_of_sine_plus_cosine_l203_203112

noncomputable def period_sine_cosine_sum (b : ℝ) : ℝ :=
  2 * Real.pi / b

theorem period_of_sine_plus_cosine (b : ℝ) (hb : b = 3) :
  period_sine_cosine_sum b = 2 * Real.pi / 3 :=
by
  rw [hb]
  apply rfl

end period_of_sine_plus_cosine_l203_203112


namespace c_investment_ratio_l203_203067

-- Conditions as definitions
variables (x : ℕ) (m : ℕ) (total_profit a_share : ℕ)
variables (h_total_profit : total_profit = 19200)
variables (h_a_share : a_share = 6400)

-- Definition of total investment (investments weighted by time)
def total_investment (x m : ℕ) : ℕ :=
  (12 * x) + (6 * 2 * x) + (4 * m * x)

-- Definition of A's share in terms of total investment
def a_share_in_terms_of_total_investment (x : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (12 * x * total_profit) / total_investment

-- The theorem stating the ratio of C's investment to A's investment
theorem c_investment_ratio (x m total_profit a_share : ℕ) (h_total_profit : total_profit = 19200)
  (h_a_share : a_share = 6400) (h_a_share_eq : a_share_in_terms_of_total_investment x (total_investment x m) total_profit = a_share) :
  m = 3 :=
by sorry

end c_investment_ratio_l203_203067


namespace number_of_pieces_l203_203263

def area_of_pan (length : ℕ) (width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

theorem number_of_pieces (length width side : ℕ) (h_length : length = 24) (h_width : width = 15) (h_side : side = 3) :
  (area_of_pan length width) / (area_of_piece side) = 40 :=
by
  rw [h_length, h_width, h_side]
  sorry

end number_of_pieces_l203_203263


namespace trigonometric_expression_l203_203795

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 3) : 
  ((Real.cos (α - π / 2) + Real.cos (α + π)) / (2 * Real.sin α) = 1 / 3) :=
by
  sorry

end trigonometric_expression_l203_203795


namespace min_value_expression_l203_203120

theorem min_value_expression (a b : ℝ) : 
  4 + (a + b)^2 ≥ 4 ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
sorry

end min_value_expression_l203_203120


namespace arithmetic_sequence_problem_l203_203299

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n m : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_problem
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 2 + a 3 = 32)
  (h2 : a 11 + a 12 + a 13 = 118) :
  a 4 + a 10 = 50 :=
sorry

end arithmetic_sequence_problem_l203_203299


namespace negation_of_p_l203_203707

-- Given conditions
def p : Prop := ∃ x : ℝ, x^2 + 3 * x = 4

-- The proof problem to be solved 
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^2 + 3 * x ≠ 4 := by
  sorry

end negation_of_p_l203_203707


namespace rectangle_perimeter_l203_203186

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def satisfies_relations (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  b1 + b2 = b3 ∧
  b1 + b3 = b4 ∧
  b3 + b4 = b5 ∧
  b4 + b5 = b6 ∧
  b2 + b5 = b7

def non_overlapping_squares (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  -- Placeholder for expressing that the squares are non-overlapping.
  true -- This is assumed as given in the problem.

theorem rectangle_perimeter (b1 b2 b3 b4 b5 b6 b7 : ℕ)
  (h1 : b1 = 1) (h2 : b2 = 2)
  (h_relations : satisfies_relations b1 b2 b3 b4 b5 b6 b7)
  (h_non_overlapping : non_overlapping_squares b1 b2 b3 b4 b5 b6 b7)
  (h_rel_prime : relatively_prime b6 b7) :
  2 * (b6 + b7) = 46 := by
  sorry

end rectangle_perimeter_l203_203186


namespace machine_sprockets_rate_l203_203673

theorem machine_sprockets_rate:
  ∀ (h : ℝ), h > 0 → (660 / (h + 10) = (660 / h) * 1/1.1) → (660 / 1.1 / h) = 6 :=
by
  intros h h_pos h_eq
  -- Proof will be here
  sorry

end machine_sprockets_rate_l203_203673


namespace christmas_trees_in_each_box_l203_203747

theorem christmas_trees_in_each_box
  (T : ℕ)
  (pieces_of_tinsel_in_each_box : ℕ := 4)
  (snow_globes_in_each_box : ℕ := 5)
  (total_boxes : ℕ := 12)
  (total_decorations : ℕ := 120)
  (decorations_per_box : ℕ := pieces_of_tinsel_in_each_box + T + snow_globes_in_each_box)
  (total_decorations_distributed : ℕ := total_boxes * decorations_per_box) :
  total_decorations_distributed = total_decorations → T = 1 := by
  sorry

end christmas_trees_in_each_box_l203_203747


namespace sarah_cupcakes_ratio_l203_203563

theorem sarah_cupcakes_ratio (total_cupcakes : ℕ) (cookies_from_michael : ℕ) 
    (final_desserts : ℕ) (cupcakes_given : ℕ) (h1 : total_cupcakes = 9) 
    (h2 : cookies_from_michael = 5) (h3 : final_desserts = 11) 
    (h4 : total_cupcakes - cupcakes_given + cookies_from_michael = final_desserts) : 
    cupcakes_given / total_cupcakes = 1 / 3 :=
by
  sorry

end sarah_cupcakes_ratio_l203_203563


namespace score_87_not_possible_l203_203374

def max_score := 15 * 6
def score (correct unanswered incorrect : ℕ) := 6 * correct + unanswered

theorem score_87_not_possible :
  ¬∃ (correct unanswered incorrect : ℕ), 
    correct + unanswered + incorrect = 15 ∧
    6 * correct + unanswered = 87 := 
sorry

end score_87_not_possible_l203_203374


namespace find_a_of_inequality_solution_l203_203716

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -3 < ax - 2 ∧ ax - 2 < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 := by
  sorry

end find_a_of_inequality_solution_l203_203716


namespace factorize_xy2_minus_x_l203_203430

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l203_203430


namespace domain_of_sqrt_expression_l203_203419

theorem domain_of_sqrt_expression :
  {x : ℝ | x^2 - 5 * x - 6 ≥ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 6} := by
sorry

end domain_of_sqrt_expression_l203_203419


namespace joan_travel_time_correct_l203_203986

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end joan_travel_time_correct_l203_203986


namespace minimum_p_plus_q_l203_203792

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem minimum_p_plus_q (p q : ℝ) (hpq : p ≠ q) (hf : f p + f q = 2) :
  p + q = 3 - 2 * Real.log 2 := by
  sorry

end minimum_p_plus_q_l203_203792


namespace bottle_caps_total_l203_203182

theorem bottle_caps_total (groups : ℕ) (bottle_caps_per_group : ℕ) (h1 : groups = 7) (h2 : bottle_caps_per_group = 5) : (groups * bottle_caps_per_group = 35) :=
by
  sorry

end bottle_caps_total_l203_203182


namespace ball_distribution_l203_203639

theorem ball_distribution (N a b : ℕ) (h1 : N = 6912) (h2 : N = 100 * a + b) (h3 : a < 100) (h4 : b < 100) : a + b = 81 :=
by
  sorry

end ball_distribution_l203_203639


namespace find_angle_beta_l203_203809

open Real

theorem find_angle_beta
  (α β : ℝ)
  (h1 : sin α = (sqrt 5) / 5)
  (h2 : sin (α - β) = - (sqrt 10) / 10)
  (hα_range : 0 < α ∧ α < π / 2)
  (hβ_range : 0 < β ∧ β < π / 2) :
  β = π / 4 :=
sorry

end find_angle_beta_l203_203809


namespace vowel_initial_probability_is_correct_l203_203722

-- Given conditions as definitions
def total_students : ℕ := 34
def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def vowels_count_per_vowel : ℕ := 2
def total_vowels_count := vowels.length * vowels_count_per_vowel

-- The probabilistic statement we want to prove
def vowel_probability : ℚ := total_vowels_count / total_students

-- The final statement to prove
theorem vowel_initial_probability_is_correct :
  vowel_probability = 6 / 17 :=
by
  unfold vowel_probability total_vowels_count
  -- Simplification to verify our statement.
  sorry

end vowel_initial_probability_is_correct_l203_203722


namespace total_players_is_139_l203_203808

def num_kabadi := 60
def num_kho_kho := 90
def num_soccer := 40
def num_basketball := 70
def num_volleyball := 50
def num_badminton := 30

def num_k_kh := 25
def num_k_s := 15
def num_k_b := 13
def num_k_v := 20
def num_k_ba := 10
def num_kh_s := 35
def num_kh_b := 16
def num_kh_v := 30
def num_kh_ba := 12
def num_s_b := 20
def num_s_v := 18
def num_s_ba := 7
def num_b_v := 15
def num_b_ba := 8
def num_v_ba := 10

def num_k_kh_s := 5
def num_k_b_v := 4
def num_s_b_ba := 3
def num_v_ba_kh := 2

def num_all_sports := 1

noncomputable def total_players : Nat :=
  (num_kabadi + num_kho_kho + num_soccer + num_basketball + num_volleyball + num_badminton) 
  - (num_k_kh + num_k_s + num_k_b + num_k_v + num_k_ba + num_kh_s + num_kh_b + num_kh_v + num_kh_ba + num_s_b + num_s_v + num_s_ba + num_b_v + num_b_ba + num_v_ba)
  + (num_k_kh_s + num_k_b_v + num_s_b_ba + num_v_ba_kh)
  - num_all_sports

theorem total_players_is_139 : total_players = 139 := 
  by 
    sorry

end total_players_is_139_l203_203808


namespace solve_equation_1_solve_equation_2_l203_203757

theorem solve_equation_1 (x : Real) : 
  (1/3) * (x - 3)^2 = 12 → x = 9 ∨ x = -3 :=
by
  sorry

theorem solve_equation_2 (x : Real) : 
  (2 * x - 1)^2 = (1 - x)^2 → x = 0 ∨ x = 2/3 :=
by
  sorry

end solve_equation_1_solve_equation_2_l203_203757


namespace inequality_proof_l203_203402

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
    (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) +
    (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) +
    (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤
    3 := by
  sorry

end inequality_proof_l203_203402


namespace percentage_boys_playing_soccer_is_correct_l203_203233

-- Definition of conditions 
def total_students := 420
def boys := 312
def soccer_players := 250
def girls_not_playing_soccer := 73

-- Calculated values based on conditions
def girls := total_students - boys
def girls_playing_soccer := girls - girls_not_playing_soccer
def boys_playing_soccer := soccer_players - girls_playing_soccer

-- Percentage of boys playing soccer
def percentage_boys_playing_soccer := (boys_playing_soccer / soccer_players) * 100

-- We assert the percentage of boys playing soccer is 86%
theorem percentage_boys_playing_soccer_is_correct : percentage_boys_playing_soccer = 86 := 
by
  -- Placeholder proof (use sorry as the proof is not required)
  sorry

end percentage_boys_playing_soccer_is_correct_l203_203233


namespace min_value_expression_l203_203798

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 2) ^ 2 + (b + 2) ^ 2 = 25 / 2 :=
sorry

end min_value_expression_l203_203798


namespace decreasing_interval_l203_203642

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv f x < 0 :=
by sorry

end decreasing_interval_l203_203642


namespace remaining_budget_l203_203017

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l203_203017


namespace binomial_expansion_problem_l203_203764

noncomputable def binomial_expansion_sum_coefficients (n : ℕ) : ℤ :=
  (1 - 3) ^ n

def general_term_coefficient (n r : ℕ) : ℤ :=
  (-3) ^ r * (Nat.choose n r)

theorem binomial_expansion_problem :
  ∃ (n : ℕ), binomial_expansion_sum_coefficients n = 64 ∧ general_term_coefficient 6 2 = 135 :=
by
  sorry

end binomial_expansion_problem_l203_203764


namespace intersection_complement_M_N_eq_456_l203_203556

def UniversalSet := { n : ℕ | 1 ≤ n ∧ n < 9 }
def M : Set ℕ := { 1, 2, 3 }
def N : Set ℕ := { 3, 4, 5, 6 }

theorem intersection_complement_M_N_eq_456 : 
  (UniversalSet \ M) ∩ N = { 4, 5, 6 } :=
by
  sorry

end intersection_complement_M_N_eq_456_l203_203556


namespace value_of_a_l203_203704

theorem value_of_a (a : ℕ) (A_a B_a : ℕ)
  (h1 : A_a = 10)
  (h2 : B_a = 11)
  (h3 : 2 * a^2 + 10 * a + 3 + 5 * a^2 + 7 * a + 8 = 8 * a^2 + 4 * a + 11) :
  a = 13 :=
sorry

end value_of_a_l203_203704


namespace inversely_proportional_l203_203782

theorem inversely_proportional (X Y K : ℝ) (h : X * Y = K - 1) (hK : K > 1) : 
  (∃ c : ℝ, ∀ x y : ℝ, x * y = c) :=
sorry

end inversely_proportional_l203_203782


namespace distance_to_y_axis_eq_reflection_across_x_axis_eq_l203_203796

-- Definitions based on the conditions provided
def point_P : ℝ × ℝ := (4, -2)

-- Statements we need to prove
theorem distance_to_y_axis_eq : (abs (point_P.1) = 4) := 
by
  sorry  -- Proof placeholder

theorem reflection_across_x_axis_eq : (point_P.1 = 4 ∧ -point_P.2 = 2) :=
by
  sorry  -- Proof placeholder

end distance_to_y_axis_eq_reflection_across_x_axis_eq_l203_203796


namespace distance_small_ball_to_surface_l203_203551

-- Define the main variables and conditions
variables (R : ℝ)

-- Define the conditions of the problem
def bottomBallRadius : ℝ := 2 * R
def topBallRadius : ℝ := R
def edgeLengthBaseTetrahedron : ℝ := 4 * R
def edgeLengthLateralTetrahedron : ℝ := 3 * R

-- Define the main statement in Lean format
theorem distance_small_ball_to_surface (R : ℝ) :
  (3 * R) = R + bottomBallRadius R :=
sorry

end distance_small_ball_to_surface_l203_203551


namespace find_line_eq_l203_203440

noncomputable def line_eq (x y : ℝ) : Prop :=
  (∃ a : ℝ, a ≠ 0 ∧ (a * x - y = 0 ∨ x + y - a = 0)) 

theorem find_line_eq : line_eq 2 3 :=
by
  sorry

end find_line_eq_l203_203440


namespace largest_n_exists_ints_l203_203171

theorem largest_n_exists_ints (n : ℤ) :
  (∃ x y z : ℤ, n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 10 :=
sorry

end largest_n_exists_ints_l203_203171


namespace second_horse_revolutions_l203_203575

theorem second_horse_revolutions (r1 r2 d1: ℝ) (n1 n2: ℕ) 
  (h1: r1 = 30) (h2: d1 = 36) (h3: r2 = 10) 
  (h4: 2 * Real.pi * r1 * d1 = 2 * Real.pi * r2 * n2) : 
  n2 = 108 := 
by
   sorry

end second_horse_revolutions_l203_203575


namespace minimum_sum_l203_203950

theorem minimum_sum (a b c : ℕ) (h : a * b * c = 3006) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ 105 :=
sorry

end minimum_sum_l203_203950


namespace safe_unlockable_by_five_l203_203971

def min_total_keys (num_locks : ℕ) (num_people : ℕ) (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) : ℕ :=
  num_locks * ((num_people + 1) / 2)

theorem safe_unlockable_by_five (num_locks : ℕ) (num_people : ℕ) 
  (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) :
  (∀ (P : Finset (Fin num_people)), P.card = 5 → (∀ k : Fin num_locks, ∃ p ∈ P, key_distribution k p)) →
  min_total_keys num_locks num_people key_distribution = 20 := 
by
  sorry

end safe_unlockable_by_five_l203_203971


namespace sarah_pencils_on_tuesday_l203_203240

theorem sarah_pencils_on_tuesday 
    (x : ℤ)
    (h1 : 20 + x + 3 * x = 92) : 
    x = 18 := 
by 
    sorry

end sarah_pencils_on_tuesday_l203_203240


namespace length_of_longest_side_l203_203231

variable (a b c p x l : ℝ)

-- conditions of the original problem
def original_triangle_sides (a b c : ℝ) : Prop := a = 8 ∧ b = 15 ∧ c = 17

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def similar_triangle_perimeter (a b c p x : ℝ) : Prop := (a * x) + (b * x) + (c * x) = p

-- proof target
theorem length_of_longest_side (h1: original_triangle_sides a b c) 
                               (h2: is_right_triangle a b c) 
                               (h3: similar_triangle_perimeter a b c p x) 
                               (h4: x = 4)
                               (h5: p = 160): (c * x) = 68 := by
  -- to complete the proof
  sorry

end length_of_longest_side_l203_203231


namespace max_sum_of_arithmetic_sequence_l203_203731

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (a1 : a 1 = 29) 
  (S10_eq_S20 : S 10 = S 20) :
  (∃ n, ∀ m, S n ≥ S m) ∧ ∃ n, (S n = S 15) :=
sorry

end max_sum_of_arithmetic_sequence_l203_203731


namespace range_c_of_sets_l203_203514

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_c_of_sets (c : ℝ) (h₀ : c > 0)
  (A := { x : ℝ | log2 x < 1 })
  (B := { x : ℝ | 0 < x ∧ x < c })
  (hA_union_B_eq_B : A ∪ B = B) :
  c ≥ 2 :=
by
  -- Minimum outline is provided, the proof part is replaced with "sorry" to indicate the point to be proved
  sorry

end range_c_of_sets_l203_203514


namespace quadratic_no_real_roots_l203_203505

theorem quadratic_no_real_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 :=
sorry

end quadratic_no_real_roots_l203_203505


namespace job_planned_completion_days_l203_203387

noncomputable def initial_days_planned (W D : ℝ) := 6 * (W / D) = (W - 3 * (W / D)) / 3

theorem job_planned_completion_days (W : ℝ ) : 
  ∃ D : ℝ, initial_days_planned W D ∧ D = 6 := 
sorry

end job_planned_completion_days_l203_203387


namespace sum_of_digits_of_10_pow_30_minus_36_l203_203591

def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem sum_of_digits_of_10_pow_30_minus_36 : 
  sum_of_digits (10^30 - 36) = 11 := 
by 
  -- proof goes here
  sorry

end sum_of_digits_of_10_pow_30_minus_36_l203_203591


namespace adam_has_9_apples_l203_203036

def jackie_apples : ℕ := 6
def difference : ℕ := 3

def adam_apples (j : ℕ) (d : ℕ) : ℕ := 
  j + d

theorem adam_has_9_apples : adam_apples jackie_apples difference = 9 := 
by 
  sorry

end adam_has_9_apples_l203_203036


namespace parallelogram_area_l203_203244

theorem parallelogram_area (s : ℝ) (ratio : ℝ) (A : ℝ) :
  s = 3 → ratio = 2 * Real.sqrt 2 → A = 9 → 
  (A * ratio = 18 * Real.sqrt 2) :=
by
  sorry

end parallelogram_area_l203_203244


namespace fraction_shaded_l203_203892

theorem fraction_shaded (s r : ℝ) (h : s^2 = 3 * r^2) :
    (1/2 * π * r^2) / (1/4 * π * s^2) = 2/3 := 
  sorry

end fraction_shaded_l203_203892


namespace cos_of_pi_over_3_minus_alpha_l203_203811

theorem cos_of_pi_over_3_minus_alpha (α : Real) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) :
  Real.cos (Real.pi / 3 - α) = 2 / 3 :=
by
  sorry

end cos_of_pi_over_3_minus_alpha_l203_203811


namespace interval_of_x_l203_203394

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l203_203394


namespace max_f_eq_4_monotonic_increase_interval_l203_203787

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_f_eq_4 (x : ℝ) : ∀ x : ℝ, f x ≤ 4 := 
by
  sorry

theorem monotonic_increase_interval (k : ℤ) : ∀ x : ℝ, (k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) ↔ 
  (0 ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1) :=
by
  sorry

end max_f_eq_4_monotonic_increase_interval_l203_203787


namespace number_of_packs_l203_203701

theorem number_of_packs (total_towels towels_per_pack : ℕ) (h1 : total_towels = 27) (h2 : towels_per_pack = 3) :
  total_towels / towels_per_pack = 9 :=
by
  sorry

end number_of_packs_l203_203701


namespace susan_fraction_apples_given_out_l203_203317

theorem susan_fraction_apples_given_out (frank_apples : ℕ) (frank_sold_fraction : ℚ) 
  (total_remaining_apples : ℕ) (susan_multiple : ℕ) 
  (H1 : frank_apples = 36) 
  (H2 : susan_multiple = 3) 
  (H3 : frank_sold_fraction = 1 / 3) 
  (H4 : total_remaining_apples = 78) :
  let susan_apples := susan_multiple * frank_apples
  let frank_sold_apples := frank_sold_fraction * frank_apples
  let frank_remaining_apples := frank_apples - frank_sold_apples
  let total_before_susan_gave_out := susan_apples + frank_remaining_apples
  let susan_gave_out := total_before_susan_gave_out - total_remaining_apples
  let susan_gave_fraction := susan_gave_out / susan_apples
  susan_gave_fraction = 1 / 2 :=
by
  sorry

end susan_fraction_apples_given_out_l203_203317


namespace f_x_plus_f_neg_x_eq_seven_l203_203226

variable (f : ℝ → ℝ)

-- Given conditions: 
axiom cond1 : ∀ x : ℝ, f x + f (1 - x) = 10
axiom cond2 : ∀ x : ℝ, f (1 + x) = 3 + f x

-- Prove statement:
theorem f_x_plus_f_neg_x_eq_seven : ∀ x : ℝ, f x + f (-x) = 7 := 
by
  sorry

end f_x_plus_f_neg_x_eq_seven_l203_203226


namespace largest_of_five_l203_203307

def a : ℝ := 0.994
def b : ℝ := 0.9399
def c : ℝ := 0.933
def d : ℝ := 0.9940
def e : ℝ := 0.9309

theorem largest_of_five : (a > b ∧ a > c ∧ a ≥ d ∧ a > e) := by
  -- We add sorry here to skip the proof
  sorry

end largest_of_five_l203_203307


namespace time_difference_halfway_point_l203_203367

theorem time_difference_halfway_point 
  (T_d : ℝ) 
  (T_s : ℝ := 2 * T_d) 
  (H_d : ℝ := T_d / 2) 
  (H_s : ℝ := T_s / 2) 
  (diff_time : ℝ := H_s - H_d) : 
  T_d = 35 →
  T_s = 2 * T_d →
  diff_time = 17.5 :=
by
  intros h1 h2
  sorry

end time_difference_halfway_point_l203_203367


namespace find_number_l203_203842

theorem find_number (x : ℝ) (h : ((x / 8) + 8 - 30) * 6 = 12) : x = 192 :=
sorry

end find_number_l203_203842


namespace y_intercept_of_line_b_is_minus_8_l203_203391

/-- Define a line in slope-intercept form y = mx + c --/
structure Line :=
  (m : ℝ)   -- slope
  (c : ℝ)   -- y-intercept

/-- Define a point in 2D Cartesian coordinate system --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define conditions for the problem --/
def line_b_parallel_to (l: Line) (p: Point) : Prop :=
  l.m = 2 ∧ p.x = 3 ∧ p.y = -2

/-- Define the target statement to prove --/
theorem y_intercept_of_line_b_is_minus_8 :
  ∀ (b: Line) (p: Point), line_b_parallel_to b p → b.c = -8 := by
  -- proof goes here
  sorry

end y_intercept_of_line_b_is_minus_8_l203_203391


namespace find_t_from_integral_l203_203828

theorem find_t_from_integral :
  (∫ x in (1 : ℝ)..t, (-1 / x + 2 * x)) = (3 - Real.log 2) → t = 2 :=
by
  sorry

end find_t_from_integral_l203_203828


namespace parallel_lines_perpendicular_lines_l203_203206

-- Definitions of the lines
def l1 (a x y : ℝ) := x + a * y - 2 * a - 2 = 0
def l2 (a x y : ℝ) := a * x + y - 1 - a = 0

-- Statement for parallel lines
theorem parallel_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = 0 ∨ x = 1) → a = 1 :=
by 
  -- proof outline
  sorry

-- Statement for perpendicular lines
theorem perpendicular_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = y) → a = 0 :=
by 
  -- proof outline
  sorry

end parallel_lines_perpendicular_lines_l203_203206


namespace min_S_l203_203243

-- Define the arithmetic sequence
def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

-- Define the sum of the first n terms
def S (n : ℕ) (a1 : ℤ) (d : ℤ) : ℤ :=
  (n * (a1 + a n a1 d)) / 2

-- Conditions
def a4 : ℤ := -15
def d : ℤ := 3

-- Found a1 from a4 and d
def a1 : ℤ := -24

-- Theorem stating the minimum value of the sum
theorem min_S : ∃ n, S n a1 d = -108 :=
  sorry

end min_S_l203_203243


namespace figure_100_squares_l203_203583

-- Define the initial conditions as given in the problem
def squares_in_figure (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 11
  | 2 => 25
  | 3 => 45
  | _ => sorry

-- Define the quadratic formula assumed from the problem conditions
def quadratic_formula (n : ℕ) : ℕ :=
  3 * n^2 + 5 * n + 3

-- Theorem: For figure 100, the number of squares is 30503
theorem figure_100_squares :
  squares_in_figure 100 = quadratic_formula 100 :=
by
  sorry

end figure_100_squares_l203_203583


namespace inequality_holds_for_positive_reals_l203_203005

theorem inequality_holds_for_positive_reals (x y : ℝ) (m n : ℤ) 
  (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (1 - x^n)^m + (1 - y^m)^n ≥ 1 :=
sorry

end inequality_holds_for_positive_reals_l203_203005


namespace sum_of_coefficients_l203_203308

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def polynomial (a : ℝ) (x : ℝ) : ℝ :=
  (2 + a * x) * (1 + x)^5

def x2_coefficient_condition (a : ℝ) : Prop :=
  2 * binomial_coefficient 5 2 + a * binomial_coefficient 5 1 = 15

theorem sum_of_coefficients (a : ℝ) (h : x2_coefficient_condition a) : 
  polynomial a 1 = 64 := 
sorry

end sum_of_coefficients_l203_203308


namespace Petya_wins_l203_203314

theorem Petya_wins (n : ℕ) (h₁ : n = 2016) : (∀ m : ℕ, m < n → ∀ k : ℕ, k ∣ m ∧ k ≠ m → m - k = 1 → false) :=
sorry

end Petya_wins_l203_203314


namespace number_of_bushes_l203_203974

theorem number_of_bushes (T B x y : ℕ) (h1 : B = T - 6) (h2 : x ≥ y + 10) (h3 : T * x = 128) (hT_pos : T > 0) (hx_pos : x > 0) : B = 2 :=
sorry

end number_of_bushes_l203_203974


namespace certain_event_is_A_l203_203094

def conditions (option_A option_B option_C option_D : Prop) : Prop :=
  option_A ∧ ¬option_B ∧ ¬option_C ∧ ¬option_D

theorem certain_event_is_A 
  (option_A option_B option_C option_D : Prop)
  (hconditions : conditions option_A option_B option_C option_D) : 
  ∀ e, (e = option_A) := 
by
  sorry

end certain_event_is_A_l203_203094


namespace expression_value_l203_203946

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z))

theorem expression_value
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod_nonzero : x * y + x * z + y * z ≠ 0) :
  expression x y z = -7 :=
by 
  sorry

end expression_value_l203_203946


namespace term_of_sequence_l203_203060

def S (n : ℕ) : ℚ := n^2 + 2/3

def a (n : ℕ) : ℚ :=
  if n = 1 then 5/3
  else 2 * n - 1

theorem term_of_sequence (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) :=
by
  sorry

end term_of_sequence_l203_203060


namespace semicircle_radius_l203_203253

noncomputable def radius_of_semicircle (P : ℝ) (h : P = 144) : ℝ :=
  144 / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (h : P = 144) : radius_of_semicircle P h = 144 / (Real.pi + 2) :=
  by sorry

end semicircle_radius_l203_203253


namespace max_marks_l203_203684

theorem max_marks (M: ℝ) (h1: 0.95 * M = 285):
  M = 300 :=
by
  sorry

end max_marks_l203_203684


namespace min_dot_product_l203_203859

variable {α : Type}
variables {a b : α}

noncomputable def dot (x y : α) : ℝ := sorry

axiom condition (a b : α) : abs (3 * dot a b) ≤ 4

theorem min_dot_product : dot a b = -4 / 3 :=
by
  sorry

end min_dot_product_l203_203859


namespace min_tangent_length_l203_203689

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end min_tangent_length_l203_203689


namespace k_cubed_divisible_l203_203048

theorem k_cubed_divisible (k : ℕ) (h : k = 84) : ∃ n : ℕ, k ^ 3 = 592704 * n :=
by
  sorry

end k_cubed_divisible_l203_203048


namespace discount_difference_l203_203677

theorem discount_difference :
  ∀ (original_price : ℝ),
  let initial_discount := 0.40
  let subsequent_discount := 0.25
  let claimed_discount := 0.60
  let actual_discount := 1 - (1 - initial_discount) * (1 - subsequent_discount)
  let difference := claimed_discount - actual_discount
  actual_discount = 0.55 ∧ difference = 0.05
:= by
  sorry

end discount_difference_l203_203677


namespace cost_of_traveling_roads_l203_203857

def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 40
def road_width : ℕ := 10
def cost_per_sqm : ℕ := 3

def area_road_parallel_length : ℕ := road_width * lawn_length
def area_road_parallel_breadth : ℕ := road_width * lawn_breadth
def area_intersection : ℕ := road_width * road_width

def total_area_roads : ℕ := area_road_parallel_length + area_road_parallel_breadth - area_intersection
def total_cost : ℕ := total_area_roads * cost_per_sqm

theorem cost_of_traveling_roads : total_cost = 3300 :=
by
  sorry

end cost_of_traveling_roads_l203_203857


namespace find_u_values_l203_203034

namespace MathProof

variable (u v : ℝ)
variable (h1 : u ≠ 0) (h2 : v ≠ 0)
variable (h3 : u + 1/v = 8) (h4 : v + 1/u = 16/3)

theorem find_u_values : u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 :=
by {
  sorry
}

end MathProof

end find_u_values_l203_203034


namespace g_at_pi_over_4_l203_203167

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) / 2 * Real.sin (2 * x) + (Real.sqrt 6) / 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_at_pi_over_4 : g (Real.pi / 4) = (Real.sqrt 6) / 2 := by
  sorry

end g_at_pi_over_4_l203_203167


namespace veggie_patty_percentage_l203_203062

-- Let's define the weights
def weight_total : ℕ := 150
def weight_additives : ℕ := 45

-- Let's express the proof statement as a theorem
theorem veggie_patty_percentage : (weight_total - weight_additives) * 100 / weight_total = 70 := by
  sorry

end veggie_patty_percentage_l203_203062


namespace rational_neither_positive_nor_fraction_l203_203995

def is_rational (q : ℚ) : Prop :=
  q.floor = q

def is_integer (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

def is_fraction (q : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ q = p / q

def is_positive (q : ℚ) : Prop :=
  q > 0

theorem rational_neither_positive_nor_fraction (q : ℚ) :
  (is_rational q) ∧ ¬(is_positive q) ∧ ¬(is_fraction q) ↔
  (is_integer q ∧ q ≤ 0) :=
sorry

end rational_neither_positive_nor_fraction_l203_203995


namespace gcf_palindromes_multiple_of_3_eq_3_l203_203637

-- Defining a condition that expresses a three-digit palindrome in the form 101a + 10b + a
def is_palindrome (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Defining a condition that n is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
n % 3 = 0

-- The Lean statement to prove the greatest common factor of all three-digit palindromes that are multiples of 3
theorem gcf_palindromes_multiple_of_3_eq_3 :
  ∃ gcf : ℕ, gcf = 3 ∧ ∀ n : ℕ, (is_palindrome n ∧ is_multiple_of_3 n) → gcf ∣ n :=
by
  sorry

end gcf_palindromes_multiple_of_3_eq_3_l203_203637


namespace circle_center_sum_l203_203361

theorem circle_center_sum (h k : ℝ) :
  (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = x ^ 2 + y ^ 2 - 6 * x - 8 * y + 38) → h + k = 7 :=
by sorry

end circle_center_sum_l203_203361


namespace repeating_decimal_sum_l203_203316

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9

theorem repeating_decimal_sum : 
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 - repeating_decimal_7 = -1 / 3 :=
by {
  sorry
}

end repeating_decimal_sum_l203_203316


namespace train_speed_l203_203936

theorem train_speed 
  (length_train : ℝ) (length_bridge : ℝ) (time : ℝ) 
  (h_length_train : length_train = 110)
  (h_length_bridge : length_bridge = 138)
  (h_time : time = 12.399008079353651) : 
  (length_train + length_bridge) / time * 3.6 = 72 :=
by
  sorry

end train_speed_l203_203936


namespace work_required_to_pump_liquid_l203_203138

/-- Calculation of work required to pump a liquid of density ρ out of a parabolic boiler. -/
theorem work_required_to_pump_liquid
  (ρ g H a : ℝ)
  (h_pos : 0 < H)
  (a_pos : 0 < a) :
  ∃ (A : ℝ), A = (π * ρ * g * H^3) / (6 * a^2) :=
by
  -- TODO: Provide the proof.
  sorry

end work_required_to_pump_liquid_l203_203138


namespace find_num_adults_l203_203745

-- Define the conditions
def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def eggs_per_girl : ℕ := 1
def eggs_per_boy := eggs_per_girl + 1
def num_girls : ℕ := 7
def num_boys : ℕ := 10

-- Compute total eggs given to girls
def eggs_given_to_girls : ℕ := num_girls * eggs_per_girl

-- Compute total eggs given to boys
def eggs_given_to_boys : ℕ := num_boys * eggs_per_boy

-- Compute total eggs given to children
def eggs_given_to_children : ℕ := eggs_given_to_girls + eggs_given_to_boys

-- Total number of eggs given to children
def eggs_left_for_adults : ℕ := total_eggs - eggs_given_to_children

-- Calculate the number of adults
def num_adults : ℕ := eggs_left_for_adults / eggs_per_adult

-- Finally, we want to prove that the number of adults is 3
theorem find_num_adults (h1 : total_eggs = 36) 
                        (h2 : eggs_per_adult = 3) 
                        (h3 : eggs_per_girl = 1)
                        (h4 : num_girls = 7) 
                        (h5 : num_boys = 10) : 
                        num_adults = 3 := by
  -- Using the given conditions and computations
  sorry

end find_num_adults_l203_203745


namespace intersection_distance_squared_l203_203153

-- Definitions for the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 9

-- Statement to prove
theorem intersection_distance_squared : 
  ∃ C D : ℝ × ℝ, circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧ 
  (C ≠ D) ∧ ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 224 / 9) :=
sorry

end intersection_distance_squared_l203_203153


namespace efficiency_ratio_l203_203068

variable (A_eff B_eff : ℝ)

-- Condition 1: A and B together finish a piece of work in 36 days
def combined_efficiency := A_eff + B_eff = 1 / 36

-- Condition 2: B alone finishes the work in 108 days
def B_efficiency := B_eff = 1 / 108

-- Theorem: Prove that the ratio of A's efficiency to B's efficiency is 2:1
theorem efficiency_ratio (h1 : combined_efficiency A_eff B_eff) (h2 : B_efficiency B_eff) : (A_eff / B_eff) = 2 := by
  sorry

end efficiency_ratio_l203_203068


namespace solve_equation_125_eq_5_25_exp_x_min_2_l203_203021

theorem solve_equation_125_eq_5_25_exp_x_min_2 :
    ∃ x : ℝ, 125 = 5 * (25 : ℝ)^(x - 2) ∧ x = 3 := 
by
  sorry

end solve_equation_125_eq_5_25_exp_x_min_2_l203_203021


namespace cost_whitewashing_l203_203304

theorem cost_whitewashing
  (length : ℝ) (breadth : ℝ) (height : ℝ)
  (door_height : ℝ) (door_width : ℝ)
  (window_height : ℝ) (window_width : ℝ)
  (num_windows : ℕ) (cost_per_square_foot : ℝ)
  (room_dimensions : length = 25 ∧ breadth = 15 ∧ height = 12)
  (door_dimensions : door_height = 6 ∧ door_width = 3)
  (window_dimensions : window_height = 4 ∧ window_width = 3)
  (num_windows_condition : num_windows = 3)
  (cost_condition : cost_per_square_foot = 8) :
  (2 * (length + breadth) * height - (door_height * door_width + num_windows * window_height * window_width)) * cost_per_square_foot = 7248 := 
by
  sorry

end cost_whitewashing_l203_203304


namespace part1_l203_203624

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
sorry

end part1_l203_203624


namespace draw_sequence_count_l203_203870

noncomputable def total_sequences : ℕ :=
  (Nat.choose 4 3) * (Nat.factorial 4) * 5

theorem draw_sequence_count : total_sequences = 480 := by
  sorry

end draw_sequence_count_l203_203870


namespace find_phi_l203_203270

theorem find_phi (ϕ : ℝ) (h1 : |ϕ| < π / 2)
  (h2 : ∃ k : ℤ, 3 * (π / 12) + ϕ = k * π + π / 2) :
  ϕ = π / 4 :=
by sorry

end find_phi_l203_203270


namespace avg_age_women_is_52_l203_203155

-- Definitions
def avg_age_men (A : ℚ) := 9 * A
def total_increase := 36
def combined_age_replaced := 36 + 32
def combined_age_women := combined_age_replaced + total_increase
def avg_age_women (W : ℚ) := W / 2

-- Theorem statement
theorem avg_age_women_is_52 (A : ℚ) : avg_age_women combined_age_women = 52 :=
by
  sorry

end avg_age_women_is_52_l203_203155


namespace students_in_trumpet_or_trombone_l203_203461

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end students_in_trumpet_or_trombone_l203_203461


namespace intersection_complement_A_l203_203458

def A : Set ℝ := {x | abs (x - 1) < 1}

def B : Set ℝ := {x | x < 1}

def CRB : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_A :
  (CRB ∩ A) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_l203_203458


namespace new_person_weight_l203_203515

theorem new_person_weight (avg_inc : Real) (num_persons : Nat) (old_weight new_weight : Real)
  (h1 : avg_inc = 2.5)
  (h2 : num_persons = 8)
  (h3 : old_weight = 40)
  (h4 : num_persons * avg_inc = new_weight - old_weight) :
  new_weight = 60 :=
by
  --proof will be done here
  sorry

end new_person_weight_l203_203515


namespace tim_runs_more_than_sarah_l203_203683

-- Definitions based on the conditions
def street_width : ℕ := 25
def side_length : ℕ := 450

-- Perimeters of the paths
def sarah_perimeter : ℕ := 4 * side_length
def tim_perimeter : ℕ := 4 * (side_length + 2 * street_width)

-- The theorem to prove
theorem tim_runs_more_than_sarah : tim_perimeter - sarah_perimeter = 200 := by
  -- The proof will be filled in here
  sorry

end tim_runs_more_than_sarah_l203_203683


namespace weekly_deficit_is_2800_l203_203902

def daily_intake (day : String) : ℕ :=
  if day = "Monday" then 2500 else 
  if day = "Tuesday" then 2600 else 
  if day = "Wednesday" then 2400 else 
  if day = "Thursday" then 2700 else 
  if day = "Friday" then 2300 else 
  if day = "Saturday" then 3500 else 
  if day = "Sunday" then 2400 else 0

def daily_expenditure (day : String) : ℕ :=
  if day = "Monday" then 3000 else 
  if day = "Tuesday" then 3200 else 
  if day = "Wednesday" then 2900 else 
  if day = "Thursday" then 3100 else 
  if day = "Friday" then 2800 else 
  if day = "Saturday" then 3000 else 
  if day = "Sunday" then 2700 else 0

def daily_deficit (day : String) : ℤ :=
  daily_expenditure day - daily_intake day

def weekly_caloric_deficit : ℤ :=
  daily_deficit "Monday" +
  daily_deficit "Tuesday" +
  daily_deficit "Wednesday" +
  daily_deficit "Thursday" +
  daily_deficit "Friday" +
  daily_deficit "Saturday" +
  daily_deficit "Sunday"

theorem weekly_deficit_is_2800 : weekly_caloric_deficit = 2800 := by
  sorry

end weekly_deficit_is_2800_l203_203902


namespace fraction_simplest_sum_l203_203069

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum_l203_203069


namespace asymptotes_of_hyperbola_l203_203041

-- Definition of hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- The main theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l203_203041


namespace difference_of_roots_of_quadratic_l203_203813

theorem difference_of_roots_of_quadratic :
  (∃ (r1 r2 : ℝ), 3 * r1 ^ 2 + 4 * r1 - 15 = 0 ∧
                  3 * r2 ^ 2 + 4 * r2 - 15 = 0 ∧
                  r1 + r2 = -4 / 3 ∧
                  r1 * r2 = -5 ∧
                  r1 - r2 = 14 / 3) :=
sorry

end difference_of_roots_of_quadratic_l203_203813


namespace ratio_SP2_SP1_l203_203550

variable (CP : ℝ)

-- First condition: Sold at a profit of 140%
def SP1 := 2.4 * CP

-- Second condition: Sold at a loss of 20%
def SP2 := 0.8 * CP

-- Statement: The ratio of SP2 to SP1 is 1 to 3
theorem ratio_SP2_SP1 : SP2 / SP1 = 1 / 3 :=
by
  sorry

end ratio_SP2_SP1_l203_203550


namespace point_on_y_axis_l203_203414

theorem point_on_y_axis (a : ℝ) :
  (a + 2 = 0) -> a = -2 :=
by
  intro h
  sorry

end point_on_y_axis_l203_203414


namespace solve_for_w_squared_l203_203449

-- Define the original equation
def eqn (w : ℝ) := 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)

-- Define the goal to prove w^2 = 6.7585 based on the given equation
theorem solve_for_w_squared : ∃ w : ℝ, eqn w ∧ w^2 = 6.7585 :=
by
  sorry

end solve_for_w_squared_l203_203449


namespace find_D_l203_203202

theorem find_D (A B C D : ℕ) (h₁ : A + A = 6) (h₂ : B - A = 4) (h₃ : C + B = 9) (h₄ : D - C = 7) : D = 9 :=
sorry

end find_D_l203_203202


namespace sum_of_local_values_l203_203647

def local_value (digit place_value : ℕ) : ℕ := digit * place_value

theorem sum_of_local_values :
  local_value 2 1000 + local_value 3 100 + local_value 4 10 + local_value 5 1 = 2345 :=
by
  sorry

end sum_of_local_values_l203_203647


namespace crown_distribution_l203_203754

theorem crown_distribution 
  (A B C D E : ℤ) 
  (h1 : 2 * C = 3 * A)
  (h2 : 4 * D = 3 * B)
  (h3 : 4 * E = 5 * C)
  (h4 : 5 * D = 6 * A)
  (h5 : A + B + C + D + E = 2870) : 
  A = 400 ∧ B = 640 ∧ C = 600 ∧ D = 480 ∧ E = 750 := 
by 
  sorry

end crown_distribution_l203_203754


namespace max_profit_l203_203444

noncomputable def profit (x : ℝ) : ℝ :=
  10 * (x - 40) * (100 - x)

theorem max_profit (x : ℝ) (hx : x > 40) :
  (profit 70 = 9000) ∧ ∀ y > 40, profit y ≤ 9000 := by
  sorry

end max_profit_l203_203444


namespace smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l203_203627

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  -- Proof omitted
  sorry

theorem monotonically_increasing_interval :
  ∃ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
               k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤  k * Real.pi + Real.pi / 6 →
               x ≤ y → f x ≤ f y := 
by 
  -- Proof omitted
  sorry

theorem minimum_value_a_of_triangle (A B C a b c : ℝ) 
  (h₀ : f A = 1/2) 
  (h₁ : B^2 - C^2 - B * C * Real.cos A - a^2 = 4) :
  a ≥ 2 * Real.sqrt 2 :=
by 
  -- Proof omitted
  sorry

end smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l203_203627


namespace Razorback_shop_total_revenue_l203_203785

theorem Razorback_shop_total_revenue :
  let Tshirt_price := 62
  let Jersey_price := 99
  let Hat_price := 45
  let Keychain_price := 25
  let Tshirt_sold := 183
  let Jersey_sold := 31
  let Hat_sold := 142
  let Keychain_sold := 215
  let revenue := (Tshirt_price * Tshirt_sold) + (Jersey_price * Jersey_sold) + (Hat_price * Hat_sold) + (Keychain_price * Keychain_sold)
  revenue = 26180 :=
by
  sorry

end Razorback_shop_total_revenue_l203_203785


namespace total_leaves_on_farm_l203_203325

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l203_203325


namespace camille_total_birds_count_l203_203010

theorem camille_total_birds_count :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry

end camille_total_birds_count_l203_203010


namespace math_problem_l203_203192

theorem math_problem (x y : Int)
  (hx : x = 2 - 4 + 6)
  (hy : y = 1 - 3 + 5) :
  x - y = 1 :=
by
  sorry

end math_problem_l203_203192


namespace domain_of_g_eq_7_infty_l203_203239

noncomputable def domain_function (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (x - 7 > 0)

theorem domain_of_g_eq_7_infty : 
  (∀ x : ℝ, domain_function x ↔ x > 7) :=
by 
  -- We declare the structure of our proof problem here.
  -- The detailed proof steps would follow.
  sorry

end domain_of_g_eq_7_infty_l203_203239


namespace james_weekly_earnings_l203_203496

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l203_203496


namespace compare_y_l203_203865

-- Define the points M and N lie on the graph of y = -5/x
def on_inverse_proportion_curve (x y : ℝ) : Prop :=
  y = -5 / x

-- Main theorem to be proven
theorem compare_y (x1 y1 x2 y2 : ℝ) (h1 : on_inverse_proportion_curve x1 y1) (h2 : on_inverse_proportion_curve x2 y2) (hx : x1 > 0 ∧ x2 < 0) : y1 < y2 :=
by
  sorry

end compare_y_l203_203865


namespace total_crayons_l203_203862

-- Define relevant conditions
def crayons_per_child : ℕ := 8
def number_of_children : ℕ := 7

-- Define the Lean statement to prove the total number of crayons
theorem total_crayons : crayons_per_child * number_of_children = 56 :=
by
  sorry

end total_crayons_l203_203862


namespace alice_study_time_for_average_75_l203_203117

variable (study_time : ℕ → ℚ)
variable (score : ℕ → ℚ)

def inverse_relation := ∀ n, study_time n * score n = 120

theorem alice_study_time_for_average_75
  (inverse_relation : inverse_relation study_time score)
  (study_time_1 : study_time 1 = 2)
  (score_1 : score 1 = 60)
  : study_time 2 = 4/3 := by
  sorry

end alice_study_time_for_average_75_l203_203117


namespace correct_exponentiation_l203_203038

theorem correct_exponentiation (a : ℝ) : (-2 * a^3) ^ 4 = 16 * a ^ 12 :=
by sorry

end correct_exponentiation_l203_203038


namespace number_of_integer_solutions_Q_is_one_l203_203223

def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 13 * x^2 + 3 * x - 19

theorem number_of_integer_solutions_Q_is_one : 
    (∃! x : ℤ, ∃ k : ℤ, Q x = k^2) := 
sorry

end number_of_integer_solutions_Q_is_one_l203_203223


namespace number_of_classmates_l203_203659

theorem number_of_classmates (n m : ℕ) (h₁ : n < 100) (h₂ : m = 9)
:(2 ^ 6 - 1) = 63 → 63 / m = 7 := by
  intros 
  sorry

end number_of_classmates_l203_203659


namespace calculate_diff_of_squares_l203_203845

noncomputable def diff_of_squares (a b : ℕ) : ℕ :=
  a^2 - b^2

theorem calculate_diff_of_squares :
  diff_of_squares 601 597 = 4792 :=
by
  sorry

end calculate_diff_of_squares_l203_203845


namespace notification_possible_l203_203552

-- Define the conditions
def side_length : ℝ := 2
def speed : ℝ := 3
def initial_time : ℝ := 12 -- noon
def arrival_time : ℝ := 19 -- 7 PM
def notification_time : ℝ := arrival_time - initial_time -- total available time for notification

-- Define the proof statement
theorem notification_possible :
  ∃ (partition : ℕ → ℝ) (steps : ℕ → ℝ), (∀ k, steps k * partition k < notification_time) ∧ 
  ∑' k, (steps k * partition k) ≤ 6 :=
by
  sorry

end notification_possible_l203_203552


namespace range_of_m_l203_203630

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : π < α ∧ α < 2 * π ∨ 3 * π < α ∧ α < 4 * π) 
(h2 : Real.sin α = (2 * m - 3) / (4 - m)) : 
  -1 < m ∧ m < (3 : ℝ) / 2 :=
  sorry

end range_of_m_l203_203630


namespace total_amount_divided_l203_203908

theorem total_amount_divided (A B C : ℝ) (h1 : A = (2/3) * (B + C)) (h2 : B = (2/3) * (A + C)) (h3 : A = 200) :
  A + B + C = 500 :=
by
  sorry

end total_amount_divided_l203_203908


namespace smallest_possible_l_l203_203218

theorem smallest_possible_l (a b c L : ℕ) (h1 : a * b = 7) (h2 : a * c = 27) (h3 : b * c = L) (h4 : ∃ k, a * b * c = k * k) : L = 21 := sorry

end smallest_possible_l_l203_203218


namespace hide_and_seek_friends_l203_203177

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l203_203177


namespace average_growth_rate_l203_203023

theorem average_growth_rate (x : ℝ) :
  (7200 * (1 + x)^2 = 8712) → x = 0.10 :=
by
  sorry

end average_growth_rate_l203_203023


namespace proposition_B_proposition_C_l203_203788

variable (a b c d : ℝ)

-- Proposition B: If |a| > |b|, then a² > b²
theorem proposition_B (h : |a| > |b|) : a^2 > b^2 :=
sorry

-- Proposition C: If (a - b)c² > 0, then a > b
theorem proposition_C (h : (a - b) * c^2 > 0) : a > b :=
sorry

end proposition_B_proposition_C_l203_203788


namespace winningTicketProbability_l203_203169

-- Given conditions
def sharpBallProbability : ℚ := 1 / 30
def prizeBallsProbability : ℚ := 1 / (Nat.descFactorial 50 6)

-- The target probability that we are supposed to prove
def targetWinningProbability : ℚ := 1 / 476721000

-- Main theorem stating the required probability calculation
theorem winningTicketProbability :
  sharpBallProbability * prizeBallsProbability = targetWinningProbability :=
  sorry

end winningTicketProbability_l203_203169


namespace max_vector_sum_l203_203681

open Real EuclideanSpace

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def radius : ℝ := 2
noncomputable def distance_AB : ℝ := 2 * sqrt 3

theorem max_vector_sum {A B : ℝ × ℝ} 
    (hA_on_circle : dist A circle_center = radius)
    (hB_on_circle : dist B circle_center = radius)
    (hAB_eq : dist A B = distance_AB) :
    (dist (0,0) ((A.1 + B.1, A.2 + B.2))) ≤ 8 :=
by 
  sorry

end max_vector_sum_l203_203681


namespace probability_even_sum_97_l203_203143

-- You don't need to include numbers since they are directly available in Lean's library
-- This will help to ensure broader compatibility and avoid namespace issues

theorem probability_even_sum_97 (m n : ℕ) (hmn : Nat.gcd m n = 1) 
  (hprob : (224 : ℚ) / 455 = m / n) : 
  m + n = 97 :=
sorry

end probability_even_sum_97_l203_203143


namespace value_added_to_number_l203_203571

theorem value_added_to_number (n v : ℤ) (h1 : n = 9)
  (h2 : 3 * (n + 2) = v + n) : v = 24 :=
by
  sorry

end value_added_to_number_l203_203571


namespace orchid_bushes_after_planting_l203_203640

def total_orchid_bushes (current_orchids new_orchids : Nat) : Nat :=
  current_orchids + new_orchids

theorem orchid_bushes_after_planting :
  ∀ (current_orchids new_orchids : Nat), current_orchids = 22 → new_orchids = 13 → total_orchid_bushes current_orchids new_orchids = 35 :=
by
  intros current_orchids new_orchids h_current h_new
  rw [h_current, h_new]
  exact rfl

end orchid_bushes_after_planting_l203_203640


namespace smallest_c_inequality_l203_203180

theorem smallest_c_inequality (x : ℕ → ℝ) (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10) :
  ∃ c : ℝ, (∀ x : ℕ → ℝ, x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10 →
    |x 0| + |x 1| + |x 2| + |x 3| + |x 4| + |x 5| + |x 6| + |x 7| + |x 8| ≥ c * |x 4|) ∧ c = 9 := 
by
  sorry

end smallest_c_inequality_l203_203180


namespace smallest_a_condition_l203_203984

theorem smallest_a_condition:
  ∃ a: ℝ, (∀ x y z: ℝ, (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1) → a * (x^2 + y^2 + z^2) + x * y * z ≥ 10 / 27) ∧ a = 2 / 9 :=
sorry

end smallest_a_condition_l203_203984


namespace total_porridge_l203_203953

variable {c1 c2 c3 c4 c5 c6 : ℝ}

theorem total_porridge (h1 : c3 = c1 + c2)
                      (h2 : c4 = c2 + c3)
                      (h3 : c5 = c3 + c4)
                      (h4 : c6 = c4 + c5)
                      (h5 : c5 = 10) :
                      c1 + c2 + c3 + c4 + c5 + c6 = 40 := 
by
  sorry

end total_porridge_l203_203953


namespace min_value_of_f_l203_203763

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f : ∀ (x : ℝ), x > 2 → f x ≥ 4 := by
  sorry

end min_value_of_f_l203_203763


namespace unique_solution_a_eq_sqrt3_l203_203000

theorem unique_solution_a_eq_sqrt3 (a : ℝ) :
  (∃! x : ℝ, x^2 - a * |x| + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt3_l203_203000


namespace inequality_solution_set_l203_203079

open Set -- Open the Set namespace to work with sets in Lean

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (x ∈ Icc (3 / 4) 2 \ {2}) := 
by
  sorry

end inequality_solution_set_l203_203079


namespace amount_left_in_wallet_l203_203529

theorem amount_left_in_wallet
  (initial_amount : ℝ)
  (spent_amount : ℝ)
  (h_initial : initial_amount = 94)
  (h_spent : spent_amount = 16) :
  initial_amount - spent_amount = 78 :=
by
  sorry

end amount_left_in_wallet_l203_203529


namespace hardest_work_diff_l203_203748

theorem hardest_work_diff 
  (A B C D : ℕ) 
  (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x ∧ D = 4 * x)
  (h_total : A + B + C + D = 240) :
  (D - A) = 72 :=
by
  sorry

end hardest_work_diff_l203_203748


namespace technicians_count_l203_203404

theorem technicians_count (avg_all : ℕ) (avg_tech : ℕ) (avg_other : ℕ) (total_workers : ℕ)
  (h1 : avg_all = 750) (h2 : avg_tech = 900) (h3 : avg_other = 700) (h4 : total_workers = 20) :
  ∃ T O : ℕ, (T + O = total_workers) ∧ ((T * avg_tech + O * avg_other) = total_workers * avg_all) ∧ (T = 5) :=
by
  sorry

end technicians_count_l203_203404


namespace valid_subsets_12_even_subsets_305_l203_203519

def valid_subsets_count(n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else
    valid_subsets_count (n - 1) +
    valid_subsets_count (n - 2) +
    valid_subsets_count (n - 3)
    -- Recurrence relation for valid subsets which satisfy the conditions

theorem valid_subsets_12 : valid_subsets_count 12 = 610 :=
  by sorry
  -- We need to verify recurrence and compute for n = 12 (optional step if just computing, not proving the sequence.)

theorem even_subsets_305 :
  (valid_subsets_count 12) / 2 = 305 :=
  by sorry
  -- Concludes that half the valid subsets for n = 12 are even-sized sets.

end valid_subsets_12_even_subsets_305_l203_203519


namespace mass_percentage_C_in_butanoic_acid_is_54_50_l203_203653

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_butanoic_acid : ℝ :=
  (4 * atomic_mass_C) + (8 * atomic_mass_H) + (2 * atomic_mass_O)

noncomputable def mass_of_C_in_butanoic_acid : ℝ :=
  4 * atomic_mass_C

noncomputable def mass_percentage_C : ℝ :=
  (mass_of_C_in_butanoic_acid / molar_mass_butanoic_acid) * 100

theorem mass_percentage_C_in_butanoic_acid_is_54_50 :
  mass_percentage_C = 54.50 := by
  sorry

end mass_percentage_C_in_butanoic_acid_is_54_50_l203_203653


namespace maximum_value_is_16_l203_203009

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
(x^2 - 2 * x * y + 2 * y^2) * (x^2 - 2 * x * z + 2 * z^2) * (y^2 - 2 * y * z + 2 * z^2)

theorem maximum_value_is_16 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  maximum_value x y z ≤ 16 :=
by
  sorry

end maximum_value_is_16_l203_203009


namespace reciprocal_eq_self_l203_203327

theorem reciprocal_eq_self {x : ℝ} (h : x ≠ 0) : (1 / x = x) → (x = 1 ∨ x = -1) :=
by
  intro h1
  sorry

end reciprocal_eq_self_l203_203327


namespace sum_of_ages_l203_203097

theorem sum_of_ages (J L : ℕ) (h1 : J = L + 8) (h2 : J + 5 = 3 * (L - 6)) : (J + L) = 39 :=
by {
  -- Proof steps would go here, but are omitted for this task per instructions
  sorry
}

end sum_of_ages_l203_203097


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l203_203261

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l203_203261


namespace total_unique_handshakes_l203_203022

def num_couples := 8
def num_individuals := num_couples * 2
def potential_handshakes_per_person := num_individuals - 1 - 1
def total_handshakes := num_individuals * potential_handshakes_per_person / 2

theorem total_unique_handshakes : total_handshakes = 112 := sorry

end total_unique_handshakes_l203_203022


namespace probability_five_digit_palindrome_divisible_by_11_l203_203858

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  n % 100 = 100*a + 10*b + c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_five_digit_palindrome_divisible_by_11 :
  let count_palindromes := 9 * 10 * 10
  let count_divisible_by_11 := 165
  (count_divisible_by_11 : ℚ) / count_palindromes = 11 / 60 :=
by
  sorry

end probability_five_digit_palindrome_divisible_by_11_l203_203858


namespace grooming_time_5_dogs_3_cats_l203_203694

theorem grooming_time_5_dogs_3_cats :
  (2.5 * 5 + 0.5 * 3) * 60 = 840 :=
by
  -- Prove that grooming 5 dogs and 3 cats takes 840 minutes.
  sorry

end grooming_time_5_dogs_3_cats_l203_203694


namespace square_area_with_circles_l203_203504

theorem square_area_with_circles 
  (radius : ℝ) 
  (circle_count : ℕ) 
  (side_length : ℝ) 
  (total_area : ℝ)
  (h1 : radius = 7)
  (h2 : circle_count = 4)
  (h3 : side_length = 2 * (2 * radius))
  (h4 : total_area = side_length * side_length)
  : total_area = 784 :=
sorry

end square_area_with_circles_l203_203504


namespace unique_symmetric_solutions_l203_203894

theorem unique_symmetric_solutions (a b α β : ℝ) (h_mul : α * β = a) (h_add : α + β = b) :
  ∀ (x y : ℝ), x * y = a ∧ x + y = b → (x = α ∧ y = β) ∨ (x = β ∧ y = α) :=
by
  sorry

end unique_symmetric_solutions_l203_203894


namespace find_m_l203_203868

def vec (α : Type*) := (α × α)
def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) :
  let a : vec ℝ := (1, 3)
  let b : vec ℝ := (-2, m)
  let c : vec ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  dot_product a c = 0 → m = -1 :=
by
  sorry

end find_m_l203_203868


namespace remainder_of_greatest_integer_multiple_of_9_no_repeats_l203_203658

noncomputable def greatest_integer_multiple_of_9_no_repeats : ℕ :=
  9876543210 -- this should correspond to the greatest number meeting the criteria, but it's identified via more specific logic in practice

theorem remainder_of_greatest_integer_multiple_of_9_no_repeats : 
  (greatest_integer_multiple_of_9_no_repeats % 1000) = 621 := 
  by sorry

end remainder_of_greatest_integer_multiple_of_9_no_repeats_l203_203658


namespace mixed_number_sum_l203_203846

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l203_203846


namespace multiplication_simplification_l203_203073

theorem multiplication_simplification :
  let y := 6742
  let z := 397778
  let approx_mult (a b : ℕ) := 60 * a - a
  z = approx_mult y 59 := sorry

end multiplication_simplification_l203_203073


namespace craftsman_jars_l203_203966

theorem craftsman_jars (J P : ℕ) 
  (h1 : J = 2 * P)
  (h2 : 5 * J + 15 * P = 200) : 
  J = 16 := by
  sorry

end craftsman_jars_l203_203966


namespace fish_per_bowl_l203_203794

theorem fish_per_bowl (num_bowls num_fish : ℕ) (h1 : num_bowls = 261) (h2 : num_fish = 6003) :
  num_fish / num_bowls = 23 :=
by {
  sorry
}

end fish_per_bowl_l203_203794


namespace find_z_l203_203396

open Complex

theorem find_z (z : ℂ) (h : z * (2 - I) = 5 * I) : z = -1 + 2 * I :=
sorry

end find_z_l203_203396


namespace proof_expression_value_l203_203675

theorem proof_expression_value (x y : ℝ) (h : x + 2 * y = 30) : 
  (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 := 
by 
  sorry

end proof_expression_value_l203_203675


namespace price_per_pound_of_peanuts_is_2_40_l203_203077

-- Assume the conditions
def peanuts_price_per_pound (P : ℝ) : Prop :=
  let cashews_price := 6.00
  let mixture_weight := 60
  let mixture_price_per_pound := 3.00
  let cashews_weight := 10
  let total_mixture_price := mixture_weight * mixture_price_per_pound
  let total_cashews_price := cashews_weight * cashews_price
  let total_peanuts_price := total_mixture_price - total_cashews_price
  let peanuts_weight := mixture_weight - cashews_weight
  let P := total_peanuts_price / peanuts_weight
  P = 2.40

-- Prove the price per pound of peanuts
theorem price_per_pound_of_peanuts_is_2_40 (P : ℝ) : peanuts_price_per_pound P :=
by
  sorry

end price_per_pound_of_peanuts_is_2_40_l203_203077


namespace find_triplets_l203_203876

theorem find_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧ (2 * y^3 + 1 = 3 * x * y) ∧ (2 * z^3 + 1 = 3 * y * z) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 / 2 ∧ y = -1 / 2 ∧ z = -1 / 2) :=
by 
  intro h
  sorry

end find_triplets_l203_203876


namespace percentage_of_180_out_of_360_equals_50_l203_203354

theorem percentage_of_180_out_of_360_equals_50 :
  (180 / 360 : ℚ) * 100 = 50 := 
sorry

end percentage_of_180_out_of_360_equals_50_l203_203354


namespace math_proof_problem_l203_203477

noncomputable def discriminant (a : ℝ) : ℝ := a^2 - 4 * a + 2

def is_real_roots (a : ℝ) : Prop := discriminant a ≥ 0

def solution_set_a : Set ℝ := { a | is_real_roots a ∧ (a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2) }

def f (a : ℝ) : ℝ := -3 * a^2 + 16 * a - 8

def inequality_m (m t : ℝ) : Prop := m^2 + t * m + 4 * Real.sqrt 2 + 6 ≥ f (2 + Real.sqrt 2)

theorem math_proof_problem :
  (∀ a ∈ solution_set_a, ∃ m : ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) (1 : ℝ), inequality_m m t) ∧
  (∀ m t, inequality_m m t → m ≤ -1 ∨ m = 0 ∨ m ≥ 1) :=
by
  sorry

end math_proof_problem_l203_203477


namespace petya_numbers_board_l203_203368

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end petya_numbers_board_l203_203368


namespace largest_gcd_l203_203489

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l203_203489


namespace event_day_is_Sunday_l203_203817

def days_in_week := 7

def event_day := 1500

def start_day := "Friday"

def day_of_week_according_to_mod : Nat → String 
| 0 => "Friday"
| 1 => "Saturday"
| 2 => "Sunday"
| 3 => "Monday"
| 4 => "Tuesday"
| 5 => "Wednesday"
| 6 => "Thursday"
| _ => "Invalid"

theorem event_day_is_Sunday : day_of_week_according_to_mod (event_day % days_in_week) = "Sunday" :=
sorry

end event_day_is_Sunday_l203_203817


namespace proof_problem_l203_203772

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) : 
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x)) ∧ ¬ (∀ x : ℝ, f a b x = -f a b (-x))) := 
sorry

end proof_problem_l203_203772


namespace quadratic_roots_eqn_l203_203168

theorem quadratic_roots_eqn (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h3 : b = -(x1 + x2)) (h4 : c = x1 * x2) : 
    (x^2 + b * x + c = 0) ↔ (x^2 - x - 6 = 0) :=
by
  sorry

end quadratic_roots_eqn_l203_203168


namespace fraction_value_l203_203940

theorem fraction_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1 : ℚ) / (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3/4 :=
sorry

end fraction_value_l203_203940


namespace sufficient_condition_for_m_ge_9_l203_203333

theorem sufficient_condition_for_m_ge_9
  (x m : ℝ)
  (p : |x - 4| ≤ 6)
  (q : x ≤ 1 + m)
  (h_sufficient : ∀ x, |x - 4| ≤ 6 → x ≤ 1 + m)
  (h_not_necessary : ∃ x, ¬(|x - 4| ≤ 6) ∧ x ≤ 1 + m) :
  m ≥ 9 := 
sorry

end sufficient_condition_for_m_ge_9_l203_203333


namespace integer_coordinates_midpoint_exists_l203_203789

theorem integer_coordinates_midpoint_exists (P : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧
    ∃ x y : ℤ, (2 * x = (P i).1 + (P j).1) ∧ (2 * y = (P i).2 + (P j).2) := sorry

end integer_coordinates_midpoint_exists_l203_203789


namespace sunday_saturday_ratio_is_two_to_one_l203_203422

-- Define the conditions as given in the problem
def total_pages : ℕ := 360
def saturday_morning_read : ℕ := 40
def saturday_night_read : ℕ := 10
def remaining_pages : ℕ := 210

-- Define Ethan's total pages read so far
def total_read : ℕ := total_pages - remaining_pages

-- Define pages read on Saturday
def saturday_total_read : ℕ := saturday_morning_read + saturday_night_read

-- Define pages read on Sunday
def sunday_total_read : ℕ := total_read - saturday_total_read

-- Define the ratio of pages read on Sunday to pages read on Saturday
def sunday_to_saturday_ratio : ℕ := sunday_total_read / saturday_total_read

-- Theorem statement: ratio of pages read on Sunday to pages read on Saturday is 2:1
theorem sunday_saturday_ratio_is_two_to_one : sunday_to_saturday_ratio = 2 :=
by
  -- This part should contain the detailed proof
  sorry

end sunday_saturday_ratio_is_two_to_one_l203_203422


namespace tax_calculation_l203_203147

theorem tax_calculation 
  (total_earnings : ℕ) 
  (deductions : ℕ) 
  (tax_paid : ℕ) 
  (tax_rate_10 : ℚ) 
  (tax_rate_20 : ℚ) 
  (taxable_income : ℕ)
  (X : ℕ)
  (h_total_earnings : total_earnings = 100000)
  (h_deductions : deductions = 30000)
  (h_tax_paid : tax_paid = 12000)
  (h_tax_rate_10 : tax_rate_10 = 10 / 100)
  (h_tax_rate_20 : tax_rate_20 = 20 / 100)
  (h_taxable_income : taxable_income = total_earnings - deductions)
  (h_tax_equation : tax_paid = (tax_rate_10 * X) + (tax_rate_20 * (taxable_income - X))) :
  X = 20000 := 
sorry

end tax_calculation_l203_203147


namespace value_of_m_div_x_l203_203479

variables (a b : ℝ) (k : ℝ)
-- Condition: The ratio of a to b is 4 to 5
def ratio_a_to_b : Prop := a / b = 4 / 5

-- Condition: x equals a increased by 75 percent of a
def x := a + 0.75 * a

-- Condition: m equals b decreased by 80 percent of b
def m := b - 0.80 * b

-- Prove the given question
theorem value_of_m_div_x (h1 : ratio_a_to_b a b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  m / x = 1 / 7 := by
sorry

end value_of_m_div_x_l203_203479


namespace clyde_picked_bushels_l203_203834

theorem clyde_picked_bushels (weight_per_bushel : ℕ) (weight_per_cob : ℕ) (cobs_picked : ℕ) :
  weight_per_bushel = 56 →
  weight_per_cob = 1 / 2 →
  cobs_picked = 224 →
  cobs_picked * weight_per_cob / weight_per_bushel = 2 :=
by
  intros
  sorry

end clyde_picked_bushels_l203_203834


namespace solve_quadratic_l203_203891

theorem solve_quadratic :
  (x = 0 ∨ x = 2/5) ↔ (5 * x^2 - 2 * x = 0) :=
by
  sorry

end solve_quadratic_l203_203891


namespace max_value_function_max_value_expression_l203_203580

theorem max_value_function (x a : ℝ) (hx : x > 0) (ha : a > 2 * x) : ∃ y : ℝ, y = (a^2) / 8 :=
by
  sorry

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 4) : 
   ∃ m : ℝ, m = 4 :=
by
  sorry

end max_value_function_max_value_expression_l203_203580


namespace num_squares_in_6x6_grid_l203_203431

/-- Define the number of kxk squares in an nxn grid -/
def num_squares (n k : ℕ) : ℕ := (n + 1 - k) * (n + 1 - k)

/-- Prove the total number of different squares in a 6x6 grid is 86 -/
theorem num_squares_in_6x6_grid : 
  (num_squares 6 1) + (num_squares 6 2) + (num_squares 6 3) + (num_squares 6 4) = 86 :=
by sorry

end num_squares_in_6x6_grid_l203_203431


namespace problem_conditions_l203_203581

theorem problem_conditions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 3) :
  (x * y ≤ 9 / 8) ∧ (4 ^ x + 2 ^ y ≥ 4 * Real.sqrt 2) ∧ (x / y + 1 / x ≥ 2 / 3 + 2 * Real.sqrt 3 / 3) :=
by
  -- Proof goes here
  sorry

end problem_conditions_l203_203581


namespace apples_per_slice_is_two_l203_203095

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l203_203095


namespace loisa_savings_l203_203930

namespace SavingsProof

def cost_cash : ℤ := 450
def down_payment : ℤ := 100
def payment_first_4_months : ℤ := 4 * 40
def payment_next_4_months : ℤ := 4 * 35
def payment_last_4_months : ℤ := 4 * 30

def total_installment_payment : ℤ :=
  down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

theorem loisa_savings :
  (total_installment_payment - cost_cash) = 70 := by
  sorry

end SavingsProof

end loisa_savings_l203_203930


namespace erin_days_to_receive_30_l203_203685

theorem erin_days_to_receive_30 (x : ℕ) (h : 3 * x = 30) : x = 10 :=
by
  sorry

end erin_days_to_receive_30_l203_203685


namespace total_time_is_three_hours_l203_203598

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end total_time_is_three_hours_l203_203598


namespace sum_of_products_equal_l203_203443

theorem sum_of_products_equal 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h1 : a1 + a2 + a3 = b1 + b2 + b3)
  (h2 : b1 + b2 + b3 = c1 + c2 + c3)
  (h3 : c1 + c2 + c3 = a1 + b1 + c1)
  (h4 : a1 + b1 + c1 = a2 + b2 + c2)
  (h5 : a2 + b2 + c2 = a3 + b3 + c3) :
  a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3 = a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 :=
by 
  sorry

end sum_of_products_equal_l203_203443


namespace circular_permutation_divisible_41_l203_203362

theorem circular_permutation_divisible_41 (N : ℤ) (a b c d e : ℤ) (h : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
  (h41 : 41 ∣ N) :
  ∀ (k : ℕ), 41 ∣ (10^((k % 5) * (4 - (k / 5))) * a + 10^((k % 5) * 3 + (k / 5) * 4) * b + 10^((k % 5) * 2 + (k / 5) * 3) * c + 10^((k % 5) + (k / 5) * 2) * d + 10^(k / 5) * e) :=
sorry

end circular_permutation_divisible_41_l203_203362


namespace angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l203_203903

variable (a b c A B C : ℝ)

-- Condition 1
def cond1 : Prop := b / a = (Real.cos B + 1) / (Real.sqrt 3 * Real.sin A)

-- Condition 2
def cond2 : Prop := 2 * b * Real.sin A = a * Real.tan B

-- Condition 3
def cond3 : Prop := (c - a = b * Real.cos A - a * Real.cos B)

-- Angle B and area of the triangle for Condition 1
theorem angle_B_cond1 (h : cond1 a b A B) : B = π / 3 := sorry

theorem area_range_cond1 (h : cond1 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 2
theorem angle_B_cond2 (h : cond2 a b A B) : B = π / 3 := sorry

theorem area_range_cond2 (h : cond2 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 3
theorem angle_B_cond3 (h : cond3 a b c A B) : B = π / 3 := sorry

theorem area_range_cond3 (h : cond3 a b c A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

end angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l203_203903


namespace eval_expression_l203_203799

theorem eval_expression : (538 * 538) - (537 * 539) = 1 :=
by
  sorry

end eval_expression_l203_203799


namespace convert_cylindrical_to_rectangular_l203_203576

theorem convert_cylindrical_to_rectangular (r θ z x y : ℝ) (h_r : r = 5) (h_θ : θ = (3 * Real.pi) / 2) (h_z : z = 4)
    (h_x : x = r * Real.cos θ) (h_y : y = r * Real.sin θ) :
    (x, y, z) = (0, -5, 4) :=
by
    sorry

end convert_cylindrical_to_rectangular_l203_203576


namespace smallest_collected_l203_203303

noncomputable def Yoongi_collections : ℕ := 4
noncomputable def Jungkook_collections : ℕ := 6 / 3
noncomputable def Yuna_collections : ℕ := 5

theorem smallest_collected : min (min Yoongi_collections Jungkook_collections) Yuna_collections = 2 :=
by
  sorry

end smallest_collected_l203_203303


namespace fraction_transformation_l203_203532

variables (a b : ℝ)

theorem fraction_transformation (ha : a ≠ 0) (hb : b ≠ 0) : 
  (4 * a * b) / (2 * (2 * a) + 2 * b) = 2 * (a * b) / (2 * a + b) :=
by
  sorry

end fraction_transformation_l203_203532


namespace tan_sin_cos_ratio_l203_203991

open Real

variable {α β : ℝ}

theorem tan_sin_cos_ratio (h1 : tan (α + β) = 2) (h2 : tan (α - β) = 3) :
  sin (2 * α) / cos (2 * β) = 5 / 7 := sorry

end tan_sin_cos_ratio_l203_203991


namespace benny_books_l203_203146

variable (B : ℕ) -- the number of books Benny had initially

theorem benny_books (h : B - 10 + 33 = 47) : B = 24 :=
sorry

end benny_books_l203_203146


namespace floor_abs_sum_eq_501_l203_203039

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end floor_abs_sum_eq_501_l203_203039


namespace necessarily_positive_expressions_l203_203051

theorem necessarily_positive_expressions
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  (b + b^2 > 0) ∧ (b + 3 * b^2 > 0) :=
sorry

end necessarily_positive_expressions_l203_203051


namespace modular_expression_problem_l203_203843

theorem modular_expression_problem
  (m : ℕ)
  (hm : 0 ≤ m ∧ m < 29)
  (hmod : 4 * m % 29 = 1) :
  (5^m % 29)^4 - 3 % 29 = 13 % 29 :=
by
  sorry

end modular_expression_problem_l203_203843


namespace driving_time_equation_l203_203877

theorem driving_time_equation :
  ∀ (t : ℝ), (60 * t + 90 * (3.5 - t) = 300) :=
by
  intro t
  sorry

end driving_time_equation_l203_203877


namespace coterminal_angle_neg_60_eq_300_l203_203302

theorem coterminal_angle_neg_60_eq_300 :
  ∃ k : ℤ, 0 ≤ k * 360 - 60 ∧ k * 360 - 60 < 360 ∧ (k * 360 - 60 = 300) := by
  sorry

end coterminal_angle_neg_60_eq_300_l203_203302


namespace marble_count_l203_203280

variable (initial_mar: Int) (lost_mar: Int)

def final_mar (initial_mar: Int) (lost_mar: Int) : Int :=
  initial_mar - lost_mar

theorem marble_count : final_mar 16 7 = 9 := by
  trivial

end marble_count_l203_203280


namespace locus_of_vertex_P_l203_203324

noncomputable def M : ℝ × ℝ := (0, 5)
noncomputable def N : ℝ × ℝ := (0, -5)
noncomputable def perimeter : ℝ := 36

theorem locus_of_vertex_P : ∃ (P : ℝ × ℝ), 
  (∃ (a b : ℝ), a = 13 ∧ b = 12 ∧ P ≠ (0,0) ∧
  (a^2 = b^2 + 5^2) ∧ 
  (perimeter = 2 * a + (5 - (-5))) ∧ 
  ((P.1)^2 / 144 + (P.2)^2 / 169 = 1)) :=
sorry

end locus_of_vertex_P_l203_203324


namespace number_of_chinese_l203_203432

theorem number_of_chinese (total americans australians chinese : ℕ) 
    (h_total : total = 49)
    (h_americans : americans = 16)
    (h_australians : australians = 11)
    (h_chinese : chinese = total - americans - australians) :
    chinese = 22 :=
by
    rw [h_total, h_americans, h_australians] at h_chinese
    exact h_chinese

end number_of_chinese_l203_203432


namespace bella_steps_l203_203566

-- Define the conditions and the necessary variables
variable (b : ℝ) (distance : ℝ) (steps_per_foot : ℝ)

-- Given constants
def bella_speed := b
def ella_speed := 4 * b
def combined_speed := bella_speed + ella_speed
def total_distance := 15840
def feet_per_step := 3

-- Define the main theorem to prove the number of steps Bella takes
theorem bella_steps : (total_distance / combined_speed) * bella_speed / feet_per_step = 1056 := by
  sorry

end bella_steps_l203_203566


namespace range_of_a_l203_203268

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (-x)

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (h_ineq : f a (-2) > f a (-3)) : 0 < a ∧ a < 1 :=
by {
  sorry
}

end range_of_a_l203_203268


namespace face_opposite_to_turquoise_is_pink_l203_203810

-- Declare the inductive type for the color of the face
inductive Color
| P -- Pink
| V -- Violet
| T -- Turquoise
| O -- Orange

open Color

-- Define the setup conditions of the problem
def cube_faces : List Color :=
  [P, P, P, V, V, T, O]

-- Define the positions of the faces for the particular folded cube configuration
-- Assuming the function cube_configuration gives the face opposite to a given face.
axiom cube_configuration : Color → Color

-- State the main theorem regarding the opposite face
theorem face_opposite_to_turquoise_is_pink : cube_configuration T = P :=
sorry

end face_opposite_to_turquoise_is_pink_l203_203810


namespace find_c_l203_203215

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 3 - 2 * x - 6 * x^3 + 7 * x^4

-- Define the main theorem stating that c = -5/7 makes f(x) + c*g(x) have degree 3
theorem find_c (c : ℝ) (h : ∀ x : ℝ, f x + c * g x = 0) : c = -5 / 7 := by
  sorry

end find_c_l203_203215


namespace factorize_expression_l203_203442

theorem factorize_expression (x : ℝ) : 
  x^3 - 5 * x^2 + 4 * x = x * (x - 1) * (x - 4) :=
by
  sorry

end factorize_expression_l203_203442


namespace milk_required_for_flour_l203_203945

theorem milk_required_for_flour (flour_ratio milk_ratio total_flour : ℕ) : 
  (milk_ratio * (total_flour / flour_ratio)) = 160 :=
by
  let milk_ratio := 40
  let flour_ratio := 200
  let total_flour := 800
  exact sorry

end milk_required_for_flour_l203_203945


namespace floor_e_sub_6_eq_neg_4_l203_203236

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4_l203_203236


namespace some_employees_not_managers_l203_203869

-- Definitions of the conditions
def isEmployee : Type := sorry
def isManager : isEmployee → Prop := sorry
def isShareholder : isEmployee → Prop := sorry
def isPunctual : isEmployee → Prop := sorry

-- Given conditions
axiom some_employees_not_punctual : ∃ e : isEmployee, ¬isPunctual e
axiom all_managers_punctual : ∀ m : isEmployee, isManager m → isPunctual m
axiom some_managers_shareholders : ∃ m : isEmployee, isManager m ∧ isShareholder m

-- The statement to be proved
theorem some_employees_not_managers : ∃ e : isEmployee, ¬isManager e :=
by sorry

end some_employees_not_managers_l203_203869


namespace find_values_of_a_and_b_l203_203835

-- Define the problem
theorem find_values_of_a_and_b (a b : ℚ) (h1 : a + (a / 4) = 3) (h2 : b - 2 * a = 1) :
  a = 12 / 5 ∧ b = 29 / 5 := by
  sorry

end find_values_of_a_and_b_l203_203835


namespace number_of_tables_l203_203054

-- Defining the given parameters
def linen_cost : ℕ := 25
def place_setting_cost : ℕ := 10
def rose_cost : ℕ := 5
def lily_cost : ℕ := 4
def num_place_settings : ℕ := 4
def num_roses : ℕ := 10
def num_lilies : ℕ := 15
def total_decoration_cost : ℕ := 3500

-- Defining the cost per table
def cost_per_table : ℕ := linen_cost + (num_place_settings * place_setting_cost) + (num_roses * rose_cost) + (num_lilies * lily_cost)

-- Proof problem statement: Proving number of tables is 20
theorem number_of_tables : (total_decoration_cost / cost_per_table) = 20 :=
by
  sorry

end number_of_tables_l203_203054


namespace prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l203_203384

noncomputable def P_A : ℝ := 0.5
noncomputable def P_B_not_A : ℝ := 0.3
noncomputable def P_B : ℝ := 0.6  -- given from solution step
noncomputable def P_C : ℝ := 1 - (1 - P_A) * (1 - P_B)
noncomputable def P_D : ℝ := (1 - P_A) * (1 - P_B)
noncomputable def P_E : ℝ := 3 * P_D * (P_C ^ 2)

theorem prob_insurance_A_or_B :
  P_C = 0.8 :=
by
  sorry

theorem prob_exactly_one_no_insurance_out_of_three :
  P_E = 0.384 :=
by
  sorry

end prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l203_203384


namespace people_not_in_pool_l203_203969

noncomputable def total_people_karen_donald : ℕ := 2
noncomputable def children_karen_donald : ℕ := 6
noncomputable def total_people_tom_eva : ℕ := 2
noncomputable def children_tom_eva : ℕ := 4
noncomputable def legs_in_pool : ℕ := 16

theorem people_not_in_pool : total_people_karen_donald + children_karen_donald + total_people_tom_eva + children_tom_eva - (legs_in_pool / 2) = 6 := by
  sorry

end people_not_in_pool_l203_203969


namespace max_min_f_product_of_roots_f_l203_203629

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log x / Real.log 3 - 3) * (Real.log x / Real.log 3 + 1)

theorem max_min_f
  (x : ℝ) (h : x ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ)) : 
  (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≤ 12)
  ∧ (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≥ 5) :=
sorry

theorem product_of_roots_f
  (m α β : ℝ) (h1 : f α + m = 0) (h2 : f β + m = 0) : 
  (Real.log (α * β) / Real.log 3 = 2) → (α * β = 9) :=
sorry

end max_min_f_product_of_roots_f_l203_203629


namespace midlines_tangent_fixed_circle_l203_203638

-- Definitions of geometric objects and properties
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Assumptions (conditions)
variable (ω1 ω2 : Circle)
variable (l1 l2 : Point → Prop) -- Representing line equations in terms of points
variable (angle : Point → Prop) -- Representing the given angle sides

-- Tangency conditions
axiom tangency1 : ∀ p : Point, l1 p → p ≠ ω1.center ∧ (ω1.center.x - p.x) ^ 2 + (ω1.center.y - p.y) ^ 2 = ω1.radius ^ 2
axiom tangency2 : ∀ p : Point, l2 p → p ≠ ω2.center ∧ (ω2.center.x - p.x) ^ 2 + (ω2.center.y - p.y) ^ 2 = ω2.radius ^ 2

-- Non-intersecting condition for circles
axiom nonintersecting : (ω1.center.x - ω2.center.x) ^ 2 + (ω1.center.y - ω2.center.y) ^ 2 > (ω1.radius + ω2.radius) ^ 2

-- Conditions for tangent circles and middle line being between them
axiom betweenness : ∀ p, angle p → (ω1.center.y < p.y ∧ p.y < ω2.center.y)

-- Midline definition and fixed circle condition
theorem midlines_tangent_fixed_circle :
  ∃ (O : Point) (d : ℝ), ∀ (T : Point → Prop), 
  (∃ (p1 p2 : Point), l1 p1 ∧ l2 p2 ∧ T p1 ∧ T p2) →
  (∀ (m : Point), T m ↔ ∃ (p1 p2 p3 p4 : Point), T p1 ∧ T p2 ∧ angle p3 ∧ angle p4 ∧ 
  m.x = (p1.x + p2.x + p3.x + p4.x) / 4 ∧ m.y = (p1.y + p2.y + p3.y + p4.y) / 4) → 
  (∀ (m : Point), (m.x - O.x) ^ 2 + (m.y - O.y) ^ 2 = d^2)
:= 
sorry

end midlines_tangent_fixed_circle_l203_203638


namespace intersection_M_N_l203_203814

def M : Set ℝ := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l203_203814


namespace clark_discount_l203_203294

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l203_203294


namespace problem_I_problem_II_l203_203780

-- Problem (I)
theorem problem_I (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) : 
  ∀ x, (f x < |x| + 1) → (0 < x ∧ x < 2) :=
by
  intro x hx
  have fx_def : f x = |2 * x - 1| := h x
  sorry

-- Problem (II)
theorem problem_II (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) :
  ∀ x y, (|x - y - 1| ≤ 1 / 3) → (|2 * y + 1| ≤ 1 / 6) → (f x ≤ 5 / 6) :=
by
  intro x y hx hy
  have fx_def : f x = |2 * x - 1| := h x
  sorry

end problem_I_problem_II_l203_203780


namespace sine_cosine_fraction_l203_203958

theorem sine_cosine_fraction (θ : ℝ) (h : Real.tan θ = 2) : 
    (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
by 
  sorry

end sine_cosine_fraction_l203_203958


namespace minimum_seats_occupied_l203_203386

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied_l203_203386


namespace train_lengths_l203_203856

noncomputable def train_problem : Prop :=
  let speed_T1_mps := 54 * (5/18)
  let speed_T2_mps := 72 * (5/18)
  let L_T1 := speed_T1_mps * 20
  let L_p := (speed_T1_mps * 44) - L_T1
  let L_T2 := speed_T2_mps * 16
  (L_p = 360) ∧ (L_T1 = 300) ∧ (L_T2 = 320)

theorem train_lengths : train_problem := sorry

end train_lengths_l203_203856


namespace probability_of_both_contracts_l203_203881

open Classical

variable (P_A P_B' P_A_or_B P_A_and_B : ℚ)

noncomputable def probability_hardware_contract := P_A = 3 / 4
noncomputable def probability_not_software_contract := P_B' = 5 / 9
noncomputable def probability_either_contract := P_A_or_B = 4 / 5
noncomputable def probability_both_contracts := P_A_and_B = 71 / 180

theorem probability_of_both_contracts {P_A P_B' P_A_or_B P_A_and_B : ℚ} :
  probability_hardware_contract P_A →
  probability_not_software_contract P_B' →
  probability_either_contract P_A_or_B →
  probability_both_contracts P_A_and_B :=
by
  intros
  sorry

end probability_of_both_contracts_l203_203881


namespace exists_smaller_circle_with_at_least_as_many_lattice_points_l203_203730

theorem exists_smaller_circle_with_at_least_as_many_lattice_points
  (R : ℝ) (hR : 0 < R) :
  ∃ R' : ℝ, (R' < R) ∧ (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 → ∃ (x' y' : ℤ), (x')^2 + (y')^2 ≤ (R')^2) := sorry

end exists_smaller_circle_with_at_least_as_many_lattice_points_l203_203730


namespace chocolates_remaining_l203_203725

theorem chocolates_remaining 
  (total_chocolates : ℕ)
  (ate_day1 : ℕ) (ate_day2 : ℕ) (ate_day3 : ℕ) (ate_day4 : ℕ) (ate_day5 : ℕ) (remaining_chocolates : ℕ) 
  (h_total : total_chocolates = 48)
  (h_day1 : ate_day1 = 6) 
  (h_day2 : ate_day2 = 2 * ate_day1 + 2) 
  (h_day3 : ate_day3 = ate_day1 - 3) 
  (h_day4 : ate_day4 = 2 * ate_day3 + 1) 
  (h_day5 : ate_day5 = ate_day2 / 2) 
  (h_rem : remaining_chocolates = total_chocolates - (ate_day1 + ate_day2 + ate_day3 + ate_day4 + ate_day5)) :
  remaining_chocolates = 14 :=
sorry

end chocolates_remaining_l203_203725


namespace MrMartinBought2Cups_l203_203584

theorem MrMartinBought2Cups (c b : ℝ) (x : ℝ) (h1 : 3 * c + 2 * b = 12.75)
                             (h2 : x * c + 5 * b = 14.00)
                             (hb : b = 1.5) :
  x = 2 :=
sorry

end MrMartinBought2Cups_l203_203584


namespace find_x_l203_203492

theorem find_x :
  let a := 5^3
  let b := 6^2
  a - 7 = b + 82 := 
by
  sorry

end find_x_l203_203492


namespace triangle_base_length_l203_203735

theorem triangle_base_length :
  ∀ (base height area : ℕ), height = 4 → area = 16 → area = (base * height) / 2 → base = 8 :=
by
  intros base height area h_height h_area h_formula
  sorry

end triangle_base_length_l203_203735


namespace train_pass_platform_time_l203_203365

-- Define the conditions given in the problem.
def train_length : ℕ := 1200
def platform_length : ℕ := 1100
def time_to_cross_tree : ℕ := 120

-- Define the calculation for speed.
def speed := train_length / time_to_cross_tree

-- Define the combined length of train and platform.
def combined_length := train_length + platform_length

-- Define the expected time to pass the platform.
def expected_time_to_pass_platform := combined_length / speed

-- The theorem to prove.
theorem train_pass_platform_time :
  expected_time_to_pass_platform = 230 :=
by {
  -- Placeholder for the proof.
  sorry
}

end train_pass_platform_time_l203_203365


namespace min_value_x_squared_plus_y_squared_plus_z_squared_l203_203053

theorem min_value_x_squared_plus_y_squared_plus_z_squared (x y z : ℝ) (h : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 :=
by
  sorry

end min_value_x_squared_plus_y_squared_plus_z_squared_l203_203053


namespace value_of_abc_l203_203371

theorem value_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1 / b = 5) (h2 : b + 1 / c = 2) (h3 : c + 1 / a = 3) : 
  abc = 1 :=
by
  sorry

end value_of_abc_l203_203371


namespace race_permutations_l203_203882

-- Define the number of participants
def num_participants : ℕ := 4

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n + 1) * factorial n

-- Theorem: Given 4 participants, the number of different possible orders they can finish the race is 24.
theorem race_permutations : factorial num_participants = 24 := by
  -- sorry added to skip the proof
  sorry

end race_permutations_l203_203882


namespace intersection_A_B_l203_203507

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  sorry

end intersection_A_B_l203_203507


namespace area_of_rectangular_field_l203_203710

theorem area_of_rectangular_field (length width : ℝ) (h_length: length = 5.9) (h_width: width = 3) : 
  length * width = 17.7 := 
by
  sorry

end area_of_rectangular_field_l203_203710


namespace real_solutions_count_l203_203481

theorem real_solutions_count :
  ∃ S : Set ℝ, (∀ x : ℝ, x ∈ S ↔ (|x-2| + |x-3| = 1)) ∧ (S = Set.Icc 2 3) :=
sorry

end real_solutions_count_l203_203481


namespace zero_function_l203_203824

variable (f : ℝ × ℝ × ℝ → ℝ)

theorem zero_function (h : ∀ x y z : ℝ, f (x, y, z) = 2 * f (z, x, y)) : ∀ x y z : ℝ, f (x, y, z) = 0 :=
by
  intros
  sorry

end zero_function_l203_203824


namespace abs_ineq_real_solution_range_l203_203267

theorem abs_ineq_real_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x + 3| < a) ↔ a > 7 :=
sorry

end abs_ineq_real_solution_range_l203_203267


namespace ganesh_average_speed_l203_203666

variable (D : ℝ) -- distance between the two towns in kilometers
variable (V : ℝ) -- average speed from x to y in km/hr

-- Conditions
variable (h1 : V > 0) -- Speed must be positive
variable (h2 : 30 > 0) -- Speed must be positive
variable (h3 : 40 = (2 * D) / ((D / V) + (D / 30))) -- Average speed formula

theorem ganesh_average_speed : V = 60 :=
by {
  sorry
}

end ganesh_average_speed_l203_203666


namespace Jose_share_land_l203_203657

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l203_203657


namespace maximize_profit_l203_203662

def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def sales_volume (x : ℝ) : ℝ := (12 - x)^2 * 10000
def annual_profit (x : ℝ) : ℝ := (x - cost_per_product - management_fee_per_product) * sales_volume x

theorem maximize_profit :
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x = x^3 - 30*x^2 + 288*x - 864) ∧
  annual_profit 9 = 27 * 10000 ∧
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x ≤ annual_profit 9) :=
by
  sorry

end maximize_profit_l203_203662


namespace josanna_minimum_test_score_l203_203154

theorem josanna_minimum_test_score 
  (scores : List ℕ) (target_increase : ℕ) (new_score : ℕ)
  (h_scores : scores = [92, 78, 84, 76, 88]) 
  (h_target_increase : target_increase = 5):
  (List.sum scores + new_score) / (List.length scores + 1) ≥ (List.sum scores / List.length scores + target_increase) →
  new_score = 114 :=
by
  sorry

end josanna_minimum_test_score_l203_203154


namespace geometric_series_sum_l203_203447

noncomputable def geometric_sum : ℚ :=
  let a := (2^3 : ℚ) / (3^3)
  let r := (2 : ℚ) / 3
  let n := 12 - 3 + 1
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum = 1440600 / 59049 :=
by
  sorry

end geometric_series_sum_l203_203447


namespace squares_difference_sum_l203_203045

theorem squares_difference_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by 
  sorry

end squares_difference_sum_l203_203045


namespace sequence_increasing_or_decreasing_l203_203899

theorem sequence_increasing_or_decreasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) 
  (hrec : ∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∀ n, x n < x (n + 1) ∨ x n > x (n + 1) :=
by
  sorry

end sequence_increasing_or_decreasing_l203_203899


namespace total_cost_of_breakfast_l203_203784

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l203_203784


namespace cos_double_angle_l203_203837

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l203_203837


namespace one_percent_as_decimal_l203_203663

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := 
by 
  sorry

end one_percent_as_decimal_l203_203663


namespace smallest_sector_angle_3_l203_203040

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_angles_is_360 (a : ℕ → ℕ) : Prop :=
  (Finset.range 15).sum a = 360

def smallest_possible_angle (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ i : ℕ, a i ≥ x

theorem smallest_sector_angle_3 :
  ∃ a : ℕ → ℕ,
    is_arithmetic_sequence a ∧
    sum_of_angles_is_360 a ∧
    smallest_possible_angle a 3 :=
sorry

end smallest_sector_angle_3_l203_203040


namespace roots_of_quadratic_eq_l203_203626

theorem roots_of_quadratic_eq (a b : ℝ) (h1 : a * (-2)^2 + b * (-2) = 6) (h2 : a * 3^2 + b * 3 = 6) :
    ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 3 ∧ ∀ x, a * x^2 + b * x = 6 ↔ (x = x1 ∨ x = x2) :=
by
  use -2, 3
  sorry

end roots_of_quadratic_eq_l203_203626


namespace length_of_unfenced_side_l203_203380

theorem length_of_unfenced_side
  (L W : ℝ)
  (h1 : L * W = 200)
  (h2 : 2 * W + L = 50) :
  L = 10 :=
sorry

end length_of_unfenced_side_l203_203380


namespace quadratic_sum_constants_l203_203066

theorem quadratic_sum_constants (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x = -3 ∨ x = 5)
  (h_min : ∀ x, a * x^2 + b * x + c ≥ 36) 
  (h_at : a * 1^2 + b * 1 + c = 36) :
  a + b + c = 36 :=
sorry

end quadratic_sum_constants_l203_203066


namespace solution_set_ineq_l203_203660

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_ineq (x : ℝ) : f (x^2 - 4) + f (3*x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end solution_set_ineq_l203_203660


namespace triangle_AC_range_l203_203084

noncomputable def length_AB : ℝ := 12
noncomputable def length_CD : ℝ := 6

def is_valid_AC (AC : ℝ) : Prop :=
  AC > 6 ∧ AC < 24

theorem triangle_AC_range :
  ∃ m n : ℝ, 
    (6 < m ∧ m < 24) ∧ (6 < n ∧ n < 24) ∧
    m + n = 30 ∧
    ∀ AC : ℝ, is_valid_AC AC →
      6 < AC ∧ AC < 24 :=
by
  use 6
  use 24
  simp
  sorry

end triangle_AC_range_l203_203084


namespace project_completion_time_l203_203821

theorem project_completion_time (x : ℕ) :
  (∀ (B_days : ℕ), B_days = 40 →
  (∀ (combined_work_days : ℕ), combined_work_days = 10 →
  (∀ (total_days : ℕ), total_days = 20 →
  10 * (1 / (x : ℚ) + 1 / 40) + 10 * (1 / 40) = 1))) →
  x = 20 :=
by
  sorry

end project_completion_time_l203_203821


namespace sequence_polynomial_exists_l203_203305

noncomputable def sequence_exists (k : ℕ) : Prop :=
∃ u : ℕ → ℝ,
  (∀ n : ℕ, u (n + 1) - u n = (n : ℝ) ^ k) ∧
  (∃ p : Polynomial ℝ, (∀ n : ℕ, u n = Polynomial.eval (n : ℝ) p) ∧ p.degree = k + 1 ∧ p.leadingCoeff = 1 / (k + 1))

theorem sequence_polynomial_exists (k : ℕ) : sequence_exists k :=
sorry

end sequence_polynomial_exists_l203_203305


namespace triangle_identity_l203_203266

variables (a b c h_a h_b h_c x y z : ℝ)

-- Define the given conditions
def condition1 := a / h_a = x
def condition2 := b / h_b = y
def condition3 := c / h_c = z

-- Statement of the theorem to be proved
theorem triangle_identity 
  (h1 : condition1 a h_a x) 
  (h2 : condition2 b h_b y) 
  (h3 : condition3 c h_c z) : 
  x^2 + y^2 + z^2 - 2 * x * y - 2 * y * z - 2 * z * x + 4 = 0 := 
  by 
    sorry

end triangle_identity_l203_203266


namespace natasha_average_speed_l203_203855

theorem natasha_average_speed
  (time_up time_down : ℝ)
  (speed_up distance_up total_distance total_time average_speed : ℝ)
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 3)
  (h4 : distance_up = speed_up * time_up)
  (h5 : total_distance = distance_up + distance_up)
  (h6 : total_time = time_up + time_down)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 4 := by
  sorry

end natasha_average_speed_l203_203855


namespace roots_polynomial_expression_l203_203436

theorem roots_polynomial_expression (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a * b + a * c + b * c = -1)
  (h3 : a * b * c = -2) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 0 :=
by
  sorry

end roots_polynomial_expression_l203_203436


namespace base_of_parallelogram_l203_203554

theorem base_of_parallelogram (Area Height : ℕ) (h1 : Area = 44) (h2 : Height = 11) : (Area / Height) = 4 :=
by
  sorry

end base_of_parallelogram_l203_203554


namespace range_of_4x_2y_l203_203611

theorem range_of_4x_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) :
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 := 
sorry

end range_of_4x_2y_l203_203611


namespace phone_call_answered_within_first_four_rings_l203_203926

def P1 := 0.1
def P2 := 0.3
def P3 := 0.4
def P4 := 0.1

theorem phone_call_answered_within_first_four_rings :
  P1 + P2 + P3 + P4 = 0.9 :=
by
  rw [P1, P2, P3, P4]
  norm_num
  sorry -- Proof step skipped

end phone_call_answered_within_first_four_rings_l203_203926


namespace jennie_rental_cost_is_306_l203_203093

-- Definitions for the given conditions
def weekly_rate_mid_size : ℕ := 190
def daily_rate_mid_size_upto10 : ℕ := 25
def total_rental_days : ℕ := 13
def coupon_discount : ℝ := 0.10

-- Define the cost calculation
def rental_cost (days : ℕ) : ℕ :=
  let weeks := days / 7
  let extra_days := days % 7
  let cost_weeks := weeks * weekly_rate_mid_size
  let cost_extra := extra_days * daily_rate_mid_size_upto10
  cost_weeks + cost_extra

def discount (total : ℝ) (rate : ℝ) : ℝ := total * rate

def final_amount (initial_amount : ℝ) (discount_amount : ℝ) : ℝ := initial_amount - discount_amount

-- Main theorem to prove the final payment amount
theorem jennie_rental_cost_is_306 : 
  final_amount (rental_cost total_rental_days) (discount (rental_cost total_rental_days) coupon_discount) = 306 := 
by
  sorry

end jennie_rental_cost_is_306_l203_203093


namespace omitted_angle_of_convex_polygon_l203_203358

theorem omitted_angle_of_convex_polygon (calculated_sum : ℕ) (omitted_angle : ℕ)
    (h₁ : calculated_sum = 2583) (h₂ : omitted_angle = 2700 - 2583) :
    omitted_angle = 117 :=
by
  sorry

end omitted_angle_of_convex_polygon_l203_203358


namespace least_whole_number_clock_equivalent_l203_203232

theorem least_whole_number_clock_equivalent :
  ∃ h : ℕ, h > 6 ∧ h ^ 2 % 24 = h % 24 ∧ ∀ k : ℕ, k > 6 ∧ k ^ 2 % 24 = k % 24 → h ≤ k := sorry

end least_whole_number_clock_equivalent_l203_203232


namespace number_of_ways_to_express_n_as_sum_l203_203378

noncomputable def P (n k : ℕ) : ℕ := sorry
noncomputable def Q (n k : ℕ) : ℕ := sorry

theorem number_of_ways_to_express_n_as_sum (n : ℕ) (k : ℕ) (h : k ≥ 2) : P n k = Q n k := sorry

end number_of_ways_to_express_n_as_sum_l203_203378


namespace tennis_players_l203_203335

theorem tennis_players (total_members badminton_players neither_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : neither_players = 7)
  (h4 : both_players = 21) :
  total_members - neither_players = badminton_players - both_players + (total_members - neither_players - badminton_players + both_players) + both_players →
  ((total_members - neither_players) - (badminton_players - both_players) - both_players) + both_players = 46 :=
by
  intros h
  sorry

end tennis_players_l203_203335


namespace number_of_handshakes_l203_203339

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l203_203339


namespace smallest_number_of_players_l203_203142

theorem smallest_number_of_players :
  ∃ n, n ≡ 1 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 4 [MOD 6] ∧ ∃ m, n = m * m ∧ ∀ k, (k ≡ 1 [MOD 3] ∧ k ≡ 2 [MOD 4] ∧ k ≡ 4 [MOD 6] ∧ ∃ m, k = m * m) → k ≥ n :=
sorry

end smallest_number_of_players_l203_203142


namespace maya_additional_cars_l203_203671

theorem maya_additional_cars : 
  ∃ n : ℕ, 29 + n ≥ 35 ∧ (29 + n) % 7 = 0 ∧ n = 6 :=
by
  sorry

end maya_additional_cars_l203_203671


namespace cassie_and_brian_meet_at_1111am_l203_203401

theorem cassie_and_brian_meet_at_1111am :
  ∃ t : ℕ, t = 11*60 + 11 ∧
    (∃ x : ℚ, x = 51/16 ∧ 
      14 * x + 18 * (x - 1) = 84) :=
sorry

end cassie_and_brian_meet_at_1111am_l203_203401


namespace find_deeper_depth_l203_203975

noncomputable def swimming_pool_depth_proof 
  (width : ℝ) (length : ℝ) (shallow_depth : ℝ) (volume : ℝ) : Prop :=
  volume = (1 / 2) * (shallow_depth + 4) * width * length

theorem find_deeper_depth
  (h : width = 9)
  (l : length = 12)
  (a : shallow_depth = 1)
  (V : volume = 270) :
  swimming_pool_depth_proof 9 12 1 270 := by
  sorry

end find_deeper_depth_l203_203975


namespace spanish_teams_in_final_probability_l203_203508

noncomputable def probability_of_spanish_teams_in_final : ℚ :=
  let teams := 16
  let spanish_teams := 3
  let non_spanish_teams := teams - spanish_teams
  -- Probability calculation based on given conditions and solution steps
  1 - 7 / 15 * 6 / 14

theorem spanish_teams_in_final_probability :
  probability_of_spanish_teams_in_final = 4 / 5 :=
sorry

end spanish_teams_in_final_probability_l203_203508


namespace mutually_exclusive_A_C_l203_203415

-- Definitions based on the given conditions
def all_not_defective (A : Prop) : Prop := A
def all_defective (B : Prop) : Prop := B
def at_least_one_defective (C : Prop) : Prop := C

-- Theorem to prove A and C are mutually exclusive
theorem mutually_exclusive_A_C (A B C : Prop) 
  (H1 : all_not_defective A) 
  (H2 : all_defective B) 
  (H3 : at_least_one_defective C) : 
  (A ∧ C) → False :=
sorry

end mutually_exclusive_A_C_l203_203415


namespace paintings_on_Sep27_l203_203293

-- Definitions for the problem conditions
def total_days := 6
def paintings_per_2_days := (6 : ℕ)
def paintings_per_3_days := (8 : ℕ)
def paintings_P22_to_P26 := 30

-- Function to calculate paintings over a given period
def paintings_in_days (days : ℕ) (frequency : ℕ) : ℕ := days / frequency

-- Function to calculate total paintings from the given artists
def total_paintings (d : ℕ) (p2 : ℕ) (p3 : ℕ) : ℕ :=
  p2 * paintings_in_days d 2 + p3 * paintings_in_days d 3

-- Calculate total paintings in 6 days
def total_paintings_in_6_days := total_paintings total_days paintings_per_2_days paintings_per_3_days

-- Proof problem: Show the number of paintings on the last day (September 27)
theorem paintings_on_Sep27 : total_paintings_in_6_days - paintings_P22_to_P26 = 4 :=
by
  sorry

end paintings_on_Sep27_l203_203293


namespace determine_H_zero_l203_203769

theorem determine_H_zero (E F G H : ℕ) 
  (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (add_eq : 10 * E + F + 10 * G + E = 10 * H + E)
  (sub_eq : 10 * E + F - (10 * G + E) = E) : 
  H = 0 :=
sorry

end determine_H_zero_l203_203769


namespace lowest_fraction_of_job_in_one_hour_l203_203025

-- Define the rates at which each person can work
def rate_A : ℚ := 1/3
def rate_B : ℚ := 1/4
def rate_C : ℚ := 1/6

-- Define the combined rates for each pair of people
def combined_rate_AB : ℚ := rate_A + rate_B
def combined_rate_AC : ℚ := rate_A + rate_C
def combined_rate_BC : ℚ := rate_B + rate_C

-- The Lean 4 statement to prove
theorem lowest_fraction_of_job_in_one_hour : min combined_rate_AB (min combined_rate_AC combined_rate_BC) = 5/12 :=
by 
  -- Here we state that the minimum combined rate is 5/12
  sorry

end lowest_fraction_of_job_in_one_hour_l203_203025


namespace quadratic_root_unique_l203_203732

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l203_203732


namespace car_average_speed_l203_203195

theorem car_average_speed 
  (d1 d2 d3 d5 d6 d7 d8 : ℝ) 
  (t_total : ℝ) 
  (avg_speed : ℝ)
  (h1 : d1 = 90)
  (h2 : d2 = 50)
  (h3 : d3 = 70)
  (h5 : d5 = 80)
  (h6 : d6 = 60)
  (h7 : d7 = -40)
  (h8 : d8 = -55)
  (h_t_total : t_total = 8)
  (h_avg_speed : avg_speed = (d1 + d2 + d3 + d5 + d6 + d7 + d8) / t_total) :
  avg_speed = 31.875 := 
by sorry

end car_average_speed_l203_203195


namespace mean_calculation_incorrect_l203_203156

theorem mean_calculation_incorrect (a b c : ℝ) (h : a < b) (h1 : b < c) :
  let x := (a + b) / 2
  let y := (x + c) / 2
  y < (a + b + c) / 3 :=
by 
  let x := (a + b) / 2
  let y := (x + c) / 2
  sorry

end mean_calculation_incorrect_l203_203156


namespace total_trees_l203_203372

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l203_203372


namespace problem1_problem2_l203_203742

-- Equivalent proof statement for part (1)
theorem problem1 : 2023^2 - 2022 * 2024 = 1 := by
  sorry

-- Equivalent proof statement for part (2)
theorem problem2 (m : ℝ) (h : m ≠ 1) (h1 : m ≠ -1) : 
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by
  sorry

end problem1_problem2_l203_203742


namespace coin_stack_l203_203952

def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75
def stack_height : ℝ := 14

theorem coin_stack (n_penny n_nickel n_dime n_quarter : ℕ) 
  (h : n_penny * penny_thickness + n_nickel * nickel_thickness + n_dime * dime_thickness + n_quarter * quarter_thickness = stack_height) :
  n_penny + n_nickel + n_dime + n_quarter = 8 :=
sorry

end coin_stack_l203_203952


namespace sum_of_first_six_terms_l203_203520

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms :
  sum_first_six_terms a = 63 / 32 :=
by
  sorry

end sum_of_first_six_terms_l203_203520


namespace employed_females_percentage_l203_203679

-- Definitions of the conditions
def employment_rate : ℝ := 0.60
def male_employment_rate : ℝ := 0.15

-- The theorem to prove
theorem employed_females_percentage : employment_rate - male_employment_rate = 0.45 := by
  sorry

end employed_females_percentage_l203_203679


namespace expected_allergies_correct_expected_both_correct_l203_203273

noncomputable def p_allergies : ℚ := 2 / 7
noncomputable def sample_size : ℕ := 350
noncomputable def expected_allergies : ℚ := (2 / 7) * 350

noncomputable def p_left_handed : ℚ := 3 / 10
noncomputable def expected_both : ℚ := (3 / 10) * (2 / 7) * 350

theorem expected_allergies_correct : expected_allergies = 100 := by
  sorry

theorem expected_both_correct : expected_both = 30 := by
  sorry

end expected_allergies_correct_expected_both_correct_l203_203273


namespace line_intersects_ellipse_two_points_l203_203052

theorem line_intersects_ellipse_two_points {m n : ℝ} (h1 : ¬∃ x y : ℝ, m*x + n*y = 4 ∧ x^2 + y^2 = 4)
  (h2 : m^2 + n^2 < 4) : 
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (m * p1.1 + n * p1.2 = 4) ∧ (m * p2.1 + n * p2.2 = 4) ∧ 
  (p1.1^2 / 9 + p1.2^2 / 4 = 1) ∧ (p2.1^2 / 9 + p2.2^2 / 4 = 1) :=
sorry

end line_intersects_ellipse_two_points_l203_203052


namespace percentage_increase_area_l203_203593

theorem percentage_increase_area (L W : ℝ) :
  let A := L * W
  let L' := 1.20 * L
  let W' := 1.20 * W
  let A' := L' * W'
  let percentage_increase := (A' - A) / A * 100
  L > 0 → W > 0 → percentage_increase = 44 := 
by
  sorry

end percentage_increase_area_l203_203593


namespace increasing_function_greater_at_a_squared_plus_one_l203_203956

variable (f : ℝ → ℝ) (a : ℝ)

def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_greater_at_a_squared_plus_one :
  strictly_increasing f → f (a^2 + 1) > f a :=
by
  sorry

end increasing_function_greater_at_a_squared_plus_one_l203_203956


namespace sum_angles_star_l203_203687

theorem sum_angles_star (β : ℝ) (h : β = 90) : 
  8 * β = 720 :=
by
  sorry

end sum_angles_star_l203_203687


namespace axis_of_symmetry_l203_203907

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : y = - 1 / (4 * a) :=
sorry

end axis_of_symmetry_l203_203907


namespace intersection_of_sets_l203_203086

def set_A (x : ℝ) : Prop := |x - 1| < 3
def set_B (x : ℝ) : Prop := (x - 1) / (x - 5) < 0

theorem intersection_of_sets : ∀ x : ℝ, (set_A x ∧ set_B x) ↔ 1 < x ∧ x < 4 := 
by sorry

end intersection_of_sets_l203_203086


namespace find_triples_l203_203104

theorem find_triples (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : a^3 + 9 * b^2 + 9 * c + 7 = 1997) :
  (a = 10 ∧ b = 10 ∧ c = 10) :=
by sorry

end find_triples_l203_203104


namespace original_number_of_men_l203_203536

theorem original_number_of_men (x : ℕ) (h1 : x * 10 = (x - 5) * 12) : x = 30 :=
by
  sorry

end original_number_of_men_l203_203536


namespace problem_solution_l203_203674

variable (α : ℝ)
-- Condition: α in the first quadrant (0 < α < π/2)
variable (h1 : 0 < α ∧ α < Real.pi / 2)
-- Condition: sin α + cos α = sqrt 2
variable (h2 : Real.sin α + Real.cos α = Real.sqrt 2)

theorem problem_solution : Real.tan α + Real.cos α / Real.sin α = 2 :=
by
  sorry

end problem_solution_l203_203674


namespace tea_drinking_problem_l203_203179

theorem tea_drinking_problem 
  (k b c t s : ℕ) 
  (hk : k = 1) 
  (hb : b = 15) 
  (hc : c = 3) 
  (ht : t = 2) 
  (hs : s = 1) : 
  17 = 17 := 
by {
  sorry
}

end tea_drinking_problem_l203_203179


namespace tied_in_runs_l203_203915

def aaron_runs : List ℕ := [4, 8, 15, 7, 4, 12, 11, 5]
def bonds_runs : List ℕ := [3, 5, 18, 9, 12, 14, 9, 0]

def total_runs (runs : List ℕ) : ℕ := runs.foldl (· + ·) 0

theorem tied_in_runs : total_runs aaron_runs = total_runs bonds_runs := by
  sorry

end tied_in_runs_l203_203915


namespace min_value_expression_ge_512_l203_203438

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c)

theorem min_value_expression_ge_512 {a b c : ℝ} 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  min_value_expression a b c ≥ 512 :=
by
  sorry

end min_value_expression_ge_512_l203_203438


namespace birches_count_l203_203259

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l203_203259


namespace determine_constants_l203_203546

structure Vector2D :=
(x : ℝ)
(y : ℝ)

def a := 11 / 20
def b := -7 / 20

def v1 : Vector2D := ⟨3, 2⟩
def v2 : Vector2D := ⟨-1, 6⟩
def v3 : Vector2D := ⟨2, -1⟩

def linear_combination (v1 v2 : Vector2D) (a b : ℝ) : Vector2D :=
  ⟨a * v1.x + b * v2.x, a * v1.y + b * v2.y⟩

theorem determine_constants (a b : ℝ) :
  ∃ (a b : ℝ), linear_combination v1 v2 a b = v3 :=
by
  use (11 / 20)
  use (-7 / 20)
  sorry

end determine_constants_l203_203546


namespace tan_squared_sum_geq_three_over_eight_l203_203886

theorem tan_squared_sum_geq_three_over_eight 
  (α β γ : ℝ) 
  (hα : 0 ≤ α ∧ α < π / 2) 
  (hβ : 0 ≤ β ∧ β < π / 2) 
  (hγ : 0 ≤ γ ∧ γ < π / 2) 
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := 
sorry

end tan_squared_sum_geq_three_over_eight_l203_203886


namespace ellipse_foci_condition_l203_203400

theorem ellipse_foci_condition {m : ℝ} :
  (1 < m ∧ m < 2) ↔ (∃ (x y : ℝ), (x^2 / (m - 1) + y^2 / (3 - m) = 1) ∧ (3 - m > m - 1) ∧ (m - 1 > 0) ∧ (3 - m > 0)) :=
by
  sorry

end ellipse_foci_condition_l203_203400


namespace find_original_money_sandy_took_l203_203544

noncomputable def originalMoney (remainingMoney : ℝ) (clothingPercent electronicsPercent foodPercent additionalSpendPercent salesTaxPercent : ℝ) : Prop :=
  let X := (remainingMoney / (1 - ((clothingPercent + electronicsPercent + foodPercent) + additionalSpendPercent) * (1 + salesTaxPercent)))
  abs (X - 397.73) < 0.01

theorem find_original_money_sandy_took :
  originalMoney 140 0.25 0.15 0.10 0.20 0.08 :=
sorry

end find_original_money_sandy_took_l203_203544


namespace payment_correct_l203_203579

def total_payment (hours1 hours2 hours3 : ℕ) (rate_per_hour : ℕ) (num_men : ℕ) : ℕ :=
  (hours1 + hours2 + hours3) * rate_per_hour * num_men

theorem payment_correct :
  total_payment 10 8 15 10 2 = 660 :=
by
  -- We skip the proof here
  sorry

end payment_correct_l203_203579


namespace _l203_203509

noncomputable def tan_alpha_theorem (α : ℝ) (h1 : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

noncomputable def evaluate_expression_theorem (α β : ℝ) 
  (h1 : Real.tan (Real.pi / 4 + α) = 2) 
  (h2 : Real.tan β = 1 / 2) 
  (h3 : Real.tan α = 1 / 3) : 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1 / 7 :=
by
  sorry

end _l203_203509


namespace strongest_correlation_l203_203390

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end strongest_correlation_l203_203390


namespace months_to_survive_l203_203590

theorem months_to_survive (P_survive : ℝ) (initial_population : ℕ) (expected_survivors : ℝ) (n : ℕ)
  (h1 : P_survive = 5 / 6)
  (h2 : initial_population = 200)
  (h3 : expected_survivors = 115.74)
  (h4 : initial_population * (P_survive ^ n) = expected_survivors) :
  n = 3 :=
sorry

end months_to_survive_l203_203590


namespace hyperbola_range_l203_203465

theorem hyperbola_range (m : ℝ) : m * (2 * m - 1) < 0 → 0 < m ∧ m < (1 / 2) :=
by
  intro h
  sorry

end hyperbola_range_l203_203465


namespace third_dimension_of_box_l203_203602

theorem third_dimension_of_box (h : ℕ) (H : (151^2 - 150^2) * h + 151^2 = 90000) : h = 223 :=
sorry

end third_dimension_of_box_l203_203602


namespace no_integer_triple_exists_for_10_l203_203910

theorem no_integer_triple_exists_for_10 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 :=
sorry

end no_integer_triple_exists_for_10_l203_203910


namespace converse_of_squared_positive_is_negative_l203_203113

theorem converse_of_squared_positive_is_negative (x : ℝ) :
  (∀ x : ℝ, x < 0 → x^2 > 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
sorry

end converse_of_squared_positive_is_negative_l203_203113


namespace pizza_cost_is_correct_l203_203059

noncomputable def total_pizza_cost : ℝ :=
  let triple_cheese_pizza_cost := (3 * 10) + (6 * 2 * 2.5)
  let meat_lovers_pizza_cost := (3 * 8) + (4 * 3 * 2.5)
  let veggie_delight_pizza_cost := (6 * 5) + (10 * 1 * 2.5)
  triple_cheese_pizza_cost + meat_lovers_pizza_cost + veggie_delight_pizza_cost

theorem pizza_cost_is_correct : total_pizza_cost = 169 := by
  sorry

end pizza_cost_is_correct_l203_203059


namespace max_handshakes_l203_203781

theorem max_handshakes (n m : ℕ) (cond1 : n = 30) (cond2 : m = 5) 
                       (cond3 : ∀ (i : ℕ), i < 30 → ∀ (j : ℕ), j < 30 → i ≠ j → true)
                       (cond4 : ∀ (k : ℕ), k < 5 → ∃ (s : ℕ), s ≤ 10) : 
  ∃ (handshakes : ℕ), handshakes = 325 :=
by
  sorry

end max_handshakes_l203_203781


namespace Mary_chewing_gums_count_l203_203709

variable (Mary_gums Sam_gums Sue_gums : ℕ)

-- Define the given conditions
axiom Sam_chewing_gums : Sam_gums = 10
axiom Sue_chewing_gums : Sue_gums = 15
axiom Total_chewing_gums : Mary_gums + Sam_gums + Sue_gums = 30

theorem Mary_chewing_gums_count : Mary_gums = 5 := by
  sorry

end Mary_chewing_gums_count_l203_203709


namespace aluminum_percentage_range_l203_203085

variable (x1 x2 x3 y : ℝ)

theorem aluminum_percentage_range:
  (0.15 * x1 + 0.3 * x2 = 0.2) →
  (x1 + x2 + x3 = 1) →
  y = 0.6 * x1 + 0.45 * x3 →
  (1/3 ≤ x2 ∧ x2 ≤ 2/3) →
  (0.15 ≤ y ∧ y ≤ 0.4) := by
  sorry

end aluminum_percentage_range_l203_203085


namespace krish_spent_on_sweets_l203_203347

noncomputable def initial_amount := 200.50
noncomputable def amount_per_friend := 25.20
noncomputable def remaining_amount := 114.85

noncomputable def total_given_to_friends := amount_per_friend * 2
noncomputable def amount_before_sweets := initial_amount - total_given_to_friends
noncomputable def amount_spent_on_sweets := amount_before_sweets - remaining_amount

theorem krish_spent_on_sweets : amount_spent_on_sweets = 35.25 :=
by
  sorry

end krish_spent_on_sweets_l203_203347


namespace find_angle_C_l203_203652

theorem find_angle_C (A B C : ℝ) (h1 : |Real.cos A - (Real.sqrt 3 / 2)| + (1 - Real.tan B)^2 = 0) :
  C = 105 :=
by
  sorry

end find_angle_C_l203_203652


namespace length_of_platform_l203_203648

theorem length_of_platform (l t p : ℝ) (h1 : (l / t) = (l + p) / (5 * t)) : p = 4 * l :=
by
  sorry

end length_of_platform_l203_203648


namespace find_g_x2_minus_2_l203_203467

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end find_g_x2_minus_2_l203_203467


namespace table_ratio_l203_203557

theorem table_ratio (L W : ℝ) (h1 : L * W = 128) (h2 : L + 2 * W = 32) : L / W = 2 :=
by
  sorry

end table_ratio_l203_203557


namespace ap_square_sequel_l203_203448

theorem ap_square_sequel {a b c : ℝ} (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                     (h2 : 2 * (b / (c + a)) = (a / (b + c)) + (c / (a + b))) :
  (a^2 + c^2 = 2 * b^2) :=
by
  sorry

end ap_square_sequel_l203_203448


namespace total_pieces_of_gum_l203_203321

def packages := 43
def pieces_per_package := 23
def extra_pieces := 8

theorem total_pieces_of_gum :
  (packages * pieces_per_package) + extra_pieces = 997 := sorry

end total_pieces_of_gum_l203_203321


namespace total_participants_l203_203669

theorem total_participants (F M : ℕ)
  (h1 : F / 2 = 130)
  (h2 : F / 2 + M / 4 = (F + M) / 3) : 
  F + M = 780 := 
by 
  sorry

end total_participants_l203_203669


namespace factor_expression_l203_203356

theorem factor_expression (x : ℝ) : 3 * x * (x - 5) + 7 * (x - 5) - 2 * (x - 5) = (3 * x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l203_203356


namespace product_of_solutions_abs_eq_l203_203118

theorem product_of_solutions_abs_eq (x1 x2 : ℝ) (h1 : |2 * x1 - 1| + 4 = 24) (h2 : |2 * x2 - 1| + 4 = 24) : x1 * x2 = -99.75 := 
sorry

end product_of_solutions_abs_eq_l203_203118


namespace part_a_l203_203123

open Complex

theorem part_a (z : ℂ) (hz : abs z = 1) :
  (abs (z + 1) - Real.sqrt 2) * (abs (z - 1) - Real.sqrt 2) ≤ 0 :=
by
  -- Proof will go here
  sorry

end part_a_l203_203123


namespace polar_to_rectangular_correct_l203_203254

noncomputable def polar_to_rectangular (rho theta x y : ℝ) : Prop :=
  rho = 4 * Real.sin theta + 2 * Real.cos theta ∧
  rho * Real.sin theta = y ∧
  rho * Real.cos theta = x ∧
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5

theorem polar_to_rectangular_correct {rho theta x y : ℝ} :
  (rho = 4 * Real.sin theta + 2 * Real.cos theta) →
  (rho * Real.sin theta = y) →
  (rho * Real.cos theta = x) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5 :=
by
  sorry

end polar_to_rectangular_correct_l203_203254


namespace f_g_of_3_l203_203065

def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := x^2 + 2 * x + 1

theorem f_g_of_3 : f (g 3) = 61 :=
by
  sorry

end f_g_of_3_l203_203065


namespace john_pool_cleanings_per_month_l203_203693

noncomputable def tip_percent : ℝ := 0.10
noncomputable def cost_per_cleaning : ℝ := 150
noncomputable def total_cost_per_cleaning : ℝ := cost_per_cleaning + (tip_percent * cost_per_cleaning)
noncomputable def chemical_cost_bi_monthly : ℝ := 200
noncomputable def monthly_chemical_cost : ℝ := 2 * chemical_cost_bi_monthly
noncomputable def total_monthly_pool_cost : ℝ := 2050
noncomputable def total_cleaning_cost : ℝ := total_monthly_pool_cost - monthly_chemical_cost

theorem john_pool_cleanings_per_month : total_cleaning_cost / total_cost_per_cleaning = 10 := by
  sorry

end john_pool_cleanings_per_month_l203_203693


namespace probability_of_three_specific_suits_l203_203778

noncomputable def probability_at_least_one_from_each_of_three_suits : ℚ :=
  1 - (1 / 4) ^ 5

theorem probability_of_three_specific_suits (hearts clubs diamonds : ℕ) :
  hearts = 0 ∧ clubs = 0 ∧ diamonds = 0 → 
  probability_at_least_one_from_each_of_three_suits = 1023 / 1024 := 
by 
  sorry

end probability_of_three_specific_suits_l203_203778


namespace smallest_d_l203_203922

noncomputable def d := 53361

theorem smallest_d :
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧
    10000 * d = (p * q * r) ^ 2 ∧ d = 53361 :=
  by
    sorry

end smallest_d_l203_203922


namespace find_hyperbola_equation_hyperbola_equation_l203_203511

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define the new hyperbola with unknown constant m
def new_hyperbola (x y m : ℝ) := (x^2 / (m * 2)) - (y^2 / m) = 1

variable (m : ℝ)

-- The point (2, 0)
def point_on_hyperbola (x y : ℝ) := x = 2 ∧ y = 0

theorem find_hyperbola_equation (h : ∀ (x y : ℝ), point_on_hyperbola x y → new_hyperbola x y m) :
  m = 2 :=
    sorry

theorem hyperbola_equation :
  ∀ (x y : ℝ), (x = 2 ∧ y = 0) → (x^2 / 4 - y^2 / 2 = 1) :=
    sorry

end find_hyperbola_equation_hyperbola_equation_l203_203511


namespace marble_draw_l203_203189

/-- A container holds 30 red marbles, 25 green marbles, 23 yellow marbles,
15 blue marbles, 10 white marbles, and 7 black marbles. Prove that the
minimum number of marbles that must be drawn from the container without
replacement to ensure that at least 10 marbles of a single color are drawn
is 53. -/
theorem marble_draw (R G Y B W Bl : ℕ) (hR : R = 30) (hG : G = 25)
                               (hY : Y = 23) (hB : B = 15) (hW : W = 10)
                               (hBl : Bl = 7) : 
  ∃ (n : ℕ), n = 53 ∧ (∀ (x : ℕ), x ≠ n → 
  (x ≤ R → x ≤ G → x ≤ Y → x ≤ B → x ≤ W → x ≤ Bl → x < 10)) := 
by
  sorry

end marble_draw_l203_203189


namespace simplify_expression_l203_203457

variable (x y : ℝ)

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 9 * y = 45 * x + 9 * y := 
by sorry

end simplify_expression_l203_203457


namespace max_number_of_books_laughlin_can_buy_l203_203269

-- Definitions of costs and the budget constraint
def individual_book_cost : ℕ := 3
def four_book_bundle_cost : ℕ := 10
def seven_book_bundle_cost : ℕ := 15
def budget : ℕ := 20

-- Condition that Laughlin must buy at least one 4-book bundle
def minimum_required_four_book_bundles : ℕ := 1

-- Define the function to calculate the maximum number of books Laughlin can buy
def max_books (budget : ℕ) (individual_book_cost : ℕ) 
              (four_book_bundle_cost : ℕ) (seven_book_bundle_cost : ℕ) 
              (min_four_book_bundles : ℕ) : ℕ :=
  let remaining_budget_after_four_bundle := budget - (min_four_book_bundles * four_book_bundle_cost)
  if remaining_budget_after_four_bundle >= seven_book_bundle_cost then
    min_four_book_bundles * 4 + 7
  else if remaining_budget_after_four_bundle >= individual_book_cost then
    min_four_book_bundles * 4 + remaining_budget_after_four_bundle / individual_book_cost
  else
    min_four_book_bundles * 4

-- Proof statement: Laughlin can buy a maximum of 7 books
theorem max_number_of_books_laughlin_can_buy : 
  max_books budget individual_book_cost four_book_bundle_cost seven_book_bundle_cost minimum_required_four_book_bundles = 7 :=
by
  sorry

end max_number_of_books_laughlin_can_buy_l203_203269


namespace moles_of_Cl2_combined_l203_203161

theorem moles_of_Cl2_combined (nCH4 : ℕ) (nCl2 : ℕ) (nHCl : ℕ) 
  (h1 : nCH4 = 3) 
  (h2 : nHCl = nCl2) 
  (h3 : nHCl ≤ nCH4) : 
  nCl2 = 3 :=
by
  sorry

end moles_of_Cl2_combined_l203_203161


namespace abs_x_minus_2y_is_square_l203_203211

theorem abs_x_minus_2y_is_square (x y : ℕ) (h : ∃ k : ℤ, x^2 - 4 * y + 1 = (x - 2 * y) * (1 - 2 * y) * k) : ∃ m : ℕ, x - 2 * y = m ^ 2 := by
  sorry

end abs_x_minus_2y_is_square_l203_203211


namespace correct_choice_l203_203916

theorem correct_choice
  (options : List String)
  (correct : String)
  (is_correct : correct = "that") :
  "The English spoken in the United States is only slightly different from ____ spoken in England." = 
  "The English spoken in the United States is only slightly different from that spoken in England." :=
by
  sorry

end correct_choice_l203_203916


namespace isosceles_trapezoid_inscribed_circle_ratio_l203_203455

noncomputable def ratio_perimeter_inscribed_circle (x : ℝ) : ℝ := 
  (50 * x) / (10 * Real.pi * x)

theorem isosceles_trapezoid_inscribed_circle_ratio 
  (x : ℝ)
  (h1 : x > 0)
  (r : ℝ) 
  (OK OP : ℝ) 
  (h2 : OK = 3 * x) 
  (h3 : OP = 5 * x) : 
  ratio_perimeter_inscribed_circle x = 5 / Real.pi :=
by
  sorry

end isosceles_trapezoid_inscribed_circle_ratio_l203_203455


namespace food_expenditure_increase_l203_203298

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end food_expenditure_increase_l203_203298


namespace lisa_eats_correct_number_of_pieces_l203_203527

variable (M A K R L : ℚ) -- All variables are rational numbers (real numbers could also be used)
variable (n : ℕ) -- n is a natural number (the number of pieces of lasagna)

-- Let's define the conditions succinctly
def manny_wants_one_piece := M = 1
def aaron_eats_nothing := A = 0
def kai_eats_twice_manny := K = 2 * M
def raphael_eats_half_manny := R = 0.5 * M
def lasagna_is_cut_into_6_pieces := n = 6

-- The proof goal is to show Lisa eats 2.5 pieces
theorem lisa_eats_correct_number_of_pieces (M A K R L : ℚ) (n : ℕ) :
  manny_wants_one_piece M →
  aaron_eats_nothing A →
  kai_eats_twice_manny M K →
  raphael_eats_half_manny M R →
  lasagna_is_cut_into_6_pieces n →
  L = n - (M + K + R) →
  L = 2.5 :=
by
  intros hM hA hK hR hn hL
  sorry  -- Proof omitted

end lisa_eats_correct_number_of_pieces_l203_203527


namespace geometric_sequence_a4_l203_203043

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 4)
  (h3 : a 6 = 16) : 
  a 4 = 8 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a4_l203_203043


namespace geometric_seq_an_minus_2_l203_203176

-- Definitions of conditions based on given problem
def seq_a : ℕ → ℝ := sorry -- The sequence {a_n}
def sum_s : ℕ → ℝ := sorry -- The sum of the first n terms {s_n}

axiom cond1 (n : ℕ) (hn : n > 0) : seq_a (n + 1) ≠ seq_a n
axiom cond2 (n : ℕ) (hn : n > 0) : sum_s n + seq_a n = 2 * n

-- Theorem statement
theorem geometric_seq_an_minus_2 (n : ℕ) (hn : n > 0) : 
  ∃ r : ℝ, ∀ k : ℕ, seq_a (k + 1) - 2 = r * (seq_a k - 2) := 
sorry

end geometric_seq_an_minus_2_l203_203176


namespace digits_same_l203_203221

theorem digits_same (k : ℕ) (hk : k ≥ 2) :
  (∃ n : ℕ, (10^(10^n) - 9^(9^n)) % (10^k) = 0) ↔ (k = 2 ∨ k = 3 ∨ k = 4) :=
sorry

end digits_same_l203_203221


namespace num_marbles_removed_l203_203456

theorem num_marbles_removed (total_marbles red_marbles : ℕ) (prob_neither_red : ℚ) 
  (h₁ : total_marbles = 84) (h₂ : red_marbles = 12) (h₃ : prob_neither_red = 36 / 49) : 
  total_marbles - red_marbles = 2 :=
by
  sorry

end num_marbles_removed_l203_203456


namespace geometric_sequence_constant_l203_203973

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1) (h2 : ∀ n, a (n + 1) = q * a n) (c : ℝ) :
  (∀ n, a (n + 1) + c = q * (a n + c)) → c = 0 := sorry

end geometric_sequence_constant_l203_203973


namespace tetrahedron_faces_equal_l203_203762

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l203_203762


namespace rhombus_diagonal_l203_203553

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 10) (h2 : area = 60) : 
  d1 = 12 :=
by 
  have : (d1 * d2) / 2 = area := sorry
  sorry

end rhombus_diagonal_l203_203553


namespace total_money_l203_203758

-- Define the problem with conditions and question transformed into proof statement
theorem total_money (A B : ℕ) (h1 : 2 * A / 3 = B / 2) (h2 : B = 484) : A + B = 847 :=
by
  sorry -- Proof to be filled in

end total_money_l203_203758


namespace simultaneous_messengers_l203_203713

theorem simultaneous_messengers (m n : ℕ) (h : m * n = 2010) : 
  m ≠ n → ((m, n) = (1, 2010) ∨ (m, n) = (2, 1005) ∨ (m, n) = (3, 670) ∨ 
          (m, n) = (5, 402) ∨ (m, n) = (6, 335) ∨ (m, n) = (10, 201) ∨ 
          (m, n) = (15, 134) ∨ (m, n) = (30, 67)) :=
sorry

end simultaneous_messengers_l203_203713


namespace max_perimeter_right_triangle_l203_203972

theorem max_perimeter_right_triangle (a b : ℝ) (h₁ : a^2 + b^2 = 25) :
  (a + b + 5) ≤ 5 + 5 * Real.sqrt 2 :=
by
  sorry

end max_perimeter_right_triangle_l203_203972


namespace correct_systematic_sampling_method_l203_203050

inductive SamplingMethod
| A
| B
| C
| D

def most_suitable_for_systematic_sampling (A B C D : SamplingMethod) : SamplingMethod :=
SamplingMethod.C

theorem correct_systematic_sampling_method : 
    most_suitable_for_systematic_sampling SamplingMethod.A SamplingMethod.B SamplingMethod.C SamplingMethod.D = SamplingMethod.C :=
by
  sorry

end correct_systematic_sampling_method_l203_203050


namespace intersection_P_Q_l203_203641

-- Definitions based on conditions
def P : Set ℝ := { y | ∃ x : ℝ, y = x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = 1 - x }

-- Proof statement to show P ∩ Q = Set.univ
theorem intersection_P_Q : P ∩ Q = Set.univ := by
  sorry

end intersection_P_Q_l203_203641


namespace hyperbola_eccentricity_is_sqrt_3_l203_203947

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_is_sqrt_3 (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt_3_l203_203947


namespace average_price_of_rackets_l203_203330

theorem average_price_of_rackets (total_amount : ℝ) (number_of_pairs : ℕ) (average_price : ℝ) 
  (h1 : total_amount = 588) (h2 : number_of_pairs = 60) : average_price = 9.80 :=
by
  sorry

end average_price_of_rackets_l203_203330


namespace sum_of_powers_of_four_to_50_l203_203282

theorem sum_of_powers_of_four_to_50 :
  2 * (Finset.sum (Finset.range 51) (λ x => x^4)) = 1301700 := by
  sorry

end sum_of_powers_of_four_to_50_l203_203282


namespace range_of_a_l203_203388

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, x ≥ 1 → x^2 + a * x + 9 ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l203_203388


namespace bounded_sequence_is_constant_two_l203_203096

def is_bounded (l : ℕ → ℕ) := ∃ (M : ℕ), ∀ (n : ℕ), l n ≤ M

def satisfies_condition (a : ℕ → ℕ) : Prop :=
∀ n ≥ 3, a n = (a n.pred + a (n.pred.pred)) / (Nat.gcd (a n.pred) (a (n.pred.pred)))

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) 
  (h1 : is_bounded a) 
  (h2 : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 :=
sorry

end bounded_sequence_is_constant_two_l203_203096


namespace min_value_inequality_l203_203726

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  x^2 + 4 * x * y + 9 * y^2 + 6 * y * z + 8 * z^2 + 3 * x * w + 4 * w^2

theorem min_value_inequality 
  (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_prod : x * y * z * w = 3) : 
  min_value x y z w ≥ 81.25 := 
sorry

end min_value_inequality_l203_203726


namespace solve_for_x_l203_203076

-- Define the problem with the given conditions
def sum_of_triangle_angles (x : ℝ) : Prop := x + 2 * x + 30 = 180

-- State the theorem
theorem solve_for_x : ∀ (x : ℝ), sum_of_triangle_angles x → x = 50 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l203_203076


namespace find_x_l203_203235

-- Definitions from the conditions
def isPositiveMultipleOf7 (x : ℕ) : Prop := ∃ k : ℕ, x = 7 * k ∧ x > 0
def xSquaredGreaterThan150 (x : ℕ) : Prop := x^2 > 150
def xLessThan40 (x : ℕ) : Prop := x < 40

-- Main problem statement
theorem find_x (x : ℕ) (h1 : isPositiveMultipleOf7 x) (h2 : xSquaredGreaterThan150 x) (h3 : xLessThan40 x) : x = 14 :=
sorry

end find_x_l203_203235


namespace determine_range_of_k_l203_203423

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l203_203423


namespace kaleb_gave_boxes_l203_203141

theorem kaleb_gave_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (given_boxes : ℕ)
  (h1 : total_boxes = 14) 
  (h2 : pieces_per_box = 6) 
  (h3 : pieces_left = 54) :
  given_boxes = 5 :=
by
  -- Add your proof here
  sorry

end kaleb_gave_boxes_l203_203141


namespace exponent_equality_l203_203453

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l203_203453


namespace monotonic_decreasing_interval_l203_203777

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, x < 2 → f' x < 0 :=
by
  intro x hx
  sorry

end monotonic_decreasing_interval_l203_203777


namespace alok_total_payment_l203_203265

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l203_203265


namespace function_increasing_interval_l203_203884

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x ^ 2) / Real.log 2

def domain (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem function_increasing_interval : 
  ∀ x, domain x → 0 < x ∧ x < 1 → ∀ y, domain y → 0 < y ∧ y < 1 → x < y → f x < f y :=
by 
  intros x hx h0 y hy h1 hxy
  sorry

end function_increasing_interval_l203_203884


namespace second_race_length_l203_203248

variable (T L : ℝ)
variable (V_A V_B V_C : ℝ)

variables (h1 : V_A * T = 100)
variables (h2 : V_B * T = 90)
variables (h3 : V_C * T = 87)
variables (h4 : L / V_B = (L - 6) / V_C)

theorem second_race_length :
  L = 180 :=
sorry

end second_race_length_l203_203248


namespace polynomial_expansion_l203_203172

variable (x : ℝ)

theorem polynomial_expansion : 
  (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 :=
by
  sorry

end polynomial_expansion_l203_203172


namespace solve_for_x_l203_203256

theorem solve_for_x (x : ℝ) (h : x ≠ -2) :
  (4 * x) / (x + 2) - 2 / (x + 2) = 3 / (x + 2) → x = 5 / 4 := by
  sorry

end solve_for_x_l203_203256


namespace minimum_value_reciprocals_l203_203014

theorem minimum_value_reciprocals (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : 2 / Real.sqrt (a^2 + 4 * b^2) = Real.sqrt 2) :
  (1 / a^2 + 1 / b^2) = 9 / 2 :=
sorry

end minimum_value_reciprocals_l203_203014


namespace polygon_diagonals_l203_203313

theorem polygon_diagonals (n : ℕ) (k_0 k_1 k_2 : ℕ)
  (h1 : 2 * k_2 + k_1 = n)
  (h2 : k_2 + k_1 + k_0 = n - 2) :
  k_2 ≥ 2 :=
sorry

end polygon_diagonals_l203_203313


namespace mapping_image_l203_203871

theorem mapping_image (x y l m : ℤ) (h1 : x = 4) (h2 : y = 6) (h3 : l = x + y) (h4 : m = x - y) :
  (l, m) = (10, -2) := by
  sorry

end mapping_image_l203_203871


namespace percentage_problem_l203_203802

theorem percentage_problem (P : ℕ) (n : ℕ) (h_n : n = 16)
  (h_condition : (40: ℚ) = 0.25 * n + 2) : P = 250 :=
by
  sorry

end percentage_problem_l203_203802


namespace no_intersection_pair_C_l203_203927

theorem no_intersection_pair_C :
  let y1 := fun x : ℝ => x
  let y2 := fun x : ℝ => x - 3
  ∀ x : ℝ, y1 x ≠ y2 x :=
by
  sorry

end no_intersection_pair_C_l203_203927


namespace david_dogs_left_l203_203464

def total_dogs_left (boxes_small: Nat) (dogs_per_small: Nat) (boxes_large: Nat) (dogs_per_large: Nat) (giveaway_small: Nat) (giveaway_large: Nat): Nat :=
  let total_small := boxes_small * dogs_per_small
  let total_large := boxes_large * dogs_per_large
  let remaining_small := total_small - giveaway_small
  let remaining_large := total_large - giveaway_large
  remaining_small + remaining_large

theorem david_dogs_left :
  total_dogs_left 7 4 5 3 2 1 = 40 := by
  sorry

end david_dogs_left_l203_203464


namespace T_8_equals_546_l203_203522

-- Define the sum of the first n natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the squares of the first n natural numbers
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Define T_n based on the given formula
def T (n : ℕ) : ℕ := (sum_first_n n ^ 2 - sum_squares_first_n n) / 2

-- The proof statement we need to prove
theorem T_8_equals_546 : T 8 = 546 := sorry

end T_8_equals_546_l203_203522


namespace ernie_income_ratio_l203_203826

-- Define constants and properties based on the conditions
def previous_income := 6000
def jack_income := 2 * previous_income
def combined_income := 16800

-- Lean proof statement that the ratio of Ernie's current income to his previous income is 2/3
theorem ernie_income_ratio (current_income : ℕ) (h1 : current_income + jack_income = combined_income) :
    current_income / previous_income = 2 / 3 :=
sorry

end ernie_income_ratio_l203_203826


namespace bryce_received_12_raisins_l203_203561

-- Defining the main entities for the problem
variables {x y z : ℕ} -- number of raisins Bryce, Carter, and Emma received respectively

-- Conditions:
def condition1 (x y : ℕ) : Prop := y = x - 8
def condition2 (x y : ℕ) : Prop := y = x / 3
def condition3 (y z : ℕ) : Prop := z = 2 * y

-- The goal is to prove that Bryce received 12 raisins
theorem bryce_received_12_raisins (x y z : ℕ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) 
  (h3 : condition3 y z) : 
  x = 12 :=
sorry

end bryce_received_12_raisins_l203_203561


namespace prime_divides_diff_of_cubes_l203_203698

theorem prime_divides_diff_of_cubes (a b c : ℕ) [Fact (Nat.Prime a)] [Fact (Nat.Prime b)]
  (h1 : c ∣ (a + b)) (h2 : c ∣ (a * b)) : c ∣ (a^3 - b^3) :=
by
  sorry

end prime_divides_diff_of_cubes_l203_203698


namespace inequality_k_l203_203151

variable {R : Type} [LinearOrderedField R] [Nontrivial R]

theorem inequality_k (x y z : R) (k : ℕ) (h : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) 
  (hineq : (1/x) + (1/y) + (1/z) ≥ x + y + z) :
  (1/x^k) + (1/y^k) + (1/z^k) ≥ x^k + y^k + z^k :=
sorry

end inequality_k_l203_203151


namespace num_teachers_in_Oxford_High_School_l203_203007

def classes : Nat := 15
def students_per_class : Nat := 20
def principals : Nat := 1
def total_people : Nat := 349

theorem num_teachers_in_Oxford_High_School : 
  ∃ (teachers : Nat), teachers = total_people - (classes * students_per_class + principals) :=
by
  use 48
  sorry

end num_teachers_in_Oxford_High_School_l203_203007


namespace min_value_xyz_l203_203721

-- Definition of the problem
theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 108):
  x^2 + 9 * x * y + 9 * y^2 + 3 * z^2 ≥ 324 :=
sorry

end min_value_xyz_l203_203721


namespace product_mod_7_l203_203135

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l203_203135


namespace fraction_subtraction_simplified_l203_203607

theorem fraction_subtraction_simplified :
  (8 / 21 - 3 / 63) = 1 / 3 := 
by
  sorry

end fraction_subtraction_simplified_l203_203607


namespace total_number_of_cards_l203_203863

/-- There are 9 playing cards and 4 ID cards initially.
If you add 6 more playing cards and 3 more ID cards,
then the total number of playing cards and ID cards will be 22. -/
theorem total_number_of_cards :
  let initial_playing_cards := 9
  let initial_id_cards := 4
  let additional_playing_cards := 6
  let additional_id_cards := 3
  let total_playing_cards := initial_playing_cards + additional_playing_cards
  let total_id_cards := initial_id_cards + additional_id_cards
  let total_cards := total_playing_cards + total_id_cards
  total_cards = 22 :=
by
  sorry

end total_number_of_cards_l203_203863


namespace school_year_hours_per_week_l203_203836

-- Definitions based on the conditions of the problem
def summer_weeks : ℕ := 8
def summer_hours_per_week : ℕ := 40
def summer_earnings : ℕ := 3200

def school_year_weeks : ℕ := 24
def needed_school_year_earnings : ℕ := 6400

-- Question translated to a Lean statement
theorem school_year_hours_per_week :
  let hourly_rate := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := needed_school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks = (80 / 3) :=
by {
  -- The implementation of the proof goes here
  sorry
}

end school_year_hours_per_week_l203_203836


namespace integer_values_of_f_l203_203131

theorem integer_values_of_f (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a * b ≠ 1) : 
  ∃ k ∈ ({4, 7} : Finset ℕ), 
    (a^2 + b^2 + a * b) / (a * b - 1) = k := 
by
  sorry

end integer_values_of_f_l203_203131


namespace tens_digit_36_pow_12_l203_203695

theorem tens_digit_36_pow_12 : ((36 ^ 12) % 100) / 10 % 10 = 1 := 
by 
sorry

end tens_digit_36_pow_12_l203_203695


namespace charlie_received_495_l203_203225

theorem charlie_received_495 : 
  ∃ (A B C x : ℕ), 
    A + B + C = 1105 ∧ 
    A - 10 = 11 * x ∧ 
    B - 20 = 18 * x ∧ 
    C - 15 = 24 * x ∧ 
    C = 495 := 
by
  sorry

end charlie_received_495_l203_203225


namespace difference_blue_yellow_l203_203776

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end difference_blue_yellow_l203_203776


namespace find_solutions_l203_203434

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 14*x - 8)) = 0

theorem find_solutions : {x : ℝ | equation x} = {2, -4, 1, -8} :=
  by
  sorry

end find_solutions_l203_203434


namespace problem_counts_correct_pairs_l203_203633

noncomputable def count_valid_pairs : ℝ :=
  sorry

theorem problem_counts_correct_pairs :
  count_valid_pairs = 128 :=
by
  sorry

end problem_counts_correct_pairs_l203_203633


namespace angle_between_clock_hands_at_7_25_l203_203357

theorem angle_between_clock_hands_at_7_25 : 
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  abs (hour_hand_position - minute_hand_position) = 72.5 
  := by
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  sorry

end angle_between_clock_hands_at_7_25_l203_203357


namespace expression_bounds_l203_203542

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
                     Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ∧
  (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
   Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ≤ 4 := sorry

end expression_bounds_l203_203542


namespace part1_part2_l203_203919

variable {a : ℝ} (M N : Set ℝ)

theorem part1 (h : a = 1) : M = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (hM : (M = {x : ℝ | 0 < x ∧ x < a + 1}))
              (hN : N = {x : ℝ | -1 ≤ x ∧ x ≤ 3})
              (h_union : M ∪ N = N) : 
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end part1_part2_l203_203919


namespace find_soma_cubes_for_shape_l203_203049

def SomaCubes (n : ℕ) : Type := 
  if n = 1 
  then Fin 3 
  else if 2 ≤ n ∧ n ≤ 7 
       then Fin 4 
       else Fin 0

theorem find_soma_cubes_for_shape :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  SomaCubes a = Fin 3 ∧ SomaCubes b = Fin 4 ∧ SomaCubes c = Fin 4 ∧ 
  a + b + c = 11 ∧ ((a, b, c) = (1, 3, 5) ∨ (a, b, c) = (1, 3, 6)) := 
by
  sorry

end find_soma_cubes_for_shape_l203_203049


namespace mixed_number_multiplication_l203_203428

def mixed_to_improper (a : Int) (b : Int) (c : Int) : Rat :=
  a + (b / c)

theorem mixed_number_multiplication : 
  let a := 5
  let b := mixed_to_improper 7 2 5
  a * b = (37 : Rat) :=
by
  intros
  sorry

end mixed_number_multiplication_l203_203428


namespace sum_of_remainders_l203_203543

theorem sum_of_remainders (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p)
    (a : ℕ → ℕ) (ha : ∀ k, a k = k^p % p^2) :
    (Finset.sum (Finset.range (p - 1)) a) = (p^3 - p^2) / 2 :=
by
  sorry

end sum_of_remainders_l203_203543


namespace second_term_of_arithmetic_sequence_l203_203786

-- Define the statement of the problem
theorem second_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h : a + (a + 2 * d) = 10) : 
  a + d = 5 := 
by 
  sorry

end second_term_of_arithmetic_sequence_l203_203786


namespace veronica_pits_cherries_in_2_hours_l203_203812

theorem veronica_pits_cherries_in_2_hours :
  ∀ (pounds_cherries : ℕ) (cherries_per_pound : ℕ)
    (time_first_pound : ℕ) (cherries_first_pound : ℕ)
    (time_second_pound : ℕ) (cherries_second_pound : ℕ)
    (time_third_pound : ℕ) (cherries_third_pound : ℕ)
    (minutes_per_hour : ℕ),
  pounds_cherries = 3 →
  cherries_per_pound = 80 →
  time_first_pound = 10 →
  cherries_first_pound = 20 →
  time_second_pound = 8 →
  cherries_second_pound = 20 →
  time_third_pound = 12 →
  cherries_third_pound = 20 →
  minutes_per_hour = 60 →
  ((time_first_pound / cherries_first_pound * cherries_per_pound) + 
   (time_second_pound / cherries_second_pound * cherries_per_pound) + 
   (time_third_pound / cherries_third_pound * cherries_per_pound)) / minutes_per_hour = 2 :=
by
  intros pounds_cherries cherries_per_pound
         time_first_pound cherries_first_pound
         time_second_pound cherries_second_pound
         time_third_pound cherries_third_pound
         minutes_per_hour
         pounds_eq cherries_eq
         time1_eq cherries1_eq
         time2_eq cherries2_eq
         time3_eq cherries3_eq
         mins_eq

  -- You would insert the proof here
  sorry

end veronica_pits_cherries_in_2_hours_l203_203812


namespace express_fraction_l203_203646

noncomputable def x : ℚ := 0.8571 -- This represents \( x = 0.\overline{8571} \)
noncomputable def y : ℚ := 0.142857 -- This represents \( y = 0.\overline{142857} \)
noncomputable def z : ℚ := 2 + y -- This represents \( 2 + y = 2.\overline{142857} \)

theorem express_fraction :
  (x / z) = (1 / 2) :=
by
  sorry

end express_fraction_l203_203646


namespace fraction_value_l203_203942

theorem fraction_value : (3 - (-3)) / (2 - 1) = 6 := 
by
  sorry

end fraction_value_l203_203942


namespace solve_quadratic_eq_l203_203089

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 4 * x + 2 = 0 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end solve_quadratic_eq_l203_203089


namespace inverse_proportion_quadrants_l203_203255

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ y = k / x) →
  (∀ x : ℝ, x ≠ 0 → ( (x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0) ) ) :=
by
  sorry

end inverse_proportion_quadrants_l203_203255


namespace probability_of_sequential_draws_l203_203943

theorem probability_of_sequential_draws :
  let total_cards := 52
  let num_fours := 4
  let remaining_after_first_draw := total_cards - 1
  let remaining_after_second_draw := remaining_after_first_draw - 1
  num_fours / total_cards * 1 / remaining_after_first_draw * 1 / remaining_after_second_draw = 1 / 33150 :=
by sorry

end probability_of_sequential_draws_l203_203943


namespace twelve_year_olds_count_l203_203480

theorem twelve_year_olds_count (x y z w : ℕ) 
  (h1 : x + y + z + w = 23)
  (h2 : 10 * x + 11 * y + 12 * z + 13 * w = 253)
  (h3 : z = 3 * w / 2) : 
  z = 6 :=
by sorry

end twelve_year_olds_count_l203_203480


namespace arithmetic_seq_problem_l203_203245

-- Conditions and definitions for the arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) = a_n n + d

def sum_seq (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

def T_plus_K_eq_19 (T K : ℕ) : Prop :=
  T + K = 19

-- The given problem to prove
theorem arithmetic_seq_problem (a_n S_n : ℕ → ℝ) (d : ℝ) (h1 : d > 0)
  (h2 : arithmetic_seq a_n d) (h3 : sum_seq a_n S_n)
  (h4 : ∀ T K, T_plus_K_eq_19 T K → S_n T = S_n K) :
  ∃! n, a_n n - S_n n ≥ 0 := sorry

end arithmetic_seq_problem_l203_203245


namespace simon_age_is_10_l203_203451

-- Define the conditions
def alvin_age := 30
def half_alvin_age := alvin_age / 2
def simon_age := half_alvin_age - 5

-- State the theorem
theorem simon_age_is_10 : simon_age = 10 :=
by
  sorry

end simon_age_is_10_l203_203451


namespace solve_equation_l203_203547

theorem solve_equation (x : ℝ) (hx : x ≠ 1) : (x / (x - 1) - 1 = 1) → (x = 2) :=
by
  sorry

end solve_equation_l203_203547


namespace polygon_sides_eq_six_l203_203478

theorem polygon_sides_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = (2 * 360)) 
  (h2 : exterior_sum = 360) :
  n = 6 := 
by
  sorry

end polygon_sides_eq_six_l203_203478


namespace zero_point_interval_l203_203603

variable (f : ℝ → ℝ)
variable (f_deriv : ℝ → ℝ)
variable (e : ℝ)
variable (monotonic_f : MonotoneOn f (Set.Ioi 0))

noncomputable def condition1_property (x : ℝ) (h : 0 < x) : f (f x - Real.log x) = Real.exp 1 + 1 := sorry
noncomputable def derivative_property (x : ℝ) (h : 0 < x) : f_deriv x = (deriv f) x := sorry

theorem zero_point_interval :
  ∃ x ∈ Set.Ioo 1 2, f x - f_deriv x - e = 0 := sorry

end zero_point_interval_l203_203603


namespace find_k_l203_203851

noncomputable def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

def correct_k (k : ℝ) : Prop :=
  let slope1 := -k / 2
  let slope2 := -2
  slope1 * slope2 = -1

theorem find_k (k : ℝ) (h_perpendicular : correct_k k) : k = -1 :=
sorry

end find_k_l203_203851


namespace octopus_leg_count_l203_203582

theorem octopus_leg_count :
  let num_initial_octopuses := 5
  let legs_per_normal_octopus := 8
  let num_removed_octopuses := 2
  let legs_first_mutant := 10
  let legs_second_mutant := 6
  let legs_third_mutant := 2 * legs_per_normal_octopus
  let num_initial_legs := num_initial_octopuses * legs_per_normal_octopus
  let num_removed_legs := num_removed_octopuses * legs_per_normal_octopus
  let num_mutant_legs := legs_first_mutant + legs_second_mutant + legs_third_mutant
  num_initial_legs - num_removed_legs + num_mutant_legs = 56 :=
by
  -- proof to be filled in later
  sorry

end octopus_leg_count_l203_203582


namespace complex_sum_identity_l203_203932

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity_l203_203932


namespace original_number_of_men_l203_203350

theorem original_number_of_men 
    (x : ℕ) 
    (h : x * 40 = (x - 5) * 60) : x = 15 := 
sorry

end original_number_of_men_l203_203350


namespace factor_w4_minus_16_l203_203373

theorem factor_w4_minus_16 (w : ℝ) : (w^4 - 16) = (w - 2) * (w + 2) * (w^2 + 4) :=
by
    sorry

end factor_w4_minus_16_l203_203373


namespace matches_for_ladder_l203_203157

theorem matches_for_ladder (n : ℕ) (h : n = 25) : 
  (6 + 6 * (n - 1) = 150) :=
by
  sorry

end matches_for_ladder_l203_203157


namespace solve_quadratic_l203_203982

theorem solve_quadratic : ∀ (x : ℝ), x * (x + 1) = 2014 * 2015 ↔ (x = 2014 ∨ x = -2015) := by
  sorry

end solve_quadratic_l203_203982


namespace matrices_commute_l203_203204

variable {n : Nat}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem matrices_commute (h : A * X * B + A + B = 0) : A * X * B = B * X * A :=
by
  sorry

end matrices_commute_l203_203204


namespace grassy_plot_width_l203_203572

/-- A rectangular grassy plot has a length of 100 m and a certain width. 
It has a gravel path 2.5 m wide all round it on the inside. The cost of gravelling 
the path at 0.90 rupees per square meter is 742.5 rupees. 
Prove that the width of the grassy plot is 60 meters. -/
theorem grassy_plot_width 
  (length : ℝ)
  (path_width : ℝ)
  (cost_per_sq_meter : ℝ)
  (total_cost : ℝ)
  (width : ℝ) : 
  length = 100 ∧ 
  path_width = 2.5 ∧ 
  cost_per_sq_meter = 0.9 ∧ 
  total_cost = 742.5 → 
  width = 60 := 
by sorry

end grassy_plot_width_l203_203572


namespace no_real_x_condition_l203_203761

theorem no_real_x_condition (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 3| + |x - 1| ≤ a) ↔ a < 2 := 
by
  sorry

end no_real_x_condition_l203_203761


namespace find_number_divided_l203_203082

theorem find_number_divided (x : ℝ) (h : x / 1.33 = 48) : x = 63.84 :=
by
  sorry

end find_number_divided_l203_203082


namespace fraction_of_sand_is_one_third_l203_203900

noncomputable def total_weight : ℝ := 24
noncomputable def weight_of_water (total_weight : ℝ) : ℝ := total_weight / 4
noncomputable def weight_of_gravel : ℝ := 10
noncomputable def weight_of_sand (total_weight weight_of_water weight_of_gravel : ℝ) : ℝ :=
  total_weight - weight_of_water - weight_of_gravel
noncomputable def fraction_of_sand (weight_of_sand total_weight : ℝ) : ℝ :=
  weight_of_sand / total_weight

theorem fraction_of_sand_is_one_third :
  fraction_of_sand (weight_of_sand total_weight (weight_of_water total_weight) weight_of_gravel) total_weight
  = 1/3 := by
  sorry

end fraction_of_sand_is_one_third_l203_203900


namespace johns_watermelon_weight_l203_203437

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l203_203437


namespace train_length_is_135_l203_203594

noncomputable def length_of_train (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_is_135 :
  length_of_train 54 9 = 135 := 
by
  -- Conditions: 
  -- speed_kmh = 54
  -- time_sec = 9
  sorry

end train_length_is_135_l203_203594


namespace cannot_reach_target_l203_203596

def initial_price : ℕ := 1
def annual_increment : ℕ := 1
def tripling_year (n : ℕ) : ℕ := 3 * n
def total_years : ℕ := 99
def target_price : ℕ := 152
def incremental_years : ℕ := 98

noncomputable def final_price (x : ℕ) : ℕ := 
  initial_price + incremental_years * annual_increment + tripling_year x - annual_increment

theorem cannot_reach_target (p : ℕ) (h : p = final_price p) : p ≠ target_price :=
sorry

end cannot_reach_target_l203_203596


namespace sum_of_numbers_l203_203847

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 22 := 
by 
  sorry

end sum_of_numbers_l203_203847


namespace div_factorial_result_l203_203130

-- Define the given condition
def ten_fact : ℕ := 3628800

-- Define four factorial
def four_fact : ℕ := 4 * 3 * 2 * 1

-- State the theorem to be proved
theorem div_factorial_result : ten_fact / four_fact = 151200 :=
by
  -- Sorry is used to skip the proof, only the statement is provided
  sorry

end div_factorial_result_l203_203130


namespace mean_of_three_numbers_l203_203002

theorem mean_of_three_numbers (a : Fin 12 → ℕ) (x y z : ℕ) 
  (h1 : (Finset.univ.sum a) / 12 = 40)
  (h2 : ((Finset.univ.sum a) + x + y + z) / 15 = 50) :
  (x + y + z) / 3 = 90 := 
by
  sorry

end mean_of_three_numbers_l203_203002


namespace find_A_l203_203913

/-- Given that the equation Ax + 10y = 100 has two distinct positive integer solutions, prove that A = 10. -/
theorem find_A (A x1 y1 x2 y2 : ℕ) (h1 : A > 0) (h2 : x1 > 0) (h3 : y1 > 0) 
  (h4 : x2 > 0) (h5 : y2 > 0) (distinct_solutions : x1 ≠ x2 ∧ y1 ≠ y2) 
  (eq1 : A * x1 + 10 * y1 = 100) (eq2 : A * x2 + 10 * y2 = 100) : 
  A = 10 := sorry

end find_A_l203_203913


namespace smallest_pos_int_gcd_gt_one_l203_203488

theorem smallest_pos_int_gcd_gt_one : ∃ n: ℕ, n > 0 ∧ (Nat.gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 121 :=
by
  sorry

end smallest_pos_int_gcd_gt_one_l203_203488


namespace tennis_balls_ordered_l203_203736

def original_white_balls : ℕ := sorry
def original_yellow_balls_with_error : ℕ := sorry

theorem tennis_balls_ordered 
  (W Y : ℕ)
  (h1 : W = Y)
  (h2 : Y + 70 = original_yellow_balls_with_error)
  (h3 : W = 8 / 13 * (Y + 70)):
  W + Y = 224 := sorry

end tennis_balls_ordered_l203_203736


namespace cruise_liner_travelers_l203_203994

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l203_203994


namespace min_people_wearing_both_l203_203540

theorem min_people_wearing_both (n : ℕ) (h_lcm : n % 24 = 0) 
  (h_gloves : 3 * n % 8 = 0) (h_hats : 5 * n % 6 = 0) :
  ∃ x, x = 5 := 
by
  let gloves := 3 * n / 8
  let hats := 5 * n / 6
  let both := gloves + hats - n
  have h1 : both = 5 := sorry
  exact ⟨both, h1⟩

end min_people_wearing_both_l203_203540


namespace avg_primes_between_30_and_50_l203_203353

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

def sum_primes : ℕ := primes_between_30_and_50.sum

def count_primes : ℕ := primes_between_30_and_50.length

def average_primes : ℚ := (sum_primes : ℚ) / (count_primes : ℚ)

theorem avg_primes_between_30_and_50 : average_primes = 39.8 := by
  sorry

end avg_primes_between_30_and_50_l203_203353


namespace velocity_at_t4_acceleration_is_constant_l203_203370

noncomputable def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

def v (t : ℝ) : ℝ := 6 * t - 3

def a : ℝ := 6

theorem velocity_at_t4 : v 4 = 21 := by 
  sorry

theorem acceleration_is_constant : a = 6 := by 
  sorry

end velocity_at_t4_acceleration_is_constant_l203_203370


namespace jason_home_distance_l203_203911

theorem jason_home_distance :
  let v1 := 60 -- speed in miles per hour
  let t1 := 0.5 -- time in hours
  let d1 := v1 * t1 -- distance covered in first part of the journey
  let v2 := 90 -- speed in miles per hour for the second part
  let t2 := 1.0 -- remaining time in hours
  let d2 := v2 * t2 -- distance covered in second part of the journey
  let total_distance := d1 + d2 -- total distance to Jason's home
  total_distance = 120 := 
by
  simp only
  sorry

end jason_home_distance_l203_203911


namespace compare_a_b_c_l203_203183

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l203_203183


namespace total_revenue_correct_l203_203805

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l203_203805


namespace ant_population_percentage_l203_203197

theorem ant_population_percentage (R : ℝ) 
  (h1 : 0.45 * R = 46.75) 
  (h2 : R * 0.55 = 46.75) : 
  R = 0.85 := 
by 
  sorry

end ant_population_percentage_l203_203197


namespace moles_HCl_combination_l203_203987

-- Define the conditions:
def moles_HCl (C5H12O: ℕ) (H2O: ℕ) : ℕ :=
  if H2O = 18 then 18 else 0

-- The main statement to prove:
theorem moles_HCl_combination :
  moles_HCl 1 18 = 18 :=
sorry

end moles_HCl_combination_l203_203987


namespace Xiaokang_position_l203_203252

theorem Xiaokang_position :
  let east := 150
  let west := 100
  let total_walks := 3
  (east - west - west = -50) :=
sorry

end Xiaokang_position_l203_203252


namespace bridget_gave_erasers_l203_203398

variable (p_start : ℕ) (p_end : ℕ) (e_b : ℕ)

theorem bridget_gave_erasers (h1 : p_start = 8) (h2 : p_end = 11) (h3 : p_end = p_start + e_b) :
  e_b = 3 := by
  sorry

end bridget_gave_erasers_l203_203398


namespace phil_final_quarters_l203_203033

-- Define the conditions
def initial_quarters : ℕ := 50
def doubled_initial_quarters : ℕ := 2 * initial_quarters
def quarters_collected_each_month : ℕ := 3
def months_in_year : ℕ := 12
def quarters_collected_in_a_year : ℕ := quarters_collected_each_month * months_in_year
def quarters_collected_every_third_month : ℕ := 1
def quarters_collected_in_third_months : ℕ := months_in_year / 3 * quarters_collected_every_third_month
def total_before_losing : ℕ := doubled_initial_quarters + quarters_collected_in_a_year + quarters_collected_in_third_months
def lost_quarter_of_total : ℕ := total_before_losing / 4
def quarters_left : ℕ := total_before_losing - lost_quarter_of_total

-- Prove the final result
theorem phil_final_quarters : quarters_left = 105 := by
  sorry

end phil_final_quarters_l203_203033


namespace probability_multiple_choice_and_essay_correct_l203_203819

noncomputable def probability_multiple_choice_and_essay (C : ℕ → ℕ → ℕ) : ℚ :=
    (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3)

theorem probability_multiple_choice_and_essay_correct (C : ℕ → ℕ → ℕ) :
    probability_multiple_choice_and_essay C = (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3) :=
by
  sorry

end probability_multiple_choice_and_essay_correct_l203_203819


namespace value_of_expression_l203_203406

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 2

theorem value_of_expression : f (g 3) - g (f 3) = 8 :=
by
  sorry

end value_of_expression_l203_203406


namespace intersection_point_of_lines_l203_203241

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (x - 2 * y - 4 = 0) ∧ (x + 3 * y + 6 = 0) ∧ (x = 0) ∧ (y = -2) :=
by
  sorry

end intersection_point_of_lines_l203_203241


namespace min_value_of_function_l203_203074

theorem min_value_of_function (x : ℝ) (h : x > 5 / 4) : 
  ∃ ymin : ℝ, ymin = 7 ∧ ∀ y : ℝ, y = 4 * x + 1 / (4 * x - 5) → y ≥ ymin := 
sorry

end min_value_of_function_l203_203074


namespace eval_polynomial_positive_root_l203_203343

theorem eval_polynomial_positive_root : 
  ∃ x : ℝ, (x^2 - 3 * x - 10 = 0 ∧ 0 < x ∧ (x^3 - 3 * x^2 - 9 * x + 7 = 12)) :=
sorry

end eval_polynomial_positive_root_l203_203343


namespace cosine_lt_sine_neg_four_l203_203688

theorem cosine_lt_sine_neg_four : ∀ (m n : ℝ), m = Real.cos (-4) → n = Real.sin (-4) → m < n :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end cosine_lt_sine_neg_four_l203_203688


namespace geometric_sequence_a6a7_l203_203234

theorem geometric_sequence_a6a7 (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n+1) = q * a n)
  (h1 : a 4 * a 5 = 1)
  (h2 : a 8 * a 9 = 16) : a 6 * a 7 = 4 :=
sorry

end geometric_sequence_a6a7_l203_203234


namespace calories_in_250g_of_lemonade_l203_203193

structure Lemonade :=
(lemon_juice_grams : ℕ)
(sugar_grams : ℕ)
(water_grams : ℕ)
(lemon_juice_calories_per_100g : ℕ)
(sugar_calories_per_100g : ℕ)
(water_calories_per_100g : ℕ)

def calorie_count (l : Lemonade) : ℕ :=
(l.lemon_juice_grams * l.lemon_juice_calories_per_100g / 100) +
(l.sugar_grams * l.sugar_calories_per_100g / 100) +
(l.water_grams * l.water_calories_per_100g / 100)

def total_weight (l : Lemonade) : ℕ :=
l.lemon_juice_grams + l.sugar_grams + l.water_grams

def caloric_density (l : Lemonade) : ℚ :=
calorie_count l / total_weight l

theorem calories_in_250g_of_lemonade :
  ∀ (l : Lemonade), 
  l = { lemon_juice_grams := 200, sugar_grams := 300, water_grams := 500,
        lemon_juice_calories_per_100g := 40,
        sugar_calories_per_100g := 390,
        water_calories_per_100g := 0 } →
  (caloric_density l * 250 = 312.5) :=
sorry

end calories_in_250g_of_lemonade_l203_203193


namespace inequality_solution_sets_l203_203160

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end inequality_solution_sets_l203_203160


namespace contradiction_proof_l203_203320

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l203_203320


namespace area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l203_203970

-- 1. Prove that the area enclosed by x = π/2, x = 3π/2, y = 0 and y = cos x is 2
theorem area_enclosed_by_lines_and_curve : 
  ∫ (x : ℝ) in (Real.pi / 2)..(3 * Real.pi / 2), (-Real.cos x) = 2 := sorry

-- 2. Prove that the cylindrical coordinates (sqrt(2), π/4, 1) correspond to Cartesian coordinates (1, 1, 1)
theorem cylindrical_to_cartesian_coordinates :
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  let z := 1
  (r * Real.cos θ, r * Real.sin θ, z) = (1, 1, 1) := sorry

-- 3. Prove that (3 + 2i) / (2 - 3i) - (3 - 2i) / (2 + 3i) = 2i
theorem complex_number_evaluation : 
  ((3 + 2 * Complex.I) / (2 - 3 * Complex.I)) - ((3 - 2 * Complex.I) / (2 + 3 * Complex.I)) = 2 * Complex.I := sorry

-- 4. Prove that the area of triangle AOB with given polar coordinates is 2
theorem area_of_triangle_AOB :
  let A := (2, Real.pi / 6)
  let B := (4, Real.pi / 3)
  let area := 1 / 2 * (2 * 4 * Real.sin (Real.pi / 3 - Real.pi / 6))
  area = 2 := sorry

end area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l203_203970


namespace proveCarTransportationProblem_l203_203538

def carTransportationProblem :=
  ∃ x y a b : ℕ,
  -- Conditions regarding the capabilities of the cars
  (2 * x + 3 * y = 18) ∧
  (x + 2 * y = 11) ∧
  -- Conclusion (question 1)
  (x + y = 7) ∧
  -- Conditions for the rental plan (question 2)
  (3 * a + 4 * b = 27) ∧
  -- Cost optimization
  ((100 * a + 120 * b) = 820 ∨ (100 * a + 120 * b) = 860) ∧
  -- Optimal cost verification
  (100 * a + 120 * b = 820 → a = 1 ∧ b = 6)

theorem proveCarTransportationProblem : carTransportationProblem :=
  sorry

end proveCarTransportationProblem_l203_203538


namespace train_speed_clicks_l203_203275

theorem train_speed_clicks (x : ℝ) (rail_length_feet : ℝ := 40) (clicks_per_mile : ℝ := 5280/ 40) :
  15 ≤ (2400/5280) * 60  * clicks_per_mile ∧ (2400/5280) * 60 * clicks_per_mile ≤ 30 :=
by {
  sorry
}

end train_speed_clicks_l203_203275


namespace geometric_sequence_problem_l203_203909

-- Assume {a_n} is a geometric sequence with positive terms
variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Condition: all terms are positive numbers in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 0 * r ^ n

-- Condition: a_1 * a_9 = 16
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = 16

-- Question to prove: a_2 * a_5 * a_8 = 64
theorem geometric_sequence_problem
  (h_geom : is_geometric_sequence a r)
  (h_pos : ∀ n, 0 < a n)
  (h_cond1 : condition1 a) :
  a 2 * a 5 * a 8 = 64 :=
by
  sorry

end geometric_sequence_problem_l203_203909


namespace max_value_4x_plus_3y_l203_203080

theorem max_value_4x_plus_3y :
  ∃ x y : ℝ, (x^2 + y^2 = 16 * x + 8 * y + 8) ∧ (∀ w, w = 4 * x + 3 * y → w ≤ 64) ∧ ∃ x y, 4 * x + 3 * y = 64 :=
sorry

end max_value_4x_plus_3y_l203_203080


namespace min_value_of_sum_of_sides_proof_l203_203867

noncomputable def min_value_of_sum_of_sides (a b c : ℝ) (angleC : ℝ) : ℝ :=
  if (angleC = 60 * (Real.pi / 180)) ∧ ((a + b)^2 - c^2 = 4) then 4 * Real.sqrt 3 / 3 
  else 0

theorem min_value_of_sum_of_sides_proof (a b c : ℝ) (angleC : ℝ) 
  (h1 : angleC = 60 * (Real.pi / 180)) 
  (h2 : (a + b)^2 - c^2 = 4) 
  : min_value_of_sum_of_sides a b c angleC = 4 * Real.sqrt 3 / 3 := 
by
  sorry

end min_value_of_sum_of_sides_proof_l203_203867


namespace complex_power_sum_eq_self_l203_203791

theorem complex_power_sum_eq_self (z : ℂ) (h : z^2 + z + 1 = 0) : z^100 + z^101 + z^102 + z^103 = z :=
sorry

end complex_power_sum_eq_self_l203_203791


namespace minimum_cuts_for_48_pieces_l203_203296

theorem minimum_cuts_for_48_pieces 
  (rearrange_without_folding : Prop)
  (can_cut_multiple_layers_simultaneously : Prop)
  (straight_line_cut : Prop)
  (cut_doubles_pieces : ∀ n, ∃ m, m = 2 * n) :
  ∃ n, (2^n ≥ 48 ∧ ∀ m, (m < n → 2^m < 48)) ∧ n = 6 := 
by 
  sorry

end minimum_cuts_for_48_pieces_l203_203296


namespace place_synthetic_method_l203_203606

theorem place_synthetic_method :
  "Synthetic Method" = "Direct Proof" :=
sorry

end place_synthetic_method_l203_203606


namespace alpha_identity_l203_203003

theorem alpha_identity (α : ℝ) (hα : α ≠ 0) (h_tan : Real.tan α = -α) : 
    (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := 
by
  sorry

end alpha_identity_l203_203003


namespace range_of_f_on_interval_l203_203804

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem range_of_f_on_interval :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = y} :=
by
  sorry

end range_of_f_on_interval_l203_203804


namespace gcd_polynomials_l203_203879

theorem gcd_polynomials (b : ℤ) (h: ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 :=
sorry

end gcd_polynomials_l203_203879


namespace smallest_b_l203_203534

-- Define the variables and conditions
variables {a b : ℝ}

-- Assumptions based on the problem conditions
axiom h1 : 2 < a
axiom h2 : a < b

-- The theorems for the triangle inequality violations
theorem smallest_b (h : a ≥ b / (2 * b - 1)) (h' : 2 + a ≤ b) : b = (3 + Real.sqrt 7) / 2 :=
sorry

end smallest_b_l203_203534


namespace problem_l203_203623

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem problem (h : f 10 = 756) : f 10 = 756 := 
by 
  sorry

end problem_l203_203623


namespace william_wins_10_rounds_l203_203880

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l203_203880


namespace geometric_sequence_decreasing_iff_l203_203295

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def is_decreasing_sequence (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > a (n + 1)

theorem geometric_sequence_decreasing_iff (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 0 > a 1 ∧ a 1 > a 2) ↔ is_decreasing_sequence a :=
by
  sorry

end geometric_sequence_decreasing_iff_l203_203295


namespace g_value_at_8_l203_203424

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem g_value_at_8 (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g x = (1/216) * (x - (a^3)) * (x - (b^3)) * (x - (c^3))) 
  (h2 : g 0 = 1) 
  (h3 : ∀ a b c : ℝ, f (a) = 0 ∧ f (b) = 0 ∧ f (c) = 0) : 
  g 8 = 0 :=
sorry

end g_value_at_8_l203_203424


namespace sin_150_eq_half_l203_203893

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end sin_150_eq_half_l203_203893


namespace polynomial_consecutive_integers_l203_203928

theorem polynomial_consecutive_integers (a : ℤ) (c : ℤ) (P : ℤ → ℤ)
  (hP : ∀ x : ℤ, P x = 2 * x ^ 3 - 30 * x ^ 2 + c * x)
  (h_consecutive : ∃ a : ℤ, P (a - 1) + 1 = P a ∧ P a = P (a + 1) - 1) :
  a = 5 ∧ c = 149 :=
by
  sorry

end polynomial_consecutive_integers_l203_203928


namespace smallest_percentage_boys_correct_l203_203462

noncomputable def smallest_percentage_boys (B : ℝ) : ℝ :=
  if h : 0 ≤ B ∧ B ≤ 1 then B else 0

theorem smallest_percentage_boys_correct :
  ∃ B : ℝ,
    0 ≤ B ∧ B ≤ 1 ∧
    (67.5 / 100 * B * 200 + 25 / 100 * (1 - B) * 200) ≥ 101 ∧
    B = 0.6 :=
by
  sorry

end smallest_percentage_boys_correct_l203_203462


namespace opposite_of_neg_two_l203_203717

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l203_203717


namespace english_score_is_96_l203_203178

variable (Science_score : ℕ) (Social_studies_score : ℕ) (English_score : ℕ)

/-- Jimin's social studies score is 6 points higher than his science score -/
def social_studies_score_condition := Social_studies_score = Science_score + 6

/-- The science score is 87 -/
def science_score_condition := Science_score = 87

/-- The average score for science, social studies, and English is 92 -/
def average_score_condition := (Science_score + Social_studies_score + English_score) / 3 = 92

theorem english_score_is_96
  (h1 : social_studies_score_condition Science_score Social_studies_score)
  (h2 : science_score_condition Science_score)
  (h3 : average_score_condition Science_score Social_studies_score English_score) :
  English_score = 96 :=
  by
    sorry

end english_score_is_96_l203_203178


namespace contribution_per_person_correct_l203_203382

-- Definitions from conditions
def total_fundraising_goal : ℕ := 2400
def number_of_participants : ℕ := 8
def administrative_fee_per_person : ℕ := 20

-- Desired answer
def total_contribution_per_person : ℕ := total_fundraising_goal / number_of_participants + administrative_fee_per_person

-- Proof statement
theorem contribution_per_person_correct :
  total_contribution_per_person = 320 :=
by
  sorry  -- Proof to be provided

end contribution_per_person_correct_l203_203382


namespace jacqueline_erasers_l203_203708

def num_boxes : ℕ := 4
def erasers_per_box : ℕ := 10
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end jacqueline_erasers_l203_203708


namespace number_of_saturday_sales_l203_203190

def caricatures_sold_on_saturday (total_earnings weekend_earnings price_per_drawing sunday_sales : ℕ) : ℕ :=
  (total_earnings - (sunday_sales * price_per_drawing)) / price_per_drawing

theorem number_of_saturday_sales : caricatures_sold_on_saturday 800 800 20 16 = 24 := 
by 
  sorry

end number_of_saturday_sales_l203_203190


namespace negation_of_existence_l203_203933

theorem negation_of_existence :
  ¬ (∃ (x_0 : ℝ), x_0^2 - x_0 + 1 ≤ 0) ↔ ∀ (x : ℝ), x^2 - x + 1 > 0 :=
by
  sorry

end negation_of_existence_l203_203933


namespace neg_sub_eq_sub_l203_203363

theorem neg_sub_eq_sub (a b : ℝ) : - (a - b) = b - a := 
by
  sorry

end neg_sub_eq_sub_l203_203363


namespace at_least_one_gt_one_of_sum_gt_two_l203_203013

theorem at_least_one_gt_one_of_sum_gt_two (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 := 
by sorry

end at_least_one_gt_one_of_sum_gt_two_l203_203013


namespace necessary_but_not_sufficient_conditions_l203_203310

theorem necessary_but_not_sufficient_conditions (x y : ℝ) :
  (|x| ≤ 1 ∧ |y| ≤ 1) → x^2 + y^2 ≤ 1 ∨ ¬(x^2 + y^2 ≤ 1) → 
  (|x| ≤ 1 ∧ |y| ≤ 1) → (x^2 + y^2 ≤ 1 → (|x| ≤ 1 ∧ |y| ≤ 1)) :=
by
  sorry

end necessary_but_not_sufficient_conditions_l203_203310


namespace sum_of_three_integers_with_product_5_pow_4_l203_203412

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l203_203412


namespace ellipse_eccentricity_l203_203300

theorem ellipse_eccentricity
  {a b n : ℝ}
  (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (P : ℝ × ℝ), P.1 = n ∧ P.2 = 4 ∧ (n^2 / a^2 + 16 / b^2 = 1))
  (F1 F2 : ℝ × ℝ)
  (h4 : F1 = (c, 0))        -- Placeholders for focus coordinates of the ellipse
  (h5 : F2 = (-c, 0))
  (h6 : ∃ c, 4*c = (3 / 2) * (a + c))
  : 3 * c = 5 * a → c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l203_203300


namespace temperature_drop_l203_203214

theorem temperature_drop (initial_temperature drop: ℤ) (h1: initial_temperature = 3) (h2: drop = 5) : initial_temperature - drop = -2 :=
by {
  sorry
}

end temperature_drop_l203_203214


namespace triangle_area_solution_l203_203278

noncomputable def triangle_area (a b : ℝ) : ℝ := 
  let r := 6 -- radius of each circle
  let d := 2 -- derived distance
  let s := 2 * Real.sqrt 3 * d -- side length of the equilateral triangle
  let area := (Real.sqrt 3 / 4) * s^2 
  area

theorem triangle_area_solution : ∃ a b : ℝ, 
  triangle_area a b = 3 * Real.sqrt 3 ∧ 
  a + b = 27 := 
by 
  exists 27
  exists 3
  sorry

end triangle_area_solution_l203_203278


namespace find_values_of_a_to_make_lines_skew_l203_203744

noncomputable def lines_are_skew (t u a : ℝ) : Prop :=
  ∀ t u,
    (1 + 2 * t = 4 + 5 * u ∧
     2 + 3 * t = 1 + 2 * u ∧
     a + 4 * t = u) → false

theorem find_values_of_a_to_make_lines_skew :
  ∀ a : ℝ, ¬ a = 3 ↔ lines_are_skew t u a :=
by
  sorry

end find_values_of_a_to_make_lines_skew_l203_203744


namespace jamal_bought_4_half_dozens_l203_203042

/-- Given that each crayon costs $2, the total cost is $48, and a half dozen is 6 crayons,
    prove that Jamal bought 4 half dozens of crayons. -/
theorem jamal_bought_4_half_dozens (cost_per_crayon : ℕ) (total_cost : ℕ) (half_dozen : ℕ) 
  (h1 : cost_per_crayon = 2) (h2 : total_cost = 48) (h3 : half_dozen = 6) : 
  (total_cost / cost_per_crayon) / half_dozen = 4 := 
by 
  sorry

end jamal_bought_4_half_dozens_l203_203042


namespace remainder_when_n_add_3006_divided_by_6_l203_203329

theorem remainder_when_n_add_3006_divided_by_6 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end remainder_when_n_add_3006_divided_by_6_l203_203329


namespace visitors_equal_cats_l203_203134

-- Definition for conditions
def visitors_pets_cats (V C : ℕ) : Prop :=
  (∃ P : ℕ, P = 3 * V ∧ P = 3 * C)

-- Statement of the proof problem
theorem visitors_equal_cats {V C : ℕ}
  (h : visitors_pets_cats V C) : V = C :=
by sorry

end visitors_equal_cats_l203_203134


namespace work_done_by_6_men_and_11_women_l203_203389

-- Definitions based on conditions
def work_completed_by_men (men : ℕ) (days : ℕ) : ℚ := men / (8 * days)
def work_completed_by_women (women : ℕ) (days : ℕ) : ℚ := women / (12 * days)
def combined_work_rate (men : ℕ) (women : ℕ) (days : ℕ) : ℚ := 
  work_completed_by_men men days + work_completed_by_women women days

-- Problem statement
theorem work_done_by_6_men_and_11_women :
  combined_work_rate 6 11 12 = 1 := by
  sorry

end work_done_by_6_men_and_11_women_l203_203389


namespace value_of_a_l203_203773

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem value_of_a (a : ℝ) (h : abs ((a^2) - a) = a / 2) : a = 1 / 2 ∨ a = 3 / 2 := by
  sorry

end value_of_a_l203_203773


namespace parabolic_points_l203_203078

noncomputable def A (x1 : ℝ) (y1 : ℝ) : Prop := y1 = x1^2 - 3
noncomputable def B (x2 : ℝ) (y2 : ℝ) : Prop := y2 = x2^2 - 3

theorem parabolic_points (x1 x2 y1 y2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (hA : A x1 y1) (hB : B x2 y2) : y1 < y2 :=
by
  sorry

end parabolic_points_l203_203078


namespace sum_of_A_and_B_l203_203760

theorem sum_of_A_and_B (A B : ℕ) (h1 : (1 / 6 : ℚ) * (1 / 3) = 1 / (A * 3))
                       (h2 : (1 / 6 : ℚ) * (1 / 3) = 1 / B) : A + B = 24 :=
by
  sorry

end sum_of_A_and_B_l203_203760


namespace total_weight_l203_203981

def weights (M D C : ℕ): Prop :=
  D = 46 ∧ D + C = 60 ∧ C = M / 5

theorem total_weight (M D C : ℕ) (h : weights M D C) : M + D + C = 130 :=
by
  cases h with
  | intro h1 h2 =>
    cases h2 with
    | intro h2_1 h2_2 => 
      sorry

end total_weight_l203_203981


namespace triangle_shape_statements_l203_203820

theorem triangle_shape_statements (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (h : a^2 + b^2 + c^2 = ab + bc + ca) :
  (a = b ∧ b = c ∧ a = c) :=
by
  sorry 

end triangle_shape_statements_l203_203820


namespace count_valid_three_digit_numbers_l203_203617

theorem count_valid_three_digit_numbers : 
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  valid_numbers = 90 :=
by
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  have h1 : valid_numbers = 900 - 810 := by rfl
  have h2 : 900 - 810 = 90 := by norm_num
  exact h1.trans h2

end count_valid_three_digit_numbers_l203_203617


namespace general_term_a_general_term_b_sum_c_l203_203044

-- Problem 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

-- Problem 2: General term formula for the sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b1 : b 1 = 1)
  (h_b : ∀ n, b (n + 1) = b n + a n) (h_a : ∀ n, a n = (1 / 2) ^ (n - 1)) :
  ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1) := 
sorry

-- Problem 3: Sum of the first n terms for the sequence {c_n}
theorem sum_c (c : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b : ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1)) (h_c : ∀ n, c n = n * (3 - b n)) :
  ∀ n, T n = 8 - (8 + 4 * n) * (1 / 2) ^ n := 
sorry

end general_term_a_general_term_b_sum_c_l203_203044


namespace arithmetic_sequence_a4_l203_203883

theorem arithmetic_sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) :
  S 8 = 30 → S 4 = 7 → 
      (∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) → 
      a 4 = a1 + 3 * d → 
      a 4 = 13 / 4 := by
  intros hS8 hS4 hS_formula ha4_formula
  -- Formal proof to be filled in
  sorry

end arithmetic_sequence_a4_l203_203883


namespace no_rectangular_prism_equal_measures_l203_203425

theorem no_rectangular_prism_equal_measures (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0): 
  ¬ (4 * (a + b + c) = 2 * (a * b + b * c + c * a) ∧ 2 * (a * b + b * c + c * a) = a * b * c) :=
by
  sorry

end no_rectangular_prism_equal_measures_l203_203425


namespace greatest_x_value_l203_203081

theorem greatest_x_value : 
  ∃ x : ℝ, (∀ y : ℝ, (y = (4 * x - 16) / (3 * x - 4)) → (y^2 + y = 12)) ∧ (x = 2) := by
  sorry

end greatest_x_value_l203_203081


namespace fraction_of_paint_first_week_l203_203083

-- Definitions based on conditions
def total_paint := 360
def fraction_first_week (f : ℚ) : ℚ := f * total_paint
def paint_remaining_first_week (f : ℚ) : ℚ := total_paint - fraction_first_week f
def fraction_second_week (f : ℚ) : ℚ := (1 / 5) * paint_remaining_first_week f
def total_paint_used (f : ℚ) : ℚ := fraction_first_week f + fraction_second_week f
def total_paint_used_value := 104

-- Proof problem statement
theorem fraction_of_paint_first_week (f : ℚ) (h : total_paint_used f = total_paint_used_value) : f = 1 / 9 := 
sorry

end fraction_of_paint_first_week_l203_203083


namespace number_of_girls_l203_203091

theorem number_of_girls (total_children boys girls : ℕ) 
    (total_children_eq : total_children = 60)
    (boys_eq : boys = 22)
    (compute_girls : girls = total_children - boys) : 
    girls = 38 :=
by
    rw [total_children_eq, boys_eq] at compute_girls
    simp at compute_girls
    exact compute_girls

end number_of_girls_l203_203091


namespace tetrahedron_face_area_squared_l203_203897

variables {S0 S1 S2 S3 α12 α13 α23 : ℝ}

-- State the theorem
theorem tetrahedron_face_area_squared :
  (S0)^2 = (S1)^2 + (S2)^2 + (S3)^2 - 2 * S1 * S2 * (Real.cos α12) - 2 * S1 * S3 * (Real.cos α13) - 2 * S2 * S3 * (Real.cos α23) :=
sorry

end tetrahedron_face_area_squared_l203_203897


namespace train_length_l203_203530

theorem train_length (time_crossing : ℕ) (speed_kmh : ℕ) (conversion_factor : ℕ) (expected_length : ℕ) :
  time_crossing = 4 ∧ speed_kmh = 144 ∧ conversion_factor = 1000 / 3600 * 144 →
  expected_length = 160 :=
by
  sorry

end train_length_l203_203530


namespace area_union_after_rotation_l203_203102

-- Define the sides of the triangle
def PQ : ℝ := 11
def QR : ℝ := 13
def PR : ℝ := 12

-- Define the condition that H is the centroid of the triangle PQR
def centroid (P Q R H : ℝ × ℝ) : Prop := sorry -- This definition would require geometric relationships.

-- Statement to prove the area of the union of PQR and P'Q'R' after 180° rotation about H.
theorem area_union_after_rotation (P Q R H : ℝ × ℝ) (hPQ : dist P Q = PQ) (hQR : dist Q R = QR) (hPR : dist P R = PR) (hH : centroid P Q R H) : 
  let s := (PQ + QR + PR) / 2
  let area_PQR := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  2 * area_PQR = 12 * Real.sqrt 105 :=
sorry

end area_union_after_rotation_l203_203102


namespace tan_600_eq_neg_sqrt_3_l203_203257

theorem tan_600_eq_neg_sqrt_3 : Real.tan (600 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_600_eq_neg_sqrt_3_l203_203257


namespace range_of_fx1_l203_203608

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem range_of_fx1 (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) :
    f x1 a > (1 - 2 * Real.log 2) / 4 :=
sorry

end range_of_fx1_l203_203608


namespace largest_5_digit_congruent_l203_203487

theorem largest_5_digit_congruent (n : ℕ) (h1 : 29 * n + 17 < 100000) : 29 * 3447 + 17 = 99982 :=
by
  -- Proof goes here
  sorry

end largest_5_digit_congruent_l203_203487


namespace bianca_ate_candy_l203_203815

theorem bianca_ate_candy (original_candies : ℕ) (pieces_per_pile : ℕ) 
                         (number_of_piles : ℕ) 
                         (remaining_candies : ℕ) 
                         (h_original : original_candies = 78) 
                         (h_pieces_per_pile : pieces_per_pile = 8) 
                         (h_number_of_piles : number_of_piles = 6) 
                         (h_remaining : remaining_candies = pieces_per_pile * number_of_piles) :
  original_candies - remaining_candies = 30 := by
  subst_vars
  sorry

end bianca_ate_candy_l203_203815


namespace mrs_wong_initial_valentines_l203_203962

theorem mrs_wong_initial_valentines (x : ℕ) (given left : ℕ) (h_given : given = 8) (h_left : left = 22) (h_initial : x = left + given) : x = 30 :=
by
  rw [h_left, h_given] at h_initial
  exact h_initial

end mrs_wong_initial_valentines_l203_203962


namespace SallyMcQueenCostCorrect_l203_203746

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end SallyMcQueenCostCorrect_l203_203746


namespace eating_possible_values_l203_203411

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end eating_possible_values_l203_203411


namespace train_length_l203_203055

noncomputable def speed_kmph := 90
noncomputable def time_sec := 5
noncomputable def speed_mps := speed_kmph * 1000 / 3600

theorem train_length : (speed_mps * time_sec) = 125 := by
  -- We need to assert and prove this theorem
  sorry

end train_length_l203_203055


namespace f_x_f_2x_plus_1_l203_203208

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem f_x (x : ℝ) : f x = x^2 - 2 * x - 3 := 
by sorry

theorem f_2x_plus_1 (x : ℝ) : f (2 * x + 1) = 4 * x^2 - 4 := 
by sorry

end f_x_f_2x_plus_1_l203_203208


namespace exists_another_nice_triple_l203_203201

noncomputable def is_nice_triple (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ (b - a) = (c - b)) ∧
  (Nat.gcd b a = 1 ∧ Nat.gcd b c = 1) ∧ 
  (∃ k, a * b * c = k^2)

theorem exists_another_nice_triple (a b c : ℕ) 
  (h : is_nice_triple a b c) : ∃ a' b' c', 
  (is_nice_triple a' b' c') ∧ 
  (a' = a ∨ a' = b ∨ a' = c ∨ 
   b' = a ∨ b' = b ∨ b' = c ∨ 
   c' = a ∨ c' = b ∨ c' = c) :=
by sorry

end exists_another_nice_triple_l203_203201


namespace odd_divisibility_l203_203751

theorem odd_divisibility (n : ℕ) (k : ℕ) (x y : ℤ) (h : n = 2 * k + 1) : (x^n + y^n) % (x + y) = 0 :=
by sorry

end odd_divisibility_l203_203751


namespace evaluate_custom_op_l203_203555

def custom_op (a b : ℝ) : ℝ := (a - b)^2

theorem evaluate_custom_op (x y : ℝ) : custom_op ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 :=
by
  sorry

end evaluate_custom_op_l203_203555


namespace topology_on_X_l203_203860

-- Define the universal set X
def X : Set ℕ := {1, 2, 3}

-- Sequences of candidate sets v
def v1 : Set (Set ℕ) := {∅, {1}, {3}, {1, 2, 3}}
def v2 : Set (Set ℕ) := {∅, {2}, {3}, {2, 3}, {1, 2, 3}}
def v3 : Set (Set ℕ) := {∅, {1}, {1, 2}, {1, 3}}
def v4 : Set (Set ℕ) := {∅, {1, 3}, {2, 3}, {3}, {1, 2, 3}}

-- Define the conditions that determine a topology
def isTopology (X : Set ℕ) (v : Set (Set ℕ)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧ 
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋃₀ s ∈ v) ∧
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋂₀ s ∈ v)

-- The statement we want to prove
theorem topology_on_X : 
  isTopology X v2 ∧ isTopology X v4 :=
by
  sorry

end topology_on_X_l203_203860


namespace books_leftover_l203_203901

-- Definitions of the conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def books_bought : ℕ := 26

-- The theorem stating the proof problem
theorem books_leftover : (initial_books + books_bought) - (shelves * books_per_shelf) = 2 := by
  sorry

end books_leftover_l203_203901


namespace ratio_of_volumes_l203_203992

theorem ratio_of_volumes (A B : ℝ) (h : (3 / 4) * A = (2 / 3) * B) : A / B = 8 / 9 :=
by
  sorry

end ratio_of_volumes_l203_203992


namespace student_B_more_consistent_l203_203092

noncomputable def standard_deviation_A := 5.09
noncomputable def standard_deviation_B := 3.72
def games_played := 7
noncomputable def average_score_A := 16
noncomputable def average_score_B := 16

theorem student_B_more_consistent :
  standard_deviation_B < standard_deviation_A :=
sorry

end student_B_more_consistent_l203_203092


namespace y_at_x8_l203_203210

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l203_203210


namespace standard_circle_equation_passing_through_P_l203_203733

-- Define the condition that a point P is a solution to the system of equations derived from the line
def PointPCondition (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1 = 0) ∧ (3 * x - 2 * y + 5 = 0)

-- Define the center and radius of the given circle C
def CenterCircleC : ℝ × ℝ := (2, -3)
def RadiusCircleC : ℝ := 4  -- Since the radius squared is 16

-- Define the condition that a point is on a circle with a given center and radius
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.fst)^2 + (y + center.snd)^2 = radius^2

-- State the problem
theorem standard_circle_equation_passing_through_P :
  ∃ (x y : ℝ), PointPCondition x y ∧ OnCircle CenterCircleC 5 x y :=
sorry

end standard_circle_equation_passing_through_P_l203_203733


namespace compute_expression_l203_203483

noncomputable def quadratic_roots (a b c : ℝ) :
  {x : ℝ × ℝ // a * x.fst^2 + b * x.fst + c = 0 ∧ a * x.snd^2 + b * x.snd + c = 0} :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  ⟨(root1, root2), by sorry⟩

theorem compute_expression :
  let roots := quadratic_roots 5 (-3) (-4)
  let x1 := roots.val.fst
  let x2 := roots.val.snd
  2 * x1^2 + 3 * x2^2 = (178 : ℝ) / 25 := by
  sorry

end compute_expression_l203_203483


namespace gold_copper_alloy_ratio_l203_203850

theorem gold_copper_alloy_ratio 
  (G C : ℝ) 
  (h_gold : G / weight_of_water = 19) 
  (h_copper : C / weight_of_water = 9)
  (weight_of_alloy : (G + C) / weight_of_water = 17) :
  G / C = 4 :=
sorry

end gold_copper_alloy_ratio_l203_203850


namespace annika_return_time_l203_203106

-- Define the rate at which Annika hikes.
def hiking_rate := 10 -- minutes per kilometer

-- Define the distances mentioned in the problem.
def initial_distance_east := 2.5 -- kilometers
def total_distance_east := 3.5 -- kilometers

-- Define the time calculations.
def additional_distance_east := total_distance_east - initial_distance_east

-- Calculate the total time required for Annika to get back to the start.
theorem annika_return_time (rate : ℝ) (initial_dist : ℝ) (total_dist : ℝ) (additional_dist : ℝ) : 
  initial_dist = 2.5 → total_dist = 3.5 → rate = 10 → additional_dist = total_dist - initial_dist → 
  (2.5 * rate + additional_dist * rate * 2) = 45 :=
by
-- Since this is just the statement and no proof is needed, we use sorry
sorry

end annika_return_time_l203_203106


namespace bandit_showdown_l203_203185

theorem bandit_showdown :
  ∃ b : ℕ, b ≥ 8 ∧ b < 50 ∧
         ∀ i j : ℕ, i ≠ j → (i < 50 ∧ j < 50) →
         ∃ k : ℕ, k < 50 ∧
         ∀ b : ℕ, b < 50 → 
         ∃ l m : ℕ, l ≠ m ∧ l < 50 ∧ m < 50 ∧ l ≠ b ∧ m ≠ b :=
sorry

end bandit_showdown_l203_203185


namespace pastries_sold_is_correct_l203_203399

-- Definitions of the conditions
def initial_pastries : ℕ := 56
def remaining_pastries : ℕ := 27

-- Statement of the theorem
theorem pastries_sold_is_correct : initial_pastries - remaining_pastries = 29 :=
by
  sorry

end pastries_sold_is_correct_l203_203399


namespace fraction_of_ABCD_is_shaded_l203_203162

noncomputable def squareIsDividedIntoTriangles : Type := sorry
noncomputable def areTrianglesIdentical (s : squareIsDividedIntoTriangles) : Prop := sorry
noncomputable def isFractionShadedCorrect : Prop := 
  ∃ (s : squareIsDividedIntoTriangles), 
  areTrianglesIdentical s ∧ 
  (7 / 16 : ℚ) = 7 / 16

theorem fraction_of_ABCD_is_shaded (s : squareIsDividedIntoTriangles) :
  areTrianglesIdentical s → (7 / 16 : ℚ) = 7 / 16 :=
sorry

end fraction_of_ABCD_is_shaded_l203_203162


namespace principal_made_mistake_l203_203163

-- Definitions based on given conditions
def students_per_class (x : ℤ) : Prop := x > 0
def total_students (x : ℤ) : ℤ := 2 * x
def non_failing_grades (y : ℤ) : ℤ := y
def failing_grades (y : ℤ) : ℤ := y + 11
def total_grades (x y : ℤ) : Prop := total_students x = non_failing_grades y + failing_grades y

-- Proposition stating the principal made a mistake
theorem principal_made_mistake (x y : ℤ) (hx : students_per_class x) : ¬ total_grades x y :=
by
  -- Assume the proof for the hypothesis is required here
  sorry

end principal_made_mistake_l203_203163


namespace academic_academy_pass_criteria_l203_203493

theorem academic_academy_pass_criteria :
  ∀ (total_problems : ℕ) (passing_percentage : ℕ)
  (max_missed : ℕ),
  total_problems = 35 →
  passing_percentage = 80 →
  max_missed = total_problems - (passing_percentage * total_problems) / 100 →
  max_missed = 7 :=
by 
  intros total_problems passing_percentage max_missed
  intros h_total_problems h_passing_percentage h_calculation
  rw [h_total_problems, h_passing_percentage] at h_calculation
  sorry

end academic_academy_pass_criteria_l203_203493


namespace kyler_wins_one_game_l203_203807

theorem kyler_wins_one_game :
  ∃ (Kyler_wins : ℕ),
    (Kyler_wins + 3 + 2 + 2 = 6 ∧
    Kyler_wins + 3 = 6 ∧
    Kyler_wins = 1) := by
  sorry

end kyler_wins_one_game_l203_203807


namespace ceiling_fraction_evaluation_l203_203114

theorem ceiling_fraction_evaluation :
  (Int.ceil ((19 : ℚ) / 8 - Int.ceil ((45 : ℚ) / 19)) / Int.ceil ((45 : ℚ) / 8 + Int.ceil ((8 * 19 : ℚ) / 45))) = 0 :=
by
  sorry

end ceiling_fraction_evaluation_l203_203114


namespace time_for_each_student_l203_203996

-- Define the conditions as variables
variables (num_students : ℕ) (period_length : ℕ) (num_periods : ℕ)
-- Assume the conditions from the problem
def conditions := num_students = 32 ∧ period_length = 40 ∧ num_periods = 4

-- Define the total time available
def total_time (num_periods period_length : ℕ) := num_periods * period_length

-- Define the time per student
def time_per_student (total_time num_students : ℕ) := total_time / num_students

-- State the theorem to be proven
theorem time_for_each_student : 
  conditions num_students period_length num_periods →
  time_per_student (total_time num_periods period_length) num_students = 5 := sorry

end time_for_each_student_l203_203996


namespace square_adjacent_to_multiple_of_5_l203_203441

theorem square_adjacent_to_multiple_of_5 (n : ℤ) (h : n % 5 ≠ 0) : (∃ k : ℤ, n^2 = 5 * k + 1) ∨ (∃ k : ℤ, n^2 = 5 * k - 1) := 
by
  sorry

end square_adjacent_to_multiple_of_5_l203_203441


namespace sticker_probability_l203_203306

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l203_203306


namespace initial_pokemon_cards_l203_203783

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l203_203783


namespace sale_in_first_month_is_5420_l203_203696

-- Definitions of the sales in months 2 to 6
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month4 : ℕ := 6350
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470

-- Definition of the average sale goal
def average_sale_goal : ℕ := 6100

-- Calculating the total needed sales to achieve the average sale goal
def total_required_sales := 6 * average_sale_goal

-- Known sales for months 2 to 6
def known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6

-- Definition of the sale in the first month
def sale_month1 := total_required_sales - known_sales

-- The proof statement that the sale in the first month is 5420
theorem sale_in_first_month_is_5420 : sale_month1 = 5420 := by
  sorry

end sale_in_first_month_is_5420_l203_203696


namespace prob_correct_l203_203247

-- Define the individual probabilities.
def prob_first_ring := 1 / 10
def prob_second_ring := 3 / 10
def prob_third_ring := 2 / 5
def prob_fourth_ring := 1 / 10

-- Define the total probability of answering within the first four rings.
def prob_answer_within_four_rings := 
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring

-- State the theorem.
theorem prob_correct : prob_answer_within_four_rings = 9 / 10 :=
by
  -- We insert a placeholder for the proof.
  sorry

end prob_correct_l203_203247


namespace segment_length_after_reflection_l203_203844

structure Point :=
(x : ℝ)
(y : ℝ)

def reflect_over_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

def distance (p1 p2 : Point) : ℝ :=
abs (p1.y - p2.y)

theorem segment_length_after_reflection :
  let C : Point := {x := -3, y := 2}
  let C' : Point := reflect_over_x_axis C
  distance C C' = 4 :=
by
  sorry

end segment_length_after_reflection_l203_203844


namespace infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l203_203237

-- Define x, y, z to be natural numbers
def has_infinitely_many_solutions : Prop :=
  ∃ (x y z : ℕ), x^2 + 2 * y^2 = z^2

-- Prove that there are infinitely many such x, y, z
theorem infinite_solutions_x2_plus_2y2_eq_z2 : has_infinitely_many_solutions :=
  sorry

-- Define x, y, z, t to be integers and non-zero
def no_nontrivial_integer_quadruplets : Prop :=
  ∀ (x y z t : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) → 
    ¬((x^2 + 2 * y^2 = z^2) ∧ (2 * x^2 + y^2 = t^2))

-- Prove that no nontrivial integer quadruplets exist
theorem no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2 : no_nontrivial_integer_quadruplets :=
  sorry

end infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l203_203237


namespace cafeteria_pies_l203_203923

theorem cafeteria_pies (total_apples handed_out_per_student apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_per_student = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_per_student) / apples_per_pie = 5 := by
  sorry

end cafeteria_pies_l203_203923


namespace parity_of_magazines_and_celebrities_l203_203222

-- Define the main problem statement using Lean 4

theorem parity_of_magazines_and_celebrities {m c : ℕ}
  (h1 : ∀ i, i < m → ∃ d_i, d_i % 2 = 1)
  (h2 : ∀ j, j < c → ∃ e_j, e_j % 2 = 1) :
  (m % 2 = c % 2) ∧ (∃ ways, ways = 2 ^ ((m - 1) * (c - 1))) :=
by
  sorry

end parity_of_magazines_and_celebrities_l203_203222


namespace problem_statement_l203_203475

theorem problem_statement (x y : ℝ) (p : x > 0 ∧ y > 0) : (∃ p, p → xy > 0) ∧ ¬(xy > 0 → x > 0 ∧ y > 0) :=
by
  sorry

end problem_statement_l203_203475


namespace direct_proportion_m_value_l203_203526

theorem direct_proportion_m_value (m : ℝ) : 
  (∀ x: ℝ, y = -7 * x + 2 + m -> y = k * x) -> m = -2 :=
by
  sorry

end direct_proportion_m_value_l203_203526


namespace solve_for_b_l203_203497

/-- 
Given the ellipse \( x^2 + \frac{y^2}{b^2 + 1} = 1 \) where \( b > 0 \),
and the eccentricity of the ellipse is \( \frac{\sqrt{10}}{10} \),
prove that \( b = \frac{1}{3} \).
-/
theorem solve_for_b (b : ℝ) (hb : b > 0) (heccentricity : b / (Real.sqrt (b^2 + 1)) = Real.sqrt 10 / 10) : 
  b = 1 / 3 :=
sorry

end solve_for_b_l203_203497


namespace point_in_fourth_quadrant_l203_203848

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l203_203848


namespace ratio_of_surface_areas_l203_203537

theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 4 :=
by
  sorry

end ratio_of_surface_areas_l203_203537


namespace sufficient_but_not_necessary_condition_l203_203912

def p (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def q (x a : ℝ) : Prop := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬ p x) ↔ a ≤ -1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l203_203912


namespace factor_expression_l203_203198

-- Define the expression to be factored
def expr (b : ℝ) := 348 * b^2 + 87 * b + 261

-- Define the supposedly factored form of the expression
def factored_expr (b : ℝ) := 87 * (4 * b^2 + b + 3)

-- The theorem stating that the original expression is equal to its factored form
theorem factor_expression (b : ℝ) : expr b = factored_expr b := 
by
  unfold expr factored_expr
  sorry

end factor_expression_l203_203198


namespace base_h_addition_eq_l203_203548

theorem base_h_addition_eq (h : ℕ) (h_eq : h = 9) : 
  (8 * h^3 + 3 * h^2 + 7 * h + 4) + (6 * h^3 + 9 * h^2 + 2 * h + 5) = 1 * h^4 + 5 * h^3 + 3 * h^2 + 0 * h + 9 :=
by
  rw [h_eq]
  sorry

end base_h_addition_eq_l203_203548


namespace pyramid_total_blocks_l203_203574

-- Define the number of layers in the pyramid
def num_layers : ℕ := 8

-- Define the block multiplier for each subsequent layer
def block_multiplier : ℕ := 5

-- Define the number of blocks in the top layer
def top_layer_blocks : ℕ := 3

-- Define the total number of sandstone blocks
def total_blocks_pyramid : ℕ :=
  let rec total_blocks (layer : ℕ) (blocks : ℕ) :=
    if layer = 0 then blocks
    else blocks + total_blocks (layer - 1) (blocks * block_multiplier)
  total_blocks (num_layers - 1) top_layer_blocks

theorem pyramid_total_blocks :
  total_blocks_pyramid = 312093 :=
by
  -- Proof omitted
  sorry

end pyramid_total_blocks_l203_203574


namespace base_conversion_sum_l203_203997

noncomputable def A : ℕ := 10

noncomputable def base11_to_nat (x y z : ℕ) : ℕ :=
  x * 11^2 + y * 11^1 + z * 11^0

noncomputable def base12_to_nat (x y z : ℕ) : ℕ :=
  x * 12^2 + y * 12^1 + z * 12^0

theorem base_conversion_sum :
  base11_to_nat 3 7 9 + base12_to_nat 3 9 A = 999 :=
by
  sorry

end base_conversion_sum_l203_203997


namespace ducks_to_total_ratio_l203_203429

-- Definitions based on the given conditions
def totalBirds : ℕ := 15
def costPerChicken : ℕ := 2
def totalCostForChickens : ℕ := 20

-- Proving the desired ratio of ducks to total number of birds
theorem ducks_to_total_ratio : (totalCostForChickens / costPerChicken) + d = totalBirds → d = 15 - (totalCostForChickens / costPerChicken) → 
  (totalCostForChickens / costPerChicken) + d = totalBirds → d = totalBirds - (totalCostForChickens / costPerChicken) →
  d = 5 → (totalBirds - (totalCostForChickens / costPerChicken)) / totalBirds = 1 / 3 :=
by
  sorry

end ducks_to_total_ratio_l203_203429


namespace sum_and_count_even_l203_203954

-- Sum of integers from a to b (inclusive)
def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Number of even integers from a to b (inclusive)
def count_even_integers (a b : ℕ) : ℕ :=
  ((b - if b % 2 == 0 then 0 else 1) - (a + if a % 2 == 0 then 0 else 1)) / 2 + 1

theorem sum_and_count_even (x y : ℕ) :
  x = sum_of_integers 20 40 →
  y = count_even_integers 20 40 →
  x + y = 641 :=
by
  intros
  sorry

end sum_and_count_even_l203_203954


namespace average_movers_l203_203898

noncomputable def average_people_per_hour (total_people : ℕ) (total_hours : ℕ) : ℝ :=
  total_people / total_hours

theorem average_movers :
  average_people_per_hour 5000 168 = 29.76 :=
by
  sorry

end average_movers_l203_203898


namespace nested_expression_evaluation_l203_203650

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 :=
by
  sorry

end nested_expression_evaluation_l203_203650


namespace even_numbers_count_l203_203955

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end even_numbers_count_l203_203955


namespace parameter_for_three_distinct_solutions_l203_203914

open Polynomial

theorem parameter_for_three_distinct_solutions (a : ℝ) :
  (∀ x : ℝ, x^4 - 40 * x^2 + 144 = a * (x^2 + 4 * x - 12)) →
  (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  (x1^4 - 40 * x1^2 + 144 = a * (x1^2 + 4 * x1 - 12) ∧ 
   x2^4 - 40 * x2^2 + 144 = a * (x2^2 + 4 * x2 - 12) ∧ 
   x3^4 - 40 * x3^2 + 144 = a * (x3^2 + 4 * x3 - 12) ∧
   x4^4 - 40 * x4^2 + 144 = a * (x4^2 + 4 * x4 - 12))) → a = 48 :=
by
  sorry

end parameter_for_three_distinct_solutions_l203_203914


namespace number_of_divisors_of_2018_or_2019_is_7_l203_203589

theorem number_of_divisors_of_2018_or_2019_is_7 (h1 : Prime 673) (h2 : Prime 1009) : 
  Nat.card {d : Nat | d ∣ 2018 ∨ d ∣ 2019} = 7 := 
  sorry

end number_of_divisors_of_2018_or_2019_is_7_l203_203589


namespace tim_biking_time_l203_203645

theorem tim_biking_time
  (work_days : ℕ := 5) 
  (distance_to_work : ℕ := 20) 
  (weekend_ride : ℕ := 200) 
  (speed : ℕ := 25) 
  (weekly_work_distance := 2 * distance_to_work * work_days)
  (total_distance := weekly_work_distance + weekend_ride) : 
  (total_distance / speed = 16) := 
by
  sorry

end tim_biking_time_l203_203645


namespace correct_statements_l203_203518

namespace ProofProblem

variable (f : ℕ+ × ℕ+ → ℕ+)
variable (h1 : f (1, 1) = 1)
variable (h2 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2)
variable (h3 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1))

theorem correct_statements :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by
  sorry

end ProofProblem

end correct_statements_l203_203518


namespace fff1_eq_17_l203_203460

def f (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3 * n + 2
  else 2 * n - 1

theorem fff1_eq_17 : f (f (f 1)) = 17 :=
  by sorry

end fff1_eq_17_l203_203460


namespace sequence_value_at_99_l203_203501

theorem sequence_value_at_99 :
  ∃ a : ℕ → ℚ, (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = a n + n / 2) ∧ (a 99 = 2427.5) :=
by
  sorry

end sequence_value_at_99_l203_203501


namespace appropriate_presentation_length_l203_203341

-- Definitions and conditions
def ideal_speaking_rate : ℕ := 160
def min_minutes : ℕ := 20
def max_minutes : ℕ := 40
def appropriate_words_range (words : ℕ) : Prop :=
  words ≥ (min_minutes * ideal_speaking_rate) ∧ words ≤ (max_minutes * ideal_speaking_rate)

-- Statement to prove
theorem appropriate_presentation_length : appropriate_words_range 5000 :=
by sorry

end appropriate_presentation_length_l203_203341


namespace total_area_of_squares_l203_203355

-- Condition 1: Definition of the side length
def side_length (s : ℝ) : Prop := s = 12

-- Condition 2: Definition of the center of one square coinciding with the vertex of another
-- Here, we assume the positions are fixed so this condition is given
def coincide_center_vertex (s₁ s₂ : ℝ) : Prop := s₁ = s₂ 

-- The main theorem statement
theorem total_area_of_squares
  (s₁ s₂ : ℝ) 
  (h₁ : side_length s₁)
  (h₂ : side_length s₂)
  (h₃ : coincide_center_vertex s₁ s₂) :
  (2 * s₁^2) - (s₁^2 / 4) = 252 :=
by
  sorry

end total_area_of_squares_l203_203355


namespace value_of_xyz_l203_203140

theorem value_of_xyz (x y z : ℂ) 
  (h1 : x * y + 5 * y = -20)
  (h2 : y * z + 5 * z = -20)
  (h3 : z * x + 5 * x = -20) :
  x * y * z = 80 := 
by
  sorry

end value_of_xyz_l203_203140


namespace problem_statement_l203_203887

-- Definitions based on conditions
def position_of_3_in_8_063 := "thousandths"
def representation_of_3_in_8_063 : ℝ := 3 * 0.001
def unit_in_0_48 : ℝ := 0.01

theorem problem_statement :
  (position_of_3_in_8_063 = "thousandths") ∧
  (representation_of_3_in_8_063 = 3 * 0.001) ∧
  (unit_in_0_48 = 0.01) :=
sorry

end problem_statement_l203_203887


namespace pipe_cut_l203_203485

theorem pipe_cut (x : ℝ) (h1 : x + 2 * x = 177) : 2 * x = 118 :=
by
  sorry

end pipe_cut_l203_203485


namespace payment_to_y_l203_203219

theorem payment_to_y (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 580) : Y = 263.64 :=
by
  sorry

end payment_to_y_l203_203219


namespace ratio_of_radii_l203_203541

variables (a b : ℝ) (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2)

theorem ratio_of_radii (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : 
  a / b = Real.sqrt 5 / 5 :=
sorry

end ratio_of_radii_l203_203541


namespace greatest_even_integer_leq_z_l203_203597

theorem greatest_even_integer_leq_z (z : ℝ) (z_star : ℝ → ℝ)
  (h1 : ∀ z, z_star z = z_star (z - (z - z_star z))) -- (This is to match the definition given)
  (h2 : 6.30 - z_star 6.30 = 0.2999999999999998) : z_star 6.30 ≤ 6.30 := by
sorry

end greatest_even_integer_leq_z_l203_203597


namespace length_of_train_l203_203213

-- We state the conditions as definitions.
def length_of_train_equals_length_of_platform (l_train l_platform : ℝ) : Prop :=
l_train = l_platform

def speed_of_train (s : ℕ) : Prop :=
s = 216

def crossing_time (t : ℕ) : Prop :=
t = 1

-- Defining the goal according to the problem statement.
theorem length_of_train (l_train l_platform : ℝ) (s t : ℕ) 
  (h1 : length_of_train_equals_length_of_platform l_train l_platform) 
  (h2 : speed_of_train s) 
  (h3 : crossing_time t) : 
  l_train = 1800 :=
by
  sorry

end length_of_train_l203_203213


namespace isosceles_trapezoid_ratio_ab_cd_l203_203712

theorem isosceles_trapezoid_ratio_ab_cd (AB CD : ℝ) (P : ℝ → ℝ → Prop)
  (area1 area2 area3 area4 : ℝ)
  (h1 : AB > CD)
  (h2 : area1 = 5)
  (h3 : area2 = 7)
  (h4 : area3 = 3)
  (h5 : area4 = 9) :
  AB / CD = 1 + 2 * Real.sqrt 2 :=
sorry

end isosceles_trapezoid_ratio_ab_cd_l203_203712


namespace train_speed_is_correct_l203_203271

-- Define the conditions
def length_of_train : ℕ := 140 -- length in meters
def time_to_cross_pole : ℕ := 7 -- time in seconds

-- Define the expected speed in km/h
def expected_speed_in_kmh : ℕ := 72 -- speed in km/h

-- Prove that the speed of the train in km/h is 72
theorem train_speed_is_correct :
  (length_of_train / time_to_cross_pole) * 36 / 10 = expected_speed_in_kmh :=
by
  sorry

end train_speed_is_correct_l203_203271


namespace fraction_computation_l203_203344

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l203_203344


namespace TruckY_average_speed_is_63_l203_203392

noncomputable def average_speed_TruckY (initial_gap : ℕ) (extra_distance : ℕ) (hours : ℕ) (distance_X_per_hour : ℕ) : ℕ :=
  let distance_X := distance_X_per_hour * hours
  let total_distance_Y := distance_X + initial_gap + extra_distance
  total_distance_Y / hours

theorem TruckY_average_speed_is_63 
  (initial_gap : ℕ := 14) 
  (extra_distance : ℕ := 4) 
  (hours : ℕ := 3)
  (distance_X_per_hour : ℕ := 57) : 
  average_speed_TruckY initial_gap extra_distance hours distance_X_per_hour = 63 :=
by
  -- Proof goes here
  sorry

end TruckY_average_speed_is_63_l203_203392


namespace toys_produced_per_week_l203_203533

-- Definitions corresponding to the conditions
def days_per_week : ℕ := 2
def toys_per_day : ℕ := 2170

-- Theorem statement corresponding to the question and correct answer
theorem toys_produced_per_week : days_per_week * toys_per_day = 4340 := 
by 
  -- placeholders for the proof steps
  sorry

end toys_produced_per_week_l203_203533


namespace number_of_students_l203_203585

theorem number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = 80 * N)
  (h2 : (T - 160) / (N - 8) = 90) :
  N = 56 :=
sorry

end number_of_students_l203_203585


namespace truck_travel_distance_l203_203318

noncomputable def truck_distance (gallons: ℕ) : ℕ :=
  let efficiency_10_gallons := 300 / 10 -- miles per gallon
  let efficiency_initial := efficiency_10_gallons
  let efficiency_decreased := efficiency_initial * 9 / 10 -- 10% decrease
  if gallons <= 12 then
    gallons * efficiency_initial
  else
    12 * efficiency_initial + (gallons - 12) * efficiency_decreased

theorem truck_travel_distance (gallons: ℕ) :
  gallons = 15 → truck_distance gallons = 441 :=
by
  intros h
  rw [h]
  -- skipping proof
  sorry

end truck_travel_distance_l203_203318


namespace wrapping_paper_area_l203_203937

variable (w h : ℝ)

theorem wrapping_paper_area : ∃ A, A = 4 * (w + h) ^ 2 :=
by
  sorry

end wrapping_paper_area_l203_203937


namespace lcm_proof_l203_203979

theorem lcm_proof (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) : Nat.lcm b c = 540 :=
sorry

end lcm_proof_l203_203979


namespace chord_length_through_focus_l203_203568

theorem chord_length_through_focus (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1)
  (h_perp : (x = 1) ∨ (x = -1)) : abs (2 * y) = 3 :=
by {
  sorry
}

end chord_length_through_focus_l203_203568


namespace total_days_spent_on_islands_l203_203369

-- Define the conditions and question in Lean 4
def first_expedition_A_weeks := 3
def second_expedition_A_weeks := first_expedition_A_weeks + 2
def last_expedition_A_weeks := second_expedition_A_weeks * 2

def first_expedition_B_weeks := 5
def second_expedition_B_weeks := first_expedition_B_weeks - 3
def last_expedition_B_weeks := first_expedition_B_weeks

def total_weeks_on_island_A := first_expedition_A_weeks + second_expedition_A_weeks + last_expedition_A_weeks
def total_weeks_on_island_B := first_expedition_B_weeks + second_expedition_B_weeks + last_expedition_B_weeks

def total_weeks := total_weeks_on_island_A + total_weeks_on_island_B
def total_days := total_weeks * 7

theorem total_days_spent_on_islands : total_days = 210 :=
by
  -- We skip the proof part
  sorry

end total_days_spent_on_islands_l203_203369


namespace investment_amount_l203_203264

theorem investment_amount (P: ℝ) (q_investment: ℝ) (ratio_pq: ℝ) (ratio_qp: ℝ) 
  (h1: ratio_pq = 4) (h2: ratio_qp = 6) (q_investment: ℝ) (h3: q_investment = 90000): 
  P = 60000 :=
by 
  -- Sorry is used here to skip the actual proof
  sorry

end investment_amount_l203_203264


namespace find_product_l203_203839

def a : ℕ := 4
def g : ℕ := 8
def d : ℕ := 10

theorem find_product (A B C D E F : ℕ) (hA : A % 2 = 0) (hB : B % 3 = 0) (hC : C % 4 = 0) 
  (hD : D % 5 = 0) (hE : E % 6 = 0) (hF : F % 7 = 0) :
  a * g * d = 320 :=
by
  sorry

end find_product_l203_203839


namespace no_solution_fraction_eq_l203_203513

theorem no_solution_fraction_eq (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (a * x / (x - 1) + 3 / (1 - x) = 2) → false) ↔ a = 2 :=
by
  sorry

end no_solution_fraction_eq_l203_203513


namespace ray_nickels_left_l203_203619

theorem ray_nickels_left (h1 : 285 % 5 = 0) (h2 : 55 % 5 = 0) (h3 : 3 * 55 % 5 = 0) (h4 : 45 % 5 = 0) : 
  285 / 5 - ((55 / 5) + (3 * 55 / 5) + (45 / 5)) = 4 := sorry

end ray_nickels_left_l203_203619


namespace polygon_sides_l203_203137

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
by
  sorry

end polygon_sides_l203_203137


namespace adding_2_to_odd_integer_can_be_prime_l203_203158

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem adding_2_to_odd_integer_can_be_prime :
  ∃ n : ℤ, is_odd n ∧ is_prime (n + 2) :=
by
  sorry

end adding_2_to_odd_integer_can_be_prime_l203_203158


namespace roundness_of_hundred_billion_l203_203071

def roundness (n : ℕ) : ℕ :=
  let pf := n.factorization
  pf 2 + pf 5

theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by
  sorry

end roundness_of_hundred_billion_l203_203071


namespace coefficient_x2_is_negative_40_l203_203405

noncomputable def x2_coefficient_in_expansion (a : ℕ) : ℤ :=
  (-1)^3 * a^2 * Nat.choose 5 2

theorem coefficient_x2_is_negative_40 :
  x2_coefficient_in_expansion 2 = -40 :=
by
  sorry

end coefficient_x2_is_negative_40_l203_203405


namespace smallest_digit_to_correct_sum_l203_203920

theorem smallest_digit_to_correct_sum (x y z w : ℕ) (hx : x = 753) (hy : y = 946) (hz : z = 821) (hw : w = 2420) :
  ∃ d, d = 7 ∧ (753 + 946 + 821 - 100 * d = 2420) :=
by sorry

end smallest_digit_to_correct_sum_l203_203920


namespace cost_per_play_l203_203472

-- Conditions
def initial_money : ℝ := 3
def points_per_red_bucket : ℝ := 2
def points_per_green_bucket : ℝ := 3
def rings_per_play : ℕ := 5
def games_played : ℕ := 2
def red_buckets : ℕ := 4
def green_buckets : ℕ := 5
def total_games : ℕ := 3
def total_points : ℝ := 38

-- Point calculations
def points_from_red_buckets : ℝ := red_buckets * points_per_red_bucket
def points_from_green_buckets : ℝ := green_buckets * points_per_green_bucket
def current_points : ℝ := points_from_red_buckets + points_from_green_buckets
def points_needed : ℝ := total_points - current_points

-- Define the theorem statement
theorem cost_per_play :
  (initial_money / (games_played : ℝ)) = 1.50 :=
  sorry

end cost_per_play_l203_203472


namespace valid_three_digit_card_numbers_count_l203_203874

def card_numbers : List (ℕ × ℕ) := [(0, 1), (2, 3), (4, 5), (7, 8)]

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 -- Ensures it's three digits

def three_digit_numbers : List ℕ := 
  [201, 210, 102, 120, 301, 310, 103, 130, 401, 410, 104, 140,
   501, 510, 105, 150, 601, 610, 106, 160, 701, 710, 107, 170,
   801, 810, 108, 180, 213, 231, 312, 321, 413, 431, 512, 521,
   613, 631, 714, 741, 813, 831, 214, 241, 315, 351, 415, 451,
   514, 541, 615, 651, 716, 761, 815, 851, 217, 271, 317, 371,
   417, 471, 517, 571, 617, 671, 717, 771, 817, 871, 217, 271,
   321, 371, 421, 471, 521, 571, 621, 671, 721, 771, 821, 871]

def count_valid_three_digit_numbers : ℕ :=
  three_digit_numbers.length

theorem valid_three_digit_card_numbers_count :
    count_valid_three_digit_numbers = 168 :=
by
  -- proof goes here
  sorry

end valid_three_digit_card_numbers_count_l203_203874


namespace AH_HD_ratio_l203_203133

-- Given conditions
variables {A B C H D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited D]
variables (BC : ℝ) (AC : ℝ) (angle_C : ℝ)
-- We assume the values provided in the problem
variables (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4)

-- Altitudes and orthocenter assumption, representing intersections at orthocenter H
variables (A D H : Type) -- Points to represent A, D, and orthocenter H

noncomputable def AH_H_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) : ℝ :=
  if BC = 6 ∧ AC = 4 * Real.sqrt 2 ∧ angle_C = Real.pi / 4 then 2 else 0

-- We need to prove the ratio AH:HD equals 2 given the conditions
theorem AH_HD_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) :
  AH_H_ratio BC AC angle_C BC_eq AC_eq angle_C_eq = 2 :=
by {
  -- the statement will be proved here
  sorry
}

end AH_HD_ratio_l203_203133


namespace systematic_sampling_id_fourth_student_l203_203315

theorem systematic_sampling_id_fourth_student (n : ℕ) (a b c d : ℕ) (h1 : n = 54) 
(h2 : a = 3) (h3 : b = 29) (h4 : c = 42) (h5 : d = a + 13) : d = 16 :=
by
  sorry

end systematic_sampling_id_fourth_student_l203_203315


namespace no_consecutive_nat_mul_eq_25k_plus_1_l203_203334

theorem no_consecutive_nat_mul_eq_25k_plus_1 (k : ℕ) : 
  ¬ ∃ n : ℕ, n * (n + 1) = 25 * k + 1 :=
sorry

end no_consecutive_nat_mul_eq_25k_plus_1_l203_203334


namespace find_m_l203_203618

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) (h : dot_prod a (b m) / magnitude a = 3) : m = sqrt 3 :=
by
  sorry

end find_m_l203_203618


namespace correct_operation_l203_203766

theorem correct_operation : 
  (3 - Real.sqrt 2) ^ 2 = 11 - 6 * Real.sqrt 2 :=
sorry

end correct_operation_l203_203766


namespace find_x_l203_203957

theorem find_x
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h : a = (Real.sqrt 3, 0))
  (h1 : b = (x, -2))
  (h2 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) :
  x = Real.sqrt 3 / 2 :=
sorry

end find_x_l203_203957


namespace total_games_played_l203_203238

theorem total_games_played (months : ℕ) (games_per_month : ℕ) (h1 : months = 17) (h2 : games_per_month = 19) : 
  months * games_per_month = 323 :=
by
  sorry

end total_games_played_l203_203238


namespace coordinates_equidistant_l203_203558

-- Define the condition of equidistance
theorem coordinates_equidistant (x y : ℝ) :
  (x + 2) ^ 2 + (y - 2) ^ 2 = (x - 2) ^ 2 + y ^ 2 →
  y = 2 * x + 1 :=
  sorry  -- Proof is omitted

end coordinates_equidistant_l203_203558


namespace distance_is_660_km_l203_203159

def distance_between_cities (x y : ℝ) : ℝ :=
  3.3 * (x + y)

def train_A_dep_earlier (x y : ℝ) : Prop :=
  3.4 * (x + y) = 3.3 * (x + y) + 14

def train_B_dep_earlier (x y : ℝ) : Prop :=
  3.6 * (x + y) = 3.3 * (x + y) + 9

theorem distance_is_660_km (x y : ℝ) (hx : train_A_dep_earlier x y) (hy : train_B_dep_earlier x y) :
    distance_between_cities x y = 660 :=
sorry

end distance_is_660_km_l203_203159


namespace x_intercept_of_translated_line_l203_203965

theorem x_intercept_of_translated_line :
  let line_translation (y : ℝ) := y + 4
  let new_line_eq := fun (x : ℝ) => 2 * x - 2
  new_line_eq 1 = 0 :=
by
  sorry

end x_intercept_of_translated_line_l203_203965


namespace find_triangle_sides_l203_203545

theorem find_triangle_sides (k : ℕ) (k_pos : k = 6) 
  {x y z : ℝ} (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) 
  (h : k * (x * y + y * z + z * x) > 5 * (x ^ 2 + y ^ 2 + z ^ 2)) :
  ∃ x' y' z', (x = x') ∧ (y = y') ∧ (z = z') ∧ ((x' + y' > z') ∧ (x' + z' > y') ∧ (y' + z' > x')) :=
by
  sorry

end find_triangle_sides_l203_203545


namespace solve_equation_l203_203319

theorem solve_equation : ∀ x : ℝ, (2 / 3 * x - 2 = 4) → x = 9 :=
by
  intro x
  intro h
  sorry

end solve_equation_l203_203319


namespace evaluate_expression_l203_203100

theorem evaluate_expression : 3 * (3 * (3 * (3 + 2) + 2) + 2) + 2 = 161 := sorry

end evaluate_expression_l203_203100


namespace quadratic_to_vertex_form_l203_203610

theorem quadratic_to_vertex_form :
  ∀ (x a h k : ℝ), (x^2 - 7*x = a*(x - h)^2 + k) → k = -49 / 4 :=
by
  intros x a h k
  sorry

end quadratic_to_vertex_form_l203_203610


namespace hidden_prime_average_correct_l203_203592

noncomputable def hidden_prime_average : ℚ :=
  (13 + 17 + 59) / 3

theorem hidden_prime_average_correct :
  hidden_prime_average = 29.6 :=
by
  sorry

end hidden_prime_average_correct_l203_203592


namespace cake_and_milk_tea_cost_l203_203490

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l203_203490


namespace certain_number_is_10000_l203_203686

theorem certain_number_is_10000 (n : ℕ) (h1 : n - 999 = 9001) : n = 10000 :=
by
  sorry

end certain_number_is_10000_l203_203686


namespace additional_cost_per_pint_proof_l203_203539

-- Definitions based on the problem conditions
def pints_sold := 54
def total_revenue_on_sale := 216
def revenue_difference := 108

-- Derived definitions
def revenue_if_not_on_sale := total_revenue_on_sale + revenue_difference
def cost_per_pint_on_sale := total_revenue_on_sale / pints_sold
def cost_per_pint_not_on_sale := revenue_if_not_on_sale / pints_sold
def additional_cost_per_pint := cost_per_pint_not_on_sale - cost_per_pint_on_sale

-- Proof statement
theorem additional_cost_per_pint_proof :
  additional_cost_per_pint = 2 :=
by
  -- Placeholder to indicate that the proof is not provided
  sorry

end additional_cost_per_pint_proof_l203_203539


namespace number_of_individuals_left_at_zoo_l203_203740

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end number_of_individuals_left_at_zoo_l203_203740


namespace storybooks_sciencebooks_correct_l203_203872

-- Given conditions
def total_books : ℕ := 144
def ratio_storybooks_sciencebooks := (7, 5)
def fraction_storybooks := 7 / (7 + 5)
def fraction_sciencebooks := 5 / (7 + 5)

-- Prove the number of storybooks and science books
def number_of_storybooks : ℕ := 84
def number_of_sciencebooks : ℕ := 60

theorem storybooks_sciencebooks_correct :
  (fraction_storybooks * total_books = number_of_storybooks) ∧
  (fraction_sciencebooks * total_books = number_of_sciencebooks) :=
by
  sorry

end storybooks_sciencebooks_correct_l203_203872


namespace cookie_cost_proof_l203_203409

def cost_per_cookie (total_spent : ℕ) (days : ℕ) (cookies_per_day : ℕ) : ℕ :=
  total_spent / (days * cookies_per_day)

theorem cookie_cost_proof : cost_per_cookie 1395 31 3 = 15 := by
  sorry

end cookie_cost_proof_l203_203409


namespace domain_of_composite_function_l203_203199

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → -1 ≤ x + 1) →
  (∀ x, -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3 → 0 ≤ x ∧ x ≤ 1) :=
by
  sorry

end domain_of_composite_function_l203_203199


namespace bicycle_count_l203_203895

theorem bicycle_count (B T : ℕ) (hT : T = 20) (h_wheels : 2 * B + 3 * T = 160) : B = 50 :=
by
  sorry

end bicycle_count_l203_203895


namespace linear_equation_conditions_l203_203750

theorem linear_equation_conditions (m n : ℤ) :
  (∀ x y : ℝ, 4 * x^(m - n) - 5 * y^(m + n) = 6 → 
    m - n = 1 ∧ m + n = 1) →
  m = 1 ∧ n = 0 :=
by
  sorry

end linear_equation_conditions_l203_203750


namespace rate_is_15_l203_203516

variable (sum : ℝ) (interest12 : ℝ) (interest_r : ℝ) (r : ℝ)

-- Given conditions
def conditions : Prop :=
  sum = 7000 ∧
  interest12 = 7000 * 0.12 * 2 ∧
  interest_r = 7000 * (r / 100) * 2 ∧
  interest_r = interest12 + 420

-- The rate to prove
def rate_to_prove : Prop := r = 15

theorem rate_is_15 : conditions sum interest12 interest_r r → rate_to_prove r := 
by
  sorry

end rate_is_15_l203_203516


namespace common_difference_arithmetic_sequence_l203_203336

variable (n d : ℝ) (a : ℝ := 7 - 2 * d) (an : ℝ := 37) (Sn : ℝ := 198)

theorem common_difference_arithmetic_sequence :
  7 + (n - 3) * d = 37 ∧ 
  396 = n * (44 - 2 * d) ∧
  Sn = n / 2 * (a + an) →
  (∃ d : ℝ, 7 + (n - 3) * d = 37 ∧ 396 = n * (44 - 2 * d)) :=
by
  sorry

end common_difference_arithmetic_sequence_l203_203336


namespace average_speed_over_ride_l203_203904

theorem average_speed_over_ride :
  let speed1 := 12 -- speed in km/h
  let time1 := 5 / 60 -- time in hours
  
  let speed2 := 15 -- speed in km/h
  let time2 := 10 / 60 -- time in hours
  
  let speed3 := 18 -- speed in km/h
  let time3 := 15 / 60 -- time in hours
  
  let distance1 := speed1 * time1 -- distance for the first segment
  let distance2 := speed2 * time2 -- distance for the second segment
  let distance3 := speed3 * time3 -- distance for the third segment
  
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  let avg_speed := total_distance / total_time
  
  avg_speed = 16 :=
by
  sorry

end average_speed_over_ride_l203_203904


namespace inequality_transformation_l203_203070

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_transformation_l203_203070


namespace initial_women_count_l203_203588

-- Let x be the initial number of women.
-- Let y be the initial number of men.

theorem initial_women_count (x y : ℕ) (h1 : y = 2 * (x - 15)) (h2 : (y - 45) * 5 = (x - 15)) :
  x = 40 :=
by
  -- sorry to skip the proof
  sorry

end initial_women_count_l203_203588


namespace solution_is_correct_l203_203613

-- Initial conditions
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.40
def target_concentration : ℝ := 0.50

-- Given that we start with 2.4 liters of pure alcohol in a 6-liter solution
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Expected result after adding x liters of pure alcohol
def final_solution_volume (x : ℝ) : ℝ := initial_volume + x
def final_pure_alcohol (x : ℝ) : ℝ := initial_pure_alcohol + x

-- Equation to prove
theorem solution_is_correct (x : ℝ) :
  (final_pure_alcohol x) / (final_solution_volume x) = target_concentration ↔ 
  x = 1.2 := 
sorry

end solution_is_correct_l203_203613


namespace geometric_series_sum_l203_203352

theorem geometric_series_sum :
  (1 / 3 - 1 / 6 + 1 / 12 - 1 / 24 + 1 / 48 - 1 / 96) = 7 / 32 :=
by
  sorry

end geometric_series_sum_l203_203352


namespace max_marks_l203_203122

theorem max_marks (M : ℝ) (h : 0.92 * M = 460) : M = 500 :=
by
  sorry

end max_marks_l203_203122


namespace perfect_square_K_l203_203938

-- Definitions based on the conditions of the problem
variables (Z K : ℕ)
variables (h1 : 1000 < Z ∧ Z < 5000)
variables (h2 : K > 1)
variables (h3 : Z = K^3)

-- The statement we need to prove
theorem perfect_square_K :
  (∃ K : ℕ, 1000 < K^3 ∧ K^3 < 5000 ∧ K^3 = Z ∧ (∃ a : ℕ, K = a^2)) → K = 16 :=
sorry

end perfect_square_K_l203_203938


namespace total_lunch_bill_l203_203379

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l203_203379


namespace eval_fraction_power_l203_203397

theorem eval_fraction_power : (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end eval_fraction_power_l203_203397


namespace problem_statement_l203_203715

def setS : Set (ℝ × ℝ) := {p | p.1 * p.2 > 0}
def setT : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0}

theorem problem_statement : setS ∪ setT = setS ∧ setS ∩ setT = setT :=
by
  -- To be proved
  sorry

end problem_statement_l203_203715


namespace sculpt_cost_in_mxn_l203_203229

variable (usd_to_nad usd_to_mxn cost_nad cost_mxn : ℝ)

theorem sculpt_cost_in_mxn (h1 : usd_to_nad = 8) (h2 : usd_to_mxn = 20) (h3 : cost_nad = 160) : cost_mxn = 400 :=
by
  sorry

end sculpt_cost_in_mxn_l203_203229


namespace range_of_f_x_minus_2_l203_203605

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + 1 else if x > 0 then -(x + 1) else 0

theorem range_of_f_x_minus_2 :
  ∀ x : ℝ, f (x - 2) < 0 ↔ x ∈ Set.union (Set.Iio 1) (Set.Ioo 2 3) := by
sorry

end range_of_f_x_minus_2_l203_203605


namespace min_blocks_for_wall_l203_203978

theorem min_blocks_for_wall (len height : ℕ) (blocks : ℕ → ℕ → ℕ)
  (block_1 : ℕ) (block_2 : ℕ) (block_3 : ℕ) :
  len = 120 → height = 9 →
  block_3 = 1 → block_2 = 2 → block_1 = 3 →
  blocks 5 41 + blocks 4 40 = 365 :=
by
  sorry

end min_blocks_for_wall_l203_203978


namespace cloth_meters_sold_l203_203665

-- Conditions as definitions
def total_selling_price : ℝ := 4500
def profit_per_meter : ℝ := 14
def cost_price_per_meter : ℝ := 86

-- The statement of the problem
theorem cloth_meters_sold (SP : ℝ := cost_price_per_meter + profit_per_meter) :
  total_selling_price / SP = 45 := by
  sorry

end cloth_meters_sold_l203_203665


namespace div_pow_two_sub_one_l203_203332

theorem div_pow_two_sub_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  (3^k ∣ 2^n - 1) ↔ (∃ m : ℕ, n = 2 * 3^(k-1) * m) :=
by
  sorry

end div_pow_two_sub_one_l203_203332


namespace find_first_type_cookies_l203_203830

section CookiesProof

variable (x : ℕ)

-- Conditions
def box_first_type_cookies : ℕ := x
def box_second_type_cookies : ℕ := 20
def box_third_type_cookies : ℕ := 16
def boxes_first_type_sold : ℕ := 50
def boxes_second_type_sold : ℕ := 80
def boxes_third_type_sold : ℕ := 70
def total_cookies_sold : ℕ := 3320

-- Theorem to prove
theorem find_first_type_cookies 
  (h1 : 50 * x + 80 * box_second_type_cookies + 70 * box_third_type_cookies = total_cookies_sold) :
  x = 12 := by
    sorry

end CookiesProof

end find_first_type_cookies_l203_203830


namespace hens_count_l203_203132

theorem hens_count (H C : ℕ) (h_heads : H + C = 60) (h_feet : 2 * H + 4 * C = 200) : H = 20 :=
by
  sorry

end hens_count_l203_203132


namespace Francine_not_working_days_l203_203224

-- Conditions
variables (d : ℕ) -- Number of days Francine works each week
def distance_per_day : ℕ := 140 -- Distance Francine drives each day
def total_distance_4_weeks : ℕ := 2240 -- Total distance in 4 weeks
def days_per_week : ℕ := 7 -- Days in a week

-- Proving that the number of days she does not go to work every week is 3
theorem Francine_not_working_days :
  (4 * distance_per_day * d = total_distance_4_weeks) →
  ((days_per_week - d) = 3) :=
by sorry

end Francine_not_working_days_l203_203224


namespace winning_candidate_percentage_l203_203348

theorem winning_candidate_percentage
  (majority_difference : ℕ)
  (total_valid_votes : ℕ)
  (P : ℕ)
  (h1 : majority_difference = 192)
  (h2 : total_valid_votes = 480)
  (h3 : 960 * P = 67200) : 
  P = 70 := by
  sorry

end winning_candidate_percentage_l203_203348


namespace rotameter_percentage_l203_203573

theorem rotameter_percentage (l_inch_flow : ℝ) (l_liters_flow : ℝ) (g_inch_flow : ℝ) (g_liters_flow : ℝ) :
  l_inch_flow = 2.5 → l_liters_flow = 60 → g_inch_flow = 4 → g_liters_flow = 192 → 
  (g_liters_flow / g_inch_flow) / (l_liters_flow / l_inch_flow) * 100 = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_percentage_l203_203573


namespace solve_for_x_l203_203614

theorem solve_for_x (x : ℚ) (h : (1 / 2 - 1 / 3) = 3 / x) : x = 18 :=
sorry

end solve_for_x_l203_203614


namespace coupon_value_l203_203471

theorem coupon_value
  (bill : ℝ)
  (milk_cost : ℝ)
  (bread_cost : ℝ)
  (detergent_cost : ℝ)
  (banana_cost_per_pound : ℝ)
  (banana_weight : ℝ)
  (half_off : ℝ)
  (amount_left : ℝ)
  (total_without_coupon : ℝ)
  (total_spent : ℝ)
  (coupon_value : ℝ) :
  bill = 20 →
  milk_cost = 4 →
  bread_cost = 3.5 →
  detergent_cost = 10.25 →
  banana_cost_per_pound = 0.75 →
  banana_weight = 2 →
  half_off = 0.5 →
  amount_left = 4 →
  total_without_coupon = milk_cost * half_off + bread_cost + detergent_cost + banana_cost_per_pound * banana_weight →
  total_spent = bill - amount_left →
  coupon_value = total_without_coupon - total_spent →
  coupon_value = 1.25 :=
by
  sorry

end coupon_value_l203_203471


namespace line_equation_l203_203890

theorem line_equation (m : ℝ) (x1 y1 : ℝ) (b : ℝ) :
  m = -3 → x1 = -2 → y1 = 0 → 
  (∀ x y, y - y1 = m * (x - x1) ↔ 3 * x + y + 6 = 0) :=
sorry

end line_equation_l203_203890


namespace mika_jogging_speed_l203_203346

theorem mika_jogging_speed 
  (s : ℝ)  -- Mika's constant jogging speed in meters per second.
  (r : ℝ)  -- Radius of the inner semicircle.
  (L : ℝ)  -- Length of each straight section.
  (h1 : 8 > 0) -- Overall width of the track is 8 meters.
  (h2 : (2 * L + 2 * π * (r + 8)) / s = (2 * L + 2 * π * r) / s + 48) -- Time difference equation.
  : s = π / 3 := 
sorry

end mika_jogging_speed_l203_203346


namespace Mark_jump_rope_hours_l203_203595

theorem Mark_jump_rope_hours 
  (record : ℕ) 
  (jump_rate : ℕ) 
  (seconds_per_hour : ℕ) 
  (h_record : record = 54000) 
  (h_jump_rate : jump_rate = 3) 
  (h_seconds_per_hour : seconds_per_hour = 3600) 
  : (record / jump_rate) / seconds_per_hour = 5 := 
by
  sorry

end Mark_jump_rope_hours_l203_203595


namespace percent_paddyfield_warblers_l203_203999

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end percent_paddyfield_warblers_l203_203999


namespace additional_students_needed_l203_203058

theorem additional_students_needed 
  (n : ℕ) 
  (r : ℕ) 
  (t : ℕ) 
  (h_n : n = 82) 
  (h_r : r = 2) 
  (h_t : t = 49) : 
  (t - n / r) * r = 16 := 
by 
  sorry

end additional_students_needed_l203_203058


namespace how_many_pens_l203_203963

theorem how_many_pens
  (total_cost : ℝ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (avg_pen_price : ℝ)
  (total_cost := 510)
  (num_pencils := 75)
  (avg_pencil_price := 2)
  (avg_pen_price := 12)
  : ∃ (num_pens : ℕ), num_pens = 30 :=
by
  sorry

end how_many_pens_l203_203963


namespace candy_box_price_l203_203217

theorem candy_box_price (c s : ℝ) 
  (h1 : 1.50 * s = 6) 
  (h2 : c + s = 16) 
  (h3 : ∀ c, 1.25 * c = 1.25 * 12) : 
  (1.25 * c = 15) :=
by
  sorry

end candy_box_price_l203_203217


namespace cos_ninety_degrees_l203_203466

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l203_203466


namespace percent_difference_l203_203326

theorem percent_difference : 0.12 * 24.2 - 0.10 * 14.2 = 1.484 := by
  sorry

end percent_difference_l203_203326


namespace total_luggage_l203_203628

theorem total_luggage (ne nb nf : ℕ)
  (leconomy lbusiness lfirst : ℕ)
  (Heconomy : ne = 10) 
  (Hbusiness : nb = 7) 
  (Hfirst : nf = 3)
  (Heconomy_luggage : leconomy = 5)
  (Hbusiness_luggage : lbusiness = 8)
  (Hfirst_luggage : lfirst = 12) : 
  (ne * leconomy + nb * lbusiness + nf * lfirst) = 142 :=
by
  sorry

end total_luggage_l203_203628


namespace tiling_tetromino_divisibility_l203_203203

theorem tiling_tetromino_divisibility (n : ℕ) : 
  (∃ (t : ℕ), n = 4 * t) ↔ (∃ (k : ℕ), n * n = 4 * k) :=
by
  sorry

end tiling_tetromino_divisibility_l203_203203


namespace one_gt_one_others_lt_one_l203_203703

theorem one_gt_one_others_lt_one 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_prod : a * b * c = 1)
  (h_ineq : a + b + c > (1 / a) + (1 / b) + (1 / c)) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
sorry

end one_gt_one_others_lt_one_l203_203703


namespace domain_ln_l203_203719

def domain_of_ln (x : ℝ) : Prop := x^2 - x > 0

theorem domain_ln (x : ℝ) :
  domain_of_ln x ↔ (x < 0 ∨ x > 1) :=
by sorry

end domain_ln_l203_203719


namespace nick_coin_collection_l203_203651

theorem nick_coin_collection
  (total_coins : ℕ)
  (quarters_coins : ℕ)
  (dimes_coins : ℕ)
  (nickels_coins : ℕ)
  (state_quarters : ℕ)
  (pa_state_quarters : ℕ)
  (roosevelt_dimes : ℕ)
  (h_total : total_coins = 50)
  (h_quarters : quarters_coins = total_coins * 3 / 10)
  (h_dimes : dimes_coins = total_coins * 40 / 100)
  (h_nickels : nickels_coins = total_coins - (quarters_coins + dimes_coins))
  (h_state_quarters : state_quarters = quarters_coins * 2 / 5)
  (h_pa_state_quarters : pa_state_quarters = state_quarters * 3 / 8)
  (h_roosevelt_dimes : roosevelt_dimes = dimes_coins * 75 / 100) :
  pa_state_quarters = 2 ∧ roosevelt_dimes = 15 ∧ nickels_coins = 15 :=
by
  sorry

end nick_coin_collection_l203_203651


namespace parallel_vectors_l203_203107

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_b : b = (-1, m)) (h_parallel : ∃ k : ℝ, b = k • a) : m = 2 :=
by {
  sorry
}

end parallel_vectors_l203_203107


namespace calculate_expression_l203_203832

theorem calculate_expression : 
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 :=
by
  sorry

end calculate_expression_l203_203832


namespace part1_part2_l203_203889

-- Definitions as per the conditions
def A (a b : ℚ) := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) := - a^2 + (1/2) * a * b + 2 / 3

-- Part (1)
theorem part1 (a b : ℚ) (h1 : a = -1) (h2 : b = -2) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3 := 
by 
  sorry

-- Part (2)
theorem part2 (a : ℚ) : 
  (∀ a : ℚ, 4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3) → 
  b = 1/2 :=
by 
  sorry

end part1_part2_l203_203889


namespace maximal_number_of_coins_l203_203200

noncomputable def largest_number_of_coins (n k : ℕ) : Prop :=
n < 100 ∧ n = 12 * k + 3

theorem maximal_number_of_coins (n k : ℕ) : largest_number_of_coins n k → n = 99 :=
by
  sorry

end maximal_number_of_coins_l203_203200


namespace probability_of_black_ball_l203_203753

theorem probability_of_black_ball (P_red P_white : ℝ) (h_red : P_red = 0.43) (h_white : P_white = 0.27) : 
  (1 - P_red - P_white) = 0.3 := 
by
  sorry

end probability_of_black_ball_l203_203753


namespace correct_operation_l203_203985

-- Defining the options as hypotheses
variable {a b : ℕ}

theorem correct_operation (hA : 4*a + 3*b ≠ 7*a*b)
    (hB : a^4 * a^3 = a^7)
    (hC : (3*a)^3 ≠ 9*a^3)
    (hD : a^6 / a^2 ≠ a^3) :
    a^4 * a^3 = a^7 := by
  sorry

end correct_operation_l203_203985


namespace eraser_cost_l203_203964

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end eraser_cost_l203_203964


namespace difference_of_two_numbers_l203_203420

theorem difference_of_two_numbers
  (L : ℕ) (S : ℕ) 
  (hL : L = 1596) 
  (hS : 6 * S + 15 = 1596) : 
  L - S = 1333 := 
by
  sorry

end difference_of_two_numbers_l203_203420


namespace value_equation_l203_203644

noncomputable def quarter_value := 25
noncomputable def dime_value := 10
noncomputable def half_dollar_value := 50

theorem value_equation (n : ℕ) :
  25 * quarter_value + 20 * dime_value = 15 * quarter_value + 10 * dime_value + n * half_dollar_value → 
  n = 7 :=
by
  sorry

end value_equation_l203_203644


namespace positive_reals_inequality_l203_203099

variable {a b c : ℝ}

theorem positive_reals_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a * b)^(1/4) + (b * c)^(1/4) + (c * a)^(1/4) < 1/4 := 
sorry

end positive_reals_inequality_l203_203099


namespace boat_current_ratio_l203_203506

noncomputable def boat_speed_ratio (b c : ℝ) (d : ℝ) : Prop :=
  let time_upstream := 6
  let time_downstream := 10
  d = time_upstream * (b - c) ∧ 
  d = time_downstream * (b + c) → 
  b / c = 4

theorem boat_current_ratio (b c d : ℝ) (h1 : d = 6 * (b - c)) (h2 : d = 10 * (b + c)) : b / c = 4 :=
by sorry

end boat_current_ratio_l203_203506


namespace emily_necklaces_for_friends_l203_203116

theorem emily_necklaces_for_friends (n b B : ℕ)
  (h1 : n = 26)
  (h2 : b = 2)
  (h3 : B = 52)
  (h4 : n * b = B) : 
  n = 26 :=
by
  sorry

end emily_necklaces_for_friends_l203_203116


namespace smallest_x_for_gx_eq_g1458_l203_203446

noncomputable def g : ℝ → ℝ := sorry -- You can define the function later.

theorem smallest_x_for_gx_eq_g1458 :
  (∀ x : ℝ, x > 0 → g (3 * x) = 4 * g x) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → g x = 2 - 2 * |x - 2|)
  → ∃ x : ℝ, x ≥ 0 ∧ g x = g 1458 ∧ ∀ y : ℝ, y ≥ 0 ∧ g y = g 1458 → x ≤ y ∧ x = 162 := 
by
  sorry

end smallest_x_for_gx_eq_g1458_l203_203446


namespace triangle_median_equiv_l203_203797

-- Assuming necessary non-computable definitions (e.g., α for angles, R for real numbers) and non-computable nature of some geometric properties.

noncomputable def triangle (A B C : ℝ) := 
A + B + C = Real.pi

noncomputable def length_a (R A : ℝ) : ℝ := 2 * R * Real.sin A
noncomputable def length_b (R B : ℝ) : ℝ := 2 * R * Real.sin B
noncomputable def length_c (R C : ℝ) : ℝ := 2 * R * Real.sin C

noncomputable def median_a (b c A : ℝ) : ℝ := (2 * b * c) / (b + c) * Real.cos (A / 2)

theorem triangle_median_equiv (A B C R : ℝ) (hA : triangle A B C) :
  (1 / (length_a R A) + 1 / (length_b R B) = 1 / (median_a (length_b R B) (length_c R C) A)) ↔ (C = 2 * Real.pi / 3) := 
by sorry

end triangle_median_equiv_l203_203797


namespace medium_supermarkets_in_sample_l203_203512

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l203_203512


namespace polar_line_through_centers_l203_203115

-- Definition of the given circles in polar coordinates
def Circle1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def Circle2 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Statement of the problem
theorem polar_line_through_centers (ρ θ : ℝ) :
  (∃ c1 c2 : ℝ × ℝ, Circle1 c1.fst c1.snd ∧ Circle2 c2.fst c2.snd ∧ θ = Real.pi / 4) :=
sorry

end polar_line_through_centers_l203_203115


namespace graph_representation_l203_203720

theorem graph_representation {x y : ℝ} (h : x^2 * (x - y - 2) = y^2 * (x - y - 2)) :
  ( ∃ a : ℝ, ∀ (x : ℝ), y = a * x ) ∨ 
  ( ∃ b : ℝ, ∀ (x : ℝ), y = b * x ) ∨ 
  ( ∃ c : ℝ, ∀ (x : ℝ), y = x - 2 ) ∧ 
  (¬ ∃ d : ℝ, ∀ (x : ℝ), y = d * x ∧ y = d * x - 2) :=
sorry

end graph_representation_l203_203720


namespace remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l203_203216

theorem remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero (x : ℝ) :
  (x + 1) ^ 2025 % (x ^ 2 + 1) = 0 :=
  sorry

end remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l203_203216


namespace boat_travel_time_l203_203631

theorem boat_travel_time (x : ℝ) (T : ℝ) (h0 : 0 ≤ x) (h1 : x ≠ 15.6) 
    (h2 : 96 = (15.6 - x) * T) 
    (h3 : 96 = (15.6 + x) * 5) : 
    T = 8 :=
by 
  sorry

end boat_travel_time_l203_203631


namespace inequality_holds_for_positive_reals_equality_condition_l203_203486

theorem inequality_holds_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_holds_for_positive_reals_equality_condition_l203_203486


namespace field_dimension_solution_l203_203026

theorem field_dimension_solution (m : ℤ) (H1 : (3 * m + 11) * m = 100) : m = 5 :=
sorry

end field_dimension_solution_l203_203026


namespace arithmetic_expression_eval_l203_203205

theorem arithmetic_expression_eval : 3 + (12 / 3 - 1) ^ 2 = 12 := by
  sorry

end arithmetic_expression_eval_l203_203205


namespace weight_of_second_square_l203_203184

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  side_length^2 * density

theorem weight_of_second_square :
  let s1 := 4
  let m1 := 20
  let s2 := 7
  let density := m1 / (s1 ^ 2)
  ∃ (m2 : ℝ), m2 = 61.25 :=
by
  have s1 := 4
  have m1 := 20
  have s2 := 7
  let density := m1 / (s1 ^ 2)
  have m2 := weight_of_square s2 density
  use m2
  sorry

end weight_of_second_square_l203_203184


namespace sum_of_possible_values_l203_203220

theorem sum_of_possible_values 
  (x y : ℝ) 
  (h : x * y - x / y^2 - y / x^2 = 3) :
  (x = 0 ∨ y = 0 → False) → 
  ((x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4) → 
  ((x - 1) * (y - 1) = 1 → (x - 1) * (y - 1) = 1) → 
  ((x - 1) * (y - 1) = 4 → (x - 1) * (y - 1) = 4) → 
  (1 + 4 = 5) := 
by 
  sorry

end sum_of_possible_values_l203_203220


namespace distance_between_first_and_last_pots_l203_203498

theorem distance_between_first_and_last_pots (n : ℕ) (d : ℕ) 
  (h₁ : n = 8) 
  (h₂ : d = 100) : 
  ∃ total_distance : ℕ, total_distance = 175 := 
by 
  sorry

end distance_between_first_and_last_pots_l203_203498


namespace sum_of_nonnegative_numbers_eq_10_l203_203678

theorem sum_of_nonnegative_numbers_eq_10 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 48)
  (h2 : ab + bc + ca = 26)
  (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) : a + b + c = 10 := 
by
  sorry

end sum_of_nonnegative_numbers_eq_10_l203_203678


namespace diagonal_length_l203_203765

noncomputable def length_of_diagonal (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_length
  (a b c : ℝ)
  (h1 : 2 * (a * b + a * c + b * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  length_of_diagonal a b c = 5 := by
  sorry

end diagonal_length_l203_203765


namespace initial_pencils_count_l203_203714

theorem initial_pencils_count (pencils_taken : ℕ) (pencils_left : ℕ) (h1 : pencils_taken = 4) (h2 : pencils_left = 75) : 
  pencils_left + pencils_taken = 79 :=
by
  sorry

end initial_pencils_count_l203_203714


namespace max_students_l203_203864

open Nat

theorem max_students (B G : ℕ) (h1 : 11 * B = 7 * G) (h2 : G = B + 72) (h3 : B + G ≤ 550) : B + G = 324 := by
  sorry

end max_students_l203_203864


namespace sum_a_t_l203_203998

theorem sum_a_t (a : ℝ) (t : ℝ) 
  (h₁ : a = 6)
  (h₂ : t = a^2 - 1) : a + t = 41 :=
by
  sorry

end sum_a_t_l203_203998


namespace unique_plants_count_1320_l203_203129

open Set

variable (X Y Z : Finset ℕ)

def total_plants_X : ℕ := 600
def total_plants_Y : ℕ := 480
def total_plants_Z : ℕ := 420
def shared_XY : ℕ := 60
def shared_YZ : ℕ := 70
def shared_XZ : ℕ := 80
def shared_XYZ : ℕ := 30

theorem unique_plants_count_1320 : X.card = total_plants_X →
                                Y.card = total_plants_Y →
                                Z.card = total_plants_Z →
                                (X ∩ Y).card = shared_XY →
                                (Y ∩ Z).card = shared_YZ →
                                (X ∩ Z).card = shared_XZ →
                                (X ∩ Y ∩ Z).card = shared_XYZ →
                                (X ∪ Y ∪ Z).card = 1320 := 
by {
  sorry
}

end unique_plants_count_1320_l203_203129


namespace total_students_left_l203_203827

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l203_203827


namespace geometric_body_view_circle_l203_203803

theorem geometric_body_view_circle (P : Type) (is_circle : P → Prop) (is_sphere : P → Prop)
  (is_cylinder : P → Prop) (is_cone : P → Prop) (is_rectangular_prism : P → Prop) :
  (∀ x, is_sphere x → is_circle x) →
  (∃ x, is_cylinder x ∧ is_circle x) →
  (∃ x, is_cone x ∧ is_circle x) →
  ¬ (∃ x, is_rectangular_prism x ∧ is_circle x) :=
by
  intros h_sphere h_cylinder h_cone h_rectangular_prism
  sorry

end geometric_body_view_circle_l203_203803


namespace equal_animals_per_aquarium_l203_203328

theorem equal_animals_per_aquarium (aquariums animals : ℕ) (h1 : aquariums = 26) (h2 : animals = 52) (h3 : ∀ a, a = animals / aquariums) : a = 2 := 
by
  sorry

end equal_animals_per_aquarium_l203_203328


namespace quadratic_inequality_solutions_l203_203664

theorem quadratic_inequality_solutions (a x : ℝ) :
  (x^2 - (2+a)*x + 2*a > 0) → (
    (a < 2  → (x < a ∨ x > 2)) ∧
    (a = 2  → (x ≠ 2)) ∧
    (a > 2  → (x < 2 ∨ x > a))
  ) :=
by sorry

end quadratic_inequality_solutions_l203_203664


namespace find_two_digit_number_with_cubic_ending_in_9_l203_203724

theorem find_two_digit_number_with_cubic_ending_in_9:
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n^3 % 10 = 9 ∧ n = 19 := 
by
  sorry

end find_two_digit_number_with_cubic_ending_in_9_l203_203724


namespace sarah_total_pencils_l203_203567

-- Define the number of pencils Sarah buys on each day
def pencils_monday : ℕ := 35
def pencils_tuesday : ℕ := 42
def pencils_wednesday : ℕ := 3 * pencils_tuesday
def pencils_thursday : ℕ := pencils_wednesday / 2
def pencils_friday : ℕ := 2 * pencils_monday

-- Define the total number of pencils
def total_pencils : ℕ :=
  pencils_monday + pencils_tuesday + pencils_wednesday + pencils_thursday + pencils_friday

-- Theorem statement to prove the total number of pencils equals 336
theorem sarah_total_pencils : total_pencils = 336 :=
by
  -- here goes the proof, but it is not required
  sorry

end sarah_total_pencils_l203_203567


namespace existence_of_unusual_100_digit_numbers_l203_203360

theorem existence_of_unusual_100_digit_numbers :
  ∃ (n₁ n₂ : ℕ), 
  (n₁ = 10^100 - 1) ∧ (n₂ = 5 * 10^99 - 1) ∧ 
  (∀ x : ℕ, x = n₁ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) ∧
  (∀ x : ℕ, x = n₂ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) := 
sorry

end existence_of_unusual_100_digit_numbers_l203_203360


namespace total_cost_correct_l203_203838

def sandwich_cost : ℝ := 2.44
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_correct :
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 8.36 := by
  sorry

end total_cost_correct_l203_203838


namespace exponent_product_l203_203381

theorem exponent_product (a : ℝ) (m n : ℕ)
  (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 :=
sorry

end exponent_product_l203_203381


namespace coefficient_x99_is_zero_l203_203774

open Polynomial

noncomputable def P (x : ℤ) : Polynomial ℤ := sorry
noncomputable def Q (x : ℤ) : Polynomial ℤ := sorry

theorem coefficient_x99_is_zero : 
    (P 0 = 1) → 
    ((P x)^2 = 1 + x + x^100 * Q x) → 
    (Polynomial.coeff ((P x + 1)^100) 99 = 0) :=
by
    -- Proof omitted
    sorry

end coefficient_x99_is_zero_l203_203774


namespace distance_from_D_to_plane_B1EF_l203_203020

theorem distance_from_D_to_plane_B1EF :
  let D := (0, 0, 0)
  let B₁ := (1, 1, 1)
  let E := (1, 1/2, 0)
  let F := (1/2, 1, 0)
  ∃ (d : ℝ), d = 1 := by
  sorry

end distance_from_D_to_plane_B1EF_l203_203020


namespace min_exponent_binomial_l203_203705

theorem min_exponent_binomial (n : ℕ) (h1 : n > 0)
  (h2 : ∃ r : ℕ, (n.choose r) / (n.choose (r + 1)) = 5 / 7) : n = 11 :=
by {
-- Note: We are merely stating the theorem here according to the instructions,
-- the proof body is omitted and hence the use of 'sorry'.
sorry
}

end min_exponent_binomial_l203_203705


namespace max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l203_203063

open Real

noncomputable def max_value_b_minus_inv_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
b - (1 / a)

noncomputable def min_value_inv_3a_plus_1_plus_inv_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
(1 / (3 * a + 1)) + (1 / (a + b))

theorem max_value_b_minus_inv_a_is_minus_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  max_value_b_minus_inv_a a b ha hb h = -1 :=
sorry

theorem min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  min_value_inv_3a_plus_1_plus_inv_a_plus_b a b ha hb h = 1 :=
sorry

end max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l203_203063


namespace wire_cut_ratio_l203_203939

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l203_203939


namespace find_sin_θ_l203_203297

open Real

noncomputable def θ_in_range_and_sin_2θ (θ : ℝ) : Prop :=
  (θ ∈ Set.Icc (π / 4) (π / 2)) ∧ (sin (2 * θ) = 3 * sqrt 7 / 8)

theorem find_sin_θ (θ : ℝ) (h : θ_in_range_and_sin_2θ θ) : sin θ = 3 / 4 :=
  sorry

end find_sin_θ_l203_203297


namespace simple_interest_rate_l203_203741

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) :
  (T = 20) →
  (SI = P) →
  (SI = P * R * T / 100) →
  R = 5 :=
by
  sorry

end simple_interest_rate_l203_203741


namespace correct_statement_b_l203_203818

open Set 

variables {Point Line Plane : Type}
variable (m n : Line)
variable (α : Plane)
variable (perpendicular_to_plane : Line → Plane → Prop) 
variable (parallel_to_plane : Line → Plane → Prop)
variable (is_subline_of_plane : Line → Plane → Prop)
variable (perpendicular_to_line : Line → Line → Prop)

theorem correct_statement_b (hm : perpendicular_to_plane m α) (hn : is_subline_of_plane n α) : perpendicular_to_line m n :=
sorry

end correct_statement_b_l203_203818


namespace common_ratio_of_geo_seq_l203_203702

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem common_ratio_of_geo_seq :
  (∀ n, 0 < a n) →
  geometric_sequence a q →
  a 6 = a 5 + 2 * a 4 →
  q = 2 :=
by
  intros
  sorry

end common_ratio_of_geo_seq_l203_203702


namespace functional_eq_unique_solution_l203_203250

theorem functional_eq_unique_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_eq_unique_solution_l203_203250


namespace simplify_expression_l203_203622

theorem simplify_expression
  (x y : ℝ)
  (h : (x + 2)^3 ≠ (y - 2)^3) :
  ( (x + 2)^3 + (y + x)^3 ) / ( (x + 2)^3 - (y - 2)^3 ) = (2 * x + y + 2) / (x - y + 4) :=
sorry

end simplify_expression_l203_203622


namespace find_k_value_l203_203109

theorem find_k_value (k : ℚ) :
  (∀ x y : ℚ, (x = 1/3 ∧ y = -8 → -3/4 - 3 * k * x = 7 * y)) → k = 55.25 :=
by
  sorry

end find_k_value_l203_203109


namespace smallest_AAB_value_exists_l203_203718

def is_consecutive_digits (A B : ℕ) : Prop :=
  (B = A + 1 ∨ A = B + 1) ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9

def two_digit_to_int (A B : ℕ) : ℕ :=
  10 * A + B

def three_digit_to_int (A B : ℕ) : ℕ :=
  110 * A + B

theorem smallest_AAB_value_exists :
  ∃ (A B: ℕ), is_consecutive_digits A B ∧ two_digit_to_int A B = (1 / 7 : ℝ) * ↑(three_digit_to_int A B) ∧ three_digit_to_int A B = 889 :=
sorry

end smallest_AAB_value_exists_l203_203718


namespace students_in_both_clubs_l203_203285

theorem students_in_both_clubs
  (T R B total_club_students : ℕ)
  (hT : T = 85) (hR : R = 120)
  (hTotal : T + R - B = total_club_students)
  (hTotalVal : total_club_students = 180) :
  B = 25 :=
by
  -- Placeholder for proof
  sorry

end students_in_both_clubs_l203_203285


namespace pigeons_count_l203_203128

theorem pigeons_count :
  let initial_pigeons := 1
  let additional_pigeons := 1
  (initial_pigeons + additional_pigeons) = 2 :=
by
  sorry

end pigeons_count_l203_203128


namespace marked_price_percentage_fixed_l203_203127

-- Definitions based on the conditions
def discount_percentage : ℝ := 0.18461538461538467
def profit_percentage : ℝ := 0.06

-- The final theorem statement
theorem marked_price_percentage_fixed (CP MP SP : ℝ) 
  (h1 : SP = CP * (1 + profit_percentage))  
  (h2 : SP = MP * (1 - discount_percentage)) :
  (MP / CP - 1) * 100 = 30 := 
sorry

end marked_price_percentage_fixed_l203_203127


namespace first_term_of_arithmetic_sequence_l203_203672

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)
  (h_arith : ∀ n, a n = a1 + ↑n - 1) 
  (h_sum : ∀ n, S n = n / 2 * (2 * a1 + (n - 1))) 
  (h_min : ∀ n, S 2022 ≤ S n) : 
  -2022 < a1 ∧ a1 < -2021 :=
by
  sorry

end first_term_of_arithmetic_sequence_l203_203672


namespace deck_card_count_l203_203854

theorem deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 :=
by
  sorry

end deck_card_count_l203_203854


namespace dollars_sum_l203_203383

theorem dollars_sum : 
  (5 / 8 : ℝ) + (2 / 5) = 1.025 :=
by
  sorry

end dollars_sum_l203_203383


namespace number_of_divisible_factorials_l203_203439

theorem number_of_divisible_factorials:
  ∃ (count : ℕ), count = 36 ∧ ∀ n, 1 ≤ n ∧ n ≤ 50 → (∃ k : ℕ, n! = k * (n * (n + 1)) / 2) ↔ n ≤ n - 14 :=
sorry

end number_of_divisible_factorials_l203_203439


namespace sticks_in_100th_stage_l203_203587

theorem sticks_in_100th_stage : 
  ∀ (n a₁ d : ℕ), a₁ = 5 → d = 4 → n = 100 → a₁ + (n - 1) * d = 401 :=
by
  sorry

end sticks_in_100th_stage_l203_203587


namespace minimum_guests_needed_l203_203018

theorem minimum_guests_needed (total_food : ℕ) (max_food_per_guest : ℕ) (guests_needed : ℕ) : 
  total_food = 323 → max_food_per_guest = 2 → guests_needed = Nat.ceil (323 / 2) → guests_needed = 162 :=
by
  intros
  sorry

end minimum_guests_needed_l203_203018


namespace probability_diff_digits_l203_203560

open Finset

def two_digit_same_digit (n : ℕ) : Prop :=
  n / 10 = n % 10

def three_digit_same_digit (n : ℕ) : Prop :=
  (n % 100) / 10 = n / 100 ∧ (n / 100) = (n % 10)

def same_digit (n : ℕ) : Prop :=
  two_digit_same_digit n ∨ three_digit_same_digit n

def total_numbers : ℕ :=
  (199 - 10 + 1)

def same_digit_count : ℕ :=
  9 + 9

theorem probability_diff_digits : 
  ((total_numbers - same_digit_count) / total_numbers : ℚ) = 86 / 95 :=
by
  sorry

end probability_diff_digits_l203_203560


namespace triangle_area_l203_203187

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 ∧ 0.5 * a * b = 24 :=
by {
  sorry
}

end triangle_area_l203_203187


namespace num_expr_div_by_10_l203_203931

theorem num_expr_div_by_10 : (11^11 + 12^12 + 13^13) % 10 = 0 := by
  sorry

end num_expr_div_by_10_l203_203931


namespace ratio_of_adults_to_children_is_24_over_25_l203_203345

theorem ratio_of_adults_to_children_is_24_over_25
  (a c : ℕ) (h₁ : a ≥ 1) (h₂ : c ≥ 1) 
  (h₃ : 30 * a + 18 * c = 2340) 
  (h₄ : c % 5 = 0) :
  a = 48 ∧ c = 50 ∧ (a / c : ℚ) = 24 / 25 :=
sorry

end ratio_of_adults_to_children_is_24_over_25_l203_203345


namespace circle_equation_condition_l203_203822

theorem circle_equation_condition (m : ℝ) : 
  (∃ h k r : ℝ, (r > 0) ∧ ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + y^2 - 2*x - 4*y + m = 0) ↔ m < 5 :=
sorry

end circle_equation_condition_l203_203822


namespace Charles_speed_with_music_l203_203101

theorem Charles_speed_with_music (S : ℝ) (h1 : 40 / 60 + 30 / 60 = 70 / 60) (h2 : S * (40 / 60) + 4 * (30 / 60) = 6) : S = 8 :=
by
  sorry

end Charles_speed_with_music_l203_203101


namespace find_chemistry_marks_l203_203768

theorem find_chemistry_marks 
    (marks_english : ℕ := 70)
    (marks_math : ℕ := 63)
    (marks_physics : ℕ := 80)
    (marks_biology : ℕ := 65)
    (average_marks : ℚ := 68.2) :
    ∃ (marks_chemistry : ℕ), 
      (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) = 5 * average_marks 
      → marks_chemistry = 63 :=
by
  sorry

end find_chemistry_marks_l203_203768


namespace train_speed_conversion_l203_203706

def km_per_hour_to_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 108) :
  km_per_hour_to_m_per_s speed_kmph = 30 :=
by
  rw [h]
  sorry

end train_speed_conversion_l203_203706


namespace technicians_count_l203_203495

-- Define the number of workers
def total_workers : ℕ := 21

-- Define the average salaries
def avg_salary_all : ℕ := 8000
def avg_salary_technicians : ℕ := 12000
def avg_salary_rest : ℕ := 6000

-- Define the number of technicians and rest of workers
variable (T R : ℕ)

-- Define the equations based on given conditions
def equation1 := T + R = total_workers
def equation2 := (T * avg_salary_technicians) + (R * avg_salary_rest) = total_workers * avg_salary_all

-- Prove the number of technicians
theorem technicians_count : T = 7 :=
by
  sorry

end technicians_count_l203_203495


namespace find_f_difference_l203_203230

variable {α : Type*}
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_period : ∀ x, f (x + 5) = f x)
variable (h_value : f (-2) = 2)

theorem find_f_difference : f 2012 - f 2010 = -2 :=
by {
  sorry
}

end find_f_difference_l203_203230


namespace systematic_sampling_probabilities_l203_203246

-- Define the total number of students
def total_students : ℕ := 1005

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of individuals removed
def individuals_removed : ℕ := 5

-- Define the probability of an individual being removed
def probability_removed : ℚ := individuals_removed / total_students

-- Define the probability of an individual being selected in the sample
def probability_selected : ℚ := sample_size / total_students

-- The statement we need to prove
theorem systematic_sampling_probabilities :
  probability_removed = 5 / 1005 ∧ probability_selected = 50 / 1005 :=
sorry

end systematic_sampling_probabilities_l203_203246


namespace fruit_platter_has_thirty_fruits_l203_203349

-- Define the conditions
def at_least_five_apples (g_apple r_apple y_apple : ℕ) : Prop :=
  g_apple + r_apple + y_apple ≥ 5

def at_most_five_oranges (r_orange y_orange : ℕ) : Prop :=
  r_orange + y_orange ≤ 5

def kiwi_grape_constraints (g_kiwi p_grape : ℕ) : Prop :=
  g_kiwi + p_grape ≥ 8 ∧ g_kiwi + p_grape ≤ 12 ∧ g_kiwi = p_grape

def at_least_one_each_grape (g_grape p_grape : ℕ) : Prop :=
  g_grape ≥ 1 ∧ p_grape ≥ 1

-- The final statement to prove
theorem fruit_platter_has_thirty_fruits :
  ∃ (g_apple r_apple y_apple r_orange y_orange g_kiwi p_grape g_grape : ℕ),
    at_least_five_apples g_apple r_apple y_apple ∧
    at_most_five_oranges r_orange y_orange ∧
    kiwi_grape_constraints g_kiwi p_grape ∧
    at_least_one_each_grape g_grape p_grape ∧
    g_apple + r_apple + y_apple + r_orange + y_orange + g_kiwi + p_grape + g_grape = 30 :=
sorry

end fruit_platter_has_thirty_fruits_l203_203349


namespace find_valid_pairs_l203_203427

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_pair (p q : ℕ) : Prop :=
  p < 2005 ∧ q < 2005 ∧ is_prime p ∧ is_prime q ∧ q ∣ p^2 + 8 ∧ p ∣ q^2 + 8

theorem find_valid_pairs :
  ∀ p q, valid_pair p q → (p, q) = (2, 2) ∨ (p, q) = (881, 89) ∨ (p, q) = (89, 881) :=
sorry

end find_valid_pairs_l203_203427


namespace volume_frustum_l203_203262

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

theorem volume_frustum (original_base_edge original_height small_base_edge small_height : ℝ)
  (h_orig : original_base_edge = 10) (h_orig_height : original_height = 10)
  (h_small : small_base_edge = 5) (h_small_height : small_height = 5) :
  volume_pyramid original_base_edge original_height - volume_pyramid small_base_edge small_height
  = 875 / 3 := by
    simp [volume_pyramid, h_orig, h_orig_height, h_small, h_small_height]
    sorry

end volume_frustum_l203_203262


namespace optionB_unfactorable_l203_203934

-- Definitions for the conditions
def optionA (a b : ℝ) : ℝ := -a^2 + b^2
def optionB (x y : ℝ) : ℝ := x^2 + y^2
def optionC (z : ℝ) : ℝ := 49 - z^2
def optionD (m : ℝ) : ℝ := 16 - 25 * m^2

-- The proof statement that option B cannot be factored over the real numbers
theorem optionB_unfactorable (x y : ℝ) : ¬ ∃ (p q : ℝ → ℝ), p x * q y = x^2 + y^2 :=
sorry -- Proof to be filled in

end optionB_unfactorable_l203_203934


namespace two_digit_numbers_satisfying_l203_203006

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_numbers_satisfying (n : ℕ) : 
  is_two_digit n → n = P n + S n ↔ (n % 10 = 9) :=
by
  sorry

end two_digit_numbers_satisfying_l203_203006


namespace subset_sum_bounds_l203_203312

theorem subset_sum_bounds (M m n : ℕ) (A : Finset ℕ)
  (h1 : 1 ≤ m) (h2 : m ≤ n) (h3 : 1 ≤ M) (h4 : M ≤ (m * (m + 1)) / 2) (hA : A.card = m) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (n + 1)) :
  ∃ B ⊆ A, 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m :=
by
  sorry

end subset_sum_bounds_l203_203312


namespace peggy_dolls_ratio_l203_203289

noncomputable def peggy_dolls_original := 6
noncomputable def peggy_dolls_from_grandmother := 30
noncomputable def peggy_dolls_total := 51

theorem peggy_dolls_ratio :
  ∃ x, peggy_dolls_original + peggy_dolls_from_grandmother + x = peggy_dolls_total ∧ x / peggy_dolls_from_grandmother = 1 / 2 :=
by {
  sorry
}

end peggy_dolls_ratio_l203_203289


namespace cuts_for_20_pentagons_l203_203888

theorem cuts_for_20_pentagons (K : ℕ) : 20 * 540 + (K - 19) * 180 ≤ 360 * K + 540 ↔ K ≥ 38 :=
by
  sorry

end cuts_for_20_pentagons_l203_203888


namespace sum_S11_l203_203990

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {a1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom condition : a 3 + 4 = a 2 + a 7

theorem sum_S11 : S 11 = 44 := by
  sorry

end sum_S11_l203_203990


namespace quadrilateral_area_correct_l203_203004

noncomputable def area_of_quadrilateral (n : ℕ) (hn : n > 0) : ℚ :=
  (2 * n^3) / (4 * n^2 - 1)

theorem quadrilateral_area_correct (n : ℕ) (hn : n > 0) :
  ∃ area : ℚ, area = (2 * n^3) / (4 * n^2 - 1) :=
by
  use area_of_quadrilateral n hn
  sorry

end quadrilateral_area_correct_l203_203004


namespace perpendicular_lines_l203_203756

theorem perpendicular_lines (m : ℝ) : 
  (m = -2 → (2-m) * (-(m+3)/(2-m)) + m * (m-3) / (-(m+3)) = 0) → 
  (m = -2 ∨ m = 1) := 
sorry

end perpendicular_lines_l203_203756


namespace find_expression_value_l203_203072

theorem find_expression_value (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3 * y^3) / 9 = 73 / 3 :=
by
  sorry

end find_expression_value_l203_203072


namespace number_problem_l203_203976

theorem number_problem (x : ℤ) (h1 : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := by
  sorry

end number_problem_l203_203976


namespace maddie_milk_usage_l203_203948

-- Define the constants based on the problem conditions
def cups_per_day : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def bag_cost : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def weekly_coffee_expense : ℝ := 18
def gallon_milk_cost : ℝ := 4

-- Define the proof problem
theorem maddie_milk_usage : 
  (0.5 : ℝ) = (weekly_coffee_expense - 2 * ((cups_per_day * ounces_per_cup * 7) / ounces_per_bag * bag_cost)) / gallon_milk_cost :=
by 
  sorry

end maddie_milk_usage_l203_203948


namespace parallel_vectors_x_value_l203_203790

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  vectors_parallel (1, -2) (x, 1) → x = -1 / 2 :=
by
  sorry

end parallel_vectors_x_value_l203_203790


namespace hyperbola_standard_eq_line_eq_AB_l203_203103

noncomputable def fixed_points : (Real × Real) × (Real × Real) := ((-Real.sqrt 2, 0.0), (Real.sqrt 2, 0.0))

def locus_condition (P : Real × Real) (F1 F2 : Real × Real) : Prop :=
  abs (dist P F2 - dist P F1) = 2

def curve_E (P : Real × Real) : Prop :=
  (P.1 < 0) ∧ (P.1 * P.1 - P.2 * P.2 = 1)

theorem hyperbola_standard_eq :
  ∃ P : Real × Real, locus_condition P (fixed_points.1) (fixed_points.2) ↔ curve_E P :=
sorry

def line_intersects_hyperbola (P : Real × Real) (k : Real) : Prop :=
  P.2 = k * P.1 - 1 ∧ curve_E P

def dist_A_B (A B : Real × Real) : Real :=
  dist A B

theorem line_eq_AB :
  ∃ k : Real, k = -Real.sqrt 5 / 2 ∧
              ∃ A B : Real × Real, line_intersects_hyperbola A k ∧ 
              line_intersects_hyperbola B k ∧ 
              dist_A_B A B = 6 * Real.sqrt 3 ∧
              ∀ x y : Real, y = k * x - 1 ↔ x * (Real.sqrt 5/2) + y + 1 = 0 :=
sorry

end hyperbola_standard_eq_line_eq_AB_l203_203103


namespace profit_percentage_is_22_percent_l203_203929

-- Define the given conditions
def scooter_cost (C : ℝ) := C
def repair_cost (C : ℝ) := 0.10 * C
def repair_cost_value := 500
def profit := 1100

-- Let's state the main theorem
theorem profit_percentage_is_22_percent (C : ℝ) 
  (h1 : repair_cost C = repair_cost_value)
  (h2 : profit = 1100) : 
  (profit / C) * 100 = 22 :=
by
  sorry

end profit_percentage_is_22_percent_l203_203929


namespace find_a_l203_203816

-- Definition of * in terms of 2a - b^2
def custom_mul (a b : ℤ) := 2 * a - b^2

-- The proof statement
theorem find_a (a : ℤ) : custom_mul a 3 = 3 → a = 6 :=
by
  sorry

end find_a_l203_203816


namespace mark_more_hours_l203_203279

-- Definitions based on the conditions
variables (Pat Kate Mark Alex : ℝ)
variables (total_hours : ℝ)
variables (h1 : Pat + Kate + Mark + Alex = 350)
variables (h2 : Pat = 2 * Kate)
variables (h3 : Pat = (1 / 3) * Mark)
variables (h4 : Alex = 1.5 * Kate)

-- Theorem statement with the desired proof target
theorem mark_more_hours (Pat Kate Mark Alex : ℝ) (h1 : Pat + Kate + Mark + Alex = 350) 
(h2 : Pat = 2 * Kate) (h3 : Pat = (1 / 3) * Mark) (h4 : Alex = 1.5 * Kate) : 
Mark - (Kate + Alex) = 116.66666666666667 := sorry

end mark_more_hours_l203_203279


namespace height_drawn_to_hypotenuse_l203_203961

-- Definitions for the given problem
variables {A B C D : Type}
variables {area : ℝ}
variables {angle_ratio : ℝ}
variables {h : ℝ}

-- Given conditions
def is_right_triangle (A B C : Type) : Prop := -- definition for the right triangle
sorry

def area_of_triangle (A B C : Type) (area: ℝ) : Prop := 
area = ↑(2 : ℝ) * Real.sqrt 3  -- area given as 2√3 cm²

def angle_bisector_ratios (A B C D : Type) (ratio: ℝ) : Prop :=
ratio = 1 / 2  -- given ratio 1:2

-- Question statement
theorem height_drawn_to_hypotenuse (A B C D : Type) 
  (right_triangle : is_right_triangle A B C)
  (area_cond : area_of_triangle A B C area)
  (angle_ratio_cond : angle_bisector_ratios A B C D angle_ratio):
  h = Real.sqrt 3 :=
sorry

end height_drawn_to_hypotenuse_l203_203961


namespace distance_to_left_focus_l203_203395

theorem distance_to_left_focus (P : ℝ × ℝ) 
  (h1 : P.1^2 / 100 + P.2^2 / 36 = 1) 
  (h2 : dist P (50 - 100 / 9, P.2) = 17 / 2) :
  dist P (-50 - 100 / 9, P.2) = 66 / 5 :=
sorry

end distance_to_left_focus_l203_203395


namespace find_ab_l203_203667

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l203_203667


namespace number_of_lines_intersecting_circle_l203_203697

theorem number_of_lines_intersecting_circle : 
  ∃ l : ℕ, 
  (∀ a b x y : ℤ, (x^2 + y^2 = 50 ∧ (x / a + y / b = 1))) → 
  (∃ n : ℕ, n = 60) :=
sorry

end number_of_lines_intersecting_circle_l203_203697


namespace ratio_of_girls_to_boys_l203_203634

theorem ratio_of_girls_to_boys (g b : ℕ) (h1 : g = b + 6) (h2 : g + b = 36) : g / b = 7 / 5 := by sorry

end ratio_of_girls_to_boys_l203_203634


namespace sum_of_geometric_sequence_eq_31_over_16_l203_203517

theorem sum_of_geometric_sequence_eq_31_over_16 (n : ℕ) :
  let a := 1
  let r := (1 / 2 : ℝ)
  let S_n := 2 - 2 * r^n
  (S_n = (31 / 16 : ℝ)) ↔ (n = 5) := by
{
  sorry
}

end sum_of_geometric_sequence_eq_31_over_16_l203_203517


namespace calculate_total_cost_l203_203925

def cost_of_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def cost_of_non_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def total_cost (p_l1 p_l2 np_l1 np_l2 ppf np_pf : ℕ) : ℕ :=
  cost_of_parallel_sides p_l1 p_l2 ppf + cost_of_non_parallel_sides np_l1 np_l2 np_pf

theorem calculate_total_cost :
  total_cost 25 37 20 24 48 60 = 5616 :=
by
  -- Assuming the conditions are correctly applied, the statement aims to validate that the calculated
  -- sum of the costs for the specified fence sides equal Rs 5616.
  sorry

end calculate_total_cost_l203_203925


namespace toms_total_out_of_pocket_is_680_l203_203951

namespace HealthCosts

def doctor_visit_cost : ℝ := 300
def cast_cost : ℝ := 200
def initial_insurance_coverage : ℝ := 0.60
def therapy_session_cost : ℝ := 100
def number_of_sessions : ℕ := 8
def therapy_insurance_coverage : ℝ := 0.40

def total_initial_cost : ℝ :=
  doctor_visit_cost + cast_cost

def initial_out_of_pocket : ℝ :=
  total_initial_cost * (1 - initial_insurance_coverage)

def total_therapy_cost : ℝ :=
  therapy_session_cost * number_of_sessions

def therapy_out_of_pocket : ℝ :=
  total_therapy_cost * (1 - therapy_insurance_coverage)

def total_out_of_pocket : ℝ :=
  initial_out_of_pocket + therapy_out_of_pocket

theorem toms_total_out_of_pocket_is_680 :
  total_out_of_pocket = 680 := by
  sorry

end HealthCosts

end toms_total_out_of_pocket_is_680_l203_203951


namespace cassidy_grounded_days_l203_203989

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l203_203989


namespace associates_more_than_two_years_l203_203061

-- Definitions based on the given conditions
def total_associates := 100
def second_year_associates_percent := 25
def not_first_year_associates_percent := 75

-- The theorem to prove
theorem associates_more_than_two_years :
  not_first_year_associates_percent - second_year_associates_percent = 50 :=
by
  -- The proof is omitted
  sorry

end associates_more_than_two_years_l203_203061


namespace determine_list_price_l203_203759

theorem determine_list_price (x : ℝ) :
  0.12 * (x - 15) = 0.15 * (x - 25) → x = 65 :=
by 
  sorry

end determine_list_price_l203_203759


namespace largest_angle_triangl_DEF_l203_203670

theorem largest_angle_triangl_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -8) : 
  ∃ (F : ℝ), F = 109.47 ∧ (F > 90) := by sorry

end largest_angle_triangl_DEF_l203_203670


namespace find_a3_l203_203416

open Nat

def seq (a : ℕ → ℕ) : Prop := 
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a (n + 1) - a n = n)

theorem find_a3 (a : ℕ → ℕ) (h : seq a) : a 3 = 4 := by
  sorry

end find_a3_l203_203416


namespace min_squares_to_cover_staircase_l203_203290

-- Definition of the staircase and the constraints
def is_staircase (n : ℕ) (s : ℕ → ℕ) : Prop :=
  ∀ i, i < n → s i = i + 1

-- The proof problem statement
theorem min_squares_to_cover_staircase : 
  ∀ n : ℕ, n = 15 →
  ∀ s : ℕ → ℕ, is_staircase n s →
  ∃ k : ℕ, k = 15 ∧ (∀ i, i < n → ∃ a b : ℕ, a ≤ i ∧ b ≤ s a ∧ ∃ (l : ℕ), l = 1) :=
by
  sorry

end min_squares_to_cover_staircase_l203_203290


namespace probability_sales_greater_than_10000_l203_203323

/-- Define the probability that the sales of new energy vehicles in a randomly selected city are greater than 10000 -/
theorem probability_sales_greater_than_10000 :
  (1 / 2) * (2 / 10) + (1 / 2) * (6 / 10) = 2 / 5 :=
by sorry

end probability_sales_greater_than_10000_l203_203323


namespace solve_linear_equation_one_variable_with_parentheses_l203_203027

/--
Theorem: Solving a linear equation in one variable that contains parentheses
is equivalent to the process of:
1. Removing the parentheses,
2. Moving terms,
3. Combining like terms, and
4. Making the coefficient of the unknown equal to 1.

Given: a linear equation in one variable that contains parentheses
Prove: The process of solving it is to remove the parentheses, move terms, combine like terms, and make the coefficient of the unknown equal to 1.
-/
theorem solve_linear_equation_one_variable_with_parentheses
  (eq : String) :
  ∃ instructions : String,
    instructions = "remove the parentheses; move terms; combine like terms; make the coefficient of the unknown equal to 1" :=
by
  sorry

end solve_linear_equation_one_variable_with_parentheses_l203_203027


namespace relationship_between_y1_y2_l203_203616

theorem relationship_between_y1_y2 (y1 y2 : ℝ)
  (h1 : y1 = -2 * (-2) + 3)
  (h2 : y2 = -2 * 3 + 3) :
  y1 > y2 := by
  sorry

end relationship_between_y1_y2_l203_203616


namespace saree_original_price_l203_203896

theorem saree_original_price
  (sale_price : ℝ)
  (P : ℝ)
  (h_discount : sale_price = 0.80 * P * 0.95)
  (h_sale_price : sale_price = 266) :
  P = 350 :=
by
  -- Proof to be completed later
  sorry

end saree_original_price_l203_203896


namespace polygon_intersections_inside_circle_l203_203564

noncomputable def number_of_polygon_intersections
    (polygonSides: List Nat) : Nat :=
  let pairs := [(4,5), (4,7), (4,9), (5,7), (5,9), (7,9)]
  pairs.foldl (λ acc (p1, p2) => acc + 2 * min p1 p2) 0

theorem polygon_intersections_inside_circle :
  number_of_polygon_intersections [4, 5, 7, 9] = 58 :=
by
  sorry

end polygon_intersections_inside_circle_l203_203564


namespace income_scientific_notation_l203_203521

theorem income_scientific_notation (avg_income_per_acre : ℝ) (acres : ℝ) (a n : ℝ) :
  avg_income_per_acre = 20000 →
  acres = 8000 → 
  (avg_income_per_acre * acres = a * 10 ^ n ↔ (a = 1.6 ∧ n = 8)) :=
by
  sorry

end income_scientific_notation_l203_203521


namespace problem_1_problem_2_l203_203331

open Set

-- First problem: when a = 2
theorem problem_1:
  ∀ (x : ℝ), 2 * x^2 - x - 1 > 0 ↔ (x < -(1 / 2) ∨ x > 1) :=
by
  sorry

-- Second problem: when a > -1
theorem problem_2 (a : ℝ) (h : a > -1) :
  ∀ (x : ℝ), 
    (if a = 0 then x - 1 > 0 else if a > 0 then  a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (x < -1 / a ∨ x > 1) 
    else a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (1 < x ∧ x < -1 / a)) :=
by
  sorry

end problem_1_problem_2_l203_203331


namespace share_of_A_l203_203056

-- Definitions corresponding to the conditions
variables (A B C : ℝ)
variable (total : ℝ := 578)
variable (share_ratio_B_C : ℝ := 1 / 4)
variable (share_ratio_A_B : ℝ := 2 / 3)

-- Conditions
def condition1 : B = share_ratio_B_C * C := by sorry
def condition2 : A = share_ratio_A_B * B := by sorry
def condition3 : A + B + C = total := by sorry

-- The equivalent math proof problem statement
theorem share_of_A :
  A = 68 :=
by sorry

end share_of_A_l203_203056


namespace initial_bacteria_count_l203_203030

theorem initial_bacteria_count (n : ℕ) : 
  (n * 4^10 = 4194304) → n = 4 :=
by
  sorry

end initial_bacteria_count_l203_203030


namespace probability_Hugo_first_roll_is_six_l203_203284

/-
In a dice game, each of 5 players, including Hugo, rolls a standard 6-sided die. 
The winner is the player who rolls the highest number. 
In the event of a tie for the highest roll, those involved in the tie roll again until a clear winner emerges.
-/
variable (HugoRoll : Nat) (A1 B1 C1 D1 : Nat)
variable (W : Bool)

-- Conditions in the problem
def isWinner (HugoRoll : Nat) (W : Bool) : Prop := (W = true)
def firstRollAtLeastFour (HugoRoll : Nat) : Prop := HugoRoll >= 4
def firstRollIsSix (HugoRoll : Nat) : Prop := HugoRoll = 6

-- Hypotheses: Hugo's event conditions
axiom HugoWonAndRollsAtLeastFour : isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll

-- Target probability based on problem statement
noncomputable def probability (p : ℚ) : Prop := p = 625 / 4626

-- Main statement
theorem probability_Hugo_first_roll_is_six (HugoRoll : Nat) (A1 B1 C1 D1 : Nat) (W : Bool) :
  isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll → 
  probability (625 / 4626) := by
  sorry


end probability_Hugo_first_roll_is_six_l203_203284


namespace mrs_mcpherson_percentage_l203_203861

def total_rent : ℕ := 1200
def mr_mcpherson_amount : ℕ := 840
def mrs_mcpherson_amount : ℕ := total_rent - mr_mcpherson_amount

theorem mrs_mcpherson_percentage : (mrs_mcpherson_amount.toFloat / total_rent.toFloat) * 100 = 30 :=
by
  sorry

end mrs_mcpherson_percentage_l203_203861


namespace uki_cupcakes_per_day_l203_203125

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def daily_cookies : ℝ := 10
def daily_biscuits : ℝ := 20
def total_earnings : ℝ := 350
def days : ℝ := 5

-- Define the number of cupcakes baked per day
def cupcakes_per_day (x : ℝ) : Prop :=
  let earnings_cupcakes := price_cupcake * x * days
  let earnings_cookies := price_cookie * daily_cookies * days
  let earnings_biscuits := price_biscuit * daily_biscuits * days
  earnings_cupcakes + earnings_cookies + earnings_biscuits = total_earnings

-- The statement to be proven
theorem uki_cupcakes_per_day : cupcakes_per_day 20 :=
by 
  sorry

end uki_cupcakes_per_day_l203_203125


namespace other_divisor_l203_203149

theorem other_divisor (x : ℕ) (h1 : 266 % 33 = 2) (h2 : 266 % x = 2) : x = 132 :=
sorry

end other_divisor_l203_203149


namespace Z_4_3_eq_37_l203_203632

def Z (a b : ℕ) : ℕ :=
  a^2 + a * b + b^2

theorem Z_4_3_eq_37 : Z 4 3 = 37 :=
  by
    sorry

end Z_4_3_eq_37_l203_203632


namespace Jolene_charge_per_car_l203_203620

theorem Jolene_charge_per_car (babysitting_families cars_washed : ℕ) (charge_per_family total_raised babysitting_earnings car_charge : ℕ) :
  babysitting_families = 4 →
  charge_per_family = 30 →
  cars_washed = 5 →
  total_raised = 180 →
  babysitting_earnings = babysitting_families * charge_per_family →
  car_charge = (total_raised - babysitting_earnings) / cars_washed →
  car_charge = 12 :=
by
  intros
  sorry

end Jolene_charge_per_car_l203_203620


namespace train_bus_difference_l203_203711

variable (T : ℝ)  -- T is the cost of a train ride

-- conditions
def cond1 := T + 1.50 = 9.85
def cond2 := 1.50 = 1.50

theorem train_bus_difference (h1 : cond1 T) (h2 : cond2) : T - 1.50 = 6.85 := 
sorry

end train_bus_difference_l203_203711


namespace cube_edge_length_close_to_six_l203_203046

theorem cube_edge_length_close_to_six
  (a V S : ℝ)
  (h1 : V = a^3)
  (h2 : S = 6 * a^2)
  (h3 : V = S + 1) : abs (a - 6) < 1 :=
by
  sorry

end cube_edge_length_close_to_six_l203_203046


namespace tangent_line_at_01_l203_203525

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_01 : ∃ (m b : ℝ), (m = 1) ∧ (b = 1) ∧ (∀ x, tangent_line_equation x = m * x + b) :=
by
  sorry

end tangent_line_at_01_l203_203525


namespace calculate_expression_l203_203164

theorem calculate_expression : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := 
by sorry

end calculate_expression_l203_203164


namespace systematic_sampling_draw_l203_203873

theorem systematic_sampling_draw
  (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 8)
  (h2 : 160 ≥ 8 * 20)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 → 
    160 ≥ ((k - 1) * 8 + 1 + 7))
  (h4 : ∀ y : ℕ, y = 1 + (15 * 8) → y = 126)
: x = 6 := 
sorry

end systematic_sampling_draw_l203_203873


namespace complement_of_P_in_U_l203_203655

def universal_set : Set ℝ := Set.univ
def set_P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_in_U (U : Set ℝ) (P : Set ℝ) : Set ℝ := U \ P

theorem complement_of_P_in_U :
  complement_in_U universal_set set_P = { x | -1 < x ∧ x < 6 } :=
by
  sorry

end complement_of_P_in_U_l203_203655


namespace negation_equivalence_l203_203700

theorem negation_equivalence : (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
  sorry

end negation_equivalence_l203_203700


namespace Craig_initial_apples_l203_203385

variable (j : ℕ) (shared : ℕ) (left : ℕ)

theorem Craig_initial_apples (HJ : j = 11) (HS : shared = 7) (HL : left = 13) :
  shared + left = 20 := by
  sorry

end Craig_initial_apples_l203_203385


namespace find_m_l203_203126

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : ∃ s : ℝ, (s = (m + 1 - 4) / (2 - m)) ∧ s = Real.sqrt 5) :
  m = (10 - Real.sqrt 5) / 4 :=
by
  sorry

end find_m_l203_203126


namespace find_k_value_l203_203905

-- Definitions based on conditions
variables {k b x y : ℝ} -- k, b, x, and y are real numbers

-- Conditions given in the problem
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Proposition: Given the conditions, prove that k = 2
theorem find_k_value (h₁ : ∀ x y, y = linear_function k b x → y + 6 = linear_function k b (x + 3)) : k = 2 :=
by
  sorry

end find_k_value_l203_203905


namespace caterpillar_prob_A_l203_203047

-- Define the probabilities involved
def prob_move_to_A_from_1 (x y z : ℚ) : ℚ :=
  (1/3 : ℚ) * 1 + (1/3 : ℚ) * y + (1/3 : ℚ) * z

def prob_move_to_A_from_2 (x y u : ℚ) : ℚ :=
  (1/3 : ℚ) * 0 + (1/3 : ℚ) * x + (1/3 : ℚ) * u

def prob_move_to_A_from_0 (x y : ℚ) : ℚ :=
  (2/3 : ℚ) * x + (1/3 : ℚ) * y

def prob_move_to_A_from_3 (y u : ℚ) : ℚ :=
  (2/3 : ℚ) * y + (1/3 : ℚ) * u

theorem caterpillar_prob_A :
  exists (x y z u : ℚ), 
    x = prob_move_to_A_from_1 x y z ∧
    y = prob_move_to_A_from_2 x y y ∧
    z = prob_move_to_A_from_0 x y ∧
    u = prob_move_to_A_from_3 y y ∧
    u = y ∧
    x = 9/14 :=
sorry

end caterpillar_prob_A_l203_203047


namespace system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l203_203604

theorem system_of_equations_solution_non_negative (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x ≥ 0) (h4 : y ≥ 0) : x = 1 ∧ y = 0 :=
sorry

theorem system_of_equations_solution_positive_sum (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x + y > 0) : x = 1 ∧ y = 0 :=
sorry

end system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l203_203604


namespace find_k_l203_203562

variable (k : ℝ) (t : ℝ) (a : ℝ)

theorem find_k (h1 : t = (5 / 9) * (k - 32) + a * k) (h2 : t = 20) (h3 : a = 3) : k = 10.625 := by
  sorry

end find_k_l203_203562


namespace line_circle_no_intersection_l203_203463

theorem line_circle_no_intersection :
  (∀ (x y : ℝ), 3 * x + 4 * y = 12 ∨ x^2 + y^2 = 4) →
  (∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4) →
  false :=
by
  sorry

end line_circle_no_intersection_l203_203463


namespace sin_neg_1740_eq_sqrt3_div_2_l203_203090

theorem sin_neg_1740_eq_sqrt3_div_2 : Real.sin (-1740 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_1740_eq_sqrt3_div_2_l203_203090


namespace sum_of_arithmetic_sequence_l203_203454

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7/60) :
  S 9 = 63/2 :=
by
  sorry

end sum_of_arithmetic_sequence_l203_203454


namespace pairwise_sums_l203_203473

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end pairwise_sums_l203_203473


namespace david_pushups_difference_l203_203601

-- Definitions based on conditions
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- The number of push-ups David did more than Zachary
def david_more_pushups_than_zachary (D : ℕ) := D - zachary_pushups

-- The theorem we need to prove
theorem david_pushups_difference :
  ∃ D : ℕ, D > zachary_pushups ∧ D + zachary_pushups = total_pushups ∧ david_more_pushups_than_zachary D = 58 :=
by
  -- We leave the proof as an exercise or for further filling.
  sorry

end david_pushups_difference_l203_203601


namespace valid_numbers_count_l203_203474

-- Define a predicate that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that counts how many numbers between 100 and 999 are multiples of 13
def count_multiples_of_13 (start finish : ℕ) : ℕ :=
  (finish - start) / 13 + 1

-- Define a function that checks if a permutation of digits of n is a multiple of 13
-- (actual implementation would require digit manipulation, but we assume its existence here)
def is_permutation_of_digits_multiple_of_13 (n : ℕ) : Prop :=
  ∃ (perm : ℕ), is_three_digit perm ∧ perm % 13 = 0

noncomputable def count_valid_permutations (multiples_of_13 : ℕ) : ℕ :=
  multiples_of_13 * 3 -- Assuming on average

-- Problem statement: Prove that there are 207 valid numbers satisfying the condition
theorem valid_numbers_count : (count_valid_permutations (count_multiples_of_13 104 988)) = 207 := 
by {
  -- Place for proof which is omitted here
  sorry
}

end valid_numbers_count_l203_203474


namespace probability_equal_2s_after_4040_rounds_l203_203468

/-- 
Given three players Diana, Nathan, and Olivia each starting with $2, each player (with at least $1) 
simultaneously gives $1 to one of the other two players randomly every 20 seconds. 
Prove that the probability that after the bell has rung 4040 times, 
each player will have $2$ is $\frac{1}{4}$.
-/
theorem probability_equal_2s_after_4040_rounds 
  (n_rounds : ℕ) (start_money : ℕ) (probability_outcome : ℚ) :
  n_rounds = 4040 →
  start_money = 2 →
  probability_outcome = 1 / 4 :=
by
  sorry

end probability_equal_2s_after_4040_rounds_l203_203468


namespace length_of_AB_l203_203600

theorem length_of_AB (V : ℝ) (r : ℝ) :
  V = 216 * Real.pi →
  r = 3 →
  ∃ (len_AB : ℝ), len_AB = 20 :=
by
  intros hV hr
  have volume_cylinder := V - 36 * Real.pi
  have height_cylinder := volume_cylinder / (Real.pi * r^2)
  exists height_cylinder
  exact sorry

end length_of_AB_l203_203600


namespace number_of_girls_in_group_l203_203968

open Finset

/-- Given that a tech group consists of 6 students, and 3 people are to be selected to visit an exhibition,
    if there are at least 1 girl among the selected, the number of different selection methods is 16,
    then the number of girls in the group is 2. -/
theorem number_of_girls_in_group :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 6 ∧ 
            (Nat.choose 6 3 - Nat.choose (6 - n) 3 = 16)) → n = 2 :=
by
  sorry

end number_of_girls_in_group_l203_203968


namespace multiply_polynomials_l203_203274

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 8 * x^2 + 64) * (x^2 - 8) = x^4 + 16 * x^2 :=
by
  sorry

end multiply_polynomials_l203_203274


namespace num_boys_on_playground_l203_203636

-- Define the conditions using Lean definitions
def num_girls : Nat := 28
def total_children : Nat := 63

-- Define a theorem to prove the number of boys
theorem num_boys_on_playground : total_children - num_girls = 35 :=
by
  -- proof steps would go here
  sorry

end num_boys_on_playground_l203_203636


namespace find_third_number_l203_203407

-- Define the conditions
def equation1_valid : Prop := (5 * 3 = 15) ∧ (5 * 2 = 10) ∧ (2 * 1000 + 3 * 100 + 5 = 1022)
def equation2_valid : Prop := (9 * 2 = 18) ∧ (9 * 4 = 36) ∧ (4 * 1000 + 2 * 100 + 9 = 3652)

-- The theorem to prove
theorem find_third_number (h1 : equation1_valid) (h2 : equation2_valid) : (7 * 2 = 14) ∧ (7 * 5 = 35) ∧ (5 * 1000 + 2 * 100 + 7 = 547) :=
by 
  sorry

end find_third_number_l203_203407


namespace Aiden_sleep_fraction_l203_203993

theorem Aiden_sleep_fraction (minutes_slept : ℕ) (hour_minutes : ℕ) (h : minutes_slept = 15) (k : hour_minutes = 60) :
  (minutes_slept : ℚ) / hour_minutes = 1/4 :=
by
  sorry

end Aiden_sleep_fraction_l203_203993


namespace max_value_of_z_l203_203194

theorem max_value_of_z
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + y ≤ 1)
  (h3 : y ≥ -1) :
  ∃ x y, (y ≥ x) ∧ (x + y ≤ 1) ∧ (y ≥ -1) ∧ (2 * x - y = 1 / 2) := by 
  sorry

end max_value_of_z_l203_203194


namespace prove_a_pow_minus_b_l203_203578

-- Definitions of conditions
variables (x a b : ℝ)

def condition_1 : Prop := x - a > 2
def condition_2 : Prop := 2 * x - b < 0
def solution_set_condition : Prop := -1 < x ∧ x < 1
def derived_a : Prop := a + 2 = -1
def derived_b : Prop := b / 2 = 1

-- The main theorem to prove
theorem prove_a_pow_minus_b (h1 : condition_1 x a) (h2 : condition_2 x b) (h3 : solution_set_condition x) (ha : derived_a a) (hb : derived_b b) : a^(-b) = (1 / 9) :=
by
  sorry

end prove_a_pow_minus_b_l203_203578


namespace latest_departure_time_l203_203980

noncomputable def minutes_in_an_hour : ℕ := 60
noncomputable def departure_time : ℕ := 20 * minutes_in_an_hour -- 8:00 pm in minutes
noncomputable def checkin_time : ℕ := 2 * minutes_in_an_hour -- 2 hours in minutes
noncomputable def drive_time : ℕ := 45 -- 45 minutes
noncomputable def parking_time : ℕ := 15 -- 15 minutes
noncomputable def total_time_needed : ℕ := checkin_time + drive_time + parking_time -- Total time in minutes

theorem latest_departure_time : departure_time - total_time_needed = 17 * minutes_in_an_hour :=
by
  sorry

end latest_departure_time_l203_203980


namespace sum_of_lengths_of_square_sides_l203_203935

theorem sum_of_lengths_of_square_sides (side_length : ℕ) (h1 : side_length = 9) : 
  (4 * side_length) = 36 :=
by
  -- Here we would normally write the proof
  sorry

end sum_of_lengths_of_square_sides_l203_203935


namespace custom_deck_card_selection_l203_203728

theorem custom_deck_card_selection :
  let cards := 60
  let suits := 4
  let cards_per_suit := 15
  let red_suits := 2
  let black_suits := 2
  -- Total number of ways to pick two cards with the second of a different color
  ∃ (ways : ℕ), ways = 60 * 30 ∧ ways = 1800 := by
  sorry

end custom_deck_card_selection_l203_203728


namespace intersection_point_l203_203825

def satisfies_first_line (p : ℝ × ℝ) : Prop :=
  8 * p.1 - 5 * p.2 = 40

def satisfies_second_line (p : ℝ × ℝ) : Prop :=
  6 * p.1 + 2 * p.2 = 14

theorem intersection_point :
  satisfies_first_line (75 / 23, -64 / 23) ∧ satisfies_second_line (75 / 23, -64 / 23) :=
by 
  sorry

end intersection_point_l203_203825


namespace mrs_hilt_walks_240_feet_l203_203037

-- Define the distances and trips as given conditions
def distance_to_fountain : ℕ := 30
def trips_to_fountain : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain
def total_distance_walked (round_trip_distance trips_to_fountain : ℕ) : ℕ :=
  round_trip_distance * trips_to_fountain

-- State the theorem
theorem mrs_hilt_walks_240_feet :
  total_distance_walked round_trip_distance trips_to_fountain = 240 :=
by
  sorry

end mrs_hilt_walks_240_feet_l203_203037


namespace problem_given_conditions_l203_203609

theorem problem_given_conditions (x y z : ℝ) 
  (h : x / 3 = y / (-4) ∧ y / (-4) = z / 7) : (3 * x + y + z) / y = -3 := 
by 
  sorry

end problem_given_conditions_l203_203609


namespace largest_multiple_of_15_less_than_500_is_495_l203_203258

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l203_203258


namespace minimum_box_value_l203_203207

theorem minimum_box_value :
  ∃ (a b : ℤ), a * b = 36 ∧ (a^2 + b^2 = 72 ∧ ∀ (a' b' : ℤ), a' * b' = 36 → a'^2 + b'^2 ≥ 72) :=
by
  sorry

end minimum_box_value_l203_203207


namespace find_vector_b_l203_203949

structure Vec2 where
  x : ℝ
  y : ℝ

def is_parallel (a b : Vec2) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b.x = k * a.x ∧ b.y = k * a.y

def vec_a : Vec2 := { x := 2, y := 3 }
def vec_b : Vec2 := { x := -2, y := -3 }

theorem find_vector_b :
  is_parallel vec_a vec_b := 
sorry

end find_vector_b_l203_203949


namespace range_of_m_l203_203057

theorem range_of_m (m : ℝ) (x : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2023) * x₁ + m + 2023) > ((m - 2023) * x₂ + m + 2023)) → m < 2023 :=
by
  sorry

end range_of_m_l203_203057


namespace find_other_number_l203_203691

theorem find_other_number (w : ℕ) (x : ℕ) 
    (h1 : w = 468)
    (h2 : x * w = 2^4 * 3^3 * 13^3) 
    : x = 2028 :=
by
  sorry

end find_other_number_l203_203691


namespace total_points_scored_l203_203152

-- Definitions based on the conditions
def three_point_shots := 13
def two_point_shots := 20
def free_throws := 5
def missed_free_throws := 2
def points_per_three_point_shot := 3
def points_per_two_point_shot := 2
def points_per_free_throw := 1
def penalty_per_missed_free_throw := 1

-- Main statement proving the total points James scored
theorem total_points_scored :
  three_point_shots * points_per_three_point_shot +
  two_point_shots * points_per_two_point_shot +
  free_throws * points_per_free_throw -
  missed_free_throws * penalty_per_missed_free_throw = 82 :=
by
  sorry

end total_points_scored_l203_203152


namespace no_positive_ints_cube_l203_203450

theorem no_positive_ints_cube (n : ℕ) : ¬ ∃ y : ℕ, 3 * n^2 + 3 * n + 7 = y^3 := 
sorry

end no_positive_ints_cube_l203_203450


namespace no_faces_painted_two_or_three_faces_painted_l203_203281

-- Define the dimensions of the cuboid
def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

-- Define the number of small cubes
def small_cubes_total : ℕ := 60

-- Define the number of small cubes with no faces painted
def small_cubes_no_faces_painted : ℕ := (cuboid_length - 2) * (cuboid_width - 2) * (cuboid_height - 2)

-- Define the number of small cubes with 2 faces painted
def small_cubes_two_faces_painted : ℕ := (cuboid_length - 2) * cuboid_width +
                                          (cuboid_width - 2) * cuboid_length +
                                          (cuboid_height - 2) * cuboid_width

-- Define the number of small cubes with 3 faces painted
def small_cubes_three_faces_painted : ℕ := 8

-- Define the probabilities
def probability_no_faces_painted : ℚ := small_cubes_no_faces_painted / small_cubes_total
def probability_two_or_three_faces_painted : ℚ := (small_cubes_two_faces_painted + small_cubes_three_faces_painted) / small_cubes_total

-- Theorems to prove
theorem no_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                           small_cubes_total = 60 ∧ small_cubes_no_faces_painted = 6) :
  probability_no_faces_painted = 1 / 10 := by
  sorry

theorem two_or_three_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                                    small_cubes_total = 60 ∧ small_cubes_two_faces_painted = 24 ∧
                                    small_cubes_three_faces_painted = 8) :
  probability_two_or_three_faces_painted = 8 / 15 := by
  sorry

end no_faces_painted_two_or_three_faces_painted_l203_203281


namespace sum_n_div_n4_add_16_eq_9_div_320_l203_203105

theorem sum_n_div_n4_add_16_eq_9_div_320 :
  ∑' n : ℕ, n / (n^4 + 16) = 9 / 320 :=
sorry

end sum_n_div_n4_add_16_eq_9_div_320_l203_203105


namespace speed_of_other_train_l203_203012

theorem speed_of_other_train (len1 len2 time : ℝ) (v1 v_other : ℝ) :
  len1 = 200 ∧ len2 = 300 ∧ time = 17.998560115190788 ∧ v1 = 40 →
  v_other = ((len1 + len2) / 1000) / (time / 3600) - v1 :=
by
  intros
  sorry

end speed_of_other_train_l203_203012


namespace function_relation_l203_203064

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + c

theorem function_relation (c : ℝ) :
  f 1 c > f 0 c ∧ f 0 c > f (-2) c := by
  sorry

end function_relation_l203_203064


namespace consecutive_even_integers_sum_l203_203287

theorem consecutive_even_integers_sum :
  ∀ (y : Int), (y = 2 * (y + 2)) → y + (y + 2) = -6 :=
by
  intro y
  intro h
  sorry

end consecutive_even_integers_sum_l203_203287


namespace percent_students_own_only_cats_l203_203482

theorem percent_students_own_only_cats (total_students : ℕ) (students_owning_cats : ℕ) (students_owning_dogs : ℕ) (students_owning_both : ℕ) (h_total : total_students = 500) (h_cats : students_owning_cats = 80) (h_dogs : students_owning_dogs = 150) (h_both : students_owning_both = 40) : 
  (students_owning_cats - students_owning_both) * 100 / total_students = 8 := 
by
  sorry

end percent_students_own_only_cats_l203_203482


namespace inequality_pgcd_l203_203535

theorem inequality_pgcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) : 
  (a + 1) / (b + 1) ≤ Nat.gcd a b + 1 := 
sorry

end inequality_pgcd_l203_203535


namespace roots_in_interval_l203_203340

theorem roots_in_interval (f : ℝ → ℝ)
  (h : ∀ x, f x = 4 * x ^ 2 - (3 * m + 1) * x - m - 2) :
  (forall (x1 x2 : ℝ), (f x1 = 0 ∧ f x2 = 0) → -1 < x1 ∧ x1 < 2 ∧ -1 < x2 ∧ x2 < 2) ↔ -1 < m ∧ m < 12 / 7 :=
sorry

end roots_in_interval_l203_203340


namespace find_sticker_price_l203_203941

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l203_203941


namespace volume_frustum_fraction_l203_203309

-- Define the base edge and initial altitude of the pyramid.
def base_edge := 32 -- in inches
def altitude_original := 1 -- in feet

-- Define the fractional part representing the altitude of the smaller pyramid.
def altitude_fraction := 1/4

-- Define the volume of the original pyramid being V.
noncomputable def volume_original : ℝ := (1/3) * (base_edge ^ 2) * altitude_original

-- Define the volume of the smaller pyramid being removed.
noncomputable def volume_smaller : ℝ := (1/3) * ((altitude_fraction * base_edge) ^ 2) * (altitude_fraction * altitude_original)

-- We now state the proof
theorem volume_frustum_fraction : 
  (volume_original - volume_smaller) / volume_original = 63/64 :=
by
  sorry

end volume_frustum_fraction_l203_203309


namespace exponentiation_evaluation_l203_203188

theorem exponentiation_evaluation :
  (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end exponentiation_evaluation_l203_203188


namespace problem_l203_203249

theorem problem (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
sorry

end problem_l203_203249


namespace container_capacity_l203_203831

theorem container_capacity (C : ℝ) (h1 : C > 0) (h2 : 0.40 * C + 14 = 0.75 * C) : C = 40 := 
by 
  -- Would contain the proof here
  sorry

end container_capacity_l203_203831


namespace grandpa_max_movies_l203_203075

-- Definition of the conditions
def movie_duration : ℕ := 90

def tuesday_total_minutes : ℕ := 4 * 60 + 30

def tuesday_movies_watched : ℕ := tuesday_total_minutes / movie_duration

def wednesday_movies_watched : ℕ := 2 * tuesday_movies_watched

def total_movies_watched : ℕ := tuesday_movies_watched + wednesday_movies_watched

theorem grandpa_max_movies : total_movies_watched = 9 := by
  sorry

end grandpa_max_movies_l203_203075


namespace set_inter_complement_eq_l203_203612

-- Given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 < 1}
def B : Set ℝ := {x | x^2 - 2 * x > 0}

-- Question translated to proof problem statement
theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_inter_complement_eq_l203_203612


namespace smallest_pos_int_b_for_factorization_l203_203755

theorem smallest_pos_int_b_for_factorization :
  ∃ b : ℤ, 0 < b ∧ ∀ (x : ℤ), ∃ r s : ℤ, r * s = 4032 ∧ r + s = b ∧ x^2 + b * x + 4032 = (x + r) * (x + s) ∧
    (∀ b' : ℤ, 0 < b' → b' ≠ b → ∃ rr ss : ℤ, rr * ss = 4032 ∧ rr + ss = b' ∧ x^2 + b' * x + 4032 = (x + rr) * (x + ss) → b < b') := 
sorry

end smallest_pos_int_b_for_factorization_l203_203755


namespace polygon_sides_l203_203523

theorem polygon_sides
  (n : ℕ)
  (h1 : 180 * (n - 2) - (2 * (2790 / (n - 1)) - 20) = 2790) :
  n = 18 := sorry

end polygon_sides_l203_203523


namespace cost_of_item_D_is_30_usd_l203_203136

noncomputable def cost_of_item_D_in_usd (total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate : ℝ) : ℝ :=
  let total_spent_with_fee := total_spent * (1 + service_fee_rate)
  let item_D_cost_FC := total_spent_with_fee - items_ABC_spent
  item_D_cost_FC * exchange_rate

theorem cost_of_item_D_is_30_usd
  (total_spent : ℝ)
  (items_ABC_spent : ℝ)
  (tax_paid : ℝ)
  (service_fee_rate : ℝ)
  (exchange_rate : ℝ)
  (h_total_spent : total_spent = 500)
  (h_items_ABC_spent : items_ABC_spent = 450)
  (h_tax_paid : tax_paid = 60)
  (h_service_fee_rate : service_fee_rate = 0.02)
  (h_exchange_rate : exchange_rate = 0.5) :
  cost_of_item_D_in_usd total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate = 30 :=
by
  have h1 : total_spent * (1 + service_fee_rate) = 500 * 1.02 := sorry
  have h2 : 500 * 1.02 - 450 = 60 := sorry
  have h3 : 60 * 0.5 = 30 := sorry
  sorry

end cost_of_item_D_is_30_usd_l203_203136


namespace solve_for_b_l203_203752

theorem solve_for_b (b : ℝ) : 
  let slope1 := -(3 / 4 : ℝ)
  let slope2 := -(b / 6 : ℝ)
  slope1 * slope2 = -1 → b = -8 :=
by
  intro h
  sorry

end solve_for_b_l203_203752


namespace actual_estate_area_l203_203917

theorem actual_estate_area (map_scale : ℝ) (length_inches : ℝ) (width_inches : ℝ) 
  (actual_length : ℝ) (actual_width : ℝ) (area_square_miles : ℝ) 
  (h_scale : map_scale = 300)
  (h_length : length_inches = 4)
  (h_width : width_inches = 3)
  (h_actual_length : actual_length = length_inches * map_scale)
  (h_actual_width : actual_width = width_inches * map_scale)
  (h_area : area_square_miles = actual_length * actual_width) :
  area_square_miles = 1080000 :=
sorry

end actual_estate_area_l203_203917


namespace maximum_value_of_expression_l203_203499

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l203_203499


namespace total_pears_l203_203417

def jason_pears : Nat := 46
def keith_pears : Nat := 47
def mike_pears : Nat := 12

theorem total_pears : jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_l203_203417


namespace PRINT_3_3_2_l203_203452

def PRINT (a b : Nat) : Nat × Nat := (a, b)

theorem PRINT_3_3_2 :
  PRINT 3 (3 + 2) = (3, 5) :=
by
  sorry

end PRINT_3_3_2_l203_203452


namespace gcd_of_105_1001_2436_l203_203793

noncomputable def gcd_problem : ℕ :=
  Nat.gcd (Nat.gcd 105 1001) 2436

theorem gcd_of_105_1001_2436 : gcd_problem = 7 :=
by {
  sorry
}

end gcd_of_105_1001_2436_l203_203793


namespace number_of_possible_flags_l203_203124

def colors : List String := ["purple", "gold"]

noncomputable def num_choices_per_stripe (colors : List String) : Nat := 
  colors.length

theorem number_of_possible_flags :
  (num_choices_per_stripe colors) ^ 3 = 8 := 
by
  -- Proof
  sorry

end number_of_possible_flags_l203_203124


namespace max_quarters_l203_203260

theorem max_quarters (q : ℕ) (h1 : q + q + q / 2 = 20): q ≤ 11 :=
by
  sorry

end max_quarters_l203_203260


namespace rope_subdivision_length_l203_203166

theorem rope_subdivision_length 
  (initial_length : ℕ) 
  (num_parts : ℕ) 
  (num_subdivided_parts : ℕ) 
  (final_subdivision_factor : ℕ) 
  (initial_length_eq : initial_length = 200) 
  (num_parts_eq : num_parts = 4) 
  (num_subdivided_parts_eq : num_subdivided_parts = num_parts / 2) 
  (final_subdivision_factor_eq : final_subdivision_factor = 2) :
  initial_length / num_parts / final_subdivision_factor = 25 := 
by 
  sorry

end rope_subdivision_length_l203_203166


namespace teachers_on_field_trip_l203_203029

-- Definitions for conditions in the problem
def number_of_students := 12
def cost_per_student_ticket := 1
def cost_per_adult_ticket := 3
def total_cost_of_tickets := 24

-- Main statement
theorem teachers_on_field_trip :
  ∃ (T : ℕ), number_of_students * cost_per_student_ticket + T * cost_per_adult_ticket = total_cost_of_tickets ∧ T = 4 :=
by
  use 4
  sorry

end teachers_on_field_trip_l203_203029


namespace find_number_l203_203503

def number_of_faces : ℕ := 6

noncomputable def probability (n : ℕ) : ℚ :=
  (number_of_faces - n : ℕ) / number_of_faces

theorem find_number (n : ℕ) (h: n < number_of_faces) :
  probability n = 1 / 3 → n = 4 :=
by
  -- proof goes here
  sorry

end find_number_l203_203503


namespace value_of_m_l203_203301

-- Define the function given m
def f (x m : ℝ) : ℝ := x^2 - 2 * (abs x) + 2 - m

-- State the theorem to be proved
theorem value_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, f x1 m = 0 ∧ f x2 m = 0 ∧ f x3 m = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1) →
  m = 2 :=
by
  sorry

end value_of_m_l203_203301


namespace symmetric_circle_eq_l203_203559

/-- Define the equation of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Define the equation of the line l -/
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- 
The symmetric circle to C with respect to line l 
has the equation (x - 1)^2 + (y - 1)^2 = 4.
-/
theorem symmetric_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, circle_equation x y) → 
  (∃ x y : ℝ, line_equation x y) →
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l203_203559


namespace ab2_plus_bc2_plus_ca2_le_27_div_8_l203_203829

theorem ab2_plus_bc2_plus_ca2_le_27_div_8 (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end ab2_plus_bc2_plus_ca2_le_27_div_8_l203_203829


namespace average_speed_of_trip_l203_203098

theorem average_speed_of_trip :
  let distance_local := 60
  let speed_local := 20
  let distance_highway := 120
  let speed_highway := 60
  let total_distance := distance_local + distance_highway
  let time_local := distance_local / speed_local
  let time_highway := distance_highway / speed_highway
  let total_time := time_local + time_highway
  let average_speed := total_distance / total_time
  average_speed = 36 := 
by 
  sorry

end average_speed_of_trip_l203_203098


namespace triangle_right_if_condition_l203_203413

variables (a b c : ℝ) (A B C : ℝ)
-- Condition: Given 1 + cos A = (b + c) / c
axiom h1 : 1 + Real.cos A = (b + c) / c 

-- To prove: a^2 + b^2 = c^2
theorem triangle_right_if_condition (h1 : 1 + Real.cos A = (b + c) / c) : a^2 + b^2 = c^2 :=
  sorry

end triangle_right_if_condition_l203_203413


namespace common_number_in_lists_l203_203586

theorem common_number_in_lists (nums : List ℚ) (h_len : nums.length = 9)
  (h_first_five_avg : (nums.take 5).sum / 5 = 7)
  (h_last_five_avg : (nums.drop 4).sum / 5 = 9)
  (h_total_avg : nums.sum / 9 = 73/9) :
  ∃ x, x ∈ nums.take 5 ∧ x ∈ nums.drop 4 ∧ x = 7 := 
sorry

end common_number_in_lists_l203_203586


namespace new_volume_of_cylinder_l203_203643

theorem new_volume_of_cylinder (r h : ℝ) (π : ℝ := Real.pi) (V : ℝ := π * r^2 * h) (hV : V = 15) :
  let r_new := 3 * r
  let h_new := 4 * h
  let V_new := π * (r_new)^2 * h_new
  V_new = 540 :=
by
  sorry

end new_volume_of_cylinder_l203_203643


namespace relationship_ab_l203_203565

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 1) = -f x)
variable (a b : ℝ)
variable (h_ex : ∃ x : ℝ, f (a + x) = f (b - x))

-- State the conclusion we need to prove
theorem relationship_ab : ∃ k : ℕ, k > 0 ∧ (a + b) = 2 * k + 1 :=
by
  sorry

end relationship_ab_l203_203565


namespace streamers_for_price_of_confetti_l203_203875

variable (p q : ℝ) (x y : ℝ)

theorem streamers_for_price_of_confetti (h1 : x * (1 + p / 100) = y) 
                                   (h2 : y * (1 - q / 100) = x)
                                   (h3 : |p - q| = 90) :
  10 * (y * 0.4) = 4 * y :=
sorry

end streamers_for_price_of_confetti_l203_203875


namespace total_cost_of_fruit_l203_203181

theorem total_cost_of_fruit (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 58) 
  (h2 : 3 * x + 2 * y = 72) : 
  3 * x + 3 * y = 78 := 
by
  sorry

end total_cost_of_fruit_l203_203181


namespace radius_intersection_xy_plane_l203_203251

noncomputable def center_sphere : ℝ × ℝ × ℝ := (3, 3, 3)

def radius_xz_circle : ℝ := 2

def xz_center : ℝ × ℝ × ℝ := (3, 0, 3)

def xy_center : ℝ × ℝ × ℝ := (3, 3, 0)

theorem radius_intersection_xy_plane (r : ℝ) (s : ℝ) 
(h_center : center_sphere = (3, 3, 3)) 
(h_xz : xz_center = (3, 0, 3))
(h_r_xz : radius_xz_circle = 2)
(h_xy : xy_center = (3, 3, 0)):
s = 3 := 
sorry

end radius_intersection_xy_plane_l203_203251


namespace prime_square_minus_one_divisible_by_24_l203_203311

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : 5 ≤ p) : 24 ∣ (p^2 - 1) := 
by 
sorry

end prime_square_minus_one_divisible_by_24_l203_203311


namespace will_money_left_l203_203032

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end will_money_left_l203_203032


namespace work_days_B_l203_203028

theorem work_days_B (A B: ℕ) (work_per_day_B: ℕ) (total_days : ℕ) (total_units : ℕ) :
  (A = 2 * B) → (work_per_day_B = 1) → (total_days = 36) → (B = 1) → (total_units = total_days * (A + B)) → 
  total_units / work_per_day_B = 108 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_days_B_l203_203028


namespace find_digit_B_l203_203723

theorem find_digit_B (A B : ℕ) (h1 : A3B = 100 * A + 30 + B)
  (h2 : 0 ≤ A ∧ A ≤ 9)
  (h3 : 0 ≤ B ∧ B ≤ 9)
  (h4 : A3B - 41 = 591) : 
  B = 2 := 
by sorry

end find_digit_B_l203_203723


namespace interest_rate_correct_l203_203983

namespace InterestProblem

variable (P : ℤ) (SI : ℤ) (T : ℤ)

def rate_of_interest (P : ℤ) (SI : ℤ) (T : ℤ) : ℚ :=
  (SI * 100) / (P * T)

theorem interest_rate_correct :
  rate_of_interest 400 140 2 = 17.5 := by
  sorry

end InterestProblem

end interest_rate_correct_l203_203983


namespace good_eggs_collected_l203_203031

/-- 
Uncle Ben has 550 chickens on his farm, consisting of 49 roosters and the rest being hens. 
Out of these hens, there are three types:
1. Type A: 25 hens do not lay eggs at all.
2. Type B: 155 hens lay 2 eggs per day.
3. Type C: The remaining hens lay 4 eggs every three days.

Moreover, Uncle Ben found that 3% of the eggs laid by Type B and Type C hens go bad before being collected. 
Prove that the total number of good eggs collected by Uncle Ben after one day is 716.
-/
theorem good_eggs_collected 
    (total_chickens : ℕ) (roosters : ℕ) (typeA_hens : ℕ) (typeB_hens : ℕ) 
    (typeB_eggs_per_day : ℕ) (typeC_eggs_per_3days : ℕ) (percent_bad_eggs : ℚ) :
  total_chickens = 550 →
  roosters = 49 →
  typeA_hens = 25 →
  typeB_hens = 155 →
  typeB_eggs_per_day = 2 →
  typeC_eggs_per_3days = 4 →
  percent_bad_eggs = 0.03 →
  (total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day) - 
  round (percent_bad_eggs * ((total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day))) = 716 :=
by
  intros
  sorry

end good_eggs_collected_l203_203031


namespace fraction_of_single_female_students_l203_203173

variables (total_students : ℕ) (male_students married_students married_male_students female_students single_female_students : ℕ)

-- Given conditions
def condition1 : male_students = (7 * total_students) / 10 := sorry
def condition2 : married_students = (3 * total_students) / 10 := sorry
def condition3 : married_male_students = male_students / 7 := sorry

-- Derived conditions
def condition4 : female_students = total_students - male_students := sorry
def condition5 : married_female_students = married_students - married_male_students := sorry
def condition6 : single_female_students = female_students - married_female_students := sorry

-- The proof goal
theorem fraction_of_single_female_students 
  (h1 : male_students = (7 * total_students) / 10)
  (h2 : married_students = (3 * total_students) / 10)
  (h3 : married_male_students = male_students / 7)
  (h4 : female_students = total_students - male_students)
  (h5 : married_female_students = married_students - married_male_students)
  (h6 : single_female_students = female_students - married_female_students) :
  (single_female_students : ℚ) / (female_students : ℚ) = 1 / 3 :=
sorry

end fraction_of_single_female_students_l203_203173


namespace find_base_and_digit_sum_l203_203191

theorem find_base_and_digit_sum (n d : ℕ) (h1 : 4 * n^2 + 5 * n + d = 392) (h2 : 4 * n^2 + 5 * n + 7 = 740 + 7 * d) : n + d = 12 :=
by
  sorry

end find_base_and_digit_sum_l203_203191


namespace counseling_rooms_l203_203661

theorem counseling_rooms (n : ℕ) (x : ℕ)
  (h1 : n = 20 * x + 32)
  (h2 : n = 24 * (x - 1)) : x = 14 :=
by
  sorry

end counseling_rooms_l203_203661


namespace smallest_n_not_divisible_by_10_l203_203435

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l203_203435


namespace salad_dressing_oil_percentage_l203_203727

theorem salad_dressing_oil_percentage 
  (vinegar_P : ℝ) (vinegar_Q : ℝ) (oil_Q : ℝ)
  (new_vinegar : ℝ) (proportion_P : ℝ) :
  vinegar_P = 0.30 ∧ vinegar_Q = 0.10 ∧ oil_Q = 0.90 ∧ new_vinegar = 0.12 ∧ proportion_P = 0.10 →
  (1 - vinegar_P) = 0.70 :=
by
  intro h
  sorry

end salad_dressing_oil_percentage_l203_203727


namespace length_DE_l203_203111

open Classical

noncomputable def triangle_base_length (ABC_base : ℝ) : ℝ :=
15

noncomputable def is_parallel (DE BC : ℝ) : Prop :=
DE = BC

noncomputable def area_ratio (triangle_small triangle_large : ℝ) : ℝ :=
0.25

theorem length_DE 
  (ABC_base : ℝ)
  (DE : ℝ)
  (BC : ℝ)
  (triangle_small : ℝ)
  (triangle_large : ℝ)
  (h_base : triangle_base_length ABC_base = 15)
  (h_parallel : is_parallel DE BC)
  (h_area : area_ratio triangle_small triangle_large = 0.25)
  (h_similar : true):
  DE = 7.5 :=
by
  sorry

end length_DE_l203_203111


namespace greatest_possible_perimeter_l203_203682

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l203_203682


namespace quadratic_no_ten_powers_of_2_values_l203_203342

theorem quadratic_no_ten_powers_of_2_values 
  (a b : ℝ) :
  ¬ ∃ (j : ℤ), ∀ k : ℤ, j ≤ k ∧ k < j + 10 → ∃ n : ℕ, (k^2 + a * k + b) = 2 ^ n :=
by sorry

end quadratic_no_ten_powers_of_2_values_l203_203342


namespace derivative_of_m_l203_203528

noncomputable def m (x : ℝ) : ℝ := (2 : ℝ)^x / (1 + x)

theorem derivative_of_m (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by
  sorry

end derivative_of_m_l203_203528


namespace football_team_birthday_collision_moscow_birthday_collision_l203_203209

theorem football_team_birthday_collision (n : ℕ) (k : ℕ) (h1 : n ≥ 11) (h2 : k = 7) : 
  ∃ (d : ℕ) (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ n ∧ p2 ≤ n ∧ d ≤ k :=
by sorry

theorem moscow_birthday_collision (population : ℕ) (days : ℕ) (h1 : population > 10000000) (h2 : days = 366) :
  ∃ (day : ℕ) (count : ℕ), count ≥ 10000 ∧ count ≤ population / days :=
by sorry

end football_team_birthday_collision_moscow_birthday_collision_l203_203209


namespace ball_bouncing_height_l203_203276

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l203_203276


namespace range_of_a_iff_condition_l203_203494

theorem range_of_a_iff_condition (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3 * a) ↔ (a ≥ -2 ∧ a ≤ 5) :=
by
  sorry

end range_of_a_iff_condition_l203_203494


namespace find_p_q_l203_203510

noncomputable def roots_of_polynomial (a b c : ℝ) :=
  a^3 - 2018 * a + 2018 = 0 ∧ b^3 - 2018 * b + 2018 = 0 ∧ c^3 - 2018 * c + 2018 = 0

theorem find_p_q (a b c : ℝ) (p q : ℕ) 
  (h1 : roots_of_polynomial a b c)
  (h2 : 0 < p ∧ p ≤ q) 
  (h3 : (a^(p+q) + b^(p+q) + c^(p+q))/(p+q) = (a^p + b^p + c^p)/p * (a^q + b^q + c^q)/q) : 
  p^2 + q^2 = 20 := 
sorry

end find_p_q_l203_203510


namespace sphere_weight_dependence_l203_203921

theorem sphere_weight_dependence 
  (r1 r2 SA1 SA2 weight1 weight2 : ℝ) 
  (h1 : r1 = 0.15) 
  (h2 : r2 = 2 * r1) 
  (h3 : SA1 = 4 * Real.pi * r1^2) 
  (h4 : SA2 = 4 * Real.pi * r2^2) 
  (h5 : weight1 = 8) 
  (h6 : weight1 / SA1 = weight2 / SA2) : 
  weight2 = 32 :=
by
  sorry

end sphere_weight_dependence_l203_203921


namespace simplify_expr_l203_203852

noncomputable def expr : ℝ := (18 * 10^10) / (6 * 10^4) * 2

theorem simplify_expr : expr = 6 * 10^6 := sorry

end simplify_expr_l203_203852


namespace range_of_a_for_three_zeros_l203_203011

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l203_203011


namespace smallest_S_value_l203_203272

def num_list := {x : ℕ // 1 ≤ x ∧ x ≤ 9}

def S (a b c : num_list) (d e f : num_list) (g h i : num_list) : ℕ :=
  a.val * b.val * c.val + d.val * e.val * f.val + g.val * h.val * i.val

theorem smallest_S_value :
  ∃ a b c d e f g h i : num_list,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  S a b c d e f g h i = 214 :=
sorry

end smallest_S_value_l203_203272


namespace chocolate_chip_difference_l203_203476

noncomputable def V_v : ℕ := 20 -- Viviana's vanilla chips
noncomputable def S_c : ℕ := 25 -- Susana's chocolate chips
noncomputable def S_v : ℕ := 3 * V_v / 4 -- Susana's vanilla chips

theorem chocolate_chip_difference (V_c : ℕ) (h1 : V_c + V_v + S_c + S_v = 90) :
  V_c - S_c = 5 := by sorry

end chocolate_chip_difference_l203_203476


namespace flat_fee_l203_203145

theorem flat_fee (f n : ℝ) (h1 : f + 4 * n = 320) (h2 : f + 7 * n = 530) : f = 40 := by
  -- Proof goes here
  sorry

end flat_fee_l203_203145


namespace overall_average_speed_l203_203849

-- Define the conditions for Mark's travel
def time_cycling : ℝ := 1
def speed_cycling : ℝ := 20
def time_walking : ℝ := 2
def speed_walking : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ :=
  (time_cycling * speed_cycling) + (time_walking * speed_walking)

def total_time : ℝ :=
  time_cycling + time_walking

-- Define the proved statement for the average speed
theorem overall_average_speed : total_distance / total_time = 8.67 :=
by
  sorry

end overall_average_speed_l203_203849


namespace required_large_loans_l203_203393

-- We start by introducing the concepts of the number of small, medium, and large loans
def small_loans : Type := ℕ
def medium_loans : Type := ℕ
def large_loans : Type := ℕ

-- Definition of the conditions as two scenarios
def Scenario1 (m s b : ℕ) : Prop := (m = 9 ∧ s = 6 ∧ b = 1)
def Scenario2 (m s b : ℕ) : Prop := (m = 3 ∧ s = 2 ∧ b = 3)

-- Definition of the problem
theorem required_large_loans (m s b : ℕ) (H1 : Scenario1 m s b) (H2 : Scenario2 m s b) :
  b = 4 :=
sorry

end required_large_loans_l203_203393


namespace total_tickets_correct_l203_203375

-- Let's define the conditions given in the problem
def student_tickets (adult_tickets : ℕ) := 2 * adult_tickets
def adult_tickets := 122
def total_tickets := adult_tickets + student_tickets adult_tickets

-- We now state the theorem to be proved
theorem total_tickets_correct : total_tickets = 366 :=
by 
  sorry

end total_tickets_correct_l203_203375


namespace average_difference_l203_203376

theorem average_difference : 
  (500 + 1000) / 2 - (100 + 500) / 2 = 450 := 
by
  sorry

end average_difference_l203_203376


namespace problem_1_problem_2_l203_203418

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end problem_1_problem_2_l203_203418


namespace largest_whole_number_l203_203770

theorem largest_whole_number (x : ℕ) (h1 : 9 * x < 150) : x ≤ 16 :=
by sorry

end largest_whole_number_l203_203770


namespace linear_function_no_third_quadrant_l203_203364

theorem linear_function_no_third_quadrant (m : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ -2 * x + 1 - m) : 
  m ≤ 1 :=
by
  sorry

end linear_function_no_third_quadrant_l203_203364


namespace jerseys_sold_l203_203322

theorem jerseys_sold (unit_price_jersey : ℕ) (total_revenue_jersey : ℕ) (n : ℕ) 
  (h_unit_price : unit_price_jersey = 165) 
  (h_total_revenue : total_revenue_jersey = 25740) 
  (h_eq : n * unit_price_jersey = total_revenue_jersey) : 
  n = 156 :=
by
  rw [h_unit_price, h_total_revenue] at h_eq
  sorry

end jerseys_sold_l203_203322


namespace angle_C_is_150_degrees_l203_203676

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l203_203676


namespace find_smallest_value_l203_203426

noncomputable def smallest_value (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2

theorem find_smallest_value (a b c d : ℝ) (h1: a + b = 18)
  (h2: ab + c + d = 85) (h3: ad + bc = 180) (h4: cd = 104) :
  smallest_value a b c d = 484 :=
sorry

end find_smallest_value_l203_203426


namespace part1_l203_203680

noncomputable def P : Set ℝ := {x | (1 / 2) ≤ x ∧ x ≤ 1}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}
def U : Set ℝ := Set.univ
noncomputable def complement_P : Set ℝ := {x | x < (1 / 2)} ∪ {x | x > 1}

theorem part1 (a : ℝ) (h : a = 1) : 
  (complement_P ∩ Q a) = {x | 1 < x ∧ x ≤ 2} :=
sorry

end part1_l203_203680


namespace find_z_plus_one_over_y_l203_203165

theorem find_z_plus_one_over_y 
  (x y z : ℝ) 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1/z = 4)
  (h6 : y + 1/x = 20) :
  z + 1/y = 26 / 79 :=
by
  sorry

end find_z_plus_one_over_y_l203_203165


namespace solution_correctness_l203_203139

theorem solution_correctness:
  ∀ (x1 : ℝ) (θ : ℝ), (θ = (5 * Real.pi / 13)) →
  (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) →
  ∃ (x2 : ℝ), (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ 
  (Real.sin x1 - 2 * Real.sin (x2 + θ) = -1) :=
by 
  intros x1 θ hθ hx1;
  sorry

end solution_correctness_l203_203139


namespace sum_of_reciprocals_of_squares_l203_203150

open Nat

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h_prod : a * b = 5) : 
  (1 : ℚ) / (a * a) + (1 : ℚ) / (b * b) = 26 / 25 :=
by
  -- proof steps skipping with sorry
  sorry

end sum_of_reciprocals_of_squares_l203_203150


namespace gcd_sum_of_cubes_l203_203228

-- Define the problem conditions
variables (n : ℕ) (h_pos : n > 27)

-- Define the goal to prove
theorem gcd_sum_of_cubes (h : n > 27) : 
  gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end gcd_sum_of_cubes_l203_203228


namespace find_value_of_x_y_l203_203408

theorem find_value_of_x_y (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : |y| + x - y = 12) : x + y = 18 / 5 :=
by
  sorry

end find_value_of_x_y_l203_203408


namespace sequence_pattern_l203_203866

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern_l203_203866


namespace scientific_notation_of_463_4_billion_l203_203121

theorem scientific_notation_of_463_4_billion :
  (463.4 * 10^9) = (4.634 * 10^11) := by
  sorry

end scientific_notation_of_463_4_billion_l203_203121


namespace race_winner_l203_203625

theorem race_winner
  (faster : String → String → Prop)
  (Minyoung Yoongi Jimin Yuna : String)
  (cond1 : faster Minyoung Yoongi)
  (cond2 : faster Yoongi Jimin)
  (cond3 : faster Yuna Jimin)
  (cond4 : faster Yuna Minyoung) :
  ∀ s, s ≠ Yuna → faster Yuna s :=
by
  sorry

end race_winner_l203_203625


namespace find_values_l203_203470

theorem find_values (a b c : ℝ)
  (h1 : 0.005 * a = 0.8)
  (h2 : 0.0025 * b = 0.6)
  (h3 : c = 0.5 * a - 0.1 * b) :
  a = 160 ∧ b = 240 ∧ c = 56 :=
by sorry

end find_values_l203_203470


namespace camels_horses_oxen_elephants_l203_203170

theorem camels_horses_oxen_elephants :
  ∀ (C H O E : ℝ),
  10 * C = 24 * H →
  H = 4 * O →
  6 * O = 4 * E →
  10 * E = 170000 →
  C = 4184.615384615385 →
  (4 * O) / H = 1 :=
by
  intros C H O E h1 h2 h3 h4 h5
  sorry

end camels_horses_oxen_elephants_l203_203170


namespace compute_expression_l203_203654

theorem compute_expression (w : ℂ) (hw : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) (hwp : w^11 = 1) :
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -2) :=
sorry

end compute_expression_l203_203654


namespace solve_quadratic_identity_l203_203743

theorem solve_quadratic_identity (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) :
  (14 * y - 5) ^ 2 = 333 :=
by sorry

end solve_quadratic_identity_l203_203743


namespace example_theorem_l203_203001

noncomputable def P (A : Set ℕ) : ℝ := sorry

variable (A1 A2 A3 : Set ℕ)

axiom prob_A1 : P A1 = 0.2
axiom prob_A2 : P A2 = 0.3
axiom prob_A3 : P A3 = 0.5

theorem example_theorem : P (A1 ∪ A2) ≤ 0.5 := 
by {
  sorry
}

end example_theorem_l203_203001


namespace cloak_change_in_silver_l203_203570

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l203_203570


namespace total_sessions_l203_203767

theorem total_sessions (p1 p2 p3 p4 : ℕ) 
(h1 : p1 = 6) 
(h2 : p2 = p1 + 5) 
(h3 : p3 = 8) 
(h4 : p4 = 8) : 
p1 + p2 + p3 + p4 = 33 := 
by
  sorry

end total_sessions_l203_203767


namespace team_total_points_l203_203967

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l203_203967


namespace major_axis_length_of_intersecting_ellipse_l203_203924

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l203_203924


namespace solve_for_y_l203_203960

theorem solve_for_y (y : ℝ) (h : -3 * y - 9 = 6 * y + 3) : y = -4 / 3 :=
by
  sorry

end solve_for_y_l203_203960


namespace base_conversion_l203_203377

theorem base_conversion (b : ℕ) (h_pos : b > 0) :
  (1 * 6 ^ 2 + 2 * 6 ^ 1 + 5 * 6 ^ 0 = 2 * b ^ 2 + 2 * b + 1) → b = 4 :=
by
  sorry

end base_conversion_l203_203377


namespace range_of_a_l203_203291

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
sorry

end range_of_a_l203_203291


namespace track_length_l203_203840

theorem track_length (x : ℝ) (hb hs : ℝ) (h_opposite : hs = x / 2 - 120) (h_first_meet : hb = 120) (h_second_meet : hs + 180 = x / 2 + 60) : x = 600 := 
by
  sorry

end track_length_l203_203840


namespace x_intercept_of_line_l203_203833

theorem x_intercept_of_line : ∃ x : ℝ, ∃ y : ℝ, 4 * x + 7 * y = 28 ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_of_line_l203_203833


namespace ratio_sqrt_2_l203_203445

theorem ratio_sqrt_2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6 * a * b) :
  (a + b) / (a - b) = Real.sqrt 2 :=
by
  sorry

end ratio_sqrt_2_l203_203445


namespace hyperbolas_same_asymptotes_l203_203649

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (y^2 / 25 - x^2 / M = 1)) → M = 225 / 16 :=
by
  sorry

end hyperbolas_same_asymptotes_l203_203649


namespace common_difference_arithmetic_sequence_l203_203286

noncomputable def a_n (n : ℕ) : ℤ := 5 - 4 * n

theorem common_difference_arithmetic_sequence :
  ∀ n ≥ 1, a_n n - a_n (n - 1) = -4 :=
by
  intros n hn
  unfold a_n
  sorry

end common_difference_arithmetic_sequence_l203_203286


namespace kopecks_problem_l203_203944

theorem kopecks_problem (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b :=
sorry

end kopecks_problem_l203_203944


namespace remainder_of_n_div_1000_l203_203087

noncomputable def setS : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

def n : ℕ :=
  let T := {x | 4 ≤ x ∧ x ≤ 15}
  (3^12 - 2^12) / 2

theorem remainder_of_n_div_1000 : (n % 1000) = 672 := 
  by sorry

end remainder_of_n_div_1000_l203_203087


namespace num_two_digit_numbers_with_digit_sum_10_l203_203771

theorem num_two_digit_numbers_with_digit_sum_10 : 
  ∃ n, n = 9 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 10 → ∃ m, 10 * a + b = m :=
sorry

end num_two_digit_numbers_with_digit_sum_10_l203_203771


namespace coordinates_of_point_l203_203569

theorem coordinates_of_point (x y : ℝ) (hx : x < 0) (hy : y > 0) (dx : |x| = 3) (dy : |y| = 2) :
  (x, y) = (-3, 2) := 
sorry

end coordinates_of_point_l203_203569


namespace complementary_angle_difference_l203_203779

theorem complementary_angle_difference (a b : ℝ) (h1 : a = 4 * b) (h2 : a + b = 90) : (a - b) = 54 :=
by
  -- Proof is intentionally omitted
  sorry

end complementary_angle_difference_l203_203779


namespace laura_annual_income_l203_203035

theorem laura_annual_income (I T : ℝ) (q : ℝ)
  (h1 : I > 50000) 
  (h2 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000))
  (h3 : T = 0.01 * (q + 0.5) * I) : I = 56000 := 
by sorry

end laura_annual_income_l203_203035


namespace furniture_cost_final_price_l203_203906

theorem furniture_cost_final_price 
  (table_cost : ℤ := 140)
  (chair_ratio : ℚ := 1/7)
  (sofa_ratio : ℕ := 2)
  (discount : ℚ := 0.10)
  (tax : ℚ := 0.07)
  (exchange_rate : ℚ := 1.2) :
  let chair_cost := table_cost * chair_ratio
  let sofa_cost := table_cost * sofa_ratio
  let total_cost_before_discount := table_cost + 4 * chair_cost + sofa_cost
  let table_discount := discount * table_cost
  let discounted_table_cost := table_cost - table_discount
  let total_cost_after_discount := discounted_table_cost + 4 * chair_cost + sofa_cost
  let sales_tax := tax * total_cost_after_discount
  let final_cost := total_cost_after_discount + sales_tax
  final_cost = 520.02 
:= sorry

end furniture_cost_final_price_l203_203906


namespace prime_factor_difference_duodecimal_l203_203599

theorem prime_factor_difference_duodecimal (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 11) (hB : 0 ≤ B ∧ B ≤ 11) (h : A ≠ B) : 
  ∃ k : ℤ, (12 * A + B - (12 * B + A)) = 11 * k := 
by sorry

end prime_factor_difference_duodecimal_l203_203599


namespace map_distance_8_cm_l203_203227

-- Define the conditions
def scale : ℕ := 5000000
def actual_distance_km : ℕ := 400
def actual_distance_cm : ℕ := 40000000
def map_distance_cm (x : ℕ) : Prop := x * scale = actual_distance_cm

-- The theorem to be proven
theorem map_distance_8_cm : ∃ x : ℕ, map_distance_cm x ∧ x = 8 :=
by
  use 8
  unfold map_distance_cm
  norm_num
  sorry

end map_distance_8_cm_l203_203227


namespace cows_in_group_l203_203015

variable (c h : ℕ)

theorem cows_in_group (hcow : 4 * c + 2 * h = 2 * (c + h) + 18) : c = 9 := 
by 
  sorry

end cows_in_group_l203_203015


namespace probability_letter_in_mathematics_l203_203918

/-- 
Given that Lisa picks one letter randomly from the alphabet, 
prove that the probability that Lisa picks a letter in "MATHEMATICS" is 4/13.
-/
theorem probability_letter_in_mathematics :
  (8 : ℚ) / 26 = 4 / 13 :=
by
  sorry

end probability_letter_in_mathematics_l203_203918


namespace move_point_A_l203_203885

theorem move_point_A :
  let A := (-5, 6)
  let A_right := (A.1 + 5, A.2)
  let A_upwards := (A_right.1, A_right.2 + 6)
  A_upwards = (0, 12) := by
  sorry

end move_point_A_l203_203885


namespace total_coin_value_l203_203823

theorem total_coin_value (total_coins : ℕ) (two_dollar_coins : ℕ) (one_dollar_value : ℕ)
  (two_dollar_value : ℕ) (h_total_coins : total_coins = 275)
  (h_two_dollar_coins : two_dollar_coins = 148)
  (h_one_dollar_value : one_dollar_value = 1)
  (h_two_dollar_value : two_dollar_value = 2) :
  total_coins - two_dollar_coins = 275 - 148
  ∧ ((total_coins - two_dollar_coins) * one_dollar_value + two_dollar_coins * two_dollar_value) = 423 :=
by
  sorry

end total_coin_value_l203_203823


namespace find_angle_C_l203_203351

open Real

theorem find_angle_C (a b C A B : ℝ) 
  (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sin A * sin B) :
  C = π / 3 := 
  sorry

end find_angle_C_l203_203351


namespace compare_exponents_l203_203148

noncomputable def a : ℝ := 20 ^ 22
noncomputable def b : ℝ := 21 ^ 21
noncomputable def c : ℝ := 22 ^ 20

theorem compare_exponents : a > b ∧ b > c :=
by {
  sorry
}

end compare_exponents_l203_203148


namespace negation_proposition_l203_203699

theorem negation_proposition :
  ¬(∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_proposition_l203_203699


namespace slope_intercept_condition_l203_203024

theorem slope_intercept_condition (m b : ℚ) (h_m : m = 1/3) (h_b : b = -3/4) : -1 < m * b ∧ m * b < 0 := by
  sorry

end slope_intercept_condition_l203_203024


namespace triangle_side_length_sum_l203_203491

theorem triangle_side_length_sum :
  ∃ (a b c : ℕ), (5: ℝ) ^ 2 + (7: ℝ) ^ 2 - 2 * (5: ℝ) * (7: ℝ) * (Real.cos (Real.pi * 80 / 180)) = (a: ℝ) + Real.sqrt b + Real.sqrt c ∧
  b = 62 ∧ c = 0 :=
sorry

end triangle_side_length_sum_l203_203491


namespace max_sides_in_subpolygon_l203_203283

/-- In a convex 1950-sided polygon with all its diagonals drawn, the polygon with the greatest number of sides among these smaller polygons can have at most 1949 sides. -/
theorem max_sides_in_subpolygon (n : ℕ) (hn : n = 1950) : 
  ∃ p : ℕ, p = 1949 ∧ ∀ m, m ≤ n-2 → m ≤ 1949 :=
sorry

end max_sides_in_subpolygon_l203_203283


namespace rhombus_side_length_l203_203615

-- Define the conditions including the diagonals and area of the rhombus
def diagonal_ratio (d1 d2 : ℝ) : Prop := d1 = 3 * d2
def area_rhombus (b : ℝ) (K : ℝ) : Prop := K = (1 / 2) * b * (3 * b)

-- Define the side length of the rhombus in terms of K
noncomputable def side_length (K : ℝ) : ℝ := Real.sqrt (5 * K / 3)

-- The main theorem statement
theorem rhombus_side_length (K : ℝ) (b : ℝ) (h1 : diagonal_ratio (3 * b) b) (h2 : area_rhombus b K) : 
  side_length K = Real.sqrt (5 * K / 3) := 
sorry

end rhombus_side_length_l203_203615


namespace factorization_a_minus_b_l203_203729

theorem factorization_a_minus_b (a b : ℤ) (y : ℝ) 
  (h1 : 3 * y ^ 2 - 7 * y - 6 = (3 * y + a) * (y + b)) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) : 
  a - b = 5 :=
sorry

end factorization_a_minus_b_l203_203729


namespace not_perfect_square_T_l203_203668

noncomputable def operation (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom associative {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  operation x (operation y z) = operation (operation x y) z

noncomputable def T (n : ℕ) : ℝ :=
  if h : n ≥ 4 then
    (List.range (n - 2)).foldr (λ x acc => operation (x + 3) acc) 3
  else 0

theorem not_perfect_square_T (n : ℕ) (h : n ≥ 4) :
  ¬ (∃ k : ℕ, (96 / (T n - 2) : ℝ) = k ^ 2) :=
sorry

end not_perfect_square_T_l203_203668


namespace figures_can_be_drawn_l203_203577

structure Figure :=
  (degrees : List ℕ) -- List of degrees of the vertices in the graph associated with the figure.

-- Define a predicate to check if a figure can be drawn without lifting the pencil and without retracing
def canBeDrawnWithoutLifting (fig : Figure) : Prop :=
  let odd_degree_vertices := fig.degrees.filter (λ d => d % 2 = 1)
  odd_degree_vertices.length = 0 ∨ odd_degree_vertices.length = 2

-- Define the figures A, B, C, D with their degrees (examples, these should match the problem's context)
def figureA : Figure := { degrees := [2, 2, 2, 2] }
def figureB : Figure := { degrees := [2, 2, 2, 2, 4] }
def figureC : Figure := { degrees := [3, 3, 3, 3] }
def figureD : Figure := { degrees := [4, 4, 2, 2] }

-- State the theorem that figures A, B, and D can be drawn without lifting the pencil
theorem figures_can_be_drawn :
  canBeDrawnWithoutLifting figureA ∧ canBeDrawnWithoutLifting figureB ∧ canBeDrawnWithoutLifting figureD :=
  by sorry -- Proof to be completed

end figures_can_be_drawn_l203_203577


namespace quadratic_roots_max_value_l203_203242

theorem quadratic_roots_max_value (t q u₁ u₂ : ℝ)
  (h1 : u₁ + u₂ = t)
  (h2 : u₁ * u₂ = q)
  (h3 : u₁ + u₂ = u₁^2 + u₂^2)
  (h4 : u₁ + u₂ = u₁^4 + u₂^4) :
  (1 / u₁^2009 + 1 / u₂^2009) ≤ 2 :=
sorry

-- Explaination: 
-- This theorem states that given the conditions on the roots u₁ and u₂ of the quadratic equation, 
-- the maximum possible value of the expression (1 / u₁^2009 + 1 / u₂^2009) is 2.

end quadratic_roots_max_value_l203_203242


namespace gcd_lcm_product_l203_203690

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  Nat.gcd a b * Nat.lcm a b = 12000 := by
  sorry

end gcd_lcm_product_l203_203690


namespace arithmetic_sequence_sum_l203_203144

theorem arithmetic_sequence_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (m : ℕ) 
  (h1 : S_n m = 0) (h2 : S_n (m - 1) = -2) (h3 : S_n (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l203_203144


namespace possible_r_values_l203_203549

noncomputable def triangle_area (r : ℝ) : ℝ := (r - 3) ^ (3 / 2)

theorem possible_r_values :
  {r : ℝ | 16 ≤ triangle_area r ∧ triangle_area r ≤ 128} = {r : ℝ | 7 ≤ r ∧ r ≤ 19} :=
by
  sorry

end possible_r_values_l203_203549


namespace find_b_l203_203016

open Real

theorem find_b (b : ℝ) (h : b + ⌈b⌉ = 21.5) : b = 10.5 :=
sorry

end find_b_l203_203016


namespace ship_illuminated_by_lighthouse_l203_203469

theorem ship_illuminated_by_lighthouse (d v : ℝ) (hv : v > 0) (ship_speed : ℝ) 
    (hship_speed : ship_speed ≤ v / 8) (rock_distance : ℝ) 
    (hrock_distance : rock_distance = d):
    ∀ t : ℝ, ∃ t' : ℝ, t' ≤ t ∧ t' = (d * t / v) := sorry

end ship_illuminated_by_lighthouse_l203_203469


namespace eval_power_expression_l203_203801

theorem eval_power_expression : (3^3)^2 / 3^2 = 81 := by
  sorry -- Proof omitted as instructed

end eval_power_expression_l203_203801


namespace triangle_inequalities_l203_203433

theorem triangle_inequalities (a b c : ℝ) (h : a < b + c) : b < a + c ∧ c < a + b := 
  sorry

end triangle_inequalities_l203_203433


namespace B_take_time_4_hours_l203_203739

theorem B_take_time_4_hours (A_rate B_rate C_rate D_rate : ℚ) :
  (A_rate = 1 / 4) →
  (B_rate + C_rate = 1 / 2) →
  (A_rate + C_rate = 1 / 2) →
  (D_rate = 1 / 8) →
  (A_rate + B_rate + D_rate = 1 / 1.6) →
  (B_rate = 1 / 4) ∧ (1 / B_rate = 4) :=
by
  sorry

end B_take_time_4_hours_l203_203739


namespace smallest_base_for_100_l203_203737

theorem smallest_base_for_100 :
  ∃ b : ℕ, b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
sorry

end smallest_base_for_100_l203_203737


namespace p_plus_q_l203_203806

-- Define the problem conditions
def p (x : ℝ) : ℝ := 4 * (x - 2)
def q (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- Main theorem to prove the answer
theorem p_plus_q (x : ℝ) : p x + q x = x^2 + 4 * x - 12 := 
by
  sorry

end p_plus_q_l203_203806


namespace probability_factor_120_lt_8_l203_203878

theorem probability_factor_120_lt_8 :
  let n := 120
  let total_factors := 16
  let favorable_factors := 6
  (6 / 16 : ℚ) = 3 / 8 :=
by 
  sorry

end probability_factor_120_lt_8_l203_203878


namespace find_sum_of_m1_m2_l203_203196

-- Define the quadratic equation and the conditions
def quadratic (m : ℂ) (x : ℂ) : ℂ := m * x^2 - (3 * m - 2) * x + 7

-- Define the roots a and b
def are_roots (m a b : ℂ) : Prop := quadratic m a = 0 ∧ quadratic m b = 0

-- The condition given in the problem
def root_condition (a b : ℂ) : Prop := a / b + b / a = 3 / 2

-- Main theorem to be proved
theorem find_sum_of_m1_m2 (m1 m2 a1 a2 b1 b2 : ℂ) 
  (h1 : are_roots m1 a1 b1) 
  (h2 : are_roots m2 a2 b2) 
  (hc1 : root_condition a1 b1) 
  (hc2 : root_condition a2 b2) : 
  m1 + m2 = 73 / 18 :=
by sorry

end find_sum_of_m1_m2_l203_203196


namespace angles_with_same_terminal_side_pi_div_3_l203_203841

noncomputable def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * Real.pi

theorem angles_with_same_terminal_side_pi_div_3 :
  { α : ℝ | same_terminal_side α (Real.pi / 3) } =
  { α : ℝ | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 } :=
by
  sorry

end angles_with_same_terminal_side_pi_div_3_l203_203841


namespace sheila_earning_per_hour_l203_203403

def sheila_hours_per_day_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_per_day_tt : ℕ := 6
def sheila_days_tt : ℕ := 2
def sheila_total_earnings : ℕ := 432

theorem sheila_earning_per_hour : (sheila_total_earnings / (sheila_hours_per_day_mwf * sheila_days_mwf + sheila_hours_per_day_tt * sheila_days_tt)) = 12 := by
  sorry

end sheila_earning_per_hour_l203_203403


namespace smallest_divisible_by_3_and_4_is_12_l203_203484

theorem smallest_divisible_by_3_and_4_is_12 
  (n : ℕ) 
  (h1 : ∃ k1 : ℕ, n = 3 * k1) 
  (h2 : ∃ k2 : ℕ, n = 4 * k2) 
  : n ≥ 12 := sorry

end smallest_divisible_by_3_and_4_is_12_l203_203484


namespace percentage_slump_in_business_l203_203288

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end percentage_slump_in_business_l203_203288


namespace minimum_shots_to_hit_ship_l203_203635

def is_ship_hit (shots : Finset (Fin 7 × Fin 7)) : Prop :=
  -- Assuming the ship can be represented by any 4 consecutive points in a row
  ∀ r : Fin 7, ∃ c1 c2 c3 c4 : Fin 7, 
    (0 ≤ c1.1 ∧ c1.1 ≤ 6 ∧ c1.1 + 3 = c4.1) ∧
    (0 ≤ c2.1 ∧ c2.1 ≤ 6 ∧ c2.1 = c1.1 + 1) ∧
    (0 ≤ c3.1 ∧ c3.1 ≤ 6 ∧ c3.1 = c1.1 + 2) ∧
    (r, c1) ∈ shots ∧ (r, c2) ∈ shots ∧ (r, c3) ∈ shots ∧ (r, c4) ∈ shots

theorem minimum_shots_to_hit_ship : ∃ shots : Finset (Fin 7 × Fin 7), 
  shots.card = 12 ∧ is_ship_hit shots :=
by 
  sorry

end minimum_shots_to_hit_ship_l203_203635


namespace num_divisible_by_10_in_range_correct_l203_203500

noncomputable def num_divisible_by_10_in_range : ℕ :=
  let a1 := 100
  let d := 10
  let an := 500
  (an - a1) / d + 1

theorem num_divisible_by_10_in_range_correct :
  num_divisible_by_10_in_range = 41 := by
  sorry

end num_divisible_by_10_in_range_correct_l203_203500


namespace linear_function_points_relation_l203_203175

theorem linear_function_points_relation (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = 5 * x1 - 3) 
  (h2 : y2 = 5 * x2 - 3) 
  (h3 : x1 < x2) : 
  y1 < y2 :=
sorry

end linear_function_points_relation_l203_203175


namespace inequality_among_three_vars_l203_203410

theorem inequality_among_three_vars 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x + y + z ≥ 3) : 
  (
    1 / (x + y + z ^ 2) + 
    1 / (y + z + x ^ 2) + 
    1 / (z + x + y ^ 2) 
  ) ≤ 1 := 
  sorry

end inequality_among_three_vars_l203_203410


namespace find_slope_l3_l203_203019

/-- Conditions --/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line2 (x y : ℝ) : Prop := y = 2
def A : Prod ℝ ℝ := (0, -3)
def area_ABC : ℝ := 5

noncomputable def B : Prod ℝ ℝ := (2, 2)  -- Simultaneous solution of line1 and line2

theorem find_slope_l3 (C : ℝ × ℝ) (slope_l3 : ℝ) :
  line2 C.1 C.2 ∧
  ((0 : ℝ), -3) ∈ {p : ℝ × ℝ | line1 p.1 p.2 → line2 p.1 p.2 } ∧
  C.2 = 2 ∧
  0 ≤ slope_l3 ∧
  area_ABC = 5 →
  slope_l3 = 5 / 4 :=
sorry

end find_slope_l3_l203_203019


namespace proof_moles_HNO3_proof_molecular_weight_HNO3_l203_203621

variable (n_CaO : ℕ) (molar_mass_H : ℕ) (molar_mass_N : ℕ) (molar_mass_O : ℕ)

def verify_moles_HNO3 (n_CaO : ℕ) : ℕ :=
  2 * n_CaO

def verify_molecular_weight_HNO3 (molar_mass_H molar_mass_N molar_mass_O : ℕ) : ℕ :=
  molar_mass_H + molar_mass_N + 3 * molar_mass_O

theorem proof_moles_HNO3 :
  n_CaO = 7 →
  verify_moles_HNO3 n_CaO = 14 :=
sorry

theorem proof_molecular_weight_HNO3 :
  molar_mass_H = 101 / 100 ∧ molar_mass_N = 1401 / 100 ∧ molar_mass_O = 1600 / 100 →
  verify_molecular_weight_HNO3 molar_mass_H molar_mass_N molar_mass_O = 6302 / 100 :=
sorry

end proof_moles_HNO3_proof_molecular_weight_HNO3_l203_203621


namespace maximize_profit_l203_203366

noncomputable def selling_price_to_maximize_profit (original_price selling_price : ℝ) (units units_sold_decrease : ℝ) : ℝ :=
  let x := 5
  let optimal_selling_price := selling_price + x
  optimal_selling_price

theorem maximize_profit :
  selling_price_to_maximize_profit 80 90 400 20 = 95 :=
by
  sorry

end maximize_profit_l203_203366


namespace area_of_polygon_l203_203277

theorem area_of_polygon (side_length n : ℕ) (h1 : n = 36) (h2 : 36 * side_length = 72) (h3 : ∀ i, i < n → (∃ a, ∃ b, (a + b = 4) ∧ (i = 4 * a + b))) :
  (n / 4) * side_length ^ 2 = 144 :=
by
  sorry

end area_of_polygon_l203_203277


namespace find_fourth_vertex_l203_203292

-- Given three vertices of a tetrahedron
def v1 : ℤ × ℤ × ℤ := (1, 1, 2)
def v2 : ℤ × ℤ × ℤ := (4, 2, 1)
def v3 : ℤ × ℤ × ℤ := (3, 1, 5)

-- The side length squared of the tetrahedron (computed from any pair of given points)
def side_length_squared : ℤ := 11

-- The goal is to find the fourth vertex with integer coordinates which maintains the distance
def is_fourth_vertex (x y z : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 + (z - 2)^2 = side_length_squared ∧
  (x - 4)^2 + (y - 2)^2 + (z - 1)^2 = side_length_squared ∧
  (x - 3)^2 + (y - 1)^2 + (z - 5)^2 = side_length_squared

theorem find_fourth_vertex : is_fourth_vertex 4 1 3 :=
  sorry

end find_fourth_vertex_l203_203292


namespace number_of_valid_paths_l203_203775

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_valid_paths (n : ℕ) :
  let valid_paths := binomial (2 * n) n / (n + 1)
  valid_paths = binomial (2 * n) n - binomial (2 * n) (n + 1) := 
sorry

end number_of_valid_paths_l203_203775


namespace slope_of_line_det_by_two_solutions_l203_203212

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l203_203212


namespace smallest_prime_divisor_524_plus_718_l203_203110

theorem smallest_prime_divisor_524_plus_718 (x y : ℕ) (h1 : x = 5 ^ 24) (h2 : y = 7 ^ 18) :
  ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ p ∣ (x + y) :=
by
  sorry

end smallest_prime_divisor_524_plus_718_l203_203110


namespace handshake_problem_l203_203800

theorem handshake_problem :
  let team_size := 6
  let teams := 2
  let referees := 3
  let handshakes_between_teams := team_size * team_size
  let handshakes_within_teams := teams * (team_size * (team_size - 1)) / 2
  let handshakes_with_referees := (teams * team_size) * referees
  handshakes_between_teams + handshakes_within_teams + handshakes_with_referees = 102 := by
  sorry

end handshake_problem_l203_203800


namespace total_value_proof_l203_203338

def total_bills : ℕ := 126
def five_dollar_bills : ℕ := 84
def ten_dollar_bills : ℕ := total_bills - five_dollar_bills
def value_five_dollar_bills : ℕ := five_dollar_bills * 5
def value_ten_dollar_bills : ℕ := ten_dollar_bills * 10
def total_value : ℕ := value_five_dollar_bills + value_ten_dollar_bills

theorem total_value_proof : total_value = 840 := by
  unfold total_value value_five_dollar_bills value_ten_dollar_bills
  unfold five_dollar_bills ten_dollar_bills total_bills
  -- Calculation steps to show that value_five_dollar_bills + value_ten_dollar_bills = 840
  sorry

end total_value_proof_l203_203338


namespace gcd_of_sum_and_squares_l203_203088

theorem gcd_of_sum_and_squares {a b : ℤ} (h : Int.gcd a b = 1) : 
  Int.gcd (a^2 + b^2) (a + b) = 1 ∨ Int.gcd (a^2 + b^2) (a + b) = 2 := 
by
  sorry

end gcd_of_sum_and_squares_l203_203088


namespace daily_wage_of_c_l203_203734

-- Define the conditions
variables (a b c : ℝ)
variables (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5)
variables (h_days : 6 * a + 9 * b + 4 * c = 1702)

-- Define the proof problem; to prove c = 115
theorem daily_wage_of_c (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) (h_days : 6 * a + 9 * b + 4 * c = 1702) : 
  c = 115 :=
sorry

end daily_wage_of_c_l203_203734


namespace total_oil_volume_l203_203959

theorem total_oil_volume (total_bottles : ℕ) (bottles_250ml : ℕ) (bottles_300ml : ℕ)
    (volume_250ml : ℕ) (volume_300ml : ℕ) (total_volume_ml : ℚ) 
    (total_volume_l : ℚ) (h1 : total_bottles = 35)
    (h2 : bottles_250ml = 17) (h3 : bottles_300ml = total_bottles - bottles_250ml)
    (h4 : volume_250ml = 250) (h5 : volume_300ml = 300) 
    (h6 : total_volume_ml = bottles_250ml * volume_250ml + bottles_300ml * volume_300ml)
    (h7 : total_volume_l = total_volume_ml / 1000) : 
    total_volume_l = 9.65 := 
by 
  sorry

end total_oil_volume_l203_203959


namespace height_eight_times_initial_maximum_growth_year_l203_203459

noncomputable def t : ℝ := 2^(-2/3 : ℝ)
noncomputable def f (n : ℕ) (A a b t : ℝ) : ℝ := 9 * A / (a + b * t^n)

theorem height_eight_times_initial (A : ℝ) : 
  ∀ n : ℕ, f n A 1 8 t = 8 * A ↔ n = 9 :=
sorry

theorem maximum_growth_year (A : ℝ) :
  ∃ n : ℕ, (∀ k : ℕ, (f n A 1 8 t - f (n-1) A 1 8 t) ≥ (f k A 1 8 t - f (k-1) A 1 8 t))
  ∧ n = 5 :=
sorry

end height_eight_times_initial_maximum_growth_year_l203_203459


namespace average_of_rest_equals_40_l203_203359

-- Defining the initial conditions
def total_students : ℕ := 20
def high_scorers : ℕ := 2
def low_scorers : ℕ := 3
def class_average : ℚ := 40

-- The target function to calculate the average of the rest of the students
def average_rest_students (total_students high_scorers low_scorers : ℕ) (class_average : ℚ) : ℚ :=
  let total_marks := total_students * class_average
  let high_scorer_marks := 100 * high_scorers
  let low_scorer_marks := 0 * low_scorers
  let rest_marks := total_marks - (high_scorer_marks + low_scorer_marks)
  let rest_students := total_students - high_scorers - low_scorers
  rest_marks / rest_students

-- The theorem to prove that the average of the rest of the students is 40
theorem average_of_rest_equals_40 : average_rest_students total_students high_scorers low_scorers class_average = 40 := 
by
  sorry

end average_of_rest_equals_40_l203_203359


namespace base_7_multiplication_addition_l203_203853

theorem base_7_multiplication_addition :
  (25 * 3 + 144) % 7^3 = 303 :=
by sorry

end base_7_multiplication_addition_l203_203853


namespace find_equation_of_line_l_l203_203738

-- Define the conditions
def point_P : ℝ × ℝ := (2, 3)

noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

def intercept_condition (a b : ℝ) : Prop := a + b = 0

-- The proof statement
theorem find_equation_of_line_l :
  ∃ (k : ℝ), k = Real.tan angle_of_inclination ∧
  ∃ (C : ℝ), ∀ (x y : ℝ), (y - 3 = k * (x - 2)) ∧ C = (3 + 2 * (Real.sqrt 3)) ∨ 
             (intercept_condition (x / point_P.1) (y / point_P.2) ∧ C = 1) ∨ 
             -- The standard forms of the line equation
             ((Real.sqrt 3 * x + y - C = 0) ∨ (x - y + 1 = 0)) :=
sorry

end find_equation_of_line_l_l203_203738
