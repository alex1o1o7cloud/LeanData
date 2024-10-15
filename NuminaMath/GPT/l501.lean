import Mathlib

namespace NUMINAMATH_GPT_probability_perfect_square_sum_l501_50171

def is_perfect_square_sum (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def count_perfect_square_sums : ℕ :=
  let possible_outcomes := 216
  let favorable_outcomes := 32
  favorable_outcomes

theorem probability_perfect_square_sum :
  (count_perfect_square_sums : ℚ) / 216 = 4 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_perfect_square_sum_l501_50171


namespace NUMINAMATH_GPT_probability_twice_correct_l501_50102

noncomputable def probability_at_least_twice (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 3000) then
  if y ≥ 2*x then (1/6 : ℝ) else 0
else 0

theorem probability_twice_correct : probability_at_least_twice 500 1000 = (1/6 : ℝ) :=
sorry

end NUMINAMATH_GPT_probability_twice_correct_l501_50102


namespace NUMINAMATH_GPT_min_value_expression_l501_50181

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (H : 1 / a + 1 / b = 1) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → 1 / a + 1 / b = 1 → c ≤ 4 / (a - 1) + 9 / (b - 1)) ∧ (c = 6) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l501_50181


namespace NUMINAMATH_GPT_negation_exists_cube_positive_l501_50129

theorem negation_exists_cube_positive :
  ¬ (∃ x : ℝ, x^3 > 0) ↔ ∀ x : ℝ, x^3 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_exists_cube_positive_l501_50129


namespace NUMINAMATH_GPT_compute_expression_l501_50150

theorem compute_expression (y : ℕ) (h : y = 3) : 
  (y^8 + 18 * y^4 + 81) / (y^4 + 9) = 90 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l501_50150


namespace NUMINAMATH_GPT_distance_house_to_market_l501_50155

-- Define each of the given conditions
def distance_to_school := 50
def distance_to_park_from_school := 25
def return_distance := 60
def total_distance_walked := 220

-- Proven distance to the market
def distance_to_market := 85

-- Statement to prove
theorem distance_house_to_market (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = distance_to_school) 
  (h2 : d2 = distance_to_park_from_school) 
  (h3 : d3 = return_distance) 
  (h4 : d4 = total_distance_walked) :
  d4 - (d1 + d2 + d3) = distance_to_market := 
by
  sorry

end NUMINAMATH_GPT_distance_house_to_market_l501_50155


namespace NUMINAMATH_GPT_scientific_notation_l501_50193

theorem scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) (h₁ : x = 5853) (h₂ : 1 ≤ |a|) (h₃ : |a| < 10) (h₄ : x = a * 10^n) : 
  a = 5.853 ∧ n = 3 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_l501_50193


namespace NUMINAMATH_GPT_square_root_of_9_eq_pm_3_l501_50158

theorem square_root_of_9_eq_pm_3 (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_square_root_of_9_eq_pm_3_l501_50158


namespace NUMINAMATH_GPT_odd_coefficients_in_polynomial_l501_50175

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  (2^n - 1) / 3 * 4 + 1

theorem odd_coefficients_in_polynomial (n : ℕ) (hn : 0 < n) :
  (x^2 + x + 1)^n = number_of_odd_coefficients n :=
sorry

end NUMINAMATH_GPT_odd_coefficients_in_polynomial_l501_50175


namespace NUMINAMATH_GPT_graph_shift_l501_50186

theorem graph_shift (f : ℝ → ℝ) (h : f 0 = 2) : f (-1 + 1) = 2 :=
by
  have h1 : f 0 = 2 := h
  sorry

end NUMINAMATH_GPT_graph_shift_l501_50186


namespace NUMINAMATH_GPT_matches_between_withdrawn_players_l501_50106

theorem matches_between_withdrawn_players (n r : ℕ) (h : 50 = (n - 3).choose 2 + (6 - r) + r) : r = 1 :=
sorry

end NUMINAMATH_GPT_matches_between_withdrawn_players_l501_50106


namespace NUMINAMATH_GPT_solve_inequality_l501_50195

theorem solve_inequality (x : ℝ) (h1: 3 * x - 8 ≠ 0) :
  5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10 ↔ (8 / 3) < x ∧ x ≤ (20 / 7) := 
sorry

end NUMINAMATH_GPT_solve_inequality_l501_50195


namespace NUMINAMATH_GPT_parallel_line_passing_through_point_l501_50144

theorem parallel_line_passing_through_point :
  ∃ m b : ℝ, (∀ x y : ℝ, 4 * x + 2 * y = 8 → y = -2 * x + 4) ∧ b = 1 ∧ m = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_GPT_parallel_line_passing_through_point_l501_50144


namespace NUMINAMATH_GPT_ines_bought_3_pounds_l501_50127

-- Define initial and remaining money of Ines
def initial_money : ℕ := 20
def remaining_money : ℕ := 14

-- Define the cost per pound of peaches
def cost_per_pound : ℕ := 2

-- The total money spent on peaches
def money_spent := initial_money - remaining_money

-- The number of pounds of peaches bought
def pounds_of_peaches := money_spent / cost_per_pound

-- The proof problem
theorem ines_bought_3_pounds :
  pounds_of_peaches = 3 :=
by
  sorry

end NUMINAMATH_GPT_ines_bought_3_pounds_l501_50127


namespace NUMINAMATH_GPT_find_principal_amount_l501_50114

variable {P R T : ℝ} -- variables for principal, rate, and time
variable (H1: R = 25)
variable (H2: T = 2)
variable (H3: (P * (0.5625) - P * (0.5)) = 225)

theorem find_principal_amount
    (H1 : R = 25)
    (H2 : T = 2)
    (H3 : (P * 0.0625) = 225) : 
    P = 3600 := 
  sorry

end NUMINAMATH_GPT_find_principal_amount_l501_50114


namespace NUMINAMATH_GPT_min_value_l501_50196

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  ∃ c : ℝ, c = 4 ∧ 
  ∀ x y : ℝ, (x = 1 / (a - 1) ∧ y = 4 / (b - 1)) → (x + y ≥ c) :=
sorry

end NUMINAMATH_GPT_min_value_l501_50196


namespace NUMINAMATH_GPT_speed_of_stream_l501_50177

variable (x : ℝ) -- Let the speed of the stream be x kmph

-- Conditions
variable (speed_of_boat_in_still_water : ℝ)
variable (time_upstream_twice_time_downstream : Prop)

-- Given conditions
axiom h1 : speed_of_boat_in_still_water = 48
axiom h2 : time_upstream_twice_time_downstream → 1 / (speed_of_boat_in_still_water - x) = 2 * (1 / (speed_of_boat_in_still_water + x))

-- Theorem to prove
theorem speed_of_stream (h2: time_upstream_twice_time_downstream) : x = 16 := by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l501_50177


namespace NUMINAMATH_GPT_find_x_of_equation_l501_50133

-- Defining the condition and setting up the proof goal
theorem find_x_of_equation
  (h : (1/2)^25 * (1/x)^12.5 = 1/(18^25)) :
  x = 0.1577 := 
sorry

end NUMINAMATH_GPT_find_x_of_equation_l501_50133


namespace NUMINAMATH_GPT_complex_calculation_l501_50183

theorem complex_calculation (i : ℂ) (hi : i * i = -1) : (1 - i)^2 * i = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_calculation_l501_50183


namespace NUMINAMATH_GPT_problem_statement_l501_50107

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l501_50107


namespace NUMINAMATH_GPT_opposite_of_negative_fraction_l501_50157

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end NUMINAMATH_GPT_opposite_of_negative_fraction_l501_50157


namespace NUMINAMATH_GPT_overall_percentage_supporting_increased_funding_l501_50140

-- Definitions for the conditions
def percent_of_men_supporting (percent_men_supporting : ℕ := 60) : ℕ := percent_men_supporting
def percent_of_women_supporting (percent_women_supporting : ℕ := 80) : ℕ := percent_women_supporting
def number_of_men_surveyed (men_surveyed : ℕ := 100) : ℕ := men_surveyed
def number_of_women_surveyed (women_surveyed : ℕ := 900) : ℕ := women_surveyed

-- Theorem: the overall percent of people surveyed who supported increased funding is 78%
theorem overall_percentage_supporting_increased_funding : 
  (percent_of_men_supporting * number_of_men_surveyed + percent_of_women_supporting * number_of_women_surveyed) / 
  (number_of_men_surveyed + number_of_women_surveyed) = 78 := 
sorry

end NUMINAMATH_GPT_overall_percentage_supporting_increased_funding_l501_50140


namespace NUMINAMATH_GPT_max_tied_teams_for_most_wins_l501_50154

theorem max_tied_teams_for_most_wins 
  (n : ℕ) 
  (h₀ : n = 6)
  (total_games : ℕ := n * (n - 1) / 2)
  (game_result : Π (i j : ℕ), i ≠ j → (0 = 1 → false) ∨ (1 = 1))
  (rank_by_wins : ℕ → ℕ) : true := sorry

end NUMINAMATH_GPT_max_tied_teams_for_most_wins_l501_50154


namespace NUMINAMATH_GPT_nth_term_sequence_sum_first_n_terms_l501_50192

def a_n (n : ℕ) : ℕ :=
  (2 * n - 1) * (2 * n + 2)

def S_n (n : ℕ) : ℚ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6 + n * (n + 1) - 2 * n

theorem nth_term_sequence (n : ℕ) : a_n n = 4 * n^2 + 2 * n - 2 :=
  sorry

theorem sum_first_n_terms (n : ℕ) : S_n n = (4 * n^3 + 9 * n^2 - n) / 3 :=
  sorry

end NUMINAMATH_GPT_nth_term_sequence_sum_first_n_terms_l501_50192


namespace NUMINAMATH_GPT_final_probability_l501_50161

-- Define the structure of the problem
structure GameRound :=
  (green_ball : ℕ)
  (red_ball : ℕ)
  (blue_ball : ℕ)
  (white_ball : ℕ)

structure GameState :=
  (coins : ℕ)
  (players : ℕ)

-- Define the game rules and initial conditions
noncomputable def initial_coins := 5
noncomputable def rounds := 5

-- Probability-related functions and game logic
noncomputable def favorable_outcome_count : ℕ := 6
noncomputable def total_outcomes_per_round : ℕ := 120
noncomputable def probability_per_round : ℚ := favorable_outcome_count / total_outcomes_per_round

theorem final_probability :
  probability_per_round ^ rounds = 1 / 3200000 :=
by
  sorry

end NUMINAMATH_GPT_final_probability_l501_50161


namespace NUMINAMATH_GPT_brenda_total_erasers_l501_50105

theorem brenda_total_erasers (number_of_groups : ℕ) (erasers_per_group : ℕ) (h1 : number_of_groups = 3) (h2 : erasers_per_group = 90) : number_of_groups * erasers_per_group = 270 := 
by
  sorry

end NUMINAMATH_GPT_brenda_total_erasers_l501_50105


namespace NUMINAMATH_GPT_decreasing_function_iff_a_range_l501_50180

noncomputable def f (a x : ℝ) : ℝ := (1 - 2 * a) ^ x

theorem decreasing_function_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 0 < a ∧ a < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_iff_a_range_l501_50180


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_abs_eq_one_l501_50117

theorem sufficient_not_necessary_condition_abs_eq_one (a : ℝ) :
  (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_abs_eq_one_l501_50117


namespace NUMINAMATH_GPT_trioball_play_time_l501_50143

theorem trioball_play_time (total_duration : ℕ) (num_children : ℕ) (players_at_a_time : ℕ) 
  (equal_play_time : ℕ) (H1 : total_duration = 120) (H2 : num_children = 3) (H3 : players_at_a_time = 2)
  (H4 : equal_play_time = 240 / num_children)
  : equal_play_time = 80 := 
by 
  sorry

end NUMINAMATH_GPT_trioball_play_time_l501_50143


namespace NUMINAMATH_GPT_distinct_triangles_from_chord_intersections_l501_50132

theorem distinct_triangles_from_chord_intersections :
  let points := 9
  let chords := (points.choose 2)
  let intersections := (points.choose 4)
  let triangles := (points.choose 6)
  (chords > 0 ∧ intersections > 0 ∧ triangles > 0) →
  triangles = 84 :=
by
  intros
  sorry

end NUMINAMATH_GPT_distinct_triangles_from_chord_intersections_l501_50132


namespace NUMINAMATH_GPT_regular_icosahedron_edges_l501_50149

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end NUMINAMATH_GPT_regular_icosahedron_edges_l501_50149


namespace NUMINAMATH_GPT_closest_to_fraction_is_2000_l501_50125

-- Define the original fractions and their approximations
def numerator : ℝ := 410
def denominator : ℝ := 0.21
def approximated_numerator : ℝ := 400
def approximated_denominator : ℝ := 0.2

-- Define the options to choose from
def options : List ℝ := [100, 500, 1900, 2000, 2500]

-- Statement to prove that the closest value to numerator / denominator is 2000
theorem closest_to_fraction_is_2000 : 
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 100) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 500) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 1900) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 2500) :=
sorry

end NUMINAMATH_GPT_closest_to_fraction_is_2000_l501_50125


namespace NUMINAMATH_GPT_regular_pentagon_cannot_tessellate_l501_50139

-- Definitions of polygons
def is_regular_triangle (angle : ℝ) : Prop := angle = 60
def is_square (angle : ℝ) : Prop := angle = 90
def is_regular_pentagon (angle : ℝ) : Prop := angle = 108
def is_hexagon (angle : ℝ) : Prop := angle = 120

-- Tessellation condition
def divides_evenly (a b : ℝ) : Prop := ∃ k : ℕ, b = k * a

-- The main statement
theorem regular_pentagon_cannot_tessellate :
  ¬ divides_evenly 108 360 :=
sorry

end NUMINAMATH_GPT_regular_pentagon_cannot_tessellate_l501_50139


namespace NUMINAMATH_GPT_youngest_child_age_l501_50174

theorem youngest_child_age 
  (x : ℕ)
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : 
  x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_youngest_child_age_l501_50174


namespace NUMINAMATH_GPT_find_value_of_a_l501_50198

theorem find_value_of_a (a : ℝ) 
  (h : (2 * a + 16 + 3 * a - 8) / 2 = 69) : a = 26 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l501_50198


namespace NUMINAMATH_GPT_ashley_age_l501_50103

theorem ashley_age (A M : ℕ) (h1 : 4 * M = 7 * A) (h2 : A + M = 22) : A = 8 :=
sorry

end NUMINAMATH_GPT_ashley_age_l501_50103


namespace NUMINAMATH_GPT_palindrome_clock_count_l501_50112

-- Definitions based on conditions from the problem statement.
def is_valid_hour (h : ℕ) : Prop := h < 24
def is_valid_minute (m : ℕ) : Prop := m < 60
def is_palindrome (h m : ℕ) : Prop :=
  (h < 10 ∧ m / 10 = h ∧ m % 10 = h) ∨
  (h >= 10 ∧ (h / 10) = (m % 10) ∧ (h % 10) = (m / 10 % 10))

-- Main theorem statement
theorem palindrome_clock_count : 
  (∃ n : ℕ, n = 66 ∧ ∀ (h m : ℕ), is_valid_hour h → is_valid_minute m → is_palindrome h m) := 
sorry

end NUMINAMATH_GPT_palindrome_clock_count_l501_50112


namespace NUMINAMATH_GPT_problem_statement_l501_50173

-- Definitions of conditions
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Statement of the problem
theorem problem_statement (a : ℝ) (h1 : p a) (h2 : q a) : (¬ p a) → (¬ q a) → ∃ x, ¬ (¬ q x) → (¬ (¬ p x)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l501_50173


namespace NUMINAMATH_GPT_factor_expression_l501_50130

theorem factor_expression (a b c : ℝ) :
  3*a^3*(b^2 - c^2) - 2*b^3*(c^2 - a^2) + c^3*(a^2 - b^2) =
  (a - b)*(b - c)*(c - a)*(3*a^2 - 2*b^2 - 3*a^3/c + c) :=
sorry

end NUMINAMATH_GPT_factor_expression_l501_50130


namespace NUMINAMATH_GPT_car_mileage_proof_l501_50170

noncomputable def car_average_mpg 
  (odometer_start: ℝ) (odometer_end: ℝ) 
  (fuel1: ℝ) (fuel2: ℝ) (odometer2: ℝ) 
  (fuel3: ℝ) (odometer3: ℝ) (final_fuel: ℝ) 
  (final_odometer: ℝ): ℝ :=
  (odometer_end - odometer_start) / 
  ((fuel1 + fuel2 + fuel3 + final_fuel): ℝ)

theorem car_mileage_proof:
  car_average_mpg 56200 57150 6 14 56600 10 56880 20 57150 = 19 :=
by
  sorry

end NUMINAMATH_GPT_car_mileage_proof_l501_50170


namespace NUMINAMATH_GPT_praveen_hari_profit_ratio_l501_50184

theorem praveen_hari_profit_ratio
  (praveen_capital : ℕ := 3360)
  (hari_capital : ℕ := 8640)
  (time_praveen_invested : ℕ := 12)
  (time_hari_invested : ℕ := 7)
  (praveen_shares_full_time : ℕ := praveen_capital * time_praveen_invested)
  (hari_shares_full_time : ℕ := hari_capital * time_hari_invested)
  (gcd_common : ℕ := Nat.gcd praveen_shares_full_time hari_shares_full_time) :
  (praveen_shares_full_time / gcd_common) * 2 = 2 ∧ (hari_shares_full_time / gcd_common) * 2 = 3 := by
    sorry

end NUMINAMATH_GPT_praveen_hari_profit_ratio_l501_50184


namespace NUMINAMATH_GPT_circle_radius_equivalence_l501_50108

theorem circle_radius_equivalence (OP_radius : ℝ) (QR : ℝ) (a : ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ)
  (h1 : P = (12, 5))
  (h2 : S = (a, 0))
  (h3 : QR = 5)
  (h4 : OP_radius = 13) :
  a = 8 := 
sorry

end NUMINAMATH_GPT_circle_radius_equivalence_l501_50108


namespace NUMINAMATH_GPT_problem_statement_l501_50126

theorem problem_statement (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) (h₃ : x + y + z = 0) (h₄ : xy + xz + yz ≠ 0) : 
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z)) = -7 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l501_50126


namespace NUMINAMATH_GPT_convert_157_base_10_to_base_7_l501_50159

-- Given
def base_10_to_base_7(n : ℕ) : String := "313"

-- Prove
theorem convert_157_base_10_to_base_7 : base_10_to_base_7 157 = "313" := by
  sorry

end NUMINAMATH_GPT_convert_157_base_10_to_base_7_l501_50159


namespace NUMINAMATH_GPT_cone_volume_l501_50121

theorem cone_volume (diameter height : ℝ) (h_diam : diameter = 14) (h_height : height = 12) :
  (1 / 3 : ℝ) * Real.pi * ((diameter / 2) ^ 2) * height = 196 * Real.pi := by
  sorry

end NUMINAMATH_GPT_cone_volume_l501_50121


namespace NUMINAMATH_GPT_ellipse_k_values_l501_50136

theorem ellipse_k_values (k : ℝ) :
  (∃ k, (∃ e, e = 1/2 ∧
    (∃ a b : ℝ, a = Real.sqrt (k+8) ∧ b = 3 ∧
      ∃ c, (c = Real.sqrt (abs ((a^2) - (b^2)))) ∧ (e = c/b ∨ e = c/a)) ∧
      k = 4 ∨ k = -5/4)) :=
  sorry

end NUMINAMATH_GPT_ellipse_k_values_l501_50136


namespace NUMINAMATH_GPT_hulk_jump_geometric_seq_l501_50128

theorem hulk_jump_geometric_seq :
  ∃ n : ℕ, (2 * 3^(n-1) > 2000) ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_hulk_jump_geometric_seq_l501_50128


namespace NUMINAMATH_GPT_largest_sequence_sum_45_l501_50199

theorem largest_sequence_sum_45 
  (S: ℕ → ℕ)
  (h_S: ∀ n, S n = n * (n + 1) / 2)
  (h_sum: ∃ m: ℕ, S m = 45):
  (∃ k: ℕ, k ≤ 9 ∧ S k = 45) ∧ (∀ m: ℕ, S m ≤ 45 → m ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_largest_sequence_sum_45_l501_50199


namespace NUMINAMATH_GPT_graph_quadrant_exclusion_l501_50160

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end NUMINAMATH_GPT_graph_quadrant_exclusion_l501_50160


namespace NUMINAMATH_GPT_perfect_square_conditions_l501_50135

theorem perfect_square_conditions (x y k : ℝ) :
  (∃ a : ℝ, x^2 + k * x * y + 81 * y^2 = a^2) ↔ (k = 18 ∨ k = -18) :=
sorry

end NUMINAMATH_GPT_perfect_square_conditions_l501_50135


namespace NUMINAMATH_GPT_find_investment_duration_l501_50153

theorem find_investment_duration :
  ∀ (A P R I : ℝ) (T : ℝ),
    A = 1344 →
    P = 1200 →
    R = 5 →
    I = A - P →
    I = (P * R * T) / 100 →
    T = 2.4 :=
by
  intros A P R I T hA hP hR hI1 hI2
  sorry

end NUMINAMATH_GPT_find_investment_duration_l501_50153


namespace NUMINAMATH_GPT_compare_expressions_l501_50190

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end NUMINAMATH_GPT_compare_expressions_l501_50190


namespace NUMINAMATH_GPT_toothpicks_at_150th_stage_l501_50162

theorem toothpicks_at_150th_stage (a₁ d n : ℕ) (h₁ : a₁ = 6) (hd : d = 5) (hn : n = 150) :
  (n * (2 * a₁ + (n - 1) * d)) / 2 = 56775 :=
by
  sorry -- Proof to be completed.

end NUMINAMATH_GPT_toothpicks_at_150th_stage_l501_50162


namespace NUMINAMATH_GPT_probability_succeeding_third_attempt_l501_50178

theorem probability_succeeding_third_attempt :
  let total_keys := 5
  let successful_keys := 2
  let attempts := 3
  let prob := successful_keys / total_keys * (successful_keys / (total_keys - 1)) * (successful_keys / (total_keys - 2))
  prob = 1 / 5 := by
sorry

end NUMINAMATH_GPT_probability_succeeding_third_attempt_l501_50178


namespace NUMINAMATH_GPT_lottery_probability_correct_l501_50119

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_correct :
  let MegaBall_probability := 1 / 30
  let WinnerBalls_probability := 1 / (combination 50 6)
  MegaBall_probability * WinnerBalls_probability = 1 / 476721000 :=
by
  sorry

end NUMINAMATH_GPT_lottery_probability_correct_l501_50119


namespace NUMINAMATH_GPT_jacob_peter_age_ratio_l501_50116

theorem jacob_peter_age_ratio
  (Drew Maya Peter John Jacob : ℕ)
  (h1: Drew = Maya + 5)
  (h2: Peter = Drew + 4)
  (h3: John = 2 * Maya)
  (h4: John = 30)
  (h5: Jacob = 11) :
  Jacob + 2 = 1 / 2 * (Peter + 2) := by
  sorry

end NUMINAMATH_GPT_jacob_peter_age_ratio_l501_50116


namespace NUMINAMATH_GPT_geometric_series_ratio_l501_50147

theorem geometric_series_ratio (a r : ℝ) 
  (h_series : ∑' n : ℕ, a * r^n = 18 )
  (h_odd_series : ∑' n : ℕ, a * r^(2*n + 1) = 8 ) : 
  r = 4 / 5 := 
sorry

end NUMINAMATH_GPT_geometric_series_ratio_l501_50147


namespace NUMINAMATH_GPT_shooting_enthusiast_l501_50118

variables {P : ℝ} -- Declare P as a real number

-- Define the conditions where X follows a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) :=
  n * p * (1 - p)

-- State the theorem in Lean 4
theorem shooting_enthusiast (h : binomial_variance 3 P = 3 / 4) : 
  P = 1 / 2 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_shooting_enthusiast_l501_50118


namespace NUMINAMATH_GPT_find_interest_rate_l501_50131

-- Translating the identified conditions into Lean definitions
def initial_deposit (P : ℝ) : Prop := P > 0
def compounded_semiannually (n : ℕ) : Prop := n = 2
def growth_in_sum (A : ℝ) (P : ℝ) : Prop := A = 1.1592740743 * P
def time_period (t : ℝ) : Prop := t = 2.5

theorem find_interest_rate (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (A : ℝ)
  (h_init : initial_deposit P)
  (h_n : compounded_semiannually n)
  (h_A : growth_in_sum A P)
  (h_t : time_period t) :
  r = 0.06 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l501_50131


namespace NUMINAMATH_GPT_find_all_functions_l501_50188

theorem find_all_functions (n : ℕ) (h_pos : 0 < n) (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x)^n * f (x + y) = (f x)^(n + 1) + x^n * f y) ↔
  (if n % 2 = 1 then ∀ x, f x = 0 ∨ f x = x else ∀ x, f x = 0 ∨ f x = x ∨ f x = -x) :=
sorry

end NUMINAMATH_GPT_find_all_functions_l501_50188


namespace NUMINAMATH_GPT_sequence_general_formula_l501_50120

theorem sequence_general_formula :
  (∃ a : ℕ → ℕ, a 1 = 4 ∧ a 2 = 6 ∧ a 3 = 8 ∧ a 4 = 10 ∧ (∀ n : ℕ, a n = 2 * (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l501_50120


namespace NUMINAMATH_GPT_line_through_point_equidistant_l501_50182

open Real

structure Point where
  x : ℝ
  y : ℝ

def line_equation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def equidistant (p1 p2 : Point) (l : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := l
  let dist_from_p1 := abs (a * p1.x + b * p1.y + c) / sqrt (a^2 + b^2)
  let dist_from_p2 := abs (a * p2.x + b * p2.y + c) / sqrt (a^2 + b^2)
  dist_from_p1 = dist_from_p2

theorem line_through_point_equidistant (a b c : ℝ)
  (P : Point) (A : Point) (B : Point) :
  (P = ⟨1, 2⟩) →
  (A = ⟨2, 2⟩) →
  (B = ⟨4, -6⟩) →
  line_equation a b c P →
  equidistant A B (a, b, c) →
  (a = 2 ∧ b = 1 ∧ c = -4) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_equidistant_l501_50182


namespace NUMINAMATH_GPT_math_problem_proof_l501_50197

variable (Zhang Li Wang Zhao Liu : Prop)
variable (n : ℕ)
variable (reviewed_truth : Zhang → n = 0 ∧ Li → n = 1 ∧ Wang → n = 2 ∧ Zhao → n = 3 ∧ Liu → n = 4)
variable (reviewed_lie : ¬Zhang → ¬(n = 0) ∧ ¬Li → ¬(n = 1) ∧ ¬Wang → ¬(n = 2) ∧ ¬Zhao → ¬(n = 3) ∧ ¬Liu → ¬(n = 4))
variable (some_reviewed : ∃ x, x ∧ ¬x)

theorem math_problem_proof: n = 1 :=
by
  -- Proof omitted, insert logic here
  sorry

end NUMINAMATH_GPT_math_problem_proof_l501_50197


namespace NUMINAMATH_GPT_gum_cost_example_l501_50194

def final_cost (pieces : ℕ) (cost_per_piece : ℕ) (discount_percentage : ℕ) : ℕ :=
  let total_cost := pieces * cost_per_piece
  let discount := total_cost * discount_percentage / 100
  total_cost - discount

theorem gum_cost_example :
  final_cost 1500 2 10 / 100 = 27 :=
by sorry

end NUMINAMATH_GPT_gum_cost_example_l501_50194


namespace NUMINAMATH_GPT_proportion_third_number_l501_50163

theorem proportion_third_number
  (x : ℝ) (y : ℝ)
  (h1 : 0.60 * 4 = x * y)
  (h2 : x = 0.39999999999999997) :
  y = 6 :=
by
  sorry

end NUMINAMATH_GPT_proportion_third_number_l501_50163


namespace NUMINAMATH_GPT_find_k_of_quadratic_polynomial_l501_50110

variable (k : ℝ)

theorem find_k_of_quadratic_polynomial (h1 : (k - 2) = 0) (h2 : k ≠ 0) : k = 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_k_of_quadratic_polynomial_l501_50110


namespace NUMINAMATH_GPT_squirrel_rainy_days_l501_50165

theorem squirrel_rainy_days (s r : ℕ) (h1 : 20 * s + 12 * r = 112) (h2 : s + r = 8) : r = 6 :=
by {
  -- sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_squirrel_rainy_days_l501_50165


namespace NUMINAMATH_GPT_youngest_child_age_l501_50168

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_youngest_child_age_l501_50168


namespace NUMINAMATH_GPT_find_a8_l501_50124

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

theorem find_a8
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_terms a_n S)
  (h_S15 : S 15 = 45) :
  a_n 8 = 3 :=
sorry

end NUMINAMATH_GPT_find_a8_l501_50124


namespace NUMINAMATH_GPT_inequality_system_solution_l501_50113

theorem inequality_system_solution:
  ∀ (x : ℝ),
  (1 - (2*x - 1) / 2 > (3*x - 1) / 4) ∧ (2 - 3*x ≤ 4 - x) →
  -1 ≤ x ∧ x < 1 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l501_50113


namespace NUMINAMATH_GPT_fit_nine_cross_pentominoes_on_chessboard_l501_50141

def cross_pentomino (A B C D E : Prop) :=
  A ∧ B ∧ C ∧ D ∧ E -- A cross pentomino is five connected 1x1 squares

def square1x1 : Prop := sorry -- a placeholder for a 1x1 square

def eight_by_eight_chessboard := Fin 8 × Fin 8 -- an 8x8 chessboard using finitely indexed squares

noncomputable def can_cut_nine_cross_pentominoes : Prop := sorry -- a placeholder proof verification

theorem fit_nine_cross_pentominoes_on_chessboard : can_cut_nine_cross_pentominoes  :=
by 
  -- Assume each cross pentomino consists of 5 connected 1x1 squares
  let cross := cross_pentomino square1x1 square1x1 square1x1 square1x1 square1x1
  -- We need to prove that we can cut out nine such crosses from the 8x8 chessboard
  sorry

end NUMINAMATH_GPT_fit_nine_cross_pentominoes_on_chessboard_l501_50141


namespace NUMINAMATH_GPT_track_length_l501_50172

theorem track_length
  (x : ℕ)
  (run1_Brenda : x / 2 + 80 = a)
  (run2_Sally : x / 2 + 100 = b)
  (run1_ratio : 80 / (x / 2 - 80) = c)
  (run2_ratio : (x / 2 - 100) / (x / 2 + 100) = c)
  : x = 520 :=
by sorry

end NUMINAMATH_GPT_track_length_l501_50172


namespace NUMINAMATH_GPT_cricket_matches_total_l501_50176

theorem cricket_matches_total
  (n : ℕ)
  (avg_all : ℝ)
  (avg_first4 : ℝ)
  (avg_last3 : ℝ)
  (h_avg_all : avg_all = 56)
  (h_avg_first4 : avg_first4 = 46)
  (h_avg_last3 : avg_last3 = 69.33333333333333)
  (h_total_runs : n * avg_all = 4 * avg_first4 + 3 * avg_last3) :
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_cricket_matches_total_l501_50176


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l501_50185

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 - 4 * x = 0 → (x, y) = (1, 0) :=
by
  -- Use the equivalence given by the problem
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l501_50185


namespace NUMINAMATH_GPT_gcd_1213_1985_eq_1_l501_50145

theorem gcd_1213_1985_eq_1
  (h1: ¬ (1213 % 2 = 0))
  (h2: ¬ (1213 % 3 = 0))
  (h3: ¬ (1213 % 5 = 0))
  (h4: ¬ (1985 % 2 = 0))
  (h5: ¬ (1985 % 3 = 0))
  (h6: ¬ (1985 % 5 = 0)):
  Nat.gcd 1213 1985 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_1213_1985_eq_1_l501_50145


namespace NUMINAMATH_GPT_trigonometric_identity_l501_50101

theorem trigonometric_identity (x : ℝ) (h₁ : Real.sin x = 4 / 5) (h₂ : π / 2 ≤ x ∧ x ≤ π) :
  Real.cos x = -3 / 5 ∧ (Real.cos (-x) / (Real.sin (π / 2 - x) - Real.sin (2 * π - x)) = -3) := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l501_50101


namespace NUMINAMATH_GPT_exponent_zero_nonneg_l501_50156

theorem exponent_zero_nonneg (a : ℝ) (h : a ≠ -1) : (a + 1) ^ 0 = 1 :=
sorry

end NUMINAMATH_GPT_exponent_zero_nonneg_l501_50156


namespace NUMINAMATH_GPT_total_ridges_on_all_records_l501_50138

theorem total_ridges_on_all_records :
  let ridges_per_record := 60
  let cases := 4
  let shelves_per_case := 3
  let records_per_shelf := 20
  let shelf_fullness_ratio := 0.60

  let total_capacity := cases * shelves_per_case * records_per_shelf
  let actual_records := total_capacity * shelf_fullness_ratio
  let total_ridges := actual_records * ridges_per_record
  
  total_ridges = 8640 :=
by
  sorry

end NUMINAMATH_GPT_total_ridges_on_all_records_l501_50138


namespace NUMINAMATH_GPT_compute_cd_l501_50122

variable (c d : ℝ)

theorem compute_cd (h1 : c + d = 10) (h2 : c^3 + d^3 = 370) : c * d = 21 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_compute_cd_l501_50122


namespace NUMINAMATH_GPT_arithmetic_geometric_progression_l501_50111

theorem arithmetic_geometric_progression (a d : ℝ)
    (h1 : 2 * (a - d) * a * (a + d + 7) = 1000)
    (h2 : a^2 = 2 * (a - d) * (a + d + 7)) :
    d = 8 ∨ d = -8 := 
    sorry

end NUMINAMATH_GPT_arithmetic_geometric_progression_l501_50111


namespace NUMINAMATH_GPT_sons_ages_l501_50123

theorem sons_ages (x y : ℕ) (h1 : 2 * x = x + y + 18) (h2 : y = (x - y) - 6) : 
  x = 30 ∧ y = 12 := by
  sorry

end NUMINAMATH_GPT_sons_ages_l501_50123


namespace NUMINAMATH_GPT_product_four_integers_sum_to_50_l501_50151

theorem product_four_integers_sum_to_50 (E F G H : ℝ) 
  (h₀ : E + F + G + H = 50)
  (h₁ : E - 3 = F + 3)
  (h₂ : E - 3 = G * 3)
  (h₃ : E - 3 = H / 3) :
  E * F * G * H = 7461.9140625 := 
sorry

end NUMINAMATH_GPT_product_four_integers_sum_to_50_l501_50151


namespace NUMINAMATH_GPT_hyperbola_equation_l501_50146

variable (a b : ℝ)
variable (c : ℝ) (h1 : c = 4)
variable (h2 : b / a = Real.sqrt 3)
variable (h3 : a ^ 2 + b ^ 2 = c ^ 2)

theorem hyperbola_equation : (a ^ 2 = 4) ∧ (b ^ 2 = 12) ↔ (∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → (x ^ 2 / 4) - (y ^ 2 / 12) = 1) := by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l501_50146


namespace NUMINAMATH_GPT_number_of_dogs_is_correct_l501_50169

variable (D C B : ℕ)
variable (k : ℕ)

def validRatio (D C B : ℕ) : Prop := D = 7 * k ∧ C = 7 * k ∧ B = 8 * k
def totalDogsAndBunnies (D B : ℕ) : Prop := D + B = 330
def correctNumberOfDogs (D : ℕ) : Prop := D = 154

theorem number_of_dogs_is_correct (D C B k : ℕ) 
  (hRatio : validRatio D C B k)
  (hTotal : totalDogsAndBunnies D B) :
  correctNumberOfDogs D :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_is_correct_l501_50169


namespace NUMINAMATH_GPT_company_total_employees_l501_50148

def total_employees_after_hiring (T : ℕ) (before_hiring_female_percentage : ℚ) (additional_male_workers : ℕ) (after_hiring_female_percentage : ℚ) : ℕ :=
  T + additional_male_workers

theorem company_total_employees (T : ℕ)
  (before_hiring_female_percentage : ℚ)
  (additional_male_workers : ℕ)
  (after_hiring_female_percentage : ℚ)
  (h_before_percent : before_hiring_female_percentage = 0.60)
  (h_additional_male : additional_male_workers = 28)
  (h_after_percent : after_hiring_female_percentage = 0.55)
  (h_equation : (before_hiring_female_percentage * T)/(T + additional_male_workers) = after_hiring_female_percentage) :
  total_employees_after_hiring T before_hiring_female_percentage additional_male_workers after_hiring_female_percentage = 336 :=
by {
  -- This is where you add the proof steps.
  sorry
}

end NUMINAMATH_GPT_company_total_employees_l501_50148


namespace NUMINAMATH_GPT_find_number_l501_50115

theorem find_number {x : ℝ} (h : 0.5 * x - 10 = 25) : x = 70 :=
sorry

end NUMINAMATH_GPT_find_number_l501_50115


namespace NUMINAMATH_GPT_right_triangle_inradius_height_ratio_l501_50166

-- Define a right triangle with sides a, b, and hypotenuse c
variables {a b c : ℝ}
-- Define the altitude from the right angle vertex
variables {h : ℝ}
-- Define the inradius of the triangle
variables {r : ℝ}

-- Define the conditions: right triangle 
-- and the relationships for h and r
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def altitude (h : ℝ) (a b c : ℝ) : Prop := h = (a * b) / c
def inradius (r : ℝ) (a b c : ℝ) : Prop := r = (a + b - c) / 2

theorem right_triangle_inradius_height_ratio {a b c h r : ℝ} 
  (Hrt : is_right_triangle a b c)
  (Hh : altitude h a b c)
  (Hr : inradius r a b c) : 
  0.4 < r / h ∧ r / h < 0.5 :=
sorry

end NUMINAMATH_GPT_right_triangle_inradius_height_ratio_l501_50166


namespace NUMINAMATH_GPT_charge_R_12_5_percent_more_l501_50167

-- Let R be the charge for a single room at hotel R.
-- Let G be the charge for a single room at hotel G.
-- Let P be the charge for a single room at hotel P.

def charge_R (R : ℝ) : Prop := true
def charge_G (G : ℝ) : Prop := true
def charge_P (P : ℝ) : Prop := true

axiom hotel_P_20_less_R (R P : ℝ) : charge_R R → charge_P P → P = 0.80 * R
axiom hotel_P_10_less_G (G P : ℝ) : charge_G G → charge_P P → P = 0.90 * G

theorem charge_R_12_5_percent_more (R G : ℝ) :
  charge_R R → charge_G G → (∃ P, charge_P P ∧ P = 0.80 * R ∧ P = 0.90 * G) → R = 1.125 * G :=
by sorry

end NUMINAMATH_GPT_charge_R_12_5_percent_more_l501_50167


namespace NUMINAMATH_GPT_find_k_l501_50191

variable {S : ℕ → ℤ} -- Assuming the sum function S for the arithmetic sequence 
variable {k : ℕ} -- k is a natural number

theorem find_k (h1 : S (k - 2) = -4) (h2 : S k = 0) (h3 : S (k + 2) = 8) (hk2 : k > 2) (hnaturalk : k ∈ Set.univ) : k = 6 := by
  sorry

end NUMINAMATH_GPT_find_k_l501_50191


namespace NUMINAMATH_GPT_find_natural_numbers_l501_50189

open Nat

theorem find_natural_numbers (n : ℕ) (h : ∃ m : ℤ, 2^n + 33 = m^2) : n = 4 ∨ n = 8 :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_l501_50189


namespace NUMINAMATH_GPT_estimate_white_balls_l501_50100

theorem estimate_white_balls :
  (∃ x : ℕ, (6 / (x + 6) : ℝ) = 0.2 ∧ x = 24) :=
by
  use 24
  sorry

end NUMINAMATH_GPT_estimate_white_balls_l501_50100


namespace NUMINAMATH_GPT_auction_starting_price_l501_50134

-- Defining the conditions
def bid_increment := 5         -- The dollar increment per bid
def bids_per_person := 5       -- Number of bids per person
def total_bidders := 2         -- Number of people bidding
def final_price := 65          -- Final price of the desk after all bids

-- Calculate derived conditions
def total_bids := bids_per_person * total_bidders
def total_increment := total_bids * bid_increment

-- The statement to be proved
theorem auction_starting_price : (final_price - total_increment) = 15 :=
by
  sorry

end NUMINAMATH_GPT_auction_starting_price_l501_50134


namespace NUMINAMATH_GPT_value_of_a_l501_50152

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^2 - a * x + 4) (h₂ : ∀ x, f (x + 1) = f (1 - x)) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_l501_50152


namespace NUMINAMATH_GPT_two_digit_numbers_l501_50104

theorem two_digit_numbers :
  ∃ (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ x < y ∧ 2000 + x + y = x * y := 
sorry

end NUMINAMATH_GPT_two_digit_numbers_l501_50104


namespace NUMINAMATH_GPT_series_converges_to_one_l501_50109

noncomputable def infinite_series := ∑' n, (3^n) / (3^(2^n) + 2)

theorem series_converges_to_one :
  infinite_series = 1 := by
  sorry

end NUMINAMATH_GPT_series_converges_to_one_l501_50109


namespace NUMINAMATH_GPT_ab_bc_cd_da_le_four_l501_50187

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_bc_cd_da_le_four_l501_50187


namespace NUMINAMATH_GPT_NY_Mets_fans_count_l501_50137

noncomputable def NY_Yankees_fans (M: ℝ) : ℝ := (3/2) * M
noncomputable def Boston_Red_Sox_fans (M: ℝ) : ℝ := (5/4) * M
noncomputable def LA_Dodgers_fans (R: ℝ) : ℝ := (2/7) * R

theorem NY_Mets_fans_count :
  ∃ M : ℕ, let Y := NY_Yankees_fans M
           let R := Boston_Red_Sox_fans M
           let D := LA_Dodgers_fans R
           Y + M + R + D = 780 ∧ M = 178 :=
by
  sorry

end NUMINAMATH_GPT_NY_Mets_fans_count_l501_50137


namespace NUMINAMATH_GPT_part_one_part_two_l501_50164

variable {a : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom recurrence_relation : ∀ n, a (n + 1) = 2 * (a n) + 1

-- Proof of the first part
theorem part_one: ∀ n, (a (n + 1) + 1) = 2 * (a n + 1) :=
by
  sorry

-- General formula for the sequence
theorem part_two: ∀ n, a n = 2^(n + 1) - 1 :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l501_50164


namespace NUMINAMATH_GPT_ratio_difference_l501_50179

variables (p q r : ℕ) (x : ℕ)
noncomputable def shares_p := 3 * x
noncomputable def shares_q := 7 * x
noncomputable def shares_r := 12 * x

theorem ratio_difference (h1 : shares_q - shares_p = 2400) : shares_r - shares_q = 3000 :=
by sorry

end NUMINAMATH_GPT_ratio_difference_l501_50179


namespace NUMINAMATH_GPT_arithmetic_sequences_ratio_l501_50142

theorem arithmetic_sequences_ratio (a b S T : ℕ → ℕ) (h : ∀ n, S n / T n = 2 * n / (3 * n + 1)) :
  (a 2) / (b 3 + b 7) + (a 8) / (b 4 + b 6) = 9 / 14 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequences_ratio_l501_50142
