import Mathlib

namespace NUMINAMATH_GPT_largest_angle_of_pentagon_l1631_163140

theorem largest_angle_of_pentagon (a d : ℝ) (h1 : a = 100) (h2 : d = 2) :
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d
  let angle5 := a + 4 * d
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧ angle5 = 116 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_pentagon_l1631_163140


namespace NUMINAMATH_GPT_complete_the_square_solution_l1631_163190

theorem complete_the_square_solution (x : ℝ) :
  (∃ x, x^2 + 2 * x - 1 = 0) → (x + 1)^2 = 2 :=
sorry

end NUMINAMATH_GPT_complete_the_square_solution_l1631_163190


namespace NUMINAMATH_GPT_idempotent_elements_are_zero_l1631_163182

-- Definitions based on conditions specified in the problem
variables {R : Type*} [Ring R] [CharZero R]
variable {e f g : R}

def idempotent (x : R) : Prop := x * x = x

-- The theorem to be proved
theorem idempotent_elements_are_zero (h_e : idempotent e) (h_f : idempotent f) (h_g : idempotent g) (h_sum : e + f + g = 0) : 
  e = 0 ∧ f = 0 ∧ g = 0 := 
sorry

end NUMINAMATH_GPT_idempotent_elements_are_zero_l1631_163182


namespace NUMINAMATH_GPT_sectionBSeats_l1631_163134

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end NUMINAMATH_GPT_sectionBSeats_l1631_163134


namespace NUMINAMATH_GPT_required_blue_balls_to_remove_l1631_163105

-- Define the constants according to conditions
def total_balls : ℕ := 120
def red_balls : ℕ := 54
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℚ := 0.75 -- ℚ is the type for rational numbers

-- Lean theorem statement
theorem required_blue_balls_to_remove (x : ℕ) : 
    (red_balls:ℚ) / (total_balls - x : ℚ) = desired_percentage_red → x = 48 :=
by
  sorry

end NUMINAMATH_GPT_required_blue_balls_to_remove_l1631_163105


namespace NUMINAMATH_GPT_coordinates_of_point_in_fourth_quadrant_l1631_163197

theorem coordinates_of_point_in_fourth_quadrant 
  (P : ℝ × ℝ)
  (h₁ : P.1 > 0) -- P is in the fourth quadrant, so x > 0
  (h₂ : P.2 < 0) -- P is in the fourth quadrant, so y < 0
  (dist_x_axis : P.2 = -5) -- Distance from P to x-axis is 5 (absolute value of y)
  (dist_y_axis : P.1 = 3)  -- Distance from P to y-axis is 3 (absolute value of x)
  : P = (3, -5) :=
sorry

end NUMINAMATH_GPT_coordinates_of_point_in_fourth_quadrant_l1631_163197


namespace NUMINAMATH_GPT_ratio_division_l1631_163121

theorem ratio_division
  (A B C : ℕ)
  (h : (A : ℚ) / B = 3 / 2 ∧ (B : ℚ) / C = 1 / 3) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_division_l1631_163121


namespace NUMINAMATH_GPT_equilateral_triangle_black_area_l1631_163106

theorem equilateral_triangle_black_area :
  let initial_black_area := 1
  let change_fraction := 5/6 * 9/10
  let area_after_n_changes (n : Nat) : ℚ := initial_black_area * (change_fraction ^ n)
  area_after_n_changes 3 = 27/64 := 
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_black_area_l1631_163106


namespace NUMINAMATH_GPT_pseudo_symmetry_abscissa_l1631_163187

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4 * Real.log x

theorem pseudo_symmetry_abscissa :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧
    (∀ x : ℝ, x ≠ x0 → (f x - ((2*x0 + 4/x0 - 6)*(x - x0) + x0^2 - 6*x0 + 4*Real.log x0)) / (x - x0) > 0) :=
sorry

end NUMINAMATH_GPT_pseudo_symmetry_abscissa_l1631_163187


namespace NUMINAMATH_GPT_Ian_kept_1_rose_l1631_163152

def initial_roses : ℕ := 20
def roses_given_to_mother : ℕ := 6
def roses_given_to_grandmother : ℕ := 9
def roses_given_to_sister : ℕ := 4
def total_roses_given : ℕ := roses_given_to_mother + roses_given_to_grandmother + roses_given_to_sister
def roses_kept (initial: ℕ) (given: ℕ) : ℕ := initial - given

theorem Ian_kept_1_rose :
  roses_kept initial_roses total_roses_given = 1 :=
by
  sorry

end NUMINAMATH_GPT_Ian_kept_1_rose_l1631_163152


namespace NUMINAMATH_GPT_four_digit_palindrome_perfect_squares_l1631_163151

theorem four_digit_palindrome_perfect_squares : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → 
            ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
            n = 1001 * a + 110 * b ∧ 
            ∃ k : ℕ, k * k = n) → count = 2 := by
  sorry

end NUMINAMATH_GPT_four_digit_palindrome_perfect_squares_l1631_163151


namespace NUMINAMATH_GPT_color_crafter_secret_codes_l1631_163137

theorem color_crafter_secret_codes :
  8^5 = 32768 := by
  sorry

end NUMINAMATH_GPT_color_crafter_secret_codes_l1631_163137


namespace NUMINAMATH_GPT_years_to_rise_to_chief_l1631_163114

-- Definitions based on the conditions
def ageWhenRetired : ℕ := 46
def ageWhenJoined : ℕ := 18
def additionalYearsAsMasterChief : ℕ := 10
def multiplierForChiefToMasterChief : ℚ := 1.25

-- Total years spent in the military
def totalYearsInMilitary : ℕ := ageWhenRetired - ageWhenJoined

-- Given conditions and correct answer
theorem years_to_rise_to_chief (x : ℚ) (h : totalYearsInMilitary = x + multiplierForChiefToMasterChief * x + additionalYearsAsMasterChief) :
  x = 8 := by
  sorry

end NUMINAMATH_GPT_years_to_rise_to_chief_l1631_163114


namespace NUMINAMATH_GPT_andrey_stamps_count_l1631_163178

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end NUMINAMATH_GPT_andrey_stamps_count_l1631_163178


namespace NUMINAMATH_GPT_second_player_always_wins_l1631_163199

open Nat

theorem second_player_always_wins (cards : Finset ℕ) (h_card_count : cards.card = 16) :
  ∃ strategy : ℕ → ℕ, ∀ total_score : ℕ,
  total_score ≤ 22 → (total_score + strategy total_score > 22 ∨ 
  (∃ next_score : ℕ, total_score + next_score ≤ 22 ∧ strategy (total_score + next_score) = 1)) :=
sorry

end NUMINAMATH_GPT_second_player_always_wins_l1631_163199


namespace NUMINAMATH_GPT_hyperbola_line_intersection_l1631_163136

theorem hyperbola_line_intersection
  (A B m : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hm : m ≠ 0) :
  ∃ x y : ℝ, A^2 * x^2 - B^2 * y^2 = 1 ∧ Ax - By = m ∧ Bx + Ay ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_line_intersection_l1631_163136


namespace NUMINAMATH_GPT_restaurant_dinners_sold_on_Monday_l1631_163154

theorem restaurant_dinners_sold_on_Monday (M : ℕ) 
  (h1 : ∀ tues_dinners, tues_dinners = M + 40) 
  (h2 : ∀ wed_dinners, wed_dinners = (M + 40) / 2)
  (h3 : ∀ thurs_dinners, thurs_dinners = ((M + 40) / 2) + 3)
  (h4 : M + (M + 40) + ((M + 40) / 2) + (((M + 40) / 2) + 3) = 203) : 
  M = 40 := 
sorry

end NUMINAMATH_GPT_restaurant_dinners_sold_on_Monday_l1631_163154


namespace NUMINAMATH_GPT_conditional_probability_of_A_given_target_hit_l1631_163126

theorem conditional_probability_of_A_given_target_hit :
  (3 / 5 : ℚ) * ( ( 4 / 5 + 1 / 5) ) = (15 / 23 : ℚ) :=
  sorry

end NUMINAMATH_GPT_conditional_probability_of_A_given_target_hit_l1631_163126


namespace NUMINAMATH_GPT_range_of_constant_c_in_quadrant_I_l1631_163175

theorem range_of_constant_c_in_quadrant_I (c : ℝ) (x y : ℝ)
  (h1 : x - 2 * y = 4)
  (h2 : 2 * c * x + y = 5)
  (hx_pos : x > 0)
  (hy_pos : y > 0) : 
  -1 / 4 < c ∧ c < 5 / 8 := 
sorry

end NUMINAMATH_GPT_range_of_constant_c_in_quadrant_I_l1631_163175


namespace NUMINAMATH_GPT_purple_coincide_pairs_l1631_163108

theorem purple_coincide_pairs
    (yellow_triangles_upper : ℕ)
    (yellow_triangles_lower : ℕ)
    (green_triangles_upper : ℕ)
    (green_triangles_lower : ℕ)
    (purple_triangles_upper : ℕ)
    (purple_triangles_lower : ℕ)
    (yellow_coincide_pairs : ℕ)
    (green_coincide_pairs : ℕ)
    (yellow_purple_pairs : ℕ) :
    yellow_triangles_upper = 4 →
    yellow_triangles_lower = 4 →
    green_triangles_upper = 6 →
    green_triangles_lower = 6 →
    purple_triangles_upper = 10 →
    purple_triangles_lower = 10 →
    yellow_coincide_pairs = 3 →
    green_coincide_pairs = 4 →
    yellow_purple_pairs = 3 →
    (∃ purple_coincide_pairs : ℕ, purple_coincide_pairs = 5) :=
by sorry

end NUMINAMATH_GPT_purple_coincide_pairs_l1631_163108


namespace NUMINAMATH_GPT_union_of_sets_l1631_163164

-- Definition for set M
def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

-- Definition for set N
def N : Set ℝ := {x | 2 * x + 1 < 5}

-- The theorem linking M and N
theorem union_of_sets : M ∪ N = {x | x < 3} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_union_of_sets_l1631_163164


namespace NUMINAMATH_GPT_four_leaved_clovers_percentage_l1631_163124

noncomputable def percentage_of_four_leaved_clovers (clovers total_clovers purple_four_leaved_clovers : ℕ ) : ℝ := 
  (purple_four_leaved_clovers * 4 * 100) / total_clovers 

theorem four_leaved_clovers_percentage :
  percentage_of_four_leaved_clovers 500 500 25 = 20 := 
by
  -- application of conditions and arithmetic simplification.
  sorry

end NUMINAMATH_GPT_four_leaved_clovers_percentage_l1631_163124


namespace NUMINAMATH_GPT_geometric_sequence_product_l1631_163181

theorem geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (hA_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (hA_not_zero : ∀ n, a n ≠ 0)
  (h_condition : a 4 - 2 * (a 7)^2 + 3 * a 8 = 0)
  (hB_seq : ∀ n, b n = b 1 * (b 2 / b 1)^(n - 1))
  (hB7 : b 7 = a 7) :
  b 3 * b 7 * b 11 = 8 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1631_163181


namespace NUMINAMATH_GPT_edward_can_buy_candies_l1631_163167

theorem edward_can_buy_candies (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ)
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 :=
by
  sorry

end NUMINAMATH_GPT_edward_can_buy_candies_l1631_163167


namespace NUMINAMATH_GPT_percent_gain_on_transaction_l1631_163129

theorem percent_gain_on_transaction :
  ∀ (x : ℝ), (850 : ℝ) * x + (50 : ℝ) * (1.10 * ((850 : ℝ) * x / 800)) = 850 * x * (1 + 0.06875) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_percent_gain_on_transaction_l1631_163129


namespace NUMINAMATH_GPT_games_needed_in_single_elimination_l1631_163141

theorem games_needed_in_single_elimination (teams : ℕ) (h : teams = 23) : 
  ∃ games : ℕ, games = teams - 1 ∧ games = 22 :=
by
  existsi (teams - 1)
  sorry

end NUMINAMATH_GPT_games_needed_in_single_elimination_l1631_163141


namespace NUMINAMATH_GPT_shape_of_r_eq_c_in_cylindrical_coords_l1631_163100

variable {c : ℝ}

theorem shape_of_r_eq_c_in_cylindrical_coords (h : c > 0) :
  ∀ (r θ z : ℝ), (r = c) ↔ ∃ (cylinder : ℝ), cylinder = r ∧ cylinder = c :=
by
  sorry

end NUMINAMATH_GPT_shape_of_r_eq_c_in_cylindrical_coords_l1631_163100


namespace NUMINAMATH_GPT_part1_part2_part3_l1631_163102

noncomputable def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 3/2 else 
if n = 3 then 5/4 else 
sorry

noncomputable def S (n : ℕ) : ℝ := sorry

axiom recurrence {n : ℕ} (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Part 1
theorem part1 : a 4 = 7 / 8 :=
sorry

-- Part 2
theorem part2 : ∃ (r : ℝ) (b : ℕ → ℝ), (r = 1/2) ∧ (∀ n ≥ 1, a (n + 1) - r * a n = b n) :=
sorry

-- Part 3
theorem part3 : ∀ n, a n = (2 * n - 1) / 2^(n - 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1631_163102


namespace NUMINAMATH_GPT_emily_and_berengere_contribution_l1631_163135

noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def euro_to_gbp : ℝ := 0.85

noncomputable def cake_cost_euros : ℝ := 12
noncomputable def cookies_cost_euros : ℝ := 5
noncomputable def total_cost_euros : ℝ := cake_cost_euros + cookies_cost_euros

noncomputable def emily_usd : ℝ := 10
noncomputable def liam_gbp : ℝ := 10

noncomputable def emily_euros : ℝ := emily_usd / euro_to_usd
noncomputable def liam_euros : ℝ := liam_gbp / euro_to_gbp

noncomputable def total_available_euros : ℝ := emily_euros + liam_euros

theorem emily_and_berengere_contribution : total_available_euros >= total_cost_euros := by
  sorry

end NUMINAMATH_GPT_emily_and_berengere_contribution_l1631_163135


namespace NUMINAMATH_GPT_quadratic_roots_difference_l1631_163171

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2 ∧ x1 * x2 = q ∧ x1 + x2 = -p) → p = 2 * Real.sqrt (q + 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_difference_l1631_163171


namespace NUMINAMATH_GPT_find_beta_l1631_163170

theorem find_beta 
  (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) :
  β = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_find_beta_l1631_163170


namespace NUMINAMATH_GPT_sqrt_360000_eq_600_l1631_163145

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_GPT_sqrt_360000_eq_600_l1631_163145


namespace NUMINAMATH_GPT_exists_six_distinct_naturals_l1631_163192

theorem exists_six_distinct_naturals :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c + d + e + f = 3528 ∧
    (1/a + 1/b + 1/c + 1/d + 1/e + 1/f : ℝ) = 3528 / 2012 :=
sorry

end NUMINAMATH_GPT_exists_six_distinct_naturals_l1631_163192


namespace NUMINAMATH_GPT_minimum_value_l1631_163161

theorem minimum_value (a : ℝ) (h₀ : 0 < a) (h₁ : a < 3) :
  ∃ a : ℝ, (0 < a ∧ a < 3) ∧ (1 / a + 4 / (8 - a) = 9 / 8) := by
sorry

end NUMINAMATH_GPT_minimum_value_l1631_163161


namespace NUMINAMATH_GPT_coeff_x5_in_expansion_l1631_163166

noncomputable def binomial_expansion_coeff (n k : ℕ) (x : ℝ) : ℝ :=
  Real.sqrt x ^ (n - k) * 2 ^ k * (Nat.choose n k)

theorem coeff_x5_in_expansion :
  (binomial_expansion_coeff 12 2 x) = 264 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x5_in_expansion_l1631_163166


namespace NUMINAMATH_GPT_train_stoppages_l1631_163112

variables (sA sA' sB sB' sC sC' : ℝ)
variables (x y z : ℝ)

-- Conditions
def conditions : Prop :=
  sA = 80 ∧ sA' = 60 ∧
  sB = 100 ∧ sB' = 75 ∧
  sC = 120 ∧ sC' = 90

-- Goal that we need to prove
def goal : Prop :=
  x = 15 ∧ y = 15 ∧ z = 15

-- Main statement
theorem train_stoppages : conditions sA sA' sB sB' sC sC' → goal x y z :=
by
  sorry

end NUMINAMATH_GPT_train_stoppages_l1631_163112


namespace NUMINAMATH_GPT_mabel_initial_daisies_l1631_163123

theorem mabel_initial_daisies (D: ℕ) (h1: 8 * (D - 2) = 24) : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_mabel_initial_daisies_l1631_163123


namespace NUMINAMATH_GPT_mean_score_is_74_l1631_163183

theorem mean_score_is_74 (σ q : ℝ)
  (h1 : 58 = q - 2 * σ)
  (h2 : 98 = q + 3 * σ) :
  q = 74 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_is_74_l1631_163183


namespace NUMINAMATH_GPT_count_perfect_fourth_powers_l1631_163158

theorem count_perfect_fourth_powers: 
  ∃ n_count: ℕ, n_count = 4 ∧ ∀ n: ℕ, (50 ≤ n^4 ∧ n^4 ≤ 2000) → (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_perfect_fourth_powers_l1631_163158


namespace NUMINAMATH_GPT_largest_hexagon_angle_l1631_163156

-- We define the conditions first
def angle_ratios (x : ℝ) := [3*x, 3*x, 3*x, 4*x, 5*x, 6*x]
def sum_of_angles (angles : List ℝ) := angles.sum = 720

-- Now we state our proof goal
theorem largest_hexagon_angle :
  ∀ (x : ℝ), sum_of_angles (angle_ratios x) → 6 * x = 180 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_largest_hexagon_angle_l1631_163156


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1631_163104

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 + a 5 = 20)
  (h2 : a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 + a 6 = 34 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1631_163104


namespace NUMINAMATH_GPT_algebra_ineq_l1631_163107

theorem algebra_ineq (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b + b * c + c * a = 1) : a + b + c ≥ 2 := 
by sorry

end NUMINAMATH_GPT_algebra_ineq_l1631_163107


namespace NUMINAMATH_GPT_find_h_l1631_163120

theorem find_h (j k h : ℕ) (h₁ : 2013 = 3 * h^2 + j) (h₂ : 2014 = 2 * h^2 + k)
  (pos_int_x_intercepts_1 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0))
  (pos_int_x_intercepts_2 : ∃ y1 y2 : ℕ, y1 ≠ y2 ∧ y1 > 0 ∧ y2 > 0 ∧ (2 * (y1 - h)^2 + k = 0 ∧ 2 * (y2 - h)^2 + k = 0)):
  h = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_h_l1631_163120


namespace NUMINAMATH_GPT_books_loaned_out_l1631_163133

theorem books_loaned_out (initial_books loaned_books returned_percentage end_books missing_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : end_books = 66)
  (h3 : returned_percentage = 70)
  (h4 : initial_books - end_books = missing_books)
  (h5 : missing_books = (loaned_books * (100 - returned_percentage)) / 100):
  loaned_books = 30 :=
by
  sorry

end NUMINAMATH_GPT_books_loaned_out_l1631_163133


namespace NUMINAMATH_GPT_other_girl_age_l1631_163159

theorem other_girl_age (x : ℕ) (h1 : 13 + x = 27) : x = 14 := by
  sorry

end NUMINAMATH_GPT_other_girl_age_l1631_163159


namespace NUMINAMATH_GPT_min_value_z_l1631_163176

theorem min_value_z (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 := 
sorry

end NUMINAMATH_GPT_min_value_z_l1631_163176


namespace NUMINAMATH_GPT_range_of_k_l1631_163177

noncomputable def triangle_range (A B C : ℝ) (a b c k : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (B = Real.pi / 3) ∧       -- From arithmetic sequence and solving for B
  a^2 + c^2 = k * b^2 ∧
  (1 < k ∧ k <= 2)

theorem range_of_k (A B C a b c k : ℝ) :
  A + B + C = Real.pi →
  (B = Real.pi - (A + C)) →
  (B = Real.pi / 3) →
  a^2 + c^2 = k * b^2 →
  0 < A ∧ A < 2*Real.pi/3 →
  1 < k ∧ k <= 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1631_163177


namespace NUMINAMATH_GPT_problem_statement_l1631_163122

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_statement (m : ℝ) : (A ∩ (B m) = B m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1631_163122


namespace NUMINAMATH_GPT_area_of_triangle_l1631_163185

theorem area_of_triangle:
  let line1 := λ x => 3 * x - 6
  let line2 := λ x => -2 * x + 18
  let y_axis: ℝ → ℝ := λ _ => 0
  let intersection := (4.8, line1 4.8)
  let y_intercept1 := (0, -6)
  let y_intercept2 := (0, 18)
  (1/2) * 24 * 4.8 = 57.6 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1631_163185


namespace NUMINAMATH_GPT_problem_solution_l1631_163169

theorem problem_solution (x y z : ℝ) (h1 : x * y + y * z + z * x = 4) (h2 : x * y * z = 6) :
  (x * y - (3 / 2) * (x + y)) * (y * z - (3 / 2) * (y + z)) * (z * x - (3 / 2) * (z + x)) = 81 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1631_163169


namespace NUMINAMATH_GPT_sinC_calculation_maxArea_calculation_l1631_163109

noncomputable def sinC_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  Real.sin C

theorem sinC_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2) 
  (h4 : Real.sin B = Real.sqrt 5 / 3) : 
  sinC_given_sides_and_angles A B C a b c h1 h2 h3 = 2 / 3 := by sorry

noncomputable def maxArea_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem maxArea_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2)
  (h4 : Real.sin B = Real.sqrt 5 / 3) 
  (h5 : a * c ≤ 15 / 2) : 
  maxArea_given_sides_and_angles A B C a b c h1 h2 h3 = 5 * Real.sqrt 5 / 4 := by sorry

end NUMINAMATH_GPT_sinC_calculation_maxArea_calculation_l1631_163109


namespace NUMINAMATH_GPT_table_mat_length_l1631_163172

noncomputable def calculate_y (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let y_side := 2 * r * Real.sin (θ / 2)
  y_side

theorem table_mat_length :
  calculate_y 6 8 1 = 3 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_table_mat_length_l1631_163172


namespace NUMINAMATH_GPT_problem_l1631_163162

theorem problem (x : ℝ) (h : x + 2 / x = 4) : - (5 * x) / (x^2 + 2) = -5 / 4 := 
sorry

end NUMINAMATH_GPT_problem_l1631_163162


namespace NUMINAMATH_GPT_fernanda_savings_calculation_l1631_163174

theorem fernanda_savings_calculation :
  ∀ (aryan_debt kyro_debt aryan_payment kyro_payment savings total_savings : ℝ),
    aryan_debt = 1200 ∧
    aryan_debt = 2 * kyro_debt ∧
    aryan_payment = (60 / 100) * aryan_debt ∧
    kyro_payment = (80 / 100) * kyro_debt ∧
    savings = 300 ∧
    total_savings = savings + aryan_payment + kyro_payment →
    total_savings = 1500 := by
    sorry

end NUMINAMATH_GPT_fernanda_savings_calculation_l1631_163174


namespace NUMINAMATH_GPT_ral_age_is_26_l1631_163127

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end NUMINAMATH_GPT_ral_age_is_26_l1631_163127


namespace NUMINAMATH_GPT_find_m_l1631_163113

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (m^2 - 5*m + 7)*x^(m-2)) 
  (h2 : ∀ x, f (-x) = - f x) : 
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1631_163113


namespace NUMINAMATH_GPT_max_area_triangle_l1631_163149

noncomputable def max_area (QA QB QC BC : ℝ) : ℝ :=
  1 / 2 * ((QA^2 + QB^2 - QC^2) / (2 * BC) + 3) * BC

theorem max_area_triangle (QA QB QC BC : ℝ) (hQA : QA = 3) (hQB : QB = 4) (hQC : QC = 5) (hBC : BC = 6) :
  max_area QA QB QC BC = 19 := by
  sorry

end NUMINAMATH_GPT_max_area_triangle_l1631_163149


namespace NUMINAMATH_GPT_value_of_expression_l1631_163128

theorem value_of_expression (a b m n x : ℝ) 
    (hab : a * b = 1) 
    (hmn : m + n = 0) 
    (hxsq : x^2 = 1) : 
    2022 * (m + n) + 2018 * x^2 - 2019 * (a * b) = -1 := 
by 
    sorry

end NUMINAMATH_GPT_value_of_expression_l1631_163128


namespace NUMINAMATH_GPT_compare_a_b_c_l1631_163101

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end NUMINAMATH_GPT_compare_a_b_c_l1631_163101


namespace NUMINAMATH_GPT_number_of_valid_lines_l1631_163117

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def lines_passing_through_point (x_int : ℕ) (y_int : ℕ) (p : ℕ × ℕ) : Prop :=
  p.1 * y_int + p.2 * x_int = x_int * y_int

theorem number_of_valid_lines (p : ℕ × ℕ) : 
  ∃! l : ℕ × ℕ, is_prime (l.1) ∧ is_power_of_two (l.2) ∧ lines_passing_through_point l.1 l.2 p :=
sorry

end NUMINAMATH_GPT_number_of_valid_lines_l1631_163117


namespace NUMINAMATH_GPT_probability_of_at_least_one_die_shows_2_is_correct_l1631_163180

-- Definitions for the conditions
def total_outcomes : ℕ := 64
def neither_die_shows_2_outcomes : ℕ := 49
def favorability (total : ℕ) (exclusion : ℕ) : ℕ := total - exclusion
def favorable_outcomes : ℕ := favorability total_outcomes neither_die_shows_2_outcomes
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Mathematically equivalent proof problem statement
theorem probability_of_at_least_one_die_shows_2_is_correct : 
  probability favorable_outcomes total_outcomes = 15 / 64 :=
sorry

end NUMINAMATH_GPT_probability_of_at_least_one_die_shows_2_is_correct_l1631_163180


namespace NUMINAMATH_GPT_weight_first_watermelon_l1631_163115

-- We define the total weight and the weight of the second watermelon
def total_weight := 14.02
def second_watermelon := 4.11

-- We need to prove that the weight of the first watermelon is 9.91 pounds
theorem weight_first_watermelon : total_weight - second_watermelon = 9.91 := by
  -- Insert mathematical steps here (omitted in this case)
  sorry

end NUMINAMATH_GPT_weight_first_watermelon_l1631_163115


namespace NUMINAMATH_GPT_y_real_for_all_x_l1631_163165

theorem y_real_for_all_x (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_y_real_for_all_x_l1631_163165


namespace NUMINAMATH_GPT_positive_difference_of_squares_l1631_163143

theorem positive_difference_of_squares {x y : ℕ} (hx : x > y) (hxy_sum : x + y = 70) (hxy_diff : x - y = 20) :
  x^2 - y^2 = 1400 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l1631_163143


namespace NUMINAMATH_GPT_annie_accident_chance_l1631_163195

def temperature_effect (temp: ℤ) : ℚ := ((32 - temp) / 3 * 5)

def road_condition_effect (condition: ℚ) : ℚ := condition

def wind_speed_effect (speed: ℤ) : ℚ := if (speed > 20) then ((speed - 20) / 10 * 3) else 0

def skid_chance (temp: ℤ) (condition: ℚ) (speed: ℤ) : ℚ :=
  temperature_effect temp + road_condition_effect condition + wind_speed_effect speed

def accident_chance (skid_chance: ℚ) (tire_effect: ℚ) : ℚ :=
  skid_chance * tire_effect

theorem annie_accident_chance :
  (temperature_effect 8 + road_condition_effect 15 + wind_speed_effect 35) * 0.75 = 43.5 :=
by sorry

end NUMINAMATH_GPT_annie_accident_chance_l1631_163195


namespace NUMINAMATH_GPT_width_of_lawn_is_30_m_l1631_163116

-- Define the conditions
def lawn_length : ℕ := 70
def lawn_width : ℕ := 30
def road_width : ℕ := 5
def gravel_rate_per_sqm : ℕ := 4
def gravel_cost : ℕ := 1900

-- Mathematically equivalent proof problem statement
theorem width_of_lawn_is_30_m 
  (H1 : lawn_length = 70)
  (H2 : road_width = 5)
  (H3 : gravel_rate_per_sqm = 4)
  (H4 : gravel_cost = 1900)
  (H5 : 2*road_width*5 + (lawn_length - road_width) * 5 * gravel_rate_per_sqm = gravel_cost) :
  lawn_width = 30 := 
sorry

end NUMINAMATH_GPT_width_of_lawn_is_30_m_l1631_163116


namespace NUMINAMATH_GPT_compound_interest_amount_l1631_163103

theorem compound_interest_amount 
  (P_si : ℝ := 3225) 
  (R_si : ℝ := 8) 
  (T_si : ℝ := 5) 
  (R_ci : ℝ := 15) 
  (T_ci : ℝ := 2) 
  (SI : ℝ := P_si * R_si * T_si / 100) 
  (CI : ℝ := 2 * SI) 
  (CI_formula : ℝ := P_ci * ((1 + R_ci / 100)^T_ci - 1))
  (P_ci := 516 / 0.3225) :
  P_ci = 1600 := 
by
  sorry

end NUMINAMATH_GPT_compound_interest_amount_l1631_163103


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1631_163110

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1631_163110


namespace NUMINAMATH_GPT_sum_of_digits_l1631_163160

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 4 + 258 = 7 * 100 + b * 10 + 2) (h2 : (7 * 100 + b * 10 + 2) % 3 = 0) :
  a + b = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l1631_163160


namespace NUMINAMATH_GPT_frac_series_simplification_l1631_163186

theorem frac_series_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 : ℚ) / (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2 : ℚ) = 1 / 113 := 
by
  sorry

end NUMINAMATH_GPT_frac_series_simplification_l1631_163186


namespace NUMINAMATH_GPT_text_messages_December_l1631_163111

-- Definitions of the number of text messages sent each month
def text_messages_November := 1
def text_messages_January := 4
def text_messages_February := 8
def doubling_pattern (a b : ℕ) : Prop := b = 2 * a

-- Prove that Jared sent 2 text messages in December
theorem text_messages_December : ∃ x : ℕ, 
  doubling_pattern text_messages_November x ∧ 
  doubling_pattern x text_messages_January ∧ 
  doubling_pattern text_messages_January text_messages_February ∧ 
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_text_messages_December_l1631_163111


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1631_163147

theorem perfect_square_trinomial (k : ℝ) : 
  ∃ a : ℝ, (x^2 - k*x + 1 = (x + a)^2) → (k = 2 ∨ k = -2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1631_163147


namespace NUMINAMATH_GPT_probability_correct_l1631_163191

-- Define the conditions of the problem
def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 5
def total_balls : ℕ := total_white_balls + total_black_balls
def total_ways_draw_two_balls : ℕ := Nat.choose total_balls 2
def ways_choose_one_white_ball : ℕ := Nat.choose total_white_balls 1
def ways_choose_one_black_ball : ℕ := Nat.choose total_black_balls 1
def total_successful_outcomes : ℕ := ways_choose_one_white_ball * ways_choose_one_black_ball

-- Define the probability calculation
def probability_drawing_one_white_one_black : ℚ := total_successful_outcomes / total_ways_draw_two_balls

-- State the theorem
theorem probability_correct :
  probability_drawing_one_white_one_black = 6 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_l1631_163191


namespace NUMINAMATH_GPT_number_of_zeros_f_l1631_163157

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2 * x + 5

theorem number_of_zeros_f : 
  (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ 0 < a ∧ 0 < b ∧ a ≠ b) ∧ ∀ c, f c = 0 → c = a ∨ c = b :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_f_l1631_163157


namespace NUMINAMATH_GPT_jacks_walking_rate_l1631_163196

variable (distance : ℝ) (hours : ℝ) (minutes : ℝ)

theorem jacks_walking_rate (h_distance : distance = 4) (h_hours : hours = 1) (h_minutes : minutes = 15) :
  distance / (hours + minutes / 60) = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_jacks_walking_rate_l1631_163196


namespace NUMINAMATH_GPT_solve_x_l1631_163148

theorem solve_x (x : ℝ) (h : 2 - 2 / (1 - x) = 2 / (1 - x)) : x = -2 := 
by
  sorry

end NUMINAMATH_GPT_solve_x_l1631_163148


namespace NUMINAMATH_GPT_solution_set_inequality_l1631_163138

theorem solution_set_inequality (x : ℝ) : (x^2-2*x-3)*(x^2+1) < 0 ↔ -1 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1631_163138


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1631_163118

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y a : ℝ) : Prop := x + a * y - 2 = 0

def p (a : ℝ) : Prop := ∀ x y : ℝ, line1 x y → line2 x y a
def q (a : ℝ) : Prop := a = -1

theorem necessary_and_sufficient_condition (a : ℝ) : (p a) ↔ (q a) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1631_163118


namespace NUMINAMATH_GPT_soccer_ball_diameter_l1631_163144

theorem soccer_ball_diameter 
  (h : ℝ)
  (s : ℝ)
  (d : ℝ)
  (h_eq : h = 1.25)
  (s_eq : s = 1)
  (d_eq : d = 0.23) : 2 * (d * h / (s - h)) = 0.46 :=
by
  sorry

end NUMINAMATH_GPT_soccer_ball_diameter_l1631_163144


namespace NUMINAMATH_GPT_train_length_l1631_163163

def train_speed_kmph := 25 -- speed of train in km/h
def man_speed_kmph := 2 -- speed of man in km/h
def crossing_time_sec := 52 -- time to cross in seconds

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph -- relative speed in km/h
  let relative_speed_mps := relative_speed_kmph * (5 / 18) -- convert to m/s
  relative_speed_mps * crossing_time_sec -- length of train in meters

theorem train_length : length_of_train = 390 :=
  by sorry -- proof omitted

end NUMINAMATH_GPT_train_length_l1631_163163


namespace NUMINAMATH_GPT_sin_inequality_in_triangle_l1631_163155

theorem sin_inequality_in_triangle (A B C : ℝ) (h_sum : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  Real.sin A * Real.sin (A / 2) + Real.sin B * Real.sin (B / 2) + Real.sin C * Real.sin (C / 2) ≤ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_inequality_in_triangle_l1631_163155


namespace NUMINAMATH_GPT_maximal_q_for_broken_line_l1631_163125

theorem maximal_q_for_broken_line :
  ∃ q : ℝ, (∀ i : ℕ, 0 ≤ i → i < 5 → ∀ A_i : ℝ, (A_i = q ^ i)) ∧ 
  (q = (1 + Real.sqrt 5) / 2) := sorry

end NUMINAMATH_GPT_maximal_q_for_broken_line_l1631_163125


namespace NUMINAMATH_GPT_midpoint_in_polar_coordinates_l1631_163119

theorem midpoint_in_polar_coordinates :
  let A := (9, Real.pi / 3)
  let B := (9, 2 * Real.pi / 3)
  let mid := (Real.sqrt (3) * 9 / 2, Real.pi / 2)
  (mid = (Real.sqrt (3) * 9 / 2, Real.pi / 2)) :=
by 
  sorry

end NUMINAMATH_GPT_midpoint_in_polar_coordinates_l1631_163119


namespace NUMINAMATH_GPT_trigonometric_identity_l1631_163130

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : tan α + 1 / tan α = 10 / 3)
  (h₂ : π / 4 < α ∧ α < π / 2) :
  sin (2 * α + π / 4) + 2 * cos (π / 4) * sin α ^ 2 = 4 * sqrt 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1631_163130


namespace NUMINAMATH_GPT_number_of_subsets_l1631_163168

theorem number_of_subsets (x y : Type) :  ∃ s : Finset (Finset Type), s.card = 4 := 
sorry

end NUMINAMATH_GPT_number_of_subsets_l1631_163168


namespace NUMINAMATH_GPT_min_value_expr_l1631_163184

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ x : ℝ, x = 6 ∧ x = (2 * a + b) / c + (2 * a + c) / b + (2 * b + c) / a :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1631_163184


namespace NUMINAMATH_GPT_geoff_election_l1631_163153

theorem geoff_election (Votes: ℝ) (Percent: ℝ) (ExtraVotes: ℝ) (x: ℝ) 
  (h1 : Votes = 6000) 
  (h2 : Percent = 1) 
  (h3 : ExtraVotes = 3000) 
  (h4 : ReceivedVotes = (Percent / 100) * Votes) 
  (h5 : TotalVotesNeeded = ReceivedVotes + ExtraVotes) 
  (h6 : x = (TotalVotesNeeded / Votes) * 100) :
  x = 51 := 
  by 
    sorry

end NUMINAMATH_GPT_geoff_election_l1631_163153


namespace NUMINAMATH_GPT_find_pair_l1631_163198

noncomputable def x_n (n : ℕ) : ℝ := n / (n + 2016)

theorem find_pair :
  ∃ (m n : ℕ), x_n 2016 = (x_n m) * (x_n n) ∧ (m = 6048 ∧ n = 4032) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pair_l1631_163198


namespace NUMINAMATH_GPT_pencils_to_make_profit_l1631_163131

theorem pencils_to_make_profit
  (total_pencils : ℕ)
  (cost_per_pencil : ℝ)
  (selling_price_per_pencil : ℝ)
  (desired_profit : ℝ)
  (pencils_to_be_sold : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.08 →
  selling_price_per_pencil = 0.20 →
  desired_profit = 160 →
  pencils_to_be_sold = 1600 :=
sorry

end NUMINAMATH_GPT_pencils_to_make_profit_l1631_163131


namespace NUMINAMATH_GPT_total_cost_of_ads_l1631_163188

-- Define the conditions
def cost_ad1 := 3500
def minutes_ad1 := 2
def cost_ad2 := 4500
def minutes_ad2 := 3
def cost_ad3 := 3000
def minutes_ad3 := 3
def cost_ad4 := 4000
def minutes_ad4 := 2
def cost_ad5 := 5500
def minutes_ad5 := 5

-- Define the function to calculate the total cost
def total_cost :=
  (cost_ad1 * minutes_ad1) +
  (cost_ad2 * minutes_ad2) +
  (cost_ad3 * minutes_ad3) +
  (cost_ad4 * minutes_ad4) +
  (cost_ad5 * minutes_ad5)

-- The statement to prove
theorem total_cost_of_ads : total_cost = 66000 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_ads_l1631_163188


namespace NUMINAMATH_GPT_smaller_angle_measure_l1631_163173

theorem smaller_angle_measure (α β : ℝ) (h1 : α + β = 90) (h2 : α = 4 * β) : β = 18 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_measure_l1631_163173


namespace NUMINAMATH_GPT_honda_cars_in_city_l1631_163179

variable (H N : ℕ)

theorem honda_cars_in_city (total_cars : ℕ)
                         (total_red_car_ratio : ℚ)
                         (honda_red_car_ratio : ℚ)
                         (non_honda_red_car_ratio : ℚ)
                         (total_red_cars : ℕ)
                         (h : total_cars = 9000)
                         (h1 : total_red_car_ratio = 0.6)
                         (h2 : honda_red_car_ratio = 0.9)
                         (h3 : non_honda_red_car_ratio = 0.225)
                         (h4 : total_red_cars = 5400)
                         (h5 : H + N = total_cars)
                         (h6 : honda_red_car_ratio * H + non_honda_red_car_ratio * N = total_red_cars) :
  H = 5000 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_honda_cars_in_city_l1631_163179


namespace NUMINAMATH_GPT_theta_in_third_quadrant_l1631_163193

theorem theta_in_third_quadrant (θ : ℝ) (h1 : Real.tan θ > 0) (h2 : Real.sin θ < 0) : 
  ∃ q : ℕ, q = 3 := 
sorry

end NUMINAMATH_GPT_theta_in_third_quadrant_l1631_163193


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1631_163142

noncomputable def fifth_term (x y : ℝ) : ℝ :=
  let a1 := x^2 + y^2
  let a2 := x^2 - y^2
  let a3 := x^2 * y^2
  let a4 := x^2 / y^2
  let d := -2 * y^2
  a4 + d

theorem arithmetic_sequence_fifth_term (x y : ℝ) (hy : y ≠ 0) (hx2 : x ^ 2 = 3 * y ^ 2 / (y ^ 2 - 1)) :
  fifth_term x y = 3 / (y ^ 2 - 1) - 2 * y ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1631_163142


namespace NUMINAMATH_GPT_fraction_simplification_l1631_163189

theorem fraction_simplification : 1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1631_163189


namespace NUMINAMATH_GPT_fraction_paint_left_after_third_day_l1631_163150

noncomputable def original_paint : ℝ := 2
noncomputable def paint_after_first_day : ℝ := original_paint - (1 / 2 * original_paint)
noncomputable def paint_after_second_day : ℝ := paint_after_first_day - (1 / 4 * paint_after_first_day)
noncomputable def paint_after_third_day : ℝ := paint_after_second_day - (1 / 2 * paint_after_second_day)

theorem fraction_paint_left_after_third_day :
  paint_after_third_day / original_ppaint = 3 / 8 :=
sorry

end NUMINAMATH_GPT_fraction_paint_left_after_third_day_l1631_163150


namespace NUMINAMATH_GPT_surface_area_of_given_cube_l1631_163139

-- Define the edge length condition
def edge_length_of_cube (sum_edge_lengths : ℕ) :=
  sum_edge_lengths / 12

-- Define the surface area of a cube given an edge length
def surface_area_of_cube (edge_length : ℕ) :=
  6 * (edge_length * edge_length)

-- State the theorem
theorem surface_area_of_given_cube : 
  edge_length_of_cube 36 = 3 ∧ surface_area_of_cube 3 = 54 :=
by
  -- We leave the proof as an exercise.
  sorry

end NUMINAMATH_GPT_surface_area_of_given_cube_l1631_163139


namespace NUMINAMATH_GPT_locus_of_p_ratio_distances_l1631_163194

theorem locus_of_p_ratio_distances :
  (∀ (P : ℝ × ℝ), (dist P (1, 0) = (1 / 3) * abs (P.1 - 9)) →
  (P.1^2 / 9 + P.2^2 / 8 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_p_ratio_distances_l1631_163194


namespace NUMINAMATH_GPT_solve_system_of_equations_l1631_163146

theorem solve_system_of_equations :
  ∃ x y : ℝ, 
  (4 * x - 3 * y = -0.5) ∧ 
  (5 * x + 7 * y = 10.3) ∧ 
  (|x - 0.6372| < 1e-4) ∧ 
  (|y - 1.0163| < 1e-4) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1631_163146


namespace NUMINAMATH_GPT_binomial_coefficient_10_3_l1631_163132

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_10_3_l1631_163132
