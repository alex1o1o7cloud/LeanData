import Mathlib

namespace NUMINAMATH_GPT_total_games_in_season_l2136_213663

theorem total_games_in_season (teams: ℕ) (division_teams: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) (total_games: ℕ) : 
  teams = 18 → division_teams = 9 → intra_division_games = 3 → inter_division_games = 2 → total_games = 378 :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_season_l2136_213663


namespace NUMINAMATH_GPT_rectangle_area_l2136_213696

theorem rectangle_area (AB AC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) : ∃ Area : ℝ, Area = 120 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2136_213696


namespace NUMINAMATH_GPT_Sahil_purchase_price_l2136_213639

theorem Sahil_purchase_price :
  ∃ P : ℝ, (1.5 * (P + 6000) = 25500) → P = 11000 :=
sorry

end NUMINAMATH_GPT_Sahil_purchase_price_l2136_213639


namespace NUMINAMATH_GPT_fraction_reduction_l2136_213657

theorem fraction_reduction (x y : ℝ) : 
  (4 * x - 4 * y) / (4 * x * 4 * y) = (1 / 4) * ((x - y) / (x * y)) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_reduction_l2136_213657


namespace NUMINAMATH_GPT_necessary_condition_l2136_213693

theorem necessary_condition (x : ℝ) : x = 1 → x^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_l2136_213693


namespace NUMINAMATH_GPT_table_legs_l2136_213609

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end NUMINAMATH_GPT_table_legs_l2136_213609


namespace NUMINAMATH_GPT_max_area_rectangle_l2136_213641

theorem max_area_rectangle (P : ℝ) (hP : P = 60) (a b : ℝ) (h1 : b = 3 * a) (h2 : 2 * a + 2 * b = P) : a * b = 168.75 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l2136_213641


namespace NUMINAMATH_GPT_integer_points_on_line_l2136_213689

/-- Given a line that passes through points C(3, 3) and D(150, 250),
prove that the number of other points with integer coordinates
that lie strictly between C and D is 48. -/
theorem integer_points_on_line {C D : ℝ × ℝ} (hC : C = (3, 3)) (hD : D = (150, 250)) :
  ∃ (n : ℕ), n = 48 ∧ 
  ∀ p : ℝ × ℝ, C.1 < p.1 ∧ p.1 < D.1 ∧ 
  C.2 < p.2 ∧ p.2 < D.2 → 
  (∃ (k : ℤ), p.1 = ↑k ∧ p.2 = (5/3) * p.1 - 2) :=
sorry

end NUMINAMATH_GPT_integer_points_on_line_l2136_213689


namespace NUMINAMATH_GPT_option_B_equals_six_l2136_213612

theorem option_B_equals_six :
  (3 - (-3)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_option_B_equals_six_l2136_213612


namespace NUMINAMATH_GPT_depth_of_melted_ice_cream_l2136_213630

theorem depth_of_melted_ice_cream
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ)
  (h : ℝ)
  (sphere_volume_eq : V_sphere = (4 / 3) * Real.pi * r_sphere^3)
  (cylinder_volume_eq : V_sphere = Real.pi * r_cylinder^2 * h)
  (r_sphere_eq : r_sphere = 3)
  (r_cylinder_eq : r_cylinder = 9)
  : h = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_depth_of_melted_ice_cream_l2136_213630


namespace NUMINAMATH_GPT_treaty_signed_on_saturday_l2136_213686

-- Define the start day and the total days until the treaty.
def start_day_of_week : Nat := 4 -- Thursday is the 4th day (0 = Sunday, ..., 6 = Saturday)
def days_until_treaty : Nat := 919

-- Calculate the final day of the week after 919 days since start_day_of_week.
def treaty_day_of_week : Nat := (start_day_of_week + days_until_treaty) % 7

-- The goal is to prove that the treaty was signed on a Saturday.
theorem treaty_signed_on_saturday : treaty_day_of_week = 6 :=
by
  -- Implement the proof steps
  sorry

end NUMINAMATH_GPT_treaty_signed_on_saturday_l2136_213686


namespace NUMINAMATH_GPT_sam_age_two_years_ago_l2136_213640

theorem sam_age_two_years_ago (J S : ℕ) (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9)) : S - 2 = 7 :=
sorry

end NUMINAMATH_GPT_sam_age_two_years_ago_l2136_213640


namespace NUMINAMATH_GPT_line_parallel_xaxis_l2136_213655

theorem line_parallel_xaxis (x y : ℝ) : y = 2 ↔ (∃ a b : ℝ, a = 4 ∧ b = 2 ∧ y = 2) :=
by 
  sorry

end NUMINAMATH_GPT_line_parallel_xaxis_l2136_213655


namespace NUMINAMATH_GPT_substitution_result_l2136_213611

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_substitution_result_l2136_213611


namespace NUMINAMATH_GPT_initial_books_l2136_213671

theorem initial_books (sold_books : ℕ) (given_books : ℕ) (remaining_books : ℕ) 
                      (h1 : sold_books = 11)
                      (h2 : given_books = 35)
                      (h3 : remaining_books = 62) :
  (sold_books + given_books + remaining_books = 108) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_initial_books_l2136_213671


namespace NUMINAMATH_GPT_cube_volume_equality_l2136_213618

open BigOperators Real

-- Definitions
def initial_volume : ℝ := 1

def removed_volume (x : ℝ) : ℝ := x^2

def removed_volume_with_overlap (x y : ℝ) : ℝ := x^2 - (x^2 * y)

def remaining_volume (a b c : ℝ) : ℝ := 
  initial_volume - removed_volume c - removed_volume_with_overlap b c - removed_volume_with_overlap a c - removed_volume_with_overlap a b + (c^2 * b)

-- Main theorem to prove
theorem cube_volume_equality (c b a : ℝ) (hcb : c < b) (hba : b < a) (ha1 : a < 1):
  (c = 1 / 2) ∧ 
  (b = (1 + Real.sqrt 17) / 8) ∧ 
  (a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64) :=
sorry

end NUMINAMATH_GPT_cube_volume_equality_l2136_213618


namespace NUMINAMATH_GPT_part1_part2_l2136_213606

theorem part1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (1 - 4 / (2 * a^0 + a)) = 0) : a = 2 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x : ℝ, (2^x + 1) * (1 - 2 / (2^x + 1)) + k = 0) : k < 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2136_213606


namespace NUMINAMATH_GPT_quotient_of_division_l2136_213670

theorem quotient_of_division (dividend divisor remainder : ℕ) (h_dividend : dividend = 127) (h_divisor : divisor = 14) (h_remainder : remainder = 1) :
  (dividend - remainder) / divisor = 9 :=
by 
  -- Proof follows
  sorry

end NUMINAMATH_GPT_quotient_of_division_l2136_213670


namespace NUMINAMATH_GPT_Vasya_numbers_l2136_213644

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_Vasya_numbers_l2136_213644


namespace NUMINAMATH_GPT_total_players_l2136_213694

-- Definitions based on problem conditions.
def players_kabadi : Nat := 10
def players_kho_kho_only : Nat := 20
def players_both_games : Nat := 5

-- Proof statement for the total number of players.
theorem total_players : (players_kabadi + players_kho_kho_only - players_both_games) = 25 := by
  sorry

end NUMINAMATH_GPT_total_players_l2136_213694


namespace NUMINAMATH_GPT_ordering_of_a_b_c_l2136_213646

theorem ordering_of_a_b_c (a b c : ℝ)
  (ha : a = Real.exp (1 / 2))
  (hb : b = Real.log (1 / 2))
  (hc : c = Real.sin (1 / 2)) :
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_GPT_ordering_of_a_b_c_l2136_213646


namespace NUMINAMATH_GPT_rain_in_both_areas_l2136_213684

variable (P1 P2 : ℝ)
variable (hP1 : 0 < P1 ∧ P1 < 1)
variable (hP2 : 0 < P2 ∧ P2 < 1)

theorem rain_in_both_areas :
  ∀ P1 P2, (0 < P1 ∧ P1 < 1) → (0 < P2 ∧ P2 < 1) → (1 - P1) * (1 - P2) = (1 - P1) * (1 - P2) :=
by
  intros P1 P2 hP1 hP2
  sorry

end NUMINAMATH_GPT_rain_in_both_areas_l2136_213684


namespace NUMINAMATH_GPT_emily_olivia_books_l2136_213662

theorem emily_olivia_books (shared_books total_books_emily books_olivia_not_in_emily : ℕ)
  (h1 : shared_books = 15)
  (h2 : total_books_emily = 23)
  (h3 : books_olivia_not_in_emily = 8) : (total_books_emily - shared_books + books_olivia_not_in_emily = 16) :=
by
  sorry

end NUMINAMATH_GPT_emily_olivia_books_l2136_213662


namespace NUMINAMATH_GPT_new_socks_bought_l2136_213632

-- Given conditions:
def initial_socks : ℕ := 11
def socks_thrown_away : ℕ := 4
def final_socks : ℕ := 33

-- Theorem proof statement:
theorem new_socks_bought : (final_socks - (initial_socks - socks_thrown_away)) = 26 :=
by
  sorry

end NUMINAMATH_GPT_new_socks_bought_l2136_213632


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_3_l2136_213660

open Polynomial

noncomputable def p : ℝ[X] := 4 * X^3 - 12 * X^2 + 16 * X - 20

theorem remainder_when_divided_by_x_minus_3 : eval 3 p = 28 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_3_l2136_213660


namespace NUMINAMATH_GPT_range_of_a_l2136_213652

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end NUMINAMATH_GPT_range_of_a_l2136_213652


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2136_213665

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 5*x + 6 > 0) ↔ (x < -3 ∨ x > -2) :=
  by
    sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2136_213665


namespace NUMINAMATH_GPT_team_team_count_correct_l2136_213645

/-- Number of ways to select a team of three students from 20,
    one for each subject: math, Russian language, and informatics. -/
def ways_to_form_team (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

theorem team_team_count_correct : ways_to_form_team 20 = 6840 :=
by sorry

end NUMINAMATH_GPT_team_team_count_correct_l2136_213645


namespace NUMINAMATH_GPT_waiter_tips_earned_l2136_213661

theorem waiter_tips_earned (total_customers tips_left no_tip_customers tips_per_customer : ℕ) :
  no_tip_customers + tips_left = total_customers ∧ tips_per_customer = 3 ∧ no_tip_customers = 5 ∧ total_customers = 7 → 
  tips_left * tips_per_customer = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_waiter_tips_earned_l2136_213661


namespace NUMINAMATH_GPT_circle_radius_5_l2136_213643

theorem circle_radius_5 (k x y : ℝ) : x^2 + 8 * x + y^2 + 10 * y - k = 0 → (x + 4) ^ 2 + (y + 5) ^ 2 = 25 → k = -16 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_5_l2136_213643


namespace NUMINAMATH_GPT_chebyshev_birth_year_l2136_213627

theorem chebyshev_birth_year :
  ∃ (a b : ℕ),
  a > b ∧ 
  a + b = 3 ∧ 
  (1821 = 1800 + 10 * a + 1 * b) ∧
  (1821 + 73) < 1900 :=
by sorry

end NUMINAMATH_GPT_chebyshev_birth_year_l2136_213627


namespace NUMINAMATH_GPT_average_calls_per_day_l2136_213687

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end NUMINAMATH_GPT_average_calls_per_day_l2136_213687


namespace NUMINAMATH_GPT_quadratic_function_m_value_l2136_213691

theorem quadratic_function_m_value :
  ∃ m : ℝ, (m - 3 ≠ 0) ∧ (m^2 - 7 = 2) ∧ m = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_m_value_l2136_213691


namespace NUMINAMATH_GPT_max_abs_sum_l2136_213628

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_sum_l2136_213628


namespace NUMINAMATH_GPT_solve_for_nabla_l2136_213634

theorem solve_for_nabla : (∃ (nabla : ℤ), 5 * (-3) + 4 = nabla + 7) → (∃ (nabla : ℤ), nabla = -18) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_nabla_l2136_213634


namespace NUMINAMATH_GPT_Duke_three_pointers_impossible_l2136_213666

theorem Duke_three_pointers_impossible (old_record : ℤ)
  (points_needed_to_tie : ℤ)
  (points_broken_record : ℤ)
  (free_throws : ℕ)
  (regular_baskets : ℕ)
  (three_pointers : ℕ)
  (normal_three_pointers_per_game : ℕ)
  (max_attempts : ℕ)
  (last_minutes : ℕ)
  (points_per_free_throw : ℤ)
  (points_per_regular_basket : ℤ)
  (points_per_three_pointer : ℤ) :
  free_throws = 5 → regular_baskets = 4 → normal_three_pointers_per_game = 2 → max_attempts = 10 → 
  points_per_free_throw = 1 → points_per_regular_basket = 2 → points_per_three_pointer = 3 →
  old_record = 257 → points_needed_to_tie = 17 → points_broken_record = 5 →
  (free_throws + regular_baskets + three_pointers ≤ max_attempts) →
  last_minutes = 6 → 
  ¬(free_throws + regular_baskets + (points_needed_to_tie + points_broken_record - 
  (free_throws * points_per_free_throw + regular_baskets * points_per_regular_basket)) / points_per_three_pointer ≤ max_attempts) := sorry

end NUMINAMATH_GPT_Duke_three_pointers_impossible_l2136_213666


namespace NUMINAMATH_GPT_rational_solutions_quadratic_l2136_213680

theorem rational_solutions_quadratic (k : ℕ) (h_pos : 0 < k) :
  (∃ (x : ℚ), k * x^2 + 24 * x + k = 0) ↔ k = 12 :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_quadratic_l2136_213680


namespace NUMINAMATH_GPT_minimum_value_expression_l2136_213617

theorem minimum_value_expression {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : 
  a^2 + 4 * a * b + 9 * b^2 + 3 * b * c + c^2 ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l2136_213617


namespace NUMINAMATH_GPT_value_of_f_l2136_213672

variable {x t : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0
  else (1 : ℝ) / x

theorem value_of_f (h1 : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
                   (h2 : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  f (Real.tan t ^ 2 + 1) = Real.sin (2 * t) ^ 2 / 4 :=
sorry

end NUMINAMATH_GPT_value_of_f_l2136_213672


namespace NUMINAMATH_GPT_total_tomato_seeds_l2136_213613

theorem total_tomato_seeds (morn_mike morn_morning ted_morning sarah_morning : ℕ)
    (aft_mike aft_ted aft_sarah : ℕ)
    (H1 : morn_mike = 50)
    (H2 : ted_morning = 2 * morn_mike)
    (H3 : sarah_morning = morn_mike + 30)
    (H4 : aft_mike = 60)
    (H5 : aft_ted = aft_mike - 20)
    (H6 : aft_sarah = sarah_morning + 20) :
    morn_mike + aft_mike + ted_morning + aft_ted + sarah_morning + aft_sarah = 430 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  sorry

end NUMINAMATH_GPT_total_tomato_seeds_l2136_213613


namespace NUMINAMATH_GPT_number_of_three_digit_multiples_of_6_l2136_213614

theorem number_of_three_digit_multiples_of_6 : 
  let lower_bound := 100
  let upper_bound := 999
  let multiple := 6
  let smallest_n := Nat.ceil (100 / multiple)
  let largest_n := Nat.floor (999 / multiple)
  let count_multiples := largest_n - smallest_n + 1
  count_multiples = 150 := by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_multiples_of_6_l2136_213614


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2136_213605

variable (a b : ℝ)

theorem necessary_but_not_sufficient : 
  ¬ (a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) ∧ (a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2136_213605


namespace NUMINAMATH_GPT_num_regions_of_lines_l2136_213648

theorem num_regions_of_lines (R : ℕ → ℕ) :
  R 1 = 2 ∧ 
  (∀ n, R (n + 1) = R n + (n + 1)) →
  (∀ n, R n = (n * (n + 1)) / 2 + 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_num_regions_of_lines_l2136_213648


namespace NUMINAMATH_GPT_solve_inequality_l2136_213668

theorem solve_inequality (x : ℝ) : abs ((3 - x) / 4) < 1 ↔ 2 < x ∧ x < 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l2136_213668


namespace NUMINAMATH_GPT_arccos_one_over_sqrt_two_eq_pi_over_four_l2136_213631

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_over_sqrt_two_eq_pi_over_four_l2136_213631


namespace NUMINAMATH_GPT_split_enthusiasts_into_100_sections_l2136_213690

theorem split_enthusiasts_into_100_sections :
  ∃ (sections : Fin 100 → Set ℕ),
    (∀ i, sections i ≠ ∅) ∧
    (∀ i j, i ≠ j → sections i ∩ sections j = ∅) ∧
    (⋃ i, sections i) = {n : ℕ | n < 5000} :=
sorry

end NUMINAMATH_GPT_split_enthusiasts_into_100_sections_l2136_213690


namespace NUMINAMATH_GPT_roots_of_equation_l2136_213620

theorem roots_of_equation :
  (∃ (x_1 x_2 : ℝ), x_1 > x_2 ∧ (∀ x, x^2 - |x-1| - 1 = 0 ↔ x = x_1 ∨ x = x_2)) :=
sorry

end NUMINAMATH_GPT_roots_of_equation_l2136_213620


namespace NUMINAMATH_GPT_thomas_worked_hours_l2136_213669

theorem thomas_worked_hours (Toby Thomas Rebecca : ℕ) 
  (h_total : Thomas + Toby + Rebecca = 157) 
  (h_toby : Toby = 2 * Thomas - 10) 
  (h_rebecca_1 : Rebecca = Toby - 8) 
  (h_rebecca_2 : Rebecca = 56) : Thomas = 37 :=
by
  sorry

end NUMINAMATH_GPT_thomas_worked_hours_l2136_213669


namespace NUMINAMATH_GPT_initial_walnuts_l2136_213602

theorem initial_walnuts (W : ℕ) (boy_effective : ℕ) (girl_effective : ℕ) (total_walnuts : ℕ) :
  boy_effective = 5 → girl_effective = 3 → total_walnuts = 20 → W + boy_effective + girl_effective = total_walnuts → W = 12 :=
by
  intros h_boy h_girl h_total h_eq
  rw [h_boy, h_girl, h_total] at h_eq
  linarith

end NUMINAMATH_GPT_initial_walnuts_l2136_213602


namespace NUMINAMATH_GPT_only_pairs_satisfying_conditions_l2136_213619

theorem only_pairs_satisfying_conditions (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (b^2 + b + 1) % a = 0 ∧ (a^2 + a + 1) % b = 0 → a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_pairs_satisfying_conditions_l2136_213619


namespace NUMINAMATH_GPT_find_x_l2136_213622

theorem find_x (x : ℕ) : (x % 9 = 0) ∧ (x^2 > 144) ∧ (x < 30) → (x = 18 ∨ x = 27) :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l2136_213622


namespace NUMINAMATH_GPT_total_annual_interest_l2136_213658

def total_amount : ℝ := 4000
def P1 : ℝ := 2800
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

def P2 : ℝ := total_amount - P1
def I1 : ℝ := P1 * Rate1
def I2 : ℝ := P2 * Rate2
def I_total : ℝ := I1 + I2

theorem total_annual_interest : I_total = 144 := by
  sorry

end NUMINAMATH_GPT_total_annual_interest_l2136_213658


namespace NUMINAMATH_GPT_cannot_determine_red_marbles_l2136_213675

variable (Jason_blue : ℕ) (Tom_blue : ℕ) (Total_blue : ℕ)

-- Conditions
axiom Jason_has_44_blue : Jason_blue = 44
axiom Tom_has_24_blue : Tom_blue = 24
axiom Together_have_68_blue : Total_blue = 68

theorem cannot_determine_red_marbles (Jason_blue Tom_blue Total_blue : ℕ) : ¬ ∃ (Jason_red : ℕ), True := by
  sorry

end NUMINAMATH_GPT_cannot_determine_red_marbles_l2136_213675


namespace NUMINAMATH_GPT_june_eggs_count_l2136_213698

theorem june_eggs_count :
  (2 * 5) + 3 + 4 = 17 := 
by 
  sorry

end NUMINAMATH_GPT_june_eggs_count_l2136_213698


namespace NUMINAMATH_GPT_factorize_expression_l2136_213695

theorem factorize_expression (a : ℝ) : 
  a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l2136_213695


namespace NUMINAMATH_GPT_find_limpet_shells_l2136_213651

variable (L L_shells E_shells J_shells totalShells : ℕ)

def Ed_and_Jacob_initial_shells := 2
def Ed_oyster_shells := 2
def Ed_conch_shells := 4
def Jacob_more_shells := 2
def total_shells := 30

def Ed_total_shells := L + Ed_oyster_shells + Ed_conch_shells
def Jacob_total_shells := Ed_total_shells + Jacob_more_shells

theorem find_limpet_shells
  (H : Ed_and_Jacob_initial_shells + Ed_total_shells + Jacob_total_shells = total_shells) :
  L = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_limpet_shells_l2136_213651


namespace NUMINAMATH_GPT_limestone_amount_l2136_213604

theorem limestone_amount (L S : ℝ) (h1 : L + S = 100) (h2 : 3 * L + 5 * S = 425) : L = 37.5 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_limestone_amount_l2136_213604


namespace NUMINAMATH_GPT_more_likely_condition_l2136_213608

-- Definitions for the problem
def total_placements (n : ℕ) := n * n * (n * n - 1)

def not_same_intersection_placements (n : ℕ) := n * n * (n * n - 1)

def same_row_or_column_exclusions (n : ℕ) := 2 * n * (n - 1) * n

def not_same_street_placements (n : ℕ) := total_placements n - same_row_or_column_exclusions n

def probability_not_same_intersection (n : ℕ) := not_same_intersection_placements n / total_placements n

def probability_not_same_street (n : ℕ) := not_same_street_placements n / total_placements n

-- Main proposition
theorem more_likely_condition (n : ℕ) (h : n = 7) :
  probability_not_same_intersection n > probability_not_same_street n := 
by 
  sorry

end NUMINAMATH_GPT_more_likely_condition_l2136_213608


namespace NUMINAMATH_GPT_evaluate_expression_l2136_213629

open Nat

theorem evaluate_expression : 
  (3 * 4 * 5 * 6) * (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) = 342 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2136_213629


namespace NUMINAMATH_GPT_pen_tip_movement_l2136_213653

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end NUMINAMATH_GPT_pen_tip_movement_l2136_213653


namespace NUMINAMATH_GPT_average_rate_of_change_l2136_213682

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l2136_213682


namespace NUMINAMATH_GPT_gingerbread_to_bagels_l2136_213677

theorem gingerbread_to_bagels (gingerbread drying_rings bagels : ℕ) 
  (h1 : gingerbread = 1 → drying_rings = 6) 
  (h2 : drying_rings = 9 → bagels = 4) 
  (h3 : gingerbread = 3) : bagels = 8 :=
by
  sorry

end NUMINAMATH_GPT_gingerbread_to_bagels_l2136_213677


namespace NUMINAMATH_GPT_greatest_whole_number_solution_l2136_213601

theorem greatest_whole_number_solution :
  ∃ (x : ℕ), (5 * x - 4 < 3 - 2 * x) ∧ ∀ (y : ℕ), (5 * y - 4 < 3 - 2 * y) → y ≤ x ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_greatest_whole_number_solution_l2136_213601


namespace NUMINAMATH_GPT_proof_x_squared_plus_y_squared_l2136_213683

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end NUMINAMATH_GPT_proof_x_squared_plus_y_squared_l2136_213683


namespace NUMINAMATH_GPT_triangular_difference_l2136_213674

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Main theorem: the difference between the 30th and 29th triangular numbers is 30 -/
theorem triangular_difference : triangular 30 - triangular 29 = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangular_difference_l2136_213674


namespace NUMINAMATH_GPT_cube_expansion_l2136_213656

theorem cube_expansion : 101^3 + 3 * 101^2 + 3 * 101 + 1 = 1061208 :=
by
  sorry

end NUMINAMATH_GPT_cube_expansion_l2136_213656


namespace NUMINAMATH_GPT_sequence_a_n_l2136_213688

theorem sequence_a_n {n : ℕ} (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = (2/3 : ℚ) * n^2 - (1/3 : ℚ) * n)
  (ha : ∀ n, a n = if n = 1 then S n else S n - S (n - 1)) :
  ∀ n, a n = (4/3 : ℚ) * n - 1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_l2136_213688


namespace NUMINAMATH_GPT_ratio_second_part_l2136_213654

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end NUMINAMATH_GPT_ratio_second_part_l2136_213654


namespace NUMINAMATH_GPT_initial_population_l2136_213650

theorem initial_population (P : ℝ) (h : 0.72 * P = 3168) : P = 4400 :=
sorry

end NUMINAMATH_GPT_initial_population_l2136_213650


namespace NUMINAMATH_GPT_line_through_points_l2136_213642

theorem line_through_points (m n p : ℝ) 
  (h1 : m = 4 * n + 5) 
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_line_through_points_l2136_213642


namespace NUMINAMATH_GPT_find_abc_value_l2136_213600

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : a * b = 30)
variable (h5 : b * c = 54)
variable (h6 : c * a = 45)

theorem find_abc_value : a * b * c = 270 := by
  sorry

end NUMINAMATH_GPT_find_abc_value_l2136_213600


namespace NUMINAMATH_GPT_no_even_threes_in_circle_l2136_213679

theorem no_even_threes_in_circle (arr : ℕ → ℕ) (h1 : ∀ i, 1 ≤ arr i ∧ arr i ≤ 2017)
  (h2 : ∀ i, (arr i + arr ((i + 1) % 2017) + arr ((i + 2) % 2017)) % 2 = 0) : false :=
sorry

end NUMINAMATH_GPT_no_even_threes_in_circle_l2136_213679


namespace NUMINAMATH_GPT_garden_area_l2136_213621

def radius : ℝ := 0.6
def pi_approx : ℝ := 3
def circle_area (r : ℝ) (π : ℝ) := π * r^2

theorem garden_area : circle_area radius pi_approx = 1.08 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_l2136_213621


namespace NUMINAMATH_GPT_find_center_angle_l2136_213635

noncomputable def pi : ℝ := Real.pi
/-- Given conditions from the math problem -/
def radius : ℝ := 12
def area : ℝ := 67.88571428571429

theorem find_center_angle (θ : ℝ) 
  (area_def : area = (θ / 360) * pi * radius ^ 2) : 
  θ = 54 :=
sorry

end NUMINAMATH_GPT_find_center_angle_l2136_213635


namespace NUMINAMATH_GPT_graphene_scientific_notation_l2136_213678

def scientific_notation (n : ℝ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * 10 ^ exp ∧ 1 ≤ abs a ∧ abs a < 10

theorem graphene_scientific_notation :
  scientific_notation 0.00000000034 3.4 (-10) :=
by {
  sorry
}

end NUMINAMATH_GPT_graphene_scientific_notation_l2136_213678


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l2136_213625

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l2136_213625


namespace NUMINAMATH_GPT_max_minus_min_depends_on_a_not_b_l2136_213676

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a * x + b

theorem max_minus_min_depends_on_a_not_b (a b : ℝ) :
  let f := quadratic_function a b
  let M := max (f 0) (f 1)
  let m := min (f 0) (f 1)
  M - m == |a| :=
sorry

end NUMINAMATH_GPT_max_minus_min_depends_on_a_not_b_l2136_213676


namespace NUMINAMATH_GPT_rem_fraction_of_66_l2136_213667

noncomputable def n : ℝ := 22.142857142857142
noncomputable def s : ℝ := n + 5
noncomputable def p : ℝ := s * 7
noncomputable def q : ℝ := p / 5
noncomputable def r : ℝ := q - 5

theorem rem_fraction_of_66 : r = 33 ∧ r / 66 = 1 / 2 := by 
  sorry

end NUMINAMATH_GPT_rem_fraction_of_66_l2136_213667


namespace NUMINAMATH_GPT_rectangle_area_ratio_k_l2136_213624

theorem rectangle_area_ratio_k (d : ℝ) (l w : ℝ) (h1 : l / w = 5 / 2) (h2 : d^2 = l^2 + w^2) :
  ∃ k : ℝ, k = 10 / 29 ∧ (l * w = k * d^2) :=
by {
  -- proof steps will go here
  sorry
}

end NUMINAMATH_GPT_rectangle_area_ratio_k_l2136_213624


namespace NUMINAMATH_GPT_possible_values_of_sum_of_reciprocals_l2136_213699

theorem possible_values_of_sum_of_reciprocals {a b : ℝ} (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_possible_values_of_sum_of_reciprocals_l2136_213699


namespace NUMINAMATH_GPT_general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l2136_213673

section ArithmeticSequence

-- Given conditions
def a1 : Int := 13
def a4 : Int := 7
def d : Int := (a4 - a1) / 3

-- General formula for a_n
def a_n (n : Int) : Int := a1 + (n - 1) * d

-- Sum of the first n terms S_n
def S_n (n : Int) : Int := n * (a1 + a_n n) / 2

-- Maximum value of S_n and corresponding term
def S_max : Int := 49
def n_max_S : Int := 7

-- Sum of the absolute values of the first n terms T_n
def T_n (n : Int) : Int :=
  if n ≤ 7 then n^2 + 12 * n
  else 98 - 12 * n - n^2

-- Statements to prove
theorem general_formula (n : Int) : a_n n = 15 - 2 * n := sorry

theorem sum_of_first_n_terms (n : Int) : S_n n = 14 * n - n^2 := sorry

theorem max_sum_of_S_n : (S_n n_max_S = S_max) := sorry

theorem sum_of_absolute_values (n : Int) : T_n n = 
  if n ≤ 7 then n^2 + 12 * n else 98 - 12 * n - n^2 := sorry

end ArithmeticSequence

end NUMINAMATH_GPT_general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l2136_213673


namespace NUMINAMATH_GPT_ratio_of_e_to_l_l2136_213607

-- Define the conditions
def e (S : ℕ) : ℕ := 4 * S
def l (S : ℕ) : ℕ := 8 * S

-- Prove the main statement
theorem ratio_of_e_to_l (S : ℕ) (h_e : e S = 4 * S) (h_l : l S = 8 * S) : e S / gcd (e S) (l S) / l S / gcd (e S) (l S) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_e_to_l_l2136_213607


namespace NUMINAMATH_GPT_tiles_needed_l2136_213615

def hallway_length : ℕ := 14
def hallway_width : ℕ := 20
def border_tile_side : ℕ := 2
def interior_tile_side : ℕ := 3

theorem tiles_needed :
  let border_length_tiles := ((hallway_length - 2 * border_tile_side) / border_tile_side) * 2
  let border_width_tiles := ((hallway_width - 2 * border_tile_side) / border_tile_side) * 2
  let corner_tiles := 4
  let total_border_tiles := border_length_tiles + border_width_tiles + corner_tiles
  let interior_length := hallway_length - 2 * border_tile_side
  let interior_width := hallway_width - 2 * border_tile_side
  let interior_area := interior_length * interior_width
  let interior_tiles_needed := (interior_area + interior_tile_side * interior_tile_side - 1) / (interior_tile_side * interior_tile_side)
  total_border_tiles + interior_tiles_needed = 48 := 
by {
  sorry
}

end NUMINAMATH_GPT_tiles_needed_l2136_213615


namespace NUMINAMATH_GPT_not_777_integers_l2136_213637

theorem not_777_integers (p : ℕ) (hp : Nat.Prime p) :
  ¬ (∃ count : ℕ, count = 777 ∧ ∀ n : ℕ, ∃ k : ℕ, (n ^ 3 + n * p + 1 = k * (n + p + 1))) :=
by
  sorry

end NUMINAMATH_GPT_not_777_integers_l2136_213637


namespace NUMINAMATH_GPT_order_of_activities_l2136_213664

noncomputable def fraction_liking_activity_dodgeball : ℚ := 8 / 24
noncomputable def fraction_liking_activity_barbecue : ℚ := 10 / 30
noncomputable def fraction_liking_activity_archery : ℚ := 9 / 18

theorem order_of_activities :
  (fraction_liking_activity_archery > fraction_liking_activity_dodgeball) ∧
  (fraction_liking_activity_archery > fraction_liking_activity_barbecue) ∧
  (fraction_liking_activity_dodgeball = fraction_liking_activity_barbecue) :=
by
  sorry

end NUMINAMATH_GPT_order_of_activities_l2136_213664


namespace NUMINAMATH_GPT_volume_of_prism_l2136_213633

theorem volume_of_prism (x y z : ℝ) (hx : x * y = 28) (hy : x * z = 45) (hz : y * z = 63) : x * y * z = 282 := by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l2136_213633


namespace NUMINAMATH_GPT_super12_teams_l2136_213685

theorem super12_teams :
  ∃ n : ℕ, (n * (n - 1) = 132) ∧ n = 12 := by
  sorry

end NUMINAMATH_GPT_super12_teams_l2136_213685


namespace NUMINAMATH_GPT_safer_four_engine_airplane_l2136_213692

theorem safer_four_engine_airplane (P : ℝ) (hP : 0 < P ∧ P < 1):
  (∃ p : ℝ, p = 1 - P ∧ (p^4 + 4 * p^3 * (1 - p) + 6 * p^2 * (1 - p)^2 > p^2 + 2 * p * (1 - p) ↔ P > 2 / 3)) :=
sorry

end NUMINAMATH_GPT_safer_four_engine_airplane_l2136_213692


namespace NUMINAMATH_GPT_find_f_10_l2136_213610

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end NUMINAMATH_GPT_find_f_10_l2136_213610


namespace NUMINAMATH_GPT_right_triangle_area_l2136_213681

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end NUMINAMATH_GPT_right_triangle_area_l2136_213681


namespace NUMINAMATH_GPT_student_rank_from_right_l2136_213603

theorem student_rank_from_right (n m : ℕ) (h1 : n = 8) (h2 : m = 20) : m - (n - 1) = 13 :=
by
  sorry

end NUMINAMATH_GPT_student_rank_from_right_l2136_213603


namespace NUMINAMATH_GPT_find_x_y_l2136_213623

theorem find_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : (x * y / 7) ^ (3 / 2) = x) 
  (h2 : (x * y / 7) = y) : 
  x = 7 ∧ y = 7 ^ (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l2136_213623


namespace NUMINAMATH_GPT_knights_probability_l2136_213636

theorem knights_probability :
  let knights : Nat := 30
  let chosen : Nat := 4
  let probability (n k : Nat) := 1 - (((n - k + 1) * (n - k - 1) * (n - k - 3) * (n - k - 5)) / 
                                      ((n - 0) * (n - 1) * (n - 2) * (n - 3)))
  probability knights chosen = (389 / 437) := sorry

end NUMINAMATH_GPT_knights_probability_l2136_213636


namespace NUMINAMATH_GPT_mod_inverse_3_40_l2136_213647

theorem mod_inverse_3_40 : 3 * 27 % 40 = 1 := by
  sorry

end NUMINAMATH_GPT_mod_inverse_3_40_l2136_213647


namespace NUMINAMATH_GPT_tangent_through_points_l2136_213659

theorem tangent_through_points :
  ∀ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ, y₁ = x₁^2 + 1 → y₂ = x₂^2 + 1 → 
    (2 * x₁ * (x₂ - x₁) + y₁ = 0 → x₂ = -x₁) ∧ 
    (2 * x₂ * (x₁ - x₂) + y₂ = 0 → x₁ = -x₂)) →
  (x₁ = 1 / Real.sqrt 3 ∧ x₂ = -1 / Real.sqrt 3 ∧
   (x₁^2 + 1 = (1 / 3) + 1) ∧ (x₂^2 + 1 = (1 / 3) + 1)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_through_points_l2136_213659


namespace NUMINAMATH_GPT_polynomial_factor_pq_l2136_213638

theorem polynomial_factor_pq (p q : ℝ) (h : ∀ x : ℝ, (x^2 + 2*x + 5) ∣ (x^4 + p*x^2 + q)) : p + q = 31 :=
sorry

end NUMINAMATH_GPT_polynomial_factor_pq_l2136_213638


namespace NUMINAMATH_GPT_common_chord_line_l2136_213649

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem common_chord_line : 
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - y + 1 = 0) := 
by sorry

end NUMINAMATH_GPT_common_chord_line_l2136_213649


namespace NUMINAMATH_GPT_find_quadratic_function_l2136_213697

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

theorem find_quadratic_function (a b : ℝ) :
  (∀ x, (quadratic_function a b (quadratic_function a b x - x)) / (quadratic_function a b x) = x^2 + 2023 * x + 1777) →
  a = 2025 ∧ b = 249 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l2136_213697


namespace NUMINAMATH_GPT_condition_is_necessary_but_not_sufficient_l2136_213616

noncomputable def sequence_satisfies_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 7 = 2 * a 5

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem condition_is_necessary_but_not_sufficient (a : ℕ → ℤ) :
  (sequence_satisfies_condition a ∧ (¬ arithmetic_sequence a)) ∨
  (arithmetic_sequence a → sequence_satisfies_condition a) :=
sorry

end NUMINAMATH_GPT_condition_is_necessary_but_not_sufficient_l2136_213616


namespace NUMINAMATH_GPT_primes_sum_divisible_by_60_l2136_213626

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_sum_divisible_by_60 (p q r s : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (hs : is_prime s) 
  (h_cond1 : 5 < p) 
  (h_cond2 : p < q) 
  (h_cond3 : q < r) 
  (h_cond4 : r < s) 
  (h_cond5 : s < p + 10) : 
  (p + q + r + s) % 60 = 0 :=
sorry

end NUMINAMATH_GPT_primes_sum_divisible_by_60_l2136_213626
