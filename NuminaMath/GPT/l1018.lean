import Mathlib

namespace NUMINAMATH_GPT_find_packs_of_yellow_bouncy_balls_l1018_101825

noncomputable def packs_of_yellow_bouncy_balls (red_packs : ℕ) (balls_per_pack : ℕ) (extra_balls : ℕ) : ℕ :=
  (red_packs * balls_per_pack - extra_balls) / balls_per_pack

theorem find_packs_of_yellow_bouncy_balls :
  packs_of_yellow_bouncy_balls 5 18 18 = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_packs_of_yellow_bouncy_balls_l1018_101825


namespace NUMINAMATH_GPT_circle_properties_l1018_101899

theorem circle_properties :
  ∃ p q s : ℝ, 
  (∀ x y : ℝ, x^2 + 16 * y + 89 = -y^2 - 12 * x ↔ (x + p)^2 + (y + q)^2 = s^2) ∧ 
  p + q + s = -14 + Real.sqrt 11 :=
by
  use -6, -8, Real.sqrt 11
  sorry

end NUMINAMATH_GPT_circle_properties_l1018_101899


namespace NUMINAMATH_GPT_alicia_tax_cents_per_hour_l1018_101807

-- Define Alicia's hourly wage in dollars.
def alicia_hourly_wage_dollars : ℝ := 25
-- Define the conversion rate from dollars to cents.
def cents_per_dollar : ℝ := 100
-- Define the local tax rate as a percentage.
def tax_rate_percent : ℝ := 2

-- Convert Alicia's hourly wage to cents.
def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * cents_per_dollar

-- Define the theorem that needs to be proved.
theorem alicia_tax_cents_per_hour : alicia_hourly_wage_cents * (tax_rate_percent / 100) = 50 := by
  sorry

end NUMINAMATH_GPT_alicia_tax_cents_per_hour_l1018_101807


namespace NUMINAMATH_GPT_trig_identity_l1018_101864

theorem trig_identity :
  (Real.tan (30 * Real.pi / 180) * Real.cos (60 * Real.pi / 180) + Real.tan (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180)) = (2 * Real.sqrt 3) / 3 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_trig_identity_l1018_101864


namespace NUMINAMATH_GPT_find_m_for_increasing_graph_l1018_101892

theorem find_m_for_increasing_graph (m : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → (m + 1) * x ^ (3 - m^2) < (m + 1) * y ^ (3 - m^2) → x < y) ↔ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_increasing_graph_l1018_101892


namespace NUMINAMATH_GPT_problem_geometric_description_of_set_T_l1018_101813

open Complex

def set_T (a b : ℝ) : ℂ := a + b * I

theorem problem_geometric_description_of_set_T :
  {w : ℂ | ∃ a b : ℝ, w = set_T a b ∧
    (im ((5 - 3 * I) * w) = 2 * re ((5 - 3 * I) * w))} =
  {w : ℂ | ∃ a : ℝ, w = set_T a (-(13/5) * a)} :=
sorry

end NUMINAMATH_GPT_problem_geometric_description_of_set_T_l1018_101813


namespace NUMINAMATH_GPT_quadratic_function_solution_l1018_101809

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1/2 * x

theorem quadratic_function_solution (f : ℝ → ℝ)
  (h1 : ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c))
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x+1) = f x + x + 1) :
  ∀ x, f x = 1/2 * x^2 + 1/2 * x :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_solution_l1018_101809


namespace NUMINAMATH_GPT_similar_segments_areas_proportional_to_chords_squares_l1018_101851

variables {k k₁ Δ Δ₁ r r₁ a a₁ S S₁ : ℝ}

-- Conditions given in the problem
def similar_segments (r r₁ a a₁ Δ Δ₁ k k₁ : ℝ) :=
  (Δ / Δ₁ = (a^2 / a₁^2) ∧ (Δ / Δ₁ = r^2 / r₁^2)) ∧ (k / k₁ = r^2 / r₁^2)

-- Given the areas of the segments in terms of sectors and triangles
def area_of_segment (k Δ : ℝ) := k - Δ

-- Theorem statement proving the desired relationship
theorem similar_segments_areas_proportional_to_chords_squares
  (h : similar_segments r r₁ a a₁ Δ Δ₁ k k₁) :
  (S = area_of_segment k Δ) → (S₁ = area_of_segment k₁ Δ₁) → (S / S₁ = a^2 / a₁^2) :=
by
  sorry

end NUMINAMATH_GPT_similar_segments_areas_proportional_to_chords_squares_l1018_101851


namespace NUMINAMATH_GPT_weavers_in_first_group_l1018_101817

theorem weavers_in_first_group 
  (W : ℕ)
  (H1 : 4 / (W * 4) = 1 / W) 
  (H2 : (9 / 6) / 6 = 0.25) :
  W = 4 :=
sorry

end NUMINAMATH_GPT_weavers_in_first_group_l1018_101817


namespace NUMINAMATH_GPT_probability_red_side_l1018_101874

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_red_side_l1018_101874


namespace NUMINAMATH_GPT_linear_eq_implies_m_eq_1_l1018_101896

theorem linear_eq_implies_m_eq_1 (x y m : ℝ) (h : 3 * (x ^ |m|) + (m + 1) * y = 6) (hm_abs : |m| = 1) (hm_ne_zero : m + 1 ≠ 0) : m = 1 :=
  sorry

end NUMINAMATH_GPT_linear_eq_implies_m_eq_1_l1018_101896


namespace NUMINAMATH_GPT_total_games_proof_l1018_101826

def num_teams : ℕ := 20
def num_games_per_team_regular_season : ℕ := 38
def total_regular_season_games : ℕ := num_teams * (num_games_per_team_regular_season / 2)
def num_games_per_team_mid_season : ℕ := 3
def total_mid_season_games : ℕ := num_teams * num_games_per_team_mid_season
def quarter_finals_teams : ℕ := 8
def quarter_finals_matchups : ℕ := quarter_finals_teams / 2
def quarter_finals_games : ℕ := quarter_finals_matchups * 2
def semi_finals_teams : ℕ := quarter_finals_matchups
def semi_finals_matchups : ℕ := semi_finals_teams / 2
def semi_finals_games : ℕ := semi_finals_matchups * 2
def final_teams : ℕ := semi_finals_matchups
def final_games : ℕ := final_teams * 2
def total_playoff_games : ℕ := quarter_finals_games + semi_finals_games + final_games

def total_season_games : ℕ := total_regular_season_games + total_mid_season_games + total_playoff_games

theorem total_games_proof : total_season_games = 454 := by
  -- The actual proof will go here
  sorry

end NUMINAMATH_GPT_total_games_proof_l1018_101826


namespace NUMINAMATH_GPT_sum_first_10_terms_eq_65_l1018_101845

section ArithmeticSequence

variables (a d : ℕ) (S : ℕ → ℕ) 

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Condition 1: nth term at n = 3
axiom a3_eq_4 : nth_term 3 = 4

-- Condition 2: difference in sums between n = 9 and n = 6
axiom S9_minus_S6_eq_27 : sum_first_n_terms 9 - sum_first_n_terms 6 = 27

-- To prove: sum of the first 10 terms equals 65
theorem sum_first_10_terms_eq_65 : sum_first_n_terms 10 = 65 :=
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_sum_first_10_terms_eq_65_l1018_101845


namespace NUMINAMATH_GPT_base_conversion_problem_l1018_101811

variable (A C : ℕ)
variable (h1 : 0 ≤ A ∧ A < 8)
variable (h2 : 0 ≤ C ∧ C < 5)

theorem base_conversion_problem (h : 8 * A + C = 5 * C + A) : 8 * A + C = 39 := 
sorry

end NUMINAMATH_GPT_base_conversion_problem_l1018_101811


namespace NUMINAMATH_GPT_square_of_binomial_l1018_101842

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 30 * x + a) → a = 25 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l1018_101842


namespace NUMINAMATH_GPT_vector_dot_product_sum_l1018_101822

noncomputable def points_in_plane (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) : Prop :=
  dist_AB = 3 ∧ dist_BC = 5 ∧ dist_CA = 6

theorem vector_dot_product_sum (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) (HA : points_in_plane A B C dist_AB dist_BC dist_CA) :
    ∃ (AB BC CA : ℝ), AB * BC + BC * CA + CA * AB = -35 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_sum_l1018_101822


namespace NUMINAMATH_GPT_proportion_correct_l1018_101860

theorem proportion_correct (x y : ℝ) (h : 3 * x = 2 * y) (hy : y ≠ 0) : x / 2 = y / 3 :=
by
  sorry

end NUMINAMATH_GPT_proportion_correct_l1018_101860


namespace NUMINAMATH_GPT_first_place_clay_l1018_101837

def Clay := "Clay"
def Allen := "Allen"
def Bart := "Bart"
def Dick := "Dick"

-- Statements made by the participants
def Allen_statements := ["I finished right before Bart", "I am not the first"]
def Bart_statements := ["I finished right before Clay", "I am not the second"]
def Clay_statements := ["I finished right before Dick", "I am not the third"]
def Dick_statements := ["I finished right before Allen", "I am not the last"]

-- Conditions
def only_two_true_statements : Prop := sorry -- This represents the condition that only two of these statements are true.
def first_place_told_truth : Prop := sorry -- This represents the condition that the person who got first place told at least one truth.

def person_first_place := Clay

theorem first_place_clay : person_first_place = Clay ∧ only_two_true_statements ∧ first_place_told_truth := 
sorry

end NUMINAMATH_GPT_first_place_clay_l1018_101837


namespace NUMINAMATH_GPT_find_d_l1018_101888

theorem find_d
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd4 : 4 = a * Real.sin 0 + d)
  (hdm2 : -2 = a * Real.sin (π) + d) :
  d = 1 := by
  sorry

end NUMINAMATH_GPT_find_d_l1018_101888


namespace NUMINAMATH_GPT_total_watermelons_l1018_101866

theorem total_watermelons 
  (A B C : ℕ) 
  (h1 : A + B = C - 6) 
  (h2 : B + C = A + 16) 
  (h3 : C + A = B + 8) :
  A + B + C = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_watermelons_l1018_101866


namespace NUMINAMATH_GPT_elizabeth_money_l1018_101861

theorem elizabeth_money :
  (∀ (P N : ℝ), P = 5 → N = 6 → 
    (P * 1.60 + N * 2.00) = 20.00) :=
by
  sorry

end NUMINAMATH_GPT_elizabeth_money_l1018_101861


namespace NUMINAMATH_GPT_james_vs_combined_l1018_101890

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687
def olivia_balloons : ℕ := 395
def combined_balloons : ℕ := amy_balloons + felix_balloons + olivia_balloons

theorem james_vs_combined :
  1222 = 1222 ∧ 513 = 513 ∧ 687 = 687 ∧ 395 = 395 → combined_balloons - james_balloons = 373 := by
  sorry

end NUMINAMATH_GPT_james_vs_combined_l1018_101890


namespace NUMINAMATH_GPT_smallest_three_digit_solution_l1018_101875

theorem smallest_three_digit_solution (n : ℕ) : 
  75 * n ≡ 225 [MOD 345] → 100 ≤ n ∧ n ≤ 999 → n = 118 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_smallest_three_digit_solution_l1018_101875


namespace NUMINAMATH_GPT_incorrect_statement_l1018_101852

noncomputable def function_y (x : ℝ) : ℝ := 4 / x

theorem incorrect_statement (x : ℝ) (hx : x ≠ 0) : ¬(∀ x1 x2 : ℝ, (hx1 : x1 ≠ 0) → (hx2 : x2 ≠ 0) → x1 < x2 → function_y x1 > function_y x2) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_l1018_101852


namespace NUMINAMATH_GPT_acute_triangle_angle_A_range_of_bc_l1018_101859

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}
variable (Δ : ∃ (A B C : ℝ), a = sqrt 2 ∧ ∀ (a b c A B C : ℝ), 
  (a = sqrt 2) ∧ (b = b) ∧ (c = c) ∧ 
  (sin A * cos A / cos (A + C) = a * c / (b^2 - a^2 - c^2)))

-- Problem statement
theorem acute_triangle_angle_A (h : Δ) : A = π / 4 :=
sorry

theorem range_of_bc (h : Δ) : 0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
sorry

end NUMINAMATH_GPT_acute_triangle_angle_A_range_of_bc_l1018_101859


namespace NUMINAMATH_GPT_money_left_after_shopping_l1018_101800

def initial_amount : ℕ := 26
def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5

theorem money_left_after_shopping : initial_amount - (cost_jumper + cost_tshirt + cost_heels) = 8 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l1018_101800


namespace NUMINAMATH_GPT_investor_share_price_l1018_101839

theorem investor_share_price (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) (price_per_share : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 40 →
  roi = 0.25 →
  ((dividend_rate * face_value) / price_per_share) = roi →
  price_per_share = 20 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_investor_share_price_l1018_101839


namespace NUMINAMATH_GPT_no_partition_equal_product_l1018_101884

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end NUMINAMATH_GPT_no_partition_equal_product_l1018_101884


namespace NUMINAMATH_GPT_example_problem_l1018_101870

def operation (a b : ℕ) : ℕ := (a + b) * (a - b)

theorem example_problem : 50 - operation 8 5 = 11 := by
  sorry

end NUMINAMATH_GPT_example_problem_l1018_101870


namespace NUMINAMATH_GPT_rainfall_in_may_l1018_101818

-- Define the rainfalls for the months
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def june_rain : ℝ := 3.09
def july_rain : ℝ := 4.67

-- Define the average rainfall over five months
def avg_rain : ℝ := 4

-- Define total rainfall calculation
def calc_total_rain (may_rain : ℝ) : ℝ :=
  march_rain + april_rain + may_rain + june_rain + july_rain

-- Problem statement: proving the rainfall in May
theorem rainfall_in_may : ∃ (may_rain : ℝ), calc_total_rain may_rain = avg_rain * 5 ∧ may_rain = 3.95 :=
sorry

end NUMINAMATH_GPT_rainfall_in_may_l1018_101818


namespace NUMINAMATH_GPT_primitive_root_exists_mod_pow_of_two_l1018_101833

theorem primitive_root_exists_mod_pow_of_two (n : ℕ) : 
  (∃ x : ℤ, ∀ k : ℕ, 1 ≤ k → x^k % (2^n) ≠ 1 % (2^n)) ↔ (n ≤ 2) := sorry

end NUMINAMATH_GPT_primitive_root_exists_mod_pow_of_two_l1018_101833


namespace NUMINAMATH_GPT_proof_problem_l1018_101895

theorem proof_problem (α : ℝ) (h1 : 0 < α ∧ α < π)
    (h2 : Real.sin α + Real.cos α = 1 / 5) :
    (Real.tan α = -4 / 3) ∧ 
    ((Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) * (Real.tan (Real.pi - α))^3) / 
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1018_101895


namespace NUMINAMATH_GPT_part1_part2_l1018_101873

-- Part 1: Positive integers with leading digit 6 that become 1/25 of the original number when the leading digit is removed.
theorem part1 (n : ℕ) (m : ℕ) (h1 : m = 6 * 10^n + m) (h2 : m = (6 * 10^n + m) / 25) :
  m = 625 * 10^(n - 2) ∨
  m = 625 * 10^(n - 2 + 1) ∨
  ∃ k : ℕ, m = 625 * 10^(n - 2 + k) :=
sorry

-- Part 2: No positive integer exists which becomes 1/35 of the original number when its leading digit is removed.
theorem part2 (n : ℕ) (m : ℕ) (h : m = 6 * 10^n + m) :
  m ≠ (6 * 10^n + m) / 35 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1018_101873


namespace NUMINAMATH_GPT_expression_equals_value_l1018_101841

theorem expression_equals_value : 97^3 + 3 * (97^2) + 3 * 97 + 1 = 940792 := 
by
  sorry

end NUMINAMATH_GPT_expression_equals_value_l1018_101841


namespace NUMINAMATH_GPT_profit_calculation_l1018_101877

variable (x y : ℝ)

-- Conditions
def fabric_constraints_1 : Prop := (0.5 * x + 0.9 * (50 - x) ≤ 38)
def fabric_constraints_2 : Prop := (x + 0.2 * (50 - x) ≤ 26)
def x_range : Prop := (17.5 ≤ x ∧ x ≤ 20)

-- Goal
def profit_expression : ℝ := 15 * x + 1500

theorem profit_calculation (h1 : fabric_constraints_1 x) (h2 : fabric_constraints_2 x) (h3 : x_range x) : y = profit_expression x :=
by
  sorry

end NUMINAMATH_GPT_profit_calculation_l1018_101877


namespace NUMINAMATH_GPT_min_vertices_in_hex_grid_l1018_101856

-- Define a hexagonal grid and the condition on the midpoint property.
def hexagonal_grid (p : ℤ × ℤ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

-- Statement: Prove that among any 9 points in a hexagonal grid, there are two points whose midpoint is also a grid point.
theorem min_vertices_in_hex_grid :
  ∀ points : Finset (ℤ × ℤ), points.card = 9 →
  (∃ p1 p2 : (ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ 
  (∃ midpoint : ℤ × ℤ, hexagonal_grid midpoint ∧ midpoint = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))) :=
by
  intros points h_points_card
  sorry

end NUMINAMATH_GPT_min_vertices_in_hex_grid_l1018_101856


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1018_101879

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
(h_arith : 2 * a 1 * q = a 0 + a 0 * q * q) :
  q = 2 + Real.sqrt 3 ∨ q = 2 - Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1018_101879


namespace NUMINAMATH_GPT_positive_real_solutions_unique_l1018_101815

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (x y z : ℝ)

theorem positive_real_solutions_unique :
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = abc →
    (x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_positive_real_solutions_unique_l1018_101815


namespace NUMINAMATH_GPT_students_at_start_of_year_l1018_101887

theorem students_at_start_of_year (S : ℝ) (h1 : S + 46.0 = 56) : S = 10 :=
sorry

end NUMINAMATH_GPT_students_at_start_of_year_l1018_101887


namespace NUMINAMATH_GPT_distinct_integers_division_l1018_101834

theorem distinct_integers_division (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ), a = n^2 + n + 1 ∧ b = n^2 + 2 ∧ c = n^2 + 1 ∧
  n^2 < a ∧ a < (n + 1)^2 ∧ 
  n^2 < b ∧ b < (n + 1)^2 ∧ 
  n^2 < c ∧ c < (n + 1)^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ∣ (a ^ 2 + b ^ 2) := 
by
  sorry

end NUMINAMATH_GPT_distinct_integers_division_l1018_101834


namespace NUMINAMATH_GPT_ratio_of_areas_l1018_101853

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1018_101853


namespace NUMINAMATH_GPT_ratio_of_M_to_R_l1018_101898

variable (M Q P N R : ℝ)

theorem ratio_of_M_to_R :
      M = 0.40 * Q →
      Q = 0.25 * P →
      N = 0.60 * P →
      R = 0.30 * N →
      M / R = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_of_M_to_R_l1018_101898


namespace NUMINAMATH_GPT_blake_initial_amount_l1018_101829

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end NUMINAMATH_GPT_blake_initial_amount_l1018_101829


namespace NUMINAMATH_GPT_flag_design_combinations_l1018_101847

-- Definitions
def colors : Nat := 3  -- Number of colors: purple, gold, and silver
def stripes : Nat := 3  -- Number of horizontal stripes in the flag

-- The Lean statement
theorem flag_design_combinations :
  (colors ^ stripes) = 27 :=
by
  sorry

end NUMINAMATH_GPT_flag_design_combinations_l1018_101847


namespace NUMINAMATH_GPT_saplings_problem_l1018_101836

theorem saplings_problem (x : ℕ) :
  (∃ n : ℕ, 5 * x + 3 = n ∧ 6 * x - 4 = n) ↔ 5 * x + 3 = 6 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_saplings_problem_l1018_101836


namespace NUMINAMATH_GPT_longest_interval_between_friday_13ths_l1018_101812

theorem longest_interval_between_friday_13ths
  (friday_the_13th : ℕ → ℕ → Prop)
  (at_least_once_per_year : ∀ year, ∃ month, friday_the_13th year month)
  (friday_occurs : ℕ) :
  ∃ (interval : ℕ), interval = 14 :=
by
  sorry

end NUMINAMATH_GPT_longest_interval_between_friday_13ths_l1018_101812


namespace NUMINAMATH_GPT_total_ttaki_count_l1018_101835

noncomputable def total_ttaki_used (n : ℕ): ℕ := n * n

theorem total_ttaki_count {n : ℕ} (h : 4 * n - 4 = 240) : total_ttaki_used n = 3721 := by
  sorry

end NUMINAMATH_GPT_total_ttaki_count_l1018_101835


namespace NUMINAMATH_GPT_total_pencils_correct_l1018_101885
  
def original_pencils : ℕ := 2
def added_pencils : ℕ := 3
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 5 := 
by
  -- proof state will be filled here 
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l1018_101885


namespace NUMINAMATH_GPT_range_of_a_for_monotonic_function_l1018_101838

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), is_monotonic_on (f · a) (Set.Iic (-1)) → a ≤ 3 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_range_of_a_for_monotonic_function_l1018_101838


namespace NUMINAMATH_GPT_find_x_l1018_101883

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def lcm_of_10_to_15 : ℕ :=
  leastCommonMultiple 10 (leastCommonMultiple 11 (leastCommonMultiple 12 (leastCommonMultiple 13 (leastCommonMultiple 14 15))))

theorem find_x :
  (lcm_of_10_to_15 / 2310 = 26) := by
  sorry

end NUMINAMATH_GPT_find_x_l1018_101883


namespace NUMINAMATH_GPT_probability_of_event_A_l1018_101872

def probability_event_A : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes

-- Statement of the theorem
theorem probability_of_event_A :
  probability_event_A = 1 / 6 :=
by
  -- This is where the proof would go, replaced with sorry for now.
  sorry

end NUMINAMATH_GPT_probability_of_event_A_l1018_101872


namespace NUMINAMATH_GPT_company_fund_initial_amount_l1018_101805

theorem company_fund_initial_amount (n : ℕ) 
  (h : 45 * n + 95 = 50 * n - 5) : 50 * n - 5 = 995 := by
  sorry

end NUMINAMATH_GPT_company_fund_initial_amount_l1018_101805


namespace NUMINAMATH_GPT_number_in_central_region_l1018_101806

theorem number_in_central_region (a b c d : ℤ) :
  a + b + c + d = -4 →
  ∃ x : ℤ, x = -4 + 2 :=
by
  intros h
  use -2
  sorry

end NUMINAMATH_GPT_number_in_central_region_l1018_101806


namespace NUMINAMATH_GPT_probability_of_drawing_jingyuetan_ticket_l1018_101823

-- Definitions from the problem
def num_jingyuetan_tickets : ℕ := 3
def num_changying_tickets : ℕ := 2
def total_tickets : ℕ := num_jingyuetan_tickets + num_changying_tickets
def num_envelopes : ℕ := total_tickets

-- Probability calculation
def probability_jingyuetan : ℚ := (num_jingyuetan_tickets : ℚ) / (num_envelopes : ℚ)

-- Theorem statement
theorem probability_of_drawing_jingyuetan_ticket : probability_jingyuetan = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_jingyuetan_ticket_l1018_101823


namespace NUMINAMATH_GPT_surface_area_of_T_is_630_l1018_101804

noncomputable def s : ℕ := 582
noncomputable def t : ℕ := 42
noncomputable def u : ℕ := 6

theorem surface_area_of_T_is_630 : s + t + u = 630 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_T_is_630_l1018_101804


namespace NUMINAMATH_GPT_sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l1018_101889

noncomputable def P (x : ℝ) : Prop := (x - 1)^2 > 16
noncomputable def Q (x a : ℝ) : Prop := x^2 + (a - 8) * x - 8 * a ≤ 0

theorem sufficient_not_necessary (a : ℝ) (x : ℝ) :
  a = 3 →
  (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem necessary_and_sufficient (a : ℝ) :
  (-5 ≤ a ∧ a ≤ 3) ↔ ∀ x, (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8) :=
sorry

theorem P_inter_Q (a : ℝ) (x : ℝ) :
  (a > 3 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a) ∨ (5 < x ∧ x ≤ 8)) ∧
  (-5 ≤ a ∧ a ≤ 3 → (P x ∧ Q x a) ↔ (5 < x ∧ x ≤ 8)) ∧
  (-8 ≤ a ∧ a < -5 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) ∧
  (a < -8 → (P x ∧ Q x a) ↔ (8 < x ∧ x ≤ -a)) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_necessary_and_sufficient_P_inter_Q_l1018_101889


namespace NUMINAMATH_GPT_price_of_table_l1018_101808

variable (C T : ℝ)

theorem price_of_table :
  2 * C + T = 0.6 * (C + 2 * T) ∧
  C + T = 96 →
  T = 84 := by
sorry

end NUMINAMATH_GPT_price_of_table_l1018_101808


namespace NUMINAMATH_GPT_volume_of_sand_pile_l1018_101802

theorem volume_of_sand_pile (d h : ℝ) (π : ℝ) (r : ℝ) (vol : ℝ) :
  d = 8 →
  h = (3 / 4) * d →
  r = d / 2 →
  vol = (1 / 3) * π * r^2 * h →
  vol = 32 * π :=
by
  intros hd hh hr hv
  subst hd
  subst hh
  subst hr
  subst hv
  sorry

end NUMINAMATH_GPT_volume_of_sand_pile_l1018_101802


namespace NUMINAMATH_GPT_largest_two_digit_number_with_remainder_2_div_13_l1018_101867

theorem largest_two_digit_number_with_remainder_2_div_13 : 
  ∃ (N : ℕ), (10 ≤ N ∧ N ≤ 99) ∧ N % 13 = 2 ∧ ∀ (M : ℕ), (10 ≤ M ∧ M ≤ 99) ∧ M % 13 = 2 → M ≤ N :=
  sorry

end NUMINAMATH_GPT_largest_two_digit_number_with_remainder_2_div_13_l1018_101867


namespace NUMINAMATH_GPT_white_pairs_coincide_l1018_101869

def triangles_in_each_half (red blue white: Nat) : Prop :=
  red = 5 ∧ blue = 6 ∧ white = 9

def folding_over_centerline (r_pairs b_pairs rw_pairs bw_pairs: Nat) : Prop :=
  r_pairs = 3 ∧ b_pairs = 2 ∧ rw_pairs = 3 ∧ bw_pairs = 1

theorem white_pairs_coincide
    (red_triangles blue_triangles white_triangles : Nat)
    (r_pairs b_pairs rw_pairs bw_pairs : Nat) :
    triangles_in_each_half red_triangles blue_triangles white_triangles →
    folding_over_centerline r_pairs b_pairs rw_pairs bw_pairs →
    ∃ coinciding_white_pairs, coinciding_white_pairs = 5 :=
by
  intros half_cond fold_cond
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l1018_101869


namespace NUMINAMATH_GPT_abs_sum_a_to_7_l1018_101844

-- Sequence definition with domain
def a (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Lean's ℕ includes 0, so use (n + 1) instead of n here.

-- Prove absolute value sum of first seven terms
theorem abs_sum_a_to_7 : (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 25) :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_abs_sum_a_to_7_l1018_101844


namespace NUMINAMATH_GPT_integer_solutions_yk_eq_x2_plus_x_l1018_101850

-- Define the problem in Lean
theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ (x y : ℤ), y^k = x^2 + x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_yk_eq_x2_plus_x_l1018_101850


namespace NUMINAMATH_GPT_sin2x_value_l1018_101819

theorem sin2x_value (x : ℝ) (h : Real.sin (x + π / 4) = 3 / 5) : 
  Real.sin (2 * x) = 8 * Real.sqrt 2 / 25 := 
by sorry

end NUMINAMATH_GPT_sin2x_value_l1018_101819


namespace NUMINAMATH_GPT_print_time_l1018_101827

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end NUMINAMATH_GPT_print_time_l1018_101827


namespace NUMINAMATH_GPT_distance_CD_l1018_101855

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

def major_axis_distance : ℝ := 4
def minor_axis_distance : ℝ := 2

theorem distance_CD : ∃ (d : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 → d = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_CD_l1018_101855


namespace NUMINAMATH_GPT_multiplier_for_obsolete_books_l1018_101862

theorem multiplier_for_obsolete_books 
  (x : ℕ) 
  (total_books_removed number_of_damaged_books : ℕ) 
  (h1 : total_books_removed = 69) 
  (h2 : number_of_damaged_books = 11) 
  (h3 : number_of_damaged_books + (x * number_of_damaged_books - 8) = total_books_removed) 
  : x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_multiplier_for_obsolete_books_l1018_101862


namespace NUMINAMATH_GPT_deductible_amount_l1018_101824

-- This definition represents the conditions of the problem.
def current_annual_deductible_is_increased (D : ℝ) : Prop :=
  (2 / 3) * D = 2000

-- This is the Lean statement, expressing the problem that needs to be proven.
theorem deductible_amount (D : ℝ) (h : current_annual_deductible_is_increased D) : D = 3000 :=
by
  sorry

end NUMINAMATH_GPT_deductible_amount_l1018_101824


namespace NUMINAMATH_GPT_intersection_with_xz_plane_l1018_101894

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def parametric_eqn (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + t * d.x) (p.y + t * d.y) (p.z + t * d.z)

theorem intersection_with_xz_plane (p1 p2 : Point3D) :
  let d := direction_vector p1 p2
  let t := (p1.y / d.y)
  parametric_eqn p1 d t = Point3D.mk 4 0 9 :=
sorry

#check intersection_with_xz_plane

end NUMINAMATH_GPT_intersection_with_xz_plane_l1018_101894


namespace NUMINAMATH_GPT_apples_in_boxes_l1018_101858

theorem apples_in_boxes (apples_per_box : ℕ) (number_of_boxes : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_box = 12) (h2 : number_of_boxes = 90) : total_apples = 1080 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_boxes_l1018_101858


namespace NUMINAMATH_GPT_price_increase_for_desired_profit_l1018_101803

/--
In Xianyou Yonghui Supermarket, the profit from selling Pomelos is 10 yuan per kilogram.
They can sell 500 kilograms per day. Market research has found that, with a constant cost price, if the price per kilogram increases by 1 yuan, the daily sales volume will decrease by 20 kilograms.
Now, the supermarket wants to ensure a daily profit of 6000 yuan while also offering the best deal to the customers.
-/
theorem price_increase_for_desired_profit :
  ∃ x : ℝ, (10 + x) * (500 - 20 * x) = 6000 ∧ x = 5 :=
sorry

end NUMINAMATH_GPT_price_increase_for_desired_profit_l1018_101803


namespace NUMINAMATH_GPT_range_m_l1018_101881

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_m (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ (-1 ≤ m ∧ m ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_m_l1018_101881


namespace NUMINAMATH_GPT_find_ab_l1018_101863

theorem find_ab (a b q r : ℕ) (h : a > 0) (h2 : b > 0) (h3 : (a^2 + b^2) / (a + b) = q) (h4 : (a^2 + b^2) % (a + b) = r) (h5 : q^2 + r = 2010) : a * b = 1643 :=
sorry

end NUMINAMATH_GPT_find_ab_l1018_101863


namespace NUMINAMATH_GPT_probability_red_ball_is_correct_l1018_101868

noncomputable def probability_red_ball : ℚ :=
  let prob_A := 1 / 3
  let prob_B := 1 / 3
  let prob_C := 1 / 3
  let prob_red_A := 3 / 10
  let prob_red_B := 7 / 10
  let prob_red_C := 5 / 11
  (prob_A * prob_red_A) + (prob_B * prob_red_B) + (prob_C * prob_red_C)

theorem probability_red_ball_is_correct : probability_red_ball = 16 / 33 := 
by
  sorry

end NUMINAMATH_GPT_probability_red_ball_is_correct_l1018_101868


namespace NUMINAMATH_GPT_sum_of_first_50_digits_is_216_l1018_101893

noncomputable def sum_first_50_digits_of_fraction : Nat :=
  let repeating_block := [0, 0, 0, 9, 9, 9]
  let full_cycles := 8
  let remaining_digits := [0, 0]
  let sum_full_cycles := full_cycles * (repeating_block.sum)
  let sum_remaining_digits := remaining_digits.sum
  sum_full_cycles + sum_remaining_digits

theorem sum_of_first_50_digits_is_216 :
  sum_first_50_digits_of_fraction = 216 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_50_digits_is_216_l1018_101893


namespace NUMINAMATH_GPT_correct_statement_l1018_101814

-- Defining the conditions
def freq_eq_prob : Prop :=
  ∀ (f p : ℝ), f = p

def freq_objective : Prop :=
  ∀ (f : ℝ) (n : ℕ), f = f

def freq_stabilizes : Prop :=
  ∀ (p : ℝ), ∃ (f : ℝ) (n : ℕ), f = p

def prob_random : Prop :=
  ∀ (p : ℝ), p = p

-- The statement we need to prove
theorem correct_statement :
  ¬freq_eq_prob ∧ ¬freq_objective ∧ freq_stabilizes ∧ ¬prob_random :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l1018_101814


namespace NUMINAMATH_GPT_product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l1018_101832

variable {x y : ℝ}

-- The formal statement in Lean
theorem product_pos_implies_pos_or_neg (h : x * y > 0) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
sorry

theorem pos_pair_implies_product_pos (hx : x > 0) (hy : y > 0) : x * y > 0 :=
sorry

theorem product_pos_necessary_for_pos (h : x > 0 ∧ y > 0) : x * y > 0 :=
pos_pair_implies_product_pos h.1 h.2

theorem product_pos_not_sufficient_for_pos (h : x * y > 0) : ¬ (x > 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l1018_101832


namespace NUMINAMATH_GPT_stations_equation_l1018_101821

theorem stations_equation (x : ℕ) (h : x * (x - 1) = 1482) : true :=
by
  sorry

end NUMINAMATH_GPT_stations_equation_l1018_101821


namespace NUMINAMATH_GPT_smallest_n_square_smallest_n_cube_l1018_101801

theorem smallest_n_square (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 2) ↔ n = 3 := 
by sorry

theorem smallest_n_cube (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 3) ↔ n = 2 := 
by sorry

end NUMINAMATH_GPT_smallest_n_square_smallest_n_cube_l1018_101801


namespace NUMINAMATH_GPT_power_function_monotonic_l1018_101828

theorem power_function_monotonic (m : ℝ) :
  2 * m^2 + m > 0 ∧ m > 0 → m = 1 / 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_power_function_monotonic_l1018_101828


namespace NUMINAMATH_GPT_find_solutions_l1018_101830

theorem find_solutions (x : ℝ) : (x = -9 ∨ x = -3 ∨ x = 3) →
  (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_solutions_l1018_101830


namespace NUMINAMATH_GPT_watch_cost_price_l1018_101854

noncomputable def cost_price : ℝ := 1166.67

theorem watch_cost_price (CP : ℝ) (loss_percent gain_percent : ℝ) (delta : ℝ) 
  (h1 : loss_percent = 0.10) 
  (h2 : gain_percent = 0.02) 
  (h3 : delta = 140) 
  (h4 : (1 - loss_percent) * CP + delta = (1 + gain_percent) * CP) : 
  CP = cost_price := 
by 
  sorry

end NUMINAMATH_GPT_watch_cost_price_l1018_101854


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1018_101886

class ParallelLines (A B c1 c2 : ℝ)

theorem distance_between_parallel_lines (A B c1 c2 : ℝ)
  [h : ParallelLines A B c1 c2] : 
  A = 4 → B = 3 → c1 = 1 → c2 = -9 → 
  (|c1 - c2| / Real.sqrt (A^2 + B^2)) = 2 :=
by
  intros hA hB hc1 hc2
  rw [hA, hB, hc1, hc2]
  norm_num
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1018_101886


namespace NUMINAMATH_GPT_unique_fraction_representation_l1018_101865

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  ∃! (x y : ℕ), (x ≠ y) ∧ (2 * x * y = p * (x + y)) :=
by
  sorry

end NUMINAMATH_GPT_unique_fraction_representation_l1018_101865


namespace NUMINAMATH_GPT_product_of_five_consecutive_numbers_not_square_l1018_101871

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_numbers_not_square_l1018_101871


namespace NUMINAMATH_GPT_checkerboard_problem_l1018_101857

def checkerboard_rectangles : ℕ := 2025
def checkerboard_squares : ℕ := 285

def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem checkerboard_problem :
  ∃ m n : ℕ, relatively_prime m n ∧ m + n = 154 ∧ (285 : ℚ) / 2025 = m / n :=
by {
  sorry
}

end NUMINAMATH_GPT_checkerboard_problem_l1018_101857


namespace NUMINAMATH_GPT_value_of_inverse_product_l1018_101846

theorem value_of_inverse_product (x y : ℝ) (h1 : x * y > 0) (h2 : 1/x + 1/y = 15) (h3 : (x + y) / 5 = 0.6) :
  1 / (x * y) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_inverse_product_l1018_101846


namespace NUMINAMATH_GPT_Triamoeba_Count_After_One_Week_l1018_101891

def TriamoebaCount (n : ℕ) : ℕ :=
  3 ^ n

theorem Triamoeba_Count_After_One_Week : TriamoebaCount 7 = 2187 :=
by
  -- This is the statement to be proved
  sorry

end NUMINAMATH_GPT_Triamoeba_Count_After_One_Week_l1018_101891


namespace NUMINAMATH_GPT_jason_manager_years_l1018_101880

-- Definitions based on the conditions
def jason_bartender_years : ℕ := 9
def jason_total_months : ℕ := 150
def additional_months_excluded : ℕ := 6

-- Conversion from months to years
def total_years := jason_total_months / 12
def excluded_years := additional_months_excluded / 12

-- Lean statement for the proof problem
theorem jason_manager_years :
  total_years - jason_bartender_years - excluded_years = 3 := by
  sorry

end NUMINAMATH_GPT_jason_manager_years_l1018_101880


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1018_101849

theorem sum_of_reciprocals_of_roots 
  (r₁ r₂ : ℝ)
  (h_roots : ∀ (x : ℝ), x^2 - 17*x + 8 = 0 → (∃ r, (r = r₁ ∨ r = r₂) ∧ x = r))
  (h_sum : r₁ + r₂ = 17)
  (h_prod : r₁ * r₂ = 8) :
  1/r₁ + 1/r₂ = 17/8 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1018_101849


namespace NUMINAMATH_GPT_factorization_problem_l1018_101897

theorem factorization_problem 
  (C D : ℤ)
  (h1 : 15 * y ^ 2 - 76 * y + 48 = (C * y - 16) * (D * y - 3))
  (h2 : C * D = 15)
  (h3 : C * (-3) + D * (-16) = -76)
  (h4 : (-16) * (-3) = 48) : 
  C * D + C = 20 :=
by { sorry }

end NUMINAMATH_GPT_factorization_problem_l1018_101897


namespace NUMINAMATH_GPT_rhombus_properties_l1018_101876

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
noncomputable def side_length_of_rhombus (d1 d2 : ℝ) : ℝ := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 16) :
  area_of_rhombus d1 d2 = 144 ∧ side_length_of_rhombus d1 d2 = Real.sqrt 145 := by
  sorry

end NUMINAMATH_GPT_rhombus_properties_l1018_101876


namespace NUMINAMATH_GPT_circle_condition_l1018_101820

-- Define the given equation
def equation (m x y : ℝ) : Prop := x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0

-- Define the condition for the equation to represent a circle
def represents_circle (m x y : ℝ) : Prop :=
  (x + 2 * m)^2 + (y - 1)^2 = 4 * m^2 - 5 * m + 1 ∧ 4 * m^2 - 5 * m + 1 > 0

-- The main theorem to be proven
theorem circle_condition (m : ℝ) : represents_circle m x y → (m < 1/4 ∨ m > 1) := 
sorry

end NUMINAMATH_GPT_circle_condition_l1018_101820


namespace NUMINAMATH_GPT_distribute_problems_l1018_101810

theorem distribute_problems :
  (12 ^ 6) = 2985984 := by
  sorry

end NUMINAMATH_GPT_distribute_problems_l1018_101810


namespace NUMINAMATH_GPT_geologists_probability_l1018_101831

theorem geologists_probability :
  let r := 4 -- speed of each geologist in km/h
  let d := 6 -- distance in km
  let sectors := 8 -- number of sectors (roads)
  let total_outcomes := sectors * sectors
  let favorable_outcomes := sectors * 3 -- when distance > 6 km

  -- Calculating probability
  let P := (favorable_outcomes: ℝ) / (total_outcomes: ℝ)

  P = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_geologists_probability_l1018_101831


namespace NUMINAMATH_GPT_leak_empties_tank_in_4_hours_l1018_101840

theorem leak_empties_tank_in_4_hours
  (A_fills_in : ℝ)
  (A_with_leak_fills_in : ℝ) : 
  (∀ (L : ℝ), A_fills_in = 2 ∧ A_with_leak_fills_in = 4 → L = (1 / 4) → 1 / L = 4) :=
by 
  sorry

end NUMINAMATH_GPT_leak_empties_tank_in_4_hours_l1018_101840


namespace NUMINAMATH_GPT_natural_number_between_squares_l1018_101848

open Nat

theorem natural_number_between_squares (n m k l : ℕ)
  (h1 : n > m^2)
  (h2 : n < (m+1)^2)
  (h3 : n - k = m^2)
  (h4 : n + l = (m+1)^2) : ∃ x : ℕ, n - k * l = x^2 := by
  sorry

end NUMINAMATH_GPT_natural_number_between_squares_l1018_101848


namespace NUMINAMATH_GPT_most_reasonable_sampling_method_is_stratified_l1018_101816

def population_has_significant_differences 
    (grades : List String)
    (understanding : String → ℕ)
    : Prop := sorry -- This would be defined based on the details of "significant differences"

theorem most_reasonable_sampling_method_is_stratified
    (grades : List String)
    (understanding : String → ℕ)
    (h : population_has_significant_differences grades understanding)
    : (method : String) → (method = "Stratified sampling") :=
sorry

end NUMINAMATH_GPT_most_reasonable_sampling_method_is_stratified_l1018_101816


namespace NUMINAMATH_GPT_roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l1018_101843

variables {α : Type*} [Field α] (a b c x1 x2 : α)

theorem roots_quadratic_eq_identity1 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2 :=
sorry

theorem roots_quadratic_eq_identity2 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l1018_101843


namespace NUMINAMATH_GPT_min_value_n_l1018_101882

theorem min_value_n (n : ℕ) (h1 : 4 ∣ 60 * n) (h2 : 8 ∣ 60 * n) : n = 1 := 
  sorry

end NUMINAMATH_GPT_min_value_n_l1018_101882


namespace NUMINAMATH_GPT_symmetric_line_equation_l1018_101878

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y : ℝ, x - 3 * y + 5 = 0 ↔ 3 * x - y - 5 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1018_101878
