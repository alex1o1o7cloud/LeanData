import Mathlib

namespace NUMINAMATH_GPT_sum_of_squares_base_case_l673_67359

theorem sum_of_squares_base_case : 1^2 + 2^2 = (1 * 3 * 5) / 3 := by sorry

end NUMINAMATH_GPT_sum_of_squares_base_case_l673_67359


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l673_67398

theorem necessary_not_sufficient_condition (x : ℝ) : (x < 2) → (x^2 - x - 2 < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l673_67398


namespace NUMINAMATH_GPT_ratio_city_XY_l673_67313

variable (popZ popY popX : ℕ)

-- Definition of the conditions
def condition1 := popY = 2 * popZ
def condition2 := popX = 16 * popZ

-- The goal to prove
theorem ratio_city_XY 
  (h1 : condition1 popY popZ)
  (h2 : condition2 popX popZ) :
  popX / popY = 8 := 
  by sorry

end NUMINAMATH_GPT_ratio_city_XY_l673_67313


namespace NUMINAMATH_GPT_coefficients_square_sum_l673_67378

theorem coefficients_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1000 * x ^ 3 + 27 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end NUMINAMATH_GPT_coefficients_square_sum_l673_67378


namespace NUMINAMATH_GPT_goose_eggs_laid_l673_67332

theorem goose_eggs_laid (E : ℕ) 
    (H1 : ∃ h, h = (2 / 5) * E)
    (H2 : ∃ m, m = (11 / 15) * h)
    (H3 : ∃ s, s = (1 / 4) * m)
    (H4 : ∃ y, y = (2 / 7) * s)
    (H5 : y = 150) : 
    E = 7160 := 
sorry

end NUMINAMATH_GPT_goose_eggs_laid_l673_67332


namespace NUMINAMATH_GPT_bowling_ball_weight_l673_67342

theorem bowling_ball_weight :
  (∀ b c : ℝ, 9 * b = 2 * c → c = 35 → b = 70 / 9) :=
by
  intros b c h1 h2
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l673_67342


namespace NUMINAMATH_GPT_envelope_width_l673_67371

theorem envelope_width (L W A : ℝ) (hL : L = 4) (hA : A = 16) (hArea : A = L * W) : W = 4 := 
by
  -- We state the problem
  sorry

end NUMINAMATH_GPT_envelope_width_l673_67371


namespace NUMINAMATH_GPT_original_ratio_l673_67377

theorem original_ratio (x y : ℕ) (h1 : y = 15) (h2 : x + 10 = y) : x / y = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_ratio_l673_67377


namespace NUMINAMATH_GPT_chickens_egg_production_l673_67323

/--
Roberto buys 4 chickens for $20 each. The chickens cost $1 in total per week to feed.
Roberto used to buy 1 dozen eggs (12 eggs) a week, spending $2 per dozen.
After 81 weeks, the total cost of raising chickens will be cheaper than buying the eggs.
Prove that each chicken produces 3 eggs per week.
-/
theorem chickens_egg_production:
  let chicken_cost := 20
  let num_chickens := 4
  let weekly_feed_cost := 1
  let weekly_eggs_cost := 2
  let dozen_eggs := 12
  let weeks := 81

  -- Cost calculations
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weekly_feed_cost * weeks
  let total_raising_cost := total_chicken_cost + total_feed_cost
  let total_buying_cost := weekly_eggs_cost * weeks

  -- Ensure cost condition
  (total_raising_cost <= total_buying_cost) →
  
  -- Egg production calculation
  (dozen_eggs / num_chickens) = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chickens_egg_production_l673_67323


namespace NUMINAMATH_GPT_extinction_prob_one_l673_67340

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_extinction_prob_one_l673_67340


namespace NUMINAMATH_GPT_alloy_mixing_l673_67392

theorem alloy_mixing (x : ℕ) :
  (2 / 5) * 60 + (1 / 5) * x = 44 → x = 100 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_alloy_mixing_l673_67392


namespace NUMINAMATH_GPT_probability_of_selecting_one_male_and_one_female_l673_67328

noncomputable def probability_one_male_one_female : ℚ :=
  let total_ways := (Nat.choose 6 2) -- Total number of ways to select 2 out of 6
  let ways_one_male_one_female := (Nat.choose 3 1) * (Nat.choose 3 1) -- Ways to select 1 male and 1 female
  ways_one_male_one_female / total_ways

theorem probability_of_selecting_one_male_and_one_female :
  probability_one_male_one_female = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_male_and_one_female_l673_67328


namespace NUMINAMATH_GPT_geometric_prod_eight_l673_67311

theorem geometric_prod_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : ∀ n, a n ≠ 0)
  (h_eq : a 4 + 3 * a 8 = 2 * (a 7)^2)
  (h_geom : ∀ {m n : ℕ}, b m * b (m + n) = b (2 * m + n))
  (h_b_eq_a : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_prod_eight_l673_67311


namespace NUMINAMATH_GPT_count_solutions_inequalities_l673_67366

theorem count_solutions_inequalities :
  {x : ℤ | -5 * x ≥ 2 * x + 10} ∩ {x : ℤ | -3 * x ≤ 15} ∩ {x : ℤ | -6 * x ≥ 3 * x + 21} = {x : ℤ | x = -5 ∨ x = -4 ∨ x = -3} :=
by 
  sorry

end NUMINAMATH_GPT_count_solutions_inequalities_l673_67366


namespace NUMINAMATH_GPT_total_air_removed_after_5_strokes_l673_67301

theorem total_air_removed_after_5_strokes:
  let initial_air := 1
  let remaining_air_after_first_stroke := initial_air * (2 / 3)
  let remaining_air_after_second_stroke := remaining_air_after_first_stroke * (3 / 4)
  let remaining_air_after_third_stroke := remaining_air_after_second_stroke * (4 / 5)
  let remaining_air_after_fourth_stroke := remaining_air_after_third_stroke * (5 / 6)
  let remaining_air_after_fifth_stroke := remaining_air_after_fourth_stroke * (6 / 7)
  initial_air - remaining_air_after_fifth_stroke = 5 / 7 := by
  sorry

end NUMINAMATH_GPT_total_air_removed_after_5_strokes_l673_67301


namespace NUMINAMATH_GPT_part1_a_range_part2_x_range_l673_67384
open Real

-- Definitions based on given conditions
def quad_func (a b x : ℝ) : ℝ :=
  a * x^2 + b * x + 2

def y_at_x1 (a b : ℝ) : Prop :=
  quad_func a b 1 = 1

def pos_on_interval (a b l r : ℝ) (x : ℝ) : Prop :=
  l < x ∧ x < r → 0 < quad_func a b x

-- Part 1 proof statement in Lean 4
theorem part1_a_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ x : ℝ, pos_on_interval a b 2 5 x) :
  a > 3 - 2 * sqrt 2 :=
sorry

-- Part 2 proof statement in Lean 4
theorem part2_x_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ a' : ℝ, -2 ≤ a' ∧ a' ≤ -1 → 0 < quad_func a' b x) :
  (1 - sqrt 17) / 4 < x ∧ x < (1 + sqrt 17) / 4 :=
sorry

end NUMINAMATH_GPT_part1_a_range_part2_x_range_l673_67384


namespace NUMINAMATH_GPT_hyperbola_asymptote_l673_67349

theorem hyperbola_asymptote (a : ℝ) (h_cond : 0 < a)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2 - y^2 / 9 = 1) → (y = (3 / 5) * x))
  : a = 5 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l673_67349


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l673_67335

variable (a_n : ℕ → ℕ)

theorem arithmetic_sequence_sum (h1: a_n 1 + a_n 2 = 5) (h2 : a_n 3 + a_n 4 = 7) (arith : ∀ n, a_n (n + 1) - a_n n = a_n 2 - a_n 1) :
  a_n 5 + a_n 6 = 9 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l673_67335


namespace NUMINAMATH_GPT_triangle_angle_opposite_c_l673_67376

theorem triangle_angle_opposite_c (a b c : ℝ) (x : ℝ) 
  (ha : a = 2) (hb : b = 2) (hc : c = 4) : x = 180 :=
by 
  -- proof steps are not required as per the instruction
  sorry

end NUMINAMATH_GPT_triangle_angle_opposite_c_l673_67376


namespace NUMINAMATH_GPT_one_eighth_of_two_pow_36_eq_two_pow_y_l673_67326

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end NUMINAMATH_GPT_one_eighth_of_two_pow_36_eq_two_pow_y_l673_67326


namespace NUMINAMATH_GPT_inequality_proof_l673_67343

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l673_67343


namespace NUMINAMATH_GPT_range_of_m_l673_67356

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l673_67356


namespace NUMINAMATH_GPT_abs_diff_61st_term_l673_67374

-- Define sequences C and D
def seqC (n : ℕ) : ℤ := 20 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 20 - 15 * (n - 1)

-- Prove the absolute value of the difference between the 61st terms is 1800
theorem abs_diff_61st_term : (abs (seqC 61 - seqD 61) = 1800) :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_61st_term_l673_67374


namespace NUMINAMATH_GPT_overlapping_area_is_correct_l673_67358

-- Defining the coordinates of the grid points
def topLeft : (ℝ × ℝ) := (0, 2)
def topMiddle : (ℝ × ℝ) := (1.5, 2)
def topRight : (ℝ × ℝ) := (3, 2)
def middleLeft : (ℝ × ℝ) := (0, 1)
def center : (ℝ × ℝ) := (1.5, 1)
def middleRight : (ℝ × ℝ) := (3, 1)
def bottomLeft : (ℝ × ℝ) := (0, 0)
def bottomMiddle : (ℝ × ℝ) := (1.5, 0)
def bottomRight : (ℝ × ℝ) := (3, 0)

-- Defining the vertices of the triangles
def triangle1_points : List (ℝ × ℝ) := [topLeft, middleRight, bottomMiddle]
def triangle2_points : List (ℝ × ℝ) := [bottomLeft, topMiddle, middleRight]

-- Function to calculate the area of a polygon given the vertices -- placeholder here
noncomputable def area_of_overlapped_region (tr1 tr2 : List (ℝ × ℝ)) : ℝ := 
  -- Placeholder for the actual computation of the overlapped area
  1.2

-- Statement to prove
theorem overlapping_area_is_correct : 
  area_of_overlapped_region triangle1_points triangle2_points = 1.2 := sorry

end NUMINAMATH_GPT_overlapping_area_is_correct_l673_67358


namespace NUMINAMATH_GPT_difference_of_interchanged_digits_l673_67329

theorem difference_of_interchanged_digits (X Y : ℕ) (h1 : X - Y = 3) :
  (10 * X + Y) - (10 * Y + X) = 27 := by
  sorry

end NUMINAMATH_GPT_difference_of_interchanged_digits_l673_67329


namespace NUMINAMATH_GPT_susan_remaining_money_l673_67391

theorem susan_remaining_money :
  let initial_amount := 90
  let food_spent := 20
  let game_spent := 3 * food_spent
  let total_spent := food_spent + game_spent
  initial_amount - total_spent = 10 :=
by 
  sorry

end NUMINAMATH_GPT_susan_remaining_money_l673_67391


namespace NUMINAMATH_GPT_average_age_of_team_l673_67381

/--
The captain of a cricket team of 11 members is 26 years old and the wicket keeper is 
3 years older. If the ages of these two are excluded, the average age of the remaining 
players is one year less than the average age of the whole team. Prove that the average 
age of the whole team is 32 years.
-/
theorem average_age_of_team 
  (captain_age : Nat) (wicket_keeper_age : Nat) (remaining_9_average_age : Nat)
  (team_size : Nat) (total_team_age : Nat) (remaining_9_total_age : Nat)
  (A : Nat) :
  captain_age = 26 →
  wicket_keeper_age = captain_age + 3 →
  team_size = 11 →
  total_team_age = team_size * A →
  total_team_age = remaining_9_total_age + captain_age + wicket_keeper_age →
  remaining_9_total_age = 9 * (A - 1) →
  A = 32 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_team_l673_67381


namespace NUMINAMATH_GPT_num_tables_l673_67395

theorem num_tables (T : ℕ) : 
  (6 * T = (17 / 3) * T) → 
  T = 6 :=
sorry

end NUMINAMATH_GPT_num_tables_l673_67395


namespace NUMINAMATH_GPT_value_of_m_l673_67345

theorem value_of_m (m : ℚ) : 
  (m = - -(-(1/3) : ℚ) → m = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l673_67345


namespace NUMINAMATH_GPT_part1_part2_l673_67330

noncomputable def f (a x : ℝ) : ℝ := a * x + x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x, x ≥ Real.exp 1 → (a + 1 + Real.log x) ≥ 0) →
  a ≥ -2 :=
by
  sorry

theorem part2 (k : ℤ) :
  (∀ x, 1 < x → (k : ℝ) * (x - 1) < f 1 x) →
  k ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l673_67330


namespace NUMINAMATH_GPT_loga_increasing_loga_decreasing_l673_67361

noncomputable def loga (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem loga_increasing (a : ℝ) (h₁ : a > 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a x < loga a y := by
  sorry 

theorem loga_decreasing (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a y < loga a x := by
  sorry

end NUMINAMATH_GPT_loga_increasing_loga_decreasing_l673_67361


namespace NUMINAMATH_GPT_find_length_of_BC_l673_67322

-- Define the geometrical objects and lengths
variable {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable (AB AC AM BC : ℝ)
variable (is_midpoint : Midpoint M B C)
variable (known_AB : AB = 7)
variable (known_AC : AC = 6)
variable (known_AM : AM = 4)

theorem find_length_of_BC : BC = Real.sqrt 106 := by
  sorry

end NUMINAMATH_GPT_find_length_of_BC_l673_67322


namespace NUMINAMATH_GPT_smallest_n_l673_67353

theorem smallest_n (n : ℕ) (h : 0 < n) : 
  (1 / (n : ℝ)) - (1 / (n + 1 : ℝ)) < 1 / 15 → n = 4 := sorry

end NUMINAMATH_GPT_smallest_n_l673_67353


namespace NUMINAMATH_GPT_range_of_a_l673_67319

theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 - x ≤ 0 → 2^(1 - x) + a ≤ 0)) ↔ (a ≤ -2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l673_67319


namespace NUMINAMATH_GPT_first_negative_term_at_14_l673_67300

-- Define the n-th term of the arithmetic sequence
def a_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Given values
def a₁ := 51
def d := -4

-- Proof statement
theorem first_negative_term_at_14 : ∃ n : ℕ, a_n a₁ d n < 0 ∧ ∀ m < n, a_n a₁ d m ≥ 0 :=
  by sorry

end NUMINAMATH_GPT_first_negative_term_at_14_l673_67300


namespace NUMINAMATH_GPT_average_cookies_per_package_l673_67304

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies_per_package :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 125 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_cookies_per_package_l673_67304


namespace NUMINAMATH_GPT_yield_percentage_of_stock_is_8_percent_l673_67308

theorem yield_percentage_of_stock_is_8_percent :
  let face_value := 100
  let dividend_rate := 0.20
  let market_price := 250
  annual_dividend = dividend_rate * face_value →
  yield_percentage = (annual_dividend / market_price) * 100 →
  yield_percentage = 8 := 
by
  sorry

end NUMINAMATH_GPT_yield_percentage_of_stock_is_8_percent_l673_67308


namespace NUMINAMATH_GPT_intersection_A_complement_B_l673_67346

def A := { x : ℝ | x ≥ -1 }
def B := { x : ℝ | x > 2 }
def complement_B := { x : ℝ | x ≤ 2 }

theorem intersection_A_complement_B :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } :=
sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l673_67346


namespace NUMINAMATH_GPT_range_of_a_l673_67318

-- Definitions based on conditions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 4

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → f a x ≤ f a 4) → a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l673_67318


namespace NUMINAMATH_GPT_scientific_notation_of_area_l673_67302

theorem scientific_notation_of_area : 2720000 = 2.72 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_area_l673_67302


namespace NUMINAMATH_GPT_expected_value_of_X_is_5_over_3_l673_67307

-- Define the probabilities of getting an interview with company A, B, and C
def P_A : ℚ := 2 / 3
def P_BC (p : ℚ) : ℚ := p

-- Define the random variable X representing the number of interview invitations
def X (P_A P_BC : ℚ) : ℚ := sorry

-- Define the probability of receiving no interview invitations
def P_X_0 (P_A P_BC : ℚ) : ℚ := (1 - P_A) * (1 - P_BC)^2

-- Given condition that P(X=0) is 1/12
def condition_P_X_0 (P_A P_BC : ℚ) : Prop := P_X_0 P_A P_BC = 1 / 12

-- Given p = 1/2 as per the problem solution
def p : ℚ := 1 / 2

-- Expected value of X
def E_X (P_A P_BC : ℚ) : ℚ := (1 * (2 * P_BC * (1 - P_BC) + 2 * P_BC^2 * (1 - P_BC) + (1 - P_A) * P_BC^2)) +
                               (2 * (P_A * P_BC * (1 - P_BC) + P_A * (1 - P_BC)^2 + P_BC * P_BC * (1 - P_A))) +
                               (3 * (P_A * P_BC^2))

-- Theorem proving the expected value of X given the above conditions
theorem expected_value_of_X_is_5_over_3 : E_X P_A (P_BC p) = 5 / 3 :=
by
  -- here you will write the proof later
  sorry

end NUMINAMATH_GPT_expected_value_of_X_is_5_over_3_l673_67307


namespace NUMINAMATH_GPT_arithmetic_sequence_ratios_l673_67360

theorem arithmetic_sequence_ratios
  (a : ℕ → ℝ) (b : ℕ → ℝ) (A : ℕ → ℝ) (B : ℕ → ℝ)
  (d1 d2 a1 b1 : ℝ)
  (hA_sum : ∀ n : ℕ, A n = n * a1 + (n * (n - 1)) * d1 / 2)
  (hB_sum : ∀ n : ℕ, B n = n * b1 + (n * (n - 1)) * d2 / 2)
  (h_ratio : ∀ n : ℕ, B n ≠ 0 → A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b n ≠ 0 → a n / b n = (4 * n - 3) / (6 * n - 2) := sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratios_l673_67360


namespace NUMINAMATH_GPT_cylinder_volume_l673_67309

noncomputable def volume_cylinder (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) : ℝ :=
  let ratio_r := r_cylinder / r_cone
  let ratio_h := h_cylinder / h_cone
  (3 : ℝ) * ratio_r^2 * ratio_h * V_cone

theorem cylinder_volume (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) :
    r_cylinder / r_cone = 2 / 3 →
    h_cylinder / h_cone = 4 / 3 →
    V_cone = 5.4 →
    volume_cylinder V_cone r_cylinder r_cone h_cylinder h_cone = 3.2 :=
by
  intros h1 h2 h3
  rw [volume_cylinder, h1, h2, h3]
  sorry

end NUMINAMATH_GPT_cylinder_volume_l673_67309


namespace NUMINAMATH_GPT_ratio_of_sides_l673_67399

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l673_67399


namespace NUMINAMATH_GPT_bobby_shoes_l673_67383

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end NUMINAMATH_GPT_bobby_shoes_l673_67383


namespace NUMINAMATH_GPT_queenie_daily_earnings_l673_67344

/-- Define the overtime earnings per hour. -/
def overtime_pay_per_hour : ℤ := 5

/-- Define the total amount received. -/
def total_received : ℤ := 770

/-- Define the number of days worked. -/
def days_worked : ℤ := 5

/-- Define the number of overtime hours. -/
def overtime_hours : ℤ := 4

/-- State the theorem to find out Queenie's daily earnings. -/
theorem queenie_daily_earnings :
  ∃ D : ℤ, days_worked * D + overtime_hours * overtime_pay_per_hour = total_received ∧ D = 150 :=
by
  use 150
  sorry

end NUMINAMATH_GPT_queenie_daily_earnings_l673_67344


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l673_67368

theorem simplify_and_evaluate_expression (x y : ℚ) (h_x : x = -2) (h_y : y = 1/2) :
  (x + 2 * y)^2 - (x + y) * (x - y) = -11/4 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l673_67368


namespace NUMINAMATH_GPT_find_radius_l673_67339

theorem find_radius :
  ∃ (r : ℝ), 
  (∀ (x : ℝ), y = x^2 + r) ∧ 
  (∀ (x : ℝ), y = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 0) → 
  r = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l673_67339


namespace NUMINAMATH_GPT_simplify_expression_l673_67341

theorem simplify_expression (y : ℝ) : (y - 2)^2 + 2 * (y - 2) * (5 + y) + (5 + y)^2 = (2*y + 3)^2 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l673_67341


namespace NUMINAMATH_GPT_michael_laps_to_pass_donovan_l673_67350

theorem michael_laps_to_pass_donovan (track_length : ℕ) (donovan_lap_time : ℕ) (michael_lap_time : ℕ) 
  (h1 : track_length = 400) (h2 : donovan_lap_time = 48) (h3 : michael_lap_time = 40) : 
  michael_lap_time * 6 = donovan_lap_time * (michael_lap_time * 6 / track_length * michael_lap_time) :=
by
  sorry

end NUMINAMATH_GPT_michael_laps_to_pass_donovan_l673_67350


namespace NUMINAMATH_GPT_total_pastries_l673_67336

variable (P x : ℕ)

theorem total_pastries (h1 : P = 28 * (10 + x)) (h2 : P = 49 * (4 + x)) : P = 392 := 
by 
  sorry

end NUMINAMATH_GPT_total_pastries_l673_67336


namespace NUMINAMATH_GPT_elementary_sampling_count_l673_67382

theorem elementary_sampling_count :
  ∃ (a : ℕ), (a + (a + 600) + (a + 1200) = 3600) ∧
             (a = 600) ∧
             (a + 1200 = 1800) ∧
             (1800 * 1 / 100 = 18) :=
by {
  sorry
}

end NUMINAMATH_GPT_elementary_sampling_count_l673_67382


namespace NUMINAMATH_GPT_probability_of_blue_candy_l673_67367

theorem probability_of_blue_candy (green blue red : ℕ) (h1 : green = 5) (h2 : blue = 3) (h3 : red = 4) :
  (blue : ℚ) / (green + blue + red : ℚ) = 1 / 4 :=
by
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_GPT_probability_of_blue_candy_l673_67367


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l673_67362

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 162) : a = 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l673_67362


namespace NUMINAMATH_GPT_trajectory_midpoint_chord_l673_67394

theorem trajectory_midpoint_chord (x y : ℝ) 
  (h₀ : y^2 = 4 * x) : (y^2 = 2 * x - 2) :=
sorry

end NUMINAMATH_GPT_trajectory_midpoint_chord_l673_67394


namespace NUMINAMATH_GPT_segment_length_is_15_l673_67369

theorem segment_length_is_15 : 
  ∀ (x : ℝ), 
  ∀ (y1 y2 : ℝ), 
  x = 3 → 
  y1 = 5 → 
  y2 = 20 → 
  abs (y2 - y1) = 15 := by 
sorry

end NUMINAMATH_GPT_segment_length_is_15_l673_67369


namespace NUMINAMATH_GPT_real_solutions_count_l673_67303

noncomputable def number_of_real_solutions : ℕ := 2

theorem real_solutions_count (x : ℝ) :
  (x^2 - 5)^2 = 36 → number_of_real_solutions = 2 := by
  sorry

end NUMINAMATH_GPT_real_solutions_count_l673_67303


namespace NUMINAMATH_GPT_find_expression_l673_67387

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end NUMINAMATH_GPT_find_expression_l673_67387


namespace NUMINAMATH_GPT_find_y_when_x_is_minus_2_l673_67363

theorem find_y_when_x_is_minus_2 :
  ∀ (x y t : ℝ), (x = 3 - 2 * t) → (y = 5 * t + 6) → (x = -2) → y = 37 / 2 :=
by
  intros x y t h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_minus_2_l673_67363


namespace NUMINAMATH_GPT_final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l673_67393

-- Define the travel distances
def travel_distances : List ℤ := [17, -9, 7, 11, -15, -3]

-- Define the fuel consumption rate
def fuel_consumption_rate : ℝ := 0.08

-- Theorem stating the final position
theorem final_position_is_east_8km :
  List.sum travel_distances = 8 :=
by
  sorry

-- Theorem stating the total fuel consumption
theorem total_fuel_consumption_is_4_96liters :
  (List.sum (travel_distances.map fun x => |x| : List ℝ)) * fuel_consumption_rate = 4.96 :=
by
  sorry

end NUMINAMATH_GPT_final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l673_67393


namespace NUMINAMATH_GPT_part1_part2_l673_67337

variable {a b : ℝ}

noncomputable def in_interval (x: ℝ) : Prop :=
  -1/2 < x ∧ x < 1/2

theorem part1 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1/3 * a + 1/6 * b) < 1/4 := 
by sorry

theorem part2 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1 - 4 * a * b) > 2 * abs (a - b) := 
by sorry

end NUMINAMATH_GPT_part1_part2_l673_67337


namespace NUMINAMATH_GPT_claire_initial_balloons_l673_67380

theorem claire_initial_balloons (B : ℕ) (h : B - 12 - 9 + 11 = 39) : B = 49 :=
by sorry

end NUMINAMATH_GPT_claire_initial_balloons_l673_67380


namespace NUMINAMATH_GPT_least_num_subtracted_l673_67373

theorem least_num_subtracted 
  {x : ℤ} 
  (h5 : (642 - x) % 5 = 4) 
  (h7 : (642 - x) % 7 = 4) 
  (h9 : (642 - x) % 9 = 4) : 
  x = 4 := 
sorry

end NUMINAMATH_GPT_least_num_subtracted_l673_67373


namespace NUMINAMATH_GPT_solve_for_x_l673_67396

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9*x)^18 = (27*x)^9 ↔ x = 1/3 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l673_67396


namespace NUMINAMATH_GPT_train_length_l673_67321

theorem train_length (v : ℝ) (t : ℝ) (l_b : ℝ) (v_r : v = 52) (t_r : t = 34.61538461538461) (l_b_r : l_b = 140) : 
  ∃ l_t : ℝ, l_t = 360 :=
by
  have speed_ms := v * (1000 / 3600)
  have total_distance := speed_ms * t
  have length_train := total_distance - l_b
  use length_train
  sorry

end NUMINAMATH_GPT_train_length_l673_67321


namespace NUMINAMATH_GPT_ratio_of_drinking_speeds_l673_67333

def drinking_ratio(mala_portion usha_portion : ℚ) (same_time: Bool) (usha_fraction: ℚ) : ℚ :=
if same_time then mala_portion / usha_portion else 0

theorem ratio_of_drinking_speeds
  (mala_portion : ℚ)
  (usha_portion : ℚ)
  (same_time : Bool)
  (usha_fraction : ℚ)
  (usha_drank : usha_fraction = 2 / 10)
  (mala_drank : mala_portion = 1 - usha_fraction)
  (equal_time : same_time = tt)
  (ratio : drinking_ratio mala_portion usha_portion same_time usha_fraction = 4) :
  mala_portion / usha_portion = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_drinking_speeds_l673_67333


namespace NUMINAMATH_GPT_sum_of_28_terms_l673_67306

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_of_28_terms
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h2 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∃ d ≠ 0, ∃ a₁, ∀ n, a (n + 1) = a₁ + n * d)
  (h4 : f (a 6) = f (a 23)) :
  sum_arithmetic_sequence 28 (a 1) ((a 2) - (a 1)) = 28 :=
by sorry

end NUMINAMATH_GPT_sum_of_28_terms_l673_67306


namespace NUMINAMATH_GPT_sales_tax_difference_l673_67390

-- Definitions for the conditions
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.075
def tax_rate2 : ℝ := 0.05

-- Calculations based on the conditions
def tax1 := item_price * tax_rate1
def tax2 := item_price * tax_rate2

-- The proof statement
theorem sales_tax_difference :
  tax1 - tax2 = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l673_67390


namespace NUMINAMATH_GPT_total_profit_is_27_l673_67331

noncomputable def total_profit : ℕ :=
  let natasha_money := 60
  let carla_money := natasha_money / 3
  let cosima_money := carla_money / 2
  let sergio_money := 3 * cosima_money / 2

  let natasha_spent := 4 * 15
  let carla_spent := 6 * 10
  let cosima_spent := 5 * 8
  let sergio_spent := 3 * 12

  let natasha_profit := natasha_spent * 10 / 100
  let carla_profit := carla_spent * 15 / 100
  let cosima_profit := cosima_spent * 12 / 100
  let sergio_profit := sergio_spent * 20 / 100

  natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_is_27 : total_profit = 27 := by
  sorry

end NUMINAMATH_GPT_total_profit_is_27_l673_67331


namespace NUMINAMATH_GPT_correct_operation_l673_67310

theorem correct_operation : ¬ (-2 * x + 5 * x = -7 * x) 
                          ∧ (y * x - 3 * x * y = -2 * x * y) 
                          ∧ ¬ (-x^2 - x^2 = 0) 
                          ∧ ¬ (x^2 - x = x) := 
by {
    sorry
}

end NUMINAMATH_GPT_correct_operation_l673_67310


namespace NUMINAMATH_GPT_anne_find_bottle_caps_l673_67364

theorem anne_find_bottle_caps 
  (n_i n_f : ℕ) (h_initial : n_i = 10) (h_final : n_f = 15) : n_f - n_i = 5 :=
by
  sorry

end NUMINAMATH_GPT_anne_find_bottle_caps_l673_67364


namespace NUMINAMATH_GPT_range_of_x_l673_67379

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) : x > 1/3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l673_67379


namespace NUMINAMATH_GPT_resulting_ratio_correct_l673_67312

-- Define initial conditions
def initial_coffee : ℕ := 20
def joe_drank : ℕ := 3
def joe_added_cream : ℕ := 4
def joAnn_added_cream : ℕ := 3
def joAnn_drank : ℕ := 4

-- Define the resulting amounts of cream
def joe_cream : ℕ := joe_added_cream
def joAnn_initial_cream_frac : ℚ := joAnn_added_cream / (initial_coffee + joAnn_added_cream)
def joAnn_cream_drank : ℚ := (joAnn_drank : ℚ) * joAnn_initial_cream_frac
def joAnn_cream_left : ℚ := joAnn_added_cream - joAnn_cream_drank

-- Define the resulting ratio of cream in Joe's coffee to JoAnn's coffee
def resulting_ratio : ℚ := joe_cream / joAnn_cream_left

-- Theorem stating the resulting ratio is 92/45
theorem resulting_ratio_correct : resulting_ratio = 92 / 45 :=
by
  unfold resulting_ratio joe_cream joAnn_cream_left joAnn_cream_drank joAnn_initial_cream_frac
  norm_num
  sorry

end NUMINAMATH_GPT_resulting_ratio_correct_l673_67312


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l673_67388

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l673_67388


namespace NUMINAMATH_GPT_mixture_solution_l673_67351

theorem mixture_solution (x y : ℝ) :
  (0.30 * x + 0.40 * y = 32) →
  (x + y = 100) →
  (x = 80) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_mixture_solution_l673_67351


namespace NUMINAMATH_GPT_sequence_sum_l673_67355

theorem sequence_sum (a b : ℤ) (h1 : ∃ d, d = 5 ∧ (∀ n : ℕ, (3 + n * d) = a ∨ (3 + (n-1) * d) = b ∨ (3 + (n-2) * d) = 33)) : 
  a + b = 51 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l673_67355


namespace NUMINAMATH_GPT_range_of_a_l673_67320

-- Definitions for the conditions
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0

-- Main theorem
theorem range_of_a (a : ℝ) (h : a < 0) : (¬ (∃ x, prop_p a x)) → (¬ (∃ x, ¬ prop_q x)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l673_67320


namespace NUMINAMATH_GPT_max_constant_N_l673_67372

theorem max_constant_N (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0):
  (c^2 + d^2) ≠ 0 → ∃ N, N = 1 ∧ (a^2 + b^2) / (c^2 + d^2) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_constant_N_l673_67372


namespace NUMINAMATH_GPT_put_letters_in_mailboxes_l673_67389

theorem put_letters_in_mailboxes :
  (3:ℕ)^4 = 81 :=
by
  sorry

end NUMINAMATH_GPT_put_letters_in_mailboxes_l673_67389


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_is_18_l673_67347

/-- Problem statement:
The length of a rectangular plot is thrice its breadth. 
If the area of the rectangular plot is 972 sq m, 
this theorem proves that the breadth of the rectangular plot is 18 meters.
-/
theorem breadth_of_rectangular_plot_is_18 (b l : ℝ) (h_length : l = 3 * b) (h_area : l * b = 972) : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_is_18_l673_67347


namespace NUMINAMATH_GPT_matt_climbing_speed_l673_67324

theorem matt_climbing_speed :
  ∃ (x : ℝ), (12 * 7 = 7 * x + 42) ∧ x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_matt_climbing_speed_l673_67324


namespace NUMINAMATH_GPT_ned_games_l673_67325

theorem ned_games (F: ℕ) (bought_from_friend garage_sale non_working good total_games: ℕ) 
  (h₁: bought_from_friend = F)
  (h₂: garage_sale = 27)
  (h₃: non_working = 74)
  (h₄: good = 3)
  (h₅: total_games = non_working + good)
  (h₆: total_games = bought_from_friend + garage_sale) :
  F = 50 :=
by
  sorry

end NUMINAMATH_GPT_ned_games_l673_67325


namespace NUMINAMATH_GPT_find_point_C_find_area_triangle_ABC_l673_67334

noncomputable section

-- Given points and equations
def point_B : ℝ × ℝ := (4, 4)
def eq_angle_bisector : ℝ × ℝ → Prop := λ p => p.2 = 0
def eq_altitude : ℝ × ℝ → Prop := λ p => p.1 - 2 * p.2 + 2 = 0

-- Target coordinates of point C
def point_C : ℝ × ℝ := (10, -8)

-- Coordinates of point A derived from given conditions
def point_A : ℝ × ℝ := (-2, 0)

-- Line equations derived from conditions
def eq_line_BC : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 - 12 = 0
def eq_line_AC : ℝ × ℝ → Prop := λ p => 2 * p.1 + 3 * p.2 + 4 = 0

-- Prove the coordinates of point C
theorem find_point_C : ∃ C : ℝ × ℝ, eq_line_BC C ∧ eq_line_AC C ∧ C = point_C := by
  sorry

-- Prove the area of triangle ABC.
theorem find_area_triangle_ABC : ∃ S : ℝ, S = 48 := by
  sorry

end NUMINAMATH_GPT_find_point_C_find_area_triangle_ABC_l673_67334


namespace NUMINAMATH_GPT_remainder_2023_div_73_l673_67305

theorem remainder_2023_div_73 : 2023 % 73 = 52 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_2023_div_73_l673_67305


namespace NUMINAMATH_GPT_value_of_m_l673_67357

theorem value_of_m : 5^2 + 7 = 4^3 + m → m = -32 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_m_l673_67357


namespace NUMINAMATH_GPT_depth_of_first_hole_l673_67338

theorem depth_of_first_hole (n1 t1 n2 t2 : ℕ) (D : ℝ) (r : ℝ) 
  (h1 : n1 = 45) (h2 : t1 = 8) (h3 : n2 = 90) (h4 : t2 = 6) 
  (h5 : r = 1 / 12) (h6 : D = n1 * t1 * r) (h7 : n2 * t2 * r = 45) : 
  D = 30 := 
by 
  sorry

end NUMINAMATH_GPT_depth_of_first_hole_l673_67338


namespace NUMINAMATH_GPT_no_solution_inequality_l673_67385

theorem no_solution_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| < 4 * x - 1 ∧ x < a) ↔ a ≤ (2/3) := by sorry

end NUMINAMATH_GPT_no_solution_inequality_l673_67385


namespace NUMINAMATH_GPT_distance_by_land_l673_67397

theorem distance_by_land (distance_by_sea total_distance distance_by_land : ℕ)
  (h1 : total_distance = 601)
  (h2 : distance_by_sea = 150)
  (h3 : total_distance = distance_by_land + distance_by_sea) : distance_by_land = 451 := by
  sorry

end NUMINAMATH_GPT_distance_by_land_l673_67397


namespace NUMINAMATH_GPT_min_isosceles_triangle_area_l673_67315

theorem min_isosceles_triangle_area 
  (x y n : ℕ)
  (h1 : 2 * x * y = 7 * n^2)
  (h2 : ∃ m k, m = n / 2 ∧ k = 2 * m) 
  (h3 : n % 3 = 0) : 
  x = 4 * n / 3 ∧ y = n / 3 ∧ 
  ∃ A, A = 21 / 4 := 
sorry

end NUMINAMATH_GPT_min_isosceles_triangle_area_l673_67315


namespace NUMINAMATH_GPT_october_profit_condition_l673_67386

noncomputable def calculate_profit (price_reduction : ℝ) : ℝ :=
  (50 - price_reduction) * (500 + 20 * price_reduction)

theorem october_profit_condition (x : ℝ) (h : calculate_profit x = 28000) : x = 10 ∨ x = 15 := 
by
  sorry

end NUMINAMATH_GPT_october_profit_condition_l673_67386


namespace NUMINAMATH_GPT_distance_between_centers_of_externally_tangent_circles_l673_67352

noncomputable def external_tangent_distance (R r : ℝ) (hR : R = 2) (hr : r = 3) (tangent : R > 0 ∧ r > 0) : ℝ :=
  R + r

theorem distance_between_centers_of_externally_tangent_circles :
  external_tangent_distance 2 3 (by rfl) (by rfl) (by norm_num) = 5 :=
sorry

end NUMINAMATH_GPT_distance_between_centers_of_externally_tangent_circles_l673_67352


namespace NUMINAMATH_GPT_last_three_digits_of_3_pow_5000_l673_67365

theorem last_three_digits_of_3_pow_5000 : (3 ^ 5000) % 1000 = 1 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_last_three_digits_of_3_pow_5000_l673_67365


namespace NUMINAMATH_GPT_polynomial_roots_l673_67317

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l673_67317


namespace NUMINAMATH_GPT_tom_four_times_cindy_years_ago_l673_67348

variables (t c x : ℕ)

-- Conditions
axiom cond1 : t + 5 = 2 * (c + 5)
axiom cond2 : t - 13 = 3 * (c - 13)

-- Question to prove
theorem tom_four_times_cindy_years_ago :
  t - x = 4 * (c - x) → x = 19 :=
by
  intros h
  -- simply skip the proof for now
  sorry

end NUMINAMATH_GPT_tom_four_times_cindy_years_ago_l673_67348


namespace NUMINAMATH_GPT_frequency_of_group_5_l673_67327

theorem frequency_of_group_5 (total_students freq1 freq2 freq3 freq4 : ℕ)
  (h_total: total_students = 50) 
  (h_freq1: freq1 = 7) 
  (h_freq2: freq2 = 12) 
  (h_freq3: freq3 = 13) 
  (h_freq4: freq4 = 8) :
  (50 - (7 + 12 + 13 + 8)) / 50 = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_group_5_l673_67327


namespace NUMINAMATH_GPT_black_white_tile_ratio_l673_67316

theorem black_white_tile_ratio :
  let original_black_tiles := 10
  let original_white_tiles := 15
  let total_tiles_in_original_square := original_black_tiles + original_white_tiles
  let side_length_of_original_square := Int.sqrt total_tiles_in_original_square -- this should be 5
  let side_length_of_extended_square := side_length_of_original_square + 2
  let total_black_tiles_in_border := 4 * (side_length_of_extended_square - 1) / 2 -- Each border side starts and ends with black
  let total_white_tiles_in_border := (side_length_of_extended_square * 4 - 4) - total_black_tiles_in_border 
  let new_total_black_tiles := original_black_tiles + total_black_tiles_in_border
  let new_total_white_tiles := original_white_tiles + total_white_tiles_in_border
  (new_total_black_tiles / gcd new_total_black_tiles new_total_white_tiles) / 
  (new_total_white_tiles / gcd new_total_black_tiles new_total_white_tiles) = 26 / 23 :=
by
  sorry

end NUMINAMATH_GPT_black_white_tile_ratio_l673_67316


namespace NUMINAMATH_GPT_man_savings_percentage_l673_67314

theorem man_savings_percentage
  (salary expenses : ℝ)
  (increase_percentage : ℝ)
  (current_savings : ℝ)
  (P : ℝ)
  (h1 : salary = 7272.727272727273)
  (h2 : increase_percentage = 0.05)
  (h3 : current_savings = 400)
  (h4 : current_savings + (increase_percentage * salary) = (P / 100) * salary) :
  P = 10.5 := 
sorry

end NUMINAMATH_GPT_man_savings_percentage_l673_67314


namespace NUMINAMATH_GPT_stationery_cost_l673_67375

theorem stationery_cost (cost_per_pencil cost_per_pen : ℕ)
    (boxes : ℕ)
    (pencils_per_box pens_offset : ℕ)
    (total_cost : ℕ) :
    cost_per_pencil = 4 →
    boxes = 15 →
    pencils_per_box = 80 →
    pens_offset = 300 →
    cost_per_pen = 5 →
    total_cost = (boxes * pencils_per_box * cost_per_pencil) +
                 ((2 * (boxes * pencils_per_box + pens_offset)) * cost_per_pen) →
    total_cost = 18300 :=
by
  intros
  sorry

end NUMINAMATH_GPT_stationery_cost_l673_67375


namespace NUMINAMATH_GPT_equal_white_black_balls_l673_67354

theorem equal_white_black_balls (b w n x : ℕ) 
(h1 : x = n - x)
: (x = b + w - n + x - w) := sorry

end NUMINAMATH_GPT_equal_white_black_balls_l673_67354


namespace NUMINAMATH_GPT_rectangle_ratio_constant_l673_67370

theorem rectangle_ratio_constant (length width : ℝ) (d k : ℝ)
  (h1 : length/width = 5/2)
  (h2 : 2 * (length + width) = 28)
  (h3 : d^2 = length^2 + width^2)
  (h4 : (length * width) = k * d^2) :
  k = (10/29) := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_constant_l673_67370
