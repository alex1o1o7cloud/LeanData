import Mathlib

namespace NUMINAMATH_GPT_probability_of_chosen_primes_l1107_110769

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def total_ways : ℕ := Nat.choose 30 2
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def primes_not_divisible_by_5 : List ℕ := [2, 3, 7, 11, 13, 17, 19, 23, 29]

def chosen_primes (s : Finset ℕ) : Prop :=
  s.card = 2 ∧
  (∀ n ∈ s, n ∈ primes_not_divisible_by_5)  ∧
  (∀ n ∈ s, n ≠ 5) -- (5 is already excluded in the prime list, but for completeness)

def favorable_ways : ℕ := Nat.choose 9 2  -- 9 primes not divisible by 5

def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_chosen_primes:
  probability = (12 / 145 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_chosen_primes_l1107_110769


namespace NUMINAMATH_GPT_Christina_driving_time_l1107_110798

theorem Christina_driving_time 
  (speed_Christina : ℕ) 
  (speed_friend : ℕ) 
  (total_distance : ℕ)
  (friend_driving_time : ℕ) 
  (distance_by_Christina : ℕ) 
  (time_driven_by_Christina : ℕ) 
  (total_driving_time : ℕ)
  (h1 : speed_Christina = 30)
  (h2 : speed_friend = 40) 
  (h3 : total_distance = 210)
  (h4 : friend_driving_time = 3)
  (h5 : speed_friend * friend_driving_time = 120)
  (h6 : total_distance - 120 = distance_by_Christina)
  (h7 : distance_by_Christina = 90)
  (h8 : distance_by_Christina / speed_Christina = 3)
  (h9 : time_driven_by_Christina = 3)
  (h10 : time_driven_by_Christina * 60 = 180) :
    total_driving_time = 180 := 
by
  sorry

end NUMINAMATH_GPT_Christina_driving_time_l1107_110798


namespace NUMINAMATH_GPT_total_amount_invested_l1107_110791

-- Define the conditions and specify the correct answer
theorem total_amount_invested (x y : ℝ) (h8 : y = 600) 
  (h_income_diff : 0.10 * (x - 600) - 0.08 * 600 = 92) : 
  x + y = 2000 := sorry

end NUMINAMATH_GPT_total_amount_invested_l1107_110791


namespace NUMINAMATH_GPT_gcd_m_n_is_one_l1107_110744

open Int
open Nat

-- Define m and n based on the given conditions
def m : ℤ := 130^2 + 240^2 + 350^2
def n : ℤ := 129^2 + 239^2 + 351^2

-- State the theorem to be proven
theorem gcd_m_n_is_one : gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_is_one_l1107_110744


namespace NUMINAMATH_GPT_range_of_k_l1107_110735

theorem range_of_k (k : ℤ) (a : ℤ → ℤ) (h_a : ∀ n : ℕ, a n = |n - k| + |n + 2 * k|)
  (h_a3_equal_a4 : a 3 = a 4) : k ≤ -2 ∨ k ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1107_110735


namespace NUMINAMATH_GPT_profit_share_difference_l1107_110709

noncomputable def ratio (x y : ℕ) : ℕ := x / Nat.gcd x y

def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1900
def total_parts : ℕ := 15  -- Sum of the ratio parts (4 for A, 5 for B, 6 for C)
def part_amount : ℕ := profit_share_B / 5  -- 5 parts of B

def profit_share_A : ℕ := 4 * part_amount
def profit_share_C : ℕ := 6 * part_amount

theorem profit_share_difference :
  (profit_share_C - profit_share_A) = 760 := by
  sorry

end NUMINAMATH_GPT_profit_share_difference_l1107_110709


namespace NUMINAMATH_GPT_no_perfect_squares_exist_l1107_110737

theorem no_perfect_squares_exist (x y : ℕ) :
  ¬(∃ k1 k2 : ℕ, x^2 + y = k1^2 ∧ y^2 + x = k2^2) :=
sorry

end NUMINAMATH_GPT_no_perfect_squares_exist_l1107_110737


namespace NUMINAMATH_GPT_Elmer_vs_Milton_food_l1107_110775

def Penelope_daily_food := 20  -- Penelope eats 20 pounds per day
def Greta_to_Penelope_ratio := 1 / 10  -- Greta eats 1/10 of what Penelope eats
def Milton_to_Greta_ratio := 1 / 100  -- Milton eats 1/100 of what Greta eats
def Elmer_to_Penelope_difference := 60  -- Elmer eats 60 pounds more than Penelope

def Greta_daily_food := Penelope_daily_food * Greta_to_Penelope_ratio
def Milton_daily_food := Greta_daily_food * Milton_to_Greta_ratio
def Elmer_daily_food := Penelope_daily_food + Elmer_to_Penelope_difference

theorem Elmer_vs_Milton_food :
  Elmer_daily_food = 4000 * Milton_daily_food := by
  sorry

end NUMINAMATH_GPT_Elmer_vs_Milton_food_l1107_110775


namespace NUMINAMATH_GPT_unique_digit_sum_l1107_110776

theorem unique_digit_sum (Y M E T : ℕ) (h1 : Y ≠ M) (h2 : Y ≠ E) (h3 : Y ≠ T)
    (h4 : M ≠ E) (h5 : M ≠ T) (h6 : E ≠ T) (h7 : 10 * Y + E = YE) (h8 : 10 * M + E = ME)
    (h9 : YE * ME = T * T * T) (hT_even : T % 2 = 0) : 
    Y + M + E + T = 10 :=
  sorry

end NUMINAMATH_GPT_unique_digit_sum_l1107_110776


namespace NUMINAMATH_GPT_temperature_on_friday_l1107_110730

-- Define the temperatures on different days
variables (T W Th F : ℝ)

-- Define the conditions
def condition1 : Prop := (T + W + Th) / 3 = 32
def condition2 : Prop := (W + Th + F) / 3 = 34
def condition3 : Prop := T = 38

-- State the theorem to prove the temperature on Friday
theorem temperature_on_friday (h1 : condition1 T W Th) (h2 : condition2 W Th F) (h3 : condition3 T) : F = 44 :=
  sorry

end NUMINAMATH_GPT_temperature_on_friday_l1107_110730


namespace NUMINAMATH_GPT_lean_proof_l1107_110783

noncomputable def proof_problem (a b c d : ℝ) (habcd : a * b * c * d = 1) : Prop :=
  (1 + a * b) / (1 + a) ^ 2008 +
  (1 + b * c) / (1 + b) ^ 2008 +
  (1 + c * d) / (1 + c) ^ 2008 +
  (1 + d * a) / (1 + d) ^ 2008 ≥ 4

theorem lean_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_abcd : a * b * c * d = 1) : proof_problem a b c d h_abcd :=
  sorry

end NUMINAMATH_GPT_lean_proof_l1107_110783


namespace NUMINAMATH_GPT_roger_and_friend_fraction_l1107_110745

theorem roger_and_friend_fraction 
  (total_distance : ℝ) 
  (fraction_driven_before_lunch : ℝ) 
  (lunch_time : ℝ) 
  (total_time : ℝ) 
  (same_speed : Prop) 
  (driving_time_before_lunch : ℝ)
  (driving_time_after_lunch : ℝ) :
  total_distance = 200 ∧
  lunch_time = 1 ∧
  total_time = 5 ∧
  driving_time_before_lunch = 1 ∧
  driving_time_after_lunch = (total_time - lunch_time - driving_time_before_lunch) ∧
  same_speed = (total_distance * fraction_driven_before_lunch / driving_time_before_lunch = total_distance * (1 - fraction_driven_before_lunch) / driving_time_after_lunch) →
  fraction_driven_before_lunch = 1 / 4 :=
sorry

end NUMINAMATH_GPT_roger_and_friend_fraction_l1107_110745


namespace NUMINAMATH_GPT_minimize_expr_l1107_110763

theorem minimize_expr : ∃ c : ℝ, (∀ d : ℝ, (3/4 * c^2 - 9 * c + 5) ≤ (3/4 * d^2 - 9 * d + 5)) ∧ c = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_minimize_expr_l1107_110763


namespace NUMINAMATH_GPT_exclude_13_code_count_l1107_110779

/-- The number of 5-digit codes (00000 to 99999) that don't contain the sequence "13". -/
theorem exclude_13_code_count :
  let total_codes := 100000
  let excluded_codes := 3970
  total_codes - excluded_codes = 96030 :=
by
  let total_codes := 100000
  let excluded_codes := 3970
  have h : total_codes - excluded_codes = 96030 := by
    -- Provide mathematical proof or use sorry for placeholder
    sorry
  exact h

end NUMINAMATH_GPT_exclude_13_code_count_l1107_110779


namespace NUMINAMATH_GPT_prob_c_not_adjacent_to_a_or_b_l1107_110786

-- Definitions for the conditions
def num_students : ℕ := 7
def a_and_b_together : Prop := true
def c_on_edge : Prop := true

-- Main theorem: probability c not adjacent to a or b under given conditions
theorem prob_c_not_adjacent_to_a_or_b
  (h1 : a_and_b_together)
  (h2 : c_on_edge) :
  ∃ (p : ℚ), p = 0.8 := by
  sorry

end NUMINAMATH_GPT_prob_c_not_adjacent_to_a_or_b_l1107_110786


namespace NUMINAMATH_GPT_polynomial_evaluation_at_8_l1107_110792

def P (x : ℝ) : ℝ := x^3 + 2*x^2 + x - 1

theorem polynomial_evaluation_at_8 : P 8 = 647 :=
by sorry

end NUMINAMATH_GPT_polynomial_evaluation_at_8_l1107_110792


namespace NUMINAMATH_GPT_minute_hand_rotation_l1107_110741

theorem minute_hand_rotation (minutes : ℕ) (degrees_per_minute : ℝ) (radian_conversion_factor : ℝ) : 
  minutes = 10 → 
  degrees_per_minute = 360 / 60 → 
  radian_conversion_factor = π / 180 → 
  (-(degrees_per_minute * minutes * radian_conversion_factor) = -(π / 3)) := 
by
  intros hminutes hdegrees hfactor
  rw [hminutes, hdegrees, hfactor]
  simp
  sorry

end NUMINAMATH_GPT_minute_hand_rotation_l1107_110741


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1107_110771

-- Define the arithmetic sequence parameters
def first_term : Int := 2
def common_difference : Int := 5

-- Define the formula to calculate the n-th term of the sequence
def nth_term (n : Nat) : Int :=
  first_term + (n - 1) * common_difference

-- Prove that the 50th term of the sequence is 247
theorem arithmetic_sequence_50th_term : nth_term 50 = 247 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1107_110771


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l1107_110762

theorem last_four_digits_of_5_pow_2011 :
  (5 ^ 5) % 10000 = 3125 ∧
  (5 ^ 6) % 10000 = 5625 ∧
  (5 ^ 7) % 10000 = 8125 →
  (5 ^ 2011) % 10000 = 8125 :=
by
  sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l1107_110762


namespace NUMINAMATH_GPT_inverse_proportion_shift_l1107_110768

theorem inverse_proportion_shift (x : ℝ) : 
  (∀ x, y = 6 / x) -> (y = 6 / (x - 3)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inverse_proportion_shift_l1107_110768


namespace NUMINAMATH_GPT_scaled_polynomial_roots_l1107_110796

noncomputable def polynomial_with_scaled_roots : Polynomial ℂ :=
  Polynomial.X^3 - 3*Polynomial.X^2 + 5

theorem scaled_polynomial_roots :
  (∃ r1 r2 r3 : ℂ, polynomial_with_scaled_roots.eval r1 = 0 ∧ polynomial_with_scaled_roots.eval r2 = 0 ∧ polynomial_with_scaled_roots.eval r3 = 0 ∧
  (∃ q : Polynomial ℂ, q = Polynomial.X^3 - 9*Polynomial.X^2 + 135 ∧
  ∀ y, (q.eval y = 0 ↔ (polynomial_with_scaled_roots.eval (y / 3) = 0)))) := sorry

end NUMINAMATH_GPT_scaled_polynomial_roots_l1107_110796


namespace NUMINAMATH_GPT_arithmetic_sequence_positive_l1107_110781

theorem arithmetic_sequence_positive (d a_1 : ℤ) (n : ℤ) :
  (a_11 - a_8 = 3) -> 
  (S_11 - S_8 = 33) ->
  (n > 0) ->
  a_1 + (n-1) * d > 0 ->
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_positive_l1107_110781


namespace NUMINAMATH_GPT_choose_president_and_committee_l1107_110702

-- Define the condition of the problem
def total_people := 10
def committee_size := 3

-- Define the function to calculate the number of combinations
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Proving the number of ways to choose the president and the committee
theorem choose_president_and_committee :
  (total_people * comb (total_people - 1) committee_size) = 840 :=
by
  sorry

end NUMINAMATH_GPT_choose_president_and_committee_l1107_110702


namespace NUMINAMATH_GPT_john_calories_eaten_l1107_110742

def servings : ℕ := 3
def calories_per_serving : ℕ := 120
def fraction_eaten : ℚ := 1 / 2

theorem john_calories_eaten : 
  (servings * calories_per_serving : ℕ) * fraction_eaten = 180 :=
  sorry

end NUMINAMATH_GPT_john_calories_eaten_l1107_110742


namespace NUMINAMATH_GPT_seq_eventually_reaches_one_l1107_110765

theorem seq_eventually_reaches_one (a : ℕ → ℤ) (h₁ : a 1 > 0) :
  (∀ n, n % 4 = 0 → a (n + 1) = a n / 2) →
  (∀ n, n % 4 = 1 → a (n + 1) = 3 * a n + 1) →
  (∀ n, n % 4 = 2 → a (n + 1) = 2 * a n - 1) →
  (∀ n, n % 4 = 3 → a (n + 1) = (a n + 1) / 4) →
  ∃ m, a m = 1 :=
by
  sorry

end NUMINAMATH_GPT_seq_eventually_reaches_one_l1107_110765


namespace NUMINAMATH_GPT_smallest_b_greater_than_5_perfect_cube_l1107_110734

theorem smallest_b_greater_than_5_perfect_cube : ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 3 = n ^ 3 ∧ b = 6 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_b_greater_than_5_perfect_cube_l1107_110734


namespace NUMINAMATH_GPT_Juwella_reads_pages_l1107_110784

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end NUMINAMATH_GPT_Juwella_reads_pages_l1107_110784


namespace NUMINAMATH_GPT_carpenter_needs_more_logs_l1107_110789

-- Define the given conditions in Lean 4
def total_woodblocks_needed : ℕ := 80
def logs_on_hand : ℕ := 8
def woodblocks_per_log : ℕ := 5

-- Statement: Proving the number of additional logs the carpenter needs
theorem carpenter_needs_more_logs :
  let woodblocks_available := logs_on_hand * woodblocks_per_log
  let additional_woodblocks := total_woodblocks_needed - woodblocks_available
  additional_woodblocks / woodblocks_per_log = 8 :=
by
  sorry

end NUMINAMATH_GPT_carpenter_needs_more_logs_l1107_110789


namespace NUMINAMATH_GPT_maximum_value_g_on_interval_l1107_110764

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g_on_interval : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 := by
  sorry

end NUMINAMATH_GPT_maximum_value_g_on_interval_l1107_110764


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_eq_70_l1107_110714

-- Define the conditions
def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 60
def isosceles_triangle_base : ℕ := 30

-- Calculate the side of equilateral triangle
def equilateral_triangle_side : ℕ := equilateral_triangle_perimeter / 3

-- Lean 4 statement
theorem isosceles_triangle_perimeter_eq_70 :
  ∃ (a b c : ℕ), is_equilateral_triangle a b c ∧ 
  a + b + c = equilateral_triangle_perimeter →
  (is_isosceles_triangle a a isosceles_triangle_base) →
  a + a + isosceles_triangle_base = 70 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_isosceles_triangle_perimeter_eq_70_l1107_110714


namespace NUMINAMATH_GPT_gcd_of_228_and_1995_l1107_110723

theorem gcd_of_228_and_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_228_and_1995_l1107_110723


namespace NUMINAMATH_GPT_find_g6_minus_g2_div_g3_l1107_110704

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition (a c : ℝ) : c^3 * g a = a^3 * g c
axiom g_nonzero : g 3 ≠ 0

theorem find_g6_minus_g2_div_g3 : (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end NUMINAMATH_GPT_find_g6_minus_g2_div_g3_l1107_110704


namespace NUMINAMATH_GPT_bat_wings_area_l1107_110732

-- Defining a rectangle and its properties.
structure Rectangle where
  PQ : ℝ
  QR : ℝ
  PT : ℝ
  TR : ℝ
  RQ : ℝ

-- Example rectangle from the problem
def PQRS : Rectangle := { PQ := 5, QR := 3, PT := 1, TR := 1, RQ := 1 }

-- Calculate area of "bat wings" if the rectangle is specified as in the above structure.
-- Expected result is 3.5
theorem bat_wings_area (r : Rectangle) (hPQ : r.PQ = 5) (hQR : r.QR = 3) 
    (hPT : r.PT = 1) (hTR : r.TR = 1) (hRQ : r.RQ = 1) : 
    ∃ area : ℝ, area = 3.5 :=
by
  -- Adding the proof would involve geometric calculations.
  -- Skipping the proof for now.
  sorry

end NUMINAMATH_GPT_bat_wings_area_l1107_110732


namespace NUMINAMATH_GPT_prob_not_rain_correct_l1107_110760

noncomputable def prob_not_rain_each_day (prob_rain : ℚ) : ℚ :=
  1 - prob_rain

noncomputable def prob_not_rain_four_days (prob_not_rain : ℚ) : ℚ :=
  prob_not_rain ^ 4

theorem prob_not_rain_correct :
  prob_not_rain_four_days (prob_not_rain_each_day (2/3)) = 1 / 81 :=
by 
  sorry

end NUMINAMATH_GPT_prob_not_rain_correct_l1107_110760


namespace NUMINAMATH_GPT_parabola_vertex_y_coord_l1107_110770

theorem parabola_vertex_y_coord (a b c x y : ℝ) (h : a = 2 ∧ b = 16 ∧ c = 35 ∧ y = a*x^2 + b*x + c ∧ x = -b / (2 * a)) : y = 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_y_coord_l1107_110770


namespace NUMINAMATH_GPT_equality_or_neg_equality_of_eq_l1107_110736

theorem equality_or_neg_equality_of_eq
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^3 / a = b^2 + a^3 / b) : a = b ∨ a = -b := 
  by
  sorry

end NUMINAMATH_GPT_equality_or_neg_equality_of_eq_l1107_110736


namespace NUMINAMATH_GPT_total_feathers_needed_l1107_110799

theorem total_feathers_needed 
  (animals_group1 : ℕ) (feathers_group1 : ℕ)
  (animals_group2 : ℕ) (feathers_group2 : ℕ) 
  (total_feathers : ℕ) :
  animals_group1 = 934 →
  feathers_group1 = 7 →
  animals_group2 = 425 →
  feathers_group2 = 12 →
  total_feathers = 11638 :=
by sorry

end NUMINAMATH_GPT_total_feathers_needed_l1107_110799


namespace NUMINAMATH_GPT_minimum_value_quadratic_expression_l1107_110750

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end NUMINAMATH_GPT_minimum_value_quadratic_expression_l1107_110750


namespace NUMINAMATH_GPT_editors_min_count_l1107_110747

theorem editors_min_count
  (writers : ℕ)
  (P : ℕ)
  (S : ℕ)
  (W : ℕ)
  (H1 : writers = 45)
  (H2 : P = 90)
  (H3 : ∀ x : ℕ, x ≤ 6 → (90 = (writers + W - x) + 2 * x) → W ≥ P - 51)
  : W = 39 := by
  sorry

end NUMINAMATH_GPT_editors_min_count_l1107_110747


namespace NUMINAMATH_GPT_problem_N_lowest_terms_l1107_110715

theorem problem_N_lowest_terms :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 2500 ∧ ∃ k : ℕ, k ∣ 128 ∧ (n + 11) % k = 0 ∧ (Nat.gcd (n^2 + 7) (n + 11)) > 1) →
  ∃ cnt : ℕ, cnt = 168 :=
by
  sorry

end NUMINAMATH_GPT_problem_N_lowest_terms_l1107_110715


namespace NUMINAMATH_GPT_cross_out_number_l1107_110720

theorem cross_out_number (n : ℤ) (h1 : 5 * n + 10 = 10085) : n = 2015 → (n + 5 = 2020) :=
by
  sorry

end NUMINAMATH_GPT_cross_out_number_l1107_110720


namespace NUMINAMATH_GPT_vegetarian_family_member_count_l1107_110718

variable (total_family : ℕ) (vegetarian_only : ℕ) (non_vegetarian_only : ℕ)
variable (both_vegetarian_nonvegetarian : ℕ) (vegan_only : ℕ)
variable (pescatarian : ℕ) (specific_vegetarian : ℕ)

theorem vegetarian_family_member_count :
  total_family = 35 →
  vegetarian_only = 11 →
  non_vegetarian_only = 6 →
  both_vegetarian_nonvegetarian = 9 →
  vegan_only = 3 →
  pescatarian = 4 →
  specific_vegetarian = 2 →
  vegetarian_only + both_vegetarian_nonvegetarian + vegan_only + pescatarian + specific_vegetarian = 29 :=
by
  intros
  sorry

end NUMINAMATH_GPT_vegetarian_family_member_count_l1107_110718


namespace NUMINAMATH_GPT_smallest_share_arith_seq_l1107_110749

theorem smallest_share_arith_seq (a1 d : ℚ) (h1 : 5 * a1 + 10 * d = 100) (h2 : (3 * a1 + 9 * d) * (1 / 7) = 2 * a1 + d) : a1 = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_share_arith_seq_l1107_110749


namespace NUMINAMATH_GPT_roots_of_equation_l1107_110778

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1107_110778


namespace NUMINAMATH_GPT_school_band_fundraising_l1107_110773

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end NUMINAMATH_GPT_school_band_fundraising_l1107_110773


namespace NUMINAMATH_GPT_partition_exists_min_n_in_A_l1107_110794

-- Definition of subsets and their algebraic properties
variable (A B C : Set ℕ)

-- The Initial conditions
axiom A_squared_eq_A : ∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)
axiom B_squared_eq_C : ∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)
axiom C_squared_eq_B : ∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)
axiom AB_eq_B : ∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)
axiom AC_eq_C : ∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)
axiom BC_eq_A : ∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)

-- Statement for the partition existence with given conditions
theorem partition_exists :
  ∃ A B C : Set ℕ, (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
               (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
               (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
               (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
               (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
               (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) :=
sorry

-- Statement for the minimum n in A such that n and n+1 are both in A is at most 77
theorem min_n_in_A :
  ∀ A B C : Set ℕ,
    (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
    (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
    (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
    (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
    (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
    (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) →
    ∃ n : ℕ, (n ∈ A) ∧ (n + 1 ∈ A) ∧ n ≤ 77 :=
sorry

end NUMINAMATH_GPT_partition_exists_min_n_in_A_l1107_110794


namespace NUMINAMATH_GPT_range_of_a_l1107_110759

noncomputable def f (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 2, (a - 1 / x) ≥ 0) ↔ (a ≥ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1107_110759


namespace NUMINAMATH_GPT_find_subtracted_value_l1107_110719

theorem find_subtracted_value (n x : ℕ) (h1 : n = 120) (h2 : n / 6 - x = 5) : x = 15 := by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1107_110719


namespace NUMINAMATH_GPT_probability_of_interval_is_one_third_l1107_110751

noncomputable def probability_in_interval (total_start total_end inner_start inner_end : ℝ) : ℝ :=
  (inner_end - inner_start) / (total_end - total_start)

theorem probability_of_interval_is_one_third :
  probability_in_interval 1 7 5 8 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_interval_is_one_third_l1107_110751


namespace NUMINAMATH_GPT_white_pairs_coincide_l1107_110733

theorem white_pairs_coincide 
    (red_triangles : ℕ)
    (blue_triangles : ℕ)
    (white_triangles : ℕ)
    (red_pairs : ℕ)
    (blue_pairs : ℕ)
    (red_white_pairs : ℕ)
    (coinciding_white_pairs : ℕ) :
    red_triangles = 4 → 
    blue_triangles = 6 →
    white_triangles = 10 →
    red_pairs = 3 →
    blue_pairs = 4 →
    red_white_pairs = 3 →
    coinciding_white_pairs = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l1107_110733


namespace NUMINAMATH_GPT_Δy_over_Δx_l1107_110710

-- Conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 4
def y1 : ℝ := f 1
def y2 (Δx : ℝ) : ℝ := f (1 + Δx)
def Δy (Δx : ℝ) : ℝ := y2 Δx - y1

-- Theorem statement
theorem Δy_over_Δx (Δx : ℝ) : Δy Δx / Δx = 4 + 2 * Δx := by
  sorry

end NUMINAMATH_GPT_Δy_over_Δx_l1107_110710


namespace NUMINAMATH_GPT_solve_inequality_l1107_110716

theorem solve_inequality (a : ℝ) : 
  (a > 0 → {x : ℝ | x < -a / 4 ∨ x > a / 3 } = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a = 0 → {x : ℝ | x ≠ 0} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a < 0 → {x : ℝ | x < a / 3 ∨ x > -a / 4} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1107_110716


namespace NUMINAMATH_GPT_find_c_plus_d_l1107_110711

theorem find_c_plus_d (c d : ℝ) :
  (∀ x y, (x = (1 / 3) * y + c) → (y = (1 / 3) * x + d) → (x, y) = (3, 3)) → 
  c + d = 4 :=
by
  -- ahead declaration to meet the context requirements in Lean 4
  intros h
  -- Proof steps would go here, but they are omitted
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l1107_110711


namespace NUMINAMATH_GPT_gcf_45_135_90_l1107_110766

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end NUMINAMATH_GPT_gcf_45_135_90_l1107_110766


namespace NUMINAMATH_GPT_solve_problem_l1107_110754

noncomputable def find_z_values (x : ℝ) : ℝ :=
  (x - 3)^2 * (x + 4) / (2 * x - 4)

theorem solve_problem (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  find_z_values x = 64.8 ∨ find_z_values x = -10.125 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1107_110754


namespace NUMINAMATH_GPT_pirate_rick_digging_time_l1107_110748

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_pirate_rick_digging_time_l1107_110748


namespace NUMINAMATH_GPT_shooting_prob_l1107_110727

theorem shooting_prob (p : ℝ) (h₁ : (1 / 3) * (1 / 2) * (1 - p) + (1 / 3) * (1 / 2) * p + (2 / 3) * (1 / 2) * p = 7 / 18) :
  p = 2 / 3 :=
sorry

end NUMINAMATH_GPT_shooting_prob_l1107_110727


namespace NUMINAMATH_GPT_new_weekly_income_l1107_110722

-- Define the conditions
def original_income : ℝ := 60
def raise_percentage : ℝ := 0.20

-- Define the question and the expected answer
theorem new_weekly_income : original_income * (1 + raise_percentage) = 72 := 
by
  sorry

end NUMINAMATH_GPT_new_weekly_income_l1107_110722


namespace NUMINAMATH_GPT_income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l1107_110797

-- Define the conditions
def annual_income (year : ℕ) : ℝ := 0.0124 * (1 + 0.2) ^ (year - 1)
def annual_repayment : ℝ := 0.05

-- Proof Problem 1: Show that the subway's annual operating income exceeds the annual repayment at year 9
theorem income_exceeds_repayment_after_9_years :
  ∀ n ≥ 9, annual_income n > annual_repayment :=
by
  sorry

-- Define the cumulative payment function for the municipal government
def cumulative_payment (years : ℕ) : ℝ :=
  (annual_repayment * years) - (List.sum (List.map annual_income (List.range years)))

-- Proof Problem 2: Show the cumulative payment by the municipal government up to year 8 is 19,541,135 RMB
theorem cumulative_payment_up_to_year_8 :
  cumulative_payment 8 = 0.1954113485 :=
by
  sorry

end NUMINAMATH_GPT_income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l1107_110797


namespace NUMINAMATH_GPT_cos_alpha_second_quadrant_l1107_110790

theorem cos_alpha_second_quadrant (α : ℝ) (h₁ : (π / 2) < α ∧ α < π) (h₂ : Real.sin α = 5 / 13) :
  Real.cos α = -12 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_second_quadrant_l1107_110790


namespace NUMINAMATH_GPT_garrett_total_spent_l1107_110729

/-- Garrett bought 6 oatmeal raisin granola bars, each costing $1.25. -/
def oatmeal_bars_count : Nat := 6
def oatmeal_bars_cost_per_unit : ℝ := 1.25

/-- Garrett bought 8 peanut granola bars, each costing $1.50. -/
def peanut_bars_count : Nat := 8
def peanut_bars_cost_per_unit : ℝ := 1.50

/-- The total amount spent on granola bars is $19.50. -/
theorem garrett_total_spent : oatmeal_bars_count * oatmeal_bars_cost_per_unit + peanut_bars_count * peanut_bars_cost_per_unit = 19.50 :=
by
  sorry

end NUMINAMATH_GPT_garrett_total_spent_l1107_110729


namespace NUMINAMATH_GPT_cube_volume_l1107_110708

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end NUMINAMATH_GPT_cube_volume_l1107_110708


namespace NUMINAMATH_GPT_square_side_length_l1107_110701

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l1107_110701


namespace NUMINAMATH_GPT_avg_problem_l1107_110761

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Formulate the proof problem statement
theorem avg_problem : avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end NUMINAMATH_GPT_avg_problem_l1107_110761


namespace NUMINAMATH_GPT_remaining_black_cards_l1107_110739

def total_black_cards_per_deck : ℕ := 26
def num_decks : ℕ := 5
def removed_black_face_cards : ℕ := 7
def removed_black_number_cards : ℕ := 12

theorem remaining_black_cards : total_black_cards_per_deck * num_decks - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_remaining_black_cards_l1107_110739


namespace NUMINAMATH_GPT_min_third_side_length_l1107_110706

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end NUMINAMATH_GPT_min_third_side_length_l1107_110706


namespace NUMINAMATH_GPT_xyz_logarithm_sum_l1107_110757

theorem xyz_logarithm_sum :
  ∃ (X Y Z : ℕ), X > 0 ∧ Y > 0 ∧ Z > 0 ∧
  Nat.gcd X (Nat.gcd Y Z) = 1 ∧ 
  (↑X * Real.log 3 / Real.log 180 + ↑Y * Real.log 5 / Real.log 180 = ↑Z) ∧ 
  (X + Y + Z = 4) :=
by
  sorry

end NUMINAMATH_GPT_xyz_logarithm_sum_l1107_110757


namespace NUMINAMATH_GPT_find_constants_l1107_110717

open Matrix 

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem find_constants :
  ∃ c d : ℝ, c = 1/12 ∧ d = 1/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1107_110717


namespace NUMINAMATH_GPT_bean_inside_inscribed_circle_l1107_110712

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a * a

noncomputable def inscribed_circle_radius (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r * r

noncomputable def probability_inside_circle (s_triangle s_circle : ℝ) : ℝ :=
  s_circle / s_triangle

theorem bean_inside_inscribed_circle :
  let a := 2
  let s_triangle := equilateral_triangle_area a
  let r := inscribed_circle_radius a
  let s_circle := circle_area r
  probability_inside_circle s_triangle s_circle = (Real.sqrt 3 * Real.pi / 9) :=
by
  sorry

end NUMINAMATH_GPT_bean_inside_inscribed_circle_l1107_110712


namespace NUMINAMATH_GPT_find_k_for_solutions_l1107_110740

theorem find_k_for_solutions (k : ℝ) :
  (∀ x: ℝ, x = 3 ∨ x = 5 → k * x^2 - 8 * x + 15 = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_solutions_l1107_110740


namespace NUMINAMATH_GPT_price_of_fruit_juice_l1107_110731

theorem price_of_fruit_juice (F : ℝ)
  (Sandwich_price : ℝ := 2)
  (Hamburger_price : ℝ := 2)
  (Hotdog_price : ℝ := 1)
  (Selene_purchases : ℝ := 3 * Sandwich_price + F)
  (Tanya_purchases : ℝ := 2 * Hamburger_price + 2 * F)
  (Total_spent : Selene_purchases + Tanya_purchases = 16) :
  F = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_of_fruit_juice_l1107_110731


namespace NUMINAMATH_GPT_count_games_l1107_110724

def total_teams : ℕ := 20
def games_per_pairing : ℕ := 7
def total_games := (total_teams * (total_teams - 1)) / 2 * games_per_pairing

theorem count_games : total_games = 1330 := by
  sorry

end NUMINAMATH_GPT_count_games_l1107_110724


namespace NUMINAMATH_GPT_interior_diagonals_of_dodecahedron_l1107_110788

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end NUMINAMATH_GPT_interior_diagonals_of_dodecahedron_l1107_110788


namespace NUMINAMATH_GPT_find_contributions_before_johns_l1107_110793

-- Definitions based on the conditions provided
def avg_contrib_size_after (A : ℝ) := A + 0.5 * A = 75
def johns_contribution := 100
def total_amount_before (n : ℕ) (A : ℝ) := n * A
def total_amount_after (n : ℕ) (A : ℝ) := (n * A + johns_contribution)

-- Proposition we need to prove
theorem find_contributions_before_johns (n : ℕ) (A : ℝ) :
  avg_contrib_size_after A →
  total_amount_before n A + johns_contribution = (n + 1) * 75 →
  n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_contributions_before_johns_l1107_110793


namespace NUMINAMATH_GPT_butterflies_count_l1107_110738

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end NUMINAMATH_GPT_butterflies_count_l1107_110738


namespace NUMINAMATH_GPT_boys_other_communities_l1107_110752

/-- 
In a school of 850 boys, 44% are Muslims, 28% are Hindus, 
10% are Sikhs, and the remaining belong to other communities.
Prove that the number of boys belonging to other communities is 153.
-/
theorem boys_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℚ)
  (h_total_boys : total_boys = 850)
  (h_percentage_muslims : percentage_muslims = 44)
  (h_percentage_hindus : percentage_hindus = 28)
  (h_percentage_sikhs : percentage_sikhs = 10) :
  let percentage_others := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_others := (percentage_others / 100) * total_boys
  number_others = 153 := 
by
  sorry

end NUMINAMATH_GPT_boys_other_communities_l1107_110752


namespace NUMINAMATH_GPT_slower_speed_percentage_l1107_110721

theorem slower_speed_percentage (S S' T T' D : ℝ) (h1 : T = 8) (h2 : T' = T + 24) (h3 : D = S * T) (h4 : D = S' * T') : 
  (S' / S) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_slower_speed_percentage_l1107_110721


namespace NUMINAMATH_GPT_oranges_for_price_of_apples_l1107_110707

-- Given definitions based on the conditions provided
def cost_of_apples_same_as_bananas (a b : ℕ) : Prop := 12 * a = 6 * b
def cost_of_bananas_same_as_cucumbers (b c : ℕ) : Prop := 3 * b = 5 * c
def cost_of_cucumbers_same_as_oranges (c o : ℕ) : Prop := 2 * c = 1 * o

-- The theorem to prove
theorem oranges_for_price_of_apples (a b c o : ℕ) 
  (hab : cost_of_apples_same_as_bananas a b)
  (hbc : cost_of_bananas_same_as_cucumbers b c)
  (hco : cost_of_cucumbers_same_as_oranges c o) : 
  24 * a = 10 * o :=
sorry

end NUMINAMATH_GPT_oranges_for_price_of_apples_l1107_110707


namespace NUMINAMATH_GPT_rulers_added_initially_46_finally_71_l1107_110774

theorem rulers_added_initially_46_finally_71 : 
  ∀ (initial final added : ℕ), initial = 46 → final = 71 → added = final - initial → added = 25 :=
by
  intros initial final added h_initial h_final h_added
  rw [h_initial, h_final] at h_added
  exact h_added

end NUMINAMATH_GPT_rulers_added_initially_46_finally_71_l1107_110774


namespace NUMINAMATH_GPT_first_line_shift_time_l1107_110772

theorem first_line_shift_time (x y : ℝ) (h1 : (1 / x) + (1 / (x - 2)) + (1 / y) = 1.5 * ((1 / x) + (1 / (x - 2)))) 
  (h2 : x - 24 / 5 = (1 / ((1 / (x - 2)) + (1 / y)))) :
  x = 8 :=
sorry

end NUMINAMATH_GPT_first_line_shift_time_l1107_110772


namespace NUMINAMATH_GPT_geom_seq_decreasing_l1107_110713

variable {a : ℕ → ℝ}
variable {a₁ q : ℝ}

theorem geom_seq_decreasing (h : ∀ n, a n = a₁ * q^n) (h₀ : a₁ * (q - 1) < 0) (h₁ : q > 0) :
  ∀ n, a (n + 1) < a n := 
sorry

end NUMINAMATH_GPT_geom_seq_decreasing_l1107_110713


namespace NUMINAMATH_GPT_complex_number_in_fourth_quadrant_l1107_110753

theorem complex_number_in_fourth_quadrant (i : ℂ) (z : ℂ) (hx : z = -2 * i + 1) (hy : (z.re, z.im) = (1, -2)) :
  (1, -2).1 > 0 ∧ (1, -2).2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_in_fourth_quadrant_l1107_110753


namespace NUMINAMATH_GPT_collinear_points_l1107_110795

variable (α β γ δ E : Type)
variables {A B C D K L P Q : α}
variables (convex : α → α → α → α → Prop)
variables (not_parallel : α → α → Prop)
variables (internal_bisector : α → α → α → Prop)
variables (external_bisector : α → α → α → Prop)
variables (collinear : α → α → α → α → Prop)

axiom convex_quad : convex A B C D
axiom AD_not_parallel_BC : not_parallel A D ∧ not_parallel B C

axiom internal_bisectors :
  internal_bisector A B K ∧ internal_bisector B A K ∧ internal_bisector C D P ∧ internal_bisector D C P

axiom external_bisectors :
  external_bisector A B L ∧ external_bisector B A L ∧ external_bisector C D Q ∧ external_bisector D C Q

theorem collinear_points : collinear K L P Q := 
sorry

end NUMINAMATH_GPT_collinear_points_l1107_110795


namespace NUMINAMATH_GPT_problem_statement_l1107_110785

theorem problem_statement : (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1107_110785


namespace NUMINAMATH_GPT_survey_response_total_l1107_110782

theorem survey_response_total
  (X Y Z : ℕ)
  (h_ratio : X / 4 = Y / 2 ∧ X / 4 = Z)
  (h_X : X = 200) :
  X + Y + Z = 350 :=
sorry

end NUMINAMATH_GPT_survey_response_total_l1107_110782


namespace NUMINAMATH_GPT_reduced_price_per_dozen_is_approx_2_95_l1107_110780

noncomputable def original_price : ℚ := 16 / 39
noncomputable def reduced_price := 0.6 * original_price
noncomputable def reduced_price_per_dozen := reduced_price * 12

theorem reduced_price_per_dozen_is_approx_2_95 :
  abs (reduced_price_per_dozen - 2.95) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_is_approx_2_95_l1107_110780


namespace NUMINAMATH_GPT_find_f_of_given_g_and_odd_l1107_110743

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_given_g_and_odd_l1107_110743


namespace NUMINAMATH_GPT_min_words_to_learn_l1107_110703

theorem min_words_to_learn (n : ℕ) (p_guess : ℝ) (required_score : ℝ)
  (h_n : n = 600) (h_p : p_guess = 0.1) (h_score : required_score = 0.9) :
  ∃ x : ℕ, (x + p_guess * (n - x)) / n ≥ required_score ∧ x = 534 :=
by
  sorry

end NUMINAMATH_GPT_min_words_to_learn_l1107_110703


namespace NUMINAMATH_GPT_pencils_bought_l1107_110700

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end NUMINAMATH_GPT_pencils_bought_l1107_110700


namespace NUMINAMATH_GPT_find_rho_squared_l1107_110725

theorem find_rho_squared:
  ∀ (a b : ℝ), (0 < a) → (0 < b) →
  (a^2 - 2 * b^2 = 0) →
  (∃ (x y : ℝ), 
    (0 ≤ x ∧ x < a) ∧ 
    (0 ≤ y ∧ y < b) ∧ 
    (a^2 + y^2 = b^2 + x^2) ∧ 
    ((a - x)^2 + (b - y)^2 = b^2 + x^2) ∧ 
    (x^2 + y^2 = b^2)) → 
  (∃ (ρ : ℝ), ρ = a / b ∧ ρ^2 = 2) :=
by
  intros a b ha hb hab hsol
  sorry  -- Proof to be provided later

end NUMINAMATH_GPT_find_rho_squared_l1107_110725


namespace NUMINAMATH_GPT_pencils_total_l1107_110705

/-- The students in class 5A had a total of 2015 pencils. One of them lost a box containing five pencils and replaced it with a box containing 50 pencils. Prove the final number of pencils is 2060. -/
theorem pencils_total {initial_pencils lost_pencils gained_pencils final_pencils : ℕ} 
  (h1 : initial_pencils = 2015) 
  (h2 : lost_pencils = 5) 
  (h3 : gained_pencils = 50) 
  (h4 : final_pencils = (initial_pencils - lost_pencils + gained_pencils)) 
  : final_pencils = 2060 :=
sorry

end NUMINAMATH_GPT_pencils_total_l1107_110705


namespace NUMINAMATH_GPT_maxRegions100Parabolas_l1107_110746

-- Define the number of parabolas of each type
def numberOfParabolas1 := 50
def numberOfParabolas2 := 50

-- Define the function that counts the number of regions formed by n parabolas intersecting at most m times
def maxRegions (n m : Nat) : Nat :=
  (List.range (m+1)).foldl (λ acc k => acc + Nat.choose n k) 0

-- Specify the intersection properties for each type of parabolas
def intersectionsParabolas1 := 2
def intersectionsParabolas2 := 2
def intersectionsBetweenSets := 4

-- Calculate the number of regions formed by each set of 50 parabolas
def regionsSet1 := maxRegions numberOfParabolas1 intersectionsParabolas1
def regionsSet2 := maxRegions numberOfParabolas2 intersectionsParabolas2

-- Calculate the additional regions created by intersections between the sets
def additionalIntersections := numberOfParabolas1 * numberOfParabolas2 * intersectionsBetweenSets

-- Combine the regions
def totalRegions := regionsSet1 + regionsSet2 + additionalIntersections + 1

-- Prove the final result
theorem maxRegions100Parabolas : totalRegions = 15053 :=
  sorry

end NUMINAMATH_GPT_maxRegions100Parabolas_l1107_110746


namespace NUMINAMATH_GPT_income_of_A_l1107_110755

theorem income_of_A (x y : ℝ) 
    (ratio_income : 5 * x = y * 4)
    (ratio_expenditure : 3 * x = y * 2)
    (savings_A : 5 * x - 3 * y = 1600)
    (savings_B : 4 * x - 2 * y = 1600) : 
    5 * x = 4000 := 
by
  sorry

end NUMINAMATH_GPT_income_of_A_l1107_110755


namespace NUMINAMATH_GPT_total_initial_collection_l1107_110728

variable (marco strawberries father strawberries_lost : ℕ)
variable (marco : ℕ := 12)
variable (father : ℕ := 16)
variable (strawberries_lost : ℕ := 8)
variable (total_initial_weight : ℕ := marco + father + strawberries_lost)

theorem total_initial_collection : total_initial_weight = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_initial_collection_l1107_110728


namespace NUMINAMATH_GPT_winner_won_by_288_votes_l1107_110767

theorem winner_won_by_288_votes (V : ℝ) (votes_won : ℝ) (perc_won : ℝ) 
(h1 : perc_won = 0.60)
(h2 : votes_won = 864)
(h3 : votes_won = perc_won * V) : 
votes_won - (1 - perc_won) * V = 288 := 
sorry

end NUMINAMATH_GPT_winner_won_by_288_votes_l1107_110767


namespace NUMINAMATH_GPT_score_difference_proof_l1107_110777

variable (α β γ δ : ℝ)

theorem score_difference_proof
  (h1 : α + β = γ + δ + 17)
  (h2 : α = β - 4)
  (h3 : γ = δ + 5) :
  β - δ = 13 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_score_difference_proof_l1107_110777


namespace NUMINAMATH_GPT_third_term_geometric_series_l1107_110758

variable {b1 b3 q : ℝ}
variable (hb1 : b1 * (-1/4) = -1/2)
variable (hs : b1 / (1 - q) = 8/5)
variable (hq : |q| < 1)

theorem third_term_geometric_series (hb1 : b1 * (-1 / 4) = -1 / 2)
  (hs : b1 / (1 - q) = 8 / 5)
  (hq : |q| < 1)
  : b3 = b1 * q^2 := by
    sorry

end NUMINAMATH_GPT_third_term_geometric_series_l1107_110758


namespace NUMINAMATH_GPT_number_of_men_first_group_l1107_110756

theorem number_of_men_first_group :
  (∃ M : ℕ, 30 * 3 * (M : ℚ) * (84 / 30) / 3 = 112 / 6) → ∃ M : ℕ, M = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_men_first_group_l1107_110756


namespace NUMINAMATH_GPT_bowling_team_avg_weight_l1107_110726

noncomputable def total_weight (weights : List ℕ) : ℕ :=
  weights.foldr (· + ·) 0

noncomputable def average_weight (weights : List ℕ) : ℚ :=
  total_weight weights / weights.length

theorem bowling_team_avg_weight :
  let original_weights := [76, 76, 76, 76, 76, 76, 76]
  let new_weights := [110, 60, 85, 65, 100]
  let combined_weights := original_weights ++ new_weights
  average_weight combined_weights = 79.33 := 
by 
  sorry

end NUMINAMATH_GPT_bowling_team_avg_weight_l1107_110726


namespace NUMINAMATH_GPT_part_I_part_II_l1107_110787

theorem part_I (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2) (h4 : a + b ≤ m) : m ≥ 3 := by
  sorry

theorem part_II (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2)
  (h4 : 2 * |x - 1| + |x| ≥ a + b) : (x ≤ -1 / 3 ∨ x ≥ 5 / 3) := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1107_110787
