import Mathlib

namespace NUMINAMATH_GPT_increase_in_average_commission_l1619_161992

theorem increase_in_average_commission :
  ∀ (new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 : ℕ),
    new_avg = 400 → 
    n1 = 6 → 
    n2 = n1 - 1 → 
    big_sale = 1300 →
    total_earnings = new_avg * n1 →
    commission = total_earnings - big_sale →
    old_avg = commission / n2 →
    new_avg - old_avg = 180 :=
by 
  intros new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_increase_in_average_commission_l1619_161992


namespace NUMINAMATH_GPT_problem_conditions_l1619_161932

theorem problem_conditions (x y : ℝ) (hx : x * (Real.exp x + Real.log x + x) = 1) (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  (0 < x ∧ x < 1) ∧ (y - x > 1) ∧ (y - x < 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l1619_161932


namespace NUMINAMATH_GPT_selling_price_of_radio_l1619_161983

theorem selling_price_of_radio
  (cost_price : ℝ)
  (loss_percentage : ℝ) :
  loss_percentage = 13 → cost_price = 1500 → 
  (cost_price - (loss_percentage / 100) * cost_price) = 1305 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_selling_price_of_radio_l1619_161983


namespace NUMINAMATH_GPT_find_k_l1619_161922

theorem find_k 
  (k : ℤ) 
  (h : 2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997) : 
  k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l1619_161922


namespace NUMINAMATH_GPT_midpoint_chord_hyperbola_l1619_161978

theorem midpoint_chord_hyperbola (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (∃ (mx my : ℝ), (mx / a^2 + my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2))) →
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) →
  ∃ (mx my : ℝ), (mx / a^2 - my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2) := 
sorry

end NUMINAMATH_GPT_midpoint_chord_hyperbola_l1619_161978


namespace NUMINAMATH_GPT_complex_number_solution_l1619_161938

theorem complex_number_solution (z : ℂ) (i : ℂ) (H1 : i * i = -1) (H2 : z * i = 2 - 2 * i) : z = -2 - 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1619_161938


namespace NUMINAMATH_GPT_average_rst_l1619_161979

theorem average_rst (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 :=
sorry

end NUMINAMATH_GPT_average_rst_l1619_161979


namespace NUMINAMATH_GPT_massive_crate_chocolate_bars_l1619_161988

theorem massive_crate_chocolate_bars :
  (54 * 24 * 37 = 47952) :=
by
  sorry

end NUMINAMATH_GPT_massive_crate_chocolate_bars_l1619_161988


namespace NUMINAMATH_GPT_digital_root_8_pow_n_l1619_161956

-- Define the conditions
def n : ℕ := 1989

-- Define the simplified problem
def digital_root (x : ℕ) : ℕ := if x % 9 = 0 then 9 else x % 9

-- Statement of the problem
theorem digital_root_8_pow_n : digital_root (8 ^ n) = 8 := by
  have mod_nine_eq : 8^n % 9 = 8 := by
    sorry
  simp [digital_root, mod_nine_eq]

end NUMINAMATH_GPT_digital_root_8_pow_n_l1619_161956


namespace NUMINAMATH_GPT_composite_p_squared_plus_36_l1619_161943

theorem composite_p_squared_plus_36 (p : ℕ) (h_prime : Prime p) : 
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ (k * m = p^2 + 36) :=
by {
  sorry
}

end NUMINAMATH_GPT_composite_p_squared_plus_36_l1619_161943


namespace NUMINAMATH_GPT_no_nat_number_with_perfect_square_l1619_161902

theorem no_nat_number_with_perfect_square (n : Nat) : 
  ¬ ∃ m : Nat, m * m = n^6 + 3 * n^5 - 5 * n^4 - 15 * n^3 + 4 * n^2 + 12 * n + 3 := 
  by
  sorry

end NUMINAMATH_GPT_no_nat_number_with_perfect_square_l1619_161902


namespace NUMINAMATH_GPT_upper_bound_of_n_l1619_161905

theorem upper_bound_of_n (m n : ℕ) (h_m : m ≥ 2)
  (h_div : ∀ a : ℕ, gcd a n = 1 → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := 
sorry

end NUMINAMATH_GPT_upper_bound_of_n_l1619_161905


namespace NUMINAMATH_GPT_race_time_l1619_161987

theorem race_time (v_A v_B : ℝ) (t tB : ℝ) (h1 : 200 / v_A = t) (h2 : 144 / v_B = t) (h3 : 200 / v_B = t + 7) : t = 18 :=
by
  sorry

end NUMINAMATH_GPT_race_time_l1619_161987


namespace NUMINAMATH_GPT_books_in_bin_after_actions_l1619_161931

theorem books_in_bin_after_actions (x y : ℕ) (z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 250) : x - y + (z / 100) * x = 11 :=
by
  rw [hx, hy, hz]
  -- x - y + (z / 100) * x = 4 - 3 + (250 / 100) * 4
  norm_num
  sorry

end NUMINAMATH_GPT_books_in_bin_after_actions_l1619_161931


namespace NUMINAMATH_GPT_certain_number_div_5000_l1619_161962

theorem certain_number_div_5000 (num : ℝ) (h : num / 5000 = 0.0114) : num = 57 :=
sorry

end NUMINAMATH_GPT_certain_number_div_5000_l1619_161962


namespace NUMINAMATH_GPT_cirrus_clouds_count_l1619_161935

theorem cirrus_clouds_count 
  (cirrus_clouds cumulus_clouds cumulonimbus_clouds : ℕ)
  (h1 : cirrus_clouds = 4 * cumulus_clouds)
  (h2 : cumulus_clouds = 12 * cumulonimbus_clouds)
  (h3 : cumulonimbus_clouds = 3) : 
  cirrus_clouds = 144 :=
by sorry

end NUMINAMATH_GPT_cirrus_clouds_count_l1619_161935


namespace NUMINAMATH_GPT_decimal_to_binary_correct_l1619_161928

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end NUMINAMATH_GPT_decimal_to_binary_correct_l1619_161928


namespace NUMINAMATH_GPT_remainder_of_n_plus_2024_l1619_161930

-- Define the assumptions
def n : ℤ := sorry  -- n will be some integer
def k : ℤ := sorry  -- k will be some integer

-- Main statement to be proved
theorem remainder_of_n_plus_2024 (h : n % 8 = 3) : (n + 2024) % 8 = 3 := sorry

end NUMINAMATH_GPT_remainder_of_n_plus_2024_l1619_161930


namespace NUMINAMATH_GPT_abc_divisible_by_6_l1619_161963

theorem abc_divisible_by_6 (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_abc_divisible_by_6_l1619_161963


namespace NUMINAMATH_GPT_yogurt_packs_ordered_l1619_161918

theorem yogurt_packs_ordered (P : ℕ) (price_per_pack refund_amount : ℕ) (expired_percentage : ℚ)
  (h1 : price_per_pack = 12)
  (h2 : refund_amount = 384)
  (h3 : expired_percentage = 0.40)
  (h4 : refund_amount / price_per_pack = 32)
  (h5 : 32 / expired_percentage = P) :
  P = 80 :=
sorry

end NUMINAMATH_GPT_yogurt_packs_ordered_l1619_161918


namespace NUMINAMATH_GPT_total_rats_l1619_161995

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end NUMINAMATH_GPT_total_rats_l1619_161995


namespace NUMINAMATH_GPT_at_least_two_consecutive_heads_probability_l1619_161991

theorem at_least_two_consecutive_heads_probability :
  let outcomes := ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
  let favorable_outcomes := ["HHH", "HHT", "THH"]
  let total_outcomes := outcomes.length
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_at_least_two_consecutive_heads_probability_l1619_161991


namespace NUMINAMATH_GPT_original_proposition_converse_negation_contrapositive_l1619_161946

variable {a b : ℝ}

-- Original Proposition: If \( x^2 + ax + b \leq 0 \) has a non-empty solution set, then \( a^2 - 4b \geq 0 \)
theorem original_proposition (h : ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b ≥ 0 := sorry

-- Converse: If \( a^2 - 4b \geq 0 \), then \( x^2 + ax + b \leq 0 \) has a non-empty solution set
theorem converse (h : a^2 - 4 * b ≥ 0) : ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

-- Negation: If \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set, then \( a^2 - 4b < 0 \)
theorem negation (h : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b < 0 := sorry

-- Contrapositive: If \( a^2 - 4b < 0 \), then \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set
theorem contrapositive (h : a^2 - 4 * b < 0) : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

end NUMINAMATH_GPT_original_proposition_converse_negation_contrapositive_l1619_161946


namespace NUMINAMATH_GPT_radio_loss_percentage_l1619_161993

theorem radio_loss_percentage (CP SP : ℝ) (h_CP : CP = 2400) (h_SP : SP = 2100) :
  ((CP - SP) / CP) * 100 = 12.5 :=
by
  -- Given cost price
  have h_CP : CP = 2400 := h_CP
  -- Given selling price
  have h_SP : SP = 2100 := h_SP
  sorry

end NUMINAMATH_GPT_radio_loss_percentage_l1619_161993


namespace NUMINAMATH_GPT_prob_4_consecutive_baskets_prob_exactly_4_baskets_l1619_161960

theorem prob_4_consecutive_baskets 
  (p : ℝ) (h : p = 1/2) : 
  (p^4 * (1 - p) + (1 - p) * p^4) = 1/16 :=
by sorry

theorem prob_exactly_4_baskets 
  (p : ℝ) (h : p = 1/2) : 
  5 * p^4 * (1 - p) = 5/32 :=
by sorry

end NUMINAMATH_GPT_prob_4_consecutive_baskets_prob_exactly_4_baskets_l1619_161960


namespace NUMINAMATH_GPT_area_triangle_PZQ_l1619_161920

/-- 
In rectangle PQRS, side PQ measures 8 units and side QR measures 4 units.
Points X and Y are on side RS such that segment RX measures 2 units and
segment SY measures 3 units. Lines PX and QY intersect at point Z.
Prove the area of triangle PZQ is 128/3 square units.
-/

theorem area_triangle_PZQ {PQ QR RX SY : ℝ} (h1 : PQ = 8) (h2 : QR = 4) (h3 : RX = 2) (h4 : SY = 3) :
  let area_PZQ : ℝ := 8 * 4 / 2 * 8 / (3 * 2)
  area_PZQ = 128 / 3 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_PZQ_l1619_161920


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1619_161959

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_9 : ℝ)
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 3 + a 6 + a 9 = 3)
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) :
  S_9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1619_161959


namespace NUMINAMATH_GPT_power_function_monotonic_decreasing_l1619_161964

theorem power_function_monotonic_decreasing (α : ℝ) (h : ∀ x y : ℝ, 0 < x → x < y → x^α > y^α) : α < 0 :=
sorry

end NUMINAMATH_GPT_power_function_monotonic_decreasing_l1619_161964


namespace NUMINAMATH_GPT_tanner_savings_in_november_l1619_161910

theorem tanner_savings_in_november(savings_sep : ℕ) (savings_oct : ℕ) 
(spending : ℕ) (leftover : ℕ) (N : ℕ) :
savings_sep = 17 →
savings_oct = 48 →
spending = 49 →
leftover = 41 →
((savings_sep + savings_oct + N - spending) = leftover) →
N = 25 :=
by
  intros h_sep h_oct h_spending h_leftover h_equation
  sorry

end NUMINAMATH_GPT_tanner_savings_in_november_l1619_161910


namespace NUMINAMATH_GPT_power_function_odd_l1619_161924

-- Define the conditions
def f : ℝ → ℝ := sorry
def condition1 (f : ℝ → ℝ) : Prop := f 1 = 3

-- Define the statement of the problem as a Lean theorem
theorem power_function_odd (f : ℝ → ℝ) (h : condition1 f) : ∀ x, f (-x) = -f x := sorry

end NUMINAMATH_GPT_power_function_odd_l1619_161924


namespace NUMINAMATH_GPT_initially_calculated_average_l1619_161941

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end NUMINAMATH_GPT_initially_calculated_average_l1619_161941


namespace NUMINAMATH_GPT_floor_neg_seven_four_is_neg_two_l1619_161911

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_four_is_neg_two_l1619_161911


namespace NUMINAMATH_GPT_fraction_inequality_l1619_161961

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end NUMINAMATH_GPT_fraction_inequality_l1619_161961


namespace NUMINAMATH_GPT_work_days_l1619_161975

theorem work_days (Dx Dy : ℝ) (H1 : Dy = 45) (H2 : 8 / Dx + 36 / Dy = 1) : Dx = 40 :=
by
  sorry

end NUMINAMATH_GPT_work_days_l1619_161975


namespace NUMINAMATH_GPT_complaints_over_3_days_l1619_161934

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end NUMINAMATH_GPT_complaints_over_3_days_l1619_161934


namespace NUMINAMATH_GPT_correct_computation_gives_l1619_161927

variable (x : ℝ)

theorem correct_computation_gives :
  ((3 * x - 12) / 6 = 60) → ((x / 3) + 12 = 160 / 3) :=
by
  sorry

end NUMINAMATH_GPT_correct_computation_gives_l1619_161927


namespace NUMINAMATH_GPT_max_value_ineq_l1619_161969

theorem max_value_ineq (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 1) :
  (a + 3 * b + 5 * c) * (a + b / 3 + c / 5) ≤ 9 / 5 :=
sorry

end NUMINAMATH_GPT_max_value_ineq_l1619_161969


namespace NUMINAMATH_GPT_product_remainder_31_l1619_161901

theorem product_remainder_31 (m n : ℕ) (h₁ : m % 31 = 7) (h₂ : n % 31 = 12) : (m * n) % 31 = 22 :=
by
  sorry

end NUMINAMATH_GPT_product_remainder_31_l1619_161901


namespace NUMINAMATH_GPT_find_b_l1619_161968

variables (a b : ℕ)

theorem find_b
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 25 * 315 * b) :
  b = 7 :=
sorry

end NUMINAMATH_GPT_find_b_l1619_161968


namespace NUMINAMATH_GPT_average_age_of_girls_l1619_161923

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℚ)
    (girls_count : ℕ) (total_age_school : ℝ) (boys_count : ℕ) 
    (total_age_boys : ℝ) (total_age_girls : ℝ): (total_students = 640) →
    (boys_avg_age = 12) →
    (school_avg_age = 47 / 4) →
    (girls_count = 160) →
    (total_students - girls_count = boys_count) →
    (boys_avg_age * boys_count = total_age_boys) →
    (school_avg_age * total_students = total_age_school) →
    (total_age_school - total_age_boys = total_age_girls) →
    total_age_girls / girls_count = 11 :=
by
  intros h_total_students h_boys_avg_age h_school_avg_age h_girls_count 
         h_boys_count h_total_age_boys h_total_age_school h_total_age_girls
  sorry

end NUMINAMATH_GPT_average_age_of_girls_l1619_161923


namespace NUMINAMATH_GPT_solve_for_x_and_n_l1619_161933

theorem solve_for_x_and_n (x n : ℕ) : 2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := 
sorry

end NUMINAMATH_GPT_solve_for_x_and_n_l1619_161933


namespace NUMINAMATH_GPT_average_length_of_strings_l1619_161914

-- Define lengths of the three strings
def length1 := 4  -- length of the first string in inches
def length2 := 5  -- length of the second string in inches
def length3 := 7  -- length of the third string in inches

-- Define the total length and number of strings
def total_length := length1 + length2 + length3
def num_strings := 3

-- Define the average length calculation
def average_length := total_length / num_strings

-- The proof statement
theorem average_length_of_strings : average_length = 16 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_average_length_of_strings_l1619_161914


namespace NUMINAMATH_GPT_difference_of_numbers_l1619_161999

theorem difference_of_numbers : 
  ∃ (L S : ℕ), L = 1631 ∧ L = 6 * S + 35 ∧ L - S = 1365 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l1619_161999


namespace NUMINAMATH_GPT_radius_comparison_l1619_161976

theorem radius_comparison 
  (a b c : ℝ)
  (da db dc r ρ : ℝ)
  (h₁ : da ≤ r)
  (h₂ : db ≤ r)
  (h₃ : dc ≤ r)
  (h₄ : 1 / 2 * (a * da + b * db + c * dc) = ρ * ((a + b + c) / 2)) :
  r ≥ ρ := 
sorry

end NUMINAMATH_GPT_radius_comparison_l1619_161976


namespace NUMINAMATH_GPT_prob_high_quality_correct_l1619_161984

noncomputable def prob_high_quality_seeds :=
  let p_first := 0.955
  let p_second := 0.02
  let p_third := 0.015
  let p_fourth := 0.01
  let p_hq_first := 0.5
  let p_hq_second := 0.15
  let p_hq_third := 0.1
  let p_hq_fourth := 0.05
  let p_hq := p_first * p_hq_first + p_second * p_hq_second + p_third * p_hq_third + p_fourth * p_hq_fourth
  p_hq

theorem prob_high_quality_correct : prob_high_quality_seeds = 0.4825 :=
  by sorry

end NUMINAMATH_GPT_prob_high_quality_correct_l1619_161984


namespace NUMINAMATH_GPT_total_questions_in_two_hours_l1619_161949

theorem total_questions_in_two_hours (r : ℝ) : 
  let Fiona_questions := 36 
  let Shirley_questions := Fiona_questions * r
  let Kiana_questions := (Fiona_questions + Shirley_questions) / 2
  let one_hour_total := Fiona_questions + Shirley_questions + Kiana_questions
  let two_hour_total := 2 * one_hour_total
  two_hour_total = 108 + 108 * r :=
by
  sorry

end NUMINAMATH_GPT_total_questions_in_two_hours_l1619_161949


namespace NUMINAMATH_GPT_infinite_solutions_distinct_natural_numbers_l1619_161965

theorem infinite_solutions_distinct_natural_numbers :
  ∃ (x y z : ℕ), (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (x ^ 2015 + y ^ 2015 = z ^ 2016) :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_distinct_natural_numbers_l1619_161965


namespace NUMINAMATH_GPT_correct_proposition_l1619_161907

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_l1619_161907


namespace NUMINAMATH_GPT_min_value_of_2x_plus_y_l1619_161966

theorem min_value_of_2x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 8 / y = 2) : 2 * x + y ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_value_of_2x_plus_y_l1619_161966


namespace NUMINAMATH_GPT_tower_height_l1619_161947

theorem tower_height (h : ℝ) (hd : ¬ (h ≥ 200)) (he : ¬ (h ≤ 150)) (hf : ¬ (h ≤ 180)) : 180 < h ∧ h < 200 := 
by 
  sorry

end NUMINAMATH_GPT_tower_height_l1619_161947


namespace NUMINAMATH_GPT_diagonal_of_square_l1619_161977

-- Definitions based on conditions
def square_area := 8 -- Area of the square is 8 square centimeters

def diagonal_length (x : ℝ) : Prop :=
  (1/2) * x ^ 2 = square_area

-- Proof problem statement
theorem diagonal_of_square : ∃ x : ℝ, diagonal_length x ∧ x = 4 := 
sorry  -- statement only, proof skipped

end NUMINAMATH_GPT_diagonal_of_square_l1619_161977


namespace NUMINAMATH_GPT_translation_result_l1619_161972

-- Define the original point A
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 3, y := -2 }

-- Define the translation function
def translate_right (p : Point) (dx : ℤ) : Point :=
  { x := p.x + dx, y := p.y }

-- Prove that translating point A 2 units to the right gives point A'
theorem translation_result :
  translate_right A 2 = { x := 5, y := -2 } :=
by sorry

end NUMINAMATH_GPT_translation_result_l1619_161972


namespace NUMINAMATH_GPT_problem_l1619_161954

theorem problem : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end NUMINAMATH_GPT_problem_l1619_161954


namespace NUMINAMATH_GPT_rectangle_width_length_ratio_l1619_161926

theorem rectangle_width_length_ratio (w : ℕ) (h : ℕ) (P : ℕ) (H1 : h = 10) (H2 : P = 30) (H3 : 2 * w + 2 * h = P) :
  w / h = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_length_ratio_l1619_161926


namespace NUMINAMATH_GPT_trip_duration_17_hours_l1619_161953

theorem trip_duration_17_hours :
  ∃ T : ℝ, 
    (∀ d₁ d₂ : ℝ,
      (d₁ / 30 + 1 + (150 - d₁) / 4 = T) ∧ 
      (d₁ / 30 + d₂ / 30 + (150 - (d₁ - d₂)) / 30 = T) ∧ 
      ((d₁ - d₂) / 4 + (150 - (d₁ - d₂)) / 30 = T))
  → T = 17 :=
by
  sorry

end NUMINAMATH_GPT_trip_duration_17_hours_l1619_161953


namespace NUMINAMATH_GPT_ball_hits_ground_at_l1619_161939

variable (t : ℚ) 

def height_eqn (t : ℚ) : ℚ :=
  -16 * t^2 + 30 * t + 50

theorem ball_hits_ground_at :
  (height_eqn t = 0) -> t = 47 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_l1619_161939


namespace NUMINAMATH_GPT_geom_seq_product_arith_seq_l1619_161994

theorem geom_seq_product_arith_seq (a b c r : ℝ) (h1 : c = b * r)
  (h2 : b = a * r)
  (h3 : a * b * c = 512)
  (h4 : b = 8)
  (h5 : 2 * b = (a - 2) + (c - 2)) :
  (a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4) :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_product_arith_seq_l1619_161994


namespace NUMINAMATH_GPT_find_m_of_parabola_and_line_l1619_161982

theorem find_m_of_parabola_and_line (k m x1 x2 : ℝ) 
  (h_parabola_line : ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} → 
                                   y = k * x + m → true)
  (h_intersection : x1 * x2 = -4) : m = 1 := 
sorry

end NUMINAMATH_GPT_find_m_of_parabola_and_line_l1619_161982


namespace NUMINAMATH_GPT_fifth_team_points_l1619_161900

theorem fifth_team_points (points_A points_B points_C points_D points_E : ℕ) 
(hA : points_A = 1) 
(hB : points_B = 2) 
(hC : points_C = 5) 
(hD : points_D = 7) 
(h_sum : points_A + points_B + points_C + points_D + points_E = 20) : 
points_E = 5 := 
sorry

end NUMINAMATH_GPT_fifth_team_points_l1619_161900


namespace NUMINAMATH_GPT_y_intercept_exists_l1619_161904

def line_eq (x y : ℝ) : Prop := x + 2 * y + 2 = 0

theorem y_intercept_exists : ∃ y : ℝ, line_eq 0 y ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_exists_l1619_161904


namespace NUMINAMATH_GPT_math_students_count_l1619_161986

noncomputable def students_in_math (total_students history_students english_students all_three_classes two_classes : ℕ) : ℕ :=
total_students - history_students - english_students + (two_classes - all_three_classes)

theorem math_students_count :
  students_in_math 68 21 34 3 7 = 14 :=
by
  sorry

end NUMINAMATH_GPT_math_students_count_l1619_161986


namespace NUMINAMATH_GPT_max_area_of_triangle_l1619_161937

noncomputable def maxAreaTriangle (m_a m_b m_c : ℝ) : ℝ :=
  1/3 * Real.sqrt (2 * (m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4))

theorem max_area_of_triangle (m_a m_b m_c : ℝ) (h1 : m_a ≤ 2) (h2 : m_b ≤ 3) (h3 : m_c ≤ 4) :
  maxAreaTriangle m_a m_b m_c ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1619_161937


namespace NUMINAMATH_GPT_complex_fifth_roots_wrong_statement_l1619_161948

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 5)

theorem complex_fifth_roots_wrong_statement :
  ¬(x^5 + y^5 = 1) :=
sorry

end NUMINAMATH_GPT_complex_fifth_roots_wrong_statement_l1619_161948


namespace NUMINAMATH_GPT_bowling_average_decrease_l1619_161906

theorem bowling_average_decrease 
  (original_average : ℚ) 
  (wickets_last_match : ℚ) 
  (runs_last_match : ℚ) 
  (original_wickets : ℚ) 
  (original_total_runs : ℚ := original_wickets * original_average) 
  (new_total_wickets : ℚ := original_wickets + wickets_last_match) 
  (new_total_runs : ℚ := original_total_runs + runs_last_match)
  (new_average : ℚ := new_total_runs / new_total_wickets) :
  original_wickets = 85 → original_average = 12.4 → wickets_last_match = 5 → runs_last_match = 26 → new_average = 12 →
  original_average - new_average = 0.4 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_bowling_average_decrease_l1619_161906


namespace NUMINAMATH_GPT_gabby_mom_gave_20_l1619_161913

theorem gabby_mom_gave_20 (makeup_set_cost saved_money more_needed total_needed mom_money : ℕ)
  (h1 : makeup_set_cost = 65)
  (h2 : saved_money = 35)
  (h3 : more_needed = 10)
  (h4 : total_needed = makeup_set_cost - saved_money)
  (h5 : total_needed - mom_money = more_needed) :
  mom_money = 20 :=
by
  sorry

end NUMINAMATH_GPT_gabby_mom_gave_20_l1619_161913


namespace NUMINAMATH_GPT_number_division_l1619_161997

theorem number_division (x : ℚ) (h : x / 2 = 100 + x / 5) : x = 1000 / 3 := 
by
  sorry

end NUMINAMATH_GPT_number_division_l1619_161997


namespace NUMINAMATH_GPT_min_value_expression_l1619_161916

theorem min_value_expression (α β : ℝ) :
  ∃ x y, x = 3 * Real.cos α + 6 * Real.sin β ∧
         y = 3 * Real.sin α + 6 * Real.cos β ∧
         (x - 10)^2 + (y - 18)^2 = 121 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1619_161916


namespace NUMINAMATH_GPT_r_earns_per_day_l1619_161990

variables (P Q R S : ℝ)

theorem r_earns_per_day
  (h1 : P + Q + R + S = 240)
  (h2 : P + R + S = 160)
  (h3 : Q + R = 150)
  (h4 : Q + R + S = 650 / 3) :
  R = 70 :=
by
  sorry

end NUMINAMATH_GPT_r_earns_per_day_l1619_161990


namespace NUMINAMATH_GPT_percentage_exceeds_l1619_161903

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end NUMINAMATH_GPT_percentage_exceeds_l1619_161903


namespace NUMINAMATH_GPT_sequence_periodic_l1619_161970

theorem sequence_periodic (a : ℕ → ℕ) (h : ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10) :
  ∃ n₀, ∀ k, a (n₀ + k) = a (n₀ + k + 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_periodic_l1619_161970


namespace NUMINAMATH_GPT_S_11_eq_22_l1619_161989

variable {S : ℕ → ℕ}

-- Condition: given that S_8 - S_3 = 10
axiom h : S 8 - S 3 = 10

-- Proof goal: we want to show that S_11 = 22
theorem S_11_eq_22 : S 11 = 22 :=
by
  sorry

end NUMINAMATH_GPT_S_11_eq_22_l1619_161989


namespace NUMINAMATH_GPT_quadratic_root_conditions_l1619_161971

theorem quadratic_root_conditions : ∃ p q : ℝ, (p - 1)^2 - 4 * q > 0 ∧ (p + 1)^2 - 4 * q > 0 ∧ p^2 - 4 * q < 0 := 
sorry

end NUMINAMATH_GPT_quadratic_root_conditions_l1619_161971


namespace NUMINAMATH_GPT_smallest_c1_in_arithmetic_sequence_l1619_161909

theorem smallest_c1_in_arithmetic_sequence (S3 S7 : ℕ) (S3_natural : S3 > 0) (S7_natural : S7 > 0)
    (c1_geq_one_third : ∀ d : ℚ, (c1 : ℚ) = (7*S3 - S7) / 14 → c1 ≥ 1/3) : 
    ∃ c1 : ℚ, c1 = 5/14 ∧ c1 ≥ 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_c1_in_arithmetic_sequence_l1619_161909


namespace NUMINAMATH_GPT_sum_of_cube_faces_l1619_161980

theorem sum_of_cube_faces (a d b e c f : ℕ) (h1: a > 0) (h2: d > 0) (h3: b > 0) (h4: e > 0) (h5: c > 0) (h6: f > 0)
(h7 : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1491) :
  a + d + b + e + c + f = 41 := 
sorry

end NUMINAMATH_GPT_sum_of_cube_faces_l1619_161980


namespace NUMINAMATH_GPT_problem_statement_l1619_161950

-- Given conditions
noncomputable def S : ℕ → ℝ := sorry
axiom S_3_eq_2 : S 3 = 2
axiom S_6_eq_6 : S 6 = 6

-- Prove that a_{13} + a_{14} + a_{15} = 32
theorem problem_statement : (S 15 - S 12) = 32 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1619_161950


namespace NUMINAMATH_GPT_sum_of_digits_of_63_l1619_161981

theorem sum_of_digits_of_63 (x y : ℕ) (h : 10 * x + y = 63) (h1 : x + y = 9) (h2 : x - y = 3) : x + y = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_63_l1619_161981


namespace NUMINAMATH_GPT_max_value_of_f_l1619_161929

def f (x : ℝ) : ℝ := 10 * x - 2 * x ^ 2

theorem max_value_of_f : ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) :=
  ⟨12.5, sorry⟩

end NUMINAMATH_GPT_max_value_of_f_l1619_161929


namespace NUMINAMATH_GPT_weight_of_one_bowling_ball_l1619_161973

-- Definitions from the problem conditions
def weight_canoe := 36
def num_canoes := 4
def num_bowling_balls := 9

-- Calculate the total weight of the canoes
def total_weight_canoes := num_canoes * weight_canoe

-- Prove the weight of one bowling ball
theorem weight_of_one_bowling_ball : (total_weight_canoes / num_bowling_balls) = 16 := by
  sorry

end NUMINAMATH_GPT_weight_of_one_bowling_ball_l1619_161973


namespace NUMINAMATH_GPT_simplify_expression_l1619_161952

variable {x : ℝ}

theorem simplify_expression : 8 * x - 3 + 2 * x - 7 + 4 * x + 15 = 14 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1619_161952


namespace NUMINAMATH_GPT_ab_times_65_eq_48ab_l1619_161996

theorem ab_times_65_eq_48ab (a b : ℕ) (h_ab : 0 ≤ a ∧ a < 10) (h_b : 0 ≤ b ∧ b < 10) :
  (10 * a + b) * 65 = 4800 + 10 * a + b ↔ 10 * a + b = 75 := by
sorry

end NUMINAMATH_GPT_ab_times_65_eq_48ab_l1619_161996


namespace NUMINAMATH_GPT_six_digit_number_condition_l1619_161942

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_condition_l1619_161942


namespace NUMINAMATH_GPT_Marcy_sips_interval_l1619_161940

theorem Marcy_sips_interval:
  ∀ (total_volume_ml sip_volume_ml total_time min_per_sip: ℕ),
  total_volume_ml = 2000 →
  sip_volume_ml = 40 →
  total_time = 250 →
  min_per_sip = total_time / (total_volume_ml / sip_volume_ml) →
  min_per_sip = 5 :=
by
  intros total_volume_ml sip_volume_ml total_time min_per_sip hv hs ht hm
  rw [hv, hs, ht] at hm
  simp at hm
  exact hm

end NUMINAMATH_GPT_Marcy_sips_interval_l1619_161940


namespace NUMINAMATH_GPT_smallest_boxes_l1619_161919

-- Definitions based on the conditions:
def divisible_by (n d : Nat) : Prop := ∃ k, n = d * k

-- The statement to be proved:
theorem smallest_boxes (n : Nat) : 
  divisible_by n 5 ∧ divisible_by n 24 -> n = 120 :=
by sorry

end NUMINAMATH_GPT_smallest_boxes_l1619_161919


namespace NUMINAMATH_GPT_smallest_special_number_l1619_161974

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end NUMINAMATH_GPT_smallest_special_number_l1619_161974


namespace NUMINAMATH_GPT_carl_lawn_area_l1619_161921

theorem carl_lawn_area :
  ∃ (width height : ℤ), 
    (width + 1) + (height + 1) - 4 = 24 ∧
    3 * width = height ∧
    3 * ((width + 1) * 3) * ((height + 1) * 3) = 243 :=
by
  sorry

end NUMINAMATH_GPT_carl_lawn_area_l1619_161921


namespace NUMINAMATH_GPT_garden_roller_area_l1619_161908

theorem garden_roller_area (length : ℝ) (area_5rev : ℝ) (d1 d2 : ℝ) (π : ℝ) :
  length = 4 ∧ area_5rev = 88 ∧ π = 22 / 7 ∧ d2 = 1.4 →
  let circumference := π * d2
  let area_rev := circumference * length
  let new_area_5rev := 5 * area_rev
  new_area_5rev = 88 :=
by
  sorry

end NUMINAMATH_GPT_garden_roller_area_l1619_161908


namespace NUMINAMATH_GPT_barney_extra_weight_l1619_161915

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the combined weight of five regular dinosaurs
def five_regular_dinosaurs_weight : ℕ := 5 * regular_dinosaur_weight

-- Define the total weight of Barney and the five regular dinosaurs together
def total_combined_weight : ℕ := 9500

-- Define the weight of Barney
def barney_weight : ℕ := total_combined_weight - five_regular_dinosaurs_weight

-- The proof statement
theorem barney_extra_weight : barney_weight - five_regular_dinosaurs_weight = 1500 :=
by sorry

end NUMINAMATH_GPT_barney_extra_weight_l1619_161915


namespace NUMINAMATH_GPT_value_of_a_l1619_161951

theorem value_of_a (m n a : ℚ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + a) + 5) : 
  a = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1619_161951


namespace NUMINAMATH_GPT_polygon_number_of_sides_and_interior_sum_l1619_161998

-- Given conditions
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)
def exterior_angle_sum : ℝ := 360

-- Proof problem statement
theorem polygon_number_of_sides_and_interior_sum (n : ℕ)
  (h : interior_angle_sum n = 3 * exterior_angle_sum) :
  n = 8 ∧ interior_angle_sum n = 1080 :=
by
  sorry

end NUMINAMATH_GPT_polygon_number_of_sides_and_interior_sum_l1619_161998


namespace NUMINAMATH_GPT_Fedya_age_statement_l1619_161957

theorem Fedya_age_statement (d a : ℕ) (today : ℕ) (birthday : ℕ) 
    (H1 : d + 2 = a) 
    (H2 : a + 2 = birthday + 3) 
    (H3 : birthday = today + 1) :
    ∃ sameYear y, (birthday < today + 2 ∨ today < birthday) ∧ ((sameYear ∧ y - today = 1) ∨ (¬ sameYear ∧ y - today = 0)) :=
by
  sorry

end NUMINAMATH_GPT_Fedya_age_statement_l1619_161957


namespace NUMINAMATH_GPT_Chris_has_6_Teslas_l1619_161925

theorem Chris_has_6_Teslas (x y z : ℕ) (h1 : z = 13) (h2 : z = x + 10) (h3 : x = y / 2):
  y = 6 :=
by
  sorry

end NUMINAMATH_GPT_Chris_has_6_Teslas_l1619_161925


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1619_161917

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (n : ℕ) (an : ℕ → α) : α :=
  (n : α) * an 1 + (n * (n - 1) / 2 * (an 2 - an 1))

theorem common_difference_of_arithmetic_sequence (S : ℕ → ℕ) (d : ℕ) (a1 a2 : ℕ)
  (h1 : ∀ n, S n = 4 * n ^ 2 - n)
  (h2 : a1 = S 1)
  (h3 : a2 = S 2 - S 1) :
  d = a2 - a1 → d = 8 := by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1619_161917


namespace NUMINAMATH_GPT_largest_rhombus_diagonal_in_circle_l1619_161944

theorem largest_rhombus_diagonal_in_circle (r : ℝ) (h : r = 10) : (2 * r = 20) :=
by
  sorry

end NUMINAMATH_GPT_largest_rhombus_diagonal_in_circle_l1619_161944


namespace NUMINAMATH_GPT_correct_option_C_l1619_161945

theorem correct_option_C (m n : ℤ) : 
  (4 * m + 1) * 2 * m = 8 * m^2 + 2 * m :=
by
  sorry

end NUMINAMATH_GPT_correct_option_C_l1619_161945


namespace NUMINAMATH_GPT_tangent_slope_at_zero_l1619_161936

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_slope_at_zero :
  (deriv f 0) = 1 := by 
  sorry

end NUMINAMATH_GPT_tangent_slope_at_zero_l1619_161936


namespace NUMINAMATH_GPT_find_number_l1619_161967

theorem find_number (x : ℤ) : 17 * (x + 99) = 3111 → x = 84 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1619_161967


namespace NUMINAMATH_GPT_find_special_four_digit_square_l1619_161955

theorem find_special_four_digit_square :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    n = 8281 ∧
    a = c ∧
    b + 1 = d ∧
    n = (91 : ℕ) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_special_four_digit_square_l1619_161955


namespace NUMINAMATH_GPT_incircle_tangent_distance_l1619_161912

theorem incircle_tangent_distance (a b c : ℝ) (M : ℝ) (BM : ℝ) (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : BM = y1 + z1)
  (h2 : BM = y2 + z2)
  (h3 : x1 + y1 = x2 + y2)
  (h4 : x1 + z1 = c)
  (h5 : x2 + z2 = a) :
  |y1 - y2| = |(a - c) / 2| := by 
  sorry

end NUMINAMATH_GPT_incircle_tangent_distance_l1619_161912


namespace NUMINAMATH_GPT_waiter_customers_before_lunch_l1619_161958

theorem waiter_customers_before_lunch (X : ℕ) (A : X + 20 = 49) : X = 29 := by
  -- The proof is omitted based on the instructions
  sorry

end NUMINAMATH_GPT_waiter_customers_before_lunch_l1619_161958


namespace NUMINAMATH_GPT_not_factorial_tail_numbers_lt_1992_l1619_161985

noncomputable def factorial_tail_number_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + factorial_tail_number_count (n / 5)

theorem not_factorial_tail_numbers_lt_1992 :
  ∃ n, n < 1992 ∧ n = 1992 - (1992 / 5 + (1992 / 25 + (1992 / 125 + (1992 / 625 + 0)))) :=
sorry

end NUMINAMATH_GPT_not_factorial_tail_numbers_lt_1992_l1619_161985
