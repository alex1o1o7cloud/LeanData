import Mathlib

namespace NUMINAMATH_GPT_count_positive_integers_x_satisfying_inequality_l315_31544

theorem count_positive_integers_x_satisfying_inequality :
  ∃ n : ℕ, n = 6 ∧ (∀ x : ℕ, (144 ≤ x^2 ∧ x^2 ≤ 289) → (x = 12 ∨ x = 13 ∨ x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17)) :=
sorry

end NUMINAMATH_GPT_count_positive_integers_x_satisfying_inequality_l315_31544


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l315_31526

variable (x y : ℚ)

theorem simplify_and_evaluate_expression (hx : x = 1) (hy : y = 1 / 2) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x - y) ^ 2 = 31 / 4 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l315_31526


namespace NUMINAMATH_GPT_coefficient_of_m5n4_in_expansion_l315_31508

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_of_m5n4_in_expansion : binomial_coefficient 9 5 = 126 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_m5n4_in_expansion_l315_31508


namespace NUMINAMATH_GPT_fraction_value_l315_31553

def op_at (a b : ℤ) : ℤ := a * b - b ^ 2
def op_sharp (a b : ℤ) : ℤ := a + b - a * b ^ 2

theorem fraction_value : (op_at 7 3) / (op_sharp 7 3) = -12 / 53 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l315_31553


namespace NUMINAMATH_GPT_refreshment_stand_distance_l315_31504

theorem refreshment_stand_distance 
  (A B S : ℝ) -- Positions of the camps and refreshment stand
  (dist_A_highway : A = 400) -- Distance from the first camp to the highway
  (dist_B_A : B = 700) -- Distance from the second camp directly across the highway
  (equidistant : ∀ x, S = x ∧ dist (S, A) = dist (S, B)) : 
  S = 500 := -- Distance from the refreshment stand to each camp is 500 meters
sorry

end NUMINAMATH_GPT_refreshment_stand_distance_l315_31504


namespace NUMINAMATH_GPT_polynomial_two_distinct_negative_real_roots_l315_31510

theorem polynomial_two_distinct_negative_real_roots :
  ∀ (p : ℝ), 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
    (x1^4 + p*x1^3 + 3*x1^2 + p*x1 + 4 = 0) ∧ 
    (x2^4 + p*x2^3 + 3*x2^2 + p*x2 + 4 = 0)) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_two_distinct_negative_real_roots_l315_31510


namespace NUMINAMATH_GPT_alice_current_age_l315_31595

def alice_age_twice_eve (a b : Nat) : Prop := a = 2 * b

def eve_age_after_10_years (a b : Nat) : Prop := a = b + 10

theorem alice_current_age (a b : Nat) (h1 : alice_age_twice_eve a b) (h2 : eve_age_after_10_years a b) : a = 20 := by
  sorry

end NUMINAMATH_GPT_alice_current_age_l315_31595


namespace NUMINAMATH_GPT_find_angle_C_l315_31576

theorem find_angle_C (A B C : ℝ) (h1 : A = 88) (h2 : B - C = 20) (angle_sum : A + B + C = 180) : C = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_l315_31576


namespace NUMINAMATH_GPT_maximizing_sum_of_arithmetic_sequence_l315_31599

theorem maximizing_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_decreasing : ∀ n, a n > a (n + 1))
  (h_sum : S 5 = S 10) :
  (S 7 >= S n ∧ S 8 >= S n) := sorry

end NUMINAMATH_GPT_maximizing_sum_of_arithmetic_sequence_l315_31599


namespace NUMINAMATH_GPT_total_weight_proof_l315_31554

-- Define molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Define moles of elements in each compound
def moles_C4H10 : ℕ := 8
def moles_C3H8 : ℕ := 5
def moles_CH4 : ℕ := 3

-- Define the molar masses of each compound
def molar_mass_C4H10 : ℝ := 4 * molar_mass_C + 10 * molar_mass_H
def molar_mass_C3H8 : ℝ := 3 * molar_mass_C + 8 * molar_mass_H
def molar_mass_CH4 : ℝ := 1 * molar_mass_C + 4 * molar_mass_H

-- Define the total weight
def total_weight : ℝ :=
  moles_C4H10 * molar_mass_C4H10 +
  moles_C3H8 * molar_mass_C3H8 +
  moles_CH4 * molar_mass_CH4

theorem total_weight_proof :
  total_weight = 733.556 := by
  sorry

end NUMINAMATH_GPT_total_weight_proof_l315_31554


namespace NUMINAMATH_GPT_range_of_a_l315_31535

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, a*x^2 - 2*a*x + 3 ≤ 0) ↔ (0 ≤ a ∧ a < 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l315_31535


namespace NUMINAMATH_GPT_alice_ride_top_speed_l315_31511

-- Define the conditions
variables (x y : Real) -- x is the hours at 25 mph, y is the hours at 15 mph.
def distance_eq : Prop := 25 * x + 15 * y + 10 * (9 - x - y) = 162
def time_eq : Prop := x + y ≤ 9

-- Define the final answer
def final_answer : Prop := x = 2.7

-- The statement to prove
theorem alice_ride_top_speed : distance_eq x y ∧ time_eq x y → final_answer x := sorry

end NUMINAMATH_GPT_alice_ride_top_speed_l315_31511


namespace NUMINAMATH_GPT_semicircle_radius_l315_31533

noncomputable def radius_of_inscribed_semicircle (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21) : ℝ :=
  let AB := Real.sqrt (21^2 + 10^2)
  let s := 2 * Real.sqrt 541
  let area := 20 * 21
  (area) / (s * 2)

theorem semicircle_radius (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21)
  : radius_of_inscribed_semicircle BD height h_base h_height = 210 / Real.sqrt 541 :=
sorry

end NUMINAMATH_GPT_semicircle_radius_l315_31533


namespace NUMINAMATH_GPT_simplify_expression_l315_31585

theorem simplify_expression (x : ℝ) : 
  ( ( (x^(16/8))^(1/4) )^3 * ( (x^(16/4))^(1/8) )^5 ) = x^4 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l315_31585


namespace NUMINAMATH_GPT_q_value_l315_31528

-- Define the conditions and the problem statement
theorem q_value (a b m p q : ℚ) (h1 : a * b = 3) 
  (h2 : (a + 1 / b) * (b + 1 / a) = q) : 
  q = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_q_value_l315_31528


namespace NUMINAMATH_GPT_sangwoo_gave_away_notebooks_l315_31509

variables (n : ℕ)

theorem sangwoo_gave_away_notebooks
  (h1 : 12 - n + 34 - 3 * n = 30) :
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_sangwoo_gave_away_notebooks_l315_31509


namespace NUMINAMATH_GPT_max_value_of_expression_achieve_max_value_l315_31512

theorem max_value_of_expression : 
  ∀ x : ℝ, -3 * x ^ 2 + 18 * x - 4 ≤ 77 :=
by
  -- Placeholder proof
  sorry

theorem achieve_max_value : 
  ∃ x : ℝ, -3 * x ^ 2 + 18 * x - 4 = 77 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_max_value_of_expression_achieve_max_value_l315_31512


namespace NUMINAMATH_GPT_Shara_savings_l315_31538

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end NUMINAMATH_GPT_Shara_savings_l315_31538


namespace NUMINAMATH_GPT_fraction_meaningful_range_l315_31589

variable (x : ℝ)

theorem fraction_meaningful_range (h : x - 2 ≠ 0) : x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_range_l315_31589


namespace NUMINAMATH_GPT_p_iff_q_l315_31518

theorem p_iff_q (a b : ℝ) : (a > b) ↔ (a^3 > b^3) :=
sorry

end NUMINAMATH_GPT_p_iff_q_l315_31518


namespace NUMINAMATH_GPT_Rachel_total_books_l315_31501

theorem Rachel_total_books :
  (8 * 15) + (4 * 15) + (3 * 15) + (5 * 15) = 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_Rachel_total_books_l315_31501


namespace NUMINAMATH_GPT_min_bottles_to_fill_large_bottle_l315_31529

theorem min_bottles_to_fill_large_bottle (large_bottle_ml : Nat) (small_bottle1_ml : Nat) (small_bottle2_ml : Nat) (total_bottles : Nat) :
  large_bottle_ml = 800 ∧ small_bottle1_ml = 45 ∧ small_bottle2_ml = 60 ∧ total_bottles = 14 →
  ∃ x y : Nat, x * small_bottle1_ml + y * small_bottle2_ml = large_bottle_ml ∧ x + y = total_bottles :=
by
  intro h
  sorry

end NUMINAMATH_GPT_min_bottles_to_fill_large_bottle_l315_31529


namespace NUMINAMATH_GPT_sequence_periodicity_a5_a2019_l315_31531

theorem sequence_periodicity_a5_a2019 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → a n * a (n + 2) = 3 * a (n + 1)) :
  a 5 * a 2019 = 27 :=
sorry

end NUMINAMATH_GPT_sequence_periodicity_a5_a2019_l315_31531


namespace NUMINAMATH_GPT_find_k_l315_31552

theorem find_k (a b : ℤ × ℤ) (k : ℤ) 
  (h₁ : a = (2, 1)) 
  (h₂ : a.1 + b.1 = 1 ∧ a.2 + b.2 = k)
  (h₃ : a.1 * b.1 + a.2 * b.2 = 0) : k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l315_31552


namespace NUMINAMATH_GPT_probability_three_non_red_purple_balls_l315_31578

def total_balls : ℕ := 150
def prob_white : ℝ := 0.15
def prob_green : ℝ := 0.20
def prob_yellow : ℝ := 0.30
def prob_red : ℝ := 0.30
def prob_purple : ℝ := 0.05
def prob_not_red_purple : ℝ := 1 - (prob_red + prob_purple)

theorem probability_three_non_red_purple_balls :
  (prob_not_red_purple * prob_not_red_purple * prob_not_red_purple) = 0.274625 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_non_red_purple_balls_l315_31578


namespace NUMINAMATH_GPT_find_balloons_given_to_Fred_l315_31569

variable (x : ℝ)
variable (Sam_initial_balance : ℝ := 46.0)
variable (Dan_balance : ℝ := 16.0)
variable (total_balance : ℝ := 52.0)

theorem find_balloons_given_to_Fred
  (h : Sam_initial_balance - x + Dan_balance = total_balance) :
  x = 10.0 :=
by
  sorry

end NUMINAMATH_GPT_find_balloons_given_to_Fred_l315_31569


namespace NUMINAMATH_GPT_family_trip_eggs_l315_31558

theorem family_trip_eggs (adults girls boys : ℕ)
  (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) (extra_eggs_for_boy : ℕ) :
  adults = 3 →
  eggs_per_adult = 3 →
  girls = 7 →
  eggs_per_girl = 1 →
  boys = 10 →
  extra_eggs_for_boy = 1 →
  (adults * eggs_per_adult + girls * eggs_per_girl + boys * (eggs_per_girl + extra_eggs_for_boy)) = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_family_trip_eggs_l315_31558


namespace NUMINAMATH_GPT_lines_parallel_l315_31515

def line1 (x y : ℝ) : Prop := x - y + 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem lines_parallel : 
  (∀ x y, line1 x y ↔ y = x + 2) ∧ 
  (∀ x y, line2 x y ↔ y = x + 1) ∧ 
  ∃ m₁ m₂ c₁ c₂, (∀ x y, (y = m₁ * x + c₁) ↔ line1 x y) ∧ (∀ x y, (y = m₂ * x + c₂) ↔ line2 x y) ∧ m₁ = m₂ ∧ c₁ ≠ c₂ :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_l315_31515


namespace NUMINAMATH_GPT_carrots_thrown_out_l315_31580

def initial_carrots := 19
def additional_carrots := 46
def total_current_carrots := 61

def total_picked := initial_carrots + additional_carrots

theorem carrots_thrown_out : total_picked - total_current_carrots = 4 := by
  sorry

end NUMINAMATH_GPT_carrots_thrown_out_l315_31580


namespace NUMINAMATH_GPT_problem_statement_l315_31590

theorem problem_statement (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℤ, (1/2 + 1/3 + 1/11 + 1/n : ℚ) = k) : ¬ (n > 66) := 
sorry

end NUMINAMATH_GPT_problem_statement_l315_31590


namespace NUMINAMATH_GPT_min_value_of_a_and_b_l315_31588

theorem min_value_of_a_and_b (a b : ℝ) (h : a ^ 2 + 2 * b ^ 2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 6 → x + y ≥ m) ∧ (a + b = m) :=
sorry

end NUMINAMATH_GPT_min_value_of_a_and_b_l315_31588


namespace NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l315_31506

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (h_prime : Prime p) (h_gt_3 : p > 3) : 
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
sorry

end NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l315_31506


namespace NUMINAMATH_GPT_number_of_attendees_choosing_water_l315_31560

variables {total_attendees : ℕ} (juice_percent water_percent : ℚ)

-- Conditions
def attendees_juice (total_attendees : ℕ) : ℚ := 0.7 * total_attendees
def attendees_water (total_attendees : ℕ) : ℚ := 0.3 * total_attendees
def attendees_juice_given := (attendees_juice total_attendees) = 140

-- Theorem statement
theorem number_of_attendees_choosing_water 
  (h1 : juice_percent = 0.7) 
  (h2 : water_percent = 0.3) 
  (h3 : attendees_juice total_attendees = 140) : 
  attendees_water total_attendees = 60 :=
sorry

end NUMINAMATH_GPT_number_of_attendees_choosing_water_l315_31560


namespace NUMINAMATH_GPT_simplify_expression_l315_31502

theorem simplify_expression (a b : ℚ) : (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l315_31502


namespace NUMINAMATH_GPT_candy_difference_l315_31524

def given_away : ℕ := 6
def left : ℕ := 5
def difference : ℕ := given_away - left

theorem candy_difference :
  difference = 1 :=
by
  sorry

end NUMINAMATH_GPT_candy_difference_l315_31524


namespace NUMINAMATH_GPT_polynomial_simplification_l315_31540

variable (x : ℝ)

theorem polynomial_simplification :
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l315_31540


namespace NUMINAMATH_GPT_difference_of_numbers_l315_31555

theorem difference_of_numbers (x y : ℕ) (h1 : x + y = 64) (h2 : y = 26) : x - y = 12 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l315_31555


namespace NUMINAMATH_GPT_find_a_value_l315_31519

theorem find_a_value (a : ℝ) (x : ℝ) :
  (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 →
  a + 1 ≠ 0 →
  a^2 + 1 = 9 →
  a = 2 * Real.sqrt 2 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_a_value_l315_31519


namespace NUMINAMATH_GPT_mono_sum_eq_five_l315_31513

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end NUMINAMATH_GPT_mono_sum_eq_five_l315_31513


namespace NUMINAMATH_GPT_m_plus_n_is_23_l315_31532

noncomputable def find_m_plus_n : ℕ := 
  let A := 12
  let B := 4
  let C := 3
  let D := 3

  -- Declare the radius of E
  let radius_E : ℚ := (21 / 2)
  
  -- Let radius_E be written as m / n where m and n are relatively prime
  let (m : ℕ) := 21
  let (n : ℕ) := 2

  -- Calculate m + n
  m + n

theorem m_plus_n_is_23 : find_m_plus_n = 23 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_m_plus_n_is_23_l315_31532


namespace NUMINAMATH_GPT_find_e_value_l315_31573

theorem find_e_value : (14 ^ 2) * (5 ^ 3) * 568 = 13916000 := by
  sorry

end NUMINAMATH_GPT_find_e_value_l315_31573


namespace NUMINAMATH_GPT_sqrt_30_estimate_l315_31520

theorem sqrt_30_estimate : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_30_estimate_l315_31520


namespace NUMINAMATH_GPT_sin_double_angle_l315_31503

variable {α : Real}

theorem sin_double_angle (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l315_31503


namespace NUMINAMATH_GPT_cylinder_height_l315_31536

variable (r h : ℝ) (SA : ℝ)

theorem cylinder_height (h : ℝ) (r : ℝ) (SA : ℝ) (h_eq : h = 2) (r_eq : r = 3) (SA_eq : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h → h = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cylinder_height_l315_31536


namespace NUMINAMATH_GPT_greatest_large_chips_l315_31548

theorem greatest_large_chips :
  ∃ (l : ℕ), (∃ (s : ℕ), ∃ (p : ℕ), s + l = 70 ∧ s = l + p ∧ Nat.Prime p) ∧ 
  (∀ (l' : ℕ), (∃ (s' : ℕ), ∃ (p' : ℕ), s' + l' = 70 ∧ s' = l' + p' ∧ Nat.Prime p') → l' ≤ 34) :=
sorry

end NUMINAMATH_GPT_greatest_large_chips_l315_31548


namespace NUMINAMATH_GPT_determine_ab_l315_31586

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2 * a * x + b * 2^x

theorem determine_ab (a b : ℕ) (h : ∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) :
  (a, b) = (0, 0) ∨ (a, b) = (1, 0) :=
by
  sorry

end NUMINAMATH_GPT_determine_ab_l315_31586


namespace NUMINAMATH_GPT_journey_time_calculation_l315_31574

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end NUMINAMATH_GPT_journey_time_calculation_l315_31574


namespace NUMINAMATH_GPT_div_operation_example_l315_31542

theorem div_operation_example : ((180 / 6) / 3) = 10 := by
  sorry

end NUMINAMATH_GPT_div_operation_example_l315_31542


namespace NUMINAMATH_GPT_arun_completes_work_alone_in_70_days_l315_31516

def arun_days (A : ℕ) : Prop :=
  ∃ T : ℕ, (A > 0) ∧ (T > 0) ∧ 
           (∀ (work_done_by_arun_in_1_day work_done_by_tarun_in_1_day : ℝ),
            work_done_by_arun_in_1_day = 1 / A ∧
            work_done_by_tarun_in_1_day = 1 / T ∧
            (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day = 1 / 10) ∧
            (4 * (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day) = 4 / 10) ∧
            (42 * work_done_by_arun_in_1_day = 6 / 10) )

theorem arun_completes_work_alone_in_70_days : arun_days 70 :=
  sorry

end NUMINAMATH_GPT_arun_completes_work_alone_in_70_days_l315_31516


namespace NUMINAMATH_GPT_sum_of_possible_values_of_g_l315_31523

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (x : ℝ) : ℝ := 3 * x - 4

theorem sum_of_possible_values_of_g :
  let x1 := (9 + 3 * Real.sqrt 5) / 2
  let x2 := (9 - 3 * Real.sqrt 5) / 2
  g x1 + g x2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_g_l315_31523


namespace NUMINAMATH_GPT_opposite_of_neg_two_thirds_l315_31597

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_thirds_l315_31597


namespace NUMINAMATH_GPT_indira_cricket_minutes_l315_31591

def totalMinutesSeanPlayed (sean_minutes_per_day : ℕ) (days : ℕ) : ℕ :=
  sean_minutes_per_day * days

def totalMinutesIndiraPlayed (total_minutes_together : ℕ) (total_minutes_sean : ℕ) : ℕ :=
  total_minutes_together - total_minutes_sean

theorem indira_cricket_minutes :
  totalMinutesIndiraPlayed 1512 (totalMinutesSeanPlayed 50 14) = 812 :=
by
  sorry

end NUMINAMATH_GPT_indira_cricket_minutes_l315_31591


namespace NUMINAMATH_GPT_value_of_P_dot_Q_l315_31514

def P : Set ℝ := {x | Real.log x / Real.log 2 < 1}
def Q : Set ℝ := {x | abs (x - 2) < 1}
def P_dot_Q (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∧ x ∉ Q}

theorem value_of_P_dot_Q : P_dot_Q P Q = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_value_of_P_dot_Q_l315_31514


namespace NUMINAMATH_GPT_probability_of_multiple_of_3_is_1_5_l315_31563

-- Definition of the problem conditions
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Function to calculate the probability
noncomputable def probability_of_multiple_of_3 : ℚ := 
  let total_permutations := (Nat.factorial 5) / (Nat.factorial (5 - 4))  -- i.e., 120
  let valid_permutations := Nat.factorial 4  -- i.e., 24, for the valid combination
  valid_permutations / total_permutations 

-- Statement to be proved
theorem probability_of_multiple_of_3_is_1_5 :
  probability_of_multiple_of_3 = 1 / 5 := 
by
  -- Skeleton for the proof
  sorry

end NUMINAMATH_GPT_probability_of_multiple_of_3_is_1_5_l315_31563


namespace NUMINAMATH_GPT_triangle_area_from_altitudes_l315_31594

noncomputable def triangleArea (altitude1 altitude2 altitude3 : ℝ) : ℝ :=
  sorry

theorem triangle_area_from_altitudes
  (h1 : altitude1 = 15)
  (h2 : altitude2 = 21)
  (h3 : altitude3 = 35) :
  triangleArea 15 21 35 = 245 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_from_altitudes_l315_31594


namespace NUMINAMATH_GPT_inequality_of_triangle_tangents_l315_31579

theorem inequality_of_triangle_tangents
  (a b c x y z : ℝ)
  (h1 : a = y + z)
  (h2 : b = x + z)
  (h3 : c = x + y)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_tangents : z ≥ y ∧ y ≥ x) :
  (a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2) ∧
  ((a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z) :=
sorry

end NUMINAMATH_GPT_inequality_of_triangle_tangents_l315_31579


namespace NUMINAMATH_GPT_average_is_equal_l315_31505

theorem average_is_equal (x : ℝ) :
  (1 / 3) * (2 * x + 4 + 5 * x + 3 + 3 * x + 8) = 3 * x - 5 → 
  x = -30 :=
by
  sorry

end NUMINAMATH_GPT_average_is_equal_l315_31505


namespace NUMINAMATH_GPT_min_dot_product_l315_31581

-- Define the conditions of the ellipse and focal points
variables (P : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define vectors
def OP (P : ℝ × ℝ) : ℝ × ℝ := P
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Prove that the minimum value of the dot product is 2
theorem min_dot_product (hP : ellipse P.1 P.2) : 
  ∃ (P : ℝ × ℝ), dot_product (OP P) (FP P) = 2 := sorry

end NUMINAMATH_GPT_min_dot_product_l315_31581


namespace NUMINAMATH_GPT_moles_C2H6_for_HCl_l315_31583

theorem moles_C2H6_for_HCl 
  (form_HCl : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : ℕ) : 
  (6 * (reaction * moles_Cl2)) = form_HCl * (6 * reaction) :=
by
  -- The necessary proof steps will go here
  sorry

end NUMINAMATH_GPT_moles_C2H6_for_HCl_l315_31583


namespace NUMINAMATH_GPT_ratio_of_areas_of_circles_l315_31562

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_circles_l315_31562


namespace NUMINAMATH_GPT_valid_integer_pairs_l315_31567

theorem valid_integer_pairs :
  ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → a ^ (b ^ 2) = b ^ a → (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end NUMINAMATH_GPT_valid_integer_pairs_l315_31567


namespace NUMINAMATH_GPT_hydrochloric_acid_moles_l315_31575

theorem hydrochloric_acid_moles (amyl_alcohol moles_required : ℕ) 
  (h_ratio : amyl_alcohol = moles_required) 
  (h_balanced : amyl_alcohol = 3) :
  moles_required = 3 :=
by
  sorry

end NUMINAMATH_GPT_hydrochloric_acid_moles_l315_31575


namespace NUMINAMATH_GPT_a_n_divisible_by_11_l315_31565

-- Define the sequence
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

-- Main statement
theorem a_n_divisible_by_11 (a : ℕ → ℤ) (h : seq a) :
  ∀ n, ∃ k : ℕ, a n % 11 = 0 ↔ n = 4 + 11 * k :=
sorry

end NUMINAMATH_GPT_a_n_divisible_by_11_l315_31565


namespace NUMINAMATH_GPT_simplify_sqrt8_minus_sqrt2_l315_31500

theorem simplify_sqrt8_minus_sqrt2 :
  (Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_simplify_sqrt8_minus_sqrt2_l315_31500


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_l315_31572

theorem line_through_point_equal_intercepts (a b : ℝ) : 
  ((∃ (k : ℝ), k ≠ 0 ∧ (3 = 2 * k) ∧ b = k) ∨ ((a ≠ 0) ∧ (5/a = 1))) → 
  (a = 1 ∧ b = 1) ∨ (3 * a - 2 * b = 0) := 
by 
  sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_l315_31572


namespace NUMINAMATH_GPT_units_digit_7_pow_3_pow_4_l315_31571

theorem units_digit_7_pow_3_pow_4 :
  (7 ^ (3 ^ 4)) % 10 = 7 :=
by
  -- Here's the proof placeholder
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_3_pow_4_l315_31571


namespace NUMINAMATH_GPT_cricket_initial_overs_l315_31521

/-- Prove that the number of initial overs played was 10. -/
theorem cricket_initial_overs 
  (target : ℝ) 
  (initial_run_rate : ℝ) 
  (remaining_run_rate : ℝ) 
  (remaining_overs : ℕ)
  (h_target : target = 282)
  (h_initial_run_rate : initial_run_rate = 4.6)
  (h_remaining_run_rate : remaining_run_rate = 5.9)
  (h_remaining_overs : remaining_overs = 40) 
  : ∃ x : ℝ, x = 10 := 
by
  sorry

end NUMINAMATH_GPT_cricket_initial_overs_l315_31521


namespace NUMINAMATH_GPT_A_can_finish_remaining_work_in_6_days_l315_31539

-- Condition: A can finish the work in 18 days
def A_work_rate := 1 / 18

-- Condition: B can finish the work in 15 days
def B_work_rate := 1 / 15

-- Given B worked for 10 days
def B_days_worked := 10

-- Calculation of the remaining work
def remaining_work := 1 - B_days_worked * B_work_rate

-- Calculation of the time for A to finish the remaining work
def A_remaining_days := remaining_work / A_work_rate

-- The theorem to prove
theorem A_can_finish_remaining_work_in_6_days : A_remaining_days = 6 := 
by 
  -- The proof is not required, so we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_A_can_finish_remaining_work_in_6_days_l315_31539


namespace NUMINAMATH_GPT_eval_7_star_3_l315_31559

def operation (a b : ℕ) : ℕ := (4 * a + 5 * b - a * b)

theorem eval_7_star_3 : operation 7 3 = 22 :=
  by {
    -- substitution and calculation steps
    sorry
  }

end NUMINAMATH_GPT_eval_7_star_3_l315_31559


namespace NUMINAMATH_GPT_find_m_n_l315_31525

theorem find_m_n (x m n : ℤ) : (x + 2) * (x + 3) = x^2 + m * x + n → m = 5 ∧ n = 6 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_m_n_l315_31525


namespace NUMINAMATH_GPT_solve_fractional_equation_l315_31587

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l315_31587


namespace NUMINAMATH_GPT_bicycle_speed_B_l315_31550

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_speed_B_l315_31550


namespace NUMINAMATH_GPT_part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l315_31537

-- Define the game rules and conditions for the proof
def takeMatches (total_matches : Nat) (taken_matches : Nat) : Nat :=
  total_matches - taken_matches

-- Part (a) statement
theorem part_a_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (b) statement
theorem part_b_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (c) generalized statement for game type (a)
theorem part_c_winner_a (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

-- Part (c) generalized statement for game type (b)
theorem part_c_winner_b (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

end NUMINAMATH_GPT_part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l315_31537


namespace NUMINAMATH_GPT_ratio_a_d_l315_31556

theorem ratio_a_d (a b c d : ℕ) 
  (hab : a * 4 = b * 3) 
  (hbc : b * 9 = c * 7) 
  (hcd : c * 7 = d * 5) : 
  a * 12 = d :=
sorry

end NUMINAMATH_GPT_ratio_a_d_l315_31556


namespace NUMINAMATH_GPT_savings_if_together_l315_31517

def window_price : ℕ := 100

def free_windows_for_six_purchased : ℕ := 2

def windows_needed_Dave : ℕ := 9
def windows_needed_Doug : ℕ := 10

def total_individual_cost (windows_purchased : ℕ) : ℕ :=
  100 * windows_purchased

def total_cost_with_deal (windows_purchased: ℕ) : ℕ :=
  let sets_of_6 := windows_purchased / 6
  let remaining_windows := windows_purchased % 6
  100 * (sets_of_6 * 6 + remaining_windows)

def combined_savings (windows_needed_Dave: ℕ) (windows_needed_Doug: ℕ) : ℕ :=
  let total_windows := windows_needed_Dave + windows_needed_Doug
  total_individual_cost windows_needed_Dave 
  + total_individual_cost windows_needed_Doug 
  - total_cost_with_deal total_windows

theorem savings_if_together : combined_savings windows_needed_Dave windows_needed_Doug = 400 :=
by
  sorry

end NUMINAMATH_GPT_savings_if_together_l315_31517


namespace NUMINAMATH_GPT_negation_of_prop1_equiv_l315_31596

-- Given proposition: if x > 1 then x > 0
def prop1 (x : ℝ) : Prop := x > 1 → x > 0

-- Negation of the given proposition: if x ≤ 1 then x ≤ 0
def neg_prop1 (x : ℝ) : Prop := x ≤ 1 → x ≤ 0

-- The theorem to prove that the negation of the proposition "If x > 1, then x > 0" 
-- is "If x ≤ 1, then x ≤ 0"
theorem negation_of_prop1_equiv (x : ℝ) : ¬(prop1 x) ↔ neg_prop1 x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop1_equiv_l315_31596


namespace NUMINAMATH_GPT_no_integer_solution_l315_31546

theorem no_integer_solution (x y : ℤ) : ¬(x^4 + y^2 = 4 * y + 4) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_l315_31546


namespace NUMINAMATH_GPT_real_part_of_complex_div_l315_31561

noncomputable def complexDiv (c1 c2 : ℂ) := c1 / c2

theorem real_part_of_complex_div (i_unit : ℂ) (h_i : i_unit = Complex.I) :
  (Complex.re (complexDiv (2 * i_unit) (1 + i_unit)) = 1) :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_complex_div_l315_31561


namespace NUMINAMATH_GPT_correct_option_is_C_l315_31566

-- Define the polynomial expressions and their expected values as functions
def optionA (x : ℝ) : Prop := (x + 2) * (x - 5) = x^2 - 2 * x - 3
def optionB (x : ℝ) : Prop := (x + 3) * (x - 1 / 3) = x^2 + x - 1
def optionC (x : ℝ) : Prop := (x - 2 / 3) * (x + 1 / 2) = x^2 - 1 / 6 * x - 1 / 3
def optionD (x : ℝ) : Prop := (x - 2) * (-x - 2) = x^2 - 4

-- Problem Statement: Verify that the polynomial multiplication in Option C is correct
theorem correct_option_is_C (x : ℝ) : optionC x :=
by
  -- Statement indicating the proof goes here
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l315_31566


namespace NUMINAMATH_GPT_complex_calculation_l315_31564

def complex_add (a b : ℂ) : ℂ := a + b
def complex_mul (a b : ℂ) : ℂ := a * b

theorem complex_calculation :
  let z1 := (⟨2, -3⟩ : ℂ)
  let z2 := (⟨4, 6⟩ : ℂ)
  let z3 := (⟨-1, 2⟩ : ℂ)
  complex_mul (complex_add z1 z2) z3 = (⟨-12, 9⟩ : ℂ) :=
by 
  sorry

end NUMINAMATH_GPT_complex_calculation_l315_31564


namespace NUMINAMATH_GPT_jack_jill_same_speed_l315_31577

-- Definitions for Jack and Jill's conditions
def jacks_speed (x : ℝ) : ℝ := x^2 - 13*x - 48
def jills_distance (x : ℝ) : ℝ := x^2 - 5*x - 84
def jills_time (x : ℝ) : ℝ := x + 8

-- Theorem stating the same walking speed given the conditions
theorem jack_jill_same_speed (x : ℝ) (h : jacks_speed x = jills_distance x / jills_time x) : 
  jacks_speed x = 6 :=
by
  sorry

end NUMINAMATH_GPT_jack_jill_same_speed_l315_31577


namespace NUMINAMATH_GPT_dorothy_age_relation_l315_31543

theorem dorothy_age_relation (D S : ℕ) (h1: S = 5) (h2: D + 5 = 2 * (S + 5)) : D = 3 * S :=
by
  -- implement the proof here
  sorry

end NUMINAMATH_GPT_dorothy_age_relation_l315_31543


namespace NUMINAMATH_GPT_book_pages_total_l315_31507

-- Define the conditions as hypotheses
def total_pages (P : ℕ) : Prop :=
  let read_first_day := P / 2
  let read_second_day := P / 4
  let read_third_day := P / 6
  let read_total := read_first_day + read_second_day + read_third_day
  let remaining_pages := P - read_total
  remaining_pages = 20

-- The proof statement
theorem book_pages_total (P : ℕ) (h : total_pages P) : P = 240 := sorry

end NUMINAMATH_GPT_book_pages_total_l315_31507


namespace NUMINAMATH_GPT_total_baseball_cards_l315_31551

-- Define the number of baseball cards each person has
def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

-- The total number of baseball cards they have
theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards + john_cards + sarah_cards + emma_cards = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_baseball_cards_l315_31551


namespace NUMINAMATH_GPT_billy_music_book_songs_l315_31598

theorem billy_music_book_songs (can_play : ℕ) (needs_to_learn : ℕ) (total_songs : ℕ) 
  (h1 : can_play = 24) (h2 : needs_to_learn = 28) : 
  total_songs = can_play + needs_to_learn ↔ total_songs = 52 :=
by
  sorry

end NUMINAMATH_GPT_billy_music_book_songs_l315_31598


namespace NUMINAMATH_GPT_num_two_digit_multiples_5_and_7_l315_31557

/-- 
    Theorem: There are exactly 2 positive two-digit integers that are multiples of both 5 and 7.
-/
theorem num_two_digit_multiples_5_and_7 : 
  ∃ (count : ℕ), count = 2 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → 
    (n % 5 = 0 ∧ n % 7 = 0) ↔ (n = 35 ∨ n = 70) := 
by
  sorry

end NUMINAMATH_GPT_num_two_digit_multiples_5_and_7_l315_31557


namespace NUMINAMATH_GPT_sequence_a_n_correctness_l315_31522

theorem sequence_a_n_correctness (a : ℕ → ℚ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = 2 * a n + 1) : a 2 = 1.5 := by
  sorry

end NUMINAMATH_GPT_sequence_a_n_correctness_l315_31522


namespace NUMINAMATH_GPT_heaviest_tv_l315_31549

theorem heaviest_tv :
  let area (width height : ℝ) := width * height
  let weight (area : ℝ) := area * 4
  let weight_in_pounds (weight : ℝ) := weight / 16
  let bill_area := area 48 100
  let bob_area := area 70 60
  let steve_area := area 84 92
  let bill_weight := weight bill_area
  let bob_weight := weight bob_area
  let steve_weight := weight steve_area
  let bill_weight_pounds := weight_in_pounds (weight bill_area)
  let bob_weight_pounds := weight_in_pounds (weight bob_area)
  let steve_weight_pounds := weight_in_pounds (weight steve_area)
  bob_weight_pounds + bill_weight_pounds < steve_weight_pounds
  ∧ abs ((steve_weight_pounds) - (bill_weight_pounds + bob_weight_pounds)) = 318 :=
by
  sorry

end NUMINAMATH_GPT_heaviest_tv_l315_31549


namespace NUMINAMATH_GPT_three_million_times_three_million_l315_31592

theorem three_million_times_three_million : 
  (3 * 10^6) * (3 * 10^6) = 9 * 10^12 := 
by
  sorry

end NUMINAMATH_GPT_three_million_times_three_million_l315_31592


namespace NUMINAMATH_GPT_area_EFCD_l315_31534

-- Defining the geometrical setup and measurements of the trapezoid
variables (AB CD AD BC : ℝ) (h1 : AB = 10) (h2 : CD = 30) (h_altitude : ∃ h : ℝ, h = 18)

-- Defining the midpoints E and F of AD and BC respectively
variables (E F : ℝ) (h_E : E = AD / 2) (h_F : F = BC / 2)

-- Define the intersection of diagonals and the ratio condition
variables (AC BD G : ℝ) (h_ratio : ∃ r : ℝ, r = 1/2)

-- Proving the area of quadrilateral EFCD
theorem area_EFCD : EFCD_area = 225 :=
sorry

end NUMINAMATH_GPT_area_EFCD_l315_31534


namespace NUMINAMATH_GPT_find_particular_number_l315_31568

theorem find_particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 :=
by {
  -- The proof will be written here.
  sorry
}

end NUMINAMATH_GPT_find_particular_number_l315_31568


namespace NUMINAMATH_GPT_exist_six_subsets_of_six_elements_l315_31547

theorem exist_six_subsets_of_six_elements (n m : ℕ) (X : Finset ℕ) (A : Fin m → Finset ℕ) :
    n > 6 →
    X.card = n →
    (∀ i, (A i).card = 5 ∧ (A i ⊆ X)) →
    m > (n * (n-1) * (n-2) * (n-3) * (4*n-15)) / 600 →
    ∃ i1 i2 i3 i4 i5 i6 : Fin m,
      i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧
      (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end NUMINAMATH_GPT_exist_six_subsets_of_six_elements_l315_31547


namespace NUMINAMATH_GPT_initial_cost_of_smartphone_l315_31584

theorem initial_cost_of_smartphone 
(C : ℝ) 
(h : 0.85 * C = 255) : 
C = 300 := 
sorry

end NUMINAMATH_GPT_initial_cost_of_smartphone_l315_31584


namespace NUMINAMATH_GPT_equal_circle_radius_l315_31527

theorem equal_circle_radius (r R : ℝ) (h1: r > 0) (h2: R > 0)
  : ∃ x : ℝ, x = r * R / (R + r) :=
by 
  sorry

end NUMINAMATH_GPT_equal_circle_radius_l315_31527


namespace NUMINAMATH_GPT_complement_set_example_l315_31582

open Set

variable (U M : Set ℕ)

def complement (U M : Set ℕ) := U \ M

theorem complement_set_example :
  (U = {1, 2, 3, 4, 5, 6}) → 
  (M = {1, 3, 5}) → 
  (complement U M = {2, 4, 6}) := by
  intros hU hM
  rw [complement, hU, hM]
  sorry

end NUMINAMATH_GPT_complement_set_example_l315_31582


namespace NUMINAMATH_GPT_opposite_of_neg_nine_is_nine_l315_31545

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end NUMINAMATH_GPT_opposite_of_neg_nine_is_nine_l315_31545


namespace NUMINAMATH_GPT_sequence_a_n_definition_l315_31530

theorem sequence_a_n_definition (a : ℕ+ → ℝ) 
  (h₀ : ∀ n : ℕ+, a (n + 1) = 2016 * a n / (2014 * a n + 2016))
  (h₁ : a 1 = 1) : 
  a 2017 = 1008 / (1007 * 2017 + 1) :=
sorry

end NUMINAMATH_GPT_sequence_a_n_definition_l315_31530


namespace NUMINAMATH_GPT_possible_values_l315_31570

def expression (m n : ℕ) : ℤ :=
  (m^2 + m * n + n^2) / (m * n - 1)

theorem possible_values (m n : ℕ) (h : m * n ≠ 1) : 
  ∃ (N : ℤ), N = expression m n → N = 0 ∨ N = 4 ∨ N = 7 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_l315_31570


namespace NUMINAMATH_GPT_glue_needed_l315_31593

-- Definitions based on conditions
def num_friends : ℕ := 7
def clippings_per_friend : ℕ := 3
def drops_per_clipping : ℕ := 6

-- Calculation
def total_clippings : ℕ := num_friends * clippings_per_friend
def total_drops_of_glue : ℕ := drops_per_clipping * total_clippings

-- Theorem statement
theorem glue_needed : total_drops_of_glue = 126 := by
  sorry

end NUMINAMATH_GPT_glue_needed_l315_31593


namespace NUMINAMATH_GPT_number_of_customers_per_month_l315_31541

-- Define the constants and conditions
def price_lettuce_per_head : ℝ := 1
def price_tomato_per_piece : ℝ := 0.5
def num_lettuce_per_customer : ℕ := 2
def num_tomato_per_customer : ℕ := 4
def monthly_sales : ℝ := 2000

-- Calculate the cost per customer
def cost_per_customer : ℝ := 
  (num_lettuce_per_customer * price_lettuce_per_head) + 
  (num_tomato_per_customer * price_tomato_per_piece)

-- Prove the number of customers per month
theorem number_of_customers_per_month : monthly_sales / cost_per_customer = 500 :=
  by
    -- Here, we would write the proof steps
    sorry

end NUMINAMATH_GPT_number_of_customers_per_month_l315_31541
