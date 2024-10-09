import Mathlib

namespace lcm_of_numbers_l215_21577

theorem lcm_of_numbers (x : Nat) (h_ratio : x ≠ 0) (h_hcf : Nat.gcd (5 * x) (Nat.gcd (7 * x) (9 * x)) = 11) :
    Nat.lcm (5 * x) (Nat.lcm (7 * x) (9 * x)) = 99 :=
by
  sorry

end lcm_of_numbers_l215_21577


namespace calculate_a_minus_b_l215_21583

theorem calculate_a_minus_b (a b c : ℝ) (h1 : a - b - c = 3) (h2 : a - b + c = 11) : a - b = 7 :=
by 
  -- The proof would be fleshed out here.
  sorry

end calculate_a_minus_b_l215_21583


namespace pow_mod_eleven_l215_21557

theorem pow_mod_eleven : 
  ∀ (n : ℕ), (n ≡ 5 ^ 1 [MOD 11] → n ≡ 5 [MOD 11]) ∧ 
             (n ≡ 5 ^ 2 [MOD 11] → n ≡ 3 [MOD 11]) ∧ 
             (n ≡ 5 ^ 3 [MOD 11] → n ≡ 4 [MOD 11]) ∧ 
             (n ≡ 5 ^ 4 [MOD 11] → n ≡ 9 [MOD 11]) ∧ 
             (n ≡ 5 ^ 5 [MOD 11] → n ≡ 1 [MOD 11]) →
  5 ^ 1233 ≡ 4 [MOD 11] :=
by
  intro n h
  sorry

end pow_mod_eleven_l215_21557


namespace initial_number_of_men_l215_21545

theorem initial_number_of_men (M : ℕ) (F : ℕ) (h1 : F = M * 20) (h2 : (M - 100) * 10 = M * 15) : 
  M = 200 :=
  sorry

end initial_number_of_men_l215_21545


namespace no_nat_pairs_divisibility_l215_21527

theorem no_nat_pairs_divisibility (a b : ℕ) (hab : b^a ∣ a^b - 1) : false :=
sorry

end no_nat_pairs_divisibility_l215_21527


namespace larry_jogs_first_week_days_l215_21588

-- Defining the constants and conditions
def daily_jogging_time := 30 -- Larry jogs for 30 minutes each day
def total_jogging_time_in_hours := 4 -- Total jogging time in two weeks in hours
def total_jogging_time_in_minutes := total_jogging_time_in_hours * 60 -- Convert hours to minutes
def jogging_days_in_second_week := 5 -- Larry jogs 5 days in the second week
def daily_jogging_time_in_week2 := jogging_days_in_second_week * daily_jogging_time -- Total jogging time in minutes in the second week

-- Theorem statement
theorem larry_jogs_first_week_days : 
  (total_jogging_time_in_minutes - daily_jogging_time_in_week2) / daily_jogging_time = 3 :=
by
  -- Definitions and conditions used above should directly appear from the problem statement
  sorry

end larry_jogs_first_week_days_l215_21588


namespace value_of_expression_l215_21540

variables {x1 x2 x3 x4 x5 x6 : ℝ}

theorem value_of_expression
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 = 1)
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 = 14)
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 = 135) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 = 832 :=
by
  sorry

end value_of_expression_l215_21540


namespace x_axis_intercept_of_line_l215_21586

theorem x_axis_intercept_of_line (x : ℝ) : (∃ x, 2*x + 1 = 0) → x = - 1 / 2 :=
  by
    intro h
    obtain ⟨x, h1⟩ := h
    have : 2 * x + 1 = 0 := h1
    linarith [this]

end x_axis_intercept_of_line_l215_21586


namespace sufficient_but_not_necessary_l215_21504

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x + y = 1 → xy ≤ 1 / 4) ∧ (∃ x y : ℝ, xy ≤ 1 / 4 ∧ x + y ≠ 1) := by
  sorry

end sufficient_but_not_necessary_l215_21504


namespace train_speed_is_36_0036_kmph_l215_21522

noncomputable def train_length : ℝ := 130
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 27.997760179185665
noncomputable def speed_in_kmph : ℝ := (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_36_0036_kmph :
  abs (speed_in_kmph - 36.0036) < 0.001 :=
by
  sorry

end train_speed_is_36_0036_kmph_l215_21522


namespace arithmetic_sequence_n_l215_21514

theorem arithmetic_sequence_n (a_n : ℕ → ℕ) (S_n : ℕ) (n : ℕ) 
  (h1 : ∀ i, a_n i = 20 + (i - 1) * (54 - 20) / (n - 1)) 
  (h2 : S_n = 37 * n) 
  (h3 : S_n = 999) : 
  n = 27 :=
by sorry

end arithmetic_sequence_n_l215_21514


namespace train_length_l215_21512

open Real

/--
A train of a certain length can cross an electric pole in 30 sec with a speed of 43.2 km/h.
Prove that the length of the train is 360 meters.
-/
theorem train_length (t : ℝ) (v_kmh : ℝ) (length : ℝ) 
  (h_time : t = 30) 
  (h_speed_kmh : v_kmh = 43.2) 
  (h_length : length = v_kmh * (t * (1000 / 3600))) : 
  length = 360 := 
by
  -- skip the actual proof steps
  sorry

end train_length_l215_21512


namespace stratified_sampling_city_B_l215_21570

theorem stratified_sampling_city_B (sales_points_A : ℕ) (sales_points_B : ℕ) (sales_points_C : ℕ) (total_sales_points : ℕ) (sample_size : ℕ)
(h_total : total_sales_points = 450)
(h_sample : sample_size = 90)
(h_sales_points_A : sales_points_A = 180)
(h_sales_points_B : sales_points_B = 150)
(h_sales_points_C : sales_points_C = 120) :
  (sample_size * sales_points_B / total_sales_points) = 30 := 
by
  sorry

end stratified_sampling_city_B_l215_21570


namespace sales_this_month_l215_21566

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l215_21566


namespace maximum_value_of_piecewise_function_l215_21518

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function_l215_21518


namespace length_of_chord_EF_l215_21561

noncomputable def chord_length (theta_1 theta_2 : ℝ) : ℝ :=
  let x_1 := 2 * Real.cos theta_1
  let y_1 := Real.sin theta_1
  let x_2 := 2 * Real.cos theta_2
  let y_2 := Real.sin theta_2
  Real.sqrt ((x_2 - x_1)^2 + (y_2 - y_1)^2)

theorem length_of_chord_EF :
  ∀ (theta_1 theta_2 : ℝ), 
  (2 * Real.cos theta_1) + (Real.sin theta_1) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_2) + (Real.sin theta_2) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_1)^2 + 4 * (Real.sin theta_1)^2 = 4 →
  (2 * Real.cos theta_2)^2 + 4 * (Real.sin theta_2)^2 = 4 →
  chord_length theta_1 theta_2 = 8 / 5 :=
by
  intros theta_1 theta_2 h1 h2 h3 h4
  sorry

end length_of_chord_EF_l215_21561


namespace fraction_married_men_l215_21553

-- Define the problem conditions
def num_faculty : ℕ := 100
def women_perc : ℕ := 60
def married_perc : ℕ := 60
def single_men_perc : ℚ := 3/4

-- We need to calculate the fraction of men who are married.
theorem fraction_married_men :
  (60 : ℚ) / 100 = women_perc / num_faculty →
  (60 : ℚ) / 100 = married_perc / num_faculty →
  (3/4 : ℚ) = single_men_perc →
  ∃ (fraction : ℚ), fraction = 1/4 :=
by
  intro h1 h2 h3
  sorry

end fraction_married_men_l215_21553


namespace probability_A_and_B_selected_l215_21534

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l215_21534


namespace probability_of_specific_event_l215_21541

noncomputable def adam_probability := 1 / 5
noncomputable def beth_probability := 2 / 9
noncomputable def jack_probability := 1 / 6
noncomputable def jill_probability := 1 / 7
noncomputable def sandy_probability := 1 / 8

theorem probability_of_specific_event :
  (1 - adam_probability) * beth_probability * (1 - jack_probability) * jill_probability * sandy_probability = 1 / 378 := by
  sorry

end probability_of_specific_event_l215_21541


namespace reaction_produces_correct_moles_l215_21573

-- Define the variables and constants
def moles_CO2 := 2
def moles_H2O := 2
def moles_H2CO3 := moles_CO2 -- based on the balanced reaction CO2 + H2O → H2CO3

-- The theorem we need to prove
theorem reaction_produces_correct_moles :
  moles_H2CO3 = 2 :=
by
  -- Mathematical reasoning goes here
  sorry

end reaction_produces_correct_moles_l215_21573


namespace binom_14_11_l215_21580

open Nat

theorem binom_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end binom_14_11_l215_21580


namespace cost_pants_shirt_l215_21552

variable (P S C : ℝ)

theorem cost_pants_shirt (h1 : P + C = 244) (h2 : C = 5 * S) (h3 : C = 180) : P + S = 100 := by
  sorry

end cost_pants_shirt_l215_21552


namespace trapezoid_area_l215_21584

theorem trapezoid_area (x : ℝ) :
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  area = 9 * x^2 / 2 :=
by
  -- Definitions based on conditions
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  -- Proof of the theorem, currently omitted
  sorry

end trapezoid_area_l215_21584


namespace initial_percentage_reduction_l215_21546

theorem initial_percentage_reduction
  (x: ℕ)
  (h1: ∀ P: ℝ, P * (1 - x / 100) * 0.85 * 1.5686274509803921 = P) :
  x = 25 :=
by
  sorry

end initial_percentage_reduction_l215_21546


namespace number_of_people_l215_21569

theorem number_of_people (n k : ℕ) (h₁ : k * n * (n - 1) = 440) : n = 11 :=
sorry

end number_of_people_l215_21569


namespace scientific_notation_of_150000000000_l215_21555

theorem scientific_notation_of_150000000000 :
  150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_of_150000000000_l215_21555


namespace final_weights_are_correct_l215_21576

-- Definitions of initial weights and reduction percentages per week
def initial_weight_A : ℝ := 300
def initial_weight_B : ℝ := 450
def initial_weight_C : ℝ := 600
def initial_weight_D : ℝ := 750

def reduction_A_week1 : ℝ := 0.20 * initial_weight_A
def reduction_B_week1 : ℝ := 0.15 * initial_weight_B
def reduction_C_week1 : ℝ := 0.30 * initial_weight_C
def reduction_D_week1 : ℝ := 0.25 * initial_weight_D

def weight_A_after_week1 : ℝ := initial_weight_A - reduction_A_week1
def weight_B_after_week1 : ℝ := initial_weight_B - reduction_B_week1
def weight_C_after_week1 : ℝ := initial_weight_C - reduction_C_week1
def weight_D_after_week1 : ℝ := initial_weight_D - reduction_D_week1

def reduction_A_week2 : ℝ := 0.25 * weight_A_after_week1
def reduction_B_week2 : ℝ := 0.30 * weight_B_after_week1
def reduction_C_week2 : ℝ := 0.10 * weight_C_after_week1
def reduction_D_week2 : ℝ := 0.20 * weight_D_after_week1

def weight_A_after_week2 : ℝ := weight_A_after_week1 - reduction_A_week2
def weight_B_after_week2 : ℝ := weight_B_after_week1 - reduction_B_week2
def weight_C_after_week2 : ℝ := weight_C_after_week1 - reduction_C_week2
def weight_D_after_week2 : ℝ := weight_D_after_week1 - reduction_D_week2

def reduction_A_week3 : ℝ := 0.15 * weight_A_after_week2
def reduction_B_week3 : ℝ := 0.10 * weight_B_after_week2
def reduction_C_week3 : ℝ := 0.20 * weight_C_after_week2
def reduction_D_week3 : ℝ := 0.30 * weight_D_after_week2

def weight_A_after_week3 : ℝ := weight_A_after_week2 - reduction_A_week3
def weight_B_after_week3 : ℝ := weight_B_after_week2 - reduction_B_week3
def weight_C_after_week3 : ℝ := weight_C_after_week2 - reduction_C_week3
def weight_D_after_week3 : ℝ := weight_D_after_week2 - reduction_D_week3

def reduction_A_week4 : ℝ := 0.10 * weight_A_after_week3
def reduction_B_week4 : ℝ := 0.20 * weight_B_after_week3
def reduction_C_week4 : ℝ := 0.25 * weight_C_after_week3
def reduction_D_week4 : ℝ := 0.15 * weight_D_after_week3

def final_weight_A : ℝ := weight_A_after_week3 - reduction_A_week4
def final_weight_B : ℝ := weight_B_after_week3 - reduction_B_week4
def final_weight_C : ℝ := weight_C_after_week3 - reduction_C_week4
def final_weight_D : ℝ := weight_D_after_week3 - reduction_D_week4

theorem final_weights_are_correct :
  final_weight_A = 137.7 ∧ 
  final_weight_B = 192.78 ∧ 
  final_weight_C = 226.8 ∧ 
  final_weight_D = 267.75 :=
by
  unfold final_weight_A final_weight_B final_weight_C final_weight_D
  sorry

end final_weights_are_correct_l215_21576


namespace no_valid_pairs_l215_21500

theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b) :=
by {
  sorry
}

end no_valid_pairs_l215_21500


namespace variance_of_data_set_l215_21507

open Real

def dataSet := [11, 12, 15, 18, 13, 15]

theorem variance_of_data_set :
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  variance = 16 / 3 :=
by
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  have h : mean = 14 := sorry
  have h_variance : variance = 16 / 3 := sorry
  exact h_variance

end variance_of_data_set_l215_21507


namespace eva_marks_difference_l215_21520

theorem eva_marks_difference 
    (m2 : ℕ) (a2 : ℕ) (s2 : ℕ) (total_marks : ℕ)
    (h_m2 : m2 = 80) (h_a2 : a2 = 90) (h_s2 : s2 = 90) (h_total_marks : total_marks = 485)
    (m1 a1 s1 : ℕ)
    (h_m1 : m1 = m2 + 10)
    (h_a1 : a1 = a2 - 15)
    (h_s1 : s1 = s2 - 1 / 3 * s2)
    (total_semesters : ℕ)
    (h_total_semesters : total_semesters = m1 + a1 + s1 + m2 + a2 + s2)
    : m1 = m2 + 10 := by
  sorry

end eva_marks_difference_l215_21520


namespace incorrect_expression_D_l215_21517

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end incorrect_expression_D_l215_21517


namespace prism_volume_l215_21515

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 :=
by
  sorry

end prism_volume_l215_21515


namespace no_nat_triplet_exists_l215_21509

theorem no_nat_triplet_exists (x y z : ℕ) : ¬ (x ^ 2 + y ^ 2 = 7 * z ^ 2) := 
sorry

end no_nat_triplet_exists_l215_21509


namespace equation_of_midpoint_trajectory_l215_21502

theorem equation_of_midpoint_trajectory
  (M : ℝ × ℝ)
  (hM : M.1 ^ 2 + M.2 ^ 2 = 1)
  (N : ℝ × ℝ := (2, 0))
  (P : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) :
  (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 / 4 := 
sorry

end equation_of_midpoint_trajectory_l215_21502


namespace identify_person_l215_21530

variable (Person : Type) (Tweedledum Tralyalya : Person)
variable (has_black_card : Person → Prop)
variable (statement_true : Person → Prop)
variable (statement_made_by : Person)

-- Condition: The statement made: "Either I am Tweedledum, or I have a card of a black suit in my pocket."
def statement (p : Person) : Prop := p = Tweedledum ∨ has_black_card p

-- Condition: Anyone with a black card making a true statement is not possible.
axiom black_card_truth_contradiction : ∀ p : Person, has_black_card p → ¬ statement_true p

theorem identify_person :
statement_made_by = Tralyalya ∧ ¬ has_black_card statement_made_by :=
by
  sorry

end identify_person_l215_21530


namespace product_zero_when_a_is_2_l215_21582

theorem product_zero_when_a_is_2 : 
  ∀ (a : ℤ), a = 2 → (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  intros a ha
  sorry

end product_zero_when_a_is_2_l215_21582


namespace inequality_system_solution_exists_l215_21513

theorem inequality_system_solution_exists (a : ℝ) : (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := 
sorry

end inequality_system_solution_exists_l215_21513


namespace stratified_sampling_l215_21524

theorem stratified_sampling (N : ℕ) (r1 r2 r3 : ℕ) (sample_size : ℕ) 
  (ratio_given : r1 = 5 ∧ r2 = 2 ∧ r3 = 3) 
  (total_sample_size : sample_size = 200) :
  sample_size * r3 / (r1 + r2 + r3) = 60 := 
by
  sorry

end stratified_sampling_l215_21524


namespace find_X_sum_coordinates_l215_21533

/- Define points and their coordinates -/
variables (X Y Z : ℝ × ℝ)
variable  (XY XZ ZY : ℝ)
variable  (k : ℝ)
variable  (hxz : XZ = (3/4) * XY)
variable  (hzy : ZY = (1/4) * XY)
variable  (hy : Y = (2, 9))
variable  (hz : Z = (1, 5))

/-- Lean 4 statement for the proof problem -/
theorem find_X_sum_coordinates :
  (Y.1 = 2) ∧ (Y.2 = 9) ∧ (Z.1 = 1) ∧ (Z.2 = 5) ∧
  XZ = (3/4) * XY ∧ ZY = (1/4) * XY →
  (X.1 + X.2) = -9 := 
by
  sorry

end find_X_sum_coordinates_l215_21533


namespace cos_probability_ge_one_half_in_range_l215_21593

theorem cos_probability_ge_one_half_in_range :
  let interval_length := (Real.pi / 2) - (- (Real.pi / 2))
  let favorable_length := (Real.pi / 3) - (- (Real.pi / 3))
  (favorable_length / interval_length) = (2 / 3) := by
  sorry

end cos_probability_ge_one_half_in_range_l215_21593


namespace inequality_x_y_l215_21564

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l215_21564


namespace prime_factor_of_reversed_difference_l215_21532

theorem prime_factor_of_reversed_difference (A B C : ℕ) (hA : A ≠ C) (hA_d : 1 ≤ A ∧ A ≤ 9) (hB_d : 0 ≤ B ∧ B ≤ 9) (hC_d : 1 ≤ C ∧ C ≤ 9) :
  ∃ p, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 11 := 
by
  sorry

end prime_factor_of_reversed_difference_l215_21532


namespace remainder_17_pow_63_mod_7_l215_21568

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l215_21568


namespace simplify_fraction_l215_21572

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (10 * x * y^2) / (5 * x * y) = 2 * y := 
by
  sorry

end simplify_fraction_l215_21572


namespace necessary_but_not_sufficient_l215_21554

theorem necessary_but_not_sufficient (a b : ℕ) : 
  (a ≠ 1 ∨ b ≠ 2) → ¬ (a + b = 3) → ¬(a = 1 ∧ b = 2) ∧ ((a = 1 ∧ b = 2) → (a + b = 3)) := sorry

end necessary_but_not_sufficient_l215_21554


namespace min_value_y1_y2_sq_l215_21599

theorem min_value_y1_y2_sq (k : ℝ) (y1 y2 : ℝ) :
  ∃ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = -4 ∧ y1^2 + y2^2 = 8 :=
sorry

end min_value_y1_y2_sq_l215_21599


namespace pizzas_bought_l215_21523

def slices_per_pizza := 8
def total_slices := 16

theorem pizzas_bought : total_slices / slices_per_pizza = 2 := by
  sorry

end pizzas_bought_l215_21523


namespace total_seeds_in_watermelons_l215_21595

def slices1 : ℕ := 40
def seeds_per_slice1 : ℕ := 60
def slices2 : ℕ := 30
def seeds_per_slice2 : ℕ := 80
def slices3 : ℕ := 50
def seeds_per_slice3 : ℕ := 40

theorem total_seeds_in_watermelons :
  (slices1 * seeds_per_slice1) + (slices2 * seeds_per_slice2) + (slices3 * seeds_per_slice3) = 6800 := by
  sorry

end total_seeds_in_watermelons_l215_21595


namespace series_sum_eq_1_div_400_l215_21567

theorem series_sum_eq_1_div_400 :
  (∑' n : ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 := 
sorry

end series_sum_eq_1_div_400_l215_21567


namespace calculate_expr_equals_243_l215_21548

theorem calculate_expr_equals_243 :
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by
  sorry

end calculate_expr_equals_243_l215_21548


namespace solution_set_of_inequality_l215_21589

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x - 14 < 0} = {x : ℝ | -2 < x ∧ x < 7} :=
by
  sorry

end solution_set_of_inequality_l215_21589


namespace ball_hits_ground_time_l215_21510

theorem ball_hits_ground_time :
  ∀ t : ℝ, y = -20 * t^2 + 30 * t + 60 → y = 0 → t = (3 + Real.sqrt 57) / 4 := by
  sorry

end ball_hits_ground_time_l215_21510


namespace term_300_is_neg_8_l215_21551

noncomputable def geom_seq (a r : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => r * geom_seq a r n

-- First term and second term are given as conditions.
def a1 : ℤ := 8
def a2 : ℤ := -8

-- Define the common ratio based on the conditions
def r : ℤ := a2 / a1

-- Theorem stating the 300th term is -8
theorem term_300_is_neg_8 : geom_seq a1 r 299 = -8 :=
by
  have h_r : r = -1 := by
    rw [r, a2, a1]
    norm_num
  rw [h_r]
  sorry

end term_300_is_neg_8_l215_21551


namespace max_gcd_a_is_25_l215_21535

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 100 + n^2 + 2 * n

-- Define the gcd function
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Define the theorem to prove the maximum value of d_n as 25
theorem max_gcd_a_is_25 : ∃ n : ℕ, d n = 25 := 
sorry

end max_gcd_a_is_25_l215_21535


namespace part1_solve_inequality_part2_range_of_a_l215_21598

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1_solve_inequality (x : ℝ) (h : -2 < x ∧ x < -2/3) :
    f x 1 > 1 :=
by
  sorry

theorem part2_range_of_a (h : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x (a : ℝ) > 0) :
    -5/2 < a ∧ a < -2 :=
by
  sorry

end part1_solve_inequality_part2_range_of_a_l215_21598


namespace ratio_w_y_l215_21505

open Real

theorem ratio_w_y (w x y z : ℝ) (h1 : w / x = 5 / 2) (h2 : y / z = 3 / 2) (h3 : z / x = 1 / 4) : w / y = 20 / 3 :=
by
  sorry

end ratio_w_y_l215_21505


namespace like_terms_to_exponents_matching_l215_21526

theorem like_terms_to_exponents_matching (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : m^n = 27 := by
  sorry

end like_terms_to_exponents_matching_l215_21526


namespace binomial_arithmetic_sequence_iff_l215_21581

open Nat

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k 

-- Conditions
def is_arithmetic_sequence (n k : ℕ) : Prop :=
  binomial n (k-1) - 2 * binomial n k + binomial n (k+1) = 0

-- Statement to prove
theorem binomial_arithmetic_sequence_iff (u : ℕ) (u_gt2 : u > 2) :
  ∃ (n k : ℕ), (n = u^2 - 2) ∧ (k = binomial u 2 - 1 ∨ k = binomial (u+1) 2 - 1) 
  ↔ is_arithmetic_sequence n k := 
sorry

end binomial_arithmetic_sequence_iff_l215_21581


namespace values_of_x_and_y_l215_21575

theorem values_of_x_and_y (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) : x < -2 ∧ y < -1 :=
by
  -- Proof goes here
  sorry

end values_of_x_and_y_l215_21575


namespace sandy_money_l215_21562

theorem sandy_money (x : ℝ) (h : 0.70 * x = 210) : x = 300 := by
sorry

end sandy_money_l215_21562


namespace trapezium_height_l215_21549

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l215_21549


namespace multiples_of_4_in_sequence_l215_21539

-- Define the arithmetic sequence terms
def nth_term (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the conditions
def cond_1 : ℤ := 200 -- first term
def cond_2 : ℤ := -6 -- common difference
def smallest_term : ℤ := 2

-- Define the count of terms function
def num_terms (a d min : ℤ) : ℤ := (a - min) / -d + 1

-- The total number of terms in the sequence
def total_terms : ℤ := num_terms cond_1 cond_2 smallest_term

-- Define a function to get the ith term that is a multiple of 4
def ith_multiple_of_4 (n : ℤ) : ℤ := cond_1 + 18 * (n - 1)

-- Define the count of multiples of 4 within the given number of terms
def count_multiples_of_4 (total : ℤ) : ℤ := (total / 3) + 1

-- Final theorem statement
theorem multiples_of_4_in_sequence : count_multiples_of_4 total_terms = 12 := sorry

end multiples_of_4_in_sequence_l215_21539


namespace find_k_l215_21525

theorem find_k (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2) = (B.1 + B.2) / 2 ∧ (A.1^2 + A.2^2 - 6 * A.1 - 4 * A.2 + 9 = 0) ∧ (B.1^2 + B.2^2 - 6 * B.1 - 4 * B.2 + 9 = 0)
     ∧ dist A B = 2 * Real.sqrt 3)
  (h3 : ∀ x y : ℝ, y = k * x + 3 → (x^2 + y^2 - 6 * x - 4 * y + 9) = 0)
  : k = 1 := sorry

end find_k_l215_21525


namespace ratio_of_perimeters_is_one_l215_21592

-- Definitions based on the given conditions
def original_rectangle : ℝ × ℝ := (6, 8)
def folded_rectangle : ℝ × ℝ := (3, 8)
def small_rectangle : ℝ × ℝ := (3, 4)
def large_rectangle : ℝ × ℝ := (3, 4)

-- The perimeter function for a rectangle given its dimensions (length, width)
def perimeter (r : ℝ × ℝ) : ℝ := 2 * (r.1 + r.2)

-- The main theorem to prove
theorem ratio_of_perimeters_is_one : 
  perimeter small_rectangle / perimeter large_rectangle = 1 :=
by
  sorry

end ratio_of_perimeters_is_one_l215_21592


namespace ratio_of_A_to_B_is_4_l215_21565

noncomputable def A_share : ℝ := 360
noncomputable def B_share : ℝ := 90
noncomputable def ratio_A_B : ℝ := A_share / B_share

theorem ratio_of_A_to_B_is_4 : ratio_A_B = 4 :=
by
  -- This is the proof that we are skipping
  sorry

end ratio_of_A_to_B_is_4_l215_21565


namespace min_value_inequality_l215_21529

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / x + 1 / y) * (4 * x + y) ≥ 9 ∧ ((1 / x + 1 / y) * (4 * x + y) = 9 ↔ y / x = 2) :=
by
  sorry

end min_value_inequality_l215_21529


namespace some_students_are_not_club_members_l215_21519

variable (U : Type) -- U represents the universe of students and club members
variables (Student ClubMember StudyLate : U → Prop)

-- Conditions derived from the problem
axiom h1 : ∃ s, Student s ∧ ¬ StudyLate s -- Some students do not study late
axiom h2 : ∀ c, ClubMember c → StudyLate c -- All club members study late

theorem some_students_are_not_club_members :
  ∃ s, Student s ∧ ¬ ClubMember s :=
by
  sorry

end some_students_are_not_club_members_l215_21519


namespace solve_equation_l215_21594

theorem solve_equation (x : ℝ) : 
  (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = -3 - Real.sqrt 6) ↔ 
  (x^4 / (2 * x + 1) + x^2 = 6 * (2 * x + 1)) := by
  sorry

end solve_equation_l215_21594


namespace annie_diorama_time_l215_21544

theorem annie_diorama_time (P B : ℕ) (h1 : B = 3 * P - 5) (h2 : B = 49) : P + B = 67 :=
sorry

end annie_diorama_time_l215_21544


namespace cost_to_treat_dog_l215_21596

variable (D : ℕ)
variable (cost_cat : ℕ := 40)
variable (num_dogs : ℕ := 20)
variable (num_cats : ℕ := 60)
variable (total_paid : ℕ := 3600)

theorem cost_to_treat_dog : 20 * D + 60 * cost_cat = total_paid → D = 60 := by
  intros h
  -- Proof goes here
  sorry

end cost_to_treat_dog_l215_21596


namespace complex_power_sum_l215_21578

noncomputable def z : ℂ := sorry

theorem complex_power_sum (hz : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 :=
sorry

end complex_power_sum_l215_21578


namespace gcd_of_polynomial_l215_21531

theorem gcd_of_polynomial (b : ℕ) (hb : b % 780 = 0) : Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 65) b = 65 := by
  sorry

end gcd_of_polynomial_l215_21531


namespace relation_of_a_and_b_l215_21556

theorem relation_of_a_and_b (a b : ℝ) (h : 2^a + Real.log a / Real.log 2 = 4^b + 2 * Real.log b / Real.log 4) : a < 2 * b :=
sorry

end relation_of_a_and_b_l215_21556


namespace numValidRoutesJackToJill_l215_21590

noncomputable def numPaths (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

theorem numValidRoutesJackToJill : 
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  totalRoutes - pathsViaDanger = 32 :=
by
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  show totalRoutes - pathsViaDanger = 32
  sorry

end numValidRoutesJackToJill_l215_21590


namespace max_AMC_expression_l215_21536

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 24) :
  A * M * C + A * M + M * C + C * A ≤ 704 :=
sorry

end max_AMC_expression_l215_21536


namespace problem_statement_l215_21571

noncomputable def f (x : ℝ) := 2 * x + 3
noncomputable def g (x : ℝ) := 3 * x - 2

theorem problem_statement : (f (g (f 3)) / g (f (g 3))) = 53 / 49 :=
by
  -- The proof is not provided as requested.
  sorry

end problem_statement_l215_21571


namespace mean_of_other_two_numbers_l215_21579

-- Definitions based on conditions in the problem.
def mean_of_four (numbers : List ℕ) : ℝ := 2187.25
def sum_of_numbers : ℕ := 1924 + 2057 + 2170 + 2229 + 2301 + 2365
def sum_of_four_numbers : ℝ := 4 * 2187.25
def sum_of_two_numbers := sum_of_numbers - sum_of_four_numbers

-- Theorem to assert the mean of the other two numbers.
theorem mean_of_other_two_numbers : (4297 / 2) = 2148.5 := by
  sorry

end mean_of_other_two_numbers_l215_21579


namespace find_z_l215_21550

noncomputable def z := {z : ℂ | ∃ i : ℂ, i^2 = -1 ∧ i * z = i - 1}

theorem find_z (i : ℂ) (hi : i^2 = -1) : ∃ z : ℂ, i * z = i - 1 ∧ z = 1 + i := by
  use 1 + i
  sorry

end find_z_l215_21550


namespace find_b_l215_21547

def perpendicular_vectors (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_b (b : ℝ) :
  perpendicular_vectors ⟨-5, 11⟩ ⟨b, 3⟩ →
  b = 33 / 5 :=
by
  sorry

end find_b_l215_21547


namespace problem_x_sq_plus_y_sq_l215_21542

variables {x y : ℝ}

theorem problem_x_sq_plus_y_sq (h₁ : x - y = 12) (h₂ : x * y = 9) : x^2 + y^2 = 162 := 
sorry

end problem_x_sq_plus_y_sq_l215_21542


namespace tan_alpha_eq_one_third_l215_21537

variable (α : ℝ)

theorem tan_alpha_eq_one_third (h : Real.tan (α + Real.pi / 4) = 2) : Real.tan α = 1 / 3 :=
sorry

end tan_alpha_eq_one_third_l215_21537


namespace polynomial_decomposition_l215_21559

theorem polynomial_decomposition :
  (x^3 - 2*x^2 + 3*x + 5) = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 :=
by sorry

end polynomial_decomposition_l215_21559


namespace smallest_b_value_l215_21560

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l215_21560


namespace area_enclosed_is_one_third_l215_21597

theorem area_enclosed_is_one_third :
  ∫ x in (0:ℝ)..1, (x^(1/2) - x^2 : ℝ) = (1/3 : ℝ) :=
by
  sorry

end area_enclosed_is_one_third_l215_21597


namespace odds_of_picking_blue_marble_l215_21543

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end odds_of_picking_blue_marble_l215_21543


namespace min_value_a_plus_9b_l215_21585

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : a + 9 * b ≥ 16 :=
  sorry

end min_value_a_plus_9b_l215_21585


namespace geometric_sequence_sum_l215_21521

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q
def cond1 := a 0 + a 1 = 3
def cond2 := a 2 + a 3 = 12
def cond3 := is_geometric_sequence a

theorem geometric_sequence_sum :
  cond1 a →
  cond2 a →
  cond3 a q →
  a 4 + a 5 = 48 :=
by
  intro h1 h2 h3
  sorry

end geometric_sequence_sum_l215_21521


namespace value_of_x_l215_21563

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l215_21563


namespace simplify_expression_l215_21528

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 :=
by
  -- The proof is omitted, so use sorry to skip it
  sorry

end simplify_expression_l215_21528


namespace bird_population_in_1997_l215_21508

theorem bird_population_in_1997 
  (k : ℝ)
  (pop_1995 pop_1996 pop_1998 : ℝ)
  (h1 : pop_1995 = 45)
  (h2 : pop_1996 = 70)
  (h3 : pop_1998 = 145)
  (h4 : pop_1997 - pop_1995 = k * pop_1996)
  (h5 : pop_1998 - pop_1996 = k * pop_1997) : 
  pop_1997 = 105 :=
by
  sorry

end bird_population_in_1997_l215_21508


namespace xiao_ming_runs_distance_l215_21506

theorem xiao_ming_runs_distance 
  (num_trees : ℕ) 
  (first_tree : ℕ) 
  (last_tree : ℕ) 
  (distance_between_trees : ℕ) 
  (gap_count : ℕ) 
  (total_distance : ℕ)
  (h1 : num_trees = 200) 
  (h2 : first_tree = 1) 
  (h3 : last_tree = 200) 
  (h4 : distance_between_trees = 6) 
  (h5 : gap_count = last_tree - first_tree)
  (h6 : total_distance = gap_count * distance_between_trees) :
  total_distance = 1194 :=
sorry

end xiao_ming_runs_distance_l215_21506


namespace minimum_value_frac_l215_21587

theorem minimum_value_frac (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (2 / a) + (3 / b) ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end minimum_value_frac_l215_21587


namespace sequence_formula_l215_21574

theorem sequence_formula (a : ℕ → ℤ) (h0 : a 0 = 1) (h1 : a 1 = 5)
    (h_rec : ∀ n, n ≥ 2 → a n = (2 * (a (n - 1))^2 - 3 * (a (n - 1)) - 9) / (2 * a (n - 2))) :
  ∀ n, a n = 2^(n + 2) - 3 :=
by
  intros
  sorry

end sequence_formula_l215_21574


namespace inequality_solution_set_l215_21516

theorem inequality_solution_set (x : ℝ) : |x - 5| + |x + 3| ≤ 10 ↔ -4 ≤ x ∧ x ≤ 6 :=
by
  sorry

end inequality_solution_set_l215_21516


namespace inverse_proportion_k_value_l215_21591

theorem inverse_proportion_k_value (k : ℝ) (h₁ : k ≠ 0) (h₂ : (2, -1) ∈ {p : ℝ × ℝ | ∃ (k' : ℝ), k' = k ∧ p.snd = k' / p.fst}) :
  k = -2 := 
by
  sorry

end inverse_proportion_k_value_l215_21591


namespace tangent_line_through_external_point_l215_21558

theorem tangent_line_through_external_point (x y : ℝ) (h_circle : x^2 + y^2 = 1) (P : ℝ × ℝ) (h_P : P = (1, 2)) : 
  (∃ k : ℝ, (y = 2 + k * (x - 1)) ∧ (x = 1 ∨ (3 * x - 4 * y + 5 = 0))) :=
by
  sorry

end tangent_line_through_external_point_l215_21558


namespace line_in_slope_intercept_form_l215_21511

variable (x y : ℝ)

def line_eq (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 1) = 0

theorem line_in_slope_intercept_form (x y : ℝ) (h: line_eq x y) :
  y = (3 / 4) * x - 5 / 2 :=
sorry

end line_in_slope_intercept_form_l215_21511


namespace incorrect_statement_D_l215_21503

theorem incorrect_statement_D
  (passes_through_center : ∀ (x_vals y_vals : List ℝ), ∃ (regression_line : ℝ → ℝ), 
    regression_line (x_vals.sum / x_vals.length) = (y_vals.sum / y_vals.length))
  (higher_r2_better_fit : ∀ (r2 : ℝ), r2 > 0 → ∃ (residual_sum_squares : ℝ), residual_sum_squares < (1 - r2))
  (slope_interpretation : ∀ (x : ℝ), (0.2 * x + 0.8) - (0.2 * (x - 1) + 0.8) = 0.2)
  (chi_squared_k2 : ∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), (k > 0) → 
    ∃ (confidence : ℝ), confidence > 0) :
  ¬(∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), k > 0 → 
    ∃ (confidence : ℝ), confidence < 0) :=
by
  sorry

end incorrect_statement_D_l215_21503


namespace necessary_but_not_sufficient_condition_l215_21538

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 2 → x^2 - a > 0) → (a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l215_21538


namespace maximum_distance_between_balls_l215_21501

theorem maximum_distance_between_balls 
  (a b c : ℝ) 
  (aluminum_ball_heavier : true) -- Implicitly understood property rather than used in calculation directly
  (wood_ball_lighter : true) -- Implicitly understood property rather than used in calculation directly
  : ∃ d : ℝ, d = Real.sqrt (a^2 + b^2 + c^2) → d = Real.sqrt (3^2 + 4^2 + 2^2) := 
by
  use Real.sqrt (3^2 + 4^2 + 2^2)
  sorry

end maximum_distance_between_balls_l215_21501
