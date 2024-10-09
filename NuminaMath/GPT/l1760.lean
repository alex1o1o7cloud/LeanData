import Mathlib

namespace abs_eq_neg_of_le_zero_l1760_176081

theorem abs_eq_neg_of_le_zero (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_of_le_zero_l1760_176081


namespace parking_spots_full_iff_num_sequences_l1760_176079

noncomputable def num_parking_sequences (n : ℕ) : ℕ :=
  (n + 1) ^ (n - 1)

-- Statement of the theorem
theorem parking_spots_full_iff_num_sequences (n : ℕ) :
  ∀ (a : ℕ → ℕ), (∀ (i : ℕ), i < n → a i ≤ n) → 
  (∀ (j : ℕ), j ≤ n → (∃ i, i < n ∧ a i = j)) ↔ 
  num_parking_sequences n = (n + 1) ^ (n - 1) :=
sorry

end parking_spots_full_iff_num_sequences_l1760_176079


namespace value_at_one_positive_l1760_176060

-- Define the conditions
variable {f : ℝ → ℝ} 

-- f is a monotonically increasing function
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement: proving that f(1) > 0
theorem value_at_one_positive (h1 : monotone_increasing f) (h2 : odd_function f) : f 1 > 0 :=
sorry

end value_at_one_positive_l1760_176060


namespace circle_area_from_tangency_conditions_l1760_176049

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - 20 * y^2 = 24

-- Tangency to the x-axis implies the circle's lowest point touches the x-axis
def tangent_to_x_axis (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ r y₀, circle 0 y₀ ∧ y₀ = r

-- The circle is given as having tangency conditions to derive from
theorem circle_area_from_tangency_conditions (circle : ℝ → ℝ → Prop) :
  (∀ x y, circle x y → (x = 0 ∨ hyperbola x y)) →
  tangent_to_x_axis circle →
  ∃ area, area = 504 * Real.pi :=
by
  sorry

end circle_area_from_tangency_conditions_l1760_176049


namespace expected_value_l1760_176042

theorem expected_value (p1 p2 p3 p4 p5 p6 : ℕ) (hp1 : p1 = 1) (hp2 : p2 = 5) (hp3 : p3 = 10) 
(hp4 : p4 = 25) (hp5 : p5 = 50) (hp6 : p6 = 100) :
  (p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + p5 / 2 + p6 / 2 : ℝ) = 95.5 := by
  sorry

end expected_value_l1760_176042


namespace son_age_is_18_l1760_176030

theorem son_age_is_18
  (S F : ℕ)
  (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 18 :=
by sorry

end son_age_is_18_l1760_176030


namespace scientific_notation_of_2102000_l1760_176000

theorem scientific_notation_of_2102000 : ∃ (x : ℝ) (n : ℤ), 2102000 = x * 10 ^ n ∧ x = 2.102 ∧ n = 6 :=
by
  sorry

end scientific_notation_of_2102000_l1760_176000


namespace inequality_proof_l1760_176048

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c :=
by
  sorry

end inequality_proof_l1760_176048


namespace horses_added_l1760_176074

-- Define the problem parameters and conditions.
def horses_initial := 3
def water_per_horse_drinking_per_day := 5
def water_per_horse_bathing_per_day := 2
def days := 28
def total_water := 1568

-- Define the assumption based on the given problem.
def total_water_per_horse_per_day := water_per_horse_drinking_per_day + water_per_horse_bathing_per_day
def total_water_initial_horses := horses_initial * total_water_per_horse_per_day * days
def water_for_new_horses := total_water - total_water_initial_horses
def daily_water_consumption_new_horses := water_for_new_horses / days
def number_of_new_horses := daily_water_consumption_new_horses / total_water_per_horse_per_day

-- The theorem to prove number of horses added.
theorem horses_added : number_of_new_horses = 5 := 
  by {
    -- This is where you would put the proof steps.
    sorry -- skipping the proof for now
  }

end horses_added_l1760_176074


namespace dog_probability_l1760_176098

def prob_machine_A_transforms_cat_to_dog : ℚ := 1 / 3
def prob_machine_B_transforms_cat_to_dog : ℚ := 2 / 5
def prob_machine_C_transforms_cat_to_dog : ℚ := 1 / 4

def prob_cat_remains_after_A : ℚ := 1 - prob_machine_A_transforms_cat_to_dog
def prob_cat_remains_after_B : ℚ := 1 - prob_machine_B_transforms_cat_to_dog
def prob_cat_remains_after_C : ℚ := 1 - prob_machine_C_transforms_cat_to_dog

def prob_cat_remains : ℚ := prob_cat_remains_after_A * prob_cat_remains_after_B * prob_cat_remains_after_C

def prob_dog_out_of_C : ℚ := 1 - prob_cat_remains

theorem dog_probability : prob_dog_out_of_C = 7 / 10 := by
  -- Proof goes here
  sorry

end dog_probability_l1760_176098


namespace fourth_power_nested_sqrt_l1760_176013

noncomputable def nested_sqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt : nested_sqrt ^ 4 = 16 := by
  sorry

end fourth_power_nested_sqrt_l1760_176013


namespace tree_growth_l1760_176005

theorem tree_growth (x : ℝ) : 4*x + 4*2*x + 4*2 + 4*3 = 32 → x = 1 :=
by
  intro h
  sorry

end tree_growth_l1760_176005


namespace max_a2_b2_c2_d2_l1760_176025

-- Define the conditions for a, b, c, d
variables (a b c d : ℝ) 

-- Define the hypotheses from the problem
variables (h₁ : a + b = 17)
variables (h₂ : ab + c + d = 94)
variables (h₃ : ad + bc = 195)
variables (h₄ : cd = 120)

-- Define the final statement to be proved
theorem max_a2_b2_c2_d2 : ∃ (a b c d : ℝ), a + b = 17 ∧ ab + c + d = 94 ∧ ad + bc = 195 ∧ cd = 120 ∧ (a^2 + b^2 + c^2 + d^2) = 918 :=
by sorry

end max_a2_b2_c2_d2_l1760_176025


namespace farmer_shipped_67_dozens_l1760_176073

def pomelos_in_box (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 20 else if box_type = "large" then 30 else 0

def total_pomelos_last_week : ℕ := 360

def boxes_this_week (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 8 else if box_type = "large" then 7 else 0

def damage_boxes (box_type : String) : ℕ :=
  if box_type = "small" then 3 else if box_type = "medium" then 2 else if box_type = "large" then 2 else 0

def loss_percentage (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 15 else if box_type = "large" then 20 else 0

def total_pomelos_shipped_this_week : ℕ :=
  (boxes_this_week "small") * (pomelos_in_box "small") +
  (boxes_this_week "medium") * (pomelos_in_box "medium") +
  (boxes_this_week "large") * (pomelos_in_box "large")

def total_pomelos_lost_this_week : ℕ :=
  (damage_boxes "small") * (pomelos_in_box "small") * (loss_percentage "small") / 100 +
  (damage_boxes "medium") * (pomelos_in_box "medium") * (loss_percentage "medium") / 100 +
  (damage_boxes "large") * (pomelos_in_box "large") * (loss_percentage "large") / 100

def total_pomelos_shipped_successfully_this_week : ℕ :=
  total_pomelos_shipped_this_week - total_pomelos_lost_this_week

def total_pomelos_for_both_weeks : ℕ :=
  total_pomelos_last_week + total_pomelos_shipped_successfully_this_week

def total_dozens_shipped : ℕ :=
  total_pomelos_for_both_weeks / 12

theorem farmer_shipped_67_dozens :
  total_dozens_shipped = 67 := 
by sorry

end farmer_shipped_67_dozens_l1760_176073


namespace passing_percentage_is_correct_l1760_176036

theorem passing_percentage_is_correct :
  ∀ (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ),
    marks_obtained = 59 →
    marks_failed_by = 40 →
    max_marks = 300 →
    (marks_obtained + marks_failed_by) / max_marks * 100 = 33 :=
by
  intros marks_obtained marks_failed_by max_marks h1 h2 h3
  sorry

end passing_percentage_is_correct_l1760_176036


namespace cat_finishes_food_on_next_wednesday_l1760_176014

def cat_food_consumption_per_day : ℚ :=
  (1 / 4) + (1 / 6)

def total_food_on_day (n : ℕ) : ℚ :=
  n * cat_food_consumption_per_day

def total_cans : ℚ := 8

theorem cat_finishes_food_on_next_wednesday :
  total_food_on_day 10 = total_cans := sorry

end cat_finishes_food_on_next_wednesday_l1760_176014


namespace circumference_of_tank_a_l1760_176044

def is_circumference_of_tank_a (h_A h_B C_B : ℝ) (V_A_eq : ℝ → Prop) : Prop :=
  ∃ (C_A : ℝ), 
    C_B = 10 ∧ 
    h_A = 10 ∧
    h_B = 7 ∧
    V_A_eq 0.7 ∧ 
    C_A = 7

theorem circumference_of_tank_a (h_A : ℝ) (h_B : ℝ) (C_B : ℝ) (V_A_eq : ℝ → Prop) : 
  is_circumference_of_tank_a h_A h_B C_B V_A_eq := 
by
  sorry

end circumference_of_tank_a_l1760_176044


namespace ratio_of_first_term_to_common_difference_l1760_176093

theorem ratio_of_first_term_to_common_difference 
  (a d : ℤ) 
  (h : 15 * a + 105 * d = 3 * (10 * a + 45 * d)) :
  a = -2 * d :=
by 
  sorry

end ratio_of_first_term_to_common_difference_l1760_176093


namespace third_altitude_is_less_than_15_l1760_176008

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (A : ℝ)

def triangle_area (side : ℝ) (height : ℝ) : ℝ := 0.5 * side * height

axiom ha_eq : ha = 10
axiom hb_eq : hb = 6

theorem third_altitude_is_less_than_15 : hc < 15 :=
sorry

end third_altitude_is_less_than_15_l1760_176008


namespace allison_upload_rate_l1760_176077

theorem allison_upload_rate (x : ℕ) (h1 : 15 * x + 30 * x = 450) : x = 10 :=
by
  sorry

end allison_upload_rate_l1760_176077


namespace sum_of_digits_of_greatest_prime_divisor_l1760_176015

-- Define the number 32767
def number := 32767

-- Find the greatest prime divisor of 32767
def greatest_prime_divisor : ℕ :=
  127

-- Prove the sum of the digits of the greatest prime divisor is 10
theorem sum_of_digits_of_greatest_prime_divisor (h : greatest_prime_divisor = 127) : (1 + 2 + 7) = 10 :=
  sorry

end sum_of_digits_of_greatest_prime_divisor_l1760_176015


namespace noah_sales_value_l1760_176068

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end noah_sales_value_l1760_176068


namespace smallest_four_digit_multiple_of_18_l1760_176019

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l1760_176019


namespace james_music_listening_hours_l1760_176009

theorem james_music_listening_hours (BPM : ℕ) (beats_per_week : ℕ) (hours_per_day : ℕ) 
  (h1 : BPM = 200) 
  (h2 : beats_per_week = 168000) 
  (h3 : hours_per_day * 200 * 60 * 7 = beats_per_week) : 
  hours_per_day = 2 := 
by
  sorry

end james_music_listening_hours_l1760_176009


namespace arithmetic_sequence_a4_a7_div2_eq_10_l1760_176078

theorem arithmetic_sequence_a4_a7_div2_eq_10 (a : ℕ → ℝ) (h : a 4 + a 6 = 20) : (a 3 + a 6) / 2 = 10 :=
  sorry

end arithmetic_sequence_a4_a7_div2_eq_10_l1760_176078


namespace total_length_of_pencil_l1760_176092

def purple := 3
def black := 2
def blue := 1
def total_length := purple + black + blue

theorem total_length_of_pencil : total_length = 6 := 
by 
  sorry -- proof not needed

end total_length_of_pencil_l1760_176092


namespace volunteer_assignment_correct_l1760_176001

def volunteerAssignment : ℕ := 5
def pavilions : ℕ := 4

def numberOfWays (volunteers pavilions : ℕ) : ℕ := 72 -- This is based on the provided correct answer.

theorem volunteer_assignment_correct : 
  numberOfWays volunteerAssignment pavilions = 72 := 
by
  sorry

end volunteer_assignment_correct_l1760_176001


namespace problem_statement_l1760_176099

def f (x : ℝ) : ℝ := x^3 + x^2 + 2

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement : odd_function f → f (-2) = -14 := by
  intro h
  sorry

end problem_statement_l1760_176099


namespace steves_initial_emails_l1760_176085

theorem steves_initial_emails (E : ℝ) (ht : E / 2 = (0.6 * E) + 120) : E = 400 :=
  by sorry

end steves_initial_emails_l1760_176085


namespace cans_restocked_after_second_day_l1760_176062

theorem cans_restocked_after_second_day :
  let initial_cans := 2000
  let first_day_taken := 500 
  let first_day_restock := 1500
  let second_day_taken := 1000 * 2
  let total_given_away := 2500
  let remaining_after_second_day_before_restock := initial_cans - first_day_taken + first_day_restock - second_day_taken
  (total_given_away - remaining_after_second_day_before_restock) = 1500 := 
by {
  sorry
}

end cans_restocked_after_second_day_l1760_176062


namespace real_root_exists_for_all_K_l1760_176058

theorem real_root_exists_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
sorry

end real_root_exists_for_all_K_l1760_176058


namespace system_of_equations_correct_l1760_176087

variable (x y : ℝ)

def correct_system_of_equations : Prop :=
  (3 / 60) * x + (5 / 60) * y = 1.2 ∧ x + y = 16

theorem system_of_equations_correct :
  correct_system_of_equations x y :=
sorry

end system_of_equations_correct_l1760_176087


namespace area_gray_region_in_terms_of_pi_l1760_176027

variable (r : ℝ)

theorem area_gray_region_in_terms_of_pi 
    (h1 : ∀ (r : ℝ), ∃ (outer_r : ℝ), outer_r = r + 3)
    (h2 : width_gray_region = 3)
    : ∃ (area_gray : ℝ), area_gray = π * (6 * r + 9) := 
sorry

end area_gray_region_in_terms_of_pi_l1760_176027


namespace must_divisor_of_a_l1760_176057

-- The statement
theorem must_divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18)
    (h2 : Nat.gcd b c = 45) (h3 : Nat.gcd c d = 60) (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
    5 ∣ a := 
sorry

end must_divisor_of_a_l1760_176057


namespace five_integers_sum_to_first_set_impossible_second_set_sum_l1760_176004

theorem five_integers_sum_to_first_set :
  ∃ (a b c d e : ℤ), 
    (a + b = 0) ∧ (a + c = 2) ∧ (b + c = 4) ∧ (a + d = 4) ∧ (b + d = 6) ∧
    (a + e = 8) ∧ (b + e = 9) ∧ (c + d = 11) ∧ (c + e = 13) ∧ (d + e = 15) ∧ 
    (a + b + c + d + e = 18) := 
sorry

theorem impossible_second_set_sum : 
  ¬∃ (a b c d e : ℤ), 
    (a + b = 12) ∧ (a + c = 13) ∧ (a + d = 14) ∧ (a + e = 15) ∧ (b + c = 16) ∧
    (b + d = 16) ∧ (b + e = 17) ∧ (c + d = 17) ∧ (c + e = 18) ∧ (d + e = 20) ∧
    (a + b + c + d + e = 39) :=
sorry

end five_integers_sum_to_first_set_impossible_second_set_sum_l1760_176004


namespace find_speed_A_l1760_176038

-- Defining the distance between the two stations as 155 km.
def distance := 155

-- Train A starts from station A at 7 a.m. and meets Train B at 11 a.m.
-- Therefore, Train A travels for 4 hours.
def time_A := 4

-- Train B starts from station B at 8 a.m. and meets Train A at 11 a.m.
-- Therefore, Train B travels for 3 hours.
def time_B := 3

-- Speed of Train B is given as 25 km/h.
def speed_B := 25

-- Condition that the total distance covered by both trains equals the distance between the two stations.
def meet_condition (v_A : ℕ) := (time_A * v_A) + (time_B * speed_B) = distance

-- The Lean theorem statement to be proved
theorem find_speed_A (v_A := 20) : meet_condition v_A :=
by
  -- Using 'sorrry' to skip the proof
  sorry

end find_speed_A_l1760_176038


namespace suitable_survey_is_D_l1760_176026

-- Define the surveys
def survey_A := "Survey on the viewing of the movie 'The Long Way Home' by middle school students in our city"
def survey_B := "Survey on the germination rate of a batch of rose seeds"
def survey_C := "Survey on the water quality of the Jialing River"
def survey_D := "Survey on the health codes of students during the epidemic"

-- Define what it means for a survey to be suitable for a comprehensive census
def suitable_for_census (survey : String) : Prop :=
  survey = survey_D

-- Define the main theorem statement
theorem suitable_survey_is_D : suitable_for_census survey_D :=
by
  -- We assume sorry here to skip the proof
  sorry

end suitable_survey_is_D_l1760_176026


namespace compute_value_3_std_devs_less_than_mean_l1760_176051

noncomputable def mean : ℝ := 15
noncomputable def std_dev : ℝ := 1.5
noncomputable def skewness : ℝ := 0.5
noncomputable def kurtosis : ℝ := 0.6

theorem compute_value_3_std_devs_less_than_mean : 
  ¬∃ (value : ℝ), value = mean - 3 * std_dev :=
sorry

end compute_value_3_std_devs_less_than_mean_l1760_176051


namespace translation_correct_l1760_176054

theorem translation_correct : 
  ∀ (x y : ℝ), (y = -(x-1)^2 + 3) → (x, y) = (0, 0) ↔ (x - 1, y - 3) = (0, 0) :=
by 
  sorry

end translation_correct_l1760_176054


namespace number_of_rocks_chosen_l1760_176052

open Classical

theorem number_of_rocks_chosen
  (total_rocks : ℕ)
  (slate_rocks : ℕ)
  (pumice_rocks : ℕ)
  (granite_rocks : ℕ)
  (probability_both_slate : ℚ) :
  total_rocks = 44 →
  slate_rocks = 14 →
  pumice_rocks = 20 →
  granite_rocks = 10 →
  probability_both_slate = (14 / 44) * (13 / 43) →
  2 = 2 := 
by {
  sorry
}

end number_of_rocks_chosen_l1760_176052


namespace k_h_5_eq_148_l1760_176084

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end k_h_5_eq_148_l1760_176084


namespace initial_ratio_of_milk_to_water_l1760_176050

variable (M W : ℕ) -- M represents the amount of milk, W represents the amount of water

theorem initial_ratio_of_milk_to_water (h1 : M + W = 45) (h2 : 8 * M = 9 * (W + 23)) :
  M / W = 4 :=
by
  sorry

end initial_ratio_of_milk_to_water_l1760_176050


namespace smallest_positive_value_of_expression_l1760_176017

theorem smallest_positive_value_of_expression :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^3 + b^3 + c^3 - 3 * a * b * c = 4) :=
by
  sorry

end smallest_positive_value_of_expression_l1760_176017


namespace find_length_of_train_l1760_176040

def speed_kmh : Real := 60
def time_to_cross_bridge : Real := 26.997840172786177
def length_of_bridge : Real := 340

noncomputable def speed_ms : Real := speed_kmh * (1000 / 3600)
noncomputable def total_distance : Real := speed_ms * time_to_cross_bridge
noncomputable def length_of_train : Real := total_distance - length_of_bridge

theorem find_length_of_train :
  length_of_train = 109.9640028797695 := 
sorry

end find_length_of_train_l1760_176040


namespace sum_of_factorization_constants_l1760_176059

theorem sum_of_factorization_constants (p q r s t : ℤ) (y : ℤ) :
  (512 * y ^ 3 + 27 = (p * y + q) * (r * y ^ 2 + s * y + t)) →
  p + q + r + s + t = 60 :=
by
  intro h
  sorry

end sum_of_factorization_constants_l1760_176059


namespace problem_statement_l1760_176063

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^7 - 6 * x^5 + 5 * x^3 - x = 0 :=
sorry

end problem_statement_l1760_176063


namespace wage_recovery_l1760_176037

theorem wage_recovery (W : ℝ) (h : W > 0) : (1 - 0.3) * W * (1 + 42.86 / 100) = W :=
by
  sorry

end wage_recovery_l1760_176037


namespace oshea_large_planters_l1760_176091

theorem oshea_large_planters {total_seeds small_planter_capacity num_small_planters large_planter_capacity : ℕ} 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : large_planter_capacity = 20) :
  (total_seeds - num_small_planters * small_planter_capacity) / large_planter_capacity = 4 :=
by
  sorry

end oshea_large_planters_l1760_176091


namespace cristine_lemons_left_l1760_176018

theorem cristine_lemons_left (initial_lemons : ℕ) (given_fraction : ℚ) (exchanged_lemons : ℕ) (h1 : initial_lemons = 12) (h2 : given_fraction = 1/4) (h3 : exchanged_lemons = 2) : 
  initial_lemons - initial_lemons * given_fraction - exchanged_lemons = 7 :=
by 
  sorry

end cristine_lemons_left_l1760_176018


namespace general_term_formula_l1760_176088

variable {a_n : ℕ → ℕ} -- Sequence {a_n}
variable {S_n : ℕ → ℕ} -- Sum of the first n terms

-- Condition given in the problem
def S_n_condition (n : ℕ) : ℕ :=
  2 * n^2 + n

theorem general_term_formula (n : ℕ) (h₀ : ∀ (n : ℕ), S_n n = 2 * n^2 + n) :
  a_n n = 4 * n - 1 :=
sorry

end general_term_formula_l1760_176088


namespace number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l1760_176007

def Jungkook_cards : Real := 0.8
def Yoongi_cards : Real := 0.5

theorem number_of_people_with_cards_leq_0_point_3 : 
  (Jungkook_cards <= 0.3 ∨ Yoongi_cards <= 0.3) = False := 
by 
  -- neither Jungkook nor Yoongi has number cards less than or equal to 0.3
  sorry

theorem number_of_people_with_cards_leq_0_point_3_count :
  (if (Jungkook_cards <= 0.3) then 1 else 0) + (if (Yoongi_cards <= 0.3) then 1 else 0) = 0 :=
by 
  -- calculate number of people with cards less than or equal to 0.3
  sorry

end number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l1760_176007


namespace find_locus_of_M_l1760_176020

variables {P : Type*} [MetricSpace P] 
variables (A B C M : P)

def on_perpendicular_bisector (A B M : P) : Prop := 
  dist A M = dist B M

def on_circle (center : P) (radius : ℝ) (M : P) : Prop := 
  dist center M = radius

def M_AB (A B M : P) : Prop :=
  (on_perpendicular_bisector A B M) ∨ (on_circle A (dist A B) M) ∨ (on_circle B (dist A B) M)

def M_BC (B C M : P) : Prop :=
  (on_perpendicular_bisector B C M) ∨ (on_circle B (dist B C) M) ∨ (on_circle C (dist B C) M)

theorem find_locus_of_M :
  {M : P | M_AB A B M} ∩ {M : P | M_BC B C M} = {M : P | M_AB A B M ∧ M_BC B C M} :=
by sorry

end find_locus_of_M_l1760_176020


namespace calculate_expected_value_of_S_l1760_176076

-- Define the problem context
variables (boys girls : ℕ)
variable (boy_girl_pair_at_start : Bool)

-- Define the expected value function
def expected_S (boys girls : ℕ) (boy_girl_pair_at_start : Bool) : ℕ :=
  if boy_girl_pair_at_start then 10 else sorry  -- we only consider the given scenario

-- The theorem to prove
theorem calculate_expected_value_of_S :
  expected_S 5 15 true = 10 :=
by
  -- proof needs to be filled in
  sorry

end calculate_expected_value_of_S_l1760_176076


namespace intersection_is_negative_real_l1760_176053

def setA : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1}
def setB : Set ℝ := {y : ℝ | ∃ x : ℝ, y = - x ^ 2}

theorem intersection_is_negative_real :
  setA ∩ setB = {y : ℝ | y ≤ 0} := 
sorry

end intersection_is_negative_real_l1760_176053


namespace red_pill_cost_l1760_176041

theorem red_pill_cost :
  ∃ (r : ℚ) (b : ℚ), (∀ (d : ℕ), d = 21 → 3 * r - 2 = 39) ∧
                      (1 ≤ d → r = b + 1) ∧
                      (21 * (r + 2 * b) = 819) → 
                      r = 41 / 3 :=
by sorry

end red_pill_cost_l1760_176041


namespace maria_first_stop_distance_is_280_l1760_176012

noncomputable def maria_travel_distance : ℝ := 560
noncomputable def first_stop_distance (x : ℝ) : ℝ := x
noncomputable def distance_after_first_stop (x : ℝ) : ℝ := maria_travel_distance - first_stop_distance x
noncomputable def second_stop_distance (x : ℝ) : ℝ := (1 / 4) * distance_after_first_stop x
noncomputable def remaining_distance : ℝ := 210

theorem maria_first_stop_distance_is_280 :
  ∃ x, first_stop_distance x = 280 ∧ second_stop_distance x + remaining_distance = distance_after_first_stop x := sorry

end maria_first_stop_distance_is_280_l1760_176012


namespace dart_not_land_in_circle_probability_l1760_176035

theorem dart_not_land_in_circle_probability :
  let side_length := 1
  let radius := side_length / 2
  let area_square := side_length * side_length
  let area_circle := π * radius * radius
  let prob_inside_circle := area_circle / area_square
  let prob_outside_circle := 1 - prob_inside_circle
  prob_outside_circle = 1 - (π / 4) :=
by
  sorry

end dart_not_land_in_circle_probability_l1760_176035


namespace integer_bases_not_divisible_by_5_l1760_176055

theorem integer_bases_not_divisible_by_5 :
  ∀ b ∈ ({3, 5, 7, 10, 12} : Set ℕ), (b - 1) ^ 2 % 5 ≠ 0 :=
by sorry

end integer_bases_not_divisible_by_5_l1760_176055


namespace factorial_division_l1760_176021

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l1760_176021


namespace complex_number_in_third_quadrant_l1760_176071

open Complex

noncomputable def complex_number : ℂ := (1 - 3 * I) / (1 + 2 * I)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_number_in_third_quadrant : in_third_quadrant complex_number :=
sorry

end complex_number_in_third_quadrant_l1760_176071


namespace functional_eq_solution_l1760_176022

theorem functional_eq_solution (f : ℤ → ℤ) (h : ∀ x y : ℤ, x ≠ 0 →
  x * f (2 * f y - x) + y^2 * f (2 * x - f y) = (f x ^ 2) / x + f (y * f y)) :
  (∀ x: ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end functional_eq_solution_l1760_176022


namespace dishonest_dealer_profit_percent_l1760_176090

theorem dishonest_dealer_profit_percent
  (C : ℝ) -- assumed cost price for 1 kg of goods
  (SP_600 : ℝ := C) -- selling price for 600 grams is equal to the cost price for 1 kg
  (CP_600 : ℝ := 0.6 * C) -- cost price for 600 grams
  : (SP_600 - CP_600) / CP_600 * 100 = 66.67 := by
  sorry

end dishonest_dealer_profit_percent_l1760_176090


namespace line_quadrants_condition_l1760_176047

theorem line_quadrants_condition (m n : ℝ) (h : m * n < 0) :
  (m > 0 ∧ n < 0) :=
sorry

end line_quadrants_condition_l1760_176047


namespace symmetric_line_equation_l1760_176096

-- Define the original line as an equation in ℝ².
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the line of symmetry.
def line_of_symmetry (x : ℝ) : Prop := x = 1

-- The theorem stating the equation of the symmetric line.
theorem symmetric_line_equation (x y : ℝ) :
  original_line x y → line_of_symmetry x → (x + 2 * y - 3 = 0) :=
by
  intros h₁ h₂
  sorry

end symmetric_line_equation_l1760_176096


namespace prod_sum_rel_prime_l1760_176070

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end prod_sum_rel_prime_l1760_176070


namespace incorrect_simplification_l1760_176003

theorem incorrect_simplification :
  (-(1 + 1/2) ≠ 1 + 1/2) := 
by sorry

end incorrect_simplification_l1760_176003


namespace team_size_per_team_l1760_176083

theorem team_size_per_team (managers employees teams people_per_team : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) 
  (h4 : people_per_team = (managers + employees) / teams) : 
  people_per_team = 5 :=
by 
  sorry

end team_size_per_team_l1760_176083


namespace no_solution_iff_n_eq_minus_half_l1760_176089

theorem no_solution_iff_n_eq_minus_half (n x y z : ℝ) :
  (¬∃ x y z : ℝ, 2 * n * x + y = 2 ∧ n * y + z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1 / 2 :=
by
  sorry

end no_solution_iff_n_eq_minus_half_l1760_176089


namespace certain_number_correct_l1760_176061

theorem certain_number_correct : 
  (h1 : 29.94 / 1.45 = 17.9) -> (2994 / 14.5 = 1790) :=
by 
  sorry

end certain_number_correct_l1760_176061


namespace stream_speed_l1760_176045

variables (v : ℝ) (swimming_speed : ℝ) (ratio : ℝ)

theorem stream_speed (hs : swimming_speed = 4.5) (hr : ratio = 2) (h : (swimming_speed - v) / (swimming_speed + v) = 1 / ratio) :
  v = 1.5 :=
sorry

end stream_speed_l1760_176045


namespace tim_movie_marathon_l1760_176097

variables (first_movie second_movie third_movie fourth_movie fifth_movie sixth_movie seventh_movie : ℝ)

/-- Tim's movie marathon --/
theorem tim_movie_marathon
  (first_movie_duration : first_movie = 2)
  (second_movie_duration : second_movie = 1.5 * first_movie)
  (third_movie_duration : third_movie = 0.8 * (first_movie + second_movie))
  (fourth_movie_duration : fourth_movie = 2 * second_movie)
  (fifth_movie_duration : fifth_movie = third_movie - 0.5)
  (sixth_movie_duration : sixth_movie = (second_movie + fourth_movie) / 2)
  (seventh_movie_duration : seventh_movie = 45 / fifth_movie) :
  first_movie + second_movie + third_movie + fourth_movie + fifth_movie + sixth_movie + seventh_movie = 35.8571 :=
sorry

end tim_movie_marathon_l1760_176097


namespace cars_cost_between_15000_and_20000_l1760_176023

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end cars_cost_between_15000_and_20000_l1760_176023


namespace negation_of_inverse_true_l1760_176080

variables (P : Prop)

theorem negation_of_inverse_true (h : ¬P → false) : ¬P := by
  sorry

end negation_of_inverse_true_l1760_176080


namespace area_of_field_l1760_176032

-- Define the given conditions and the problem
theorem area_of_field (L W A : ℝ) (hL : L = 20) (hFencing : 2 * W + L = 88) (hA : A = L * W) : 
  A = 680 :=
by
  sorry

end area_of_field_l1760_176032


namespace jungkook_biggest_l1760_176028

noncomputable def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_biggest :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number :=
by
  unfold jungkook_number yoongi_number yuna_number
  sorry

end jungkook_biggest_l1760_176028


namespace max_distance_is_15_l1760_176016

noncomputable def max_distance_between_cars (v_A v_B: ℝ) (a: ℝ) (D: ℝ) : ℝ :=
  if v_A > v_B ∧ D = a + 60 then (a * (1 - a / 60)) else 0

theorem max_distance_is_15 (v_A v_B: ℝ) (a: ℝ) (D: ℝ) :
  v_A > v_B ∧ D = a + 60 → max_distance_between_cars v_A v_B a D = 15 :=
by
  sorry

end max_distance_is_15_l1760_176016


namespace identify_counterfeit_in_three_weighings_l1760_176033

def CoinType := {x // x = "gold" ∨ x = "silver"}

structure Coins where
  golds: Fin 13
  silvers: Fin 14
  is_counterfeit: CoinType
  counterfeit_weight: Int

def is_lighter (c1 c2: Coins): Prop := sorry
def is_heavier (c1 c2: Coins): Prop := sorry
def balance (c1 c2: Coins): Prop := sorry

def find_counterfeit_coin (coins: Coins): Option Coins := sorry

theorem identify_counterfeit_in_three_weighings (coins: Coins) :
  ∃ (identify: Coins → Option Coins),
  ∀ coins, ( identify coins ≠ none ) :=
sorry

end identify_counterfeit_in_three_weighings_l1760_176033


namespace exam_full_marks_l1760_176002

variables {A B C D F : ℝ}

theorem exam_full_marks
  (hA : A = 0.90 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.80 * D)
  (hA_val : A = 360)
  (hD : D = 0.80 * F) 
  : F = 500 :=
sorry

end exam_full_marks_l1760_176002


namespace inverse_value_of_f_l1760_176006

theorem inverse_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2^x - 2) : f⁻¹ 2 = 3 :=
sorry

end inverse_value_of_f_l1760_176006


namespace total_candies_l1760_176056

theorem total_candies (Linda_candies Chloe_candies : ℕ) (h1 : Linda_candies = 34) (h2 : Chloe_candies = 28) :
  Linda_candies + Chloe_candies = 62 := by
  sorry

end total_candies_l1760_176056


namespace minimum_F_l1760_176031

noncomputable def F (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + (0.5 * x)

theorem minimum_F : ∃ x : ℝ, x ≥ 0 ∧ F x = 57.5 ∧ ∀ y ≥ 0, F y ≥ F x := by
  use 55
  sorry

end minimum_F_l1760_176031


namespace sasha_studies_more_avg_4_l1760_176064

-- Define the differences recorded over the five days
def differences : List ℤ := [20, 0, 30, -20, -10]

-- Calculate the average difference
def average_difference (diffs : List ℤ) : ℚ :=
  (List.sum diffs : ℚ) / (List.length diffs : ℚ)

-- The statement to prove
theorem sasha_studies_more_avg_4 :
  average_difference differences = 4 := by
  sorry

end sasha_studies_more_avg_4_l1760_176064


namespace find_x_plus_y_l1760_176066

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 16) :
  x + y = 4 := 
by
  sorry

end find_x_plus_y_l1760_176066


namespace geometric_series_sum_l1760_176046

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 6
  let sum := a * ((1 - r ^ n) / (1 - r))
  sum = (4095 / 5120 : ℚ) :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l1760_176046


namespace no_integer_roots_of_quadratic_l1760_176072

theorem no_integer_roots_of_quadratic
  (a b c : ℤ) (f : ℤ → ℤ)
  (h_def : ∀ x, f x = a * x * x + b * x + c)
  (h_a_nonzero : a ≠ 0)
  (h_f0_odd : Odd (f 0))
  (h_f1_odd : Odd (f 1)) :
  ∀ x : ℤ, f x ≠ 0 :=
by
  sorry

end no_integer_roots_of_quadratic_l1760_176072


namespace fraction_positive_implies_x_greater_than_seven_l1760_176095

variable (x : ℝ)

theorem fraction_positive_implies_x_greater_than_seven (h : -6 / (7 - x) > 0) : x > 7 := by
  sorry

end fraction_positive_implies_x_greater_than_seven_l1760_176095


namespace calc_expression_l1760_176029

theorem calc_expression :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 :=
sorry

end calc_expression_l1760_176029


namespace arithmetic_sequence_S_15_l1760_176043

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ}

theorem arithmetic_sequence_S_15 :
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 1 + a 15 = 2 * a 8) →
  (a 4 + a 12 = 2 * a 8) →
  S 15 a = -30 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S_15_l1760_176043


namespace probability_odd_80_heads_l1760_176024

noncomputable def coin_toss_probability_odd (n : ℕ) (p : ℝ) : ℝ :=
  (1 / 2) * (1 - (1 / 3^n))

theorem probability_odd_80_heads :
  coin_toss_probability_odd 80 (3 / 4) = (1 / 2) * (1 - 1 / 3^80) :=
by
  sorry

end probability_odd_80_heads_l1760_176024


namespace blueberries_in_blue_box_l1760_176067

theorem blueberries_in_blue_box (S B : ℕ) (h1 : S - B = 15) (h2 : S + B = 87) : B = 36 :=
by sorry

end blueberries_in_blue_box_l1760_176067


namespace Jake_has_8_peaches_l1760_176010

variables (Jake Steven Jill : ℕ)

-- The conditions
def condition1 : Steven = 15 := sorry
def condition2 : Steven = Jill + 14 := sorry
def condition3 : Jake = Steven - 7 := sorry

-- The proof statement
theorem Jake_has_8_peaches 
  (h1 : Steven = 15) 
  (h2 : Steven = Jill + 14) 
  (h3 : Jake = Steven - 7) : Jake = 8 :=
by
  -- The proof will go here
  sorry

end Jake_has_8_peaches_l1760_176010


namespace profit_percentage_l1760_176094

theorem profit_percentage (cost_price selling_price : ℝ) (h₁ : cost_price = 32) (h₂ : selling_price = 56) : 
  ((selling_price - cost_price) / cost_price) * 100 = 75 :=
by
  sorry

end profit_percentage_l1760_176094


namespace correct_range_a_l1760_176011

noncomputable def proposition_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def proposition_q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem correct_range_a (a : ℝ) :
  (¬ ∃ x, proposition_p a x → ¬ ∃ x, proposition_q x) →
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end correct_range_a_l1760_176011


namespace weekly_caloric_deficit_l1760_176075

-- Define the conditions
def daily_calories (day : String) : Nat :=
  if day = "Saturday" then 3500 else 2500

def daily_burn : Nat := 3000

-- Define the total calories consumed in a week
def total_weekly_consumed : Nat :=
  (2500 * 6) + 3500

-- Define the total calories burned in a week
def total_weekly_burned : Nat :=
  daily_burn * 7

-- Define the weekly deficit
def weekly_deficit : Nat :=
  total_weekly_burned - total_weekly_consumed

-- The proof goal
theorem weekly_caloric_deficit : weekly_deficit = 2500 :=
by
  -- Proof steps would go here; however, per instructions, we use sorry
  sorry

end weekly_caloric_deficit_l1760_176075


namespace no_prime_divisible_by_77_l1760_176039

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l1760_176039


namespace rope_segments_after_folds_l1760_176086

theorem rope_segments_after_folds (n : ℕ) : 
  (if n = 1 then 3 else 
   if n = 2 then 5 else 
   if n = 3 then 9 else 2^n + 1) = 2^n + 1 :=
by sorry

end rope_segments_after_folds_l1760_176086


namespace find_abc_l1760_176082

theorem find_abc :
  ∃ a b c : ℝ, 
    -- Conditions
    (a + b + c = 12) ∧ 
    (2 * b = a + c) ∧ 
    ((a + 2) * (c + 5) = (b + 2) * (b + 2)) ∧ 
    -- Correct answers
    ((a = 1 ∧ b = 4 ∧ c = 7) ∨ 
     (a = 10 ∧ b = 4 ∧ c = -2)) := 
  by 
    sorry

end find_abc_l1760_176082


namespace proof_problem_l1760_176069

-- Define sets
def N_plus : Set ℕ := {x | x > 0}  -- Positive integers
def Z : Set ℤ := {x | true}        -- Integers
def Q : Set ℚ := {x | true}        -- Rational numbers

-- Lean problem statement
theorem proof_problem : 
  (0 ∉ N_plus) ∧ 
  (((-1)^3 : ℤ) ∈ Z) ∧ 
  (π ∉ Q) :=
by
  sorry

end proof_problem_l1760_176069


namespace part_a_l1760_176065

theorem part_a (x y : ℝ) (hx : 1 > x ∧ x ≥ 0) (hy : 1 > y ∧ y ≥ 0) : 
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := sorry

end part_a_l1760_176065


namespace evaluate_sum_of_powers_of_i_l1760_176034

-- Definition of the imaginary unit i with property i^2 = -1.
def i : ℂ := Complex.I

lemma i_pow_2 : i^2 = -1 := by
  sorry

lemma i_pow_4n (n : ℤ) : i^(4 * n) = 1 := by
  sorry

-- Problem statement: Evaluate i^13 + i^18 + i^23 + i^28 + i^33 + i^38.
theorem evaluate_sum_of_powers_of_i : 
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by
  sorry

end evaluate_sum_of_powers_of_i_l1760_176034
