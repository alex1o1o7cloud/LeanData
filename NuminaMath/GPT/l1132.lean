import Mathlib

namespace NUMINAMATH_GPT_probability_window_opens_correct_l1132_113218

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_probability_window_opens_correct_l1132_113218


namespace NUMINAMATH_GPT_find_x_l1132_113201

open Real

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 6 / (x / 3)) : x = 18 ∨ x = -18 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1132_113201


namespace NUMINAMATH_GPT_spinner_sections_equal_size_l1132_113232

theorem spinner_sections_equal_size 
  (p : ℕ → Prop)
  (h1 : ∀ n, p n ↔ (1 - (1: ℝ) / n) ^ 2 = 0.5625) : 
  p 4 :=
by
  sorry

end NUMINAMATH_GPT_spinner_sections_equal_size_l1132_113232


namespace NUMINAMATH_GPT_scientific_notation_21600_l1132_113286

theorem scientific_notation_21600 : ∃ (a : ℝ) (n : ℤ), 21600 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.16 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_21600_l1132_113286


namespace NUMINAMATH_GPT_smallest_y_l1132_113294

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end NUMINAMATH_GPT_smallest_y_l1132_113294


namespace NUMINAMATH_GPT_simplify_expression_l1132_113276

theorem simplify_expression (x y : ℝ) : x^2 * y - 3 * x * y^2 + 2 * y * x^2 - y^2 * x = 3 * x^2 * y - 4 * x * y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1132_113276


namespace NUMINAMATH_GPT_ratio_of_areas_l1132_113291

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1132_113291


namespace NUMINAMATH_GPT_a_wins_by_200_meters_l1132_113272

-- Define the conditions
def race_distance : ℕ := 600
def speed_ratio_a_to_b := 5 / 4
def head_start_a : ℕ := 100

-- Define the proof statement
theorem a_wins_by_200_meters (x : ℝ) (ha_speed : ℝ := 5 * x) (hb_speed : ℝ := 4 * x)
  (ha_distance_to_win : ℝ := race_distance - head_start_a) :
  (ha_distance_to_win / ha_speed) = (400 / hb_speed) → 
  600 - (400) = 200 :=
by
  -- For now, skip the proof, focus on the statement.
  sorry

end NUMINAMATH_GPT_a_wins_by_200_meters_l1132_113272


namespace NUMINAMATH_GPT_find_books_second_purchase_profit_l1132_113266

-- For part (1)
theorem find_books (x y : ℕ) (h₁ : 12 * x + 10 * y = 1200) (h₂ : 3 * x + 2 * y = 270) :
  x = 50 ∧ y = 60 :=
by 
  sorry

-- For part (2)
theorem second_purchase_profit (m : ℕ) (h₃ : 50 * (m - 12) + 2 * 60 * (12 - 10) ≥ 340) :
  m ≥ 14 :=
by 
  sorry

end NUMINAMATH_GPT_find_books_second_purchase_profit_l1132_113266


namespace NUMINAMATH_GPT_max_AMC_AM_MC_CA_l1132_113217

theorem max_AMC_AM_MC_CA (A M C : ℕ) (h_sum : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_AMC_AM_MC_CA_l1132_113217


namespace NUMINAMATH_GPT_cat_finishes_food_on_next_monday_l1132_113216

noncomputable def cat_food_consumption_per_day : ℚ := (1 / 4) + (1 / 6)

theorem cat_finishes_food_on_next_monday :
  ∃ n : ℕ, n = 8 ∧ (n * cat_food_consumption_per_day > 8) := sorry

end NUMINAMATH_GPT_cat_finishes_food_on_next_monday_l1132_113216


namespace NUMINAMATH_GPT_total_seashells_l1132_113282

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end NUMINAMATH_GPT_total_seashells_l1132_113282


namespace NUMINAMATH_GPT_set_intersection_l1132_113295

def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1, 2}

theorem set_intersection :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1132_113295


namespace NUMINAMATH_GPT_simplest_form_l1132_113253

theorem simplest_form (b : ℝ) (h : b ≠ 2) : 2 - (2 / (2 + b / (2 - b))) = 4 / (4 - b) :=
by sorry

end NUMINAMATH_GPT_simplest_form_l1132_113253


namespace NUMINAMATH_GPT_range_of_a_sufficient_but_not_necessary_condition_l1132_113290

theorem range_of_a_sufficient_but_not_necessary_condition (a : ℝ) : 
  (-2 < x ∧ x < -1) → ((x + a) * (x + 1) < 0) → (a > 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_sufficient_but_not_necessary_condition_l1132_113290


namespace NUMINAMATH_GPT_second_dog_average_miles_l1132_113213

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end NUMINAMATH_GPT_second_dog_average_miles_l1132_113213


namespace NUMINAMATH_GPT_min_distance_between_lines_t_l1132_113236

theorem min_distance_between_lines_t (t : ℝ) :
  (∀ x y : ℝ, x + 2 * y + t^2 = 0) ∧ (∀ x y : ℝ, 2 * x + 4 * y + 2 * t - 3 = 0) →
  t = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_min_distance_between_lines_t_l1132_113236


namespace NUMINAMATH_GPT_problem_1992_AHSME_43_l1132_113275

theorem problem_1992_AHSME_43 (a b c : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : Odd a) (h2 : Odd b) : Odd (3^a + (b-1)^2 * c) :=
sorry

end NUMINAMATH_GPT_problem_1992_AHSME_43_l1132_113275


namespace NUMINAMATH_GPT_algebraic_expression_value_l1132_113234

noncomputable def a : ℝ := 2 * Real.sin (Real.pi / 4) + 1
noncomputable def b : ℝ := 2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_value :
  ((a^2 + b^2) / (2 * a * b) - 1) / ((a^2 - b^2) / (a^2 * b + a * b^2)) = 1 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1132_113234


namespace NUMINAMATH_GPT_sufficient_condition_l1132_113277

theorem sufficient_condition (a : ℝ) (h : a > 0) : a^2 + a ≥ 0 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_l1132_113277


namespace NUMINAMATH_GPT_binom_10_4_l1132_113248

theorem binom_10_4 : Nat.choose 10 4 = 210 := 
by sorry

end NUMINAMATH_GPT_binom_10_4_l1132_113248


namespace NUMINAMATH_GPT_Joey_age_l1132_113259

-- Define the basic data
def ages : List ℕ := [4, 6, 8, 10, 12]

-- Define the conditions
def cinema_ages (x y : ℕ) : Prop := x + y = 18
def soccer_ages (x y : ℕ) : Prop := x < 11 ∧ y < 11
def stays_home (x : ℕ) : Prop := x = 6

-- The goal is to prove Joey's age
theorem Joey_age : ∃ j, j ∈ ages ∧ stays_home 6 ∧ (∀ x y, cinema_ages x y → x ≠ j ∧ y ≠ j) ∧ 
(∃ x y, soccer_ages x y ∧ x ≠ 6 ∧ y ≠ 6) ∧ j = 8 := by
  sorry

end NUMINAMATH_GPT_Joey_age_l1132_113259


namespace NUMINAMATH_GPT_arrival_time_l1132_113251

def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem arrival_time (departure_time : ℕ) (stop1 stop2 stop3 travel_hours : ℕ) (stops_total_time := stop1 + stop2 + stop3) (stops_total_hours := minutes_to_hours stops_total_time) : 
  departure_time = 7 → 
  stop1 = 25 → 
  stop2 = 10 → 
  stop3 = 25 → 
  travel_hours = 12 → 
  (departure_time + (travel_hours - stops_total_hours)) % 24 = 18 :=
by
  sorry

end NUMINAMATH_GPT_arrival_time_l1132_113251


namespace NUMINAMATH_GPT_cos_sq_sub_sin_sq_l1132_113220

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end NUMINAMATH_GPT_cos_sq_sub_sin_sq_l1132_113220


namespace NUMINAMATH_GPT_solve_trig_problem_l1132_113245

theorem solve_trig_problem (α : ℝ) (h : Real.tan α = 1 / 3) :
  (Real.cos α)^2 - 2 * (Real.sin α)^2 / (Real.cos α)^2 = 7 / 9 := 
sorry

end NUMINAMATH_GPT_solve_trig_problem_l1132_113245


namespace NUMINAMATH_GPT_half_abs_diff_of_squares_l1132_113246

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end NUMINAMATH_GPT_half_abs_diff_of_squares_l1132_113246


namespace NUMINAMATH_GPT_geom_seq_min_value_l1132_113223

theorem geom_seq_min_value (r : ℝ) : 
  (1 : ℝ) = a_1 → a_2 = r → a_3 = r^2 → ∃ r : ℝ, 6 * a_2 + 7 * a_3 = -9/7 := 
by 
  intros h1 h2 h3 
  use -3/7 
  rw [h2, h3] 
  ring 
  sorry

end NUMINAMATH_GPT_geom_seq_min_value_l1132_113223


namespace NUMINAMATH_GPT_sin_70_eq_1_minus_2k_squared_l1132_113274

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by
  sorry

end NUMINAMATH_GPT_sin_70_eq_1_minus_2k_squared_l1132_113274


namespace NUMINAMATH_GPT_find_initial_apples_l1132_113233

theorem find_initial_apples (A : ℤ)
  (h1 : 6 * ((A / 8) + 8 - 30) = 12) :
  A = 192 :=
sorry

end NUMINAMATH_GPT_find_initial_apples_l1132_113233


namespace NUMINAMATH_GPT_locus_of_intersection_l1132_113299

-- Define the conditions
def line_e (m_e x y : ℝ) : Prop := y = m_e * (x - 1) + 1
def line_f (m_f x y : ℝ) : Prop := y = m_f * (x + 1) + 1
def slope_diff_cond (m_e m_f : ℝ) : Prop := (m_e - m_f = 2 ∨ m_f - m_e = 2)
def not_at_points (x y : ℝ) : Prop := (x, y) ≠ (1, 1) ∧ (x, y) ≠ (-1, 1)

-- Define the proof problem
theorem locus_of_intersection (x y m_e m_f : ℝ) :
  line_e m_e x y → line_f m_f x y → slope_diff_cond m_e m_f → not_at_points x y →
  (y = x^2 ∨ y = 2 - x^2) :=
by
  intros he hf h_diff h_not_at
  sorry

end NUMINAMATH_GPT_locus_of_intersection_l1132_113299


namespace NUMINAMATH_GPT_max_value_a7_b7_c7_d7_l1132_113203

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end NUMINAMATH_GPT_max_value_a7_b7_c7_d7_l1132_113203


namespace NUMINAMATH_GPT_prove_equation_l1132_113244

theorem prove_equation (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (2 * x + 5) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_prove_equation_l1132_113244


namespace NUMINAMATH_GPT_mary_shirt_fraction_l1132_113288

theorem mary_shirt_fraction (f : ℝ) : 
  26 * (1 - f) + 36 - 36 / 3 = 37 → f = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_shirt_fraction_l1132_113288


namespace NUMINAMATH_GPT_expected_value_is_correct_l1132_113283

noncomputable def expected_value_of_heads : ℝ :=
  let penny := 1 / 2 * 1
  let nickel := 1 / 2 * 5
  let dime := 1 / 2 * 10
  let quarter := 1 / 2 * 25
  let half_dollar := 1 / 2 * 50
  (penny + nickel + dime + quarter + half_dollar : ℝ)

theorem expected_value_is_correct : expected_value_of_heads = 45.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l1132_113283


namespace NUMINAMATH_GPT_Billy_weight_is_159_l1132_113254

def Carl_weight : ℕ := 145
def Brad_weight : ℕ := Carl_weight + 5
def Billy_weight : ℕ := Brad_weight + 9

theorem Billy_weight_is_159 : Billy_weight = 159 := by
  sorry

end NUMINAMATH_GPT_Billy_weight_is_159_l1132_113254


namespace NUMINAMATH_GPT_probability_of_sum_greater_than_15_l1132_113285

-- Definition of the dice and outcomes
def total_outcomes : ℕ := 6 * 6 * 6
def favorable_outcomes : ℕ := 10

-- Probability calculation
def probability_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

-- Theorem to be proven
theorem probability_of_sum_greater_than_15 : probability_sum_gt_15 = 5 / 108 := by
  sorry

end NUMINAMATH_GPT_probability_of_sum_greater_than_15_l1132_113285


namespace NUMINAMATH_GPT_trains_clear_time_l1132_113273

theorem trains_clear_time :
  ∀ (length_A length_B length_C : ℕ)
    (speed_A_kmph speed_B_kmph speed_C_kmph : ℕ)
    (distance_AB distance_BC : ℕ),
  length_A = 160 ∧ length_B = 320 ∧ length_C = 480 ∧
  speed_A_kmph = 42 ∧ speed_B_kmph = 30 ∧ speed_C_kmph = 48 ∧
  distance_AB = 200 ∧ distance_BC = 300 →
  ∃ (time_clear : ℚ), time_clear = 50.78 :=
by
  intros length_A length_B length_C
         speed_A_kmph speed_B_kmph speed_C_kmph
         distance_AB distance_BC h
  sorry

end NUMINAMATH_GPT_trains_clear_time_l1132_113273


namespace NUMINAMATH_GPT_part_1_part_2_l1132_113227

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def A_def : A = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  ext x
  sorry
  
def B_def : B = {x : ℝ | x^2 + 2*x - 3 > 0} := by
  ext x
  sorry

theorem part_1 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0} := by
  rw [hA, hB]
  sorry

theorem part_2 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  (compl A ∩ B) = {x | x > 1 ∨ x < -3} := by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1132_113227


namespace NUMINAMATH_GPT_N_is_85714_l1132_113257

theorem N_is_85714 (N : ℕ) (hN : 10000 ≤ N ∧ N < 100000) 
  (P : ℕ := 200000 + N) 
  (Q : ℕ := 10 * N + 2) 
  (hQ_eq_3P : Q = 3 * P) 
  : N = 85714 := 
by 
  sorry

end NUMINAMATH_GPT_N_is_85714_l1132_113257


namespace NUMINAMATH_GPT_least_five_digit_is_15625_l1132_113263

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_is_15625_l1132_113263


namespace NUMINAMATH_GPT_barry_pretzels_l1132_113258

theorem barry_pretzels (A S B : ℕ) (h1 : A = 3 * S) (h2 : S = B / 2) (h3 : A = 18) : B = 12 :=
  by
  sorry

end NUMINAMATH_GPT_barry_pretzels_l1132_113258


namespace NUMINAMATH_GPT_triangle_BD_length_l1132_113279

noncomputable def triangle_length_BD : ℝ :=
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  BD

theorem triangle_BD_length : triangle_length_BD = 63 :=
by
  -- Definitions and assumptions
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)

  -- Formal proof logic corresponding to solution steps
  sorry

end NUMINAMATH_GPT_triangle_BD_length_l1132_113279


namespace NUMINAMATH_GPT_kids_still_awake_l1132_113214

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end NUMINAMATH_GPT_kids_still_awake_l1132_113214


namespace NUMINAMATH_GPT_pies_baked_l1132_113260

theorem pies_baked (days : ℕ) (eddie_rate : ℕ) (sister_rate : ℕ) (mother_rate : ℕ)
  (H1 : eddie_rate = 3) (H2 : sister_rate = 6) (H3 : mother_rate = 8) (days_eq : days = 7) :
  eddie_rate * days + sister_rate * days + mother_rate * days = 119 :=
by
  sorry

end NUMINAMATH_GPT_pies_baked_l1132_113260


namespace NUMINAMATH_GPT_find_all_waldo_time_l1132_113204

theorem find_all_waldo_time (b : ℕ) (p : ℕ) (t : ℕ) :
  b = 15 → p = 30 → t = 3 → b * p * t = 1350 := by
sorry

end NUMINAMATH_GPT_find_all_waldo_time_l1132_113204


namespace NUMINAMATH_GPT_find_fourth_term_l1132_113252

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (a_1 a_4 d : ℕ)

-- Conditions
axiom sum_first_5 : S_n 5 = 35
axiom sum_first_9 : S_n 9 = 117
axiom sum_closed_form_first_5 : 5 * a_1 + (5 * (5 - 1)) / 2 * d = 35
axiom sum_closed_form_first_9 : 9 * a_1 + (9 * (9 - 1)) / 2 * d = 117
axiom nth_term_closed_form : ∀ n, a_n n = a_1 + (n-1)*d

-- Target
theorem find_fourth_term : a_4 = 10 := by
  sorry

end NUMINAMATH_GPT_find_fourth_term_l1132_113252


namespace NUMINAMATH_GPT_max_pieces_l1132_113284

namespace CakeProblem

-- Define the dimensions of the cake and the pieces.
def cake_side : ℕ := 16
def piece_side : ℕ := 4

-- Define the areas of the cake and the pieces.
def cake_area : ℕ := cake_side * cake_side
def piece_area : ℕ := piece_side * piece_side

-- State the main problem to prove.
theorem max_pieces : cake_area / piece_area = 16 :=
by
  -- The proof is omitted.
  sorry

end CakeProblem

end NUMINAMATH_GPT_max_pieces_l1132_113284


namespace NUMINAMATH_GPT_polynomial_equivalence_l1132_113264

def polynomial_expression (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 2 * x - 5) * (x - 2) - (x - 2) * (x ^ 2 - 5 * x + 28) + (4 * x - 7) * (x - 2) * (x + 4)

theorem polynomial_equivalence (x : ℝ) : 
  polynomial_expression x = 6 * x ^ 3 + 4 * x ^ 2 - 93 * x + 122 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_equivalence_l1132_113264


namespace NUMINAMATH_GPT_no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l1132_113212

theorem no_integer_solution_2_to_2x_minus_3_to_2y_eq_58
  (x y : ℕ)
  (h1 : 2 ^ (2 * x) - 3 ^ (2 * y) = 58) : false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l1132_113212


namespace NUMINAMATH_GPT_find_subsequence_with_sum_n_l1132_113268

theorem find_subsequence_with_sum_n (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, a i ∈ Finset.range n) 
  (h2 : (Finset.univ.sum a) < 2 * n) : 
  ∃ s : Finset (Fin n), s.sum a = n := 
sorry

end NUMINAMATH_GPT_find_subsequence_with_sum_n_l1132_113268


namespace NUMINAMATH_GPT_distinct_real_numbers_satisfying_system_l1132_113219

theorem distinct_real_numbers_satisfying_system :
  ∃! (x y z : ℝ),
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x^2 + y^2 = -x + 3 * y + z) ∧
  (y^2 + z^2 = x + 3 * y - z) ∧
  (x^2 + z^2 = 2 * x + 2 * y - z) ∧
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
sorry

end NUMINAMATH_GPT_distinct_real_numbers_satisfying_system_l1132_113219


namespace NUMINAMATH_GPT_ratio_difference_l1132_113205

theorem ratio_difference (x : ℕ) (h : 7 * x = 70) : 70 - 3 * x = 40 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_ratio_difference_l1132_113205


namespace NUMINAMATH_GPT_dodecagon_diagonals_l1132_113269

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_GPT_dodecagon_diagonals_l1132_113269


namespace NUMINAMATH_GPT_marcus_point_value_l1132_113261

theorem marcus_point_value 
  (team_total_points : ℕ)
  (marcus_percentage : ℚ)
  (three_point_goals : ℕ)
  (num_goals_type2 : ℕ)
  (score_type1 : ℕ)
  (score_type2 : ℕ)
  (total_marcus_points : ℚ)
  (points_type2 : ℚ)
  (three_point_value : ℕ := 3):
  team_total_points = 70 →
  marcus_percentage = 0.5 →
  three_point_goals = 5 →
  num_goals_type2 = 10 →
  total_marcus_points = marcus_percentage * team_total_points →
  score_type1 = three_point_goals * three_point_value →
  points_type2 = total_marcus_points - score_type1 →
  score_type2 = points_type2 / num_goals_type2 →
  score_type2 = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_marcus_point_value_l1132_113261


namespace NUMINAMATH_GPT_moles_H2O_formed_l1132_113243

-- Define the balanced equation as a struct
structure Reaction :=
(reactants : List (String × ℕ)) -- List of reactants with their stoichiometric coefficients
(products : List (String × ℕ)) -- List of products with their stoichiometric coefficients

-- Example reaction: NaHCO3 + HC2H3O2 -> NaC2H3O2 + H2O + CO2
def example_reaction : Reaction :=
{ reactants := [("NaHCO3", 1), ("HC2H3O2", 1)],
  products := [("NaC2H3O2", 1), ("H2O", 1), ("CO2", 1)] }

-- We need a predicate to determine the number of moles of a product based on the reaction
def moles_of_product (reaction : Reaction) (product : String) (moles_reactant₁ moles_reactant₂ : ℕ) : ℕ :=
if product = "H2O" then moles_reactant₁ else 0  -- Only considering H2O for simplicity

-- Now we define our main theorem
theorem moles_H2O_formed : 
  moles_of_product example_reaction "H2O" 3 3 = 3 :=
by
  -- The proof will go here; for now, we use sorry to skip it
  sorry

end NUMINAMATH_GPT_moles_H2O_formed_l1132_113243


namespace NUMINAMATH_GPT_sin_2pi_minus_alpha_l1132_113281

noncomputable def alpha_condition (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi) ∧ (Real.cos (Real.pi + α) = -1 / 2)

theorem sin_2pi_minus_alpha (α : ℝ) (h : alpha_condition α) : Real.sin (2 * Real.pi - α) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_2pi_minus_alpha_l1132_113281


namespace NUMINAMATH_GPT_solution_set_l1132_113265

noncomputable def solve_inequality : Set ℝ :=
  {x | (1 / (x - 1)) >= -1}

theorem solution_set :
  solve_inequality = {x | x ≤ 0} ∪ {x | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1132_113265


namespace NUMINAMATH_GPT_f_log₂_20_l1132_113210

noncomputable def f (x : ℝ) : ℝ := sorry -- This is a placeholder for the function f.

lemma f_neg (x : ℝ) : f (-x) = -f (x) := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (1 - x) := sorry
lemma f_special (x : ℝ) (hx : -1 < x ∧ x < 0) : f (x) = 2^x + 6 / 5 := sorry

theorem f_log₂_20 : f (Real.log 20 / Real.log 2) = -2 := by
  -- Proof details would go here.
  sorry

end NUMINAMATH_GPT_f_log₂_20_l1132_113210


namespace NUMINAMATH_GPT_dog_catches_rabbit_in_4_minutes_l1132_113224

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end NUMINAMATH_GPT_dog_catches_rabbit_in_4_minutes_l1132_113224


namespace NUMINAMATH_GPT_square_root_ratio_area_l1132_113249

theorem square_root_ratio_area (side_length_C side_length_D : ℕ) (hC : side_length_C = 45) (hD : side_length_D = 60) : 
  Real.sqrt ((side_length_C^2 : ℝ) / (side_length_D^2 : ℝ)) = 3 / 4 :=
by
  rw [hC, hD]
  sorry

end NUMINAMATH_GPT_square_root_ratio_area_l1132_113249


namespace NUMINAMATH_GPT_f_1987_is_3_l1132_113278

noncomputable def f : ℕ → ℕ :=
sorry

axiom f_is_defined : ∀ x : ℕ, f x ≠ 0
axiom f_initial : f 1 = 3
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b) + 1

theorem f_1987_is_3 : f 1987 = 3 :=
by
  -- Here we would provide the mathematical proof
  sorry

end NUMINAMATH_GPT_f_1987_is_3_l1132_113278


namespace NUMINAMATH_GPT_eval_expression_l1132_113292

theorem eval_expression : 68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1132_113292


namespace NUMINAMATH_GPT_theatre_fraction_l1132_113267

noncomputable def fraction_theatre_took_elective_last_year (T P Th M : ℕ) : Prop :=
  (P = 1 / 2 * T) ∧
  (Th + M = T - P) ∧
  (1 / 3 * P + M = 2 / 3 * T) ∧
  (Th = 1 / 6 * T)

theorem theatre_fraction (T P Th M : ℕ) :
  fraction_theatre_took_elective_last_year T P Th M →
  Th / T = 1 / 6 :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_theatre_fraction_l1132_113267


namespace NUMINAMATH_GPT_hyperbola_asymptotes_angle_l1132_113296

-- Define the given conditions and the proof problem
theorem hyperbola_asymptotes_angle (a b c : ℝ) (e : ℝ) (h1 : e = 2) 
  (h2 : e = c / a) (h3 : c = 2 * a) (h4 : b^2 + a^2 = c^2) : 
  ∃ θ : ℝ, θ = 60 :=
by 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_hyperbola_asymptotes_angle_l1132_113296


namespace NUMINAMATH_GPT_functional_solution_l1132_113202

def functional_property (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (x * f y + 2 * x) = x * y + 2 * f x

theorem functional_solution (f : ℝ → ℝ) (h : functional_property f) : f 1 = 0 :=
by sorry

end NUMINAMATH_GPT_functional_solution_l1132_113202


namespace NUMINAMATH_GPT_even_function_exists_l1132_113255

def f (x m : ℝ) : ℝ := x^2 + m * x

theorem even_function_exists : ∃ m : ℝ, ∀ x : ℝ, f x m = f (-x) m :=
by
  use 0
  intros x
  unfold f
  simp

end NUMINAMATH_GPT_even_function_exists_l1132_113255


namespace NUMINAMATH_GPT_class_duration_l1132_113280

theorem class_duration (x : ℝ) (h : 3 * x = 6) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_class_duration_l1132_113280


namespace NUMINAMATH_GPT_distinct_constructions_l1132_113207

def num_cube_constructions (white_cubes : Nat) (blue_cubes : Nat) : Nat :=
  if white_cubes = 5 ∧ blue_cubes = 3 then 5 else 0

theorem distinct_constructions : num_cube_constructions 5 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_distinct_constructions_l1132_113207


namespace NUMINAMATH_GPT_A_intersect_B_l1132_113271

def A : Set ℝ := { x | abs x < 2 }
def B : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }

theorem A_intersect_B : A ∩ B = { x | -1 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_A_intersect_B_l1132_113271


namespace NUMINAMATH_GPT_longest_segment_in_cylinder_l1132_113228

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt (h^2 + (2*r)^2) :=
by
  sorry

end NUMINAMATH_GPT_longest_segment_in_cylinder_l1132_113228


namespace NUMINAMATH_GPT_total_money_collected_l1132_113262

def hourly_wage : ℕ := 10 -- Marta's hourly wage 
def tips_collected : ℕ := 50 -- Tips collected by Marta
def hours_worked : ℕ := 19 -- Hours Marta worked

theorem total_money_collected : (hourly_wage * hours_worked + tips_collected = 240) :=
  sorry

end NUMINAMATH_GPT_total_money_collected_l1132_113262


namespace NUMINAMATH_GPT_total_number_of_matches_l1132_113250

-- Define the total number of teams
def numberOfTeams : ℕ := 10

-- Define the number of matches each team competes against each other team
def matchesPerPair : ℕ := 4

-- Calculate the total number of unique matches
def calculateUniqueMatches (teams : ℕ) : ℕ :=
  (teams * (teams - 1)) / 2

-- Main statement to be proved
theorem total_number_of_matches : calculateUniqueMatches numberOfTeams * matchesPerPair = 180 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_number_of_matches_l1132_113250


namespace NUMINAMATH_GPT_marble_draw_probability_l1132_113235

theorem marble_draw_probability :
  let total_marbles := 12
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 3

  let p_red_first := (red_marbles / total_marbles : ℚ)
  let p_white_second := (white_marbles / (total_marbles - 1) : ℚ)
  let p_blue_third := (blue_marbles / (total_marbles - 2) : ℚ)
  
  p_red_first * p_white_second * p_blue_third = (1/22 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_marble_draw_probability_l1132_113235


namespace NUMINAMATH_GPT_anticipated_sedans_l1132_113293

theorem anticipated_sedans (sales_sports_cars sedans_ratio sports_ratio sports_forecast : ℕ) 
  (h_ratio : sports_ratio = 5) (h_sedans_ratio : sedans_ratio = 8) (h_sports_forecast : sports_forecast = 35)
  (h_eq : sales_sports_cars = sports_ratio * sports_forecast) :
  sales_sports_cars * 8 / 5 = 56 :=
by
  sorry

end NUMINAMATH_GPT_anticipated_sedans_l1132_113293


namespace NUMINAMATH_GPT_max_modulus_l1132_113287

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end NUMINAMATH_GPT_max_modulus_l1132_113287


namespace NUMINAMATH_GPT_trigonometric_identity_l1132_113239

theorem trigonometric_identity
    (α φ : ℝ) :
    4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = Real.cos (2 * α) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1132_113239


namespace NUMINAMATH_GPT_pipe_b_fills_tank_7_times_faster_l1132_113256

theorem pipe_b_fills_tank_7_times_faster 
  (time_A : ℝ) 
  (time_B : ℝ)
  (combined_time : ℝ) 
  (hA : time_A = 30)
  (h_combined : combined_time = 3.75) 
  (hB : time_B = time_A / 7) :
  time_B =  30 / 7 :=
by
  sorry

end NUMINAMATH_GPT_pipe_b_fills_tank_7_times_faster_l1132_113256


namespace NUMINAMATH_GPT_molecular_weight_calculation_l1132_113225

theorem molecular_weight_calculation :
  let atomic_weight_K := 39.10
  let atomic_weight_Br := 79.90
  let atomic_weight_O := 16.00
  let num_K := 1
  let num_Br := 1
  let num_O := 3
  let molecular_weight := (num_K * atomic_weight_K) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)
  molecular_weight = 167.00 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l1132_113225


namespace NUMINAMATH_GPT_find_numbers_l1132_113289

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1132_113289


namespace NUMINAMATH_GPT_Courtney_total_marbles_l1132_113247

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end NUMINAMATH_GPT_Courtney_total_marbles_l1132_113247


namespace NUMINAMATH_GPT_factor_expression_correct_l1132_113208

variable (y : ℝ)

def expression := 4 * y * (y + 2) + 6 * (y + 2)

theorem factor_expression_correct : expression y = (y + 2) * (2 * (2 * y + 3)) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_correct_l1132_113208


namespace NUMINAMATH_GPT_exponential_function_solution_l1132_113297

theorem exponential_function_solution (a : ℝ) (h : a > 1)
  (h_max_min_diff : a - a⁻¹ = 1) : a = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_exponential_function_solution_l1132_113297


namespace NUMINAMATH_GPT_pizza_slices_left_l1132_113211

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end NUMINAMATH_GPT_pizza_slices_left_l1132_113211


namespace NUMINAMATH_GPT_finance_specialization_percentage_l1132_113241

theorem finance_specialization_percentage (F : ℝ) :
  (76 - 43.333333333333336) = (90 - F) → 
  F = 57.333333333333336 :=
by
  sorry

end NUMINAMATH_GPT_finance_specialization_percentage_l1132_113241


namespace NUMINAMATH_GPT_probability_exactly_two_heads_and_two_tails_l1132_113221

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_probability_exactly_two_heads_and_two_tails_l1132_113221


namespace NUMINAMATH_GPT_digit_2567_l1132_113238

def nth_digit_in_concatenation (n : ℕ) : ℕ :=
  sorry

theorem digit_2567 : nth_digit_in_concatenation 2567 = 8 :=
by
  sorry

end NUMINAMATH_GPT_digit_2567_l1132_113238


namespace NUMINAMATH_GPT_jessica_total_payment_l1132_113200

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_jessica_total_payment_l1132_113200


namespace NUMINAMATH_GPT_probability_no_shaded_square_l1132_113242

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_probability_no_shaded_square_l1132_113242


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1132_113237

def U (a : ℕ) : Set ℕ := { x | x > 0 ∧ x ≤ a }
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}
def C_U (S : Set ℕ) (a : ℕ) : Set ℕ := U a ∩ Sᶜ

theorem necessary_and_sufficient_condition (a : ℕ) (h : 6 ≤ a ∧ a < 7) : 
  C_U P a = Q ↔ (6 ≤ a ∧ a < 7) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1132_113237


namespace NUMINAMATH_GPT_total_roses_tom_sent_l1132_113206

theorem total_roses_tom_sent
  (roses_in_dozen : ℕ := 12)
  (dozens_per_day : ℕ := 2)
  (days_in_week : ℕ := 7) :
  7 * (2 * 12) = 168 := by
  sorry

end NUMINAMATH_GPT_total_roses_tom_sent_l1132_113206


namespace NUMINAMATH_GPT_length_third_altitude_l1132_113209

theorem length_third_altitude (a b c : ℝ) (S : ℝ) 
  (h_altitude_a : 4 = 2 * S / a)
  (h_altitude_b : 12 = 2 * S / b)
  (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_third_integer : ∃ n : ℕ, h = n):
  h = 5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_length_third_altitude_l1132_113209


namespace NUMINAMATH_GPT_awareness_survey_sampling_l1132_113298

theorem awareness_survey_sampling
  (students : Set ℝ) -- assumption that defines the set of students
  (grades : Set ℝ) -- assumption that defines the set of grades
  (awareness : ℝ → ℝ) -- assumption defining the awareness function
  (significant_differences : ∀ g1 g2 : ℝ, g1 ≠ g2 → awareness g1 ≠ awareness g2) -- significant differences in awareness among grades
  (first_grade_students : Set ℝ) -- assumption defining the set of first grade students
  (second_grade_students : Set ℝ) -- assumption defining the set of second grade students
  (third_grade_students : Set ℝ) -- assumption defining the set of third grade students
  (students_from_grades : students = first_grade_students ∪ second_grade_students ∪ third_grade_students) -- assumption that the students are from first, second, and third grades
  (representative_method : (simple_random_sampling → False) ∧ (systematic_sampling_method → False))
  : stratified_sampling_method := 
sorry

end NUMINAMATH_GPT_awareness_survey_sampling_l1132_113298


namespace NUMINAMATH_GPT_cos_of_angle_between_lines_l1132_113240

noncomputable def cosTheta (a b : ℝ × ℝ) : ℝ :=
  let dotProduct := a.1 * b.1 + a.2 * b.2
  let magA := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dotProduct / (magA * magB)

theorem cos_of_angle_between_lines :
  cosTheta (3, 4) (1, 3) = 3 / Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_between_lines_l1132_113240


namespace NUMINAMATH_GPT_store_profit_l1132_113226

theorem store_profit {C : ℝ} (h₁ : C > 0) : 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  SPF - C = 0.20 * C := 
by 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  sorry

end NUMINAMATH_GPT_store_profit_l1132_113226


namespace NUMINAMATH_GPT_sequence_perfect_square_l1132_113222

variable (a : ℕ → ℤ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n ≥ 3, a n = 7 * (a (n - 1)) - (a (n - 2))

theorem sequence_perfect_square (n : ℕ) (hn : n > 0) : ∃ k : ℤ, a n + a (n + 1) + 2 = k * k :=
by
  sorry

end NUMINAMATH_GPT_sequence_perfect_square_l1132_113222


namespace NUMINAMATH_GPT_combination_mod_100_l1132_113270

def totalDistinctHands : Nat := Nat.choose 60 12

def remainder (n : Nat) (m : Nat) : Nat := n % m

theorem combination_mod_100 :
  remainder totalDistinctHands 100 = R :=
sorry

end NUMINAMATH_GPT_combination_mod_100_l1132_113270


namespace NUMINAMATH_GPT_arithmetic_sequence_a101_eq_52_l1132_113231

theorem arithmetic_sequence_a101_eq_52 (a : ℕ → ℝ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = 1 / 2) :
  a 101 = 52 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a101_eq_52_l1132_113231


namespace NUMINAMATH_GPT_fixed_monthly_fee_l1132_113230

-- Define the problem parameters and assumptions
variables (x y : ℝ)
axiom february_bill : x + y = 20.72
axiom march_bill : x + 3 * y = 35.28

-- State the Lean theorem that we want to prove
theorem fixed_monthly_fee : x = 13.44 :=
by
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l1132_113230


namespace NUMINAMATH_GPT_pencils_given_l1132_113215

-- Define the conditions
def a : Nat := 9
def b : Nat := 65

-- Define the goal statement: the number of pencils Kathryn gave to Anthony
theorem pencils_given (a b : Nat) (h₁ : a = 9) (h₂ : b = 65) : b - a = 56 :=
by
  -- Omitted proof part
  sorry

end NUMINAMATH_GPT_pencils_given_l1132_113215


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1132_113229

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 1) : (1 / a < 1) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1132_113229
