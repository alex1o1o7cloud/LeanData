import Mathlib

namespace NUMINAMATH_GPT_line_equation_k_value_l2206_220610

theorem line_equation_k_value (m n k : ℝ) 
    (h1 : m = 2 * n + 5) 
    (h2 : m + 5 = 2 * (n + k) + 5) : 
    k = 2.5 :=
by sorry

end NUMINAMATH_GPT_line_equation_k_value_l2206_220610


namespace NUMINAMATH_GPT_abs_div_nonzero_l2206_220622

theorem abs_div_nonzero (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  ¬ (|a| / a + |b| / b = 1) :=
by
  sorry

end NUMINAMATH_GPT_abs_div_nonzero_l2206_220622


namespace NUMINAMATH_GPT_find_fraction_l2206_220618

theorem find_fraction (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4)
  (h3 : b = 1) : a / b = (17 + Real.sqrt 269) / 10 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l2206_220618


namespace NUMINAMATH_GPT_find_Allyson_age_l2206_220660

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end NUMINAMATH_GPT_find_Allyson_age_l2206_220660


namespace NUMINAMATH_GPT_right_triangle_conditions_l2206_220640

-- Definitions
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

-- Conditions
def cond1 (A B C : ℝ) : Prop := A + B = C
def cond2 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def cond3 (A B C : ℝ) : Prop := A = B ∧ B = 2 * C
def cond4 (A B C : ℝ) : Prop := A = 2 * B ∧ B = 3 * C

-- Problem statement
theorem right_triangle_conditions (A B C : ℝ) :
  (cond1 A B C → is_right_triangle A B C) ∧
  (cond2 A B C → is_right_triangle A B C) ∧
  ¬(cond3 A B C → is_right_triangle A B C) ∧
  ¬(cond4 A B C → is_right_triangle A B C) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_conditions_l2206_220640


namespace NUMINAMATH_GPT_angle_of_skew_lines_in_range_l2206_220672

noncomputable def angle_between_skew_lines (θ : ℝ) (θ_range : 0 < θ ∧ θ ≤ 90) : Prop :=
  θ ∈ (Set.Ioc 0 90)

-- We assume the existence of such an angle θ formed by two skew lines
theorem angle_of_skew_lines_in_range (θ : ℝ) (h_skew : true) : angle_between_skew_lines θ (⟨sorry, sorry⟩) :=
  sorry

end NUMINAMATH_GPT_angle_of_skew_lines_in_range_l2206_220672


namespace NUMINAMATH_GPT_range_of_m_l2206_220695

noncomputable def y (m x : ℝ) := m * (1/4)^x - (1/2)^x + 1

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, y m x = 0) → (m ≤ 0 ∨ m = 1 / 4) := sorry

end NUMINAMATH_GPT_range_of_m_l2206_220695


namespace NUMINAMATH_GPT_laps_needed_l2206_220632

theorem laps_needed (r1 r2 : ℕ) (laps1 : ℕ) (h1 : r1 = 30) (h2 : r2 = 10) (h3 : laps1 = 40) : 
  (r1 * laps1) / r2 = 120 := by
  sorry

end NUMINAMATH_GPT_laps_needed_l2206_220632


namespace NUMINAMATH_GPT_flight_duration_problem_l2206_220612

def problem_conditions : Prop :=
  let la_departure_pst := (7, 15) -- 7:15 AM PST
  let ny_arrival_est := (17, 40) -- 5:40 PM EST (17:40 in 24-hour format)
  let time_difference := 3 -- Hours difference (EST is 3 hours ahead of PST)
  let dst_adjustment := 1 -- Daylight saving time adjustment in hours
  ∃ (h m : ℕ), (0 < m ∧ m < 60) ∧ ((h = 7 ∧ m = 25) ∧ (h + m = 32))

theorem flight_duration_problem :
  problem_conditions :=
by
  -- Placeholder for the proof that shows the conditions established above imply h + m = 32
  sorry

end NUMINAMATH_GPT_flight_duration_problem_l2206_220612


namespace NUMINAMATH_GPT_siblings_count_l2206_220630

noncomputable def Masud_siblings (M : ℕ) : Prop :=
  (4 * M - 60 = (3 * M) / 4 + 135) → M = 60

theorem siblings_count (M : ℕ) : Masud_siblings M :=
  by
  sorry

end NUMINAMATH_GPT_siblings_count_l2206_220630


namespace NUMINAMATH_GPT_combined_weight_of_student_and_sister_l2206_220604

theorem combined_weight_of_student_and_sister
  (S : ℝ) (R : ℝ)
  (h1 : S = 90)
  (h2 : S - 6 = 2 * R) :
  S + R = 132 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_student_and_sister_l2206_220604


namespace NUMINAMATH_GPT_wendy_first_day_miles_l2206_220629

-- Define the variables for the problem
def total_miles : ℕ := 493
def miles_day2 : ℕ := 223
def miles_day3 : ℕ := 145

-- Define the proof problem
theorem wendy_first_day_miles :
  total_miles = miles_day2 + miles_day3 + 125 :=
sorry

end NUMINAMATH_GPT_wendy_first_day_miles_l2206_220629


namespace NUMINAMATH_GPT_correct_meteor_passing_time_l2206_220670

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end NUMINAMATH_GPT_correct_meteor_passing_time_l2206_220670


namespace NUMINAMATH_GPT_solution_set_inequality_l2206_220641

theorem solution_set_inequality :
  {x : ℝ | (x^2 - 4) * (x - 6)^2 ≤ 0} = {x : ℝ | (-2 ≤ x ∧ x ≤ 2) ∨ x = 6} :=
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l2206_220641


namespace NUMINAMATH_GPT_trapezium_distance_l2206_220661

theorem trapezium_distance (h : ℝ) (a b A : ℝ) 
  (h_area : A = 95) (h_a : a = 20) (h_b : b = 18) :
  A = (1/2 * (a + b) * h) → h = 5 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_distance_l2206_220661


namespace NUMINAMATH_GPT_avg_score_is_94_l2206_220674

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_avg_score_is_94_l2206_220674


namespace NUMINAMATH_GPT_loss_of_30_yuan_is_minus_30_yuan_l2206_220662

def profit (p : ℤ) : Prop := p = 20
def loss (l : ℤ) : Prop := l = -30

theorem loss_of_30_yuan_is_minus_30_yuan (p : ℤ) (l : ℤ) (h : profit p) : loss l :=
by
  sorry

end NUMINAMATH_GPT_loss_of_30_yuan_is_minus_30_yuan_l2206_220662


namespace NUMINAMATH_GPT_coordinates_with_respect_to_origin_l2206_220678

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_with_respect_to_origin_l2206_220678


namespace NUMINAMATH_GPT_part_1_part_2_l2206_220681

theorem part_1 (a b A B : ℝ)
  (h : b * (Real.sin A)^2 = Real.sqrt 3 * a * Real.cos A * Real.sin B) 
  (h_sine_law : b / Real.sin B = a / Real.sin A)
  (A_in_range: A ∈ Set.Ioo 0 Real.pi):
  A = Real.pi / 3 := 
sorry

theorem part_2 (x : ℝ)
  (A : ℝ := Real.pi / 3)
  (h_sin_cos : ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
                f x = (Real.sin A * (Real.cos x)^2) - (Real.sin (A / 2))^2 * (Real.sin (2 * x))) :
  Set.image f (Set.Icc 0 (Real.pi / 2)) = Set.Icc ((Real.sqrt 3 - 2)/4) (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l2206_220681


namespace NUMINAMATH_GPT_diophantine_3x_5y_diophantine_3x_5y_indefinite_l2206_220642

theorem diophantine_3x_5y (n : ℕ) (h_n_pos : n > 0) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n) ↔ 
    (∃ k : ℕ, (n = 3 * k ∧ n ≥ 15) ∨ 
              (n = 3 * k + 1 ∧ n ≥ 13) ∨ 
              (n = 3 * k + 2 ∧ n ≥ 11) ∨ 
              (n = 8)) :=
sorry

theorem diophantine_3x_5y_indefinite (n m : ℕ) (h_n_large : n > 40 * m):
  ∃ (N : ℕ), ∀ k ≤ N, ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n + k :=
sorry

end NUMINAMATH_GPT_diophantine_3x_5y_diophantine_3x_5y_indefinite_l2206_220642


namespace NUMINAMATH_GPT_part1_part2_l2206_220616

section

variable (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_seq : ∀ n, a_seq (n + 1) = (5 * a_seq n - 8) / (a_seq n - 1))
variable (h_initial : a_seq 1 = a)

-- Part 1:
theorem part1 (h_a : a = 3) : 
  ∃ r : ℝ, ∀ n, (a_seq n - 2) / (a_seq n - 4) = r ^ n ∧ a_seq n = (4 * 3 ^ (n - 1) + 2) / (3 ^ (n - 1) + 1) := 
sorry

-- Part 2:
theorem part2 (h_pos : ∀ n, a_seq n > 3) : 3 < a := 
sorry

end

end NUMINAMATH_GPT_part1_part2_l2206_220616


namespace NUMINAMATH_GPT_two_color_K6_contains_monochromatic_triangle_l2206_220649

theorem two_color_K6_contains_monochromatic_triangle (V : Type) [Fintype V] [DecidableEq V]
  (hV : Fintype.card V = 6)
  (color : V → V → Fin 2) :
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (color a b = color b c ∧ color b c = color c a) := by
  sorry

end NUMINAMATH_GPT_two_color_K6_contains_monochromatic_triangle_l2206_220649


namespace NUMINAMATH_GPT_g_neg_eleven_eq_neg_two_l2206_220638

def f (x : ℝ) : ℝ := 2 * x - 7
def g (y : ℝ) : ℝ := 3 * y^2 + 4 * y - 6

theorem g_neg_eleven_eq_neg_two : g (-11) = -2 := by
  sorry

end NUMINAMATH_GPT_g_neg_eleven_eq_neg_two_l2206_220638


namespace NUMINAMATH_GPT_cos_double_angle_l2206_220685

theorem cos_double_angle (α : ℝ) (h : Real.sin (π/6 - α) = 1/3) :
  Real.cos (2 * (π/3 + α)) = -7/9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2206_220685


namespace NUMINAMATH_GPT_employee_pays_204_l2206_220631

-- Definitions based on conditions
def wholesale_cost : ℝ := 200
def markup_percent : ℝ := 0.20
def discount_percent : ℝ := 0.15

def retail_price := wholesale_cost * (1 + markup_percent)
def employee_payment := retail_price * (1 - discount_percent)

-- Theorem with the expected result
theorem employee_pays_204 : employee_payment = 204 := by
  -- Proof not required, we add sorry to avoid the proof details
  sorry

end NUMINAMATH_GPT_employee_pays_204_l2206_220631


namespace NUMINAMATH_GPT_max_product_sum_1988_l2206_220680

theorem max_product_sum_1988 :
  ∃ (n : ℕ) (a : ℕ), n + a = 1988 ∧ a = 1 ∧ n = 662 ∧ (3^n * 2^a) = 2 * 3^662 :=
by
  sorry

end NUMINAMATH_GPT_max_product_sum_1988_l2206_220680


namespace NUMINAMATH_GPT_functions_not_exist_l2206_220687

theorem functions_not_exist :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by
  sorry

end NUMINAMATH_GPT_functions_not_exist_l2206_220687


namespace NUMINAMATH_GPT_text_messages_in_march_l2206_220624

/-
Jared sent text messages each month according to the formula:
  T_n = n^3 - n^2 + n
We need to prove that the number of text messages Jared will send in March
(which is the 5th month) is given by T_5 = 105.
-/

def T (n : ℕ) : ℕ := n^3 - n^2 + n

theorem text_messages_in_march : T 5 = 105 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_text_messages_in_march_l2206_220624


namespace NUMINAMATH_GPT_abs_inequality_k_ge_neg3_l2206_220637

theorem abs_inequality_k_ge_neg3 (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 :=
sorry

end NUMINAMATH_GPT_abs_inequality_k_ge_neg3_l2206_220637


namespace NUMINAMATH_GPT_find_side_a_in_triangle_l2206_220646

noncomputable def triangle_side_a (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  cosA = 4/5 ∧ b = 2 ∧ S = 3 → a = Real.sqrt 13

-- Theorem statement with explicit conditions and proof goal
theorem find_side_a_in_triangle
  (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) :
  cosA = 4 / 5 → b = 2 → S = 3 → a = Real.sqrt 13 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_find_side_a_in_triangle_l2206_220646


namespace NUMINAMATH_GPT_base7_addition_l2206_220668

theorem base7_addition (X Y : ℕ) (h1 : X + 5 = 9) (h2 : Y + 2 = 4) : X + Y = 6 :=
by
  sorry

end NUMINAMATH_GPT_base7_addition_l2206_220668


namespace NUMINAMATH_GPT_output_value_is_16_l2206_220652

def f (x : ℤ) : ℤ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem output_value_is_16 : f 5 = 16 := by
  sorry

end NUMINAMATH_GPT_output_value_is_16_l2206_220652


namespace NUMINAMATH_GPT_fraction_of_top10_lists_l2206_220654

theorem fraction_of_top10_lists (total_members : ℕ) (min_lists : ℝ) (H1 : total_members = 795) (H2 : min_lists = 198.75) :
  (min_lists / total_members) = 1 / 4 :=
by
  -- The proof is omitted as requested
  sorry

end NUMINAMATH_GPT_fraction_of_top10_lists_l2206_220654


namespace NUMINAMATH_GPT_integral_root_of_equation_l2206_220696

theorem integral_root_of_equation : 
  ∀ x : ℤ, (x - 8 / (x - 4)) = 2 - 8 / (x - 4) ↔ x = 2 := 
sorry

end NUMINAMATH_GPT_integral_root_of_equation_l2206_220696


namespace NUMINAMATH_GPT_proportion_of_line_segments_l2206_220617

theorem proportion_of_line_segments (a b c d : ℕ)
  (h_proportion : a * d = b * c)
  (h_a : a = 2)
  (h_b : b = 4)
  (h_c : c = 3) :
  d = 6 :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_line_segments_l2206_220617


namespace NUMINAMATH_GPT_smallest_number_is_32_l2206_220679

theorem smallest_number_is_32 (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 25) (h3 : c = 25 + 8) : a = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_is_32_l2206_220679


namespace NUMINAMATH_GPT_time_walking_each_day_l2206_220634

variable (days : Finset ℕ) (d1 : ℕ) (d2 : ℕ) (W : ℕ)

def time_spent_parking (days : Finset ℕ) : ℕ :=
  5 * days.card

def time_spent_metal_detector : ℕ :=
  2 * 30 + 3 * 10

def total_timespent (d1 d2 W : ℕ) : ℕ :=
  d1 + d2 + W

theorem time_walking_each_day (total_minutes : ℕ) (total_days : ℕ):
  total_timespent (time_spent_parking days) (time_spent_metal_detector) (total_minutes - time_spent_metal_detector - 5 * total_days)
  = total_minutes → W = 3 := by
  sorry

end NUMINAMATH_GPT_time_walking_each_day_l2206_220634


namespace NUMINAMATH_GPT_find_m_plus_b_l2206_220647

-- Define the given equation
def given_line (x y : ℝ) : Prop := x - 3 * y + 11 = 0

-- Define the reflection of the given line about the x-axis
def reflected_line (x y : ℝ) : Prop := x + 3 * y + 11 = 0

-- Define the slope-intercept form of the reflected line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- State the theorem to prove
theorem find_m_plus_b (m b : ℝ) :
  (∀ x y : ℝ, reflected_line x y ↔ slope_intercept_form m b x y) → m + b = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_plus_b_l2206_220647


namespace NUMINAMATH_GPT_vertex_of_parabola_l2206_220605

theorem vertex_of_parabola :
  ∀ (x y : ℝ), y = (1 / 3) * (x - 7) ^ 2 + 5 → ∃ h k : ℝ, h = 7 ∧ k = 5 ∧ y = (1 / 3) * (x - h) ^ 2 + k :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l2206_220605


namespace NUMINAMATH_GPT_chatterboxes_total_jokes_l2206_220658

theorem chatterboxes_total_jokes :
  let num_chatterboxes := 10
  let jokes_increasing := (100 * (100 + 1)) / 2
  let jokes_decreasing := (99 * (99 + 1)) / 2
  (jokes_increasing + jokes_decreasing) / num_chatterboxes = 1000 :=
by
  sorry

end NUMINAMATH_GPT_chatterboxes_total_jokes_l2206_220658


namespace NUMINAMATH_GPT_find_initial_money_l2206_220645
 
theorem find_initial_money (x : ℕ) (gift_grandma gift_aunt_uncle gift_parents total_money : ℕ) 
  (h1 : gift_grandma = 25) 
  (h2 : gift_aunt_uncle = 20) 
  (h3 : gift_parents = 75) 
  (h4 : total_money = 279) 
  (h : x + (gift_grandma + gift_aunt_uncle + gift_parents) = total_money) : 
  x = 159 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_money_l2206_220645


namespace NUMINAMATH_GPT_find_divisor_l2206_220688

-- Define the conditions as hypotheses and the main problem as a theorem
theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 6) / y = 6) : y = 8 := sorry

end NUMINAMATH_GPT_find_divisor_l2206_220688


namespace NUMINAMATH_GPT_display_total_cans_l2206_220601

def row_num_cans (row : ℕ) : ℕ :=
  if row < 7 then 19 - 3 * (7 - row)
  else 19 + 3 * (row - 7)

def total_cans : ℕ :=
  List.sum (List.map row_num_cans (List.range 10))

theorem display_total_cans : total_cans = 145 := 
  sorry

end NUMINAMATH_GPT_display_total_cans_l2206_220601


namespace NUMINAMATH_GPT_expression_evaluation_l2206_220613

theorem expression_evaluation : 2^2 - Real.tan (Real.pi / 3) + abs (Real.sqrt 3 - 1) - (3 - Real.pi)^0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2206_220613


namespace NUMINAMATH_GPT_find_n_values_l2206_220648

-- Define a function to sum the first n consecutive natural numbers starting from k
def sum_consecutive_numbers (n k : ℕ) : ℕ :=
  n * k + (n * (n - 1)) / 2

-- Define a predicate to check if a number is a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the theorem statement
theorem find_n_values (n : ℕ) (k : ℕ) :
  is_prime (sum_consecutive_numbers n k) →
  n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_values_l2206_220648


namespace NUMINAMATH_GPT_max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l2206_220657

theorem max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h₁ : a ≤ b) (h₂ : b ≤ c) (h₃ : c ≤ 2 * a) :
    b / a + c / b + a / c ≤ 7 / 2 := 
  sorry

end NUMINAMATH_GPT_max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l2206_220657


namespace NUMINAMATH_GPT_dance_problem_l2206_220627

theorem dance_problem :
  ∃ (G : ℝ) (B T : ℝ),
    B / G = 3 / 4 ∧
    T = 0.20 * B ∧
    B + G + T = 114 ∧
    G = 60 :=
by
  sorry

end NUMINAMATH_GPT_dance_problem_l2206_220627


namespace NUMINAMATH_GPT_camera_lens_distance_l2206_220635

theorem camera_lens_distance (f u : ℝ) (h_fu : f ≠ u) (h_f : f ≠ 0) (h_u : u ≠ 0) :
  (∃ v : ℝ, (1 / f) = (1 / u) + (1 / v) ∧ v = (f * u) / (u - f)) :=
by {
  sorry
}

end NUMINAMATH_GPT_camera_lens_distance_l2206_220635


namespace NUMINAMATH_GPT_gcd_lcm_888_1147_l2206_220666

theorem gcd_lcm_888_1147 :
  Nat.gcd 888 1147 = 37 ∧ Nat.lcm 888 1147 = 27528 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_888_1147_l2206_220666


namespace NUMINAMATH_GPT_problem1_problem2_l2206_220673

-- Problem 1
theorem problem1 (x : ℝ) : 
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (x + 3)^2 = (1 - 2 * x)^2 ↔ x = 4 ∨ x = -2 / 3 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2206_220673


namespace NUMINAMATH_GPT_K_time_correct_l2206_220683

open Real

noncomputable def K_speed : ℝ := sorry
noncomputable def M_speed : ℝ := K_speed - 1 / 2
noncomputable def K_time : ℝ := 45 / K_speed
noncomputable def M_time : ℝ := 45 / M_speed

theorem K_time_correct (K_speed_correct : 45 / K_speed - 45 / M_speed = 1 / 2) : K_time = 45 / K_speed :=
by
  sorry

end NUMINAMATH_GPT_K_time_correct_l2206_220683


namespace NUMINAMATH_GPT_rectangular_sheet_integer_side_l2206_220690

theorem rectangular_sheet_integer_side
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_cut_a : ∀ x, x ≤ a → ∃ n : ℕ, x = n ∨ x = n + 1)
  (h_cut_b : ∀ y, y ≤ b → ∃ n : ℕ, y = n ∨ y = n + 1) :
  ∃ n m : ℕ, a = n ∨ b = m := 
sorry

end NUMINAMATH_GPT_rectangular_sheet_integer_side_l2206_220690


namespace NUMINAMATH_GPT_div_by_7_l2206_220623

theorem div_by_7 (k : ℕ) : (2^(6*k + 1) + 3^(6*k + 1) + 5^(6*k + 1)) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_div_by_7_l2206_220623


namespace NUMINAMATH_GPT_overtime_hours_correct_l2206_220628

def regular_pay_rate : ℕ := 3
def max_regular_hours : ℕ := 40
def total_pay_received : ℕ := 192
def overtime_pay_rate : ℕ := 2 * regular_pay_rate
def regular_earnings : ℕ := regular_pay_rate * max_regular_hours
def additional_earnings : ℕ := total_pay_received - regular_earnings
def calculated_overtime_hours : ℕ := additional_earnings / overtime_pay_rate

theorem overtime_hours_correct :
  calculated_overtime_hours = 12 :=
by
  sorry

end NUMINAMATH_GPT_overtime_hours_correct_l2206_220628


namespace NUMINAMATH_GPT_polynomial_sum_l2206_220692

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l2206_220692


namespace NUMINAMATH_GPT_wire_problem_l2206_220689

theorem wire_problem (a b : ℝ) (h_perimeter : a = b) : a / b = 1 := by
  sorry

end NUMINAMATH_GPT_wire_problem_l2206_220689


namespace NUMINAMATH_GPT_ratio_nonupgraded_to_upgraded_l2206_220606

-- Define the initial conditions and properties
variable (S : ℝ) (N : ℝ)
variable (h1 : ∀ N, N = S / 32)
variable (h2 : ∀ S, 0.25 * S = 0.25 * S)
variable (h3 : S > 0)

-- Define the theorem to show the required ratio
theorem ratio_nonupgraded_to_upgraded (h3 : 24 * N = 0.75 * S) : (N / (0.25 * S) = 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_ratio_nonupgraded_to_upgraded_l2206_220606


namespace NUMINAMATH_GPT_minimum_n_for_3_zeros_l2206_220691

theorem minimum_n_for_3_zeros :
  ∃ n : ℕ, (∀ m : ℕ, (m < n → ∀ k < 10, m + k ≠ 5 * m ∧ m + k ≠ 5 * m + 25)) ∧
  (∀ k < 10, n + k = 16 ∨ n + k = 16 + 9) ∧
  n = 16 :=
sorry

end NUMINAMATH_GPT_minimum_n_for_3_zeros_l2206_220691


namespace NUMINAMATH_GPT_no_pairs_xy_perfect_square_l2206_220626

theorem no_pairs_xy_perfect_square :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ ∃ k : ℕ, (xy + 1) * (xy + x + 2) = k^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_no_pairs_xy_perfect_square_l2206_220626


namespace NUMINAMATH_GPT_vectors_coplanar_l2206_220621

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (1, -3, -7)
def vector_c : ℝ × ℝ × ℝ := (1, 2, 3)

def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product vector_a vector_b vector_c = 0 := 
by
  sorry

end NUMINAMATH_GPT_vectors_coplanar_l2206_220621


namespace NUMINAMATH_GPT_cos_sq_minus_exp_equals_neg_one_fourth_l2206_220664

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end NUMINAMATH_GPT_cos_sq_minus_exp_equals_neg_one_fourth_l2206_220664


namespace NUMINAMATH_GPT_sum_f_always_negative_l2206_220611

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_always_negative
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_sum_f_always_negative_l2206_220611


namespace NUMINAMATH_GPT_tangent_line_characterization_l2206_220643

theorem tangent_line_characterization 
  (α β m n : ℝ) 
  (h_pos_α : 0 < α) 
  (h_pos_β : 0 < β) 
  (h_alpha_beta : 1/α + 1/β = 1)
  (h_pos_m : 0 < m)
  (h_pos_n : 0 < n) :
  (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ x^α + y^α = 1 → mx + ny = 1) ↔ (m^β + n^β = 1) := 
sorry

end NUMINAMATH_GPT_tangent_line_characterization_l2206_220643


namespace NUMINAMATH_GPT_binomial_expansion_sum_l2206_220602

theorem binomial_expansion_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 :=
sorry

end NUMINAMATH_GPT_binomial_expansion_sum_l2206_220602


namespace NUMINAMATH_GPT_counterexample_exists_l2206_220697

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem equivalently in Lean
theorem counterexample_exists : (sum_of_digits 33 % 6 = 0) ∧ (33 % 6 ≠ 0) := by
  sorry

end NUMINAMATH_GPT_counterexample_exists_l2206_220697


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l2206_220615

theorem number_of_sides_of_polygon (n : ℕ) (h1 : (n * (n - 3)) = 340) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l2206_220615


namespace NUMINAMATH_GPT_pat_peano_maximum_pages_l2206_220625

noncomputable def count_fives_in_range : ℕ → ℕ := sorry

theorem pat_peano_maximum_pages (n : ℕ) : 
  (count_fives_in_range 54) = 15 → n ≤ 54 :=
sorry

end NUMINAMATH_GPT_pat_peano_maximum_pages_l2206_220625


namespace NUMINAMATH_GPT_passenger_speed_relative_forward_correct_l2206_220671

-- Define the conditions
def train_speed : ℝ := 60     -- Train's speed in km/h
def passenger_speed_inside_train : ℝ := 3  -- Passenger's speed inside the train in km/h

-- Define the effective speed of the passenger relative to the railway track when moving forward
def passenger_speed_relative_forward (train_speed passenger_speed_inside_train : ℝ) : ℝ :=
  train_speed + passenger_speed_inside_train

-- Prove that the passenger's speed relative to the railway track is 63 km/h when moving forward
theorem passenger_speed_relative_forward_correct :
  passenger_speed_relative_forward train_speed passenger_speed_inside_train = 63 := by
  sorry

end NUMINAMATH_GPT_passenger_speed_relative_forward_correct_l2206_220671


namespace NUMINAMATH_GPT_two_colonies_reach_limit_in_same_time_l2206_220665

theorem two_colonies_reach_limit_in_same_time (d : ℕ) (h : 16 = d): 
  d = 16 :=
by
  /- Asserting that if one colony takes 16 days, two starting together will also take 16 days -/
  sorry

end NUMINAMATH_GPT_two_colonies_reach_limit_in_same_time_l2206_220665


namespace NUMINAMATH_GPT_identify_conic_section_l2206_220633

theorem identify_conic_section (x y : ℝ) :
  (x + 7)^2 = (5 * y - 6)^2 + 125 →
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e * x * y + f = 0 ∧
  (a > 0) ∧ (b < 0) := sorry

end NUMINAMATH_GPT_identify_conic_section_l2206_220633


namespace NUMINAMATH_GPT_option_D_correct_l2206_220677

-- Definitions representing conditions
variables (a b : Line) (α : Plane)

-- Conditions
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_in_plane (b : Line) (α : Plane) : Prop := sorry

-- Theorem stating the correctness of option D
theorem option_D_correct (h1 : line_parallel_plane a α)
                         (h2 : line_parallel_line a b) :
                         (line_in_plane b α) ∨ (line_parallel_plane b α) :=
by
  sorry

end NUMINAMATH_GPT_option_D_correct_l2206_220677


namespace NUMINAMATH_GPT_smallest_class_size_l2206_220676

theorem smallest_class_size :
  ∀ (x : ℕ), 4 * x + 3 > 50 → 4 * x + 3 = 51 :=
by
  sorry

end NUMINAMATH_GPT_smallest_class_size_l2206_220676


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l2206_220607

theorem sum_of_roots_of_quadratic : 
  ∀ x1 x2 : ℝ, 
  (3 * x1^2 - 6 * x1 - 7 = 0 ∧ 3 * x2^2 - 6 * x2 - 7 = 0) → 
  (x1 + x2 = 2) := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l2206_220607


namespace NUMINAMATH_GPT_rectangle_perimeter_from_square_l2206_220609

theorem rectangle_perimeter_from_square (d : ℝ)
  (h : d = 6) :
  ∃ (p : ℝ), p = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_from_square_l2206_220609


namespace NUMINAMATH_GPT_staffing_correct_l2206_220655

-- The number of ways to staff a battle station with constraints.
def staffing_ways (total_applicants unsuitable_fraction: ℕ) (job_openings: ℕ): ℕ :=
  let suitable_candidates := total_applicants * (1 - unsuitable_fraction)
  if suitable_candidates < job_openings then
    0 
  else
    (List.range' (suitable_candidates - job_openings + 1) job_openings).prod

-- Definitions of the problem conditions
def total_applicants := 30
def unsuitable_fraction := 2/3
def job_openings := 5
-- Expected result
def expected_result := 30240

-- The theorem to prove the number of ways to staff the battle station equals the given result.
theorem staffing_correct : staffing_ways total_applicants unsuitable_fraction job_openings = expected_result := by
  sorry

end NUMINAMATH_GPT_staffing_correct_l2206_220655


namespace NUMINAMATH_GPT_sin_alpha_plus_beta_eq_33_by_65_l2206_220653

theorem sin_alpha_plus_beta_eq_33_by_65 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hcosα : Real.cos α = 12 / 13) 
  (hcos_2α_β : Real.cos (2 * α + β) = 3 / 5) :
  Real.sin (α + β) = 33 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_beta_eq_33_by_65_l2206_220653


namespace NUMINAMATH_GPT_speed_of_B_l2206_220619

theorem speed_of_B 
    (initial_distance : ℕ)
    (speed_of_A : ℕ)
    (time : ℕ)
    (distance_covered_by_A : ℕ)
    (distance_covered_by_B : ℕ)
    : initial_distance = 24 → speed_of_A = 5 → time = 2 → distance_covered_by_A = speed_of_A * time → distance_covered_by_B = initial_distance - distance_covered_by_A → distance_covered_by_B / time = 7 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_B_l2206_220619


namespace NUMINAMATH_GPT_correct_option_D_l2206_220636

theorem correct_option_D (defect_rate_products : ℚ)
                         (rain_probability : ℚ)
                         (cure_rate_hospital : ℚ)
                         (coin_toss_heads_probability : ℚ)
                         (coin_toss_tails_probability : ℚ):
  defect_rate_products = 1/10 →
  rain_probability = 0.9 →
  cure_rate_hospital = 0.1 →
  coin_toss_heads_probability = 0.5 →
  coin_toss_tails_probability = 0.5 →
  coin_toss_tails_probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5
  exact h5

end NUMINAMATH_GPT_correct_option_D_l2206_220636


namespace NUMINAMATH_GPT_concentric_but_different_radius_l2206_220620

noncomputable def circleF (x y : ℝ) : ℝ :=
  x^2 + y^2 - 1

def pointP (x : ℝ) : ℝ × ℝ :=
  (x, x)

def circleEquation (x y : ℝ) : Prop :=
  circleF x y = 0

def circleEquation' (x y : ℝ) : Prop :=
  circleF x y - circleF x y = 0

theorem concentric_but_different_radius (x : ℝ) (hP : circleF x x ≠ 0) (hCenter : x ≠ 0):
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧
    ∀ x y, (circleEquation x y ↔ x^2 + y^2 = 1) ∧ 
           (circleEquation' x y ↔ x^2 + y^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_concentric_but_different_radius_l2206_220620


namespace NUMINAMATH_GPT_exist_positive_abc_with_nonzero_integer_roots_l2206_220639

theorem exist_positive_abc_with_nonzero_integer_roots :
  ∃ (a b c : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 + b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) :=
sorry

end NUMINAMATH_GPT_exist_positive_abc_with_nonzero_integer_roots_l2206_220639


namespace NUMINAMATH_GPT_selection_and_arrangement_l2206_220667

-- Defining the problem conditions
def volunteers : Nat := 5
def roles : Nat := 4
def A_excluded_role : String := "music_composer"
def total_methods : Nat := 96

theorem selection_and_arrangement (h1 : volunteers = 5) (h2 : roles = 4) (h3 : A_excluded_role = "music_composer") :
  total_methods = 96 :=
by
  sorry

end NUMINAMATH_GPT_selection_and_arrangement_l2206_220667


namespace NUMINAMATH_GPT_polynomial_expression_value_l2206_220600

theorem polynomial_expression_value (a : ℕ → ℤ) (x : ℤ) :
  (x + 2)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 →
  ((a 1 + 3 * a 3 + 5 * a 5 + 7 * a 7 + 9 * a 9)^2 - (2 * a 2 + 4 * a 4 + 6 * a 6 + 8 * a 8)^2) = 3^12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expression_value_l2206_220600


namespace NUMINAMATH_GPT_k_even_l2206_220650

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end NUMINAMATH_GPT_k_even_l2206_220650


namespace NUMINAMATH_GPT_max_workers_l2206_220698

-- Each worker produces 10 bricks a day and steals as many bricks per day as there are workers at the factory.
def worker_bricks_produced_per_day : ℕ := 10
def worker_bricks_stolen_per_day (n : ℕ) : ℕ := n

-- The factory must have at least 13 more bricks at the end of the day.
def factory_brick_surplus_requirement : ℕ := 13

-- Prove the maximum number of workers that can be hired so that the factory has at least 13 more bricks than at the beginning:
theorem max_workers
  (n : ℕ) -- Let \( n \) be the number of workers at the brick factory.
  (h : worker_bricks_produced_per_day * n - worker_bricks_stolen_per_day n + 13 ≥ factory_brick_surplus_requirement): 
  n = 8 := 
sorry

end NUMINAMATH_GPT_max_workers_l2206_220698


namespace NUMINAMATH_GPT_find_m_in_function_l2206_220684

noncomputable def f (m : ℝ) (x : ℝ) := (1 / 3) * x^3 - x^2 - x + m

theorem find_m_in_function {m : ℝ} (h : ∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f m x ≥ (1/3)) :
  m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_in_function_l2206_220684


namespace NUMINAMATH_GPT_subset_to_union_eq_l2206_220608

open Set

variable {α : Type*} (A B : Set α)

theorem subset_to_union_eq (h : A ∩ B = A) : A ∪ B = B :=
by
  sorry

end NUMINAMATH_GPT_subset_to_union_eq_l2206_220608


namespace NUMINAMATH_GPT_arccos_of_one_over_sqrt_two_l2206_220656

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_arccos_of_one_over_sqrt_two_l2206_220656


namespace NUMINAMATH_GPT_additional_height_last_two_floors_l2206_220603

-- Definitions of the problem conditions
def num_floors : ℕ := 20
def height_per_floor : ℕ := 3
def building_total_height : ℤ := 61

-- Condition on the height of first 18 floors
def height_first_18_floors : ℤ := 18 * 3

-- Height of the last two floors
def height_last_two_floors : ℤ := building_total_height - height_first_18_floors
def height_each_last_two_floor : ℤ := height_last_two_floors / 2

-- Height difference between the last two floors and the first 18 floors
def additional_height : ℤ := height_each_last_two_floor - height_per_floor

-- Theorem to prove
theorem additional_height_last_two_floors :
  additional_height = 1 / 2 := 
sorry

end NUMINAMATH_GPT_additional_height_last_two_floors_l2206_220603


namespace NUMINAMATH_GPT_flat_fee_shipping_l2206_220663

theorem flat_fee_shipping (w : ℝ) (c : ℝ) (C : ℝ) (F : ℝ) 
  (h_w : w = 5) 
  (h_c : c = 0.80) 
  (h_C : C = 9)
  (h_shipping : C = F + (c * w)) :
  F = 5 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_flat_fee_shipping_l2206_220663


namespace NUMINAMATH_GPT_find_m_l2206_220694

noncomputable def given_hyperbola (x y : ℝ) (m : ℝ) : Prop :=
    x^2 / m - y^2 / 3 = 1

noncomputable def hyperbola_eccentricity (m : ℝ) (e : ℝ) : Prop :=
    e = Real.sqrt (1 + 3 / m)

theorem find_m (m : ℝ) (h1 : given_hyperbola 1 1 m) (h2 : hyperbola_eccentricity m 2) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2206_220694


namespace NUMINAMATH_GPT_gcd_48_72_120_l2206_220659

theorem gcd_48_72_120 : Nat.gcd (Nat.gcd 48 72) 120 = 24 :=
by
  sorry

end NUMINAMATH_GPT_gcd_48_72_120_l2206_220659


namespace NUMINAMATH_GPT_percent_increase_first_quarter_l2206_220614

theorem percent_increase_first_quarter (S : ℝ) (P : ℝ) :
  (S * 1.75 = (S + (P / 100) * S) * 1.346153846153846) → P = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percent_increase_first_quarter_l2206_220614


namespace NUMINAMATH_GPT_alyssa_games_this_year_l2206_220686

theorem alyssa_games_this_year : 
    ∀ (X: ℕ), 
    (13 + X + 15 = 39) → 
    X = 11 := 
by
  intros X h
  have h₁ : 13 + 15 = 28 := by norm_num
  have h₂ : X + 28 = 39 := by linarith
  have h₃ : X = 11 := by linarith
  exact h₃

end NUMINAMATH_GPT_alyssa_games_this_year_l2206_220686


namespace NUMINAMATH_GPT_positive_difference_of_numbers_l2206_220651

theorem positive_difference_of_numbers :
  ∃ x y : ℕ, x + y = 50 ∧ 3 * y - 4 * x = 10 ∧ y - x = 10 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_numbers_l2206_220651


namespace NUMINAMATH_GPT_parallel_segment_length_l2206_220699

/-- In \( \triangle ABC \), given side lengths AB = 500, BC = 550, and AC = 650,
there exists an interior point P such that each segment drawn parallel to the
sides of the triangle and passing through P splits the sides into segments proportional
to the overall sides of the triangle. Prove that the length \( d \) of each segment
parallel to the sides is 28.25 -/
theorem parallel_segment_length
  (A B C P : Type)
  (d AB BC AC : ℝ)
  (ha : AB = 500)
  (hb : BC = 550)
  (hc : AC = 650)
  (hp : AB * BC = AC * 550) -- This condition ensures proportionality of segments
  : d = 28.25 :=
sorry

end NUMINAMATH_GPT_parallel_segment_length_l2206_220699


namespace NUMINAMATH_GPT_parabola_intersection_ratios_l2206_220682

noncomputable def parabola_vertex_x1 (a b c : ℝ) := -b / (2 * a)
noncomputable def parabola_vertex_y1 (a b c : ℝ) := (4 * a * c - b^2) / (4 * a)
noncomputable def parabola_vertex_x2 (a d e : ℝ) := d / (2 * a)
noncomputable def parabola_vertex_y2 (a d e : ℝ) := (4 * a * e + d^2) / (4 * a)

theorem parabola_intersection_ratios
  (a b c d e : ℝ)
  (h1 : 144 * a + 12 * b + c = 21)
  (h2 : 784 * a + 28 * b + c = 3)
  (h3 : -144 * a + 12 * d + e = 21)
  (h4 : -784 * a + 28 * d + e = 3) :
  (parabola_vertex_x1 a b c + parabola_vertex_x2 a d e) / 
  (parabola_vertex_y1 a b c + parabola_vertex_y2 a d e) = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_parabola_intersection_ratios_l2206_220682


namespace NUMINAMATH_GPT_common_difference_range_l2206_220693

noncomputable def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

theorem common_difference_range :
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  (a10 > 0) ∧ (a9 <= 0) → 8 / 3 < d ∧ d <= 3 :=
by
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  intro h
  sorry

end NUMINAMATH_GPT_common_difference_range_l2206_220693


namespace NUMINAMATH_GPT_length_CK_angle_BCA_l2206_220675

variables {A B C O O₁ O₂ K K₁ K₂ K₃ : Point}
variables {r R : ℝ}
variables {AC CK AK₁ AK₂ : ℝ}

-- Definitions and conditions
def triangle_ABC (A B C : Point) : Prop := True
def incenter (A B C O : Point) : Prop := True
def in_radius_is_equal (O₁ O₂ : Point) (r : ℝ) : Prop := True
def circle_touches_side (circle_center : Point) (side_point : Point) (distance : ℝ) : Prop := True
def circumcenter (A C B O₁ : Point) : Prop := True
def angle (A B C : Point) (θ : ℝ) : Prop := True

-- Conditions from the problem
axiom cond1 : triangle_ABC A B C
axiom cond2 : in_radius_is_equal O₁ O₂ r
axiom cond3 : incenter A B C O
axiom cond4 : circle_touches_side O₁ K₁ 6
axiom cond5 : circle_touches_side O₂ K₂ 8
axiom cond6 : AC = 21
axiom cond7 : circle_touches_side O K 9
axiom cond8 : circumcenter O K₁ K₃ O₁

-- Statements to prove
theorem length_CK : CK = 9 := by
  sorry

theorem angle_BCA : angle B C A 60 := by
  sorry

end NUMINAMATH_GPT_length_CK_angle_BCA_l2206_220675


namespace NUMINAMATH_GPT_rectangle_breadth_l2206_220644

theorem rectangle_breadth (length radius side breadth: ℝ)
  (h1: length = (2/5) * radius)
  (h2: radius = side)
  (h3: side ^ 2 = 1600)
  (h4: length * breadth = 160) :
  breadth = 10 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_breadth_l2206_220644


namespace NUMINAMATH_GPT_problem_solution_l2206_220669

-- Define the variables and the conditions
variable (a b c : ℝ)
axiom h1 : a^2 + 2 * b = 7
axiom h2 : b^2 - 2 * c = -1
axiom h3 : c^2 - 6 * a = -17

-- State the theorem to be proven
theorem problem_solution : a + b + c = 3 := 
by sorry

end NUMINAMATH_GPT_problem_solution_l2206_220669
