import Mathlib

namespace NUMINAMATH_GPT_late_fisherman_arrival_l179_17942

-- Definitions of conditions
variables (n d : ℕ) -- n is the number of fishermen on Monday, d is the number of days the late fisherman fished
variable (total_fish : ℕ := 370)
variable (fish_per_day_per_fisherman : ℕ := 10)
variable (days_fished : ℕ := 5) -- From Monday to Friday

-- Condition in Lean: total fish caught from Monday to Friday
def total_fish_caught (n d : ℕ) := 50 * n + 10 * d

theorem late_fisherman_arrival (n d : ℕ) (h : total_fish_caught n d = 370) : 
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_late_fisherman_arrival_l179_17942


namespace NUMINAMATH_GPT_box_volume_increase_l179_17901

theorem box_volume_increase (l w h : ℝ)
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
  sorry

end NUMINAMATH_GPT_box_volume_increase_l179_17901


namespace NUMINAMATH_GPT_tangent_line_ellipse_l179_17962

variable (a b x0 y0 : ℝ)
variable (x y : ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

theorem tangent_line_ellipse :
  ellipse x y a b ∧ a > b ∧ (x0 ≠ 0 ∨ y0 ≠ 0) ∧ (x0 ^ 2) / (a ^ 2) + (y0 ^ 2) / (b ^ 2) > 1 →
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1 :=
  sorry

end NUMINAMATH_GPT_tangent_line_ellipse_l179_17962


namespace NUMINAMATH_GPT_no_solution_for_x_l179_17934

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345

theorem no_solution_for_x : proof_problem :=
  by
    intro x
    sorry

end NUMINAMATH_GPT_no_solution_for_x_l179_17934


namespace NUMINAMATH_GPT_condition_for_ellipse_l179_17914

theorem condition_for_ellipse (m : ℝ) : 
  (3 < m ∧ m < 7) ↔ (7 - m > 0 ∧ m - 3 > 0 ∧ (7 - m) ≠ (m - 3)) :=
by sorry

end NUMINAMATH_GPT_condition_for_ellipse_l179_17914


namespace NUMINAMATH_GPT_find_triplets_geometric_and_arithmetic_prog_l179_17938

theorem find_triplets_geometric_and_arithmetic_prog :
  ∃ a1 a2 b1 b2,
    (a2 = a1 * ((12:ℚ) / a1) ∧ 12 = a1 * ((12:ℚ) / a1)^2) ∧
    (b2 = b1 + ((9:ℚ) - b1) / 2 ∧ 9 = b1 + 2 * (((9:ℚ) - b1) / 2)) ∧
    ((a1 = b1) ∧ (a2 = b2)) ∧ 
    (∀ (a1 a2 : ℚ), ((a1 = -9) ∧ (a2 = -6)) ∨ ((a1 = 15) ∧ (a2 = 12))) :=
by sorry

end NUMINAMATH_GPT_find_triplets_geometric_and_arithmetic_prog_l179_17938


namespace NUMINAMATH_GPT_original_speed_of_person_B_l179_17945

-- Let v_A and v_B be the speeds of person A and B respectively
variable (v_A v_B : ℝ)

-- Conditions for problem
axiom initial_ratio : v_A / v_B = (5 / 4 * v_A) / (v_B + 10)

-- The goal: Prove that v_B = 40
theorem original_speed_of_person_B : v_B = 40 := 
  sorry

end NUMINAMATH_GPT_original_speed_of_person_B_l179_17945


namespace NUMINAMATH_GPT_complex_power_difference_l179_17960

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 40 - (1 - i) ^ 40 = 0 := by 
  sorry

end NUMINAMATH_GPT_complex_power_difference_l179_17960


namespace NUMINAMATH_GPT_geometric_seq_sum_l179_17976

theorem geometric_seq_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1_pos : a 1 > 0)
  (h_a4_7 : a 4 + a 7 = 2)
  (h_a5_6 : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 := 
sorry

end NUMINAMATH_GPT_geometric_seq_sum_l179_17976


namespace NUMINAMATH_GPT_total_amount_paid_l179_17920

theorem total_amount_paid (grapes_kg mangoes_kg rate_grapes rate_mangoes : ℕ) 
    (h1 : grapes_kg = 8) (h2 : mangoes_kg = 8) 
    (h3 : rate_grapes = 70) (h4 : rate_mangoes = 55) : 
    (grapes_kg * rate_grapes + mangoes_kg * rate_mangoes) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l179_17920


namespace NUMINAMATH_GPT_no_four_distinct_integers_with_product_plus_2006_perfect_square_l179_17964

theorem no_four_distinct_integers_with_product_plus_2006_perfect_square : 
  ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ k1 k2 k3 k4 k5 k6 : ℕ, a * b + 2006 = k1^2 ∧ 
                          a * c + 2006 = k2^2 ∧ 
                          a * d + 2006 = k3^2 ∧ 
                          b * c + 2006 = k4^2 ∧ 
                          b * d + 2006 = k5^2 ∧ 
                          c * d + 2006 = k6^2) := 
sorry

end NUMINAMATH_GPT_no_four_distinct_integers_with_product_plus_2006_perfect_square_l179_17964


namespace NUMINAMATH_GPT_find_angle_C_find_sum_a_b_l179_17961

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = 7 / 2 ∧
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 ∧
  (Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1))

theorem find_angle_C (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : C = Real.pi / 3 :=
  sorry

theorem find_sum_a_b (a b c A B C : ℝ) (h : triangle_condition a b c A B C) (hC : C = Real.pi / 3) : a + b = 11 / 2 :=
  sorry

end NUMINAMATH_GPT_find_angle_C_find_sum_a_b_l179_17961


namespace NUMINAMATH_GPT_distinct_integer_pairs_l179_17952

theorem distinct_integer_pairs :
  ∃ pairs : (Nat × Nat) → Prop,
  (∀ x y : Nat, pairs (x, y) → 0 < x ∧ x < y ∧ (8 * Real.sqrt 31 = Real.sqrt x + Real.sqrt y))
  ∧ (∃! p, pairs p) → (∃! q, pairs q) → (∃! r, pairs r) → true := sorry

end NUMINAMATH_GPT_distinct_integer_pairs_l179_17952


namespace NUMINAMATH_GPT_euler_totient_inequality_l179_17957

variable {n : ℕ}
def even (n : ℕ) := ∃ k : ℕ, n = 2 * k
def positive (n : ℕ) := n > 0

theorem euler_totient_inequality (h_even : even n) (h_positive : positive n) : 
  Nat.totient n ≤ n / 2 :=
sorry

end NUMINAMATH_GPT_euler_totient_inequality_l179_17957


namespace NUMINAMATH_GPT_diameter_twice_radius_l179_17954

theorem diameter_twice_radius (r d : ℝ) (h : d = 2 * r) : d = 2 * r :=
by
  exact h

end NUMINAMATH_GPT_diameter_twice_radius_l179_17954


namespace NUMINAMATH_GPT_max_last_place_score_l179_17921

theorem max_last_place_score (n : ℕ) (h : n ≥ 4) :
  ∃ k, (∀ m, m < n -> (k + m) < (n * 3)) ∧ 
     (∀ i, ∃ j, j < n ∧ i = k + j) ∧
     (n * 2 - 2) = (k + n - 1) ∧ 
     k = n - 2 := 
sorry

end NUMINAMATH_GPT_max_last_place_score_l179_17921


namespace NUMINAMATH_GPT_chord_bisected_by_point_l179_17924

theorem chord_bisected_by_point (x y : ℝ) (h : (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ (∀ x y : ℝ, (a * x + b * y + c = 0 ↔ (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1)) := by
  sorry

end NUMINAMATH_GPT_chord_bisected_by_point_l179_17924


namespace NUMINAMATH_GPT_diamondsuit_result_l179_17925

def diam (a b : ℕ) : ℕ := a

theorem diamondsuit_result : (diam 7 (diam 4 8)) = 7 :=
by sorry

end NUMINAMATH_GPT_diamondsuit_result_l179_17925


namespace NUMINAMATH_GPT_arc_length_l179_17944

theorem arc_length (circumference : ℝ) (angle_degrees : ℝ) (h : circumference = 90) (θ : angle_degrees = 45) :
  (angle_degrees / 360) * circumference = 11.25 := 
  by 
    sorry

end NUMINAMATH_GPT_arc_length_l179_17944


namespace NUMINAMATH_GPT_work_days_l179_17927

theorem work_days (p_can : ℕ → ℝ) (q_can : ℕ → ℝ) (together_can: ℕ → ℝ) :
  (together_can 6 = 1) ∧ (q_can 10 = 1) → (1 / (p_can x) + 1 / (q_can 10) = 1 / (together_can 6)) → (x = 15) :=
by
  sorry

end NUMINAMATH_GPT_work_days_l179_17927


namespace NUMINAMATH_GPT_solve_equation_l179_17975

theorem solve_equation :
  ∀ x : ℝ,
  (1 / (x^2 + 12 * x - 9) + 
   1 / (x^2 + 3 * x - 9) + 
   1 / (x^2 - 12 * x - 9) = 0) ↔ 
  (x = 1 ∨ x = -9 ∨ x = 3 ∨ x = -3) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l179_17975


namespace NUMINAMATH_GPT_sum_of_coefficients_l179_17910

theorem sum_of_coefficients (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℝ) :
  (5 * 1 - 2)^6 = b_6 * 1^6 + b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0
  → b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 = 729 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l179_17910


namespace NUMINAMATH_GPT_sequence_m_l179_17941

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We usually start sequences from n = 1; hence, a_0 is irrelevant
  else (n * n) - n + 1

theorem sequence_m (m : ℕ) (h_positive : m > 0) (h_bound : 43 < a m ∧ a m < 73) : m = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_m_l179_17941


namespace NUMINAMATH_GPT_horse_food_calculation_l179_17904

theorem horse_food_calculation
  (num_sheep : ℕ)
  (ratio_sheep_horses : ℕ)
  (total_horse_food : ℕ)
  (H : ℕ)
  (num_sheep_eq : num_sheep = 56)
  (ratio_eq : ratio_sheep_horses = 7)
  (total_food_eq : total_horse_food = 12880)
  (num_horses : H = num_sheep * 1 / ratio_sheep_horses)
  : num_sheep = ratio_sheep_horses → total_horse_food / H = 230 :=
by
  sorry

end NUMINAMATH_GPT_horse_food_calculation_l179_17904


namespace NUMINAMATH_GPT_surfers_ratio_l179_17926

theorem surfers_ratio (S1 : ℕ) (S3 : ℕ) : S1 = 1500 → 
  (∀ S2 : ℕ, S2 = S1 + 600 → (1400 * 3 = S1 + S2 + S3) → 
  S3 = 600) → (S3 / S1 = 2 / 5) :=
sorry

end NUMINAMATH_GPT_surfers_ratio_l179_17926


namespace NUMINAMATH_GPT_solve_for_x_l179_17949

-- Define the custom operation
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Main statement to prove
theorem solve_for_x : (∃ x : ℝ, custom_mul 3 (custom_mul 4 x) = 10) ↔ (x = 7.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l179_17949


namespace NUMINAMATH_GPT_find_m_n_difference_l179_17911

theorem find_m_n_difference (x y m n : ℤ)
  (hx : x = 2)
  (hy : y = -3)
  (hm : x + y = m)
  (hn : 2 * x - y = n) :
  m - n = -8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_n_difference_l179_17911


namespace NUMINAMATH_GPT_A_plus_B_zero_l179_17907

def f (A B x : ℝ) : ℝ := 3 * A * x + 2 * B
def g (A B x : ℝ) : ℝ := 2 * B * x + 3 * A

theorem A_plus_B_zero (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = 3 * (B - A)) :
  A + B = 0 :=
sorry

end NUMINAMATH_GPT_A_plus_B_zero_l179_17907


namespace NUMINAMATH_GPT_seven_digit_divisible_by_11_l179_17933

def is_digit (d : ℕ) : Prop := d ≤ 9

def valid7DigitNumber (b n : ℕ) : Prop :=
  let sum_odd := 3 + 5 + 6
  let sum_even := b + n + 7 + 8
  let diff := sum_odd - sum_even
  diff % 11 = 0

theorem seven_digit_divisible_by_11 (b n : ℕ) (hb : is_digit b) (hn : is_digit n)
  (h_valid : valid7DigitNumber b n) : b + n = 10 := 
sorry

end NUMINAMATH_GPT_seven_digit_divisible_by_11_l179_17933


namespace NUMINAMATH_GPT_complex_number_evaluation_l179_17985

noncomputable def i := Complex.I

theorem complex_number_evaluation :
  (1 - i) * (i * i) / (1 + 2 * i) = (1/5 : ℂ) + (3/5 : ℂ) * i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_evaluation_l179_17985


namespace NUMINAMATH_GPT_product_of_odd_and_even_is_odd_l179_17940

theorem product_of_odd_and_even_is_odd {f g : ℝ → ℝ} 
  (hf : ∀ x : ℝ, f (-x) = -f x)
  (hg : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x) * (g x) = -(f (-x) * g (-x)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_odd_and_even_is_odd_l179_17940


namespace NUMINAMATH_GPT_tournament_rounds_l179_17991

/-- 
Given a tournament where each participant plays several games with every other participant
and a total of 224 games were played, prove that the number of rounds in the competition is 8.
-/
theorem tournament_rounds (x y : ℕ) (hx : x > 1) (hy : y > 0) (h : x * (x - 1) * y = 448) : y = 8 :=
sorry

end NUMINAMATH_GPT_tournament_rounds_l179_17991


namespace NUMINAMATH_GPT_roots_of_polynomial_l179_17999

theorem roots_of_polynomial : ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_roots_of_polynomial_l179_17999


namespace NUMINAMATH_GPT_time_for_completion_l179_17989

noncomputable def efficiency_b : ℕ := 100

noncomputable def efficiency_a := 130

noncomputable def total_work := efficiency_a * 23

noncomputable def combined_efficiency := efficiency_a + efficiency_b

noncomputable def time_taken := total_work / combined_efficiency

theorem time_for_completion (h1 : efficiency_a = 130)
                           (h2 : efficiency_b = 100)
                           (h3 : total_work = 2990)
                           (h4 : combined_efficiency = 230) :
  time_taken = 13 := by
  sorry

end NUMINAMATH_GPT_time_for_completion_l179_17989


namespace NUMINAMATH_GPT_length_of_room_l179_17997

theorem length_of_room 
  (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
  (h_width : width = 3.75) 
  (h_total_cost : total_cost = 16500) 
  (h_rate_per_sq_meter : rate_per_sq_meter = 800) : 
  ∃ length : ℝ, length = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_room_l179_17997


namespace NUMINAMATH_GPT_find_k_l179_17979

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_l179_17979


namespace NUMINAMATH_GPT_find_nonzero_c_l179_17968

def quadratic_has_unique_solution (c b : ℝ) : Prop :=
  (b^4 + (1 - 4 * c) * b^2 + 1 = 0) ∧ (1 - 4 * c)^2 - 4 = 0

theorem find_nonzero_c (c : ℝ) (b : ℝ) (h_nonzero : c ≠ 0) (h_unique_sol : quadratic_has_unique_solution c b) : 
  c = 3 / 4 := 
sorry

end NUMINAMATH_GPT_find_nonzero_c_l179_17968


namespace NUMINAMATH_GPT_series_sum_eq_50_l179_17998

noncomputable def series_sum (x : ℝ) : ℝ :=
  2 + 6 * x + 10 * x^2 + 14 * x^3 -- This represents the series

theorem series_sum_eq_50 : 
  ∃ x : ℝ, series_sum x = 50 ∧ x = 0.59 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_eq_50_l179_17998


namespace NUMINAMATH_GPT_num_green_hats_l179_17909

-- Definitions
def total_hats : ℕ := 85
def blue_hat_cost : ℕ := 6
def green_hat_cost : ℕ := 7
def total_cost : ℕ := 548

-- Prove the number of green hats (g) is 38 given the conditions
theorem num_green_hats (b g : ℕ) 
  (h₁ : b + g = total_hats)
  (h₂ : blue_hat_cost * b + green_hat_cost * g = total_cost) : 
  g = 38 := by
  sorry

end NUMINAMATH_GPT_num_green_hats_l179_17909


namespace NUMINAMATH_GPT_final_total_cost_l179_17987

def initial_spiral_cost : ℝ := 15
def initial_planner_cost : ℝ := 10
def spiral_discount_rate : ℝ := 0.20
def planner_discount_rate : ℝ := 0.15
def num_spirals : ℝ := 4
def num_planners : ℝ := 8
def sales_tax_rate : ℝ := 0.07

theorem final_total_cost :
  let discounted_spiral_cost := initial_spiral_cost * (1 - spiral_discount_rate)
  let discounted_planner_cost := initial_planner_cost * (1 - planner_discount_rate)
  let total_before_tax := num_spirals * discounted_spiral_cost + num_planners * discounted_planner_cost
  let total_tax := total_before_tax * sales_tax_rate
  let total_cost := total_before_tax + total_tax
  total_cost = 124.12 :=
by
  sorry

end NUMINAMATH_GPT_final_total_cost_l179_17987


namespace NUMINAMATH_GPT_jane_babysitting_start_l179_17906

-- Definitions based on the problem conditions
def jane_current_age := 32
def years_since_babysitting := 10
def oldest_current_child_age := 24

-- Definition for the starting babysitting age
def starting_babysitting_age : ℕ := 8

-- Theorem statement to prove
theorem jane_babysitting_start (h1 : jane_current_age - years_since_babysitting = 22)
  (h2 : oldest_current_child_age - years_since_babysitting = 14)
  (h3 : ∀ (age_jane age_child : ℕ), age_child ≤ age_jane / 2) :
  starting_babysitting_age = 8 :=
by
  sorry

end NUMINAMATH_GPT_jane_babysitting_start_l179_17906


namespace NUMINAMATH_GPT_national_flag_length_l179_17908

-- Definitions from the conditions specified in the problem
def width : ℕ := 128
def ratio_length_to_width (L W : ℕ) : Prop := L / W = 3 / 2

-- The main theorem to prove
theorem national_flag_length (L : ℕ) (H : ratio_length_to_width L width) : L = 192 :=
by
  sorry

end NUMINAMATH_GPT_national_flag_length_l179_17908


namespace NUMINAMATH_GPT_books_inequality_system_l179_17922

theorem books_inequality_system (x : ℕ) (n : ℕ) (h1 : x = 5 * n + 6) (h2 : (1 ≤ x - 7 * (x - 6) / 5 + 7)) :
  1 ≤ x - 7 * (x - 6) / 5 + 7 ∧ x - 7 * (x - 6) / 5 + 7 < 7 := 
by
  sorry

end NUMINAMATH_GPT_books_inequality_system_l179_17922


namespace NUMINAMATH_GPT_intersection_eq_T_l179_17978

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end NUMINAMATH_GPT_intersection_eq_T_l179_17978


namespace NUMINAMATH_GPT_union_A_B_complement_A_l179_17929

-- Definition of Universe U
def U : Set ℝ := Set.univ

-- Definition of set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Definition of set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem 1: Proving the union A ∪ B
theorem union_A_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := 
sorry

-- Theorem 2: Proving the complement of A with respect to U
theorem complement_A : (U \ A) = {x | x < -1 ∨ x > 3} := 
sorry

end NUMINAMATH_GPT_union_A_B_complement_A_l179_17929


namespace NUMINAMATH_GPT_actual_number_of_children_l179_17937

theorem actual_number_of_children (N : ℕ) (B : ℕ) 
  (h1 : B = 2 * N)
  (h2 : ∀ k : ℕ, k = N - 330)
  (h3 : B = 4 * (N - 330)) : 
  N = 660 :=
by 
  sorry

end NUMINAMATH_GPT_actual_number_of_children_l179_17937


namespace NUMINAMATH_GPT_trajectory_of_M_l179_17982

variables {x y : ℝ}

theorem trajectory_of_M (h : y / (x + 2) + y / (x - 2) = 2) (hx : x ≠ 2) (hx' : x ≠ -2) :
  x * y - x^2 + 4 = 0 :=
by sorry

end NUMINAMATH_GPT_trajectory_of_M_l179_17982


namespace NUMINAMATH_GPT_find_m_l179_17990

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 5 * y + 20 = 0
def l2 (m x y : ℝ) : Prop := m * x + 2 * y - 10 = 0

-- Define the condition of perpendicularity
def lines_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Proving the value of m given the conditions
theorem find_m (m : ℝ) :
  (∃ x y : ℝ, l1 x y) → (∃ x y : ℝ, l2 m x y) → lines_perpendicular 2 (-5 : ℝ) m 2 → m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l179_17990


namespace NUMINAMATH_GPT_ratio_of_screws_l179_17994

def initial_screws : Nat := 8
def total_required_screws : Nat := 4 * 6
def screws_to_buy : Nat := total_required_screws - initial_screws

theorem ratio_of_screws :
  (screws_to_buy : ℚ) / initial_screws = 2 :=
by
  simp [initial_screws, total_required_screws, screws_to_buy]
  sorry

end NUMINAMATH_GPT_ratio_of_screws_l179_17994


namespace NUMINAMATH_GPT_sonya_falls_6_l179_17977

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end NUMINAMATH_GPT_sonya_falls_6_l179_17977


namespace NUMINAMATH_GPT_point_on_line_and_equidistant_l179_17900

theorem point_on_line_and_equidistant {x y : ℝ} :
  (4 * x + 3 * y = 12) ∧ (x = y) ∧ (x ≥ 0) ∧ (y ≥ 0) ↔ x = 12 / 7 ∧ y = 12 / 7 :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_and_equidistant_l179_17900


namespace NUMINAMATH_GPT_remainder_of_multiple_l179_17973

theorem remainder_of_multiple (m k : ℤ) (h1 : m % 5 = 2) (h2 : (2 * k) % 5 = 1) : 
  (k * m) % 5 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_of_multiple_l179_17973


namespace NUMINAMATH_GPT_distance_first_to_last_tree_l179_17943

theorem distance_first_to_last_tree 
    (n_trees : ℕ) 
    (distance_first_to_fifth : ℕ)
    (h1 : n_trees = 8)
    (h2 : distance_first_to_fifth = 80) 
    : ∃ distance_first_to_last, distance_first_to_last = 140 := by
  sorry

end NUMINAMATH_GPT_distance_first_to_last_tree_l179_17943


namespace NUMINAMATH_GPT_demand_change_for_revenue_l179_17972

theorem demand_change_for_revenue (P D D' : ℝ)
  (h1 : D' = (1.10 * D) / 1.20)
  (h2 : P' = 1.20 * P)
  (h3 : P * D = P' * D') :
  (D' - D) / D * 100 = -8.33 := by
sorry

end NUMINAMATH_GPT_demand_change_for_revenue_l179_17972


namespace NUMINAMATH_GPT_total_capacity_of_schools_l179_17902

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end NUMINAMATH_GPT_total_capacity_of_schools_l179_17902


namespace NUMINAMATH_GPT_circles_are_separate_l179_17983

def circle_center (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circles_are_separate :
  circle_center 0 0 1 x y → 
  circle_center 3 (-4) 3 x' y' →
  dist (0, 0) (3, -4) > 1 + 3 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_circles_are_separate_l179_17983


namespace NUMINAMATH_GPT_car_travel_first_hour_l179_17966

-- Define the conditions as variables and the ultimate equality to be proved
theorem car_travel_first_hour (x : ℕ) (h : 12 * x + 132 = 612) : x = 40 :=
by
  -- Proof will be completed here
  sorry

end NUMINAMATH_GPT_car_travel_first_hour_l179_17966


namespace NUMINAMATH_GPT_cupcakes_frosted_l179_17930

def Cagney_rate := 1 / 25
def Lacey_rate := 1 / 35
def time_duration := 600
def combined_rate := Cagney_rate + Lacey_rate
def total_cupcakes := combined_rate * time_duration

theorem cupcakes_frosted (Cagney_rate Lacey_rate time_duration combined_rate total_cupcakes : ℝ) 
  (hC: Cagney_rate = 1 / 25)
  (hL: Lacey_rate = 1 / 35)
  (hT: time_duration = 600)
  (hCR: combined_rate = Cagney_rate + Lacey_rate)
  (hTC: total_cupcakes = combined_rate * time_duration) :
  total_cupcakes = 41 :=
sorry

end NUMINAMATH_GPT_cupcakes_frosted_l179_17930


namespace NUMINAMATH_GPT_correct_multiplication_result_l179_17936

theorem correct_multiplication_result (x : ℕ) (h : 9 * x = 153) : 6 * x = 102 :=
by {
  -- We would normally provide a detailed proof here, but as per instruction, we add sorry.
  sorry
}

end NUMINAMATH_GPT_correct_multiplication_result_l179_17936


namespace NUMINAMATH_GPT_problem1_problem2_l179_17956

-- Assume x and y are positive numbers
variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

-- Prove that x^3 + y^3 >= x^2*y + y^2*x
theorem problem1 : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
by sorry

-- Prove that m ≤ 2 given the additional condition
variables (m : ℝ)
theorem problem2 (cond : (x/y^2 + y/x^2) ≥ m/2 * (1/x + 1/y)) : m ≤ 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l179_17956


namespace NUMINAMATH_GPT_mark_paired_with_mike_prob_l179_17953

def total_students := 16
def other_students := 15
def prob_pairing (mark: Nat) (mike: Nat) : ℚ := 1 / other_students

theorem mark_paired_with_mike_prob : prob_pairing 1 2 = 1 / 15 := 
sorry

end NUMINAMATH_GPT_mark_paired_with_mike_prob_l179_17953


namespace NUMINAMATH_GPT_range_of_a_l179_17955

def A (x : ℝ) : Prop := abs (x - 4) < 2 * x

def B (x a : ℝ) : Prop := x * (x - a) ≥ (a + 6) * (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) → a ≤ -14 / 3 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l179_17955


namespace NUMINAMATH_GPT_ribbon_leftover_correct_l179_17951

def initial_ribbon : ℕ := 84
def used_ribbon : ℕ := 46
def leftover_ribbon : ℕ := 38

theorem ribbon_leftover_correct : initial_ribbon - used_ribbon = leftover_ribbon :=
by
  sorry

end NUMINAMATH_GPT_ribbon_leftover_correct_l179_17951


namespace NUMINAMATH_GPT_problem_statement_l179_17946

   def f (a : ℤ) : ℤ := a - 2
   def F (a b : ℤ) : ℤ := b^2 + a

   theorem problem_statement : F 3 (f 4) = 7 := by
     sorry
   
end NUMINAMATH_GPT_problem_statement_l179_17946


namespace NUMINAMATH_GPT_radius_of_semicircle_l179_17903

theorem radius_of_semicircle (P : ℝ) (π_val : ℝ) (h1 : P = 162) (h2 : π_val = Real.pi) : 
  ∃ r : ℝ, r = 162 / (π + 2) :=
by
  use 162 / (Real.pi + 2)
  sorry

end NUMINAMATH_GPT_radius_of_semicircle_l179_17903


namespace NUMINAMATH_GPT_halfway_between_ratios_l179_17958

theorem halfway_between_ratios :
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_GPT_halfway_between_ratios_l179_17958


namespace NUMINAMATH_GPT_eval_expression_l179_17923

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l179_17923


namespace NUMINAMATH_GPT_inequality_comparison_l179_17932

theorem inequality_comparison 
  (a b : ℝ) (x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) :
  a^2 + b^2 ≥ (x + y)^2 :=
sorry

end NUMINAMATH_GPT_inequality_comparison_l179_17932


namespace NUMINAMATH_GPT_coloring_ways_l179_17988

-- Definitions of the problem:
def column1 := 1
def column2 := 2
def column3 := 3
def column4 := 4
def column5 := 3
def column6 := 2
def column7 := 1
def total_colors := 3 -- Blue, Yellow, Green

-- Adjacent coloring constraints:
def adjacent_constraints (c1 c2 : ℕ) : Prop := c1 ≠ c2

-- Number of ways to color figure:
theorem coloring_ways : 
  (∃ (n : ℕ), n = 2^5) ∧ 
  n = 32 :=
by 
  sorry

end NUMINAMATH_GPT_coloring_ways_l179_17988


namespace NUMINAMATH_GPT_cost_of_two_books_and_one_magazine_l179_17947

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := 3 * x + 2 * y = 18.40
def condition2 (x y : ℝ) : Prop := 2 * x + 3 * y = 17.60

-- Proof problem
theorem cost_of_two_books_and_one_magazine (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  2 * x + y = 11.20 :=
sorry

end NUMINAMATH_GPT_cost_of_two_books_and_one_magazine_l179_17947


namespace NUMINAMATH_GPT_cost_price_per_meter_l179_17916

-- Given conditions
def total_selling_price : ℕ := 18000
def total_meters_sold : ℕ := 400
def loss_per_meter : ℕ := 5

-- Statement to be proven
theorem cost_price_per_meter : 
    ((total_selling_price + (loss_per_meter * total_meters_sold)) / total_meters_sold) = 50 := 
by
    sorry

end NUMINAMATH_GPT_cost_price_per_meter_l179_17916


namespace NUMINAMATH_GPT_fixed_point_through_1_neg2_l179_17993

noncomputable def fixed_point (a : ℝ) (x : ℝ) : ℝ :=
a^(x - 1) - 3

-- The statement to prove
theorem fixed_point_through_1_neg2 (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  fixed_point a 1 = -2 :=
by
  unfold fixed_point
  sorry

end NUMINAMATH_GPT_fixed_point_through_1_neg2_l179_17993


namespace NUMINAMATH_GPT_smallest_number_remainder_problem_l179_17970

theorem smallest_number_remainder_problem :
  ∃ N : ℕ, (N % 13 = 2) ∧ (N % 15 = 4) ∧ (∀ n : ℕ, (n % 13 = 2 ∧ n % 15 = 4) → n ≥ N) :=
sorry

end NUMINAMATH_GPT_smallest_number_remainder_problem_l179_17970


namespace NUMINAMATH_GPT_product_expression_evaluates_to_32_l179_17963

theorem product_expression_evaluates_to_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  -- The proof itself is not required, hence we can put sorry here
  sorry

end NUMINAMATH_GPT_product_expression_evaluates_to_32_l179_17963


namespace NUMINAMATH_GPT_smallest_N_divisors_of_8_l179_17912

theorem smallest_N_divisors_of_8 (N : ℕ) (h0 : N % 10 = 0) (h8 : ∃ (divisors : ℕ), divisors = 8 ∧ (∀ k, k ∣ N → k ≤ divisors)) : N = 30 := 
sorry

end NUMINAMATH_GPT_smallest_N_divisors_of_8_l179_17912


namespace NUMINAMATH_GPT_Corey_goal_reachable_l179_17931

theorem Corey_goal_reachable :
  ∀ (goal balls_found_saturday balls_found_sunday additional_balls : ℕ),
    goal = 48 →
    balls_found_saturday = 16 →
    balls_found_sunday = 18 →
    additional_balls = goal - (balls_found_saturday + balls_found_sunday) →
    additional_balls = 14 :=
by
  intros goal balls_found_saturday balls_found_sunday additional_balls
  intro goal_eq
  intro saturday_eq
  intro sunday_eq
  intro additional_eq
  sorry

end NUMINAMATH_GPT_Corey_goal_reachable_l179_17931


namespace NUMINAMATH_GPT_sum_of_digits_8_pow_2003_l179_17974

noncomputable def units_digit (n : ℕ) : ℕ :=
n % 10

noncomputable def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

noncomputable def sum_of_tens_and_units_digits (n : ℕ) : ℕ :=
units_digit n + tens_digit n

theorem sum_of_digits_8_pow_2003 :
  sum_of_tens_and_units_digits (8 ^ 2003) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_8_pow_2003_l179_17974


namespace NUMINAMATH_GPT_zero_of_function_l179_17959

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

theorem zero_of_function (x : ℝ) (h : f x = 0) (x1 x2 : ℝ)
  (h1 : -1 < x1 ∧ x1 < x)
  (h2 : x < x2 ∧ x2 < 2) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_function_l179_17959


namespace NUMINAMATH_GPT_problem1_correctness_problem2_correctness_l179_17996

noncomputable def problem1_solution_1 (x : ℝ) : Prop := x = Real.sqrt 5 - 1
noncomputable def problem1_solution_2 (x : ℝ) : Prop := x = -Real.sqrt 5 - 1
noncomputable def problem2_solution_1 (x : ℝ) : Prop := x = 5
noncomputable def problem2_solution_2 (x : ℝ) : Prop := x = -1 / 3

theorem problem1_correctness (x : ℝ) :
  (x^2 + 2*x - 4 = 0) → (problem1_solution_1 x ∨ problem1_solution_2 x) :=
by sorry

theorem problem2_correctness (x : ℝ) :
  (3 * x * (x - 5) = 5 - x) → (problem2_solution_1 x ∨ problem2_solution_2 x) :=
by sorry

end NUMINAMATH_GPT_problem1_correctness_problem2_correctness_l179_17996


namespace NUMINAMATH_GPT_face_value_shares_l179_17905

theorem face_value_shares (market_value : ℝ) (dividend_rate desired_rate : ℝ) (FV : ℝ) 
  (h1 : dividend_rate = 0.09)
  (h2 : desired_rate = 0.12)
  (h3 : market_value = 36.00000000000001)
  (h4 : (dividend_rate * FV) = (desired_rate * market_value)) :
  FV = 48.00000000000001 :=
by
  sorry

end NUMINAMATH_GPT_face_value_shares_l179_17905


namespace NUMINAMATH_GPT_correct_statement_l179_17965

variables {α β γ : ℝ → ℝ → ℝ → Prop} -- planes
variables {a b c : ℝ → ℝ → ℝ → Prop} -- lines

def is_parallel (P Q : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, (P x y z → Q x y z) ∧ (Q x y z → P x y z)

def is_perpendicular (L : ℝ → ℝ → ℝ → Prop) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, L x y z ↔ ¬ P x y z 

theorem correct_statement : 
  (is_perpendicular a α) → 
  (is_parallel b β) → 
  (is_parallel α β) → 
  (is_perpendicular a b) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l179_17965


namespace NUMINAMATH_GPT_find_a_find_b_find_T_l179_17971

open Real

def S (n : ℕ) : ℝ := 2 * n^2 + n

def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 2^(n - 1)

def T (n : ℕ) : ℝ := (4 * n - 5) * 2^n + 5

theorem find_a (n : ℕ) (hn : n > 0) : a n = 4 * n - 1 :=
by sorry

theorem find_b (n : ℕ) (hn : n > 0) : b n = 2^(n-1) :=
by sorry

theorem find_T (n : ℕ) (hn : n > 0) (a_def : ∀ n, a n = 4 * n - 1) (b_def : ∀ n, b n = 2^(n-1)) : T n = (4 * n - 5) * 2^n + 5 :=
by sorry

end NUMINAMATH_GPT_find_a_find_b_find_T_l179_17971


namespace NUMINAMATH_GPT_base_five_to_decimal_l179_17935

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2 => 2 * 5^0
  | 3 => 3 * 5^1
  | 1 => 1 * 5^2
  | _ => 0

theorem base_five_to_decimal : base5_to_base10 2 + base5_to_base10 3 + base5_to_base10 1 = 42 :=
by sorry

end NUMINAMATH_GPT_base_five_to_decimal_l179_17935


namespace NUMINAMATH_GPT_metals_inductive_reasoning_l179_17950

def conducts_electricity (metal : String) : Prop :=
  metal = "Gold" ∨ metal = "Silver" ∨ metal = "Copper" ∨ metal = "Iron"

def all_metals_conduct_electricity (metals : List String) : Prop :=
  ∀ metal, metal ∈ metals → conducts_electricity metal

theorem metals_inductive_reasoning 
  (h1 : conducts_electricity "Gold")
  (h2 : conducts_electricity "Silver")
  (h3 : conducts_electricity "Copper")
  (h4 : conducts_electricity "Iron") :
  (all_metals_conduct_electricity ["Gold", "Silver", "Copper", "Iron"] → 
  all_metals_conduct_electricity ["All metals"]) :=
  sorry -- Proof skipped, as per instructions.

end NUMINAMATH_GPT_metals_inductive_reasoning_l179_17950


namespace NUMINAMATH_GPT_ratio_c_b_l179_17913

theorem ratio_c_b (x y a b c : ℝ) (h1 : x ≥ 1) (h2 : x + y ≤ 4) (h3 : a * x + b * y + c ≤ 0) 
    (h_max : ∀ x y, (x,y) = (2, 2) → 2 * x + y = 6) (h_min : ∀ x y, (x,y) = (1, -1) → 2 * x + y = 1) (h_b : b ≠ 0) :
    c / b = 4 := sorry

end NUMINAMATH_GPT_ratio_c_b_l179_17913


namespace NUMINAMATH_GPT_range_of_m_l179_17995

def A (x : ℝ) : Prop := 1/2 < x ∧ x < 1

def B (x : ℝ) (m : ℝ) : Prop := x^2 + 2 * x + 1 - m ≤ 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, A x → B x m) → 4 ≤ m := by
  sorry

end NUMINAMATH_GPT_range_of_m_l179_17995


namespace NUMINAMATH_GPT_chromosomes_mitosis_late_stage_l179_17986

/-- A biological cell with 24 chromosomes at the late stage of the second meiotic division. -/
def cell_chromosomes_meiosis_late_stage : ℕ := 24

/-- The number of chromosomes in this organism at the late stage of mitosis is double that at the late stage of the second meiotic division. -/
theorem chromosomes_mitosis_late_stage : cell_chromosomes_meiosis_late_stage * 2 = 48 :=
by
  -- We will add the necessary proof here.
  sorry

end NUMINAMATH_GPT_chromosomes_mitosis_late_stage_l179_17986


namespace NUMINAMATH_GPT_mr_brown_net_result_l179_17939

noncomputable def C1 := 1.50 / 1.3
noncomputable def C2 := 1.50 / 0.9
noncomputable def profit_from_first_pen := 1.50 - C1
noncomputable def tax := 0.05 * profit_from_first_pen
noncomputable def total_cost := C1 + C2
noncomputable def total_revenue := 3.00
noncomputable def net_result := total_revenue - total_cost - tax

theorem mr_brown_net_result : net_result = 0.16 :=
by
  sorry

end NUMINAMATH_GPT_mr_brown_net_result_l179_17939


namespace NUMINAMATH_GPT_new_ratio_of_partners_to_associates_l179_17980

theorem new_ratio_of_partners_to_associates
  (partners associates : ℕ)
  (rat_partners_associates : 2 * associates = 63 * partners)
  (partners_count : partners = 18)
  (add_assoc : associates + 45 = 612) :
  (partners:ℚ) / (associates + 45) = 1 / 34 :=
by
  -- Actual proof goes here
  sorry

end NUMINAMATH_GPT_new_ratio_of_partners_to_associates_l179_17980


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_at_two_points_l179_17967

theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) :
  (k < 1 ∧ k ≠ 0) ↔ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (kx1^2 + 2 * x1 + 1 = 0) ∧ (kx2^2 + 2 * x2 + 1 = 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_x_axis_at_two_points_l179_17967


namespace NUMINAMATH_GPT_not_possible_155_cents_five_coins_l179_17928

/-- It is not possible to achieve a total value of 155 cents using exactly five coins 
    from a piggy bank containing only pennies (1 cent), nickels (5 cents), 
    quarters (25 cents), and half-dollars (50 cents). -/
theorem not_possible_155_cents_five_coins (n_pennies n_nickels n_quarters n_half_dollars : ℕ) 
    (h : n_pennies + n_nickels + n_quarters + n_half_dollars = 5) : 
    n_pennies * 1 + n_nickels * 5 + n_quarters * 25 + n_half_dollars * 50 ≠ 155 := 
sorry

end NUMINAMATH_GPT_not_possible_155_cents_five_coins_l179_17928


namespace NUMINAMATH_GPT_solve_quadratic_sum_l179_17992

theorem solve_quadratic_sum (a b : ℕ) (x : ℝ) (h₁ : x^2 + 10 * x = 93)
  (h₂ : x = Real.sqrt a - b) (ha_pos : 0 < a) (hb_pos : 0 < b) : a + b = 123 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_sum_l179_17992


namespace NUMINAMATH_GPT_func_identity_equiv_l179_17917

theorem func_identity_equiv (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x) + f (y)) ↔ (∀ x y : ℝ, f (xy + x + y) = f (xy) + f (x) + f (y)) :=
by
  sorry

end NUMINAMATH_GPT_func_identity_equiv_l179_17917


namespace NUMINAMATH_GPT_parameterization_of_line_l179_17919

theorem parameterization_of_line (t : ℝ) (g : ℝ → ℝ) 
  (h : ∀ t, (g t - 10) / 2 = t ) :
  g t = 5 * t + 10 := by
  sorry

end NUMINAMATH_GPT_parameterization_of_line_l179_17919


namespace NUMINAMATH_GPT_min_a_plus_b_l179_17918

theorem min_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 := sorry

end NUMINAMATH_GPT_min_a_plus_b_l179_17918


namespace NUMINAMATH_GPT_river_flow_rate_l179_17948

noncomputable def volume_per_minute : ℝ := 3200
noncomputable def depth_of_river : ℝ := 3
noncomputable def width_of_river : ℝ := 32
noncomputable def cross_sectional_area : ℝ := depth_of_river * width_of_river

noncomputable def flow_rate_m_per_minute : ℝ := volume_per_minute / cross_sectional_area
-- Conversion factors
noncomputable def minutes_per_hour : ℝ := 60
noncomputable def meters_per_km : ℝ := 1000

noncomputable def flow_rate_kmph : ℝ := (flow_rate_m_per_minute * minutes_per_hour) / meters_per_km

theorem river_flow_rate :
  flow_rate_kmph = 2 :=
by
  -- We skip the proof and use sorry to focus on the statement structure.
  sorry

end NUMINAMATH_GPT_river_flow_rate_l179_17948


namespace NUMINAMATH_GPT_xiao_ying_performance_l179_17981

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50

def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

-- Define the function that calculates the weighted average
def semester_performance (rw mw fw rs ms fs : ℝ) : ℝ :=
  rw * rs + mw * ms + fw * fs

-- The theorem that the weighted average of the scores is 90
theorem xiao_ying_performance : semester_performance regular_weight midterm_weight final_weight regular_score midterm_score final_score = 90 := by
  sorry

end NUMINAMATH_GPT_xiao_ying_performance_l179_17981


namespace NUMINAMATH_GPT_largest_angle_sine_of_C_l179_17915

-- Given conditions
def side_a : ℝ := 7
def side_b : ℝ := 3
def side_c : ℝ := 5

-- 1. Prove the largest angle
theorem largest_angle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) : 
  ∃ A : ℝ, A = 120 :=
by
  sorry

-- 2. Prove the sine value of angle C
theorem sine_of_C (a b c A : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) (h₄ : A = 120) : 
  ∃ sinC : ℝ, sinC = 5 * (Real.sqrt 3) / 14 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_sine_of_C_l179_17915


namespace NUMINAMATH_GPT_avery_egg_cartons_filled_l179_17984

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end NUMINAMATH_GPT_avery_egg_cartons_filled_l179_17984


namespace NUMINAMATH_GPT_lights_on_after_2011_toggles_l179_17969

-- Definitions for light states and index of lights
inductive Light : Type
| A | B | C | D | E | F | G
deriving DecidableEq

-- Initial light state: function from Light to Bool (true means the light is on)
def initialState : Light → Bool
| Light.A => true
| Light.B => false
| Light.C => true
| Light.D => false
| Light.E => true
| Light.F => false
| Light.G => true

-- Toggling function: toggles the state of a given light
def toggleState (state : Light → Bool) (light : Light) : Light → Bool :=
  fun l => if l = light then ¬ (state l) else state l

-- Toggling sequence: sequentially toggle lights in the given list
def toggleSequence (state : Light → Bool) (seq : List Light) : Light → Bool :=
  seq.foldl toggleState state

-- Toggles the sequence n times
def toggleNTimes (state : Light → Bool) (seq : List Light) (n : Nat) : Light → Bool :=
  let rec aux (state : Light → Bool) (n : Nat) : Light → Bool :=
    if n = 0 then state
    else aux (toggleSequence state seq) (n - 1)
  aux state n

-- Toggling sequence: A, B, C, D, E, F, G
def toggleSeq : List Light := [Light.A, Light.B, Light.C, Light.D, Light.E, Light.F, Light.G]

-- Determine the final state after 2011 toggles
def finalState : Light → Bool := toggleNTimes initialState toggleSeq 2011

-- Proof statement: the state of the lights after 2011 toggles is such that lights A, D, F are on
theorem lights_on_after_2011_toggles :
  finalState Light.A = true ∧
  finalState Light.D = true ∧
  finalState Light.F = true ∧
  finalState Light.B = false ∧
  finalState Light.C = false ∧
  finalState Light.E = false ∧
  finalState Light.G = false :=
by
  sorry

end NUMINAMATH_GPT_lights_on_after_2011_toggles_l179_17969
