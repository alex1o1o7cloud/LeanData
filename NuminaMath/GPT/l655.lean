import Mathlib

namespace NUMINAMATH_GPT_right_triangle_congruence_l655_65582

theorem right_triangle_congruence (A B C D : Prop) :
  (A → true) → (C → true) → (D → true) → (¬ B) → B :=
by
sorry

end NUMINAMATH_GPT_right_triangle_congruence_l655_65582


namespace NUMINAMATH_GPT_find_x_l655_65567

theorem find_x (x : ℝ) : 0.6 * x = (x / 3) + 110 → x = 412.5 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l655_65567


namespace NUMINAMATH_GPT_nails_remaining_proof_l655_65503

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end NUMINAMATH_GPT_nails_remaining_proof_l655_65503


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_10_0_l655_65557

theorem line_intersects_x_axis_at_10_0 :
  let x1 := 9
  let y1 := 1
  let x2 := 5
  let y2 := 5
  let slope := (y2 - y1) / (x2 - x1)
  let y := 0
  ∃ x, (x - x1) * slope = y - y1 ∧ y = 0 → x = 10 := by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_10_0_l655_65557


namespace NUMINAMATH_GPT_integer_for_finitely_many_n_l655_65590

theorem integer_for_finitely_many_n (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, N < n → ¬ ∃ k : ℤ, (a + 1 / 2) ^ n + (b + 1 / 2) ^ n = k := 
sorry

end NUMINAMATH_GPT_integer_for_finitely_many_n_l655_65590


namespace NUMINAMATH_GPT_mean_total_sample_variance_total_sample_expected_final_score_l655_65526

section SeagrassStatistics

variables (m n : ℕ) (mean_x mean_y: ℝ) (var_x var_y: ℝ) (A_win_A B_win_A : ℝ)

-- Assumptions from the conditions
variable (hp1 : m = 12)
variable (hp2 : mean_x = 18)
variable (hp3 : var_x = 19)
variable (hp4 : n = 18)
variable (hp5 : mean_y = 36)
variable (hp6 : var_y = 70)
variable (hp7 : A_win_A = 3 / 5)
variable (hp8 : B_win_A = 1 / 2)

-- Statements to prove
theorem mean_total_sample (m n : ℕ) (mean_x mean_y : ℝ) : 
  m * mean_x + n * mean_y = (m + n) * 28.8 := sorry

theorem variance_total_sample (m n : ℕ) (mean_x mean_y var_x var_y : ℝ) :
  m * (var_x + (mean_x - 28.8)^2) + n * (var_y + (mean_y - 28.8)^2) = (m + n) * 127.36 := sorry

theorem expected_final_score (A_win_A B_win_A : ℝ) :
  2 * ((6/25) * 1 + (15/25) * 2 + (4/25) * 0) = 36 / 25 := sorry

end SeagrassStatistics

end NUMINAMATH_GPT_mean_total_sample_variance_total_sample_expected_final_score_l655_65526


namespace NUMINAMATH_GPT_find_first_number_l655_65514

/-- The Least Common Multiple (LCM) of two numbers A and B is 2310,
    and their Highest Common Factor (HCF) is 30.
    Given one of the numbers B is 180, find the other number A. -/
theorem find_first_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 30) (h3 : B = 180) (h4 : A * B = LCM * HCF) :
  A = 385 :=
by sorry

end NUMINAMATH_GPT_find_first_number_l655_65514


namespace NUMINAMATH_GPT_equal_roots_of_quadratic_l655_65549

theorem equal_roots_of_quadratic (k : ℝ) : 
  (∃ x, (x^2 + 2 * x + k = 0) ∧ (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_equal_roots_of_quadratic_l655_65549


namespace NUMINAMATH_GPT_number_of_sides_is_15_l655_65545

variable {n : ℕ} -- n is the number of sides

-- Define the conditions
def sum_of_all_but_one_angle (n : ℕ) : Prop :=
  180 * (n - 2) - 2190 > 0 ∧ 180 * (n - 2) - 2190 < 180

-- State the theorem to be proven
theorem number_of_sides_is_15 (n : ℕ) (h : sum_of_all_but_one_angle n) : n = 15 :=
sorry

end NUMINAMATH_GPT_number_of_sides_is_15_l655_65545


namespace NUMINAMATH_GPT_temperature_range_l655_65535

-- Define the highest and lowest temperature conditions
variable (t : ℝ)
def highest_temp := t ≤ 30
def lowest_temp := 20 ≤ t

-- The theorem to prove the range of temperature change
theorem temperature_range (t : ℝ) (h_high : highest_temp t) (h_low : lowest_temp t) : 20 ≤ t ∧ t ≤ 30 :=
by 
  -- Insert the proof or leave as sorry for now
  sorry

end NUMINAMATH_GPT_temperature_range_l655_65535


namespace NUMINAMATH_GPT_function_is_odd_and_increasing_l655_65511

-- Define the function y = x^(3/5)
def f (x : ℝ) : ℝ := x ^ (3 / 5)

-- Define what it means for the function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for the function to be increasing in its domain
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- The proposition to prove
theorem function_is_odd_and_increasing :
  is_odd f ∧ is_increasing f :=
by
  sorry

end NUMINAMATH_GPT_function_is_odd_and_increasing_l655_65511


namespace NUMINAMATH_GPT_original_price_calc_l655_65520

theorem original_price_calc (h : 1.08 * x = 2) : x = 100 / 54 := by
  sorry

end NUMINAMATH_GPT_original_price_calc_l655_65520


namespace NUMINAMATH_GPT_total_fare_for_100_miles_l655_65501

theorem total_fare_for_100_miles (b c : ℝ) (h₁ : 200 = b + 80 * c) : 240 = b + 100 * c :=
sorry

end NUMINAMATH_GPT_total_fare_for_100_miles_l655_65501


namespace NUMINAMATH_GPT_sum_of_g1_values_l655_65530

noncomputable def g : Polynomial ℝ := sorry

theorem sum_of_g1_values :
  (∀ x : ℝ, x ≠ 0 → g.eval (x-1) + g.eval x + g.eval (x+1) = (g.eval x)^2 / (4036 * x)) →
  g.degree ≠ 0 →
  g.eval 1 = 12108 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_g1_values_l655_65530


namespace NUMINAMATH_GPT_simplify_expression_l655_65578

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x ^ 2 + 10 - (5 - 4 * x + 8 * x ^ 2) = -16 * x ^ 2 + 8 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l655_65578


namespace NUMINAMATH_GPT_john_hourly_wage_l655_65509

theorem john_hourly_wage (days_off: ℕ) (hours_per_day: ℕ) (weekly_wage: ℕ) 
  (days_off_eq: days_off = 3) (hours_per_day_eq: hours_per_day = 4) (weekly_wage_eq: weekly_wage = 160):
  (weekly_wage / ((7 - days_off) * hours_per_day) = 10) :=
by
  /-
  Given:
  days_off = 3
  hours_per_day = 4
  weekly_wage = 160

  To prove:
  weekly_wage / ((7 - days_off) * hours_per_day) = 10
  -/
  sorry

end NUMINAMATH_GPT_john_hourly_wage_l655_65509


namespace NUMINAMATH_GPT_ratio_of_buttons_to_magnets_per_earring_l655_65537

-- Definitions related to the problem statement
def gemstones_per_button : ℕ := 3
def magnets_per_earring : ℕ := 2
def sets_of_earrings : ℕ := 4
def required_gemstones : ℕ := 24

-- Problem statement translation into Lean 4
theorem ratio_of_buttons_to_magnets_per_earring :
  (required_gemstones / gemstones_per_button / (sets_of_earrings * 2)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_buttons_to_magnets_per_earring_l655_65537


namespace NUMINAMATH_GPT_weeks_of_exercise_l655_65531

def hours_per_day : ℕ := 1
def days_per_week : ℕ := 5
def total_hours : ℕ := 40

def weekly_hours : ℕ := hours_per_day * days_per_week

theorem weeks_of_exercise (W : ℕ) (h : total_hours = weekly_hours * W) : W = 8 :=
by
  sorry

end NUMINAMATH_GPT_weeks_of_exercise_l655_65531


namespace NUMINAMATH_GPT_greatest_integer_b_for_no_real_roots_l655_65543

theorem greatest_integer_b_for_no_real_roots (b : ℤ) :
  (∀ x : ℝ, x^2 + (b:ℝ)*x + 10 ≠ 0) ↔ b ≤ 6 :=
sorry

end NUMINAMATH_GPT_greatest_integer_b_for_no_real_roots_l655_65543


namespace NUMINAMATH_GPT_problem1_l655_65534

theorem problem1 (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end NUMINAMATH_GPT_problem1_l655_65534


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l655_65532

theorem hyperbola_real_axis_length
    (a b : ℝ) 
    (h_pos_a : a > 0) 
    (h_pos_b : b > 0) 
    (h_eccentricity : a * Real.sqrt 5 = Real.sqrt (a^2 + b^2))
    (h_distance : b * a * Real.sqrt 5 / Real.sqrt (a^2 + b^2) = 8) :
    2 * a = 8 :=
sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l655_65532


namespace NUMINAMATH_GPT_min_top_block_sum_l655_65524

theorem min_top_block_sum : 
  ∀ (assign_numbers : ℕ → ℕ) 
  (layer_1 : Fin 16 → ℕ) (layer_2 : Fin 9 → ℕ) (layer_3 : Fin 4 → ℕ) (top_block : ℕ),
  (∀ i, layer_3 i = layer_2 (i / 2) + layer_2 ((i / 2) + 1) + layer_2 ((i / 2) + 3) + layer_2 ((i / 2) + 4)) →
  (∀ i, layer_2 i = layer_1 (i / 2) + layer_1 ((i / 2) + 1) + layer_1 ((i / 2) + 3) + layer_1 ((i / 2) + 4)) →
  (top_block = layer_3 0 + layer_3 1 + layer_3 2 + layer_3 3) →
  top_block = 40 :=
sorry

end NUMINAMATH_GPT_min_top_block_sum_l655_65524


namespace NUMINAMATH_GPT_length_of_BC_l655_65513

-- Define the given conditions and the theorem using Lean
theorem length_of_BC 
  (A B C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hB : ∃ b : ℝ, B = (-b, -b^2)) 
  (hC : ∃ b : ℝ, C = (b, -b^2)) 
  (hBC_parallel_x_axis : ∀ b : ℝ, C.2 = B.2)
  (hArea : ∀ b : ℝ, b^3 = 72) 
  : ∀ b : ℝ, (BC : ℝ) = 2 * b := 
by
  sorry

end NUMINAMATH_GPT_length_of_BC_l655_65513


namespace NUMINAMATH_GPT_sqrt_of_sqrt_81_l655_65517

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end NUMINAMATH_GPT_sqrt_of_sqrt_81_l655_65517


namespace NUMINAMATH_GPT_peter_total_dogs_l655_65500

def num_german_shepherds_sam : ℕ := 3
def num_french_bulldogs_sam : ℕ := 4
def num_german_shepherds_peter := 3 * num_german_shepherds_sam
def num_french_bulldogs_peter := 2 * num_french_bulldogs_sam

theorem peter_total_dogs : num_german_shepherds_peter + num_french_bulldogs_peter = 17 :=
by {
  -- adding proofs later
  sorry
}

end NUMINAMATH_GPT_peter_total_dogs_l655_65500


namespace NUMINAMATH_GPT_inequality_not_always_hold_l655_65593

variable (a b c : ℝ)

theorem inequality_not_always_hold (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) : ¬ (∀ (a b : ℝ), |a - b| + 1 / (a - b) ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_always_hold_l655_65593


namespace NUMINAMATH_GPT_factor_expression_l655_65564

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l655_65564


namespace NUMINAMATH_GPT_sequence_sum_l655_65510

theorem sequence_sum :
  1 - 4 + 7 - 10 + 13 - 16 + 19 - 22 + 25 - 28 + 31 - 34 + 37 - 40 + 43 - 46 + 49 - 52 + 55 = 28 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l655_65510


namespace NUMINAMATH_GPT_Mr_Deane_filled_today_l655_65591

theorem Mr_Deane_filled_today :
  ∀ (x : ℝ),
    (25 * (1.4 - 0.4) + 1.4 * x = 39) →
    x = 10 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_Mr_Deane_filled_today_l655_65591


namespace NUMINAMATH_GPT_Ben_sales_value_l655_65559

noncomputable def value_of_sale (old_salary new_salary commission_ratio sales_required : ℝ) (diff_salary: ℝ) :=
  ∃ x : ℝ, 0.15 * x * sales_required = diff_salary ∧ x = 750

theorem Ben_sales_value (old_salary new_salary commission_ratio sales_required diff_salary: ℝ)
  (h1: old_salary = 75000)
  (h2: new_salary = 45000)
  (h3: commission_ratio = 0.15)
  (h4: sales_required = 266.67)
  (h5: diff_salary = old_salary - new_salary) :
  value_of_sale old_salary new_salary commission_ratio sales_required diff_salary :=
by
  sorry

end NUMINAMATH_GPT_Ben_sales_value_l655_65559


namespace NUMINAMATH_GPT_bonus_tasks_l655_65560

-- Definition for earnings without bonus
def earnings_without_bonus (tasks : ℕ) : ℕ := tasks * 2

-- Definition for calculating the total bonus received
def total_bonus (tasks : ℕ) (earnings : ℕ) : ℕ := earnings - earnings_without_bonus tasks

-- Definition for the number of bonuses received given the total bonus and a single bonus amount
def number_of_bonuses (total_bonus : ℕ) (bonus_amount : ℕ) : ℕ := total_bonus / bonus_amount

-- The theorem we want to prove
theorem bonus_tasks (tasks : ℕ) (earnings : ℕ) (bonus_amount : ℕ) (bonus_tasks : ℕ) :
  earnings = 78 →
  tasks = 30 →
  bonus_amount = 6 →
  bonus_tasks = tasks / (number_of_bonuses (total_bonus tasks earnings) bonus_amount) →
  bonus_tasks = 10 :=
by
  intros h_earnings h_tasks h_bonus_amount h_bonus_tasks
  sorry

end NUMINAMATH_GPT_bonus_tasks_l655_65560


namespace NUMINAMATH_GPT_min_angle_B_l655_65597

-- Definitions using conditions from part a)
def triangle (A B C : ℝ) : Prop := A + B + C = Real.pi
def arithmetic_sequence_prop (A B C : ℝ) : Prop := 
  Real.tan A + Real.tan C = 2 * (1 + Real.sqrt 2) * Real.tan B

-- Main theorem to prove
theorem min_angle_B (A B C : ℝ) (h1 : triangle A B C) (h2 : arithmetic_sequence_prop A B C) :
  B ≥ Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_min_angle_B_l655_65597


namespace NUMINAMATH_GPT_total_amount_shared_l655_65565

-- Define the initial conditions
def ratioJohn : ℕ := 2
def ratioJose : ℕ := 4
def ratioBinoy : ℕ := 6
def JohnShare : ℕ := 2000
def partValue : ℕ := JohnShare / ratioJohn

-- Define the shares based on the ratio and part value
def JoseShare := ratioJose * partValue
def BinoyShare := ratioBinoy * partValue

-- Prove the total amount shared is Rs. 12000
theorem total_amount_shared : (JohnShare + JoseShare + BinoyShare) = 12000 :=
  by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l655_65565


namespace NUMINAMATH_GPT_tax_percentage_l655_65553

theorem tax_percentage (C T : ℝ) (h1 : C + 10 = 90) (h2 : 1 = 90 - C - T * 90) : T = 0.1 := 
by 
  -- We provide the conditions using sorry to indicate the steps would go here
  sorry

end NUMINAMATH_GPT_tax_percentage_l655_65553


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l655_65548

theorem solve_equation_1 (x : ℝ) (h₁ : x - 4 = -5) : x = -1 :=
sorry

theorem solve_equation_2 (x : ℝ) (h₂ : (1/2) * x + 2 = 6) : x = 8 :=
sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l655_65548


namespace NUMINAMATH_GPT_suresh_borrowed_amount_l655_65541

theorem suresh_borrowed_amount 
  (P: ℝ)
  (i1 i2 i3: ℝ)
  (t1 t2 t3: ℝ)
  (total_interest: ℝ)
  (h1 : i1 = 0.12) 
  (h2 : t1 = 3)
  (h3 : i2 = 0.09)
  (h4 : t2 = 5)
  (h5 : i3 = 0.13)
  (h6 : t3 = 3)
  (h_total : total_interest = 8160) 
  (h_interest_eq : total_interest = P * i1 * t1 + P * i2 * t2 + P * i3 * t3)
  : P = 6800 :=
by
  sorry

end NUMINAMATH_GPT_suresh_borrowed_amount_l655_65541


namespace NUMINAMATH_GPT_quadratic_solution_eq_l655_65546

theorem quadratic_solution_eq (c d : ℝ) 
  (h_eq : ∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = c ∨ x = d))
  (h_order : c ≥ d) :
  c + 2*d = 9 - Real.sqrt 23 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_eq_l655_65546


namespace NUMINAMATH_GPT_total_participants_l655_65562

theorem total_participants (x : ℕ) (h1 : 800 / x + 60 = 800 / (x - 3)) : x = 8 :=
sorry

end NUMINAMATH_GPT_total_participants_l655_65562


namespace NUMINAMATH_GPT_operation_result_l655_65551

-- Define the operation
def operation (a b : ℝ) : ℝ := (a - b) ^ 3

theorem operation_result (x y : ℝ) : operation ((x - y) ^ 3) ((y - x) ^ 3) = -8 * (y - x) ^ 9 := 
  sorry

end NUMINAMATH_GPT_operation_result_l655_65551


namespace NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l655_65579

/-- 
  Problem: Determine the surface area of the sphere circumscribed about a cube with edge length 2.

  Given:
  - The edge length of the cube is 2.
  - The space diagonal of a cube with edge length \(a\) is given by \(d = \sqrt{3} \cdot a\).
  - The diameter of the circumscribed sphere is equal to the space diagonal of the cube.
  - The surface area \(S\) of a sphere with radius \(R\) is given by \(S = 4\pi R^2\).

  To Prove:
  - The surface area of the sphere circumscribed about the cube is \(12\pi\).
-/
theorem surface_area_of_circumscribed_sphere (a : ℝ) (π : ℝ) (h1 : a = 2) 
  (h2 : ∀ a, d = Real.sqrt 3 * a) (h3 : ∀ d, R = d / 2) (h4 : ∀ R, S = 4 * π * R^2) : 
  S = 12 * π := 
by
  sorry

end NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l655_65579


namespace NUMINAMATH_GPT_train_speed_is_45_km_per_hr_l655_65598

/-- 
  Given the length of the train (135 m), the time to cross a bridge (30 s),
  and the length of the bridge (240 m), we want to prove that the speed of the 
  train is 45 km/hr.
--/

def length_of_train : ℕ := 135
def time_to_cross_bridge : ℕ := 30
def length_of_bridge : ℕ := 240
def speed_of_train_in_km_per_hr (L_t t L_b : ℕ) : ℕ := 
  ((L_t + L_b) * 36 / 10) / t

theorem train_speed_is_45_km_per_hr : 
  speed_of_train_in_km_per_hr length_of_train time_to_cross_bridge length_of_bridge = 45 :=
by 
  -- Assuming the calculations are correct, the expected speed is provided here directly
  sorry

end NUMINAMATH_GPT_train_speed_is_45_km_per_hr_l655_65598


namespace NUMINAMATH_GPT_domain_of_sqrt_l655_65519

theorem domain_of_sqrt (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by sorry

end NUMINAMATH_GPT_domain_of_sqrt_l655_65519


namespace NUMINAMATH_GPT_sum_of_digits_l655_65587

variable (a b c d e f : ℕ)

theorem sum_of_digits :
  ∀ (a b c d e f : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 →
    a + b + c + d + e + f = 28 := 
by
  intros a b c d e f h
  sorry

end NUMINAMATH_GPT_sum_of_digits_l655_65587


namespace NUMINAMATH_GPT_part1_part2_l655_65525

def A (x : ℤ) := ∃ m n : ℤ, x = m^2 - n^2
def B (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

theorem part1 (h1: A 8) (h2: A 9) (h3: ¬ A 10) : 
  (A 8) ∧ (A 9) ∧ (¬ A 10) :=
by {
  sorry
}

theorem part2 (x : ℤ) (h : A x) : B x :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l655_65525


namespace NUMINAMATH_GPT_minimize_f_minimize_f_exact_l655_65518

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 14 * x - 20

-- State the theorem that x = -7 minimizes the function f(x)
theorem minimize_f : ∀ x : ℝ, f x ≥ f (-7) :=
by
  intro x
  unfold f
  sorry

-- An alternative statement could include the exact condition for the minimum value
theorem minimize_f_exact : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ x = -7 :=
by
  use -7
  intro y
  unfold f
  sorry

end NUMINAMATH_GPT_minimize_f_minimize_f_exact_l655_65518


namespace NUMINAMATH_GPT_lamp_count_and_profit_l655_65540

-- Define the parameters given in the problem
def total_lamps : ℕ := 50
def total_cost : ℕ := 2500
def cost_A : ℕ := 40
def cost_B : ℕ := 65
def marked_A : ℕ := 60
def marked_B : ℕ := 100
def discount_A : ℕ := 10 -- percent
def discount_B : ℕ := 30 -- percent

-- Derived definitions from the solution
def lamps_A : ℕ := 30
def lamps_B : ℕ := 20
def selling_price_A : ℕ := marked_A * (100 - discount_A) / 100
def selling_price_B : ℕ := marked_B * (100 - discount_B) / 100
def profit_A : ℕ := selling_price_A - cost_A
def profit_B : ℕ := selling_price_B - cost_B
def total_profit : ℕ := (profit_A * lamps_A) + (profit_B * lamps_B)

-- Lean statement
theorem lamp_count_and_profit :
  lamps_A + lamps_B = total_lamps ∧
  (cost_A * lamps_A + cost_B * lamps_B) = total_cost ∧
  total_profit = 520 := by
  -- proofs will go here
  sorry

end NUMINAMATH_GPT_lamp_count_and_profit_l655_65540


namespace NUMINAMATH_GPT_tan_315_degrees_l655_65573

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_315_degrees_l655_65573


namespace NUMINAMATH_GPT_PropositionA_necessary_not_sufficient_l655_65584

variable (a : ℝ)

def PropositionA : Prop := a < 2
def PropositionB : Prop := a^2 < 4

theorem PropositionA_necessary_not_sufficient : 
  (PropositionA a → PropositionB a) ∧ ¬ (PropositionB a → PropositionA a) :=
sorry

end NUMINAMATH_GPT_PropositionA_necessary_not_sufficient_l655_65584


namespace NUMINAMATH_GPT_slope_of_parallel_line_l655_65523

theorem slope_of_parallel_line (a b c : ℝ) (h: 3*a + 6*b = -24) :
  ∃ m : ℝ, (a * 3 + b * 6 = c) → m = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l655_65523


namespace NUMINAMATH_GPT_length_of_AE_l655_65538

theorem length_of_AE (AF CE ED : ℝ) (ABCD_area : ℝ) (hAF : AF = 30) (hCE : CE = 40) (hED : ED = 50) (hABCD_area : ABCD_area = 7200) : ∃ AE : ℝ, AE = 322.5 := sorry

end NUMINAMATH_GPT_length_of_AE_l655_65538


namespace NUMINAMATH_GPT_evaluate_expression_l655_65508

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) : 2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l655_65508


namespace NUMINAMATH_GPT_gel_pen_price_relation_b_l655_65516

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end NUMINAMATH_GPT_gel_pen_price_relation_b_l655_65516


namespace NUMINAMATH_GPT_four_P_plus_five_square_of_nat_l655_65529

theorem four_P_plus_five_square_of_nat 
  (a b : ℕ)
  (P : ℕ)
  (hP : P = (Nat.lcm a b) / (a + 1) + (Nat.lcm a b) / (b + 1))
  (h_prime : Nat.Prime P) : 
  ∃ n : ℕ, 4 * P + 5 = (2 * n + 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_four_P_plus_five_square_of_nat_l655_65529


namespace NUMINAMATH_GPT_triangle_AC_length_l655_65506

open Real

theorem triangle_AC_length (A : ℝ) (AB AC S : ℝ) (h1 : A = π / 3) (h2 : AB = 2) (h3 : S = sqrt 3 / 2) : AC = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_AC_length_l655_65506


namespace NUMINAMATH_GPT_percent_increase_sales_l655_65550

theorem percent_increase_sales (sales_this_year sales_last_year : ℝ) (h1 : sales_this_year = 460) (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_sales_l655_65550


namespace NUMINAMATH_GPT_solve_for_x_l655_65533

-- declare an existential quantifier to encapsulate the condition and the answer.
theorem solve_for_x : ∃ x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := 
by 
  -- begin sorry to skip the proof part
  sorry

end NUMINAMATH_GPT_solve_for_x_l655_65533


namespace NUMINAMATH_GPT_find_x_condition_l655_65558

theorem find_x_condition (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := 
by
  sorry

end NUMINAMATH_GPT_find_x_condition_l655_65558


namespace NUMINAMATH_GPT_eval_expression_l655_65588

theorem eval_expression : -20 + 12 * ((5 + 15) / 4) = 40 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l655_65588


namespace NUMINAMATH_GPT_bear_cubs_count_l655_65583

theorem bear_cubs_count (total_meat : ℕ) (meat_per_cub : ℕ) (rabbits_per_day : ℕ) (weeks_days : ℕ) (meat_per_rabbit : ℕ)
  (mother_total_meat : ℕ) (number_of_cubs : ℕ) : 
  total_meat = 210 →
  meat_per_cub = 35 →
  rabbits_per_day = 10 →
  weeks_days = 7 →
  meat_per_rabbit = 5 →
  mother_total_meat = rabbits_per_day * weeks_days * meat_per_rabbit →
  meat_per_cub * number_of_cubs + mother_total_meat = total_meat →
  number_of_cubs = 4 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end NUMINAMATH_GPT_bear_cubs_count_l655_65583


namespace NUMINAMATH_GPT_sum_adjacent_to_49_l655_65563

noncomputable def sum_of_adjacent_divisors : ℕ :=
  let divisors := [5, 7, 35, 49, 245]
  -- We assume an arrangement such that adjacent pairs to 49 are {35, 245}
  35 + 245

theorem sum_adjacent_to_49 : sum_of_adjacent_divisors = 280 := by
  sorry

end NUMINAMATH_GPT_sum_adjacent_to_49_l655_65563


namespace NUMINAMATH_GPT_largest_integer_x_l655_65522

theorem largest_integer_x (x : ℤ) :
  (x ^ 2 - 11 * x + 28 < 0) → x ≤ 6 := sorry

end NUMINAMATH_GPT_largest_integer_x_l655_65522


namespace NUMINAMATH_GPT_cost_to_marked_price_ratio_l655_65580

variables (p : ℝ) (discount : ℝ := 0.20) (cost_ratio : ℝ := 0.60)

theorem cost_to_marked_price_ratio :
  (cost_ratio * (1 - discount) * p) / p = 0.48 :=
by sorry

end NUMINAMATH_GPT_cost_to_marked_price_ratio_l655_65580


namespace NUMINAMATH_GPT_find_2a_plus_b_l655_65595

open Real

theorem find_2a_plus_b (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
    (h1 : 4 * (cos a)^3 - 3 * (cos b)^3 = 2) 
    (h2 : 4 * cos (2 * a) + 3 * cos (2 * b) = 1) : 
    2 * a + b = π / 2 :=
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l655_65595


namespace NUMINAMATH_GPT_min_value_of_squares_find_p_l655_65570

open Real

theorem min_value_of_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (eqn : a + sqrt 2 * b + sqrt 3 * c = 2 * sqrt 3) :
  a^2 + b^2 + c^2 = 2 :=
by sorry

theorem find_p (m : ℝ) (hm : m = 2) (p q : ℝ) :
  (∀ x, |x - 3| ≥ m ↔ x^2 + p * x + q ≥ 0) → p = -6 :=
by sorry

end NUMINAMATH_GPT_min_value_of_squares_find_p_l655_65570


namespace NUMINAMATH_GPT_sum_mod_9_l655_65555

theorem sum_mod_9 :
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := 
by sorry

end NUMINAMATH_GPT_sum_mod_9_l655_65555


namespace NUMINAMATH_GPT_simplify_fraction_l655_65594

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h_cond : y^3 - 1/x ≠ 0) :
  (x^3 - 1/y) / (y^3 - 1/x) = x / y :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l655_65594


namespace NUMINAMATH_GPT_friends_carrying_bananas_l655_65507

theorem friends_carrying_bananas :
  let total_friends := 35
  let friends_with_pears := 14
  let friends_with_oranges := 8
  let friends_with_apples := 5
  total_friends - (friends_with_pears + friends_with_oranges + friends_with_apples) = 8 := 
by
  sorry

end NUMINAMATH_GPT_friends_carrying_bananas_l655_65507


namespace NUMINAMATH_GPT_compare_neg_third_and_neg_point_three_l655_65515

/-- Compare two numbers -1/3 and -0.3 -/
theorem compare_neg_third_and_neg_point_three : (-1 / 3 : ℝ) < -0.3 :=
sorry

end NUMINAMATH_GPT_compare_neg_third_and_neg_point_three_l655_65515


namespace NUMINAMATH_GPT_find_multiplier_l655_65569

-- Define the variables x and y
variables (x y : ℕ)

-- Define the conditions
def condition1 := (x / 6) * y = 12
def condition2 := x = 6

-- State the theorem to prove
theorem find_multiplier (h1 : condition1 x y) (h2 : condition2 x) : y = 12 :=
sorry

end NUMINAMATH_GPT_find_multiplier_l655_65569


namespace NUMINAMATH_GPT_green_ish_count_l655_65521

theorem green_ish_count (total : ℕ) (blue_ish : ℕ) (both : ℕ) (neither : ℕ) (green_ish : ℕ) :
  total = 150 ∧ blue_ish = 90 ∧ both = 40 ∧ neither = 30 → green_ish = 70 :=
by
  sorry

end NUMINAMATH_GPT_green_ish_count_l655_65521


namespace NUMINAMATH_GPT_scouts_attended_l655_65568

def chocolate_bar_cost : ℝ := 1.50
def total_spent : ℝ := 15
def sections_per_bar : ℕ := 3
def smores_per_scout : ℕ := 2

theorem scouts_attended (bars : ℝ) (sections : ℕ) (smores : ℕ) (scouts : ℕ) :
  bars = total_spent / chocolate_bar_cost →
  sections = bars * sections_per_bar →
  smores = sections →
  scouts = smores / smores_per_scout →
  scouts = 15 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_scouts_attended_l655_65568


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_proof_l655_65536

-- Define the function f(x) with a given value of a.
def f (x : ℝ) (a : ℝ) : ℝ := |x + a|

-- Problem 1: Solve the inequality f(x) ≥ 5 - |x - 2| when a = 1.
theorem problem1_solution_set (x : ℝ) :
  f x 1 ≥ 5 - |x - 2| ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

-- Problem 2: Given the solution set of f(x) ≤ 5 is [-9, 1] and the equation 1/m + 1/(2n) = a, prove m + 2n ≥ 1
theorem problem2_proof (a m n : ℝ) (hma : a = 4) (hmpos : m > 0) (hnpos : n > 0) :
  (1 / m + 1 / (2 * n) = a) → m + 2 * n ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_proof_l655_65536


namespace NUMINAMATH_GPT_intersection_M_N_l655_65552

-- Define the universal set U, and subsets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- Prove that the intersection of M and N is as stated
theorem intersection_M_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 1} :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_intersection_M_N_l655_65552


namespace NUMINAMATH_GPT_gcd_of_lcm_l655_65577

noncomputable def gcd (A B C : ℕ) : ℕ := Nat.gcd (Nat.gcd A B) C
noncomputable def lcm (A B C : ℕ) : ℕ := Nat.lcm (Nat.lcm A B) C

theorem gcd_of_lcm (A B C : ℕ) (LCM_ABC : ℕ) (Product_ABC : ℕ) :
  lcm A B C = LCM_ABC →
  A * B * C = Product_ABC →
  gcd A B C = 20 :=
by
  intros lcm_eq product_eq
  sorry

end NUMINAMATH_GPT_gcd_of_lcm_l655_65577


namespace NUMINAMATH_GPT_domain_of_h_l655_65561

-- Definition of the function domain of f(x) and h(x)
def f_domain := Set.Icc (-10: ℝ) 6
def h_domain := Set.Icc (-2: ℝ) (10/3)

-- Definition of f and h
def f (x: ℝ) : ℝ := sorry  -- f is assumed to be defined on the interval [-10, 6]
def h (x: ℝ) : ℝ := f (-3 * x)

-- Theorem statement: Given the domain of f(x), the domain of h(x) is as follows
theorem domain_of_h :
  (∀ x, x ∈ f_domain ↔ (-3 * x) ∈ h_domain) :=
sorry

end NUMINAMATH_GPT_domain_of_h_l655_65561


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l655_65544

theorem sum_and_product_of_roots (a b : ℝ) (h1 : a * a * a - 4 * a * a - a + 4 = 0)
  (h2 : b * b * b - 4 * b * b - b + 4 = 0) :
  a + b + a * b = -1 :=
sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l655_65544


namespace NUMINAMATH_GPT_booksReadPerDay_l655_65547

-- Mrs. Hilt read 14 books in a week.
def totalBooksReadInWeek : ℕ := 14

-- There are 7 days in a week.
def daysInWeek : ℕ := 7

-- We need to prove that the number of books read per day is 2.
theorem booksReadPerDay :
  totalBooksReadInWeek / daysInWeek = 2 :=
by
  sorry

end NUMINAMATH_GPT_booksReadPerDay_l655_65547


namespace NUMINAMATH_GPT_beth_coins_sold_l655_65528

def initial_coins : ℕ := 250
def additional_coins : ℕ := 75
def percentage_sold : ℚ := 60 / 100
def total_coins : ℕ := initial_coins + additional_coins
def coins_sold : ℚ := percentage_sold * total_coins

theorem beth_coins_sold : coins_sold = 195 :=
by
  -- Sorry is used to skip the proof as requested
  sorry

end NUMINAMATH_GPT_beth_coins_sold_l655_65528


namespace NUMINAMATH_GPT_find_constants_a_b_l655_65556

variables (x a b : ℝ)

theorem find_constants_a_b (h : (x - a) / (x + b) = (x^2 - 45 * x + 504) / (x^2 + 66 * x - 1080)) :
  a + b = 48 :=
sorry

end NUMINAMATH_GPT_find_constants_a_b_l655_65556


namespace NUMINAMATH_GPT_bowling_ball_weight_l655_65527

theorem bowling_ball_weight (b c : ℝ) (h1 : 9 * b = 6 * c) (h2 : 4 * c = 120) : b = 20 :=
sorry

end NUMINAMATH_GPT_bowling_ball_weight_l655_65527


namespace NUMINAMATH_GPT_min_value_f_at_3_f_increasing_for_k_neg4_l655_65576

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + k / (x - 1)

-- Problem (1): If k = 4, find the minimum value of f(x) and the corresponding value of x.
theorem min_value_f_at_3 : ∃ x > 1, @f x 4 = 5 ∧ x = 3 :=
  sorry

-- Problem (2): If k = -4, prove that f(x) is an increasing function for x > 1.
theorem f_increasing_for_k_neg4 : ∀ ⦃x y : ℝ⦄, 1 < x → x < y → f x (-4) < f y (-4) :=
  sorry

end NUMINAMATH_GPT_min_value_f_at_3_f_increasing_for_k_neg4_l655_65576


namespace NUMINAMATH_GPT_find_number_l655_65539

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l655_65539


namespace NUMINAMATH_GPT_GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l655_65571

noncomputable def GCD (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCD_17_51 : GCD 17 51 = 17 := by
  sorry

theorem LCM_17_51 : LCM 17 51 = 51 := by
  sorry

theorem GCD_6_8 : GCD 6 8 = 2 := by
  sorry

theorem LCM_8_9 : LCM 8 9 = 72 := by
  sorry

end NUMINAMATH_GPT_GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l655_65571


namespace NUMINAMATH_GPT_jessica_coins_worth_l655_65599

theorem jessica_coins_worth :
  ∃ (n d : ℕ), n + d = 30 ∧ 5 * (30 - d) + 10 * d = 165 :=
by {
  sorry
}

end NUMINAMATH_GPT_jessica_coins_worth_l655_65599


namespace NUMINAMATH_GPT_payment_proof_l655_65586

theorem payment_proof (X Y : ℝ) 
  (h₁ : X + Y = 572) 
  (h₂ : X = 1.20 * Y) 
  : Y = 260 := 
by 
  sorry

end NUMINAMATH_GPT_payment_proof_l655_65586


namespace NUMINAMATH_GPT_pressure_increases_when_block_submerged_l655_65512

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end NUMINAMATH_GPT_pressure_increases_when_block_submerged_l655_65512


namespace NUMINAMATH_GPT_ratio_of_share_l655_65574

/-- A certain amount of money is divided amongst a, b, and c. 
The share of a is $122, and the total amount of money is $366. 
Prove that the ratio of a's share to the combined share of b and c is 1 / 2. -/
theorem ratio_of_share (a b c : ℝ) (total share_a : ℝ) (h1 : a + b + c = total) 
  (h2 : total = 366) (h3 : share_a = 122) : share_a / (total - share_a) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_share_l655_65574


namespace NUMINAMATH_GPT_cos_angles_difference_cos_angles_sum_l655_65502

-- Part (a)
theorem cos_angles_difference: 
  (Real.cos (36 * Real.pi / 180) - Real.cos (72 * Real.pi / 180) = 1 / 2) :=
sorry

-- Part (b)
theorem cos_angles_sum: 
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2) :=
sorry

end NUMINAMATH_GPT_cos_angles_difference_cos_angles_sum_l655_65502


namespace NUMINAMATH_GPT_triplet_unique_solution_l655_65589

theorem triplet_unique_solution {x y z : ℝ} :
  x^2 - 2*x - 4*z = 3 →
  y^2 - 2*y - 2*x = -14 →
  z^2 - 4*y - 4*z = -18 →
  (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry

end NUMINAMATH_GPT_triplet_unique_solution_l655_65589


namespace NUMINAMATH_GPT_farmer_planting_problem_l655_65596

theorem farmer_planting_problem (total_acres : ℕ) (flax_acres : ℕ) (sunflower_acres : ℕ)
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : sunflower_acres = total_acres - flax_acres) :
  sunflower_acres - flax_acres = 80 := by
  sorry

end NUMINAMATH_GPT_farmer_planting_problem_l655_65596


namespace NUMINAMATH_GPT_top_card_is_red_l655_65505

noncomputable def standard_deck (ranks : ℕ) (suits : ℕ) : ℕ := ranks * suits

def red_cards_in_deck (hearts : ℕ) (diamonds : ℕ) : ℕ := hearts + diamonds

noncomputable def probability_red_card (red_cards : ℕ) (total_cards : ℕ) : ℚ := red_cards / total_cards

theorem top_card_is_red (hearts diamonds spades clubs : ℕ) (deck_size : ℕ)
  (H1 : hearts = 13) (H2 : diamonds = 13) (H3 : spades = 13) (H4 : clubs = 13) (H5 : deck_size = 52):
  probability_red_card (red_cards_in_deck hearts diamonds) deck_size = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_top_card_is_red_l655_65505


namespace NUMINAMATH_GPT_bicyclist_speed_remainder_l655_65575

theorem bicyclist_speed_remainder (total_distance first_distance remainder_distance first_speed avg_speed remainder_speed time_total time_first time_remainder : ℝ) 
  (H1 : total_distance = 350)
  (H2 : first_distance = 200)
  (H3 : remainder_distance = total_distance - first_distance)
  (H4 : first_speed = 20)
  (H5 : avg_speed = 17.5)
  (H6 : time_total = total_distance / avg_speed)
  (H7 : time_first = first_distance / first_speed)
  (H8 : time_remainder = time_total - time_first)
  (H9 : remainder_speed = remainder_distance / time_remainder) :
  remainder_speed = 15 := 
sorry

end NUMINAMATH_GPT_bicyclist_speed_remainder_l655_65575


namespace NUMINAMATH_GPT_distance_from_P_to_focus_l655_65572

-- Definition of a parabola y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of distance from P to y-axis
def distance_to_y_axis (x : ℝ) : ℝ := abs x

-- Definition of the focus of the parabola y^2 = 8x
def focus : (ℝ × ℝ) := (2, 0)

-- Definition of Euclidean distance
def euclidean_distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  (P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 

theorem distance_from_P_to_focus (x y : ℝ) (h₁ : parabola x y) (h₂ : distance_to_y_axis x = 4) :
  abs (euclidean_distance (x, y) focus) = 6 :=
sorry

end NUMINAMATH_GPT_distance_from_P_to_focus_l655_65572


namespace NUMINAMATH_GPT_systematic_sampling_l655_65566

theorem systematic_sampling (N : ℕ) (k : ℕ) (interval : ℕ) (seq : List ℕ) : 
  N = 70 → k = 7 → interval = 10 → 
  seq = [3, 13, 23, 33, 43, 53, 63] := 
by 
  intros hN hk hInt;
  sorry

end NUMINAMATH_GPT_systematic_sampling_l655_65566


namespace NUMINAMATH_GPT_min_x_plus_y_l655_65504

theorem min_x_plus_y (x y : ℝ) (h1 : x * y = 2 * x + y + 2) (h2 : x > 1) :
  x + y ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_l655_65504


namespace NUMINAMATH_GPT_find_f_l655_65581

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2 - 4) :
  ∀ x : ℝ, f x = x^2 - 2 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_find_f_l655_65581


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l655_65592

theorem isosceles_triangle_angles (α β γ : ℝ) 
  (h1 : α = 50)
  (h2 : α + β + γ = 180)
  (isosceles : (α = β ∨ α = γ ∨ β = γ)) :
  (β = 50 ∧ γ = 80) ∨ (γ = 50 ∧ β = 80) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l655_65592


namespace NUMINAMATH_GPT_no_hexagon_cross_section_l655_65542

-- Define the shape of the cross-section resulting from cutting a triangular prism with a plane
inductive Shape
| triangle
| quadrilateral
| pentagon
| hexagon

-- Define the condition of cutting a triangular prism
structure TriangularPrism where
  cut : Shape

-- The theorem stating that cutting a triangular prism with a plane cannot result in a hexagon
theorem no_hexagon_cross_section (P : TriangularPrism) : P.cut ≠ Shape.hexagon :=
by
  sorry

end NUMINAMATH_GPT_no_hexagon_cross_section_l655_65542


namespace NUMINAMATH_GPT_alice_journey_duration_l655_65585
noncomputable def journey_duration (start_hour start_minute end_hour end_minute : ℕ) : ℕ :=
  let start_in_minutes := start_hour * 60 + start_minute
  let end_in_minutes := end_hour * 60 + end_minute
  if end_in_minutes >= start_in_minutes then end_in_minutes - start_in_minutes
  else end_in_minutes + 24 * 60 - start_in_minutes
  
theorem alice_journey_duration :
  ∃ start_hour start_minute end_hour end_minute,
  (7 ≤ start_hour ∧ start_hour < 8 ∧ start_minute = 38) ∧
  (16 ≤ end_hour ∧ end_hour < 17 ∧ end_minute = 35) ∧
  journey_duration start_hour start_minute end_hour end_minute = 537 :=
by {
  sorry
}

end NUMINAMATH_GPT_alice_journey_duration_l655_65585


namespace NUMINAMATH_GPT_fraction_halfway_between_l655_65554

theorem fraction_halfway_between : 
  ∃ (x : ℚ), (x = (1 / 6 + 1 / 4) / 2) ∧ x = 5 / 24 :=
by
  sorry

end NUMINAMATH_GPT_fraction_halfway_between_l655_65554
