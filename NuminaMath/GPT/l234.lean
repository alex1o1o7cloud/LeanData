import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l234_23459

noncomputable def ratFunc (x : ℝ) : ℝ := 
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) : 
  (ratFunc x > 0) ↔ 
  ((x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l234_23459


namespace NUMINAMATH_GPT_find_x_plus_y_l234_23471

theorem find_x_plus_y
  (x y : ℝ)
  (hx : x^3 - 3 * x^2 + 5 * x - 17 = 0)
  (hy : y^3 - 3 * y^2 + 5 * y + 11 = 0) :
  x + y = 2 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l234_23471


namespace NUMINAMATH_GPT_area_triangle_le_quarter_l234_23437

theorem area_triangle_le_quarter (S : ℝ) (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ)
  (h₁ : S₃ + (S₂ + S₇) = S / 2)
  (h₂ : S₁ + S₆ + (S₂ + S₇) = S / 2) :
  S₁ ≤ S / 4 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_area_triangle_le_quarter_l234_23437


namespace NUMINAMATH_GPT_k_ge_a_l234_23497

theorem k_ge_a (a k : ℕ) (h_pos_a : 0 < a) (h_pos_k : 0 < k) 
  (h_div : (a ^ 2 + k) ∣ (a - 1) * a * (a + 1)) : k ≥ a := 
sorry

end NUMINAMATH_GPT_k_ge_a_l234_23497


namespace NUMINAMATH_GPT_solve_for_y_l234_23447

variable (k y : ℝ)

-- Define the first equation for x
def eq1 (x : ℝ) : Prop := (1 / 2023) * x - 2 = 3 * x + k

-- Define the condition that x = -5 satisfies eq1
def condition1 : Prop := eq1 k (-5)

-- Define the second equation for y
def eq2 : Prop := (1 / 2023) * (2 * y + 1) - 5 = 6 * y + k

-- Prove that given condition1, y = -3 satisfies eq2
theorem solve_for_y : condition1 k → eq2 k (-3) :=
sorry

end NUMINAMATH_GPT_solve_for_y_l234_23447


namespace NUMINAMATH_GPT_depth_of_canal_l234_23448

/-- The cross-section of a canal is a trapezium with a top width of 12 meters, 
a bottom width of 8 meters, and an area of 840 square meters. 
Prove that the depth of the canal is 84 meters.
-/
theorem depth_of_canal (top_width bottom_width area : ℝ) (h : ℝ) :
  top_width = 12 → bottom_width = 8 → area = 840 → 1 / 2 * (top_width + bottom_width) * h = area → h = 84 :=
by
  intros ht hb ha h_area
  sorry

end NUMINAMATH_GPT_depth_of_canal_l234_23448


namespace NUMINAMATH_GPT_avg_salary_increases_by_150_l234_23406

def avg_salary_increase
  (emp_avg_salary : ℕ) (num_employees : ℕ) (mgr_salary : ℕ) : ℕ :=
  let total_salary_employees := emp_avg_salary * num_employees
  let total_salary_with_mgr := total_salary_employees + mgr_salary
  let new_avg_salary := total_salary_with_mgr / (num_employees + 1)
  new_avg_salary - emp_avg_salary

theorem avg_salary_increases_by_150 :
  avg_salary_increase 1800 15 4200 = 150 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_increases_by_150_l234_23406


namespace NUMINAMATH_GPT_maximum_possible_shortest_piece_length_l234_23451

theorem maximum_possible_shortest_piece_length :
  ∃ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ 
  C = 140 ∧ (A + B + C + D + E = 640) ∧ A = 80 :=
by
  sorry

end NUMINAMATH_GPT_maximum_possible_shortest_piece_length_l234_23451


namespace NUMINAMATH_GPT_ironed_clothing_count_l234_23450

theorem ironed_clothing_count : 
  (4 * 2 + 5 * 3) + (3 * 3 + 4 * 2) + (2 * 1 + 3 * 1) = 45 := by
  sorry

end NUMINAMATH_GPT_ironed_clothing_count_l234_23450


namespace NUMINAMATH_GPT_xy_system_solution_l234_23443

theorem xy_system_solution (x y : ℝ) (h₁ : x + 5 * y = 6) (h₂ : 3 * x - y = 2) : x + y = 2 := 
by 
  sorry

end NUMINAMATH_GPT_xy_system_solution_l234_23443


namespace NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_l234_23460

theorem greatest_divisor_of_arithmetic_sequence (x c : ℤ) (h_odd : x % 2 = 1) (h_even : c % 2 = 0) :
  15 ∣ (15 * (x + 7 * c)) :=
sorry

end NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sequence_l234_23460


namespace NUMINAMATH_GPT_find_divisor_l234_23411

theorem find_divisor (D N : ℕ) (k l : ℤ)
  (h1 : N % D = 255)
  (h2 : (2 * N) % D = 112) :
  D = 398 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_find_divisor_l234_23411


namespace NUMINAMATH_GPT_evaluate_expression_l234_23492

theorem evaluate_expression : 2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l234_23492


namespace NUMINAMATH_GPT_degree_sum_interior_angles_of_star_l234_23436

-- Definitions based on conditions provided.
def extended_polygon_star (n : Nat) (h : n ≥ 6) : Nat := 
  180 * (n - 2)

-- Theorem to prove the degree-sum of the interior angles.
theorem degree_sum_interior_angles_of_star (n : Nat) (h : n ≥ 6) : 
  extended_polygon_star n h = 180 * (n - 2) :=
by
  sorry

end NUMINAMATH_GPT_degree_sum_interior_angles_of_star_l234_23436


namespace NUMINAMATH_GPT_meaningful_expr_l234_23480

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_GPT_meaningful_expr_l234_23480


namespace NUMINAMATH_GPT_triple_g_eq_nineteen_l234_23455

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 3 else 2 * n + 1

theorem triple_g_eq_nineteen : g (g (g 1)) = 19 := by
  sorry

end NUMINAMATH_GPT_triple_g_eq_nineteen_l234_23455


namespace NUMINAMATH_GPT_digit_for_multiple_of_9_l234_23410

theorem digit_for_multiple_of_9 (d : ℕ) : (23450 + d) % 9 = 0 ↔ d = 4 := by
  sorry

end NUMINAMATH_GPT_digit_for_multiple_of_9_l234_23410


namespace NUMINAMATH_GPT_solution_set_of_inequality_l234_23440

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  { x : ℝ | |f (x - 2)| > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l234_23440


namespace NUMINAMATH_GPT_correct_subtraction_l234_23489

/-- Given a number n where subtracting 63 results in 8,
we aim to find the result of subtracting 36 from n
and proving that the result is 35. -/
theorem correct_subtraction (n : ℕ) (h : n - 63 = 8) : n - 36 = 35 :=
by
  sorry

end NUMINAMATH_GPT_correct_subtraction_l234_23489


namespace NUMINAMATH_GPT_solution_set_of_f_gt_7_minimum_value_of_m_n_l234_23427

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem solution_set_of_f_gt_7 :
  { x : ℝ | f x > 7 } = { x | x > 4 ∨ x < -3 } :=
by
  ext x
  sorry

theorem minimum_value_of_m_n (m n : ℝ) (h : 0 < m ∧ 0 < n) (hfmin : ∀ x : ℝ, f x ≥ m + n) :
  m = n ∧ m = 3 / 2 ∧ m^2 + n^2 = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_f_gt_7_minimum_value_of_m_n_l234_23427


namespace NUMINAMATH_GPT_no_positive_integer_solution_l234_23428

theorem no_positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ¬ (∃ (k : ℕ), (xy + 1) * (xy + x + 2) = k^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_positive_integer_solution_l234_23428


namespace NUMINAMATH_GPT_min_value_expression_min_value_expression_achieved_at_1_l234_23421

noncomputable def min_value_expr (a b : ℝ) (n : ℕ) : ℝ :=
  (1 / (1 + a^n)) + (1 / (1 + b^n))

theorem min_value_expression (a b : ℝ) (n : ℕ) (h1 : a + b = 2) (h2 : 0 < a) (h3 : 0 < b) : 
  (min_value_expr a b n) ≥ 1 :=
sorry

theorem min_value_expression_achieved_at_1 (n : ℕ) :
  (min_value_expr 1 1 n = 1) :=
sorry

end NUMINAMATH_GPT_min_value_expression_min_value_expression_achieved_at_1_l234_23421


namespace NUMINAMATH_GPT_jane_payment_per_bulb_l234_23478

theorem jane_payment_per_bulb :
  let tulip_bulbs := 20
  let iris_bulbs := tulip_bulbs / 2
  let daffodil_bulbs := 30
  let crocus_bulbs := 3 * daffodil_bulbs
  let total_bulbs := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let total_earned := 75
  let payment_per_bulb := total_earned / total_bulbs
  payment_per_bulb = 0.50 := 
by
  sorry

end NUMINAMATH_GPT_jane_payment_per_bulb_l234_23478


namespace NUMINAMATH_GPT_hours_to_seconds_l234_23432

theorem hours_to_seconds : 
  (3.5 * 60 * 60) = 12600 := 
by 
  sorry

end NUMINAMATH_GPT_hours_to_seconds_l234_23432


namespace NUMINAMATH_GPT_algebraic_expression_correct_l234_23434

theorem algebraic_expression_correct (a b : ℝ) (h : a = 7 - 3 * b) : a^2 + 6 * a * b + 9 * b^2 = 49 := 
by sorry

end NUMINAMATH_GPT_algebraic_expression_correct_l234_23434


namespace NUMINAMATH_GPT_inequality_a_inequality_b_l234_23415

theorem inequality_a (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A + R_B + R_C + R_D) * (1 / d_A + 1 / d_B + 1 / d_C + 1 / d_D) ≥ 48 :=
sorry

theorem inequality_b (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A^2 + R_B^2 + R_C^2 + R_D^2) * (1 / d_A^2 + 1 / d_B^2 + 1 / d_C^2 + 1 / d_D^2) ≥ 144 :=
sorry

end NUMINAMATH_GPT_inequality_a_inequality_b_l234_23415


namespace NUMINAMATH_GPT_total_distance_covered_l234_23417

-- Define the distances for each segment of Biker Bob's journey
def distance1 : ℕ := 45 -- 45 miles west
def distance2 : ℕ := 25 -- 25 miles northwest
def distance3 : ℕ := 35 -- 35 miles south
def distance4 : ℕ := 50 -- 50 miles east

-- Statement to prove that the total distance covered is 155 miles
theorem total_distance_covered : distance1 + distance2 + distance3 + distance4 = 155 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_distance_covered_l234_23417


namespace NUMINAMATH_GPT_satisfies_conditions_l234_23493

open Real

def point_P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

def condition1 (a : ℝ) : Prop := (point_P a).fst = 0

def condition2 (a : ℝ) : Prop := (point_P a).snd = 5

def condition3 (a : ℝ) : Prop := abs ((point_P a).fst) = abs ((point_P a).snd)

theorem satisfies_conditions :
  ∃ P : ℝ × ℝ, P = (12, 12) ∨ P = (-12, -12) ∨ P = (4, -4) ∨ P = (-4, 4) :=
by
  sorry

end NUMINAMATH_GPT_satisfies_conditions_l234_23493


namespace NUMINAMATH_GPT_meeting_time_coincides_l234_23464

variables (distance_ab : ℕ) (speed_train_a : ℕ) (start_time_train_a : ℕ) (distance_at_9am : ℕ) (speed_train_b : ℕ) (start_time_train_b : ℕ)

def total_distance_ab := 465
def train_a_speed := 60
def train_b_speed := 75
def start_time_a := 8
def start_time_b := 9
def distance_train_a_by_9am := train_a_speed * (start_time_b - start_time_a)
def remaining_distance := total_distance_ab - distance_train_a_by_9am
def relative_speed := train_a_speed + train_b_speed
def time_to_meet := remaining_distance / relative_speed

theorem meeting_time_coincides :
  time_to_meet = 3 → (start_time_b + time_to_meet = 12) :=
by
  sorry

end NUMINAMATH_GPT_meeting_time_coincides_l234_23464


namespace NUMINAMATH_GPT_compute_HHHH_of_3_l234_23470

def H (x : ℝ) : ℝ := -0.5 * x^2 + 3 * x

theorem compute_HHHH_of_3 :
  H (H (H (H 3))) = 2.689453125 := by
  sorry

end NUMINAMATH_GPT_compute_HHHH_of_3_l234_23470


namespace NUMINAMATH_GPT_units_digit_base8_of_sum_34_8_47_8_l234_23431

def is_units_digit (n m : ℕ) (u : ℕ) := (n + m) % 8 = u

theorem units_digit_base8_of_sum_34_8_47_8 :
  ∀ (n m : ℕ), n = 34 ∧ m = 47 → (is_units_digit (n % 8) (m % 8) 3) :=
by
  intros n m h
  rw [h.1, h.2]
  sorry

end NUMINAMATH_GPT_units_digit_base8_of_sum_34_8_47_8_l234_23431


namespace NUMINAMATH_GPT_joan_writing_time_l234_23494

theorem joan_writing_time
  (total_time : ℕ)
  (time_piano : ℕ)
  (time_reading : ℕ)
  (time_exerciser : ℕ)
  (h1 : total_time = 120)
  (h2 : time_piano = 30)
  (h3 : time_reading = 38)
  (h4 : time_exerciser = 27) : 
  total_time - (time_piano + time_reading + time_exerciser) = 25 :=
by
  sorry

end NUMINAMATH_GPT_joan_writing_time_l234_23494


namespace NUMINAMATH_GPT_certain_number_l234_23408

theorem certain_number (x : ℝ) (h : x - 4 = 2) : x^2 - 3 * x = 18 :=
by
  -- Proof yet to be completed
  sorry

end NUMINAMATH_GPT_certain_number_l234_23408


namespace NUMINAMATH_GPT_initial_ratio_l234_23484

-- Define the initial number of horses and cows
def initial_horses (H : ℕ) : Prop := H = 120
def initial_cows (C : ℕ) : Prop := C = 20

-- Define the conditions of the problem
def condition1 (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)
def condition2 (H C : ℕ) : Prop := H - 15 = C + 15 + 70

-- The statement that initial ratio is 6:1
theorem initial_ratio (H C : ℕ) (h1 : condition1 H C) (h2 : condition2 H C) : 
  H = 6 * C :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_ratio_l234_23484


namespace NUMINAMATH_GPT_car_speed_l234_23419

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end NUMINAMATH_GPT_car_speed_l234_23419


namespace NUMINAMATH_GPT_solution_set_l234_23401

-- Define the two conditions as hypotheses
variables (x : ℝ)

def condition1 : Prop := x + 6 ≤ 8
def condition2 : Prop := x - 7 < 2 * (x - 3)

-- The statement to prove
theorem solution_set (h1 : condition1 x) (h2 : condition2 x) : -1 < x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l234_23401


namespace NUMINAMATH_GPT_teal_sold_pumpkin_pies_l234_23442

def pies_sold 
  (pumpkin_pie_slices : ℕ) (pumpkin_pie_price : ℕ) 
  (custard_pie_slices : ℕ) (custard_pie_price : ℕ) 
  (custard_pies_sold : ℕ) (total_revenue : ℕ) : ℕ :=
  total_revenue / (pumpkin_pie_slices * pumpkin_pie_price)

theorem teal_sold_pumpkin_pies : 
  pies_sold 8 5 6 6 5 340 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_teal_sold_pumpkin_pies_l234_23442


namespace NUMINAMATH_GPT_sum_of_squares_eq_two_l234_23472

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end NUMINAMATH_GPT_sum_of_squares_eq_two_l234_23472


namespace NUMINAMATH_GPT_exists_positive_integer_m_such_that_sqrt_8m_is_integer_l234_23433

theorem exists_positive_integer_m_such_that_sqrt_8m_is_integer :
  ∃ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), 8 * m = k^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_m_such_that_sqrt_8m_is_integer_l234_23433


namespace NUMINAMATH_GPT_total_cookies_l234_23496

theorem total_cookies (x y : Nat) (h1 : x = 137) (h2 : y = 251) : x * y = 34387 := by
  sorry

end NUMINAMATH_GPT_total_cookies_l234_23496


namespace NUMINAMATH_GPT_pure_imaginary_k_l234_23426

theorem pure_imaginary_k (k : ℝ) :
  (2 * k^2 - 3 * k - 2 = 0) → (k^2 - 2 * k ≠ 0) → k = -1 / 2 :=
by
  intro hr hi
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_pure_imaginary_k_l234_23426


namespace NUMINAMATH_GPT_union_complement_l234_23402

-- Definitions of the sets
def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Statement of the proof problem
theorem union_complement : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end NUMINAMATH_GPT_union_complement_l234_23402


namespace NUMINAMATH_GPT_time_after_4350_minutes_is_march_6_00_30_l234_23416

-- Define the start time as a date
def startDate := (2015, 3, 3, 0, 0) -- March 3, 2015 at midnight (00:00)

-- Define the total minutes to add
def totalMinutes := 4350

-- Function to convert minutes to a date and time given a start date
def addMinutes (date : (Nat × Nat × Nat × Nat × Nat)) (minutes : Nat) : (Nat × Nat × Nat × Nat × Nat) :=
  let hours := minutes / 60
  let remainMinutes := minutes % 60
  let days := hours / 24
  let remainHours := hours % 24
  let (year, month, day, hour, min) := date
  (year, month, day + days, remainHours, remainMinutes)

-- Expected result date and time
def expectedDate := (2015, 3, 6, 0, 30) -- March 6, 2015 at 00:30 AM

theorem time_after_4350_minutes_is_march_6_00_30 :
  addMinutes startDate totalMinutes = expectedDate :=
by
  sorry

end NUMINAMATH_GPT_time_after_4350_minutes_is_march_6_00_30_l234_23416


namespace NUMINAMATH_GPT_neg_forall_sin_gt_zero_l234_23465

theorem neg_forall_sin_gt_zero :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 := 
sorry

end NUMINAMATH_GPT_neg_forall_sin_gt_zero_l234_23465


namespace NUMINAMATH_GPT_subtract_base8_l234_23458

theorem subtract_base8 (a b : ℕ) (h₁ : a = 0o2101) (h₂ : b = 0o1245) :
  a - b = 0o634 := sorry

end NUMINAMATH_GPT_subtract_base8_l234_23458


namespace NUMINAMATH_GPT_rotated_squares_overlap_area_l234_23466

noncomputable def total_overlap_area (side_length : ℝ) : ℝ :=
  let base_area := side_length ^ 2
  3 * base_area

theorem rotated_squares_overlap_area : total_overlap_area 8 = 192 := by
  sorry

end NUMINAMATH_GPT_rotated_squares_overlap_area_l234_23466


namespace NUMINAMATH_GPT_sqrt_last_digit_l234_23456

-- Definitions related to the problem
def is_p_adic_number (α : ℕ) (p : ℕ) := true -- assume this definition captures p-adic number system

-- Problem statement in Lean 4
theorem sqrt_last_digit (p α a1 b1 : ℕ) (hα : is_p_adic_number α p) (h_last_digit_α : α % p = a1)
  (h_sqrt : (b1 * b1) % p = α % p) :
  (b1 * b1) % p = a1 :=
by sorry

end NUMINAMATH_GPT_sqrt_last_digit_l234_23456


namespace NUMINAMATH_GPT_bus_remaining_distance_l234_23420

noncomputable def final_distance (z x : ℝ) : ℝ :=
  z - (z * x / 5)

theorem bus_remaining_distance (z : ℝ) :
  (z / 2) / (z - 19.2) = x ∧ (z - 12) / (z / 2) = x → final_distance z x = 6.4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bus_remaining_distance_l234_23420


namespace NUMINAMATH_GPT_wax_current_amount_l234_23473

theorem wax_current_amount (wax_needed wax_total : ℕ) (h : wax_needed + 11 = wax_total) : 11 = wax_total - wax_needed :=
by
  sorry

end NUMINAMATH_GPT_wax_current_amount_l234_23473


namespace NUMINAMATH_GPT_double_acute_angle_l234_23409

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end NUMINAMATH_GPT_double_acute_angle_l234_23409


namespace NUMINAMATH_GPT_cube_less_than_three_times_square_l234_23407

theorem cube_less_than_three_times_square (x : ℤ) : x^3 < 3 * x^2 → x = 1 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_less_than_three_times_square_l234_23407


namespace NUMINAMATH_GPT_find_number_l234_23413

theorem find_number (x : ℕ) (h : x - 18 = 3 * (86 - x)) : x = 69 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l234_23413


namespace NUMINAMATH_GPT_parabola_ordinate_l234_23412

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end NUMINAMATH_GPT_parabola_ordinate_l234_23412


namespace NUMINAMATH_GPT_only_solution_is_two_l234_23487

theorem only_solution_is_two :
  ∀ n : ℕ, (Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_only_solution_is_two_l234_23487


namespace NUMINAMATH_GPT_initial_pieces_of_fruit_l234_23441

-- Definitions for the given problem
def pieces_eaten_in_first_four_days : ℕ := 5
def pieces_kept_for_next_week : ℕ := 2
def pieces_brought_to_school : ℕ := 3

-- Problem statement
theorem initial_pieces_of_fruit 
  (pieces_eaten : ℕ)
  (pieces_kept : ℕ)
  (pieces_brought : ℕ)
  (h1 : pieces_eaten = pieces_eaten_in_first_four_days)
  (h2 : pieces_kept = pieces_kept_for_next_week)
  (h3 : pieces_brought = pieces_brought_to_school) :
  pieces_eaten + pieces_kept + pieces_brought = 10 := 
sorry

end NUMINAMATH_GPT_initial_pieces_of_fruit_l234_23441


namespace NUMINAMATH_GPT_total_wheels_in_parking_lot_l234_23424

-- Definitions (conditions)
def cars := 14
def wheels_per_car := 4
def missing_wheels_per_missing_car := 1
def missing_cars := 2

def bikes := 5
def wheels_per_bike := 2

def unicycles := 3
def wheels_per_unicycle := 1

def twelve_wheeler_trucks := 2
def wheels_per_twelve_wheeler_truck := 12
def damaged_wheels_per_twelve_wheeler_truck := 3
def damaged_twelve_wheeler_trucks := 1

def eighteen_wheeler_trucks := 1
def wheels_per_eighteen_wheeler_truck := 18

-- The total wheels calculation proof
theorem total_wheels_in_parking_lot :
  ((cars * wheels_per_car - missing_cars * missing_wheels_per_missing_car) +
   (bikes * wheels_per_bike) +
   (unicycles * wheels_per_unicycle) +
   (twelve_wheeler_trucks * wheels_per_twelve_wheeler_truck - damaged_twelve_wheeler_trucks * damaged_wheels_per_twelve_wheeler_truck) +
   (eighteen_wheeler_trucks * wheels_per_eighteen_wheeler_truck)) = 106 := by
  sorry

end NUMINAMATH_GPT_total_wheels_in_parking_lot_l234_23424


namespace NUMINAMATH_GPT_plant_ways_count_l234_23452

theorem plant_ways_count :
  ∃ (solutions : Finset (Fin 7 → ℕ)), 
    (∀ x ∈ solutions, (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 = 10) ∧ 
                       (100 * x 0 + 200 * x 1 + 300 * x 2 + 150 * x 3 + 125 * x 4 + 125 * x 5 = 2500)) ∧
    (solutions.card = 8) :=
sorry

end NUMINAMATH_GPT_plant_ways_count_l234_23452


namespace NUMINAMATH_GPT_fraction_of_loss_is_correct_l234_23483

-- Definitions based on the conditions
def selling_price : ℕ := 18
def cost_price : ℕ := 19

-- Calculating the loss
def loss : ℕ := cost_price - selling_price

-- Fraction of the loss compared to the cost price
def fraction_of_loss : ℚ := loss / cost_price

-- The theorem we want to prove
theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 19 := by
  sorry

end NUMINAMATH_GPT_fraction_of_loss_is_correct_l234_23483


namespace NUMINAMATH_GPT_smallest_n_satisfies_conditions_l234_23486

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfies_conditions_l234_23486


namespace NUMINAMATH_GPT_rectangle_area_l234_23430

theorem rectangle_area (P : ℕ) (a : ℕ) (b : ℕ) (h₁ : P = 2 * (a + b)) (h₂ : P = 40) (h₃ : a = 5) : a * b = 75 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l234_23430


namespace NUMINAMATH_GPT_milk_production_l234_23423

theorem milk_production (a b c x y z w : ℕ) : 
  ((b:ℝ) / c) * w + ((y:ℝ) / z) * w = (bw / c) + (yw / z) := sorry

end NUMINAMATH_GPT_milk_production_l234_23423


namespace NUMINAMATH_GPT_min_square_value_l234_23463

theorem min_square_value (a b : ℤ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ r : ℤ, r^2 = 15 * a + 16 * b)
  (h2 : ∃ s : ℤ, s^2 = 16 * a - 15 * b) : 
  231361 ≤ min (15 * a + 16 * b) (16 * a - 15 * b) :=
sorry

end NUMINAMATH_GPT_min_square_value_l234_23463


namespace NUMINAMATH_GPT_liquor_and_beer_cost_l234_23445

-- Define the variables and conditions
variables (p_liquor p_beer : ℕ)

-- Main theorem to prove
theorem liquor_and_beer_cost (h1 : 2 * p_liquor + 12 * p_beer = 56)
                             (h2 : p_liquor = 8 * p_beer) :
  p_liquor + p_beer = 18 :=
sorry

end NUMINAMATH_GPT_liquor_and_beer_cost_l234_23445


namespace NUMINAMATH_GPT_number_of_mango_trees_l234_23449

-- Define the conditions
variable (M : Nat) -- Number of mango trees
def num_papaya_trees := 2
def papayas_per_tree := 10
def mangos_per_tree := 20
def total_fruits := 80

-- Prove that the number of mango trees M is equal to 3
theorem number_of_mango_trees : 20 + (mangos_per_tree * M) = total_fruits -> M = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_mango_trees_l234_23449


namespace NUMINAMATH_GPT_min_total_bananas_l234_23444

noncomputable def total_bananas_condition (b1 b2 b3 : ℕ) : Prop :=
  let m1 := (5/8 : ℚ) * b1 + (5/16 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m2 := (3/16 : ℚ) * b1 + (3/8 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m3 := (3/16 : ℚ) * b1 + (5/16 : ℚ) * b2 + (1/24 : ℚ) * b3
  (((m1 : ℚ) * 4) = ((m2 : ℚ) * 3)) ∧ (((m1 : ℚ) * 4) = ((m3 : ℚ) * 2))

theorem min_total_bananas : ∃ (b1 b2 b3 : ℕ), b1 + b2 + b3 = 192 ∧ total_bananas_condition b1 b2 b3 :=
sorry

end NUMINAMATH_GPT_min_total_bananas_l234_23444


namespace NUMINAMATH_GPT_remy_water_usage_l234_23418

theorem remy_water_usage :
  ∃ R : ℕ, (Remy = 3 * R + 1) ∧ 
    (Riley = R + (3 * R + 1) - 2) ∧ 
    (R + (3 * R + 1) + (R + (3 * R + 1) - 2) = 48) ∧ 
    (Remy = 19) :=
sorry

end NUMINAMATH_GPT_remy_water_usage_l234_23418


namespace NUMINAMATH_GPT_percentage_unloaded_at_second_store_l234_23435

theorem percentage_unloaded_at_second_store
  (initial_weight : ℝ)
  (percent_unloaded_first : ℝ)
  (remaining_weight_after_deliveries : ℝ)
  (remaining_weight_after_first : ℝ)
  (weight_unloaded_second : ℝ)
  (percent_unloaded_second : ℝ) :
  initial_weight = 50000 →
  percent_unloaded_first = 0.10 →
  remaining_weight_after_deliveries = 36000 →
  remaining_weight_after_first = initial_weight * (1 - percent_unloaded_first) →
  weight_unloaded_second = remaining_weight_after_first - remaining_weight_after_deliveries →
  percent_unloaded_second = (weight_unloaded_second / remaining_weight_after_first) * 100 →
  percent_unloaded_second = 20 :=
by
  intros _
  sorry

end NUMINAMATH_GPT_percentage_unloaded_at_second_store_l234_23435


namespace NUMINAMATH_GPT_r_at_6_l234_23405

-- Define the monic quintic polynomial r(x) with given conditions
def r (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + 2 

-- Given conditions
axiom r_1 : r 1 = 3
axiom r_2 : r 2 = 7
axiom r_3 : r 3 = 13
axiom r_4 : r 4 = 21
axiom r_5 : r 5 = 31

-- Proof goal
theorem r_at_6 : r 6 = 158 :=
by
  sorry

end NUMINAMATH_GPT_r_at_6_l234_23405


namespace NUMINAMATH_GPT_f_shift_l234_23491

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem f_shift (x : ℝ) : f (x - 1) = x^2 - 4 * x + 3 := by
  sorry

end NUMINAMATH_GPT_f_shift_l234_23491


namespace NUMINAMATH_GPT_john_new_total_lifting_capacity_is_correct_l234_23477

def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50

def new_clean_and_jerk : ℕ := 2 * initial_clean_and_jerk
def new_snatch : ℕ := initial_snatch + (initial_snatch * 8 / 10)

def new_combined_total_capacity : ℕ := new_clean_and_jerk + new_snatch

theorem john_new_total_lifting_capacity_is_correct : 
  new_combined_total_capacity = 250 := by
  sorry

end NUMINAMATH_GPT_john_new_total_lifting_capacity_is_correct_l234_23477


namespace NUMINAMATH_GPT_total_marbles_left_is_correct_l234_23429

def marbles_left_after_removal : ℕ :=
  let red_initial := 80
  let blue_initial := 120
  let green_initial := 75
  let yellow_initial := 50
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10
  let yellow_removed := 25
  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed
  red_left + blue_left + green_left + yellow_left

theorem total_marbles_left_is_correct :
  marbles_left_after_removal = 213 :=
  by
    sorry

end NUMINAMATH_GPT_total_marbles_left_is_correct_l234_23429


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l234_23485

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I: When a = 2, solve the inequality f(x) < 4
theorem part_I_solution (x : ℝ) : f x 2 < 4 ↔ x > -1/2 ∧ x < 7/2 :=
by sorry

-- Part II: Range of values for a such that f(x) ≥ 2 for all x
theorem part_II_solution (a : ℝ) : (∀ x, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l234_23485


namespace NUMINAMATH_GPT_minimum_value_of_expression_l234_23425

theorem minimum_value_of_expression {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l234_23425


namespace NUMINAMATH_GPT_problem_statement_l234_23454

noncomputable def a (n : ℕ) := n^2

theorem problem_statement (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  x + a n / x ^ n ≥ n + 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l234_23454


namespace NUMINAMATH_GPT_finite_fraction_n_iff_l234_23476

theorem finite_fraction_n_iff (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℕ), n * (n + 1) = 2^a * 5^b) ↔ (n = 1 ∨ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_finite_fraction_n_iff_l234_23476


namespace NUMINAMATH_GPT_problem1_l234_23404

noncomputable def sqrt7_minus_1_pow_0 : ℝ := (Real.sqrt 7 - 1)^0
noncomputable def minus_half_pow_neg_2 : ℝ := (-1 / 2)^(-2 : ℤ)
noncomputable def sqrt3_tan_30 : ℝ := Real.sqrt 3 * Real.tan (Real.pi / 6)

theorem problem1 : sqrt7_minus_1_pow_0 - minus_half_pow_neg_2 + sqrt3_tan_30 = -2 := by
  sorry

end NUMINAMATH_GPT_problem1_l234_23404


namespace NUMINAMATH_GPT_value_of_m_l234_23462

theorem value_of_m 
  (m : ℝ)
  (h1 : |m - 1| = 1)
  (h2 : m - 2 ≠ 0) : 
  m = 0 :=
  sorry

end NUMINAMATH_GPT_value_of_m_l234_23462


namespace NUMINAMATH_GPT_opposite_of_6_is_neg_6_l234_23490

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end NUMINAMATH_GPT_opposite_of_6_is_neg_6_l234_23490


namespace NUMINAMATH_GPT_set_intersection_complement_l234_23467

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}
def complement_B : Set ℝ := U \ B
def expected_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem set_intersection_complement :
  A ∩ complement_B = expected_set :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l234_23467


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_14_l234_23498

theorem sum_of_squares_of_roots_eq_14 {α β γ : ℝ}
  (h1: ∀ x: ℝ, (x^3 - 6*x^2 + 11*x - 6 = 0) → (x = α ∨ x = β ∨ x = γ)) :
  α^2 + β^2 + γ^2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_14_l234_23498


namespace NUMINAMATH_GPT_even_function_iff_b_zero_l234_23468

theorem even_function_iff_b_zero (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c) = ((-x)^2 + b * (-x) + c)) ↔ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_function_iff_b_zero_l234_23468


namespace NUMINAMATH_GPT_range_of_a_l234_23488

variable {x : ℝ} {a : ℝ}

theorem range_of_a (h : ∀ x : ℝ, ¬ (x^2 - 5*x + (5/4)*a > 0)) : 5 < a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l234_23488


namespace NUMINAMATH_GPT_sheilas_hours_mwf_is_24_l234_23474

-- Define Sheila's earning conditions and working hours
def sheilas_hours_mwf (H : ℕ) : Prop :=
  let hours_tu_th := 6 * 2
  let earnings_tu_th := hours_tu_th * 14
  let earnings_mwf := 504 - earnings_tu_th
  H = earnings_mwf / 14

-- The theorem to state that Sheila works 24 hours on Monday, Wednesday, and Friday
theorem sheilas_hours_mwf_is_24 : sheilas_hours_mwf 24 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sheilas_hours_mwf_is_24_l234_23474


namespace NUMINAMATH_GPT_ratio_of_green_to_blue_l234_23439

def balls (total blue red green yellow : ℕ) : Prop :=
  total = 36 ∧ blue = 6 ∧ red = 4 ∧ yellow = 2 * red ∧ green = total - (blue + red + yellow)

theorem ratio_of_green_to_blue (total blue red green yellow : ℕ) (h : balls total blue red green yellow) :
  (green / blue = 3) :=
by
  -- Unpack the conditions
  obtain ⟨total_eq, blue_eq, red_eq, yellow_eq, green_eq⟩ := h
  -- Simplify values based on the given conditions
  have blue_val := blue_eq
  have green_val := green_eq
  rw [blue_val, green_val]
  sorry

end NUMINAMATH_GPT_ratio_of_green_to_blue_l234_23439


namespace NUMINAMATH_GPT_lloyd_total_hours_worked_l234_23453

-- Conditions
def regular_hours_per_day : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.5
def total_earnings : ℝ := 67.5

-- Proof problem
theorem lloyd_total_hours_worked :
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_earnings := regular_hours_per_day * regular_rate
  let earnings_from_overtime := total_earnings - regular_earnings
  let hours_of_overtime := earnings_from_overtime / overtime_rate
  let total_hours := regular_hours_per_day + hours_of_overtime
  total_hours = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_lloyd_total_hours_worked_l234_23453


namespace NUMINAMATH_GPT_garden_ratio_length_to_width_l234_23495

theorem garden_ratio_length_to_width (width length : ℕ) (area : ℕ) 
  (h1 : area = 507) 
  (h2 : width = 13) 
  (h3 : length * width = area) :
  length / width = 3 :=
by
  -- Proof to be filled in.
  sorry

end NUMINAMATH_GPT_garden_ratio_length_to_width_l234_23495


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l234_23499

theorem geometric_sequence_sixth_term (a : ℕ) (a2 : ℝ) (aₖ : ℕ → ℝ) (r : ℝ) (k : ℕ) (h1 : a = 3) (h2 : a2 = -1/6) (h3 : ∀ n, aₖ n = a * r^(n-1)) (h4 : r = a2 / a) (h5 : k = 6) :
  aₖ k = -1 / 629856 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l234_23499


namespace NUMINAMATH_GPT_solve_abs_eqn_l234_23400

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 11) ↔ (y = 3.5) := by
  sorry

end NUMINAMATH_GPT_solve_abs_eqn_l234_23400


namespace NUMINAMATH_GPT_find_second_bag_weight_l234_23479

variable (initialWeight : ℕ) (firstBagWeight : ℕ) (totalWeight : ℕ)

theorem find_second_bag_weight 
  (h1: initialWeight = 15)
  (h2: firstBagWeight = 15)
  (h3: totalWeight = 40) :
  totalWeight - (initialWeight + firstBagWeight) = 10 :=
  sorry

end NUMINAMATH_GPT_find_second_bag_weight_l234_23479


namespace NUMINAMATH_GPT_f_15_equals_227_l234_23469

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem f_15_equals_227 : f 15 = 227 := by
  sorry

end NUMINAMATH_GPT_f_15_equals_227_l234_23469


namespace NUMINAMATH_GPT_vets_recommend_yummy_dog_kibble_l234_23414

theorem vets_recommend_yummy_dog_kibble :
  (let total_vets := 1000
   let percentage_puppy_kibble := 20
   let vets_puppy_kibble := (percentage_puppy_kibble * total_vets) / 100
   let diff_yummy_puppy := 100
   let vets_yummy_kibble := vets_puppy_kibble + diff_yummy_puppy
   let percentage_yummy_kibble := (vets_yummy_kibble * 100) / total_vets
   percentage_yummy_kibble = 30) :=
by
  sorry

end NUMINAMATH_GPT_vets_recommend_yummy_dog_kibble_l234_23414


namespace NUMINAMATH_GPT_arctan_sum_property_l234_23457

open Real

theorem arctan_sum_property (x y z : ℝ) :
  arctan x + arctan y + arctan z = π / 2 → x * y + y * z + x * z = 1 :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_property_l234_23457


namespace NUMINAMATH_GPT_points_per_game_l234_23446

theorem points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  sorry

end NUMINAMATH_GPT_points_per_game_l234_23446


namespace NUMINAMATH_GPT_lena_candy_bars_l234_23403

/-- Lena has some candy bars. She needs 5 more candy bars to have 3 times as many as Kevin,
and Kevin has 4 candy bars less than Nicole. Lena has 5 more candy bars than Nicole.
How many candy bars does Lena have? -/
theorem lena_candy_bars (L K N : ℕ) 
  (h1 : L + 5 = 3 * K)
  (h2 : K = N - 4)
  (h3 : L = N + 5) : 
  L = 16 :=
sorry

end NUMINAMATH_GPT_lena_candy_bars_l234_23403


namespace NUMINAMATH_GPT_value_of_x_in_logarithm_equation_l234_23461

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_x_in_logarithm_equation (n : ℝ) (h1 : n = 343) : 
  ∃ (x : ℝ), log_base x n + log_base 7 n = log_base 1 n :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_in_logarithm_equation_l234_23461


namespace NUMINAMATH_GPT_existence_of_function_implies_a_le_1_l234_23422

open Real

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem existence_of_function_implies_a_le_1 (a : ℝ) :
  (∃ f : positive_reals → positive_reals, ∀ x : positive_reals, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_existence_of_function_implies_a_le_1_l234_23422


namespace NUMINAMATH_GPT_set_intersection_l234_23475

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l234_23475


namespace NUMINAMATH_GPT_range_of_a_l234_23482

-- Definitions of sets and the problem conditions
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}
def condition (a : ℝ) : Prop := P ∪ M a = P

-- The theorem stating what needs to be proven
theorem range_of_a (a : ℝ) (h : condition a) : -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l234_23482


namespace NUMINAMATH_GPT_find_u_minus_v_l234_23481

theorem find_u_minus_v (u v : ℚ) (h1 : 5 * u - 6 * v = 31) (h2 : 3 * u + 5 * v = 4) : u - v = 5.3 := by
  sorry

end NUMINAMATH_GPT_find_u_minus_v_l234_23481


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l234_23438

noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

theorem instantaneous_velocity_at_2 :
  (deriv S 2) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l234_23438
