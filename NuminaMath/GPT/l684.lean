import Mathlib

namespace NUMINAMATH_GPT_weighted_average_AC_l684_68469

theorem weighted_average_AC (avgA avgB avgC wA wB wC total_weight: ℝ)
  (h_avgA : avgA = 7.3)
  (h_avgB : avgB = 7.6) 
  (h_avgC : avgC = 7.2)
  (h_wA : wA = 3)
  (h_wB : wB = 4)
  (h_wC : wC = 1)
  (h_total_weight : total_weight = 5) :
  ((avgA * wA + avgC * wC) / total_weight) = 5.82 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_AC_l684_68469


namespace NUMINAMATH_GPT_base_conversion_l684_68432

theorem base_conversion (b : ℝ) (h : 2 * b^2 + 3 = 51) : b = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_l684_68432


namespace NUMINAMATH_GPT_parametric_equation_of_line_passing_through_M_l684_68441

theorem parametric_equation_of_line_passing_through_M (
  t : ℝ
) : 
    ∃ x y : ℝ, 
      x = 1 + (t * (Real.cos (Real.pi / 3))) ∧ 
      y = 5 + (t * (Real.sin (Real.pi / 3))) ∧ 
      x = 1 + (1/2) * t ∧ 
      y = 5 + (Real.sqrt 3 / 2) * t := 
by
  sorry

end NUMINAMATH_GPT_parametric_equation_of_line_passing_through_M_l684_68441


namespace NUMINAMATH_GPT_clock_90_degree_angle_times_l684_68431

noncomputable def first_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 90

noncomputable def second_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 270

theorem clock_90_degree_angle_times :
  ∃ t₁ t₂ : ℝ,
  first_time_90_degree_angle t₁ ∧ 
  second_time_90_degree_angle t₂ ∧ 
  t₁ = (180 / 11 : ℝ) ∧ 
  t₂ = (540 / 11 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_clock_90_degree_angle_times_l684_68431


namespace NUMINAMATH_GPT_son_l684_68429

-- Define the context of the problem with conditions
variables (S M : ℕ)

-- Condition 1: The man is 28 years older than his son
def condition1 : Prop := M = S + 28

-- Condition 2: In two years, the man's age will be twice the son's age
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The final statement to prove the son's present age
theorem son's_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 26 :=
by
  sorry

end NUMINAMATH_GPT_son_l684_68429


namespace NUMINAMATH_GPT_surface_area_sphere_l684_68444

-- Definitions based on conditions
def SA : ℝ := 3
def SB : ℝ := 4
def SC : ℝ := 5
def vertices_perpendicular : Prop := ∀ (a b c : ℝ), (a = SA ∧ b = SB ∧ c = SC) → (a * b * c = SA * SB * SC)

-- Definition of the theorem based on problem and correct answer
theorem surface_area_sphere (h1 : vertices_perpendicular) : 
  4 * Real.pi * ((Real.sqrt (SA^2 + SB^2 + SC^2)) / 2)^2 = 50 * Real.pi :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_surface_area_sphere_l684_68444


namespace NUMINAMATH_GPT_nat_representable_as_sequence_or_difference_l684_68467

theorem nat_representable_as_sequence_or_difference
  (a : ℕ → ℕ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, a n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, k ≠ l ∧ (m = a k ∨ m = a k - a l) :=
by
  sorry

end NUMINAMATH_GPT_nat_representable_as_sequence_or_difference_l684_68467


namespace NUMINAMATH_GPT_cookies_in_box_l684_68418

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end NUMINAMATH_GPT_cookies_in_box_l684_68418


namespace NUMINAMATH_GPT_central_cell_value_l684_68413

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end NUMINAMATH_GPT_central_cell_value_l684_68413


namespace NUMINAMATH_GPT_function_has_three_zeros_l684_68453

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end NUMINAMATH_GPT_function_has_three_zeros_l684_68453


namespace NUMINAMATH_GPT_find_number_l684_68456

variable (x : ℝ)

theorem find_number (h : 0.20 * x = 0.40 * 140 + 80) : x = 680 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l684_68456


namespace NUMINAMATH_GPT_abs_diff_ge_abs_sum_iff_non_positive_prod_l684_68477

theorem abs_diff_ge_abs_sum_iff_non_positive_prod (a b : ℝ) : 
  |a - b| ≥ |a| + |b| ↔ a * b ≤ 0 := 
by sorry

end NUMINAMATH_GPT_abs_diff_ge_abs_sum_iff_non_positive_prod_l684_68477


namespace NUMINAMATH_GPT_units_digit_of_N_is_8_l684_68497

def product_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens * units

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

theorem units_digit_of_N_is_8 (N : ℕ) (hN_range : 10 ≤ N ∧ N < 100)
    (hN_eq : N = product_of_digits N * sum_of_digits N) : N % 10 = 8 :=
sorry

end NUMINAMATH_GPT_units_digit_of_N_is_8_l684_68497


namespace NUMINAMATH_GPT_conservation_of_mass_l684_68455

def molecular_weight_C := 12.01
def molecular_weight_H := 1.008
def molecular_weight_O := 16.00
def molecular_weight_Na := 22.99

def molecular_weight_C9H8O4 := (9 * molecular_weight_C) + (8 * molecular_weight_H) + (4 * molecular_weight_O)
def molecular_weight_NaOH := molecular_weight_Na + molecular_weight_O + molecular_weight_H
def molecular_weight_C7H6O3 := (7 * molecular_weight_C) + (6 * molecular_weight_H) + (3 * molecular_weight_O)
def molecular_weight_CH3COONa := (2 * molecular_weight_C) + (3 * molecular_weight_H) + (2 * molecular_weight_O) + molecular_weight_Na

theorem conservation_of_mass :
  (molecular_weight_C9H8O4 + molecular_weight_NaOH) = (molecular_weight_C7H6O3 + molecular_weight_CH3COONa) := by
  sorry

end NUMINAMATH_GPT_conservation_of_mass_l684_68455


namespace NUMINAMATH_GPT_solution_to_equation1_solution_to_equation2_l684_68401

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := 3 * x^3 + 4 = -20

-- State the theorems with the correct answers
theorem solution_to_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = -3) :=
by
  sorry

theorem solution_to_equation2 (x : ℝ) : equation2 x ↔ (x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_equation1_solution_to_equation2_l684_68401


namespace NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l684_68443

theorem largest_n_for_factorable_polynomial :
  ∃ (n : ℤ), (∀ A B : ℤ, 7 * A * B = 56 → n ≤ 7 * B + A) ∧ n = 393 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l684_68443


namespace NUMINAMATH_GPT_discriminant_of_quadratic_eq_l684_68436

/-- The discriminant of a quadratic equation -/
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_eq : discriminant 1 3 (-1) = 13 := by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_eq_l684_68436


namespace NUMINAMATH_GPT_emma_average_speed_last_segment_l684_68445

open Real

theorem emma_average_speed_last_segment :
  ∀ (d1 d2 d3 : ℝ) (t1 t2 t3 : ℝ),
    d1 + d2 + d3 = 120 →
    t1 + t2 + t3 = 2 →
    t1 = 2 / 3 → t2 = 2 / 3 → 
    t1 = d1 / 50 → t2 = d2 / 55 → 
    ∃ x : ℝ, t3 = d3 / x ∧ x = 75 := 
by
  intros d1 d2 d3 t1 t2 t3 h1 h2 ht1 ht2 hs1 hs2
  use 75 / (2 / 3)
  -- skipped proof for simplicity
  sorry

end NUMINAMATH_GPT_emma_average_speed_last_segment_l684_68445


namespace NUMINAMATH_GPT_fg_of_3_is_83_l684_68450

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_is_83_l684_68450


namespace NUMINAMATH_GPT_students_not_picked_correct_l684_68485

-- Define the total number of students and the number of students picked for the team
def total_students := 17
def students_picked := 3 * 4

-- Define the number of students who didn't get picked based on the conditions
noncomputable def students_not_picked : ℕ := total_students - students_picked

-- The theorem stating the problem
theorem students_not_picked_correct : students_not_picked = 5 := 
by 
  sorry

end NUMINAMATH_GPT_students_not_picked_correct_l684_68485


namespace NUMINAMATH_GPT_two_marbles_different_colors_probability_l684_68419

-- Definitions
def red_marbles : Nat := 3
def green_marbles : Nat := 4
def white_marbles : Nat := 5
def blue_marbles : Nat := 3
def total_marbles : Nat := red_marbles + green_marbles + white_marbles + blue_marbles

-- Combinations of different colored marbles
def red_green : Nat := red_marbles * green_marbles
def red_white : Nat := red_marbles * white_marbles
def red_blue : Nat := red_marbles * blue_marbles
def green_white : Nat := green_marbles * white_marbles
def green_blue : Nat := green_marbles * blue_marbles
def white_blue : Nat := white_marbles * blue_marbles

-- Total favorable outcomes
def total_favorable : Nat := red_green + red_white + red_blue + green_white + green_blue + white_blue

-- Total outcomes when drawing 2 marbles from the jar
def total_outcomes : Nat := Nat.choose total_marbles 2

-- Probability calculation
noncomputable def probability_different_colors : Rat := total_favorable / total_outcomes

-- Proof that the probability is 83/105
theorem two_marbles_different_colors_probability :
  probability_different_colors = 83 / 105 := by
  sorry

end NUMINAMATH_GPT_two_marbles_different_colors_probability_l684_68419


namespace NUMINAMATH_GPT_emails_in_inbox_l684_68424

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end NUMINAMATH_GPT_emails_in_inbox_l684_68424


namespace NUMINAMATH_GPT_doves_count_l684_68457

theorem doves_count 
  (num_doves : ℕ)
  (num_eggs_per_dove : ℕ)
  (hatch_rate : ℚ)
  (initial_doves : num_doves = 50)
  (eggs_per_dove : num_eggs_per_dove = 5)
  (hatch_fraction : hatch_rate = 7/9) :
  (num_doves + Int.toNat ((hatch_rate * num_doves * num_eggs_per_dove).floor)) = 244 :=
by
  sorry

end NUMINAMATH_GPT_doves_count_l684_68457


namespace NUMINAMATH_GPT_theorem1_theorem2_theorem3_l684_68409

-- Given conditions as definitions
variables {x y p q : ℝ}

-- Condition definitions
def condition1 : x + y = -p := sorry
def condition2 : x * y = q := sorry

-- Theorems to be proved
theorem theorem1 (h1 : x + y = -p) (h2 : x * y = q) : x^2 + y^2 = p^2 - 2 * q := sorry

theorem theorem2 (h1 : x + y = -p) (h2 : x * y = q) : x^3 + y^3 = -p^3 + 3 * p * q := sorry

theorem theorem3 (h1 : x + y = -p) (h2 : x * y = q) : x^4 + y^4 = p^4 - 4 * p^2 * q + 2 * q^2 := sorry

end NUMINAMATH_GPT_theorem1_theorem2_theorem3_l684_68409


namespace NUMINAMATH_GPT_car_avg_speed_l684_68494

def avg_speed_problem (d1 d2 t : ℕ) : ℕ :=
  (d1 + d2) / t

theorem car_avg_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 70) (h2 : d2 = 90) (ht : t = 2) :
  avg_speed_problem d1 d2 t = 80 := by
  sorry

end NUMINAMATH_GPT_car_avg_speed_l684_68494


namespace NUMINAMATH_GPT_line_intersects_hyperbola_l684_68499

theorem line_intersects_hyperbola (k : Real) : 
  (∃ x y : Real, y = k * x ∧ (x^2) / 9 - (y^2) / 4 = 1) ↔ (-2 / 3 < k ∧ k < 2 / 3) := 
sorry

end NUMINAMATH_GPT_line_intersects_hyperbola_l684_68499


namespace NUMINAMATH_GPT_evaluate_expression_l684_68430

theorem evaluate_expression : (-2)^3 - (-3)^2 = -17 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l684_68430


namespace NUMINAMATH_GPT_initial_pollykawgs_computation_l684_68473

noncomputable def initial_pollykawgs_in_pond (daily_rate_matured : ℕ) (daily_rate_caught : ℕ)
  (total_days : ℕ) (catch_days : ℕ) : ℕ :=
let first_phase := (daily_rate_matured + daily_rate_caught) * catch_days
let second_phase := daily_rate_matured * (total_days - catch_days)
first_phase + second_phase

theorem initial_pollykawgs_computation :
  initial_pollykawgs_in_pond 50 10 44 20 = 2400 :=
by sorry

end NUMINAMATH_GPT_initial_pollykawgs_computation_l684_68473


namespace NUMINAMATH_GPT_relay_race_l684_68484

theorem relay_race (n : ℕ) (H1 : 2004 % n = 0) (H2 : n ≤ 168) (H3 : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6 ∧ n ≠ 12): n = 167 :=
by
  sorry

end NUMINAMATH_GPT_relay_race_l684_68484


namespace NUMINAMATH_GPT_average_first_21_multiples_of_8_l684_68422

noncomputable def average_of_multiples (n : ℕ) (a : ℕ) : ℕ :=
  let sum := (n * (a + a * n)) / 2
  sum / n

theorem average_first_21_multiples_of_8 : average_of_multiples 21 8 = 88 :=
by
  sorry

end NUMINAMATH_GPT_average_first_21_multiples_of_8_l684_68422


namespace NUMINAMATH_GPT_radius_ratio_l684_68435

noncomputable def volume_large_sphere : ℝ := 432 * Real.pi

noncomputable def volume_small_sphere : ℝ := 0.08 * volume_large_sphere

noncomputable def radius_large_sphere : ℝ :=
  (3 * volume_large_sphere / (4 * Real.pi)) ^ (1 / 3)

noncomputable def radius_small_sphere : ℝ :=
  (3 * volume_small_sphere / (4 * Real.pi)) ^ (1 / 3)

theorem radius_ratio (V_L V_s : ℝ) (hL : V_L = 432 * Real.pi) (hS : V_s = 0.08 * V_L) :
  (radius_small_sphere / radius_large_sphere) = (2/5)^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_radius_ratio_l684_68435


namespace NUMINAMATH_GPT_total_hours_eq_52_l684_68496

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end NUMINAMATH_GPT_total_hours_eq_52_l684_68496


namespace NUMINAMATH_GPT_even_iff_a_zero_monotonous_iff_a_range_max_value_l684_68482

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 2

-- (I) Prove that f(x) is even on [-5, 5] if and only if a = 0
theorem even_iff_a_zero (a : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a = f (-x) a) ↔ a = 0 := sorry

-- (II) Prove that f(x) is monotonous on [-5, 5] if and only if a ≥ 10 or a ≤ -10
theorem monotonous_iff_a_range (a : ℝ) : (∀ x y : ℝ, -5 ≤ x ∧ x ≤ y ∧ y ≤ 5 → f x a ≤ f y a) ↔ (a ≥ 10 ∨ a ≤ -10) := sorry

-- (III) Prove the maximum value of f(x) in the interval [-5, 5]
theorem max_value (a : ℝ) : (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ (∀ y : ℝ, -5 ≤ y ∧ y ≤ 5 → f y a ≤ f x a)) ∧  
                           ((a ≥ 0 → f 5 a = 27 + 5 * a) ∧ (a < 0 → f (-5) a = 27 - 5 * a)) := sorry

end NUMINAMATH_GPT_even_iff_a_zero_monotonous_iff_a_range_max_value_l684_68482


namespace NUMINAMATH_GPT_all_points_on_single_quadratic_l684_68476

theorem all_points_on_single_quadratic (points : Fin 100 → (ℝ × ℝ)) :
  (∀ (p1 p2 p3 p4 : Fin 100),
    ∃ a b c : ℝ, 
      ∀ (i : Fin 100), 
        (i = p1 ∨ i = p2 ∨ i = p3 ∨ i = p4) →
          (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c) → 
  ∃ a b c : ℝ, ∀ i : Fin 100, (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c :=
by 
  sorry

end NUMINAMATH_GPT_all_points_on_single_quadratic_l684_68476


namespace NUMINAMATH_GPT_harper_water_intake_l684_68408

theorem harper_water_intake
  (cases_cost : ℕ := 12)
  (cases_count : ℕ := 24)
  (total_spent : ℕ)
  (days : ℕ)
  (total_days_spent : ℕ := 240)
  (total_money_spent: ℕ := 60)
  (total_water: ℕ := 5 * 24)
  (water_per_day : ℝ := 0.5):
  total_spent = total_money_spent ->
  days = total_days_spent ->
  water_per_day = (total_water : ℝ) / total_days_spent :=
by
  sorry

end NUMINAMATH_GPT_harper_water_intake_l684_68408


namespace NUMINAMATH_GPT_sequence_sum_l684_68437

theorem sequence_sum :
  (3 + 13 + 23 + 33 + 43 + 53) + (5 + 15 + 25 + 35 + 45 + 55) = 348 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_l684_68437


namespace NUMINAMATH_GPT_base_of_triangle_is_24_l684_68438

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end NUMINAMATH_GPT_base_of_triangle_is_24_l684_68438


namespace NUMINAMATH_GPT_find_a_find_b_plus_c_l684_68488

variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Condition: Given that 2a / cos A = (3c - 2b) / cos B
axiom condition1 : 2 * a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B)

-- Condition 1: b = sqrt(5) * sin B
axiom condition2 : b = Real.sqrt 5 * (Real.sin B)

-- Proof problem for finding a
theorem find_a : a = 5 / 3 := by
  sorry

-- Condition 2: a = sqrt(6) and the area is sqrt(5) / 2
axiom condition3 : a = Real.sqrt 6
axiom condition4 : 1 / 2 * b * c * (Real.sin A) = Real.sqrt 5 / 2

-- Proof problem for finding b + c
theorem find_b_plus_c : b + c = 4 := by
  sorry

end NUMINAMATH_GPT_find_a_find_b_plus_c_l684_68488


namespace NUMINAMATH_GPT_dice_probability_same_face_l684_68423

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end NUMINAMATH_GPT_dice_probability_same_face_l684_68423


namespace NUMINAMATH_GPT_original_water_amount_in_mixture_l684_68495

-- Define heat calculations and conditions
def latentHeatOfFusionIce : ℕ := 80       -- Latent heat of fusion for ice in cal/g
def initialTempWaterAdded : ℕ := 20      -- Initial temperature of added water in °C
def finalTempMixture : ℕ := 5            -- Final temperature of the mixture in °C
def specificHeatWater : ℕ := 1           -- Specific heat of water in cal/g°C

-- Define the known parameters of the problem
def totalMass : ℕ := 250               -- Total mass of the initial mixture in grams
def addedMassWater : ℕ := 1000         -- Mass of added water in grams
def initialTempMixtureIceWater : ℕ := 0  -- Initial temperature of the ice-water mixture in °C

-- Define the equation that needs to be solved
theorem original_water_amount_in_mixture (x : ℝ) :
  (250 - x) * 80 + (250 - x) * 5 + x * 5 = 15000 →
  x = 90.625 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_water_amount_in_mixture_l684_68495


namespace NUMINAMATH_GPT_number_of_square_free_odd_integers_between_1_and_200_l684_68414

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end NUMINAMATH_GPT_number_of_square_free_odd_integers_between_1_and_200_l684_68414


namespace NUMINAMATH_GPT_repair_cost_l684_68493

theorem repair_cost (purchase_price transport_charges selling_price profit_percentage R : ℝ)
  (h1 : purchase_price = 10000)
  (h2 : transport_charges = 1000)
  (h3 : selling_price = 24000)
  (h4 : profit_percentage = 0.5)
  (h5 : selling_price = (1 + profit_percentage) * (purchase_price + R + transport_charges)) :
  R = 5000 :=
by
  sorry

end NUMINAMATH_GPT_repair_cost_l684_68493


namespace NUMINAMATH_GPT_find_other_number_l684_68480

variable (A B : ℕ)
variable (LCM : ℕ → ℕ → ℕ)
variable (HCF : ℕ → ℕ → ℕ)

theorem find_other_number (h1 : LCM A B = 2310) 
  (h2 : HCF A B = 30) (h3 : A = 210) : B = 330 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l684_68480


namespace NUMINAMATH_GPT_negation_universal_prop_l684_68425

theorem negation_universal_prop:
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
  sorry

end NUMINAMATH_GPT_negation_universal_prop_l684_68425


namespace NUMINAMATH_GPT_range_of_x_l684_68461

def star (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x (x : ℝ) (h : star x (x - 2) < 0) : -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l684_68461


namespace NUMINAMATH_GPT_integer_solutions_of_xyz_equation_l684_68458

/--
  Find all integer solutions of the equation \( x + y + z = xyz \).
  The integer solutions are expected to be:
  \[
  (1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (1, 3, 2), (2, 3, 1), (-a, 0, a) \text{ for } (a : ℤ).
  \]
-/
theorem integer_solutions_of_xyz_equation (x y z : ℤ) :
    x + y + z = x * y * z ↔ 
    (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
    (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ 
    ∃ a : ℤ, (x = -a ∧ y = 0 ∧ z = a) := by
  sorry


end NUMINAMATH_GPT_integer_solutions_of_xyz_equation_l684_68458


namespace NUMINAMATH_GPT_unique_prime_sum_8_l684_68452
-- Import all necessary mathematical libraries

-- Prime number definition
def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Function definition for f(y), number of unique ways to sum primes to form y
def f (y : Nat) : Nat :=
  if y = 8 then 2 else sorry -- We're assuming the correct answer to state the theorem; in a real proof, we would define this correctly.

theorem unique_prime_sum_8 :
  f 8 = 2 :=
by
  -- The proof goes here, but for now, we leave it as a placeholder.
  sorry

end NUMINAMATH_GPT_unique_prime_sum_8_l684_68452


namespace NUMINAMATH_GPT_scientific_notation_of_12000000000_l684_68463

theorem scientific_notation_of_12000000000 :
  12000000000 = 1.2 * 10^10 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_12000000000_l684_68463


namespace NUMINAMATH_GPT_sin_cos_tan_l684_68460

theorem sin_cos_tan (α : ℝ) (h1 : Real.tan α = 3) : Real.sin α * Real.cos α = 3 / 10 := 
sorry

end NUMINAMATH_GPT_sin_cos_tan_l684_68460


namespace NUMINAMATH_GPT_ratio_equality_proof_l684_68411

theorem ratio_equality_proof
  (m n k a b c x y z : ℝ)
  (h : x / (m * (n * b + k * c - m * a)) = y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = k / (z * (a * x + b * y - c * z)) :=
by
  sorry

end NUMINAMATH_GPT_ratio_equality_proof_l684_68411


namespace NUMINAMATH_GPT_trigonometric_identity_l684_68417

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = 10 / 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l684_68417


namespace NUMINAMATH_GPT_expansion_eq_coeff_sum_l684_68486

theorem expansion_eq_coeff_sum (a : ℕ → ℤ) (m : ℤ) 
  (h : (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7)
  (h_coeff : a 4 = -35) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1 ∧ a 1 + a 3 + a 5 + a 7 = 26 := 
by 
  sorry

end NUMINAMATH_GPT_expansion_eq_coeff_sum_l684_68486


namespace NUMINAMATH_GPT_problem_statement_l684_68478

theorem problem_statement 
  (x1 y1 x2 y2 x3 y3 x4 y4 a b c : ℝ)
  (h1 : x1 > 0) (h2 : y1 > 0)
  (h3 : x2 < 0) (h4 : y2 > 0)
  (h5 : x3 < 0) (h6 : y3 < 0)
  (h7 : x4 > 0) (h8 : y4 < 0)
  (h9 : (x1 - a)^2 + (y1 - b)^2 ≤ c^2)
  (h10 : (x2 - a)^2 + (y2 - b)^2 ≤ c^2)
  (h11 : (x3 - a)^2 + (y3 - b)^2 ≤ c^2)
  (h12 : (x4 - a)^2 + (y4 - b)^2 ≤ c^2) : a^2 + b^2 < c^2 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l684_68478


namespace NUMINAMATH_GPT_range_of_sum_of_products_l684_68410

theorem range_of_sum_of_products (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)
  (h_sum : a + b + c = (Real.sqrt 3) / 2) :
  0 < (a * b + b * c + c * a) ∧ (a * b + b * c + c * a) ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_sum_of_products_l684_68410


namespace NUMINAMATH_GPT_amusement_park_ticket_cost_l684_68475

/-- Jeremie is going to an amusement park with 3 friends. 
    The cost of a set of snacks is $5. 
    The total cost for everyone to go to the amusement park and buy snacks is $92.
    Prove that the cost of one ticket is $18.
-/
theorem amusement_park_ticket_cost 
  (number_of_people : ℕ)
  (snack_cost_per_person : ℕ)
  (total_cost : ℕ)
  (ticket_cost : ℕ) :
  number_of_people = 4 → 
  snack_cost_per_person = 5 → 
  total_cost = 92 → 
  ticket_cost = 18 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_amusement_park_ticket_cost_l684_68475


namespace NUMINAMATH_GPT_sum_of_intersections_l684_68405

theorem sum_of_intersections :
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (∀ x y : ℝ, y = (x - 2)^2 ↔ x + 1 = (y - 2)^2) ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 20) :=
sorry

end NUMINAMATH_GPT_sum_of_intersections_l684_68405


namespace NUMINAMATH_GPT_cost_of_dozen_pens_l684_68470

variable (x : ℝ) (pen_cost pencil_cost : ℝ)
variable (h1 : 3 * pen_cost + 5 * pencil_cost = 260)
variable (h2 : pen_cost / pencil_cost = 5)

theorem cost_of_dozen_pens (x_pos : 0 < x) 
    (pen_cost_def : pen_cost = 5 * x) 
    (pencil_cost_def : pencil_cost = x) :
    12 * pen_cost = 780 := by
  sorry

end NUMINAMATH_GPT_cost_of_dozen_pens_l684_68470


namespace NUMINAMATH_GPT_range_a_I_range_a_II_l684_68407

variable (a: ℝ)

-- Define the proposition p and q
def p := (Real.sqrt (a^2 + 13) > Real.sqrt 17)
def q := ∀ x, (0 < x ∧ x < 3) → (x^2 - 2 * a * x - 2 = 0)

-- Prove question (I): If proposition p is true, find the range of the real number $a$
theorem range_a_I (h_p : p a) : a < -2 ∨ a > 2 :=
by sorry

-- Prove question (II): If both the proposition "¬q" and "p ∧ q" are false, find the range of the real number $a$
theorem range_a_II (h_neg_q : ¬ q a) (h_p_and_q : ¬ (p a ∧ q a)) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_range_a_I_range_a_II_l684_68407


namespace NUMINAMATH_GPT_cannot_be_expressed_as_x_squared_plus_y_fifth_l684_68468

theorem cannot_be_expressed_as_x_squared_plus_y_fifth :
  ¬ ∃ x y : ℤ, 59121 = x^2 + y^5 :=
by sorry

end NUMINAMATH_GPT_cannot_be_expressed_as_x_squared_plus_y_fifth_l684_68468


namespace NUMINAMATH_GPT_combined_total_cost_is_correct_l684_68446

-- Define the number and costs of balloons for each person
def Fred_yellow_count : ℕ := 5
def Fred_red_count : ℕ := 3
def Fred_yellow_cost_per : ℕ := 3
def Fred_red_cost_per : ℕ := 4

def Sam_yellow_count : ℕ := 6
def Sam_red_count : ℕ := 4
def Sam_yellow_cost_per : ℕ := 4
def Sam_red_cost_per : ℕ := 5

def Mary_yellow_count : ℕ := 7
def Mary_red_count : ℕ := 5
def Mary_yellow_cost_per : ℕ := 5
def Mary_red_cost_per : ℕ := 6

def Susan_yellow_count : ℕ := 4
def Susan_red_count : ℕ := 6
def Susan_yellow_cost_per : ℕ := 6
def Susan_red_cost_per : ℕ := 7

def Tom_yellow_count : ℕ := 10
def Tom_red_count : ℕ := 8
def Tom_yellow_cost_per : ℕ := 2
def Tom_red_cost_per : ℕ := 3

-- Formula to calculate total cost for a given person
def total_cost (yellow_count red_count yellow_cost_per red_cost_per : ℕ) : ℕ :=
  (yellow_count * yellow_cost_per) + (red_count * red_cost_per)

-- Total costs for each person
def Fred_total_cost := total_cost Fred_yellow_count Fred_red_count Fred_yellow_cost_per Fred_red_cost_per
def Sam_total_cost := total_cost Sam_yellow_count Sam_red_count Sam_yellow_cost_per Sam_red_cost_per
def Mary_total_cost := total_cost Mary_yellow_count Mary_red_count Mary_yellow_cost_per Mary_red_cost_per
def Susan_total_cost := total_cost Susan_yellow_count Susan_red_count Susan_yellow_cost_per Susan_red_cost_per
def Tom_total_cost := total_cost Tom_yellow_count Tom_red_count Tom_yellow_cost_per Tom_red_cost_per

-- Combined total cost
def combined_total_cost : ℕ :=
  Fred_total_cost + Sam_total_cost + Mary_total_cost + Susan_total_cost + Tom_total_cost

-- Lean statement to prove
theorem combined_total_cost_is_correct : combined_total_cost = 246 :=
by
  dsimp [combined_total_cost, Fred_total_cost, Sam_total_cost, Mary_total_cost, Susan_total_cost, Tom_total_cost, total_cost]
  sorry

end NUMINAMATH_GPT_combined_total_cost_is_correct_l684_68446


namespace NUMINAMATH_GPT_project_time_for_A_l684_68462

/--
A can complete a project in some days and B can complete the same project in 30 days.
If A and B start working on the project together and A quits 5 days before the project is 
completed, the project will be completed in 15 days.
Prove that A can complete the project alone in 20 days.
-/
theorem project_time_for_A (x : ℕ) (h : 10 * (1 / x + 1 / 30) + 5 * (1 / 30) = 1) : x = 20 :=
sorry

end NUMINAMATH_GPT_project_time_for_A_l684_68462


namespace NUMINAMATH_GPT_route_length_is_140_l684_68420

-- Conditions of the problem
variable (D : ℝ)  -- Length of the route
variable (Vx Vy t : ℝ)  -- Speeds of Train X and Train Y, and time to meet

-- Given conditions
axiom train_X_trip_time : D / Vx = 4
axiom train_Y_trip_time : D / Vy = 3
axiom train_X_distance_when_meet : Vx * t = 60
axiom total_distance_covered_on_meeting : Vx * t + Vy * t = D

-- Goal: Prove that the length of the route is 140 kilometers
theorem route_length_is_140 : D = 140 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_route_length_is_140_l684_68420


namespace NUMINAMATH_GPT_RachelFurnitureAssemblyTime_l684_68434

/-- Rachel bought seven new chairs and three new tables for her house.
    She spent four minutes on each piece of furniture putting it together.
    Prove that it took her 40 minutes to finish putting together all the furniture. -/
theorem RachelFurnitureAssemblyTime :
  let chairs := 7
  let tables := 3
  let time_per_piece := 4
  let total_time := (chairs + tables) * time_per_piece
  total_time = 40 := by
    sorry

end NUMINAMATH_GPT_RachelFurnitureAssemblyTime_l684_68434


namespace NUMINAMATH_GPT_truth_values_of_p_and_q_l684_68421

theorem truth_values_of_p_and_q (p q : Prop) (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬p) : ¬p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_truth_values_of_p_and_q_l684_68421


namespace NUMINAMATH_GPT_sum_reciprocal_l684_68415

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end NUMINAMATH_GPT_sum_reciprocal_l684_68415


namespace NUMINAMATH_GPT_workers_new_daily_wage_l684_68427

def wage_before : ℝ := 25
def increase_percentage : ℝ := 0.40

theorem workers_new_daily_wage : wage_before * (1 + increase_percentage) = 35 :=
by
  -- sorry will be replaced by the actual proof steps
  sorry

end NUMINAMATH_GPT_workers_new_daily_wage_l684_68427


namespace NUMINAMATH_GPT_simple_interest_rate_l684_68406

theorem simple_interest_rate (P T SI : ℝ) (hP : P = 10000) (hT : T = 1) (hSI : SI = 400) :
    (SI = P * 0.04 * T) := by
  rw [hP, hT, hSI]
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l684_68406


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l684_68472

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) : 
  x / y = 37 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l684_68472


namespace NUMINAMATH_GPT_inverse_g_of_neg_92_l684_68440

noncomputable def g (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 1

theorem inverse_g_of_neg_92 : g (-3) = -92 :=
by 
-- This would be the proof but we are skipping it as requested
sorry

end NUMINAMATH_GPT_inverse_g_of_neg_92_l684_68440


namespace NUMINAMATH_GPT_infinite_solutions_xyz_t_l684_68492

theorem infinite_solutions_xyz_t (x y z t : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : t ≠ 0) (h5 : gcd (gcd x y) (gcd z t) = 1) :
  ∃ (x y z t : ℕ), x^3 + y^3 + z^3 = t^4 ∧ gcd (gcd x y) (gcd z t) = 1 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_xyz_t_l684_68492


namespace NUMINAMATH_GPT_perfect_square_A_plus_2B_plus_4_l684_68451

theorem perfect_square_A_plus_2B_plus_4 (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9 : ℚ) * (10 ^ (2 * n) - 1)
  let B := (8 / 9 : ℚ) * (10 ^ n - 1)
  ∃ k : ℚ, A + 2 * B + 4 = k^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_perfect_square_A_plus_2B_plus_4_l684_68451


namespace NUMINAMATH_GPT_num_convex_quadrilateral_angles_arith_prog_l684_68403

theorem num_convex_quadrilateral_angles_arith_prog :
  ∃ (S : Finset (Finset ℤ)), S.card = 29 ∧
    ∀ {a b c d : ℤ}, {a, b, c, d} ∈ S →
      a + b + c + d = 360 ∧
      a < b ∧ b < c ∧ c < d ∧
      ∃ (m d_diff : ℤ), 
        m - d_diff = a ∧
        m = b ∧
        m + d_diff = c ∧
        m + 2 * d_diff = d ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end NUMINAMATH_GPT_num_convex_quadrilateral_angles_arith_prog_l684_68403


namespace NUMINAMATH_GPT_kelly_chris_boxes_ratio_l684_68447

theorem kelly_chris_boxes_ratio (X : ℝ) (h : X > 0) :
  (0.4 * X) / (0.6 * X) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_kelly_chris_boxes_ratio_l684_68447


namespace NUMINAMATH_GPT_waiter_earned_total_tips_l684_68428

def tips (c1 c2 c3 c4 c5 : ℝ) := c1 + c2 + c3 + c4 + c5

theorem waiter_earned_total_tips :
  tips 1.50 2.75 3.25 4.00 5.00 = 16.50 := 
by 
  sorry

end NUMINAMATH_GPT_waiter_earned_total_tips_l684_68428


namespace NUMINAMATH_GPT_range_of_a_l684_68449

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + a * x - 2 < 0) → a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l684_68449


namespace NUMINAMATH_GPT_value_of_x_div_y_l684_68498

theorem value_of_x_div_y (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, x = t * y ∧ t = -2 := 
sorry

end NUMINAMATH_GPT_value_of_x_div_y_l684_68498


namespace NUMINAMATH_GPT_exists_unique_c_for_a_equals_3_l684_68466

theorem exists_unique_c_for_a_equals_3 :
  ∃! c : ℝ, ∀ x ∈ Set.Icc (3 : ℝ) 9, ∃ y ∈ Set.Icc (3 : ℝ) 27, Real.log x / Real.log 3 + Real.log y / Real.log 3 = c :=
sorry

end NUMINAMATH_GPT_exists_unique_c_for_a_equals_3_l684_68466


namespace NUMINAMATH_GPT_vertex_in_second_quadrant_l684_68490

-- Theorems and properties regarding quadratic functions and their roots.
theorem vertex_in_second_quadrant (c : ℝ) (h : 4 + 4 * c < 0) : 
  (1:ℝ) * -1^2 + 2 * -1 - c > 0 :=
sorry

end NUMINAMATH_GPT_vertex_in_second_quadrant_l684_68490


namespace NUMINAMATH_GPT_xy_plus_one_ge_four_l684_68479

theorem xy_plus_one_ge_four {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x + 1) * (y + 1) >= 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_xy_plus_one_ge_four_l684_68479


namespace NUMINAMATH_GPT_union_eq_l684_68400

-- Define the sets M and N
def M : Finset ℕ := {0, 3}
def N : Finset ℕ := {1, 2, 3}

-- Define the proof statement
theorem union_eq : M ∪ N = {0, 1, 2, 3} := 
by
  sorry

end NUMINAMATH_GPT_union_eq_l684_68400


namespace NUMINAMATH_GPT_increase_factor_l684_68433

noncomputable def old_plates : ℕ := 26 * 10^3
noncomputable def new_plates : ℕ := 26^4 * 10^4
theorem increase_factor : (new_plates / old_plates) = 175760 := by
  sorry

end NUMINAMATH_GPT_increase_factor_l684_68433


namespace NUMINAMATH_GPT_regular_polygon_enclosure_l684_68459

theorem regular_polygon_enclosure (m n : ℕ) (h₁: m = 6) (h₂: (m + 1) = 7): n = 6 :=
by
  -- Lean code to include the problem hypothesis and conclude the theorem
  sorry

end NUMINAMATH_GPT_regular_polygon_enclosure_l684_68459


namespace NUMINAMATH_GPT_increasing_function_when_a_eq_2_range_of_a_for_solution_set_l684_68464

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * (x - 1) / (x + 1)

theorem increasing_function_when_a_eq_2 :
  ∀ ⦃x⦄, x > 0 → (f 2 x - f 2 1) * (x - 1) > 0 := sorry

theorem range_of_a_for_solution_set :
  ∀ ⦃a x⦄, f a x ≥ 0 ↔ (x ≥ 1) → a ≤ 1 := sorry

end NUMINAMATH_GPT_increasing_function_when_a_eq_2_range_of_a_for_solution_set_l684_68464


namespace NUMINAMATH_GPT_sandy_position_l684_68489

structure Position :=
  (x : ℤ)
  (y : ℤ)

def initial_position : Position := { x := 0, y := 0 }
def after_south : Position := { x := 0, y := -20 }
def after_east : Position := { x := 20, y := -20 }
def after_north : Position := { x := 20, y := 0 }
def final_position : Position := { x := 30, y := 0 }

theorem sandy_position :
  final_position.x - initial_position.x = 10 ∧ final_position.y - initial_position.y = 0 :=
by
  sorry

end NUMINAMATH_GPT_sandy_position_l684_68489


namespace NUMINAMATH_GPT_ab_bc_ca_value_a4_b4_c4_value_l684_68471

theorem ab_bc_ca_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ab + bc + ca = -1/2 :=
sorry

theorem a4_b4_c4_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1/2 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_value_a4_b4_c4_value_l684_68471


namespace NUMINAMATH_GPT_find_c_l684_68483

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 19 * x - 84
noncomputable def g (x : ℝ) : ℝ := 4 * x ^ 2 - 12 * x + 5

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ f x = 0)
  (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ g x = 0) :
  c = -23 / 2 := by
  sorry

end NUMINAMATH_GPT_find_c_l684_68483


namespace NUMINAMATH_GPT_sum_of_three_numbers_eq_zero_l684_68416

theorem sum_of_three_numbers_eq_zero (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : (a + b + c) / 3 = a + 20) (h3 : (a + b + c) / 3 = c - 10) (h4 : b = 10) : 
  a + b + c = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_eq_zero_l684_68416


namespace NUMINAMATH_GPT_part1_part2_l684_68412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part1 (a : ℝ) (x : ℝ) (hx1 : 1 ≤ x) (hx2 : x ≤ Real.exp 1) :
  a = 1 →
  (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 + (Real.exp 1)^2 / 2) ∧ (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 / 2) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part2 (a : ℝ) :
  (-1/2 ≤ a ∧ a ≤ 1/2) ↔
  ∀ x, 1 < x → g a x < 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l684_68412


namespace NUMINAMATH_GPT_arithmetic_sequence_l684_68454

theorem arithmetic_sequence (a : ℕ → ℝ) 
    (h : ∀ m n, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
    ∃ d, ∀ k, a k = k * d := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l684_68454


namespace NUMINAMATH_GPT_inequality_proof_l684_68448

theorem inequality_proof
  (a b c d : ℝ)
  (hpos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (hcond: (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2 * a + b + c) * (2 * b + c + d) * (2 * c + d + a) * (2 * d + a + b) * (a * b * c * d) ^ 2 ≤ 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l684_68448


namespace NUMINAMATH_GPT_findMonthlyIncome_l684_68404

-- Variables and conditions
variable (I : ℝ) -- Raja's monthly income
variable (saving : ℝ) (r1 r2 r3 r4 r5 : ℝ) -- savings and monthly percentages

-- Conditions
def condition1 : r1 = 0.45 := by sorry
def condition2 : r2 = 0.12 := by sorry
def condition3 : r3 = 0.08 := by sorry
def condition4 : r4 = 0.15 := by sorry
def condition5 : r5 = 0.10 := by sorry
def conditionSaving : saving = 5000 := by sorry

-- Define the main equation
def mainEquation (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) : Prop :=
  (r1 * I) + (r2 * I) + (r3 * I) + (r4 * I) + (r5 * I) + saving = I

-- Main theorem to prove
theorem findMonthlyIncome (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) 
  (h1 : r1 = 0.45) (h2 : r2 = 0.12) (h3 : r3 = 0.08) (h4 : r4 = 0.15) (h5 : r5 = 0.10) (hSaving : saving = 5000) :
  mainEquation I r1 r2 r3 r4 r5 saving → I = 50000 :=
  by sorry

end NUMINAMATH_GPT_findMonthlyIncome_l684_68404


namespace NUMINAMATH_GPT_fraction_comparison_l684_68439

theorem fraction_comparison (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l684_68439


namespace NUMINAMATH_GPT_baseball_card_distribution_l684_68487

theorem baseball_card_distribution (total_cards : ℕ) (capacity_4 : ℕ) (capacity_6 : ℕ) (capacity_8 : ℕ) :
  total_cards = 137 →
  capacity_4 = 4 →
  capacity_6 = 6 →
  capacity_8 = 8 →
  (total_cards % capacity_4) % capacity_6 = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_baseball_card_distribution_l684_68487


namespace NUMINAMATH_GPT_sum_of_roots_expression_involving_roots_l684_68491

variables {a b : ℝ}

axiom roots_of_quadratic :
  (a^2 + 3 * a - 2 = 0) ∧ (b^2 + 3 * b - 2 = 0)

theorem sum_of_roots :
  a + b = -3 :=
by 
  sorry

theorem expression_involving_roots :
  a^3 + 3 * a^2 + 2 * b = -6 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_expression_involving_roots_l684_68491


namespace NUMINAMATH_GPT_total_bees_count_l684_68426

-- Definitions
def initial_bees : ℕ := 16
def additional_bees : ℕ := 7

-- Problem statement to prove
theorem total_bees_count : initial_bees + additional_bees = 23 := by
  -- The proof will be given here
  sorry

end NUMINAMATH_GPT_total_bees_count_l684_68426


namespace NUMINAMATH_GPT_fish_remaining_correct_l684_68481

def guppies := 225
def angelfish := 175
def tiger_sharks := 200
def oscar_fish := 140
def discus_fish := 120

def guppies_sold := 3/5 * guppies
def angelfish_sold := 3/7 * angelfish
def tiger_sharks_sold := 1/4 * tiger_sharks
def oscar_fish_sold := 1/2 * oscar_fish
def discus_fish_sold := 2/3 * discus_fish

def guppies_remaining := guppies - guppies_sold
def angelfish_remaining := angelfish - angelfish_sold
def tiger_sharks_remaining := tiger_sharks - tiger_sharks_sold
def oscar_fish_remaining := oscar_fish - oscar_fish_sold
def discus_fish_remaining := discus_fish - discus_fish_sold

def total_remaining_fish := guppies_remaining + angelfish_remaining + tiger_sharks_remaining + oscar_fish_remaining + discus_fish_remaining

theorem fish_remaining_correct : total_remaining_fish = 450 := 
by 
  -- insert the necessary steps of the proof here
  sorry

end NUMINAMATH_GPT_fish_remaining_correct_l684_68481


namespace NUMINAMATH_GPT_find_k_l684_68402

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Vectors expressions
def k_a_add_b (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def a_sub_3b : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)

-- Condition of collinearity
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 = 0 ∨ v2.1 = 0 ∨ v1.1 * v2.2 = v1.2 * v2.1)

-- Statement to prove
theorem find_k :
  collinear (k_a_add_b (-1/3)) a_sub_3b :=
sorry

end NUMINAMATH_GPT_find_k_l684_68402


namespace NUMINAMATH_GPT_fish_count_and_total_l684_68465

-- Definitions of each friend's number of fish
def max_fish : ℕ := 6
def sam_fish : ℕ := 3 * max_fish
def joe_fish : ℕ := 9 * sam_fish
def harry_fish : ℕ := 5 * joe_fish

-- Total number of fish for all friends combined
def total_fish : ℕ := max_fish + sam_fish + joe_fish + harry_fish

-- The theorem stating the problem and corresponding solution
theorem fish_count_and_total :
  max_fish = 6 ∧
  sam_fish = 3 * max_fish ∧
  joe_fish = 9 * sam_fish ∧
  harry_fish = 5 * joe_fish ∧
  total_fish = (max_fish + sam_fish + joe_fish + harry_fish) :=
by
  repeat { sorry }

end NUMINAMATH_GPT_fish_count_and_total_l684_68465


namespace NUMINAMATH_GPT_frac_sum_diff_l684_68474

theorem frac_sum_diff (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : (a + b) / (a - b) = -1001 :=
sorry

end NUMINAMATH_GPT_frac_sum_diff_l684_68474


namespace NUMINAMATH_GPT_work_problem_l684_68442

theorem work_problem (A B C : ℝ) (hB : B = 3) (h1 : 1 / B + 1 / C = 1 / 2) (h2 : 1 / A + 1 / C = 1 / 2) : A = 3 := by
  sorry

end NUMINAMATH_GPT_work_problem_l684_68442
