import Mathlib

namespace NUMINAMATH_GPT_remainder_of_sum_of_integers_mod_15_l2269_226963

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_integers_mod_15_l2269_226963


namespace NUMINAMATH_GPT_average_salary_is_8000_l2269_226943

def average_salary_all_workers (A : ℝ) :=
  let total_workers := 30
  let technicians := 10
  let technician_salary := 12000
  let rest_workers := total_workers - technicians
  let rest_salary := 6000
  let total_salary := (technicians * technician_salary) + (rest_workers * rest_salary)
  A = total_salary / total_workers

theorem average_salary_is_8000 : average_salary_all_workers 8000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_is_8000_l2269_226943


namespace NUMINAMATH_GPT_number_of_students_l2269_226931

theorem number_of_students : 
    ∃ (n : ℕ), 
      (∃ (x : ℕ), 
        (∀ (k : ℕ), x = 4 * k ∧ 5 * x + 1 = n)
      ) ∧ 
      (∃ (y : ℕ), 
        (∀ (k : ℕ), y = 5 * k ∧ 4 * y + 1 = n)
      ) ∧
      n ≤ 30 ∧ 
      n = 21 :=
  sorry

end NUMINAMATH_GPT_number_of_students_l2269_226931


namespace NUMINAMATH_GPT_Sam_balloons_correct_l2269_226910

def Fred_balloons : Nat := 10
def Dan_balloons : Nat := 16
def Total_balloons : Nat := 72

def Sam_balloons : Nat := Total_balloons - Fred_balloons - Dan_balloons

theorem Sam_balloons_correct : Sam_balloons = 46 := by 
  have H : Sam_balloons = 72 - 10 - 16 := rfl
  simp at H
  exact H

end NUMINAMATH_GPT_Sam_balloons_correct_l2269_226910


namespace NUMINAMATH_GPT_is_isosceles_triangle_l2269_226906

theorem is_isosceles_triangle 
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a * Real.cos B + b * Real.cos C + c * Real.cos A = b * Real.cos A + c * Real.cos B + a * Real.cos C) : 
  (A = B ∨ B = C ∨ A = C) :=
sorry

end NUMINAMATH_GPT_is_isosceles_triangle_l2269_226906


namespace NUMINAMATH_GPT_find_biology_marks_l2269_226904

variable (english mathematics physics chemistry average_marks : ℕ)

theorem find_biology_marks
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_physics : physics = 92)
  (h_chemistry : chemistry = 87)
  (h_average_marks : average_marks = 89) : 
  (english + mathematics + physics + chemistry + (445 - (english + mathematics + physics + chemistry))) / 5 = average_marks :=
by
  sorry

end NUMINAMATH_GPT_find_biology_marks_l2269_226904


namespace NUMINAMATH_GPT_find_max_value_l2269_226948

noncomputable def max_value (x y z : ℝ) : ℝ := (x + y) / (x * y * z)

theorem find_max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 2) :
  max_value x y z ≤ 13.5 :=
sorry

end NUMINAMATH_GPT_find_max_value_l2269_226948


namespace NUMINAMATH_GPT_farmer_apples_l2269_226979

theorem farmer_apples (initial_apples : ℕ) (given_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 127) (h2 : given_apples = 88) 
  (h3 : final_apples = initial_apples - given_apples) : final_apples = 39 :=
by {
  -- proof steps would go here, but since only the statement is needed, we use 'sorry' to skip the proof
  sorry
}

end NUMINAMATH_GPT_farmer_apples_l2269_226979


namespace NUMINAMATH_GPT_uki_total_earnings_l2269_226925

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end NUMINAMATH_GPT_uki_total_earnings_l2269_226925


namespace NUMINAMATH_GPT_Jane_indisposed_days_l2269_226996

-- Definitions based on conditions
def John_completion_days := 18
def Jane_completion_days := 12
def total_task_days := 10.8
def work_per_day_by_john := 1 / John_completion_days
def work_per_day_by_jane := 1 / Jane_completion_days
def work_per_day_together := work_per_day_by_john + work_per_day_by_jane

-- Equivalent proof problem
theorem Jane_indisposed_days : 
  ∃ (x : ℝ), 
    (10.8 - x) * work_per_day_together + x * work_per_day_by_john = 1 ∧
    x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Jane_indisposed_days_l2269_226996


namespace NUMINAMATH_GPT_vehicle_A_no_speed_increase_needed_l2269_226909

noncomputable def V_A := 60 -- Speed of Vehicle A in mph
noncomputable def V_B := 70 -- Speed of Vehicle B in mph
noncomputable def V_C := 50 -- Speed of Vehicle C in mph
noncomputable def dist_AB := 100 -- Initial distance between A and B in ft
noncomputable def dist_AC := 300 -- Initial distance between A and C in ft

theorem vehicle_A_no_speed_increase_needed 
  (V_A V_B V_C : ℝ)
  (dist_AB dist_AC : ℝ)
  (h1 : V_A > V_C)
  (h2 : V_A = 60)
  (h3 : V_B = 70)
  (h4 : V_C = 50)
  (h5 : dist_AB = 100)
  (h6 : dist_AC = 300) : 
  ∀ ΔV : ℝ, ΔV = 0 :=
by
  sorry -- Proof to be filled out

end NUMINAMATH_GPT_vehicle_A_no_speed_increase_needed_l2269_226909


namespace NUMINAMATH_GPT_unpacked_boxes_l2269_226939

-- Definitions of boxes per case
def boxesPerCaseLemonChalet : Nat := 12
def boxesPerCaseThinMints : Nat := 15
def boxesPerCaseSamoas : Nat := 10
def boxesPerCaseTrefoils : Nat := 18

-- Definitions of boxes sold by Deborah
def boxesSoldLemonChalet : Nat := 31
def boxesSoldThinMints : Nat := 26
def boxesSoldSamoas : Nat := 17
def boxesSoldTrefoils : Nat := 44

-- The theorem stating the number of boxes that will not be packed to a case
theorem unpacked_boxes :
  boxesSoldLemonChalet % boxesPerCaseLemonChalet = 7 ∧
  boxesSoldThinMints % boxesPerCaseThinMints = 11 ∧
  boxesSoldSamoas % boxesPerCaseSamoas = 7 ∧
  boxesSoldTrefoils % boxesPerCaseTrefoils = 8 := 
by
  sorry

end NUMINAMATH_GPT_unpacked_boxes_l2269_226939


namespace NUMINAMATH_GPT_smallest_b_l2269_226968

open Real

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 2 ∧ B = a ∧ C = b ∨ A = 2 ∧ B = b ∧ C = a ∨ A = a ∧ B = b ∧ C = 2) ∧ A + B > C ∧ A + C > B ∧ B + C > A)
  (h4 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 1 / b ∧ B = 1 / a ∧ C = 2 ∨ A = 1 / a ∧ B = 1 / b ∧ C = 2 ∨ A = 1 / b ∧ B = 2 ∧ C = 1 / a ∨ A = 1 / a ∧ B = 2 ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / a ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / b ∧ C = 1 / a) ∧ A + B > C ∧ A + C > B ∧ B + C > A) :
  b = 2 := 
sorry

end NUMINAMATH_GPT_smallest_b_l2269_226968


namespace NUMINAMATH_GPT_function_property_l2269_226900

variable (g : ℝ × ℝ → ℝ)
variable (cond : ∀ x y : ℝ, g (x, y) = - g (y, x))

theorem function_property (x : ℝ) : g (x, x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_function_property_l2269_226900


namespace NUMINAMATH_GPT_incorrect_inequality_l2269_226945

theorem incorrect_inequality : ¬ (-2 < -3) :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_incorrect_inequality_l2269_226945


namespace NUMINAMATH_GPT_find_x_l2269_226976

theorem find_x : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2269_226976


namespace NUMINAMATH_GPT_molecular_weight_proof_l2269_226965

noncomputable def molecular_weight_C7H6O2 := 
  (7 * 12.01) + (6 * 1.008) + (2 * 16.00) -- molecular weight of one mole of C7H6O2

noncomputable def total_molecular_weight_9_moles := 
  9 * molecular_weight_C7H6O2 -- total molecular weight of 9 moles of C7H6O2

theorem molecular_weight_proof : 
  total_molecular_weight_9_moles = 1099.062 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_proof_l2269_226965


namespace NUMINAMATH_GPT_calc_fraction_l2269_226987

theorem calc_fraction:
  (125: ℕ) = 5 ^ 3 →
  (25: ℕ) = 5 ^ 2 →
  (25 ^ 40) / (125 ^ 20) = 5 ^ 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_calc_fraction_l2269_226987


namespace NUMINAMATH_GPT_radius_of_shorter_cylinder_l2269_226977

theorem radius_of_shorter_cylinder (h r : ℝ) (V_s V_t : ℝ) (π : ℝ) : 
  V_s = 500 → 
  V_t = 500 → 
  V_t = π * 5^2 * 4 * h → 
  V_s = π * r^2 * h → 
  r = 10 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_shorter_cylinder_l2269_226977


namespace NUMINAMATH_GPT_joe_lists_count_l2269_226908

def num_options (n : ℕ) (k : ℕ) : ℕ := n ^ k

theorem joe_lists_count : num_options 12 3 = 1728 := by
  unfold num_options
  sorry

end NUMINAMATH_GPT_joe_lists_count_l2269_226908


namespace NUMINAMATH_GPT_range_of_a_l2269_226953

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

-- Main theorem to prove
theorem range_of_a (a : ℝ) (h : a < 0)
  (h_necessary : ∀ x, ¬ p x a → ¬ q x) 
  (h_not_sufficient : ∃ x, ¬ p x a ∧ q x): 
  a ≤ -4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2269_226953


namespace NUMINAMATH_GPT_removed_number_is_24_l2269_226956

theorem removed_number_is_24
  (S9 : ℕ) (S8 : ℕ) (avg_9 : ℕ) (avg_8 : ℕ) (h1 : avg_9 = 72) (h2 : avg_8 = 78) (h3 : S9 = avg_9 * 9) (h4 : S8 = avg_8 * 8) :
  S9 - S8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_removed_number_is_24_l2269_226956


namespace NUMINAMATH_GPT_zhou_yu_age_equation_l2269_226933

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end NUMINAMATH_GPT_zhou_yu_age_equation_l2269_226933


namespace NUMINAMATH_GPT_how_many_rocks_l2269_226930

section see_saw_problem

-- Conditions
def Jack_weight : ℝ := 60
def Anna_weight : ℝ := 40
def rock_weight : ℝ := 4

-- Theorem statement
theorem how_many_rocks : (Jack_weight - Anna_weight) / rock_weight = 5 :=
by
  -- Proof is omitted, just ensuring the theorem statement
  sorry

end see_saw_problem

end NUMINAMATH_GPT_how_many_rocks_l2269_226930


namespace NUMINAMATH_GPT_sufficient_condition_for_negation_l2269_226935

theorem sufficient_condition_for_negation {A B : Prop} (h : B → A) : ¬ A → ¬ B :=
by
  intro hA
  intro hB
  apply hA
  exact h hB

end NUMINAMATH_GPT_sufficient_condition_for_negation_l2269_226935


namespace NUMINAMATH_GPT_points_on_ellipse_satisfying_dot_product_l2269_226972

theorem points_on_ellipse_satisfying_dot_product :
  ∃ P1 P2 : ℝ × ℝ,
    P1 = (0, 3) ∧ P2 = (0, -3) ∧
    ∀ P : ℝ × ℝ, 
    (P ∈ ({p : ℝ × ℝ | (p.1 / 5)^2 + (p.2 / 3)^2 = 1}) → 
     ((P.1 - (-4)) * (P.1 - 4) + P.2^2 = -7) →
     (P = P1 ∨ P = P2))
:=
sorry

end NUMINAMATH_GPT_points_on_ellipse_satisfying_dot_product_l2269_226972


namespace NUMINAMATH_GPT_correct_growth_rate_equation_l2269_226954

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end NUMINAMATH_GPT_correct_growth_rate_equation_l2269_226954


namespace NUMINAMATH_GPT_obtuse_angle_in_second_quadrant_l2269_226918

-- Let θ be an angle in degrees
def angle_in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def angle_terminal_side_same (θ₁ θ₂ : ℝ) : Prop := θ₁ % 360 = θ₂ % 360

def angle_in_fourth_quadrant (θ : ℝ) : Prop := -360 < θ ∧ θ < 0 ∧ (θ + 360) > 270

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement D: An obtuse angle is definitely in the second quadrant
theorem obtuse_angle_in_second_quadrant (θ : ℝ) (h : is_obtuse_angle θ) :
  90 < θ ∧ θ < 180 := by
    sorry

end NUMINAMATH_GPT_obtuse_angle_in_second_quadrant_l2269_226918


namespace NUMINAMATH_GPT_cos2x_quadratic_eq_specific_values_l2269_226919

variable (a b c x : ℝ)

axiom eqn1 : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0

noncomputable def quadratic_equation_cos2x 
  (a b c : ℝ) : ℝ × ℝ × ℝ := 
  (a^2, 2*a^2 + 2*a*c - b^2, a^2 + 2*a*c - b^2 + 4*c^2)

theorem cos2x_quadratic_eq 
  (a b c x : ℝ) 
  (h: a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0) :
  (a^2) * (Real.cos (2*x))^2 + 
  (2*a^2 + 2*a*c - b^2) * Real.cos (2*x) + 
  (a^2 + 2*a*c - b^2 + 4*c^2) = 0 :=
sorry

theorem specific_values : 
  quadratic_equation_cos2x 4 2 (-1) = (4, 2, -1) :=
by
  unfold quadratic_equation_cos2x
  simp
  sorry

end NUMINAMATH_GPT_cos2x_quadratic_eq_specific_values_l2269_226919


namespace NUMINAMATH_GPT_train_speed_is_72_km_per_hr_l2269_226971

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_72_km_per_hr_l2269_226971


namespace NUMINAMATH_GPT_tg_gamma_half_eq_2_div_5_l2269_226941

theorem tg_gamma_half_eq_2_div_5
  (α β γ : ℝ)
  (a b c : ℝ)
  (triangle_angles : α + β + γ = π)
  (tg_half_alpha : Real.tan (α / 2) = 5/6)
  (tg_half_beta : Real.tan (β / 2) = 10/9)
  (ac_eq_2b : a + c = 2 * b):
  Real.tan (γ / 2) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_tg_gamma_half_eq_2_div_5_l2269_226941


namespace NUMINAMATH_GPT_certain_number_l2269_226924

theorem certain_number (p q x : ℝ) (h1 : 3 / p = x) (h2 : 3 / q = 15) (h3 : p - q = 0.3) : x = 6 :=
sorry

end NUMINAMATH_GPT_certain_number_l2269_226924


namespace NUMINAMATH_GPT_gcd_n_cube_plus_27_n_plus_3_l2269_226952

theorem gcd_n_cube_plus_27_n_plus_3 (n : ℕ) (h : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
sorry

end NUMINAMATH_GPT_gcd_n_cube_plus_27_n_plus_3_l2269_226952


namespace NUMINAMATH_GPT_pie_eating_contest_l2269_226942

theorem pie_eating_contest :
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  max (max first_student second_student) third_student - 
  min (min first_student second_student) third_student = 1 / 6 :=
by
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  sorry

end NUMINAMATH_GPT_pie_eating_contest_l2269_226942


namespace NUMINAMATH_GPT_find_corresponding_element_l2269_226944

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem find_corresponding_element :
  f (-1, 2) = (-3, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_corresponding_element_l2269_226944


namespace NUMINAMATH_GPT_positive_difference_in_x_coordinates_l2269_226929

-- Define points for line l
def point_l1 : ℝ × ℝ := (0, 10)
def point_l2 : ℝ × ℝ := (2, 0)

-- Define points for line m
def point_m1 : ℝ × ℝ := (0, 3)
def point_m2 : ℝ × ℝ := (10, 0)

-- Define the proof statement with the given problem
theorem positive_difference_in_x_coordinates :
  let y := 20
  let slope_l := (point_l2.2 - point_l1.2) / (point_l2.1 - point_l1.1)
  let intersection_l_x := (y - point_l1.2) / slope_l + point_l1.1
  let slope_m := (point_m2.2 - point_m1.2) / (point_m2.1 - point_m1.1)
  let intersection_m_x := (y - point_m1.2) / slope_m + point_m1.1
  abs (intersection_l_x - intersection_m_x) = 54.67 := 
  sorry -- Proof goes here

end NUMINAMATH_GPT_positive_difference_in_x_coordinates_l2269_226929


namespace NUMINAMATH_GPT_acute_triangle_B_area_l2269_226975

-- Basic setup for the problem statement
variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to respective angles

-- The theorem to be proven
theorem acute_triangle_B_area (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) 
                              (h_sides : a = 2 * b * Real.sin A)
                              (h_a : a = 3 * Real.sqrt 3) 
                              (h_c : c = 5) : 
  B = π / 6 ∧ (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_B_area_l2269_226975


namespace NUMINAMATH_GPT_unlock_probability_l2269_226973

/--
Xiao Ming set a six-digit passcode for his phone using the numbers 0-9, but he forgot the last digit.
The probability that Xiao Ming can unlock his phone with just one try is 1/10.
-/
theorem unlock_probability (n : ℕ) (h : n ≥ 0 ∧ n ≤ 9) : 
  1 / 10 = 1 / (10 : ℝ) :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_unlock_probability_l2269_226973


namespace NUMINAMATH_GPT_square_of_negative_is_positive_l2269_226997

-- Define P as a negative integer
variable (P : ℤ) (hP : P < 0)

-- Theorem statement that P² is always positive.
theorem square_of_negative_is_positive : P^2 > 0 :=
sorry

end NUMINAMATH_GPT_square_of_negative_is_positive_l2269_226997


namespace NUMINAMATH_GPT_area_correct_l2269_226934

noncomputable def area_bounded_curves : ℝ := sorry

theorem area_correct :
  ∃ S, S = area_bounded_curves ∧ S = 12 * pi + 16 := sorry

end NUMINAMATH_GPT_area_correct_l2269_226934


namespace NUMINAMATH_GPT_numbers_not_equal_l2269_226981

theorem numbers_not_equal
  (a b c S : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a + b^2 + c^2 = S)
  (h2 : b + a^2 + c^2 = S)
  (h3 : c + a^2 + b^2 = S) :
  ¬ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_GPT_numbers_not_equal_l2269_226981


namespace NUMINAMATH_GPT_minimum_value_l2269_226932

theorem minimum_value (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 2) 
  (h4 : a + b = 1) : 
  ∃ L, L = (3 * a * c / b) + (c / (a * b)) + (6 / (c - 2)) ∧ L = 1 / (a * (1 - a)) := sorry

end NUMINAMATH_GPT_minimum_value_l2269_226932


namespace NUMINAMATH_GPT_general_form_equation_l2269_226964

theorem general_form_equation (x : ℝ) : 
  x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_general_form_equation_l2269_226964


namespace NUMINAMATH_GPT_crayons_per_box_l2269_226983

theorem crayons_per_box (total_crayons : ℕ) (total_boxes : ℕ)
  (h1 : total_crayons = 321)
  (h2 : total_boxes = 45) :
  (total_crayons / total_boxes) = 7 :=
by
  sorry

end NUMINAMATH_GPT_crayons_per_box_l2269_226983


namespace NUMINAMATH_GPT_rent_increase_percentage_l2269_226998

theorem rent_increase_percentage :
  ∀ (initial_avg new_avg rent : ℝ) (num_friends : ℝ),
    num_friends = 4 →
    initial_avg = 800 →
    new_avg = 850 →
    rent = 800 →
    ((num_friends * new_avg) - (num_friends * initial_avg)) / rent * 100 = 25 :=
by
  intros initial_avg new_avg rent num_friends h_num h_initial h_new h_rent
  sorry

end NUMINAMATH_GPT_rent_increase_percentage_l2269_226998


namespace NUMINAMATH_GPT_value_of_a_l2269_226967

theorem value_of_a {a : ℝ} (A : Set ℝ) (B : Set ℝ) (hA : A = {-1, 0, 2}) (hB : B = {2^a}) (hSub : B ⊆ A) : a = 1 := 
sorry

end NUMINAMATH_GPT_value_of_a_l2269_226967


namespace NUMINAMATH_GPT_number_is_160_l2269_226993

theorem number_is_160 (x : ℝ) (h : x / 5 + 4 = x / 4 - 4) : x = 160 :=
by
  sorry

end NUMINAMATH_GPT_number_is_160_l2269_226993


namespace NUMINAMATH_GPT_train_speed_54_kmh_l2269_226920

theorem train_speed_54_kmh
  (train_length : ℕ)
  (tunnel_length : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ := train_length + tunnel_length)
  (speed_mps : ℚ := total_distance / time_seconds)
  (conversion_factor : ℚ := 3.6) :
  train_length = 300 →
  tunnel_length = 1200 →
  time_seconds = 100 →
  speed_mps * conversion_factor = 54 := 
by
  intros h_train_length h_tunnel_length h_time_seconds
  sorry

end NUMINAMATH_GPT_train_speed_54_kmh_l2269_226920


namespace NUMINAMATH_GPT_smallest_n_satisfying_ratio_l2269_226995

-- Definitions and conditions from problem
def sum_first_n_odd_numbers_starting_from_3 (n : ℕ) : ℕ := n^2 + 2 * n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)

theorem smallest_n_satisfying_ratio :
  ∃ n : ℕ, n > 0 ∧ (sum_first_n_odd_numbers_starting_from_3 n : ℚ) / (sum_first_n_even_numbers n : ℚ) = 49 / 50 ∧ n = 51 :=
by
  use 51
  exact sorry

end NUMINAMATH_GPT_smallest_n_satisfying_ratio_l2269_226995


namespace NUMINAMATH_GPT_inverse_of_B_cubed_l2269_226961

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
def B_inv := Matrix.of ![![3, -2], ![0, -1]]
noncomputable def B_cubed_inv := ((B_inv) 3)^3

theorem inverse_of_B_cubed :
  B_inv = Matrix.of ![![27, -24], ![0, -1]] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_B_cubed_l2269_226961


namespace NUMINAMATH_GPT_solution_set_absolute_value_sum_eq_three_l2269_226955

theorem solution_set_absolute_value_sum_eq_three (m n : ℝ) (h : ∀ x : ℝ, (|2 * x - 3| ≤ 1) ↔ (m ≤ x ∧ x ≤ n)) : m + n = 3 :=
sorry

end NUMINAMATH_GPT_solution_set_absolute_value_sum_eq_three_l2269_226955


namespace NUMINAMATH_GPT_sum_of_n_terms_l2269_226959

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end NUMINAMATH_GPT_sum_of_n_terms_l2269_226959


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2269_226922

theorem quadratic_inequality_solution (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2269_226922


namespace NUMINAMATH_GPT_colored_sectors_overlap_l2269_226905

/--
Given two disks each divided into 1985 equal sectors, with 200 sectors on each disk colored arbitrarily,
and one disk is rotated by angles that are multiples of 360 degrees / 1985, 
prove that there are at least 80 positions where no more than 20 colored sectors coincide.
-/
theorem colored_sectors_overlap :
  ∀ (disks : ℕ → ℕ) (sectors_colored : ℕ),
  disks 1 = 1985 → disks 2 = 1985 →
  sectors_colored = 200 →
  ∃ (p : ℕ), p ≥ 80 ∧ (∀ (i : ℕ), (i < p → sectors_colored ≤ 20)) := 
sorry

end NUMINAMATH_GPT_colored_sectors_overlap_l2269_226905


namespace NUMINAMATH_GPT_line_slope_point_l2269_226928

theorem line_slope_point (m b : ℝ) (h_slope : m = -4) (h_point : ∃ x y : ℝ, (x, y) = (5, 2) ∧ y = m * x + b) : 
  m + b = 18 := by
  sorry

end NUMINAMATH_GPT_line_slope_point_l2269_226928


namespace NUMINAMATH_GPT_angle_B_l2269_226988

/-- 
  Given that the area of triangle ABC is (sqrt 3 / 2) 
  and the dot product of vectors AB and BC is 3, 
  prove that the measure of angle B is 5π/6. 
--/
theorem angle_B (A B C : ℝ) (a c : ℝ) (h1 : 0 ≤ B ∧ B ≤ π)
  (h_area : (1 / 2) * a * c * (Real.sin B) = (Real.sqrt 3 / 2))
  (h_dot : a * c * (Real.cos B) = -3) :
  B = 5 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_angle_B_l2269_226988


namespace NUMINAMATH_GPT_sum_of_fifth_powers_cannot_conclude_fourth_powers_l2269_226927

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_cannot_conclude_fourth_powers_l2269_226927


namespace NUMINAMATH_GPT_james_spends_on_pistachios_per_week_l2269_226926

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end NUMINAMATH_GPT_james_spends_on_pistachios_per_week_l2269_226926


namespace NUMINAMATH_GPT_max_servings_l2269_226917

-- Define available chunks for each type of fruit
def available_cantaloupe := 150
def available_honeydew := 135
def available_pineapple := 60
def available_watermelon := 220

-- Define the required chunks per serving for each type of fruit
def chunks_per_serving_cantaloupe := 3
def chunks_per_serving_honeydew := 2
def chunks_per_serving_pineapple := 1
def chunks_per_serving_watermelon := 4

-- Define the minimum required servings
def minimum_servings := 50

-- Prove the greatest number of servings that can be made while maintaining the specific ratio
theorem max_servings : 
  ∀ s : ℕ, 
  s * chunks_per_serving_cantaloupe ≤ available_cantaloupe ∧
  s * chunks_per_serving_honeydew ≤ available_honeydew ∧
  s * chunks_per_serving_pineapple ≤ available_pineapple ∧
  s * chunks_per_serving_watermelon ≤ available_watermelon ∧ 
  s ≥ minimum_servings → 
  s = 50 :=
by
  sorry

end NUMINAMATH_GPT_max_servings_l2269_226917


namespace NUMINAMATH_GPT_estimate_sqrt_expression_l2269_226901

theorem estimate_sqrt_expression :
  5 < 3 * Real.sqrt 5 - 1 ∧ 3 * Real.sqrt 5 - 1 < 6 :=
by
  sorry

end NUMINAMATH_GPT_estimate_sqrt_expression_l2269_226901


namespace NUMINAMATH_GPT_molecular_weight_one_mole_l2269_226923

variable (molecular_weight : ℕ → ℕ)

theorem molecular_weight_one_mole (h : molecular_weight 7 = 2856) :
  molecular_weight 1 = 408 :=
sorry

end NUMINAMATH_GPT_molecular_weight_one_mole_l2269_226923


namespace NUMINAMATH_GPT_cubic_polynomial_solution_l2269_226958

theorem cubic_polynomial_solution 
  (p : ℚ → ℚ) 
  (h1 : p 1 = 1)
  (h2 : p 2 = 1 / 4)
  (h3 : p 3 = 1 / 9)
  (h4 : p 4 = 1 / 16)
  (h6 : p 6 = 1 / 36)
  (h0 : p 0 = -1 / 25) : 
  p 5 = 20668 / 216000 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_solution_l2269_226958


namespace NUMINAMATH_GPT_find_x_l2269_226992
-- Lean 4 equivalent problem setup

-- Assuming a and b are the tens and units digits respectively.
def number (a b : ℕ) := 10 * a + b
def interchangedNumber (a b : ℕ) := 10 * b + a
def digitsDifference (a b : ℕ) := a - b

-- Given conditions
variable (a b k : ℕ)

def condition1 := number a b = k * digitsDifference a b
def condition2 (x : ℕ) := interchangedNumber a b = x * digitsDifference a b

-- Theorem to prove
theorem find_x (h1 : condition1 a b k) : ∃ x, condition2 a b x ∧ x = k - 9 := 
by sorry

end NUMINAMATH_GPT_find_x_l2269_226992


namespace NUMINAMATH_GPT_gabby_additional_money_needed_l2269_226912

theorem gabby_additional_money_needed
  (cost_makeup : ℕ := 65)
  (cost_skincare : ℕ := 45)
  (cost_hair_tool : ℕ := 55)
  (initial_savings : ℕ := 35)
  (money_from_mom : ℕ := 20)
  (money_from_dad : ℕ := 30)
  (money_from_chores : ℕ := 25) :
  (cost_makeup + cost_skincare + cost_hair_tool) - (initial_savings + money_from_mom + money_from_dad + money_from_chores) = 55 := 
by
  sorry

end NUMINAMATH_GPT_gabby_additional_money_needed_l2269_226912


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l2269_226969

theorem arithmetic_sequence_tenth_term :
  ∀ (a : ℚ) (a_20 : ℚ) (a_10 : ℚ),
    a = 5 / 11 →
    a_20 = 9 / 11 →
    a_10 = a + (9 * ((a_20 - a) / 19)) →
    a_10 = 1233 / 2309 :=
by
  intros a a_20 a_10 h_a h_a_20 h_a_10
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l2269_226969


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l2269_226970

theorem rowing_speed_in_still_water (speed_of_current : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) (S : ℝ)
  (h_current : speed_of_current = 3) 
  (h_time : time_seconds = 9.390553103577801) 
  (h_distance : distance_meters = 60) 
  (h_S : S = 20) : 
  (distance_meters / 1000) / (time_seconds / 3600) - speed_of_current = S :=
by 
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l2269_226970


namespace NUMINAMATH_GPT_find_b_plus_c_l2269_226978

-- Definitions based on the given conditions.
variables {A : ℝ} {a b c : ℝ}

-- The conditions in the problem
theorem find_b_plus_c
  (h_cosA : Real.cos A = 1 / 3)
  (h_a : a = Real.sqrt 3)
  (h_bc : b * c = 3 / 2) :
  b + c = Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_b_plus_c_l2269_226978


namespace NUMINAMATH_GPT_baker_extra_cakes_l2269_226994

-- Defining the conditions
def original_cakes : ℕ := 78
def total_cakes : ℕ := 87
def extra_cakes := total_cakes - original_cakes

-- The statement to prove
theorem baker_extra_cakes : extra_cakes = 9 := by
  sorry

end NUMINAMATH_GPT_baker_extra_cakes_l2269_226994


namespace NUMINAMATH_GPT_actual_distance_traveled_l2269_226951

theorem actual_distance_traveled (D : ℝ) (h : D / 10 = (D + 20) / 20) : D = 20 :=
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l2269_226951


namespace NUMINAMATH_GPT_total_cuts_length_eq_60_l2269_226936

noncomputable def total_length_of_cuts (side_length : ℝ) (num_rectangles : ℕ) : ℝ :=
  if side_length = 36 ∧ num_rectangles = 3 then 60 else 0

theorem total_cuts_length_eq_60 :
  ∀ (side_length : ℝ) (num_rectangles : ℕ),
    side_length = 36 ∧ num_rectangles = 3 →
    total_length_of_cuts side_length num_rectangles = 60 := by
  intros
  simp [total_length_of_cuts]
  sorry

end NUMINAMATH_GPT_total_cuts_length_eq_60_l2269_226936


namespace NUMINAMATH_GPT_distance_between_cities_l2269_226985

-- Definitions
def map_distance : ℝ := 120 -- Distance on the map in cm
def scale_factor : ℝ := 10  -- Scale factor in km per cm

-- Theorem statement
theorem distance_between_cities :
  map_distance * scale_factor = 1200 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l2269_226985


namespace NUMINAMATH_GPT_commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l2269_226938

def star (x y : ℕ) : ℕ := (x + 2) * (y + 2) - 2

theorem commutative_star : ∀ x y : ℕ, star x y = star y x := by
  sorry

theorem not_distributive_star : ∃ x y z : ℕ, star x (y + z) ≠ star x y + star x z := by
  sorry

theorem special_case_star_false : ∀ x : ℕ, star (x - 2) (x + 2) ≠ star x x - 2 := by
  sorry

theorem no_identity_star : ¬∃ e : ℕ, ∀ x : ℕ, star x e = x ∧ star e x = x := by
  sorry

-- Associativity requires further verification and does not have a definitive statement yet.

end NUMINAMATH_GPT_commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l2269_226938


namespace NUMINAMATH_GPT_log_identity_l2269_226913

theorem log_identity : (Real.log 2)^3 + 3 * (Real.log 2) * (Real.log 5) + (Real.log 5)^3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_identity_l2269_226913


namespace NUMINAMATH_GPT_customers_left_correct_l2269_226902

-- Define the initial conditions
def initial_customers : ℕ := 8
def remaining_customers : ℕ := 5

-- Define the statement regarding customers left
def customers_left : ℕ := initial_customers - remaining_customers

-- The theorem we need to prove
theorem customers_left_correct : customers_left = 3 := by
    -- Skipping the actual proof
    sorry

end NUMINAMATH_GPT_customers_left_correct_l2269_226902


namespace NUMINAMATH_GPT_binar_operation_correct_l2269_226974

theorem binar_operation_correct : 
  let a := 13  -- 1101_2 in decimal
  let b := 15  -- 1111_2 in decimal
  let c := 9   -- 1001_2 in decimal
  let d := 2   -- 10_2 in decimal
  a + b - c * d = 10 ↔ "1010" = "1010" := 
by 
  intros
  simp
  sorry

end NUMINAMATH_GPT_binar_operation_correct_l2269_226974


namespace NUMINAMATH_GPT_expression_simplification_l2269_226911

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l2269_226911


namespace NUMINAMATH_GPT_det_dilation_matrix_l2269_226914

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![12, 0], ![0, 12]]

theorem det_dilation_matrix : Matrix.det E = 144 := by
  sorry

end NUMINAMATH_GPT_det_dilation_matrix_l2269_226914


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_a6_l2269_226984

theorem arithmetic_sequence_a1_a6
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) : a 1 * a 6 = 14 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_a6_l2269_226984


namespace NUMINAMATH_GPT_category_B_count_solution_hiring_probability_l2269_226999

-- Definitions and conditions
def category_A_count : Nat := 12

def total_selected_housekeepers : Nat := 20
def category_B_selected_housekeepers : Nat := 16
def category_A_selected_housekeepers := total_selected_housekeepers - category_B_selected_housekeepers

-- The value of x
def category_B_count (x : Nat) : Prop :=
  (category_A_selected_housekeepers * x) / category_A_count = category_B_selected_housekeepers

-- Assertion for the value of x
theorem category_B_count_solution : category_B_count 48 :=
by sorry

-- Conditions for the second part of the problem
def remaining_category_A : Nat := 3
def remaining_category_B : Nat := 2
def total_remaining := remaining_category_A + remaining_category_B

def possible_choices := remaining_category_A * (remaining_category_A - 1) / 2 + remaining_category_A * remaining_category_B + remaining_category_B * (remaining_category_B - 1) / 2
def successful_choices := remaining_category_A * remaining_category_B

def probability (a b : Nat) := (successful_choices % total_remaining) / (possible_choices % total_remaining)

-- Assertion for the probability
theorem hiring_probability : probability remaining_category_A remaining_category_B = 3 / 5 :=
by sorry

end NUMINAMATH_GPT_category_B_count_solution_hiring_probability_l2269_226999


namespace NUMINAMATH_GPT_conic_section_is_hyperbola_l2269_226907

theorem conic_section_is_hyperbola (x y : ℝ) :
  (x - 3)^2 = (3 * y + 4)^2 - 75 → 
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 :=
sorry

end NUMINAMATH_GPT_conic_section_is_hyperbola_l2269_226907


namespace NUMINAMATH_GPT_prime_cubed_plus_prime_plus_one_not_square_l2269_226982

theorem prime_cubed_plus_prime_plus_one_not_square (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ k : ℕ, k * k = p^3 + p + 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_cubed_plus_prime_plus_one_not_square_l2269_226982


namespace NUMINAMATH_GPT_rock_paper_scissors_l2269_226947

open Nat

-- Definitions based on problem conditions
def personA_movement (x y z : ℕ) : ℤ :=
  3 * (x : ℤ) - 2 * (y : ℤ) + (z : ℤ)

def personB_movement (x y z : ℕ) : ℤ :=
  3 * (y : ℤ) - 2 * (x : ℤ) + (z : ℤ)

def total_rounds (x y z : ℕ) : ℕ :=
  x + y + z

-- Problem statement
theorem rock_paper_scissors (x y z : ℕ) 
  (h1 : total_rounds x y z = 15)
  (h2 : personA_movement x y z = 17)
  (h3 : personB_movement x y z = 2) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_rock_paper_scissors_l2269_226947


namespace NUMINAMATH_GPT_maximum_discount_rate_l2269_226991

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_maximum_discount_rate_l2269_226991


namespace NUMINAMATH_GPT_num_five_digit_integers_l2269_226990

theorem num_five_digit_integers
  (total_digits : ℕ := 8)
  (repeat_3 : ℕ := 2)
  (repeat_6 : ℕ := 3)
  (repeat_8 : ℕ := 2)
  (arrangements : ℕ := Nat.factorial total_digits / (Nat.factorial repeat_3 * Nat.factorial repeat_6 * Nat.factorial repeat_8)) :
  arrangements = 1680 := by
  sorry

end NUMINAMATH_GPT_num_five_digit_integers_l2269_226990


namespace NUMINAMATH_GPT_boys_without_calculators_l2269_226957

theorem boys_without_calculators 
  (total_students : ℕ)
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (H_total_students : total_students = 30)
  (H_total_boys : total_boys = 20)
  (H_students_with_calculators : students_with_calculators = 25)
  (H_girls_with_calculators : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 13 :=
by
  sorry

end NUMINAMATH_GPT_boys_without_calculators_l2269_226957


namespace NUMINAMATH_GPT_problem1_problem2_l2269_226921

variable (x y a b c d : ℝ)
variable (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)

-- Problem 1: Prove (x + y) * (x^2 - x * y + y^2) = x^3 + y^3
theorem problem1 : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

-- Problem 2: Prove ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6)
theorem problem2 (a b c d : ℝ) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : 
  ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6) := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2269_226921


namespace NUMINAMATH_GPT_net_change_correct_l2269_226915
-- Import the necessary library

-- Price calculation function
def price_after_changes (initial_price: ℝ) (changes: List (ℝ -> ℝ)): ℝ :=
  changes.foldl (fun price change => change price) initial_price

-- Define each model's price changes
def modelA_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.9, 
  fun price => price * 1.3, 
  fun price => price * 0.85
]

def modelB_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.85, 
  fun price => price * 1.25, 
  fun price => price * 0.80
]

def modelC_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.80, 
  fun price => price * 1.20, 
  fun price => price * 0.95
]

-- Calculate final prices
def final_price_modelA := price_after_changes 1000 modelA_changes
def final_price_modelB := price_after_changes 1500 modelB_changes
def final_price_modelC := price_after_changes 2000 modelC_changes

-- Calculate net changes
def net_change_modelA := final_price_modelA - 1000
def net_change_modelB := final_price_modelB - 1500
def net_change_modelC := final_price_modelC - 2000

-- Set up theorem
theorem net_change_correct:
  net_change_modelA = -5.5 ∧ net_change_modelB = -225 ∧ net_change_modelC = -176 := by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_net_change_correct_l2269_226915


namespace NUMINAMATH_GPT_positive_difference_arithmetic_sequence_l2269_226903

theorem positive_difference_arithmetic_sequence :
  let a := 3
  let d := 5
  let a₁₀₀ := a + (100 - 1) * d
  let a₁₁₀ := a + (110 - 1) * d
  a₁₁₀ - a₁₀₀ = 50 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_arithmetic_sequence_l2269_226903


namespace NUMINAMATH_GPT_rectangle_area_l2269_226960

variable (l w : ℕ)

def length_is_three_times_width := l = 3 * w

def perimeter_is_160 := 2 * l + 2 * w = 160

theorem rectangle_area : 
  length_is_three_times_width l w → 
  perimeter_is_160 l w → 
  l * w = 1200 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_rectangle_area_l2269_226960


namespace NUMINAMATH_GPT_derivative_of_f_eval_deriv_at_pi_over_6_l2269_226916

noncomputable def f (x : Real) : Real := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem derivative_of_f : ∀ x, deriv f x = -Real.sin (4 * x) :=
by
  intro x
  sorry

theorem eval_deriv_at_pi_over_6 : deriv f (Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  rw [derivative_of_f]
  sorry

end NUMINAMATH_GPT_derivative_of_f_eval_deriv_at_pi_over_6_l2269_226916


namespace NUMINAMATH_GPT_constant_value_l2269_226949

theorem constant_value (x y z C : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : x > y) (h4 : y > z) (h5 : z = 2) (h6 : 2 * x + 3 * y + 3 * z = 5 * y + C) : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_constant_value_l2269_226949


namespace NUMINAMATH_GPT_a3_value_l2269_226946

variable {a : ℕ → ℤ} -- Arithmetic sequence as a function from natural numbers to integers
variable {S : ℕ → ℤ} -- Sum of the first n terms

-- Conditions
axiom a1_eq : a 1 = -11
axiom a4_plus_a6_eq : a 4 + a 6 = -6
-- Common difference d
variable {d : ℤ}
axiom d_def : ∀ n, a (n + 1) = a n + d

theorem a3_value : a 3 = -7 := by
  sorry -- Proof not required as per the instructions

end NUMINAMATH_GPT_a3_value_l2269_226946


namespace NUMINAMATH_GPT_find_x_l2269_226986

theorem find_x (a b x : ℝ) (h1 : ∀ a b, a * b = 2 * a - b) (h2 : 2 * (6 * x) = 2) : x = 10 := 
sorry

end NUMINAMATH_GPT_find_x_l2269_226986


namespace NUMINAMATH_GPT_positive_integer_solutions_count_l2269_226989

theorem positive_integer_solutions_count :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 24 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 64) ∧ s.card = 4 := 
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_count_l2269_226989


namespace NUMINAMATH_GPT_total_surface_area_l2269_226937

noncomputable def calculate_surface_area
  (radius : ℝ) (reflective : Bool) : ℝ :=
  let base_area := (radius^2 * Real.pi)
  let curved_surface_area := (4 * Real.pi * (radius^2)) / 2
  let effective_surface_area := if reflective then 2 * curved_surface_area else curved_surface_area
  effective_surface_area

theorem total_surface_area (r : ℝ) (h₁_reflective : Bool) (h₂_reflective : Bool) :
  r = 8 →
  h₁_reflective = false →
  h₂_reflective = true →
  (calculate_surface_area r h₁_reflective + calculate_surface_area r h₂_reflective) = 384 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_total_surface_area_l2269_226937


namespace NUMINAMATH_GPT_percentage_decrease_l2269_226940

theorem percentage_decrease (x y : ℝ) :
  let x' := 0.8 * x
  let y' := 0.7 * y
  let original_expr := x^2 * y^3
  let new_expr := (x')^2 * (y')^3
  let perc_decrease := (original_expr - new_expr) / original_expr * 100
  perc_decrease = 78.048 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l2269_226940


namespace NUMINAMATH_GPT_value_of_seventh_observation_l2269_226962

-- Given conditions
def sum_of_first_six_observations : ℕ := 90
def new_total_sum : ℕ := 98

-- Problem: prove the value of the seventh observation
theorem value_of_seventh_observation : new_total_sum - sum_of_first_six_observations = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_seventh_observation_l2269_226962


namespace NUMINAMATH_GPT_fritz_has_40_dollars_l2269_226980

variable (F S R : ℝ)
variable (h1 : S = (1 / 2) * F + 4)
variable (h2 : R = 3 * S)
variable (h3 : R + S = 96)

theorem fritz_has_40_dollars : F = 40 :=
by
  sorry

end NUMINAMATH_GPT_fritz_has_40_dollars_l2269_226980


namespace NUMINAMATH_GPT_number_of_questions_in_exam_l2269_226966

theorem number_of_questions_in_exam :
  ∀ (typeA : ℕ) (typeB : ℕ) (timeA : ℝ) (timeB : ℝ) (totalTime : ℝ),
    typeA = 100 →
    timeA = 1.2 →
    timeB = 0.6 →
    totalTime = 180 →
    120 = typeA * timeA →
    totalTime - 120 = typeB * timeB →
    typeA + typeB = 200 :=
by
  intros typeA typeB timeA timeB totalTime h_typeA h_timeA h_timeB h_totalTime h_timeA_calc h_remaining_time
  sorry

end NUMINAMATH_GPT_number_of_questions_in_exam_l2269_226966


namespace NUMINAMATH_GPT_john_new_weekly_earnings_l2269_226950

theorem john_new_weekly_earnings :
  ∀ (original_earnings : ℤ) (percentage_increase : ℝ),
  original_earnings = 60 →
  percentage_increase = 66.67 →
  (original_earnings + (percentage_increase / 100 * original_earnings)) = 100 := 
by
  intros original_earnings percentage_increase h_earnings h_percentage
  rw [h_earnings, h_percentage]
  norm_num
  sorry

end NUMINAMATH_GPT_john_new_weekly_earnings_l2269_226950
