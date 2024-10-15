import Mathlib

namespace NUMINAMATH_GPT_tan_pi_div_4_add_alpha_l508_50863

theorem tan_pi_div_4_add_alpha (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.tan (π / 4 + α) = -3 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_div_4_add_alpha_l508_50863


namespace NUMINAMATH_GPT_trig_identity_simplification_l508_50869

theorem trig_identity_simplification (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end NUMINAMATH_GPT_trig_identity_simplification_l508_50869


namespace NUMINAMATH_GPT_parabola_line_intersection_l508_50840

theorem parabola_line_intersection :
  let a := (3 + Real.sqrt 11) / 2
  let b := (3 - Real.sqrt 11) / 2
  let p1 := (a, (9 + Real.sqrt 11) / 2)
  let p2 := (b, (9 - Real.sqrt 11) / 2)
  (3 * a^2 - 9 * a + 4 = (9 + Real.sqrt 11) / 2) ∧
  (-a^2 + 3 * a + 6 = (9 + Real.sqrt 11) / 2) ∧
  ((9 + Real.sqrt 11) / 2 = a + 3) ∧
  (3 * b^2 - 9 * b + 4 = (9 - Real.sqrt 11) / 2) ∧
  (-b^2 + 3 * b + 6 = (9 - Real.sqrt 11) / 2) ∧
  ((9 - Real.sqrt 11) / 2 = b + 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_l508_50840


namespace NUMINAMATH_GPT_find_x_l508_50855

theorem find_x (x y : ℤ) (h₁ : x + 3 * y = 10) (h₂ : y = 3) : x = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l508_50855


namespace NUMINAMATH_GPT_weight_problem_l508_50807

variable (M T : ℕ)

theorem weight_problem
  (h1 : 220 = 3 * M + 10)
  (h2 : T = 2 * M)
  (h3 : 2 * T = 220) :
  M = 70 ∧ T = 140 :=
by
  sorry

end NUMINAMATH_GPT_weight_problem_l508_50807


namespace NUMINAMATH_GPT_geometric_sequence_tenth_fifth_terms_l508_50850

variable (a r : ℚ) (n : ℕ)

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem geometric_sequence_tenth_fifth_terms :
  (geometric_sequence 4 (4/3) 10 = 1048576 / 19683) ∧ (geometric_sequence 4 (4/3) 5 = 1024 / 81) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_tenth_fifth_terms_l508_50850


namespace NUMINAMATH_GPT_exists_x_eq_1_l508_50842

theorem exists_x_eq_1 (x y z t : ℕ) (h : x + y + z + t = 10) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  ∃ x, x = 1 :=
sorry

end NUMINAMATH_GPT_exists_x_eq_1_l508_50842


namespace NUMINAMATH_GPT_no_solutions_ordered_triples_l508_50832

theorem no_solutions_ordered_triples :
  ¬ ∃ (x y z : ℤ), 
    x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
    -x^2 + 5 * y * z + 3 * z^2 = 55 ∧
    x^2 + 2 * x * y + 9 * z^2 = 150 :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_ordered_triples_l508_50832


namespace NUMINAMATH_GPT_quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l508_50835

theorem quadratic_has_negative_root_sufficiency 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (a < 0) :=
sorry

theorem quadratic_has_negative_root_necessity 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0) :=
sorry

end NUMINAMATH_GPT_quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l508_50835


namespace NUMINAMATH_GPT_geometric_sequence_sum_l508_50866

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 + a 3 = 20)
  (h2 : a 2 + a 4 = 40)
  :
  a 3 + a 5 = 80 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l508_50866


namespace NUMINAMATH_GPT_ticket_cost_l508_50859

theorem ticket_cost 
  (V G : ℕ)
  (h1 : V + G = 320)
  (h2 : V = G - 212) :
  40 * V + 15 * G = 6150 := 
by
  sorry

end NUMINAMATH_GPT_ticket_cost_l508_50859


namespace NUMINAMATH_GPT_rectangle_same_color_l508_50800

/-- In a 3 × 7 grid where each square is either black or white, 
  there exists a rectangle whose four corners are of the same color. -/
theorem rectangle_same_color (grid : Fin 3 × Fin 7 → Bool) :
  ∃ (r1 r2 : Fin 3) (c1 c2 : Fin 7), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid (r1, c1) = grid (r1, c2) ∧ grid (r2, c1) = grid (r2, c2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_same_color_l508_50800


namespace NUMINAMATH_GPT_derivative_at_2_l508_50846

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_at_2 : deriv f 2 = (1 - Real.log 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l508_50846


namespace NUMINAMATH_GPT_remainder_div_3005_95_l508_50868

theorem remainder_div_3005_95 : 3005 % 95 = 60 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_div_3005_95_l508_50868


namespace NUMINAMATH_GPT_number_divisible_by_k_cube_l508_50822

theorem number_divisible_by_k_cube (k : ℕ) (h : k = 42) : ∃ n, (k^3) % n = 0 ∧ n = 74088 := by
  sorry

end NUMINAMATH_GPT_number_divisible_by_k_cube_l508_50822


namespace NUMINAMATH_GPT_go_stones_perimeter_l508_50836

-- Define the conditions for the problem
def stones_wide : ℕ := 4
def stones_tall : ℕ := 8

-- Define what we want to prove based on the conditions
theorem go_stones_perimeter : 2 * stones_wide + 2 * stones_tall - 4 = 20 :=
by
  -- Proof would normally go here
  sorry

end NUMINAMATH_GPT_go_stones_perimeter_l508_50836


namespace NUMINAMATH_GPT_value_of_k_l508_50853

theorem value_of_k {k : ℝ} :
  (∀ x : ℝ, (x^2 + k * x + 24 > 0) ↔ (x < -6 ∨ x > 4)) →
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l508_50853


namespace NUMINAMATH_GPT_arrangement_correct_l508_50828

def A := 4
def B := 1
def C := 2
def D := 5
def E := 6
def F := 3

def sum1 := A + B + C
def sum2 := A + D + F
def sum3 := B + E + D
def sum4 := C + F + E
def sum5 := A + E + F
def sum6 := B + D + C
def sum7 := B + C + F

theorem arrangement_correct :
  sum1 = 15 ∧ sum2 = 15 ∧ sum3 = 15 ∧ sum4 = 15 ∧ sum5 = 15 ∧ sum6 = 15 ∧ sum7 = 15 := 
by
  unfold sum1 sum2 sum3 sum4 sum5 sum6 sum7 
  unfold A B C D E F
  sorry

end NUMINAMATH_GPT_arrangement_correct_l508_50828


namespace NUMINAMATH_GPT_percent_of_day_is_hours_l508_50874

theorem percent_of_day_is_hours (h : ℝ) (day_hours : ℝ) (percent : ℝ) 
  (day_hours_def : day_hours = 24)
  (percent_def : percent = 29.166666666666668) :
  h = 7 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_day_is_hours_l508_50874


namespace NUMINAMATH_GPT_verify_quadratic_eq_l508_50852

def is_quadratic (eq : String) : Prop :=
  eq = "ax^2 + bx + c = 0"

theorem verify_quadratic_eq :
  is_quadratic "x^2 - 1 = 0" :=
by
  -- Auxiliary functions or steps can be introduced if necessary, but proof is omitted here.
  sorry

end NUMINAMATH_GPT_verify_quadratic_eq_l508_50852


namespace NUMINAMATH_GPT_ellipse_focus_value_l508_50808

theorem ellipse_focus_value (k : ℝ) (hk : 5 * (0:ℝ)^2 - k * (2:ℝ)^2 = 5) : k = -1 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focus_value_l508_50808


namespace NUMINAMATH_GPT_length_of_second_train_l508_50811

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (clear_time_seconds : ℝ)
  (relative_speed_kmph : ℝ) :
  speed_first_train_kmph + speed_second_train_kmph = relative_speed_kmph →
  relative_speed_kmph * (5 / 18) * clear_time_seconds = length_first_train + 280 :=
by
  let length_first_train := 120
  let speed_first_train_kmph := 42
  let speed_second_train_kmph := 30
  let clear_time_seconds := 20
  let relative_speed_kmph := 72
  sorry

end NUMINAMATH_GPT_length_of_second_train_l508_50811


namespace NUMINAMATH_GPT_find_abs_3h_minus_4k_l508_50871

theorem find_abs_3h_minus_4k
  (h k : ℤ)
  (factor1_eq_zero : 3 * (-3)^3 - h * (-3) - 3 * k = 0)
  (factor2_eq_zero : 3 * 2^3 - h * 2 - 3 * k = 0) :
  |3 * h - 4 * k| = 615 :=
by
  sorry

end NUMINAMATH_GPT_find_abs_3h_minus_4k_l508_50871


namespace NUMINAMATH_GPT_right_triangle_third_side_product_l508_50838

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_product_l508_50838


namespace NUMINAMATH_GPT_individual_weight_l508_50892

def total_students : ℕ := 1500
def sampled_students : ℕ := 100

def individual := "the weight of each student"

theorem individual_weight :
  (total_students = 1500) →
  (sampled_students = 100) →
  individual = "the weight of each student" :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_individual_weight_l508_50892


namespace NUMINAMATH_GPT_people_present_l508_50888

-- Number of parents, pupils, teachers, staff members, and volunteers
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_teachers : ℕ := 35
def num_staff_members : ℕ := 20
def num_volunteers : ℕ := 50

-- The total number of people present in the program
def total_people : ℕ := num_parents + num_pupils + num_teachers + num_staff_members + num_volunteers

-- Proof statement
theorem people_present : total_people = 908 := by
  -- Proof goes here, but adding sorry for now
  sorry

end NUMINAMATH_GPT_people_present_l508_50888


namespace NUMINAMATH_GPT_sale_in_second_month_l508_50895

def sale_first_month : ℕ := 6435
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6191
def average_sale : ℕ := 6700

theorem sale_in_second_month : 
  ∀ (sale_second_month : ℕ), 
    (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month = 6700 * 6) → 
    sale_second_month = 6927 :=
by
  intro sale_second_month h
  sorry

end NUMINAMATH_GPT_sale_in_second_month_l508_50895


namespace NUMINAMATH_GPT_checkered_rectangles_containing_one_gray_cell_l508_50896

theorem checkered_rectangles_containing_one_gray_cell 
  (num_gray_cells : ℕ) 
  (num_blue_cells : ℕ) 
  (num_red_cells : ℕ)
  (blue_containing_rectangles : ℕ) 
  (red_containing_rectangles : ℕ) :
  num_gray_cells = 40 →
  num_blue_cells = 36 →
  num_red_cells = 4 →
  blue_containing_rectangles = 4 →
  red_containing_rectangles = 8 →
  num_blue_cells * blue_containing_rectangles + num_red_cells * red_containing_rectangles = 176 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_checkered_rectangles_containing_one_gray_cell_l508_50896


namespace NUMINAMATH_GPT_color_cartridge_cost_l508_50847

theorem color_cartridge_cost :
  ∃ C : ℝ, 
  (1 * 27) + (3 * C) = 123 ∧ C = 32 :=
by
  sorry

end NUMINAMATH_GPT_color_cartridge_cost_l508_50847


namespace NUMINAMATH_GPT_rational_coefficients_count_l508_50834

theorem rational_coefficients_count : 
  ∃ n, n = 84 ∧ ∀ k, (0 ≤ k ∧ k ≤ 500) → 
            (k % 3 = 0 ∧ (500 - k) % 2 = 0) → 
            n = 84 :=
by
  sorry

end NUMINAMATH_GPT_rational_coefficients_count_l508_50834


namespace NUMINAMATH_GPT_cookie_weight_l508_50899

theorem cookie_weight :
  ∀ (pounds_per_box cookies_per_box ounces_per_pound : ℝ),
    pounds_per_box = 40 →
    cookies_per_box = 320 →
    ounces_per_pound = 16 →
    (pounds_per_box * ounces_per_pound) / cookies_per_box = 2 := 
by 
  intros pounds_per_box cookies_per_box ounces_per_pound hpounds hcookies hounces
  rw [hpounds, hcookies, hounces]
  norm_num

end NUMINAMATH_GPT_cookie_weight_l508_50899


namespace NUMINAMATH_GPT_smaller_cone_volume_ratio_l508_50867

theorem smaller_cone_volume_ratio :
  let r := 12
  let theta1 := 120
  let theta2 := 240
  let arc_length_small := (theta1 / 360) * (2 * Real.pi * r)
  let arc_length_large := (theta2 / 360) * (2 * Real.pi * r)
  let r1 := arc_length_small / (2 * Real.pi)
  let r2 := arc_length_large / (2 * Real.pi)
  let l := r
  let h1 := Real.sqrt (l^2 - r1^2)
  let h2 := Real.sqrt (l^2 - r2^2)
  let V1 := (1 / 3) * Real.pi * r1^2 * h1
  let V2 := (1 / 3) * Real.pi * r2^2 * h2
  V1 / V2 = Real.sqrt 10 / 10 := sorry

end NUMINAMATH_GPT_smaller_cone_volume_ratio_l508_50867


namespace NUMINAMATH_GPT_find_n_l508_50815

open Nat

theorem find_n (n : ℕ) (d : ℕ → ℕ) (h1 : d 1 = 1) (hk : d 6^2 + d 7^2 - 1 = n) :
  n = 1984 ∨ n = 144 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l508_50815


namespace NUMINAMATH_GPT_find_a_plus_b_l508_50881

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hx : x = a + Real.sqrt b)
  (hxeq : x^2 + 5*x + 5/x + 1/(x^2) = 42) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l508_50881


namespace NUMINAMATH_GPT_largest_consecutive_odd_number_sum_is_27_l508_50819

theorem largest_consecutive_odd_number_sum_is_27
  (a b c : ℤ)
  (h1 : a + b + c = 75)
  (h2 : c - a = 4)
  (h3 : a % 2 = 1)
  (h4 : b % 2 = 1)
  (h5 : c % 2 = 1) :
  c = 27 := 
sorry

end NUMINAMATH_GPT_largest_consecutive_odd_number_sum_is_27_l508_50819


namespace NUMINAMATH_GPT_series_inequality_l508_50823

open BigOperators

theorem series_inequality :
  (∑ k in Finset.range 2012, (1 / (((k + 1) * Real.sqrt k) + (k * Real.sqrt (k + 1))))) > 0.97 :=
sorry

end NUMINAMATH_GPT_series_inequality_l508_50823


namespace NUMINAMATH_GPT_quadratic_roots_equal_integral_l508_50898

theorem quadratic_roots_equal_integral (c : ℝ) (h : (6^2 - 4 * 3 * c) = 0) : 
  ∃ x : ℝ, (3 * x^2 - 6 * x + c = 0) ∧ (x = 1) := 
by sorry

end NUMINAMATH_GPT_quadratic_roots_equal_integral_l508_50898


namespace NUMINAMATH_GPT_rose_can_afford_l508_50864

noncomputable def total_cost_before_discount : ℝ :=
  2.40 + 9.20 + 6.50 + 12.25 + 4.75

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def budget : ℝ :=
  30.00

noncomputable def remaining_budget : ℝ :=
  budget - total_cost_after_discount

theorem rose_can_afford :
  remaining_budget = 0.165 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_rose_can_afford_l508_50864


namespace NUMINAMATH_GPT_greatest_x_value_l508_50875

noncomputable def greatest_possible_value (x : ℕ) : ℕ :=
  if (x % 5 = 0) ∧ (x^3 < 3375) then x else 0

theorem greatest_x_value :
  ∃ x, greatest_possible_value x = 10 ∧ (∀ y, ((y % 5 = 0) ∧ (y^3 < 3375)) → y ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_value_l508_50875


namespace NUMINAMATH_GPT_smallest_sum_of_three_l508_50897

open Finset

-- Define the set of numbers
def my_set : Finset ℤ := {10, 2, -4, 15, -7}

-- Statement of the problem: Prove the smallest sum of any three different numbers from the set is -9
theorem smallest_sum_of_three :
  ∃ (a b c : ℤ), a ∈ my_set ∧ b ∈ my_set ∧ c ∈ my_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_three_l508_50897


namespace NUMINAMATH_GPT_root_in_interval_l508_50812

def f (x : ℝ) : ℝ := x^3 + 5 * x^2 - 3 * x + 1

theorem root_in_interval : ∃ A B : ℤ, B = A + 1 ∧ (∃ ξ : ℝ, f ξ = 0 ∧ (A : ℝ) < ξ ∧ ξ < (B : ℝ)) ∧ A = -6 ∧ B = -5 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l508_50812


namespace NUMINAMATH_GPT_g_neg_2_eq_3_l508_50827

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem g_neg_2_eq_3 : g (-2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_g_neg_2_eq_3_l508_50827


namespace NUMINAMATH_GPT_domain_of_g_cauchy_schwarz_inequality_l508_50810

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Question 1: Prove the domain of g(x) = log(f(x) - 2) is {x | 0.5 < x < 2.5}
theorem domain_of_g : {x : ℝ | 0.5 < x ∧ x < 2.5} = {x : ℝ | 0.5 < x ∧ x < 2.5} :=
by
  sorry

-- Minimum value of f(x)
def m : ℝ := 1

-- Question 2: Prove a^2 + b^2 + c^2 ≥ 1/3 given a + b + c = m
theorem cauchy_schwarz_inequality (a b c : ℝ) (h : a + b + c = m) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_g_cauchy_schwarz_inequality_l508_50810


namespace NUMINAMATH_GPT_length_of_CD_l508_50830

theorem length_of_CD (x y: ℝ) (h1: 5 * x = 3 * y) (u v: ℝ) (h2: u = x + 3) (h3: v = y - 3) (h4: 7 * u = 4 * v) : x + y = 264 :=
by
  sorry

end NUMINAMATH_GPT_length_of_CD_l508_50830


namespace NUMINAMATH_GPT_proof_problem_l508_50854

-- Conditions
def op1 := (15 + 3) / (8 - 2) = 3
def op2 := (9 + 4) / (14 - 7)

-- Statement
theorem proof_problem : op1 → op2 = 13 / 7 :=
by 
  intro h
  unfold op2
  sorry

end NUMINAMATH_GPT_proof_problem_l508_50854


namespace NUMINAMATH_GPT_find_deepaks_age_l508_50858

variable (R D : ℕ)

theorem find_deepaks_age
  (h1 : R / D = 4 / 3)
  (h2 : R + 2 = 26) :
  D = 18 := by
  sorry

end NUMINAMATH_GPT_find_deepaks_age_l508_50858


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l508_50848

open Real

theorem basic_astrophysics_degrees :
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  basic_astrophysics_percentage / 100 * circle_degrees = 43.2 :=
by
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  exact sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l508_50848


namespace NUMINAMATH_GPT_find_t_l508_50804

variable {a b c r s t : ℝ}

-- Conditions from part a)
def first_polynomial_has_roots (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c)) : Prop :=
  ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = 0 → x = a ∨ x = b ∨ x = c

def second_polynomial_has_roots (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) : Prop :=
  ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = 0 → x = (a + b) ∨ x = (b + c) ∨ x = (c + a)

-- Translate problem (find t) with conditions
theorem find_t (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c))
    (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
    (sum_roots : a + b + c = -3) 
    (prod_roots : a * b * c = -11):
  t = 23 := 
sorry

end NUMINAMATH_GPT_find_t_l508_50804


namespace NUMINAMATH_GPT_find_b_in_quadratic_eqn_l508_50837

theorem find_b_in_quadratic_eqn :
  ∃ (b : ℝ), ∃ (p : ℝ), 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → 
  b = 8 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_in_quadratic_eqn_l508_50837


namespace NUMINAMATH_GPT_repeating_decimal_division_l508_50841

-- Define x and y as the repeating decimals.
noncomputable def x : ℚ := 84 / 99
noncomputable def y : ℚ := 21 / 99

-- Proof statement of the equivalence.
theorem repeating_decimal_division : (x / y) = 4 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_division_l508_50841


namespace NUMINAMATH_GPT_scientific_notation_of_8200000_l508_50833

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_8200000_l508_50833


namespace NUMINAMATH_GPT_compressor_station_distances_compressor_station_distances_when_a_is_30_l508_50886

theorem compressor_station_distances (a : ℝ) (h : 0 < a ∧ a < 60) :
  ∃ x y z : ℝ, x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

theorem compressor_station_distances_when_a_is_30 :
  ∃ x y z : ℝ, 
  (x + y = 3 * z) ∧ (z + y = x + 30) ∧ (x + z = 60) ∧ 
  (x = 35) ∧ (y = 40) ∧ (z = 25) :=
sorry

end NUMINAMATH_GPT_compressor_station_distances_compressor_station_distances_when_a_is_30_l508_50886


namespace NUMINAMATH_GPT_maximize_profit_l508_50856

noncomputable def I (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if h' : 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0 -- default case for Lean to satisfy definition

noncomputable def P (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * x * (x - 1) * Real.exp (x - 2) - 448 * x - 180
  else if h' : 2 < x ∧ x ≤ 50 then -10 * x - 9000 / x + 2870
  else 0 -- default case for Lean to satisfy definition

theorem maximize_profit :
  (∀ x : ℝ, 0 < x ∧ x ≤ 50 → P x ≤ 2270) ∧ P 30 = 2270 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l508_50856


namespace NUMINAMATH_GPT_find_value_of_y_l508_50878

theorem find_value_of_y (x y : ℕ) 
    (h1 : 2^x - 2^y = 3 * 2^12) 
    (h2 : x = 14) : 
    y = 13 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_y_l508_50878


namespace NUMINAMATH_GPT_sequence_value_l508_50820

noncomputable def f : ℝ → ℝ := sorry

theorem sequence_value :
  ∃ a : ℕ → ℝ, 
    (a 1 = f 1) ∧ 
    (∀ n : ℕ, f (a (n + 1)) = f (2 * a n + 1)) ∧ 
    (a 2017 = 2 ^ 2016 - 1) := sorry

end NUMINAMATH_GPT_sequence_value_l508_50820


namespace NUMINAMATH_GPT_interest_amount_eq_750_l508_50891

-- Definitions
def P : ℕ := 3000
def R : ℕ := 5
def T : ℕ := 5

-- Condition
def interest_less_than_sum := 2250

-- Simple interest formula
def simple_interest (P R T : ℕ) := (P * R * T) / 100

-- Theorem
theorem interest_amount_eq_750 : simple_interest P R T = P - interest_less_than_sum :=
by
  -- We assert that we need to prove the equality holds.
  sorry

end NUMINAMATH_GPT_interest_amount_eq_750_l508_50891


namespace NUMINAMATH_GPT_algebra_expression_opposite_l508_50845

theorem algebra_expression_opposite (a : ℚ) :
  3 * a + 1 = -(3 * (a - 1)) → a = 1 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_algebra_expression_opposite_l508_50845


namespace NUMINAMATH_GPT_interest_rate_second_part_l508_50861

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_second_part_l508_50861


namespace NUMINAMATH_GPT_sum_of_x_coordinates_l508_50802

def exists_common_point (x : ℕ) : Prop :=
  (3 * x + 5) % 9 = (7 * x + 3) % 9

theorem sum_of_x_coordinates :
  ∃ x : ℕ, exists_common_point x ∧ x % 9 = 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_l508_50802


namespace NUMINAMATH_GPT_infinite_pairs_m_n_l508_50894

theorem infinite_pairs_m_n :
  ∃ (f : ℕ → ℕ × ℕ), (∀ k, (f k).1 > 0 ∧ (f k).2 > 0 ∧ ((f k).1 ∣ (f k).2 ^ 2 + 1) ∧ ((f k).2 ∣ (f k).1 ^ 2 + 1)) :=
sorry

end NUMINAMATH_GPT_infinite_pairs_m_n_l508_50894


namespace NUMINAMATH_GPT_sides_of_triangle_l508_50826

variable (a b c : ℝ)

theorem sides_of_triangle (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
  sorry

end NUMINAMATH_GPT_sides_of_triangle_l508_50826


namespace NUMINAMATH_GPT_larger_number_l508_50821

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_number_l508_50821


namespace NUMINAMATH_GPT_number_of_pieces_of_bubble_gum_l508_50843

theorem number_of_pieces_of_bubble_gum (cost_per_piece total_cost : ℤ) (h1 : cost_per_piece = 18) (h2 : total_cost = 2448) :
  total_cost / cost_per_piece = 136 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_number_of_pieces_of_bubble_gum_l508_50843


namespace NUMINAMATH_GPT_minimum_m_value_l508_50824

theorem minimum_m_value (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 24 * m = n^4) : m = 54 := sorry

end NUMINAMATH_GPT_minimum_m_value_l508_50824


namespace NUMINAMATH_GPT_susan_walked_9_miles_l508_50873

theorem susan_walked_9_miles (E S : ℕ) (h1 : E + S = 15) (h2 : E = S - 3) : S = 9 :=
by
  sorry

end NUMINAMATH_GPT_susan_walked_9_miles_l508_50873


namespace NUMINAMATH_GPT_area_of_region_l508_50884

theorem area_of_region : ∃ A, (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y = 12 → A = 25 * Real.pi) :=
by
  -- Completing the square and identifying the circle
  -- We verify that the given equation represents a circle
  existsi (25 * Real.pi)
  intros x y h
  sorry

end NUMINAMATH_GPT_area_of_region_l508_50884


namespace NUMINAMATH_GPT_n_pow4_sub_n_pow2_divisible_by_12_l508_50813

theorem n_pow4_sub_n_pow2_divisible_by_12 (n : ℤ) (h : n > 1) : 12 ∣ (n^4 - n^2) :=
by sorry

end NUMINAMATH_GPT_n_pow4_sub_n_pow2_divisible_by_12_l508_50813


namespace NUMINAMATH_GPT_calc_a_squared_plus_b_squared_and_ab_l508_50849

theorem calc_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) :
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_calc_a_squared_plus_b_squared_and_ab_l508_50849


namespace NUMINAMATH_GPT_Finn_initial_goldfish_l508_50814

variable (x : ℕ)

-- Defining the conditions
def number_of_goldfish_initial (x : ℕ) : Prop :=
  ∃ y z : ℕ, y = 32 ∧ z = 57 ∧ x = y + z 

-- Theorem statement to prove Finn's initial number of goldfish
theorem Finn_initial_goldfish (x : ℕ) (h : number_of_goldfish_initial x) : x = 89 := by
  sorry

end NUMINAMATH_GPT_Finn_initial_goldfish_l508_50814


namespace NUMINAMATH_GPT_complex_z24_condition_l508_50801

open Complex

theorem complex_z24_condition (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * π / 180)) : 
  z^24 + z⁻¹^24 = -1 := sorry

end NUMINAMATH_GPT_complex_z24_condition_l508_50801


namespace NUMINAMATH_GPT_determine_b_when_lines_parallel_l508_50825

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_when_lines_parallel_l508_50825


namespace NUMINAMATH_GPT_bell_pepper_slices_l508_50890

theorem bell_pepper_slices :
  ∀ (num_peppers : ℕ) (slices_per_pepper : ℕ) (total_slices_pieces : ℕ) (half_slices : ℕ),
  num_peppers = 5 → slices_per_pepper = 20 → total_slices_pieces = 200 →
  half_slices = (num_peppers * slices_per_pepper) / 2 →
  (total_slices_pieces - (num_peppers * slices_per_pepper)) / half_slices = 2 :=
by
  intros num_peppers slices_per_pepper total_slices_pieces half_slices h1 h2 h3 h4
  -- skip the proof with sorry as instructed
  sorry

end NUMINAMATH_GPT_bell_pepper_slices_l508_50890


namespace NUMINAMATH_GPT_derek_bought_more_cars_l508_50844

-- Define conditions
variables (d₆ c₆ d₁₆ c₁₆ : ℕ)

-- Given conditions
def initial_conditions :=
  (d₆ = 90) ∧
  (d₆ = 3 * c₆) ∧
  (d₁₆ = 120) ∧
  (c₁₆ = 2 * d₁₆)

-- Prove the number of cars Derek bought in ten years
theorem derek_bought_more_cars (h : initial_conditions d₆ c₆ d₁₆ c₁₆) : c₁₆ - c₆ = 210 :=
by sorry

end NUMINAMATH_GPT_derek_bought_more_cars_l508_50844


namespace NUMINAMATH_GPT_distinct_int_divisible_by_12_l508_50877

variable {a b c d : ℤ}

theorem distinct_int_divisible_by_12 (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by
  sorry

end NUMINAMATH_GPT_distinct_int_divisible_by_12_l508_50877


namespace NUMINAMATH_GPT_tan_range_l508_50883

theorem tan_range :
  ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ Real.pi / 4 → -1 ≤ Real.tan x ∧ Real.tan x < 0 ∨ 0 < Real.tan x ∧ Real.tan x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_range_l508_50883


namespace NUMINAMATH_GPT_compare_2_roses_3_carnations_l508_50860

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end NUMINAMATH_GPT_compare_2_roses_3_carnations_l508_50860


namespace NUMINAMATH_GPT_purchase_price_is_60_l508_50879

variable (P S D : ℝ)
variable (GP : ℝ := 4)

theorem purchase_price_is_60
  (h1 : S = P + 0.25 * S)
  (h2 : D = 0.80 * S)
  (h3 : GP = D - P) :
  P = 60 :=
by
  sorry

end NUMINAMATH_GPT_purchase_price_is_60_l508_50879


namespace NUMINAMATH_GPT_inequality_holds_l508_50829

theorem inequality_holds (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  Real.sin (2 * x)^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l508_50829


namespace NUMINAMATH_GPT_length_of_platform_l508_50818

theorem length_of_platform (length_of_train : ℕ) (speed_kmph : ℕ) (time_s : ℕ) (L : ℕ) :
  length_of_train = 160 → speed_kmph = 72 → time_s = 25 → (L = 340) :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l508_50818


namespace NUMINAMATH_GPT_profit_percentage_on_cost_price_l508_50806

theorem profit_percentage_on_cost_price (CP MP SP : ℝ)
    (h1 : CP = 100)
    (h2 : MP = 131.58)
    (h3 : SP = 0.95 * MP) :
    ((SP - CP) / CP) * 100 = 25 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_profit_percentage_on_cost_price_l508_50806


namespace NUMINAMATH_GPT_goods_train_speed_l508_50882

theorem goods_train_speed (train_length platform_length : ℝ) (time_sec : ℝ) : 
  train_length = 270.0416 ∧ platform_length = 250 ∧ time_sec = 26 → 
  (train_length + platform_length) / time_sec * 3.6 = 72.00576 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l508_50882


namespace NUMINAMATH_GPT_problem_divisibility_l508_50851

theorem problem_divisibility 
  (a b c : ℕ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : b ∣ a^3)
  (h2 : c ∣ b^3)
  (h3 : a ∣ c^3) : 
  (a + b + c) ^ 13 ∣ a * b * c := 
sorry

end NUMINAMATH_GPT_problem_divisibility_l508_50851


namespace NUMINAMATH_GPT_nonneg_solution_iff_m_range_l508_50839

theorem nonneg_solution_iff_m_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 1) + 3 / (1 - x) = 1)) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_GPT_nonneg_solution_iff_m_range_l508_50839


namespace NUMINAMATH_GPT_find_f_at_one_l508_50817

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

theorem find_f_at_one (h_cond : f a b (-1) = 10) : f a b (1) = 14 := by
  sorry

end NUMINAMATH_GPT_find_f_at_one_l508_50817


namespace NUMINAMATH_GPT_woman_completion_days_l508_50893

variable (M W : ℚ)
variable (work_days_man work_days_total : ℚ)

-- Given conditions
def condition1 : Prop :=
  (10 * M + 15 * W) * 7 = 1

def condition2 : Prop :=
  M * 100 = 1

-- To prove
def one_woman_days : ℚ := 350

theorem woman_completion_days (h1 : condition1 M W) (h2 : condition2 M) :
  1 / W = one_woman_days :=
by
  sorry

end NUMINAMATH_GPT_woman_completion_days_l508_50893


namespace NUMINAMATH_GPT_problem1_problem2_l508_50816

-- Proof Problem 1: Prove that (x-y)^2 - (x+y)(x-y) = -2xy + 2y^2
theorem problem1 (x y : ℝ) : (x - y) ^ 2 - (x + y) * (x - y) = -2 * x * y + 2 * y ^ 2 := 
by
  sorry

-- Proof Problem 2: Prove that (12a^2b - 6ab^2) / (-3ab) = -4a + 2b
theorem problem2 (a b : ℝ) (h : -3 * a * b ≠ 0) : (12 * a^2 * b - 6 * a * b^2) / (-3 * a * b) = -4 * a + 2 * b := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l508_50816


namespace NUMINAMATH_GPT_find_number_of_violas_l508_50870

theorem find_number_of_violas (cellos : ℕ) (pairs : ℕ) (probability : ℚ) 
    (h1 : cellos = 800) 
    (h2 : pairs = 100) 
    (h3 : probability = 0.00020833333333333335) : 
    ∃ V : ℕ, V = 600 := 
by 
    sorry

end NUMINAMATH_GPT_find_number_of_violas_l508_50870


namespace NUMINAMATH_GPT_min_blocks_to_remove_l508_50862

theorem min_blocks_to_remove (n : ℕ) (h₁ : n = 59) : ∃ k, ∃ m, (m*m*m ≤ n ∧ n < (m+1)*(m+1)*(m+1)) ∧ k = n - m*m*m ∧ k = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_blocks_to_remove_l508_50862


namespace NUMINAMATH_GPT_fraction_comparison_l508_50805

theorem fraction_comparison :
  (2 : ℝ) * (4 : ℝ) > (7 : ℝ) → (4 / 7 : ℝ) > (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l508_50805


namespace NUMINAMATH_GPT_find_u_plus_v_l508_50885

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 7 * v = 17) (h2 : 5 * u + 3 * v = 1) : 
  u + v = - 6 / 11 :=
  sorry

end NUMINAMATH_GPT_find_u_plus_v_l508_50885


namespace NUMINAMATH_GPT_initial_shipment_robot_rascals_l508_50876

theorem initial_shipment_robot_rascals 
(T : ℝ) 
(h1 : (0.7 * T = 168)) : 
  T = 240 :=
sorry

end NUMINAMATH_GPT_initial_shipment_robot_rascals_l508_50876


namespace NUMINAMATH_GPT_find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l508_50803

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end NUMINAMATH_GPT_find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l508_50803


namespace NUMINAMATH_GPT_final_number_lt_one_l508_50831

theorem final_number_lt_one :
  ∀ (numbers : Finset ℕ),
    (numbers = Finset.range 3000 \ Finset.range 1000) →
    (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≤ b →
    ∃ (numbers' : Finset ℕ), numbers' = (numbers \ {a, b}) ∪ {a / 2}) →
    ∃ (x : ℕ), x ∈ numbers ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_final_number_lt_one_l508_50831


namespace NUMINAMATH_GPT_sufficient_and_necessary_l508_50872

theorem sufficient_and_necessary (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_GPT_sufficient_and_necessary_l508_50872


namespace NUMINAMATH_GPT_inequality_problem_l508_50865

theorem inequality_problem 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
by sorry

end NUMINAMATH_GPT_inequality_problem_l508_50865


namespace NUMINAMATH_GPT_exists_mutual_shooters_l508_50887

theorem exists_mutual_shooters (n : ℕ) (h : 0 ≤ n) (d : Fin (2 * n + 1) → Fin (2 * n + 1) → ℝ)
  (hdistinct : ∀ i j k l : Fin (2 * n + 1), i ≠ j → k ≠ l → d i j ≠ d k l)
  (hc : ∀ i : Fin (2 * n + 1), ∃ j : Fin (2 * n + 1), i ≠ j ∧ (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k)) :
  ∃ i j : Fin (2 * n + 1), i ≠ j ∧
  (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k) ∧
  (∀ k : Fin (2 * n + 1), k ≠ i → d j i < d j k) :=
by
  sorry

end NUMINAMATH_GPT_exists_mutual_shooters_l508_50887


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l508_50857

theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := 
by
  sorry

theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := 
by 
  sorry

theorem problem3 : ( (1/2) + (1/3) - (1/6) ) / (-1/18) = -12 := 
by 
  sorry

theorem problem4 : - 1 ^ 4 - (1 + 0.5) * (1/3) * (-4) ^ 2 = -33 / 32 := 
by 
  sorry


end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l508_50857


namespace NUMINAMATH_GPT_pauls_total_cost_is_252_l508_50809

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end NUMINAMATH_GPT_pauls_total_cost_is_252_l508_50809


namespace NUMINAMATH_GPT_largest_c_in_range_of_f_l508_50889

theorem largest_c_in_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 - 6 * x + c = 2) -> c ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_c_in_range_of_f_l508_50889


namespace NUMINAMATH_GPT_ratio_of_terms_l508_50880

theorem ratio_of_terms (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ) :
  (∀ n, S_n n = (n * (2 * a_n n - (n - 1))) / 2) → 
  (∀ n, T_n n = (n * (2 * b_n n - (n - 1))) / 2) → 
  (∀ n, S_n n / T_n n = (n + 3) / (2 * n + 1)) → 
  S_n 6 / T_n 6 = 14 / 23 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_terms_l508_50880
