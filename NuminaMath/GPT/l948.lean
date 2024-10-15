import Mathlib

namespace NUMINAMATH_GPT_interval_width_and_count_l948_94870

def average_income_intervals := [3000, 4000, 5000, 6000, 7000]
def frequencies := [5, 9, 4, 2]

theorem interval_width_and_count:
  (average_income_intervals[1] - average_income_intervals[0] = 1000) ∧
  (frequencies.length = 4) :=
by
  sorry

end NUMINAMATH_GPT_interval_width_and_count_l948_94870


namespace NUMINAMATH_GPT_height_in_meters_l948_94878

theorem height_in_meters (h: 1 * 100 + 36 = 136) : 1.36 = 1 + 36 / 100 :=
by 
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_height_in_meters_l948_94878


namespace NUMINAMATH_GPT_inequality_implies_l948_94888

theorem inequality_implies:
  ∀ (x y : ℝ), (x > y) → (2 * x - 1 > 2 * y - 1) :=
by
  intro x y hxy
  sorry

end NUMINAMATH_GPT_inequality_implies_l948_94888


namespace NUMINAMATH_GPT_problem1_problem2_l948_94863

theorem problem1 : 3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3)^0 + abs (Real.sqrt 3 - 2) = 3 := 
by
  sorry

theorem problem2 : (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / Real.sqrt 3 = 28 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l948_94863


namespace NUMINAMATH_GPT_find_age_of_b_l948_94892

variables (A B C : ℕ)

def average_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 28
def average_ac (A C : ℕ) : Prop := (A + C) / 2 = 29

theorem find_age_of_b (h1 : average_abc A B C) (h2 : average_ac A C) : B = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_age_of_b_l948_94892


namespace NUMINAMATH_GPT_P_at_3_l948_94865

noncomputable def P (x : ℝ) : ℝ := 1 * x^5 + 0 * x^4 + 0 * x^3 + 2 * x^2 + 1 * x + 4

theorem P_at_3 : P 3 = 268 := by
  sorry

end NUMINAMATH_GPT_P_at_3_l948_94865


namespace NUMINAMATH_GPT_inequality_proof_l948_94812

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) >= (2 / 3) * (a^2 + b^2 + c^2) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l948_94812


namespace NUMINAMATH_GPT_polygon_side_possibilities_l948_94866

theorem polygon_side_possibilities (n : ℕ) (h : (n-2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_polygon_side_possibilities_l948_94866


namespace NUMINAMATH_GPT_polar_equation_of_circle_c_range_of_op_oq_l948_94838

noncomputable def circle_param_eq (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_kl_eq (θ : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

theorem polar_equation_of_circle_c :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * Real.cos θ :=
by sorry

theorem range_of_op_oq (θ₁ : ℝ) (hθ : 0 < θ₁ ∧ θ₁ < Real.pi / 2) :
  0 < (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) ∧
  (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) < 6 :=
by sorry

end NUMINAMATH_GPT_polar_equation_of_circle_c_range_of_op_oq_l948_94838


namespace NUMINAMATH_GPT_remainder_m_n_mod_1000_l948_94897

noncomputable def m : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2009 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

noncomputable def n : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2000 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

theorem remainder_m_n_mod_1000 : (m - n) % 1000 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_m_n_mod_1000_l948_94897


namespace NUMINAMATH_GPT_sum_remainders_l948_94854

theorem sum_remainders (n : ℤ) (h : n % 20 = 14) : (n % 4) + (n % 5) = 6 :=
  by
  sorry

end NUMINAMATH_GPT_sum_remainders_l948_94854


namespace NUMINAMATH_GPT_find_table_height_l948_94883

theorem find_table_height (b r g h : ℝ) (h1 : h + b - g = 111) (h2 : h + r - b = 80) (h3 : h + g - r = 82) : h = 91 := 
by
  sorry

end NUMINAMATH_GPT_find_table_height_l948_94883


namespace NUMINAMATH_GPT_x_expression_l948_94835

noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

theorem x_expression {x y : ℝ} (hx : x ≠ 1) (hy : y = f x) : x = y / (1 + y) :=
by {
  sorry
}

end NUMINAMATH_GPT_x_expression_l948_94835


namespace NUMINAMATH_GPT_find_solutions_of_equation_l948_94880

theorem find_solutions_of_equation (m n : ℝ) 
  (h1 : ∀ x, (x - m)^2 + n = 0 ↔ (x = -1 ∨ x = 3)) :
  (∀ x, (x - 1)^2 + m^2 = 2 * m * (x - 1) - n ↔ (x = 0 ∨ x = 4)) :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_of_equation_l948_94880


namespace NUMINAMATH_GPT_imaginary_part_of_z_l948_94828

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 1 - 3 * I) : z.im = -2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l948_94828


namespace NUMINAMATH_GPT_cost_of_white_washing_l948_94847

-- Definitions for room dimensions, doors, windows, and cost per square foot
def length : ℕ := 25
def width : ℕ := 15
def height1 : ℕ := 12
def height2 : ℕ := 8
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def cost_per_sq_ft : ℕ := 10
def ceiling_decoration_area : ℕ := 10

-- Definitions for the areas calculation
def area_walls_height1 : ℕ := 2 * (length * height1)
def area_walls_height2 : ℕ := 2 * (width * height2)
def total_wall_area : ℕ := area_walls_height1 + area_walls_height2

def area_one_door : ℕ := door_height * door_width
def total_doors_area : ℕ := 2 * area_one_door

def area_one_window : ℕ := window_height * window_width
def total_windows_area : ℕ := 3 * area_one_window

def adjusted_wall_area : ℕ := total_wall_area - total_doors_area - total_windows_area - ceiling_decoration_area

def total_cost : ℕ := adjusted_wall_area * cost_per_sq_ft

-- The theorem we want to prove
theorem cost_of_white_washing : total_cost = 7580 := by
  sorry

end NUMINAMATH_GPT_cost_of_white_washing_l948_94847


namespace NUMINAMATH_GPT_joint_probability_l948_94856

noncomputable def P (A B : Prop) : ℝ := sorry
def A : Prop := sorry
def B : Prop := sorry

axiom prob_A : P A true = 0.005
axiom prob_B_given_A : P B true = 0.99

theorem joint_probability :
  P A B = 0.00495 :=
by sorry

end NUMINAMATH_GPT_joint_probability_l948_94856


namespace NUMINAMATH_GPT_sufficient_condition_l948_94862

variable (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, -1 ≤ x → x ≤ 2 → x^2 - a ≥ 0) : a ≤ -1 := 
sorry

end NUMINAMATH_GPT_sufficient_condition_l948_94862


namespace NUMINAMATH_GPT_vector_collinearity_l948_94869

variables (a b : ℝ × ℝ)

def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_collinearity : collinear (-1, 2) (1, -2) :=
by
  sorry

end NUMINAMATH_GPT_vector_collinearity_l948_94869


namespace NUMINAMATH_GPT_point_slope_form_l948_94843

theorem point_slope_form (k : ℝ) (p : ℝ × ℝ) (h_slope : k = 2) (h_point : p = (2, -3)) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ y = 2 * (x - 2) + (-3)) := 
sorry

end NUMINAMATH_GPT_point_slope_form_l948_94843


namespace NUMINAMATH_GPT_new_cylinder_height_percentage_l948_94877

variables (r h h_new : ℝ)

theorem new_cylinder_height_percentage :
  (7 / 8) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new →
  (h_new / h) = 14 / 15 :=
by
  intro h_volume_eq
  sorry

end NUMINAMATH_GPT_new_cylinder_height_percentage_l948_94877


namespace NUMINAMATH_GPT_sin_x_eq_2ab_div_a2_plus_b2_l948_94850

theorem sin_x_eq_2ab_div_a2_plus_b2
  (a b : ℝ) (x : ℝ)
  (h_tan : Real.tan x = 2 * a * b / (a^2 - b^2))
  (h_pos : 0 < b) (h_lt : b < a) (h_x : 0 < x ∧ x < Real.pi / 2) :
  Real.sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end NUMINAMATH_GPT_sin_x_eq_2ab_div_a2_plus_b2_l948_94850


namespace NUMINAMATH_GPT_find_c_l948_94845

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : 
  c = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l948_94845


namespace NUMINAMATH_GPT_crafts_sold_l948_94824

theorem crafts_sold (x : ℕ) 
  (h1 : ∃ (n : ℕ), 12 * n = x * 12)
  (h2 : x * 12 + 7 - 18 = 25):
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_crafts_sold_l948_94824


namespace NUMINAMATH_GPT_kitchen_chairs_count_l948_94891

-- Define the conditions
def total_chairs : ℕ := 9
def living_room_chairs : ℕ := 3

-- Prove the number of kitchen chairs
theorem kitchen_chairs_count : total_chairs - living_room_chairs = 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_kitchen_chairs_count_l948_94891


namespace NUMINAMATH_GPT_correct_avg_weight_of_class_l948_94885

theorem correct_avg_weight_of_class :
  ∀ (n : ℕ) (avg_wt : ℝ) (mis_A mis_B mis_C actual_A actual_B actual_C : ℝ),
  n = 30 →
  avg_wt = 60.2 →
  mis_A = 54 → actual_A = 64 →
  mis_B = 58 → actual_B = 68 →
  mis_C = 50 → actual_C = 60 →
  (n * avg_wt + (actual_A - mis_A) + (actual_B - mis_B) + (actual_C - mis_C)) / n = 61.2 :=
by
  intros n avg_wt mis_A mis_B mis_C actual_A actual_B actual_C h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_correct_avg_weight_of_class_l948_94885


namespace NUMINAMATH_GPT_sequence_equality_l948_94855

noncomputable def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_equality (x : ℝ) (hx : x = 0 ∨ x = 1 ∨ x = -1) (n : ℕ) (hn : n ≥ 3) :
  (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end NUMINAMATH_GPT_sequence_equality_l948_94855


namespace NUMINAMATH_GPT_triangle_circumscribed_circle_diameter_l948_94896

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem triangle_circumscribed_circle_diameter :
  let a := 16
  let A := Real.pi / 4   -- 45 degrees in radians
  circumscribed_circle_diameter a A = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_circumscribed_circle_diameter_l948_94896


namespace NUMINAMATH_GPT_equivalent_proof_problem_l948_94842

def option_A : ℚ := 14 / 10
def option_B : ℚ := 1 + 2 / 5
def option_C : ℚ := 1 + 6 / 15
def option_D : ℚ := 1 + 3 / 8
def option_E : ℚ := 1 + 28 / 20
def target : ℚ := 7 / 5

theorem equivalent_proof_problem : option_D ≠ target :=
by {
  sorry
}

end NUMINAMATH_GPT_equivalent_proof_problem_l948_94842


namespace NUMINAMATH_GPT_calculate_m_l948_94894

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end NUMINAMATH_GPT_calculate_m_l948_94894


namespace NUMINAMATH_GPT_fencing_required_l948_94829

theorem fencing_required (L W : ℕ) (hL : L = 40) (hA : 40 * W = 680) : 2 * W + L = 74 :=
by sorry

end NUMINAMATH_GPT_fencing_required_l948_94829


namespace NUMINAMATH_GPT_parabola_coefficients_l948_94811

theorem parabola_coefficients (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (y = (x + 2)^2 + 5) ∧ y = 9 ↔ x = 0) →
  (a, b, c) = (1, 4, 9) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parabola_coefficients_l948_94811


namespace NUMINAMATH_GPT_choir_membership_l948_94893

theorem choir_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 8 = 3) (h3 : n ≥ 100) (h4 : n ≤ 200) :
  n = 123 ∨ n = 179 :=
by
  sorry

end NUMINAMATH_GPT_choir_membership_l948_94893


namespace NUMINAMATH_GPT_lisa_total_spoons_l948_94817

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end NUMINAMATH_GPT_lisa_total_spoons_l948_94817


namespace NUMINAMATH_GPT_tree_planting_problem_l948_94834

variables (n t : ℕ)

theorem tree_planting_problem (h1 : 4 * n = t + 11) (h2 : 2 * n = t - 13) : n = 12 ∧ t = 37 :=
by
  sorry

end NUMINAMATH_GPT_tree_planting_problem_l948_94834


namespace NUMINAMATH_GPT_pictures_at_museum_l948_94830

-- Define the given conditions
def z : ℕ := 24
def k : ℕ := 14
def p : ℕ := 22

-- Define the number of pictures taken at the museum
def M : ℕ := 12

-- The theorem to be proven
theorem pictures_at_museum :
  z + M - k = p ↔ M = 12 :=
by
  sorry

end NUMINAMATH_GPT_pictures_at_museum_l948_94830


namespace NUMINAMATH_GPT_jinho_remaining_money_l948_94833

def jinho_initial_money : ℕ := 2500
def cost_per_eraser : ℕ := 120
def erasers_bought : ℕ := 5
def cost_per_pencil : ℕ := 350
def pencils_bought : ℕ := 3

theorem jinho_remaining_money :
  jinho_initial_money - (erasers_bought * cost_per_eraser + pencils_bought * cost_per_pencil) = 850 :=
by
  sorry

end NUMINAMATH_GPT_jinho_remaining_money_l948_94833


namespace NUMINAMATH_GPT_surface_area_of_reassembled_solid_l948_94886

noncomputable def total_surface_area : ℕ :=
let height_E := 1/4
let height_F := 1/6
let height_G := 1/9 
let height_H := 1 - (height_E + height_F + height_G)
let face_area := 2 * 1
(face_area * 2)     -- Top and bottom surfaces
+ 2                -- Side surfaces (1 foot each side * 2 sides)
+ (face_area * 2)   -- Front and back surfaces 

theorem surface_area_of_reassembled_solid :
  total_surface_area = 10 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_reassembled_solid_l948_94886


namespace NUMINAMATH_GPT_simplify_expression_l948_94876

variable {a b c : ℤ}

theorem simplify_expression (a b c : ℤ) : 3 * a - (4 * a - 6 * b - 3 * c) - 5 * (c - b) = -a + 11 * b - 2 * c :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l948_94876


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l948_94818

def simple_prop (p q : Prop) :=
  (¬ (p ∧ q)) → (¬ (p ∨ q))

theorem necessary_but_not_sufficient (p q : Prop) (h : simple_prop p q) :
  ((¬ (p ∧ q)) → (¬ (p ∨ q))) ∧ ¬ ((¬ (p ∨ q)) → (¬ (p ∧ q))) := by
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l948_94818


namespace NUMINAMATH_GPT_problem_statement_l948_94872

variable (m : ℝ) -- We declare m as a real number

theorem problem_statement (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := 
by 
  sorry -- The proof is omitted

end NUMINAMATH_GPT_problem_statement_l948_94872


namespace NUMINAMATH_GPT_find_number_l948_94803

theorem find_number (x n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l948_94803


namespace NUMINAMATH_GPT_select_student_for_performance_and_stability_l948_94861

def average_score_A : ℝ := 6.2
def average_score_B : ℝ := 6.0
def average_score_C : ℝ := 5.8
def average_score_D : ℝ := 6.2

def variance_A : ℝ := 0.32
def variance_B : ℝ := 0.58
def variance_C : ℝ := 0.12
def variance_D : ℝ := 0.25

theorem select_student_for_performance_and_stability :
  (average_score_A ≤ average_score_D ∧ variance_D < variance_A) →
  (average_score_B < average_score_A ∧ average_score_B < average_score_D) →
  (average_score_C < average_score_A ∧ average_score_C < average_score_D) →
  "D" = "D" :=
by
  intros h₁ h₂ h₃
  exact rfl

end NUMINAMATH_GPT_select_student_for_performance_and_stability_l948_94861


namespace NUMINAMATH_GPT_greatest_int_value_not_satisfy_condition_l948_94820

/--
For the inequality 8 - 6x > 26, the greatest integer value 
of x that satisfies this is -4.
-/
theorem greatest_int_value (x : ℤ) : 8 - 6 * x > 26 → x ≤ -4 :=
by sorry

theorem not_satisfy_condition (x : ℤ) : x > -4 → ¬ (8 - 6 * x > 26) :=
by sorry

end NUMINAMATH_GPT_greatest_int_value_not_satisfy_condition_l948_94820


namespace NUMINAMATH_GPT_second_sum_is_1704_l948_94852

theorem second_sum_is_1704
    (total_sum : ℝ)
    (x : ℝ)
    (interest_rate_first_part : ℝ)
    (time_first_part : ℝ)
    (interest_rate_second_part : ℝ)
    (time_second_part : ℝ)
    (h1 : total_sum = 2769)
    (h2 : interest_rate_first_part = 3)
    (h3 : time_first_part = 8)
    (h4 : interest_rate_second_part = 5)
    (h5 : time_second_part = 3)
    (h6 : 24 * x / 100 = (total_sum - x) * 15 / 100) :
    total_sum - x = 1704 :=
  by
    sorry

end NUMINAMATH_GPT_second_sum_is_1704_l948_94852


namespace NUMINAMATH_GPT_symmetry_axis_is_neg_pi_over_12_l948_94801

noncomputable def symmetry_axis_of_sine_function : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, (3 * x + 3 * Real.pi / 4 = Real.pi / 2 + k * Real.pi) ↔ (x = - Real.pi / 12 + k * Real.pi / 3)

theorem symmetry_axis_is_neg_pi_over_12 : symmetry_axis_of_sine_function := sorry

end NUMINAMATH_GPT_symmetry_axis_is_neg_pi_over_12_l948_94801


namespace NUMINAMATH_GPT_expand_expression_l948_94868

variable (x : ℝ)

theorem expand_expression : (9 * x + 4) * (2 * x ^ 2) = 18 * x ^ 3 + 8 * x ^ 2 :=
by sorry

end NUMINAMATH_GPT_expand_expression_l948_94868


namespace NUMINAMATH_GPT_students_enrolled_both_english_and_german_l948_94849

def total_students : ℕ := 32
def enrolled_german : ℕ := 22
def only_english : ℕ := 10
def students_enrolled_at_least_one_subject := total_students

theorem students_enrolled_both_english_and_german :
  ∃ (e_g : ℕ), e_g = enrolled_german - only_english :=
by
  sorry

end NUMINAMATH_GPT_students_enrolled_both_english_and_german_l948_94849


namespace NUMINAMATH_GPT_power_division_l948_94890

theorem power_division (a b : ℕ) (h : 64 = 8^2) : (8 ^ 15) / (64 ^ 7) = 8 := by
  sorry

end NUMINAMATH_GPT_power_division_l948_94890


namespace NUMINAMATH_GPT_find_x_such_that_l948_94802

theorem find_x_such_that {x : ℝ} (h : ⌈x⌉ * x + 15 = 210) : x = 195 / 14 :=
by
  sorry

end NUMINAMATH_GPT_find_x_such_that_l948_94802


namespace NUMINAMATH_GPT_side_length_of_square_l948_94807

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l948_94807


namespace NUMINAMATH_GPT_exists_C_a_n1_minus_a_n_l948_94822

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| 2 => 8
| (n+1) => a (n - 1) + (4 / n) * a n

theorem exists_C (C : ℕ) (hC : C = 2) : ∃ C > 0, ∀ n > 0, a n ≤ C * n^2 := by
  use 2
  sorry

theorem a_n1_minus_a_n (n : ℕ) (h : n > 0) : a (n + 1) - a n ≤ 4 * n + 3 := by
  sorry

end NUMINAMATH_GPT_exists_C_a_n1_minus_a_n_l948_94822


namespace NUMINAMATH_GPT_number_of_zeros_in_factorial_30_l948_94846

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end NUMINAMATH_GPT_number_of_zeros_in_factorial_30_l948_94846


namespace NUMINAMATH_GPT_quadratic_function_properties_l948_94839

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem quadratic_function_properties :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f x ≤ f y) ∧
  (∃ x : ℝ, x = 1.5 ∧ ∀ y : ℝ, f x ≤ f y) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l948_94839


namespace NUMINAMATH_GPT_x_range_condition_l948_94873

-- Define the inequality and conditions
def inequality (x : ℝ) : Prop := x^2 + 2 * x < 8

-- The range of x must be (-4, 2)
theorem x_range_condition (x : ℝ) : inequality x → x > -4 ∧ x < 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_x_range_condition_l948_94873


namespace NUMINAMATH_GPT_pears_value_equivalence_l948_94867

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end NUMINAMATH_GPT_pears_value_equivalence_l948_94867


namespace NUMINAMATH_GPT_derivative_at_1_l948_94814

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_1 : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_derivative_at_1_l948_94814


namespace NUMINAMATH_GPT_median_is_70_74_l948_94844

-- Define the histogram data as given
def histogram : List (ℕ × ℕ) :=
  [(85, 5), (80, 15), (75, 18), (70, 22), (65, 20), (60, 10), (55, 10)]

-- Function to calculate the cumulative sum at each interval
def cumulativeSum (hist : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  hist.scanl (λ acc pair => (pair.1, acc.2 + pair.2)) (0, 0)

-- Function to find the interval where the median lies
def medianInterval (hist : List (ℕ × ℕ)) : ℕ :=
  let cumSum := cumulativeSum hist
  -- The median is the 50th and 51st scores
  let medianPos := 50
  -- Find the interval that contains the median position
  List.find? (λ pair => medianPos ≤ pair.2) cumSum |>.getD (0, 0) |>.1

-- The theorem stating that the median interval is 70-74
theorem median_is_70_74 : medianInterval histogram = 70 :=
  by sorry

end NUMINAMATH_GPT_median_is_70_74_l948_94844


namespace NUMINAMATH_GPT_find_S_l948_94813

theorem find_S (x y : ℝ) (h : x + y = 4) : 
  ∃ S, (∀ x y, x + y = 4 → 3*x^2 + y^2 = 12) → S = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_S_l948_94813


namespace NUMINAMATH_GPT_total_amount_received_l948_94889

theorem total_amount_received
  (total_books : ℕ := 500)
  (novels_price : ℕ := 8)
  (biographies_price : ℕ := 12)
  (science_books_price : ℕ := 10)
  (novels_discount : ℚ := 0.25)
  (biographies_discount : ℚ := 0.30)
  (science_books_discount : ℚ := 0.20)
  (sales_tax : ℚ := 0.05)
  (remaining_novels : ℕ := 60)
  (remaining_biographies : ℕ := 65)
  (remaining_science_books : ℕ := 50)
  (novel_ratio_sold : ℚ := 3/5)
  (biography_ratio_sold : ℚ := 2/3)
  (science_book_ratio_sold : ℚ := 7/10)
  (original_novels : ℕ := 150)
  (original_biographies : ℕ := 195)
  (original_science_books : ℕ := 167) -- Rounded from 166.67
  (sold_novels : ℕ := 90)
  (sold_biographies : ℕ := 130)
  (sold_science_books : ℕ := 117)
  (total_revenue_before_discount : ℚ := (90 * 8 + 130 * 12 + 117 * 10))
  (total_revenue_after_discount : ℚ := (720 * (1 - 0.25) + 1560 * (1 - 0.30) + 1170 * (1 - 0.20)))
  (total_revenue_after_tax : ℚ := (2568 * 1.05)) :
  total_revenue_after_tax = 2696.4 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_received_l948_94889


namespace NUMINAMATH_GPT_ten_times_six_x_plus_fourteen_pi_l948_94853

theorem ten_times_six_x_plus_fourteen_pi (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) : 
  10 * (6 * x + 14 * Real.pi) = 4 * Q :=
by
  sorry

end NUMINAMATH_GPT_ten_times_six_x_plus_fourteen_pi_l948_94853


namespace NUMINAMATH_GPT_union_is_equivalent_l948_94851

def A (x : ℝ) : Prop := x ^ 2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem union_is_equivalent (x : ℝ) :
  (A x ∨ B x) ↔ (-2 ≤ x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_union_is_equivalent_l948_94851


namespace NUMINAMATH_GPT_final_apples_count_l948_94827

-- Definitions from the problem conditions
def initialApples : ℕ := 150
def soldToJill (initial : ℕ) : ℕ := initial * 30 / 100
def remainingAfterJill (initial : ℕ) := initial - soldToJill initial
def soldToJune (remaining : ℕ) : ℕ := remaining * 20 / 100
def remainingAfterJune (remaining : ℕ) := remaining - soldToJune remaining
def givenToFriend (current : ℕ) : ℕ := current - 2
def soldAfterFriend (current : ℕ) : ℕ := current * 10 / 100
def remainingAfterAll (current : ℕ) := current - soldAfterFriend current

theorem final_apples_count : remainingAfterAll (givenToFriend (remainingAfterJune (remainingAfterJill initialApples))) = 74 :=
by
  sorry

end NUMINAMATH_GPT_final_apples_count_l948_94827


namespace NUMINAMATH_GPT_possible_integer_radii_l948_94882

theorem possible_integer_radii (r : ℕ) (h : r < 140) : 
  (3 * 2 * r * π = 2 * 140 * π) → ∃ rs : Finset ℕ, rs.card = 10 := by
  sorry

end NUMINAMATH_GPT_possible_integer_radii_l948_94882


namespace NUMINAMATH_GPT_simplify_expression_l948_94816

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l948_94816


namespace NUMINAMATH_GPT_number_of_integer_solutions_l948_94875

theorem number_of_integer_solutions :
  ∃ (n : ℕ), 
  (∀ (x y : ℤ), 2 * x + 3 * y = 7 ∧ 5 * x + n * y = n ^ 2) ∧
  (n = 8) := 
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l948_94875


namespace NUMINAMATH_GPT_cos_arcsin_l948_94821

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_cos_arcsin_l948_94821


namespace NUMINAMATH_GPT_total_dogs_l948_94879

axiom brown_dogs : ℕ
axiom white_dogs : ℕ
axiom black_dogs : ℕ

theorem total_dogs (b w bl : ℕ) (h1 : b = 20) (h2 : w = 10) (h3 : bl = 15) : (b + w + bl) = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_dogs_l948_94879


namespace NUMINAMATH_GPT_quadrant_of_point_C_l948_94815

theorem quadrant_of_point_C
  (a b : ℝ)
  (h1 : -(a-2) = -1)
  (h2 : b+5 = 3) :
  a = 3 ∧ b = -2 ∧ 0 < a ∧ b < 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadrant_of_point_C_l948_94815


namespace NUMINAMATH_GPT_odd_function_neg_expression_l948_94806

theorem odd_function_neg_expression (f : ℝ → ℝ) (h₀ : ∀ x > 0, f x = x^3 + x + 1)
    (h₁ : ∀ x, f (-x) = -f x) : ∀ x < 0, f x = x^3 + x - 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_neg_expression_l948_94806


namespace NUMINAMATH_GPT_certain_time_in_seconds_l948_94899

theorem certain_time_in_seconds
  (ratio : ℕ) (minutes : ℕ) (time_in_minutes : ℕ) (seconds_in_a_minute : ℕ)
  (h_ratio : ratio = 8)
  (h_minutes : minutes = 4)
  (h_time : time_in_minutes = minutes)
  (h_conversion : seconds_in_a_minute = 60) :
  time_in_minutes * seconds_in_a_minute = 240 :=
by
  sorry

end NUMINAMATH_GPT_certain_time_in_seconds_l948_94899


namespace NUMINAMATH_GPT_vase_net_gain_l948_94809

theorem vase_net_gain 
  (selling_price : ℝ)
  (V1_cost : ℝ)
  (V2_cost : ℝ)
  (hyp1 : selling_price = 2.50)
  (hyp2 : 1.25 * V1_cost = selling_price)
  (hyp3 : 0.85 * V2_cost = selling_price) :
  (selling_price + selling_price) - (V1_cost + V2_cost) = 0.06 := 
by 
  sorry

end NUMINAMATH_GPT_vase_net_gain_l948_94809


namespace NUMINAMATH_GPT_complex_number_corresponding_to_OB_l948_94860

theorem complex_number_corresponding_to_OB :
  let OA : ℂ := 6 + 5 * Complex.I
  let AB : ℂ := 4 + 5 * Complex.I
  OB = OA + AB -> OB = 10 + 10 * Complex.I := by
  sorry

end NUMINAMATH_GPT_complex_number_corresponding_to_OB_l948_94860


namespace NUMINAMATH_GPT_no_primes_in_range_l948_94836

theorem no_primes_in_range (n : ℕ) (hn : n > 2) : 
  ∀ k, n! + 2 < k ∧ k < n! + n + 1 → ¬Prime k := 
sorry

end NUMINAMATH_GPT_no_primes_in_range_l948_94836


namespace NUMINAMATH_GPT_mower_next_tangent_point_l948_94864

theorem mower_next_tangent_point (r_garden r_mower : ℝ) (h_garden : r_garden = 15) (h_mower : r_mower = 5) :
    ∃ θ : ℝ, θ = (2 * π * r_mower / (2 * π * r_garden)) * 360 ∧ θ = 120 :=
sorry

end NUMINAMATH_GPT_mower_next_tangent_point_l948_94864


namespace NUMINAMATH_GPT_servings_of_honey_l948_94832

theorem servings_of_honey :
  let total_ounces := 37 + 1/3
  let serving_size := 1 + 1/2
  total_ounces / serving_size = 24 + 8/9 :=
by
  sorry

end NUMINAMATH_GPT_servings_of_honey_l948_94832


namespace NUMINAMATH_GPT_find_a_l948_94810

theorem find_a {a : ℝ} (h : ∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5 → x ≤ 3) : a = 8 :=
sorry

end NUMINAMATH_GPT_find_a_l948_94810


namespace NUMINAMATH_GPT_InfinitePairsExist_l948_94823

theorem InfinitePairsExist (a b : ℕ) : (∀ n : ℕ, ∃ a b : ℕ, a ∣ b^2 + 1 ∧ b ∣ a^2 + 1) :=
sorry

end NUMINAMATH_GPT_InfinitePairsExist_l948_94823


namespace NUMINAMATH_GPT_hannah_total_spending_l948_94871

def sweatshirt_price : ℕ := 15
def sweatshirt_quantity : ℕ := 3
def t_shirt_price : ℕ := 10
def t_shirt_quantity : ℕ := 2
def socks_price : ℕ := 5
def socks_quantity : ℕ := 4
def jacket_price : ℕ := 50
def discount_rate : ℚ := 0.10

noncomputable def total_cost_before_discount : ℕ :=
  (sweatshirt_quantity * sweatshirt_price) +
  (t_shirt_quantity * t_shirt_price) +
  (socks_quantity * socks_price) +
  jacket_price

noncomputable def total_cost_after_discount : ℚ :=
  total_cost_before_discount - (discount_rate * total_cost_before_discount)

theorem hannah_total_spending : total_cost_after_discount = 121.50 := by
  sorry

end NUMINAMATH_GPT_hannah_total_spending_l948_94871


namespace NUMINAMATH_GPT_trig_identity_l948_94804

noncomputable def sin_40 := Real.sin (40 * Real.pi / 180)
noncomputable def tan_10 := Real.tan (10 * Real.pi / 180)
noncomputable def sqrt_3 := Real.sqrt 3

theorem trig_identity : sin_40 * (tan_10 - sqrt_3) = -1 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l948_94804


namespace NUMINAMATH_GPT_total_stuffed_animals_l948_94825

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end NUMINAMATH_GPT_total_stuffed_animals_l948_94825


namespace NUMINAMATH_GPT_john_pays_percentage_of_srp_l948_94848

theorem john_pays_percentage_of_srp (P MP : ℝ) (h1 : P = 1.20 * MP) (h2 : MP > 0): 
  (0.60 * MP / P) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_john_pays_percentage_of_srp_l948_94848


namespace NUMINAMATH_GPT_find_m_l948_94831

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l948_94831


namespace NUMINAMATH_GPT_actual_height_of_boy_l948_94826

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end NUMINAMATH_GPT_actual_height_of_boy_l948_94826


namespace NUMINAMATH_GPT_loan_difference_l948_94874

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def monthly_compounding : ℝ :=
  future_value 8000 0.10 12 5

noncomputable def semi_annual_compounding : ℝ :=
  future_value 8000 0.10 2 5

noncomputable def interest_difference : ℝ :=
  monthly_compounding - semi_annual_compounding

theorem loan_difference (P : ℝ) (r : ℝ) (n_m n_s t : ℝ) :
    interest_difference = 745.02 := by sorry

end NUMINAMATH_GPT_loan_difference_l948_94874


namespace NUMINAMATH_GPT_find_cos_beta_l948_94800

variable {α β : ℝ}
variable (h_acute_α : 0 < α ∧ α < π / 2)
variable (h_acute_β : 0 < β ∧ β < π / 2)
variable (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
variable (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5)

theorem find_cos_beta 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
  (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := 
sorry

end NUMINAMATH_GPT_find_cos_beta_l948_94800


namespace NUMINAMATH_GPT_incorrect_statement_maximum_value_l948_94857

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_statement_maximum_value :
  ∃ (a b c : ℝ), 
    (quadratic_function a b c 1 = -40) ∧
    (quadratic_function a b c (-1) = -8) ∧
    (quadratic_function a b c (-3) = 8) ∧
    (∀ (x_max : ℝ), (x_max = -b / (2 * a)) →
      (quadratic_function a b c x_max = 10) ∧
      (quadratic_function a b c x_max ≠ 8)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_maximum_value_l948_94857


namespace NUMINAMATH_GPT_total_rent_paid_l948_94858

theorem total_rent_paid
  (weekly_rent : ℕ) (num_weeks : ℕ) 
  (hrent : weekly_rent = 388)
  (hweeks : num_weeks = 1359) :
  weekly_rent * num_weeks = 527292 := 
by
  sorry

end NUMINAMATH_GPT_total_rent_paid_l948_94858


namespace NUMINAMATH_GPT_plane_equidistant_from_B_and_C_l948_94805

-- Define points B and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def B : Point3D := { x := 4, y := 1, z := 0 }
def C : Point3D := { x := 2, y := 0, z := 3 }

-- Define the predicate for a plane equation
def plane_eq (a b c d : ℝ) (P : Point3D) : Prop :=
  a * P.x + b * P.y + c * P.z + d = 0

-- The problem statement
theorem plane_equidistant_from_B_and_C :
  ∃ D : ℝ, plane_eq (-2) (-1) 3 D { x := B.x, y := B.y, z := B.z } ∧
            plane_eq (-2) (-1) 3 D { x := C.x, y := C.y, z := C.z } :=
sorry

end NUMINAMATH_GPT_plane_equidistant_from_B_and_C_l948_94805


namespace NUMINAMATH_GPT_gcd_21_eq_7_count_l948_94808

theorem gcd_21_eq_7_count : Nat.card {n : Fin 200 // Nat.gcd 21 n = 7} = 19 := 
by
  sorry

end NUMINAMATH_GPT_gcd_21_eq_7_count_l948_94808


namespace NUMINAMATH_GPT_find_certain_number_l948_94840

theorem find_certain_number 
  (num : ℝ)
  (h1 : num / 14.5 = 177)
  (h2 : 29.94 / 1.45 = 17.7) : 
  num = 2566.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_certain_number_l948_94840


namespace NUMINAMATH_GPT_jenny_profit_l948_94895

-- Definitions for the conditions
def cost_per_pan : ℝ := 10.0
def pans_sold : ℕ := 20
def selling_price_per_pan : ℝ := 25.0

-- Definition for the profit calculation based on the given conditions
def total_revenue : ℝ := pans_sold * selling_price_per_pan
def total_cost : ℝ := pans_sold * cost_per_pan
def profit : ℝ := total_revenue - total_cost

-- The actual theorem statement
theorem jenny_profit : profit = 300.0 := by
  sorry

end NUMINAMATH_GPT_jenny_profit_l948_94895


namespace NUMINAMATH_GPT_numberOfColoringWays_l948_94884

-- Define the problem parameters
def totalBalls : Nat := 5
def redBalls : Nat := 1
def blueBalls : Nat := 1
def yellowBalls : Nat := 2
def whiteBalls : Nat := 1

-- Show that the number of permutations of the multiset is 60
theorem numberOfColoringWays : (Nat.factorial totalBalls) / ((Nat.factorial redBalls) * (Nat.factorial blueBalls) * (Nat.factorial yellowBalls) * (Nat.factorial whiteBalls)) = 60 :=
  by
  simp [totalBalls, redBalls, blueBalls, yellowBalls, whiteBalls]
  sorry

end NUMINAMATH_GPT_numberOfColoringWays_l948_94884


namespace NUMINAMATH_GPT_train_cross_duration_l948_94859

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 162
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_duration :
  time_to_cross_pole = 250 / (162 * (1000 / 3600)) :=
by
  -- The detailed proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_train_cross_duration_l948_94859


namespace NUMINAMATH_GPT_find_m_l948_94887

theorem find_m (x m : ℝ) (h1 : 4 * x + 2 * m = 5 * x + 1) (h2 : 3 * x = 6 * x - 1) : m = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l948_94887


namespace NUMINAMATH_GPT_tubs_from_usual_vendor_l948_94898

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end NUMINAMATH_GPT_tubs_from_usual_vendor_l948_94898


namespace NUMINAMATH_GPT_tan_pi_add_alpha_eq_two_l948_94837

theorem tan_pi_add_alpha_eq_two
  (α : ℝ)
  (h : Real.tan (Real.pi + α) = 2) :
  (2 * Real.sin α - Real.cos α) / (3 * Real.sin α + 2 * Real.cos α) = 3 / 8 :=
sorry

end NUMINAMATH_GPT_tan_pi_add_alpha_eq_two_l948_94837


namespace NUMINAMATH_GPT_abs_a_plus_2_always_positive_l948_94841

theorem abs_a_plus_2_always_positive (a : ℝ) : |a| + 2 > 0 := 
sorry

end NUMINAMATH_GPT_abs_a_plus_2_always_positive_l948_94841


namespace NUMINAMATH_GPT_initial_avg_weight_proof_l948_94881

open Classical

variable (A B C D E : ℝ) (W : ℝ)

-- Given conditions
def initial_avg_weight_A_B_C : Prop := W = (A + B + C) / 3
def avg_with_D : Prop := (A + B + C + D) / 4 = 80
def E_weighs_D_plus_8 : Prop := E = D + 8
def avg_with_E_replacing_A : Prop := (B + C + D + E) / 4 = 79
def weight_of_A : Prop := A = 80

-- Question to prove
theorem initial_avg_weight_proof (h1 : initial_avg_weight_A_B_C W A B C)
                                 (h2 : avg_with_D A B C D)
                                 (h3 : E_weighs_D_plus_8 D E)
                                 (h4 : avg_with_E_replacing_A B C D E)
                                 (h5 : weight_of_A A) :
  W = 84 := by
  sorry

end NUMINAMATH_GPT_initial_avg_weight_proof_l948_94881


namespace NUMINAMATH_GPT_seeds_in_bucket_A_l948_94819

theorem seeds_in_bucket_A (A B C : ℕ) (h_total : A + B + C = 100) (h_B : B = 30) (h_C : C = 30) : A = 40 :=
by
  sorry

end NUMINAMATH_GPT_seeds_in_bucket_A_l948_94819
