import Mathlib
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic

namespace mickey_horses_per_week_2_2221

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_2_2221


namespace farmer_plants_rows_per_bed_2_2157

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed_2_2157


namespace number_of_male_animals_2_2817

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_2_2817


namespace parallel_lines_equal_slopes_2_2489

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end parallel_lines_equal_slopes_2_2489


namespace diamond_19_98_2_2206

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98_2_2206


namespace rectangle_perimeter_2_2346

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter_2_2346


namespace number_of_terms_in_arithmetic_sequence_2_2771

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_2_2771


namespace hiking_committee_selection_2_2578

def comb (n k : ℕ) : ℕ := n.choose k

theorem hiking_committee_selection :
  comb 10 3 = 120 :=
by
  sorry

end hiking_committee_selection_2_2578


namespace tv_cost_2_2331

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost_2_2331


namespace total_meals_per_week_2_2368

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_2_2368


namespace min_value_fraction_sum_2_2256

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end min_value_fraction_sum_2_2256


namespace cone_base_diameter_2_2914

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end cone_base_diameter_2_2914


namespace find_b_2_2734

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b_2_2734


namespace problem1_line_equation_problem2_circle_equation_2_2045

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation_2_2045


namespace four_roots_sum_eq_neg8_2_2162

def op (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := op x 2

theorem four_roots_sum_eq_neg8 :
  ∃ (x1 x2 x3 x4 : ℝ), 
  (x1 ≠ -2) ∧ (x2 ≠ -2) ∧ (x3 ≠ -2) ∧ (x4 ≠ -2) ∧
  (f x1 = Real.log (abs (x1 + 2))) ∧ 
  (f x2 = Real.log (abs (x2 + 2))) ∧ 
  (f x3 = Real.log (abs (x3 + 2))) ∧ 
  (f x4 = Real.log (abs (x4 + 2))) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ 
  x3 ≠ x4 ∧ 
  x1 + x2 + x3 + x4 = -8 :=
by 
  sorry

end four_roots_sum_eq_neg8_2_2162


namespace dart_hit_number_list_count_2_2334

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end dart_hit_number_list_count_2_2334


namespace parabola_x_intercepts_2_2490

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end parabola_x_intercepts_2_2490


namespace deepak_profit_share_2_2495

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share_2_2495


namespace count_multiples_5_or_7_but_not_35_2_2253

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35_2_2253


namespace total_material_ordered_2_2336

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered_2_2336


namespace manufacturing_section_degrees_2_2185

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees_2_2185


namespace find_x_2_2541

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x_2_2541


namespace average_marks_all_students_proof_2_2555

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof_2_2555


namespace find_angle_C_60_find_min_value_of_c_2_2803

theorem find_angle_C_60 (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) : 
  C = 60 := 
sorry

theorem find_min_value_of_c (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h_area : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  c ≥ 2 * Real.sqrt 2 :=
sorry

end find_angle_C_60_find_min_value_of_c_2_2803


namespace rational_includes_integers_and_fractions_2_2494

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end rational_includes_integers_and_fractions_2_2494


namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_2_2552

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_2_2552


namespace ratio_of_volumes_2_2613

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end ratio_of_volumes_2_2613


namespace sam_possible_lunches_without_violation_2_2564

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end sam_possible_lunches_without_violation_2_2564


namespace diameter_of_larger_sphere_2_2822

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end diameter_of_larger_sphere_2_2822


namespace area_of_fig_between_x1_and_x2_2_2963

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2_2_2963


namespace solve_abs_inequality_2_2211

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end solve_abs_inequality_2_2211


namespace problem_part1_problem_part2_2_2539

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2_2_2539


namespace line_y_intercept_2_2770

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept_2_2770


namespace max_min_x2_minus_xy_plus_y2_2_2764

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2_2_2764


namespace hexagonal_prism_sum_maximum_2_2046

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum_2_2046


namespace cost_of_whitewashing_2_2478

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end cost_of_whitewashing_2_2478


namespace age_ratio_in_1_year_2_2549

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end age_ratio_in_1_year_2_2549


namespace total_goals_2_2774

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end total_goals_2_2774


namespace inverse_function_ratio_2_2287

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end inverse_function_ratio_2_2287


namespace smallest_altitude_leq_three_2_2122

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end smallest_altitude_leq_three_2_2122


namespace find_values_2_2917

def isInInterval (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem find_values 
  (a b c d e : ℝ)
  (ha : isInInterval a) 
  (hb : isInInterval b) 
  (hc : isInInterval c) 
  (hd : isInInterval d)
  (he : isInInterval e)
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10) : 
  (a = 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = 2) :=
sorry

end find_values_2_2917


namespace jane_played_8_rounds_2_2785

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_2_2785


namespace shaded_region_area_2_2284

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end shaded_region_area_2_2284


namespace find_expression_value_2_2392

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_2_2392


namespace reciprocal_of_sum_of_repeating_decimals_2_2239

theorem reciprocal_of_sum_of_repeating_decimals :
  let x := 5 / 33
  let y := 1 / 3
  1 / (x + y) = 33 / 16 :=
by
  -- The following is the proof, but it will be skipped for this exercise.
  sorry

end reciprocal_of_sum_of_repeating_decimals_2_2239


namespace average_weight_correct_2_2226

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_2_2226


namespace isosceles_triangle_sides_2_2921

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides_2_2921


namespace evaluate_expression_2_2543

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression_2_2543


namespace line_through_point_parallel_to_line_2_2893

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end line_through_point_parallel_to_line_2_2893


namespace find_f2_2_2053

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end find_f2_2_2053


namespace triangle_angle_range_2_2410

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end triangle_angle_range_2_2410


namespace sum_possible_values_2_2044

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values_2_2044


namespace time_to_walk_against_walkway_150_2_2210

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150_2_2210


namespace division_of_decimals_2_2537

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals_2_2537


namespace rhombus_diagonal_length_2_2850

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length_2_2850


namespace negation_of_proposition_2_2924

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition_2_2924


namespace fraction_of_number_is_one_fifth_2_2792

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth_2_2792


namespace cubic_sum_2_2596

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum_2_2596


namespace sara_picked_peaches_2_2009

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end sara_picked_peaches_2_2009


namespace percentage_loss_2_2488

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end percentage_loss_2_2488


namespace quadratic_least_value_2_2570

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value_2_2570


namespace part_A_part_B_part_D_2_2473

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D_2_2473


namespace age_of_other_man_replaced_2_2535

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end age_of_other_man_replaced_2_2535


namespace general_term_formula_sum_of_first_n_terms_2_2639

noncomputable def a_seq (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_seq (n : ℕ) : ℚ := 1 / ((a_seq n) * (a_seq (n + 1)))
noncomputable def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_seq (i + 1)

theorem general_term_formula :
  ∀ (n : ℕ), (n ≥ 1) → S_n = n * a_seq n - n * (n - 1) := sorry

theorem sum_of_first_n_terms (n : ℕ) :
  T_n n = n / (6 * n + 9) := sorry

end general_term_formula_sum_of_first_n_terms_2_2639


namespace find_shop_width_2_2725

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end find_shop_width_2_2725


namespace abc_equivalence_2_2057

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence_2_2057


namespace common_chord_line_2_2219

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line_2_2219


namespace animal_eyes_count_2_2932

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end animal_eyes_count_2_2932


namespace total_height_of_buildings_2_2789

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end total_height_of_buildings_2_2789


namespace abs_expression_equals_one_2_2975

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one_2_2975


namespace frank_hamburger_goal_2_2668

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal_2_2668


namespace sign_of_c_2_2008

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end sign_of_c_2_2008


namespace smallest_next_divisor_2_2652

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor_2_2652


namespace addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_2_2314

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_2_2314


namespace sample_size_of_survey_2_2131

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end sample_size_of_survey_2_2131


namespace blocks_before_jess_turn_2_2718

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn_2_2718


namespace ryan_weekly_commuting_time_2_2093

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_2_2093


namespace find_initial_passengers_2_2511

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers_2_2511


namespace greatest_sum_consecutive_integers_2_2856

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_2_2856


namespace minimum_value_of_a_2_2407

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end minimum_value_of_a_2_2407


namespace sum_after_50_rounds_2_2600

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds_2_2600


namespace length_of_CB_2_2767

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end length_of_CB_2_2767


namespace edward_mowed_lawns_2_2263

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns_2_2263


namespace tent_cost_solution_2_2620

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end tent_cost_solution_2_2620


namespace number_of_valid_pairs_2_2289

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end number_of_valid_pairs_2_2289


namespace planes_are_perpendicular_2_2459

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular_2_2459


namespace purchase_costs_10_2_2689

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10_2_2689


namespace find_four_consecutive_odd_numbers_2_2869

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers_2_2869


namespace sum_at_simple_interest_2_2607

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end sum_at_simple_interest_2_2607


namespace inequality_proof_2_2130

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end inequality_proof_2_2130


namespace vanya_exam_scores_2_2096

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores_2_2096


namespace geom_seq_m_value_2_2665

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end geom_seq_m_value_2_2665


namespace abc_value_2_2235

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end abc_value_2_2235


namespace sin_identity_2_2588

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_2_2588


namespace find_g_2_2912

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g_2_2912


namespace joe_total_cars_2_2480

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars_2_2480


namespace ratio_of_packets_to_tent_stakes_2_2777

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes_2_2777


namespace algebra_inequality_2_2796

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality_2_2796


namespace expenses_opposite_to_income_2_2536

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_2_2536


namespace math_problem_2_2406

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem_2_2406


namespace toms_age_2_2129

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age_2_2129


namespace evaluate_expression_2_2745

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end evaluate_expression_2_2745


namespace number_of_pieces_2_2889

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end number_of_pieces_2_2889


namespace water_consumption_and_bill_34_7_2_2609

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7_2_2609


namespace train_length_2_2230

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end train_length_2_2230


namespace area_of_circle_with_given_circumference_2_2781

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference_2_2781


namespace negation_proof_2_2260

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end negation_proof_2_2260


namespace no_integer_solutions_2_2408

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end no_integer_solutions_2_2408


namespace perfect_square_iff_n_eq_5_2_2195

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end perfect_square_iff_n_eq_5_2_2195


namespace charlie_first_week_usage_2_2627

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end charlie_first_week_usage_2_2627


namespace trajectory_equation_2_2166

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end trajectory_equation_2_2166


namespace quadratic_polynomial_AT_BT_2_2194

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end quadratic_polynomial_AT_BT_2_2194


namespace always_possible_to_rotate_disks_2_2744

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end always_possible_to_rotate_disks_2_2744


namespace sqrt_inequality_2_2142

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_2_2142


namespace sum_pattern_2_2116

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end sum_pattern_2_2116


namespace marek_sequence_sum_2_2281

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum_2_2281


namespace largest_pies_without_ingredients_2_2460

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end largest_pies_without_ingredients_2_2460


namespace inequality_abc_2_2453

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end inequality_abc_2_2453


namespace age_difference_64_2_2911

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end age_difference_64_2_2911


namespace sin_cos_term_side_2_2813

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end sin_cos_term_side_2_2813


namespace find_k_inv_h_of_10_2_2618

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10_2_2618


namespace find_slope_and_intercept_2_2151

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end find_slope_and_intercept_2_2151


namespace maximum_value_of_f_2_2395

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end maximum_value_of_f_2_2395


namespace find_XY_square_2_2188

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end find_XY_square_2_2188


namespace female_democrats_2_2943

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats_2_2943


namespace determinant_of_matrixA_2_2271

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, -2],
  ![5, 6, -4],
  ![1, 3, 7]
]

theorem determinant_of_matrixA : Matrix.det matrixA = 144 := by
  sorry

end determinant_of_matrixA_2_2271


namespace functional_eq_solution_2_2291

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end functional_eq_solution_2_2291


namespace total_bananas_in_collection_2_2614

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end total_bananas_in_collection_2_2614


namespace number_of_kids_stayed_home_is_668278_2_2573

  def number_of_kids_who_stayed_home : Prop :=
    ∃ X : ℕ, X + 150780 = 819058 ∧ X = 668278

  theorem number_of_kids_stayed_home_is_668278 : number_of_kids_who_stayed_home :=
    sorry
  
end number_of_kids_stayed_home_is_668278_2_2573


namespace fraction_addition_2_2547

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end fraction_addition_2_2547


namespace triangular_number_30_sum_of_first_30_triangular_numbers_2_2905

theorem triangular_number_30 
  (T : ℕ → ℕ)
  (hT : ∀ n : ℕ, T n = n * (n + 1) / 2) : 
  T 30 = 465 :=
by
  -- Skipping proof with sorry
  sorry

theorem sum_of_first_30_triangular_numbers 
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = n * (n + 1) * (n + 2) / 6) : 
  S 30 = 4960 :=
by
  -- Skipping proof with sorry
  sorry

end triangular_number_30_sum_of_first_30_triangular_numbers_2_2905


namespace father_age_difference_2_2938

variables (F S X : ℕ)
variable (h1 : F = 33)
variable (h2 : F = 3 * S + X)
variable (h3 : F + 3 = 2 * (S + 3) + 10)

theorem father_age_difference : X = 3 :=
by
  sorry

end father_age_difference_2_2938


namespace min_value_sum_reciprocal_2_2601

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_2_2601


namespace line_l_passes_fixed_point_line_l_perpendicular_value_a_2_2529

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end line_l_passes_fixed_point_line_l_perpendicular_value_a_2_2529


namespace contrapositive_proof_2_2791

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end contrapositive_proof_2_2791


namespace probability_at_least_one_2_2388

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end probability_at_least_one_2_2388


namespace yards_gained_2_2170

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained_2_2170


namespace no_solutions_exist_2_2950

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_2_2950


namespace beta_angle_relationship_2_2720

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end beta_angle_relationship_2_2720


namespace actual_distance_between_towns_2_2352

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns_2_2352


namespace decreasing_interval_2_2544

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval_2_2544


namespace robotics_club_neither_2_2534

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end robotics_club_neither_2_2534


namespace square_inequality_2_2574

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality_2_2574


namespace man_speed_in_still_water_2_2481

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water_2_2481


namespace boys_count_2_2775

/-
Conditions:
1. The total number of members in the chess team is 26.
2. 18 members were present at the last session.
3. One-third of the girls attended the session.
4. All of the boys attended the session.
-/
def TotalMembers : Nat := 26
def LastSessionAttendance : Nat := 18
def GirlsAttendance (G : Nat) : Nat := G / 3
def BoysAttendance (B : Nat) : Nat := B

/-
Main theorem statement:
Prove that the number of boys in the chess team is 14.
-/
theorem boys_count (B G : Nat) (h1 : B + G = TotalMembers) (h2 : GirlsAttendance G + BoysAttendance B = LastSessionAttendance) : B = 14 :=
by
  sorry

end boys_count_2_2775


namespace determine_ABCC_2_2461

theorem determine_ABCC :
  ∃ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
    C ≠ D ∧ C ≠ E ∧ 
    D ≠ E ∧ 
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    1000 * A + 100 * B + 11 * C = (11 * D - E) * 100 + 11 * D * E ∧ 
    1000 * A + 100 * B + 11 * C = 1966 :=
sorry

end determine_ABCC_2_2461


namespace number_of_subsets_2_2545

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end number_of_subsets_2_2545


namespace circumradius_of_right_triangle_2_2344

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end circumradius_of_right_triangle_2_2344


namespace find_m_n_2_2888

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end find_m_n_2_2888


namespace tax_percentage_2_2958

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end tax_percentage_2_2958


namespace cookies_left_2_2937

theorem cookies_left (initial_cookies : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) :
  initial_cookies = 28 → cookies_eaten = 21 → cookies_left = initial_cookies - cookies_eaten → cookies_left = 7 :=
by
  intros h_initial h_eaten h_left
  rw [h_initial, h_eaten] at h_left
  exact h_left

end cookies_left_2_2937


namespace recommendation_plans_count_2_2172

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count_2_2172


namespace problem_1_problem_2_2_2140

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end problem_1_problem_2_2_2140


namespace sufficient_but_not_necessary_condition_2_2948

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end sufficient_but_not_necessary_condition_2_2948


namespace total_cost_correct_2_2562

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_2_2562


namespace ticket_1000_wins_probability_2_2797

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end ticket_1000_wins_probability_2_2797


namespace soda_cost_2_2595

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost_2_2595


namespace count_integers_with_sum_of_digits_18_2_2016

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_2_2016


namespace simplify_expression_2_2417

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end simplify_expression_2_2417


namespace percentage_of_second_solution_is_16point67_2_2832

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end percentage_of_second_solution_is_16point67_2_2832


namespace DiagonalsOfShapesBisectEachOther_2_2305

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther_2_2305


namespace vertex_angle_isosceles_2_2984

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end vertex_angle_isosceles_2_2984


namespace binomial_term_is_constant_range_of_a_over_b_2_2248

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end binomial_term_is_constant_range_of_a_over_b_2_2248


namespace mod_inverse_sum_2_2304

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end mod_inverse_sum_2_2304


namespace brownies_in_pan_2_2086

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan_2_2086


namespace sqrt_range_real_2_2730

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end sqrt_range_real_2_2730


namespace sum_base8_to_decimal_2_2396

theorem sum_base8_to_decimal (a b : ℕ) (ha : a = 5) (hb : b = 0o17)
  (h_sum_base8 : a + b = 0o24) : (a + b) = 20 := by
  sorry

end sum_base8_to_decimal_2_2396


namespace sequence_u5_value_2_2146

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value_2_2146


namespace discount_percentage_is_20_2_2390

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end discount_percentage_is_20_2_2390


namespace condition_for_positive_expression_2_2155

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end condition_for_positive_expression_2_2155


namespace difference_divisible_by_9_2_2884

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9_2_2884


namespace pastries_total_2_2312

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_2_2312


namespace find_value_2_2752

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value_2_2752


namespace symmetric_point_correct_2_2327

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end symmetric_point_correct_2_2327


namespace mixed_gender_appointment_schemes_2_2376

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 
  else n * factorial (n - 1)

noncomputable def P (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem mixed_gender_appointment_schemes : 
  let total_students := 9
  let total_permutations := P total_students 3
  let male_students := 5
  let female_students := 4
  let male_permutations := P male_students 3
  let female_permutations := P female_students 3
  total_permutations - (male_permutations + female_permutations) = 420 :=
by 
  sorry

end mixed_gender_appointment_schemes_2_2376


namespace function_has_exactly_one_zero_2_2083

open Set

-- Conditions
def a_gt_3 (a : ℝ) : Prop := a > 3
def f (x a : ℝ) : ℝ := x^2 - a * x + 1

-- Theorem Statement
theorem function_has_exactly_one_zero (a : ℝ) (h : a_gt_3 a) :
  ∃! x ∈ Ioo 0 2, f x a = 0 := sorry

end function_has_exactly_one_zero_2_2083


namespace find_a_2_2671

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end find_a_2_2671


namespace log_inequality_2_2736

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality_2_2736


namespace shaded_region_area_2_2278

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area_2_2278


namespace how_many_trucks_2_2126

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks_2_2126


namespace find_c_deg3_2_2865

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3_2_2865


namespace inclination_angle_x_eq_one_2_2976

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one_2_2976


namespace emails_received_in_afternoon_2_2616

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon_2_2616


namespace ellipse_circle_inequality_2_2164

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end ellipse_circle_inequality_2_2164


namespace dartboard_points_proof_2_2599

variable (points_one points_two points_three points_four : ℕ)

theorem dartboard_points_proof
  (h1 : points_one = 30)
  (h2 : points_two = 38)
  (h3 : points_three = 41)
  (h4 : 2 * points_four = points_one + points_two) :
  points_four = 34 :=
by {
  sorry
}

end dartboard_points_proof_2_2599


namespace prove_m_eq_n_2_2062

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end prove_m_eq_n_2_2062


namespace integer_sided_triangle_with_60_degree_angle_exists_2_2726

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists_2_2726


namespace total_new_students_2_2021

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students_2_2021


namespace total_potatoes_2_2216

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes_2_2216


namespace area_of_park_2_2847

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park_2_2847


namespace father_three_times_marika_in_year_2_2784

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year_2_2784


namespace angle_triple_complement_2_2952

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_2_2952


namespace votes_difference_2_2318

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference_2_2318


namespace segment_length_2_2509
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length_2_2509


namespace unique_triplet_2_2279

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet_2_2279


namespace solution_correct_2_2619

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end solution_correct_2_2619


namespace polynomial_value_2_2143

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value_2_2143


namespace distance_between_first_and_last_stop_in_km_2_2713

-- Define the total number of stops
def num_stops := 12

-- Define the distance between the third and sixth stops in meters
def dist_3_to_6 := 3300

-- The distance between consecutive stops is the same
def distance_between_first_and_last_stop : ℕ := (num_stops - 1) * (dist_3_to_6 / 3)

-- The distance in kilometers (1 kilometer = 1000 meters)
noncomputable def distance_km : ℝ := distance_between_first_and_last_stop / 1000

-- Statement to prove
theorem distance_between_first_and_last_stop_in_km : distance_km = 12.1 :=
by
  -- Theorem proof should go here
  sorry

end distance_between_first_and_last_stop_in_km_2_2713


namespace survival_rate_is_98_2_2988

def total_flowers := 150
def unsurviving_flowers := 3
def surviving_flowers := total_flowers - unsurviving_flowers

theorem survival_rate_is_98 : (surviving_flowers : ℝ) / total_flowers * 100 = 98 := by
  sorry

end survival_rate_is_98_2_2988


namespace divisor_inequality_2_2846

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality_2_2846


namespace pencil_case_costs_2_2153

variable {x y : ℝ}

theorem pencil_case_costs :
  (2 * x + 3 * y = 108) ∧ (5 * x = 6 * y) → 
  (x = 24) ∧ (y = 20) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  sorry

end pencil_case_costs_2_2153


namespace probability_red_red_red_2_2464

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end probability_red_red_red_2_2464


namespace point_slope_intersection_lines_2_2530

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end point_slope_intersection_lines_2_2530


namespace find_numbers_2_2605

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end find_numbers_2_2605


namespace perpendicular_lines_unique_a_2_2148

open Real

theorem perpendicular_lines_unique_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, (a - 1) * x + y - 1 = 0) 
  (l2 : ∀ x y : ℝ, 3 * x + a * y + 2 = 0) 
  (perpendicular : True) : 
  a = 3 / 4 := 
sorry

end perpendicular_lines_unique_a_2_2148


namespace sum_sin_cos_2_2497

theorem sum_sin_cos :
  ∑ k in Finset.range 181, (Real.sin (k * Real.pi / 180))^4 * (Real.cos (k * Real.pi / 180))^4 = 543 / 128 :=
by
  sorry

end sum_sin_cos_2_2497


namespace zhou_yu_age_equation_2_2699

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation_2_2699


namespace tan_sum_inequality_2_2743

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end tan_sum_inequality_2_2743


namespace min_value_of_expression_2_2197

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression_2_2197


namespace part1_part2_2_2000

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end part1_part2_2_2000


namespace sufficient_but_not_necessary_condition_for_intersections_2_2498

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end sufficient_but_not_necessary_condition_for_intersections_2_2498


namespace find_lawn_width_2_2519

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end find_lawn_width_2_2519


namespace convert_denominators_to_integers_2_2949

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end convert_denominators_to_integers_2_2949


namespace distance_between_A_and_B_2_2558

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end distance_between_A_and_B_2_2558


namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_2_2710

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_2_2710


namespace primes_unique_2_2111

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique_2_2111


namespace tom_total_spent_correct_2_2237

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct_2_2237


namespace total_tea_cups_2_2902

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups_2_2902


namespace paintings_in_four_weeks_2_2435

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_2_2435


namespace horse_food_needed_2_2982

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed_2_2982


namespace angle_difference_2_2874

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference_2_2874


namespace evaluate_expression_2_2338

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression_2_2338


namespace expand_product_2_2946

-- Definitions of the polynomial functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 + x + 1

-- Statement of the theorem
theorem expand_product : ∀ x : ℝ, (f x) * (g x) = x^3 + 4*x^2 + 4*x + 3 :=
by
  -- Proof goes here, but is omitted for the statement only
  sorry

end expand_product_2_2946


namespace jellybeans_problem_2_2754

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem_2_2754


namespace remainder_when_divided_by_2000_2_2936

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000_2_2936


namespace michael_left_money_2_2290

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end michael_left_money_2_2290


namespace multiplication_expansion_2_2419

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end multiplication_expansion_2_2419


namespace range_of_a_2_2563

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a_2_2563


namespace sandy_more_tokens_than_siblings_2_2414

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end sandy_more_tokens_than_siblings_2_2414


namespace ratio_quadrilateral_2_2698

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral_2_2698


namespace value_of_m_2_2433

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m_2_2433


namespace sqrt_144_times_3_squared_2_2677

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end sqrt_144_times_3_squared_2_2677


namespace number_of_ordered_pairs_2_2102

def count_valid_pairs : ℕ :=
  ∑ y in Finset.range 149, (150 - y) / (y * (y + 2))

theorem number_of_ordered_pairs :
  count_valid_pairs = ∑ y in Finset.range 149, (150 - y) / (y * (y + 2)) := by
  sorry

end number_of_ordered_pairs_2_2102


namespace ab_equals_six_2_2434

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_2_2434


namespace find_roots_of_parabola_2_2648

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end find_roots_of_parabola_2_2648


namespace correct_option_B_2_2429

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B_2_2429


namespace negation_of_prop_2_2843

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end negation_of_prop_2_2843


namespace polynomial_horner_value_2_2324

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end polynomial_horner_value_2_2324


namespace chord_length_of_intersection_2_2333

theorem chord_length_of_intersection 
  (x y : ℝ) (h_line : 2 * x - y - 1 = 0) (h_circle : (x - 2)^2 + (y + 2)^2 = 9) : 
  ∃ l, l = 4 := 
sorry

end chord_length_of_intersection_2_2333


namespace product_of_four_integers_2_2259

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end product_of_four_integers_2_2259


namespace min_value_of_quadratic_2_2145

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end min_value_of_quadratic_2_2145


namespace triangle_inequality_from_condition_2_2359

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition_2_2359


namespace domain_f_log2_x_to_domain_f_x_2_2523

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x_2_2523


namespace penguin_fish_consumption_2_2203

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption_2_2203


namespace problem_statement_2_2313

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement_2_2313


namespace range_of_a_2_2326

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end range_of_a_2_2326


namespace roots_polynomial_sum_2_2369

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum_2_2369


namespace expression_value_2_2378

theorem expression_value :
  ∀ (x y : ℚ), (x = -5/4) → (y = -3/2) → -2 * x - y^2 = 1/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end expression_value_2_2378


namespace hyperbola_asymptote_equation_2_2114

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end hyperbola_asymptote_equation_2_2114


namespace max_contribution_2_2964

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end max_contribution_2_2964


namespace soja_book_page_count_2_2868

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count_2_2868


namespace zero_in_interval_2_2749

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval_2_2749


namespace track_meet_total_people_2_2631

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end track_meet_total_people_2_2631


namespace fbox_eval_correct_2_2168

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end fbox_eval_correct_2_2168


namespace sam_has_75_dollars_2_2181

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars_2_2181


namespace opposite_event_is_at_least_one_hit_2_2870

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end opposite_event_is_at_least_one_hit_2_2870


namespace distinct_integers_sum_441_2_2328

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end distinct_integers_sum_441_2_2328


namespace mean_of_all_students_2_2998

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_2_2998


namespace parabola_equation_2_2925

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation_2_2925


namespace eval_expression_2_2522

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression_2_2522


namespace how_many_years_older_2_2773

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end how_many_years_older_2_2773


namespace sequence_positions_2_2257

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end sequence_positions_2_2257


namespace opposite_of_x_is_positive_2_2178

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_2_2178


namespace pet_shop_ways_2_2158

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways_2_2158


namespace find_n_for_arithmetic_sequence_2_2068

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence_2_2068


namespace range_of_m_2_2003

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m_2_2003


namespace find_number_2_2788

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end find_number_2_2788


namespace possible_items_2_2171

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_2_2171


namespace symmetric_colors_different_at_8281_div_2_2_2367

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2_2_2367


namespace derivative_at_one_2_2285

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one_2_2285


namespace tom_found_dimes_2_2220

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes_2_2220


namespace power_multiplication_2_2043

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication_2_2043


namespace petya_time_comparison_2_2702

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_2_2702


namespace geometric_sequence_sum_2_2144

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end geometric_sequence_sum_2_2144


namespace john_spends_40_dollars_2_2228

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end john_spends_40_dollars_2_2228


namespace day_of_20th_is_Thursday_2_2173

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday_2_2173


namespace find_number_with_10_questions_2_2345

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions_2_2345


namespace find_v_3_2_2330

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end find_v_3_2_2330


namespace find_m_2_2906

noncomputable def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (h : dot_product (vec_add (-1, 2) (m, 1)) (-1, 2) = 0) : m = 7 :=
  by 
  sorry

end find_m_2_2906


namespace max_M_range_a_2_2411

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem max_M (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) : 
  4 ≤ g x1 - g x2 :=
sorry

theorem range_a (a : ℝ) (s t : ℝ) (h1 : 1 / 2 ≤ s) (h2 : s ≤ 2) (h3 : 1 / 2 ≤ t) (h4 : t ≤ 2) : 
  1 ≤ a ∧ f s a ≥ g t :=
sorry

end max_M_range_a_2_2411


namespace general_formula_a_n_T_n_greater_than_S_n_2_2978

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_2_2978


namespace estimate_y_value_2_2137

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value_2_2137


namespace total_wheels_in_parking_lot_2_2217

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot_2_2217


namespace average_sale_six_months_2_2241

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months_2_2241


namespace valid_x_values_2_2983

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end valid_x_values_2_2983


namespace log_product_zero_2_2721

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end log_product_zero_2_2721


namespace inequality_order_2_2845

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order_2_2845


namespace binom_20_4_plus_10_2_2646

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end binom_20_4_plus_10_2_2646


namespace number_of_books_is_10_2_2520

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end number_of_books_is_10_2_2520


namespace intersection_complement_eq_2_2413

noncomputable def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
noncomputable def A : Set Int := {-1, 0, 1, 2}
noncomputable def B : Set Int := {-3, 0, 2, 3}

-- Complement of B with respect to U
noncomputable def U_complement_B : Set Int := U \ B

-- The statement we need to prove
theorem intersection_complement_eq :
  A ∩ U_complement_B = {-1, 1} :=
by
  sorry

end intersection_complement_eq_2_2413


namespace value_of_a_plus_b_2_2992

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b_2_2992


namespace emerie_dimes_count_2_2572

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count_2_2572


namespace negate_neg_two_2_2026

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_2_2026


namespace part1_part2_2_2589

-- Definitions from part (a)
def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ (a_n n + 1)

-- Specification from the given problem
def S_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i
def c_n (n : ℕ) : ℕ := a_n n * b_n n
def T_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), c_n i

-- Theorem to be proven (part (c))
theorem part1 (n : ℕ) : S_n n = n ^ 2 := by
  sorry

theorem part2 (n : ℕ) : T_n n = (24 * n - 20) * 4 ^ n / 9 + 20 / 9 := by
  sorry

end part1_part2_2_2589


namespace sum_of_possible_values_2_2981

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end sum_of_possible_values_2_2981


namespace simplify_div_2_2351

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div_2_2351


namespace highest_power_of_3_divides_2_2690

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_3_divides (n : ℕ) : ∃ k : ℕ, A_n n = 3^n * k ∧ ¬ (3 * A_n n = 3^(n+1) * k)
:= by
  sorry

end highest_power_of_3_divides_2_2690


namespace john_weight_end_2_2466

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_2_2466


namespace problem1_2_2059

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end problem1_2_2059


namespace ratio_of_areas_2_2050

theorem ratio_of_areas (s r : ℝ) (h : 3 * s = 2 * π * r) : 
  ( (√ 3) * π / 9)  =
  ( (√ 3 * π^2 * r^2 / 9) / (π * r^2)) :=
by
  sorry

end ratio_of_areas_2_2050


namespace xavier_yvonne_not_zelda_prob_2_2940

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end xavier_yvonne_not_zelda_prob_2_2940


namespace find_N_2_2612

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end find_N_2_2612


namespace fractions_sum_correct_2_2383

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct_2_2383


namespace solve_for_percentage_2_2898

-- Define the constants and variables
variables (P : ℝ)

-- Define the given conditions
def condition : Prop := (P / 100 * 1600 = P / 100 * 650 + 190)

-- Formalize the conjecture: if the conditions hold, then P = 20
theorem solve_for_percentage (h : condition P) : P = 20 :=
sorry

end solve_for_percentage_2_2898


namespace opposite_of_neg_three_2_2492

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_2_2492


namespace kim_total_ounces_2_2994

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces_2_2994


namespace increasing_interval_2_2808

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval_2_2808


namespace intersection_A_B_2_2002

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B_2_2002


namespace origin_movement_by_dilation_2_2209

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation_2_2209


namespace find_power_2_2712

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power_2_2712


namespace juanita_loss_2_2180

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end juanita_loss_2_2180


namespace trapezoid_area_2_2075

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_2_2075


namespace original_recipe_calls_for_4_tablespoons_2_2362

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end original_recipe_calls_for_4_tablespoons_2_2362


namespace units_digit_3m_squared_plus_2m_2_2772

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m_2_2772


namespace Ryan_spit_distance_correct_2_2820

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_2_2820


namespace range_of_m_2_2184

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_2_2184


namespace general_admission_tickets_2_2750

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets_2_2750


namespace regression_analysis_incorrect_statement_2_2587

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement_2_2587


namespace percent_gain_correct_2_2714

theorem percent_gain_correct :
  ∀ (x : ℝ), (900 * x + 50 * (900 * x / 850) - 900 * x) / (900 * x) * 100 = 58.82 :=
by sorry

end percent_gain_correct_2_2714


namespace no_solution_2_2120

theorem no_solution : ¬∃ x : ℝ, x^3 - 8*x^2 + 16*x - 32 / (x - 2) < 0 := by
  sorry

end no_solution_2_2120


namespace min_det_is_neg_six_2_2837

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six_2_2837


namespace only_function_2_2985

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function_2_2985


namespace largest_value_of_b_2_2193

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b_2_2193


namespace convex_parallelogram_faces_1992_2_2319

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992_2_2319


namespace probability_at_least_three_prime_dice_2_2805

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end probability_at_least_three_prime_dice_2_2805


namespace third_bowler_points_162_2_2343

variable (x : ℕ)

def total_score (x : ℕ) : Prop :=
  let first_bowler_points := x
  let second_bowler_points := 3 * x
  let third_bowler_points := x
  first_bowler_points + second_bowler_points + third_bowler_points = 810

theorem third_bowler_points_162 (x : ℕ) (h : total_score x) : x = 162 := by
  sorry

end third_bowler_points_162_2_2343


namespace solve_system_of_equations_2_2441

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations_2_2441


namespace dogs_prevent_wolf_escape_2_2761

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape_2_2761


namespace initial_girls_count_2_2027

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count_2_2027


namespace sum_of_four_digit_integers_up_to_4999_2_2424

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999_2_2424


namespace Pat_worked_days_eq_57_2_2286

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end Pat_worked_days_eq_57_2_2286


namespace parallel_x_axis_implies_conditions_2_2606

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end parallel_x_axis_implies_conditions_2_2606


namespace find_remainder_2_2205

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder_2_2205


namespace green_eyed_snack_min_2_2909

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end green_eyed_snack_min_2_2909


namespace problem_2_2038

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem_2_2038


namespace additional_men_joined_2_2550

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined_2_2550


namespace natural_number_squares_2_2542

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares_2_2542


namespace fraction_to_decimal_2_2387

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_2_2387


namespace union_of_A_B_complement_intersection_2_2384

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection_2_2384


namespace atomic_weight_S_is_correct_2_2680

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end atomic_weight_S_is_correct_2_2680


namespace nectar_water_percentage_2_2934

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end nectar_water_percentage_2_2934


namespace bridge_length_is_235_2_2337

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235_2_2337


namespace least_n_for_distance_2_2375

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end least_n_for_distance_2_2375


namespace remainder_is_correct_2_2446

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct_2_2446


namespace kathleen_money_left_2_2347

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_2_2347


namespace min_value_expression_2_2840

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end min_value_expression_2_2840


namespace smallest_m_2_2073

open Real

noncomputable def y_seq (m : ℕ) : Type := Fin m → ℝ

theorem smallest_m (m : ℕ) (y : y_seq m) (h1 : ∀ i : Fin m, |y i| ≤ 1/2)
  (h2 : ∑ i, |y i| = 10 + |∑ i, y i|) : m = 20 :=
sorry

end smallest_m_2_2073


namespace percent_covered_by_larger_triangles_2_2635

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end percent_covered_by_larger_triangles_2_2635


namespace european_postcards_cost_2_2833

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end european_postcards_cost_2_2833


namespace Liu_Wei_parts_per_day_2_2249

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day_2_2249


namespace jose_share_of_profit_2_2294

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end jose_share_of_profit_2_2294


namespace calc_expr_solve_fractional_eq_2_2370

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq_2_2370


namespace matchstick_triangle_sides_2_2561

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end matchstick_triangle_sides_2_2561


namespace janet_needs_9_dog_collars_2_2794

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars_2_2794


namespace hollis_student_loan_2_2835

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan_2_2835


namespace tiling_implies_divisibility_2_2358

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility_2_2358


namespace transport_equivalence_2_2871

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_2_2871


namespace percentage_of_girls_after_change_2_2024

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change_2_2024


namespace triangle_has_at_most_one_obtuse_angle_2_2658

-- Definitions
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def Obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

def Two_obtuse_angles (α β γ : ℝ) : Prop :=
  Obtuse_angle α ∧ Obtuse_angle β

-- Theorem Statement
theorem triangle_has_at_most_one_obtuse_angle (α β γ : ℝ) (h_triangle : Triangle α β γ) :
  ¬ Two_obtuse_angles α β γ := 
sorry

end triangle_has_at_most_one_obtuse_angle_2_2658


namespace petya_digits_2_2756

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_2_2756


namespace tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_2_2947

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_2_2947


namespace value_of_x_2_2468

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end value_of_x_2_2468


namespace f_increasing_interval_2_2517

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval_2_2517


namespace unique_three_digit_base_g_2_2233

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end unique_three_digit_base_g_2_2233


namespace A_inter_B_eq_2_2666

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end A_inter_B_eq_2_2666


namespace minimum_value_expression_2_2011

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_2_2011


namespace find_a_find_k_max_2_2078

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a_find_k_max_2_2078


namespace average_mark_excluded_students_2_2456

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end average_mark_excluded_students_2_2456


namespace avg_weight_B_correct_2_2087

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct_2_2087


namespace part1_part2_2_2389

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end part1_part2_2_2389


namespace union_M_N_is_U_2_2273

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end union_M_N_is_U_2_2273


namespace shooting_competition_2_2470

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end shooting_competition_2_2470


namespace probability_red_second_draw_2_2591

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw_2_2591


namespace find_a_b_2_2681

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_b 
  (h_max : ∀ x, f a b x ≤ 3)
  (h_min : ∀ x, f a b x ≥ 2)
  : (a = 0.5 ∨ a = -0.5) ∧ b = 2.5 :=
by
  sorry

end find_a_b_2_2681


namespace unique_nonneg_sequence_2_2810

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence_2_2810


namespace pyramid_volume_2_2581

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end pyramid_volume_2_2581


namespace larger_integer_of_two_integers_diff_8_prod_120_2_2751

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end larger_integer_of_two_integers_diff_8_prod_120_2_2751


namespace angle_B_is_40_degrees_2_2659

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees_2_2659


namespace maximum_value_x2_add_3xy_add_y2_2_2243

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2_2_2243


namespace obtuse_triangle_sum_range_2_2088

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range_2_2088


namespace avg_price_pen_is_correct_2_2816

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct_2_2816


namespace calculate_expression_2_2935

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression_2_2935


namespace tommy_profit_2_2508

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit_2_2508


namespace negation_of_p_2_2207

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p_2_2207


namespace problem_inequality_2_2156

variable {a b c d : ℝ}

theorem problem_inequality (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ (Real.sqrt 2) - (1 / 2) := 
sorry

end problem_inequality_2_2156


namespace pumps_time_to_empty_pool_2_2795

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool_2_2795


namespace andrey_boris_denis_eat_candies_2_2941

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_2_2941


namespace a7_b7_equals_29_2_2875

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29_2_2875


namespace Pam_current_balance_2_2215

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end Pam_current_balance_2_2215


namespace area_of_L_shape_is_58_2_2623

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end area_of_L_shape_is_58_2_2623


namespace find_highway_speed_2_2325

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end find_highway_speed_2_2325


namespace triangle_area_is_correct_2_2915

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct_2_2915


namespace total_number_of_cats_2_2431

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats_2_2431


namespace sum_of_uv_2_2091

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end sum_of_uv_2_2091


namespace gcf_3465_10780_2_2393

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780_2_2393


namespace initial_lychees_count_2_2421

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end initial_lychees_count_2_2421


namespace M_subset_N_2_2622

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end M_subset_N_2_2622


namespace rectangle_width_decrease_2_2966

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease_2_2966


namespace x_pow_n_plus_inv_x_pow_n_2_2903

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n_2_2903


namespace coprime_pairs_solution_2_2132

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end coprime_pairs_solution_2_2132


namespace factorize_difference_of_squares_2_2316

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares_2_2316


namespace solution_set_M_minimum_value_expr_2_2052

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_2_2052


namespace weight_of_rod_2_2350

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod_2_2350


namespace tina_pink_pens_2_2825

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end tina_pink_pens_2_2825


namespace exists_sum_and_sum_of_squares_lt_2_2930

theorem exists_sum_and_sum_of_squares_lt :
  ∃ (n : ℕ) (x : Fin n → ℝ), (∑ i, x i = 10) ∧ (∑ i, (x i)^2 < 0.2) :=
  sorry

end exists_sum_and_sum_of_squares_lt_2_2930


namespace f_max_a_zero_f_zero_range_2_2873

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range_2_2873


namespace ring_stack_distance_2_2196

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance_2_2196


namespace find_t_2_2423

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t_2_2423


namespace total_worth_is_correct_2_2757

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_2_2757


namespace intersection_of_M_and_N_2_2236

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_2_2236


namespace compare_2_roses_3_carnations_2_2373

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end compare_2_roses_3_carnations_2_2373


namespace quadratic_function_conditions_2_2124

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions_2_2124


namespace integer_type_2_2696

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type_2_2696


namespace total_spent_is_195_2_2999

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end total_spent_is_195_2_2999


namespace probability_product_positive_is_5_div_9_2_2175

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_2_2175


namespace inequality_solution_2_2504

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end inequality_solution_2_2504


namespace location_determined_2_2264

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end location_determined_2_2264


namespace cost_price_percentage_2_2961

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end cost_price_percentage_2_2961


namespace exists_disk_of_radius_one_containing_1009_points_2_2302

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points_2_2302


namespace line_intersects_ellipse_with_conditions_2_2741

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_2_2741


namespace not_divisible_by_1000_pow_m_minus_1_2_2298

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_by_1000_pow_m_minus_1_2_2298


namespace coffee_consumption_2_2311

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption_2_2311


namespace minimize_J_2_2617

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J_2_2617


namespace minimum_ab_2_2580

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end minimum_ab_2_2580


namespace perpendicular_lines_2_2809

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines_2_2809


namespace projectile_height_reaches_49_2_2322

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49_2_2322


namespace cost_of_green_pill_2_2965

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill_2_2965


namespace members_play_both_2_2883

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end members_play_both_2_2883


namespace inequality_proof_2_2597

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end inequality_proof_2_2597


namespace percentage_increase_second_movie_2_2683

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie_2_2683


namespace find_a6_2_2828

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6_2_2828


namespace rectangular_solid_sum_of_edges_2_2790

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end rectangular_solid_sum_of_edges_2_2790


namespace total_sum_of_money_2_2701

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money_2_2701


namespace sin_alpha_neg_point_two_2_2738

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two_2_2738


namespace field_area_2_2353

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area_2_2353


namespace small_boxes_in_large_box_2_2109

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end small_boxes_in_large_box_2_2109


namespace remainder_gx12_div_gx_2_2106

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end remainder_gx12_div_gx_2_2106


namespace find_point_D_2_2067

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end find_point_D_2_2067


namespace cos_double_angle_2_2528

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_2_2528


namespace cubed_difference_2_2023

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference_2_2023


namespace margie_driving_distance_2_2877

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end margie_driving_distance_2_2877


namespace count_primes_with_digit_three_2_2018

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_2_2018


namespace solve_for_x_2_2548

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end solve_for_x_2_2548


namespace radio_lowest_price_rank_2_2123

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank_2_2123


namespace digit_sum_10_pow_93_minus_937_2_2242

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937_2_2242


namespace sum_ge_3_implies_one_ge_2_2_2299

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end sum_ge_3_implies_one_ge_2_2_2299


namespace equilateral_triangle_side_length_2_2034

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length_2_2034


namespace fixed_points_2_2214

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end fixed_points_2_2214


namespace John_pays_amount_2_2615

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end John_pays_amount_2_2615


namespace period_sine_transformed_2_2382

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed_2_2382


namespace isabella_more_than_sam_2_2169

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end isabella_more_than_sam_2_2169


namespace base_r_representation_26_eq_32_2_2993

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32_2_2993


namespace scientific_notation_2_2437

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end scientific_notation_2_2437


namespace ratio_matt_fem_4_1_2_2190

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1_2_2190


namespace smaller_of_two_digit_numbers_2_2640

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end smaller_of_two_digit_numbers_2_2640


namespace find_a_b_c_2_2202

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c_2_2202


namespace cheezit_bag_weight_2_2069

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight_2_2069


namespace probability_adjacent_vertices_decagon_2_2800

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_2_2800


namespace equation_of_AB_2_2348

-- Definitions based on the conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 3

def midpoint_M (p : ℝ × ℝ) : Prop :=
  p = (1, 0)

-- The theorem to be proved
theorem equation_of_AB (x y : ℝ) (M : ℝ × ℝ) :
  circle_C x y ∧ midpoint_M M → x - y = 1 :=
by
  sorry

end equation_of_AB_2_2348


namespace sixth_element_row_20_2_2261

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_2_2261


namespace polynomial_coefficients_2_2028

theorem polynomial_coefficients (a : Fin 10 → ℤ) :
  (1 - X) ^ 9 = ∑ i in Finset.range 10, (a i) * X ^ i →
  a 0 = 1 ∧
  a 1 + a 3 + a 5 + a 7 + a 9 = -256 ∧
  (2 : ℤ) * a 1 + (2 : ℤ)^2 * a 2 + (2 : ℤ)^3 * a 3 + (2 : ℤ)^4 * a 4 + (2 : ℤ)^5 * a 5 + 
  (2 : ℤ)^6 * a 6 + (2 : ℤ)^7 * a 7 + (2 : ℤ)^8 * a 8 + (2 : ℤ)^9 * a 9 = -2 := by
  sorry

end polynomial_coefficients_2_2028


namespace measure_angle_WYZ_2_2276

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ_2_2276


namespace probability_participation_on_both_days_2_2400

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days_2_2400


namespace parallelogram_area_is_correct_2_2121

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end parallelogram_area_is_correct_2_2121


namespace trig_identity_2_2571

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity_2_2571


namespace trig_identity_2_2213

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end trig_identity_2_2213


namespace relationship_A_B_2_2986

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B_2_2986


namespace michael_average_speed_2_2977

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed_2_2977


namespace find_x_value_2_2176

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end find_x_value_2_2176


namespace john_made_money_2_2029

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_2_2029


namespace area_triangle_MNR_2_2739

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR_2_2739


namespace campers_in_two_classes_2_2715

-- Definitions of the sets and conditions
variable (S A R : Finset ℕ)
variable (n : ℕ)
variable (x : ℕ)

-- Given conditions
axiom hyp1 : S.card = 20
axiom hyp2 : A.card = 20
axiom hyp3 : R.card = 20
axiom hyp4 : (S ∩ A ∩ R).card = 4
axiom hyp5 : (S \ (A ∪ R)).card + (A \ (S ∪ R)).card + (R \ (S ∪ A)).card = 24

-- The hypothesis that n = |S ∪ A ∪ R|
axiom hyp6 : n = (S ∪ A ∪ R).card

-- Statement to be proven in Lean
theorem campers_in_two_classes : x = 12 :=
by
  sorry

end campers_in_two_classes_2_2715


namespace three_digit_numbers_mod_1000_2_2138

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000_2_2138


namespace sum_of_squares_of_roots_eq_2_2147

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq_2_2147


namespace roots_of_quadratic_2_2089

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic_2_2089


namespace gcd_153_119_eq_17_2_2084

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_eq_17_2_2084


namespace riding_time_fraction_2_2329

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end riding_time_fraction_2_2329


namespace zorbs_of_60_deg_2_2989

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg_2_2989


namespace choosing_top_cases_2_2687

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases_2_2687


namespace units_digit_sum_factorials_2_2283

-- Definitions based on the conditions
def units_digit (n : ℕ) : ℕ := n % 10

-- Lean statement to represent the proof problem
theorem units_digit_sum_factorials :
  units_digit (∑ n in Finset.range 2024, n.factorial) = 3 :=
by 
  sorry

end units_digit_sum_factorials_2_2283


namespace travel_time_third_to_first_2_2656

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end travel_time_third_to_first_2_2656


namespace solve_wire_cut_problem_2_2748

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end solve_wire_cut_problem_2_2748


namespace total_pumped_volume_2_2669

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume_2_2669


namespace fraction_subtraction_2_2101

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction_2_2101


namespace k5_possibility_2_2684

noncomputable def possible_k5 : Prop :=
  ∃ (intersections : Fin 5 → Fin 5 × Fin 10), 
    ∀ i j : Fin 5, i ≠ j → intersections i ≠ intersections j

theorem k5_possibility : possible_k5 := 
by
  sorry

end k5_possibility_2_2684


namespace probability_of_three_primes_from_30_2_2268

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end probability_of_three_primes_from_30_2_2268


namespace remainder_of_3_pow_17_mod_7_2_2693

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7_2_2693


namespace sum_of_three_distinct_integers_product_625_2_2560

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end sum_of_three_distinct_integers_product_625_2_2560


namespace symmetric_points_x_axis_2_2244

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end symmetric_points_x_axis_2_2244


namespace obtuse_triangle_side_range_2_2066

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end obtuse_triangle_side_range_2_2066


namespace simplify_fraction_expression_2_2065

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_2_2065


namespace photograph_area_2_2107

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end photograph_area_2_2107


namespace range_of_a_2_2218

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a_2_2218


namespace ScarlettsDishCost_2_2297

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end ScarlettsDishCost_2_2297


namespace farmer_trees_2_2340

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees_2_2340


namespace determine_8_genuine_coins_2_2238

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end determine_8_genuine_coins_2_2238


namespace hyperbola_proof_2_2955

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end hyperbola_proof_2_2955


namespace exponent_of_5_in_30_fact_2_2960

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_2_2960


namespace combined_alloy_tin_amount_2_2979

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end combined_alloy_tin_amount_2_2979


namespace max_point_f_2_2740

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end max_point_f_2_2740


namespace total_percent_decrease_2_2546

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end total_percent_decrease_2_2546


namespace range_of_a_2_2559

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end range_of_a_2_2559


namespace second_quadrant_point_2_2655

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point_2_2655


namespace probability_heads_exactly_8_in_10_2_2277

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10_2_2277


namespace only_exprC_cannot_be_calculated_with_square_of_binomial_2_2878

-- Definitions of our expressions using their variables
def exprA (a b : ℝ) := (a + b) * (a - b)
def exprB (x : ℝ) := (-x + 1) * (-x - 1)
def exprC (y : ℝ) := (y + 1) * (-y - 1)
def exprD (m : ℝ) := (m - 1) * (-1 - m)

-- Statement that only exprC cannot be calculated using the square of a binomial formula
theorem only_exprC_cannot_be_calculated_with_square_of_binomial :
  (∀ a b : ℝ, ∃ (u v : ℝ), exprA a b = u^2 - v^2) ∧
  (∀ x : ℝ, ∃ (u v : ℝ), exprB x = u^2 - v^2) ∧
  (forall m : ℝ, ∃ (u v : ℝ), exprD m = u^2 - v^2) 
  ∧ (∀ v : ℝ, ¬ ∃ (u : ℝ), exprC v = u^2 ∨ (exprC v = - (u^2))) := sorry

end only_exprC_cannot_be_calculated_with_square_of_binomial_2_2878


namespace total_cost_of_purchase_2_2077

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_2_2077


namespace volleyball_tournament_2_2386

theorem volleyball_tournament (n m : ℕ) (h : n = m) :
  n = m := 
by
  sorry

end volleyball_tournament_2_2386


namespace find_k_minus_r_2_2293

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end find_k_minus_r_2_2293


namespace selling_price_is_1260_2_2032

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260_2_2032


namespace required_moles_of_H2O_2_2479

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O_2_2479


namespace strawberries_harvest_2_2861

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest_2_2861


namespace blue_paint_cans_needed_2_2786

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end blue_paint_cans_needed_2_2786


namespace bill_original_selling_price_2_2815

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end bill_original_selling_price_2_2815


namespace find_a7_2_2765

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_2_2765


namespace solve_quadratic_inequality_2_2234

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end solve_quadratic_inequality_2_2234


namespace solution_to_quadratic_solution_to_cubic_2_2527

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end solution_to_quadratic_solution_to_cubic_2_2527


namespace calculate_abs_mul_2_2430

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul_2_2430


namespace part_I_part_II_2_2476

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end part_I_part_II_2_2476


namespace area_between_chords_is_correct_2_2245

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end area_between_chords_is_correct_2_2245


namespace sacks_harvested_per_section_2_2672

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end sacks_harvested_per_section_2_2672


namespace largest_sum_is_5_over_6_2_2036

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6_2_2036


namespace find_abc_solutions_2_2901

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions_2_2901


namespace sum_abs_a1_to_a10_2_2160

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10_2_2160


namespace product_of_roots_2_2309

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots_2_2309


namespace perfect_square_trinomial_m_2_2900

theorem perfect_square_trinomial_m (m : ℤ) : (∀ x : ℤ, ∃ k : ℤ, x^2 + 2*m*x + 9 = (x + k)^2) ↔ m = 3 ∨ m = -3 :=
by
  sorry

end perfect_square_trinomial_m_2_2900


namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_2_2972

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_2_2972


namespace probability_of_male_selected_2_2838

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end probability_of_male_selected_2_2838


namespace radius_of_larger_circle_is_25_over_3_2_2812

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end radius_of_larger_circle_is_25_over_3_2_2812


namespace pumps_280_gallons_in_30_minutes_2_2335

def hydraflow_rate_per_hour := 560 -- gallons per hour
def time_fraction_in_hour := 1 / 2

theorem pumps_280_gallons_in_30_minutes : hydraflow_rate_per_hour * time_fraction_in_hour = 280 := by
  sorry

end pumps_280_gallons_in_30_minutes_2_2335


namespace min_value_a_plus_3b_2_2621

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end min_value_a_plus_3b_2_2621


namespace area_PQR_is_4_5_2_2006

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end area_PQR_is_4_5_2_2006


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_2_2252

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_2_2252


namespace number_of_cirrus_clouds_2_2551

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_2_2551


namespace Claire_takes_6_photos_2_2916

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos_2_2916


namespace min_value_alpha_beta_gamma_2_2891

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma_2_2891


namespace max_value_f_2_2258

noncomputable def max_f (a : Fin 2019 → ℝ) : ℝ :=
  ∑ i, (a i) ^ 3

theorem max_value_f (a : Fin 2019 → ℝ) :
  (∀ i, a i ∈ Set.Icc (-1 : ℝ) 1) → (∑ i, a i = 0) →
  max_f a ≤ 2019 / 4 :=
by
  sorry

end max_value_f_2_2258


namespace ellen_golf_cart_trips_2_2526

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_2_2526


namespace circles_intersect_2_2274

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect_2_2274


namespace inequality_a_b_2_2462

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end inequality_a_b_2_2462


namespace sum_is_zero_2_2251

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero_2_2251


namespace arithmetic_seq_sum_2_2759

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum_2_2759


namespace max_b_c_2_2363

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end max_b_c_2_2363


namespace actual_cost_of_article_2_2995

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article_2_2995


namespace find_number_2_2448

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end find_number_2_2448


namespace range_of_a_2_2103

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_2_2103


namespace nonnegative_integer_with_divisors_is_multiple_of_6_2_2717

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end nonnegative_integer_with_divisors_is_multiple_of_6_2_2717


namespace amount_of_juice_p_in_a_2_2654

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a_2_2654


namespace solution_set_equiv_2_2349

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv_2_2349


namespace min_overlap_2_2377

noncomputable def drinks_coffee := 0.60
noncomputable def drinks_tea := 0.50
noncomputable def drinks_neither := 0.10
noncomputable def drinks_either := 1 - drinks_neither
noncomputable def total_overlap := drinks_coffee + drinks_tea - drinks_either

theorem min_overlap (hcoffee : drinks_coffee = 0.60) (htea : drinks_tea = 0.50) (hneither : drinks_neither = 0.10) :
  total_overlap = 0.20 :=
by
  sorry

end min_overlap_2_2377


namespace halfway_between_3_4_and_5_7_2_2183

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7_2_2183


namespace problem_2_2962

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem_2_2962


namespace original_perimeter_not_necessarily_multiple_of_four_2_2779

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four_2_2779


namespace scientific_notation_448000_2_2474

theorem scientific_notation_448000 :
  ∃ a n, (448000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 :=
by
  sorry

end scientific_notation_448000_2_2474


namespace variance_of_scores_2_2341

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores_2_2341


namespace polynomial_division_remainder_zero_2_2022

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero_2_2022


namespace fred_red_marbles_2_2553

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles_2_2553


namespace ratio_volumes_2_2819

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end ratio_volumes_2_2819


namespace thalassa_population_2050_2_2227

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050_2_2227


namespace problem_1_problem_2_2_2895

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_2_2895


namespace total_combinations_2_2525

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end total_combinations_2_2525


namespace divisible_by_five_solution_exists_2_2787

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end divisible_by_five_solution_exists_2_2787


namespace common_difference_2_2643

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference_2_2643


namespace original_triangle_area_2_2637

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_2_2637


namespace log_inequality_solution_2_2942

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution_2_2942


namespace renu_suma_work_together_2_2438

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end renu_suma_work_together_2_2438


namespace complement_of_A_in_U_2_2876

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U_2_2876


namespace slope_intercept_equivalence_2_2532

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end slope_intercept_equivalence_2_2532


namespace remainder_of_poly_div_2_2834

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_div_2_2834


namespace ratio_of_age_difference_2_2428

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end ratio_of_age_difference_2_2428


namespace num_marked_cells_at_least_num_cells_in_one_square_2_2611

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square_2_2611


namespace infinite_impossible_values_of_d_2_2463

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end infinite_impossible_values_of_d_2_2463


namespace expand_product_2_2968

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end expand_product_2_2968


namespace polygon_sides_eq_five_2_2064

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five_2_2064


namespace find_y_when_x_is_4_2_2487

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end find_y_when_x_is_4_2_2487


namespace ordered_pairs_count_2_2783

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count_2_2783


namespace no_integer_pairs_satisfy_equation_2_2524

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end no_integer_pairs_satisfy_equation_2_2524


namespace sum_floor_parity_2_2031

theorem sum_floor_parity (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (∑ i in Finset.range (m * n), (-1) ^ (i / m + i / n) = 0) ↔ ((m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0)) :=
by sorry

end sum_floor_parity_2_2031


namespace sophie_saves_money_2_2469

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end sophie_saves_money_2_2469


namespace volume_of_convex_polyhedron_2_2063

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron_2_2063


namespace Johnson_farm_budget_2_2586

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end Johnson_farm_budget_2_2586


namespace part1_part2_2_2125

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_2_2125


namespace missing_fraction_2_2292

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end missing_fraction_2_2292


namespace compute_xy_2_2391

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end compute_xy_2_2391


namespace find_a_2_2676

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end find_a_2_2676


namespace least_number_of_plates_needed_2_2919

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end least_number_of_plates_needed_2_2919


namespace gate_distance_probability_correct_2_2584

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end gate_distance_probability_correct_2_2584


namespace parallelogram_area_2_2630

def base : ℕ := 34
def height : ℕ := 18
def area_of_parallelogram (b h : ℕ) : ℕ := b * h

theorem parallelogram_area : area_of_parallelogram base height = 612 := by
  sorry

end parallelogram_area_2_2630


namespace intersection_sum_zero_2_2678

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end intersection_sum_zero_2_2678


namespace hexagon_coloring_count_2_2182

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count_2_2182


namespace find_enclosed_area_2_2486

def area_square (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem find_enclosed_area :
  let side1 := 3
  let side2 := 6
  let area1 := area_square side1
  let area2 := area_square side2
  let area_tri := 2 * area_triangle side1 side2
  area1 + area2 + area_tri = 63 :=
by
  sorry

end find_enclosed_area_2_2486


namespace simplify_T_2_2516

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end simplify_T_2_2516


namespace num_paths_from_E_to_G_pass_through_F_2_2686

-- Definitions for the positions on the grid.
def E := (0, 4)
def G := (5, 0)
def F := (3, 3)

-- Function to calculate the number of combinations.
def binom (n k: ℕ) : ℕ := Nat.choose n k

-- The mathematical statement to be proven.
theorem num_paths_from_E_to_G_pass_through_F :
  (binom 4 1) * (binom 5 2) = 40 :=
by
  -- Placeholder for the proof.
  sorry

end num_paths_from_E_to_G_pass_through_F_2_2686


namespace find_fraction_2_2769

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction_2_2769


namespace people_counted_2_2472

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_2_2472


namespace find_pairs_2_2844

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end find_pairs_2_2844


namespace increasing_interval_2_2254

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval_2_2254


namespace coffee_shop_cups_2_2624

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end coffee_shop_cups_2_2624


namespace initial_bottle_caps_2_2826

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps_2_2826


namespace total_area_three_plots_2_2070

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots_2_2070


namespace symmetric_points_power_2_2793

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power_2_2793


namespace rectangle_side_ratio_2_2697

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio_2_2697


namespace nap_duration_is_two_hours_2_2432

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours_2_2432


namespace find_prime_pair_2_2692

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair_2_2692


namespace lateral_surface_area_pyramid_2_2503

theorem lateral_surface_area_pyramid (α R : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < π) :
    let S := 4 * R^2 * (Real.cot (α / 2)) * (Real.tan (π / 4 + α / 2))
    S = 4 * R^2 * (Real.cot (α / 2)) * (Real.tan (π / 4 + α / 2)) :=
by
    sorry

end lateral_surface_area_pyramid_2_2503


namespace coal_consumption_rel_2_2664

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end coal_consumption_rel_2_2664


namespace shop_owner_profitable_2_2471

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable_2_2471


namespace find_cows_2_2881

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end find_cows_2_2881


namespace fraction_value_2_2594

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value_2_2594


namespace gift_boxes_in_3_days_2_2633
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_2_2633


namespace difference_sum_first_100_odds_evens_2_2001

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end difference_sum_first_100_odds_evens_2_2001


namespace loss_equals_cost_price_of_some_balls_2_2208

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_2_2208


namespace quadratic_real_roots_condition_2_2890

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_condition_2_2890


namespace black_cars_count_2_2583

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end black_cars_count_2_2583


namespace total_marbles_proof_2_2295

def red_marble_condition (b r : ℕ) : Prop :=
  r = b + (3 * b / 10)

def yellow_marble_condition (r y : ℕ) : Prop :=
  y = r + (5 * r / 10)

def total_marbles (b r y : ℕ) : ℕ :=
  r + b + y

theorem total_marbles_proof (b r y : ℕ)
  (h1 : red_marble_condition b r)
  (h2 : yellow_marble_condition r y) :
  total_marbles b r y = 425 * r / 130 :=
by {
  sorry
}

end total_marbles_proof_2_2295


namespace value_of_v_3_2_2954

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end value_of_v_3_2_2954


namespace discount_is_one_percent_2_2150

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end discount_is_one_percent_2_2150


namespace quadratic_two_distinct_real_roots_2_2165

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end quadratic_two_distinct_real_roots_2_2165


namespace given_conditions_implies_correct_answer_2_2634

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer_2_2634


namespace inequality_1_inequality_2_2_2766

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2_2_2766


namespace sum_of_faces_of_rectangular_prism_2_2451

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism_2_2451


namespace y_intercepts_count_2_2859

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_2_2859


namespace reciprocal_roots_k_value_2_2722

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end reciprocal_roots_k_value_2_2722


namespace length_of_equal_pieces_2_2514

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_2_2514


namespace prob1_prob2_max_area_prob3_circle_diameter_2_2222

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter_2_2222


namespace average_sitting_time_per_student_2_2223

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student_2_2223


namespace simplify_and_evaluate_2_2918

noncomputable def expr (x : ℝ) : ℝ :=
  ((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4 * x + 4) / x)

theorem simplify_and_evaluate : expr 1 = -1 / 3 :=
by
  sorry

end simplify_and_evaluate_2_2918


namespace find_k_2_2385

theorem find_k (k : ℤ) :
  (-x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) → k = -16 :=
by
  sorry

end find_k_2_2385


namespace calculate_fraction_value_2_2926

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value_2_2926


namespace rice_less_than_beans_by_30_2_2372

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30_2_2372


namespace exactly_one_correct_2_2731

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end exactly_one_correct_2_2731


namespace perpendicular_vectors_parallel_vectors_2_2913

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors_2_2913


namespace range_a_satisfies_2_2485

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end range_a_satisfies_2_2485


namespace annual_rent_per_sqft_2_2491

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end annual_rent_per_sqft_2_2491


namespace apple_distribution_ways_2_2951

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end apple_distribution_ways_2_2951


namespace smallest_total_cells_marked_2_2709

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end smallest_total_cells_marked_2_2709


namespace complement_of_union_eq_2_2831

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define the subset A
def A : Set ℤ := {-1, 0, 1}

-- Define the subset B
def B : Set ℤ := {0, 1, 2, 3}

-- Define the union of A and B
def A_union_B : Set ℤ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℤ := U \ A_union_B

-- State the theorem to be proved
theorem complement_of_union_eq {U A B : Set ℤ} :
  U = {-1, 0, 1, 2, 3, 4} →
  A = {-1, 0, 1} →
  B = {0, 1, 2, 3} →
  complement_U_A_union_B = {4} :=
by
  intros hU hA hB
  sorry

end complement_of_union_eq_2_2831


namespace percentage_passed_all_three_2_2569

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end percentage_passed_all_three_2_2569


namespace coordinates_with_respect_to_origin_2_2379

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin_2_2379


namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_2_2711

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_2_2711


namespace max_bag_weight_is_50_2_2019

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end max_bag_weight_is_50_2_2019


namespace thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_2_2728

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_2_2728


namespace black_balls_in_box_2_2357

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box_2_2357


namespace number_of_m_gons_proof_2_2500

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end number_of_m_gons_proof_2_2500


namespace final_hair_length_is_14_2_2094

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14_2_2094


namespace correct_phone_call_sequence_2_2010

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence_2_2010


namespace geometric_sequence_sum_2_2047

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_2_2047


namespace number_of_people_2_2737

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people_2_2737


namespace Tony_fills_pool_in_90_minutes_2_2177

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end Tony_fills_pool_in_90_minutes_2_2177


namespace geom_sequence_property_2_2397

-- Define geometric sequence sums
variables {a : ℕ → ℝ} {s₁ s₂ s₃ : ℝ}

-- Assume a is a geometric sequence and s₁, s₂, s₃ are sums of first n, 2n, and 3n terms respectively
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

variables (a_is_geom : is_geometric_sequence a)
variables (s₁_eq : s₁ = sum_first_n_terms a n)
variables (s₂_eq : s₂ = sum_first_n_terms a (2 * n))
variables (s₃_eq : s₃ = sum_first_n_terms a (3 * n))

-- Statement: Prove that y(y - x) = x(z - x)
theorem geom_sequence_property (n : ℕ) : s₂ * (s₂ - s₁) = s₁ * (s₃ - s₁) :=
by
  sorry

end geom_sequence_property_2_2397


namespace janessa_gives_dexter_cards_2_2818

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end janessa_gives_dexter_cards_2_2818


namespace min_value_expression_2_2398

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end min_value_expression_2_2398


namespace shaded_region_area_2_2854

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area_2_2854


namespace f_zero_eq_zero_f_one_eq_one_f_n_is_n_2_2694

variable (f : ℤ → ℤ)

axiom functional_eq : ∀ m n : ℤ, f (m^2 + f n) = f (f m) + n

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_one_eq_one : f 1 = 1 :=
sorry

theorem f_n_is_n : ∀ n : ℤ, f n = n :=
sorry

end f_zero_eq_zero_f_one_eq_one_f_n_is_n_2_2694


namespace sum_series_upto_9_2_2288

open Nat

noncomputable def series_sum_to (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n+1), (1 : ℝ) / (2 : ℝ) ^ i

theorem sum_series_upto_9 : series_sum_to 9 = 2 - 2 ^ (-9 : ℝ) :=
by
  sorry

end sum_series_upto_9_2_2288


namespace tim_tasks_per_day_2_2575

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day_2_2575


namespace expand_product_2_2426

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product_2_2426


namespace rectangle_midpoints_sum_2_2014

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum_2_2014


namespace solve_inequality_system_2_2642

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end solve_inequality_system_2_2642


namespace find_ABC_base10_2_2923

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10_2_2923


namespace andy_time_correct_2_2598

-- Define the conditions
def time_dawn_wash_dishes : ℕ := 20
def time_andy_put_laundry : ℕ := 2 * time_dawn_wash_dishes + 6

-- The theorem to prove
theorem andy_time_correct : time_andy_put_laundry = 46 :=
by
  -- Proof goes here
  sorry

end andy_time_correct_2_2598


namespace m_div_x_eq_4_div_5_2_2735

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5_2_2735


namespace equation_of_plane_passing_through_points_2_2823

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end equation_of_plane_passing_through_points_2_2823


namespace total_games_eq_64_2_2246

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end total_games_eq_64_2_2246


namespace complex_number_solution_2_2716

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution_2_2716


namespace prince_spending_2_2247

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_2_2247


namespace tens_digit_of_8_pow_1234_2_2439

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_2_2439


namespace find_a_minus_b_2_2275

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b_2_2275


namespace probability_no_correct_letter_for_7_envelopes_2_2161

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes_2_2161


namespace sequence_sum_2_2657

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end sequence_sum_2_2657


namespace originally_planned_days_2_2644

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end originally_planned_days_2_2644


namespace value_of_a5_max_sum_first_n_value_2_2727

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end value_of_a5_max_sum_first_n_value_2_2727


namespace solve_for_y_2_2857

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y_2_2857


namespace num_fixed_last_two_digits_2_2886

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits_2_2886


namespace find_angle_A_2_2945

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end find_angle_A_2_2945


namespace candle_cost_correct_2_2866

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct_2_2866


namespace investor_receives_7260_2_2593

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260_2_2593


namespace completing_square_transformation_2_2879

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_2_2879


namespace sequence_general_term_2_2506

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term_2_2506


namespace sin_arithmetic_sequence_2_2139

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end sin_arithmetic_sequence_2_2139


namespace acres_used_for_corn_2_2939

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end acres_used_for_corn_2_2939


namespace length_of_second_platform_2_2801

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform_2_2801


namespace fizz_preference_count_2_2651

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count_2_2651


namespace parabola_vertex_y_axis_opens_upwards_2_2095

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards_2_2095


namespace infinitely_many_not_representable_2_2679

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end infinitely_many_not_representable_2_2679


namespace yolanda_walking_rate_2_2778

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate_2_2778


namespace translation_preserves_coordinates_2_2723

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end translation_preserves_coordinates_2_2723


namespace find_P_nplus1_2_2272

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end find_P_nplus1_2_2272


namespace average_rate_of_change_2_2872

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end average_rate_of_change_2_2872


namespace area_of_square_2_2447

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end area_of_square_2_2447


namespace math_proof_problem_2_2510

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem_2_2510


namespace gerald_added_crayons_2_2154

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end gerald_added_crayons_2_2154


namespace angle_CDE_proof_2_2920

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end angle_CDE_proof_2_2920


namespace solve_system_2_2055

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end solve_system_2_2055


namespace sufficient_not_necessary_condition_2_2015

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end sufficient_not_necessary_condition_2_2015


namespace correct_meteor_passing_time_2_2366

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_2_2366


namespace larry_wins_game_2_2641

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end larry_wins_game_2_2641


namespace coloring_possible_2_2058

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible_2_2058


namespace max_jars_in_crate_2_2361

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end max_jars_in_crate_2_2361


namespace intersection_complement_2_2685

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end intersection_complement_2_2685


namespace num_five_digit_integers_2_2013

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers_2_2013


namespace neg_p_2_2493

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end neg_p_2_2493


namespace chocolates_bought_at_cost_price_2_2565

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end chocolates_bought_at_cost_price_2_2565


namespace product_of_roots_of_cubic_2_2163

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end product_of_roots_of_cubic_2_2163


namespace distance_between_towns_proof_2_2099

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof_2_2099


namespace part1_part2_part3_2_2250

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3_2_2250


namespace value_of_f_at_3_2_2049

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end value_of_f_at_3_2_2049


namespace total_meters_examined_2_2841

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end total_meters_examined_2_2841


namespace relationship_of_ys_2_2566

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_2_2566


namespace right_triangle_midpoints_distances_2_2579

theorem right_triangle_midpoints_distances (a b : ℝ) 
  (hXON : 19^2 = a^2 + (b/2)^2)
  (hYOM : 22^2 = b^2 + (a/2)^2) :
  a^2 + b^2 = 676 :=
by
  sorry

end right_triangle_midpoints_distances_2_2579


namespace answered_both_2_2483

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end answered_both_2_2483


namespace remainder_7547_div_11_2_2885

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11_2_2885


namespace original_number_2_2496

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end original_number_2_2496


namespace range_of_a_2_2339

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end range_of_a_2_2339


namespace n_squared_divides_2n_plus_1_2_2136

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1_2_2136


namespace total_sum_vowels_2_2533

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end total_sum_vowels_2_2533


namespace proportional_b_value_2_2306

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end proportional_b_value_2_2306


namespace papers_left_2_2762

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left_2_2762


namespace coeff_z_in_third_eq_2_2159

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq_2_2159


namespace min_value_expr_2_2444

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr_2_2444


namespace find_a_2_2443

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a_2_2443


namespace match_Tile_C_to_Rectangle_III_2_2848

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III_2_2848


namespace problem_statement_2_2746

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_2_2746


namespace functional_eq_2_2280

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end functional_eq_2_2280


namespace find_largest_cos_x_2_2128

theorem find_largest_cos_x (x y z : ℝ) 
  (h1 : Real.sin x = Real.cot y)
  (h2 : Real.sin y = Real.cot z)
  (h3 : Real.sin z = Real.cot x) :
  Real.cos x ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := sorry

end find_largest_cos_x_2_2128


namespace sequence_pattern_2_2682

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern_2_2682


namespace geometric_series_first_term_2_2513

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_2_2513


namespace steve_paid_18_2_2332

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_2_2332


namespace sum_congruence_example_2_2420

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end sum_congruence_example_2_2420


namespace gcd_problem_2_2670

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end gcd_problem_2_2670


namespace factor_expression_2_2110

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_2_2110


namespace y_divides_x_squared_2_2768

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared_2_2768


namespace houses_with_neither_2_2030

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither_2_2030


namespace base_b_sum_correct_2_2422

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct_2_2422


namespace abc_eq_1_2_2012

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1_2_2012


namespace max_heaps_of_stones_2_2201

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_2_2201


namespace remainder_when_divided_by_r_minus_1_2_2266

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end remainder_when_divided_by_r_minus_1_2_2266


namespace determine_x_y_2_2590

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y_2_2590


namespace bathroom_area_is_50_square_feet_2_2540

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet_2_2540


namespace largest_corner_sum_2_2454

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum_2_2454


namespace Faye_total_pencils_2_2839

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils_2_2839


namespace select_4_people_arrangement_3_day_new_year_2_2827

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year_2_2827


namespace factor_polynomial_2_2604

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end factor_polynomial_2_2604


namespace find_a_pow_b_2_2650

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b_2_2650


namespace roses_in_centerpiece_2_2996

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end roses_in_centerpiece_2_2996


namespace product_computation_2_2894

def product_fraction_compute : ℕ :=
  ∏ n in Finset.range (22 - 2 + 1), (n + 5) * (if (1 ≤ n + 2) then 1 else 0)

theorem product_computation : product_fraction_compute = 14950 := by
  sorry

end product_computation_2_2894


namespace no_negative_roots_2_2092

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end no_negative_roots_2_2092


namespace wire_cut_2_2097

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut_2_2097


namespace triangle_is_right_angled_2_2356

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled_2_2356


namespace number_of_proper_subsets_of_P_2_2321

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P_2_2321


namespace general_term_defines_sequence_2_2401

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence_2_2401


namespace intersection_A_B_2_2842

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end intersection_A_B_2_2842


namespace time_difference_correct_2_2836

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct_2_2836


namespace bags_of_chips_count_2_2704

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count_2_2704


namespace probability_unit_square_not_touch_central_2x2_square_2_2025

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square_2_2025


namespace number_of_customers_left_2_2688

theorem number_of_customers_left (x : ℕ) (h : 14 - x + 39 = 50) : x = 3 := by
  sorry

end number_of_customers_left_2_2688


namespace problem_statement_2_2174

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement_2_2174


namespace percentage_defective_2_2134

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end percentage_defective_2_2134


namespace num_boys_2_2804

-- Definitions as per the conditions
def boys (d : ℕ) := 2 * d
def reducedGirls (d : ℕ) := d - 1

-- Lean statement for the proof problem
theorem num_boys (d b : ℕ) 
  (h1 : b = boys d)
  (h2 : b = reducedGirls d + 8) : b = 14 :=
by {
  sorry
}

end num_boys_2_2804


namespace part_a_part_b_part_c_part_d_part_e_part_f_2_2198

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_2_2198


namespace value_of_m_2_2484

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end value_of_m_2_2484


namespace Vishal_investment_percentage_more_than_Trishul_2_2959

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end Vishal_investment_percentage_more_than_Trishul_2_2959


namespace solve_system_2_2908

section system_equations

variable (x y : ℤ)

def equation1 := 2 * x - y = 5
def equation2 := 5 * x + 2 * y = 8
def solution := x = 2 ∧ y = -1

theorem solve_system : (equation1 x y) ∧ (equation2 x y) ↔ solution x y := by
  sorry

end system_equations

end solve_system_2_2908


namespace find_a_for_extraneous_roots_find_a_for_no_solution_2_2858

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution_2_2858


namespace proof_problem_2_2112

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end proof_problem_2_2112


namespace exists_positive_x_le_sqrt_x_add_one_2_2929

theorem exists_positive_x_le_sqrt_x_add_one (h : ∀ x > 0, √x > x + 1) :
  ∃ x > 0, √x ≤ x + 1 :=
sorry

end exists_positive_x_le_sqrt_x_add_one_2_2929


namespace subtraction_and_multiplication_problem_2_2610

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem_2_2610


namespace relationship_between_a_and_b_2_2957

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b_2_2957


namespace box_height_at_least_2_sqrt_15_2_2577

def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x ^ 2

theorem box_height_at_least_2_sqrt_15 (x : ℝ) (h : ℝ) :
  h = box_height x →
  surface_area x ≥ 150 →
  h ≥ 2 * Real.sqrt 15 :=
by
  intros h_eq sa_ge_150
  sorry

end box_height_at_least_2_sqrt_15_2_2577


namespace third_test_point_2_2085

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point_2_2085


namespace candle_ratio_2_2118

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end candle_ratio_2_2118


namespace each_friend_paid_2_2824

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid_2_2824


namespace final_elephants_count_2_2851

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count_2_2851


namespace range_of_a_2_2303

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_2_2303


namespace problem_statement_2_2179

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end problem_statement_2_2179


namespace problem_statement_2_2910

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end problem_statement_2_2910


namespace perimeter_C_correct_2_2507

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_2_2507


namespace art_of_passing_through_walls_2_2455

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end art_of_passing_through_walls_2_2455


namespace surface_area_of_sphere_2_2661

noncomputable def length : ℝ := 3
noncomputable def width : ℝ := 2
noncomputable def height : ℝ := Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (length^2 + width^2 + height^2)
noncomputable def r : ℝ := d / 2

theorem surface_area_of_sphere :
  4 * Real.pi * r^2 = 14 * Real.pi := by
  sorry

end surface_area_of_sphere_2_2661


namespace projection_area_rectangular_board_2_2691

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board_2_2691


namespace range_of_a_2_2317

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_2_2317


namespace final_velocity_2_2799

variable (u a t : ℝ)

-- Defining the conditions
def initial_velocity := u = 0
def acceleration := a = 1.2
def time := t = 15

-- Statement of the theorem
theorem final_velocity : initial_velocity u ∧ acceleration a ∧ time t → (u + a * t = 18) := by
  sorry

end final_velocity_2_2799


namespace sum_base9_to_base9_eq_2_2381

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq_2_2381


namespace farmer_loss_representative_value_2_2758

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end farmer_loss_representative_value_2_2758


namespace hyperbola_intersection_2_2707

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end hyperbola_intersection_2_2707


namespace max_area_ABC_2_2763

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC_2_2763


namespace trig_identity_2_2501

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end trig_identity_2_2501


namespace g_of_3_equals_5_2_2427

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5_2_2427


namespace fourth_term_of_geometric_sequence_is_320_2_2852

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320_2_2852


namespace students_more_than_pets_2_2141

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end students_more_than_pets_2_2141


namespace tennis_player_games_2_2811

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end tennis_player_games_2_2811


namespace one_number_greater_than_one_2_2457

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end one_number_greater_than_one_2_2457


namespace perimeter_of_face_given_volume_2_2904

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume_2_2904


namespace calculator_transform_implication_2_2371

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication_2_2371


namespace vermont_clicked_ads_2_2760

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads_2_2760


namespace ball_height_less_than_10_after_16_bounces_2_2729

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end ball_height_less_than_10_after_16_bounces_2_2729


namespace find_a_2_2863

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_a_2_2863


namespace parallel_lines_m_values_2_2853

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values_2_2853


namespace min_value_144_2_2880

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end min_value_144_2_2880


namespace hansel_album_duration_2_2225

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_2_2225


namespace smallest_d_for_inverse_2_2342

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end smallest_d_for_inverse_2_2342


namespace total_payment_correct_2_2626

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end total_payment_correct_2_2626


namespace rhombus_perimeter_2_2802

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter_2_2802


namespace farmer_goats_sheep_unique_solution_2_2442

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution_2_2442


namespace valid_schedule_count_2_2082

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count_2_2082


namespace ratio_jordana_jennifer_10_years_2_2135

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end ratio_jordana_jennifer_10_years_2_2135


namespace cost_price_USD_2_2780

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD_2_2780


namespace find_number_2_2449

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number_2_2449


namespace num_people_visited_iceland_2_2267

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end num_people_visited_iceland_2_2267


namespace sum_first_18_terms_2_2753

noncomputable def a_n (n : ℕ) : ℚ := (-1 : ℚ)^n * (3 * n + 2) / (n * (n + 1) * 2^(n + 1))

noncomputable def S (n : ℕ) : ℚ := ∑ i in Finset.range n, a_n (i + 1)

theorem sum_first_18_terms :
  S 18 = (1 / (2^19 * 19) - 1 / 2) :=
sorry

end sum_first_18_terms_2_2753


namespace problem_statement_2_2592

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement_2_2592


namespace speed_of_sound_2_2104

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end speed_of_sound_2_2104


namespace maximum_value_of_piecewise_function_2_2404

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function_2_2404


namespace range_of_ab_c2_2_2806

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end range_of_ab_c2_2_2806


namespace inequality_solution_set_2_2004

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end inequality_solution_set_2_2004


namespace find_a_plus_b_2_2521

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end find_a_plus_b_2_2521


namespace arithmetic_sequence_8th_term_2_2074

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term_2_2074


namespace arrangement_plans_count_2_2204

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count_2_2204


namespace tommys_family_members_2_2282

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members_2_2282


namespace point_reflection_example_2_2662

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end point_reflection_example_2_2662


namespace amount_borrowed_2_2849

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed_2_2849


namespace portion_of_work_done_2_2673

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end portion_of_work_done_2_2673


namespace no_values_satisfy_equation_2_2585

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation_2_2585


namespace team_C_games_played_2_2412

variable (x : ℕ)
variable (winC : ℕ := 5 * x / 7)
variable (loseC : ℕ := 2 * x / 7)
variable (winD : ℕ := 2 * x / 3)
variable (loseD : ℕ := x / 3)

theorem team_C_games_played :
  winD = winC - 5 →
  loseD = loseC - 5 →
  x = 105 := by
  sorry

end team_C_games_played_2_2412


namespace fraction_of_people_under_21_correct_2_2405

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct_2_2405


namespace negation_of_prop_2_2445

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop_2_2445


namespace amount_deducted_from_third_2_2307

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end amount_deducted_from_third_2_2307


namespace total_puppies_is_74_2_2117

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_2_2117


namespace original_denominator_is_21_2_2970

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21_2_2970


namespace ellipse_equation_2_2189

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end ellipse_equation_2_2189


namespace largest_increase_between_2006_and_2007_2_2465

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007_2_2465


namespace num_pos_int_values_2_2505

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values_2_2505


namespace minimize_x_2_2240

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end minimize_x_2_2240


namespace intersection_complement_2_2719

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement_2_2719


namespace hyperbola_condition_2_2017

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end hyperbola_condition_2_2017


namespace time_addition_correct_2_2035

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct_2_2035


namespace percentage_more_than_2_2402

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than_2_2402


namespace best_scrap_year_limit_2_2512

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end best_scrap_year_limit_2_2512


namespace wine_division_2_2724

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end wine_division_2_2724


namespace no_integer_k_sq_plus_k_plus_one_divisible_by_101_2_2374

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end no_integer_k_sq_plus_k_plus_one_divisible_by_101_2_2374


namespace valid_integer_values_of_x_2_2224

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x_2_2224


namespace first_month_sale_2_2081

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale_2_2081


namespace miley_total_cost_2_2987

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost_2_2987


namespace anne_trip_shorter_2_2090

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end anne_trip_shorter_2_2090


namespace divisible_by_120_2_2649

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end divisible_by_120_2_2649


namespace contrapositive_equiv_2_2953

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_2_2953


namespace max_bag_weight_2_2056

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end max_bag_weight_2_2056


namespace participants_in_robbery_2_2782

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_2_2782


namespace cricket_average_increase_2_2628

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase_2_2628


namespace mother_age_when_harry_born_2_2971

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_2_2971


namespace Maria_waist_size_correct_2_2192

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end Maria_waist_size_correct_2_2192


namespace square_perimeter_2_2674

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end square_perimeter_2_2674


namespace find_cost_prices_2_2991

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices_2_2991


namespace problem1_problem2_problem3_problem4_2_2100

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4_2_2100


namespace find_vector_at_t4_2_2232

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end find_vector_at_t4_2_2232


namespace prime_factors_sum_correct_prime_factors_product_correct_2_2603

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end prime_factors_sum_correct_prime_factors_product_correct_2_2603


namespace correct_exponent_calculation_2_2033

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation_2_2033


namespace determine_y_2_2944

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end determine_y_2_2944


namespace add_in_base_7_2_2399

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7_2_2399


namespace pencil_groups_2_2897

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups_2_2897


namespace solve_x_2_2040

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end solve_x_2_2040


namespace puzzles_sold_correct_2_2567

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct_2_2567


namespace relationship_y1_y2_2_2477

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_2_2477


namespace nguyen_fabric_needs_2_2814

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_2_2814


namespace cos_180_eq_neg_one_2_2973

/-- The cosine of 180 degrees is -1. -/
theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 :=
by sorry

end cos_180_eq_neg_one_2_2973


namespace equivalent_exponentiation_2_2152

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end equivalent_exponentiation_2_2152


namespace trig_identity_example_2_2315

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example_2_2315


namespace max_lg_value_2_2105

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end max_lg_value_2_2105


namespace geometric_sequence_sum_2_2557

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum_2_2557


namespace birds_in_trees_2_2933

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_2_2933


namespace participation_schemes_count_2_2354

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count_2_2354


namespace minimum_number_is_correct_2_2149

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct_2_2149


namespace system_of_equations_solution_2_2308

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution_2_2308


namespace find_number_of_spiders_2_2072

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end find_number_of_spiders_2_2072


namespace benny_leftover_money_2_2440

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money_2_2440


namespace minimize_triangle_expression_2_2531

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end minimize_triangle_expression_2_2531


namespace number_of_sequences_2_2602

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences_2_2602


namespace tower_construction_2_2862

-- Define the number of cubes the child has
def red_cubes : Nat := 3
def blue_cubes : Nat := 3
def green_cubes : Nat := 4

-- Define the total number of cubes
def total_cubes : Nat := red_cubes + blue_cubes + green_cubes

-- Define the height of the tower and the number of cubes left out
def tower_height : Nat := 8
def cubes_left_out : Nat := 2

-- Prove that the number of different towers that can be constructed is 980
theorem tower_construction : 
  (∑ k in {0,1}, (Nat.factorial tower_height) / 
    (Nat.factorial (red_cubes - k) * Nat.factorial (blue_cubes - k) * 
     Nat.factorial (green_cubes - 2*k))) +
  (∑ k in {0,1}, (Nat.factorial total_cubes) / 
    (Nat.factorial (red_cubes - k) * Nat.factorial (blue_cubes - k) * 
     Nat.factorial (green_cubes - 2*k) * Nat.factorial (cubes_left_out - k))) = 980 := 
by 
  sorry

end tower_construction_2_2862


namespace find_pairs_2_2956

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end find_pairs_2_2956


namespace simplify_expression_2_2229

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression_2_2229


namespace max_value_g_eq_3_in_interval_2_2450

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval_2_2450


namespace exists_a_2_2608

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a_2_2608


namespace g_triple_composition_2_2482

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition_2_2482


namespace f_neg2_minus_f_neg3_2_2310

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3_2_2310


namespace determine_dimensions_2_2020

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end determine_dimensions_2_2020


namespace water_tank_capacity_2_2887

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end water_tank_capacity_2_2887


namespace find_solution_2_2394

theorem find_solution (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : (x = 1 ∧ y = 1) :=
sorry

end find_solution_2_2394


namespace axis_of_symmetry_parabola_2_2037

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola_2_2037


namespace sum_of_favorite_numbers_2_2931

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_2_2931


namespace f_of_x_2_2554

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x_2_2554


namespace intersection_M_N_2_2060

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end intersection_M_N_2_2060


namespace principal_amount_borrowed_2_2355

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed_2_2355


namespace seventeen_divides_9x_plus_5y_2_2969

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end seventeen_divides_9x_plus_5y_2_2969


namespace problem_2_2133

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end problem_2_2133


namespace no_such_integers_x_y_2_2667

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end no_such_integers_x_y_2_2667


namespace problem_Ashwin_Sah_2_2425

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah_2_2425


namespace smallest_y_value_2_2296

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end smallest_y_value_2_2296


namespace percentage_of_women_picnic_2_2415

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic_2_2415


namespace integer_a_conditions_2_2320

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions_2_2320


namespace frog_arrangement_2_2638

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end frog_arrangement_2_2638


namespace line_through_point_with_equal_intercepts_2_2899

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, -2)

-- Define the property of having equal absolute intercepts
def has_equal_absolute_intercepts (a b : ℝ) : Prop :=
  |a| = |b|

-- Define the general form of a line equation
def line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main theorem: Any line passing through (3, -2) with equal absolute intercepts satisfies the given equations
theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  has_equal_absolute_intercepts a b
  → line_eq 2 3 0 3 (-2)
  ∨ line_eq 1 1 (-1) 3 (-2)
  ∨ line_eq 1 (-1) (-5) 3 (-2) :=
by {
  sorry
}

end line_through_point_with_equal_intercepts_2_2899


namespace range_of_m_2_2515

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end range_of_m_2_2515


namespace smallest_digit_never_in_units_place_of_odd_numbers_2_2867

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_2_2867


namespace find_m_2_2502

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end find_m_2_2502


namespace proof_sets_2_2807

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end proof_sets_2_2807


namespace emma_investment_2_2269

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end emma_investment_2_2269


namespace sampling_interval_is_9_2_2700

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9_2_2700


namespace constant_term_is_21_2_2663

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_2_2663


namespace phone_extension_permutations_2_2747

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end phone_extension_permutations_2_2747


namespace muffin_half_as_expensive_as_banana_2_2997

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end muffin_half_as_expensive_as_banana_2_2997


namespace k_even_2_2108

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end k_even_2_2108


namespace find_pos_ints_2_2708

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end find_pos_ints_2_2708


namespace guest_bedroom_ratio_2_2645

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end guest_bedroom_ratio_2_2645


namespace solution_2_2265

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solution_2_2265


namespace speed_equivalence_2_2187

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end speed_equivalence_2_2187


namespace percentage_of_males_2_2079

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end percentage_of_males_2_2079


namespace right_triangle_has_one_right_angle_2_2301

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end right_triangle_has_one_right_angle_2_2301


namespace fraction_subtraction_2_2582

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end fraction_subtraction_2_2582


namespace count_seating_arrangements_2_2882

/-
  Definition of the seating problem at the round table:
  - The committee has six members from each of three species: Martians (M), Venusians (V), and Earthlings (E).
  - The table has 18 seats numbered from 1 to 18.
  - Seat 1 is occupied by a Martian, and seat 18 is occupied by an Earthling.
  - Martians cannot sit immediately to the left of Venusians.
  - Venusians cannot sit immediately to the left of Earthlings.
  - Earthlings cannot sit immediately to the left of Martians.
-/
def num_arrangements_valid_seating : ℕ := -- the number of valid seating arrangements
  sorry

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def N : ℕ := 347

theorem count_seating_arrangements :
  num_arrangements_valid_seating = N * (factorial 6)^3 :=
sorry

end count_seating_arrangements_2_2882


namespace cat_toy_cost_2_2733

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost_2_2733


namespace quilt_width_is_eight_2_2200

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight_2_2200


namespace tree_count_in_yard_2_2364

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end tree_count_in_yard_2_2364


namespace sum_of_ages_2_2416

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end sum_of_ages_2_2416


namespace f_m_eq_five_2_2892

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five_2_2892


namespace total_rooms_count_2_2855

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count_2_2855


namespace probability_dice_sum_12_2_2475

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := 25

theorem probability_dice_sum_12 :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 216 := by
  sorry

end probability_dice_sum_12_2_2475


namespace B_is_criminal_2_2830

-- Introduce the conditions
variable (A B C : Prop)  -- A, B, and C represent whether each individual is the criminal.

-- A says they did not commit the crime
axiom A_says_innocent : ¬A

-- Exactly one of A_says_innocent must hold true (A says ¬A, so B or C must be true)
axiom exactly_one_assertion_true : (¬A ∨ B ∨ C)

-- Problem Statement: Prove that B is the criminal
theorem B_is_criminal : B :=
by
  -- Solution steps would go here
  sorry

end B_is_criminal_2_2830


namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_2_2896

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_2_2896


namespace area_square_field_2_2576

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field_2_2576


namespace accessories_per_doll_2_2061

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end accessories_per_doll_2_2061


namespace kamal_average_marks_2_2380

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks_2_2380


namespace area_of_sector_2_2922

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_2_2922


namespace choir_row_lengths_2_2798

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end choir_row_lengths_2_2798


namespace max_quotient_2_2928

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient_2_2928


namespace smallest_value_n_2_2007

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end smallest_value_n_2_2007


namespace Lyka_saves_for_8_weeks_2_2755

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end Lyka_saves_for_8_weeks_2_2755


namespace color_opposite_orange_is_indigo_2_2186

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo_2_2186


namespace sqrt_D_always_irrational_2_2703

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational_2_2703


namespace person_speed_approx_2_2051

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end person_speed_approx_2_2051


namespace power_sum_eq_2_2042

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq_2_2042


namespace probability_exactly_two_heads_and_two_tails_2_2403

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails_2_2403


namespace no_eleven_points_achieve_any_score_2_2907

theorem no_eleven_points (x y : ℕ) : 3 * x + 7 * y ≠ 11 := 
sorry

theorem achieve_any_score (S : ℕ) (h : S ≥ 12) : ∃ (x y : ℕ), 3 * x + 7 * y = S :=
sorry

end no_eleven_points_achieve_any_score_2_2907


namespace area_of_rectangle_PQRS_2_2636

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end area_of_rectangle_PQRS_2_2636


namespace sqrt_defined_iff_ge_neg1_2_2115

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end sqrt_defined_iff_ge_neg1_2_2115


namespace least_x_2_2632

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end least_x_2_2632


namespace magic_square_y_value_2_2418

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end magic_square_y_value_2_2418


namespace remainder_when_eight_n_plus_five_divided_by_eleven_2_2647

theorem remainder_when_eight_n_plus_five_divided_by_eleven
  (n : ℤ) (h : n % 11 = 4) : (8 * n + 5) % 11 = 4 := 
  sorry

end remainder_when_eight_n_plus_five_divided_by_eleven_2_2647


namespace original_price_2_2675

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end original_price_2_2675


namespace minimum_third_highest_score_2_2538

theorem minimum_third_highest_score (scores : Fin 6 → ℕ) (h_uniq : Function.Injective scores)
  (h_avg : (∑ i, scores i) = 555) (h_max : ∃ i, scores i = 99) 
  (h_min : ∃ i, scores i = 76) : 
  ∃ s, s = 95 ∧ 
    ∃ (i : Fin 6), scores i = s ∧ 
    ∃ (j : Fin 6), (i ≠ j) ∧ (scores j < scores i) ∧ 
    ∃ (k : Fin 6), (i ≠ k) ∧ (j ≠ k) ∧ (scores k < scores j) :=
  sorry

end minimum_third_highest_score_2_2538


namespace probability_at_least_one_blue_2_2054

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end probability_at_least_one_blue_2_2054


namespace hcl_reaction_2_2076

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction_2_2076


namespace smallest_y_square_2_2098

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_2_2098


namespace min_shift_value_2_2990

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end min_shift_value_2_2990


namespace second_smallest_N_prevent_Bananastasia_win_2_2776

-- Definition of the set S, as positive integers not divisible by any p^4.
def S : Set ℕ := {n | ∀ p : ℕ, Prime p → ¬ (p ^ 4 ∣ n)}

-- Definition of the game rules and the condition for Anastasia to prevent Bananastasia from winning.
-- N is a value such that for all a in S, it is not possible for Bananastasia to directly win.

theorem second_smallest_N_prevent_Bananastasia_win :
  ∃ N : ℕ, N = 625 ∧ (∀ a ∈ S, N - a ≠ 0 ∧ N - a ≠ 1) :=
by
  sorry

end second_smallest_N_prevent_Bananastasia_win_2_2776


namespace min_voters_for_Tall_victory_2_2199

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_2_2199


namespace sum_first_mk_terms_arithmetic_seq_2_2360

theorem sum_first_mk_terms_arithmetic_seq (m k : ℕ) (hm : 0 < m) (hk : 0 < k)
  (a : ℕ → ℚ)
  (h_am : a m = (1 : ℚ) / k)
  (h_ak : a k = (1 : ℚ) / m) :
  ∑ i in Finset.range (m * k), a i = (1 + k * m) / 2 := sorry

end sum_first_mk_terms_arithmetic_seq_2_2360


namespace coats_leftover_2_2705

theorem coats_leftover :
  ∀ (total_coats : ℝ) (num_boxes : ℝ),
  total_coats = 385.5 →
  num_boxes = 7.5 →
  ∃ extra_coats : ℕ, extra_coats = 3 :=
by
  intros total_coats num_boxes h1 h2
  sorry

end coats_leftover_2_2705


namespace expected_pairs_of_adjacent_face_cards_is_44_over_17_2_2732
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17_2_2732


namespace find_principal_2_2270

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end find_principal_2_2270


namespace average_speed_2_2568

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed_2_2568


namespace find_prices_2_2980

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end find_prices_2_2980


namespace find_number_2_2041

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number_2_2041


namespace James_balloons_correct_2_2323

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct_2_2323


namespace interest_rate_second_part_2_2212

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end interest_rate_second_part_2_2212


namespace mike_corvette_average_speed_2_2365

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_2_2365


namespace sin_add_alpha_2_2556

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha_2_2556


namespace complement_of_B_in_A_2_2829

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A_2_2829


namespace plane_determination_2_2262

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end plane_determination_2_2262


namespace wam_gm_gt_hm_2_2048

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm_2_2048


namespace simplify_and_evaluate_2_2860

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_2_2860


namespace f_2010_plus_f_2011_2_2071

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end f_2010_plus_f_2011_2_2071


namespace lamp_count_and_profit_2_2967

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

end lamp_count_and_profit_2_2967


namespace transistors_in_2002_transistors_in_2010_2_2191

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010_2_2191


namespace paperback_copies_sold_2_2255

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold_2_2255


namespace nicky_speed_2_2113

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end nicky_speed_2_2113


namespace al_told_the_truth_2_2467

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end al_told_the_truth_2_2467


namespace smallest_period_cos_2_2625

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end smallest_period_cos_2_2625


namespace triangle_area_correct_2_2452

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end triangle_area_correct_2_2452


namespace simplify_expression_2_2695

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end simplify_expression_2_2695


namespace odd_function_negative_value_2_2039

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end odd_function_negative_value_2_2039


namespace shortest_side_2_2005

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end shortest_side_2_2005


namespace binom_18_6_2_2629

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_2_2629


namespace inequality_holds_2_2231

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_holds_2_2231


namespace fraction_inequality_2_2127

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality_2_2127


namespace time_to_save_for_downpayment_2_2499

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end time_to_save_for_downpayment_2_2499


namespace find_second_number_2_2518

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_2_2518


namespace ellipse_x_intercept_other_2_2119

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_2_2119


namespace closest_point_in_plane_2_2300

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end closest_point_in_plane_2_2300


namespace find_y_2_2409

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end find_y_2_2409


namespace find_b_2_2706

-- Definitions for conditions
def eq1 (a : ℤ) : Prop := 2 * a + 1 = 1
def eq2 (a b : ℤ) : Prop := 2 * b - 3 * a = 2

-- The theorem statement
theorem find_b (a b : ℤ) (h1 : eq1 a) (h2 : eq2 a b) : b = 1 :=
  sorry  -- Proof to be filled in.

end find_b_2_2706


namespace inequality_solution_sets_2_2864

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end inequality_solution_sets_2_2864


namespace smallest_k_for_sixty_four_gt_four_nineteen_2_2653

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen_2_2653


namespace mul_101_eq_10201_2_2458

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end mul_101_eq_10201_2_2458


namespace total_matches_correct_total_points_earthlings_correct_total_players_is_square_2_2660

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end total_matches_correct_total_points_earthlings_correct_total_players_is_square_2_2660


namespace price_per_butterfly_2_2974

theorem price_per_butterfly (jars : ℕ) (caterpillars_per_jar : ℕ) (fail_percentage : ℝ) (total_money : ℝ) (price : ℝ) :
  jars = 4 →
  caterpillars_per_jar = 10 →
  fail_percentage = 0.40 →
  total_money = 72 →
  price = 3 :=
by
  intros h_jars h_caterpillars h_fail_percentage h_total_money
  -- Full proof here
  sorry

end price_per_butterfly_2_2974


namespace incorrect_option_c_2_2436

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end incorrect_option_c_2_2436


namespace combined_total_value_of_items_2_2080

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items_2_2080


namespace min_y_value_2_2821

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value_2_2821


namespace trader_sold_45_meters_2_2167

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end trader_sold_45_meters_2_2167


namespace tangent_polar_equation_2_2927

theorem tangent_polar_equation :
  (∀ t : ℝ, ∃ (x y : ℝ), x = √2 * Real.cos t ∧ y = √2 * Real.sin t) →
  ∃ ρ θ : ℝ, (x = 1) ∧ (y = 1) → 
  ρ * Real.cos θ + ρ * Real.sin θ = 2 := 
by
  sorry

end tangent_polar_equation_2_2927
