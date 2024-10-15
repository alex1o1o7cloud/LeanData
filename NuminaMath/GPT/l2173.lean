import Mathlib

namespace NUMINAMATH_GPT_inequality_with_xy_l2173_217325

theorem inequality_with_xy
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3)) + (1 / (y + 3)) ≤ 2 / 5 :=
sorry

end NUMINAMATH_GPT_inequality_with_xy_l2173_217325


namespace NUMINAMATH_GPT_marks_in_chemistry_l2173_217338

-- Define the given conditions
def marks_english := 76
def marks_math := 65
def marks_physics := 82
def marks_biology := 85
def average_marks := 75
def number_subjects := 5

-- Define the theorem statement to prove David's marks in Chemistry
theorem marks_in_chemistry :
  let total_marks := marks_english + marks_math + marks_physics + marks_biology
  let total_marks_all_subjects := average_marks * number_subjects
  let marks_chemistry := total_marks_all_subjects - total_marks
  marks_chemistry = 67 :=
sorry

end NUMINAMATH_GPT_marks_in_chemistry_l2173_217338


namespace NUMINAMATH_GPT_find_natrual_numbers_l2173_217376

theorem find_natrual_numbers (k n : ℕ) (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : k ≥ 1) 
  (h2 : n ≥ 2) 
  (h3 : A ^ 3 = 0) 
  (h4 : A ^ k * B + B * A = 1) : 
  k = 1 ∧ Even n := 
sorry

end NUMINAMATH_GPT_find_natrual_numbers_l2173_217376


namespace NUMINAMATH_GPT_exponent_rule_l2173_217315

variable (a : ℝ) (m n : ℕ)

theorem exponent_rule (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_exponent_rule_l2173_217315


namespace NUMINAMATH_GPT_evaluate_expression_l2173_217369

theorem evaluate_expression : - (16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2173_217369


namespace NUMINAMATH_GPT_find_middle_integer_l2173_217320

theorem find_middle_integer (a b c : ℕ) (h1 : a^2 = 97344) (h2 : c^2 = 98596) (h3 : c = a + 2) : b = a + 1 ∧ b = 313 :=
by
  sorry

end NUMINAMATH_GPT_find_middle_integer_l2173_217320


namespace NUMINAMATH_GPT_long_diagonal_length_l2173_217353

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end NUMINAMATH_GPT_long_diagonal_length_l2173_217353


namespace NUMINAMATH_GPT_square_side_length_l2173_217316

variable (x : ℝ) (π : ℝ) (hπ: π = Real.pi)

theorem square_side_length (h1: 4 * x = 10 * π) : 
  x = (5 * π) / 2 := 
by
  sorry

end NUMINAMATH_GPT_square_side_length_l2173_217316


namespace NUMINAMATH_GPT_angle_terminal_side_equiv_l2173_217327

theorem angle_terminal_side_equiv (α : ℝ) (k : ℤ) :
  (∃ k : ℤ, α = 30 + k * 360) ↔ (∃ β : ℝ, β = 30 ∧ α % 360 = β % 360) :=
by
  sorry

end NUMINAMATH_GPT_angle_terminal_side_equiv_l2173_217327


namespace NUMINAMATH_GPT_union_A_B_equals_x_lt_3_l2173_217388

theorem union_A_B_equals_x_lt_3 :
  let A := { x : ℝ | 3 - x > 0 ∧ x + 2 > 0 }
  let B := { x : ℝ | 3 > 2*x - 1 }
  A ∪ B = { x : ℝ | x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_equals_x_lt_3_l2173_217388


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2173_217336

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2173_217336


namespace NUMINAMATH_GPT_find_time_to_fill_tank_l2173_217372

noncomputable def time_to_fill_tanker (TA : ℝ) : Prop :=
  let RB := 1 / 40
  let fill_time := 29.999999999999993
  let half_fill_time := fill_time / 2
  let RAB := (1 / TA) + RB
  (RAB * half_fill_time = 1 / 2) → (TA = 120)

theorem find_time_to_fill_tank : ∃ TA, time_to_fill_tanker TA :=
by
  use 120
  sorry

end NUMINAMATH_GPT_find_time_to_fill_tank_l2173_217372


namespace NUMINAMATH_GPT_find_p_l2173_217399

-- Define the coordinates as given in the problem
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Defining the function to calculate area of triangle given three points
def area_of_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (P1.fst * (P2.snd - P3.snd) + P2.fst * (P3.snd - P1.snd) + P3.fst * (P1.snd - P2.snd))

-- The statement we need to prove
theorem find_p :
  ∃ p : ℝ, area_of_triangle A B (C p) = 42 ∧ p = 11.75 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2173_217399


namespace NUMINAMATH_GPT_net_change_in_collection_is_94_l2173_217389

-- Definitions for the given conditions
def thrown_away_caps : Nat := 6
def initially_found_caps : Nat := 50
def additionally_found_caps : Nat := 44 + thrown_away_caps

-- Definition of the total found bottle caps
def total_found_caps : Nat := initially_found_caps + additionally_found_caps

-- Net change in Bottle Cap collection
def net_change_in_collection : Nat := total_found_caps - thrown_away_caps

-- Proof statement
theorem net_change_in_collection_is_94 : net_change_in_collection = 94 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_net_change_in_collection_is_94_l2173_217389


namespace NUMINAMATH_GPT_express_as_sum_of_cubes_l2173_217307

variables {a b : ℝ}

theorem express_as_sum_of_cubes (a b : ℝ) : 
  2 * a * (a^2 + 3 * b^2) = (a + b)^3 + (a - b)^3 :=
by sorry

end NUMINAMATH_GPT_express_as_sum_of_cubes_l2173_217307


namespace NUMINAMATH_GPT_steps_in_staircase_l2173_217346

theorem steps_in_staircase (h1 : 120 / 20 = 6) (h2 : 180 / 6 = 30) : 
  ∃ n : ℕ, n = 30 :=
by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_steps_in_staircase_l2173_217346


namespace NUMINAMATH_GPT_max_b_div_a_plus_c_l2173_217377

-- Given positive numbers a, b, c
-- equation: b^2 + 2(a + c)b - ac = 0
-- Prove: ∀ a b c : ℝ (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 + 2*(a + c)*b - a*c = 0),
--         b/(a + c) ≤ (Real.sqrt 5 - 2)/2

theorem max_b_div_a_plus_c (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + 2 * (a + c) * b - a * c = 0) :
  b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
sorry

end NUMINAMATH_GPT_max_b_div_a_plus_c_l2173_217377


namespace NUMINAMATH_GPT_diaries_ratio_l2173_217324

variable (initial_diaries : ℕ)
variable (final_diaries : ℕ)
variable (lost_fraction : ℚ)
variable (bought_diaries : ℕ)

theorem diaries_ratio 
  (h1 : initial_diaries = 8)
  (h2 : final_diaries = 18)
  (h3 : lost_fraction = 1 / 4)
  (h4 : ∃ x : ℕ, (initial_diaries + x - lost_fraction * (initial_diaries + x) = final_diaries) ∧ x = 16) :
  (16 / initial_diaries : ℚ) = 2 := 
by
  sorry

end NUMINAMATH_GPT_diaries_ratio_l2173_217324


namespace NUMINAMATH_GPT_elevation_after_descend_l2173_217379

theorem elevation_after_descend (initial_elevation : ℕ) (rate : ℕ) (time : ℕ) (final_elevation : ℕ) 
  (h_initial : initial_elevation = 400) 
  (h_rate : rate = 10) 
  (h_time : time = 5) 
  (h_final : final_elevation = initial_elevation - rate * time) : 
  final_elevation = 350 := 
by 
  sorry

end NUMINAMATH_GPT_elevation_after_descend_l2173_217379


namespace NUMINAMATH_GPT_find_product_abcd_l2173_217365

def prod_abcd (a b c d : ℚ) :=
  4 * a - 2 * b + 3 * c + 5 * d = 22 ∧
  2 * (d + c) = b - 2 ∧
  4 * b - c = a + 1 ∧
  c + 1 = 2 * d

theorem find_product_abcd (a b c d : ℚ) (h : prod_abcd a b c d) :
  a * b * c * d = -30751860 / 11338912 :=
sorry

end NUMINAMATH_GPT_find_product_abcd_l2173_217365


namespace NUMINAMATH_GPT_green_apples_count_l2173_217366

variables (G R : ℕ)

def total_apples_collected (G R : ℕ) : Prop :=
  R + G = 496

def relation_red_green (G R : ℕ) : Prop :=
  R = 3 * G

theorem green_apples_count (G R : ℕ) (h1 : total_apples_collected G R) (h2 : relation_red_green G R) :
  G = 124 :=
by sorry

end NUMINAMATH_GPT_green_apples_count_l2173_217366


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l2173_217305

theorem rowing_speed_in_still_water (v c : ℝ) (t : ℝ) (h1 : c = 1.1) (h2 : (v + c) * t = (v - c) * 2 * t) : v = 3.3 :=
sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l2173_217305


namespace NUMINAMATH_GPT_monochromatic_triangle_probability_l2173_217323

noncomputable def probability_of_monochromatic_triangle_in_hexagon : ℝ := 0.968324

theorem monochromatic_triangle_probability :
  ∃ (H : Hexagon), probability_of_monochromatic_triangle_in_hexagon = 0.968324 :=
sorry

end NUMINAMATH_GPT_monochromatic_triangle_probability_l2173_217323


namespace NUMINAMATH_GPT_right_triangle_perimeter_l2173_217333

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l2173_217333


namespace NUMINAMATH_GPT_plants_per_row_l2173_217308

-- Define the conditions from the problem
def rows : ℕ := 7
def extra_plants : ℕ := 15
def total_plants : ℕ := 141

-- Define the problem statement to prove
theorem plants_per_row :
  ∃ x : ℕ, rows * x + extra_plants = total_plants ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_plants_per_row_l2173_217308


namespace NUMINAMATH_GPT_value_of_b_l2173_217370

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2173_217370


namespace NUMINAMATH_GPT_correct_expression_after_removing_parentheses_l2173_217381

variable (a b c : ℝ)

theorem correct_expression_after_removing_parentheses :
  -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c :=
sorry

end NUMINAMATH_GPT_correct_expression_after_removing_parentheses_l2173_217381


namespace NUMINAMATH_GPT_total_amount_divided_l2173_217359

theorem total_amount_divided 
    (A B C : ℝ) 
    (h1 : A = (2 / 3) * (B + C)) 
    (h2 : B = (2 / 3) * (A + C)) 
    (h3 : A = 160) : 
    A + B + C = 400 := 
by 
  sorry

end NUMINAMATH_GPT_total_amount_divided_l2173_217359


namespace NUMINAMATH_GPT_minimum_combined_horses_ponies_l2173_217304

noncomputable def ranch_min_total (P H : ℕ) : ℕ :=
  P + H

theorem minimum_combined_horses_ponies (P H : ℕ) 
  (h1 : ∃ k : ℕ, P = 16 * k ∧ k ≥ 1)
  (h2 : H = P + 3) 
  (h3 : P = 80) 
  (h4 : H = 83) :
  ranch_min_total P H = 163 :=
by
  sorry

end NUMINAMATH_GPT_minimum_combined_horses_ponies_l2173_217304


namespace NUMINAMATH_GPT_equation_of_line_l2173_217303

theorem equation_of_line (a b : ℝ) (h1 : a = -2) (h2 : b = 2) :
  (∀ x y : ℝ, (x / a + y / b = 1) → x - y + 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l2173_217303


namespace NUMINAMATH_GPT_unwanted_texts_per_week_l2173_217347

-- Define the conditions as constants
def messages_per_day_old : ℕ := 20
def messages_per_day_new : ℕ := 55
def days_per_week : ℕ := 7

-- Define the theorem stating the problem
theorem unwanted_texts_per_week (messages_per_day_old messages_per_day_new days_per_week 
  : ℕ) : (messages_per_day_new - messages_per_day_old) * days_per_week = 245 :=
by
  sorry

end NUMINAMATH_GPT_unwanted_texts_per_week_l2173_217347


namespace NUMINAMATH_GPT_math_problem_l2173_217360

noncomputable def x : ℝ := -2

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}
def C1 : Set ℝ := {1, 3}
def C2 : Set ℝ := {3, 4}

theorem math_problem
  (h1 : B x ⊆ A x) :
  x = -2 ∧ (B x ∪ C1 = A x ∨ B x ∪ C2 = A x) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2173_217360


namespace NUMINAMATH_GPT_rectangle_area_l2173_217395

-- Definitions
variables {height length : ℝ} (h : height = length / 2)
variables {area perimeter : ℝ} (a : area = perimeter)

-- Problem statement
theorem rectangle_area : ∃ h : ℝ, ∃ l : ℝ, ∃ area : ℝ, 
  (l = 2 * h) ∧ (area = l * h) ∧ (area = 2 * (l + h)) ∧ (area = 18) :=
sorry

end NUMINAMATH_GPT_rectangle_area_l2173_217395


namespace NUMINAMATH_GPT_part_one_part_two_l2173_217330

def f (a x : ℝ) : ℝ := |a - 4 * x| + |2 * a + x|

theorem part_one (x : ℝ) : f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 / 5 := 
sorry

theorem part_two (a x : ℝ) : f a x + f a (-1 / x) ≥ 10 := 
sorry

end NUMINAMATH_GPT_part_one_part_two_l2173_217330


namespace NUMINAMATH_GPT_log2_monotone_l2173_217398

theorem log2_monotone (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (Real.log a / Real.log 2 > Real.log b / Real.log 2) :=
sorry

end NUMINAMATH_GPT_log2_monotone_l2173_217398


namespace NUMINAMATH_GPT_real_numbers_correspond_to_number_line_l2173_217314

noncomputable def number_line := ℝ

def real_numbers := ℝ

theorem real_numbers_correspond_to_number_line :
  ∀ (p : ℝ), ∃ (r : real_numbers), r = p ∧ ∀ (r : real_numbers), ∃ (p : ℝ), p = r :=
by
  sorry

end NUMINAMATH_GPT_real_numbers_correspond_to_number_line_l2173_217314


namespace NUMINAMATH_GPT_son_age_l2173_217396

variable (F S : ℕ)
variable (h₁ : F = 3 * S)
variable (h₂ : F - 8 = 4 * (S - 8))

theorem son_age : S = 24 := 
by 
  sorry

end NUMINAMATH_GPT_son_age_l2173_217396


namespace NUMINAMATH_GPT_championship_outcomes_l2173_217332

theorem championship_outcomes :
  ∀ (students events : ℕ), students = 4 → events = 3 → students ^ events = 64 :=
by
  intros students events h_students h_events
  rw [h_students, h_events]
  exact rfl

end NUMINAMATH_GPT_championship_outcomes_l2173_217332


namespace NUMINAMATH_GPT_initial_potatoes_count_l2173_217311

theorem initial_potatoes_count (initial_tomatoes picked_tomatoes total_remaining : ℕ) 
    (h_initial_tomatoes : initial_tomatoes = 177)
    (h_picked_tomatoes : picked_tomatoes = 53)
    (h_total_remaining : total_remaining = 136) :
  (initial_tomatoes - picked_tomatoes + x = total_remaining) → 
  x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_initial_potatoes_count_l2173_217311


namespace NUMINAMATH_GPT_total_perimeter_of_compound_shape_l2173_217352

-- Definitions of the conditions from the original problem
def triangle1_side : ℝ := 10
def triangle2_side : ℝ := 6
def shared_side : ℝ := 6

-- A theorem to represent the mathematically equivalent proof problem
theorem total_perimeter_of_compound_shape 
  (t1s : ℝ := triangle1_side) 
  (t2s : ℝ := triangle2_side)
  (ss : ℝ := shared_side) : 
  t1s = 10 ∧ t2s = 6 ∧ ss = 6 → 3 * t1s + 3 * t2s - ss = 42 := 
by
  sorry

end NUMINAMATH_GPT_total_perimeter_of_compound_shape_l2173_217352


namespace NUMINAMATH_GPT_problem1_problem2_l2173_217339

theorem problem1 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : a > b) : a + b = 7 ∨ a + b = 3 := 
by sorry

theorem problem2 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : |a + b| = |a| - |b|) : (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2) := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2173_217339


namespace NUMINAMATH_GPT_value_of_x2_plus_9y2_l2173_217394

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end NUMINAMATH_GPT_value_of_x2_plus_9y2_l2173_217394


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2173_217342

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2173_217342


namespace NUMINAMATH_GPT_inscribed_circle_radius_is_correct_l2173_217390

noncomputable def radius_of_inscribed_circle (base height : ℝ) : ℝ := sorry

theorem inscribed_circle_radius_is_correct :
  radius_of_inscribed_circle 20 24 = 120 / 13 := sorry

end NUMINAMATH_GPT_inscribed_circle_radius_is_correct_l2173_217390


namespace NUMINAMATH_GPT_part1_part2_l2173_217392

noncomputable def A (x : ℝ) (k : ℝ) := -2 * x ^ 2 - (k - 1) * x + 1
noncomputable def B (x : ℝ) := -2 * (x ^ 2 - x + 2)

-- Part 1: If A is a quadratic binomial, then the value of k is 1
theorem part1 (x : ℝ) (k : ℝ) (h : ∀ x, A x k ≠ 0) : k = 1 :=
sorry

-- Part 2: When k = -1, C + 2A = B, then C = 2x^2 - 2x - 6
theorem part2 (x : ℝ) (C : ℝ → ℝ) (h1 : k = -1) (h2 : ∀ x, C x + 2 * A x k = B x) : (C x = 2 * x ^ 2 - 2 * x - 6) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2173_217392


namespace NUMINAMATH_GPT_infinite_solutions_l2173_217361

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_l2173_217361


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_by_8_11_15_l2173_217355

-- Define what it means for a number to be divisible by another
def divisible_by (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

-- Define a function to find the least common multiple of three numbers
noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Statement of the theorem
theorem smallest_positive_integer_divisible_by_8_11_15 : 
  ∀ n : ℕ, (n > 0) ∧ divisible_by n 8 ∧ divisible_by n 11 ∧ divisible_by n 15 ↔ n = 1320 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_smallest_positive_integer_divisible_by_8_11_15_l2173_217355


namespace NUMINAMATH_GPT_mn_value_l2173_217300

theorem mn_value (m n : ℤ) (h1 : m + n = 1) (h2 : m - n + 2 = 1) : m * n = 0 := 
by 
  sorry

end NUMINAMATH_GPT_mn_value_l2173_217300


namespace NUMINAMATH_GPT_seeds_in_bucket_C_l2173_217335

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_seeds_in_bucket_C_l2173_217335


namespace NUMINAMATH_GPT_proof_custom_operations_l2173_217302

def customOp1 (a b : ℕ) : ℕ := a * b / (a + b)
def customOp2 (a b : ℕ) : ℕ := a * a + b * b

theorem proof_custom_operations :
  customOp2 (customOp1 7 14) 2 = 200 := 
by 
  sorry

end NUMINAMATH_GPT_proof_custom_operations_l2173_217302


namespace NUMINAMATH_GPT_children_l2173_217340

variable (C : ℝ) -- Define the weight of a children's book

theorem children's_book_weight :
  (9 * 0.8 + 7 * C = 10.98) → C = 0.54 :=
by  
sorry

end NUMINAMATH_GPT_children_l2173_217340


namespace NUMINAMATH_GPT_all_fruits_fallen_by_twelfth_day_l2173_217326

noncomputable def magical_tree_falling_day : Nat :=
  let total_fruits := 58
  let initial_day_falls := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].foldl (· + ·) 0
  let continuation_falls := [1, 2].foldl (· + ·) 0
  let total_days := initial_day_falls + continuation_falls
  12

theorem all_fruits_fallen_by_twelfth_day :
  magical_tree_falling_day = 12 :=
by
  sorry

end NUMINAMATH_GPT_all_fruits_fallen_by_twelfth_day_l2173_217326


namespace NUMINAMATH_GPT_swimming_pool_width_l2173_217349

theorem swimming_pool_width
  (length : ℝ)
  (lowered_height_inches : ℝ)
  (removed_water_gallons : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (volume_for_removal : ℝ)
  (width : ℝ) :
  length = 60 → 
  lowered_height_inches = 6 →
  removed_water_gallons = 4500 →
  gallons_per_cubic_foot = 7.5 →
  volume_for_removal = removed_water_gallons / gallons_per_cubic_foot →
  width = volume_for_removal / (length * (lowered_height_inches / 12)) →
  width = 20 :=
by
  intros h_length h_lowered_height h_removed_water h_gallons_per_cubic_foot h_volume_for_removal h_width
  sorry

end NUMINAMATH_GPT_swimming_pool_width_l2173_217349


namespace NUMINAMATH_GPT_find_difference_l2173_217318

variable (f : ℝ → ℝ)

-- Conditions
axiom linear_f : ∀ x y a b, f (a * x + b * y) = a * f x + b * f y
axiom f_difference : f 6 - f 2 = 12

theorem find_difference : f 12 - f 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l2173_217318


namespace NUMINAMATH_GPT_yu_chan_walked_distance_l2173_217387

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walked_distance : step_length * steps_per_minute * walking_time = 682.5 :=
by
  sorry

end NUMINAMATH_GPT_yu_chan_walked_distance_l2173_217387


namespace NUMINAMATH_GPT_remainder_of_trailing_zeroes_in_factorials_product_l2173_217345

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc * factorial x) 1 

def trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5 + trailing_zeroes (n / 5))

def trailing_zeroes_in_product (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc + trailing_zeroes x) 0 

theorem remainder_of_trailing_zeroes_in_factorials_product :
  let N := trailing_zeroes_in_product 150
  N % 500 = 45 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_trailing_zeroes_in_factorials_product_l2173_217345


namespace NUMINAMATH_GPT_second_term_geometric_series_l2173_217306

theorem second_term_geometric_series (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 :=
by
  -- Sorry is used to finalize the theorem without providing a proof here
  sorry

end NUMINAMATH_GPT_second_term_geometric_series_l2173_217306


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_l2173_217364

-- Definition of a dodecahedron based on given conditions
structure Dodecahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (vertices_per_face : ℕ)
  (faces_per_vertex : ℕ)
  (interior_diagonals : ℕ)

-- Conditions provided in the problem
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    interior_diagonals := 130 }

-- The theorem to prove that given a dodecahedron structure, it has the correct number of interior diagonals
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : d.interior_diagonals = 130 := by
  sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_l2173_217364


namespace NUMINAMATH_GPT_circle_properties_l2173_217380

noncomputable def pi : Real := 3.14
variable (C : Real) (diameter : Real) (radius : Real) (area : Real)

theorem circle_properties (h₀ : C = 31.4) :
  radius = C / (2 * pi) ∧
  diameter = 2 * radius ∧
  area = pi * radius^2 ∧
  radius = 5 ∧
  diameter = 10 ∧
  area = 78.5 :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l2173_217380


namespace NUMINAMATH_GPT_det_my_matrix_l2173_217350

def my_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 1], ![-5, 5, -4], ![3, 3, 6]]

theorem det_my_matrix : my_matrix.det = 96 := by
  sorry

end NUMINAMATH_GPT_det_my_matrix_l2173_217350


namespace NUMINAMATH_GPT_ellipse_standard_equation_l2173_217309

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : (3 * a) / (-a) + 16 / b = 1)
  (h2 : (3 * a) / c + 16 / (-b) = 1)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : a > b)
  (h6 : a^2 = b^2 + c^2) : 
  (a = 5 ∧ b = 4 ∧ c = 3) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ (a = 5 ∧ b = 4)) := 
sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l2173_217309


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l2173_217382

def part1 (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (m * x1 - 2) * (m * x2 - 2) = 4

theorem part1_solution : part1 (1/3) 9 18 :=
by 
  sorry

def part2 (m x1 x2 : ℕ) : Prop :=
  ((m * x1 - 2) * (m * x2 - 2) = 4)

def count_pairs : ℕ := 7

theorem part2_solution 
  (m x1 x2 : ℕ) 
  (h_pos : m > 0 ∧ x1 > 0 ∧ x2 > 0) : 
  ∃ c, c = count_pairs ∧ 
  (part2 m x1 x2) :=
by 
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l2173_217382


namespace NUMINAMATH_GPT_andrew_total_days_l2173_217391

noncomputable def hours_per_day : ℝ := 2.5
noncomputable def total_hours : ℝ := 7.5

theorem andrew_total_days : total_hours / hours_per_day = 3 := 
by 
  sorry

end NUMINAMATH_GPT_andrew_total_days_l2173_217391


namespace NUMINAMATH_GPT_worksheets_turned_in_l2173_217378

def initial_worksheets : ℕ := 34
def graded_worksheets : ℕ := 7
def remaining_worksheets : ℕ := initial_worksheets - graded_worksheets
def current_worksheets : ℕ := 63

theorem worksheets_turned_in :
  current_worksheets - remaining_worksheets = 36 :=
by
  sorry

end NUMINAMATH_GPT_worksheets_turned_in_l2173_217378


namespace NUMINAMATH_GPT_num_people_present_l2173_217375

-- Given conditions
def associatePencilCount (A : ℕ) : ℕ := 2 * A
def assistantPencilCount (B : ℕ) : ℕ := B
def associateChartCount (A : ℕ) : ℕ := A
def assistantChartCount (B : ℕ) : ℕ := 2 * B

def totalPencils (A B : ℕ) : ℕ := associatePencilCount A + assistantPencilCount B
def totalCharts (A B : ℕ) : ℕ := associateChartCount A + assistantChartCount B

-- Prove the total number of people present
theorem num_people_present (A B : ℕ) (h1 : totalPencils A B = 11) (h2 : totalCharts A B = 16) : A + B = 9 :=
by
  sorry

end NUMINAMATH_GPT_num_people_present_l2173_217375


namespace NUMINAMATH_GPT_problem_l2173_217351

section Problem
variables {n : ℕ } {k : ℕ} 

theorem problem (n : ℕ) (k : ℕ) (a : ℕ) (n_i : Fin k → ℕ) (h1 : ∀ i j, i ≠ j → Nat.gcd (n_i i) (n_i j) = 1) 
  (h2 : ∀ i, a^n_i i % n_i i = 1) (h3 : ∀ i, ¬(n_i i ∣ a - 1)) :
  ∃ (x : ℕ), x > 1 ∧ a^x % x = 1 ∧ x ≥ 2^(k + 1) - 2 := by
  sorry
end Problem

end NUMINAMATH_GPT_problem_l2173_217351


namespace NUMINAMATH_GPT_sym_sum_ineq_l2173_217313

theorem sym_sum_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z = 1 / x + 1 / y + 1 / z) : x * y + y * z + z * x ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_sym_sum_ineq_l2173_217313


namespace NUMINAMATH_GPT_determine_s_l2173_217337

theorem determine_s 
  (s : ℝ) 
  (h : (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
       6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30) : 
  s = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_s_l2173_217337


namespace NUMINAMATH_GPT_ratio_of_wilted_roses_to_total_l2173_217397

-- Defining the conditions
def initial_roses := 24
def traded_roses := 12
def total_roses := initial_roses + traded_roses
def remaining_roses_after_second_night := 9
def roses_before_second_night := remaining_roses_after_second_night * 2
def wilted_roses_after_first_night := total_roses - roses_before_second_night
def ratio_wilted_to_total := wilted_roses_after_first_night / total_roses

-- Proving the ratio of wilted flowers to the total number of flowers after the first night is 1:2
theorem ratio_of_wilted_roses_to_total :
  ratio_wilted_to_total = (1/2) := by
  sorry

end NUMINAMATH_GPT_ratio_of_wilted_roses_to_total_l2173_217397


namespace NUMINAMATH_GPT_sum_of_six_consecutive_integers_l2173_217312

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_six_consecutive_integers_l2173_217312


namespace NUMINAMATH_GPT_f_2011_l2173_217310

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_defined_segment : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_2011 : f 2011 = -2 := by
  sorry

end NUMINAMATH_GPT_f_2011_l2173_217310


namespace NUMINAMATH_GPT_negation_of_p_l2173_217344

-- Define the proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of p
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := sorry

end NUMINAMATH_GPT_negation_of_p_l2173_217344


namespace NUMINAMATH_GPT_discount_policy_l2173_217362

-- Define the prices of the fruits
def lemon_price := 2
def papaya_price := 1
def mango_price := 4

-- Define the quantities Tom buys
def lemons_bought := 6
def papayas_bought := 4
def mangos_bought := 2

-- Define the total amount paid by Tom
def amount_paid := 21

-- Define the total number of fruits bought
def total_fruits_bought := lemons_bought + papayas_bought + mangos_bought

-- Define the total cost without discount
def total_cost_without_discount := 
  (lemons_bought * lemon_price) + 
  (papayas_bought * papaya_price) + 
  (mangos_bought * mango_price)

-- Calculate the discount
def discount := total_cost_without_discount - amount_paid

-- The discount policy
theorem discount_policy : discount = 3 ∧ total_fruits_bought = 12 :=
by 
  sorry

end NUMINAMATH_GPT_discount_policy_l2173_217362


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2173_217368

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x + m = 0) ↔ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2173_217368


namespace NUMINAMATH_GPT_a_gives_b_head_start_l2173_217367

theorem a_gives_b_head_start (Va Vb L H : ℝ) 
    (h1 : Va = (20 / 19) * Vb)
    (h2 : L / Va = (L - H) / Vb) : 
    H = (1 / 20) * L := sorry

end NUMINAMATH_GPT_a_gives_b_head_start_l2173_217367


namespace NUMINAMATH_GPT_arithmetic_sequence_primes_l2173_217334

theorem arithmetic_sequence_primes (a : ℕ) (d : ℕ) (primes_seq : ∀ n : ℕ, n < 15 → Nat.Prime (a + n * d))
  (distinct_primes : ∀ m n : ℕ, m < 15 → n < 15 → m ≠ n → a + m * d ≠ a + n * d) :
  d > 30000 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_primes_l2173_217334


namespace NUMINAMATH_GPT_other_root_of_equation_l2173_217358

theorem other_root_of_equation (m : ℝ) :
  (∃ (x : ℝ), 3 * x^2 + m * x = -2 ∧ x = -1) →
  (∃ (y : ℝ), 3 * y^2 + m * y + 2 = 0 ∧ y = -(-2 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_equation_l2173_217358


namespace NUMINAMATH_GPT_solution_to_problem_l2173_217371

theorem solution_to_problem
  {x y z : ℝ}
  (h1 : xy / (x + y) = 1 / 3)
  (h2 : yz / (y + z) = 1 / 5)
  (h3 : zx / (z + x) = 1 / 6) :
  xyz / (xy + yz + zx) = 1 / 7 :=
by sorry

end NUMINAMATH_GPT_solution_to_problem_l2173_217371


namespace NUMINAMATH_GPT_annual_rate_of_decrease_l2173_217319

variable (r : ℝ) (initial_population population_after_2_years : ℝ)

-- Conditions
def initial_population_eq : initial_population = 30000 := sorry
def population_after_2_years_eq : population_after_2_years = 19200 := sorry
def population_formula : population_after_2_years = initial_population * (1 - r)^2 := sorry

-- Goal: Prove that the annual rate of decrease r is 0.2
theorem annual_rate_of_decrease :
  r = 0.2 := sorry

end NUMINAMATH_GPT_annual_rate_of_decrease_l2173_217319


namespace NUMINAMATH_GPT_CorrectChoice_l2173_217386

open Classical

-- Define the integer n
variable (n : ℤ)

-- Define proposition p: 2n - 1 is always odd
def p : Prop := ∃ k : ℤ, 2 * k + 1 = 2 * n - 1

-- Define proposition q: 2n + 1 is always even
def q : Prop := ∃ k : ℤ, 2 * k = 2 * n + 1

-- The theorem we want to prove
theorem CorrectChoice : (p n ∨ q n) :=
by
  sorry

end NUMINAMATH_GPT_CorrectChoice_l2173_217386


namespace NUMINAMATH_GPT_last_digit_of_3_pow_2004_l2173_217357

theorem last_digit_of_3_pow_2004 : (3 ^ 2004) % 10 = 1 := by
  sorry

end NUMINAMATH_GPT_last_digit_of_3_pow_2004_l2173_217357


namespace NUMINAMATH_GPT_find_a2_l2173_217317

variable {a_n : ℕ → ℚ}

def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n-1) * d

theorem find_a2 (h_seq : arithmetic_seq a_n) (h3_5 : a_n 3 + a_n 5 = 15) (h6 : a_n 6 = 7) :
  a_n 2 = 8 := 
sorry

end NUMINAMATH_GPT_find_a2_l2173_217317


namespace NUMINAMATH_GPT_hike_duration_l2173_217321

def initial_water := 11
def final_water := 2
def leak_rate := 1
def water_drunk := 6

theorem hike_duration (time_hours : ℕ) :
  initial_water - final_water = water_drunk + time_hours * leak_rate →
  time_hours = 3 :=
by intro h; sorry

end NUMINAMATH_GPT_hike_duration_l2173_217321


namespace NUMINAMATH_GPT_slices_with_all_toppings_l2173_217354

-- Definitions
def slices_with_pepperoni (x y w : ℕ) : ℕ := 15 - x - y + w
def slices_with_mushrooms (x z w : ℕ) : ℕ := 16 - x - z + w
def slices_with_olives (y z w : ℕ) : ℕ := 10 - y - z + w

-- Problem's total validation condition
axiom total_slices_with_at_least_one_topping (x y z w : ℕ) :
  15 + 16 + 10 - x - y - z - 2 * w = 24

-- Statement to prove
theorem slices_with_all_toppings (x y z w : ℕ) (h : 15 + 16 + 10 - x - y - z - 2 * w = 24) : w = 2 :=
sorry

end NUMINAMATH_GPT_slices_with_all_toppings_l2173_217354


namespace NUMINAMATH_GPT_find_duplicated_page_number_l2173_217322

noncomputable def duplicated_page_number (n : ℕ) (incorrect_sum : ℕ) : ℕ :=
  incorrect_sum - n * (n + 1) / 2

theorem find_duplicated_page_number :
  ∃ n k, (1 <= k ∧ k <= n) ∧ ( ∃ n, (1 <= n) ∧ ( n * (n + 1) / 2 + k = 2550) )
  ∧ duplicated_page_number 70 2550 = 65 :=
by
  sorry

end NUMINAMATH_GPT_find_duplicated_page_number_l2173_217322


namespace NUMINAMATH_GPT_nancy_games_this_month_l2173_217331

-- Define the variables and conditions from the problem
def went_games_last_month : ℕ := 8
def plans_games_next_month : ℕ := 7
def total_games : ℕ := 24

-- Let's calculate the games this month and state the theorem
def games_last_and_next : ℕ := went_games_last_month + plans_games_next_month
def games_this_month : ℕ := total_games - games_last_and_next

-- The theorem statement
theorem nancy_games_this_month : games_this_month = 9 := by
  -- Proof is omitted for the sake of brevity
  sorry

end NUMINAMATH_GPT_nancy_games_this_month_l2173_217331


namespace NUMINAMATH_GPT_Rick_is_three_times_Sean_l2173_217348

-- Definitions and assumptions
def Fritz_money : ℕ := 40
def Sean_money : ℕ := (Fritz_money / 2) + 4
def total_money : ℕ := 96

-- Rick's money can be derived from total_money - Sean_money
def Rick_money : ℕ := total_money - Sean_money

-- Claim to be proven
theorem Rick_is_three_times_Sean : Rick_money = 3 * Sean_money := 
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_Rick_is_three_times_Sean_l2173_217348


namespace NUMINAMATH_GPT_factor_expression_l2173_217341

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2173_217341


namespace NUMINAMATH_GPT_find_c_interval_l2173_217385

theorem find_c_interval (c : ℚ) : 
  (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ (-4 ≤ c ∧ c < -3 / 2) := 
by 
  sorry

end NUMINAMATH_GPT_find_c_interval_l2173_217385


namespace NUMINAMATH_GPT_cement_percentage_first_concrete_correct_l2173_217301

open Real

noncomputable def cement_percentage_of_first_concrete := 
  let total_weight := 4500 
  let cement_percentage := 10.8 / 100
  let weight_each_type := 1125
  let total_cement_weight := cement_percentage * total_weight
  let x := 2.0 / 100
  let y := 21.6 / 100 - x
  (weight_each_type * x + weight_each_type * y = total_cement_weight) →
  (x = 2.0 / 100)

theorem cement_percentage_first_concrete_correct :
  cement_percentage_of_first_concrete := sorry

end NUMINAMATH_GPT_cement_percentage_first_concrete_correct_l2173_217301


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l2173_217373

theorem ratio_of_x_intercepts (b s t : ℝ) (h_b : b ≠ 0)
  (h1 : 0 = 8 * s + b)
  (h2 : 0 = 4 * t + b) :
  s / t = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_intercepts_l2173_217373


namespace NUMINAMATH_GPT_geom_series_sum_correct_l2173_217384

noncomputable def geometric_series_sum (b1 r : ℚ) (n : ℕ) : ℚ :=
b1 * (1 - r ^ n) / (1 - r)

theorem geom_series_sum_correct :
  geometric_series_sum (3/4) (3/4) 15 = 3177905751 / 1073741824 := by
sorry

end NUMINAMATH_GPT_geom_series_sum_correct_l2173_217384


namespace NUMINAMATH_GPT_range_of_a_l2173_217363

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define the theorem to be proved
theorem range_of_a (a : ℝ) (h₁ : A a ⊆ B) (h₂ : 2 - a < 2 + a) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2173_217363


namespace NUMINAMATH_GPT_lower_tap_used_earlier_l2173_217393

-- Define the conditions given in the problem
def capacity : ℕ := 36
def midway_capacity : ℕ := capacity / 2
def lower_tap_rate : ℕ := 4  -- minutes per litre
def upper_tap_rate : ℕ := 6  -- minutes per litre

def lower_tap_draw (minutes : ℕ) : ℕ := minutes / lower_tap_rate  -- litres drawn by lower tap
def beer_left_after_draw (initial_amount litres_drawn : ℕ) : ℕ := initial_amount - litres_drawn

-- Define the assistant's drawing condition
def assistant_draw_min : ℕ := 16
def assistant_draw_litres : ℕ := lower_tap_draw assistant_draw_min

-- Define proof statement
theorem lower_tap_used_earlier :
  let initial_amount := capacity
  let litres_when_midway := midway_capacity
  let litres_beer_left := beer_left_after_draw initial_amount assistant_draw_litres
  let additional_litres := litres_beer_left - litres_when_midway
  let time_earlier := additional_litres * upper_tap_rate
  time_earlier = 84 := 
by
  sorry

end NUMINAMATH_GPT_lower_tap_used_earlier_l2173_217393


namespace NUMINAMATH_GPT_Nina_saves_enough_to_buy_video_game_in_11_weeks_l2173_217343

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end NUMINAMATH_GPT_Nina_saves_enough_to_buy_video_game_in_11_weeks_l2173_217343


namespace NUMINAMATH_GPT_range_of_sum_l2173_217329

theorem range_of_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + b + 3 = a * b) : 
a + b ≥ 6 := 
sorry

end NUMINAMATH_GPT_range_of_sum_l2173_217329


namespace NUMINAMATH_GPT_committee_size_l2173_217374

theorem committee_size (n : ℕ)
  (h : ((n - 2 : ℕ) : ℚ) / ((n - 1) * (n - 2) / 2 : ℚ) = 0.4) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_committee_size_l2173_217374


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l2173_217328

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l2173_217328


namespace NUMINAMATH_GPT_condition_for_diff_of_roots_l2173_217356

/-- Statement: For a quadratic equation of the form x^2 + px + q = 0, if the difference of the roots is a, then the condition a^2 - p^2 = -4q holds. -/
theorem condition_for_diff_of_roots (p q a : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 - x2 = a) :
  a^2 - p^2 = -4 * q :=
sorry

end NUMINAMATH_GPT_condition_for_diff_of_roots_l2173_217356


namespace NUMINAMATH_GPT_value_of_f5_and_f_neg5_l2173_217383

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f5_and_f_neg5 (a b c : ℝ) (m : ℝ) (h : f a b c (-5) = m) :
  f a b c 5 + f a b c (-5) = 4 :=
sorry

end NUMINAMATH_GPT_value_of_f5_and_f_neg5_l2173_217383
