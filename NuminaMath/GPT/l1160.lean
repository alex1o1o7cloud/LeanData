import Mathlib

namespace NUMINAMATH_GPT_smallest_scalene_triangle_perimeter_l1160_116077

-- Define what it means for a number to be a prime number greater than 3
def prime_gt_3 (n : ℕ) : Prop := Prime n ∧ 3 < n

-- Define the main theorem
theorem smallest_scalene_triangle_perimeter : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  prime_gt_3 a ∧ prime_gt_3 b ∧ prime_gt_3 c ∧
  Prime (a + b + c) ∧ 
  (∀ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    prime_gt_3 x ∧ prime_gt_3 y ∧ prime_gt_3 z ∧
    Prime (x + y + z) → (a + b + c) ≤ (x + y + z)) ∧
  a + b + c = 23 := by
    sorry

end NUMINAMATH_GPT_smallest_scalene_triangle_perimeter_l1160_116077


namespace NUMINAMATH_GPT_intersection_of_sets_l1160_116086

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1160_116086


namespace NUMINAMATH_GPT_measure_of_angle_y_l1160_116026

theorem measure_of_angle_y (m n : ℝ) (A B C D F G H : ℝ) :
  (m = n) → (A = 40) → (B = 90) → (B = 40) → (y = 80) :=
by
  -- proof steps to be filled in
  sorry

end NUMINAMATH_GPT_measure_of_angle_y_l1160_116026


namespace NUMINAMATH_GPT_actual_height_is_191_l1160_116087

theorem actual_height_is_191 :
  ∀ (n incorrect_avg correct_avg incorrect_height x : ℝ),
  n = 20 ∧ incorrect_avg = 175 ∧ correct_avg = 173 ∧ incorrect_height = 151 ∧
  (n * incorrect_avg - n * correct_avg = x - incorrect_height) →
  x = 191 :=
by
  intros n incorrect_avg correct_avg incorrect_height x h
  -- skip the proof part
  sorry

end NUMINAMATH_GPT_actual_height_is_191_l1160_116087


namespace NUMINAMATH_GPT_determine_t_l1160_116003

theorem determine_t (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_t_l1160_116003


namespace NUMINAMATH_GPT_circle_equation_through_ABC_circle_equation_with_center_and_points_l1160_116001

-- Define points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨6, -2⟩

-- First problem: proof of the circle equation given points A, B, and C
theorem circle_equation_through_ABC :
  ∃ (D E F : ℝ), 
  (∀ (P : Point), (P = A ∨ P = B ∨ P = C) → P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0) 
  ↔ (D = -5 ∧ E = 7 ∧ F = 4) := sorry

-- Second problem: proof of the circle equation given the y-coordinate of the center and points A and B
theorem circle_equation_with_center_and_points :
  ∃ (h k r : ℝ), 
  (h = (A.x + B.x) / 2 ∧ k = 2) ∧
  ∀ (P : Point), (P = A ∨ P = B) → (P.x - h)^2 + (P.y - k)^2 = r^2
  ↔ (h = 5 / 2 ∧ k = 2 ∧ r = 5 / 2) := sorry

end NUMINAMATH_GPT_circle_equation_through_ABC_circle_equation_with_center_and_points_l1160_116001


namespace NUMINAMATH_GPT_largest_x_value_l1160_116089

noncomputable def quadratic_eq (x : ℝ) : Prop :=
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60)

theorem largest_x_value (x : ℝ) :
  quadratic_eq x → x = - ((35 - Real.sqrt 745) / 12) ∨
  x = - ((35 + Real.sqrt 745) / 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_x_value_l1160_116089


namespace NUMINAMATH_GPT_questionnaires_drawn_l1160_116076

theorem questionnaires_drawn
  (units : ℕ → ℕ)
  (h_arithmetic : ∀ n, units (n + 1) - units n = units 1 - units 0)
  (h_total : units 0 + units 1 + units 2 + units 3 = 100)
  (h_unitB : units 1 = 20) :
  units 3 = 40 :=
by
  -- Proof would go here
  -- Establish that the arithmetic sequence difference is 10, then compute unit D (units 3)
  sorry

end NUMINAMATH_GPT_questionnaires_drawn_l1160_116076


namespace NUMINAMATH_GPT_no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l1160_116024

theorem no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ a b c : ℕ, 2 * n^2 + 1 = a^2 ∧ 3 * n^2 + 1 = b^2 ∧ 6 * n^2 + 1 = c^2 := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l1160_116024


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1160_116099

theorem perfect_square_trinomial (y : ℝ) (m : ℝ) : 
  (∃ b : ℝ, y^2 - m*y + 9 = (y + b)^2) → (m = 6 ∨ m = -6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1160_116099


namespace NUMINAMATH_GPT_least_possible_value_of_squares_l1160_116000

theorem least_possible_value_of_squares (a b x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 15 * a + 16 * b = x^2) (h2 : 16 * a - 15 * b = y^2) : 
  ∃ (x : ℕ) (y : ℕ), min (x^2) (y^2) = 231361 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_squares_l1160_116000


namespace NUMINAMATH_GPT_percentage_of_part_over_whole_l1160_116054

theorem percentage_of_part_over_whole (Part Whole : ℕ) (h1 : Part = 120) (h2 : Whole = 50) :
  (Part / Whole : ℚ) * 100 = 240 := by
  sorry

end NUMINAMATH_GPT_percentage_of_part_over_whole_l1160_116054


namespace NUMINAMATH_GPT_square_area_l1160_116019

theorem square_area (A : ℝ) (s : ℝ) (prob_not_in_B : ℝ)
  (h1 : s * 4 = 32)
  (h2 : prob_not_in_B = 0.20987654320987653)
  (h3 : A - s^2 = prob_not_in_B * A) :
  A = 81 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1160_116019


namespace NUMINAMATH_GPT_opposite_sides_of_line_l1160_116006

theorem opposite_sides_of_line (a : ℝ) (h1 : 0 < a) (h2 : a < 2) : (-a) * (2 - a) < 0 :=
sorry

end NUMINAMATH_GPT_opposite_sides_of_line_l1160_116006


namespace NUMINAMATH_GPT_rationalize_denominator_correct_l1160_116071

noncomputable def rationalize_denominator : ℚ := 
  let A := 5
  let B := 49
  let C := 21
  -- Form is (5 * ∛49) / 21
  A + B + C

theorem rationalize_denominator_correct : rationalize_denominator = 75 :=
  by 
    -- The proof steps are omitted, as they are not required for this task
    sorry

end NUMINAMATH_GPT_rationalize_denominator_correct_l1160_116071


namespace NUMINAMATH_GPT_geometric_sequence_min_value_l1160_116095

theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n) 
  (h2 : a 9 = 9 * a 7)
  (exists_m_n : ∃ m n, a m * a n = 9 * (a 1)^2):
  ∀ m n, (m + n = 4) → (1 / m + 9 / n) ≥ 4 :=
by
  intros m n h
  sorry

end NUMINAMATH_GPT_geometric_sequence_min_value_l1160_116095


namespace NUMINAMATH_GPT_mk_div_km_l1160_116041

theorem mk_div_km 
  (m n k : ℕ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (h1 : m^n ∣ n^m) 
  (h2 : n^k ∣ k^n) : 
  m^k ∣ k^m := 
  sorry

end NUMINAMATH_GPT_mk_div_km_l1160_116041


namespace NUMINAMATH_GPT_fraction_of_work_left_correct_l1160_116093

-- Define the conditions for p, q, and r
def p_one_day_work : ℚ := 1 / 15
def q_one_day_work : ℚ := 1 / 20
def r_one_day_work : ℚ := 1 / 30

-- Define the total work done in one day by p, q, and r
def total_one_day_work : ℚ := p_one_day_work + q_one_day_work + r_one_day_work

-- Define the work done in 4 days
def work_done_in_4_days : ℚ := total_one_day_work * 4

-- Define the fraction of work left after 4 days
def fraction_of_work_left : ℚ := 1 - work_done_in_4_days

-- Statement to prove
theorem fraction_of_work_left_correct : fraction_of_work_left = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_of_work_left_correct_l1160_116093


namespace NUMINAMATH_GPT_value_of_7_prime_prime_l1160_116011

-- Define the function q' (written as q_prime in Lean)
def q_prime (q : ℕ) : ℕ := 3 * q - 3

-- Define the specific value problem
theorem value_of_7_prime_prime : q_prime (q_prime 7) = 51 := by
  sorry

end NUMINAMATH_GPT_value_of_7_prime_prime_l1160_116011


namespace NUMINAMATH_GPT_cost_price_of_radio_l1160_116063

theorem cost_price_of_radio (SP : ℝ) (L_p : ℝ) (C : ℝ) (h₁ : SP = 3200) (h₂ : L_p = 0.28888888888888886) 
  (h₃ : SP = C - (C * L_p)) : C = 4500 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_radio_l1160_116063


namespace NUMINAMATH_GPT_operation_evaluation_l1160_116015

theorem operation_evaluation : 65 + 5 * 12 / (180 / 3) = 66 :=
by
  -- Parentheses
  have h1 : 180 / 3 = 60 := by sorry
  -- Multiplication and Division
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  -- Addition
  exact sorry

end NUMINAMATH_GPT_operation_evaluation_l1160_116015


namespace NUMINAMATH_GPT_value_of_x_l1160_116005

variable (x y : ℕ)

-- Conditions
axiom cond1 : x / y = 15 / 3
axiom cond2 : y = 27

-- Lean statement for the problem
theorem value_of_x : x = 135 :=
by
  have h1 := cond1
  have h2 := cond2
  sorry

end NUMINAMATH_GPT_value_of_x_l1160_116005


namespace NUMINAMATH_GPT_isosceles_triangle_angles_sum_l1160_116075

theorem isosceles_triangle_angles_sum (x : ℝ) 
  (h_triangle_sum : ∀ a b c : ℝ, a + b + c = 180)
  (h_isosceles : ∃ a b : ℝ, (a = 50 ∧ b = x) ∨ (a = x ∧ b = 50)) :
  50 + x + (180 - 50 * 2) + 65 + 80 = 195 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_sum_l1160_116075


namespace NUMINAMATH_GPT_not_directly_or_inversely_proportional_l1160_116047

theorem not_directly_or_inversely_proportional
  (P : ∀ x y : ℝ, x + y = 0 → (∃ k : ℝ, x = k * y))
  (Q : ∀ x y : ℝ, 3 * x * y = 10 → ∃ k : ℝ, x * y = k)
  (R : ∀ x y : ℝ, x = 5 * y → (∃ k : ℝ, x = k * y))
  (S : ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y))
  (T : ∀ x y : ℝ, x / y = Real.sqrt 3 → (∃ k : ℝ, x = k * y)) :
  ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y) := by
  sorry

end NUMINAMATH_GPT_not_directly_or_inversely_proportional_l1160_116047


namespace NUMINAMATH_GPT_find_cost_price_l1160_116038

/-- 
Given:
- SP = 1290 (selling price)
- LossP = 14.000000000000002 (loss percentage)
Prove that: CP = 1500 (cost price)
--/
theorem find_cost_price (SP : ℝ) (LossP : ℝ) (CP : ℝ) (h1 : SP = 1290) (h2 : LossP = 14.000000000000002) : CP = 1500 :=
sorry

end NUMINAMATH_GPT_find_cost_price_l1160_116038


namespace NUMINAMATH_GPT_video_games_expenditure_l1160_116045

theorem video_games_expenditure (allowance : ℝ) (books_expense : ℝ) (snacks_expense : ℝ) (clothes_expense : ℝ) 
    (initial_allowance : allowance = 50)
    (books_fraction : books_expense = 1 / 7 * allowance)
    (snacks_fraction : snacks_expense = 1 / 2 * allowance)
    (clothes_fraction : clothes_expense = 3 / 14 * allowance) :
    50 - (books_expense + snacks_expense + clothes_expense) = 7.15 :=
by
  sorry

end NUMINAMATH_GPT_video_games_expenditure_l1160_116045


namespace NUMINAMATH_GPT_five_digit_palindromes_count_l1160_116050

theorem five_digit_palindromes_count : 
  ∃ (a b c : Fin 10), (a ≠ 0) ∧ (∃ (count : Nat), count = 9 * 10 * 10 ∧ count = 900) :=
by
  sorry

end NUMINAMATH_GPT_five_digit_palindromes_count_l1160_116050


namespace NUMINAMATH_GPT_smallest_n_div_75_eq_432_l1160_116059

theorem smallest_n_div_75_eq_432 :
  ∃ n k : ℕ, (n ∣ 75 ∧ (∃ (d : ℕ), d ∣ n → d ≠ 1 → d ≠ n → n = 75 * k ∧ ∀ x: ℕ, (x ∣ n) → (x ≠ 1 ∧ x ≠ n) → False)) → ( k =  432 ) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_div_75_eq_432_l1160_116059


namespace NUMINAMATH_GPT_largest_positive_integer_solution_l1160_116023

theorem largest_positive_integer_solution (x : ℕ) (h₁ : 1 ≤ x) (h₂ : x + 3 ≤ 6) : 
  x = 3 := by
  sorry

end NUMINAMATH_GPT_largest_positive_integer_solution_l1160_116023


namespace NUMINAMATH_GPT_side_length_of_square_l1160_116081

theorem side_length_of_square (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l1160_116081


namespace NUMINAMATH_GPT_equilateral_triangle_condition_l1160_116060

-- We define points in a plane and vectors between these points
structure Point where
  x : ℝ
  y : ℝ

-- Vector subtraction
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

-- The equation required to hold for certain type of triangles
def bisector_eq_zero (A B C A1 B1 C1 : Point) : Prop :=
  let AA1 := vector A A1
  let BB1 := vector B B1
  let CC1 := vector C C1
  AA1.x + BB1.x + CC1.x = 0 ∧ AA1.y + BB1.y + CC1.y = 0

-- Property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  let CA := vector C A
  (AB.x^2 + AB.y^2 = BC.x^2 + BC.y^2 ∧ BC.x^2 + BC.y^2 = CA.x^2 + CA.y^2)

-- Main theorem statement
theorem equilateral_triangle_condition (A B C A1 B1 C1 : Point)
  (h : bisector_eq_zero A B C A1 B1 C1) :
  is_equilateral A B C :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_condition_l1160_116060


namespace NUMINAMATH_GPT_problem_statement_l1160_116094

variable {x y z : ℝ}

-- Lean 4 statement of the problem
theorem problem_statement (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) (h₄ : 0 ≤ z) (h₅ : z ≤ 1) :
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1160_116094


namespace NUMINAMATH_GPT_arc_length_of_sector_l1160_116042

theorem arc_length_of_sector (r θ : ℝ) (A : ℝ) (h₁ : r = 4)
  (h₂ : A = 7) : (1 / 2) * r^2 * θ = A → r * θ = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1160_116042


namespace NUMINAMATH_GPT_probability_of_triangle_or_circle_l1160_116048

/-- The total number of figures -/
def total_figures : ℕ := 10

/-- The number of triangles -/
def triangles : ℕ := 3

/-- The number of circles -/
def circles : ℕ := 3

/-- The number of figures that are either triangles or circles -/
def favorable_figures : ℕ := triangles + circles

/-- The probability that the chosen figure is either a triangle or a circle -/
theorem probability_of_triangle_or_circle : (favorable_figures : ℚ) / (total_figures : ℚ) = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_triangle_or_circle_l1160_116048


namespace NUMINAMATH_GPT_second_time_apart_l1160_116033

theorem second_time_apart 
  (glen_speed : ℕ) 
  (hannah_speed : ℕ)
  (initial_distance : ℕ) 
  (initial_time : ℕ)
  (relative_speed : ℕ)
  (hours_later : ℕ) :
  glen_speed = 37 →
  hannah_speed = 15 →
  initial_distance = 130 →
  initial_time = 6 →
  relative_speed = glen_speed + hannah_speed →
  hours_later = initial_distance / relative_speed →
  initial_time + hours_later = 8 + 30 / 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_time_apart_l1160_116033


namespace NUMINAMATH_GPT_measure_angle_A_l1160_116017

open Real

def triangle_area (a b c S : ℝ) (A B C : ℝ) : Prop :=
  S = (1 / 2) * b * c * sin A

def sides_and_angles (a b c A B C : ℝ) : Prop :=
  A = 2 * B

theorem measure_angle_A (a b c S A B C : ℝ)
  (h1 : triangle_area a b c S A B C)
  (h2 : sides_and_angles a b c A B C)
  (h3 : S = (a ^ 2) / 4) :
  A = π / 2 ∨ A = π / 4 :=
  sorry

end NUMINAMATH_GPT_measure_angle_A_l1160_116017


namespace NUMINAMATH_GPT_angle_sum_at_F_l1160_116083

theorem angle_sum_at_F (x y z w v : ℝ) (h : x + y + z + w + v = 360) : 
  x = 360 - y - z - w - v := by
  sorry

end NUMINAMATH_GPT_angle_sum_at_F_l1160_116083


namespace NUMINAMATH_GPT_drug_price_reduction_l1160_116085

theorem drug_price_reduction (x : ℝ) :
    36 * (1 - x)^2 = 25 :=
sorry

end NUMINAMATH_GPT_drug_price_reduction_l1160_116085


namespace NUMINAMATH_GPT_number_of_rectangles_l1160_116065

theorem number_of_rectangles (a b : ℝ) (ha_lt_b : a < b) :
  ∃! (x y : ℝ), (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4) := 
sorry

end NUMINAMATH_GPT_number_of_rectangles_l1160_116065


namespace NUMINAMATH_GPT_flagpole_proof_l1160_116043

noncomputable def flagpole_height (AC AD DE : ℝ) (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) : ℝ :=
  let DC := AC - AD
  let h_ratio := DE / DC
  h_ratio * AC

theorem flagpole_proof (AC AD DE : ℝ) (h_AC : AC = 4) (h_AD : AD = 3) (h_DE : DE = 1.8) 
  (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) :
  flagpole_height AC AD DE h_ABC_DEC = 7.2 := by
  sorry

end NUMINAMATH_GPT_flagpole_proof_l1160_116043


namespace NUMINAMATH_GPT_evaluate_expression_l1160_116002

def f (x : ℕ) : ℕ :=
  match x with
  | 3 => 10
  | 4 => 17
  | 5 => 26
  | 6 => 37
  | 7 => 50
  | _ => 0  -- for any x not in the table, f(x) is undefined and defaults to 0

def f_inv (y : ℕ) : ℕ :=
  match y with
  | 10 => 3
  | 17 => 4
  | 26 => 5
  | 37 => 6
  | 50 => 7
  | _ => 0  -- for any y not in the table, f_inv(y) is undefined and defaults to 0

theorem evaluate_expression :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1160_116002


namespace NUMINAMATH_GPT_relationship_abc_l1160_116072

noncomputable def a : ℝ := 4 / 5
noncomputable def b : ℝ := Real.sin (2 / 3)
noncomputable def c : ℝ := Real.cos (1 / 3)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_abc_l1160_116072


namespace NUMINAMATH_GPT_money_difference_l1160_116062

-- Given conditions
def packs_per_hour_peak : Nat := 6
def packs_per_hour_low : Nat := 4
def price_per_pack : Nat := 60
def hours_per_day : Nat := 15

-- Calculate total sales in peak and low seasons
def total_sales_peak : Nat :=
  packs_per_hour_peak * price_per_pack * hours_per_day

def total_sales_low : Nat :=
  packs_per_hour_low * price_per_pack * hours_per_day

-- The Lean statement proving the correct answer
theorem money_difference :
  total_sales_peak - total_sales_low = 1800 :=
by
  sorry

end NUMINAMATH_GPT_money_difference_l1160_116062


namespace NUMINAMATH_GPT_no_such_number_l1160_116031

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end NUMINAMATH_GPT_no_such_number_l1160_116031


namespace NUMINAMATH_GPT_intersection_A_B_l1160_116030

noncomputable def A : Set ℝ := {x | 9 * x ^ 2 < 1}

noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 2 * x + 5 / 4}

theorem intersection_A_B :
  (A ∩ B) = {y | y ∈ Set.Ico (1/4 : ℝ) (1/3 : ℝ)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1160_116030


namespace NUMINAMATH_GPT_total_spent_l1160_116016

def spending (A B C : ℝ) : Prop :=
  (A = (13 / 10) * B) ∧
  (C = (4 / 5) * B) ∧
  (A = C + 15)

theorem total_spent (A B C : ℝ) (h : spending A B C) : A + B + C = 93 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l1160_116016


namespace NUMINAMATH_GPT_intersection_equiv_l1160_116027

def A : Set ℝ := { x : ℝ | x > 1 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def C : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem intersection_equiv : A ∩ B = C :=
by
  sorry

end NUMINAMATH_GPT_intersection_equiv_l1160_116027


namespace NUMINAMATH_GPT_simplify_fraction_to_9_l1160_116014

-- Define the necessary terms and expressions
def problem_expr := (3^12)^2 - (3^10)^2
def problem_denom := (3^11)^2 - (3^9)^2
def simplified_expr := problem_expr / problem_denom

-- State the theorem we want to prove
theorem simplify_fraction_to_9 : simplified_expr = 9 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_to_9_l1160_116014


namespace NUMINAMATH_GPT_cost_of_song_book_l1160_116070

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end NUMINAMATH_GPT_cost_of_song_book_l1160_116070


namespace NUMINAMATH_GPT_triangle_angle_and_area_l1160_116082

section Geometry

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def vectors_parallel (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Problem statement
theorem triangle_angle_and_area (A B C a b c : ℝ) : 
  triangle_sides_opposite_angles a b c A B C ∧ vectors_parallel a b A B ∧ a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3
  → A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end Geometry

end NUMINAMATH_GPT_triangle_angle_and_area_l1160_116082


namespace NUMINAMATH_GPT_solution_of_fractional_inequality_l1160_116068

noncomputable def solution_set_of_inequality : Set ℝ :=
  {x : ℝ | -3 < x ∨ x > 1/2 }

theorem solution_of_fractional_inequality :
  {x : ℝ | (2 * x - 1) / (x + 3) > 0} = solution_set_of_inequality :=
by
  sorry

end NUMINAMATH_GPT_solution_of_fractional_inequality_l1160_116068


namespace NUMINAMATH_GPT_roots_of_equation_l1160_116040

theorem roots_of_equation :
  {x : ℝ | -x * (x + 3) = x * (x + 3)} = {0, -3} :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1160_116040


namespace NUMINAMATH_GPT_sqrt_defined_value_l1160_116096

theorem sqrt_defined_value (x : ℝ) (h : x ≥ 4) : x = 5 → true := 
by 
  intro hx
  sorry

end NUMINAMATH_GPT_sqrt_defined_value_l1160_116096


namespace NUMINAMATH_GPT_division_of_monomials_l1160_116084

variable (x : ℝ) -- ensure x is defined as a variable, here assuming x is a real number

theorem division_of_monomials (x : ℝ) : (2 * x^3 / x^2) = 2 * x := 
by 
  sorry

end NUMINAMATH_GPT_division_of_monomials_l1160_116084


namespace NUMINAMATH_GPT_second_car_distance_l1160_116013

variables 
  (distance_apart : ℕ := 105)
  (d1 d2 d3 : ℕ := 25) -- distances 25 km, 15 km, 25 km respectively
  (d_road_back : ℕ := 15)
  (final_distance : ℕ := 20)

theorem second_car_distance 
  (car1_total_distance := d1 + d2 + d3 + d_road_back)
  (car2_distance : ℕ) :
  distance_apart - (car1_total_distance + car2_distance) = final_distance →
  car2_distance = 5 :=
sorry

end NUMINAMATH_GPT_second_car_distance_l1160_116013


namespace NUMINAMATH_GPT_evaluate_expression_l1160_116035

theorem evaluate_expression : 2 * (2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3) = 24240542 :=
by
  let a := 2009
  let b := 2010
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1160_116035


namespace NUMINAMATH_GPT_sqrt_square_eq_self_l1160_116025

theorem sqrt_square_eq_self (a : ℝ) (h : a ≥ 1/2) :
  Real.sqrt ((2 * a - 1) ^ 2) = 2 * a - 1 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_square_eq_self_l1160_116025


namespace NUMINAMATH_GPT_positive_integer_solutions_l1160_116066

theorem positive_integer_solutions (n x y z t : ℕ) (h_n : n > 0) (h_n_neq_1 : n ≠ 1) (h_x : x > 0) (h_y : y > 0) (h_z : z > 0) (h_t : t > 0) :
  (n ^ x ∣ n ^ y + n ^ z ↔ n ^ x = n ^ t) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨ (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l1160_116066


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1160_116044

theorem solution_set_of_inequality :
  { x : ℝ // (x - 2)^2 ≤ 2 * x + 11 } = { x : ℝ | -1 ≤ x ∧ x ≤ 7 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1160_116044


namespace NUMINAMATH_GPT_sandy_ordered_three_cappuccinos_l1160_116022

-- Definitions and conditions
def cost_cappuccino : ℝ := 2
def cost_iced_tea : ℝ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℝ := 1
def num_iced_teas : ℕ := 2
def num_cafe_lattes : ℕ := 2
def num_espressos : ℕ := 2
def change_received : ℝ := 3
def amount_paid : ℝ := 20

-- Calculation of costs
def total_cost_iced_teas : ℝ := num_iced_teas * cost_iced_tea
def total_cost_cafe_lattes : ℝ := num_cafe_lattes * cost_cafe_latte
def total_cost_espressos : ℝ := num_espressos * cost_espresso
def total_cost_other_drinks : ℝ := total_cost_iced_teas + total_cost_cafe_lattes + total_cost_espressos
def total_spent : ℝ := amount_paid - change_received
def cost_cappuccinos := total_spent - total_cost_other_drinks

-- Proof statement
theorem sandy_ordered_three_cappuccinos (num_cappuccinos : ℕ) : cost_cappuccinos = num_cappuccinos * cost_cappuccino → num_cappuccinos = 3 :=
by sorry

end NUMINAMATH_GPT_sandy_ordered_three_cappuccinos_l1160_116022


namespace NUMINAMATH_GPT_base_five_product_l1160_116097

theorem base_five_product (n1 n2 : ℕ) (h1 : n1 = 1 * 5^2 + 3 * 5^1 + 1 * 5^0) 
                          (h2 : n2 = 1 * 5^1 + 2 * 5^0) :
  let product_dec := (n1 * n2 : ℕ)
  let product_base5 := 2 * 125 + 1 * 25 + 2 * 5 + 2 * 1
  product_dec = 287 ∧ product_base5 = 2122 := by
                                -- calculations to verify statement omitted
                                sorry

end NUMINAMATH_GPT_base_five_product_l1160_116097


namespace NUMINAMATH_GPT_shaniqua_earnings_correct_l1160_116004

noncomputable def calc_earnings : ℝ :=
  let haircut_tuesday := 5 * 10
  let haircut_normal := 5 * 12
  let styling_vip := (6 * 25) * (1 - 0.2)
  let styling_regular := 4 * 25
  let coloring_friday := (7 * 35) * (1 - 0.15)
  let coloring_normal := 3 * 35
  let treatment_senior := (3 * 50) * (1 - 0.1)
  let treatment_other := 4 * 50
  haircut_tuesday + haircut_normal + styling_vip + styling_regular + coloring_friday + coloring_normal + treatment_senior + treatment_other

theorem shaniqua_earnings_correct : calc_earnings = 978.25 := by
  sorry

end NUMINAMATH_GPT_shaniqua_earnings_correct_l1160_116004


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l1160_116039

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) (h3 : a 3 = 3) (h5 : a 5 = -3) : a 7 = -9 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l1160_116039


namespace NUMINAMATH_GPT_crossed_out_digit_l1160_116056

theorem crossed_out_digit (N S S' x : ℕ) (hN : N % 9 = 3) (hS : S % 9 = 3) (hS' : S' % 9 = 7)
  (hS'_eq : S' = S - x) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_crossed_out_digit_l1160_116056


namespace NUMINAMATH_GPT_total_expenditure_l1160_116046

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end NUMINAMATH_GPT_total_expenditure_l1160_116046


namespace NUMINAMATH_GPT_max_souls_guaranteed_l1160_116058

def initial_nuts : ℕ := 1001

def valid_N (N : ℕ) : Prop :=
  1 ≤ N ∧ N ≤ 1001

def nuts_transferred (N : ℕ) (T : ℕ) : Prop :=
  valid_N N ∧ T ≤ 71

theorem max_souls_guaranteed : (∀ N, valid_N N → ∃ T, nuts_transferred N T) :=
sorry

end NUMINAMATH_GPT_max_souls_guaranteed_l1160_116058


namespace NUMINAMATH_GPT_total_fundamental_particles_l1160_116057

def protons := 9
def neutrons := 19 - protons
def electrons := protons
def total_particles := protons + neutrons + electrons

theorem total_fundamental_particles : total_particles = 28 := by
  sorry

end NUMINAMATH_GPT_total_fundamental_particles_l1160_116057


namespace NUMINAMATH_GPT_find_y_l1160_116088

variable (x y z : ℚ)

theorem find_y
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) :
  y = 688 / 9 :=
sorry

end NUMINAMATH_GPT_find_y_l1160_116088


namespace NUMINAMATH_GPT_inscribable_quadrilateral_l1160_116080

theorem inscribable_quadrilateral
  (a b c d : ℝ)
  (A : ℝ)
  (circumscribable : Prop)
  (area_condition : A = Real.sqrt (a * b * c * d))
  (A := Real.sqrt (a * b * c * d)) : 
  circumscribable → ∃ B D : ℝ, B + D = 180 :=
sorry

end NUMINAMATH_GPT_inscribable_quadrilateral_l1160_116080


namespace NUMINAMATH_GPT_evans_family_children_count_l1160_116032

-- Let the family consist of the mother, the father, two grandparents, and children.
-- This proof aims to show x, the number of children, is 1.

theorem evans_family_children_count
  (m g y : ℕ) -- m = mother's age, g = average age of two grandparents, y = average age of children
  (x : ℕ) -- x = number of children
  (avg_family_age : (m + 50 + 2 * g + x * y) / (4 + x) = 30)
  (father_age : 50 = 50)
  (avg_non_father_age : (m + 2 * g + x * y) / (3 + x) = 25) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_evans_family_children_count_l1160_116032


namespace NUMINAMATH_GPT_find_m_l1160_116021

def hyperbola_focus (x y : ℝ) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = -m ∧ (x - 0)^2 / a^2 - (y - 0)^2 / b^2 = 1

theorem find_m (m : ℝ) (H : hyperbola_focus 5 0 m) : m = -16 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1160_116021


namespace NUMINAMATH_GPT_no_four_digit_with_five_units_divisible_by_ten_l1160_116049

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def units_place_is_five (n : ℕ) : Prop :=
  n % 10 = 5

def divisible_by_ten (n : ℕ) : Prop :=
  n % 10 = 0

theorem no_four_digit_with_five_units_divisible_by_ten : ∀ n : ℕ, 
  is_four_digit n → units_place_is_five n → ¬ divisible_by_ten n :=
by
  intro n h1 h2
  rw [units_place_is_five] at h2
  rw [divisible_by_ten, h2]
  sorry

end NUMINAMATH_GPT_no_four_digit_with_five_units_divisible_by_ten_l1160_116049


namespace NUMINAMATH_GPT_inequality_proof_l1160_116012

theorem inequality_proof (a b c : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |c * x^2 + b * x + a| ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1160_116012


namespace NUMINAMATH_GPT_right_triangle_cosine_l1160_116079

theorem right_triangle_cosine (XY XZ YZ : ℝ) (hXY_pos : XY > 0) (hXZ_pos : XZ > 0) (hYZ_pos : YZ > 0)
  (angle_XYZ : angle_1 = 90) (tan_Z : XY / XZ = 5 / 12) : (XZ / YZ = 12 / 13) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_cosine_l1160_116079


namespace NUMINAMATH_GPT_largest_inscribed_square_length_l1160_116034

noncomputable def inscribed_square_length (s : ℝ) (n : ℕ) : ℝ :=
  let t := s / n
  let h := (Real.sqrt 3 / 2) * t
  s - 2 * h

theorem largest_inscribed_square_length :
  inscribed_square_length 12 3 = 12 - 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_inscribed_square_length_l1160_116034


namespace NUMINAMATH_GPT_find_M_l1160_116074

theorem find_M :
  (∃ (M : ℕ), (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M) → M = 1723 :=
  by
  sorry

end NUMINAMATH_GPT_find_M_l1160_116074


namespace NUMINAMATH_GPT_johns_age_l1160_116052

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end NUMINAMATH_GPT_johns_age_l1160_116052


namespace NUMINAMATH_GPT_noemi_start_amount_l1160_116018

/-
  Conditions:
    lost_roulette = -600
    won_blackjack = 400
    lost_poker = -400
    won_baccarat = 500
    meal_cost = 200
    purse_end = 1800

  Prove: start_amount == 2300
-/

noncomputable def lost_roulette : Int := -600
noncomputable def won_blackjack : Int := 400
noncomputable def lost_poker : Int := -400
noncomputable def won_baccarat : Int := 500
noncomputable def meal_cost : Int := 200
noncomputable def purse_end : Int := 1800

noncomputable def net_gain : Int := lost_roulette + won_blackjack + lost_poker + won_baccarat

noncomputable def start_amount : Int := net_gain + meal_cost + purse_end

theorem noemi_start_amount : start_amount = 2300 :=
by
  sorry

end NUMINAMATH_GPT_noemi_start_amount_l1160_116018


namespace NUMINAMATH_GPT_martian_year_length_ratio_l1160_116008

theorem martian_year_length_ratio :
  let EarthDay := 24 -- hours
  let MarsDay := EarthDay + 2 / 3 -- hours (since 40 minutes is 2/3 of an hour)
  let MartianYearDays := 668
  let EarthYearDays := 365.25
  (MartianYearDays * MarsDay) / EarthYearDays = 1.88 := by
{
  sorry
}

end NUMINAMATH_GPT_martian_year_length_ratio_l1160_116008


namespace NUMINAMATH_GPT_sum_algebra_values_l1160_116010

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | _ => 1

theorem sum_algebra_values : 
  alphabet_value 1 + 
  alphabet_value 12 + 
  alphabet_value 7 +
  alphabet_value 5 +
  alphabet_value 2 +
  alphabet_value 18 +
  alphabet_value 1 
  = 5 := by
  sorry

end NUMINAMATH_GPT_sum_algebra_values_l1160_116010


namespace NUMINAMATH_GPT_books_sold_l1160_116036

def initial_books : ℕ := 134
def given_books : ℕ := 39
def books_left : ℕ := 68

theorem books_sold : (initial_books - given_books - books_left = 27) := 
by 
  sorry

end NUMINAMATH_GPT_books_sold_l1160_116036


namespace NUMINAMATH_GPT_triangle_area_given_conditions_l1160_116092

theorem triangle_area_given_conditions (a b c A B S : ℝ) (h₁ : (2 * c - b) * Real.cos A = a * Real.cos B) (h₂ : b = 1) (h₃ : c = 2) :
  S = (1 / 2) * b * c * Real.sin A → S = Real.sqrt 3 / 2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_area_given_conditions_l1160_116092


namespace NUMINAMATH_GPT_number_of_blue_candles_l1160_116061

def total_candles : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def blue_candles : ℕ := total_candles - (yellow_candles + red_candles)

theorem number_of_blue_candles : blue_candles = 38 :=
by
  unfold blue_candles
  unfold total_candles yellow_candles red_candles
  sorry

end NUMINAMATH_GPT_number_of_blue_candles_l1160_116061


namespace NUMINAMATH_GPT_velocity_of_current_correct_l1160_116053

-- Definitions based on the conditions in the problem
def rowing_speed_in_still_water : ℝ := 10
def distance_to_place : ℝ := 24
def total_time_round_trip : ℝ := 5

-- Define the velocity of the current
def velocity_of_current : ℝ := 2

-- Main theorem statement
theorem velocity_of_current_correct :
  ∃ (v : ℝ), (v = 2) ∧ 
  (total_time_round_trip = (distance_to_place / (rowing_speed_in_still_water + v) + 
                            distance_to_place / (rowing_speed_in_still_water - v))) :=
by {
  sorry
}

end NUMINAMATH_GPT_velocity_of_current_correct_l1160_116053


namespace NUMINAMATH_GPT_find_numbers_l1160_116067

theorem find_numbers (a b c : ℕ) (h : a + b = 2015) (h' : a = 10 * b + c) (hc : 0 ≤ c ∧ c ≤ 9) :
  (a = 1832 ∧ b = 183) :=
sorry

end NUMINAMATH_GPT_find_numbers_l1160_116067


namespace NUMINAMATH_GPT_tangent_condition_l1160_116028

theorem tangent_condition (a b : ℝ) :
  (4 * a^2 + b^2 = 1) ↔ 
  ∀ x y : ℝ, (y = 2 * x + 1) → ((x^2 / a^2) + (y^2 / b^2) = 1) → (∃! y, y = 2 * x + 1 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end NUMINAMATH_GPT_tangent_condition_l1160_116028


namespace NUMINAMATH_GPT_probability_of_yellow_face_l1160_116098

theorem probability_of_yellow_face :
  let total_faces : ℕ := 10
  let yellow_faces : ℕ := 4
  (yellow_faces : ℚ) / (total_faces : ℚ) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_face_l1160_116098


namespace NUMINAMATH_GPT_total_prairie_area_l1160_116020

theorem total_prairie_area (A B C : ℕ) (Z1 Z2 Z3 : ℚ) (unaffected : ℕ) (total_area : ℕ) : 
  A = 55000 →
  B = 35000 →
  C = 45000 →
  Z1 = 0.80 →
  Z2 = 0.60 →
  Z3 = 0.95 →
  unaffected = 1500 →
  total_area = Z1 * A + Z2 * B + Z3 * C + unaffected →
  total_area = 109250 := sorry

end NUMINAMATH_GPT_total_prairie_area_l1160_116020


namespace NUMINAMATH_GPT_tens_digit_of_square_ending_in_six_odd_l1160_116090

theorem tens_digit_of_square_ending_in_six_odd 
   (N : ℤ) 
   (a : ℤ) 
   (b : ℕ) 
   (hle : 0 ≤ b) 
   (hge : b < 10) 
   (hexp : N = 10 * a + b) 
   (hsqr : (N^2) % 10 = 6) : 
   ∃ k : ℕ, (N^2 / 10) % 10 = 2 * k + 1 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_tens_digit_of_square_ending_in_six_odd_l1160_116090


namespace NUMINAMATH_GPT_transition_algebraic_expression_l1160_116069

theorem transition_algebraic_expression (k : ℕ) (hk : k > 0) :
  (k + 1 + k) * (k + 1 + k + 1) / (k + 1) = 4 * k + 2 :=
sorry

end NUMINAMATH_GPT_transition_algebraic_expression_l1160_116069


namespace NUMINAMATH_GPT_ambulance_ride_cost_is_correct_l1160_116091

-- Define all the constants and conditions
def daily_bed_cost : ℝ := 900
def bed_days : ℕ := 3
def specialist_rate_per_hour : ℝ := 250
def specialist_minutes_per_day : ℕ := 15
def specialists_count : ℕ := 2
def total_bill : ℝ := 4625

noncomputable def ambulance_cost : ℝ :=
  total_bill - ((daily_bed_cost * bed_days) + (specialist_rate_per_hour * (specialist_minutes_per_day / 60) * specialists_count))

-- The proof statement
theorem ambulance_ride_cost_is_correct : ambulance_cost = 1675 := by
  sorry

end NUMINAMATH_GPT_ambulance_ride_cost_is_correct_l1160_116091


namespace NUMINAMATH_GPT_binom_600_eq_1_l1160_116007

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end NUMINAMATH_GPT_binom_600_eq_1_l1160_116007


namespace NUMINAMATH_GPT_range_of_a_in_second_quadrant_l1160_116029

theorem range_of_a_in_second_quadrant :
  (∀ (x y : ℝ), x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0 → x < 0 ∧ y > 0) → (0 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_second_quadrant_l1160_116029


namespace NUMINAMATH_GPT_sum_ninth_power_l1160_116064

theorem sum_ninth_power (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) 
                        (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7)
                        (h5 : a^5 + b^5 = 11)
                        (h_ind : ∀ n, n ≥ 3 → a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)) :
  a^9 + b^9 = 76 :=
by
  sorry

end NUMINAMATH_GPT_sum_ninth_power_l1160_116064


namespace NUMINAMATH_GPT_determinant_zero_l1160_116078

noncomputable def matrix_A (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin θ, -Real.cos θ],
    ![-2 * Real.sin θ, 0, Real.sin φ],
    ![Real.cos θ, -Real.sin φ, 0]]

theorem determinant_zero (θ φ : ℝ) : Matrix.det (matrix_A θ φ) = 0 := by
  sorry

end NUMINAMATH_GPT_determinant_zero_l1160_116078


namespace NUMINAMATH_GPT_value_of_x_l1160_116009

theorem value_of_x (x y : ℕ) (h1 : y = 864) (h2 : x^3 * 6^3 / 432 = y) : x = 12 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1160_116009


namespace NUMINAMATH_GPT_probability_even_sum_l1160_116073

open Nat

def balls : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def even_sum_probability : ℚ :=
  let total_outcomes := 12 * 11
  let even_balls := balls.filter (λ n => n % 2 = 0)
  let odd_balls := balls.filter (λ n => n % 2 = 1)
  let even_outcomes := even_balls.length * (even_balls.length - 1)
  let odd_outcomes := odd_balls.length * (odd_balls.length - 1)
  let favorable_outcomes := even_outcomes + odd_outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_even_sum :
  even_sum_probability = 5 / 11 := by
  sorry

end NUMINAMATH_GPT_probability_even_sum_l1160_116073


namespace NUMINAMATH_GPT_possibleValues_set_l1160_116051

noncomputable def possibleValues (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 3) : Set ℝ :=
  {x | x = 1/a + 1/b}

theorem possibleValues_set :
  ∀ a b : ℝ, (0 < a ∧ 0 < b) → (a + b = 3) → possibleValues a b (by sorry) (by sorry) = {x | ∃ y, y ≥ 4/3 ∧ x = y} :=
by
  sorry

end NUMINAMATH_GPT_possibleValues_set_l1160_116051


namespace NUMINAMATH_GPT_ratio_x_to_y_l1160_116055

theorem ratio_x_to_y (x y : ℤ) (h : (10*x - 3*y) / (13*x - 2*y) = 3 / 5) : x / y = 9 / 11 := 
by sorry

end NUMINAMATH_GPT_ratio_x_to_y_l1160_116055


namespace NUMINAMATH_GPT_negation_log2_property_l1160_116037

theorem negation_log2_property :
  ¬(∃ x₀ : ℝ, Real.log x₀ / Real.log 2 ≤ 0) ↔ ∀ x : ℝ, Real.log x / Real.log 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_log2_property_l1160_116037
