import Mathlib

namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l836_83607

theorem right_triangle_hypotenuse_length (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 10 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l836_83607


namespace NUMINAMATH_GPT_solve_system_l836_83614

theorem solve_system :
  (∀ x y : ℝ, 
    (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0 ∧
     x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0) → (x, y) = (2, 1)) :=
by simp [← solve_system]; sorry

end NUMINAMATH_GPT_solve_system_l836_83614


namespace NUMINAMATH_GPT_hulk_jump_geometric_sequence_l836_83675

theorem hulk_jump_geometric_sequence (n : ℕ) (a_n : ℕ) : 
  (a_n = 3 * 2^(n - 1)) → (a_n > 3000) → n = 11 :=
by
  sorry

end NUMINAMATH_GPT_hulk_jump_geometric_sequence_l836_83675


namespace NUMINAMATH_GPT_parametric_line_segment_computation_l836_83609

theorem parametric_line_segment_computation :
  ∃ (a b c d : ℝ), 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   (-3, 10) = (a * t + b, c * t + d) ∧
   (4, 16) = (a * 1 + b, c * 1 + d)) ∧
  (b = -3) ∧ (d = 10) ∧ 
  (a + b = 4) ∧ (c + d = 16) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 194) :=
sorry

end NUMINAMATH_GPT_parametric_line_segment_computation_l836_83609


namespace NUMINAMATH_GPT_georgina_teaches_2_phrases_per_week_l836_83670

theorem georgina_teaches_2_phrases_per_week
    (total_phrases : ℕ) 
    (initial_phrases : ℕ) 
    (days_owned : ℕ)
    (phrases_per_week : ℕ):
    total_phrases = 17 → 
    initial_phrases = 3 → 
    days_owned = 49 → 
    phrases_per_week = (total_phrases - initial_phrases) / (days_owned / 7) → 
    phrases_per_week = 2 := 
by
  intros h_total h_initial h_days h_calc
  rw [h_total, h_initial, h_days] at h_calc
  sorry  -- Proof to be filled

end NUMINAMATH_GPT_georgina_teaches_2_phrases_per_week_l836_83670


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l836_83687

theorem arithmetic_sequence_difference (a d : ℕ) (n m : ℕ) (hnm : m > n) (h_a : a = 3) (h_d : d = 7) (h_n : n = 1001) (h_m : m = 1004) :
  (a + (m - 1) * d) - (a + (n - 1) * d) = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_difference_l836_83687


namespace NUMINAMATH_GPT_solve_for_F_l836_83639

variable (S W F : ℝ)

def condition1 (S W : ℝ) : Prop := S = W / 3
def condition2 (W F : ℝ) : Prop := W = F + 60
def condition3 (S W F : ℝ) : Prop := S + W + F = 150

theorem solve_for_F (S W F : ℝ) (h1 : condition1 S W) (h2 : condition2 W F) (h3 : condition3 S W F) : F = 52.5 :=
sorry

end NUMINAMATH_GPT_solve_for_F_l836_83639


namespace NUMINAMATH_GPT_container_capacity_l836_83640

theorem container_capacity (C : ℝ) 
  (h1 : (0.30 * C : ℝ) + 27 = 0.75 * C) : C = 60 :=
sorry

end NUMINAMATH_GPT_container_capacity_l836_83640


namespace NUMINAMATH_GPT_math_problem_l836_83621

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
noncomputable def y : ℝ := (Real.sqrt 5 - 1) / 2

theorem math_problem :
    x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l836_83621


namespace NUMINAMATH_GPT_fraction_evaluation_l836_83679

def number_of_primes_between_10_and_30 : ℕ := 6

theorem fraction_evaluation : (number_of_primes_between_10_and_30^2 - 4) / (number_of_primes_between_10_and_30 + 2) = 4 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l836_83679


namespace NUMINAMATH_GPT_work_completion_days_l836_83643

theorem work_completion_days (A B : ℕ) (hA : A = 20) (hB : B = 20) : A + B / (A + B) / 2 = 10 :=
by 
  rw [hA, hB]
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_work_completion_days_l836_83643


namespace NUMINAMATH_GPT_combinatorial_solution_l836_83645

theorem combinatorial_solution (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 14)
  (h3 : 0 ≤ 2 * x - 4) (h4 : 2 * x - 4 ≤ 14) : x = 4 ∨ x = 6 := by
  sorry

end NUMINAMATH_GPT_combinatorial_solution_l836_83645


namespace NUMINAMATH_GPT_son_l836_83684

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 24) 
  (h2 : M + 2 = 2 * (S + 2)) : S = 22 := 
by 
  sorry

end NUMINAMATH_GPT_son_l836_83684


namespace NUMINAMATH_GPT_factors_and_multiple_of_20_l836_83628

-- Define the relevant numbers
def a := 20
def b := 5
def c := 4

-- Given condition: the equation 20 / 5 = 4
def condition : Prop := a / b = c

-- Factors and multiples relationships to prove
def are_factors : Prop := a % b = 0 ∧ a % c = 0
def is_multiple : Prop := b * c = a

-- The main statement combining everything
theorem factors_and_multiple_of_20 (h : condition) : are_factors ∧ is_multiple :=
sorry

end NUMINAMATH_GPT_factors_and_multiple_of_20_l836_83628


namespace NUMINAMATH_GPT_largest_even_number_l836_83630

theorem largest_even_number (x : ℕ) (h : x + (x+2) + (x+4) = 1194) : x + 4 = 400 :=
by
  have : 3*x + 6 = 1194 := by linarith
  have : 3*x = 1188 := by linarith
  have : x = 396 := by linarith
  linarith

end NUMINAMATH_GPT_largest_even_number_l836_83630


namespace NUMINAMATH_GPT_min_positive_announcements_l836_83669

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 90)
  (h2 : y * (y - 1) + (x - y) * (x - y - 1) = 48) 
  : y = 3 :=
sorry

end NUMINAMATH_GPT_min_positive_announcements_l836_83669


namespace NUMINAMATH_GPT_remainder_when_four_times_n_minus_9_divided_by_11_l836_83666

theorem remainder_when_four_times_n_minus_9_divided_by_11 
  (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_when_four_times_n_minus_9_divided_by_11_l836_83666


namespace NUMINAMATH_GPT_cost_jam_l836_83653

noncomputable def cost_of_jam (N B J : ℕ) : ℝ :=
  N * J * 5 / 100

theorem cost_jam (N B J : ℕ) (h₁ : N > 1) (h₂ : 4 * N + 20 = 414) :
  cost_of_jam N B J = 2.25 := by
  sorry

end NUMINAMATH_GPT_cost_jam_l836_83653


namespace NUMINAMATH_GPT_sum_fiftieth_powers_100_gon_l836_83657

noncomputable def sum_fiftieth_powers_all_sides_and_diagonals (n : ℕ) (R : ℝ) : ℝ := sorry
-- Define the sum of 50-th powers of all the sides and diagonals for a general n-gon inscribed in a circle of radius R

theorem sum_fiftieth_powers_100_gon (R : ℝ) : 
  sum_fiftieth_powers_all_sides_and_diagonals 100 R = sorry := sorry

end NUMINAMATH_GPT_sum_fiftieth_powers_100_gon_l836_83657


namespace NUMINAMATH_GPT_cherries_used_l836_83631

theorem cherries_used (initial remaining used : ℕ) (h_initial : initial = 77) (h_remaining : remaining = 17) (h_used : used = initial - remaining) : used = 60 :=
by
  rw [h_initial, h_remaining] at h_used
  simp at h_used
  exact h_used

end NUMINAMATH_GPT_cherries_used_l836_83631


namespace NUMINAMATH_GPT_find_x_l836_83671

namespace ProofProblem

def δ (x : ℚ) : ℚ := 5 * x + 6
def φ (x : ℚ) : ℚ := 9 * x + 4

theorem find_x (x : ℚ) : (δ (φ x) = 14) ↔ (x = -4 / 15) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_find_x_l836_83671


namespace NUMINAMATH_GPT_work_rate_ab_together_l836_83620

-- Define A, B, and C as the work rates of individuals
variables (A B C : ℝ)

-- We are given the following conditions:
-- 1. a, b, and c together can finish the job in 11 days
-- 2. c alone can finish the job in 41.25 days

-- Given these conditions, we aim to prove that a and b together can finish the job in 15 days
theorem work_rate_ab_together
  (h1 : A + B + C = 1 / 11)
  (h2 : C = 1 / 41.25) :
  1 / (A + B) = 15 :=
by
  sorry

end NUMINAMATH_GPT_work_rate_ab_together_l836_83620


namespace NUMINAMATH_GPT_curved_surface_area_cone_l836_83667

theorem curved_surface_area_cone :
  let r := 8  -- base radius in cm
  let l := 19  -- lateral edge length in cm
  let π := Real.pi
  let CSA := π * r * l
  477.5 < CSA ∧ CSA < 478 := by
  sorry

end NUMINAMATH_GPT_curved_surface_area_cone_l836_83667


namespace NUMINAMATH_GPT_number_satisfying_condition_l836_83642

-- The sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem
theorem number_satisfying_condition : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 :=
by
  sorry

end NUMINAMATH_GPT_number_satisfying_condition_l836_83642


namespace NUMINAMATH_GPT_expression_remainder_l836_83600

theorem expression_remainder (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_expression_remainder_l836_83600


namespace NUMINAMATH_GPT_geometric_sum_S12_l836_83663

theorem geometric_sum_S12 (a r : ℝ) (h₁ : r ≠ 1) (S4_eq : a * (1 - r^4) / (1 - r) = 24) (S8_eq : a * (1 - r^8) / (1 - r) = 36) : a * (1 - r^12) / (1 - r) = 42 := 
sorry

end NUMINAMATH_GPT_geometric_sum_S12_l836_83663


namespace NUMINAMATH_GPT_jam_consumption_l836_83634

theorem jam_consumption (x y t : ℝ) :
  x + y = 100 →
  t = 45 * x / y →
  t = 20 * y / x →
  x = 40 ∧ y = 60 ∧ 
  (y / 45 = 4 / 3) ∧ 
  (x / 20 = 2) := by
  sorry

end NUMINAMATH_GPT_jam_consumption_l836_83634


namespace NUMINAMATH_GPT_rectangle_length_width_l836_83603

theorem rectangle_length_width (x y : ℝ) 
  (h1 : 2 * x + 2 * y = 16) 
  (h2 : x - y = 1) : 
  x = 4.5 ∧ y = 3.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_length_width_l836_83603


namespace NUMINAMATH_GPT_dragon_legs_l836_83629

variable {x y n : ℤ}

theorem dragon_legs :
  (x = 40) ∧
  (y = 9) ∧
  (220 = 40 * x + n * y) →
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_dragon_legs_l836_83629


namespace NUMINAMATH_GPT_final_temp_fahrenheit_correct_l836_83615

noncomputable def initial_temp_celsius : ℝ := 50
noncomputable def conversion_c_to_f (c: ℝ) : ℝ := (c * 9 / 5) + 32
noncomputable def final_temp_celsius := initial_temp_celsius / 2

theorem final_temp_fahrenheit_correct : conversion_c_to_f final_temp_celsius = 77 :=
  by sorry

end NUMINAMATH_GPT_final_temp_fahrenheit_correct_l836_83615


namespace NUMINAMATH_GPT_evaluate_sum_l836_83691

theorem evaluate_sum : (-1:ℤ) ^ 2010 + (-1:ℤ) ^ 2011 + (1:ℤ) ^ 2012 - (1:ℤ) ^ 2013 + (-1:ℤ) ^ 2014 = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_sum_l836_83691


namespace NUMINAMATH_GPT_taller_pot_shadow_length_l836_83665

theorem taller_pot_shadow_length
  (height1 shadow1 height2 : ℝ)
  (h1 : height1 = 20)
  (h2 : shadow1 = 10)
  (h3 : height2 = 40) :
  ∃ shadow2 : ℝ, height2 / shadow2 = height1 / shadow1 ∧ shadow2 = 20 :=
by
  -- Since Lean requires proofs for existential statements,
  -- we add "sorry" to skip the proof.
  sorry

end NUMINAMATH_GPT_taller_pot_shadow_length_l836_83665


namespace NUMINAMATH_GPT_problem_A_problem_C_l836_83646

section
variables {a b : ℝ}

-- A: If a and b are positive real numbers, and a > b, then a^3 + b^3 > a^2 * b + a * b^2.
theorem problem_A (ha : 0 < a) (hb : 0 < b) (h : a > b) : a^3 + b^3 > a^2 * b + a * b^2 := sorry

end

section
variables {a b : ℝ}

-- C: If a and b are real numbers, then "a > b > 0" is a sufficient but not necessary condition for "1/a < 1/b".
theorem problem_C (ha : 0 < a) (hb : 0 < b) (h : a > b) : 1/a < 1/b := sorry

end

end NUMINAMATH_GPT_problem_A_problem_C_l836_83646


namespace NUMINAMATH_GPT_find_x_l836_83637

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉₊ * x = 198) : x = 13.2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l836_83637


namespace NUMINAMATH_GPT_magnitude_of_z_l836_83633

open Complex

theorem magnitude_of_z (z : ℂ) (h : z + I = (2 + I) / I) : abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l836_83633


namespace NUMINAMATH_GPT_set_difference_NM_l836_83654

open Set

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_NM :
  let M := {1, 2, 3, 4, 5}
  let N := {1, 2, 3, 7}
  setDifference N M = {7} :=
by
  sorry

end NUMINAMATH_GPT_set_difference_NM_l836_83654


namespace NUMINAMATH_GPT_fabric_area_l836_83659

theorem fabric_area (length width : ℝ) (h_length : length = 8) (h_width : width = 3) : 
  length * width = 24 := 
by
  rw [h_length, h_width]
  norm_num

end NUMINAMATH_GPT_fabric_area_l836_83659


namespace NUMINAMATH_GPT_sum_of_roots_eq_14_l836_83627

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) → 11 + 3 = 14 :=
by
  intro h
  have x1 : 11 = 11 := rfl
  have x2 : 3 = 3 := rfl
  exact rfl

end NUMINAMATH_GPT_sum_of_roots_eq_14_l836_83627


namespace NUMINAMATH_GPT_prove_a2_a3_a4_sum_l836_83682

theorem prove_a2_a3_a4_sum (a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, a1 * (x-1)^4 + a2 * (x-1)^3 + a3 * (x-1)^2 + a4 * (x-1) + a5 = x^4) :
  a2 + a3 + a4 = 14 :=
sorry

end NUMINAMATH_GPT_prove_a2_a3_a4_sum_l836_83682


namespace NUMINAMATH_GPT_max_value_ab_ac_bc_l836_83678

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_ab_ac_bc_l836_83678


namespace NUMINAMATH_GPT_coin_value_l836_83658

variables (n d q : ℕ)  -- Number of nickels, dimes, and quarters
variable (total_coins : n + d + q = 30)  -- Total coins condition

-- Original value in cents
def original_value : ℕ := 5 * n + 10 * d + 25 * q

-- Swapped values in cents
def swapped_value : ℕ := 10 * n + 25 * d + 5 * q

-- Condition given about the value difference
variable (value_difference : swapped_value = original_value + 150)

-- Prove the total value of coins is $5.00 (500 cents)
theorem coin_value : original_value = 500 :=
by
  sorry

end NUMINAMATH_GPT_coin_value_l836_83658


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_is_negative_336_l836_83606

theorem sum_of_consecutive_integers_product_is_negative_336 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = -336 ∧ (n - 1) + n + (n + 1) = -21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_is_negative_336_l836_83606


namespace NUMINAMATH_GPT_probability_three_even_dice_l836_83624

theorem probability_three_even_dice :
  let p_even := 1 / 2
  let combo := Nat.choose 5 3
  let probability := combo * (p_even ^ 3) * ((1 - p_even) ^ 2)
  probability = 5 / 16 := 
by
  sorry

end NUMINAMATH_GPT_probability_three_even_dice_l836_83624


namespace NUMINAMATH_GPT_complement_of_A_in_U_l836_83644

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l836_83644


namespace NUMINAMATH_GPT_Johnny_is_8_l836_83692

-- Define Johnny's current age
def johnnys_age (x : ℕ) : Prop :=
  x + 2 = 2 * (x - 3)

theorem Johnny_is_8 (x : ℕ) (h : johnnys_age x) : x = 8 :=
sorry

end NUMINAMATH_GPT_Johnny_is_8_l836_83692


namespace NUMINAMATH_GPT_average_marks_all_students_proof_l836_83602

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

end NUMINAMATH_GPT_average_marks_all_students_proof_l836_83602


namespace NUMINAMATH_GPT_polynomial_bound_l836_83648

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (hP : ∀ x : ℝ, |x| < 1 → |P x a b c d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end NUMINAMATH_GPT_polynomial_bound_l836_83648


namespace NUMINAMATH_GPT_cube_edge_length_l836_83661

theorem cube_edge_length (sum_edges length_edge : ℝ) (cube_has_12_edges : 12 * length_edge = sum_edges) (sum_edges_eq_144 : sum_edges = 144) : length_edge = 12 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l836_83661


namespace NUMINAMATH_GPT_kamal_marks_physics_correct_l836_83610

-- Definition of the conditions
def kamal_marks_english : ℕ := 76
def kamal_marks_mathematics : ℕ := 60
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 74
def kamal_num_subjects : ℕ := 5

-- Definition of the total marks
def kamal_total_marks : ℕ := kamal_average_marks * kamal_num_subjects

-- Sum of known marks
def kamal_known_marks : ℕ := kamal_marks_english + kamal_marks_mathematics + kamal_marks_chemistry + kamal_marks_biology

-- The expected result for Physics
def kamal_marks_physics : ℕ := 82

-- Proof statement
theorem kamal_marks_physics_correct :
  kamal_total_marks - kamal_known_marks = kamal_marks_physics :=
by
  simp [kamal_total_marks, kamal_known_marks, kamal_marks_physics]
  sorry

end NUMINAMATH_GPT_kamal_marks_physics_correct_l836_83610


namespace NUMINAMATH_GPT_pat_moved_chairs_l836_83672

theorem pat_moved_chairs (total_chairs : ℕ) (carey_moved : ℕ) (left_to_move : ℕ) (pat_moved : ℕ) :
  total_chairs = 74 →
  carey_moved = 28 →
  left_to_move = 17 →
  pat_moved = total_chairs - left_to_move - carey_moved →
  pat_moved = 29 :=
by
  intros h_total h_carey h_left h_equation
  rw [h_total, h_carey, h_left] at h_equation
  exact h_equation

end NUMINAMATH_GPT_pat_moved_chairs_l836_83672


namespace NUMINAMATH_GPT_permits_increase_l836_83686

theorem permits_increase :
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  new_permits = 67600 * old_permits :=
by
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  exact sorry

end NUMINAMATH_GPT_permits_increase_l836_83686


namespace NUMINAMATH_GPT_om_4_2_eq_18_l836_83636

def om (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem om_4_2_eq_18 : om 4 2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_om_4_2_eq_18_l836_83636


namespace NUMINAMATH_GPT_total_cookies_sold_l836_83604

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end NUMINAMATH_GPT_total_cookies_sold_l836_83604


namespace NUMINAMATH_GPT_gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l836_83656

def gervais_distance_miles_per_day : Real := 315
def gervais_days : Real := 3
def gervais_km_per_mile : Real := 1.60934

def henri_total_miles : Real := 1250
def madeleine_distance_miles_per_day : Real := 100
def madeleine_days : Real := 5

def gervais_total_km := gervais_distance_miles_per_day * gervais_days * gervais_km_per_mile
def henri_total_km := henri_total_miles * gervais_km_per_mile
def madeleine_total_km := madeleine_distance_miles_per_day * madeleine_days * gervais_km_per_mile

def combined_total_km := gervais_total_km + henri_total_km + madeleine_total_km

theorem gervais_km_correct : gervais_total_km = 1520.82405 := sorry
theorem henri_km_correct : henri_total_km = 2011.675 := sorry
theorem madeleine_km_correct : madeleine_total_km = 804.67 := sorry
theorem total_km_correct : combined_total_km = 4337.16905 := sorry
theorem henri_drove_farthest : henri_total_km = 2011.675 := sorry

end NUMINAMATH_GPT_gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l836_83656


namespace NUMINAMATH_GPT_total_bill_amount_l836_83674

theorem total_bill_amount (n : ℕ) (cost_per_meal : ℕ) (gratuity_rate : ℚ) (total_bill_with_gratuity : ℚ)
  (h1 : n = 7) (h2 : cost_per_meal = 100) (h3 : gratuity_rate = 20 / 100) :
  total_bill_with_gratuity = (n * cost_per_meal : ℕ) * (1 + gratuity_rate) :=
sorry

end NUMINAMATH_GPT_total_bill_amount_l836_83674


namespace NUMINAMATH_GPT_min_y_value_l836_83613

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem min_y_value : ∀ x > -1, f x ≥ 9 :=
by sorry

end NUMINAMATH_GPT_min_y_value_l836_83613


namespace NUMINAMATH_GPT_Ali_winning_strategy_l836_83673

def Ali_and_Mohammad_game (m n : ℕ) (a : Fin m → ℕ) : Prop :=
∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ (∃ p : ℕ, Nat.Prime p ∧ m = p^k ∧ n = p^l)

theorem Ali_winning_strategy (m n : ℕ) (a : Fin m → ℕ) :
  Ali_and_Mohammad_game m n a :=
sorry

end NUMINAMATH_GPT_Ali_winning_strategy_l836_83673


namespace NUMINAMATH_GPT_simple_interest_true_discount_l836_83651

theorem simple_interest_true_discount (P R T : ℝ) 
  (h1 : 85 = (P * R * T) / 100)
  (h2 : 80 = (85 * P) / (P + 85)) : P = 1360 :=
sorry

end NUMINAMATH_GPT_simple_interest_true_discount_l836_83651


namespace NUMINAMATH_GPT_train_speed_in_km_per_hour_l836_83647

-- Definitions based on the conditions
def train_length : ℝ := 240  -- The length of the train in meters.
def time_to_pass_tree : ℝ := 8  -- The time to pass the tree in seconds.
def meters_per_second_to_kilometers_per_hour : ℝ := 3.6  -- Conversion factor from meters/second to kilometers/hour.

-- Statement based on the question and the correct answer
theorem train_speed_in_km_per_hour : (train_length / time_to_pass_tree) * meters_per_second_to_kilometers_per_hour = 108 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_per_hour_l836_83647


namespace NUMINAMATH_GPT_VIP_ticket_price_l836_83617

variable (total_savings : ℕ) 
variable (num_VIP_tickets : ℕ)
variable (num_regular_tickets : ℕ)
variable (price_per_regular_ticket : ℕ)
variable (remaining_savings : ℕ)

theorem VIP_ticket_price 
  (h1 : total_savings = 500)
  (h2 : num_VIP_tickets = 2)
  (h3 : num_regular_tickets = 3)
  (h4 : price_per_regular_ticket = 50)
  (h5 : remaining_savings = 150) :
  (total_savings - remaining_savings) - (num_regular_tickets * price_per_regular_ticket) = num_VIP_tickets * 100 := 
by
  sorry

end NUMINAMATH_GPT_VIP_ticket_price_l836_83617


namespace NUMINAMATH_GPT_children_on_bus_after_stops_l836_83694

-- Define the initial number of children and changes at each stop
def initial_children := 128
def first_stop_addition := 67
def second_stop_subtraction := 34
def third_stop_addition := 54

-- Prove that the number of children on the bus after all the stops is 215
theorem children_on_bus_after_stops :
  initial_children + first_stop_addition - second_stop_subtraction + third_stop_addition = 215 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_children_on_bus_after_stops_l836_83694


namespace NUMINAMATH_GPT_solution_of_system_l836_83608

theorem solution_of_system 
  (k : ℝ) (x y : ℝ)
  (h1 : (1 : ℝ) = 2 * 1 - 1)
  (h2 : (1 : ℝ) = k * 1)
  (h3 : k ≠ 0)
  (h4 : 2 * x - y = 1)
  (h5 : k * x - y = 0) : 
  x = 1 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l836_83608


namespace NUMINAMATH_GPT_distance_between_centers_of_circles_l836_83664

theorem distance_between_centers_of_circles (C_1 C_2 : ℝ) : 
  (∀ a : ℝ, (C_1 = a ∧ C_2 = a ∧ (4- a)^2 + (1 - a)^2 = a^2)) → 
  |C_1 - C_2| = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_centers_of_circles_l836_83664


namespace NUMINAMATH_GPT_isabella_exchange_l836_83698

theorem isabella_exchange (d : ℚ) : 
  (8 * d / 5 - 72 = 4 * d) → d = -30 :=
by
  sorry

end NUMINAMATH_GPT_isabella_exchange_l836_83698


namespace NUMINAMATH_GPT_conditions_iff_positive_l836_83625

theorem conditions_iff_positive (a b : ℝ) (h₁ : a + b > 0) (h₂ : ab > 0) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ ab > 0) :=
sorry

end NUMINAMATH_GPT_conditions_iff_positive_l836_83625


namespace NUMINAMATH_GPT_negation_equiv_l836_83649

open Classical

-- Proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 = 0

-- Negation of proposition p
def neg_p : Prop := ∀ x : ℝ, x^2 - x + 1 ≠ 0

-- Statement to prove the equivalence of the negation of p and neg_p
theorem negation_equiv :
  ¬p ↔ neg_p := 
sorry

end NUMINAMATH_GPT_negation_equiv_l836_83649


namespace NUMINAMATH_GPT_ratio_qp_l836_83695

theorem ratio_qp (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 6 → 
    P / (x + 3) + Q / (x * (x - 6)) = (x^2 - 4 * x + 15) / (x * (x + 3) * (x - 6))) : 
  Q / P = 5 := 
sorry

end NUMINAMATH_GPT_ratio_qp_l836_83695


namespace NUMINAMATH_GPT_line_parabola_one_intersection_l836_83693

theorem line_parabola_one_intersection (k : ℝ) : 
  ((∃ (x y : ℝ), y = k * x - 1 ∧ y^2 = 4 * x ∧ (∀ u v : ℝ, u ≠ x → v = k * u - 1 → v^2 ≠ 4 * u)) ↔ (k = 0 ∨ k = 1)) := 
sorry

end NUMINAMATH_GPT_line_parabola_one_intersection_l836_83693


namespace NUMINAMATH_GPT_correct_factorization_from_left_to_right_l836_83618

theorem correct_factorization_from_left_to_right 
  (x a b c m n : ℝ) : 
  (2 * a * b - 2 * a * c = 2 * a * (b - c)) :=
sorry

end NUMINAMATH_GPT_correct_factorization_from_left_to_right_l836_83618


namespace NUMINAMATH_GPT_standard_deviation_distance_l836_83626

-- Definitions and assumptions based on the identified conditions
def mean : ℝ := 12
def std_dev : ℝ := 1.2
def value : ℝ := 9.6

-- Statement to prove
theorem standard_deviation_distance : (value - mean) / std_dev = -2 :=
by sorry

end NUMINAMATH_GPT_standard_deviation_distance_l836_83626


namespace NUMINAMATH_GPT_iceberg_submersion_l836_83611

theorem iceberg_submersion (V_total V_immersed S_total S_submerged : ℝ) :
  convex_polyhedron ∧ floating_on_sea ∧
  V_total > 0 ∧ V_immersed > 0 ∧ S_total > 0 ∧ S_submerged > 0 ∧
  (V_immersed / V_total >= 0.90) ∧ ((S_total - S_submerged) / S_total >= 0.50) :=
sorry

end NUMINAMATH_GPT_iceberg_submersion_l836_83611


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l836_83685

theorem triangle_is_right_triangle 
  (A B C : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = 180)
  (h3 : A / B = 2 / 3)
  (h4 : A / C = 2 / 5) : 
  A = 36 ∧ B = 54 ∧ C = 90 := 
sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l836_83685


namespace NUMINAMATH_GPT_triangle_inequality_l836_83690

theorem triangle_inequality 
  (A B C : ℝ) -- angle measures
  (a b c : ℝ) -- side lengths
  (h1 : a = b * (Real.cos C) + c * (Real.cos B)) 
  (cos_half_C_pos : 0 < Real.cos (C/2)) 
  (cos_half_C_lt_one : Real.cos (C/2) < 1)
  (cos_half_B_pos : 0 < Real.cos (B/2)) 
  (cos_half_B_lt_one : Real.cos (B/2) < 1) :
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l836_83690


namespace NUMINAMATH_GPT_Mary_is_10_years_younger_l836_83660

theorem Mary_is_10_years_younger
  (betty_age : ℕ)
  (albert_age : ℕ)
  (mary_age : ℕ)
  (h1 : albert_age = 2 * mary_age)
  (h2 : albert_age = 4 * betty_age)
  (h_betty : betty_age = 5) :
  (albert_age - mary_age) = 10 :=
  by
  sorry

end NUMINAMATH_GPT_Mary_is_10_years_younger_l836_83660


namespace NUMINAMATH_GPT_power_sum_eq_l836_83668

theorem power_sum_eq (n : ℕ) : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end NUMINAMATH_GPT_power_sum_eq_l836_83668


namespace NUMINAMATH_GPT_solve_for_a_l836_83677

theorem solve_for_a (a : Real) (h_pos : a > 0) (h_eq : (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 18) : 
  a = Real.sqrt (Real.sqrt 14 + 2) := by 
  sorry

end NUMINAMATH_GPT_solve_for_a_l836_83677


namespace NUMINAMATH_GPT_air_conditioned_rooms_fraction_l836_83680

theorem air_conditioned_rooms_fraction (R A : ℝ) (h1 : 3/4 * R = 3/4 * R - 1/4 * R)
                                        (h2 : 2/3 * A = 2/3 * A - 1/3 * A)
                                        (h3 : 1/3 * A = 0.8 * 1/4 * R) :
    A / R = 3 / 5 :=
by
  -- Proof content goes here
  sorry

end NUMINAMATH_GPT_air_conditioned_rooms_fraction_l836_83680


namespace NUMINAMATH_GPT_inequality_solution_l836_83612

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l836_83612


namespace NUMINAMATH_GPT_salt_added_correct_l836_83635

theorem salt_added_correct (x : ℝ)
  (hx : x = 119.99999999999996)
  (initial_salt : ℝ := 0.20 * x)
  (evaporation_volume : ℝ := x - (1/4) * x)
  (additional_water : ℝ := 8)
  (final_volume : ℝ := evaporation_volume + additional_water)
  (final_concentration : ℝ := 1 / 3)
  (final_salt : ℝ := final_concentration * final_volume)
  (salt_added : ℝ := final_salt - initial_salt) :
  salt_added = 8.67 :=
sorry

end NUMINAMATH_GPT_salt_added_correct_l836_83635


namespace NUMINAMATH_GPT_solution_set_inequality_l836_83641

theorem solution_set_inequality (x : ℝ) : 
  (abs (x + 3) - abs (x - 2) ≥ 3) ↔ (x ≥ 1) := 
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_inequality_l836_83641


namespace NUMINAMATH_GPT_weave_mats_l836_83676

theorem weave_mats (m n p q : ℕ) (h1 : m * n = p * q) (h2 : ∀ k, k = n → n * 2 = k * 2) :
  (8 * 2 = 16) :=
by
  -- This is where we would traditionally include the proof steps.
  sorry

end NUMINAMATH_GPT_weave_mats_l836_83676


namespace NUMINAMATH_GPT_shirt_original_price_l836_83632

theorem shirt_original_price (original_price final_price : ℝ) (h1 : final_price = 0.5625 * original_price) 
  (h2 : final_price = 19) : original_price = 33.78 :=
by
  sorry

end NUMINAMATH_GPT_shirt_original_price_l836_83632


namespace NUMINAMATH_GPT_range_of_a_if_f_increasing_l836_83681

theorem range_of_a_if_f_increasing (a : ℝ) :
  (∀ x : ℝ, 3*x^2 + 3*a ≥ 0) → (a ≥ 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_if_f_increasing_l836_83681


namespace NUMINAMATH_GPT_no_int_a_divisible_289_l836_83655

theorem no_int_a_divisible_289 : ¬ ∃ a : ℤ, ∃ k : ℤ, a^2 - 3 * a - 19 = 289 * k :=
by
  sorry

end NUMINAMATH_GPT_no_int_a_divisible_289_l836_83655


namespace NUMINAMATH_GPT_function_decreasing_on_interval_l836_83697

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem function_decreasing_on_interval : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f x₁ ≥ f x₂ := by
  sorry

end NUMINAMATH_GPT_function_decreasing_on_interval_l836_83697


namespace NUMINAMATH_GPT_stickers_per_page_l836_83683

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end NUMINAMATH_GPT_stickers_per_page_l836_83683


namespace NUMINAMATH_GPT_this_year_sales_l836_83623

def last_year_sales : ℝ := 320 -- in millions
def percent_increase : ℝ := 0.5 -- 50%

theorem this_year_sales : (last_year_sales * (1 + percent_increase)) = 480 := by
  sorry

end NUMINAMATH_GPT_this_year_sales_l836_83623


namespace NUMINAMATH_GPT_solve_inequality_l836_83699

theorem solve_inequality :
  {x : ℝ | (x - 1) * (2 * x + 1) ≤ 0} = { x : ℝ | -1/2 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l836_83699


namespace NUMINAMATH_GPT_cauliflower_sales_l836_83619

noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := 16 + (1 / 2 * carrot_sales)
noncomputable def total_sales : ℝ := 380
noncomputable def other_sales : ℝ := broccoli_sales + carrot_sales + spinach_sales

theorem cauliflower_sales :
  total_sales - other_sales = 136 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_cauliflower_sales_l836_83619


namespace NUMINAMATH_GPT_number_of_initial_cards_l836_83696

theorem number_of_initial_cards (x : ℝ) (h1 : x + 276.0 = 580) : x = 304 :=
by
  sorry

end NUMINAMATH_GPT_number_of_initial_cards_l836_83696


namespace NUMINAMATH_GPT_min_value_of_a_l836_83688

noncomputable def smallest_root_sum : ℕ := 78

theorem min_value_of_a (r s t : ℕ) (h1 : r * s * t = 2310) (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) :
  r + s + t = smallest_root_sum :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l836_83688


namespace NUMINAMATH_GPT_no_solution_iff_n_eq_neg2_l836_83650

noncomputable def has_no_solution (n : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬ (n * x + y + z = 2 ∧ 
                  x + n * y + z = 2 ∧ 
                  x + y + n * z = 2)

theorem no_solution_iff_n_eq_neg2 (n : ℝ) : has_no_solution n ↔ n = -2 := by
  sorry

end NUMINAMATH_GPT_no_solution_iff_n_eq_neg2_l836_83650


namespace NUMINAMATH_GPT_segments_do_not_intersect_l836_83616

noncomputable def check_intersection (AP PB BQ QC CR RD DS SA : ℚ) : Bool :=
  (AP / PB) * (BQ / QC) * (CR / RD) * (DS / SA) = 1

theorem segments_do_not_intersect :
  let AP := (3 : ℚ)
  let PB := (6 : ℚ)
  let BQ := (2 : ℚ)
  let QC := (4 : ℚ)
  let CR := (1 : ℚ)
  let RD := (5 : ℚ)
  let DS := (4 : ℚ)
  let SA := (6 : ℚ)
  ¬ check_intersection AP PB BQ QC CR RD DS SA :=
by sorry

end NUMINAMATH_GPT_segments_do_not_intersect_l836_83616


namespace NUMINAMATH_GPT_tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l836_83662

-- Definition for Part (a)
theorem tetrahedron_parallelepiped_areas 
  (S1 S2 S3 S4 P1 P2 P3 : ℝ)
  (h1 : true)
  (h2 : true) :
  S1^2 + S2^2 + S3^2 + S4^2 = P1^2 + P2^2 + P3^2 := 
sorry

-- Definition for Part (b)
theorem tetrahedron_heights_distances 
  (h1 h2 h3 h4 d1 d2 d3 : ℝ)
  (h : true) :
  (1/(h1^2)) + (1/(h2^2)) + (1/(h3^2)) + (1/(h4^2)) = (1/(d1^2)) + (1/(d2^2)) + (1/(d3^2)) := 
sorry

end NUMINAMATH_GPT_tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l836_83662


namespace NUMINAMATH_GPT_cube_cut_problem_l836_83689

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end NUMINAMATH_GPT_cube_cut_problem_l836_83689


namespace NUMINAMATH_GPT_kibble_left_l836_83601

-- Define the initial amount of kibble
def initial_kibble := 3

-- Define the rate at which the cat eats kibble
def kibble_rate := 1 / 4

-- Define the time Kira was away
def time_away := 8

-- Define the amount of kibble eaten by the cat during the time away
def kibble_eaten := (time_away * kibble_rate)

-- Define the remaining kibble in the bowl
def remaining_kibble := initial_kibble - kibble_eaten

-- State and prove that the remaining amount of kibble is 1 pound
theorem kibble_left : remaining_kibble = 1 := by
  sorry

end NUMINAMATH_GPT_kibble_left_l836_83601


namespace NUMINAMATH_GPT_calculate_S_value_l836_83605

def operation_S (a b : ℕ) : ℕ := 4 * a + 7 * b

theorem calculate_S_value : operation_S 8 3 = 53 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_calculate_S_value_l836_83605


namespace NUMINAMATH_GPT_unique_y_for_star_eq_9_l836_83622

def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

theorem unique_y_for_star_eq_9 : ∃! y : ℝ, star 2 y = 9 := by
  sorry

end NUMINAMATH_GPT_unique_y_for_star_eq_9_l836_83622


namespace NUMINAMATH_GPT_find_special_numbers_l836_83652

def is_digit_sum_equal (n m : Nat) : Prop := 
  (n.digits 10).sum = (m.digits 10).sum

def is_valid_number (n : Nat) : Prop := 
  100 ≤ n ∧ n ≤ 999 ∧ is_digit_sum_equal n (6 * n)

theorem find_special_numbers :
  {n : Nat | is_valid_number n} = {117, 135} :=
sorry

end NUMINAMATH_GPT_find_special_numbers_l836_83652


namespace NUMINAMATH_GPT_find_x_unique_l836_83638

def productOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of product of digits function
  sorry

def sumOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of sum of digits function
  sorry

theorem find_x_unique : ∀ x : ℕ, (productOfDigits x = 44 * x - 86868 ∧ ∃ n : ℕ, sumOfDigits x = n^3) -> x = 1989 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_unique_l836_83638
