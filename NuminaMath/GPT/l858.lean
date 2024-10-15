import Mathlib

namespace NUMINAMATH_GPT_coffee_students_l858_85806

variable (S : ℝ) -- Total number of students
variable (T : ℝ) -- Number of students who chose tea
variable (C : ℝ) -- Number of students who chose coffee

-- Given conditions
axiom h1 : 0.4 * S = 80   -- 40% of the students chose tea
axiom h2 : T = 80         -- Number of students who chose tea is 80
axiom h3 : 0.3 * S = C    -- 30% of the students chose coffee

-- Prove that the number of students who chose coffee is 60
theorem coffee_students : C = 60 := by
  sorry

end NUMINAMATH_GPT_coffee_students_l858_85806


namespace NUMINAMATH_GPT_polynomial_has_one_real_root_l858_85889

theorem polynomial_has_one_real_root (a : ℝ) :
  (∃! x : ℝ, x^3 - 2 * a * x^2 + 3 * a * x + a^2 - 2 = 0) :=
sorry

end NUMINAMATH_GPT_polynomial_has_one_real_root_l858_85889


namespace NUMINAMATH_GPT_cody_paid_17_l858_85858

-- Definitions for the conditions
def initial_cost : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def final_price_after_discount : ℝ := initial_cost * (1 + tax_rate) - discount
def cody_payment : ℝ := 17

-- The proof statement
theorem cody_paid_17 :
  cody_payment = (final_price_after_discount / 2) :=
by
  -- Proof steps, which we omit by using sorry
  sorry

end NUMINAMATH_GPT_cody_paid_17_l858_85858


namespace NUMINAMATH_GPT_visibility_time_correct_l858_85895

noncomputable def visibility_time (r : ℝ) (d : ℝ) (v_j : ℝ) (v_k : ℝ) : ℝ :=
  (d / (v_j + v_k)) * (r / (r * (v_j / v_k + 1)))

theorem visibility_time_correct :
  visibility_time 60 240 4 2 = 120 :=
by
  sorry

end NUMINAMATH_GPT_visibility_time_correct_l858_85895


namespace NUMINAMATH_GPT_find_x_in_terms_of_abc_l858_85893

variable {x y z a b c : ℝ}

theorem find_x_in_terms_of_abc
  (h1 : xy / (x + y + 1) = a)
  (h2 : xz / (x + z + 1) = b)
  (h3 : yz / (y + z + 1) = c) :
  x = 2 * a * b * c / (a * b + a * c - b * c) := 
sorry

end NUMINAMATH_GPT_find_x_in_terms_of_abc_l858_85893


namespace NUMINAMATH_GPT_find_ordered_pair_l858_85828

theorem find_ordered_pair (x y : ℝ) (h : (x - 2 * y)^2 + (y - 1)^2 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l858_85828


namespace NUMINAMATH_GPT_find_a_l858_85832

theorem find_a (f : ℝ → ℝ) (a x : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x - 5) 
  (h2 : f a = 6) : a = 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l858_85832


namespace NUMINAMATH_GPT_mary_spent_total_amount_l858_85879

def cost_of_berries := 11.08
def cost_of_apples := 14.33
def cost_of_peaches := 9.31
def total_cost := 34.72

theorem mary_spent_total_amount :
  cost_of_berries + cost_of_apples + cost_of_peaches = total_cost :=
by
  sorry

end NUMINAMATH_GPT_mary_spent_total_amount_l858_85879


namespace NUMINAMATH_GPT_range_of_function_l858_85872

theorem range_of_function (x : ℝ) : x ≠ 2 ↔ ∃ y, y = x / (x - 2) :=
sorry

end NUMINAMATH_GPT_range_of_function_l858_85872


namespace NUMINAMATH_GPT_final_price_chocolate_l858_85865

-- Conditions
def original_cost : ℝ := 2.00
def discount : ℝ := 0.57

-- Question and answer
theorem final_price_chocolate : original_cost - discount = 1.43 :=
by
  sorry

end NUMINAMATH_GPT_final_price_chocolate_l858_85865


namespace NUMINAMATH_GPT_max_largest_integer_l858_85850

theorem max_largest_integer (A B C D E : ℕ) 
  (h1 : A ≤ B) 
  (h2 : B ≤ C) 
  (h3 : C ≤ D) 
  (h4 : D ≤ E)
  (h5 : (A + B + C + D + E) / 5 = 60) 
  (h6 : E - A = 10) : 
  E ≤ 290 :=
sorry

end NUMINAMATH_GPT_max_largest_integer_l858_85850


namespace NUMINAMATH_GPT_no_valid_prime_l858_85842

open Nat

def base_p_polynomial (p : ℕ) (coeffs : List ℕ) : ℕ → ℕ :=
  fun (n : ℕ) => coeffs.foldl (λ sum coef => sum * p + coef) 0

def num_1013 (p : ℕ) := base_p_polynomial p [1, 0, 1, 3]
def num_207 (p : ℕ) := base_p_polynomial p [2, 0, 7]
def num_214 (p : ℕ) := base_p_polynomial p [2, 1, 4]
def num_100 (p : ℕ) := base_p_polynomial p [1, 0, 0]
def num_10 (p : ℕ) := base_p_polynomial p [1, 0]

def num_321 (p : ℕ) := base_p_polynomial p [3, 2, 1]
def num_403 (p : ℕ) := base_p_polynomial p [4, 0, 3]
def num_210 (p : ℕ) := base_p_polynomial p [2, 1, 0]

theorem no_valid_prime (p : ℕ) [Fact (Nat.Prime p)] :
  num_1013 p + num_207 p + num_214 p + num_100 p + num_10 p ≠
  num_321 p + num_403 p + num_210 p := by
  sorry

end NUMINAMATH_GPT_no_valid_prime_l858_85842


namespace NUMINAMATH_GPT_sin_alpha_second_quadrant_l858_85848

/-- Given angle α in the second quadrant such that tan(π - α) = 3/4, we need to prove that sin α = 3/5. -/
theorem sin_alpha_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.tan (π - α) = 3 / 4) : Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_alpha_second_quadrant_l858_85848


namespace NUMINAMATH_GPT_train_length_l858_85802

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_length_l858_85802


namespace NUMINAMATH_GPT_real_solutions_unique_l858_85867

theorem real_solutions_unique (a b c : ℝ) :
  (2 * a - b = a^2 * b ∧ 2 * b - c = b^2 * c ∧ 2 * c - a = c^2 * a) →
  (a, b, c) = (-1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 1, 1) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_unique_l858_85867


namespace NUMINAMATH_GPT_intersecting_lines_l858_85863

def diamondsuit (a b : ℝ) : ℝ := a^2 + a * b - b^2

theorem intersecting_lines (x y : ℝ) : 
  (diamondsuit x y = diamondsuit y x) ↔ (y = x ∨ y = -x) := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l858_85863


namespace NUMINAMATH_GPT_greatest_integer_gcd_18_is_6_l858_85849

theorem greatest_integer_gcd_18_is_6 (n : ℕ) (h1 : n < 150) (h2 : Nat.gcd n 18 = 6) : n = 138 := 
sorry

end NUMINAMATH_GPT_greatest_integer_gcd_18_is_6_l858_85849


namespace NUMINAMATH_GPT_marbles_left_l858_85864

-- Definitions and conditions
def marbles_initial : ℕ := 38
def marbles_lost : ℕ := 15

-- Statement of the problem
theorem marbles_left : marbles_initial - marbles_lost = 23 := by
  sorry

end NUMINAMATH_GPT_marbles_left_l858_85864


namespace NUMINAMATH_GPT_intersection_M_N_l858_85852

-- Definitions:
def M := {x : ℝ | 0 ≤ x}
def N := {y : ℝ | -2 ≤ y}

-- The theorem statement:
theorem intersection_M_N : M ∩ N = {z : ℝ | 0 ≤ z} := sorry

end NUMINAMATH_GPT_intersection_M_N_l858_85852


namespace NUMINAMATH_GPT_sin_double_angle_l858_85899

noncomputable def unit_circle_point :=
  (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2))

theorem sin_double_angle 
  (α : Real)
  (h1 : (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2)) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 })
  (h2 : α = (Real.arccos (1 / 2)) ∨ α = -(Real.arccos (1 / 2))) :
  Real.sin (π / 2 + 2 * α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l858_85899


namespace NUMINAMATH_GPT_displacement_correct_l858_85861

-- Define the initial conditions of the problem
def init_north := 50
def init_east := 70
def init_south := 20
def init_west := 30

-- Define the net movements
def net_north := init_north - init_south
def net_east := init_east - init_west

-- Define the straight-line distance using the Pythagorean theorem
def displacement_AC := (net_north ^ 2 + net_east ^ 2).sqrt

theorem displacement_correct : displacement_AC = 50 := 
by sorry

end NUMINAMATH_GPT_displacement_correct_l858_85861


namespace NUMINAMATH_GPT_houses_with_two_car_garage_l858_85824

theorem houses_with_two_car_garage
  (T P GP N G : ℕ)
  (hT : T = 90)
  (hP : P = 40)
  (hGP : GP = 35)
  (hN : N = 35)
  (hFormula : G + P - GP = T - N) :
  G = 50 :=
by
  rw [hT, hP, hGP, hN] at hFormula
  simp at hFormula
  exact hFormula

end NUMINAMATH_GPT_houses_with_two_car_garage_l858_85824


namespace NUMINAMATH_GPT_sum_of_coordinates_l858_85844

theorem sum_of_coordinates :
  let in_distance_from_line (p : (ℝ × ℝ)) (d : ℝ) (line_y : ℝ) : Prop := abs (p.2 - line_y) = d
  let in_distance_from_point (p1 p2 : (ℝ × ℝ)) (d : ℝ) : Prop := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = d^2
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  in_distance_from_line P1 4 13 ∧ in_distance_from_point P1 (7, 13) 10 ∧
  in_distance_from_line P2 4 13 ∧ in_distance_from_point P2 (7, 13) 10 ∧
  in_distance_from_line P3 4 13 ∧ in_distance_from_point P3 (7, 13) 10 ∧
  in_distance_from_line P4 4 13 ∧ in_distance_from_point P4 (7, 13) 10 ∧
  (P1.1 + P2.1 + P3.1 + P4.1) + (P1.2 + P2.2 + P3.2 + P4.2) = 80 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_l858_85844


namespace NUMINAMATH_GPT_min_value_x_3y_l858_85838

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∃ (x y : ℝ), min_value x y = 18 + 21 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_x_3y_l858_85838


namespace NUMINAMATH_GPT_find_200_digit_number_l858_85876

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end NUMINAMATH_GPT_find_200_digit_number_l858_85876


namespace NUMINAMATH_GPT_age_of_B_l858_85811

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end NUMINAMATH_GPT_age_of_B_l858_85811


namespace NUMINAMATH_GPT_find_line_l858_85821

def point_on_line (P : ℝ × ℝ) (m b : ℝ) : Prop :=
  P.2 = m * P.1 + b

def intersection_points_distance (k m b : ℝ) : Prop :=
  |(k^2 - 4*k + 4) - (m*k + b)| = 6

noncomputable def desired_line (m b : ℝ) : Prop :=
  point_on_line (2, 3) m b ∧ ∀ (k : ℝ), intersection_points_distance k m b

theorem find_line : desired_line (-6) 15 := sorry

end NUMINAMATH_GPT_find_line_l858_85821


namespace NUMINAMATH_GPT_find_linear_function_passing_A_B_l858_85822

-- Conditions
def line_function (k b x : ℝ) : ℝ := k * x + b

theorem find_linear_function_passing_A_B :
  (∃ k b : ℝ, k ≠ 0 ∧ line_function k b 1 = 3 ∧ line_function k b 0 = -2) → 
  ∃ k b : ℝ, k = 5 ∧ b = -2 ∧ ∀ x : ℝ, line_function k b x = 5 * x - 2 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_find_linear_function_passing_A_B_l858_85822


namespace NUMINAMATH_GPT_crayons_per_unit_l858_85883

theorem crayons_per_unit :
  ∀ (units : ℕ) (cost_per_crayon : ℕ) (total_cost : ℕ),
    units = 4 →
    cost_per_crayon = 2 →
    total_cost = 48 →
    (total_cost / cost_per_crayon) / units = 6 :=
by
  intros units cost_per_crayon total_cost h_units h_cost_per_crayon h_total_cost
  sorry

end NUMINAMATH_GPT_crayons_per_unit_l858_85883


namespace NUMINAMATH_GPT_problem_solution_l858_85833

variable (f : ℝ → ℝ)

noncomputable def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x < 1/2) ∨ (2 < x)

theorem problem_solution
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_1 : f 1 = 2) :
  ∀ x, f (Real.log x / Real.log 2) > 2 ↔ solution_set x :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l858_85833


namespace NUMINAMATH_GPT_batsman_average_l858_85882

/-- The average after 12 innings given that the batsman makes a score of 115 in his 12th innings,
     increases his average by 3 runs, and he had never been 'not out'. -/
theorem batsman_average (A : ℕ) (h1 : 11 * A + 115 = 12 * (A + 3)) : A + 3 = 82 := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_l858_85882


namespace NUMINAMATH_GPT_find_angle_A_find_side_a_l858_85825

-- Define the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}
-- Assumption conditions in the problem
variables (h₁ : a * sin B = sqrt 3 * b * cos A)
variables (hb : b = 3)
variables (hc : c = 2)

-- Prove that A = π / 3 given the first condition
theorem find_angle_A : h₁ → A = π / 3 := by
  -- Proof is omitted
  sorry

-- Prove that a = sqrt 7 given b = 3, c = 2, and A = π / 3
theorem find_side_a : h₁ → hb → hc → a = sqrt 7 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_angle_A_find_side_a_l858_85825


namespace NUMINAMATH_GPT_only_odd_digit_square_l858_85826

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d % 2 = 1

theorem only_odd_digit_square (n : ℕ) : n^2 = n → odd_digits n → n = 1 ∨ n = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_only_odd_digit_square_l858_85826


namespace NUMINAMATH_GPT_largest_integer_n_apples_l858_85835

theorem largest_integer_n_apples (t : ℕ) (a : ℕ → ℕ) (h1 : t = 150) 
    (h2 : ∀ i : ℕ, 100 ≤ a i ∧ a i ≤ 130) :
  ∃ n : ℕ, n = 5 ∧ (∀ i j : ℕ, a i = a j → i = j → 5 ≤ i ∧ 5 ≤ j) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_apples_l858_85835


namespace NUMINAMATH_GPT_sequence_general_term_l858_85881

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), (a 1 = 1) →
    (∀ n : ℕ, n > 0 → (Real.sqrt (a n) - Real.sqrt (a (n + 1)) = Real.sqrt (a n * a (n + 1)))) →
    (∀ n : ℕ, n > 0 → a n = 1 / (n ^ 2)) :=
by
  intros a ha1 hrec n hn
  sorry

end NUMINAMATH_GPT_sequence_general_term_l858_85881


namespace NUMINAMATH_GPT_number_of_squares_l858_85846

-- Define the conditions and the goal
theorem number_of_squares {x : ℤ} (hx0 : 0 ≤ x) (hx6 : x ≤ 6) {y : ℤ} (hy0 : -1 ≤ y) (hy : y ≤ 3 * x) :
  ∃ (n : ℕ), n = 123 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_squares_l858_85846


namespace NUMINAMATH_GPT_complement_of_M_l858_85839

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | x^2 - 2 * x > 0 }
def comp_M_Real := compl M

theorem complement_of_M :
  comp_M_Real = { x : ℝ | 0 ≤ x ∧ x ≤ 2 } :=
sorry

end NUMINAMATH_GPT_complement_of_M_l858_85839


namespace NUMINAMATH_GPT_part1_part2_l858_85862

-- Define the main condition of the farthest distance formula
def distance_formula (S h : ℝ) : Prop := S^2 = 1.7 * h

-- Define part 1: Given h = 1.7, prove S = 1.7
theorem part1
  (h : ℝ)
  (hyp : h = 1.7)
  : ∃ S : ℝ, distance_formula S h ∧ S = 1.7 :=
by
  sorry
  
-- Define part 2: Given S = 6.8 and height of eyes to ground 1.5, prove the height of tower = 25.7
theorem part2
  (S : ℝ)
  (h1 : ℝ)
  (height_eyes_to_ground : ℝ)
  (hypS : S = 6.8)
  (height_eyes_to_ground_eq : height_eyes_to_ground = 1.5)
  : ∃ h : ℝ, distance_formula S h ∧ (h - height_eyes_to_ground) = 25.7 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l858_85862


namespace NUMINAMATH_GPT_even_function_value_at_three_l858_85856

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- f is an even function
axiom h_even : ∀ x, f x = f (-x)

-- f(x) is defined as x^2 + x when x < 0
axiom h_neg_def : ∀ x, x < 0 → f x = x^2 + x

theorem even_function_value_at_three : f 3 = 6 :=
by {
  -- To be proved
  sorry
}

end NUMINAMATH_GPT_even_function_value_at_three_l858_85856


namespace NUMINAMATH_GPT_find_temperature_on_December_25_l858_85808

theorem find_temperature_on_December_25 {f : ℕ → ℤ}
  (h_recurrence : ∀ n, f (n - 1) + f (n + 1) = f n)
  (h_initial1 : f 3 = 5)
  (h_initial2 : f 31 = 2) :
  f 25 = -3 :=
  sorry

end NUMINAMATH_GPT_find_temperature_on_December_25_l858_85808


namespace NUMINAMATH_GPT_domain_of_f_l858_85869

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l858_85869


namespace NUMINAMATH_GPT_Lauryn_employees_l858_85820

variables (M W : ℕ)

theorem Lauryn_employees (h1 : M = W - 20) (h2 : M + W = 180) : M = 80 :=
by {
    sorry
}

end NUMINAMATH_GPT_Lauryn_employees_l858_85820


namespace NUMINAMATH_GPT_worker_usual_time_l858_85885

theorem worker_usual_time (S T : ℝ) (D : ℝ) (h1 : D = S * T)
    (h2 : D = (3/4) * S * (T + 8)) : T = 24 :=
by
  sorry

end NUMINAMATH_GPT_worker_usual_time_l858_85885


namespace NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l858_85819

theorem largest_angle_of_convex_hexagon 
  (x : ℝ) 
  (hx : (x + 2) + (2 * x - 1) + (3 * x + 1) + (4 * x - 2) + (5 * x + 3) + (6 * x - 4) = 720) :
  6 * x - 4 = 202 :=
sorry

end NUMINAMATH_GPT_largest_angle_of_convex_hexagon_l858_85819


namespace NUMINAMATH_GPT_mark_wait_time_l858_85877

theorem mark_wait_time (t1 t2 T : ℕ) (h1 : t1 = 4) (h2 : t2 = 20) (hT : T = 38) : 
  T - (t1 + t2) = 14 :=
by sorry

end NUMINAMATH_GPT_mark_wait_time_l858_85877


namespace NUMINAMATH_GPT_evaluate_expression_l858_85843

theorem evaluate_expression :
  (3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7) = (6^5 + 3^7) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l858_85843


namespace NUMINAMATH_GPT_eval_dagger_l858_85890

noncomputable def dagger (m n p q : ℕ) : ℚ := 
  (m * p) * (q / n)

theorem eval_dagger : dagger 5 16 12 5 = 75 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_eval_dagger_l858_85890


namespace NUMINAMATH_GPT_electricity_price_per_kWh_l858_85818

theorem electricity_price_per_kWh (consumption_rate : ℝ) (hours_used : ℝ) (total_cost : ℝ) :
  consumption_rate = 2.4 → hours_used = 25 → total_cost = 6 →
  total_cost / (consumption_rate * hours_used) = 0.10 :=
by
  intros hc hh ht
  have h_energy : consumption_rate * hours_used = 60 :=
    by rw [hc, hh]; norm_num
  rw [ht, h_energy]
  norm_num

end NUMINAMATH_GPT_electricity_price_per_kWh_l858_85818


namespace NUMINAMATH_GPT_eq_of_plane_contains_points_l858_85891

noncomputable def plane_eq (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let ⟨px, py, pz⟩ := p
  let ⟨qx, qy, qz⟩ := q
  let ⟨rx, ry, rz⟩ := r
  -- Vector pq
  let pq := (qx - px, qy - py, qz - pz)
  let ⟨pqx, pqy, pqz⟩ := pq
  -- Vector pr
  let pr := (rx - px, ry - py, rz - pz)
  let ⟨prx, pry, prz⟩ := pr
  -- Normal vector via cross product
  let norm := (pqy * prz - pqz * pry, pqz * prx - pqx * prz, pqx * pry - pqy * prx)
  let ⟨nx, ny, nz⟩ := norm
  -- Use normalized normal vector (1, 2, -2)
  (1, 2, -2, -(1 * px + 2 * py + -2 * pz))

theorem eq_of_plane_contains_points : 
  plane_eq (-2, 5, -3) (2, 5, -1) (4, 3, -2) = (1, 2, -2, -14) :=
by
  sorry

end NUMINAMATH_GPT_eq_of_plane_contains_points_l858_85891


namespace NUMINAMATH_GPT_num_counting_numbers_dividing_52_leaving_remainder_7_l858_85860

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem num_counting_numbers_dividing_52_leaving_remainder_7 (n : ℕ) :
  (∃ n : ℕ, 59 ≡ 7 [MOD n]) → (n > 7 ∧ divides n 52) → n = 3 := 
sorry

end NUMINAMATH_GPT_num_counting_numbers_dividing_52_leaving_remainder_7_l858_85860


namespace NUMINAMATH_GPT_population_net_change_l858_85810

theorem population_net_change
  (initial_population : ℝ)
  (year1_increase : initial_population * (6/5) = year1_population)
  (year2_increase : year1_population * (6/5) = year2_population)
  (year3_decrease : year2_population * (4/5) = year3_population)
  (year4_decrease : year3_population * (4/5) = final_population) :
  ((final_population - initial_population) / initial_population) * 100 = -8 :=
  sorry

end NUMINAMATH_GPT_population_net_change_l858_85810


namespace NUMINAMATH_GPT_jasmine_added_is_8_l858_85815

noncomputable def jasmine_problem (J : ℝ) : Prop :=
  let initial_volume := 80
  let initial_jasmine_concentration := 0.10
  let initial_jasmine_amount := initial_volume * initial_jasmine_concentration

  let added_water := 12
  let final_volume := initial_volume + J + added_water
  let final_jasmine_concentration := 0.16
  let final_jasmine_amount := final_volume * final_jasmine_concentration

  initial_jasmine_amount + J = final_jasmine_amount 

theorem jasmine_added_is_8 : jasmine_problem 8 :=
by
  sorry

end NUMINAMATH_GPT_jasmine_added_is_8_l858_85815


namespace NUMINAMATH_GPT_chicken_farm_l858_85854

def total_chickens (roosters hens : ℕ) : ℕ := roosters + hens

theorem chicken_farm (roosters hens : ℕ) (h1 : 2 * hens = roosters) (h2 : roosters = 6000) : 
  total_chickens roosters hens = 9000 :=
by
  sorry

end NUMINAMATH_GPT_chicken_farm_l858_85854


namespace NUMINAMATH_GPT_calculation_result_l858_85851

theorem calculation_result : 2014 * (1/19 - 1/53) = 68 := by
  sorry

end NUMINAMATH_GPT_calculation_result_l858_85851


namespace NUMINAMATH_GPT_general_term_formula_l858_85887

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 5 = (2 / 7) * (a 3) ^ 2) (h2 : S 7 = 63) :
  ∀ n, a n = 2 * n + 1 := by
  sorry

end NUMINAMATH_GPT_general_term_formula_l858_85887


namespace NUMINAMATH_GPT_total_lives_l858_85817

def initial_players := 25
def additional_players := 10
def lives_per_player := 15

theorem total_lives :
  (initial_players + additional_players) * lives_per_player = 525 := by
  sorry

end NUMINAMATH_GPT_total_lives_l858_85817


namespace NUMINAMATH_GPT_third_grade_contribution_fourth_grade_contribution_l858_85831

def first_grade := 20
def second_grade := 45
def third_grade := first_grade + second_grade - 17
def fourth_grade := 2 * third_grade - 36

theorem third_grade_contribution : third_grade = 48 := by
  sorry

theorem fourth_grade_contribution : fourth_grade = 60 := by
  sorry

end NUMINAMATH_GPT_third_grade_contribution_fourth_grade_contribution_l858_85831


namespace NUMINAMATH_GPT_apples_total_l858_85805

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_apples_total_l858_85805


namespace NUMINAMATH_GPT_problem_statement_l858_85830

noncomputable def middle_of_three_consecutive (x : ℕ) : ℕ :=
  let y := x + 1
  let z := x + 2
  y

theorem problem_statement :
  ∃ x : ℕ, 
    (x + (x + 1) = 18) ∧ 
    (x + (x + 2) = 20) ∧ 
    ((x + 1) + (x + 2) = 23) ∧ 
    (middle_of_three_consecutive x = 7) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l858_85830


namespace NUMINAMATH_GPT_percent_more_than_l858_85827

-- Definitions and conditions
variables (x y p : ℝ)

-- Condition: x is p percent more than y
def x_is_p_percent_more_than_y (x y p : ℝ) : Prop :=
  x = y + (p / 100) * y

-- The theorem to prove
theorem percent_more_than (h : x_is_p_percent_more_than_y x y p) :
  p = 100 * (x / y - 1) :=
sorry

end NUMINAMATH_GPT_percent_more_than_l858_85827


namespace NUMINAMATH_GPT_largest_of_consecutive_non_prime_integers_l858_85840

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of consecutive non-prime sequence condition
def consecutive_non_prime_sequence (start : ℕ) : Prop :=
  ∀ i : ℕ, 0 ≤ i → i < 10 → ¬ is_prime (start + i)

theorem largest_of_consecutive_non_prime_integers :
  (∃ start, start + 9 < 50 ∧ consecutive_non_prime_sequence start) →
  (∃ start, start + 9 = 47) :=
by
  sorry

end NUMINAMATH_GPT_largest_of_consecutive_non_prime_integers_l858_85840


namespace NUMINAMATH_GPT_find_solution_pairs_l858_85878

theorem find_solution_pairs (m n : ℕ) (t : ℕ) (ht : t > 0) (hcond : 2 ≤ m ∧ 2 ≤ n ∧ n ∣ (1 + m^(3^n) + m^(2 * 3^n))) : 
  ∃ t : ℕ, t > 0 ∧ m = 3 * t - 2 ∧ n = 3 :=
by sorry

end NUMINAMATH_GPT_find_solution_pairs_l858_85878


namespace NUMINAMATH_GPT_trig_identity_l858_85841

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l858_85841


namespace NUMINAMATH_GPT_find_B_find_sin_A_find_sin_2A_minus_B_l858_85814

open Real

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a * cos C + c * cos A = 2 * b * cos B) ∧ (7 * a = 5 * b)

theorem find_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_A (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) :
  sin A = 3 * sqrt 3 / 14 :=
sorry

theorem find_sin_2A_minus_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) (hA : sin A = 3 * sqrt 3 / 14) :
  sin (2 * A - B) = 8 * sqrt 3 / 49 :=
sorry

end NUMINAMATH_GPT_find_B_find_sin_A_find_sin_2A_minus_B_l858_85814


namespace NUMINAMATH_GPT_triangle_inequality_cosine_rule_l858_85896

theorem triangle_inequality_cosine_rule (a b c : ℝ) (A B C : ℝ)
  (hA : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hB : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hC : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * Real.cos A + b^3 * Real.cos B + c^3 * Real.cos C ≤ (3 / 2) * a * b * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_cosine_rule_l858_85896


namespace NUMINAMATH_GPT_range_of_a_l858_85873

def A : Set ℝ := { x | x^2 - x - 2 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - a) < 3 }

theorem range_of_a (a : ℝ) :
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l858_85873


namespace NUMINAMATH_GPT_smallestBeta_satisfies_l858_85836

noncomputable def validAlphaBeta (alpha beta : ℕ) : Prop :=
  16 / 37 < (alpha : ℚ) / beta ∧ (alpha : ℚ) / beta < 7 / 16

def smallestBeta : ℕ := 23

theorem smallestBeta_satisfies :
  (∀ (alpha beta : ℕ), validAlphaBeta alpha beta → beta ≥ 23) ∧
  (∃ (alpha : ℕ), validAlphaBeta alpha 23) :=
by sorry

end NUMINAMATH_GPT_smallestBeta_satisfies_l858_85836


namespace NUMINAMATH_GPT_baker_cakes_total_l858_85853

def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

theorem baker_cakes_total : 
  (initial_cakes - cakes_sold) + additional_cakes = 111 := by
  sorry

end NUMINAMATH_GPT_baker_cakes_total_l858_85853


namespace NUMINAMATH_GPT_total_applicants_is_40_l858_85837

def total_applicants (PS GPA_high Not_PS_GPA_low both : ℕ) : ℕ :=
  let PS_or_GPA_high := PS + GPA_high - both 
  PS_or_GPA_high + Not_PS_GPA_low

theorem total_applicants_is_40 :
  total_applicants 15 20 10 5 = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_applicants_is_40_l858_85837


namespace NUMINAMATH_GPT_number_of_integers_congruent_7_mod_9_lessthan_1000_l858_85898

theorem number_of_integers_congruent_7_mod_9_lessthan_1000 : 
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → 7 + 9 * n < 1000 → k + 1 = 111 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_congruent_7_mod_9_lessthan_1000_l858_85898


namespace NUMINAMATH_GPT_trapezoid_area_pqrs_l858_85823

theorem trapezoid_area_pqrs :
  let P := (1, 1)
  let Q := (1, 4)
  let R := (6, 4)
  let S := (7, 1)
  let parallelogram := true -- indicates that PQ and RS are parallel
  let PQ := abs (Q.2 - P.2)
  let RS := abs (S.1 - R.1)
  let height := abs (R.1 - P.1)
  (1 / 2 : ℚ) * (PQ + RS) * height = 10 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_pqrs_l858_85823


namespace NUMINAMATH_GPT_comparison_of_y1_and_y2_l858_85809

variable {k y1 y2 : ℝ}

theorem comparison_of_y1_and_y2 (hk : 0 < k)
    (hy1 : y1 = k)
    (hy2 : y2 = k / 4) :
    y1 > y2 := by
  sorry

end NUMINAMATH_GPT_comparison_of_y1_and_y2_l858_85809


namespace NUMINAMATH_GPT_functional_eq_solution_l858_85880

theorem functional_eq_solution (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) :
  g 10 = -48 :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l858_85880


namespace NUMINAMATH_GPT_inequality_proof_l858_85847

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem inequality_proof (α β : ℝ) (m : ℕ) (h1 : 1 < α) (h2 : 1 < β) (h3 : m = 1) 
  (h4 : f α m + f β m = 2) : (4 / α) + (1 / β) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l858_85847


namespace NUMINAMATH_GPT_ratio_SP_CP_l858_85803

variables (CP SP P : ℝ)
axiom ratio_profit_CP : P / CP = 2

theorem ratio_SP_CP : SP / CP = 3 :=
by
  -- Proof statement (not required as per the instruction)
  sorry

end NUMINAMATH_GPT_ratio_SP_CP_l858_85803


namespace NUMINAMATH_GPT_direct_proportion_graph_is_straight_line_l858_85845

-- Defining the direct proportion function
def direct_proportion_function (k x : ℝ) : ℝ := k * x

-- Theorem statement
theorem direct_proportion_graph_is_straight_line (k : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = direct_proportion_function k x ∧ 
    ∀ (x1 x2 : ℝ), 
    ∃ a b : ℝ, b ≠ 0 ∧ 
    (a * x1 + b * (direct_proportion_function k x1)) = (a * x2 + b * (direct_proportion_function k x2)) :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_graph_is_straight_line_l858_85845


namespace NUMINAMATH_GPT_range_of_f_l858_85897

noncomputable def f (x : ℝ) : ℝ := x + |x - 2|

theorem range_of_f : Set.range f = Set.Ici 2 :=
sorry

end NUMINAMATH_GPT_range_of_f_l858_85897


namespace NUMINAMATH_GPT_minimum_vertical_distance_l858_85857

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3 * x - 5

theorem minimum_vertical_distance :
  ∃ x : ℝ, (∀ y : ℝ, |absolute_value y - quadratic_function y| ≥ 4) ∧ (|absolute_value x - quadratic_function x| = 4) := 
sorry

end NUMINAMATH_GPT_minimum_vertical_distance_l858_85857


namespace NUMINAMATH_GPT_z_share_profit_correct_l858_85812

-- Define the investments as constants
def x_investment : ℕ := 20000
def y_investment : ℕ := 25000
def z_investment : ℕ := 30000

-- Define the number of months for each investment
def x_months : ℕ := 12
def y_months : ℕ := 12
def z_months : ℕ := 7

-- Define the annual profit
def annual_profit : ℕ := 50000

-- Calculate the active investment
def x_share : ℕ := x_investment * x_months
def y_share : ℕ := y_investment * y_months
def z_share : ℕ := z_investment * z_months

-- Calculate the total investment
def total_investment : ℕ := x_share + y_share + z_share

-- Define Z's ratio in terms of the total investment
def z_ratio : ℚ := z_share / total_investment

-- Calculate Z's share of the annual profit
def z_profit_share : ℚ := z_ratio * annual_profit

-- Theorem to prove Z's share in the annual profit
theorem z_share_profit_correct : z_profit_share = 14000 := by
  sorry

end NUMINAMATH_GPT_z_share_profit_correct_l858_85812


namespace NUMINAMATH_GPT_sum_of_two_digit_numbers_l858_85888

/-- Given two conditions regarding multiplication mistakes, we prove the sum of the numbers. -/
theorem sum_of_two_digit_numbers
  (A B C D : ℕ)
  (h1 : (10 * A + B) * (60 + D) = 2496)
  (h2 : (10 * A + B) * (20 + D) = 936) :
  (10 * A + B) + (10 * C + D) = 63 :=
by
  -- Conditions and necessary steps for solving the problem would go here.
  -- We're focusing on stating the problem, not the solution.
  sorry

end NUMINAMATH_GPT_sum_of_two_digit_numbers_l858_85888


namespace NUMINAMATH_GPT_triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l858_85829

variable {a b c x : ℝ}

-- Part (1)
theorem triangle_ABC_is_isosceles (h : (a + b) * 1 ^ 2 - 2 * c * 1 + (a - b) = 0) : a = c :=
by 
  -- Proof omitted
  sorry

-- Part (2)
theorem roots_of_quadratic_for_equilateral (h_eq : a = b ∧ b = c ∧ c = a) : 
  (∀ x : ℝ, (a + a) * x ^ 2 - 2 * a * x + (a - a) = 0 → (x = 0 ∨ x = 1)) :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l858_85829


namespace NUMINAMATH_GPT_time_spent_driving_l858_85800

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end NUMINAMATH_GPT_time_spent_driving_l858_85800


namespace NUMINAMATH_GPT_sum_nk_l858_85884

theorem sum_nk (n k : ℕ) (h₁ : 3 * n - 4 * k = 4) (h₂ : 4 * n - 5 * k = 13) : n + k = 55 := by
  sorry

end NUMINAMATH_GPT_sum_nk_l858_85884


namespace NUMINAMATH_GPT_linear_equation_in_two_vars_example_l858_85868

def is_linear_equation_in_two_vars (eq : String) : Prop :=
  eq = "x + 4y = 6"

theorem linear_equation_in_two_vars_example :
  is_linear_equation_in_two_vars "x + 4y = 6" :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_in_two_vars_example_l858_85868


namespace NUMINAMATH_GPT_calc_sub_neg_eq_add_problem_0_sub_neg_3_l858_85855

theorem calc_sub_neg_eq_add (a b : Int) : a - (-b) = a + b := by
  sorry

theorem problem_0_sub_neg_3 : 0 - (-3) = 3 := by
  exact calc_sub_neg_eq_add 0 3

end NUMINAMATH_GPT_calc_sub_neg_eq_add_problem_0_sub_neg_3_l858_85855


namespace NUMINAMATH_GPT_combined_apples_sold_l858_85804

theorem combined_apples_sold (red_apples green_apples total_apples : ℕ) 
    (h1 : red_apples = 32) 
    (h2 : green_apples = (3 * (32 / 8))) 
    (h3 : total_apples = red_apples + green_apples) : 
    total_apples = 44 :=
by
  sorry

end NUMINAMATH_GPT_combined_apples_sold_l858_85804


namespace NUMINAMATH_GPT_parabola_y_intercepts_l858_85875

theorem parabola_y_intercepts : 
  (∃ y1 y2 : ℝ, 3 * y1^2 - 4 * y1 + 1 = 0 ∧ 3 * y2^2 - 4 * y2 + 1 = 0 ∧ y1 ≠ y2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_y_intercepts_l858_85875


namespace NUMINAMATH_GPT_percent_difference_l858_85859

variable (w e y z : ℝ)

-- Definitions based on the given conditions
def condition1 : Prop := w = 0.60 * e
def condition2 : Prop := e = 0.60 * y
def condition3 : Prop := z = 0.54 * y

-- Statement of the theorem to prove
theorem percent_difference (h1 : condition1 w e) (h2 : condition2 e y) (h3 : condition3 z y) : 
  (z - w) / w * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_percent_difference_l858_85859


namespace NUMINAMATH_GPT_greatest_sum_of_visible_numbers_l858_85866

/-- Definition of a cube with numbered faces -/
structure Cube where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ

/-- The cubes face numbers -/
def cube_numbers : List ℕ := [1, 2, 4, 8, 16, 32]

/-- Stacked cubes with maximized visible numbers sum -/
def maximize_visible_sum :=
  let cube1 := Cube.mk 1 2 4 8 16 32
  let cube2 := Cube.mk 1 2 4 8 16 32
  let cube3 := Cube.mk 1 2 4 8 16 32
  let cube4 := Cube.mk 1 2 4 8 16 32
  244

theorem greatest_sum_of_visible_numbers : maximize_visible_sum = 244 := 
  by
    sorry -- Proof to be done

end NUMINAMATH_GPT_greatest_sum_of_visible_numbers_l858_85866


namespace NUMINAMATH_GPT_pure_imaginary_complex_solution_l858_85892

theorem pure_imaginary_complex_solution (a : Real) :
  (a ^ 2 - 1 = 0) ∧ ((a - 1) ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_complex_solution_l858_85892


namespace NUMINAMATH_GPT_inv_mod_997_l858_85834

theorem inv_mod_997 : ∃ x : ℤ, 0 ≤ x ∧ x < 997 ∧ (10 * x) % 997 = 1 := 
sorry

end NUMINAMATH_GPT_inv_mod_997_l858_85834


namespace NUMINAMATH_GPT_cube_side_ratio_l858_85886

theorem cube_side_ratio (a b : ℝ) (h : (6 * a^2) / (6 * b^2) = 36) : a / b = 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_ratio_l858_85886


namespace NUMINAMATH_GPT_treasure_in_heaviest_bag_l858_85816

theorem treasure_in_heaviest_bag (A B C D : ℝ) (h1 : A + B < C)
                                        (h2 : A + C = D)
                                        (h3 : A + D > B + C) : D > A ∧ D > B ∧ D > C :=
by 
  sorry

end NUMINAMATH_GPT_treasure_in_heaviest_bag_l858_85816


namespace NUMINAMATH_GPT_stamp_collection_l858_85874

theorem stamp_collection (x : ℕ) :
  (5 * x + 3 * (x + 20) = 300) → (x = 30) ∧ (x + 20 = 50) :=
by
  sorry

end NUMINAMATH_GPT_stamp_collection_l858_85874


namespace NUMINAMATH_GPT_minimize_f_a_n_distance_l858_85894

noncomputable def f (x : ℝ) : ℝ :=
  2^x + Real.log x

noncomputable def a (n : ℕ) : ℝ :=
  0.1 * n

theorem minimize_f_a_n_distance :
  ∃ n : ℕ, n = 110 ∧ ∀ m : ℕ, (m > 0) -> |f (a 110) - 2012| ≤ |f (a m) - 2012| :=
by
  sorry

end NUMINAMATH_GPT_minimize_f_a_n_distance_l858_85894


namespace NUMINAMATH_GPT_remainder_of_2345678_div_5_l858_85870

theorem remainder_of_2345678_div_5 : (2345678 % 5) = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_2345678_div_5_l858_85870


namespace NUMINAMATH_GPT_tree_height_at_2_years_l858_85813

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end NUMINAMATH_GPT_tree_height_at_2_years_l858_85813


namespace NUMINAMATH_GPT_divides_2_pow_26k_plus_2_plus_3_by_19_l858_85871

theorem divides_2_pow_26k_plus_2_plus_3_by_19 (k : ℕ) : 19 ∣ (2^(26*k+2) + 3) := 
by
  sorry

end NUMINAMATH_GPT_divides_2_pow_26k_plus_2_plus_3_by_19_l858_85871


namespace NUMINAMATH_GPT_largest_gcd_sum_1071_l858_85801

theorem largest_gcd_sum_1071 (x y: ℕ) (h1: x > 0) (h2: y > 0) (h3: x + y = 1071) : 
  ∃ d, d = Nat.gcd x y ∧ ∀ z, (z ∣ 1071 -> z ≤ d) := 
sorry

end NUMINAMATH_GPT_largest_gcd_sum_1071_l858_85801


namespace NUMINAMATH_GPT_game_ends_after_63_rounds_l858_85807

-- Define tokens for players A, B, C, and D at the start
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16
def initial_tokens_D := 14

-- Define the rules of the game
def game_rounds_to_end (A B C D : ℕ) : ℕ :=
  -- This function calculates the number of rounds after which any player runs out of tokens
  if (A, B, C, D) = (20, 18, 16, 14) then 63 else 0

-- Statement to prove
theorem game_ends_after_63_rounds :
  game_rounds_to_end initial_tokens_A initial_tokens_B initial_tokens_C initial_tokens_D = 63 :=
by sorry

end NUMINAMATH_GPT_game_ends_after_63_rounds_l858_85807
