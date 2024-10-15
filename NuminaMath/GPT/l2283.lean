import Mathlib

namespace NUMINAMATH_GPT_least_positive_integer_n_l2283_228373

theorem least_positive_integer_n (n : ℕ) (hn : n = 10) :
  (2:ℝ)^(1 / 5 * (n * (n + 1) / 2)) > 1000 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_n_l2283_228373


namespace NUMINAMATH_GPT_tan_sum_identity_l2283_228327

theorem tan_sum_identity
  (A B C : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

end NUMINAMATH_GPT_tan_sum_identity_l2283_228327


namespace NUMINAMATH_GPT_correct_calculation_l2283_228330

theorem correct_calculation :
  (∃ (x y : ℝ), 5 * x + 2 * y ≠ 7 * x * y) ∧
  (∃ (x : ℝ), 3 * x - 2 * x ≠ 1) ∧
  (∃ (x : ℝ), x^2 + x^5 ≠ x^7) →
  (∀ (x y : ℝ), 3 * x^2 * y - 4 * y * x^2 = -x^2 * y) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2283_228330


namespace NUMINAMATH_GPT_evaluate_fraction_l2283_228386

theorem evaluate_fraction :
  (20 - 18 + 16 - 14 + 12 - 10 + 8 - 6 + 4 - 2) / (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2283_228386


namespace NUMINAMATH_GPT_salmon_at_rest_oxygen_units_l2283_228302

noncomputable def salmonSwimSpeed (x : ℝ) : ℝ := (1/2) * Real.log (x / 100 * Real.pi) / Real.log 3

theorem salmon_at_rest_oxygen_units :
  ∃ x : ℝ, salmonSwimSpeed x = 0 ∧ x = 100 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_salmon_at_rest_oxygen_units_l2283_228302


namespace NUMINAMATH_GPT_find_m_l2283_228389

noncomputable def quadratic_eq (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + 4 * x + m

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : quadratic_eq x₁ m = 0)
  (h2 : quadratic_eq x₂ m = 0)
  (h3 : 16 - 8 * m ≥ 0)
  (h4 : x₁^2 + x₂^2 + 2 * x₁ * x₂ - x₁^2 * x₂^2 = 0) 
  : m = -4 :=
sorry

end NUMINAMATH_GPT_find_m_l2283_228389


namespace NUMINAMATH_GPT_part_I_part_II_l2283_228307

theorem part_I (a b : ℝ) (h1 : 0 < a) (h2 : b * a = 2)
  (h3 : (1 + b) * a = 3) :
  (a = 1) ∧ (b = 2) :=
by {
  sorry
}

theorem part_II (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1 : ℝ) / x + 2 / y = 1)
  (k : ℝ) : 2 * x + y ≥ k^2 + k + 2 → (-3 ≤ k) ∧ (k ≤ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_part_I_part_II_l2283_228307


namespace NUMINAMATH_GPT_number_of_elements_in_set_l2283_228371

-- We define the conditions in terms of Lean definitions.
variable (n : ℕ) (S : ℕ)

-- Define the initial wrong average condition
def wrong_avg_condition : Prop := (S + 26) / n = 18

-- Define the corrected average condition
def correct_avg_condition : Prop := (S + 36) / n = 19

-- The main theorem to be proved
theorem number_of_elements_in_set (h1 : wrong_avg_condition n S) (h2 : correct_avg_condition n S) : n = 10 := 
sorry

end NUMINAMATH_GPT_number_of_elements_in_set_l2283_228371


namespace NUMINAMATH_GPT_bisect_angle_BAX_l2283_228374

-- Definitions and conditions
variables {A B C M X : Point}
variable (is_scalene_triangle : ScaleneTriangle A B C)
variable (is_midpoint : Midpoint M B C)
variable (is_parallel : Parallel (Line C X) (Line A B))
variable (angle_right : Angle AM X = 90)

-- The theorem statement to be proven
theorem bisect_angle_BAX (h1 : is_scalene_triangle)
                         (h2 : is_midpoint)
                         (h3 : is_parallel)
                         (h4 : angle_right) :
  Bisects (Line A M) (Angle B A X) :=
sorry

end NUMINAMATH_GPT_bisect_angle_BAX_l2283_228374


namespace NUMINAMATH_GPT_solve_for_y_l2283_228319

theorem solve_for_y :
  ∃ y : ℚ, 2 * y + 3 * y = 200 - (4 * y + (10 * y / 2)) ∧ y = 100 / 7 :=
by {
  -- Assertion only, proof is not required as per instructions.
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l2283_228319


namespace NUMINAMATH_GPT_value_of_x2017_l2283_228313

-- Definitions and conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

variables (f : ℝ → ℝ) (x : ℕ → ℝ)
variables (d : ℝ)
variable (h_odd : is_odd_function f)
variable (h_increasing : is_increasing_function f)
variable (h_arithmetic : arithmetic_sequence x 2)
variable (h_condition : f (x 7) + f (x 8) = 0)

-- Define the proof goal
theorem value_of_x2017 : x 2017 = 4019 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x2017_l2283_228313


namespace NUMINAMATH_GPT_matilda_jellybeans_l2283_228366

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end NUMINAMATH_GPT_matilda_jellybeans_l2283_228366


namespace NUMINAMATH_GPT_rational_root_neg_one_third_l2283_228355

def P (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_neg_one_third : P (-1/3) = 0 :=
by
  have : (-1/3 : ℚ) ≠ 0 := by norm_num
  sorry

end NUMINAMATH_GPT_rational_root_neg_one_third_l2283_228355


namespace NUMINAMATH_GPT_increase_80_by_150_percent_l2283_228321

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end NUMINAMATH_GPT_increase_80_by_150_percent_l2283_228321


namespace NUMINAMATH_GPT_perpendicular_planes_normal_vector_l2283_228304

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem perpendicular_planes_normal_vector {m : ℝ} 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (h₁ : a = (1, 2, -2)) 
  (h₂ : b = (-2, 1, m)) 
  (h₃ : dot_product a b = 0) : 
  m = 0 := 
sorry

end NUMINAMATH_GPT_perpendicular_planes_normal_vector_l2283_228304


namespace NUMINAMATH_GPT_find_point_M_l2283_228340

/-- Define the function f(x) = x^3 + x - 2. -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- Define the derivative of the function, f'(x). -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Define the condition that the slope of the tangent line is perpendicular to y = -1/4x - 1. -/
def slope_perpendicular_condition (m : ℝ) : Prop := m = 4

/-- Main theorem: The coordinates of the point M are (1, 0) and (-1, -4). -/
theorem find_point_M : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ slope_perpendicular_condition (f' x₀) ∧ 
  ((x₀ = 1 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = -4)) := 
sorry

end NUMINAMATH_GPT_find_point_M_l2283_228340


namespace NUMINAMATH_GPT_ice_cream_flavors_l2283_228393

-- We have four basic flavors and want to combine four scoops from these flavors.
def ice_cream_combinations : ℕ :=
  Nat.choose 7 3

theorem ice_cream_flavors : ice_cream_combinations = 35 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_flavors_l2283_228393


namespace NUMINAMATH_GPT_major_arc_circumference_l2283_228392

noncomputable def circumference_major_arc 
  (A B C : Point) (r : ℝ) (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) : ℝ :=
  let total_circumference := 2 * Real.pi * r
  let major_arc_angle := 360 - angle_ACB
  major_arc_angle / 360 * total_circumference

theorem major_arc_circumference (A B C : Point) (r : ℝ)
  (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) :
  circumference_major_arc A B C r angle_ACB h1 h2 = (500 / 3) * Real.pi :=
  sorry

end NUMINAMATH_GPT_major_arc_circumference_l2283_228392


namespace NUMINAMATH_GPT_num_right_angle_triangles_l2283_228353

-- Step d): Lean 4 statement
theorem num_right_angle_triangles {C : ℝ × ℝ} (hC : C.2 = 0) :
  (C = (-2, 0) ∨ C = (4, 0) ∨ C = (1, 0)) ↔ ∃ A B : ℝ × ℝ,
  (A = (-2, 3)) ∧ (B = (4, 3)) ∧ 
  (A.2 = B.2) ∧ (A.1 ≠ B.1) ∧ 
  (((C.1-A.1)*(B.1-A.1) + (C.2-A.2)*(B.2-A.2) = 0) ∨ 
   ((C.1-B.1)*(A.1-B.1) + (C.2-B.2)*(A.2-B.2) = 0)) :=
sorry

end NUMINAMATH_GPT_num_right_angle_triangles_l2283_228353


namespace NUMINAMATH_GPT_problem_l2283_228361

theorem problem (x : ℕ) (h : 2^x + 2^x + 2^x = 256) : x * (x + 1) = 72 :=
sorry

end NUMINAMATH_GPT_problem_l2283_228361


namespace NUMINAMATH_GPT_win_percentage_of_people_with_envelopes_l2283_228377

theorem win_percentage_of_people_with_envelopes (total_people : ℕ) (percent_with_envelopes : ℝ) (winners : ℕ) (num_with_envelopes : ℕ) : 
  total_people = 100 ∧ percent_with_envelopes = 0.40 ∧ num_with_envelopes = total_people * percent_with_envelopes ∧ winners = 8 → 
    (winners / num_with_envelopes) * 100 = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_win_percentage_of_people_with_envelopes_l2283_228377


namespace NUMINAMATH_GPT_camilla_jellybeans_l2283_228364

theorem camilla_jellybeans (b c : ℕ) (h1 : b = 3 * c) (h2 : b - 20 = 4 * (c - 20)) :
  b = 180 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_camilla_jellybeans_l2283_228364


namespace NUMINAMATH_GPT_fraction_evaluation_l2283_228342

theorem fraction_evaluation : (1 - (1 / 4)) / (1 - (1 / 3)) = (9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l2283_228342


namespace NUMINAMATH_GPT_triangle_sides_and_angles_l2283_228310

theorem triangle_sides_and_angles (a : Real) (α β : Real) :
  (a ≥ 0) →
  let sides := [a, a + 1, a + 2]
  let angles := [α, β, 2 * α]
  (∀ s, s ∈ sides) → (∀ θ, θ ∈ angles) →
  a = 4 ∧ a + 1 = 5 ∧ a + 2 = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_triangle_sides_and_angles_l2283_228310


namespace NUMINAMATH_GPT_find_x_l2283_228343

noncomputable def positive_real (a : ℝ) := 0 < a

theorem find_x (x y : ℝ) (h1 : positive_real x) (h2 : positive_real y)
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y)
  (h4 : x + y = 3) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2283_228343


namespace NUMINAMATH_GPT_find_x_square_l2283_228395

theorem find_x_square (x : ℝ) (h_pos : x > 0) (h_condition : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_square_l2283_228395


namespace NUMINAMATH_GPT_solve_problem_l2283_228329

open Classical

-- Definition of the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1

-- Definition of the quadratic solution considering the quadratic formula
def quadratic_solution (x : ℝ) : Prop :=
  x = (-21 + Real.sqrt 641) / 50 ∨ x = (-21 - Real.sqrt 641) / 50

-- Main theorem statement
theorem solve_problem :
  ∃ x y : ℝ, problem_conditions x y ∧ quadratic_solution x :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2283_228329


namespace NUMINAMATH_GPT_solve_inequality_l2283_228303

noncomputable def solution_set (x : ℝ) : Prop :=
  (-(9/2) ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2)

theorem solve_inequality (x : ℝ) :
  (x ≠ -2 ∧ x ≠ 9/2) →
  ( (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ) ↔ solution_set x :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2283_228303


namespace NUMINAMATH_GPT_num_users_in_china_in_2022_l2283_228380

def num_users_scientific (n : ℝ) : Prop :=
  n = 1.067 * 10^9

theorem num_users_in_china_in_2022 :
  num_users_scientific 1.067e9 :=
by
  sorry

end NUMINAMATH_GPT_num_users_in_china_in_2022_l2283_228380


namespace NUMINAMATH_GPT_sum_three_digit_integers_from_200_to_900_l2283_228316

theorem sum_three_digit_integers_from_200_to_900 : 
  let a := 200
  let l := 900
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 385550 := by
    let a := 200
    let l := 900
    let d := 1
    let n := (l - a) / d + 1
    let S := n / 2 * (a + l)
    sorry

end NUMINAMATH_GPT_sum_three_digit_integers_from_200_to_900_l2283_228316


namespace NUMINAMATH_GPT_augmented_matrix_solution_l2283_228365

theorem augmented_matrix_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), (m * x = 6 ∧ 3 * y = n) ∧ (x = -3 ∧ y = 4)) →
  m + n = 10 :=
by
  intros m n h
  sorry

end NUMINAMATH_GPT_augmented_matrix_solution_l2283_228365


namespace NUMINAMATH_GPT_select_student_based_on_variance_l2283_228344

-- Define the scores for students A and B
def scoresA : List ℚ := [12.1, 12.1, 12.0, 11.9, 11.8, 12.1]
def scoresB : List ℚ := [12.2, 12.0, 11.8, 12.0, 12.3, 11.7]

-- Define the function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ := (scores.foldr (· + ·) 0) / scores.length

-- Define the function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.foldr (λ x acc => acc + (x - m) ^ 2) 0) / scores.length

-- Prove that the variance of student A's scores is less than the variance of student B's scores
theorem select_student_based_on_variance :
  variance scoresA < variance scoresB := by
  sorry

end NUMINAMATH_GPT_select_student_based_on_variance_l2283_228344


namespace NUMINAMATH_GPT_matchstick_game_winner_a_matchstick_game_winner_b_l2283_228394

def is_winning_position (pile1 pile2 : Nat) : Bool :=
  (pile1 % 2 = 1) && (pile2 % 2 = 1)

theorem matchstick_game_winner_a : is_winning_position 101 201 = true := 
by
  -- Theorem statement for (101 matches, 201 matches)
  -- The second player wins
  sorry

theorem matchstick_game_winner_b : is_winning_position 100 201 = false := 
by
  -- Theorem statement for (100 matches, 201 matches)
  -- The first player wins
  sorry

end NUMINAMATH_GPT_matchstick_game_winner_a_matchstick_game_winner_b_l2283_228394


namespace NUMINAMATH_GPT_find_x_find_y_find_p_q_r_l2283_228339

-- Condition: The number on the line connecting two circles is the sum of the two numbers in the circles.

-- For part (a):
theorem find_x (a b : ℝ) (x : ℝ) (h1 : a + 4 = 13) (h2 : a + b = 10) (h3 : b + 4 = x) : x = 5 :=
by {
  -- Proof can be filled in here to show x = 5 by solving the equations.
  sorry
}

-- For part (b):
theorem find_y (w y : ℝ) (h1 : 3 * w + w = y) (h2 : 6 * w = 48) : y = 32 := 
by {
  -- Proof can be filled in here to show y = 32 by solving the equations.
  sorry
}

-- For part (c):
theorem find_p_q_r (p q r : ℝ) (h1 : p + r = 3) (h2 : p + q = 18) (h3 : q + r = 13) : p = 4 ∧ q = 14 ∧ r = -1 :=
by {
  -- Proof can be filled in here to show p = 4, q = 14, r = -1 by solving the equations.
  sorry
}

end NUMINAMATH_GPT_find_x_find_y_find_p_q_r_l2283_228339


namespace NUMINAMATH_GPT_ribbon_cuts_l2283_228311

theorem ribbon_cuts (rolls : ℕ) (length_per_roll : ℕ) (piece_length : ℕ) (total_rolls : rolls = 5) (roll_length : length_per_roll = 50) (piece_size : piece_length = 2) : 
  (rolls * ((length_per_roll / piece_length) - 1) = 120) :=
by
  sorry

end NUMINAMATH_GPT_ribbon_cuts_l2283_228311


namespace NUMINAMATH_GPT_largest_triangle_angle_l2283_228308

-- Define the angles
def angle_sum := (105 : ℝ) -- Degrees
def delta_angle := (36 : ℝ) -- Degrees
def total_sum := (180 : ℝ) -- Degrees

-- Theorem statement
theorem largest_triangle_angle (a b c : ℝ) (h1 : a + b = angle_sum)
  (h2 : b = a + delta_angle) (h3 : a + b + c = total_sum) : c = 75 :=
sorry

end NUMINAMATH_GPT_largest_triangle_angle_l2283_228308


namespace NUMINAMATH_GPT_other_x_intercept_of_parabola_l2283_228335

theorem other_x_intercept_of_parabola (a b c : ℝ) :
  (∃ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ (2, 10) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} ∧ (1, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)}
  → ∃ x : ℝ, x = 3 ∧ (x, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} :=
by
  sorry

end NUMINAMATH_GPT_other_x_intercept_of_parabola_l2283_228335


namespace NUMINAMATH_GPT_number_of_people_l2283_228345

variable (P M : ℕ)

-- Conditions
def cond1 : Prop := (500 = P * M)
def cond2 : Prop := (500 = (P + 5) * (M - 2))

-- Goal
theorem number_of_people (h1 : cond1 P M) (h2 : cond2 P M) : P = 33 :=
sorry

end NUMINAMATH_GPT_number_of_people_l2283_228345


namespace NUMINAMATH_GPT_periodic_sequence_condition_l2283_228314

theorem periodic_sequence_condition (m : ℕ) (a : ℕ) 
  (h_pos : 0 < m)
  (a_seq : ℕ → ℕ) (h_initial : a_seq 0 = a)
  (h_relation : ∀ n, a_seq (n + 1) = if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n + m) :
  (∃ p, ∀ k, a_seq (k + p) = a_seq k) ↔ 
  (a ∈ ({n | 1 ≤ n ∧ n ≤ m} ∪ {n | ∃ k, n = m + 2 * k + 1 ∧ n < 2 * m + 1})) :=
sorry

end NUMINAMATH_GPT_periodic_sequence_condition_l2283_228314


namespace NUMINAMATH_GPT_second_group_product_number_l2283_228387

theorem second_group_product_number (a₀ : ℕ) (h₀ : 0 ≤ a₀ ∧ a₀ < 20)
  (h₁ : 4 * 20 + a₀ = 94) : 1 * 20 + a₀ = 34 :=
by
  sorry

end NUMINAMATH_GPT_second_group_product_number_l2283_228387


namespace NUMINAMATH_GPT_find_a1_over_d_l2283_228337

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1_over_d 
  (d_ne_zero : d ≠ 0) 
  (seq : arithmetic_sequence a d) 
  (h : a 2021 = a 20 + a 21) : 
  a 1 / d = 1981 :=
by 
  sorry

end NUMINAMATH_GPT_find_a1_over_d_l2283_228337


namespace NUMINAMATH_GPT_initial_roses_count_l2283_228391

theorem initial_roses_count 
  (roses_to_mother : ℕ)
  (roses_to_grandmother : ℕ)
  (roses_to_sister : ℕ)
  (roses_kept : ℕ)
  (initial_roses : ℕ)
  (h_mother : roses_to_mother = 6)
  (h_grandmother : roses_to_grandmother = 9)
  (h_sister : roses_to_sister = 4)
  (h_kept : roses_kept = 1)
  (h_initial : initial_roses = roses_to_mother + roses_to_grandmother + roses_to_sister + roses_kept) :
  initial_roses = 20 :=
by
  rw [h_mother, h_grandmother, h_sister, h_kept] at h_initial
  exact h_initial

end NUMINAMATH_GPT_initial_roses_count_l2283_228391


namespace NUMINAMATH_GPT_totalCups_l2283_228369

-- Let's state our definitions based on the conditions:
def servingsPerBox : ℕ := 9
def cupsPerServing : ℕ := 2

-- Our goal is to prove the following statement.
theorem totalCups (hServings: servingsPerBox = 9) (hCups: cupsPerServing = 2) : servingsPerBox * cupsPerServing = 18 := by
  -- The detailed proof will go here.
  sorry

end NUMINAMATH_GPT_totalCups_l2283_228369


namespace NUMINAMATH_GPT_P_shape_points_length_10_l2283_228396

def P_shape_points (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem P_shape_points_length_10 :
  P_shape_points 10 = 31 := 
by 
  sorry

end NUMINAMATH_GPT_P_shape_points_length_10_l2283_228396


namespace NUMINAMATH_GPT_tank_filled_fraction_l2283_228372

noncomputable def initial_quantity (total_capacity : ℕ) := (3 / 4 : ℚ) * total_capacity

noncomputable def final_quantity (initial : ℚ) (additional : ℚ) := initial + additional

noncomputable def fraction_of_capacity (quantity : ℚ) (total_capacity : ℕ) := quantity / total_capacity

theorem tank_filled_fraction (total_capacity : ℕ) (additional_gas : ℚ)
  (initial_fraction : ℚ) (final_fraction : ℚ) :
  initial_fraction = initial_quantity total_capacity →
  final_fraction = fraction_of_capacity (final_quantity initial_fraction additional_gas) total_capacity →
  total_capacity = 42 →
  additional_gas = 7 →
  initial_fraction = 31.5 →
  final_fraction = (833 / 909 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_tank_filled_fraction_l2283_228372


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2283_228359

theorem sufficient_not_necessary (p q: Prop) :
  ¬ (p ∨ q) → ¬ p ∧ (¬ p → ¬(¬ p ∧ ¬ q)) := sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2283_228359


namespace NUMINAMATH_GPT_polygonal_chain_max_length_not_exceed_200_l2283_228385

-- Define the size of the board
def board_size : ℕ := 15

-- Define the concept of a polygonal chain length on a symmetric board
def polygonal_chain_length (n : ℕ) : ℕ := sorry -- length function yet to be defined

-- Define the maximum length constant to be compared with
def max_length : ℕ := 200

-- Define the theorem statement including all conditions and constraints
theorem polygonal_chain_max_length_not_exceed_200 :
  ∃ (n : ℕ), n = board_size ∧ 
             (∀ (length : ℕ),
             length = polygonal_chain_length n →
             length ≤ max_length) :=
sorry

end NUMINAMATH_GPT_polygonal_chain_max_length_not_exceed_200_l2283_228385


namespace NUMINAMATH_GPT_difference_of_squares_example_l2283_228306

theorem difference_of_squares_example (a b : ℕ) (h₁ : a = 650) (h₂ : b = 350) :
  a^2 - b^2 = 300000 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_l2283_228306


namespace NUMINAMATH_GPT_factory_workers_total_payroll_l2283_228390

theorem factory_workers_total_payroll (total_office_payroll : ℝ) (number_factory_workers : ℝ) 
(number_office_workers : ℝ) (salary_difference : ℝ) 
(average_office_salary : ℝ) (average_factory_salary : ℝ) 
(h1 : total_office_payroll = 75000) (h2 : number_factory_workers = 15)
(h3 : number_office_workers = 30) (h4 : salary_difference = 500)
(h5 : average_office_salary = total_office_payroll / number_office_workers)
(h6 : average_office_salary = average_factory_salary + salary_difference) :
  number_factory_workers * average_factory_salary = 30000 :=
by
  sorry

end NUMINAMATH_GPT_factory_workers_total_payroll_l2283_228390


namespace NUMINAMATH_GPT_base_5_to_decimal_l2283_228347

theorem base_5_to_decimal : 
  let b5 := [1, 2, 3, 4] -- base-5 number 1234 in list form
  let decimal := 194
  (b5[0] * 5^3 + b5[1] * 5^2 + b5[2] * 5^1 + b5[3] * 5^0) = decimal :=
by
  -- Proof details go here
  sorry

end NUMINAMATH_GPT_base_5_to_decimal_l2283_228347


namespace NUMINAMATH_GPT_fraction_addition_solution_is_six_l2283_228356

theorem fraction_addition_solution_is_six :
  (1 / 9) + (1 / 18) = 1 / 6 := 
sorry

end NUMINAMATH_GPT_fraction_addition_solution_is_six_l2283_228356


namespace NUMINAMATH_GPT_circle_intersection_l2283_228322

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0)) ↔ 4 < m ∧ m < 144 :=
by
  have h1 : distance (0, 0) (3, -4) = 5 := by sorry
  have h2 : ∀ m, |7 - Real.sqrt m| < 5 ↔ 4 < m ∧ m < 144 := by sorry
  exact sorry

end NUMINAMATH_GPT_circle_intersection_l2283_228322


namespace NUMINAMATH_GPT_sum_of_ages_l2283_228383

variables (P M Mo : ℕ)

theorem sum_of_ages (h1 : 5 * P = 3 * M)
                    (h2 : 5 * M = 3 * Mo)
                    (h3 : Mo - P = 32) :
  P + M + Mo = 98 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l2283_228383


namespace NUMINAMATH_GPT_xy_expr_value_l2283_228338

variable (x y : ℝ)

-- Conditions
def cond1 : Prop := x - y = 2
def cond2 : Prop := x * y = 3

-- Statement to prove
theorem xy_expr_value (h1 : cond1 x y) (h2 : cond2 x y) : x * y^2 - x^2 * y = -6 :=
by
  sorry

end NUMINAMATH_GPT_xy_expr_value_l2283_228338


namespace NUMINAMATH_GPT_distance_from_origin_to_point_on_parabola_l2283_228332

theorem distance_from_origin_to_point_on_parabola
  (y x : ℝ)
  (focus : ℝ × ℝ := (4, 0))
  (on_parabola : y^2 = 8 * x)
  (distance_to_focus : Real.sqrt ((x - 4)^2 + y^2) = 4) :
  Real.sqrt (x^2 + y^2) = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_to_point_on_parabola_l2283_228332


namespace NUMINAMATH_GPT_number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l2283_228300

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S := sorry

lemma functional_equation (x y : S) (h : (x.val + y.val) ≠ 0) :
  (f x).val + (f y).val = (f ⟨(x.val * y.val) / (x.val + y.val) * (f ⟨x.val + y.val, sorry⟩).val, sorry⟩).val := sorry

-- Prove that the number of possible values of f(3) is 1

theorem number_of_values_f3 : ∃ n : ℕ, n = 1 := sorry

-- Prove that the sum of all possible values of f(3) is 1/3

theorem sum_of_values_f3 : ∃ s : ℚ, s = 1/3 := sorry

-- Prove that n * s = 1/3

theorem product_of_n_and_s (n : ℕ) (s : ℚ) (hn : n = 1) (hs : s = 1/3) : n * s = 1/3 := by
  rw [hn, hs]
  norm_num

end NUMINAMATH_GPT_number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l2283_228300


namespace NUMINAMATH_GPT_lemonade_glasses_l2283_228320

theorem lemonade_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (glasses : ℝ) :
  total_lemons = 18.0 → lemons_per_glass = 2.0 → glasses = total_lemons / lemons_per_glass → glasses = 9 :=
by
  intro h_total_lemons h_lemons_per_glass h_glasses
  sorry

end NUMINAMATH_GPT_lemonade_glasses_l2283_228320


namespace NUMINAMATH_GPT_equivalent_problem_l2283_228362

variable (a b : ℤ)

def condition1 : Prop :=
  a * (-2)^3 + b * (-2) - 7 = 9

def condition2 : Prop :=
  8 * a + 2 * b - 7 = -23

theorem equivalent_problem (h : condition1 a b) : condition2 a b :=
sorry

end NUMINAMATH_GPT_equivalent_problem_l2283_228362


namespace NUMINAMATH_GPT_prove_min_max_A_l2283_228305

theorem prove_min_max_A : 
  ∃ (A_max A_min : ℕ), 
  (∃ B : ℕ, 
    A_max = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧
    B % 10 = 9) ∧ 
  (∃ B : ℕ, 
    A_min = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧ 
    B % 10 = 1) ∧ 
  A_max = 999999998 ∧ 
  A_min = 166666667 := sorry

end NUMINAMATH_GPT_prove_min_max_A_l2283_228305


namespace NUMINAMATH_GPT_sales_on_second_street_l2283_228363

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end NUMINAMATH_GPT_sales_on_second_street_l2283_228363


namespace NUMINAMATH_GPT_find_constant_d_l2283_228399

noncomputable def polynomial_g (d : ℝ) (x : ℝ) := d * x^4 + 17 * x^3 - 5 * d * x^2 + 45

theorem find_constant_d (d : ℝ) : polynomial_g d 5 = 0 → d = -4.34 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_d_l2283_228399


namespace NUMINAMATH_GPT_find_y_value_l2283_228325
-- Import the necessary Lean library

-- Define the conditions and the target theorem
theorem find_y_value (h : 6 * y + 3 * y + y + 4 * y = 360) : y = 180 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l2283_228325


namespace NUMINAMATH_GPT_incorrect_relation_when_agtb_l2283_228379

theorem incorrect_relation_when_agtb (a b : ℝ) (c : ℝ) (h : a > b) : c = 0 → ¬ (a * c^2 > b * c^2) :=
by
  -- Not providing the proof here as specified in the instructions.
  sorry

end NUMINAMATH_GPT_incorrect_relation_when_agtb_l2283_228379


namespace NUMINAMATH_GPT_Heather_heavier_than_Emily_l2283_228312

def Heather_weight := 87
def Emily_weight := 9

theorem Heather_heavier_than_Emily : (Heather_weight - Emily_weight = 78) :=
by sorry

end NUMINAMATH_GPT_Heather_heavier_than_Emily_l2283_228312


namespace NUMINAMATH_GPT_simplify_expression_l2283_228336

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2283_228336


namespace NUMINAMATH_GPT_tetrahedron_volume_l2283_228370

-- Definition of the required constants and variables
variables {S1 S2 S3 S4 r : ℝ}

-- The volume formula we need to prove
theorem tetrahedron_volume :
  (V = 1/3 * (S1 + S2 + S3 + S4) * r) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_l2283_228370


namespace NUMINAMATH_GPT_largest_multiple_of_8_smaller_than_neg_80_l2283_228381

theorem largest_multiple_of_8_smaller_than_neg_80 :
  ∃ n : ℤ, (8 ∣ n) ∧ n < -80 ∧ ∀ m : ℤ, (8 ∣ m ∧ m < -80 → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_8_smaller_than_neg_80_l2283_228381


namespace NUMINAMATH_GPT_hannahs_grapes_per_day_l2283_228331

-- Definitions based on conditions
def oranges_per_day : ℕ := 20
def days : ℕ := 30
def total_fruits : ℕ := 1800
def total_oranges : ℕ := oranges_per_day * days

-- The math proof problem to be targeted
theorem hannahs_grapes_per_day : 
  (total_fruits - total_oranges) / days = 40 := 
by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_hannahs_grapes_per_day_l2283_228331


namespace NUMINAMATH_GPT_total_pieces_equiv_231_l2283_228367

-- Define the arithmetic progression for rods.
def rods_arithmetic_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => 3 * (n + 1)

-- Define the sum of the first 10 terms of the sequence.
def rods_total (n : ℕ) : ℕ :=
  let a := 3
  let d := 3
  n / 2 * (2 * a + (n - 1) * d)

def rods_count : ℕ :=
  rods_total 10

-- Define the 11th triangular number for connectors.
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def connectors_count : ℕ :=
  triangular_number 11

-- Define the total number of pieces.
def total_pieces : ℕ :=
  rods_count + connectors_count

-- The theorem we aim to prove.
theorem total_pieces_equiv_231 : total_pieces = 231 := by
  sorry

end NUMINAMATH_GPT_total_pieces_equiv_231_l2283_228367


namespace NUMINAMATH_GPT_area_ratio_l2283_228346

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_l2283_228346


namespace NUMINAMATH_GPT_train_speed_is_60_kmph_l2283_228348

-- Define the distance and time
def train_length : ℕ := 400
def bridge_length : ℕ := 800
def time_to_pass_bridge : ℕ := 72

-- Define the distances and calculations
def total_distance : ℕ := train_length + bridge_length
def speed_m_per_s : ℚ := total_distance / time_to_pass_bridge
def speed_km_per_h : ℚ := speed_m_per_s * 3.6

-- State and prove the theorem
theorem train_speed_is_60_kmph : speed_km_per_h = 60 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_60_kmph_l2283_228348


namespace NUMINAMATH_GPT_batsman_average_increase_l2283_228352

theorem batsman_average_increase
  (A : ℕ)
  (h_average_after_17th : (16 * A + 90) / 17 = 42) :
  42 - A = 3 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l2283_228352


namespace NUMINAMATH_GPT_tan_of_x_is_3_l2283_228378

theorem tan_of_x_is_3 (x : ℝ) (h : Real.tan x = 3) (hx : Real.cos x ≠ 0) : 
  (Real.sin x + 3 * Real.cos x) / (2 * Real.sin x - 3 * Real.cos x) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_x_is_3_l2283_228378


namespace NUMINAMATH_GPT_find_larger_number_l2283_228326

-- Define the conditions
variables (L S : ℕ)
axiom condition1 : L - S = 1365
axiom condition2 : L = 6 * S + 35

-- State the theorem
theorem find_larger_number : L = 1631 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l2283_228326


namespace NUMINAMATH_GPT_optimal_playground_dimensions_and_area_l2283_228309

theorem optimal_playground_dimensions_and_area:
  ∃ (l w : ℝ), 2 * l + 2 * w = 380 ∧ l ≥ 100 ∧ w ≥ 60 ∧ l * w = 9000 :=
by
  sorry

end NUMINAMATH_GPT_optimal_playground_dimensions_and_area_l2283_228309


namespace NUMINAMATH_GPT_percent_of_ac_is_db_l2283_228324

variable (a b c d : ℝ)

-- Given conditions
variable (h1 : c = 0.25 * a)
variable (h2 : c = 0.10 * b)
variable (h3 : d = 0.50 * b)

-- Theorem statement: Prove the final percentage
theorem percent_of_ac_is_db : (d * b) / (a * c) * 100 = 1250 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_ac_is_db_l2283_228324


namespace NUMINAMATH_GPT_minimize_wage_l2283_228315

def totalWorkers : ℕ := 150
def wageA : ℕ := 2000
def wageB : ℕ := 3000

theorem minimize_wage : ∃ (a : ℕ), a = 50 ∧ (totalWorkers - a) ≥ 2 * a ∧ 
  (wageA * a + wageB * (totalWorkers - a) = 400000) := sorry

end NUMINAMATH_GPT_minimize_wage_l2283_228315


namespace NUMINAMATH_GPT_bus_driver_regular_rate_l2283_228333

theorem bus_driver_regular_rate (hours := 60) (total_pay := 1200) (regular_hours := 40) (overtime_rate_factor := 1.75) :
  ∃ R : ℝ, 40 * R + 20 * (1.75 * R) = 1200 ∧ R = 16 := 
by
  sorry

end NUMINAMATH_GPT_bus_driver_regular_rate_l2283_228333


namespace NUMINAMATH_GPT_keith_turnips_l2283_228384

theorem keith_turnips (Alyssa_turnips Keith_turnips : ℕ) 
  (total_turnips : Alyssa_turnips + Keith_turnips = 15) 
  (alyssa_grew : Alyssa_turnips = 9) : Keith_turnips = 6 :=
by
  sorry

end NUMINAMATH_GPT_keith_turnips_l2283_228384


namespace NUMINAMATH_GPT_arcsin_neg_half_eq_neg_pi_six_l2283_228397

theorem arcsin_neg_half_eq_neg_pi_six : 
  Real.arcsin (-1 / 2) = -Real.pi / 6 := 
sorry

end NUMINAMATH_GPT_arcsin_neg_half_eq_neg_pi_six_l2283_228397


namespace NUMINAMATH_GPT_min_sum_ab_l2283_228301

theorem min_sum_ab (a b : ℤ) (h : a * b = 196) : a + b = -197 :=
sorry

end NUMINAMATH_GPT_min_sum_ab_l2283_228301


namespace NUMINAMATH_GPT_complete_the_square_l2283_228317

theorem complete_the_square (x : ℝ) (h : x^2 - 8 * x - 1 = 0) : (x - 4)^2 = 17 :=
by
  -- proof steps would go here, but we use sorry for now
  sorry

end NUMINAMATH_GPT_complete_the_square_l2283_228317


namespace NUMINAMATH_GPT_sin_product_l2283_228349

theorem sin_product :
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (36 * Real.pi / 180)) *
  (Real.sin (72 * Real.pi / 180)) *
  (Real.sin (84 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_sin_product_l2283_228349


namespace NUMINAMATH_GPT_machine_production_in_10_seconds_l2283_228328

def items_per_minute : ℕ := 150
def seconds_per_minute : ℕ := 60
def production_rate_per_second : ℚ := items_per_minute / seconds_per_minute
def production_time_in_seconds : ℕ := 10
def expected_production_in_ten_seconds : ℚ := 25

theorem machine_production_in_10_seconds :
  (production_rate_per_second * production_time_in_seconds) = expected_production_in_ten_seconds :=
sorry

end NUMINAMATH_GPT_machine_production_in_10_seconds_l2283_228328


namespace NUMINAMATH_GPT_find_q_l2283_228388

-- Define the conditions and the statement to prove
theorem find_q (p q : ℝ) (hp1 : p > 1) (hq1 : q > 1) 
  (h1 : 1 / p + 1 / q = 3 / 2)
  (h2 : p * q = 9) : q = 6 := 
sorry

end NUMINAMATH_GPT_find_q_l2283_228388


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2283_228350

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2283_228350


namespace NUMINAMATH_GPT_percentage_favoring_all_three_l2283_228398

variable (A B C A_union_B_union_C Y X : ℝ)

-- Conditions
axiom hA : A = 0.50
axiom hB : B = 0.30
axiom hC : C = 0.20
axiom hA_union_B_union_C : A_union_B_union_C = 0.78
axiom hY : Y = 0.17

-- Question: Prove that the percentage of those asked favoring all three proposals is 5%
theorem percentage_favoring_all_three :
  A = 0.50 → B = 0.30 → C = 0.20 →
  A_union_B_union_C = 0.78 →
  Y = 0.17 →
  X = 0.05 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_favoring_all_three_l2283_228398


namespace NUMINAMATH_GPT_brownie_to_bess_ratio_l2283_228341

-- Define daily milk production
def bess_daily_milk : ℕ := 2
def daisy_daily_milk : ℕ := bess_daily_milk + 1

-- Calculate weekly milk production
def bess_weekly_milk : ℕ := bess_daily_milk * 7
def daisy_weekly_milk : ℕ := daisy_daily_milk * 7

-- Given total weekly milk production
def total_weekly_milk : ℕ := 77
def combined_bess_daisy_weekly_milk : ℕ := bess_weekly_milk + daisy_weekly_milk
def brownie_weekly_milk : ℕ := total_weekly_milk - combined_bess_daisy_weekly_milk

-- Main proof statement
theorem brownie_to_bess_ratio : brownie_weekly_milk / bess_weekly_milk = 3 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_brownie_to_bess_ratio_l2283_228341


namespace NUMINAMATH_GPT_avg_diff_l2283_228368

theorem avg_diff (n : ℕ) (m : ℝ) (mistake : ℝ) (true_value : ℝ)
   (h_n : n = 30) (h_mistake : mistake = 15) (h_true_value : true_value = 105) 
   (h_m : m = true_value - mistake) : 
   (m / n) = 3 := 
by
  sorry

end NUMINAMATH_GPT_avg_diff_l2283_228368


namespace NUMINAMATH_GPT_casey_savings_l2283_228375

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end NUMINAMATH_GPT_casey_savings_l2283_228375


namespace NUMINAMATH_GPT_evaluate_expression_l2283_228354

theorem evaluate_expression :
  let a := 5 ^ 1001
  let b := 6 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2283_228354


namespace NUMINAMATH_GPT_intersection_complement_N_M_eq_singleton_two_l2283_228334

def M : Set ℝ := {y | y ≥ 2}
def N : Set ℝ := {x | x > 2}
def C_R_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_N_M_eq_singleton_two :
  (C_R_N ∩ M = {2}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_N_M_eq_singleton_two_l2283_228334


namespace NUMINAMATH_GPT_min_value_abs_2a_minus_b_l2283_228323

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) : ∃ c : ℝ, c = |2 * a - b| ∧ c = 1 := 
sorry

end NUMINAMATH_GPT_min_value_abs_2a_minus_b_l2283_228323


namespace NUMINAMATH_GPT_maximize_profit_l2283_228318

variables (a x : ℝ) (t : ℝ := 5 - 12 / (x + 3)) (cost : ℝ := 10 + 2 * t) 
  (price : ℝ := 5 + 20 / t) (profit : ℝ := 2 * (price * t - cost - x))

-- Assume non-negativity and upper bound on promotional cost
variable (h_a_nonneg : 0 ≤ a)
variable (h_a_pos : 0 < a)

noncomputable def profit_function (x : ℝ) : ℝ := 20 - 4 / x - x

-- Prove the maximum promotional cost that maximizes the profit
theorem maximize_profit : 
  (if a ≥ 2 then ∃ y, y = 2 ∧ profit_function y = profit_function 2 
   else ∃ y, y = a ∧ profit_function y = profit_function a) := 
sorry

end NUMINAMATH_GPT_maximize_profit_l2283_228318


namespace NUMINAMATH_GPT_number_of_pictures_deleted_l2283_228360

-- Definitions based on the conditions
def total_files_deleted : ℕ := 17
def songs_deleted : ℕ := 8
def text_files_deleted : ℕ := 7

-- The question rewritten as a Lean theorem statement
theorem number_of_pictures_deleted : 
  (total_files_deleted - songs_deleted - text_files_deleted) = 2 := 
by
  sorry

end NUMINAMATH_GPT_number_of_pictures_deleted_l2283_228360


namespace NUMINAMATH_GPT_ratio_of_speeds_l2283_228358

theorem ratio_of_speeds
  (speed_of_tractor : ℝ)
  (speed_of_bike : ℝ)
  (speed_of_car : ℝ)
  (h1 : speed_of_tractor = 575 / 25)
  (h2 : speed_of_car = 331.2 / 4)
  (h3 : speed_of_bike = 2 * speed_of_tractor) :
  speed_of_car / speed_of_bike = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2283_228358


namespace NUMINAMATH_GPT_all_three_digits_same_two_digits_same_all_digits_different_l2283_228376

theorem all_three_digits_same (a : ℕ) (h1 : a < 10) (h2 : 3 * a = 24) : a = 8 :=
by sorry

theorem two_digits_same (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 2 * a + b = 24 ∨ a + 2 * b = 24) : 
  (a = 9 ∧ b = 6) ∨ (a = 6 ∧ b = 9) :=
by sorry

theorem all_digits_different (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) (h7 : a + b + c = 24) :
  (a, b, c) = (7, 8, 9) ∨ (a, b, c) = (7, 9, 8) ∨ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (8, 9, 7) ∨ (a, b, c) = (9, 7, 8) ∨ (a, b, c) = (9, 8, 7) :=
by sorry

end NUMINAMATH_GPT_all_three_digits_same_two_digits_same_all_digits_different_l2283_228376


namespace NUMINAMATH_GPT_problem_true_propositions_l2283_228357

-- Definitions
def is_square (q : ℕ) : Prop := q = 4
def is_trapezoid (q : ℕ) : Prop := q ≠ 4
def is_parallelogram (q : ℕ) : Prop := q = 2

-- Propositions
def prop_negation (p : Prop) : Prop := ¬ p
def prop_contrapositive (p q : Prop) : Prop := ¬ q → ¬ p
def prop_inverse (p q : Prop) : Prop := p → q

-- True propositions
theorem problem_true_propositions (a b c : ℕ) (h1 : ¬ (is_square 4)) (h2 : ¬ (is_parallelogram 3)) (h3 : ¬ (a * c^2 > b * c^2 → a > b)) : 
    (prop_negation (is_square 4) ∧ prop_contrapositive (is_trapezoid 3) (is_parallelogram 3)) ∧ ¬ prop_inverse (a * c^2 > b * c^2) (a > b) := 
by
    sorry

end NUMINAMATH_GPT_problem_true_propositions_l2283_228357


namespace NUMINAMATH_GPT_price_reduction_percentage_l2283_228351

theorem price_reduction_percentage (original_price new_price : ℕ) 
  (h_original : original_price = 250) 
  (h_new : new_price = 200) : 
  (original_price - new_price) * 100 / original_price = 20 := 
by 
  -- include the proof when needed
  sorry

end NUMINAMATH_GPT_price_reduction_percentage_l2283_228351


namespace NUMINAMATH_GPT_count_noncongruent_triangles_l2283_228382

theorem count_noncongruent_triangles :
  ∃ (n : ℕ), n = 13 ∧
  ∀ (a b c : ℕ), a < b ∧ b < c ∧ a + b > c ∧ a + b + c < 20 ∧ ¬(a * a + b * b = c * c)
  → n = 13 := by {
  sorry
}

end NUMINAMATH_GPT_count_noncongruent_triangles_l2283_228382
